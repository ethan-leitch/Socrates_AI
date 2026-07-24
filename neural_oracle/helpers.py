import requests
import os
import re
import json
import hashlib
import unicodedata
from difflib import SequenceMatcher

from gutenbergpy.textget import get_text_by_id, strip_headers


def fetch_book_text(book_id):
    """
    Custom fetch funciton for when get_text_by_id() fails
    """
    urls_to_try = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]
    for url in urls_to_try:
        resp = requests.get(url)
        if resp.status_code == 200:
            raw_bytes = resp.content
            try:
                return raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return raw_bytes.decode(
                    "latin-1"
                )  # latin-1 never fails, maps every byte 0-255
    raise ValueError(f"Could not fetch text for {book_id}")


def get_book_titles(cache, book_id):
    title_query = f"""
    SELECT t.name, bs.name
    FROM books b
    JOIN titles t ON t.bookid = b.id
    JOIN bookshelves bs ON bs.id = b.bookshelveid
    WHERE b.gutenbergbookid = {book_id}
    """
    title_results = cache.native_query(title_query).fetchall()
    title = title_results[0][0] if title_results else "UNKNOWN"
    bookshelf_category = title_results[0][1] if title_results else "UNKNOWN"
    return title, bookshelf_category


def get_author_info(cache, book_id):
    author_query = f"""
    SELECT a.name
    FROM books b
    JOIN book_authors ba ON ba.bookid = b.id
    JOIN authors a ON a.id = ba.authorid
    WHERE b.gutenbergbookid = {book_id}
    """
    author_results = cache.native_query(author_query).fetchall()
    authors = [r[0] for r in author_results]
    return authors


def download_bookid(book_id, output_dir):
    filepath = os.path.join(output_dir, f"{book_id}.txt")
    try:
        try:
            raw = get_text_by_id(book_id)
            text = strip_headers(raw).decode("utf-8", errors="ignore")
        except UnicodeDecodeError as e:
            print(
                f"gutenbergpy encoding failed on {book_id} ({e}), trying direct download..."
            )
            text = fetch_book_text(book_id)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved {book_id} ({len(text)} chars)")
        return len(text), None

    except Exception as e:
        print(f"Failed on {book_id}: {e}")
        return None, str(e)


START_RE = re.compile(r"^.*start of (?:the|this) project gutenberg.*$", re.IGNORECASE | re.MULTILINE)
END_RE = re.compile(r"^.*end of (?:the|this) project gutenberg.*$", re.IGNORECASE | re.MULTILINE)

def strip_gutenberg_boilerplate(text):
    """
    Removes Project Gutenberg header/footer boilerplate.
    gutenbergpy's strip_headers() already handles most files, but the
    requests-based fallback in fetch_book_text() does not, so this has
    to stay idempotent for already-stripped text.
    """
    start_match = START_RE.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()

def clean_text(text):
    """
    Light normalization for LM training data. Keeps wording/punctuation
    intact, only touches encoding artifacts and whitespace.
    """
    text = text.lstrip("﻿")  # BOM from some direct downloads
    text = strip_gutenberg_boilerplate(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)      # trailing whitespace per line
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse 3+ blank lines to 1
    return text.strip()


class CleanedDataChecks:
    """
    Sanity checks for a downloaded + cleaned philosophy corpus. Reads
    metadata.json plus the raw/clean book directories from disk, so it can
    be run standalone against whatever TheFarmer has already produced.
    """

    def __init__(self, metadata_path, output_dir, clean_dir, cache=None):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.clean_dir = clean_dir
        self.cache = cache  # only needed for check_non_english

        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

    def check_missing_and_failed(self):
        missing_raw = [
            bid for bid, info in self.metadata.items()
            if not os.path.exists(os.path.join(self.output_dir, info["filename"]))
        ]
        failed_downloads = [bid for bid, info in self.metadata.items() if info.get("error")]

        print(f"Missing raw files: {missing_raw}")
        print(f"Books with download errors: {failed_downloads}")
        return missing_raw, failed_downloads

    def check_short_texts(self, threshold=5000):
        short_books = {
            bid: info["clean_char_count"]
            for bid, info in self.metadata.items()
            if info.get("clean_char_count", 0) < threshold
        }
        print(f"Books below {threshold} chars after cleaning:")
        for bid, count in short_books.items():
            print(f"  {bid}: {self.metadata[bid]['title']} ({count} chars)")
        return short_books

    def check_leftover_boilerplate(self):
        leftover = []
        for book_id, info in self.metadata.items():
            if "clean_char_count" not in info:
                continue
            with open(os.path.join(self.clean_dir, info["filename"]), encoding="utf-8") as f:
                text = f.read()
            head, tail = text[:500].upper(), text[-500:].upper()
            if "PROJECT GUTENBERG" in head or "PROJECT GUTENBERG" in tail:
                leftover.append(book_id)

        print(f"Books with possible leftover boilerplate: {leftover}")
        return leftover

    def check_non_english(self):
        if self.cache is None:
            print("No Gutenberg cache provided, skipping language check")
            return None

        book_ids_str = ",".join(str(bid) for bid in self.metadata.keys())
        lang_query = f"""
        SELECT b.gutenbergbookid, l.name
        FROM books b
        JOIN languages l ON l.id = b.languageid
        WHERE b.gutenbergbookid IN ({book_ids_str})
        """
        non_english = [
            (bid, lang) for bid, lang in self.cache.native_query(lang_query).fetchall()
            if lang != "en"
        ]
        print(f"Non-English books: {non_english}")
        return non_english

    def check_exact_duplicates(self):
        seen_hashes = {}
        exact_duplicates = []
        for book_id, info in self.metadata.items():
            if "clean_char_count" not in info:
                continue
            with open(os.path.join(self.clean_dir, info["filename"]), encoding="utf-8") as f:
                text = f.read()
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                exact_duplicates.append((seen_hashes[text_hash], book_id))
            else:
                seen_hashes[text_hash] = book_id

        print(f"Exact duplicate texts: {exact_duplicates}")
        return exact_duplicates

    def check_near_duplicates(self, similarity_threshold=0.5):
        items = list(self.metadata.items())
        near_duplicates = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                bid_a, info_a = items[i]
                bid_b, info_b = items[j]
                if set(info_a["authors"]) & set(info_b["authors"]):
                    similarity = SequenceMatcher(None, info_a["title"], info_b["title"]).ratio()
                    if similarity > similarity_threshold:
                        near_duplicates.append(
                            (bid_a, info_a["title"], bid_b, info_b["title"], round(similarity, 2))
                        )

        print("Possible duplicate/overlapping works (verify manually before excluding):")
        for d in near_duplicates:
            print(f"  {d[0]} {d[1]!r}  <->  {d[2]} {d[3]!r}  (similarity={d[4]})")
        return near_duplicates

    def run_all(self):
        print("=== Check 1: missing/failed downloads ===")
        self.check_missing_and_failed()
        print("\n=== Check 2: suspiciously short cleaned texts ===")
        self.check_short_texts()
        print("\n=== Check 3: leftover boilerplate ===")
        self.check_leftover_boilerplate()
        print("\n=== Check 4: non-English books ===")
        self.check_non_english()
        print("\n=== Check 5: exact duplicate texts ===")
        self.check_exact_duplicates()
        print("\n=== Check 6: near-duplicate/overlapping works ===")
        self.check_near_duplicates()