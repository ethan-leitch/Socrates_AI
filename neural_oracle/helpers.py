import requests
import os

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
