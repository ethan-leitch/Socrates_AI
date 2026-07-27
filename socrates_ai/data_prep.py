import json
import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gutenbergpy.gutenbergcache import GutenbergCache, GutenbergCacheSettings

from socrates_ai.helpers import (
    CleanedDataChecks,
    clean_text,
    download_bookid,
    get_author_info,
    get_book_titles,
)
from socrates_ai.resources import PHILOSOPHY_QUERY, PHILOSOPHY_SHELVES


class TheFarmer:
    def __init__(self):
        self.created_cache = GutenbergCache.exists()
        self.cache = None
        self.output_dir = os.path.join("..", "data", "raw_books")
        self.metadata_path = os.path.join("..", "data", "metadata.json")
        self.clean_dir = os.path.join("..", "data", "clean_books")

    def set_data_destination(self):
        data_dir = os.path.abspath(
            os.path.join("..", "data", "gutenberg_catalog_cache")
        )
        os.makedirs(data_dir, exist_ok=True)

        # Deliberately NOT setting CacheUnpackDir here, even though that seems
        # like the obvious thing to do. gutenbergpy's own extraction code
        # (Utils.unpack_tarbz2) calls tar.extract(member) with no `path`
        # argument, which always extracts relative to the current working
        # directory, following the archive's own internal "cache/epub/..."
        # structure -- it never actually reads CacheUnpackDir. Only the
        # *parser* (reads FROM that path) and *cleanup* (deletes FROM that
        # path) honor it. Point it anywhere other than gutenbergpy's own
        # default and the parser looks in an empty directory (silently
        # "succeeding" with 0 books), while the real extracted files sit
        # wherever CWD actually put them, uncleaned. Confirmed this the hard
        # way: pointing it at data/gutenberg_catalog_cache/epub left 79,017
        # real files sitting in notebooks/cache/epub instead, and a
        # 0-book gutenbergindex.db that LOOKED valid enough to skip rebuilding.
        #
        # Only the final .db file's location (CacheFilename) is genuinely
        # respected end-to-end, so that's the only thing we redirect.
        GutenbergCacheSettings.set(
            CacheFilename=os.path.join(data_dir, "gutenbergindex.db"),
        )
        print("Storing cache data at:")
        print(GutenbergCacheSettings.CACHE_FILENAME)
        print(
            f"(RDF files unpack to '{GutenbergCacheSettings.CACHE_RDF_UNPACK_DIRECTORY}', "
            "relative to wherever this notebook's CWD is -- gutenbergpy always does this, "
            "regardless of settings, and cleans it up itself via deleteTemp=True)"
        )

    def retrieve_gutenberg_books(self):
        if self.created_cache:
            print(
                "Cache has already been created ensure .db & epub files are in gutenberg_cataglog_cache "
            )
        else:
            GutenbergCache.create(
                refresh=True,
                download=True,
                unpack=True,
                parse=True,
                cache=True,
                deleteTemp=True,
            )
        self.cache = GutenbergCache.get_cache()

    def farm_philosophy_books(self):
        query_results = self.cache.native_query(PHILOSOPHY_QUERY).fetchall()
        philosophy_book_ids = [r[0] for r in query_results]
        print(f"{len(philosophy_book_ids)} philosophy books found")

        os.makedirs(self.output_dir, exist_ok=True)

        metadata = {}
        # --- Step 1: gather metadata + confirm bookshelf ---
        for book_id in philosophy_book_ids:
            title, bookshelf_cat = get_book_titles(self.cache, book_id)
            authors = get_author_info(self.cache, book_id)

            # PHILOSOPHY_SHELVES, not a single hardcoded name -- a book can
            # legitimately come from either shelf, so checking against just
            # "Philosophy" would wrongly flag every book pulled from the
            # larger "Category: Philosophy & Ethics" shelf as suspicious
            is_philosophy = bookshelf_cat in PHILOSOPHY_SHELVES
            flag = "✓" if is_philosophy else "✗ CHECK THIS ONE"
            print(f"{book_id}: {title} — {authors} — {bookshelf_cat} {flag}")

            metadata[book_id] = {
                "title": title,
                "authors": authors,
                "bookshelf": bookshelf_cat,
                "is_flagged_philosophy": is_philosophy,
                "filename": f"{book_id}.txt",
            }
        # --- Step 2: download texts ---
        for book_id in philosophy_book_ids:
            filepath = os.path.join(self.output_dir, f"{book_id}.txt")

            if os.path.exists(filepath):
                print(f"Skipping {book_id}, already downloaded")
                metadata[book_id]["char_count"] = os.path.getsize(filepath)
                continue

            char_count, error = download_bookid(book_id, self.output_dir)
            metadata[book_id]["char_count"] = char_count
            if error is not None:
                metadata[book_id]["error"] = error

            time.sleep(1)

        # --- Step 3: save metadata ---
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nMetadata saved to {self.metadata_path}")
        print(f"Texts saved to {self.output_dir}")

    def clean_books(self):

        os.makedirs(self.clean_dir, exist_ok=True)
        with open(self.metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        for book_id, info in metadata.items():
            if info.get("char_count") is None:
                continue  # download failed for this book, nothing to clean
            raw_path = os.path.join(self.output_dir, info["filename"])
            clean_path = os.path.join(self.clean_dir, info["filename"])

            with open(raw_path, encoding="utf-8") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)

            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            info["clean_char_count"] = len(cleaned)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Cleaned {sum(1 for i in metadata.values() if 'clean_char_count' in i)} books -> {self.clean_dir}")


    def check_cleaned_data(self):
        checks = CleanedDataChecks(
            metadata_path=self.metadata_path,
            output_dir=self.output_dir,
            clean_dir=self.clean_dir,
            cache=self.cache,
        )
        checks.run_all()
        return checks







class EDAPainter:
    """
    Produces EDA plots for the cleaned philosophy corpus.
    Assumes TheFarmer has already run — reads data/metadata.json and
    data/clean_books/ from disk, so it works standalone even in a fresh kernel.
    """

    WORD_RE = re.compile(r"[A-Za-z']+")
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, metadata_path=None, clean_dir=None):
        self.metadata_path = metadata_path or os.path.join(
            "..", "data", "metadata.json"
        )
        self.clean_dir = clean_dir or os.path.join("..", "data", "clean_books")

        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._eda_df = None  # lazily built by self.eda_df, expensive to recompute

    @property
    def eda_df(self):
        """Word/vocab/sentence stats per book, computed once and cached."""
        if self._eda_df is None:
            self._eda_df = self._build_eda_df()
        return self._eda_df

    def _build_eda_df(self):
        rows = []
        for book_id, info in self.metadata.items():
            if "clean_char_count" not in info:
                continue

            with open(
                os.path.join(self.clean_dir, info["filename"]), encoding="utf-8"
            ) as f:
                text = f.read()

            words = self.WORD_RE.findall(text.lower())
            sentences = [
                s for s in self.SENTENCE_SPLIT_RE.split(text) if len(s.split()) > 2
            ]

            rows.append(
                {
                    "book_id": book_id,
                    "title": info["title"],
                    "word_count": len(words),
                    "unique_word_count": len(set(words)),
                    "lexical_diversity": len(set(words)) / len(words) if words else 0,
                    "sentence_count": len(sentences),
                    "avg_sentence_length": len(words) / len(sentences)
                    if sentences
                    else 0,
                }
            )

        return pd.DataFrame(rows).set_index("book_id")

    @staticmethod
    def _new_dark_axes(figsize=(10, 6)):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#1A1817")
        return fig, ax

    def plot_length_distribution(self):
        """Histogram of cleaned character counts per book."""
        clean_counts = pd.Series(
            {
                bid: info["clean_char_count"]
                for bid, info in self.metadata.items()
                if "clean_char_count" in info
            }
        )

        fig, ax = self._new_dark_axes()
        sns.histplot(clean_counts, bins=20, color="#18D5FFFF", ax=ax, alpha=0.95)

        ax.grid(True, color="#444444", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#FF8818")

        ax.set_title("Distribution of Cleaned Book Lengths")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Number of Books")

        plt.tight_layout()
        plt.show()

    def plot_cumulative_corpus_size(self):
        """
        Cumulative character count as books are "added" in download-rank order.
        Rank order comes straight from metadata's insertion order, since
        TheFarmer writes entries in the order the ranked query returned them.
        """
        ranked_ids = [
            bid for bid in self.metadata if "clean_char_count" in self.metadata[bid]
        ]
        cumulative_chars = pd.Series(
            [self.metadata[bid]["clean_char_count"] for bid in ranked_ids]
        ).cumsum()

        fig, ax = self._new_dark_axes()
        ax.plot(
            range(1, len(ranked_ids) + 1),
            cumulative_chars,
            color="#B02156FF",
            alpha=0.95,
            linewidth=2,
        )

        ax.grid(True, color="#444444", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#3E75EA")

        ax.set_title("Cumulative Corpus Size by Download Rank")
        ax.set_xlabel("Books Added (ranked by downloads, descending)")
        ax.set_ylabel("Cumulative Character Count")

        plt.tight_layout()
        plt.show()

        print(
            f"Total corpus size: {cumulative_chars.iloc[-1]:,} characters across {len(ranked_ids)} books"
        )

    def plot_lexical_diversity(self):
        """Horizontal bar chart of unique words / total words per book."""
        plot_df = self.eda_df.sort_values("lexical_diversity")

        fig, ax = self._new_dark_axes(figsize=(10, 14))
        ax.barh(
            plot_df["title"],
            plot_df["lexical_diversity"],
            color="#B02156FF",
            alpha=0.95,
        )

        ax.grid(True, color="#444444", linewidth=0.8, axis="x")
        for spine in ax.spines.values():
            spine.set_color("#3E75EA")

        ax.set_title("Lexical Diversity per Book (unique words / total words)")
        ax.set_xlabel("Lexical Diversity")
        ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        plt.show()

    def plot_avg_sentence_length(self):
        """Horizontal bar chart of average words per sentence per book."""
        plot_df = self.eda_df.sort_values("avg_sentence_length")

        fig, ax = self._new_dark_axes(figsize=(10, 14))
        ax.barh(
            plot_df["title"],
            plot_df["avg_sentence_length"],
            color="#B02156FF",
            alpha=0.95,
        )

        ax.grid(True, color="#444444", linewidth=0.8, axis="x")
        for spine in ax.spines.values():
            spine.set_color("#3E75EA")

        ax.set_title("Average Sentence Length per Book (words per sentence)")
        ax.set_xlabel("Average Words per Sentence")
        ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        plt.show()
