# Gutenberg tags philosophy content under two separate bookshelves: the older
# "Philosophy" shelf (78 EN text books) and a much larger "Category:" shelf
# introduced later (556 EN text books). Both are needed to get the full pool.
# Kept as a tuple here so data_prep.py's is_philosophy sanity check can share
# the exact same list rather than drifting out of sync with the SQL below.
PHILOSOPHY_SHELVES = ("Philosophy", "Category: Philosophy & Ethics")

PHILOSOPHY_QUERY = """
SELECT b.gutenbergbookid, b.numdownloads, bs.name, MIN(t.name) AS title
FROM books b
JOIN bookshelves bs ON bs.id = b.bookshelveid
JOIN titles t ON t.bookid = b.id
JOIN types ty ON ty.id = b.typeid
JOIN languages l ON l.id = b.languageid
WHERE bs.name IN ('Philosophy', 'Category: Philosophy & Ethics')
  AND ty.name = 'Text'
  AND l.name = 'en'
GROUP BY b.id
ORDER BY b.numdownloads DESC
"""