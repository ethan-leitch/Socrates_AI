TOP_50_PHILOSOPHY_QUERY = """
SELECT b.gutenbergbookid, b.numdownloads, bs.name, MIN(t.name) AS title
FROM books b
JOIN bookshelves bs ON bs.id = b.bookshelveid
JOIN titles t ON t.bookid = b.id
JOIN types ty ON ty.id = b.typeid
JOIN languages l ON l.id = b.languageid
WHERE bs.name = 'Philosophy'
  AND ty.name = 'Text'
  AND l.name = 'en'
GROUP BY b.id
ORDER BY b.numdownloads DESC
LIMIT 50
"""