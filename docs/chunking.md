# Document chunking

```python
from app.ingestion.pipeline import ingest_document
from app.ingestion.chunking import chunk_document

document = ingest_document("data/example.pdf")
chunks = chunk_document(document)  # max_chars=800, overlap=80
embedding_inputs = [chunk.text for chunk in chunks]
```

`chunk_document` accepts the existing `IngestedDocument` without modifying it.
The limit counts Python characters, including whitespace and overlap, rather
than tokens or bytes. Override `max_chars` and `overlap` with keyword arguments;
the overlap must be an integer from zero through `max_chars - 1`.

Each part is split independently to preserve page boundaries. When a split is
needed, the chunk ends at the last available paragraph newline within the limit,
then sentence punctuation followed by whitespace, then whitespace, then the
character limit. Sentence boundaries are heuristic; abbreviations may be treated
as sentence endings. A short remaining part is kept whole. Adjacent source spans
overlap by the requested number of characters; overlap can begin inside a word.
Text and separators remain unchanged, including indentation and table tabs.

Each `DocumentChunk` contains `text` for embedding, a stable SHA-256 `chunk_id`,
`source_path`, `filename`, `format`, an independent copy of document `metadata`,
and `locations`. Each location records the original zero-based `part_index`,
`page_number`, `page_label`, `start_char`, and exclusive `end_char`. Offsets refer
to the input part's text, after any normalization performed during ingestion.
Page fields remain `None` for unpaged formats.

Empty parts and whitespace-only windows are skipped. Exact duplicate chunk texts
within one document are emitted once, with every occurrence retained in
`locations`, including occurrences on different pages. Results follow first
occurrence order. Intentional overlap is retained; near-duplicate text is not
removed. IDs depend on source path and exact chunk text, so they remain stable
across repeated runs on the same source. This prepares embedding inputs and
provenance; it does not call an embedding model or vector database.

Run automated checks with `python -m pytest -q`.
