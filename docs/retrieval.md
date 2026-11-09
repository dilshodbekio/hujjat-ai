# Gemini embeddings and persistent local Qdrant

## Hybrid search (Gemini + BM25 + RRF)

```python
with RetrievalPipeline() as retrieval:
    hits = retrieval.hybrid_search("SOC2030 attendance rules", limit=5,
        candidate_limit=20, filters={"filename": "handbook.pdf"})
```

`search()` remains semantic-only and unchanged. `hybrid_search()` embeds the
question using `RETRIEVAL_QUERY`, then asks Qdrant to fuse a dense-vector prefetch
and a BM25 text prefetch using equal-weight native Reciprocal Rank Fusion (RRF).
The same validated metadata filter is applied to BOTH candidate branches before
their limits, and to the final query. `candidate_limit` is the number of candidates
from each branch, must be an integer at least `limit`, and defaults to 20.
Invalid inputs fail before Gemini. Native fusion returns each point once;
all original chunk text, metadata, and location records are preserved.

Scores have different meanings:

- `search().score`: cosine similarity of Gemini vectors.
- `hybrid_search().score`: RRF rank contribution summed across branches, not
  cosine similarity, BM25 score, or probability. Do not reuse cosine thresholds
  on hybrid results. This implementation uses Qdrant's default RRF constant.

Requires `qdrant-client>=1.20,<2`. Compatibility was verified with 1.20.0 in a
temporary persistent local database BEFORE implementation: a populated dense-only
collection gained BM25 and filtered RRF without recreation or modifying its
points/vectors, and BM25 still worked after reopening. No Gemini calls were made.

The first hybrid query enables a BM25-scored text index on the EXISTING `text`
payload. There is no new sparse embedding model, sparse vector, document reload,
or Gemini re-embedding. Existing and future points use their stored chunk text.
`ensure_keyword_index()` can also initialize it explicitly. Existing compatible
configuration is reused; an incompatible supported text index is not overwritten.
Local Qdrant reports an empty payload schema even for text indexes, so setup uses
a public text-query capability probe and only creates an index on the specific
missing-index error. This version-dependent behavior is covered by integration
tests and should be checked when upgrading Qdrant.

Local BM25 supports default word tokenization, lowercase matching, and
collection-wide statistics. It does not support custom stemming/stopwords or
Uzbek morphology, and scans stored text to score candidates; it is not a
server-scale inverted-index performance guarantee. Tokenization can split
punctuation in identifiers and apostrophes. Keyword matches are lexical relevance,
not substring or exact-phrase constraints. Measure latency before scaling.

If the keyword branch has no matches (including punctuation-only queries),
semantic candidates remain available. Empty collections return `[]`. Provider
and database failures raise existing errors rather than silently switching modes.

Tests exercise persistence, dense-only setup, unchanged dense vectors and semantic
results, native fusion scores, metadata constraints on both branches, incompatible
indexes, new/updated points, failures, and a tiny labeled Recall@1/MRR comparison
of semantic, keyword, and hybrid rankings. The quality fixture uses synthetic
dense vectors to verify mechanics; it does not establish real Gemini retrieval
quality. A representative labeled corpus with live Gemini embeddings remains
necessary before making a product-quality claim.

## Metadata filtering

```python
hits = retrieval.search("What are the scholarship requirements?", limit=5, filters={
    "filename": "handbook.pdf",
    "format": "pdf",
    "page_number": 12,
    "metadata": {"author": "Inha University"},
})
```

Supported top-level filters are `filename`, `source_path`, `format`, `page_number`,
`page_label`, and `metadata`. They match existing stored payloads, so no reindexing
is needed. Multiple conditions use AND and are applied by Qdrant before choosing
the top results. String matches are exact and case-sensitive; only `format` is
lowercased. Accepted formats are `pdf`, `docx`, `txt`, `md`, and `markdown`, with
`md` and `markdown` matching their respective stored extensions. Paths must match
the stored source path exactly; they are not resolved or normalized by filtering.

Page numbers must be positive integers (not booleans), and match any retained
location. Page number and label together must match the same location. Returned
chunks retain all original locations, including locations on other pages.

`metadata` accepts string keys and non-empty string values. Keys must start with
an ASCII letter or underscore and contain only ASCII letters, digits, and
underscores; payload path expressions such as `author.name` or `x[]` are rejected.
This supports simple custom metadata such as `author`, `title`, or `department`;
it does not interpret nested metadata or add fields that were never stored.
Missing fields, null page information, and conflicting conditions do not match.
No matches returns `[]`. `None`, `{}`, or `{"metadata": {}}` leave search
unfiltered. Invalid field names, structures, types, or values raise
`InvalidFilterError` from `app.retrieval.filters` before a Gemini query request.

Tests use temporary real Qdrant databases and mocked Gemini, including combined
conditions, missing metadata, page-location correlation, unchanged payloads,
unfiltered searches, and filtering after reopening storage.

Install using the project's interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add the following settings to the project-root `.env`. Keep your actual key
private. The implementation loads this file explicitly and does not modify it.
Existing environment variables override values of the same name in the file.
`GOOGLE_API_KEY` is accepted as a fallback if `GEMINI_API_KEY` is empty or absent.

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=3072
QDRANT_PATH=./.qdrant
QDRANT_COLLECTION=document_chunks
```

Relative Qdrant paths are resolved against the `.env` file's directory, so the
default database remains at the same project location even if the working
directory changes. This is `QdrantClient(path=...)` disk storage, not in-memory
storage. No server or Docker installation is needed. Vectors and payloads survive
normal program exits, laptop shutdowns, and reopens. Keep the database directory;
deleting it deletes the index. Only one client/process may open that directory
at a time. Use the context manager to close clients and release the disk lock.

```python
from app.ingestion.pipeline import ingest_document
from app.ingestion.chunking import chunk_document
from app.retrieval.pipeline import RetrievalPipeline

document = ingest_document("data/example.pdf")
chunks = chunk_document(document)

with RetrievalPipeline() as retrieval:
    indexed = retrieval.index_chunks(chunks)
    hits = retrieval.search("What does the document say about scholarships?", limit=3)
    for hit in hits:
        print(hit.score, hit.chunk.text)
        print(hit.chunk.filename, hit.chunk.metadata, hit.chunk.locations)
```

On a later run, search the existing database without reloading or re-embedding
the document:

```python
with RetrievalPipeline() as retrieval:
    hits = retrieval.search("How do I apply?", limit=5)
```

Chunks are embedded with `RETRIEVAL_DOCUMENT`; questions use `RETRIEVAL_QUERY`.
Both use the same configured model and output dimension. Requests use batches of
at most 100 texts. All embedding batches must succeed before any points from
that `index_chunks` call are written. Response count, dimension, finite values,
and nonzero vectors are checked. Cosine similarity determines result order;
the returned score is a similarity score, not a probability.

Each point stores the complete `DocumentChunk`, including text, original chunk
ID, filename, source path, format, custom metadata, and all location records
(page numbers, page labels, part indices, and character offsets). Qdrant point
UUIDs are deterministic functions of source path and chunk ID. Reindexing the
same chunks updates those points instead of adding duplicates. Upsert does not
remove obsolete chunks when a source document changes or is deleted; source
replacement/deletion is a separate future operation.

The collection's persisted named-vector schema fingerprints the embedding
model, dimension, and document/query task pair. Opening an incompatible or
unrecognized collection raises `CollectionMismatchError`; it is never silently
recreated. To change models or dimensions, choose a new collection and reindex.

Error types:

- `ConfigurationError`: missing key or invalid settings.
- `AuthenticationError`: rejected key or denied Gemini permissions.
- `EmbeddingError`: quota, model, service/network failures, or invalid vectors.
- `CollectionMismatchError`: incompatible existing index configuration.
- `StorageError`: local lock, permission, write, or search failure.

API error messages are sanitized and raw provider exceptions suppressed so they
do not expose credentials. No points are written after an embedding failure.
Database writes are not promised to be transactional if a storage operation
itself fails; retry the same chunks safely using their deterministic IDs.

Run ` .\.venv\Scripts\python.exe -m pytest -q` for automated tests. Gemini is
mocked; local Qdrant tests use real temporary disk databases and verify reopening
from another process, metadata, ranking, and duplicate indexing. A passing suite
does not establish live Gemini connectivity or semantic quality; those require a
valid key and a separate live run. Persisted vectors are local, while creating
document and query embeddings requires Gemini internet access.

References: [Gemini embedding task types](https://ai.google.dev/gemini-api/docs/embeddings),
[Google Gen AI SDK](https://googleapis.github.io/python-genai/), and
[Qdrant local mode](https://github.com/qdrant/qdrant-client#local-mode).
