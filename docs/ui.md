# hujjat-ai local Gradio UI

Install and launch from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.ui --generation-model gemini-3.1-flash-lite
```

Open **http://127.0.0.1:7860**. Gradio is pinned to the verified **6.31.0** API.
`--port 7861` selects another local port. `--database <directory>` selects a
separate persistent Qdrant directory for experiments. Without that flag the
existing configured Qdrant directory is reused, with no migration or deletion.
Close other clients using that directory before launching. The model argument
overrides generation only for this process; it does not change embedding settings
or `.env`. Credentials use the existing backend loading convention, with no key
input or credential display in the browser. Without an override, existing
generation configuration is used.

## Workflow

1. Select up to 10 documents (PDF, DOCX, TXT, MD, Markdown), at most 20 MB each.
2. Click **Index documents**. Stage progress and per-file results show what
   completed, was already present, or failed. A failure does not discard successful
   files. DOCX expanded content is capped at 100 MB before parsing.
3. Select **All indexed documents** or one document in **Search in**. Refresh
   reloads inventory from Qdrant, including pre-existing documents.
4. Ask a complete question. Chat displays the grounded answer and citations;
   **Validated supporting sources** shows only evidence/citations returned by
   successful generation validation, never arbitrary retrieved chunks.
5. **Clear chat** clears the session display, not the persistent document library.

Each question is independent. Display history is bounded to 20 question/answer
pairs and belongs to the browser session; it is not sent as conversation context.
The document library is shared within this local application. This is a local
single-user MVP, without authentication or a separate API framework.

## Identity, metadata, and partial indexing

The original filename is preserved separately from the processing filename,
including Unicode, spaces, and extension casing. Core document metadata and all
page locations pass through the existing ingestion/chunking pipeline unchanged.
Upload identities use a SHA-256 content digest plus the original filename, not
the temporary cache path. This makes repeated uploads stable across app restarts.
Identical bytes uploaded under different names are separate sources. Changed
bytes under the same filename create another version; old versions are never
silently replaced or deleted. Dropdown labels use filename plus an opaque short
ID to distinguish same-name documents. Full source paths stay on the server.

Before indexing, the backend retrieves deterministic point IDs in batches and
checks their dense vector and full chunk payload. Only absent/incomplete chunks
are embedded. If a write partially completed, re-uploading resumes missing chunks
without embedding already stored chunks. Existing payload conflicts produce a
safe failure rather than an overwrite. An after-write check confirms all expected
chunks are present. Partial writes are not represented as transactional: retry
the same file after an error. Documents with at least one saved point appear in
the inventory, including partially indexed uploads.

Pre-existing documents retain their original IDs and paths and remain selectable
without migration. Re-uploading a legacy document may create a new upload-identity
source, because old records do not contain a reliable content hash; we do not
guess whether differently identified old and new records are the same document.

## Privacy, lifecycle, and limits

Qdrant initialization, every database operation, Gemini work, and resource closure
run on one dedicated worker thread. Gradio callbacks share a serialized queue;
the worker also protects SQLite thread affinity. Shutdown stops the server,
drains accepted worker jobs, closes owned clients on that same thread, and then
removes the private upload cache. Worker initialization and shutdown failures
still terminate the executor. Ctrl+C is the normal shutdown path.

Uploads are handled as raw Gradio file metadata so `orig_name` survives. The
service therefore independently enforces private cache-root containment, link
checks, original-filename rules, file count, byte count, and extensions before
reading. It creates its own temporary processing copy and deletes that copy on
success or failure. It does not delete a browser cache file while another queued
request may still reference it. Gradio cache cleanup runs hourly, and the owned
cache directory is removed on normal process shutdown. A forced process kill may
leave OS temporary files behind. Qdrant storage is never included in cleanup.

The server binds only to loopback with sharing disabled. Project files and the
database directory are blocked from file serving. No backend path or raw exception
is returned by UI callbacks. Chat/evidence are rendered as plain text, with HTML
and file downloads disabled. Expected errors use fixed, understandable messages;
insufficient evidence stays an ordinary grounded-answer outcome. This local MVP
does not stream unvalidated partial answers or claim model-level security beyond
the existing evidence validation and system instructions.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use temporary Qdrant databases and mocked Gemini. They cover original names
and metadata, PDF page citations, duplicate uploads, recovery from partial writes,
same-name document filtering, legacy documents, safe errors/paths, processing-copy
cleanup, single-thread ownership, initialization and shutdown failure handling,
session-history isolation, Gradio component/callback compatibility, and a real
loopback HTTP server startup/file-access smoke test. This is automated UI/server
verification, not a manual browser walkthrough or a new live Gemini UI test.
On Windows a restricted sandbox may block Gradio's event-loop socket pair or old
pytest temp directories; use an allowed normal terminal with fresh temp storage.
