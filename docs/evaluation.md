# Reproducible RAG evaluation v1

This is evaluation infrastructure, not a benchmark result. No real document
corpus or manually labeled question set is bundled. Existing test fixtures are
SYNTHETIC and Gemini requests are MOCKED. Their scores only verify the machinery.

## Preserve the original database and reuse vectors

Close all Qdrant clients, including UI/server sessions using the local directory.
Every CLI command acquires the source's EXISTING `.lock` in read-only file mode,
copies the directory to temporary storage, and opens ONLY the copy. The original
lock is held until execution finishes. A locked source, absent directory/collection,
or symlink/junction in the source is rejected. No index is recreated in the source.
BM25 setup occurs only in the temporary copy. The copy is removed on exit.
Disk space for the full copy is required. Report outputs must be disjoint from
the source database; existing report paths are never overwritten.

Document vectors, payloads, chunk IDs, paths, and locations are reused directly.
No call to document embedding or indexing exists in the evaluator. A live
retrieval comparison uses ONE query vector per question, shared by semantic and
hybrid retrieval. BM25 makes no embedding request. Generation, when enabled, runs
the actual end-to-end pipeline once per question with a fresh query embedding,
so end-to-end timing does not pretend that a cached query is free.

## Prepare a real labeled dataset

First export a manifest from a CLOSED existing index (no API calls):

```powershell
python -m app.evaluation manifest --database .qdrant --output reports/corpus_manifest_v1.json
```

For non-default collections/models/dimensions, explicitly pass `--collection`,
`--embedding-model`, and `--dimensions`. Defaults match the original project
settings, not arbitrary indexes. A mismatching dense schema is rejected.
The manifest lists the exact stored chunk text and provenance to help manual
labeling. Its SHA-256 covers sorted point IDs, payloads AND stored vectors.
The evaluator does not invent source-file hashes or chunking settings: current
ingestion does not store those, so the manifest marks them unknown. Retain the
original source documents and record their hashes/chunking settings separately
when preparing a portfolio dataset. Moving source files and re-ingesting can
change IDs; copying the existing index preserves them.

Create a JSON file with this structure (placeholders are NOT real labels):

```json
{
  "version": 1,
  "kind": "manual",
  "corpus_sha256": "<exact hash from manifest>",
  "questions": [
    {
      "id": "q001",
      "question": "<manually verified question>",
      "filters": {"filename": "<stored filename>"},
      "answerable": true,
      "relevant_chunk_ids": ["<verified chunk ID>"],
      "expected_answer": "<reference answer>",
      "expected_facts": ["<required factual phrase>"],
      "supporting_sources": [{"filename": "<stored filename>", "page_number": 12}]
    },
    {
      "id": "q002",
      "question": "<verified unanswerable question>",
      "answerable": false,
      "relevant_chunk_ids": []
    }
  ]
}
```

Start with about 25 questions: 15 directly answerable, 5 paraphrase or exact-code
queries, and 5 insufficient-evidence cases. A person must examine the complete
filter-eligible corpus to label ALL relevant chunks, including overlapping chunks
that independently support an answer. Do not derive labels from a preferred
retriever's output. Negative questions have no positive labels or expected facts.
Unanswerability cannot be established automatically; the human label is a claim
that must be checked. `kind="synthetic"` is required for artificial fixtures.

Validation checks schema, unique IDs, corpus hash, existence of relevance IDs,
filter compatibility, and supporting filename/page locations:

```powershell
python -m app.evaluation validate --database .qdrant --dataset data/evaluation_v1.json
```

Validation cannot prove label completeness or factual correctness. Keep raw
documents, manifest, dataset and manual review notes together for reproducibility.

## Run measurements

```powershell
python -m app.evaluation run --database .qdrant --dataset data/evaluation_v1.json --output reports/evaluation/run_v1 --k 1 3 5 10 --live
```

`--live` explicitly allows Gemini QUERY embedding costs. To additionally measure
answers, add `--generate --generation-model gemini-3.1-flash-lite`. This model
passed the project's earlier small live generation check; availability and
performance can change. Without the model argument, existing generation config
is used. Never edit `.env` for evaluation. Credentials are loaded using the
existing convention, are omitted from artifacts, and exception messages from
providers are not serialized. Only error class names are recorded.

Client shutdown attempts both generation and embedding cleanup independently.
Cleanup failures are retained in an `ExceptionGroup`; an earlier execution error
remains available as exception context. The isolated store's context manager still
exits if client cleanup fails. CLI-facing errors remain sanitized.

`--candidate-limit` defaults to 20 and must be at least max(K).
`--generation-limit` defaults to 5 and must not exceed candidate depth.
Each method retrieves max(K) once and uses prefixes for smaller K. All three
methods use the same questions, filters, corpus, and relevance labels. Native
ranking algorithms remain unchanged. Startup/BM25 setup is timed separately.
Method order is fixed and recorded, not randomized; measure larger repeated
datasets if cache/order effects matter. There is no automatic warm-up or hidden
repeat in v1. SDK-level retries, if configured, contribute to measured call latency.

## Metric rules

For relevant-ID set R and the first K returned positions T:

- Recall@K = number of distinct relevant IDs in T divided by |R|.
- RR@K = reciprocal of the one-based first relevant rank within K, else zero.
- MRR@K = mean RR@K across answerable questions.
- Recall is macro-averaged across answerable questions too.
- Fewer than K results are not padded. Duplicate IDs consume rank positions and
  are flagged; they cannot inflate recall.
- Unanswerable questions have undefined retrieval relevance metrics and are
  excluded from those averages, not converted to perfect or zero retrieval scores.
- Errors are reported separately. Failure-inclusive averages count failed
  answerable queries as zero. Successful-query averages and both denominators
  remain in JSON so failures cannot silently improve the headline.

Cosine, BM25 and RRF scores are not directly comparable; metrics compare ranked
chunk IDs rather than mixing score thresholds across retrieval methods.

## Generation and human review

Generation reuses `GenerationPipeline.answer()` without changing its algorithms.
Deterministic diagnostics check expected answer/abstention status, normalized
expected-phrase presence in claim prose, evidence occurrence/IDs, citation payload
integrity, and coverage of expected supporting sources. Rebuilding validated
citations is an integrity check, not an independent entailment assessment.
Phrase presence can accept contradictions and miss paraphrases; it is NOT an
answer-correctness score. No LLM judge is used.

`report.md` contains each question, reference answer/facts, labels, actual answer,
all claim evidence quotes, citations and full source locations, included context,
rankings and failures. Use these side by side with the original documents to
judge correctness, faithfulness, and whether citations support each claim.
All judgments start `not_reviewed`; they are never automatically counted as pass.
Fill in `human_review.json` with `pass`, `fail`, or `not_reviewed` and notes for
each dimension. Preserve that file alongside the immutable run artifacts. V1 does
not import human edits or pretend that original machine summaries were updated;
include the filled review file when sharing results.

Errors are not abstentions. Correct abstentions, incorrect abstentions on positive
questions, status accuracy counts, and errors are separately reported.

## Latency and artifacts

Use monotonic `perf_counter()` seconds. Record shared query embedding, database
search, embedding-inclusive retrieval, generation retrieval, generation call, and
total end-to-end latency. Total includes context construction/validation and can
exceed the sum of separately measured calls. Skipped generation has a null call
time, not a fake zero sample. Failures/timeouts have separate elapsed summaries.
Mean and median include sample counts; nearest-rank p95 is only shown for at
least 20 samples and remains exploratory at 20–30. No p99 is claimed.

Each new output directory contains `results.json`, `report.md`, and
`human_review.json`. It records dataset/corpus hashes, model settings and package
versions, K/candidate/context/output limits, per-question evidence, diagnostics,
and latency. Reports can contain private document text: review before sharing.
Frozen inputs make runs comparable, but external Gemini models/services may
change and do not guarantee byte-identical output.

Run `.\.venv\Scripts\python.exe -m pytest -q` for automated checks. These use
mocked Gemini and temporary real Qdrant databases; no benchmark API calls or
claimed real-world scores are produced. The tests verify the original database
is byte-for-byte unchanged, corpus-label checks, exact metric calculations,
failure denominators, stage timing, report serialization, and CLI safeguards.
