# hujjat-ai (حجة)

**hujjat-ai** (*hujjat* / حجة: evidence, proof) is a Python-based Retrieval-Augmented Generation (RAG) system designed around strict evidence grounding, hybrid search, reproducible evaluation, and an interactive Gradio web interface.

The application pairs Google Gemini models (`gemini-3.8-flash` for generation, `gemini-embedding-001` for dense representations) with a local Qdrant instance combining vector similarity and BM25 lexical search merged via Reciprocal Rank Fusion (RRF).

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Documents (PDF, DOCX,     |
                      |          TXT, MD)           |
                      +--------------+--------------+
                                     |
                                     v
                       [ Ingestion & Normalization ]
                       - Structure-aware chunking
                       - Natural boundary windows
                       - Breadcrumb heading metadata
                                     |
                                     v
                      +-----------------------------+
                      |    Local Qdrant Storage     |
                      | - Dense Gemini Vectors      |
                      | - Native BM25 Payload Index |
                      | - Reciprocal Rank Fusion    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Strict Context Assembly   |
                      | - Budget limits (12k chars) |
                      | - Complete chunk packing    |
                      | - Provenance tracking       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Gemini Flash Generator   |
                      | - Structured Pydantic schema|
                      | - Answer status + claims    |
                      | - Claim-level quote spans   |
                      +--------------+--------------+
                                     |
                                     v
                       [ Deterministic Span Check ]
                       - Verifies contiguous text
                       - Preserves negations/numbers
                       - Replaces with exact source
                                     |
                                     v
                      +-----------------------------+
                      | Gradio Web UI / Eval CLI    |
                      +-----------------------------+
```

### Module Responsibilities

- **`app/ingestion/`**: Loads text from PDF, DOCX, TXT, and Markdown files. Chunks documents using natural paragraph, sentence, and word boundary heuristics while preserving page numbers, heading breadcrumbs, and structural hierarchy.
- **`app/retrieval/`**: Manages vector embeddings and indexing with local Qdrant. Supports dense semantic search (3072 dimensions), native BM25 text queries, metadata filtering, and equal-weight Reciprocal Rank Fusion (RRF).
- **`app/generation/`**: Formulates structured prompts with strict character budgets. Gemini (`gemini-3.8-flash`) emits structured JSON containing status (`answered` or `insufficient_evidence`) and claims with supporting quote spans. Every quote span is deterministically validated against the exact retrieved chunk text before being accepted.
- **`app/evaluation/`**: Evaluates retrieval and generation pipelines without modifying production data. Automatically creates a read-only lock, snapshots the database into a temporary directory, and runs retrieval benchmarks and generation diagnostics.
- **`app/ui/`**: Local Gradio web interface providing document upload, background indexing with progress tracking, interactive chat, and citation drill-down views.

---

## Features

- **Hybrid Retrieval**: Combines semantic embeddings (`gemini-embedding-001`) with native Qdrant BM25 keyword scoring using Reciprocal Rank Fusion (RRF).
- **Strict Evidence Grounding**: The generation layer requires the model to supply exact quote spans for each factual claim. The application deterministically matches quotes against source context, protecting against hallucinated citations, altered dates/numbers, and omitted negations.
- **Safe Evaluation Harness**: Evaluates retrieval (Recall@K, MRR across dense, BM25, and hybrid) and generation diagnostics on temporary database snapshots without altering or corrupting the persistent store.
- **Interactive Web Interface**: Single-page Gradio UI for document ingestion, real-time status reporting, interactive question answering, and inline source citation review.

---

## Prerequisites

- **Python**: 3.11 or newer (tested on Python 3.13)
- **Google Gemini API Key**: Access to `gemini-3.8-flash` and `gemini-embedding-001`

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hujjat-ai.git
   cd hujjat-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Copy `.env.example` to create your local `.env` file:

```bash
# On macOS / Linux:
cp .env.example .env

# On Windows:
copy .env.example .env
```

Edit `.env` to configure your Gemini API key and optional overrides:

```env
# Google Gemini API Credentials (required)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Generation Model (defaults to gemini-3.8-flash)
GEMINI_GENERATION_MODEL=gemini-3.8-flash

# Embedding Settings (defaults to gemini-embedding-001 with 3072 dimensions)
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSIONS=3072

# Local Qdrant Storage (must be a persistent disk directory)
QDRANT_PATH=./.qdrant
QDRANT_COLLECTION=document_chunks
```

---

## Usage

### 1. Launching the Gradio Web Interface

Run the UI module directly:

```bash
python -m app.ui
```

Open your browser to the local URL (defaults to `http://127.0.0.1:7860`). Through the interface you can:
1. Upload documents (`.pdf`, `.docx`, `.txt`, `.md`).
2. Trigger indexing into the local Qdrant collection.
3. Ask questions and inspect answers with verified quotes and source citations.

### 2. Python API

You can also use the generation pipeline directly in your Python code:

```python
from app.generation.pipeline import GenerationPipeline

with GenerationPipeline() as rag:
    result = rag.answer(
        "What are the eligibility requirements for sick leave?",
        retrieval_limit=5,
        filters={"filename": "employee_handbook.pdf"}
    )

    print(f"Status: {result.status}")
    print(f"Answer: {result.answer}")
    for citation in result.citations:
        print(f"Citation: [{citation.label}] {citation.chunk.filename} (Page {citation.chunk.locations[0].page_number})")
        print(f"Quote: \"{citation.quote}\"")
```

### 3. Evaluation Framework

The evaluation CLI operates on an isolated snapshot of an existing Qdrant database to prevent accidental data modification.

**Step A: Export a Corpus Manifest**
```bash
python -m app.evaluation manifest --database .qdrant --output reports/corpus_manifest.json
```

**Step B: Run Benchmark Evaluation**
```bash
python -m app.evaluation run \
  --dataset path/to/dataset.json \
  --database .qdrant \
  --output reports/eval_results.json \
  --ks 1,3,5,10
```

The evaluator reports:
- **Retrieval Metrics**: Recall@K and Reciprocal Rank (RR) for dense, BM25, and hybrid search.
- **Latency Diagnostics**: Mean, median, and p95 retrieval and generation timings.
- **Generation Diagnostics**: Answer status agreement, expected fact presence, and deterministic evidence quote integrity checks.

---

## Automated Tests

The test suite contains **318 tests** across 17 test modules:

- **308 Unit and Integration Tests**: Test document parsing, chunk boundary heuristics, embedding handling, metadata filtering, Qdrant vector operations, context window budgeting, Gemini generation integration, quote verification, and evaluation metrics. These tests execute in-process without requiring network access or external services.
- **10 Application & Audit Tests**: Verify full Gradio UI layout wiring, HTTP loopback communication, error sanitization, and end-to-end evaluation lifecycle.

### Running the Tests

To run the in-process test suite:

```bash
pytest tests/ --ignore=tests/test_ui_app.py --ignore=tests/test_dynamic_audit.py
```

To run all tests including loopback interface tests (requires loopback socket permissions):

```bash
pytest tests/
```

---

## Known Scope and Limitations

- **Text Extraction**: Ingestion extracts selectable text from PDF, DOCX, TXT, and Markdown files. Optical Character Recognition (OCR) for scanned PDFs or images is not included in this release.
- **Single-Node Embedded Vector Store**: Storage uses local disk-backed Qdrant. Multi-node distributed clustering is not supported in the local mode.
- **Evidence Quote Strictness**: The evidence validator rejects quotes that cannot be deterministically matched to the retrieved source text. If the LLM alters numbers, dates, negations, or rephrases quotes, the claim is rejected rather than attributing an inaccurate quote.
- **Evaluation Metrics Scope**: Evaluation focuses on deterministic metrics (Recall@K, MRR, latency, fact presence). It intentionally avoids automated LLM-as-a-judge scoring for subjective faithfulness, providing structured schemas for human review instead.

---

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
