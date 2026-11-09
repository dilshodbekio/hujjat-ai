# Grounded RAG answers — Generation v1

```python
from app.generation.pipeline import GenerationPipeline

with GenerationPipeline() as rag:
    result = rag.answer("What are the scholarship requirements?",
                        filters={"filename": "handbook.pdf"}, retrieval_limit=5)
    print(result.status)
    print(result.answer)
    for citation in result.citations:
        print(citation.label, citation.chunk.source_path, citation.chunk.locations)
```

The existing `.env` key-loading convention is reused. `.env` is not changed by
this feature. `GEMINI_GENERATION_MODEL` optionally selects a text-generation model
(default `gemini-3.8-flash`); it is separate from `GEMINI_EMBEDDING_MODEL`.
Availability depends on the account; no live generation call was made by tests.
The generation SDK uses `generate_content` with a separate system instruction,
JSON MIME type, and strict Pydantic schema. It enables no external tools.

`GenerationConfig(context_chars=12000, question_chars=4000,
max_output_tokens=2048)` controls budgets. Context uses complete chunks in
retrieval order, skips empty or repeated source/chunk IDs, and tries smaller later
chunks if a larger one does not fit. The character budget includes the exact JSON
source records and wrappers, not the question, system instruction, or token count.
Omitted chunks produce warnings. No complete usable chunk fitting the budget
raises `ContextLimitError`; no usable retrieved text returns insufficient evidence
without a generation request. Chunks are never silently truncated.

The pipeline calls existing hybrid retrieval with the same metadata filters.
Retrieval scores are not treated as confidence or proof of sufficient evidence.
Source records sent to Gemini contain local IDs and text. Trusted filenames and
locations stay in an application-owned registry. Documents and the user question
are JSON-serialized as untrusted data; system instructions prohibit following
embedded instructions or adding outside knowledge. This reduces prompt-injection
risk but cannot guarantee model compliance or semantic correctness.

The structured response has `status` (`answered` or `insufficient_evidence`) and
`claims`. Each answered claim requires non-empty text and at least one evidence
record with `source_id` and a supporting `quote`. The application checks every
evidence reference against sources actually INCLUDED in the context, then checks
that the quote occurs in the source text after collapsing whitespace runs. A
narrow PDF table exception allows an inserted colon after a short standalone
capitalized alphabetic heading to match a single source newline before its value.
No colon is accepted for inline text, paragraph breaks, numeric/punctuated labels,
or labels containing negations. This is deterministic matching with character
offsets: accepted evidence is replaced with the EXACT contiguous source slice,
so model-inserted punctuation is never displayed as an original quote. Case,
all other punctuation, numbers, dates, names, and words remain unchanged. Token
boundaries reject substrings cut from names, signed numbers, decimals, and other
numeric expressions; adjacent omitted negations are rejected conservatively.
One invalid or missing evidence record rejects the ENTIRE answer
with `AnswerValidationError`. Claims are never silently removed. Quote occurrence
does not prove entailment; real factual grounding needs evaluation.

Claim prose is deliberately restricted to keep citations in one trusted channel:
square citation brackets (including `【】`), source IDs such as `S1`, recognizable
document filenames, and numeric page-reference syntax are rejected in model claim
text, even with otherwise valid evidence. The entire response fails validation;
the renderer never silently strips these markers. This conservative rule can
reject otherwise legitimate prose containing such syntax; source references must
be expressed through evidence records and application-generated citations.

Citation labels are constructed from stored filenames and locations, never from
model-supplied labels. With page filters, the displayed location must match both
active page number and page label in the same location. Without filters, choose
the smallest available page number, then part index, offsets, and label as stable
tie-breakers. Numberless locations follow numbered locations. Use filename only
if no page information exists, or explicitly say `page label` when only a label
exists. All genuine locations, metadata, source paths, chunk IDs, and evidence
remain available in `AnswerResult` through claims and citation chunks. Context
and returned citations copy metadata so downstream edits do not mutate retrieval
results. Displayed citations are appended per claim by the application.

Empty retrieval and explicit model abstention return:
`The answer cannot be determined from the available documents.`
with `status="insufficient_evidence"`, no claims or citations. Abstention with
claims and answered status without claims are validation errors. Empty/malformed
JSON, unexpected fields, invalid evidence, blocked or truncated output, and API
failures are errors, not insufficient-evidence responses. Provider errors are
sanitized, and no automatic repair/retry request is made by this layer.
Retrieval errors retain existing retrieval exception types.

`GenerationPipeline` owns and closes the retrieval/generator clients it creates.
Injected clients are borrowed and must be closed by their caller. Generation v1
does not change ingestion, retrieval signatures, storage schemas, or indexes.

Run `.\.venv\Scripts\python.exe -m pytest -q` for all tests. Generation and Gemini
embeddings are mocked; integration uses a real temporary local Qdrant database.
Tests check evidence rejection, whitespace rules, deterministic/page-filtered
citations, context budgets, malformed/blocked output, input validation, retrieval
integration, and backward compatibility. Live answer quality and adversarial
prompt-injection behavior require separate evaluation with representative documents.
