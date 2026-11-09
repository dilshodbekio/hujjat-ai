"""Bounded source context and citations built only from retrieved payloads."""

from copy import deepcopy
import json
import re

from app.retrieval.filters import validate_filters
from app.retrieval.vector_store import SearchResult
from .evidence import extract_evidence_span
from .models import (AnswerResult, AnswerValidationError, Citation, ContextLimitError,
                     GenerationContext, INSUFFICIENT_ANSWER, ModelAnswer)


def build_context(hits: list[SearchResult], max_chars: int = 12000) -> GenerationContext:
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    records = []
    sources = {}
    seen = set()
    omitted = 0
    for hit in hits:
        chunk = hit.chunk
        if not chunk.text.strip():
            continue
        key = (chunk.source_path, chunk.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        source_id = f"S{len(sources) + 1}"
        record = {"source_id": source_id, "text": chunk.text}
        candidate = json.dumps(records + [record], ensure_ascii=False)
        if len(candidate) > max_chars:
            omitted += 1
            continue
        records.append(record)
        sources[source_id] = deepcopy(chunk)
    if not sources and seen:
        raise ContextLimitError("No complete retrieved chunk fits the context budget; increase context_chars.")
    return GenerationContext(json.dumps(records, ensure_ascii=False), sources,
                             [f"Omitted {omitted} chunk(s) due to context budget."] if omitted else [])


def validate_and_render(answer: ModelAnswer, context: GenerationContext,
                        filters: dict | None = None) -> AnswerResult:
    validated = validate_filters(filters)
    if answer.status == "insufficient_evidence":
        if answer.claims:
            raise AnswerValidationError("Insufficient-evidence responses must not contain claims.")
        return AnswerResult("insufficient_evidence", INSUFFICIENT_ANSWER,
                            warnings=list(context.warnings))
    if not answer.claims:
        raise AnswerValidationError("Answered responses must contain at least one evidenced claim.")
    citations = {}
    lines = []
    resolved_claims = deepcopy(answer.claims)
    for claim in resolved_claims:
        if not claim.text.strip() or not claim.evidence:
            raise AnswerValidationError("Every claim requires non-empty text and supporting evidence; entire answer rejected.")
        # Claim prose is not a second citation channel. Fail closed rather than
        # stripping model-written labels and potentially hiding an invalid claim.
        if (any(marker in claim.text for marker in "[]【】") or
                re.search(r"\b[\w.-]+\.(?:pdf|docx|txt|md|markdown)\b", claim.text, re.IGNORECASE) or
                re.search(r"\bS\d+\b|\bpages?\s+(?:label\s+)?\d+\b", claim.text, re.IGNORECASE)):
            raise AnswerValidationError("Claims must not contain citation markers or source labels; entire answer rejected.")
        labels = []
        for evidence in claim.evidence:
            chunk = context.sources.get(evidence.source_id)
            span = extract_evidence_span(chunk.text, evidence.quote) if chunk is not None else None
            if span is None:
                raise AnswerValidationError("Evidence must quote an included source; entire answer rejected.")
            evidence.quote = span.text
            if evidence.source_id not in citations:
                eligible = [location for location in chunk.locations if
                    ("page_number" not in validated or location.page_number == validated["page_number"])
                    and ("page_label" not in validated or location.page_label == validated["page_label"])]
                if not eligible and ("page_number" in validated or "page_label" in validated):
                    raise AnswerValidationError("Source has no location matching the active page filters.")
                primary = min(eligible, key=lambda location: (
                    location.page_number is None, location.page_number or 0,
                    location.part_index, location.start_char, location.end_char,
                    location.page_label or "")) if eligible else None
                label = chunk.filename
                if primary is not None and primary.page_number is not None:
                    label += f", page {primary.page_number}"
                elif primary is not None and primary.page_label:
                    label += f", page label {primary.page_label}"
                citations[evidence.source_id] = Citation(evidence.source_id, deepcopy(chunk), primary, label)
            label = citations[evidence.source_id].label
            if label not in labels:
                labels.append(label)
        lines.append(claim.text.strip() + " " + " ".join(f"[{label}]" for label in labels))
    return AnswerResult("answered", "\n".join(lines), resolved_claims,
                        list(citations.values()), list(context.warnings))
