from dataclasses import replace

import pytest

from app.generation.context import build_context, validate_and_render
from app.generation.models import AnswerValidationError, ContextLimitError, ModelAnswer
from app.ingestion.models import ChunkLocation
from app.retrieval.vector_store import SearchResult


def model_answer(source="S1", quote="apple fruit", text="It mentions fruit."):
    return ModelAnswer.model_validate({"status": "answered", "claims": [
        {"text": text, "evidence": [{"source_id": source, "quote": quote}]}]})


def test_context_exact_budget_and_omission(sample_chunks):
    hits = [SearchResult(chunk, 1.) for chunk in sample_chunks]
    first = build_context(hits[:1])
    assert build_context(hits, len(first.serialized)).serialized == first.serialized
    assert build_context(hits, len(first.serialized)).warnings
    with pytest.raises(ContextLimitError):
        build_context(hits[:1], len(first.serialized) - 1)
    assert len(build_context(hits).sources) == 2
    assert len(build_context(hits + hits).sources) == 2
    assert build_context([]).sources == {}


def test_oversized_first_chunk_does_not_hide_smaller_source(sample_chunks):
    large = replace(sample_chunks[0], text="x" * 1000)
    context = build_context([SearchResult(large, 2.), SearchResult(sample_chunks[1], 1.)], 100)
    assert context.sources["S1"] == sample_chunks[1]
    assert "x" * 1000 not in context.serialized


def test_citation_deterministic_and_all_locations_preserved(sample_chunks):
    chunk = replace(sample_chunks[0], locations=list(reversed(sample_chunks[0].locations)))
    context = build_context([SearchResult(chunk, 1.)])
    result = validate_and_render(model_answer(), context)
    citation = result.citations[0]
    assert citation.label == "example.pdf, page 1"
    assert citation.primary_location.page_number == 1
    assert citation.chunk.locations == chunk.locations
    assert result.answer == "It mentions fruit. [example.pdf, page 1]"
    result.citations[0].chunk.metadata["author"] = "modified"
    assert chunk.metadata["author"] == "Tester"
    assert context.sources["S1"].metadata["author"] == "Tester"


@pytest.mark.parametrize("filters,page", [({"page_number": 2}, 2),
    ({"page_label": "ii"}, 2), ({"page_number": 2, "page_label": "ii"}, 2)])
def test_active_page_filters(sample_chunks, filters, page):
    context = build_context([SearchResult(sample_chunks[0], 1.)])
    result = validate_and_render(model_answer(), context, filters)
    assert result.citations[0].primary_location.page_number == page
    assert len(result.citations[0].chunk.locations) == 2


def test_page_conditions_must_match_same_location(sample_chunks):
    context = build_context([SearchResult(sample_chunks[0], 1.)])
    with pytest.raises(AnswerValidationError):
        validate_and_render(model_answer(), context, {"page_number": 1, "page_label": "ii"})


def test_unpaged_and_page_label_only(sample_chunks):
    for location, label in [(None, "example.pdf"),
            (ChunkLocation(0, None, "Preface", 0, 11), "example.pdf, page label Preface")]:
        chunk = replace(sample_chunks[0], locations=[location] if location else [])
        result = validate_and_render(model_answer(), build_context([SearchResult(chunk, 1.)]))
        assert result.citations[0].label == label


def test_whitespace_normalization_only(sample_chunks):
    chunk = replace(sample_chunks[0], text="apple\n\t fruit")
    context = build_context([SearchResult(chunk, 1.)])
    assert validate_and_render(model_answer(quote="apple   fruit"), context).status == "answered"
    with pytest.raises(AnswerValidationError):
        validate_and_render(model_answer(quote="Apple fruit"), context)
    with pytest.raises(AnswerValidationError):
        validate_and_render(model_answer(quote="apple pear fruit"), context)


@pytest.mark.parametrize("source,quote", [("S99", "apple fruit"), ("S2", "python code"),
    ("S1", "fake quote"), ("S1", " \n")])
def test_one_invalid_claim_rejects_entire_answer(sample_chunks, source, quote):
    context = build_context([SearchResult(sample_chunks[0], 1.)])
    answer = model_answer()
    answer.claims.extend(model_answer(source, quote).claims)
    with pytest.raises(AnswerValidationError):
        validate_and_render(answer, context)


def test_missing_evidence_and_invalid_status_combinations(sample_chunks):
    context = build_context([SearchResult(sample_chunks[0], 1.)])
    for data in [{"status": "answered", "claims": []},
                 {"status": "answered", "claims": [{"text": "Claim", "evidence": []}]},
                 {"status": "answered", "claims": [{"text": " ", "evidence": [{"source_id": "S1", "quote": "apple"}]}]},
                 {"status": "insufficient_evidence", "claims": model_answer().model_dump()["claims"]}]:
        with pytest.raises(AnswerValidationError):
            validate_and_render(ModelAnswer.model_validate(data), context)
    result = validate_and_render(ModelAnswer(status="insufficient_evidence", claims=[]), context)
    assert result.status == "insufficient_evidence" and not result.citations


@pytest.mark.parametrize("text", ["It mentions fruit. [fake.pdf, page 999]",
    "Fruit (fake.pdf, page 999).", "Fruit [S99].", "Fruit 【S99】.",
    "Fruit per page 999.", "Fruit per fake.DOCX."])
def test_model_written_citations_reject_entire_answer(sample_chunks, text):
    context = build_context([SearchResult(sample_chunks[0], 1.)])
    answer = model_answer()
    answer.claims.extend(model_answer(text=text).claims)
    with pytest.raises(AnswerValidationError, match="citation markers or source labels"):
        validate_and_render(answer, context)
