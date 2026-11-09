from copy import deepcopy
from dataclasses import replace

import pytest

from app.ingestion.models import ChunkLocation
from app.retrieval.filters import InvalidFilterError, build_qdrant_filter, validate_filters
from app.retrieval.vector_store import LocalVectorStore


@pytest.mark.parametrize("filters", [[], "pdf", 5, False, {"unknown": "value"},
    {"filename": " "}, {"source_path": None}, {"page_label": 1}, {"format": "exe"},
    {"page_number": True}, {"page_number": 0}, {"page_number": -1}, {"page_number": "1"},
    {"page_number": 1.0}, {"metadata": []}, {"metadata": {"author": None}},
    {"metadata": {"author": 3}}, {"metadata": {"author": ""}},
    {"metadata": {"author.name": "Test"}}, {"metadata": {"x[]": "Test"}},
    {"metadata": {1: "Test"}}, {"metadata": {"": "Test"}}])
def test_invalid_filters(filters):
    with pytest.raises(InvalidFilterError):
        build_qdrant_filter(filters)


def test_normalization_and_no_mutation():
    original = {"format": "PDF", "filename": "Case.PDF", "metadata": {"author": " Author "}}
    before = deepcopy(original)
    validated = validate_filters(original)
    assert validated["format"] == "pdf"
    assert validated["filename"] == "Case.PDF"
    assert validated["metadata"]["author"] == " Author "
    validated["metadata"]["author"] = "changed"
    assert original == before
    for empty in (None, {}, {"metadata": {}}):
        assert build_qdrant_filter(empty) is None


@pytest.fixture
def filtered_store(retrieval_config, sample_chunks):
    # Different sources and formats, missing author metadata, and unpaged content.
    chunks = [sample_chunks[0], replace(sample_chunks[1], filename="other.docx",
        source_path="/docs/other.docx", format="docx", metadata={"title": "Other"},
        locations=[ChunkLocation(0, None, None, 0, len(sample_chunks[1].text))])]
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(chunks, [[1., 0., 0.], [0., 1., 0.]])
        yield store, chunks


@pytest.mark.parametrize("filters,index", [({"filename": "example.pdf"}, 0),
    ({"filename": "other.docx"}, 1), ({"format": "PDF"}, 0), ({"format": "docx"}, 1),
    ({"source_path": "/docs/other.docx"}, 1), ({"page_number": 2}, 0),
    ({"page_label": "ii"}, 0), ({"metadata": {"author": "Tester"}}, 0),
    ({"metadata": {"title": "Other"}}, 1),
    ({"filename": "example.pdf", "format": "pdf", "source_path": "/docs/example.pdf",
      "page_number": 2, "page_label": "ii", "metadata": {"author": "Tester"}}, 0)])
def test_individual_combined_and_metadata_preservation(filtered_store, filters, index):
    store, chunks = filtered_store
    results = store.search([1., 0., 0.], limit=1, filters=filters)
    assert len(results) == 1
    assert results[0].chunk == chunks[index]
    if index == 0:
        assert [location.page_number for location in results[0].chunk.locations] == [1, 2]


@pytest.mark.parametrize("filters", [{"page_number": 1, "page_label": "ii"},
    {"filename": "other.docx", "page_number": 1}, {"metadata": {"department": "AI"}},
    {"metadata": {"author": "Tester", "title": "Other"}}, {"filename": "EXAMPLE.PDF"},
    {"filename": "missing.pdf"}, {"format": "txt"}])
def test_missing_or_conflicting_metadata_returns_empty(filtered_store, filters):
    store, _ = filtered_store
    assert store.search([1., 0., 0.], filters=filters) == []


def test_no_filters_and_reopen(filtered_store, retrieval_config):
    store, _ = filtered_store
    baseline = store.search([1., 0., 0.])
    assert store.search([1., 0., 0.], filters=None) == baseline
    assert store.search([1., 0., 0.], filters={}) == baseline
    expected = store.search([1., 0., 0.], filters={"page_number": 2})
    store.close()
    with LocalVectorStore(retrieval_config) as reopened:
        assert reopened.search([1., 0., 0.], filters={"page_number": 2}) == expected


@pytest.mark.parametrize("format", ["pdf", "docx", "txt", "md", "markdown"])
def test_all_stored_formats(retrieval_config, sample_chunks, format):
    chunk = replace(sample_chunks[0], format=format)
    with LocalVectorStore(retrieval_config) as store:
        store.upsert([chunk], [[1., 0., 0.]])
        assert store.search([1., 0., 0.], filters={"format": format.upper()})[0].chunk == chunk
