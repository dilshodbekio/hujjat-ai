from dataclasses import replace

import pytest
from qdrant_client import models

from app.retrieval.embeddings import EmbeddingError, GeminiEmbedder
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import LocalVectorStore, StorageError


def test_dense_collection_upgrade_fusion_and_reopen(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
        baseline = store.search([1., 0., 0.])
        before, _ = store.client.scroll(retrieval_config.collection, with_vectors=True)
        # Dense branch chooses apple, lexical branch chooses code: both survive fusion.
        hits = store.hybrid_search([1., 0., 0.], "code", limit=2, candidate_limit=2)
        assert hits[0].chunk == sample_chunks[1]
        assert hits[0].score == pytest.approx(1 / 2 + 1 / 3)
        assert len({hit.chunk.chunk_id for hit in hits}) == 2
        after, _ = store.client.scroll(retrieval_config.collection, with_vectors=True)
        assert before == after
        assert store.search([1., 0., 0.]) == baseline
        store.ensure_keyword_index()  # idempotent
    with LocalVectorStore(retrieval_config) as store:
        lexical = store.client.query_points(retrieval_config.collection,
            query=models.TextQuery(text="code"), using="text").points
        assert lexical[0].payload["text"] == "python code"
        assert store.hybrid_search([1., 0., 0.], "code", limit=2, candidate_limit=2) == hits
        assert store.search([1., 0., 0.]) == baseline


@pytest.mark.parametrize("filters,expected", [({"page_number": 2}, [0]),
    ({"page_number": 3}, [1]), ({"metadata": {"author": "Tester"}}, [1, 0]),
    ({"filename": "example.pdf", "page_number": 2, "page_label": "ii"}, [0]),
    ({"page_number": 1, "page_label": "ii"}, []),
    ({"metadata": {"missing": "value"}}, []), ({"format": "docx"}, []),
    ({"source_path": "/docs/example.pdf", "format": "pdf", "page_number": 3}, [1])])
def test_filters_on_both_branches(retrieval_config, sample_chunks, filters, expected):
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
        hits = store.hybrid_search([1., 0., 0.], "code", limit=2, candidate_limit=2, filters=filters)
        assert [hit.chunk for hit in hits] == [sample_chunks[index] for index in expected]


def test_filter_before_candidate_selection(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
        # Both branches' best unfiltered point is excluded; filtering only after fusion fails.
        hits = store.hybrid_search([1., 0., 0.], "apple", limit=1, candidate_limit=1,
                                  filters={"page_number": 3})
        assert hits[0].chunk == sample_chunks[1]
        hits = store.hybrid_search([0., 1., 0.], "code", limit=1, candidate_limit=1,
                                  filters={"page_number": 2})
        assert hits[0].chunk == sample_chunks[0]


def test_empty_and_no_keyword_matches(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        assert store.hybrid_search([1., 0., 0.], "apple") == []
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
        hits = store.hybrid_search([1., 0., 0.], "notincorpus", limit=2)
        assert [hit.chunk for hit in hits] == sample_chunks
        assert store.hybrid_search([1., 0., 0.], "!!!", limit=2) == hits


def test_pipeline_and_query_task(retrieval_config, gemini_client, sample_chunks):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        pipeline.index_chunks(sample_chunks)
        result = pipeline.hybrid_search("apple", filters={"page_number": 2})
        assert result[0].chunk == sample_chunks[0]
        assert gemini_client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.parametrize("kwargs", [{"question": ""}, {"question": " "},
    {"question": "code", "limit": True}, {"question": "code", "limit": 0},
    {"question": "code", "candidate_limit": 4}, {"question": "code", "candidate_limit": True},
    {"question": "code", "filters": {"unknown": "x"}}])
def test_invalid_inputs_before_gemini(retrieval_config, gemini_client, kwargs):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        with pytest.raises(ValueError):
            pipeline.hybrid_search(**kwargs)
    gemini_client.models.embed_content.assert_not_called()


def test_incompatible_index_not_replaced(retrieval_config):
    with LocalVectorStore(retrieval_config) as store:
        params = models.TextIndexParams(type="text", lowercase=False,
                                       scoring=models.TextScoringParams(type="bm25"))
        store.client.create_payload_index(retrieval_config.collection, "text", field_schema=params)
        with pytest.raises(StorageError, match="compatibility"):
            store.ensure_keyword_index()
        with pytest.raises(NotImplementedError):
            store.client.query_points(retrieval_config.collection,
                query=models.TextQuery(text="test"), using="text")


def test_new_upsert_updates_keyword_results(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        store.ensure_keyword_index()
        store.upsert([sample_chunks[0]], [[1., 0., 0.]])
        updated = replace(sample_chunks[0], text="newuniqueterm")
        store.upsert([updated], [[1., 0., 0.]])
        assert store.client.count(retrieval_config.collection).count == 1
        assert store.client.query_points(retrieval_config.collection,
            query=models.TextQuery(text="apple"), using="text").points == []
        assert store.hybrid_search([1., 0., 0.], "newuniqueterm")[0].chunk == updated


def test_failures_are_not_silent_fallbacks(retrieval_config, sample_chunks, gemini_client, monkeypatch):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        gemini_client.models.embed_content.side_effect = OSError("network")
        with pytest.raises(EmbeddingError):
            pipeline.hybrid_search("apple")
        pipeline.store.ensure_keyword_index()
        def fail(*args, **kwargs):
            raise RuntimeError("internal")
        monkeypatch.setattr(pipeline.store.client, "query_points", fail)
        with pytest.raises(StorageError):
            pipeline.store.hybrid_search([1., 0., 0.], "apple")


def test_labeled_retrieval_quality_fixture(retrieval_config, sample_chunks):
    # Synthetic dense vectors isolate fusion mechanics, not live Gemini quality.
    queries = [("code", [1., 0., 0.], sample_chunks[1].chunk_id),
               ("apple", [0., 1., 0.], sample_chunks[0].chunk_id),
               ("fruit", [1., 0., 0.], sample_chunks[0].chunk_id)]
    rankings = {"semantic": [], "keyword": [], "hybrid": []}
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
        store.ensure_keyword_index()
        for question, vector, relevant in queries:
            rankings["semantic"].append(([hit.chunk.chunk_id for hit in store.search(vector)], relevant))
            rankings["hybrid"].append(([hit.chunk.chunk_id for hit in store.hybrid_search(vector, question)], relevant))
            lexical = store.client.query_points(retrieval_config.collection,
                query=models.TextQuery(text=question), using="text", with_payload=True).points
            rankings["keyword"].append(([hit.payload["chunk_id"] for hit in lexical], relevant))
    def metrics(rows):
        recall_at_one = sum(bool(ids) and ids[0] == relevant for ids, relevant in rows) / len(rows)
        mrr = sum(1 / (ids.index(relevant) + 1) if relevant in ids else 0
                  for ids, relevant in rows) / len(rows)
        return recall_at_one, mrr
    assert metrics(rankings["semantic"]) == pytest.approx((1 / 3, 2 / 3))
    assert metrics(rankings["keyword"]) == (1., 1.)
    assert metrics(rankings["hybrid"]) == (1., 1.)
