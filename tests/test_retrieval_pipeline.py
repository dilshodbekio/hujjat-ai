from types import SimpleNamespace

import pytest

from app.retrieval.embeddings import EmbeddingError, GeminiEmbedder
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.filters import InvalidFilterError


def test_filtered_search_pipeline(retrieval_config, sample_chunks, gemini_client):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        pipeline.index_chunks(sample_chunks)
        result = pipeline.search("apple question", limit=1, filters={"page_number": 3})
        assert result[0].chunk == sample_chunks[1]
        assert gemini_client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.parametrize("filters", [{"unknown": "x"}, {"page_number": True}, {"metadata": []}])
def test_invalid_filters_do_not_call_gemini(retrieval_config, gemini_client, filters):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        with pytest.raises(InvalidFilterError):
            pipeline.search("question", filters=filters)
    gemini_client.models.embed_content.assert_not_called()


def test_full_pipeline(retrieval_config, sample_chunks, gemini_client):
    embedder = GeminiEmbedder(retrieval_config, client=gemini_client)
    with RetrievalPipeline(retrieval_config, embedder=embedder) as pipeline:
        assert pipeline.index_chunks([]) == 0
        assert pipeline.index_chunks(sample_chunks) == 2
        result = pipeline.search("apple question", limit=1)[0]
        assert result.chunk == sample_chunks[0]
        assert result.score == pytest.approx(1.)
        assert [call.kwargs["config"].task_type for call in gemini_client.models.embed_content.call_args_list] == [
            "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    gemini_client.close.assert_called_once()


def test_later_batch_failure_keeps_existing_data(retrieval_config, sample_chunks, gemini_client):
    embedder = GeminiEmbedder(retrieval_config, client=gemini_client)
    with RetrievalPipeline(retrieval_config, embedder=embedder) as pipeline:
        pipeline.index_chunks(sample_chunks)
        gemini_client.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=[1., 0., 0.]) for _ in range(100)]),
            OSError("connection failed")]
        with pytest.raises(EmbeddingError):
            pipeline.index_chunks([sample_chunks[0]] * 101)
        assert pipeline.store.client.count(retrieval_config.collection).count == 2
        assert pipeline.store.search([0., 1., 0.], limit=1)[0].chunk == sample_chunks[1]


@pytest.mark.parametrize("question,limit", [("", 5), (" \n", 5), ("apple", 0), ("apple", True)])
def test_invalid_query_makes_no_api_request(retrieval_config, gemini_client, question, limit):
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as pipeline:
        with pytest.raises(ValueError):
            pipeline.search(question, limit=limit)
    gemini_client.models.embed_content.assert_not_called()
