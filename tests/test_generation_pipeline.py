from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.generation.config import GenerationConfig
from app.generation.models import AnswerValidationError, ContextLimitError, ModelAnswer
from app.generation.pipeline import GenerationPipeline
from app.retrieval.embeddings import GeminiEmbedder
from app.retrieval.pipeline import RetrievalPipeline


def generated_answer(source="S1", quote="apple fruit"):
    return ModelAnswer.model_validate({"status": "answered", "claims": [
        {"text": "It contains fruit.", "evidence": [{"source_id": source, "quote": quote}]}]})


def test_real_hybrid_integration_and_page_filter(retrieval_config, sample_chunks, gemini_client):
    generator = Mock()
    generator.generate.return_value = generated_answer()
    with RetrievalPipeline(retrieval_config,
            embedder=GeminiEmbedder(retrieval_config, client=gemini_client)) as retrieval:
        retrieval.index_chunks(sample_chunks)
        baseline = retrieval.search("apple")
        with GenerationPipeline(GenerationConfig(), retrieval=retrieval, generator=generator) as rag:
            result = rag.answer("What fruit?", filters={"page_number": 2})
            assert result.status == "answered"
            assert result.citations[0].label == "example.pdf, page 2"
            assert len(result.citations[0].chunk.locations) == 2
            generator.generate.reset_mock()
            assert rag.answer("question", filters={"filename": "missing.pdf"}).status == "insufficient_evidence"
            generator.generate.assert_not_called()
        assert retrieval.search("apple") == baseline  # borrowed resources remain usable
        generator.close.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"question": ""}, {"question": "x" * 4001},
    {"question": "q", "filters": {"unknown": 1}}, {"question": "q", "retrieval_limit": 0},
    {"question": "q", "candidate_limit": 1}])
def test_invalid_input_before_retrieval(kwargs):
    retrieval, generator = Mock(), Mock()
    rag = GenerationPipeline(GenerationConfig(), retrieval=retrieval, generator=generator)
    with pytest.raises(ValueError):
        rag.answer(**kwargs)
    retrieval.hybrid_search.assert_not_called()
    generator.generate.assert_not_called()


def test_budget_failure_skips_generation(sample_chunks):
    retrieval, generator = Mock(), Mock()
    retrieval.hybrid_search.return_value = [SimpleNamespace(chunk=sample_chunks[0])]
    rag = GenerationPipeline(GenerationConfig(context_chars=1), retrieval=retrieval, generator=generator)
    with pytest.raises(ContextLimitError):
        rag.answer("question")
    generator.generate.assert_not_called()


def test_retrieval_error_propagates():
    retrieval, generator = Mock(), Mock()
    retrieval.hybrid_search.side_effect = RuntimeError("storage failed")
    rag = GenerationPipeline(GenerationConfig(), retrieval=retrieval, generator=generator)
    with pytest.raises(RuntimeError, match="storage failed"):
        rag.answer("question")
    generator.generate.assert_not_called()
