from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import errors

from app.retrieval.config import ConfigurationError, load_config
from app.retrieval.embeddings import AuthenticationError, EmbeddingError, GeminiEmbedder


@pytest.fixture
def clean_environment(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_EMBEDDING_MODEL",
                 "GEMINI_EMBEDDING_DIMENSIONS", "QDRANT_PATH", "QDRANT_COLLECTION"):
        monkeypatch.delenv(name, raising=False)


def test_load_env_and_resolve_persistent_path(tmp_path, monkeypatch, clean_environment):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=file-key\nQDRANT_PATH=./vectors\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(env)
    assert config.api_key == "file-key"
    assert config.qdrant_path == tmp_path / "vectors"
    assert "file-key" not in repr(config)
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    assert load_config(env).api_key == "environment-key"


def test_missing_and_legacy_key(tmp_path, clean_environment):
    env = tmp_path / ".env"
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        load_config(env)
    env.write_text("GOOGLE_API_KEY=legacy-key\n", encoding="utf-8")
    assert load_config(env).api_key == "legacy-key"


@pytest.mark.parametrize("entry", ["GEMINI_EMBEDDING_DIMENSIONS=bad", "GEMINI_EMBEDDING_DIMENSIONS=0",
    "QDRANT_PATH=:memory:", "QDRANT_PATH=", "GEMINI_EMBEDDING_MODEL=", "QDRANT_COLLECTION="])
def test_invalid_env(tmp_path, clean_environment, entry):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=test-key\n" + entry, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(env)


def test_task_types_and_requested_dimension(retrieval_config, gemini_client):
    embedder = GeminiEmbedder(retrieval_config, client=gemini_client)
    assert embedder.embed_documents(["apple", "code"]) == [[1., 0., 0.], [0., 1., 0.]]
    assert embedder.embed_query("apple?") == [1., 0., 0.]
    calls = gemini_client.models.embed_content.call_args_list
    assert [call.kwargs["config"].task_type for call in calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert all(call.kwargs["model"] == retrieval_config.model for call in calls)
    assert all(call.kwargs["config"].output_dimensionality == 3 for call in calls)


@pytest.mark.parametrize("code,message,expected", [(400, "API key not valid secret-key", AuthenticationError),
    (401, "secret-key", AuthenticationError), (403, "secret-key", AuthenticationError),
    (429, "quota secret-key", EmbeddingError), (503, "secret-key", EmbeddingError)])
def test_api_errors_are_safe(retrieval_config, code, message, expected):
    client = Mock()
    client.models.embed_content.side_effect = errors.APIError(code, {"error": {"message": message}})
    with pytest.raises(expected) as error:
        GeminiEmbedder(retrieval_config, client=client).embed_query("question")
    assert "secret-key" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("vectors", [[], [[1., 2.]], [[0., 0., 0.]], [[float("nan"), 1., 0.]],
    [[float("inf"), 1., 0.]], [["bad", 1., 0.]]])
def test_invalid_vectors_rejected(retrieval_config, vectors):
    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors])
    with pytest.raises(EmbeddingError):
        GeminiEmbedder(retrieval_config, client=client).embed_query("question")


def test_batches_and_empty_input(retrieval_config, gemini_client):
    embedder = GeminiEmbedder(retrieval_config, client=gemini_client)
    assert embedder.embed_documents([]) == []
    assert len(embedder.embed_documents(["apple"] * 201)) == 201
    assert [len(call.kwargs["contents"]) for call in gemini_client.models.embed_content.call_args_list] == [100, 100, 1]
    with pytest.raises(EmbeddingError):
        embedder.embed_query(" \n")


def test_network_error_and_malformed_response(retrieval_config):
    client = Mock()
    client.models.embed_content.side_effect = OSError("network secret-key")
    embedder = GeminiEmbedder(retrieval_config, client=client)
    with pytest.raises(EmbeddingError, match="connectivity") as error:
        embedder.embed_query("question")
    assert "secret-key" not in str(error.value)
    client.models.embed_content.side_effect = None
    client.models.embed_content.return_value = SimpleNamespace(embeddings=None)
    with pytest.raises(EmbeddingError, match="invalid embedding response"):
        embedder.embed_query("question")
