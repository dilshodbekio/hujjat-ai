from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.models import DocumentPart, IngestedDocument
from app.retrieval.config import RetrievalConfig


@pytest.fixture
def retrieval_config(tmp_path):
    return RetrievalConfig("test-key-not-real", dimensions=3, qdrant_path=tmp_path / "qdrant")


@pytest.fixture
def sample_chunks():
    source = IngestedDocument("/docs/example.pdf", "example.pdf", "pdf", {"author": "Tester"},
        [DocumentPart("apple fruit", 1, "i"), DocumentPart("apple fruit", 2, "ii"),
         DocumentPart("python code", 3, "iii")])
    return chunk_document(source)


@pytest.fixture
def gemini_client():
    client = Mock()
    def embed_content(*, model, contents, config):
        vectors = [[1., 0., 0.] if "apple" in text else [0., 1., 0.] for text in contents]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector) for vector in vectors])
    client.models.embed_content.side_effect = embed_content
    return client
