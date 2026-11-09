from dataclasses import replace
import json
import subprocess
import sys

import pytest
from qdrant_client import QdrantClient, models

from app.retrieval.embeddings import EmbeddingError
from app.retrieval.vector_store import CollectionMismatchError, LocalVectorStore, StorageError


def test_persistence_metadata_ranking_and_duplicate_upsert(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        assert store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]]) == 2
        assert store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]]) == 2
        assert store.client.count(retrieval_config.collection).count == 2
        hits = store.search([1., 0., 0.], limit=1)
        assert hits[0].chunk == sample_chunks[0]
        assert hits[0].score == pytest.approx(1.)
    with LocalVectorStore(retrieval_config) as reopened:
        assert reopened.search([0., 1., 0.], limit=1)[0].chunk == sample_chunks[1]
        assert reopened.client.count(retrieval_config.collection).count == 2


def test_persistence_across_process(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
    script = """
import json, sys
from pathlib import Path
from app.retrieval.config import RetrievalConfig
from app.retrieval.vector_store import LocalVectorStore
config = RetrievalConfig('test-key', dimensions=3, qdrant_path=Path(sys.argv[1]))
with LocalVectorStore(config) as store:
    result = store.search([1., 0., 0.], 1)[0]
    print(json.dumps({'text': result.chunk.text, 'pages': [x.page_number for x in result.chunk.locations]}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(retrieval_config.qdrant_path)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"text": "apple fruit", "pages": [1, 2]}


@pytest.mark.parametrize("change", [{"model": "different-model"}, {"dimensions": 2}])
def test_collection_contract_survives_reopen(retrieval_config, change):
    with LocalVectorStore(retrieval_config):
        pass
    with pytest.raises(CollectionMismatchError):
        LocalVectorStore(replace(retrieval_config, **change))
    with LocalVectorStore(retrieval_config) as store:
        assert store.search([1., 0., 0.]) == []


def test_foreign_collection_is_not_overwritten(retrieval_config):
    client = QdrantClient(path=str(retrieval_config.qdrant_path))
    try:
        client.create_collection(retrieval_config.collection,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    finally:
        client.close()
    with pytest.raises(CollectionMismatchError):
        LocalVectorStore(retrieval_config)


def test_invalid_vectors_do_not_write(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        for vectors in ([], [[1., 0.], [1., 0.]], [[0., 0., 0.], [1., 0., 0.]]):
            with pytest.raises(EmbeddingError):
                store.upsert(sample_chunks, vectors)
        assert store.client.count(retrieval_config.collection).count == 0


def test_database_lock_and_failure_messages(retrieval_config, monkeypatch):
    with LocalVectorStore(retrieval_config) as store:
        with pytest.raises(StorageError, match="close other clients"):
            LocalVectorStore(retrieval_config)
        def fail(*args, **kwargs):
            raise RuntimeError("internal details")
        monkeypatch.setattr(store.client, "query_points", fail)
        with pytest.raises(StorageError, match="search failed"):
            store.search([1., 0., 0.])
