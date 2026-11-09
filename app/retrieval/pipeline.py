"""Chunk indexing and semantic retrieval orchestration."""

from .config import RetrievalConfig, load_config
from .embeddings import GeminiEmbedder
from .filters import validate_filters
from .vector_store import LocalVectorStore, SearchResult, validate_hybrid_request
from app.ingestion.models import DocumentChunk


class RetrievalPipeline:
    def __init__(self, config: RetrievalConfig | None = None, *, embedder=None):
        config = config if config is not None else load_config()
        if embedder is not None and (embedder.model != config.model or embedder.dimensions != config.dimensions):
            raise ValueError("Embedder must use the configured model and dimensions.")
        self.embedder = embedder if embedder is not None else GeminiEmbedder(config)
        try:
            self.store = LocalVectorStore(config)
        except Exception:
            self.embedder.close()
            raise

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        # Embedding failures leave the existing points untouched.
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        return self.store.upsert(chunks, vectors)

    def search(self, question: str, *, limit: int = 5, filters: dict | None = None) -> list[SearchResult]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must contain non-empty text")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        validated = validate_filters(filters)
        return self.store.search(self.embedder.embed_query(question), limit=limit, filters=validated)

    def close(self):
        try:
            self.store.close()
        finally:
            self.embedder.close()

    def hybrid_search(self, question: str, *, limit: int = 5, candidate_limit: int = 20,
                      filters: dict | None = None) -> list[SearchResult]:
        """Combine Gemini semantics and local BM25; result scores are native RRF."""
        validate_hybrid_request(question, limit, candidate_limit)
        validated = validate_filters(filters)
        return self.store.hybrid_search(self.embedder.embed_query(question), question,
            limit=limit, candidate_limit=candidate_limit, filters=validated)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
