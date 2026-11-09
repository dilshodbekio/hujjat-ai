"""Persistent local Qdrant with model-specific named vectors."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.ingestion.models import ChunkLocation, DocumentChunk
from .config import RetrievalConfig
from .embeddings import validate_vectors
from .filters import build_qdrant_filter


class StorageError(RuntimeError):
    """Local database could not be opened, read, or written."""


class CollectionMismatchError(StorageError):
    """Existing vectors belong to a different embedding configuration."""


@dataclass
class SearchResult:
    """score is cosine similarity for search(), RRF ranking score for hybrid_search()."""
    chunk: DocumentChunk
    score: float


def validate_hybrid_request(question: str, limit: int, candidate_limit: int):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-empty text")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if type(candidate_limit) is not int or candidate_limit < limit:
        raise ValueError("candidate_limit must be an integer at least as large as limit")


class LocalVectorStore:
    @staticmethod
    def point_id(chunk: DocumentChunk) -> str:
        return str(uuid5(NAMESPACE_URL, f"{chunk.source_path}:{chunk.chunk_id}"))

    def __init__(self, config: RetrievalConfig):
        self.config = config
        # Persist the model/task contract in the collection's vector schema.
        signature = f"gemini:{config.model}:{config.dimensions}:RETRIEVAL_DOCUMENT:RETRIEVAL_QUERY:v1"
        self.vector_name = "gemini_" + sha256(signature.encode()).hexdigest()
        try:
            self.client = QdrantClient(path=str(config.qdrant_path))
        except Exception:
            raise StorageError("Cannot open local Qdrant directory; check permissions and close other clients using it.") from None
        try:
            if not self.client.collection_exists(config.collection):
                self.client.create_collection(config.collection, vectors_config={
                    self.vector_name: models.VectorParams(size=config.dimensions, distance=models.Distance.COSINE)})
            self._check_collection()
        except Exception as exc:
            self.client.close()
            if isinstance(exc, StorageError):
                raise
            raise StorageError("Cannot initialize the local Qdrant collection.") from None

    def _check_collection(self):
        vectors = self.client.get_collection(self.config.collection).config.params.vectors
        if (not isinstance(vectors, dict) or set(vectors) != {self.vector_name}
                or vectors[self.vector_name].size != self.config.dimensions
                or vectors[self.vector_name].distance != models.Distance.COSINE):
            raise CollectionMismatchError("Collection embedding configuration differs; use a new collection and reindex.")

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        validate_vectors(vectors, len(chunks), self.config.dimensions)
        if not chunks:
            return 0
        points = [models.PointStruct(
            id=self.point_id(chunk),
            vector={self.vector_name: vector}, payload=asdict(chunk))
            for chunk, vector in zip(chunks, vectors)]
        try:
            self._check_collection()
            self.client.upsert(self.config.collection, points=points, wait=True)
        except StorageError:
            raise
        except Exception:
            raise StorageError("Qdrant write failed; indexing can be retried with the same chunk IDs.") from None
        return len(points)

    def search(self, vector: list[float], limit: int = 5, *, filters: dict | None = None) -> list[SearchResult]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        validate_vectors([vector], 1, self.config.dimensions)
        query_filter = build_qdrant_filter(filters)
        try:
            self._check_collection()
            hits = self.client.query_points(self.config.collection, query=vector,
                using=self.vector_name, limit=limit, with_payload=True, query_filter=query_filter).points
            results = []
            for hit in hits:
                payload = dict(hit.payload or {})
                payload["locations"] = [ChunkLocation(**location) for location in payload["locations"]]
                results.append(SearchResult(DocumentChunk(**payload), float(hit.score)))
            return results
        except StorageError:
            raise
        except Exception:
            raise StorageError("Qdrant search failed or stored chunk payload is invalid.") from None

    def close(self):
        self.client.close()

    def missing_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Find absent/incomplete points without re-embedding valid existing chunks."""
        try:
            self._check_collection()
            missing = []
            for start in range(0, len(chunks), 256):
                batch = chunks[start:start + 256]
                stored = {str(point.id): point for point in self.client.retrieve(
                    self.config.collection, ids=[self.point_id(chunk) for chunk in batch],
                    with_payload=True, with_vectors=True)}
                for chunk in batch:
                    point = stored.get(self.point_id(chunk))
                    vector = point.vector.get(self.vector_name) if point and isinstance(point.vector, dict) else None
                    if vector is None:
                        missing.append(chunk)
                        continue
                    validate_vectors([vector], 1, self.config.dimensions)
                    if point.payload != asdict(chunk):
                        raise StorageError("An existing chunk has conflicting metadata; no points were overwritten.")
            return missing
        except StorageError:
            raise
        except Exception:
            raise StorageError("Cannot verify existing indexed chunks.") from None

    def list_documents(self) -> list[dict]:
        """Server-side inventory; callers must not expose source paths in UI outputs."""
        try:
            documents = {}
            offset = None
            while True:
                points, offset = self.client.scroll(self.config.collection, offset=offset,
                    limit=256, with_payload=True, with_vectors=False)
                for point in points:
                    payload = point.payload or {}
                    source = payload["source_path"]
                    if source not in documents:
                        documents[source] = {"source_path": source, "filename": payload["filename"],
                            "format": payload["format"], "chunks": 0}
                    documents[source]["chunks"] += 1
                if offset is None:
                    break
            return sorted(documents.values(), key=lambda item: (item["filename"], item["source_path"]))
        except Exception:
            raise StorageError("Cannot load indexed document inventory.") from None

    def ensure_keyword_index(self):
        """Enable local BM25 without replacing existing points or dense vectors.

        Local Qdrant does not expose text indexes in get_collection.payload_schema.
        Probe the public TextQuery API instead. Only its explicit missing-index
        error authorizes creation; incompatible existing settings are preserved.
        """
        try:
            self._check_collection()
            try:
                self.client.query_points(self.config.collection,
                    query=models.TextQuery(text=""), using="text", limit=1,
                    with_payload=False)
            except ValueError as exc:
                if str(exc) != "A text query needs a text index with `scoring` on `text`":
                    raise
                self.client.create_payload_index(self.config.collection, "text",
                    field_schema=models.TextIndexParams(type="text",
                        scoring=models.TextScoringParams(type="bm25")), wait=True)
        except StorageError:
            raise
        except Exception:
            raise StorageError("Cannot enable BM25; check Qdrant version and existing text-index compatibility.") from None

    def hybrid_search(self, vector: list[float], question: str, *, limit: int = 5,
                      candidate_limit: int = 20, filters: dict | None = None) -> list[SearchResult]:
        """Fuse filtered semantic and BM25 candidates with equal-weight native RRF."""
        validate_hybrid_request(question, limit, candidate_limit)
        validate_vectors([vector], 1, self.config.dimensions)
        query_filter = build_qdrant_filter(filters)
        self.ensure_keyword_index()
        try:
            hits = self.client.query_points(self.config.collection,
                prefetch=[
                    models.Prefetch(query=vector, using=self.vector_name,
                                    filter=query_filter, limit=candidate_limit),
                    models.Prefetch(query=models.TextQuery(text=question), using="text",
                                    filter=query_filter, limit=candidate_limit),
                ], query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter, limit=limit, with_payload=True).points
            results = []
            for hit in hits:
                payload = dict(hit.payload or {})
                payload["locations"] = [ChunkLocation(**location) for location in payload["locations"]]
                results.append(SearchResult(DocumentChunk(**payload), float(hit.score)))
            return results
        except Exception:
            raise StorageError("Qdrant hybrid search failed or stored chunk payload is invalid.") from None

    def keyword_search(self, question: str, *, limit: int = 5,
                       filters: dict | None = None) -> list[SearchResult]:
        """BM25-only search; scores are lexical relevance, not cosine similarity."""
        validate_hybrid_request(question, limit, limit)
        query_filter = build_qdrant_filter(filters)
        self.ensure_keyword_index()
        try:
            hits = self.client.query_points(self.config.collection,
                query=models.TextQuery(text=question), using="text", limit=limit,
                query_filter=query_filter, with_payload=True).points
            results = []
            for hit in hits:
                payload = dict(hit.payload or {})
                payload["locations"] = [ChunkLocation(**location) for location in payload["locations"]]
                results.append(SearchResult(DocumentChunk(**payload), float(hit.score)))
            return results
        except Exception:
            raise StorageError("Qdrant keyword search failed or stored chunk payload is invalid.") from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
