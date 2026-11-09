"""Corpus manifests, label validation, and isolated disk snapshots."""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import portalocker

from app.retrieval.filters import build_qdrant_filter
from app.retrieval.vector_store import LocalVectorStore
from .models import EvaluationDataset


def load_dataset(path: str | Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def corpus_manifest(store: LocalVectorStore) -> dict:
    points = []
    offset = None
    while True:
        batch, offset = store.client.scroll(store.config.collection, limit=256,
            offset=offset, with_payload=True, with_vectors=True)
        points.extend({"point_id": str(point.id), "payload": point.payload, "vectors": point.vector}
                      for point in batch)
        if offset is None:
            break
    points.sort(key=lambda point: point["point_id"])
    canonical = json.dumps(points, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return {"corpus_sha256": sha256(canonical.encode()).hexdigest(), "point_count": len(points),
            "embedding_model": store.config.model, "dimensions": store.config.dimensions,
            "collection": store.config.collection,
            "chunks": [point["payload"] for point in points],
            "source_document_hashes": "unknown: not stored by ingestion",
            "chunking_settings": "unknown: not stored by ingestion"}


def source_matches(source, chunk: dict, filters: dict | None = None) -> bool:
    if source.filename != chunk.get("filename"):
        return False
    if source.source_path is not None and source.source_path != chunk.get("source_path"):
        return False
    constraints = {key: value for key, value in (filters or {}).items()
                   if key in {"page_number", "page_label"}}
    for key in ("page_number", "page_label"):
        value = getattr(source, key)
        if value is not None:
            if key in constraints and constraints[key] != value:
                return False
            constraints[key] = value
    if not constraints:
        return True
    return any(all(location.get(key) == value for key, value in constraints.items())
               for location in chunk.get("locations", []))


def validate_corpus(dataset: EvaluationDataset, store: LocalVectorStore, manifest: dict):
    if dataset.corpus_sha256 != manifest["corpus_sha256"]:
        raise ValueError("Dataset corpus hash differs from stored payloads/vectors; verify labels before running")
    chunks = {chunk["chunk_id"]: chunk for chunk in manifest["chunks"]}
    if len(chunks) != len(manifest["chunks"]):
        raise ValueError("Corpus contains duplicate chunk IDs; relevance identity is ambiguous")
    for question in dataset.questions:
        relevant = question.relevant_chunk_ids
        if any(chunk_id not in chunks for chunk_id in relevant):
            raise ValueError(f"Unknown relevant chunk ID in {question.id}")
        if relevant:
            eligible = set()
            offset = None
            while True:
                batch, offset = store.client.scroll(store.config.collection, offset=offset, limit=256,
                    scroll_filter=build_qdrant_filter(question.filters), with_payload=True)
                eligible.update(point.payload["chunk_id"] for point in batch)
                if offset is None:
                    break
            if not set(relevant) <= eligible:
                raise ValueError(f"Relevant labels conflict with filters in {question.id}")
            for source in question.supporting_sources:
                if not any(source_matches(source, chunks[item], question.filters) for item in relevant):
                    raise ValueError(f"Supporting source does not match relevant chunks in {question.id}")


@contextmanager
def isolated_store(config, source_path: str | Path):
    """Lock existing files read-only, copy while quiescent, open ONLY the copy.

    Qdrant's lock must already exist. No source files are created or changed.
    All other clients must be closed before snapshotting (same as normal local mode).
    """
    source = Path(source_path).resolve()
    if not source.is_dir() or not (source / "meta.json").is_file():
        raise ValueError("Source must be an existing persistent Qdrant directory")
    if any(path.is_symlink() or path.is_junction() for path in [source, *source.rglob("*")]):
        raise ValueError("Snapshot source must not contain symbolic links or junctions")
    with (source / ".lock").open("rb") as lock:
        try:
            portalocker.lock(lock, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            raise ValueError("Close other Qdrant clients before making the evaluation copy") from None
        try:
            with TemporaryDirectory(prefix="hujjat-evaluation-") as directory:
                target = Path(directory) / "database"
                shutil.copytree(source, target, ignore=shutil.ignore_patterns(".lock"))
                local = replace(config, qdrant_path=target)
                # Reject absent collections instead of allowing LocalVectorStore to create one.
                metadata = json.loads((target / "meta.json").read_text(encoding="utf-8"))
                if local.collection not in metadata.get("collections", {}):
                    raise ValueError("Requested collection does not exist in the snapshot")
                with LocalVectorStore(local) as store:
                    store.evaluation_isolated = True
                    yield store
        finally:
            portalocker.unlock(lock)
