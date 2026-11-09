"""Explicit .env loading; paths are stable regardless of the working directory."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Missing or invalid retrieval settings."""


@dataclass(frozen=True)
class RetrievalConfig:
    api_key: str = field(repr=False)
    model: str = "gemini-embedding-001"
    dimensions: int = 3072
    qdrant_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / ".qdrant")
    collection: str = "document_chunks"

    def __post_init__(self):
        if not self.api_key.strip():
            raise ConfigurationError("Set GEMINI_API_KEY in .env (GOOGLE_API_KEY is also supported).")
        if not self.model.strip() or not self.collection.strip():
            raise ConfigurationError("Embedding model and Qdrant collection must not be empty.")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 3072:
            raise ConfigurationError("Embedding dimensions must be an integer between 1 and 3072.")
        if str(self.qdrant_path) == ":memory:":
            raise ConfigurationError("Use a disk path for persistent Qdrant storage.")


def load_config(env_file: str | Path | None = None) -> RetrievalConfig:
    path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[2] / ".env"
    path = path.resolve()
    settings = {**dotenv_values(path), **os.environ}
    try:
        dimensions = int(settings.get("GEMINI_EMBEDDING_DIMENSIONS", "3072"))
    except (ValueError, TypeError):
        raise ConfigurationError("GEMINI_EMBEDDING_DIMENSIONS must be an integer.") from None
    storage = settings.get("QDRANT_PATH", "./.qdrant")
    if not storage or not storage.strip() or storage == ":memory:":
        raise ConfigurationError("QDRANT_PATH must be a persistent disk directory.")
    storage_path = Path(storage).expanduser()
    if not storage_path.is_absolute():
        storage_path = path.parent / storage_path
    return RetrievalConfig(
        api_key=(settings.get("GEMINI_API_KEY") or settings.get("GOOGLE_API_KEY") or "").strip(),
        model=(settings.get("GEMINI_EMBEDDING_MODEL") or "").strip()
              if "GEMINI_EMBEDDING_MODEL" in settings else "gemini-embedding-001",
        dimensions=dimensions, qdrant_path=storage_path.resolve(),
        collection=(settings.get("QDRANT_COLLECTION") or "").strip()
                   if "QDRANT_COLLECTION" in settings else "document_chunks",
    )
