"""Gemini embeddings with asymmetric retrieval task types."""

import math

from google import genai
from google.genai import errors, types

from .config import RetrievalConfig


class EmbeddingError(RuntimeError):
    """Embedding request or response failed; messages never include API keys."""


class AuthenticationError(EmbeddingError):
    """Gemini rejected the credentials or their access permissions."""


def validate_vectors(vectors, count: int, dimensions: int):
    if len(vectors) != count:
        raise EmbeddingError("Embedding response count does not match input count.")
    for vector in vectors:
        if (len(vector) != dimensions or
                any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in vector) or
                not any(value != 0 for value in vector)):
            raise EmbeddingError("Embedding vector has an invalid dimension or non-finite/zero values.")


class GeminiEmbedder:
    def __init__(self, config: RetrievalConfig, *, client=None):
        self.model = config.model
        self.dimensions = config.dimensions
        self.client = client if client is not None else genai.Client(
            api_key=config.api_key, http_options=types.HttpOptions(timeout=60000))

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must contain non-empty text.")
        vectors = []
        # Bounded requests; collect ALL successful batches before any database write.
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            try:
                response = self.client.models.embed_content(
                    model=self.model, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type,
                                                   output_dimensionality=self.dimensions))
            except errors.APIError as exc:
                if exc.code in (401, 403) or (exc.code == 400 and
                        ("API_KEY" in str(exc).upper() or "API KEY" in str(exc).upper())):
                    raise AuthenticationError("Gemini rejected the API key or its permissions; check .env.") from None
                raise EmbeddingError("Gemini embedding request failed; check model, quota, and service availability.") from None
            except Exception:
                raise EmbeddingError("Gemini embedding request failed; check connectivity and configuration.") from None
            try:
                batch_vectors = [list(item.values) for item in response.embeddings]
                validate_vectors(batch_vectors, len(batch), self.dimensions)
            except (TypeError, AttributeError):
                raise EmbeddingError("Gemini returned an invalid embedding response.") from None
            vectors.extend(batch_vectors)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def close(self):
        self.client.close()
