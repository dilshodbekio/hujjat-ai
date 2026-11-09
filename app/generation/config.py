"""Generation settings, independent of the embedding model."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "gemini-3.8-flash"
    context_chars: int = 12000
    question_chars: int = 4000
    max_output_tokens: int = 2048

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip() or "embedding" in self.model.lower():
            raise ValueError("Generation requires a non-empty text-generation model, not an embedding model")
        for name in ("context_chars", "question_chars", "max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def load_generation_config(env_file: str | Path | None = None) -> GenerationConfig:
    path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[2] / ".env"
    values = {**dotenv_values(path), **os.environ}
    return GenerationConfig(model=values.get("GEMINI_GENERATION_MODEL", "gemini-3.8-flash"))
