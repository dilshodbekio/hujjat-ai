"""Strict model response schema and application-owned answer provenance."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.ingestion.models import ChunkLocation, DocumentChunk


class GenerationError(RuntimeError):
    """Generation failed; messages contain no raw provider response or key."""


class AnswerValidationError(GenerationError):
    """The entire response was rejected because its structure/evidence is invalid."""


class ContextLimitError(GenerationError):
    """No complete usable source fits the context budget."""


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str
    quote: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    evidence: list[Evidence]


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_evidence"]
    claims: list[Claim]


@dataclass
class GenerationContext:
    serialized: str
    sources: dict[str, DocumentChunk]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Citation:
    source_id: str
    chunk: DocumentChunk
    primary_location: ChunkLocation | None
    label: str


@dataclass
class AnswerResult:
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


INSUFFICIENT_ANSWER = "The answer cannot be determined from the available documents."
