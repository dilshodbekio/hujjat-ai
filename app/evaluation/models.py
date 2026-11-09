"""Versioned, strictly validated evaluation inputs; no built-in quality labels."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.retrieval.filters import validate_filters


class SupportingSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    filename: str = Field(min_length=1)
    source_path: str | None = None
    page_number: int | None = Field(default=None, gt=0)
    page_label: str | None = None


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1)
    filters: dict = Field(default_factory=dict)
    answerable: bool
    relevant_chunk_ids: list[str]
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    supporting_sources: list[SupportingSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_labels(self):
        self.filters = validate_filters(self.filters)
        if not self.question.strip() or any(not value.strip() for value in self.expected_facts):
            raise ValueError("Questions and expected facts must not be blank")
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("Relevant chunk IDs must be unique")
        if any(not item.strip() for item in self.relevant_chunk_ids):
            raise ValueError("Relevant chunk IDs must not be blank")
        if self.answerable and (not self.relevant_chunk_ids or not self.supporting_sources):
            raise ValueError("Answerable questions require relevant IDs and verified supporting sources")
        if not self.answerable and (self.relevant_chunk_ids or self.expected_facts or
                                    self.expected_answer or self.supporting_sources):
            raise ValueError("Unanswerable questions cannot contain positive answer labels")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    kind: Literal["manual", "synthetic"]
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    questions: list[EvaluationQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return self
