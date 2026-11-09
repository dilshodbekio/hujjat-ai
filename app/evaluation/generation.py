"""Timed adapters and transparent diagnostics; no automated entailment judge."""

from dataclasses import asdict
import re

from app.generation.context import validate_and_render
from app.generation.models import ModelAnswer
from .dataset import source_matches


class TimedRetrieval:
    def __init__(self, store, embedder, clock):
        self.store, self.embedder, self.clock = store, embedder, clock
        self.elapsed = None
        self.hits = []

    def hybrid_search(self, question, **kwargs):
        start = self.clock()
        try:
            self.hits = self.store.hybrid_search(self.embedder.embed_query(question), question, **kwargs)
            return self.hits
        finally:
            self.elapsed = self.clock() - start


class TimedGenerator:
    def __init__(self, generator, clock):
        self.generator, self.clock = generator, clock
        self.elapsed = None
        self.context = None

    def generate(self, question, context):
        self.context = context
        start = self.clock()
        try:
            return self.generator.generate(question, context)
        finally:
            self.elapsed = self.clock() - start


def generation_checks(question, result, context) -> dict:
    def normalize(text):
        return re.sub(r"\s+", " ", text).strip().casefold()
    prose = normalize(" ".join(claim.text for claim in result.claims))
    checks = {"status_correct": (result.status == "answered") == question.answerable,
              "fact_presence": [{"fact": fact, "present": normalize(fact) in prose}
                                for fact in question.expected_facts],
              "evidence_integrity": None, "citation_integrity": None,
              "expected_source_coverage": None,
              "human": {"correctness": "not_reviewed", "faithfulness": "not_reviewed",
                        "citation_support": "not_reviewed", "notes": ""}}
    if result.status == "answered" and context is not None:
        try:
            rebuilt = validate_and_render(ModelAnswer(status="answered", claims=result.claims),
                                           context, question.filters)
            checks["evidence_integrity"] = True
            checks["citation_integrity"] = (asdict(rebuilt) == asdict(result))
        except Exception:
            checks["evidence_integrity"] = checks["citation_integrity"] = False
        checks["expected_source_coverage"] = [
            {"source": source.model_dump(), "cited": any(
                source_matches(source, {**asdict(citation.chunk), "locations":
                    [asdict(citation.primary_location)] if citation.primary_location else []}, question.filters)
                for citation in result.citations)} for source in question.supporting_sources]
    return checks
