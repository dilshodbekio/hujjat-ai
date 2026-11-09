"""Hybrid retrieval followed by bounded, evidence-validated generation."""

from app.retrieval.config import load_config
from app.retrieval.filters import validate_filters
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import validate_hybrid_request
from .config import GenerationConfig, load_generation_config
from .context import build_context, validate_and_render
from .gemini import GeminiAnswerGenerator
from .models import AnswerResult, INSUFFICIENT_ANSWER


class GenerationPipeline:
    def __init__(self, config: GenerationConfig | None = None, *, retrieval=None, generator=None):
        self.config = config if config is not None else load_generation_config()
        self._owns_retrieval = retrieval is None
        self._owns_generator = generator is None
        retrieval_config = load_config() if retrieval is None or generator is None else None
        self.retrieval = retrieval if retrieval is not None else RetrievalPipeline(retrieval_config)
        try:
            self.generator = generator if generator is not None else GeminiAnswerGenerator(
                retrieval_config.api_key, self.config)
        except Exception:
            if self._owns_retrieval:
                self.retrieval.close()
            raise

    def answer(self, question: str, *, filters: dict | None = None,
               retrieval_limit: int = 5, candidate_limit: int = 20) -> AnswerResult:
        validate_hybrid_request(question, retrieval_limit, candidate_limit)
        if len(question) > self.config.question_chars:
            raise ValueError("question exceeds question_chars")
        validated = validate_filters(filters)
        hits = self.retrieval.hybrid_search(question, limit=retrieval_limit,
                                           candidate_limit=candidate_limit, filters=validated)
        context = build_context(hits, self.config.context_chars)
        if not context.sources:
            return AnswerResult("insufficient_evidence", INSUFFICIENT_ANSWER,
                                warnings=list(context.warnings))
        response = self.generator.generate(question, context)
        return validate_and_render(response, context, validated)

    def close(self):
        try:
            if self._owns_retrieval:
                self.retrieval.close()
        finally:
            if self._owns_generator:
                self.generator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
