"""Evaluation on an already-isolated store; never embeds or indexes documents."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from statistics import mean
from time import perf_counter

from app.generation.pipeline import GenerationPipeline
from .dataset import corpus_manifest, validate_corpus
from .generation import TimedGenerator, TimedRetrieval, generation_checks
from .metrics import latency_summary, validate_ks
from .retrieval import compare_retrieval


def run_evaluation(dataset, store, embedder, *, ks=(1, 3, 5, 10), candidate_limit=20,
                   generator=None, generation_config=None, generation_limit=5,
                   clock=perf_counter, execution_kind="live"):
    if not getattr(store, "evaluation_isolated", False):
        raise ValueError("Evaluation requires an isolated_store snapshot, never an application store")
    ks = validate_ks(ks)
    if type(candidate_limit) is not int or candidate_limit < max(ks):
        raise ValueError("candidate_limit must be at least max(K)")
    if type(generation_limit) is not int or not 0 < generation_limit <= candidate_limit:
        raise ValueError("generation_limit must be positive and no greater than candidate_limit")
    if execution_kind not in {"live", "mocked"}:
        raise ValueError("execution_kind must be live or mocked")
    if generator is not None and generation_config is None:
        raise ValueError("Generation configuration is required when generation is enabled")
    if embedder.model != store.config.model or embedder.dimensions != store.config.dimensions:
        raise ValueError("Query embedder does not match the stored vectors")
    manifest = corpus_manifest(store)
    validate_corpus(dataset, store, manifest)
    setup_start = clock()
    store.ensure_keyword_index()
    setup_seconds = clock() - setup_start
    rows = []
    for question in dataset.questions:
        row = {"question": question.model_dump(), "retrieval": compare_retrieval(
            question, store, embedder, ks, candidate_limit, clock), "generation": None}
        if generator is not None:
            retrieval = TimedRetrieval(store, embedder, clock)
            timed_generator = TimedGenerator(generator, clock)
            pipeline = GenerationPipeline(generation_config, retrieval=retrieval, generator=timed_generator)
            start = clock()
            result = None
            error = None
            try:
                result = pipeline.answer(question.question, filters=question.filters,
                    retrieval_limit=generation_limit, candidate_limit=candidate_limit)
            except Exception as failure:
                error = type(failure).__name__
            elapsed = clock() - start
            serialized = asdict(result) if result else None
            if serialized is not None:
                serialized["claims"] = [claim.model_dump() for claim in result.claims]
            row["generation"] = {"error": error, "result": serialized,
                "checks": generation_checks(question, result, timed_generator.context) if result else None,
                "context": asdict(timed_generator.context) if timed_generator.context else None,
                "retrieval_s": retrieval.elapsed, "generation_s": timed_generator.elapsed,
                "total_s": elapsed}
        rows.append(row)
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_kind": dataset.kind, "execution_kind": execution_kind,
        "dataset_sha256": sha256(dataset.model_dump_json().encode()).hexdigest(),
        "corpus": {key: value for key, value in manifest.items() if key != "chunks"},
        "settings": {"ks": ks, "candidate_limit": candidate_limit, "generation_limit": generation_limit,
                     "generation": asdict(generation_config) if generation_config else None,
                     "qdrant_client": version("qdrant-client"), "google_genai": version("google-genai"),
                     "setup_s": setup_seconds, "method_order": ["semantic", "bm25", "hybrid"],
                     "query_vectors": "one shared vector per question for comparison; fresh query for generation"},
        "questions": rows}
    result["summary"] = summarize(rows, ks)
    return result


def summarize(rows, ks):
    summary = {"retrieval": {}, "generation": {}}
    for method in ("semantic", "bm25", "hybrid"):
        entries = [row["retrieval"]["methods"][method] for row in rows]
        labeled = [row["retrieval"]["methods"][method] for row in rows if row["question"]["answerable"]]
        successful = [item for item in labeled if item["error"] is None]
        metrics = {}
        for k in ks:
            metrics[str(k)] = {"answerable_count": len(labeled), "successful_answerable_count": len(successful)}
            for name, key in (("recall", "recall"), ("mrr", "rr")):
                metrics[str(k)][name + "_failure_inclusive"] = mean(
                    0. if item["error"] else item["metrics"][str(k)][key] for item in labeled) if labeled else None
                metrics[str(k)][name + "_successful"] = mean(
                    item["metrics"][str(k)][key] for item in successful) if successful else None
        summary["retrieval"][method] = {"metrics": metrics, "query_count": len(entries),
            "error_count": sum(item["error"] is not None for item in entries),
            "error_rate": sum(item["error"] is not None for item in entries) / len(entries),
            "successful_inclusive_latency": latency_summary([item["inclusive_s"] for item in entries if not item["error"]]),
            "successful_database_latency": latency_summary([item["database_s"] for item in entries if not item["error"]]),
            "failed_elapsed": latency_summary([item["inclusive_s"] for item in entries if item["error"]])}
    generated = [row for row in rows if row["generation"] is not None]
    summary["generation"] = {"attempted": len(generated),
        "errors": sum(row["generation"]["error"] is not None for row in generated),
        "status_correct_count": sum(bool(row["generation"]["checks"] and row["generation"]["checks"]["status_correct"]) for row in generated),
        "human_reviewed_count": 0,
        "abstention": {"expected_unanswerable": sum(not row["question"]["answerable"] for row in generated),
            "correct_abstentions": sum(not row["question"]["answerable"] and bool(row["generation"]["checks"] and row["generation"]["checks"]["status_correct"]) for row in generated),
            "incorrect_abstentions": sum(row["question"]["answerable"] and bool(row["generation"]["result"] and row["generation"]["result"]["status"] == "insufficient_evidence") for row in generated)},
        "latency": {stage: latency_summary([row["generation"][stage] for row in generated
            if row["generation"]["error"] is None and row["generation"][stage] is not None])
            for stage in ("retrieval_s", "generation_s", "total_s")},
        "failed_total_latency": latency_summary([row["generation"]["total_s"] for row in generated if row["generation"]["error"]])}
    return summary
