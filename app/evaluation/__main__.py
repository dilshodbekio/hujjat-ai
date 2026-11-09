"""CLI: explicit API opt-in, offline manifest export, isolated database copies."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from app.generation.config import load_generation_config
from app.generation.gemini import GeminiAnswerGenerator
from app.retrieval.config import RetrievalConfig, load_config
from app.retrieval.embeddings import GeminiEmbedder
from .dataset import corpus_manifest, isolated_store, load_dataset, validate_corpus
from .reporting import write_reports
from .runner import run_evaluation


def protect_output(output, database):
    output, database = Path(output).resolve(), Path(database).resolve()
    if output == database or output.is_relative_to(database) or database.is_relative_to(output):
        raise ValueError("Report output and source database paths must be disjoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated RAG evaluation; never re-embeds documents")
    parser.add_argument("command", choices=["manifest", "validate", "run"])
    parser.add_argument("--database", required=True, help="Existing CLOSED Qdrant directory; only its copy is opened")
    parser.add_argument("--dataset")
    parser.add_argument("--output", help="New report directory, or new manifest file")
    parser.add_argument("--collection", default="document_chunks")
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    parser.add_argument("--dimensions", type=int, default=3072)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--generation-limit", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Allow Gemini query embedding requests")
    parser.add_argument("--generate", action="store_true", help="Also allow one answer-generation attempt per question")
    parser.add_argument("--generation-model")
    args = parser.parse_args(argv)
    if args.command != "manifest" and not args.dataset:
        parser.error("--dataset is required")
    if args.command in {"manifest", "run"} and not args.output:
        parser.error("--output is required")
    if args.command == "run" and not args.live:
        parser.error("run requires --live to authorize query-embedding API costs; tests use mocks")
    if args.output:
        protect_output(args.output, args.database)
        if Path(args.output).exists():
            parser.error("--output must not already exist")
    config = RetrievalConfig("offline", model=args.embedding_model, dimensions=args.dimensions,
                             collection=args.collection)
    if args.command == "run":
        credentials = load_config()
        if args.output:
            protect_output(args.output, credentials.qdrant_path)
        config = replace(config, api_key=credentials.api_key)
    dataset = load_dataset(args.dataset) if args.dataset else None
    with isolated_store(config, args.database) as store:
        if args.command == "manifest":
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(corpus_manifest(store), stream, ensure_ascii=False, indent=2, allow_nan=False)
            print("Exported corpus manifest; no API calls or source database changes.")
            return
        if args.command == "validate":
            validate_corpus(dataset, store, corpus_manifest(store))
            print("Dataset schema, corpus identity, filter compatibility, and source labels validated.")
            return
        embedder = None
        generator = None
        try:
            embedder = GeminiEmbedder(config)
            generation_config = None
            if args.generate:
                generation_config = load_generation_config()
                if args.generation_model:
                    generation_config = replace(generation_config, model=args.generation_model)
                generator = GeminiAnswerGenerator(config.api_key, generation_config)
            result = run_evaluation(dataset, store, embedder, ks=args.k,
                candidate_limit=args.candidate_limit, generator=generator,
                generation_config=generation_config, generation_limit=args.generation_limit)
            write_reports(result, args.output)
            print("Evaluation reports written. Human judgments remain not_reviewed.")
        finally:
            cleanup_errors = []
            for client in (generator, embedder):
                if client is not None:
                    try:
                        client.close()
                    except Exception as error:
                        cleanup_errors.append(error)
            if cleanup_errors:
                # Both closures are attempted. Any active execution error remains
                # in __context__; multiple cleanup failures remain individually inspectable.
                raise ExceptionGroup("Evaluation client cleanup failed", cleanup_errors)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve original evaluation failure as primary, reporting cleanup errors separately.
        primary = error
        cleanup_note = ""
        if isinstance(error, ExceptionGroup):
            if error.__context__ is not None:
                primary = error.__context__
            cleanup_names = ", ".join(type(e).__name__ for e in error.exceptions)
            cleanup_note = f" (Client cleanup also failed: {cleanup_names})"
        # Errors can originate in third-party code; never echo credential-bearing messages.
        raise SystemExit(
            f"Evaluation failed: {type(primary).__name__}{cleanup_note}. "
            "Check dataset/configuration, collection schema, and that Qdrant is closed."
        ) from None
