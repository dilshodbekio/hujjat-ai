from copy import deepcopy
from hashlib import sha256
from itertools import count
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.evaluation.__main__ import main, protect_output
from app.evaluation.dataset import corpus_manifest, isolated_store, validate_corpus
from app.evaluation.models import EvaluationDataset, SupportingSource
from app.evaluation.reporting import write_reports
from app.evaluation.runner import run_evaluation
from app.generation.config import GenerationConfig
from app.generation.models import ModelAnswer
from app.retrieval.vector_store import LocalVectorStore


@pytest.fixture
def source_database(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        store.upsert(sample_chunks, [[1., 0., 0.], [0., 1., 0.]])
    return retrieval_config


def files_hashes(path):
    return {str(file.relative_to(path)): sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def labeled_dataset(store, sample_chunks):
    return EvaluationDataset.model_validate({"version": 1, "kind": "synthetic",
        "corpus_sha256": corpus_manifest(store)["corpus_sha256"], "questions": [
            {"id": "q1", "question": "apple?", "answerable": True,
             "relevant_chunk_ids": [sample_chunks[0].chunk_id],
             "expected_answer": "It contains apple fruit.", "expected_facts": ["apple fruit"],
             "supporting_sources": [{"filename": "example.pdf", "page_number": 1}]},
            {"id": "q2", "question": "What is the missing fee?", "answerable": False,
             "relevant_chunk_ids": [], "filters": {"filename": "missing.pdf"}}]})


def fake_embedder(config):
    embedder = Mock()
    embedder.model = config.model
    embedder.dimensions = config.dimensions
    embedder.embed_query.return_value = [1., 0., 0.]
    return embedder


def test_run_reports_and_original_database_unchanged(source_database, sample_chunks, tmp_path):
    config = source_database
    before = files_hashes(config.qdrant_path)
    with isolated_store(config, config.qdrant_path) as store:
        dataset = labeled_dataset(store, sample_chunks)
        embedder = fake_embedder(config)
        generator = Mock()
        generator.generate.return_value = ModelAnswer.model_validate({"status": "answered", "claims": [
            {"text": "It contains apple fruit.", "evidence": [{"source_id": "S1", "quote": "apple fruit"}]}]})
        ticks = count()
        result = run_evaluation(dataset, store, embedder, ks=[1, 3, 5, 10],
            generator=generator, generation_config=GenerationConfig(), clock=lambda: next(ticks),
                execution_kind="mocked")
        assert result["dataset_kind"] == "synthetic" and result["execution_kind"] == "mocked"
        assert result["summary"]["retrieval"]["semantic"]["metrics"]["1"]["recall_failure_inclusive"] == 1
        assert result["summary"]["generation"]["status_correct_count"] == 2
        assert result["summary"]["generation"]["abstention"]["correct_abstentions"] == 1
        assert result["questions"][0]["generation"]["checks"]["citation_integrity"]
        assert result["questions"][0]["generation"]["checks"]["fact_presence"][0]["present"]
        assert result["summary"]["generation"]["human_reviewed_count"] == 0
        embedder.embed_documents.assert_not_called()
        assert embedder.embed_query.call_count == 4  # 2 shared comparison vectors + 2 full E2E queries
        assert generator.generate.call_count == 1  # empty retrieval skips generation
        output = tmp_path / "reports"
        write_reports(result, output)
        text = (output / "report.md").read_text(encoding="utf-8")
        for value in ("apple?", "apple fruit", "It contains apple fruit.", "example.pdf", "not_reviewed", "mocked", "synthetic"):
            assert value in text
        assert json.loads((output / "results.json").read_text(encoding="utf-8"))["dataset_sha256"] == result["dataset_sha256"]
        with pytest.raises(FileExistsError):
            write_reports(result, output)
    assert files_hashes(config.qdrant_path) == before
    with LocalVectorStore(config) as store:
        assert store.client.count(config.collection).count == 2
        assert store.search([1., 0., 0.])[0].chunk == sample_chunks[0]


def test_embedding_failure_does_not_hide_bm25_or_inflate_scores(source_database, sample_chunks):
    with isolated_store(source_database, source_database.qdrant_path) as store:
        dataset = labeled_dataset(store, sample_chunks)
        embedder = fake_embedder(source_database)
        embedder.embed_query.side_effect = RuntimeError("fake-secret-key")
        result = run_evaluation(dataset, store, embedder, ks=[1], execution_kind="mocked")
        assert "fake-secret-key" not in json.dumps(result)
        for method in ("semantic", "hybrid"):
            metrics = result["summary"]["retrieval"][method]["metrics"]["1"]
            assert metrics["recall_failure_inclusive"] == 0
            assert metrics["recall_successful"] is None
        assert result["summary"]["retrieval"]["bm25"]["error_count"] == 0
        assert result["summary"]["retrieval"]["bm25"]["metrics"]["1"]["recall_failure_inclusive"] == 1


def test_generation_failure_recorded_not_abstention(source_database, sample_chunks):
    with isolated_store(source_database, source_database.qdrant_path) as store:
        generator = Mock()
        generator.generate.side_effect = RuntimeError("secret")
        result = run_evaluation(labeled_dataset(store, sample_chunks), store, fake_embedder(source_database),
            generator=generator, generation_config=GenerationConfig(), execution_kind="mocked")
        assert result["summary"]["generation"]["errors"] == 1
        assert result["summary"]["generation"]["failed_total_latency"]["count"] == 1
        assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("change", ["hash", "unknown", "filter", "source", "page"])
def test_corpus_label_validation(source_database, sample_chunks, change):
    with isolated_store(source_database, source_database.qdrant_path) as store:
        dataset = labeled_dataset(store, sample_chunks)
        if change == "hash":
            dataset.corpus_sha256 = "0" * 64
        elif change == "unknown":
            dataset.questions[0].relevant_chunk_ids = ["unknown"]
        elif change == "filter":
            dataset.questions[0].filters = {"page_number": 3}
        elif change == "source":
            dataset.questions[0].supporting_sources[0].filename = "fake.pdf"
        else:
            dataset.questions[0].filters = {"page_number": 2}
        with pytest.raises(ValueError):
            validate_corpus(dataset, store, corpus_manifest(store))


@pytest.mark.parametrize("change", ["duplicate", "unanswerable", "empty", "blank", "bad_filter"])
def test_dataset_schema_validation(source_database, sample_chunks, change):
    with isolated_store(source_database, source_database.qdrant_path) as store:
        data = labeled_dataset(store, sample_chunks).model_dump()
    if change == "duplicate":
        data["questions"].append(deepcopy(data["questions"][0]))
    elif change == "unanswerable":
        data["questions"][0]["answerable"] = False
    elif change == "empty":
        data["questions"][0]["relevant_chunk_ids"] = []
    elif change == "blank":
        data["questions"][0]["question"] = " "
    else:
        data["questions"][0]["filters"] = {"unsupported": "x"}
    with pytest.raises((ValidationError, ValueError)):
        EvaluationDataset.model_validate(data)


def test_locked_source_and_no_original_runner(source_database):
    with LocalVectorStore(source_database) as store:
        with pytest.raises(ValueError, match="Close other"):
            with isolated_store(source_database, source_database.qdrant_path):
                pass
        with pytest.raises(ValueError, match="isolated"):
            run_evaluation(None, store, None)


@pytest.mark.parametrize("filters,source,valid", [
    ({"page_number": 2}, {"page_label": "i"}, False),
    ({"page_label": "ii"}, {"page_number": 1}, False),
    ({"page_number": 2}, {"page_label": "ii"}, True),
    ({"page_label": "ii"}, {"page_number": 2}, True),
])
def test_expected_source_and_filters_share_one_location(source_database, sample_chunks, filters, source, valid):
    with isolated_store(source_database, source_database.qdrant_path) as store:
        dataset = labeled_dataset(store, sample_chunks)
        dataset.questions[0].filters = filters
        dataset.questions[0].supporting_sources = [SupportingSource(filename="example.pdf", **source)]
        if valid:
            validate_corpus(dataset, store, corpus_manifest(store))
        else:
            with pytest.raises(ValueError, match="Supporting source"):
                validate_corpus(dataset, store, corpus_manifest(store))


def test_cli_manifest_validation_and_output_safety(source_database, sample_chunks, tmp_path):
    output = tmp_path / "manifest.json"
    common = ["--database", str(source_database.qdrant_path), "--dimensions", "3"]
    before = files_hashes(source_database.qdrant_path)
    main(["manifest", *common, "--output", str(output)])
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["point_count"] == 2
    with isolated_store(source_database, source_database.qdrant_path) as store:
        dataset = labeled_dataset(store, sample_chunks)
    labels = tmp_path / "labels.json"
    labels.write_text(dataset.model_dump_json(), encoding="utf-8")
    main(["validate", *common, "--dataset", str(labels)])
    assert files_hashes(source_database.qdrant_path) == before
    with pytest.raises(SystemExit):
        main(["run", *common, "--dataset", str(labels), "--output", str(tmp_path / "run")])
    with pytest.raises(ValueError):
        protect_output(source_database.qdrant_path / "report", source_database.qdrant_path)
