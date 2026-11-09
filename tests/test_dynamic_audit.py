"""Runtime audit regressions. Synthetic fixtures are not measured RAG quality."""

from dataclasses import replace
import json
from pathlib import Path
import socket
from unittest.mock import Mock
import uuid

import httpx
import pytest

from app.generation.config import GenerationConfig, load_generation_config
from app.generation.context import build_context, validate_and_render
from app.generation.models import AnswerValidationError, ModelAnswer
from app.ingestion.chunking import chunk_document
from app.ingestion.models import DocumentPart, IngestedDocument
from app.retrieval.vector_store import LocalVectorStore, SearchResult, StorageError
from app.ui.service import UIService
from test_ui_service import FakeEmbedder, FakeGenerator


def test_realistic_multi_page_policy_and_exact_pdf_quote_rules():
    source = IngestedDocument("synthetic:policy", "policy.pdf", "pdf", parts=[
        DocumentPart("Employees receive 10 paid sick days per calendar year.\nA medical certificate is required from the third consecutive working day.", 4, "4"),
        DocumentPart("From 1 June 2026, employees with 5 or more years of continuous service receive 12 paid sick days per calendar year.", 11, "11"),
        DocumentPart("Sick leave\n5 paid days for the whole internship.", 10, "10")])
    chunks = chunk_document(source)
    context = build_context([SearchResult(chunk, 1.) for chunk in chunks])
    data = {"status": "answered", "claims": [
        {"text": "Employees receive 10 paid sick days annually.", "evidence": [{
            "source_id": "S1", "quote": "Employees receive 10 paid sick days per calendar year."}]},
        {"text": "Long-service employees receive 12 paid sick days from June 2026.", "evidence": [{
            "source_id": "S2", "quote": source.parts[1].text}]},
        {"text": "Interns receive 5 paid sick days for the whole internship.", "evidence": [{
            "source_id": "S3", "quote": "Sick leave: 5 paid days for the whole internship."}]}]}
    # Original failure now resolves the narrow label/newline formatting case to
    # an actual source span; the displayed evidence never contains the invented colon.
    original = ModelAnswer.model_validate(data)
    resolved = validate_and_render(original, context)
    assert resolved.claims[2].evidence[0].quote == source.parts[2].text
    assert original.claims[2].evidence[0].quote == "Sick leave: 5 paid days for the whole internship."
    data["claims"][2]["evidence"][0]["quote"] = "5 paid days for the whole internship."
    result = validate_and_render(ModelAnswer.model_validate(data), context)
    assert [citation.primary_location.page_number for citation in result.citations] == [4, 11, 10]
    assert all(number in result.answer for number in ("10", "12", "5"))


def test_missing_vector_recovery_and_conflicting_payload(retrieval_config, sample_chunks):
    with LocalVectorStore(retrieval_config) as store:
        vectors = [[1., 0., 0.], [0., 1., 0.]]
        store.upsert(sample_chunks, vectors)
        store.client.delete_vectors(retrieval_config.collection, vectors=[store.vector_name],
                                    points=[store.point_id(sample_chunks[0])])
        assert store.missing_chunks(sample_chunks) == [sample_chunks[0]]
        store.upsert([sample_chunks[0]], vectors[:1])
        assert store.missing_chunks(sample_chunks) == []
        conflict = replace(sample_chunks[0], metadata={"author": "Conflicting author"})
        with pytest.raises(StorageError, match="conflicting metadata"):
            store.missing_chunks([conflict])
        assert store.search(vectors[0], limit=1)[0].chunk == sample_chunks[0]


def test_generation_config_env_loading_without_real_credentials(tmp_path, monkeypatch):
    path = tmp_path / "settings.env"
    path.write_text("GEMINI_GENERATION_MODEL=gemini-3.1-flash-lite\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_GENERATION_MODEL", raising=False)
    assert load_generation_config(path).model == "gemini-3.1-flash-lite"
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
    assert load_generation_config(path).model == "gemini-3.8-flash"


def test_actual_default_backend_and_http_upload_index_chat(retrieval_config, tmp_path, monkeypatch):
    import app.retrieval.pipeline as retrieval_module
    import app.ui.service as service_module
    from app.ui.app import build_app
    root = tmp_path / "gradio-uploads"
    root.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(root))
    events = []
    # Only external Gemini calls are mocked. Default _Backend, real retrieval,
    # context/evidence validation, Qdrant, HTTP uploads and queue all execute.
    monkeypatch.setattr(retrieval_module, "GeminiEmbedder", lambda cfg: FakeEmbedder(cfg, events))
    monkeypatch.setattr(service_module, "GeminiAnswerGenerator", lambda key, cfg: FakeGenerator(events))
    service = UIService(retrieval_config, GenerationConfig(), root)
    app = build_app(service)
    dependencies = {entry["api_name"]: entry for entry in app.get_config_file()["dependencies"]}
    session = "dynamic-" + uuid.uuid4().hex
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    try:
        _, url, _ = app.launch(server_name="127.0.0.1", server_port=port,
            share=False, prevent_thread_lock=True, quiet=True, show_error=False,
            enable_monitoring=False, ssr_mode=False, max_file_size=20 * 1024 * 1024)
        with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
            def queued(name, data):
                dependency = dependencies[name]
                response = client.post('/gradio_api/queue/join', json={"data": data,
                    "fn_index": dependency["id"], "trigger_id": dependency["targets"][0][0],
                    "event_data": None, "session_hash": session})
                response.raise_for_status()
                event_id = response.json()["event_id"]
                with client.stream("GET", "/gradio_api/queue/data", params={"session_hash": session}) as stream:
                    stream.raise_for_status()
                    for line in stream.iter_lines():
                        if line.startswith("data: "):
                            message = json.loads(line[6:])
                            if message.get("event_id") == event_id and message.get("msg") == "process_completed":
                                assert message["success"] is True
                                return message["output"]["data"]
                pytest.fail("Queued callback did not complete")
            assert "refreshed" in queued("refresh", [])[1]
            response = client.post('/gradio_api/upload', files=[('files',
                ('Original handbook.txt', b'Employees receive 10 paid sick days per calendar year.', 'text/plain'))])
            response.raise_for_status()
            cached = response.json()[0]
            raw = [{"path": cached, "orig_name": "Original handbook.txt", "meta": {"_type": "gradio.FileData"}}]
            indexed = queued("index_files", [raw])
            assert "indexed 1 new chunks" in indexed[0]
            document_id = indexed[1]["choices"][1][1]
            assert "already indexed" in queued("index_files", [raw])[0]
            answered = queued("chat", ["What are the sick leave rules?", document_id, None])
            assert answered[-1] == "Answer complete."
            assert "Original handbook.txt" in answered[-2]
            assert "Employees receive 10 paid sick days" in answered[-2]
            assert "source_path" not in answered[-2]
            unsupported = queued("chat", ["What is the fee?", document_id, None])
            assert "cannot be determined" in json.dumps(unsupported)
            assert unsupported[-2] == "No supporting citations."
    finally:
        app.close()
        service.close()
    assert len({thread for _, thread in events}) == 1
    assert len(service._backend.retrieval.embedder.document_batches) == 1


def test_ui_entrypoint_launch_failure_closes_resources(retrieval_config, tmp_path, monkeypatch):
    import app.ui.__main__ as entry
    import app.ui.app as ui_app
    previous = __import__('os').environ.get("GRADIO_TEMP_DIR")
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_generation_config", GenerationConfig)
    service = Mock()
    monkeypatch.setattr(entry, "UIService", lambda *args: service)
    app = Mock()
    app.launch.side_effect = RuntimeError("simulated launch failure")
    monkeypatch.setattr(ui_app, "build_app", lambda supplied_service: app)
    with pytest.raises(RuntimeError, match="launch failure"):
        entry.main(["--generation-model", "gemini-3.1-flash-lite", "--port", "7861"])
    service.close.assert_called_once()
    app.close.assert_called_once()
    assert __import__('os').environ.get("GRADIO_TEMP_DIR") == previous


def test_evaluation_cli_closes_embedder_even_if_generator_close_fails(retrieval_config, tmp_path, monkeypatch):
    import app.evaluation.__main__ as entry
    from contextlib import nullcontext
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_generation_config", GenerationConfig)
    monkeypatch.setattr(entry, "load_dataset", lambda path: object())
    monkeypatch.setattr(entry, "isolated_store", lambda *args: nullcontext(object()))
    embedder, generator = Mock(), Mock()
    generator.close.side_effect = RuntimeError("simulated generator close failure")
    monkeypatch.setattr(entry, "GeminiEmbedder", lambda *args: embedder)
    monkeypatch.setattr(entry, "GeminiAnswerGenerator", lambda *args: generator)
    monkeypatch.setattr(entry, "run_evaluation", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "write_reports", lambda *args: None)
    with pytest.raises(ExceptionGroup, match="client cleanup failed") as failure:
        entry.main(["run", "--database", str(tmp_path / "snapshot"), "--dataset", "mocked.json",
                    "--output", str(tmp_path / "reports"), "--live", "--generate"])
    embedder.close.assert_called_once()
    assert isinstance(failure.value.exceptions[0], RuntimeError)
