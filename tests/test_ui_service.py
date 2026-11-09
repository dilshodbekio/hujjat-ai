from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

import pytest
from docx import Document

from app.generation.config import GenerationConfig
from app.generation.models import ModelAnswer
from app.generation.pipeline import GenerationPipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.vector_store import LocalVectorStore
from app.ui.service import MAX_FILE_BYTES, UIError, UIService, safe_error


class FakeEmbedder:
    def __init__(self, config, events):
        self.model, self.dimensions, self.events = config.model, config.dimensions, events
        self.document_batches = []

    def embed_documents(self, texts):
        self.events.append(("embed_documents", get_ident()))
        self.document_batches.append(list(texts))
        return [[1., 0., 0.] for _ in texts]

    def embed_query(self, text):
        self.events.append(("embed_query", get_ident()))
        return [1., 0., 0.]

    def close(self):
        self.events.append(("embedder_close", get_ident()))


class FakeGenerator:
    def __init__(self, events):
        self.events = events

    def generate(self, question, context):
        self.events.append(("generate", get_ident()))
        if "fee" in question:
            return ModelAnswer(status="insufficient_evidence", claims=[])
        return ModelAnswer.model_validate({"status": "answered", "claims": [{
            "text": "The document states its rule.", "evidence": [{
                "source_id": "S1", "quote": context.sources["S1"].text}]}]})

    def close(self):
        self.events.append(("generator_close", get_ident()))


def backend_factory(config, generation, events):
    def create():
        events.append(("init", get_ident()))
        embedder = FakeEmbedder(config, events)
        retrieval = RetrievalPipeline(config, embedder=embedder)
        generator = FakeGenerator(events)
        rag = GenerationPipeline(generation, retrieval=retrieval, generator=generator)
        def close():
            events.append(("close", get_ident()))
            try:
                retrieval.close()
            finally:
                generator.close()
        return SimpleNamespace(retrieval=retrieval, generator=generator, rag=rag, close=close)
    return create


@pytest.fixture
def ui_service(retrieval_config, tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    events = []
    generation = GenerationConfig()
    service = UIService(retrieval_config, generation, root,
        backend_factory=backend_factory(retrieval_config, generation, events))
    yield service, root, events
    service.close()


def upload(path, filename=None):
    return {"path": str(path), "orig_name": filename or path.name}


@pytest.mark.parametrize("extension", [".txt", ".md", ".markdown", ".TXT", ".docx"])
def test_upload_original_name_metadata_duplicates(ui_service, extension):
    service, root, _ = ui_service
    path = root / ("cache" + extension.lower())
    name = "Original résumé" + extension
    if extension == ".docx":
        document = Document()
        document.core_properties.author = "Original author"
        document.add_paragraph("Applicants must submit a transcript.")
        document.save(path)
    else:
        path.write_text("Applicants must submit a transcript.", encoding="utf-8")
    status, choices = service.index_uploads([upload(path, name)])
    assert "indexed 1 new chunks" in status
    assert name in choices[0][0]
    assert ":" not in choices[0][1] and str(root) not in repr(choices)
    status, _ = service.index_uploads([upload(path, name)])
    assert "already indexed" in status
    assert len(service._backend.retrieval.embedder.document_batches) == 1
    chunks = service._submit(lambda: service._backend.retrieval.store.search([1., 0., 0.])).result()
    assert chunks[0].chunk.filename == name
    assert chunks[0].chunk.source_path.startswith("upload:")
    if extension == ".docx":
        assert chunks[0].chunk.metadata["author"] == "Original author"


def test_partial_storage_failure_resumes_missing_only(ui_service):
    service, root, _ = ui_service
    path = root / "long.txt"
    path.write_text(" ".join(f"word{index}" for index in range(1000)), encoding="utf-8")
    def install_failure():
        store = service._backend.retrieval.store
        real_upsert = store.client.upsert
        def partial(collection, *, points, **kwargs):
            real_upsert(collection, points=points[:1], **kwargs)
            raise RuntimeError("simulated partial write")
        store.client.upsert = partial
        return real_upsert
    original = service._submit(install_failure).result()
    message, choices = service.index_uploads([upload(path)])
    assert "database" in message and choices
    initial = len(service._backend.retrieval.embedder.document_batches[0])
    assert initial > 1
    service._submit(lambda: setattr(service._backend.retrieval.store.client, "upsert", original)).result()
    message, _ = service.index_uploads([upload(path)])
    assert f"indexed {initial - 1} new chunks" in message
    assert len(service._backend.retrieval.embedder.document_batches[-1]) == initial - 1
    assert "already indexed" in service.index_uploads([upload(path)])[0]


def test_same_filename_distinct_documents_and_validated_sources(ui_service):
    service, root, _ = ui_service
    first, second = root / "a.txt", root / "b.txt"
    first.write_text("Apples are fruit.", encoding="utf-8")
    second.write_text("Python is a programming language.", encoding="utf-8")
    _, choices = service.index_uploads([upload(first, "same.txt"), upload(second, "same.txt")])
    assert len(choices) == 2 and choices[0] != choices[1]
    for _, token in choices:
        answer, sources = service.answer("What does it say?", token)
        assert "same.txt" in answer
        assert ("Apples" in sources) != ("Python" in sources)
        assert "source_path" not in sources and str(root) not in sources
    answer, sources = service.answer("What is the fee?")
    assert "cannot be determined" in answer
    assert sources == "No supporting citations."
    with pytest.raises(UIError):
        service.answer("question", "unrecognized-token")


def test_existing_legacy_document_accessible_and_pages(ui_service):
    service, _, _ = ui_service
    from app.ingestion.models import DocumentChunk, ChunkLocation
    chunk = DocumentChunk("legacy", "The deadline is Friday.", "C:/private/old.pdf", "old.pdf", "pdf",
        {"author": "Original"}, [ChunkLocation(0, 12, "12", 0, 23)])
    service._submit(lambda: service._backend.retrieval.store.upsert([chunk], [[1., 0., 0.]])).result()
    choices = service.documents()
    assert "C:/private" not in repr(choices)
    answer, sources = service.answer("When is the deadline?", choices[0][1])
    assert "old.pdf, page 12" in answer and "old.pdf, page 12" in sources
    assert "C:/private" not in sources + answer


def test_thread_lifecycle_and_concurrent_inventory(ui_service, retrieval_config):
    service, _, events = ui_service
    with ThreadPoolExecutor(max_workers=4) as callers:
        assert list(callers.map(lambda _: service.documents(), range(8))) == [[]] * 8
    service.close()
    service.close()
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != get_ident()
    assert [name for name, _ in events].count("close") == 1
    with pytest.raises(UIError):
        service.documents()
    with LocalVectorStore(retrieval_config):
        pass  # lock released and DB can reopen on a different owning thread


@pytest.mark.parametrize("name", ["../file.txt", "C:\\secret.txt", "bad\n.txt", "file.exe", ".env"])
def test_invalid_filenames_are_safe(ui_service, name):
    service, root, _ = ui_service
    path = root / "file.txt"
    path.write_text("Text", encoding="utf-8")
    message, _ = service.index_uploads([upload(path, name)])
    assert "invalid" in message or "only" in message
    assert "../" not in message and "secret" not in message
    assert not service._backend.retrieval.embedder.document_batches


def test_invalid_paths_size_count_and_progress(ui_service, tmp_path):
    service, root, _ = ui_service
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    assert "unavailable" in service.index_uploads([upload(outside)])[0]
    for files in ([], [upload(outside)] * 11):
        with pytest.raises(UIError):
            service.index_uploads(files)
    large = root / "large.txt"
    with large.open("wb") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    assert "20 MB" in service.index_uploads([upload(large)])[0]
    valid = root / "small.txt"
    valid.write_text("small text", encoding="utf-8")
    progress = []
    service.index_uploads([upload(valid)], lambda value, **kwargs: progress.append(value))
    assert progress and progress[-1] == 1
    assert "private-path" not in safe_error(RuntimeError("private-path secret-key"))


def test_initialization_failure_cleanup(retrieval_config, tmp_path):
    events = []
    def fail():
        events.append(get_ident())
        raise RuntimeError("private-path secret-key")
    with pytest.raises(UIError) as error:
        UIService(retrieval_config, GenerationConfig(), tmp_path, backend_factory=fail)
    assert "private-path" not in str(error.value)
    assert events[0] != get_ident()


def test_pdf_upload_citations_and_processing_cleanup(ui_service, monkeypatch):
    from test_ingestion import write_pdf
    import app.ui.service as service_module
    service, root, _ = ui_service
    path = root / "cache.pdf"
    write_pdf(path, ["The deadline is Friday.", ""])
    created = []
    actual_temporary_directory = service_module.TemporaryDirectory
    def tracked_directory(*args, **kwargs):
        directory = actual_temporary_directory(*args, **kwargs)
        created.append(directory.name)
        return directory
    monkeypatch.setattr(service_module, "TemporaryDirectory", tracked_directory)
    message, choices = service.index_uploads([upload(path, "Original handbook.PDF")])
    assert "indexed" in message and "OCR" in message
    answer, sources = service.answer("When is the deadline?", choices[0][1])
    assert "Original handbook.PDF, page 1" in answer + sources
    from pathlib import Path
    assert all(not Path(directory).exists() for directory in created)
    corrupt = root / "bad.pdf"
    corrupt.write_bytes(b"broken")
    assert "Could not read" in service.index_uploads([upload(corrupt)])[0]
    assert all(not Path(directory).exists() for directory in created)


def test_worker_shutdown_exception_still_joins(retrieval_config, tmp_path):
    events = []
    factory = backend_factory(retrieval_config, GenerationConfig(), events)
    def create():
        backend = factory()
        actual_close = backend.close
        def close():
            actual_close()
            raise RuntimeError("simulated close failure")
        backend.close = close
        return backend
    service = UIService(retrieval_config, GenerationConfig(), tmp_path, backend_factory=create)
    with pytest.raises(RuntimeError):
        service.close()
    assert all(not thread.is_alive() for thread in service._executor._threads)
    service.close()
    with LocalVectorStore(retrieval_config):
        pass


def test_index_completion_at_polling_timeout_boundary(ui_service, monkeypatch):
    from concurrent.futures import TimeoutError
    service, root, _ = ui_service
    original_submit = service._submit
    calls = []
    class CompletedAtBoundary:
        def result(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise TimeoutError()
            return "race.txt: indexed 1 new chunks; 1 total."

        def done(self):
            return True

    def submit(function, *args):
        if function.__name__ == "_index_one":
            return CompletedAtBoundary()
        return original_submit(function, *args)
    monkeypatch.setattr(service, "_submit", submit)
    message, _ = service.index_uploads([upload(root / "race.txt")])
    assert "indexed 1 new chunks" in message
    assert "could not" not in message
    assert calls == [.1, None]
