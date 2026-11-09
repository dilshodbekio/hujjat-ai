from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from app.generation.config import GenerationConfig


@pytest.mark.parametrize("operation_fails,generator_fails,embedder_fails", [
    (False, True, False), (False, False, True), (False, True, True),
    (True, False, False), (True, True, False), (True, False, True), (True, True, True)])
def test_independent_client_cleanup_and_error_context(retrieval_config, tmp_path, monkeypatch,
        operation_fails, generator_fails, embedder_fails):
    import app.evaluation.__main__ as entry
    events = []
    @contextmanager
    def store(*args):
        try:
            yield object()
        finally:
            events.append("store_closed")
    embedder, generator = Mock(), Mock()
    def close_generator():
        events.append("generator_closed")
        if generator_fails:
            raise RuntimeError("generator cleanup error")
    def close_embedder():
        events.append("embedder_closed")
        if embedder_fails:
            raise RuntimeError("embedder cleanup error")
    generator.close.side_effect = close_generator
    embedder.close.side_effect = close_embedder
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_generation_config", GenerationConfig)
    monkeypatch.setattr(entry, "load_dataset", lambda path: object())
    monkeypatch.setattr(entry, "isolated_store", store)
    monkeypatch.setattr(entry, "GeminiEmbedder", lambda *args: embedder)
    monkeypatch.setattr(entry, "GeminiAnswerGenerator", lambda *args: generator)
    operation_error = ValueError("execution error")
    def run(*args, **kwargs):
        if operation_fails:
            raise operation_error
        return {}
    monkeypatch.setattr(entry, "run_evaluation", run)
    monkeypatch.setattr(entry, "write_reports", lambda *args: None)
    expected = ExceptionGroup if generator_fails or embedder_fails else ValueError
    with pytest.raises(expected) as caught:
        entry.main(["run", "--database", str(tmp_path / "snapshot"), "--dataset", "mocked.json",
            "--output", str(tmp_path / "reports"), "--live", "--generate"])
    assert events == ["generator_closed", "embedder_closed", "store_closed"]
    if isinstance(caught.value, ExceptionGroup):
        assert len(caught.value.exceptions) == int(generator_fails) + int(embedder_fails)
        if operation_fails:
            assert caught.value.__context__ is operation_error
    else:
        assert caught.value is operation_error


def test_generator_initialization_failure_closes_embedder_and_store(retrieval_config, tmp_path, monkeypatch):
    import app.evaluation.__main__ as entry
    events = []
    @contextmanager
    def store(*args):
        try:
            yield object()
        finally:
            events.append("store_closed")
    embedder = Mock()
    embedder.close.side_effect = lambda: events.append("embedder_closed")
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_generation_config", GenerationConfig)
    monkeypatch.setattr(entry, "load_dataset", lambda path: object())
    monkeypatch.setattr(entry, "isolated_store", store)
    monkeypatch.setattr(entry, "GeminiEmbedder", lambda *args: embedder)
    def fail_generator(*args, **kwargs):
        raise RuntimeError("generator construction failed")
    monkeypatch.setattr(entry, "GeminiAnswerGenerator", fail_generator)

    with pytest.raises(RuntimeError, match="generator construction failed"):
        entry.main(["run", "--database", str(tmp_path / "snapshot"), "--dataset", "mocked.json",
                    "--output", str(tmp_path / "reports"), "--live", "--generate"])
    assert events == ["embedder_closed", "store_closed"]


def test_embedder_initialization_failure_cleans_up_store(retrieval_config, tmp_path, monkeypatch):
    import app.evaluation.__main__ as entry
    events = []
    @contextmanager
    def store(*args):
        try:
            yield object()
        finally:
            events.append("store_closed")
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_dataset", lambda path: object())
    monkeypatch.setattr(entry, "isolated_store", store)
    def fail_embedder(*args, **kwargs):
        raise RuntimeError("embedder construction failed")
    monkeypatch.setattr(entry, "GeminiEmbedder", fail_embedder)

    with pytest.raises(RuntimeError, match="embedder construction failed"):
        entry.main(["run", "--database", str(tmp_path / "snapshot"), "--dataset", "mocked.json",
                    "--output", str(tmp_path / "reports"), "--live"])
    assert events == ["store_closed"]


def test_cli_main_preserves_evaluation_error_when_cleanup_fails(retrieval_config, tmp_path, monkeypatch):
    import app.evaluation.__main__ as entry
    @contextmanager
    def store(*args):
        yield object()
    embedder, generator = Mock(), Mock()
    generator.close.side_effect = RuntimeError("generator cleanup error")
    monkeypatch.setattr(entry, "load_config", lambda: retrieval_config)
    monkeypatch.setattr(entry, "load_generation_config", GenerationConfig)
    monkeypatch.setattr(entry, "load_dataset", lambda path: object())
    monkeypatch.setattr(entry, "isolated_store", store)
    monkeypatch.setattr(entry, "GeminiEmbedder", lambda *args: embedder)
    monkeypatch.setattr(entry, "GeminiAnswerGenerator", lambda *args: generator)
    monkeypatch.setattr(entry, "run_evaluation", Mock(side_effect=ValueError("operation error")))

    # Simulate __main__ execution block
    with pytest.raises(SystemExit) as exit_info:
        try:
            entry.main(["run", "--database", str(tmp_path / "snapshot"), "--dataset", "mocked.json",
                        "--output", str(tmp_path / "reports"), "--live", "--generate"])
        except Exception as error:
            primary = error
            cleanup_note = ""
            if isinstance(error, ExceptionGroup):
                if error.__context__ is not None:
                    primary = error.__context__
                cleanup_names = ", ".join(type(e).__name__ for e in error.exceptions)
                cleanup_note = f" (Client cleanup also failed: {cleanup_names})"
            raise SystemExit(
                f"Evaluation failed: {type(primary).__name__}{cleanup_note}. "
                "Check dataset/configuration, collection schema, and that Qdrant is closed."
            ) from None

    msg = str(exit_info.value)
    assert "Evaluation failed: ValueError" in msg
    assert "Client cleanup also failed: RuntimeError" in msg
