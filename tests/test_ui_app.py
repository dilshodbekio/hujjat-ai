from unittest.mock import Mock
from copy import deepcopy


import gradio as gr

from app.ui.app import build_app


def test_installed_gradio_compatibility_and_wiring():
    service = Mock()
    service.documents.return_value = []
    app = build_app(service)
    assert gr.__version__ == "6.31.0"
    config = app.get_config_file()
    components = {component["type"]: component["props"] for component in config["components"]}
    assert components["chatbot"]["render_markdown"] is False
    assert components["chatbot"]["allow_file_downloads"] is False
    assert components["file"]["file_count"] == "multiple"
    assert app.analytics_enabled is False
    for dependency in config["dependencies"]:
        assert dependency["api_visibility"] == "private"
    service.answer.assert_not_called()
    app.close()


def test_chat_history_isolated_and_errors_sanitized():
    service = Mock()
    service.answer.return_value = ("Grounded answer. [document.pdf, page 12]", "Validated evidence")
    app = build_app(service)
    functions = {item.fn.__name__: item.fn for item in app.fns.values() if item.fn is not None}
    first_history = [{"role": "user", "content": "Previous question"}]
    before = deepcopy(first_history)
    first = functions["chat"]("New question", "opaque-id", first_history)
    second = functions["chat"]("Another session", "", [])
    assert first_history == before
    assert len(first[1]) == 3 and len(second[1]) == 2
    assert "Previous question" not in repr(second)
    service.answer.side_effect = RuntimeError("secret-key C:/private/.env")
    error = functions["chat"]("question", "", [])
    assert "secret-key" not in repr(error) and "C:/private" not in repr(error)
    assert functions["clear"]()[0:2] == ([], [])
    app.close()


def test_raw_upload_metadata_reaches_callback():
    service = Mock()
    service.index_uploads.return_value = ("Indexed", [("Original file.txt", "opaque-id")])
    app = build_app(service)
    index = next(item.fn for item in app.fns.values() if item.fn and item.fn.__name__ == "index_files")
    files = [{"path": "cache-only-path", "orig_name": "Original file.txt"}]
    progress = Mock()
    result = index(files, progress)
    service.index_uploads.assert_called_once_with(files, progress)
    assert result[0] == "Indexed"
    app.close()


def test_loopback_http_startup_and_blocked_files(tmp_path, monkeypatch):
    import socket
    from urllib.parse import quote
    import httpx
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    secret = tmp_path / "private.txt"
    secret.write_text("not-a-real-key", encoding="utf-8")
    service = Mock()
    service.documents.return_value = []
    app = build_app(service)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        _, url, _ = app.launch(server_name="127.0.0.1", server_port=port,
            share=False, prevent_thread_lock=True, show_error=False, quiet=True,
            blocked_paths=[str(secret)], allowed_paths=[], enable_monitoring=False,
            max_file_size=20 * 1024 * 1024, ssr_mode=False)
        with httpx.Client(timeout=10, trust_env=False) as client:
            assert client.get(url).status_code == 200
            assert client.get(url.rstrip("/") + "/config").json()["title"] == "hujjat-ai"
            response = client.get(url.rstrip("/") + "/gradio_api/file=" + quote(str(secret), safe=""))
            assert response.status_code in {403, 404}
            assert "not-a-real-key" not in response.text
        service.answer.assert_not_called()
    finally:
        app.close()
