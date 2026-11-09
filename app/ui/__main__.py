"""Launch only on loopback, keeping temporary uploads separate from project files."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from app.generation.config import load_generation_config
from app.retrieval.config import load_config
from .service import MAX_FILE_BYTES, UIService, safe_error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local hujjat-ai Gradio UI")
    parser.add_argument("--generation-model", help="Text-generation model override; does not edit .env")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--database", type=Path, help="Optional separate local Qdrant directory")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    retrieval = load_config()
    generation = load_generation_config()
    if args.generation_model:
        generation = replace(generation, model=args.generation_model)
    if args.database:
        retrieval = replace(retrieval, qdrant_path=args.database.resolve())
    previous_cache = os.environ.get("GRADIO_TEMP_DIR")
    with TemporaryDirectory(prefix="hujjat-ui-uploads-") as directory:
        os.environ["GRADIO_TEMP_DIR"] = directory
        service = None
        demo = None
        try:
            # Import after choosing the dedicated Gradio cache location.
            from .app import build_app
            service = UIService(retrieval, generation, directory)
            demo = build_app(service)
            project = Path(__file__).resolve().parents[2]
            demo.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                show_error=False, max_file_size=MAX_FILE_BYTES, enable_monitoring=False,
                blocked_paths=[str(project), str(retrieval.qdrant_path.resolve())],
                allowed_paths=[], footer_links=[], ssr_mode=False, inbrowser=False)
        finally:
            try:
                if demo is not None:
                    demo.close()
            finally:
                try:
                    if service is not None:
                        service.close()
                finally:
                    if previous_cache is None:
                        os.environ.pop("GRADIO_TEMP_DIR", None)
                    else:
                        os.environ["GRADIO_TEMP_DIR"] = previous_cache


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(safe_error(error)) from None
