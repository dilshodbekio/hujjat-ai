"""Public entry point for document ingestion."""

from pathlib import Path

from .loaders import load_docx, load_pdf, load_text
from .models import IngestedDocument
from .normalize import normalize_text


class IngestionError(Exception):
    """A source-specific failure, with the original error chained when present."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        super().__init__(f"Cannot ingest '{path}': {reason}")


def ingest_document(path: str | Path) -> IngestedDocument:
    source = Path(path).absolute()
    loaders = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text,
               ".md": load_text, ".markdown": load_text}
    try:
        if not source.exists():
            raise FileNotFoundError("file does not exist")
        if not source.is_file():
            raise IsADirectoryError("source is not a regular file")
        loader = loaders.get(source.suffix.lower())
        if loader is None:
            raise IngestionError(source, "unsupported extension; expected PDF, DOCX, TXT, MD, or Markdown")
        result = loader(source)
    except IngestionError:
        raise
    except UnicodeDecodeError as exc:
        raise IngestionError(source, "text must be UTF-8 encoded (an optional UTF-8 BOM is accepted)") from exc
    except OSError as exc:
        raise IngestionError(source, f"file access failed: {exc}") from exc
    except Exception as exc:
        raise IngestionError(source, f"{source.suffix.upper().lstrip('.')} extraction failed ({type(exc).__name__}): {exc}") from exc

    for part in result.parts:
        part.text = normalize_text(part.text)
        if not part.text.strip() and part.page_number is not None:
            result.warnings.append(f"Page {part.page_number} has no extractable text; it may be blank or require OCR.")
    if not any(part.text.strip() for part in result.parts):
        hint = " OCR may be required for scanned PDFs." if result.format == "pdf" else ""
        raise IngestionError(source, f"document contains no usable text.{hint}")
    return result
