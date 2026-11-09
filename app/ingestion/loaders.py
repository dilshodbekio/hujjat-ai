"""Extract text without chunking or interpreting document content."""

from collections.abc import Iterable
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from .models import DocumentPart, IngestedDocument


def _document(path: Path) -> IngestedDocument:
    return IngestedDocument(str(path), path.name, path.suffix.lower()[1:])


def load_pdf(path: Path) -> IngestedDocument:
    result = _document(path)
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF requires a password; password input is not supported")
    metadata = reader.metadata
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                result.metadata[str(key).lstrip("/")] = str(value)
        for source, target in (("/Title", "title"), ("/Author", "author"),
                               ("/CreationDate", "created"), ("/ModDate", "modified")):
            if metadata.get(source) is not None:
                result.metadata[target] = str(metadata[source])
    labels = reader.page_labels
    for index, page in enumerate(reader.pages):
        result.parts.append(DocumentPart(page.extract_text() or "", index + 1, labels[index]))
    result.warnings.append("PDF text reading order and layout depend on the source PDF; OCR is not performed.")
    return result


def _docx_blocks_text(blocks: Iterable[Paragraph | Table]) -> str:
    """Read body or cell blocks in order, including nested tables."""
    texts = []
    for block in blocks:
        if isinstance(block, Paragraph):
            texts.append(block.text)
        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                cells = [""] * row.grid_cols_before
                cells.extend(_docx_blocks_text(cell.iter_inner_content()) for cell in row.cells)
                cells.extend([""] * row.grid_cols_after)
                rows.append("\t".join(cells))
            texts.append("\n".join(rows))
    return "\n".join(texts)


def load_docx(path: Path) -> IngestedDocument:
    result = _document(path)
    document = Document(path)
    props = document.core_properties
    for key in ("title", "author", "subject", "keywords", "created", "modified"):
        value = getattr(props, key)
        if value is not None and value != "":
            result.metadata[key] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    result.parts.append(DocumentPart(_docx_blocks_text(document.iter_inner_content())))
    result.warnings.append("DOCX extraction includes body paragraphs and tables; styles, headers, footers, text boxes, and images are not extracted. Tables use tabs between cells.")
    return result


def load_text(path: Path) -> IngestedDocument:
    result = _document(path)
    # Disable automatic newline conversion so normalization has a single owner.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        result.parts.append(DocumentPart(stream.read()))
    return result
