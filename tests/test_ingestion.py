from pathlib import Path

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.ingestion.normalize import normalize_text
from app.ingestion.pipeline import IngestionError, ingest_document


def write_pdf(path, texts, encrypted=False):
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Example", "/Author": "Test Author"})
    for text in texts:
        page = writer.add_blank_page(width=300, height=300)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("secret")
    writer.write(path)


@pytest.mark.parametrize("extension", [".txt", ".TXT", ".md", ".MD", ".markdown"])
def test_text_formats_preserve_content(tmp_path, extension):
    path = tmp_path / ("sample" + extension)
    content = "  # Sarlavha  \r\n\r\n```python\r\n    print('salom')\r\n```\r\n| A | B |\r\n| - | - |\r\n\ttext  \r\n"
    path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    result = ingest_document(path)
    assert result.source_path == str(path.absolute())
    assert result.filename == path.name
    assert result.format == extension[1:].lower()
    assert result.parts[0].text == content.replace("\r\n", "\n")
    assert result.parts[0].page_number is None
    assert result.parts[0].page_label is None
    assert result.metadata == {}


def test_normalization_is_conservative():
    # Decomposed Unicode, compatibility characters, trailing spaces and tabs survive.
    text = "  e\u0301 ①\t\r\n\rnext\x00  \n"
    expected = "  e\u0301 ①\t\n\nnext  \n"
    assert normalize_text(text) == expected
    assert normalize_text(expected) == expected


def test_pdf_pages_metadata_and_empty_page(tmp_path):
    path = tmp_path / "sample.PDF"
    write_pdf(path, ["First page", "", "Third page"])
    result = ingest_document(path)
    assert [part.text.strip() for part in result.parts] == ["First page", "", "Third page"]
    assert [part.page_number for part in result.parts] == [1, 2, 3]
    assert [part.page_label for part in result.parts] == ["1", "2", "3"]
    assert result.metadata["title"] == "Example"
    assert result.metadata["author"] == "Test Author"
    assert any("Page 2" in warning for warning in result.warnings)


def test_docx_body_order_and_metadata(tmp_path):
    path = tmp_path / "sample.DOCX"
    doc = Document()
    doc.core_properties.title = "Example DOCX"
    doc.core_properties.author = "Author"
    doc.add_paragraph("  Before  ")
    table = doc.add_table(rows=2, cols=2)
    for cell, value in zip(table._cells, ["A", "B", "one\ntwo", "four"]):
        cell.text = value
    doc.add_paragraph("\tAfter")
    doc.save(path)
    result = ingest_document(path)
    assert result.parts[0].text == "  Before  \nA\tB\none\ntwo\tfour\n\tAfter"
    assert result.metadata["title"] == "Example DOCX"
    assert result.metadata["author"] == "Author"
    assert result.parts[0].page_number is None


def test_pdf_custom_labels_and_missing_metadata(tmp_path):
    path = tmp_path / "labels.pdf"
    write_pdf(path, ["Preface", "Body"])
    writer = PdfWriter(clone_from=path)
    writer.metadata = None
    writer.set_page_label(0, 0, style="/r", start=1)
    writer.set_page_label(1, 1, style="/D", prefix="Section-", start=1)
    writer.write(path)
    result = ingest_document(path)
    assert result.metadata == {}
    assert [part.page_label for part in result.parts] == ["i", "Section-1"]
    assert [part.page_number for part in result.parts] == [1, 2]


def test_docx_missing_metadata(tmp_path):
    path = tmp_path / "no_metadata.docx"
    doc = Document()
    doc.core_properties.title = ""
    doc.core_properties.author = ""
    doc.add_paragraph("Hello")
    doc.save(path)
    result = ingest_document(path)
    assert "title" not in result.metadata
    assert "author" not in result.metadata


@pytest.mark.parametrize("surrounding_text", [False, True])
def test_docx_nested_tables_preserve_text_and_order(tmp_path, surrounding_text):
    path = tmp_path / "nested.docx"
    doc = Document()
    cell = doc.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].text = "Before" if surrounding_text else ""
    nested_cell = cell.add_table(rows=1, cols=1).cell(0, 0)
    nested_cell.paragraphs[0].text = "Inside" if surrounding_text else ""
    nested_cell.add_table(rows=1, cols=1).cell(0, 0).text = "Deep text"
    nested_cell.paragraphs[-1].text = "Inside after" if surrounding_text else ""
    cell.paragraphs[-1].text = "After" if surrounding_text else ""
    doc.save(path)

    result = ingest_document(path)
    expected = ("Before\nInside\nDeep text\nInside after\nAfter"
                if surrounding_text else "\n\nDeep text\n\n")
    assert result.parts[0].text == expected


@pytest.mark.parametrize("before,after", [(1, 0), (0, 1), (1, 1)])
def test_docx_omitted_table_cells_preserve_columns(tmp_path, before, after):
    path = tmp_path / "omitted_cells.docx"
    doc = Document()
    table = doc.add_table(rows=2, cols=3)
    for cell, value in zip(table.rows[0].cells, ["Name", "Amount", "Note"]):
        cell.text = value
    row = table.rows[1]
    values = ["Alice", "100", "Paid"]
    for cell, value in zip(row.cells, values):
        cell.text = value
    # Word represents omitted cells with gridBefore/gridAfter on the row.
    cells = list(row._tr.tc_lst)
    for cell in cells[:before] + (cells[-after:] if after else []):
        row._tr.remove(cell)
    for tag, count in (("gridBefore", before), ("gridAfter", after)):
        if count:
            element = OxmlElement(f"w:{tag}")
            element.set(qn("w:val"), str(count))
            row._tr.get_or_add_trPr().append(element)
    doc.save(path)

    result = ingest_document(path)
    expected_row = [""] * before + values[before:3-after] + [""] * after
    assert result.parts[0].text == "Name\tAmount\tNote\n" + "\t".join(expected_row)


@pytest.mark.parametrize("extension", [".txt", ".md", ".pdf", ".docx"])
def test_empty_documents_rejected(tmp_path, extension):
    path = tmp_path / ("empty" + extension)
    if extension == ".pdf":
        write_pdf(path, [""])
    elif extension == ".docx":
        Document().save(path)
    else:
        path.write_text(" \t\n\x00", encoding="utf-8")
    with pytest.raises(IngestionError, match="no usable text") as error:
        ingest_document(path)
    if extension == ".pdf":
        assert "OCR" in str(error.value)


@pytest.mark.parametrize("extension", [".pdf", ".docx"])
def test_corrupt_files_preserve_exception(tmp_path, extension):
    path = tmp_path / ("broken" + extension)
    path.write_bytes(b"not a valid document")
    with pytest.raises(IngestionError, match="extraction failed") as error:
        ingest_document(path)
    assert error.value.__cause__ is not None
    assert str(path) in str(error.value)


def test_encrypted_pdf(tmp_path):
    path = tmp_path / "locked.pdf"
    write_pdf(path, ["secret"], encrypted=True)
    with pytest.raises(IngestionError, match="requires a password") as error:
        ingest_document(path)
    assert isinstance(error.value.__cause__, ValueError)


def test_invalid_encoding(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(IngestionError, match="UTF-8") as error:
        ingest_document(path)
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_missing_file(tmp_path):
    with pytest.raises(IngestionError, match="does not exist") as error:
        ingest_document(tmp_path / "missing.txt")
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_directory_rejected(tmp_path):
    with pytest.raises(IngestionError, match="not a regular file") as error:
        ingest_document(tmp_path)
    assert isinstance(error.value.__cause__, IsADirectoryError)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"example")
    with pytest.raises(IngestionError, match="unsupported extension"):
        ingest_document(path)


def test_permission_error_preserves_cause(tmp_path, monkeypatch):
    path = tmp_path / "private.txt"
    path.write_text("text", encoding="utf-8")
    def deny_open(*args, **kwargs):
        raise PermissionError("permission denied")
    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(IngestionError, match="permission denied") as error:
        ingest_document(path)
    assert isinstance(error.value.__cause__, PermissionError)
