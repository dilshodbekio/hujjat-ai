import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.models import DocumentPart, IngestedDocument
from app.ingestion.pipeline import ingest_document


def document(*texts):
    return IngestedDocument("/docs/source.pdf", "source.pdf", "pdf",
                            {"title": "Example", "author": "Author", "filename": "custom"},
                            [DocumentPart(text, index + 1, f"p-{index + 1}")
                             for index, text in enumerate(texts)])


def test_paragraph_boundary_preferred_over_sentence():
    text = "First paragraph.\n\nSecond sentence. Third sentence."
    chunks = chunk_document(document(text), max_chars=38, overlap=0)
    assert chunks[0].text == "First paragraph.\n\n"
    assert "".join(chunk.text for chunk in chunks) == text


def test_sentence_boundary_preferred_over_word():
    chunks = chunk_document(document("First sentence. Second longer sentence."),
                            max_chars=26, overlap=0)
    assert chunks[0].text == "First sentence. "


def test_word_boundary_before_hard_split():
    chunks = chunk_document(document("alpha beta gamma"), max_chars=12, overlap=0)
    assert [chunk.text for chunk in chunks] == ["alpha beta ", "gamma"]


@pytest.mark.parametrize("overlap", [0, 80, 799])
def test_limit_overlap_and_complete_coverage(overlap):
    text = "".join(chr(0x4E00 + index) for index in range(1703))
    chunks = chunk_document(document(text), overlap=overlap)
    assert all(0 < len(chunk.text) <= 800 for chunk in chunks)
    locations = [chunk.locations[0] for chunk in chunks]
    assert locations[0].start_char == 0
    assert locations[-1].end_char == len(text)
    for chunk, location in zip(chunks, locations):
        assert chunk.text == text[location.start_char:location.end_char]
    for previous, current in zip(locations, locations[1:]):
        assert current.start_char == previous.end_char - overlap
        assert current.end_char > previous.end_char
    reconstructed = chunks[0].text + "".join(chunk.text[overlap:] for chunk in chunks[1:])
    assert reconstructed == text


def test_natural_boundary_overlap():
    text = "A paragraph ends here.\n\nAnother paragraph with more words.\nLast one."
    chunks = chunk_document(document(text), max_chars=40, overlap=5)
    assert chunks[0].text == "A paragraph ends here.\n\n"
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-5:] == current.text[:5]
        assert len(current.text) <= 40


def test_metadata_pages_and_copies():
    source = document("page one", "page two")
    chunks = chunk_document(source)
    for index, chunk in enumerate(chunks):
        assert chunk.source_path == source.source_path
        assert chunk.filename == source.filename
        assert chunk.format == source.format
        assert chunk.metadata == source.metadata
        assert chunk.locations[0].part_index == index
        assert chunk.locations[0].page_number == index + 1
        assert chunk.locations[0].page_label == f"p-{index + 1}"
    chunks[0].metadata["title"] = "Changed"
    assert source.metadata["title"] == chunks[1].metadata["title"] == "Example"


def test_duplicate_text_retains_all_locations_and_stable_ids():
    source = document("repeated", "repeated", "different")
    chunks = chunk_document(source)
    assert [chunk.text for chunk in chunks] == ["repeated", "different"]
    assert [location.page_number for location in chunks[0].locations] == [1, 2]
    assert [location.part_index for location in chunks[0].locations] == [0, 1]
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in chunk_document(source)]
    other = document("repeated")
    other.source_path = "/other/source.pdf"
    assert chunk_document(other)[0].chunk_id != chunks[0].chunk_id


def test_empty_parts_and_unpaged_content():
    source = document("", " \t\n", "usable")
    source.parts[2].page_number = source.parts[2].page_label = None
    chunks = chunk_document(source)
    assert len(chunks) == 1
    assert chunks[0].locations[0].part_index == 2
    assert chunks[0].locations[0].page_number is None
    assert chunks[0].locations[0].page_label is None
    assert chunk_document(document("", " \n")) == []
    assert chunk_document(document()) == []


def test_ingestion_to_embedding_input_preserves_unicode_and_whitespace(tmp_path):
    path = tmp_path / "notes.md"
    text = "# O'zbekcha\n\n  e\u0301 matn. Salom!\n\n```python\n    print('hello')\n```\n"
    path.write_text(text, encoding="utf-8")
    source = ingest_document(path)
    chunks = chunk_document(source, max_chars=30, overlap=0)
    assert "".join(chunk.text for chunk in chunks) == text
    assert source.parts[0].text == text
    assert all(chunk.filename == "notes.md" and chunk.format == "md" for chunk in chunks)
    assert all(chunk.source_path == str(path.absolute()) for chunk in chunks)


def test_repeated_windows_and_whitespace_windows_do_not_duplicate_embeddings():
    source = document("abc" * 10, " " * 20 + "end")
    chunks = chunk_document(source, max_chars=6, overlap=0)
    assert [chunk.text for chunk in chunks] == ["abcabc", "  end"]
    assert len(chunks[0].locations) == 5
    assert chunks[1].locations[0].start_char == 18


@pytest.mark.parametrize("size", [799, 800, 801])
def test_exact_limit_and_no_overlap_only_tail(size):
    text = "x" * size
    chunks = chunk_document(document(text))
    assert len(chunks) == (1 if size <= 800 else 2)
    assert chunks[-1].locations[-1].end_char == size


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (-1, 0), (800, -1),
                         (800, 800), (800, 801), (True, 0), (800, False),
                         (1.5, 0), (800, 1.5)])
def test_invalid_configuration(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_document(document("text"), max_chars=max_chars, overlap=overlap)
