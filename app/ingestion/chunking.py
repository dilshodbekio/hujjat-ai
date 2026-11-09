"""Paragraph-first chunking with sentence, word, and character fallbacks."""

from bisect import bisect_right
from hashlib import sha256
import re

from .models import ChunkLocation, DocumentChunk, IngestedDocument


def _spans(text: str, max_chars: int, overlap: int):
    # Keep separators in the original slices so whitespace and offsets survive.
    boundaries = [
        [match.end() for match in re.finditer(pattern, text)]
        for pattern in (r"\n+", r"[.!?。！？][\"'”’)]*\s+", r"\s+")
    ]
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for positions in boundaries:
                index = bisect_right(positions, end) - 1
                # A boundary must allow new text after the requested overlap.
                if index >= 0 and positions[index] > start + overlap:
                    end = positions[index]
                    break
        yield start, end
        if end == len(text):
            break
        start = end - overlap


def chunk_document(
    document: IngestedDocument, *, max_chars: int = 800, overlap: int = 80,
) -> list[DocumentChunk]:
    """Prepare unique embedding texts without crossing document-part boundaries.

    Prefer newlines (paragraphs), then sentence punctuation, then whitespace,
    finally a hard character split. Sentence detection is heuristic, not NLP.
    Adjacent source spans overlap by exactly ``overlap`` characters; the limit
    includes that overlap. Empty parts/windows produce no embedding input.
    Identical chunk texts are embedded once, retaining ALL source locations,
    including repeated text on different pages. No source text is normalized.
    Metadata is copied so downstream updates do not mutate the input document.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_chars:
        raise ValueError("overlap must be an integer satisfying 0 <= overlap < max_chars")

    by_text: dict[str, DocumentChunk] = {}
    for part_index, part in enumerate(document.parts):
        if not part.text.strip():
            continue
        for start, end in _spans(part.text, max_chars, overlap):
            text = part.text[start:end]
            if not text.strip():
                continue
            location = ChunkLocation(part_index, part.page_number, part.page_label, start, end)
            if text in by_text:
                by_text[text].locations.append(location)
                continue
            # Length-prefix the source identity to avoid ambiguous concatenation.
            identity = f"{len(document.source_path)}:{document.source_path}{text}"
            by_text[text] = DocumentChunk(
                chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
                text=text, source_path=document.source_path, filename=document.filename,
                format=document.format, metadata=dict(document.metadata), locations=[location],
            )
    return list(by_text.values())
