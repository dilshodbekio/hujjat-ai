"""Format-independent ingestion results."""

from dataclasses import dataclass, field


@dataclass
class DocumentPart:
    text: str
    page_number: int | None = None
    page_label: str | None = None


@dataclass
class IngestedDocument:
    source_path: str
    filename: str
    format: str
    metadata: dict[str, str] = field(default_factory=dict)
    parts: list[DocumentPart] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkLocation:
    """One occurrence; offsets are Python character indices in the original part."""

    part_index: int
    page_number: int | None
    page_label: str | None
    start_char: int
    end_char: int


@dataclass
class DocumentChunk:
    """Embedding input plus source metadata and every occurrence of its text."""

    chunk_id: str
    text: str
    source_path: str
    filename: str
    format: str
    metadata: dict[str, str] = field(default_factory=dict)
    locations: list[ChunkLocation] = field(default_factory=list)
