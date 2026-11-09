"""Conservative matching that always returns an actual source-text span."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start_char: int
    end_char: int


def _normalize_char(char: str) -> str:
    """Map harmless typographical variations to canonical ASCII equivalents."""
    if char in "’‘":
        return "'"
    if char in "“”":
        return '"'
    if char in "–—":
        return "-"
    if char in ("\u00a0", "\u202f"):
        return " "
    return char


def _mapped_text(text: str, heading_colons: bool = False):
    insertions = set()
    if heading_colons:
        for match in re.finditer(r"(?m)^([^\r\n]+)(\r?\n)(?=\S)", text):
            heading = match[1].strip()
            words = heading.split()
            # A narrow PDF label/value repair for short headings and labels.
            # Preserves negations: headings with negation words never receive a colon.
            # Permits alphanumeric section labels (e.g. '3.2 Sick leave', 'Appendix C – Interns').
            clean_words = {re.sub(r"^\W+|\W+$", "", word).casefold() for word in words}
            is_valid_start = (
                heading
                and (
                    heading[0].isupper()
                    or (heading[0].isdigit() and any(c.isalpha() for c in heading))
                    or (heading[0] in "([{" and len(heading) > 1 and (heading[1].isupper() or heading[1].isdigit()))
                )
            )
            is_valid_chars = all(char.isalnum() or char.isspace() or char in "-–—/&.()," for char in heading)
            no_negations = not (clean_words & {"no", "not", "never", "neither", "nor", "without"})
            if (
                heading
                and len(heading) <= 60
                and 1 <= len(words) <= 8
                and is_valid_start
                and is_valid_chars
                and no_negations
            ):
                insertions.add(match.start(2))
    mapped = []
    for index, char in enumerate(text):
        if index in insertions:
            mapped.append((":", index, index))
        norm_char = _normalize_char(char)
        if norm_char.isspace():
            if mapped and mapped[-1][0] == " ":
                mapped[-1] = (" ", mapped[-1][1], index + 1)
            elif mapped:
                mapped.append((" ", index, index + 1))
        else:
            mapped.append((norm_char, index, index + 1))
    if mapped and mapped[-1][0] == " ":
        mapped.pop()
    return "".join(item[0] for item in mapped), mapped


def _complete_token(text: str, start: int, end: int) -> bool:
    """Do not accept substrings cut from names, signed numbers, or decimals."""
    def word(char):
        return char.isalnum() or char == "_"
    prefix = text[:start].rstrip()
    previous_word = re.search(r"(\w+)$", prefix)
    if previous_word and previous_word[1].casefold() in {"no", "not", "never", "without", "neither", "nor"}:
        return False
    if text[start].isdigit() and prefix and prefix[-1] in "+-−$£€¥":
        return False
    if word(text[start]) and start:
        previous = text[start - 1]
        if word(previous):
            return False
        if previous in "-'’" and start > 1 and text[start - 2].isalpha():
            return False
        if text[start].isdigit() and (previous in "+-−" or
                (previous in ".,/" and start > 1 and text[start - 2].isdigit())):
            return False
    if word(text[end - 1]) and end < len(text):
        following = text[end]
        if word(following):
            return False
        if following in "-'’" and end + 1 < len(text) and text[end + 1].isalpha():
            return False
        if text[end - 1].isdigit() and following in ".,/-" and end + 1 < len(text) and text[end + 1].isdigit():
            return False
    return True


def _find_span(source: str, requested: str) -> EvidenceSpan | None:
    norm_requested = "".join(_normalize_char(c) for c in requested)
    for heading_colons in (False, True):
        normalized, mapping = _mapped_text(source, heading_colons)
        start = normalized.find(norm_requested)
        while start >= 0:
            end = start + len(norm_requested)
            # Never accept matches that begin or end on a synthetically inserted colon.
            if mapping[start][1] != mapping[start][2] and mapping[end - 1][1] != mapping[end - 1][2]:
                if _complete_token(normalized, start, end):
                    first, last = mapping[start][1], mapping[end - 1][2]
                    return EvidenceSpan(source[first:last], first, last)
            start = normalized.find(norm_requested, start + 1)
    return None


def extract_evidence_span(source: str, quote: str) -> EvidenceSpan | None:
    """Whitespace-tolerant exact match, then narrowly validated heading/colon match.

    Case, numbers, dates, names, and all substantive punctuation are preserved.
    Harmless typographical variations (curly vs straight quotes/apostrophes, en-dashes)
    are normalized for matching, but an accepted quote is always replaced by the
    original contiguous source slice, never the model's rewritten text.
    This verifies occurrence, not whether a claim follows from that occurrence.
    """
    requested = re.sub(r"\s+", " ", quote).strip()
    if not requested or not any(char.isalnum() for char in requested):
        return None

    span = _find_span(source, requested)
    if span is not None:
        return span

    # Handle harmless outer quotation marks if the model quoted the excerpt with wrapping quotes.
    if len(requested) >= 2 and (
        (requested[0] == '"' and requested[-1] == '"')
        or (requested[0] == "'" and requested[-1] == "'")
        or (requested[0] == "“" and requested[-1] == "”")
        or (requested[0] == "‘" and requested[-1] == "’")
    ):
        inner = requested[1:-1].strip()
        if inner and any(char.isalnum() for char in inner):
            return _find_span(source, inner)

    return None
