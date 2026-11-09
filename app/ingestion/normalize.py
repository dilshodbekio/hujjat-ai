"""Conservative text normalization; whitespace carries meaning."""


def normalize_text(text: str) -> str:
    """Normalize newlines and remove nulls, preserving all other characters.

    Do not strip, collapse whitespace, or normalize Unicode: indentation,
    Markdown hard breaks, code, and compatibility characters may be meaningful.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
