import re

_HAS_WHITESPACE = re.compile(r"\s")
_ENDS_WITH_PUNCT = re.compile(r"[.!?]$")


def is_degenerate_output(text: str) -> bool:
    trimmed = text.strip()
    if len(trimmed) == 0:
        return True
    if len(trimmed) < 2:
        return True
    if (
        not _HAS_WHITESPACE.search(trimmed)
        and len(trimmed) < 15
        and not _ENDS_WITH_PUNCT.search(trimmed)
    ):
        return True
    return False
