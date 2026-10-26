"""Names and special terms, and clean-up of how Whisper writes them.

A glossary (one term per line) is given to Whisper as words to listen for, and used
afterwards to write each term exactly the same way every time. Spelling matters
because the translation and the voice both inherit it.
"""

from __future__ import annotations

import re
from pathlib import Path

# Whisper sometimes writes acronyms with spaces: "S .H .I .E .L .D." -> "S.H.I.E.L.D."
_SPACED_ACRONYM = re.compile(r"\b((?:[A-Z] ?\. ?){1,}[A-Z]\.?)")


def load_glossary(*paths: Path | None, extra: list[str] | None = None) -> list[str]:
    terms: list[str] = []
    for path in paths:
        if path and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and line not in terms:
                    terms.append(line)
    for term in extra or []:
        if term not in terms:
            terms.append(term)
    return terms


def fix_acronyms(text: str) -> str:
    def join(m: re.Match) -> str:
        letters = re.findall(r"[A-Z]", m.group(1))
        return ".".join(letters) + "."
    out = _SPACED_ACRONYM.sub(join, text)
    return re.sub(r"\.\.+", ".", out)


def apply_glossary(text: str, terms: list[str]) -> str:
    """Write glossary terms with the glossary's exact spelling and capitals."""
    for term in sorted(terms, key=len, reverse=True):
        # Dots must be there (so "shield" never becomes "S.H.I.E.L.D."); stray spaces are allowed.
        loose = re.escape(term).replace(r"\.", r"\.\s*").replace(r"\ ", r"\s+")
        text = re.sub(rf"(?<!\w){loose}(?!\w)", term, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, terms: list[str]) -> str:
    text = fix_acronyms(" ".join(text.split()))
    return apply_glossary(text, terms) if terms else text


def hotwords(terms: list[str]) -> str | None:
    """The glossary as a hint string for Whisper (faster-whisper 'hotwords')."""
    return ", ".join(terms) if terms else None
