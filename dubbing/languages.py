"""One table of language codes, because every model names languages differently.

Whisper uses ISO 639-1 ('hi'), NLLB uses FLORES-200 ('hin_Deva'),
XTTS v2 uses its own short codes, and MMS TTS uses ISO 639-3 ('hin').
"""

from __future__ import annotations

from dataclasses import dataclass

from dubbing.errors import PipelineError


@dataclass(frozen=True)
class Language:
    code: str                 # our code (same as Whisper's)
    name: str
    nllb: str                 # translation model code
    script: str               # writing system, used to check translations
    xtts: str | None          # None = XTTS v2 cannot speak it
    mms: str | None           # Meta MMS TTS code (fallback voice), None if unknown
    # Rough characters spoken per second in ordinary conversation. An ESTIMATE used
    # only to warn early that a translation may be too long for its time slot;
    # Phase 4 replaces it with real measured speech durations.
    chars_per_second: float = 14.0


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in [
        Language("en", "English", "eng_Latn", "Latin", "en", "eng", 14.0),
        Language("hi", "Hindi", "hin_Deva", "Devanagari", "hi", "hin", 13.0),
        Language("mr", "Marathi", "mar_Deva", "Devanagari", None, "mar", 13.0),
        Language("bn", "Bengali", "ben_Beng", "Bengali", None, "ben", 13.0),
        Language("te", "Telugu", "tel_Telu", "Telugu", None, "tel", 12.0),
        Language("ta", "Tamil", "tam_Taml", "Tamil", None, "tam", 12.0),
        Language("or", "Odia", "ory_Orya", "Oriya", None, "ory", 12.0),
        Language("es", "Spanish", "spa_Latn", "Latin", "es", "spa", 15.0),
        Language("fr", "French", "fra_Latn", "Latin", "fr", "fra", 15.0),
        Language("de", "German", "deu_Latn", "Latin", "de", "deu", 14.0),
    ]
}

# Unicode ranges for each script, used to check output really is in that script.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "Devanagari": [(0x0900, 0x097F)],
    "Bengali": [(0x0980, 0x09FF)],
    "Oriya": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
}


def get_language(code: str) -> Language:
    lang = LANGUAGES.get(code.lower().strip())
    if lang is None:
        known = ", ".join(f"{c} ({l.name})" for c, l in LANGUAGES.items())
        raise PipelineError(
            f"Language '{code}' is not in the supported list yet.\nSupported: {known}\n"
            "More can be added in dubbing/languages.py if NLLB supports them."
        )
    return lang


def script_fraction(text: str, script: str) -> float:
    """Share of letters in `text` that belong to `script` (digits/punctuation ignored)."""
    ranges = SCRIPT_RANGES[script]
    letters = [ch for ch in text if ch.isalpha() or 0x0900 <= ord(ch) <= 0x0DFF]
    if not letters:
        return 0.0
    inside = sum(1 for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in ranges))
    return inside / len(letters)
