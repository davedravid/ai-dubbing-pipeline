"""Group transcript segments into whole sentences ("dub units") before translating.

Whisper often splits one sentence across segments. Translating the halves separately
gives broken grammar, especially into Hindi, where the verb comes at the end. A unit
is a run of consecutive segments that together form a sentence and is translated,
timed and later spoken as one piece.
"""

from __future__ import annotations

import re
from typing import Any

SENTENCE_END = (".", "?", "!", "…", '"', "'", "।")
# A full stop that belongs to an abbreviation or acronym does not end the sentence:
# "...why S.H.I.E.L.D." + "is using the Tesseract..." is one sentence.
_ABBREVIATION_END = re.compile(r"(?:\b(?:[A-Z]\.){2,}|\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Prof|Capt|Gen|Lt|Sgt|vs)\.)$")


def ends_sentence(text: str) -> bool:
    text = text.rstrip()
    if not text.endswith(SENTENCE_END):
        return False
    return not _ABBREVIATION_END.search(text)
# Do not join across a pause longer than this, or into a unit longer than this.
MAX_JOIN_GAP_S = 1.0
MAX_UNIT_DURATION_S = 15.0


def build_units(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            unit = {
                "id": len(units),
                "source_segment_ids": [s["id"] for s in current],
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(s.get("text", "").strip() for s in current).strip(),
                "words": [w for s in current for w in (s.get("words") or [])],
            }
            if any("speaker" in s for s in current):
                unit["speaker"] = current[0].get("speaker")
                unit["speaker_uncertain"] = any(s.get("speaker_uncertain", True) for s in current)
            units.append(unit)
            current.clear()

    for seg in segments:
        if current:
            prev = current[-1]
            sentence_done = ends_sentence(prev.get("text", ""))
            gap = float(seg["start"]) - float(prev["end"])
            too_long = float(seg["end"]) - float(current[0]["start"]) > MAX_UNIT_DURATION_S
            # Never join two people's lines into one sentence.
            other_speaker = seg.get("speaker") is not None and seg.get("speaker") != prev.get("speaker")
            if sentence_done or gap > MAX_JOIN_GAP_S or too_long or other_speaker:
                flush()
        current.append(seg)
    flush()
    return units
