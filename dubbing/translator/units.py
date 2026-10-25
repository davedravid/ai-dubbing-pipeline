"""Group transcript segments into whole sentences ("dub units") before translating.

Whisper often splits one sentence across segments. Translating the halves separately
gives broken grammar, especially into Hindi, where the verb comes at the end. A unit
is a run of consecutive segments that together form a sentence and is translated,
timed and later spoken as one piece.
"""

from __future__ import annotations

from typing import Any

SENTENCE_END = (".", "?", "!", "…", '"', "'", "।")
# Do not join across a pause longer than this, or into a unit longer than this.
MAX_JOIN_GAP_S = 1.0
MAX_UNIT_DURATION_S = 15.0


def build_units(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            units.append({
                "id": len(units),
                "source_segment_ids": [s["id"] for s in current],
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(s.get("text", "").strip() for s in current).strip(),
                "words": [w for s in current for w in (s.get("words") or [])],
            })
            current.clear()

    for seg in segments:
        if current:
            prev = current[-1]
            ends_sentence = prev.get("text", "").rstrip().endswith(SENTENCE_END)
            gap = float(seg["start"]) - float(prev["end"])
            too_long = float(seg["end"]) - float(current[0]["start"]) > MAX_UNIT_DURATION_S
            if ends_sentence or gap > MAX_JOIN_GAP_S or too_long:
                flush()
        current.append(seg)
    flush()
    return units
