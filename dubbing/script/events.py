"""Non-speech sounds (laughs, sighs, gasps) and overlapping speech.

Sounds are recorded with exact times so the mix can keep the actor's original sound
instead of losing it or covering it with dubbed speech. Overlaps (two people talking at
once) are marked: transcription is less reliable there, and the dub must handle them.
"""

from __future__ import annotations

import re
from typing import Any

from dubbing.audio.nonverbal import candidate_regions, is_nonverbal_text

MIN_OVERLAP_S = 0.15

_KINDS = [("laugh", re.compile(r"\b(ha|hah|heh|laugh\w*|chuckl\w*)\b", re.I)),
          ("sigh", re.compile(r"\bsigh\w*\b", re.I)),
          ("gasp", re.compile(r"\bgasp\w*\b", re.I)),
          ("hum", re.compile(r"\b(hm+|mm+|uh+|um+|ah+|oh+)\b", re.I))]


def kind_of_sound(heard: str) -> str:
    for kind, pattern in _KINDS:
        if pattern.search(heard):
            return kind
    return "sound"


def find_sounds(vocals, sr: int, lines: list[dict[str, Any]], heard_fn) -> list[dict[str, Any]]:
    """Audible voice stretches outside every word. Whisper's guess says what kind of sound;
    if it hears real words, the stretch is untranscribed speech and is flagged."""
    words = [(w["start"], w["end"]) for l in lines for w in l.get("words") or []]
    events = []
    for a, b in candidate_regions(vocals, sr, words, []):
        heard = heard_fn(vocals[int(a * sr): int(b * sr)], sr)
        if is_nonverbal_text(heard):
            events.append({"type": kind_of_sound(heard), "start": a, "end": b, "heard": heard})
        else:
            events.append({"type": "untranscribed_speech", "start": a, "end": b, "heard": heard})
    return events


MIN_TURN_S = 0.5   # pyannote adds tiny turns at speaker changes; they are not real overlaps


def overlaps_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Times when two different speakers' turns overlap (from pyannote's speaker turns)."""
    turns = [t for t in turns if t["end"] - t["start"] >= MIN_TURN_S]
    points = sorted([(t["start"], 1, t["speaker"]) for t in turns] + [(t["end"], -1, t["speaker"]) for t in turns],
                    key=lambda p: (p[0], p[1]))
    active: dict[str, int] = {}
    out, start = [], None
    for time, step, speaker in points:
        active[speaker] = active.get(speaker, 0) + step
        talking = [s for s, n in active.items() if n > 0]
        if len(talking) >= 2 and start is None:
            start = time
        elif len(talking) < 2 and start is not None:
            if time - start >= MIN_OVERLAP_S:
                out.append({"type": "overlapping_speech", "start": round(start, 3), "end": round(time, 3)})
            start = None
    return out
