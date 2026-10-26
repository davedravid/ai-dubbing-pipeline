"""How each line is performed: pace, pitch, loudness, style, and stressed words.

The speech phase uses these to match the delivery (and to check it afterwards).
Style is relative to the same speaker: a "raised" line is clearly louder than that
person usually speaks, a "soft" one clearly quieter.
"""

from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from dubbing.analysis.prosody import analyze_clip

RAISED_DB = 6.0
SOFT_DB = -6.0
EMPHASIS_DB = 4.0       # a word this much louder than the line's typical word is stressed
MAX_EMPHASIS = 2


def _rms_db(clip: np.ndarray) -> float | None:
    if len(clip) == 0:
        return None
    return float(20 * np.log10(max(np.sqrt(np.mean(clip ** 2)), 1e-10)))


def measure_line(vocals: np.ndarray, sr: int, line: dict[str, Any]) -> dict[str, Any]:
    clip = vocals[int(line["start"] * sr): int(line["end"] * sr)]
    m = analyze_clip(clip, sr)
    words = line.get("words") or []
    duration = max(line["end"] - line["start"], 0.001)
    word_db = [(w["word"], _rms_db(vocals[int(w["start"] * sr): int(w["end"] * sr)])) for w in words]
    levels = [d for _, d in word_db if d is not None]
    emphasis = []
    if len(levels) >= 3:
        typical = median(levels)
        loud = sorted(((d, w) for w, d in word_db if d is not None and d >= typical + EMPHASIS_DB
                       and len(w.strip(".,!?;:'\"")) >= 3), reverse=True)
        emphasis = [w.strip(".,!?;:'\"") for _, w in loud[:MAX_EMPHASIS]]
    text = line["text"].strip()
    return {
        "words_per_s": round(len(words) / duration, 2) if words else None,
        "pitch_median_hz": m.get("pitch_median_hz"),
        "pitch_range_st": m.get("pitch_range_st"),
        "loudness_db": m.get("loudness_db"),
        "voiced_ratio": m.get("voiced_ratio"),
        "kind": "question" if text.endswith("?") else "exclamation" if text.endswith("!") else "statement",
        "emphasis": emphasis,
    }


def add_styles(lines: list[dict[str, Any]]) -> None:
    """Mark each line raised/soft/normal compared with how that speaker usually sounds."""
    by: dict[str, list[float]] = {}
    for l in lines:
        if l.get("speaker") and l["delivery"].get("loudness_db") is not None:
            by.setdefault(l["speaker"], []).append(l["delivery"]["loudness_db"])
    typical = {s: median(v) for s, v in by.items()}
    for l in lines:
        d = l["delivery"]
        base = typical.get(l.get("speaker") or "")
        if base is None or d.get("loudness_db") is None:
            d["style"] = "unknown"
            continue
        diff = d["loudness_db"] - base
        d["loudness_vs_usual_db"] = round(diff, 1)
        d["style"] = "raised" if diff >= RAISED_DB else "soft" if diff <= SOFT_DB else "normal"
