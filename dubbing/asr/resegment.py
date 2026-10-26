"""Split long transcript segments into sentences using word timings.

On clean (separated) voices Whisper returns fewer, longer segments, some over 20 s.
Dubbing needs sentence-sized pieces, so segments are cut after sentence-ending
punctuation, at clear pauses, and, if still too long, at the best comma or pause.
Every word keeps its own timing; nothing is dropped or moved.
"""

from __future__ import annotations

from typing import Any

SENTENCE_END = (".", "?", "!", "…")
CLAUSE_END = (",", ";", ":", "—", "-")


def _make(words: list[dict[str, Any]], parent: dict[str, Any]) -> dict[str, Any]:
    seg = {k: v for k, v in parent.items() if k not in ("id", "start", "end", "text", "words")}
    seg.update({"start": words[0]["start"], "end": words[-1]["end"],
                "text": " ".join(w["word"] for w in words).strip(), "words": words})
    return seg


def _split_long(words: list[dict[str, Any]], max_s: float) -> list[list[dict[str, Any]]]:
    """Cut an over-long run of words at its best break: a comma, else the longest pause."""
    if len(words) < 2 or words[-1]["end"] - words[0]["start"] <= max_s:
        return [words]
    best_i, best_score = None, -1.0
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        score = gap + (1.0 if words[i]["word"].endswith(CLAUSE_END) else 0.0)
        # avoid tiny pieces at either end
        if 0 < i < len(words) - 2 and score > best_score:
            best_i, best_score = i, score
    if best_i is None:
        return [words]
    return _split_long(words[: best_i + 1], max_s) + _split_long(words[best_i + 1:], max_s)


def split_segments(segments: list[dict[str, Any]], max_s: float = 10.0, pause_s: float = 0.6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            out.append(dict(seg))
            continue
        pieces, current = [], []
        for i, w in enumerate(words):
            current.append(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is None or w["word"].endswith(SENTENCE_END) or nxt["start"] - w["end"] > pause_s:
                pieces.append(current)
                current = []
        for piece in pieces:
            for part in _split_long(piece, max_s):
                out.append(_make(part, seg))
    for i, seg in enumerate(out):
        seg["id"] = i
    return out
