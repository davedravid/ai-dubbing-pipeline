"""Exact line timings, pauses inside lines, and breath groups.

Whisper's line edges are often padded or clipped. The real start/end is where the
actor's voice (in the separated voices track) becomes audible near the first/last word.
Pauses inside a line are kept so the dub can pause in the same places; "breath groups"
are the phrases between those pauses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FRAME_S = 0.01
EDGE_SEARCH_BEFORE_S = 0.25   # look this far before the first word / after the last word
EDGE_SEARCH_INSIDE_S = 0.30   # ...and this far inside the line
ACTIVE_BELOW_PEAK_DB = 30.0
PAUSE_MIN_S = 0.25            # a gap between words at least this long is a pause
QUIET_RUN = 4                 # 40 ms of silence marks a real start/stop of the voice


class Envelope:
    """Loudness of the voices track every 10 ms, for finding where a voice starts and stops."""

    def __init__(self, y: np.ndarray, sr: int):
        hop = int(FRAME_S * sr)
        n = len(y) // hop
        frames = y[: n * hop].reshape(n, hop)
        self.db = 20 * np.log10(np.maximum(np.sqrt((frames ** 2).mean(axis=1)), 1e-10))

    def _idx(self, t: float) -> int:
        return int(np.clip(round(t / FRAME_S), 0, len(self.db)))

    def refine(self, start: float, end: float, not_before: float = 0.0,
               not_after: float = float("inf")) -> tuple[float, float]:
        """Snap a line's edges to where the voice really starts and stops.

        An edge only moves to a true onset/offset: a voice frame next to a real moment of
        silence. If the voice runs straight on from the previous (or into the next) line,
        the word timing is kept. Edges never cross into the neighbouring lines' words."""
        lo = self._idx(max(start - EDGE_SEARCH_BEFORE_S, not_before))
        hi = self._idx(min(end + EDGE_SEARCH_BEFORE_S, not_after))
        if hi - lo < 2 * QUIET_RUN + 2:
            return start, end
        window = self.db[lo:hi]
        active = window > window.max() - ACTIVE_BELOW_PEAK_DB
        quiet = ~active
        a_hi = min(self._idx(start + EDGE_SEARCH_INSIDE_S) - lo, len(active))
        b_lo = max(self._idx(end - EDGE_SEARCH_INSIDE_S) - lo, 0)

        new_start = start
        for f in range(QUIET_RUN, a_hi):           # first voice frame right after silence
            if active[f] and quiet[f - QUIET_RUN:f].all():
                new_start = (lo + f) * FRAME_S
                break
        new_end = end
        for f in range(len(active) - QUIET_RUN - 1, b_lo - 1, -1):   # last voice frame right before silence
            if active[f] and quiet[f + 1:f + 1 + QUIET_RUN].all():
                new_end = (lo + f + 1) * FRAME_S
                break
        if new_end - new_start < 0.1:
            return start, end
        return round(new_start, 3), round(new_end, 3)


def pauses(words: list[dict[str, Any]], min_s: float = PAUSE_MIN_S) -> list[dict[str, float]]:
    return [{"start": a["end"], "end": b["start"], "length_s": round(b["start"] - a["end"], 3)}
            for a, b in zip(words, words[1:]) if b["start"] - a["end"] >= min_s]


def breath_groups(words: list[dict[str, Any]], min_pause_s: float = PAUSE_MIN_S) -> list[dict[str, Any]]:
    """The phrases between pauses: what the actor says in one go."""
    groups, current = [], []
    for i, w in enumerate(words):
        current.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or nxt["start"] - w["end"] >= min_pause_s:
            groups.append({"start": current[0]["start"], "end": current[-1]["end"],
                           "text": " ".join(x["word"] for x in current)})
            current = []
    return groups
