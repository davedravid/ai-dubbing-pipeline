"""Decide where each dubbed line plays and how much (if at all) to tighten it.

Fixes for a line that is too long for its slot, gentlest first:
  1. use the pause after it (up to the next line's start, minus a small gap);
  2. start it a little early, into the pause before it;
  3. play it slightly faster, pitch unchanged ("tighten"), up to a limit.
Words are never cut. A line that still does not fit is flagged so a person can
write a shorter translation, and the remaining overlap is reported.
Lines are never started LATER than the original, which would drift out of sync.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Placement:
    id: int
    original_start: float
    start: float          # where the line now begins
    duration: float       # length after tightening
    tighten: float        # 1.0 = unchanged; 1.1 = played 10% faster
    early_s: float        # how much earlier than the original it starts
    overlap_s: float      # how far it still runs into the next line (0 = fits)
    flags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_fit(
    lines: list[dict[str, float]],
    total_s: float,
    max_early_s: float = 0.3,
    min_gap_s: float = 0.08,
    max_tighten: float = 1.15,
) -> list[Placement]:
    """`lines`: dicts with id, start, duration (seconds), sorted by start."""
    placements: list[Placement] = []
    prev_end = 0.0
    for i, line in enumerate(lines):
        start, dur = float(line["start"]), float(line["duration"])
        next_start = float(lines[i + 1]["start"]) if i + 1 < len(lines) else total_s
        slot_end = max(next_start - (min_gap_s if i + 1 < len(lines) else 0.0), start)
        flags: list[str] = []

        new_start, early, factor = start, 0.0, 1.0
        if start + dur > slot_end:
            # 2) start early, but not before the previous line has finished (plus a gap)
            earliest = max(start - max_early_s, prev_end + min_gap_s, 0.0)
            if earliest < start:
                early = min(start - earliest, start + dur - slot_end)
                new_start = start - early
                flags.append("started_early")
            room = slot_end - new_start
            # 3) tighten
            if dur > room:
                needed = dur / room if room > 0 else float("inf")
                factor = min(needed, max_tighten)
                flags.append("tightened")
                if needed > max_tighten:
                    flags.append("needs_shorter_translation")

        final_dur = dur / factor
        overlap = max(0.0, new_start + final_dur - next_start) if i + 1 < len(lines) else 0.0
        if overlap > 0:
            flags.append("overlaps_next_line")
        placements.append(Placement(int(line["id"]), start, round(new_start, 3), round(final_dur, 3),
                                    round(factor, 3), round(early, 3), round(overlap, 3), flags))
        prev_end = new_start + final_dur
    return placements
