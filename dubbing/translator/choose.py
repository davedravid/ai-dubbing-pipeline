"""Choose the best of several translations for dubbing.

A good dubbing line keeps the meaning (checked by translating it back to the source
language), fits the time the actor took, sounds conversational, and has no source-language
words left in it. Each option gets a score; the highest wins.
"""

from __future__ import annotations

import re
from typing import Any

from dubbing.qa.text_metrics import chrf
from dubbing.translator.spoken import formality_score

FORMAL_PENALTY = 5        # per formal word
OVERRUN_PENALTY = 30      # per 100% the estimated speech exceeds 1.25x the time available
LATIN_PENALTY = 25        # per English word left in the Hindi
FIT_SLACK = 1.25          # the voice/mix steps can absorb up to 25% extra
RUNAWAY_PENALTY = 100     # repeated words or far too long: almost never chosen


def estimated_seconds(text: str, chars_per_second: float, pace: float) -> float:
    return len(text) / (chars_per_second * pace)


def _repeats(text: str, run: int = 3) -> bool:
    """The same word `run`+ times in a row beyond what the source has: a known model failure."""
    words = [w.strip(".,!?;:।\"'") for w in text.split()]
    streak = 1
    for a, b in zip(words, words[1:]):
        streak = streak + 1 if a and a == b else 1
        if streak >= run:
            return True
    return False


def score_option(text: str, source: str, back: str, slot_s: float, chars_per_second: float,
                 pace: float, penalize_latin: bool = True) -> dict[str, Any]:
    meaning = chrf(source, back)
    est = estimated_seconds(text, chars_per_second, pace)
    over = max(0.0, est / max(slot_s, 0.3) - FIT_SLACK)
    # English words left in a non-Latin-script line (Hindi). For Spanish etc. Latin letters are normal.
    latin = len(re.findall(r"[A-Za-z]{2,}", text)) if penalize_latin else 0
    # Repeating far more than the English did ("No, no, no." -> thirteen "नो"), or being far
    # longer than the English, means the model ran away.
    runaway = (_repeats(text, run=max(3, _longest_run(source) + 1))
               or len(text) > max(3.0 * len(source), len(source) + 30))
    score = (meaning - FORMAL_PENALTY * formality_score(text) - OVERRUN_PENALTY * over
             - LATIN_PENALTY * latin - (RUNAWAY_PENALTY if runaway else 0))
    return {"text": text, "back_translation": back, "meaning": meaning, "est_s": round(est, 2),
            "latin_words": latin, "runaway": runaway, "score": round(score, 1)}


def _longest_run(text: str) -> int:
    words = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    best = streak = 1 if words else 0
    for a, b in zip(words, words[1:]):
        streak = streak + 1 if a and a == b else 1
        best = max(best, streak)
    return best


def choose(options: list[str], source: str, backs: dict[str, str], slot_s: float,
           chars_per_second: float, pace: float, penalize_latin: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (chosen option, all options scored best-first)."""
    scored = [score_option(t, source, backs.get(t, ""), slot_s, chars_per_second, pace, penalize_latin)
              for t in options if t]
    scored.sort(key=lambda o: o["score"], reverse=True)
    if not scored:
        empty = {"text": "", "back_translation": "", "meaning": 0.0, "est_s": 0.0, "latin_words": 0,
                 "runaway": False, "score": -999}
        return empty, []
    return scored[0], scored
