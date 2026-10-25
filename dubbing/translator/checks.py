"""Per-segment checks on translations, plus timing facts the dubbing stage needs."""

from __future__ import annotations

from typing import Any

from dubbing.languages import Language, script_fraction

# Below this share of letters in the target script, the output is probably untranslated.
MIN_SCRIPT_FRACTION = 0.6
# Translation length / source length outside this range usually means a bad output.
LENGTH_RATIO_RANGE = (0.4, 3.0)
# Dubbed speech can be sped up a little without sounding unnatural; beyond this, flag it.
MAX_COMFORTABLE_SPEEDUP = 1.25
# Speaker pace relative to average is clamped to this range.
PACE_RANGE = (0.7, 1.8)


def _has_repetition(text: str, run: int = 4) -> bool:
    """Same word `run` or more times in a row: a known way translation models fail."""
    words = [w.strip(".,!?;:।\"'") for w in text.split()]
    streak = 1
    for prev, cur in zip(words, words[1:]):
        streak = streak + 1 if cur and cur == prev else 1
        if streak >= run:
            return True
    return False


def timing_facts(segments: list[dict[str, Any]], media_duration: float) -> list[dict[str, float]]:
    """For each segment: its length, the room until the next line, and the speaker's pace."""
    facts = []
    for i, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        prev_end = float(segments[i - 1]["end"]) if i > 0 else 0.0
        next_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else max(media_duration, end)
        duration = max(end - start, 0.001)
        text = seg.get("text", "")
        words = len(seg.get("words") or text.split())
        facts.append({
            "duration_s": round(duration, 3),
            # The dub may run into the silence before the next line, but not past it.
            "available_s": round(max(next_start, end) - start, 3),
            "pause_before_s": round(max(start - prev_end, 0.0), 3),
            "pause_after_s": round(max(next_start - end, 0.0), 3),
            "source_words_per_s": round(words / duration, 2),
            "source_chars_per_s": round(len(text) / duration, 2),
        })
    return facts


def _has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def speaker_pace(source_chars_per_s: float, source: Language) -> float:
    """How fast this speaker talks compared with an average speaker (1.0 = average).
    Clamped, because timings of very short segments are noisy."""
    return min(max(source_chars_per_s / source.chars_per_second, PACE_RANGE[0]), PACE_RANGE[1])


def check_translation(
    source_text: str,
    target_text: str,
    source: Language,
    target: Language,
    duration_s: float,
    available_s: float,
    source_chars_per_s: float,
) -> tuple[list[str], dict[str, float]]:
    """Return (flags, estimates). Flags are short codes explained in the README."""
    flags: list[str] = []
    stripped = target_text.strip()
    if not stripped:
        return ["empty_translation"], {}

    if not _has_letters(stripped):
        # Only digits/punctuation, e.g. "12.": fine as text, but speech needs words.
        flags.append("numbers_only")
    else:
        if script_fraction(stripped, target.script) < MIN_SCRIPT_FRACTION:
            flags.append("wrong_script")
        if stripped == source_text.strip():
            flags.append("unchanged")

    ratio = len(stripped) / max(len(source_text.strip()), 1)
    if not LENGTH_RATIO_RANGE[0] <= ratio <= LENGTH_RATIO_RANGE[1]:
        flags.append("unusual_length")

    if _has_repetition(stripped):
        flags.append("repetition")

    # Assume the speaker keeps their own pace in the new language.
    pace = speaker_pace(source_chars_per_s, source)
    est_duration = len(stripped) / (target.chars_per_second * pace)
    speedup = est_duration / max(duration_s, 0.001)
    if est_duration / max(available_s, 0.001) > MAX_COMFORTABLE_SPEEDUP:
        flags.append("may_not_fit")

    return flags, {
        "length_ratio": round(ratio, 2),
        "speaker_pace": round(pace, 2),
        "est_target_duration_s": round(est_duration, 2),
        "est_speedup_needed": round(speedup, 2),
    }
