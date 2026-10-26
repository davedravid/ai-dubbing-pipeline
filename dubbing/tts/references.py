"""Choose and cut the original audio each dubbed line is cloned from.

Cloning from the line's OWN original audio keeps both the voice and the delivery
(tone, energy, emotion) of that exact moment. Lines too short to clone from
(XTTS needs a few seconds) borrow the nearest long-enough line. Until speaker
detection exists (Phase 6) that neighbour may be a DIFFERENT person, so such
lines are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dubbing.audio.clips import cut, write_wav

PAD_S = 0.1  # a little context either side of the line
CLIP_LEVEL = 0.99
TARGET_PEAK = 0.89   # about -1 dB below the maximum
MAX_GAIN_DB = 24.0


@dataclass(frozen=True)
class ReferenceChoice:
    unit_id: int
    ref_unit_id: int | None   # None = no usable reference; use the built-in voice
    borrowed: bool
    distance_s: float | None


SPEAKER_REF_MAX_S = 30.0     # XTTS uses up to about 30 s of reference audio
SPEAKER_REF_MAX_CLIPS = 8
SPEAKER_REF_MIN_CLIP_S = 1.5


def build_speaker_references(units: list[dict[str, Any]], y: np.ndarray, sr: int,
                             out_dir: Path) -> dict[str, dict[str, Any]]:
    """For each detected speaker, cut their longest confidently-labelled lines (up to
    SPEAKER_REF_MAX_S in total) as one shared voice reference. Same person = same voice."""
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for u in units:
        if u.get("speaker") and not u.get("speaker_uncertain", True) and u["end"] - u["start"] >= SPEAKER_REF_MIN_CLIP_S:
            by_speaker.setdefault(u["speaker"], []).append(u)
    refs: dict[str, dict[str, Any]] = {}
    for speaker, lines in by_speaker.items():
        chosen, total = [], 0.0
        for u in sorted(lines, key=lambda u: u["end"] - u["start"], reverse=True):
            if total >= SPEAKER_REF_MAX_S or len(chosen) >= SPEAKER_REF_MAX_CLIPS:
                break
            chosen.append(u)
            total += u["end"] - u["start"]
        clips = []
        for u in sorted(chosen, key=lambda u: u["start"]):
            path, quality = cut_reference(y, sr, u, out_dir / f"speaker_{speaker}_unit_{u['id']:03d}.wav")
            clips.append({"unit_id": u["id"], "path": str(path), **quality})
        refs[speaker] = {"seconds": round(total, 1), "clips": clips}
    return refs


def plan_references(units: list[dict[str, Any]], min_ref_s: float, max_borrow_s: float) -> list[ReferenceChoice]:
    long_enough = [u for u in units if u["end"] - u["start"] >= min_ref_s and u.get("text")]
    plan = []
    for u in units:
        if u["end"] - u["start"] >= min_ref_s:
            plan.append(ReferenceChoice(u["id"], u["id"], False, 0.0))
            continue
        mid = (u["start"] + u["end"]) / 2
        best = min(long_enough, key=lambda c: abs((c["start"] + c["end"]) / 2 - mid), default=None)
        if best is not None:
            distance = abs((best["start"] + best["end"]) / 2 - mid)
            if distance <= max_borrow_s:
                plan.append(ReferenceChoice(u["id"], best["id"], True, round(distance, 2)))
                continue
        plan.append(ReferenceChoice(u["id"], None, False, None))
    return plan


def reference_quality(clip: np.ndarray, sr: int) -> dict[str, Any]:
    from dubbing.analysis.prosody import analyze_clip

    q = analyze_clip(clip, sr, with_pitch=False)
    q["clipped_ratio"] = round(float(np.mean(np.abs(clip) >= CLIP_LEVEL)), 4) if len(clip) else 0.0
    warnings = []
    if q.get("active_ratio", 0) < 0.5:
        warnings.append("mostly_silence")
    if q["clipped_ratio"] > 0.001:
        warnings.append("clipped")
    if (q.get("loudness_db") or -99) < -35:
        warnings.append("very_quiet")
    q["warnings"] = warnings
    return q


def normalize_peak(clip: np.ndarray, target_peak: float = TARGET_PEAK, max_gain_db: float = MAX_GAIN_DB) -> tuple[np.ndarray, float]:
    """Raise quiet audio so its loudest point reaches `target_peak` (never lowers it,
    never boosts more than `max_gain_db`). Returns the audio and the gain applied in dB."""
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak <= 0:
        return clip, 0.0
    gain = min(target_peak / peak, 10 ** (max_gain_db / 20))
    if gain <= 1.0:
        return clip, 0.0
    return (clip * gain).astype(np.float32), round(20 * np.log10(gain), 1)


def cut_reference(y: np.ndarray, sr: int, unit: dict[str, Any], out_path: Path) -> tuple[Path, dict[str, Any]]:
    clip = cut(y, sr, unit["start"] - PAD_S, unit["end"] + PAD_S)
    quality = reference_quality(clip, sr)  # measured on the original, before any boost
    # Film dialogue is often mixed quietly; a weak sample gives XTTS a weak voice to copy.
    boosted, gain_db = normalize_peak(clip)
    quality["gain_applied_db"] = gain_db
    write_wav(out_path, boosted, sr, comment=f"Voice reference cut from original audio, unit {unit['id']}")
    return out_path, quality
