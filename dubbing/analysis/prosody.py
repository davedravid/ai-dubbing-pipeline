"""Measure delivery of a speech clip: pitch, loudness and how much of it is voiced.

"Prosody" is the music of speech: how high or low, loud or soft, fast or slow it is.
These numbers let us describe the original line and check the dubbed line against it.
They are measurements, not emotion labels. On film audio, background music and
effects can distort them; vocal separation (Phase 7) makes them cleaner.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Typical adult speaking pitch lies between these (Hz).
PITCH_MIN_HZ = 65.0
PITCH_MAX_HZ = 500.0
# Frames quieter than this (dB below the clip's loudest frame) count as silence.
SILENCE_BELOW_PEAK_DB = 35.0
FRAME = 1024
HOP = 256


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, 1e-10))


def analyze_clip(y: np.ndarray, sr: int, with_pitch: bool = True) -> dict[str, Any]:
    """Return delivery measurements for mono audio `y` (float samples in -1..1)."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    duration = len(y) / sr if sr else 0.0
    if len(y) < FRAME:
        return {"duration_s": round(duration, 3), "too_short": True}

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = _db(rms)
    active = rms_db > (rms_db.max() - SILENCE_BELOW_PEAK_DB)
    result: dict[str, Any] = {
        "duration_s": round(duration, 3),
        "loudness_db": round(float(_db(np.sqrt(np.mean(rms[active] ** 2)))), 1) if active.any() else None,
        "peak_db": round(float(_db(np.max(np.abs(y)))), 1),
        "active_ratio": round(float(active.mean()), 3),
    }

    if with_pitch:
        f0, voiced, _ = librosa.pyin(y, fmin=PITCH_MIN_HZ, fmax=PITCH_MAX_HZ, sr=sr,
                                     frame_length=FRAME, hop_length=HOP)
        f0 = f0[voiced & ~np.isnan(f0)]
        result["voiced_ratio"] = round(float(voiced.mean()), 3)
        if len(f0) >= 5:
            p10, p50, p90 = np.percentile(f0, [10, 50, 90])
            result.update({
                "pitch_median_hz": round(float(p50), 1),
                "pitch_p10_hz": round(float(p10), 1),
                "pitch_p90_hz": round(float(p90), 1),
                # Semitones: the musical distance between low and high; wider = more expressive.
                "pitch_range_st": round(12 * math.log2(p90 / p10), 1),
            })
        else:
            result["pitch_median_hz"] = None
    return result


def compare(source: dict[str, Any], dubbed: dict[str, Any]) -> dict[str, Any]:
    """How the dubbed line differs from the original (positive = dub is higher/louder/longer)."""
    out: dict[str, Any] = {}
    if source.get("pitch_median_hz") and dubbed.get("pitch_median_hz"):
        out["pitch_shift_st"] = round(12 * math.log2(dubbed["pitch_median_hz"] / source["pitch_median_hz"]), 1)
    if source.get("pitch_range_st") is not None and dubbed.get("pitch_range_st") is not None:
        out["pitch_range_diff_st"] = round(dubbed["pitch_range_st"] - source["pitch_range_st"], 1)
    if source.get("loudness_db") is not None and dubbed.get("loudness_db") is not None:
        out["loudness_diff_db"] = round(dubbed["loudness_db"] - source["loudness_db"], 1)
    return out
