"""Loudness and clean-audio measurements for whole soundtracks.

Loudness is measured in LUFS (the broadcast standard, EBU R128 / ITU-R BS.1770):
it follows how loud people *perceive* sound, not just the raw signal level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CLIP_LEVEL = 0.999
SILENCE_DBFS = -50.0
FRAME_S = 0.05


def integrated_lufs(y: np.ndarray, sr: int) -> float | None:
    import pyloudnorm as pyln

    if len(y) < sr * 0.5:
        return None
    value = pyln.Meter(sr).integrated_loudness(y)
    return None if not np.isfinite(value) else round(float(value), 2)


def soundtrack_metrics(y: np.ndarray, sr: int) -> dict[str, Any]:
    """`y`: (samples,) mono or (samples, channels)."""
    mono = y.mean(axis=1) if y.ndim == 2 else y
    frame = max(1, int(FRAME_S * sr))
    n = len(mono) // frame
    rms = np.sqrt(np.mean(mono[: n * frame].reshape(n, frame) ** 2, axis=1)) if n else np.zeros(0)
    level = 20 * np.log10(np.maximum(rms, 1e-10))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return {
        "loudness_lufs": integrated_lufs(y, sr),
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-10)), 2),
        "clipped_ratio": round(float(np.mean(np.abs(y) >= CLIP_LEVEL)), 6) if y.size else 0.0,
        "silence_ratio": round(float(np.mean(level < SILENCE_DBFS)), 4) if n else 1.0,
        "duration_s": round(len(mono) / sr, 3),
    }
