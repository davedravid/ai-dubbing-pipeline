"""Read audio, cut clips by time, and place clips on a timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def cut(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(round(start_s * sr)))
    b = min(len(y), int(round(end_s * sr)))
    return y[a:b] if b > a else np.zeros(0, dtype=np.float32)


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y
    import librosa

    return librosa.resample(y, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


def write_wav(path: Path, y: np.ndarray, sr: int, comment: str | None = None) -> Path:
    """Write 16-bit WAV. `comment` is stored in the file's metadata (INFO chunk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=1, subtype="PCM_16", format="WAV") as f:
        if comment:
            f.comment = comment
            f.software = "AI Dubbing Pipeline"
        f.write(np.clip(y, -1.0, 1.0))
    return path


def place_on_timeline(
    clips: Iterable[tuple[float, np.ndarray]], sr: int, total_s: float
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mix (start_seconds, audio) clips into one track of length `total_s`.
    Returns the track and the time ranges where clips overlapped."""
    track = np.zeros(int(round(total_s * sr)) + 1, dtype=np.float32)
    covered = np.zeros(len(track), dtype=bool)
    overlaps: list[tuple[float, float]] = []
    for start, y in clips:
        a = int(round(start * sr))
        b = min(len(track), a + len(y))
        if b <= a:
            continue
        clash = np.flatnonzero(covered[a:b])
        if clash.size:
            overlaps.append((round((a + clash[0]) / sr, 3), round((a + clash[-1]) / sr, 3)))
        track[a:b] += y[: b - a]
        covered[a:b] = True
    return track, overlaps
