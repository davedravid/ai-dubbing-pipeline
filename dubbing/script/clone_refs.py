"""Pick the best clean samples of each speaker's voice for voice cloning.

A good sample: one speaker only (confidently labelled), nobody talking over them, no
laughs or other sounds inside, little music under it, long enough. Each speaker gets a
listenable preview so a person can confirm it is the right voice, and a grade saying
whether there is enough clean audio to clone well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dubbing.audio.clips import resample, write_wav
from dubbing.tts.references import normalize_peak

MIN_CLIP_S = 1.5
MAX_CLIP_S = 12.0
TARGET_TOTAL_S = 30.0
GOOD_TOTAL_S = 15.0
OK_TOTAL_S = 8.0
MAX_MUSIC_UNDER_DB = -12.0   # background this close to the voice level makes a muddy sample
PREVIEW_SR = 24000


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(x ** 2)) if len(x) else 0.0, 1e-10)))


def _overlaps(a0: float, a1: float, spans: list[dict[str, Any]], min_s: float = 0.05,
              min_share: float = 1.0) -> bool:
    """True if any span covers more than `min_s` seconds, or more than `min_share` of the clip."""
    for s in spans:
        shared = min(a1, s["end"]) - max(a0, s["start"])
        if shared > min_s or shared > min_share * (a1 - a0):
            return True
    return False


def choose_samples(lines: list[dict[str, Any]], events: list[dict[str, Any]], vocals: np.ndarray,
                   background: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for l in lines:
        if not l.get("speaker"):
            continue
        reasons = []
        dur = l["end"] - l["start"]
        if l.get("speaker_uncertain"):
            reasons.append("speaker_uncertain")
        if dur < MIN_CLIP_S:
            reasons.append("too_short")
        # A real overlap (over 0.4 s, or a quarter of the line) makes the sample two voices at once.
        if _overlaps(l["start"], l["end"], [e for e in events if e["type"] == "overlapping_speech"],
                     min_s=0.4, min_share=0.25):
            reasons.append("overlapping_speech")
        if _overlaps(l["start"], l["end"], [e for e in events if e["type"] != "overlapping_speech"]):
            reasons.append("sound_inside")
        a, b = int(l["start"] * sr), int(min(l["end"], l["start"] + MAX_CLIP_S) * sr)
        music_under = _db(background[a:b]) - _db(vocals[a:b])
        if music_under > MAX_MUSIC_UNDER_DB:
            reasons.append("music_under_voice")
        by.setdefault(l["speaker"], []).append({"line_id": l["id"], "start": l["start"],
                                                "end": min(l["end"], l["start"] + MAX_CLIP_S),
                                                "music_under_db": round(music_under, 1), "rejected": reasons})
    out = {}
    for spk, cands in by.items():
        usable = sorted((c for c in cands if not c["rejected"]), key=lambda c: c["end"] - c["start"], reverse=True)
        chosen, total = [], 0.0
        for c in usable:
            if total >= TARGET_TOTAL_S:
                break
            chosen.append(c)
            total += c["end"] - c["start"]
        grade = "good" if total >= GOOD_TOTAL_S else "ok" if total >= OK_TOTAL_S else "weak"
        out[spk] = {"clean_seconds": round(total, 1), "grade": grade,
                    "samples": sorted(chosen, key=lambda c: c["start"]),
                    "rejected": {r: sum(1 for c in cands if r in c["rejected"])
                                 for r in ("speaker_uncertain", "too_short", "overlapping_speech",
                                           "sound_inside", "music_under_voice")}}
    return out


def write_previews(samples: dict[str, dict[str, Any]], vocals: np.ndarray, sr: int, out_dir: Path) -> None:
    """One WAV per speaker: their chosen samples back to back, to listen and confirm the voice."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(0.3 * PREVIEW_SR), dtype=np.float32)
    for spk, info in samples.items():
        parts = []
        for c in info["samples"]:
            clip = resample(vocals[int(c["start"] * sr): int(c["end"] * sr)], sr, PREVIEW_SR)
            parts += [normalize_peak(clip)[0], gap]
        if parts:
            path = write_wav(out_dir / f"{spk}.wav", np.concatenate(parts), PREVIEW_SR,
                             comment=f"Voice samples chosen for cloning speaker {spk} (original audio, local use only)")
            info["preview"] = str(path)
