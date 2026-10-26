"""Recover speech that the first transcription pass skipped.

Whisper sometimes skips whole stretches (short lines from other speakers, quiet
greetings). We find stretches where a voice is audible but no word was transcribed,
listen to each stretch on its own (clean voices first, then the original audio), and
add any words found, with their own timings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dubbing.audio.nonverbal import candidate_regions, is_nonverbal_text

log = logging.getLogger(__name__)

PAD_S = 0.3            # listen a little either side of the stretch
MIN_GAP_S = 0.5        # ignore audible stretches shorter than this
MIN_WORD_PROB = 0.3    # drop words Whisper itself is unsure of


def find_gaps(vocals: np.ndarray, sr: int, segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    words = [(w["start"], w["end"]) for s in segments for w in (s.get("words") or [])] or \
            [(s["start"], s["end"]) for s in segments]
    return [(a, b) for a, b in candidate_regions(vocals, sr, words, []) if b - a >= MIN_GAP_S]


def _listen(model, clip: np.ndarray, offset: float, language: str | None) -> list[dict[str, Any]]:
    segments, _ = model.transcribe(clip, language=language, beam_size=5, vad_filter=False,
                                   word_timestamps=True, condition_on_previous_text=False)
    found = []
    for seg in segments:
        words = [{"start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
                  "word": w.word.strip(), "probability": round(w.probability, 3)}
                 for w in (seg.words or []) if w.probability >= MIN_WORD_PROB and w.word.strip()]
        text = " ".join(w["word"] for w in words)
        if words and not is_nonverbal_text(text) and seg.no_speech_prob < 0.6:
            found.append({"start": words[0]["start"], "end": words[-1]["end"], "text": text,
                          "avg_logprob": round(seg.avg_logprob, 4), "no_speech_prob": round(seg.no_speech_prob, 4),
                          "words": words, "recovered": True})
    return found


def fill_gaps(model, vocals16k: np.ndarray, mix16k: np.ndarray | None, segments: list[dict[str, Any]],
              language: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (all segments in time order with recovered ones added, report of each gap)."""
    sr = 16000
    report, added = [], []
    for a, b in find_gaps(vocals16k, sr, segments):
        lo, hi = max(0.0, a - PAD_S), b + PAD_S
        source, found = "vocals", _listen(model, vocals16k[int(lo * sr): int(hi * sr)], lo, language)
        if not found and mix16k is not None:
            source, found = "original_mix", _listen(model, mix16k[int(lo * sr): int(hi * sr)], lo, language)
        # Only keep words that lie inside the gap (not words already transcribed nearby).
        found = [s for s in found if s["end"] > a - 0.05 and s["start"] < b + 0.05]
        report.append({"start": a, "end": b, "recovered": [s["text"] for s in found],
                       "source": source if found else None})
        added += found
    merged = sorted(segments + added, key=lambda s: s["start"])
    for i, s in enumerate(merged):
        s["id"] = i
    if added:
        log.info("Recovered %d missed lines from %d silent-looking stretches", len(added), len(report))
    return merged, report
