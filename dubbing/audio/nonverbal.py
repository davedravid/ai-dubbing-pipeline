"""Find the actors' non-word sounds (laughs, sighs, breaths, gasps) so they can be kept.

Voice separation puts laughs in the "voices" layer, so replacing the voices with the
dub would silence them. We look for stretches where the original voices are audible
but no transcribed word is spoken, and check each with Whisper: if it hears real words,
the stretch is untranslated speech (reported, not kept, so no English leaks into the dub);
otherwise it is a non-word sound and the original is put back.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

FRAME_S = 0.05
ACTIVE_BELOW_PEAK_DB = 35.0
MIN_SOUND_S = 0.25
WORD_PAD_S = 0.12   # keep clear of the edges of spoken words
# Whisper writes laughs etc. like this; such text is not "words".
_NONVERBAL_TEXT = re.compile(r"^[\s\W]*((ha|he|hah|heh|hm+|mm+|uh+|um+|ah+|oh+|aw+|ugh|whoa|wow|"
                             r"laugh\w*|chuckl\w*|sigh\w*|gasp\w*|breath\w*|music|applause|noise)[\s\W]*)*$",
                             re.IGNORECASE)


def is_nonverbal_text(text: str) -> bool:
    return not text.strip() or bool(_NONVERBAL_TEXT.match(text.strip()))


def candidate_regions(vocals: np.ndarray, sr: int, word_spans: list[tuple[float, float]],
                      busy_spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Stretches where voices are audible, outside spoken words and outside `busy_spans`
    (where dubbed speech plays)."""
    hop = int(FRAME_S * sr)
    n = len(vocals) // hop
    if n == 0:
        return []
    level = 20 * np.log10(np.maximum(np.sqrt((vocals[: n * hop].reshape(n, hop) ** 2).mean(1)), 1e-10))
    active = level > level.max() - ACTIVE_BELOW_PEAK_DB
    blocked = np.zeros(n, dtype=bool)
    for spans, pad in ((word_spans, WORD_PAD_S), (busy_spans, 0.0)):
        for a, b in spans:
            blocked[max(0, int((a - pad) / FRAME_S)): min(n, int((b + pad) / FRAME_S) + 1)] = True
    keep = active & ~blocked
    regions, start = [], None
    for i, k in enumerate(np.append(keep, False)):
        if k and start is None:
            start = i
        elif not k and start is not None:
            if (i - start) * FRAME_S >= MIN_SOUND_S:
                regions.append((round(start * FRAME_S, 2), round(i * FRAME_S, 2)))
            start = None
    return regions


def classify_regions(regions, vocals: np.ndarray, sr: int, heard_fn) -> tuple[list, list]:
    """Split regions into (non-word sounds to keep, untranslated speech to report).
    `heard_fn(clip, sr)` returns the text Whisper hears."""
    keep, speech = [], []
    for a, b in regions:
        clip = vocals[int(a * sr): int(b * sr)]
        text = heard_fn(clip, sr)
        (keep if is_nonverbal_text(text) else speech).append({"start": a, "end": b, "heard": text})
    return keep, speech


def restore_mask(regions: list[dict[str, Any]], n_samples: int, sr: int, ramp_s: float = 0.05) -> np.ndarray:
    """0..1 envelope that is 1 inside kept regions, with short fades so nothing clicks."""
    mask = np.zeros(n_samples, dtype=np.float32)
    for r in regions:
        mask[int(r["start"] * sr): min(n_samples, int(r["end"] * sr))] = 1.0
    ramp = max(1, int(ramp_s * sr))
    return np.clip(np.convolve(mask, np.ones(ramp, dtype=np.float32) / ramp, mode="same"), 0.0, 1.0)
