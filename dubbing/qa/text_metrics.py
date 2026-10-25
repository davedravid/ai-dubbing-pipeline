"""Compare two texts, e.g. what a dubbed line should say vs. what ASR heard."""

from __future__ import annotations

import unicodedata

# Spelling variants that sound the same in Hindi: nukta dot, and chandrabindu vs anusvara.
_HINDI_FOLD = str.maketrans({"़": None, "ँ": "ं"})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(_HINDI_FOLD)
    return "".join(ch for ch in text if ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc"))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_error_rate(expected: str, heard: str) -> float:
    """0.0 = identical, 1.0 = completely different (can exceed 1 if `heard` is much longer)."""
    ref, hyp = normalize_text(expected), normalize_text(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(edit_distance(ref, hyp) / len(ref), 3)
