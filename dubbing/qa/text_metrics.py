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


def chrf(reference: str, candidate: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF), 0-100: how similar two sentences are in wording.
    Used to compare a back-translation with the original sentence."""
    from collections import Counter

    def grams(text: str, n: int) -> Counter:
        t = " ".join(text.lower().split())
        return Counter(t[i:i + n] for i in range(len(t) - n + 1))

    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        ref, hyp = grams(reference, n), grams(candidate, n)
        if not ref or not hyp:
            continue
        match = sum((ref & hyp).values())
        precisions.append(match / sum(hyp.values()))
        recalls.append(match / sum(ref.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 1)


def char_error_rate(expected: str, heard: str) -> float:
    """0.0 = identical, 1.0 = completely different (can exceed 1 if `heard` is much longer)."""
    ref, hyp = normalize_text(expected), normalize_text(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(edit_distance(ref, hyp) / len(ref), 3)
