"""Find who speaks when with pyannote, and label each transcript segment with a speaker.

pyannote/speaker-diarization-3.1 is a gated model: it needs a free Hugging Face account,
its terms accepted, and an access token in the HF_TOKEN environment variable (never in code).
pyannote's usage statistics ("telemetry", on by default) are switched off here.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

from dubbing.errors import PipelineError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
# Gated pages whose terms must be accepted. pyannote.audio 4 also fetches one file
# from the community-1 page even when running the 3.1 pipeline.
REQUIRED_REPOS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0",
                  "pyannote/speaker-diarization-community-1")


def diarize(y: np.ndarray, sr: int, model: str = DEFAULT_MODEL, device: str = "cpu",
            min_speakers: int | None = None, max_speakers: int | None = None) -> list[dict[str, Any]]:
    """Return speaker turns [{start, end, speaker}] for mono audio `y`."""
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"  # must be set before pyannote is imported
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PipelineError(
            "Speaker detection needs a Hugging Face access token in the HF_TOKEN setting.\n"
            "Create one at https://huggingface.co/settings/tokens, accept the terms on "
            + ", ".join(f"https://huggingface.co/{r}" for r in REQUIRED_REPOS)
            + ", then run:  setx HF_TOKEN \"hf_...\"  and open a new terminal."
        )
    import torch
    from pyannote.audio import Pipeline

    started = time.perf_counter()
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as exc:
        raise PipelineError(
            f"Could not load '{model}': {exc}\n"
            "Check that you accepted the terms on ALL of these pages with the same account as the token:\n"
            + "\n".join(f"  https://huggingface.co/{repo}" for repo in REQUIRED_REPOS)
        ) from exc
    if pipeline is None:
        raise PipelineError(f"Access to '{model}' was refused; accept its terms on Hugging Face first.")
    pipeline.to(torch.device(device))

    # Pass the audio itself (not a file path) so pyannote does not need to read files.
    waveform = torch.from_numpy(np.asarray(y, dtype=np.float32))[None]
    kwargs = {k: v for k, v in (("min_speakers", min_speakers), ("max_speakers", max_speakers)) if v}
    log.info("Detecting speakers with %s on %s (%.0fs of audio)...", model, device, len(y) / sr)
    output = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    annotation = getattr(output, "speaker_diarization", output)
    turns = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": str(label)}
             for t, _, label in annotation.itertracks(yield_label=True)]
    log.info("Found %d speakers in %d turns (%.1fs)", len({t["speaker"] for t in turns}), len(turns),
             time.perf_counter() - started)
    return turns


EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_EMBED_S = 0.6        # shorter lines give unreliable voice fingerprints: marked uncertain
MIN_SIMILARITY = 0.25    # a line this unlike its group's average voice is marked uncertain


class VoiceFingerprinter:
    """Turns a stretch of speech into a voice fingerprint (a vector; similar voices point the same way)."""

    def __init__(self, y: np.ndarray, sr: int, device: str = "cpu"):
        os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
        import torch
        from pyannote.audio import Inference, Model

        self._torch, self.y, self.sr = torch, y, sr
        model = Model.from_pretrained(EMBEDDING_MODEL, token=os.environ.get("HF_TOKEN"))
        if model is None:
            raise PipelineError(f"Could not load the voice fingerprint model '{EMBEDDING_MODEL}'.")
        self.inference = Inference(model, window="whole", device=torch.device(device))

    def __call__(self, start: float, end: float) -> np.ndarray:
        clip = self.y[int(start * self.sr): int(end * self.sr)]
        if len(clip) < int(0.4 * self.sr):
            clip = np.pad(clip, (0, int(0.4 * self.sr) - len(clip)))
        wav = self._torch.from_numpy(np.asarray(clip, dtype=np.float32))[None]
        e = np.asarray(self.inference({"waveform": wav, "sample_rate": self.sr})).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)


SPLIT_MIN_LINE_S = 2.0      # only look for a second voice in lines at least this long
SPLIT_MIN_SIDE_S = 0.8      # each side of a split must be at least this long
SPLIT_MIN_PAUSE_S = 0.08    # split only at a (short) pause between words
SPLIT_MAX_SIMILARITY = 0.35 # the two sides must sound this different


def split_two_voice_lines(segments: list[dict[str, Any]], fingerprint) -> list[dict[str, Any]]:
    """Split a line where the voice clearly changes at a pause between words (two people
    in one line). Each side keeps its own words and timings."""
    out: list[dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if seg["end"] - seg["start"] < SPLIT_MIN_LINE_S or len(words) < 2:
            out.append(seg)
            continue
        best = None
        for i in range(len(words) - 1):
            left_end, right_start = words[i]["end"], words[i + 1]["start"]
            if right_start - left_end < SPLIT_MIN_PAUSE_S:
                continue
            if left_end - words[0]["start"] < SPLIT_MIN_SIDE_S or words[-1]["end"] - right_start < SPLIT_MIN_SIDE_S:
                continue
            sim = float(fingerprint(words[0]["start"], left_end) @ fingerprint(right_start, words[-1]["end"]))
            if sim < SPLIT_MAX_SIMILARITY and (best is None or sim < best[1]):
                best = (i, sim)
        if best is None:
            out.append(seg)
            continue
        i = best[0]
        for part in (words[: i + 1], words[i + 1:]):
            out.append({**{k: v for k, v in seg.items() if k not in ("words", "text", "start", "end")},
                        "start": part[0]["start"], "end": part[-1]["end"], "words": part,
                        "text": " ".join(w["word"] for w in part), "split_from_two_voices": round(best[1], 3)})
    for n, seg in enumerate(out):
        seg["id"] = n
    return out


def label_segments_by_voice(y: np.ndarray, sr: int, segments: list[dict[str, Any]],
                            threshold: float = 0.7, device: str = "cpu") -> list[dict[str, Any]]:
    """Group transcript lines by voice: one voice fingerprint per line, then clustering.

    Works better than pyannote's own grouping on fast back-and-forth dialogue, because
    the transcript already says where each line starts and ends. Checked on a 6-person
    argument scene: groups matched voice pitch (the one female voice formed its own group).
    Lines containing two voices are split first."""
    from sklearn.cluster import AgglomerativeClustering

    fp = VoiceFingerprinter(y, sr, device)
    started = time.perf_counter()
    before = len(segments)
    segments = split_two_voice_lines(segments, fp)
    if len(segments) > before:
        log.info("Split %d lines that contained two voices", len(segments) - before)

    def fingerprint(seg: dict[str, Any]) -> np.ndarray:
        return fp(seg["start"], seg["end"])

    reliable = [s for s in segments if s["end"] - s["start"] >= MIN_EMBED_S]
    if len(reliable) < 2:
        return [{**s, "speaker": "S1" if reliable else None, "speaker_confidence": 0.0,
                 "speaker_uncertain": True} for s in segments]
    emb = {s["id"]: fingerprint(s) for s in segments}
    E = np.stack([emb[s["id"]] for s in reliable])
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                     linkage="average").fit_predict(E)
    centroids = {}
    for lab in set(labels):
        c = E[labels == lab].mean(axis=0)
        centroids[lab] = c / (np.linalg.norm(c) + 1e-9)

    names: dict[int, str] = {}
    out = []
    for seg in segments:
        sims = {lab: float(emb[seg["id"]] @ c) for lab, c in centroids.items()}
        best = max(sims, key=sims.get)
        names.setdefault(best, f"S{len(names) + 1}")
        short = seg["end"] - seg["start"] < MIN_EMBED_S
        out.append({**seg, "speaker": names[best], "speaker_confidence": round(sims[best], 3),
                    "speaker_uncertain": bool(short or sims[best] < MIN_SIMILARITY)})
    out = _merge_small_speakers(out, y, sr, emb, centroids, names)

    # Re-number in order of appearance (S1 = first voice heard).
    order: dict[str, str] = {}
    for s in out:
        order.setdefault(s["speaker"], f"S{len(order) + 1}")
        s["speaker"] = order[s["speaker"]]
    log.info("Grouped %d lines into %d voices by fingerprint (%.1fs)", len(segments), len(order),
             time.perf_counter() - started)
    return out


MIN_SPEAKER_S = 4.0       # a "speaker" with less speech than this is probably fingerprint noise
MAX_PITCH_GAP_ST = 5.0    # never merge a line into a speaker whose voice is this much higher/lower


def _line_pitch(y: np.ndarray, sr: int, seg: dict[str, Any]) -> float | None:
    from dubbing.analysis.prosody import analyze_clip

    return analyze_clip(y[int(seg["start"] * sr): int(seg["end"] * sr)], sr).get("pitch_median_hz")


def _merge_small_speakers(lines, y, sr, emb, centroids, names) -> list[dict[str, Any]]:
    """Fold tiny groups (short lines with noisy fingerprints) into the closest real speaker
    whose voice pitch matches; the moved lines are marked uncertain for review."""
    import math

    total: dict[str, float] = defaultdict(float)
    for l in lines:
        total[l["speaker"]] += l["end"] - l["start"]
    big = {s for s, t in total.items() if t >= MIN_SPEAKER_S}
    if not big or len(big) == len(total):
        return lines
    by_name = {names[lab]: lab for lab in names}
    pitch = {l["id"]: _line_pitch(y, sr, l) for l in lines}
    speaker_pitch = {}
    for s in big:
        values = sorted(p for l in lines if l["speaker"] == s and (p := pitch[l["id"]]))
        speaker_pitch[s] = values[len(values) // 2] if values else None
    for l in lines:
        if l["speaker"] in big:
            continue
        p = pitch[l["id"]]
        options = []
        for s in big:
            sp = speaker_pitch.get(s)
            if p and sp and abs(12 * math.log2(p / sp)) > MAX_PITCH_GAP_ST:
                continue  # e.g. never put a woman's line under a man's voice
            options.append((float(emb[l["id"]] @ centroids[by_name[s]]), s))
        if options:
            sim, target = max(options)
            l.update(speaker_reassigned_from=l["speaker"], speaker=target,
                     speaker_confidence=round(sim, 3), speaker_uncertain=True)
    return lines


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def label_segments(segments: list[dict[str, Any]], turns: list[dict[str, Any]],
                   min_confidence: float = 0.7) -> list[dict[str, Any]]:
    """Give each segment the speaker who talks most during its words.

    Confidence = share of the segment's speech time belonging to that speaker. Segments
    below `min_confidence`, or with no speaker found, are marked uncertain."""
    names: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        names.setdefault(t["speaker"], f"S{len(names) + 1}")  # S1, S2... in order of appearance

    out = []
    for seg in segments:
        pieces = [(w["start"], w["end"]) for w in seg.get("words") or []] or [(seg["start"], seg["end"])]
        time_by: dict[str, float] = defaultdict(float)
        for a, b in pieces:
            for t in turns:
                time_by[t["speaker"]] += _overlap(a, b, t["start"], t["end"])
        total = sum(time_by.values())
        if total <= 0:
            out.append({**seg, "speaker": None, "speaker_confidence": 0.0, "speaker_uncertain": True})
            continue
        best = max(time_by, key=time_by.get)
        conf = round(float(time_by[best] / total), 3)
        out.append({**seg, "speaker": names[best], "speaker_confidence": conf,
                    "speaker_uncertain": bool(conf < min_confidence)})
    return out
