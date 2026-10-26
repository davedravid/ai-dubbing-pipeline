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
