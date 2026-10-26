"""Separate voices from music and effects with Demucs (Meta, MIT licence).

Demucs splits a song or soundtrack into "stems". We keep two: the voices, and the
sum of all other stems (music, effects, ambience), which becomes the bed the dubbed
voices are mixed onto. Audio is read and written with soundfile rather than
torchaudio, which avoids the Windows audio-library issues seen with XTTS.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from dubbing.errors import PipelineError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "htdemucs"
VOICE_STEM = "vocals"


def separate(
    in_wav: Path,
    vocals_out: Path,
    background_out: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> dict[str, Any]:
    """Write voices-only and background-only WAVs. Returns timing and model details."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    started = time.perf_counter()
    try:
        model = get_model(model_name)
    except Exception as exc:
        raise PipelineError(f"Could not load the Demucs model '{model_name}': {exc}") from exc
    model.eval()
    if VOICE_STEM not in model.sources:
        raise PipelineError(f"Demucs model '{model_name}' has no '{VOICE_STEM}' stem: {model.sources}")

    audio, sr = sf.read(str(in_wav), dtype="float32", always_2d=True)  # (samples, channels)
    if sr != model.samplerate:
        raise PipelineError(f"Demucs needs {model.samplerate} Hz audio, got {sr} Hz")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    wav = torch.from_numpy(audio.T.copy())  # (channels, samples)

    # Demucs works best on audio normalised to zero mean and unit spread; undo afterwards.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    log.info("Separating voices from music with Demucs '%s' on %s (%.0fs of audio)...",
             model_name, device, audio.shape[0] / sr)
    with torch.inference_mode():
        stems = apply_model(model, ((wav - mean) / std)[None], device=device, split=True,
                            overlap=0.25, progress=False)[0]
    stems = stems * std + mean  # (sources, channels, samples)

    v = model.sources.index(VOICE_STEM)
    vocals = stems[v].numpy().T
    background = (stems.sum(0) - stems[v]).numpy().T
    for path, data in ((vocals_out, vocals), (background_out, background)):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), data, sr, subtype="FLOAT")

    return {
        "engine": "demucs",
        "model": model_name,
        "stems": list(model.sources),
        "sample_rate": sr,
        "device": device,
        "separate_s": round(time.perf_counter() - started, 2),
    }
