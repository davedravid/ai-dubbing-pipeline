"""Transcribe speech with timestamps using faster-whisper."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dubbing.config import AsrSettings
from dubbing.errors import PipelineError

log = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    audio_duration_s: float
    speech_duration_s: float | None
    segments: list[dict[str, Any]] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)


def resolve_device(device: str) -> tuple[str, str]:
    """Return (device, compute_type). int8 keeps CPU memory and time down."""
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def transcribe(
    audio_path: Path,
    asr: AsrSettings,
    device: str = "auto",
    language: str | None = None,
    hotwords: str | None = None,
) -> TranscriptionResult:
    """Run Whisper on a WAV file. `language=None` means auto-detect."""
    from faster_whisper import WhisperModel

    device, compute_type = resolve_device(device)
    log.info("Loading Whisper model '%s' on %s (%s). First use downloads it, which can take a while.",
             asr.model, device, compute_type)
    started = time.perf_counter()
    try:
        model = WhisperModel(asr.model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise PipelineError(
            f"Could not load Whisper model '{asr.model}': {exc}\n"
            "Check your internet connection for the first download, and free disk space."
        ) from exc
    log.info("Model ready in %.1fs", time.perf_counter() - started)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=asr.beam_size,
        vad_filter=asr.vad_filter,
        word_timestamps=asr.word_timestamps,
        hotwords=hotwords,  # glossary names/terms to listen for
    )
    log.info("Language: %s (probability %.2f). Audio length %.1fs. Transcribing...",
             info.language, info.language_probability, info.duration)

    segments: list[dict[str, Any]] = []
    # Segments are produced one at a time while Whisper works through the audio.
    for seg in segments_iter:
        item: dict[str, Any] = {
            "id": len(segments),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
        }
        if asr.word_timestamps and seg.words:
            item["words"] = [
                {"start": round(w.start, 3), "end": round(w.end, 3),
                 "word": w.word.strip(), "probability": round(w.probability, 3)}
                for w in seg.words
            ]
        segments.append(item)
        if info.duration:
            log.info("  %5.1f%%  [%7.2f -> %7.2f] %d segments so far",
                     min(100.0, 100 * seg.end / info.duration), seg.start, seg.end, len(segments))

    return TranscriptionResult(
        language=info.language,
        language_probability=round(float(info.language_probability), 4),
        audio_duration_s=round(float(info.duration), 3),
        speech_duration_s=round(float(info.duration_after_vad), 3) if asr.vad_filter else None,
        segments=segments,
        model_info={
            "engine": "faster-whisper",
            "model": asr.model,
            "device": device,
            "compute_type": compute_type,
            "beam_size": asr.beam_size,
            "vad_filter": asr.vad_filter,
            "word_timestamps": asr.word_timestamps,
            "language_forced": language is not None,
            "hotwords": hotwords,
        },
    )
