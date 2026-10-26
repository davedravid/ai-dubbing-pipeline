"""Stage runner: original audio -> voices-only + background-only tracks, with a leak check."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing import __version__
from dubbing.audio.clips import read_mono, resample
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.media.ffmpeg import extract_audio, find_ffmpeg
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.analyze_job import load_transcript
from dubbing.pipeline.transcribe_job import write_json
from dubbing.separation.demucs_sep import separate

log = logging.getLogger(__name__)

STEREO_SR = 44100


def ensure_stereo_source(paths: JobPaths, transcript_or_path: dict[str, Any] | Path, settings: Settings) -> Path:
    """The 44.1 kHz stereo copy of the original audio (extracted once, then reused)."""
    if paths.source_stereo.is_file():
        return paths.source_stereo
    source = Path(transcript_or_path["source"]["path"]) if isinstance(transcript_or_path, dict) else Path(transcript_or_path)
    if not source.is_file():
        raise PipelineError(f"The original video is no longer at {source}.")
    return extract_audio(find_ffmpeg(settings.ffmpeg_path), source, paths.source_stereo,
                         sample_rate=STEREO_SR, channels=2)


def leak_check(background: Path, settings: Settings, lang: str) -> dict[str, Any]:
    """Listen to the background track with Whisper: any words it can make out are
    voices that leaked through the separation ("ghost" dialogue)."""
    from faster_whisper import WhisperModel

    from dubbing.asr.transcriber import resolve_device

    device, compute_type = resolve_device(settings.device)
    model = WhisperModel(settings.asr.model, device=device, compute_type=compute_type)
    y, sr = read_mono(background)
    segments, _ = model.transcribe(resample(y, sr, 16000), language=lang, beam_size=5, vad_filter=True)
    heard = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
             for s in segments if s.text.strip() and s.no_speech_prob < 0.6]
    return {
        "segments_heard": len(heard),
        "words_heard": sum(len(h["text"].split()) for h in heard),
        "seconds_with_words": round(sum(h["end"] - h["start"] for h in heard), 1),
        "heard": heard,
    }


def run_separation(job_id: str, settings: Settings, overwrite: bool = False,
                   input_path: Path | None = None, leak_lang: str | None = None) -> Path:
    """Separate voices from music/effects. `input_path` is needed when the job has no
    transcript yet (separation now runs before transcription)."""
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    if input_path is None:
        _, transcript = load_transcript(paths, job_id)
        input_path = Path(transcript["source"]["path"])
        leak_lang = leak_lang or transcript["language"]
    if paths.separation_report.exists() and not overwrite:
        raise OutputExistsError(f"{paths.separation_report.name} already exists. Add --overwrite to separate again.")

    timings: dict[str, float] = {}
    started = time.perf_counter()
    if overwrite and paths.source_stereo.is_file():
        paths.source_stereo.unlink()  # our own extracted copy; re-extract from the given input
    source = ensure_stereo_source(paths, Path(input_path), settings)
    timings["extract_s"] = round(time.perf_counter() - started, 2)

    device = "cuda" if settings.device == "cuda" else "cpu"
    info = separate(source, paths.vocals, paths.background, settings.mix.demucs_model, device)
    timings["separate_s"] = info.pop("separate_s")

    leaks = None
    if settings.mix.leak_check:
        started = time.perf_counter()
        leaks = leak_check(paths.background, settings, leak_lang or settings.source_language or "en")
        timings["leak_check_s"] = round(time.perf_counter() - started, 2)
        log.info("Leak check: %d words of dialogue still audible in the background (%.1fs)",
                 leaks["words_heard"], leaks["seconds_with_words"])

    report = {
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_audio": str(source),
        "vocals": str(paths.vocals),
        "background": str(paths.background),
        "separation": info,
        "leak_check": leaks,
        "timings": timings,
    }
    write_json(paths.separation_report, report)
    log.info("Saved voices: %s | background: %s", paths.vocals, paths.background)
    return paths.separation_report
