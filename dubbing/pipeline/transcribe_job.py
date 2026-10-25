"""Stage runner: input video -> extracted audio -> transcript JSON."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing import __version__
from dubbing.asr.transcriber import transcribe
from dubbing.config import Settings
from dubbing.errors import OutputExistsError
from dubbing.media.ffmpeg import extract_audio, find_ffmpeg, probe
from dubbing.paths import JobPaths, validate_job_id

log = logging.getLogger(__name__)

TRANSCRIPT_SCHEMA_VERSION = 1


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Fingerprint of the input file; later phases use it to detect changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file so a crash never leaves a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_transcription(
    input_path: Path,
    job_id: str,
    settings: Settings,
    overwrite: bool = False,
) -> Path:
    """Run the Milestone 1 pipeline and return the path of the transcript JSON."""
    input_path = Path(input_path).resolve()
    job_id = validate_job_id(job_id)
    paths = JobPaths(settings.data_dir, job_id)

    existing = paths.existing_transcripts()
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise OutputExistsError(
            f"Job '{job_id}' already has a transcript ({names}).\n"
            "Use a different --job-id, or add --overwrite to replace it."
        )
    paths.ensure_dirs()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    tools = find_ffmpeg(settings.ffmpeg_path)
    media = probe(tools, input_path)
    log.info("Input: %s (%.1fs, video=%s, audio streams=%d)",
             input_path.name, media.duration_s, media.has_video, media.audio_streams)
    source_hash = file_sha256(input_path)
    timings["probe_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    audio_path = extract_audio(tools, input_path, paths.asr_audio)
    timings["extract_audio_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    result = transcribe(audio_path, settings.asr, settings.device, settings.source_language)
    timings["transcribe_s"] = round(time.perf_counter() - started, 2)

    if not result.segments:
        log.warning("Whisper found no speech in this file. The transcript will have 0 segments.")

    transcript = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(input_path),
            "sha256": source_hash,
            "duration_s": round(media.duration_s, 3),
            "has_video": media.has_video,
            "audio_codec": media.audio_codec,
            "sample_rate": media.sample_rate,
            "channels": media.channels,
        },
        "asr_audio": str(audio_path),
        "asr": result.model_info,
        "language": result.language,
        "language_probability": result.language_probability,
        "audio_duration_s": result.audio_duration_s,
        "speech_duration_s": result.speech_duration_s,
        "segment_count": len(result.segments),
        "timings": timings,
        "segments": result.segments,
    }
    out_path = paths.transcript(result.language)
    write_json(out_path, transcript)
    log.info("Saved transcript: %s (%d segments)", out_path, len(result.segments))
    log.info("Timings: %s", timings)
    return out_path
