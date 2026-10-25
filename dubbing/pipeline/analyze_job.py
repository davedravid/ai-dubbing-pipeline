"""Stage runner: original audio -> delivery measurements per sentence unit."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from dubbing import __version__
from dubbing.analysis.prosody import analyze_clip
from dubbing.audio.clips import cut, read_mono
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.media.ffmpeg import extract_audio, find_ffmpeg
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.transcribe_job import write_json
from dubbing.translator.checks import timing_facts
from dubbing.translator.units import build_units

log = logging.getLogger(__name__)

SOURCE_SR = 24000


def ensure_source_audio(paths: JobPaths, transcript: dict[str, Any], settings: Settings) -> Path:
    """Extract the 24 kHz mono copy of the original audio if it is not there yet."""
    if paths.source_audio.is_file():
        return paths.source_audio
    source = Path(transcript["source"]["path"])
    if not source.is_file():
        raise PipelineError(f"The original video is no longer at {source}; it is needed to read the voices.")
    return extract_audio(find_ffmpeg(settings.ffmpeg_path), source, paths.source_audio, sample_rate=SOURCE_SR)


def load_transcript(paths: JobPaths, job_id: str) -> tuple[Path, dict[str, Any]]:
    found = paths.existing_transcripts()
    if len(found) != 1:
        raise PipelineError(f"Expected one transcript for job '{job_id}', found {len(found)}. Run 'transcribe' first.")
    return found[0], json.loads(found[0].read_text(encoding="utf-8"))


def run_analysis(job_id: str, settings: Settings, overwrite: bool = False) -> Path:
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    _, transcript = load_transcript(paths, job_id)
    lang = transcript["language"]
    out_path = paths.features(lang)
    if out_path.exists() and not overwrite:
        raise OutputExistsError(f"{out_path.name} already exists. Add --overwrite to replace it.")

    started = time.perf_counter()
    audio_path = ensure_source_audio(paths, transcript, settings)
    y, sr = read_mono(audio_path)
    units = build_units(transcript.get("segments", []))
    facts = timing_facts(units, float(transcript["source"]["duration_s"]))

    out_units = []
    for unit, fact in zip(units, facts):
        clip = cut(y, sr, unit["start"], unit["end"])
        out_units.append({
            "id": unit["id"],
            "source_segment_ids": unit["source_segment_ids"],
            "start": unit["start"],
            "end": unit["end"],
            "text": unit["text"],
            "words_per_s": fact["source_words_per_s"],
            "pause_before_s": fact["pause_before_s"],
            "pause_after_s": fact["pause_after_s"],
            **analyze_clip(clip, sr),
        })
        if (unit["id"] + 1) % 10 == 0:
            log.info("  analysed %d/%d units", unit["id"] + 1, len(units))

    pitches = [u["pitch_median_hz"] for u in out_units if u.get("pitch_median_hz")]
    result = {
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": lang,
        "source_audio": str(audio_path),
        "note": "Measurements of the ORIGINAL audio. Background music/effects can distort them.",
        "summary": {
            "unit_count": len(out_units),
            "units_with_pitch": len(pitches),
            "median_pitch_hz": round(median(pitches), 1) if pitches else None,
        },
        "timings": {"analyze_s": round(time.perf_counter() - started, 2)},
        "units": out_units,
    }
    write_json(out_path, result)
    log.info("Saved delivery measurements: %s (%d units)", out_path, len(out_units))
    return out_path
