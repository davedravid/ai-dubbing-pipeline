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


def _separation_matches(paths: JobPaths, input_path: Path) -> bool:
    """True if this job already has voices/background separated from this same input file."""
    if not (paths.vocals.is_file() and paths.background.is_file() and paths.separation_report.is_file()):
        return False
    report = json.loads(paths.separation_report.read_text(encoding="utf-8"))
    return report.get("input_path") == str(input_path) and report.get("input_size") == input_path.stat().st_size


def _plain(value: Any) -> Any:
    """Turn NumPy numbers/booleans/arrays (from audio and model libraries) into plain Python."""
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot save a {type(value).__name__} in a JSON file")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file so a crash never leaves a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_plain), encoding="utf-8")
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

    if settings.asr.from_separated_vocals:
        # Music and effects confuse speech recognition; listen to the voices alone instead.
        from dubbing.audio.clips import read_mono, resample, write_wav
        from dubbing.pipeline.separate_job import run_separation

        # Re-use an earlier separation of the same video ('separate --overwrite' forces a new one).
        if not _separation_matches(paths, input_path):
            run_separation(job_id, settings, overwrite=True, input_path=input_path,
                           leak_lang=settings.source_language or "en")
        vocals, sr = read_mono(paths.vocals)
        audio_path = write_wav(paths.asr_audio, resample(vocals, sr, 16000), 16000)
        asr_input = "separated_vocals"
    else:
        audio_path = extract_audio(tools, input_path, paths.asr_audio)
        asr_input = "original_mix"
    timings["prepare_audio_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    result = transcribe(audio_path, settings.asr, settings.device, settings.source_language)
    timings["transcribe_s"] = round(time.perf_counter() - started, 2)

    gap_report: list[dict[str, Any]] | None = None
    if settings.asr.from_separated_vocals and settings.asr.fill_gaps and settings.asr.word_timestamps:
        from faster_whisper import WhisperModel

        from dubbing.asr.gapfill import fill_gaps
        from dubbing.asr.transcriber import resolve_device
        from dubbing.audio.clips import read_mono

        started = time.perf_counter()
        device, compute_type = resolve_device(settings.device)
        model = WhisperModel(settings.asr.model, device=device, compute_type=compute_type)
        vocals16k, _ = read_mono(audio_path)
        mix_path = extract_audio(tools, input_path, paths.temp_dir / f"{job_id}_mix16k.wav")
        mix16k, _ = read_mono(mix_path)
        result.segments, gap_report = fill_gaps(model, vocals16k, mix16k, result.segments,
                                                settings.source_language or result.language)
        timings["fill_gaps_s"] = round(time.perf_counter() - started, 2)

    if not result.segments:
        log.warning("Whisper found no speech in this file. The transcript will have 0 segments.")
    elif settings.asr.split_long_segments and settings.asr.word_timestamps:
        from dubbing.asr.resegment import split_segments

        before = len(result.segments)
        result.segments = split_segments(result.segments, settings.asr.max_segment_s, settings.asr.split_pause_s)
        log.info("Split %d Whisper segments into %d sentence-sized segments", before, len(result.segments))

    speakers_info: dict[str, Any] | None = None
    if settings.diarization.enabled and result.segments:
        from dubbing.audio.clips import read_mono
        from dubbing.diarization.speakers import diarize, label_segments

        started = time.perf_counter()
        cached = json.loads(paths.diarization.read_text(encoding="utf-8")) if paths.diarization.is_file() else None
        if cached and cached.get("source_sha256") == source_hash \
                and cached.get("model") == settings.diarization.model:
            turns = cached["turns"]  # same video, same model: reuse (it takes minutes on a CPU)
            log.info("Re-using speaker detection from %s", paths.diarization.name)
        else:
            y16, sr16 = read_mono(audio_path)
            turns = diarize(y16, sr16, settings.diarization.model, "cuda" if settings.device == "cuda" else "cpu",
                            settings.diarization.min_speakers, settings.diarization.max_speakers)
            write_json(paths.diarization, {"job_id": job_id, "model": settings.diarization.model,
                                           "source_sha256": source_hash, "turns": turns})
        result.segments = label_segments(result.segments, turns, settings.diarization.min_confidence)
        timings["diarize_s"] = round(time.perf_counter() - started, 2)
        per = {}
        for seg in result.segments:
            if seg["speaker"]:
                p = per.setdefault(seg["speaker"], {"segments": 0, "speech_s": 0.0})
                p["segments"] += 1
                p["speech_s"] = round(p["speech_s"] + seg["end"] - seg["start"], 2)
        speakers_info = {
            "model": settings.diarization.model,
            "speaker_count": len(per),
            "speakers": per,
            "uncertain_segments": [s["id"] for s in result.segments if s["speaker_uncertain"]],
        }
        log.info("Speakers: %s; %d segments uncertain", per, len(speakers_info["uncertain_segments"]))

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
        "asr_input": asr_input,
        "asr": result.model_info,
        "recovered_gaps": gap_report,
        "diarization": speakers_info,
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
