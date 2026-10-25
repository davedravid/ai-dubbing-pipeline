"""Stage runner: translation JSON -> one dubbed WAV per sentence unit + a speech preview track."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dubbing import __version__
from dubbing.analysis.prosody import analyze_clip, compare
from dubbing.audio.clips import cut, place_on_timeline, read_mono, resample, write_wav
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.languages import Language, get_language
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.analyze_job import ensure_source_audio, load_transcript
from dubbing.pipeline.transcribe_job import file_sha256, write_json
from dubbing.qa.text_metrics import char_error_rate
from dubbing.tts.references import cut_reference, plan_references
from dubbing.tts.text_norm import normalize_for_speech
from dubbing.tts.xtts import LICENSE_NOTE, Voice, XttsSynthesizer

log = logging.getLogger(__name__)

TRIM_TOP_DB = 40          # silence threshold for trimming the start/end of generated speech
TRIM_PAD_S = 0.05
OVERRUN_TOLERANCE_S = 0.05
LOW_INTELLIGIBILITY_CER = 0.35
# A take that fails the checks is generated again with a new random seed, up to this many takes.
MAX_TAKES = 3
FLATTER_THAN_ORIGINAL_ST = -4.0
# XTTS sometimes keeps "talking" after a short line. A take this many times longer than the
# text needs (at normal pace) is treated as invented speech and rejected.
MAX_DURATION_FACTOR = 2.5
MIN_SUSPECT_DURATION_S = 1.5


def _trim(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    if len(y) == 0:
        return y
    _, (a, b) = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    pad = int(TRIM_PAD_S * sr)
    return y[max(0, a - pad): min(len(y), b + pad)]


def expected_duration_s(text: str, lang: Language) -> float:
    return len(text) / lang.chars_per_second


def is_invented_speech(duration_s: float, speed: float, text: str, lang: Language) -> bool:
    """True if a take is far longer than its text could need: XTTS invented extra speech."""
    limit = max(MIN_SUSPECT_DURATION_S, expected_duration_s(text, lang) * MAX_DURATION_FACTOR / speed)
    return duration_s > limit


class _BackChecker:
    """Transcribes dubbed lines with Whisper to check they can be understood."""

    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        from dubbing.asr.transcriber import resolve_device

        device, compute_type = resolve_device(settings.device)
        self.model = WhisperModel(settings.asr.model, device=device, compute_type=compute_type)

    def heard(self, y: np.ndarray, sr: int, lang: str) -> str:
        segments, _ = self.model.transcribe(resample(y, sr, 16000), language=lang, beam_size=5, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


def _generate_line(
    synth: XttsSynthesizer,
    checker: _BackChecker | None,
    settings: Settings,
    text: str,
    target: Language,
    voice: Voice,
    available_s: float,
) -> dict[str, Any]:
    """Make up to MAX_TAKES takes of one line and return the best.

    Each take is generated at normal speed and, if too long for its slot, once more faster.
    A take passes if it is not invented speech and the back-check understood it."""
    sr = synth.sample_rate
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for take in range(MAX_TAKES if checker is not None else 1):
        speed = 1.0
        y = _trim(synth.speak(text, target.xtts, voice, speed, seed_offset=take), sr)
        if len(y) / sr > available_s and settings.tts.max_speed > 1.0:
            speed = round(min(settings.tts.max_speed, len(y) / sr / available_s * 1.03), 2)
            y = _trim(synth.speak(text, target.xtts, voice, speed, seed_offset=take), sr)
        duration = len(y) / sr
        invented = is_invented_speech(duration, speed, text, target)
        heard, cer = None, None
        if checker is not None:
            heard = checker.heard(y, sr, target.code)
            cer = char_error_rate(text, heard)
        passed = not invented and (cer is None or cer <= LOW_INTELLIGIBILITY_CER)
        attempts.append({"take": take, "speed": speed, "duration_s": round(duration, 3),
                         "char_error_rate": cer, "invented_speech": invented, "passed": passed})
        # Prefer takes without invented speech, then the lowest error rate.
        score = (invented, cer if cer is not None else 0.0)
        if best is None or score < best["score"]:
            best = {"y": y, "speed": speed, "heard": heard, "cer": cer, "invented": invented,
                    "passed": passed, "score": score}
        if passed:
            break
    assert best is not None
    best["attempts"] = attempts
    return best


def _load_previous(manifest_path: Path, translation_sha: str) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise PipelineError("--redo-flagged needs an earlier 'speak' run for this job and language.")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("source_translation_sha256") != translation_sha:
        raise PipelineError("The translation changed since the last 'speak' run; run 'speak --overwrite' instead.")
    return previous


def _needs_redo(entry: dict[str, Any] | None) -> bool:
    if entry is None or entry.get("status") != "generated":
        return True
    return bool({"low_intelligibility", "invented_speech"} & set(entry.get("flags", [])))


def run_speech(
    job_id: str,
    settings: Settings,
    lang: str = "hi",
    overwrite: bool = False,
    limit: int | None = None,
    voice_mode: str | None = None,
    redo_flagged: bool = False,
) -> Path:
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    target = get_language(lang)
    if target.xtts is None:
        raise PipelineError(
            f"XTTS v2 cannot speak {target.name}. A fallback voice (Meta MMS) is not built yet."
        )
    mode = voice_mode or settings.tts.voice_mode

    translation_path = paths.translation(target.code)
    if not translation_path.is_file():
        raise PipelineError(f"No {target.name} translation for job '{job_id}'. Run 'translate --to {lang}' first.")
    translation_sha = file_sha256(translation_path)
    manifest_path = paths.tts_manifest(target.code)
    previous: dict[int, dict[str, Any]] = {}
    references: dict[int, dict[str, Any]] = {}
    if redo_flagged:
        earlier = _load_previous(manifest_path, translation_sha)
        previous = {u["id"]: u for u in earlier["units"]}
        # JSON stores dictionary keys as text; turn them back into numbers.
        references = {int(k): v for k, v in earlier.get("references", {}).items()}
    elif manifest_path.exists() and not overwrite:
        raise OutputExistsError(
            f"{manifest_path.name} already exists. Add --overwrite to generate all speech again, "
            "or --redo-flagged to remake only the lines that failed the checks."
        )

    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    units: list[dict[str, Any]] = translation["segments"]
    selected = units if limit is None else units[:limit]
    if redo_flagged:
        selected = [u for u in selected if _needs_redo(previous.get(u["id"]))]
        log.info("Remaking %d flagged lines; keeping the other %d", len(selected), len(previous) - len(selected))
    redo_ids = {u["id"] for u in selected}
    _, transcript = load_transcript(paths, job_id)
    total_s = float(transcript["source"]["duration_s"])

    timings: dict[str, float] = {}
    started = time.perf_counter()
    source_y, sr_src = read_mono(ensure_source_audio(paths, transcript, settings))

    # Delivery of the original lines: from 'analyze' if it was run, otherwise measured now.
    features_path = paths.features(translation["source_language"])
    source_features: dict[int, dict[str, Any]] = {}
    if features_path.is_file():
        source_features = {u["id"]: u for u in json.loads(features_path.read_text(encoding="utf-8"))["units"]}

    # Voice references (cut again for the lines being made now, so improvements apply)
    plan = {c.unit_id: c for c in plan_references(units, settings.tts.min_reference_s,
                                                  settings.tts.max_borrow_distance_s)}
    if mode == "clone":
        needed = {plan[u["id"]].ref_unit_id for u in selected} - {None}
        for ref_id in sorted(needed):
            path, quality = cut_reference(source_y, sr_src, units[ref_id], paths.references_dir / f"unit_{ref_id:03d}.wav")
            references[ref_id] = {"path": str(path), **quality}
    timings["references_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    synth = XttsSynthesizer(settings)
    checker = _BackChecker(settings) if settings.tts.verify_with_asr else None
    timings["model_load_s"] = round(time.perf_counter() - started, 2)
    standard = synth.builtin_voice(settings.tts.standard_speaker)
    voices: dict[int, Voice] = {}

    out_dir = paths.tts_dir(target.code)
    out_units: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0
    for unit in (units if redo_flagged else selected):
        if unit["id"] not in redo_ids:
            out_units.append(previous[unit["id"]])  # kept from the earlier run
            continue
        done += 1
        entry: dict[str, Any] = {k: unit[k] for k in ("id", "source_segment_ids", "start", "end", "text")}
        entry["available_s"] = unit["available_s"]
        flags: list[str] = []
        if not unit["text"].strip():
            entry.update(status="skipped_no_text", flags=["no_text"])
            out_units.append(entry)
            continue

        choice = plan[unit["id"]]
        if mode == "clone" and choice.ref_unit_id is not None:
            if choice.ref_unit_id not in voices:
                ref = references[choice.ref_unit_id]
                voices[choice.ref_unit_id] = synth.voice_from_audio(
                    [Path(ref["path"])], f"cloned from original line {choice.ref_unit_id}")
            voice = voices[choice.ref_unit_id]
            entry["reference"] = {"unit_id": choice.ref_unit_id, "borrowed": choice.borrowed,
                                  "distance_s": choice.distance_s,
                                  "warnings": references[choice.ref_unit_id]["warnings"],
                                  "gain_applied_db": references[choice.ref_unit_id]["gain_applied_db"]}
            if choice.borrowed:
                flags.append("reference_borrowed")
            flags += [f"reference_{w}" for w in references[choice.ref_unit_id]["warnings"]]
        else:
            voice = standard
            if mode == "clone":
                flags.append("builtin_voice_fallback")

        spoken_text = normalize_for_speech(unit["text"], target.code)
        entry["spoken_text"] = spoken_text
        available = max(unit["available_s"], 0.001)
        result = _generate_line(synth, checker, settings, spoken_text, target, voice, available)

        # Clarity comes first: if the cloned voice could not be understood, try the clear built-in voice.
        if not result["passed"] and voice is not standard and settings.tts.clarity_fallback:
            fallback = _generate_line(synth, checker, settings, spoken_text, target, standard, available)
            result["attempts"] += [{**a, "voice": "built-in"} for a in fallback["attempts"]]
            if fallback["score"] < result["score"]:
                fallback["attempts"] = result["attempts"]
                result, voice = fallback, standard
                flags.append("clarity_fallback_builtin")
        entry["voice"] = voice.description

        y, speed = result["y"], result["speed"]
        if len(result["attempts"]) > 1:
            flags.append("retried")
        if result["invented"]:
            flags.append("invented_speech")
        if result["cer"] is not None:
            entry["back_check"] = {"heard": result["heard"], "char_error_rate": result["cer"]}
            if result["cer"] > LOW_INTELLIGIBILITY_CER:
                flags.append("low_intelligibility")
        duration = len(y) / synth.sample_rate
        overrun = duration - available
        if overrun > OVERRUN_TOLERANCE_S:
            flags.append("overruns_slot")

        wav_path = out_dir / f"unit_{unit['id']:03d}.wav"
        write_wav(wav_path, y, synth.sample_rate,
                  comment=f"AI-generated speech ({LICENSE_NOTE}). Voice: {voice.description}.")

        dubbed = analyze_clip(y, synth.sample_rate)
        original = source_features.get(unit["id"]) or analyze_clip(cut(source_y, sr_src, unit["start"], unit["end"]), sr_src)
        delivery = compare(original, dubbed)
        if delivery.get("pitch_range_diff_st", 0) < FLATTER_THAN_ORIGINAL_ST:
            flags.append("flatter_than_original")

        entry.update({
            "status": "generated",
            "path": str(wav_path),
            "duration_s": round(duration, 3),
            "overrun_s": round(max(overrun, 0.0), 3),
            "speed": speed,
            "attempts": result["attempts"],
            "original_delivery": {k: original.get(k) for k in ("pitch_median_hz", "pitch_range_st", "loudness_db")},
            "dubbed_delivery": {k: dubbed.get(k) for k in ("pitch_median_hz", "pitch_range_st", "loudness_db")},
            "delivery_diff": delivery,
            "flags": flags,
            "ai_generated": True,
        })
        out_units.append(entry)
        log.info("  %d/%d unit %d: %.2fs (slot %.2fs, speed %.2f)%s", done, len(selected), unit["id"],
                 duration, available, speed, f" flags: {', '.join(flags)}" if flags else "")
    timings["synthesize_s"] = round(time.perf_counter() - started, 2)

    generated = [u for u in out_units if u.get("status") == "generated"]
    clips = [(u["start"], read_mono(Path(u["path"]))[0]) for u in generated]
    track, overlaps = place_on_timeline(clips, synth.sample_rate, total_s)
    preview = write_wav(paths.speech_preview(target.code), track, synth.sample_rate,
                        comment=f"AI-generated dubbed speech preview ({LICENSE_NOTE}).")

    cers = [u["back_check"]["char_error_rate"] for u in generated if "back_check" in u]
    manifest = {
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": target.code,
        "source_translation": str(translation_path),
        "source_translation_sha256": translation_sha,
        "engine": {"name": "xtts_v2", "model": settings.tts.model, "device": synth.device,
                   "voice_mode": mode, "temperature": settings.tts.temperature, "seed": settings.tts.seed,
                   "license": LICENSE_NOTE},
        "ai_generated": True,
        "speech_preview": str(preview),
        "summary": {
            "units_in_translation": len(units),
            "units_in_manifest": len(out_units),
            "generated": len(generated),
            "remade_this_run": len(selected) if redo_flagged else None,
            "flag_counts": dict(Counter(f for u in out_units for f in u.get("flags", []))),
            "overlapping_ranges_in_preview": overlaps,
            "median_char_error_rate": round(float(np.median(cers)), 3) if cers else None,
            "total_speech_s": round(sum(u["duration_s"] for u in generated), 2),
        },
        "references": references,
        "timings": timings,
        "units": out_units,
    }
    write_json(manifest_path, manifest)
    log.info("Saved speech manifest: %s (%d lines). Preview: %s", manifest_path, len(generated), preview)
    return manifest_path
