"""Stage runner: background + dubbed lines -> final soundtrack -> dubbed video + QA report."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from dubbing import __version__
from dubbing.analysis.prosody import analyze_clip
from dubbing.audio.clips import cut, place_on_timeline, read_mono, resample
from dubbing.asr.listen import Listener
from dubbing.audio.fit import plan_fit
from dubbing.audio.nonverbal import candidate_regions, classify_regions, restore_mask
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.languages import get_language
from dubbing.media.ffmpeg import find_ffmpeg
from dubbing.media.mux import (AudioTrack, SubtitleTrack, build_video, decode_errors, extract_track,
                               streams, subtitle_cues, tighten, video_fingerprint)
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.analyze_job import load_transcript
from dubbing.pipeline.separate_job import ensure_stereo_source, run_separation
from dubbing.pipeline.subtitles_job import run_subtitles
from dubbing.pipeline.transcribe_job import write_json
from dubbing.qa.audio_metrics import integrated_lufs, soundtrack_metrics
from dubbing.tts.xtts import LICENSE_NOTE

log = logging.getLogger(__name__)

MIX_SR = 44100
DIP_RAMP_S = 0.15           # how gently the background dips in and out around speech
MAX_OVERALL_GAIN_DB = 12.0
# Pass/fail limits for the QA report
MAX_DURATION_DIFF_S = 0.1
MAX_LOUDNESS_DIFF_LU = 1.0
# Speech flags worth a human listen in the final video
LISTEN_FLAGS = {"low_intelligibility", "tail_cut", "clarity_fallback_builtin", "builtin_voice_fallback",
                "needs_shorter_translation", "overlaps_next_line", "flatter_than_original"}


ONSET_BELOW_PEAK_DB = 30.0
ONSET_LOOKBACK_S = 0.1


def voice_onset(vocals: np.ndarray, sr: int, start: float, end: float) -> float:
    """When the actor's voice really begins in a line (transcript times can be early).
    Searches from a moment before `start` to `end` in the separated voices; never
    returns a time before the transcript's start minus ONSET_LOOKBACK_S."""
    a = max(0, int((start - ONSET_LOOKBACK_S) * sr))
    clip = vocals[a: int(end * sr)]
    hop = int(0.01 * sr)
    n = len(clip) // hop
    if n == 0:
        return start
    level = 20 * np.log10(np.maximum(np.sqrt((clip[: n * hop].reshape(n, hop) ** 2).mean(1)), 1e-10))
    loud = np.flatnonzero(level > level.max() - ONSET_BELOW_PEAK_DB)
    return round(a / sr + (loud[0] * hop / sr if loud.size else ONSET_LOOKBACK_S), 3)


def _speech_mask(placements, n_samples: int, sr: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=np.float32)
    for p in placements:
        a, b = int(p.start * sr), min(n_samples, int((p.start + p.duration) * sr))
        mask[a:b] = 1.0
    ramp = max(1, int(DIP_RAMP_S * sr))
    kernel = np.ones(ramp, dtype=np.float32) / ramp
    return np.clip(np.convolve(mask, kernel, mode="same"), 0.0, 1.0)


def _gain_db(target: float | None, current: float | None, limit: float) -> float:
    if target is None or current is None:
        return 0.0
    return float(np.clip(target - current, -limit, limit))


def _ensure_subtitles(job_id: str, settings: Settings, paths: JobPaths, langs: list[str]) -> list[Path]:
    out = []
    for lang in langs:
        path = paths.subtitle(lang, "vtt")
        if not path.is_file():
            run_subtitles(job_id, settings, lang=lang)
        out.append(path)
    return out


def run_mix(job_id: str, settings: Settings, lang: str = "hi", overwrite: bool = False) -> Path:
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    target = get_language(lang)
    cfg = settings.mix
    for out in (paths.dub_video(lang), paths.dub_mix(lang), paths.mix_report(lang)):
        if out.exists() and not overwrite:
            raise OutputExistsError(f"{out.name} already exists. Add --overwrite to mix again.")

    manifest_path = paths.tts_manifest(lang)
    if not manifest_path.is_file():
        raise PipelineError(f"No dubbed speech for job '{job_id}' in '{lang}'. Run 'speak' first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, transcript = load_transcript(paths, job_id)
    source_lang = get_language(transcript["language"])
    video_in = Path(transcript["source"]["path"])
    if not video_in.is_file():
        raise PipelineError(f"The original video is no longer at {video_in}.")
    total_s = float(transcript["source"]["duration_s"])
    tools = find_ffmpeg(settings.ffmpeg_path)
    timings: dict[str, float] = {}

    # 1) Voices / background separation (run it now if it has not been done)
    if not paths.background.is_file() or not paths.vocals.is_file():
        log.info("No separated background yet; running 'separate' first.")
        run_separation(job_id, settings, overwrite=True, input_path=video_in, leak_lang=transcript["language"])
    separation = json.loads(paths.separation_report.read_text(encoding="utf-8")) if paths.separation_report.is_file() else {}

    started = time.perf_counter()
    background, sr = sf.read(str(paths.background), dtype="float32", always_2d=True)
    vocals_mono, sr_v = read_mono(paths.vocals)
    if sr != MIX_SR or sr_v != MIX_SR:
        raise PipelineError(f"Separated tracks must be {MIX_SR} Hz; found {sr} / {sr_v}. Run 'separate --overwrite'.")
    n = len(background)

    # 2) Timing: start each line when the actor's voice really begins, then fit it
    generated = sorted((u for u in manifest["units"] if u.get("status") == "generated"), key=lambda u: u["start"])
    lines = [{"id": u["id"], "start": voice_onset(vocals_mono, MIX_SR, u["start"], u["end"]),
              "duration": u["duration_s"]} for u in generated]
    placements = plan_fit(lines, total_s, cfg.max_early_start_s, cfg.min_gap_s, cfg.max_tighten)

    # 3) Prepare each line: tighten, resample, match the original line's loudness
    fit_dir = paths.temp_dir / f"fit_{lang}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[float, np.ndarray]] = []
    line_reports: list[dict[str, Any]] = []
    for unit, place in zip(generated, placements):
        wav = Path(unit["path"])
        if place.tighten > 1.001:
            wav = tighten(tools, wav, fit_dir / f"unit_{unit['id']:03d}.wav", place.tighten)
        y, sr_line = read_mono(wav)
        y = resample(y, sr_line, MIX_SR)
        gain = 0.0
        if cfg.match_line_loudness:
            original = analyze_clip(cut(vocals_mono, MIX_SR, unit["start"], unit["end"]), MIX_SR, with_pitch=False)
            dubbed = analyze_clip(y, MIX_SR, with_pitch=False)
            gain = _gain_db(original.get("loudness_db"), dubbed.get("loudness_db"), cfg.max_line_gain_db)
            y = y * (10 ** (gain / 20))
        clips.append((place.start, y))
        line_reports.append({**place.as_dict(), "line_gain_db": round(gain, 1),
                             "speech_flags": unit.get("flags", [])})
    timings["prepare_lines_s"] = round(time.perf_counter() - started, 2)

    # 4) Mix: speech on top of the background, which dips a little under speech
    started = time.perf_counter()
    speech, overlaps = place_on_timeline(clips, MIX_SR, n / MIX_SR)
    speech = speech[:n] if len(speech) >= n else np.pad(speech, (0, n - len(speech)))
    dip = 10 ** (-cfg.background_dip_db * _speech_mask(placements, n, MIX_SR) / 20)
    mix = background * dip[:, None] + speech[:, None]
    timings["mix_s"] = round(time.perf_counter() - started, 2)

    # 4b) Put back the actors' laughs, sighs and breaths (never untranslated words)
    started = time.perf_counter()
    kept_sounds, untranslated = [], []
    if cfg.keep_nonverbal_sounds:
        words = [(w["start"], w["end"]) for s in transcript["segments"] for w in (s.get("words") or [])] or \
                [(s["start"], s["end"]) for s in transcript["segments"]]
        busy = [(p.start, p.start + p.duration) for p in placements]
        regions = candidate_regions(vocals_mono, MIX_SR, words, busy)
        if regions:
            listener = Listener(settings, transcript["language"])
            kept_sounds, untranslated = classify_regions(regions, vocals_mono, MIX_SR, listener.heard)
            vocals_stereo, _ = sf.read(str(paths.vocals), dtype="float32", always_2d=True)
            mix = mix + vocals_stereo[:n] * restore_mask(kept_sounds, n, MIX_SR)[:, None]
        log.info("Kept %d non-word sounds (%.1fs); %d stretches of untranslated speech found",
                 len(kept_sounds), sum(r["end"] - r["start"] for r in kept_sounds), len(untranslated))
    timings["nonverbal_s"] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()

    # 5) Overall loudness: match the original soundtrack
    original_y, _ = sf.read(str(ensure_stereo_source(paths, transcript, settings)), dtype="float32", always_2d=True)
    original_lufs, mix_lufs = integrated_lufs(original_y, MIX_SR), integrated_lufs(mix, MIX_SR)
    overall_gain = _gain_db(original_lufs, mix_lufs, MAX_OVERALL_GAIN_DB)
    mix = mix * (10 ** (overall_gain / 20))
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(paths.dub_mix(lang)), "w", samplerate=MIX_SR, channels=2, subtype="FLOAT", format="WAV") as f:
        f.comment = f"AI-generated {target.name} dub ({LICENSE_NOTE})"
        f.write(mix)
    timings["loudness_s"] = round(time.perf_counter() - started, 2)

    # 6) Build the video: original picture + dubbed audio (+ original audio) + subtitles
    started = time.perf_counter()
    audio_tracks = [AudioTrack(paths.dub_mix(lang), target.mms, f"{target.name} (AI dub)", True)]
    if cfg.include_original_audio:
        audio_tracks.append(AudioTrack(None, source_lang.mms, f"{source_lang.name} (original)", False))
    subtitles = []
    if cfg.embed_subtitles:
        for code, path in zip([target.code, source_lang.code],
                              _ensure_subtitles(job_id, settings, paths, [target.code, source_lang.code])):
            language = get_language(code)
            subtitles.append(SubtitleTrack(path, language.mms, language.name))
    video_out = build_video(tools, video_in, paths.dub_video(lang), audio_tracks, subtitles,
                            bitrate=cfg.audio_bitrate, limiter_peak_db=cfg.limiter_peak_db,
                            comment=f"Contains AI-generated {target.name} speech ({LICENSE_NOTE}).")
    timings["build_video_s"] = round(time.perf_counter() - started, 2)

    # 7) Checks on the finished video
    started = time.perf_counter()
    expected_lines = sum(1 for u in manifest["units"] if u.get("status") != "skipped_no_text")
    report = _qa(tools, paths, lang, video_in, video_out, original_y, placements, line_reports,
                 overlaps, expected_lines, separation, cfg, [s.path for s in subtitles])
    report["nonverbal_sounds_kept"] = {"count": len(kept_sounds),
                                       "seconds": round(sum(r["end"] - r["start"] for r in kept_sounds), 1),
                                       "where": kept_sounds}
    report["untranslated_speech"] = untranslated
    report["checks"]["all_speech_dubbed"] = not untranslated
    report["passed"] = all(report["checks"].values())
    report["listen_to"] += [{"id": None, "at_s": r["start"], "reasons": ["untranslated_speech"]} for r in untranslated]
    timings["qa_s"] = round(time.perf_counter() - started, 2)
    report.update({
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": lang,
        "outputs": {"video": str(video_out), "soundtrack": str(paths.dub_mix(lang))},
        "levels": {"original_lufs": original_lufs, "mix_before_gain_lufs": mix_lufs,
                   "overall_gain_db": round(overall_gain, 2), "background_dip_db": cfg.background_dip_db},
        "settings": {"max_early_start_s": cfg.max_early_start_s, "min_gap_s": cfg.min_gap_s,
                     "max_tighten": cfg.max_tighten, "limiter_peak_db": cfg.limiter_peak_db},
        "timings": timings,
        "ai_generated": True,
    })
    write_json(paths.mix_report(lang), report)
    verdict = "PASSED" if report["passed"] else "NEEDS ATTENTION"
    log.info("Dubbed video: %s | checks %s: %s", video_out, verdict,
             ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in report["checks"].items()))
    return video_out


def _qa(tools, paths: JobPaths, lang: str, video_in: Path, video_out: Path, original_y: np.ndarray,
        placements, line_reports, overlaps, expected_lines: int, separation: dict, cfg,
        subtitle_files: list[Path]) -> dict[str, Any]:
    info = streams(tools, video_out)
    video_streams = [s for s in info if s["type"] == "video"]
    audio_streams = [s for s in info if s["type"] == "audio"]
    errors = decode_errors(tools, video_out)
    same_picture = video_fingerprint(tools, video_in) == video_fingerprint(tools, video_out)
    subtitle_streams = [s for s in info if s["type"] == "subtitle"]
    cues_in_video = [subtitle_cues(tools, video_out, i) for i in range(len(subtitle_streams))]
    cues_in_files = [Path(p).read_text(encoding="utf-8").count(" --> ") for p in subtitle_files]

    dub_wav = extract_track(tools, video_out, 0, paths.temp_dir / f"{paths.job_id}_{lang}_final_check.wav", MIX_SR)
    dub_y, _ = sf.read(str(dub_wav), dtype="float32", always_2d=True)
    final = soundtrack_metrics(dub_y, MIX_SR)
    original = soundtrack_metrics(original_y, MIX_SR)

    video_dur = video_streams[0]["duration_s"] if video_streams else None
    dub_dur = audio_streams[0]["duration_s"] if audio_streams else None
    duration_diff = abs(video_dur - dub_dur) if video_dur and dub_dur else None
    loud_diff = (abs(final["loudness_lufs"] - original["loudness_lufs"])
                 if final["loudness_lufs"] is not None and original["loudness_lufs"] is not None else None)

    fit_counts = Counter(f for r in line_reports for f in r["flags"])
    listen = []
    for r in line_reports:
        reasons = sorted((set(r["flags"]) | set(r["speech_flags"])) & LISTEN_FLAGS)
        if reasons:
            listen.append({"id": r["id"], "at_s": r["start"], "reasons": reasons})

    checks = {
        "plays_without_errors": not errors,
        "picture_unchanged": same_picture,
        "sound_same_length_as_picture": duration_diff is not None and duration_diff <= MAX_DURATION_DIFF_S,
        "loudness_matches_original": loud_diff is not None and loud_diff <= MAX_LOUDNESS_DIFF_LU,
        "no_distortion": final["clipped_ratio"] == 0.0,
        "subtitles_complete": cues_in_video == cues_in_files,
        # Every line the speech step produced must have been placed in the mix.
        "all_lines_in_mix": len(placements) == expected_lines,
    }
    leak = (separation or {}).get("leak_check") or {}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "streams": info,
        "decode_errors": errors[:20],
        "subtitles": {"captions_in_video": cues_in_video, "captions_in_files": cues_in_files},
        "duration": {"video_s": video_dur, "dubbed_audio_s": dub_dur, "difference_s": duration_diff},
        "soundtrack": {"original": original, "dubbed_final": final, "loudness_difference_lu": loud_diff},
        "timing_fixes": {
            "lines": len(placements),
            "lines_expected": expected_lines,
            "fit_without_changes": sum(1 for r in line_reports if not r["flags"]),
            "started_early": fit_counts.get("started_early", 0),
            "tightened": fit_counts.get("tightened", 0),
            "needs_shorter_translation": fit_counts.get("needs_shorter_translation", 0),
            "still_overlapping": fit_counts.get("overlaps_next_line", 0),
            "overlap_total_s": round(sum(r["overlap_s"] for r in line_reports), 2),
            "overlapping_ranges": overlaps,
        },
        "leaked_original_dialogue": {"words_heard": leak.get("words_heard"),
                                     "seconds_with_words": leak.get("seconds_with_words"),
                                     "where": [(h["start"], h["end"]) for h in leak.get("heard", [])]},
        "listen_to": listen,
        "lines": line_reports,
    }
