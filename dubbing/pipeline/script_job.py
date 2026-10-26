"""Stage runner: transcript + separated voices -> the source dubbing script and its review page.

The script is the single source of truth for later phases: every line with its speaker,
exact timing, pauses, breath groups, delivery, and flags; non-speech sounds and
overlapping speech; speaker profiles (with gender); and clean voice samples per speaker.
"""

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
from dubbing.asr.listen import Listener
from dubbing.audio.clips import read_mono, resample
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.analyze_job import load_transcript
from dubbing.pipeline.transcribe_job import write_json
from dubbing.script.clone_refs import choose_samples, write_previews
from dubbing.script.delivery import add_styles, measure_line
from dubbing.script.events import find_sounds, overlaps_from_turns
from dubbing.script.glossary import clean_text, load_glossary
from dubbing.script.review import LOW_CONFIDENCE, write_review
from dubbing.script.speakers import apply_merges, build_profiles, load_overrides, write_overrides_template
from dubbing.script.timing import Envelope, breath_groups, pauses
from dubbing.subtitles.vtt import write_vtt

log = logging.getLogger(__name__)

SCRIPT_SCHEMA_VERSION = 1
VERY_FAST_WPS = 4.5          # words per second; dubbing such lines in Hindi is hard
TIMING_SHIFT_FLAG_S = 0.15   # report lines whose edges moved this much after snapping to the voice


def _overlap_turns(paths: JobPaths, transcript: dict[str, Any], settings: Settings, vocals16k: np.ndarray):
    """pyannote's speaker turns (re-used if cached for this video), only to find overlaps."""
    sha = transcript["source"]["sha256"]
    if paths.diarization.is_file():
        cached = json.loads(paths.diarization.read_text(encoding="utf-8"))
        if cached.get("source_sha256") == sha:
            return cached["turns"]
    if not settings.script.detect_overlaps:
        return None
    from dubbing.diarization.speakers import diarize

    turns = diarize(vocals16k, 16000, settings.diarization.model,
                    "cuda" if settings.device == "cuda" else "cpu")
    write_json(paths.diarization, {"job_id": paths.job_id, "model": settings.diarization.model,
                                   "source_sha256": sha, "turns": turns})
    return turns


def _coverage(env: Envelope, spans: list[tuple[float, float]]) -> float:
    active = env.db > env.db.max() - 35
    covered = np.zeros_like(active)
    for a, b in spans:
        covered[int(a / 0.01): int(b / 0.01) + 1] = True
    return round(float((active & covered).sum() / max(active.sum(), 1)), 3)


def run_script(job_id: str, settings: Settings, overwrite: bool = False) -> Path:
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    if paths.script.exists() and not overwrite:
        raise OutputExistsError(f"{paths.script.name} already exists. Add --overwrite to build it again.")
    _, transcript = load_transcript(paths, job_id)
    if not paths.vocals.is_file() or not paths.background.is_file():
        raise PipelineError("The separated voices are missing. Run 'transcribe' (or 'separate') first.")
    video = Path(transcript["source"]["path"])
    timings: dict[str, float] = {}

    started = time.perf_counter()
    vocals, sr = read_mono(paths.vocals)
    background, _ = read_mono(paths.background)
    vocals16k = resample(vocals, sr, 16000)
    terms = load_glossary(paths.glossary, extra=settings.asr.glossary)
    overrides = load_overrides(paths.speaker_overrides)
    env = Envelope(vocals, sr)

    lines = []
    segs = transcript["segments"]
    seg_by_id = {s["id"]: s for s in segs}
    for i, seg in enumerate(segs):
        words = seg.get("words") or []
        # Never let a line's edge cross into the previous line's last word or the next line's first word.
        prev_end = segs[i - 1]["end"] if i > 0 else 0.0
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else float("inf")
        start, end = env.refine(seg["start"], seg["end"], not_before=min(prev_end, seg["start"]),
                                not_after=max(next_start, seg["end"]))
        line = {
            "id": seg["id"], "speaker": seg.get("speaker"),
            "speaker_confidence": seg.get("speaker_confidence"), "speaker_uncertain": seg.get("speaker_uncertain", True),
            "start": start, "end": end, "asr_start": seg["start"], "asr_end": seg["end"],
            "text": clean_text(seg["text"], terms), "words": words,
            "pauses": pauses(words), "breath_groups": breath_groups(words),
        }
        if seg.get("recovered"):
            line["recovered"] = True
        if seg.get("split_from_two_voices") is not None:
            line["split_from_two_voices"] = seg["split_from_two_voices"]
        lines.append(line)
    lines = apply_merges(lines, overrides)
    timings["lines_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    for line in lines:
        line["delivery"] = measure_line(vocals, sr, line)
    add_styles(lines)
    timings["delivery_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    listener = Listener(settings, transcript["language"])
    events = find_sounds(vocals16k, 16000, lines, listener.heard)
    turns = _overlap_turns(paths, transcript, settings, vocals16k)
    overlaps = overlaps_from_turns(turns) if turns else []
    events += overlaps
    timings["events_s"] = round(time.perf_counter() - started, 2)

    for line in lines:
        flags = []
        low = [w["word"] for w in line["words"] if (w.get("probability") or 1.0) < LOW_CONFIDENCE]
        if low:
            flags.append(f"unsure_words:{len(low)}")
        if line.get("speaker_uncertain"):
            flags.append("speaker_uncertain")
        if any(min(line["end"], o["end"]) - max(line["start"], o["start"]) > 0.4 for o in overlaps):
            flags.append("overlapping_speech")
        if line.get("recovered"):
            flags.append("recovered")
        if "split_from_two_voices" in line:
            flags.append("split_two_voices")
        if seg_by_id.get(line["id"], {}).get("speaker_reassigned_from"):
            flags.append("speaker_merged_from_tiny_group")
        if (line["delivery"].get("words_per_s") or 0) > VERY_FAST_WPS:
            flags.append("very_fast")
        if max(abs(line["start"] - line["asr_start"]), abs(line["end"] - line["asr_end"])) > TIMING_SHIFT_FLAG_S:
            flags.append("timing_adjusted")
        line["flags"] = flags

    profiles = build_profiles(lines, overrides)
    for line in lines:
        line["speaker_name"] = (profiles.get(line.get("speaker") or "") or {}).get("name")
        line["speaker_gender"] = (profiles.get(line.get("speaker") or "") or {}).get("gender")
    created_template = write_overrides_template(paths.speaker_overrides, profiles)

    started = time.perf_counter()
    samples = choose_samples(lines, events, vocals, background, sr)
    write_previews(samples, vocals, sr, paths.speaker_samples_dir)
    for spk, info in samples.items():
        if info["grade"] == "weak" and spk in profiles:
            profiles[spk].setdefault("flags", []).append("few_clean_voice_samples")
    timings["clone_samples_s"] = round(time.perf_counter() - started, 2)

    flag_counts = Counter(f.split(":")[0] for l in lines for f in l["flags"])
    event_counts = Counter(e["type"] for e in events)
    script = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": transcript["language"],
        "source": transcript["source"],
        "speaker_corrections_file": str(paths.speaker_overrides),
        "glossary": terms,
        "summary": {
            "lines": len(lines),
            "words": sum(len(l["words"]) for l in lines),
            "speakers": len(profiles),
            "voice covered by script": f'{100 * _coverage(env, [(l["start"], l["end"]) for l in lines] + [(e["start"], e["end"]) for e in events if e["type"] != "overlapping_speech"]):.1f}%',
            "lines flagged": sum(1 for l in lines if l["flags"]),
            "unsure words": sum(1 for l in lines for w in l["words"] if (w.get("probability") or 1) < LOW_CONFIDENCE),
            "uncertain speaker": flag_counts.get("speaker_uncertain", 0),
            "overlapping speech": event_counts.get("overlapping_speech", 0),
            "sounds (laugh/sigh/…)": sum(v for k, v in event_counts.items()
                                          if k not in ("overlapping_speech", "untranscribed_speech")),
            "untranscribed speech": event_counts.get("untranscribed_speech", 0),
        },
        "flag_counts": dict(flag_counts),
        "speakers": profiles,
        "clone_samples": samples,
        "events": sorted(events, key=lambda e: e["start"]),
        "lines": lines,
        "timings": timings,
    }
    write_json(paths.script, script)
    write_review(script, paths.script_review, video)
    write_vtt([{**l, "text": f'[{l.get("speaker_name") or l.get("speaker") or "?"}] {l["text"]}'} for l in lines],
              paths.scripts_dir / f"{job_id}_en.vtt")
    log.info("Saved dubbing script: %s | review page: %s", paths.script, paths.script_review)
    if created_template:
        log.info("Created %s: add speaker names/genders there and run 'script --overwrite'.", paths.speaker_overrides)
    return paths.script
