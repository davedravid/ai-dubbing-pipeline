"""Stage runner: transcript JSON -> translation JSON for each target language."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dubbing import __version__
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.languages import get_language
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.transcribe_job import file_sha256, write_json
from dubbing.translator.checks import check_translation, timing_facts
from dubbing.translator.nllb import NllbTranslator, Translator
from dubbing.translator.units import build_units

log = logging.getLogger(__name__)

TRANSLATION_SCHEMA_VERSION = 2


def _find_transcript(paths: JobPaths, job_id: str) -> Path:
    found = paths.existing_transcripts()
    if not found:
        raise PipelineError(f"Job '{job_id}' has no transcript yet. Run 'transcribe' first.")
    if len(found) > 1:
        raise PipelineError(f"Job '{job_id}' has several transcripts: {', '.join(p.name for p in found)}")
    return found[0]


def _check_complete(units: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    """Every transcript segment must belong to exactly one unit, in order."""
    covered = [i for u in units for i in u["source_segment_ids"]]
    if covered != [s["id"] for s in segments]:
        raise PipelineError("Translated units do not cover the transcript exactly; nothing was saved.")


def run_translation(
    job_id: str,
    settings: Settings,
    target_langs: list[str] | None = None,
    overwrite: bool = False,
    translator_factory: Callable[[], Translator] | None = None,
) -> list[Path]:
    """Translate a job's transcript into each target language. Returns the files written."""
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    transcript_path = _find_transcript(paths, job_id)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    source = get_language(transcript["language"])

    targets = [get_language(code) for code in (target_langs or settings.target_languages)]
    if not targets:
        raise PipelineError("No target language given. Use --to hi or set target_languages in config.yaml.")
    for target in targets:
        if target.code == source.code:
            raise PipelineError(f"The video is already in {source.name}; choose a different target language.")
        out = paths.translation(target.code)
        if out.exists() and not overwrite:
            raise OutputExistsError(f"{out.name} already exists. Add --overwrite to replace it.")

    segments: list[dict[str, Any]] = transcript.get("segments", [])
    units = build_units(segments)
    _check_complete(units, segments)
    facts = timing_facts(units, float(transcript["source"]["duration_s"]))
    to_translate = [i for i, u in enumerate(units) if u["text"]]
    log.info("Grouped %d transcript segments into %d sentence units", len(segments), len(units))

    started = time.perf_counter()
    translator = (translator_factory or (lambda: NllbTranslator(settings.translation, settings.device)))()
    load_s = round(time.perf_counter() - started, 2)

    written = []
    for target in targets:
        log.info("Translating %d units: %s -> %s", len(to_translate), source.name, target.name)
        started = time.perf_counter()
        outputs = translator.translate([units[i]["text"] for i in to_translate], source.nllb, target.nllb)
        translate_s = round(time.perf_counter() - started, 2)
        if len(outputs) != len(to_translate):
            raise PipelineError(
                f"Translator returned {len(outputs)} results for {len(to_translate)} units; nothing was saved."
            )
        translated = dict(zip(to_translate, outputs))

        out_units = []
        for i, unit in enumerate(units):
            fact = facts[i]
            if i in translated:
                flags, estimates = check_translation(
                    unit["text"], translated[i], source, target,
                    fact["duration_s"], fact["available_s"], fact["source_chars_per_s"],
                )
                status = "translated"
            else:
                flags, estimates, status = ["empty_source"], {}, "skipped_empty_source"
            out_units.append({
                "id": unit["id"],
                "source_segment_ids": unit["source_segment_ids"],
                "start": unit["start"],
                "end": unit["end"],
                "source_text": unit["text"],
                "text": translated.get(i, ""),
                "status": status,
                "flags": flags,
                **fact,
                **estimates,
            })

        flag_counts = Counter(f for u in out_units for f in u["flags"])
        result = {
            "schema_version": TRANSLATION_SCHEMA_VERSION,
            "pipeline_version": __version__,
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_transcript": str(transcript_path),
            "source_transcript_sha256": file_sha256(transcript_path),
            "source_language": source.code,
            "language": target.code,
            "target_nllb_code": target.nllb,
            "translator": translator.info(),
            "summary": {
                "source_segment_count": len(segments),
                "unit_count": len(out_units),
                "translated": sum(1 for u in out_units if u["status"] == "translated"),
                "skipped": sum(1 for u in out_units if u["status"] != "translated"),
                "flagged_units": sum(1 for u in out_units if u["flags"]),
                "flag_counts": dict(flag_counts),
            },
            "timings": {"model_load_s": load_s, "translate_s": translate_s},
            # Each entry is one sentence unit (one or more transcript segments).
            "segments": out_units,
        }
        out = paths.translation(target.code)
        write_json(out, result)
        log.info("Saved translation: %s (%d units from %d segments, %d flagged: %s)",
                 out, len(out_units), len(segments), result["summary"]["flagged_units"],
                 dict(flag_counts) or "none")
        written.append(out)
    return written
