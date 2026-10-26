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
from dubbing.qa.text_metrics import chrf
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


MIN_MEANING_SCORE = 40.0        # chrF of the back-translation against the original sentence
KEEP_MEANING_SHARE = 0.85       # an alternative must keep this share of the main wording's score
MIN_SHORTER = 0.92              # ...and be at least 8% shorter than the main wording
BAD_FLAGS = {"empty_translation", "wrong_script", "repetition", "unchanged"}


def _shorter_alternatives(translator, sources: list[str], candidates: list[list[str]],
                          source, target) -> dict[int, dict[str, Any]]:
    """For each sentence, the alternative wordings that are shorter AND keep the meaning.

    Meaning is checked by translating each wording back into the source language and
    comparing it with the original sentence (chrF score, 0-100)."""
    flat = [(k, text) for k, cands in enumerate(candidates) for text in cands]
    backs = translator.translate([t for _, t in flat], target.nllb, source.nllb)
    scored: dict[int, list[dict[str, Any]]] = {}
    for (k, text), back in zip(flat, backs):
        scored.setdefault(k, []).append({"text": text, "chars": len(text), "back_translation": back,
                                         "meaning_score": chrf(sources[k], back)})
    result = {}
    for k, options in scored.items():
        primary, rest = options[0], options[1:]
        floor = max(MIN_MEANING_SCORE, primary["meaning_score"] * KEEP_MEANING_SHARE)
        good = []
        for o in rest:
            flags, _ = check_translation(sources[k], o["text"], source, target, 1.0, 1.0, 14.0)
            if o["chars"] <= primary["chars"] * MIN_SHORTER and o["meaning_score"] >= floor \
                    and not set(flags) & BAD_FLAGS:
                good.append(o)
        result[k] = {"primary_meaning": primary["meaning_score"], "primary_back": primary["back_translation"],
                     "options": sorted(good, key=lambda o: -o["meaning_score"])}
    return result


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
        sources = [units[i]["text"] for i in to_translate]
        n_alt = settings.translation.alternatives
        if n_alt > 0 and hasattr(translator, "translate_candidates"):
            candidates = translator.translate_candidates(sources, source.nllb, target.nllb, n_alt + 1)
        else:
            candidates = [[t] for t in translator.translate(sources, source.nllb, target.nllb)]
        translate_s = round(time.perf_counter() - started, 2)
        if len(candidates) != len(to_translate):
            raise PipelineError(
                f"Translator returned {len(candidates)} results for {len(to_translate)} units; nothing was saved."
            )
        translated = {i: c[0] for i, c in zip(to_translate, candidates)}

        started = time.perf_counter()
        has_options = any(len(c) > 1 for c in candidates)
        alternatives = (_shorter_alternatives(translator, sources, candidates, source, target)
                        if n_alt > 0 and has_options else {})
        alternatives = {to_translate[k]: v for k, v in alternatives.items()}
        alternatives_s = round(time.perf_counter() - started, 2)

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
                "speaker": unit.get("speaker"),
                "speaker_uncertain": unit.get("speaker_uncertain", True),
                "source_text": unit["text"],
                "text": translated.get(i, ""),
                "status": status,
                "flags": flags,
                **fact,
                **estimates,
                **({"meaning_score": alternatives[i]["primary_meaning"],
                    "back_translation": alternatives[i]["primary_back"],
                    "shorter_alternatives": alternatives[i]["options"]} if i in alternatives else {}),
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
            "timings": {"model_load_s": load_s, "translate_s": translate_s, "alternatives_s": alternatives_s},
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
