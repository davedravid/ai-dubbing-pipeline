"""Stage runner (Phase B): approved dubbing script -> conversational translation + review page.

For every sentence (never mixing two speakers): protect names, get several wordings from the
translator, check each one's meaning by translating it back, choose the best for dubbing
(meaning, time, everyday words, no English left), then apply the spoken-Hindi layer
(speaker gender, तुम/आप, names in Hindi letters). Every choice and change is recorded.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dubbing import __version__
from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.languages import get_language
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.transcribe_job import file_sha256, write_json
from dubbing.script.glossary import load_name_spellings, load_name_variants
from dubbing.translator.checks import check_translation
from dubbing.translator.choose import FIT_SLACK, _longest_run, choose
from dubbing.translator.spoken import make_spoken, protect_names
from dubbing.translator.units import build_units

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3
TYPICAL_WORDS_PER_S = 2.8      # average English conversational pace
PACE_RANGE = (0.7, 1.8)
LOW_MEANING = 50.0             # back-translation this far from the English: check the meaning
QUALITY_FLAGS = {"wrong_script", "repetition", "unchanged", "empty_translation", "unusual_length"}


def _translator(settings: Settings, target):
    if target.code != "hi":
        raise PipelineError(f"This version of the pipeline dubs into Hindi only (asked for {target.name}).")
    engine = settings.translation.engine
    if engine == "indictrans2":
        from dubbing.translator.indictrans import IndicTransTranslator

        return IndicTransTranslator(settings.translation.indictrans_model, num_beams=settings.translation.num_beams)
    if engine == "nllb":
        from dubbing.translator.nllb import NllbTranslator

        return NllbTranslator(settings.translation, settings.device)
    raise PipelineError(f"Unknown translation engine '{engine}' (use indictrans2 or nllb)")


def run_translate_script(job_id: str, settings: Settings, lang: str = "hi", overwrite: bool = False,
                         translator=None, referee=None) -> Path:
    """`translator`/`referee` can be passed in (tests use stand-ins); normally they are loaded."""
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    if not paths.script.is_file():
        raise PipelineError(f"Job '{job_id}' has no dubbing script yet. Run 'script' first (Phase A).")
    out_path = paths.translation(lang)
    if out_path.exists() and not overwrite:
        raise OutputExistsError(f"{out_path.name} already exists. Add --overwrite to translate again.")

    script = json.loads(paths.script.read_text(encoding="utf-8"))
    source, target = get_language(script["language"]), get_language(lang)
    names = load_name_spellings(paths.glossary)
    variants = load_name_variants(paths.glossary)
    register = settings.translation.register
    lines = script["lines"]
    by_line = {l["id"]: l for l in lines}
    units = build_units(lines)
    total_s = float(script["source"]["duration_s"])
    timings: dict[str, float] = {}

    # Time available: from where this sentence starts until the next one starts.
    starts = [u["start"] for u in units]
    for k, u in enumerate(units):
        nxt = starts[k + 1] if k + 1 < len(units) else total_s
        u["available_s"] = round(max(nxt, u["end"]) - u["start"], 3)
        words = sum(len(by_line[i]["words"]) for i in u["source_segment_ids"])
        wps = words / max(u["end"] - u["start"], 0.3)
        u["pace"] = round(min(max(wps / TYPICAL_WORDS_PER_S, PACE_RANGE[0]), PACE_RANGE[1]), 2)
        u["gender"] = (script["speakers"].get(u.get("speaker") or "") or {}).get("gender")

    texts = [protect_names(u["text"], names) for u in units]
    n = settings.translation.alternatives + 1

    started = time.perf_counter()
    translator = translator or _translator(settings, target)
    candidates = translator.translate_candidates(texts, source.nllb, target.nllb, n)
    timings["translate_s"] = round(time.perf_counter() - started, 2)
    if len(candidates) != len(units):
        raise PipelineError(f"Translator returned {len(candidates)} results for {len(units)} sentences; nothing saved.")

    # One referee for meaning: translate every option back with NLLB.
    started = time.perf_counter()
    if referee is None:
        from dataclasses import replace

        from dubbing.translator.nllb import NllbTranslator

        # For checking meaning, one quick guess per option is enough (several times faster).
        quick = replace(settings.translation, num_beams=1, batch_size=16)
        referee = NllbTranslator(quick, settings.device)
    flat = sorted({t for c in candidates for t in c if t})
    backs = dict(zip(flat, referee.translate(flat, target.nllb, source.nllb)))
    timings["back_translate_s"] = round(time.perf_counter() - started, 2)

    out_units = []
    for u, options in zip(units, candidates):
        chosen, scored = choose(options, u["text"], backs, u["available_s"], target.chars_per_second, u["pace"])
        # If every option ran away (e.g. a word repeated 13 times), trim the repeats to the English's count.
        max_repeat = max(1, _longest_run(u["text"])) if chosen.get("runaway") else None
        spoken = make_spoken(chosen["text"], u["gender"], register, names, english=u["text"], variants=variants,
                             max_repeat=max_repeat)
        flags, _ = check_translation(u["text"], spoken.text, source, target, u["end"] - u["start"],
                                     u["available_s"], 14.0 * u["pace"])
        flags = [f for f in flags if f in QUALITY_FLAGS]
        est = len(spoken.text) / (target.chars_per_second * u["pace"])
        if est > u["available_s"] * FIT_SLACK:
            flags.append("may_not_fit")
        if chosen["meaning"] < LOW_MEANING:
            flags.append("check_meaning")
        if max_repeat is not None:
            flags.append("repaired_repetition")
        if u.get("speaker_uncertain"):
            flags.append("speaker_uncertain")
        if any(f.startswith("unsure_words") for i in u["source_segment_ids"] for f in by_line[i]["flags"]):
            flags.append("english_unsure")
        out_units.append({
            "id": u["id"],
            "source_segment_ids": u["source_segment_ids"],
            "speaker": u.get("speaker"),
            "speaker_gender": u["gender"],
            "speaker_uncertain": u.get("speaker_uncertain", True),
            "start": u["start"], "end": u["end"], "available_s": u["available_s"], "pace": u["pace"],
            "source_text": u["text"],
            "text": spoken.text,                 # what will be spoken
            "translator_text": chosen["text"],   # before the spoken-Hindi layer
            "changes": spoken.changes,
            "back_translation": chosen["back_translation"],
            "meaning_score": chosen["meaning"],
            "est_speech_s": round(est, 2),
            "chose_first_option": bool(options) and chosen["text"] == options[0],
            "options": scored,
            "flags": flags,
            "status": "translated" if spoken.text else "empty",
        })

    flag_counts = Counter(f for u in out_units for f in u["flags"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": script["source"],
        "source_script": str(paths.script),
        "source_script_sha256": file_sha256(paths.script),
        "source_language": source.code,
        "language": target.code,
        "translator": translator.info(),
        "register": register,
        "summary": {
            "sentences": len(out_units),
            "script lines": len(lines),
            "translated": sum(1 for u in out_units if u["status"] == "translated"),
            "fits its time": sum(1 for u in out_units if "may_not_fit" not in u["flags"]),
            "changed by spoken layer": sum(1 for u in out_units if u["changes"]),
            "median meaning score": sorted(u["meaning_score"] for u in out_units)[len(out_units) // 2],
            "flagged": sum(1 for u in out_units if u["flags"]),
        },
        "flag_counts": dict(flag_counts),
        "timings": timings,
        "segments": out_units,   # same key as earlier translation files, for the later phases
    }
    if [i for u in out_units for i in u["source_segment_ids"]] != [l["id"] for l in lines]:
        raise PipelineError("Sentences do not cover the script lines exactly once; nothing saved.")
    write_json(out_path, result)
    from dubbing.translator.review import write_translation_review

    review = write_translation_review(result, paths.translations_dir / f"{job_id}_{lang}_review.html",
                                      Path(script["source"]["path"]))
    log.info("Saved translation: %s | review page: %s", out_path, review)
    return out_path
