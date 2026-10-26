"""Stage runner (Phase A2): approved dubbing script -> story bible + review page.

Reads the whole script (in sections) with rules (part A) and a local AI model (part B):
characters and names, who each line is spoken to, the listener's gender, relationship and
formality per pair, scene and tone, names/terms, and suggested English corrections (which
wait for a person's approval in the corrections file).
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
from dubbing.context import corrections as corr
from dubbing.context.llm import read_script
from dubbing.context.merge import (addressee_gender, merge_lines, pair_registers, suggest_names,
                                   usable_corrections)
from dubbing.context.rules import analyse
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.paths import JobPaths, validate_job_id
from dubbing.pipeline.transcribe_job import file_sha256, write_json
from dubbing.script.glossary import load_glossary
from dubbing.script.speakers import load_overrides

log = logging.getLogger(__name__)

NEARBY_LINES = 6   # speakers within this many lines count as "present" for a group


def run_story(job_id: str, settings: Settings, overwrite: bool = False, ai_reader=None,
              reread: bool = False) -> Path:
    """`ai_reader(lines, speakers, terms)` can be passed in (tests use a stand-in).
    The AI's raw notes are saved and re-used for the same script unless `reread` is set,
    so changing how they are combined does not mean another long AI read."""
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))
    if not paths.script.is_file():
        raise PipelineError(f"Job '{job_id}' has no dubbing script yet. Run 'script' first (Phase A).")
    if paths.story.exists() and not overwrite:
        raise OutputExistsError(f"{paths.story.name} already exists. Add --overwrite to read the script again.")

    script = json.loads(paths.script.read_text(encoding="utf-8"))
    lines, speakers = script["lines"], script["speakers"]
    terms = load_glossary(paths.glossary, extra=settings.asr.glossary)
    overrides = load_overrides(paths.speaker_overrides)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    rules = analyse(lines, terms)
    timings["rules_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    script_sha = file_sha256(paths.script)
    notes_path = paths.story.with_name(f"{job_id}_ai_notes.json")
    cached = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.is_file() else None
    if cached and not reread and cached.get("script_sha256") == script_sha and cached.get("model") == settings.context.model:
        ai = cached["notes"]
        ai["lines"] = {int(k): v for k, v in ai.get("lines", {}).items()}
        ai["corrections"] = {int(k): v for k, v in ai.get("corrections", {}).items()}
        log.info("Re-using the AI's earlier reading of this script (%s)", notes_path.name)
    else:
        if ai_reader is None:
            def ai_reader(ls, spk, t):
                return read_script(ls, spk, t, settings.context.model, settings.context.chunk_lines)
        ai = ai_reader(lines, speakers, terms)
        write_json(notes_path, {"script_sha256": script_sha, "model": settings.context.model, "notes": ai})
    timings["ai_s"] = round(time.perf_counter() - started, 2)

    names = suggest_names(rules["name_votes"], ai.get("characters", {}), speakers)
    for spk, n in names.items():
        n["final_name"] = (overrides.get(spk) or {}).get("name")  # your name for a speaker always wins
    per_line = merge_lines(lines, rules["lines"], ai.get("lines", {}), names, speakers)
    ids = [l["id"] for l in lines]
    for i, line in enumerate(lines):
        nearby = {lines[j].get("speaker") for j in range(max(0, i - NEARBY_LINES), min(len(lines), i + NEARBY_LINES + 1))}
        nearby.discard(line.get("speaker"))
        info = per_line[line["id"]]
        info["addressee_gender"] = addressee_gender(info["addressee"], speakers, sorted(s for s in nearby if s))
        info["speaker_gender"] = (speakers.get(line.get("speaker") or "") or {}).get("gender")
        info["text"] = line["text"]
        info["start"] = line["start"]
    pairs = pair_registers(ai.get("relationships", []), speakers, settings.context.default_register)

    originals = {l["id"]: l["text"] for l in lines}
    suggestions = usable_corrections(ai.get("corrections", {}), originals)
    decisions = corr.update(paths.corrections_file, suggestions, originals)

    conf = Counter(v["confidence"] for v in per_line.values())
    story = {
        "pipeline_version": __version__,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_script": str(paths.script),
        "source_script_sha256": file_sha256(paths.script),
        "ai_model": settings.context.model,
        "summary": {
            "lines": len(lines),
            "addressee high confidence": conf.get("high", 0),
            "addressee medium": conf.get("medium", 0),
            "addressee low": conf.get("low", 0),
            "rules and AI disagree": sum(1 for v in per_line.values() if v["sources_agree"] is False),
            "said to a group": sum(1 for v in per_line.values() if v["plural"]),
            "names suggested": sum(1 for n in names.values() if n["suggested_name"]),
            "corrections waiting for you": sum(1 for c in decisions.values() if c["accept"] is None),
        },
        "characters": {spk: {**speakers[spk], **names.get(spk, {})} for spk in speakers},
        "pairs": pairs,
        "scenes": ai.get("scenes", []),
        "terms": sorted(set(terms) | set(ai.get("terms", []))),
        "lines": {str(k): per_line[k] for k in ids},
        "corrections": {str(k): v for k, v in decisions.items()},
        "corrections_file": str(paths.corrections_file),
        "ai_sections": ai.get("sections", []),
        "timings": timings,
    }
    write_json(paths.story, story)
    from dubbing.context.review import write_story_review

    write_story_review(story, script, paths.story_review, Path(script["source"]["path"]))
    log.info("Saved story bible: %s | review page: %s | corrections: %s",
             paths.story, paths.story_review, paths.corrections_file)
    return paths.story
