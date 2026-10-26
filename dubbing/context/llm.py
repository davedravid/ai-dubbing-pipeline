"""Part B of Phase A2: a local AI language model (via Ollama) reads the script like a person.

The model runs on this PC (Ollama at http://127.0.0.1:11434); nothing is sent online.
It reads the script in sections and returns structured notes. Its answers are checked
and combined with the rule-based clues; anything it changes in the English is only
ever a suggestion for a person to approve.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from dubbing.errors import PipelineError

log = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"

SYSTEM = """You are a script supervisor preparing a film scene for dubbing into other languages.
You read dialogue transcribed by speech recognition (it may contain mistakes) with speaker labels
(S1, S2...) detected from voices. The gender given for each label was estimated from voice pitch.
Answer ONLY with JSON in exactly the requested format. Be careful and honest: use null or "unknown"
when you are not sure, and never invent names that are not supported by the dialogue."""

INSTRUCTIONS = """Read this section of the script and return JSON with these keys:

"characters": an object keyed by speaker label, each {"name": a name or title the dialogue gives
  this person, or null; "evidence": short reason}. Only use names other people say to or about them.
  Each name belongs to ONE speaker only; if you are unsure, use null.
"lines": a list with one item per line id: {"id": <line id>, "addressee": list of speaker labels
  being spoken to, or ["group"] if several people, or [] if unclear; "confidence": "high", "medium" or "low"}.
"relationships": a list of {"a": label, "b": label, "relationship": short description,
  "register": "familiar" (friends, equals, arguing) or "formal" (boss to stranger, very polite, respect)}.
"scene": {"summary": one or two sentences, "tone": a few words}.
"terms": list of names and special terms that must be kept consistent.
"corrections": list of {"id": line id, "suggested": the SAME line with only the mis-heard words fixed,
  "reason": which words sound alike}. ONLY for lines where speech recognition clearly mis-heard a word
  or two (the line makes no sense, but a similar-sounding word would). Never replace a line with another
  line, never rewrite or improve wording. Most lines need no correction; an empty list is normal.
"""


def is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def chat_json(model: str, system: str, user: str, timeout: float = 1800, num_ctx: int = 8192) -> dict[str, Any]:
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0.2, "num_ctx": num_ctx},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise PipelineError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}. Start the Ollama app.") from exc
    return json.loads(reply["message"]["content"])


def _format_lines(lines: list[dict[str, Any]], speakers: dict[str, Any]) -> str:
    out = []
    for l in lines:
        g = (speakers.get(l.get("speaker") or "") or {}).get("gender", "unknown")
        out.append(f'{l["id"]} | {l.get("speaker") or "?"} ({g}) | {l["text"]}')
    return "\n".join(out)


def _check(answer: dict[str, Any], ids: set[int]) -> list[str]:
    problems = []
    for key in ("characters", "lines", "relationships", "scene", "corrections"):
        if key not in answer:
            problems.append(f"missing '{key}'")
    got = {item.get("id") for item in answer.get("lines", []) if isinstance(item, dict)}
    if ids - got:
        problems.append(f"{len(ids - got)} line ids missing")
    return problems


def read_script(lines: list[dict[str, Any]], speakers: dict[str, Any], terms: list[str], model: str,
                chunk: int = 60, overlap: int = 8) -> dict[str, Any]:
    """Read the script in sections; carry character notes forward. Returns merged notes."""
    if not is_running():
        raise PipelineError("Ollama is not running. Start the Ollama app (or run 'ollama serve') and try again.")
    merged: dict[str, Any] = {"characters": {}, "lines": {}, "relationships": [], "scenes": [], "terms": [],
                              "corrections": {}, "sections": []}
    start = 0
    while start < len(lines):
        part = lines[start:start + chunk]
        ids = {l["id"] for l in part}
        known = {s: c for s, c in merged["characters"].items() if c.get("name")}
        user = (f"{INSTRUCTIONS}\nKnown names and terms: {', '.join(terms) or 'none'}\n"
                f"Characters identified in earlier sections: {json.dumps(known) if known else 'none yet'}\n"
                f"Speaker genders (from voice pitch): "
                f"{json.dumps({s: p.get('gender') for s, p in speakers.items()})}\n\n"
                f"SCRIPT (line id | speaker (gender) | text):\n{_format_lines(part, speakers)}")
        started = time.perf_counter()
        answer, problems = {}, ["no answer"]
        for attempt in range(2):
            try:
                answer = chat_json(model, SYSTEM, user)
            except json.JSONDecodeError:
                answer = {}
            problems = _check(answer, ids)
            if not problems:
                break
            log.info("  AI answer for lines %d-%d incomplete (%s); asking again", min(ids), max(ids), "; ".join(problems))
        seconds = round(time.perf_counter() - started, 1)
        log.info("  AI read lines %d-%d in %ss%s", min(ids), max(ids), seconds,
                 f" (still incomplete: {'; '.join(problems)})" if problems else "")
        merged["sections"].append({"from": min(ids), "to": max(ids), "seconds": seconds, "problems": problems})
        for spk, info in (answer.get("characters") or {}).items():
            if isinstance(info, dict) and info.get("name") and spk not in merged["characters"]:
                merged["characters"][spk] = {"name": info["name"], "evidence": info.get("evidence", "")}
        for item in answer.get("lines") or []:
            if isinstance(item, dict) and item.get("id") in ids and item["id"] not in merged["lines"]:
                merged["lines"][item["id"]] = {"addressee": item.get("addressee") or [],
                                               "confidence": item.get("confidence", "low")}
        merged["relationships"] += [r for r in answer.get("relationships") or [] if isinstance(r, dict)]
        if isinstance(answer.get("scene"), dict):
            merged["scenes"].append({"from": min(ids), "to": max(ids), **answer["scene"]})
        merged["terms"] += [t for t in answer.get("terms") or [] if isinstance(t, str) and t not in merged["terms"]]
        for c in answer.get("corrections") or []:
            if isinstance(c, dict) and c.get("id") in ids and c.get("suggested"):
                merged["corrections"][c["id"]] = {"suggested": c["suggested"], "reason": c.get("reason", "")}
        if start + chunk >= len(lines):
            break
        start += chunk - overlap
    return merged
