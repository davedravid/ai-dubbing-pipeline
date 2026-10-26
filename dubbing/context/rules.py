"""Rule-based context clues (part A of Phase A2): fast, transparent, and usually right.

1. Names/titles said to someone ("Agent Romanoff, ...", "..., Stark?") show who is addressed,
   and are clues to that person's name (the person named usually answers next).
2. Plural words ("you people", "you boys", "kids") mean a group is addressed.
3. Turn-taking: in an A-B-A exchange, A is talking to B.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

TITLES = ["Agent", "Doctor", "Dr.", "Captain", "Mr.", "Mrs.", "Ms.", "Miss", "Sir", "Ma'am", "Madam",
          "Professor", "Detective", "Officer", "General", "Colonel", "Director"]
PLURAL_CUES = re.compile(r"\b(you people|you guys|you boys|you girls|you two|you all|y'all|all of you|"
                         r"everyone|everybody|kids|ladies|gentlemen|folks|team|boys|guys)\b", re.IGNORECASE)
GENERIC_VOCATIVES = {"man", "buddy", "pal", "dude", "sir", "ma'am", "madam", "son", "kid", "honey", "doctor"}


def vocatives(text: str, names: list[str]) -> list[str]:
    """Names or titles used to address someone: at the start ("Stark, ...") or end (", Stark?")."""
    found = []
    candidates = sorted(set(names) | set(TITLES), key=len, reverse=True)
    for name in candidates:
        n = re.escape(name).replace(r"\ ", r"\s+")
        start = re.search(rf"^\s*(?:(?:{'|'.join(map(re.escape, TITLES))})\s+)?{n}\s*[,!?]", text, re.IGNORECASE)
        end = re.search(rf",\s*(?:(?:{'|'.join(map(re.escape, TITLES))})\s+)?{n}\s*[.?!]*\s*$", text, re.IGNORECASE)
        if (start or end) and not any(name.lower() in f.lower() for f in found):
            found.append(name)
    return found


def _neighbours(lines: list[dict[str, Any]], i: int) -> tuple[str | None, str | None]:
    me = lines[i].get("speaker")
    prev = next((lines[j]["speaker"] for j in range(i - 1, -1, -1) if lines[j].get("speaker") not in (None, me)), None)
    nxt = next((lines[j]["speaker"] for j in range(i + 1, len(lines)) if lines[j].get("speaker") not in (None, me)), None)
    return prev, nxt


def _speaks_after(lines: list[dict[str, Any]], i: int, other: str) -> str | None:
    """Who speaks next after `other` has answered line i (for spotting A-B-A going forwards)."""
    seen_other = False
    for j in range(i + 1, len(lines)):
        spk = lines[j].get("speaker")
        if spk == other:
            seen_other = True
        elif seen_other and spk is not None:
            return spk
    return None


def analyse(lines: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Per line: an addressee guess with evidence; per speaker: name votes."""
    per_line: dict[int, dict[str, Any]] = {}
    name_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    proper = [n for n in names if n not in TITLES]
    for i, line in enumerate(lines):
        text = line.get("text", "")
        prev, nxt = _neighbours(lines, i)
        voc = vocatives(text, proper + TITLES)
        plural = bool(PLURAL_CUES.search(text))
        if plural:
            guess = {"addressee": ["group"], "plural": True, "confidence": "high",
                     "evidence": f"plural words: {PLURAL_CUES.search(text).group(0)!r}"}
        elif prev and nxt and prev == nxt:
            guess = {"addressee": [prev], "plural": False, "confidence": "high",
                     "evidence": f"back-and-forth with {prev}"}
        elif nxt and _speaks_after(lines, i, nxt) == line.get("speaker"):
            guess = {"addressee": [nxt], "plural": False, "confidence": "high",
                     "evidence": f"{nxt} answers, then this speaker replies"}
        elif prev or nxt:
            guess = {"addressee": [prev or nxt], "plural": False, "confidence": "low",
                     "evidence": "nearest other speaker"}
        else:
            guess = {"addressee": [], "plural": False, "confidence": "low", "evidence": "no other speaker nearby"}
        if voc:
            guess["vocatives"] = voc
            guess["evidence"] += f"; addressed as {', '.join(voc)}"
            # The person named usually answers next; otherwise it is the person just answered.
            for name in voc:
                if name.lower() in GENERIC_VOCATIVES or name in TITLES:
                    continue
                if nxt:
                    name_votes[nxt][name] += 0.6
                if prev:
                    name_votes[prev][name] += 0.4
        per_line[line["id"]] = guess
    return {"lines": per_line, "name_votes": {s: dict(v) for s, v in name_votes.items()}}
