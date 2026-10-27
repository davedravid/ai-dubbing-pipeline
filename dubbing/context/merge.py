"""Combine the rule-based clues (part A) with the AI model's reading (part B).

Agreement raises confidence; disagreement is shown to the reviewer instead of being
silently resolved. A name said in the line itself ("..., Stark?") always wins.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def suggest_names(rule_votes: dict[str, dict[str, float]], ai_chars: dict[str, Any],
                  speakers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # A name the AI gave to more than one speaker is not trusted for any of them.
    given = [str((ai_chars.get(s) or {}).get("name") or "").lower() for s in speakers]
    repeated = {n for n in given if n and given.count(n) > 1}
    out = {}
    for spk in speakers:
        votes = rule_votes.get(spk, {})
        rule_name = max(votes, key=votes.get) if votes else None
        ai_name = (ai_chars.get(spk) or {}).get("name")
        if ai_name and ai_name.lower() in repeated:
            ai_name = None
        if rule_name and ai_name and (rule_name.lower() in ai_name.lower() or ai_name.lower() in rule_name.lower()):
            name, conf, agree = ai_name if len(ai_name) >= len(rule_name) else rule_name, "high", True
        else:
            name, conf, agree = ai_name or rule_name, "medium" if (ai_name or rule_name) else "low", None
            if rule_name and ai_name:
                conf, agree = "low", False
        out[spk] = {"suggested_name": name, "confidence": conf, "rules_say": rule_name, "ai_says": ai_name,
                    "ai_evidence": (ai_chars.get(spk) or {}).get("evidence"), "sources_agree": agree}
    return out


def usable_corrections(suggestions: dict[int, dict[str, Any]], originals: dict[int, str]) -> dict[int, dict[str, Any]]:
    """Keep only real fixes of mis-heard words: changed, not a copy of another line, and still
    mostly the same line (a rewrite is not a hearing correction)."""
    from dubbing.qa.text_metrics import chrf

    others = {t.strip().lower() for t in originals.values()}
    out = {}
    for lid, c in suggestions.items():
        old, new = originals.get(lid, "").strip(), str(c.get("suggested", "")).strip()
        if not new or new.lower() == old.lower():
            continue
        if new.lower() in others:            # copied another line of the script
            continue
        if chrf(old, new) < 50:              # rewrote the line rather than fixing a few words
            continue
        out[lid] = c
    return out


def _speaker_by_name(names: dict[str, dict[str, Any]], vocative: str) -> str | None:
    v = vocative.lower()
    for spk, n in names.items():
        name = (n.get("final_name") or n.get("suggested_name") or "").lower()
        if name and (v in name or name in v):
            return spk
    return None


def merge_lines(lines: list[dict[str, Any]], rules: dict[int, dict[str, Any]], ai: dict[int, dict[str, Any]],
                names: dict[str, dict[str, Any]], speakers: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    valid = set(speakers) | {"group"}
    for line in lines:
        lid, me = line["id"], line.get("speaker")
        r = rules.get(lid, {})
        a = ai.get(lid, {})
        r_to = [x for x in r.get("addressee", []) if x in valid and x != me]
        a_to = [x for x in a.get("addressee", []) if x in valid and x != me]
        if len(a_to) > 1:
            a_to = ["group"]
        named = [s for v in r.get("vocatives", []) if (s := _speaker_by_name(names, v)) and s != me]
        if named:
            to, conf, agree, why = named[:1], "high", (a_to == named[:1]) if a_to else None, "name said in the line"
        elif r_to and a_to and r_to == a_to:
            to, conf, agree, why = r_to, "high", True, "rules and AI agree"
        elif r_to and r.get("confidence") == "high":
            # Confident rules (back-and-forth, plural words) beat the AI: on a fast multi-person
            # argument the local AI model contradicted clear turn-taking.
            to, conf, agree, why = r_to, "high", (False if a_to else None), r.get("evidence", "rules")
        elif a_to:
            to, conf, agree, why = a_to, "medium" if r_to else a.get("confidence", "low"), (False if r_to else None), "AI reading"
        else:
            to, conf, agree, why = r_to, r.get("confidence", "low"), None, r.get("evidence", "rules")
        if agree is False and conf == "high" and why != "name said in the line":
            conf = "medium"  # a name said in the line stays certain, whatever the AI thinks
        if to == ["group"] and r.get("plural") and r_to == ["group"]:
            why = r.get("evidence", why)  # keep "plural words: ..." as the reason
        out[lid] = {"speaker": me, "addressee": to, "plural": to == ["group"] or bool(r.get("plural")),
                    "confidence": conf, "sources_agree": agree, "reason": why,
                    "rules_say": r_to, "ai_says": a_to}
    return out


def share_within_runs(lines: list[dict[str, Any]], merged: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """When one person says several lines in a row, they are usually talking to the same listener.
    If one line in the run clearly says who (a name, plural words, rules+AI agree), the other
    lines in the run that are less sure take that listener."""
    runs, current = [], []
    for line in lines:
        if current and line.get("speaker") != current[-1].get("speaker"):
            runs.append(current)
            current = []
        current.append(line)
    if current:
        runs.append(current)
    for run in runs:
        anchors = [merged[l["id"]] for l in run if merged[l["id"]]["confidence"] == "high"
                   and merged[l["id"]]["reason"] in ("name said in the line", "rules and AI agree")
                   or merged[l["id"]]["reason"].startswith("plural words")]
        if not anchors:
            continue
        anchor = anchors[0]
        for l in run:
            m = merged[l["id"]]
            if m["confidence"] != "high" or m["reason"] not in ("name said in the line", "rules and AI agree") \
                    and not m["reason"].startswith("plural words"):
                if m["addressee"] != anchor["addressee"]:
                    m.update(addressee=list(anchor["addressee"]), plural=anchor["plural"],
                             reason=f'same speech as a line that says who ({anchor["reason"]})')
                m["confidence"] = "high"
    return merged


def addressee_gender(addressee: list[str], speakers: dict[str, Any], nearby: list[str]) -> str:
    """'male'/'female' for one person; for a group, 'female' only if everyone nearby is female."""
    if addressee == ["group"]:
        genders = {(speakers.get(s) or {}).get("gender") for s in nearby}
        genders.discard(None)
        return "female_group" if genders == {"female"} else "male_or_mixed_group"
    if len(addressee) == 1:
        return (speakers.get(addressee[0]) or {}).get("gender", "unknown")
    return "unknown"


def pair_registers(relationships: list[dict[str, Any]], speakers: dict[str, Any], default: str) -> dict[str, Any]:
    """'familiar'/'formal' for each speaker pair (A>B), from the AI's notes; default otherwise."""
    votes: dict[str, list[str]] = defaultdict(list)
    notes: dict[str, str] = {}
    for r in relationships:
        a, b, reg = r.get("a"), r.get("b"), r.get("register")
        if a in speakers and b in speakers and reg in ("familiar", "formal"):
            for key in (f"{a}>{b}", f"{b}>{a}"):
                votes[key].append(reg)
                notes.setdefault(key, r.get("relationship", ""))
    out = {}
    for key, v in votes.items():
        reg = max(set(v), key=v.count)
        out[key] = {"register": reg, "relationship": notes.get(key, ""), "votes": len(v)}
    out["_default"] = {"register": default, "relationship": "used for any pair not listed"}
    return out
