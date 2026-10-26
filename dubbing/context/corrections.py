"""Suggested English corrections, waiting for a person's decision (never applied silently).

The file lists each suggestion with 'accept: null'. Set it to 'yes' or 'no'. Running the
story phase again adds new suggestions but never changes a decision you already made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(k): (v or {}) for k, v in (data.get("corrections") or {}).items()}


def _decision(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    return {"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False}.get(str(value).lower())


def update(path: Path, suggestions: dict[int, dict[str, Any]], originals: dict[int, str]) -> dict[int, dict[str, Any]]:
    """Add new suggestions, keep your decisions, write the file. Returns all with decisions."""
    existing = load(path)
    # Keep every decision you made; drop undecided suggestions that are no longer suggested.
    kept = {lid for lid, c in existing.items() if _decision(c.get("accept")) is not None}
    merged: dict[int, dict[str, Any]] = {}
    for lid in sorted(kept | set(suggestions)):
        old, new = existing.get(lid, {}), suggestions.get(lid, {})
        merged[lid] = {"original": old.get("original") or originals.get(lid, ""),
                       "suggested": old.get("suggested") or new.get("suggested", ""),
                       "reason": old.get("reason") or new.get("reason", ""),
                       "accept": _decision(old.get("accept"))}
    lines = ["# Suggested corrections to the English script (from the story phase).",
             "# For each one set accept: yes  (use the suggestion) or  accept: no  (keep the original).",
             "# Only accepted corrections are used. Your decisions are kept when the phase runs again.",
             "corrections:"]
    q = lambda s: json.dumps(s or "", ensure_ascii=False)  # a double-quoted string is valid YAML
    for lid, c in merged.items():
        lines += [f"  {lid}:",
                  f"    original: {q(c['original'])}",
                  f"    suggested: {q(c['suggested'])}",
                  f"    reason: {q(c['reason'])}",
                  f"    accept: {'yes' if c['accept'] else 'no' if c['accept'] is False else 'null'}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return merged
