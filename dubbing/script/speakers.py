"""Speaker profiles (gender, pitch, how much they speak) and your corrections.

Gender is estimated from voice pitch: adult male voices usually sit around 85-155 Hz,
female around 165-255 Hz. Between the two it is "unknown" and flagged for you to check.
It matters for Hindi grammar (verbs change with the speaker's gender) and for picking
a matching backup voice. Your corrections file always wins over the estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MALE_BELOW_HZ = 150.0
FEMALE_ABOVE_HZ = 165.0


def gender_from_pitch(pitch_hz: float | None) -> tuple[str, float]:
    """Return (gender, confidence 0-1)."""
    if not pitch_hz:
        return "unknown", 0.0
    if pitch_hz < MALE_BELOW_HZ:
        return "male", round(min(1.0, (MALE_BELOW_HZ - pitch_hz) / 40 + 0.5), 2)
    if pitch_hz > FEMALE_ABOVE_HZ:
        return "female", round(min(1.0, (pitch_hz - FEMALE_ABOVE_HZ) / 40 + 0.5), 2)
    return "unknown", 0.3


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): (v or {}) for k, v in (data.get("speakers") or {}).items()}


def load_line_overrides(path: Path) -> dict[int, str]:
    """Per-line speaker corrections: 'lines: {3: S1, 5: S4}' in the corrections file."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(k): str(v) for k, v in (data.get("lines") or {}).items() if v}


def apply_line_overrides(lines: list[dict[str, Any]], line_overrides: dict[int, str]) -> list[dict[str, Any]]:
    """Your per-line speaker corrections win; a new label (e.g. S4) creates a new speaker."""
    for line in lines:
        target = line_overrides.get(line["id"])
        if target:
            line["speaker_detected"] = line.get("speaker")
            line.update(speaker=target, speaker_uncertain=False, speaker_corrected=True)
    return lines


def apply_merges(lines: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """'same_as: S2' in the corrections file relabels a speaker as another one."""
    for line in lines:
        target = (overrides.get(line.get("speaker") or "") or {}).get("same_as")
        if target:
            line["speaker_detected"] = line["speaker"]
            line["speaker"] = str(target)
    return lines


def write_overrides_template(path: Path, profiles: dict[str, dict[str, Any]]) -> bool:
    """Create the corrections file for you to edit, only if it does not exist yet."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ["# Your corrections for the detected speakers. Edit and run 'script --overwrite' again.",
           "#   name:    who it is (e.g. Banner)",
           "#   gender:  male / female (overrides the pitch-based guess)",
           "#   same_as: another label, if two labels are really the same person",
           "# Single lines with the wrong speaker go under 'lines' at the bottom, e.g.  3: S1",
           "# (a new label such as S4 creates a new speaker; give it a gender under 'speakers').",
           "speakers:"]
    for spk, p in profiles.items():
        out.append(f"  {spk}:   # detected: {p['gender_detected']} ({p['median_pitch_hz']} Hz), "
                   f"{p['speech_s']} s of speech, {p['lines']} lines")
        out.append("    name: null")
        out.append("    gender: null")
        out.append("    same_as: null")
    out += ["lines:", "  # 3: S1"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def build_profiles(lines: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        if line.get("speaker"):
            by.setdefault(line["speaker"], []).append(line)
    profiles = {}
    for spk in sorted(by, key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
        ls = by[spk]
        pitched = [(l["delivery"]["pitch_median_hz"], l["end"] - l["start"]) for l in ls
                   if l["delivery"].get("pitch_median_hz") and not l.get("speaker_uncertain")]
        if pitched:
            values = sorted(pitched)
            half, acc, pitch = sum(w for _, w in values) / 2, 0.0, None
            for v, w in values:  # duration-weighted median
                acc += w
                if acc >= half:
                    pitch = v
                    break
        else:
            pitch = None
        gender, conf = gender_from_pitch(pitch)
        o = overrides.get(spk, {})
        profiles[spk] = {
            "name": o.get("name"),
            "gender": o.get("gender") or gender,
            "gender_detected": gender,
            "gender_confidence": 1.0 if o.get("gender") else conf,
            "gender_source": "your correction" if o.get("gender") else "voice pitch",
            "median_pitch_hz": round(pitch, 1) if pitch else None,
            "lines": len(ls),
            "speech_s": round(sum(l["end"] - l["start"] for l in ls), 1),
            "uncertain_lines": sum(1 for l in ls if l.get("speaker_uncertain")),
        }
        if profiles[spk]["gender"] == "unknown":
            profiles[spk]["flags"] = ["check_gender"]
    return profiles
