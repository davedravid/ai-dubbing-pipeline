"""Write WebVTT (.vtt) subtitles: the format web players and browsers use."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Iterable

# Common subtitle guidance: at most 2 lines of about 42 characters each.
MAX_LINE_CHARS = 42
MAX_LINES = 2


def format_timestamp(seconds: float) -> str:
    """12.3456 -> '00:00:12.346' (WebVTT requires HH:MM:SS.mmm)."""
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _escape(text: str) -> str:
    # '<', '>' and '&' have special meaning inside WebVTT cue text.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap_text(text: str, width: int = MAX_LINE_CHARS) -> list[str]:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines or [text]


def build_cues(segments: Iterable[dict[str, Any]]) -> list[tuple[float, float, list[str]]]:
    """Turn segments into (start, end, lines). Long segments are split into several cues,
    with time shared out in proportion to the number of characters."""
    cues: list[tuple[float, float, list[str]]] = []
    for seg in segments:
        text = " ".join(str(seg.get("text", "")).split())
        if not text:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        lines = wrap_text(text)
        chunks = [lines[i:i + MAX_LINES] for i in range(0, len(lines), MAX_LINES)]
        total_chars = sum(len(" ".join(c)) for c in chunks)
        cursor = start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk_end = end
            else:
                chunk_end = cursor + (end - start) * len(" ".join(chunk)) / total_chars
            cues.append((cursor, chunk_end, chunk))
            cursor = chunk_end
    return cues


def render_vtt(segments: Iterable[dict[str, Any]]) -> str:
    parts = ["WEBVTT", ""]
    for number, (start, end, lines) in enumerate(build_cues(segments), start=1):
        parts.append(str(number))
        parts.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        parts.extend(_escape(line) for line in lines)
        parts.append("")
    return "\n".join(parts)


def write_vtt(segments: Iterable[dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_vtt(segments), encoding="utf-8", newline="\n")
    return out_path
