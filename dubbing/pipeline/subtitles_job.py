"""Stage runner: transcript JSON -> subtitle file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dubbing.config import Settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.paths import JobPaths, validate_job_id
from dubbing.subtitles.vtt import write_vtt

log = logging.getLogger(__name__)


def run_subtitles(job_id: str, settings: Settings, lang: str | None = None, overwrite: bool = False) -> Path:
    """Write data/subtitles/<job>_<lang>.vtt from that job's translation or transcript.

    With `lang`, a translation into that language is used if it exists, otherwise the
    transcript in that language. Without `lang`, the job's only transcript is used.
    """
    paths = JobPaths(settings.data_dir, validate_job_id(job_id))

    if lang:
        # The speech manifest holds the wording actually spoken (it may be a shorter one).
        candidates = [paths.tts_manifest(lang), paths.translation(lang), paths.transcript(lang)]
        transcript_path = next((p for p in candidates if p.is_file()), None)
        if transcript_path is None:
            raise PipelineError(
                f"Job '{job_id}' has no '{lang}' transcript or translation. "
                "Run 'transcribe' or 'translate' first."
            )
    else:
        found = paths.existing_transcripts()
        if not found:
            raise PipelineError(f"Job '{job_id}' has no transcript yet. Run 'transcribe' first.")
        if len(found) > 1:
            names = ", ".join(p.name for p in found)
            raise PipelineError(f"Job '{job_id}' has several transcripts ({names}); choose one with --lang.")
        transcript_path = found[0]

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    lang = data["language"]
    segments = data.get("segments") or data.get("units") or []

    out_path = paths.subtitle(lang, "vtt")
    if out_path.exists() and not overwrite:
        raise OutputExistsError(f"{out_path.name} already exists. Add --overwrite to replace it.")

    write_vtt(segments, out_path)
    log.info("Saved subtitles: %s (from %d segments in %s)", out_path, len(segments), transcript_path.name)
    return out_path
