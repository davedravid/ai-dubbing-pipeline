"""Where each job's files live inside the data folder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dubbing.errors import PipelineError

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_job_id(job_id: str) -> str:
    if not _JOB_ID_PATTERN.match(job_id):
        raise PipelineError(
            f"Invalid job id '{job_id}'. Use 1-64 letters, numbers, '_' or '-' (no spaces)."
        )
    return job_id


@dataclass(frozen=True)
class JobPaths:
    data_dir: Path
    job_id: str

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "temp"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def translations_dir(self) -> Path:
        return self.data_dir / "translations"

    @property
    def subtitles_dir(self) -> Path:
        return self.data_dir / "subtitles"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def asr_audio(self) -> Path:
        """16 kHz mono WAV: the format Whisper works with."""
        return self.temp_dir / f"{self.job_id}_asr16k.wav"

    @property
    def source_audio(self) -> Path:
        """24 kHz mono WAV: better quality than the ASR copy; used for analysis and voice references."""
        return self.temp_dir / f"{self.job_id}_source24k.wav"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    def features(self, lang: str) -> Path:
        return self.features_dir / f"{self.job_id}_{lang}.json"

    @property
    def references_dir(self) -> Path:
        return self.data_dir / "references" / self.job_id

    @property
    def tts_root(self) -> Path:
        return self.data_dir / "tts"

    def tts_dir(self, lang: str) -> Path:
        return self.tts_root / f"{self.job_id}_{lang}"

    def tts_manifest(self, lang: str) -> Path:
        return self.tts_root / f"{self.job_id}_{lang}.json"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    def speech_preview(self, lang: str) -> Path:
        """Dubbed speech only, placed on the video's timeline (no music/effects yet)."""
        return self.outputs_dir / f"{self.job_id}_{lang}_speech_preview.wav"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.job_id}.log"

    def transcript(self, lang: str) -> Path:
        return self.transcripts_dir / f"{self.job_id}_{lang}.json"

    def translation(self, lang: str) -> Path:
        return self.translations_dir / f"{self.job_id}_{lang}.json"

    def subtitle(self, lang: str, ext: str = "vtt") -> Path:
        return self.subtitles_dir / f"{self.job_id}_{lang}.{ext}"

    def existing_transcripts(self) -> list[Path]:
        return sorted(self.transcripts_dir.glob(f"{self.job_id}_*.json"))

    def ensure_dirs(self) -> None:
        for folder in (self.temp_dir, self.transcripts_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)
