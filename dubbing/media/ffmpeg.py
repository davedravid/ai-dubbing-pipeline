"""Find FFmpeg, inspect media files, and extract audio."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dubbing.errors import MediaError

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "Install it with:  winget install --id Gyan.FFmpeg --exact\n"
    "then open a NEW terminal window, or set paths.ffmpeg in config.yaml to the full path of ffmpeg.exe."
)


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    has_video: bool
    audio_streams: int
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None


def find_ffmpeg(configured_path: str | None = None) -> FFmpegTools:
    """Locate ffmpeg and ffprobe, preferring the path set in config.yaml."""
    if configured_path:
        ffmpeg = Path(configured_path)
        if not ffmpeg.is_file():
            raise MediaError(f"paths.ffmpeg in config.yaml points to a missing file: {ffmpeg}")
        ffprobe = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        if not ffprobe.is_file():
            raise MediaError(f"ffprobe was not found next to {ffmpeg}")
        return FFmpegTools(str(ffmpeg), str(ffprobe))

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MediaError("FFmpeg was not found on this computer's PATH.\n" + INSTALL_HINT)
    return FFmpegTools(ffmpeg, ffprobe)


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise MediaError(f"FFmpeg could not {what}.\nDetails:\n{tail}")
    return result


def probe(tools: FFmpegTools, media_path: Path) -> MediaInfo:
    """Read duration and stream details of a video or audio file."""
    media_path = Path(media_path)
    if not media_path.is_file():
        raise MediaError(f"Input file not found: {media_path}")

    result = _run(
        [tools.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(media_path)],
        f"read '{media_path.name}' (is it a valid video/audio file?)",
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    # Cover art in audio files shows up as a one-frame video stream; ignore it.
    video = [
        s for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    duration = float(data.get("format", {}).get("duration") or 0.0)
    first_audio = audio[0] if audio else {}

    return MediaInfo(
        path=media_path,
        duration_s=duration,
        has_video=bool(video),
        audio_streams=len(audio),
        audio_codec=first_audio.get("codec_name"),
        sample_rate=int(first_audio["sample_rate"]) if first_audio.get("sample_rate") else None,
        channels=first_audio.get("channels"),
    )


def extract_audio(
    tools: FFmpegTools,
    media_path: Path,
    out_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Write the first audio stream as an uncompressed WAV (16 kHz mono by default)."""
    info = probe(tools, media_path)
    if info.audio_streams == 0:
        raise MediaError(f"'{media_path.name}' has no audio track, so there is no speech to transcribe.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -y: this is a regenerated temp file, safe to replace. -vn: drop video.
    _run(
        [
            tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(media_path),
            "-map", "0:a:0", "-vn",
            "-ac", str(channels), "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        f"extract audio from '{media_path.name}'",
    )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise MediaError(f"FFmpeg finished but produced no audio file at {out_path}")
    log.info("Extracted audio: %s", out_path)
    return out_path
