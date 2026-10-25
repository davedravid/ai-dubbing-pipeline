"""Shared test helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dubbing.errors import MediaError
from dubbing.media.ffmpeg import FFmpegTools, find_ffmpeg


@pytest.fixture(scope="session")
def ffmpeg_tools() -> FFmpegTools:
    try:
        return find_ffmpeg()
    except MediaError as exc:
        pytest.skip(f"FFmpeg not available: {exc}")


def make_test_clip(tools: FFmpegTools, out: Path, seconds: float = 2.0, audio: bool = True) -> Path:
    """Build a tiny video (colour frames + a tone) to test FFmpeg handling. Contains no speech."""
    cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=blue:s=160x90:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
                "-c:a", "aac", "-ac", "2"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out
