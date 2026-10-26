"""FFmpeg steps for the final video: tighten speech, build the MP4, and check the result."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dubbing.errors import MediaError
from dubbing.media.ffmpeg import FFmpegTools, _run


def tighten(tools: FFmpegTools, in_wav: Path, out_wav: Path, factor: float) -> Path:
    """Play speech `factor` times faster without changing the pitch of the voice (Rubber Band)."""
    _run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_wav),
          "-af", f"rubberband=tempo={factor:.4f}:pitchq=quality", str(out_wav)],
         f"tighten '{in_wav.name}'")
    return out_wav


@dataclass(frozen=True)
class AudioTrack:
    path: Path | None        # None = take the original video's own audio stream
    language: str            # ISO 639-2, e.g. 'hin', 'eng'
    title: str
    default: bool


@dataclass(frozen=True)
class SubtitleTrack:
    path: Path
    language: str
    title: str


def build_video(
    tools: FFmpegTools,
    video_in: Path,
    out_path: Path,
    audio_tracks: list[AudioTrack],
    subtitles: list[SubtitleTrack],
    bitrate: str = "192k",
    limiter_peak_db: float = -1.0,
    comment: str = "",
) -> Path:
    """Copy the original picture untouched and add the given audio and subtitle tracks."""
    cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_in)]
    inputs = 1
    audio_maps: list[str] = []
    new_audio: list[int] = []
    for t in audio_tracks:
        if t.path is None:
            audio_maps.append("0:a:0")
        else:
            cmd += ["-i", str(t.path)]
            audio_maps.append(f"{inputs}:a:0")
            new_audio.append(len(audio_maps) - 1)
            inputs += 1
    sub_inputs = []
    for s in subtitles:
        cmd += ["-i", str(s.path)]
        sub_inputs.append(inputs)
        inputs += 1

    cmd += ["-map", "0:v:0"]
    for m in audio_maps:
        cmd += ["-map", m]
    for i in sub_inputs:
        cmd += ["-map", f"{i}:s:0"]

    cmd += ["-c:v", "copy"]   # the picture is copied exactly, never re-encoded
    limit = 10 ** (limiter_peak_db / 20)
    for idx, t in enumerate(audio_tracks):
        if t.path is None:
            cmd += [f"-c:a:{idx}", "copy"]
        else:
            # The limiter stops any peak above limiter_peak_db (prevents distortion).
            cmd += [f"-filter:a:{idx}", f"alimiter=limit={limit:.4f}:level=false",
                    f"-c:a:{idx}", "aac", f"-b:a:{idx}", bitrate]
        # MP4 keeps a track's name in "handler_name"; players show it in their audio menu.
        cmd += [f"-metadata:s:a:{idx}", f"language={t.language}", f"-metadata:s:a:{idx}", f"handler_name={t.title}",
                f"-disposition:a:{idx}", "default" if t.default else "0"]
    for idx, s in enumerate(subtitles):
        cmd += [f"-c:s:{idx}", "mov_text", f"-metadata:s:s:{idx}", f"language={s.language}",
                f"-metadata:s:s:{idx}", f"handler_name={s.title}", f"-disposition:s:{idx}", "0"]
    if comment:
        cmd += ["-metadata", f"comment={comment}"]
    cmd += ["-movflags", "+faststart", str(out_path)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd, f"build the dubbed video '{out_path.name}'")
    return out_path


# ---- checks on a finished file ------------------------------------------------

def streams(tools: FFmpegTools, path: Path) -> list[dict]:
    result = _run([tools.ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                   str(path)], f"read '{path.name}'")
    data = json.loads(result.stdout or "{}")
    out = []
    for s in data.get("streams", []):
        out.append({
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
            "language": (s.get("tags") or {}).get("language"),
            "title": (s.get("tags") or {}).get("handler_name") or (s.get("tags") or {}).get("title"),
            "default": bool((s.get("disposition") or {}).get("default")),
            "duration_s": round(float(s["duration"]), 3) if s.get("duration") else None,
        })
    return out


def video_fingerprint(tools: FFmpegTools, path: Path) -> str:
    """MD5 of the video stream's data packets: identical fingerprints = identical picture."""
    result = _run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                   "-c", "copy", "-f", "md5", "-"], f"fingerprint the picture of '{path.name}'")
    return result.stdout.strip().removeprefix("MD5=")


def decode_errors(tools: FFmpegTools, path: Path) -> list[str]:
    """Play all picture and sound tracks to the end like a player would; return any errors.
    (Subtitles are checked separately with subtitle_cues: this decoder mode cannot take them.)"""
    result = subprocess.run([tools.ffmpeg, "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v",
                             "-map", "0:a", "-f", "null", "-"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    errors = [line for line in result.stderr.splitlines() if line.strip()]
    if result.returncode != 0 and not errors:
        errors = [f"ffmpeg exited with code {result.returncode}"]
    return errors


def subtitle_cues(tools: FFmpegTools, path: Path, subtitle_index: int) -> int:
    """Read one subtitle track back out of the video and count its captions."""
    result = _run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
                   "-map", f"0:s:{subtitle_index}", "-f", "srt", "-"], f"read subtitle track {subtitle_index}")
    return result.stdout.count(" --> ")


def extract_track(tools: FFmpegTools, path: Path, audio_index: int, out_wav: Path, sr: int = 44100) -> Path:
    _run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-map", f"0:a:{audio_index}",
          "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(out_wav)], f"read audio track {audio_index}")
    if not out_wav.is_file():
        raise MediaError(f"Could not extract audio track {audio_index} from {path.name}")
    return out_wav
