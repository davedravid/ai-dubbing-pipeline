import wave
from pathlib import Path

import pytest

from conftest import make_test_clip
from dubbing.errors import MediaError
from dubbing.media.ffmpeg import extract_audio, find_ffmpeg, probe


def test_probe_reads_video_and_audio(ffmpeg_tools, tmp_path: Path):
    clip = make_test_clip(ffmpeg_tools, tmp_path / "clip.mp4", seconds=2.0)
    info = probe(ffmpeg_tools, clip)
    assert info.has_video
    assert info.audio_streams == 1
    assert info.channels == 2
    assert info.duration_s == pytest.approx(2.0, abs=0.2)


def test_extract_audio_makes_16k_mono_wav(ffmpeg_tools, tmp_path: Path):
    clip = make_test_clip(ffmpeg_tools, tmp_path / "clip.mp4", seconds=2.0)
    out = extract_audio(ffmpeg_tools, clip, tmp_path / "out" / "audio.wav")
    with wave.open(str(out)) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() / 16000 == pytest.approx(2.0, abs=0.2)


def test_video_without_audio_gives_clear_error(ffmpeg_tools, tmp_path: Path):
    clip = make_test_clip(ffmpeg_tools, tmp_path / "silent.mp4", audio=False)
    with pytest.raises(MediaError, match="no audio track"):
        extract_audio(ffmpeg_tools, clip, tmp_path / "audio.wav")


def test_missing_input_file(ffmpeg_tools, tmp_path: Path):
    with pytest.raises(MediaError, match="not found"):
        probe(ffmpeg_tools, tmp_path / "missing.mp4")


def test_non_media_file_gives_clear_error(ffmpeg_tools, tmp_path: Path):
    junk = tmp_path / "notes.mp4"
    junk.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(MediaError, match="could not read"):
        probe(ffmpeg_tools, junk)


def test_bad_configured_ffmpeg_path(tmp_path: Path):
    with pytest.raises(MediaError, match="missing file"):
        find_ffmpeg(str(tmp_path / "ffmpeg.exe"))
