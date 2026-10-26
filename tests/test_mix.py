import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import make_test_clip
from dubbing.audio.fit import plan_fit
from dubbing.media.mux import (AudioTrack, SubtitleTrack, build_video, decode_errors, extract_track, streams,
                               subtitle_cues, tighten, video_fingerprint)
from dubbing.qa.audio_metrics import integrated_lufs, soundtrack_metrics

SR = 44100


def _tone(freq, seconds, amp=0.3, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---- timing fixes ----------------------------------------------------------

def _line(i, start, dur):
    return {"id": i, "start": start, "duration": dur}


def test_line_that_fits_is_left_alone():
    [p] = plan_fit([_line(0, 1.0, 2.0)], total_s=10)
    assert (p.start, p.tighten, p.flags) == (1.0, 1.0, [])


def test_pause_after_the_line_is_used_first():
    # 3 s line, next line at 5 s: fits in the pause without any change.
    p = plan_fit([_line(0, 1.0, 3.0), _line(1, 5.0, 1.0)], total_s=10)[0]
    assert (p.start, p.tighten, p.flags) == (1.0, 1.0, [])


def test_early_start_before_tightening():
    # Needs 0.2 s more; there is a pause before it, so it starts 0.2 s early instead of speeding up.
    p = plan_fit([_line(0, 2.0, 1.12), _line(1, 3.0, 1.0)], total_s=10, max_early_s=0.3, min_gap_s=0.08)[0]
    assert p.flags == ["started_early"] and p.tighten == 1.0
    assert p.start == pytest.approx(1.8, abs=0.01) and p.overlap_s == 0


def test_tightening_is_capped_and_flagged():
    # 2 s of speech for a 1 s slot with no pause before: tighten to the cap, flag the rest.
    p = plan_fit([_line(0, 0.0, 2.0), _line(1, 1.08, 1.0)], total_s=10, max_early_s=0.3,
                 min_gap_s=0.08, max_tighten=1.15)[0]
    assert p.tighten == 1.15
    assert {"tightened", "needs_shorter_translation", "overlaps_next_line"} <= set(p.flags)
    assert p.overlap_s == pytest.approx(2.0 / 1.15 - 1.08, abs=0.01)


def test_lines_never_start_later_than_original_and_never_before_previous_ends():
    lines = [_line(0, 0.0, 1.0), _line(1, 1.1, 1.5), _line(2, 2.5, 1.0)]
    ps = plan_fit(lines, total_s=10)
    for p, line in zip(ps, lines):
        assert p.start <= line["start"] + 1e-9
    for a, b in zip(ps, ps[1:]):
        assert b.start >= a.start


def test_last_line_can_use_rest_of_video():
    [p] = plan_fit([_line(0, 8.0, 1.9)], total_s=10)
    assert p.flags == []


# ---- loudness measurements -------------------------------------------------

def test_louder_signal_measures_louder_lufs():
    quiet, loud = _tone(440, 2.0, 0.05), _tone(440, 2.0, 0.5)
    assert integrated_lufs(loud, SR) - integrated_lufs(quiet, SR) == pytest.approx(20.0, abs=0.5)


def test_soundtrack_metrics_detect_clipping_and_silence():
    y = np.concatenate([np.zeros(SR, np.float32), np.clip(_tone(440, 1.0, 2.0), -1, 1)])
    m = soundtrack_metrics(y, SR)
    assert m["clipped_ratio"] > 0
    assert m["silence_ratio"] == pytest.approx(0.5, abs=0.05)


# ---- FFmpeg: tighten and build the final video ----------------------------

def test_tighten_keeps_pitch_and_shortens(ffmpeg_tools, tmp_path: Path):
    from dubbing.analysis.prosody import analyze_clip

    src = tmp_path / "in.wav"
    sf.write(str(src), _tone(220, 2.0, sr=24000), 24000)
    out = tighten(ffmpeg_tools, src, tmp_path / "out.wav", 1.15)
    y, sr = sf.read(str(out), dtype="float32")
    assert len(y) / sr == pytest.approx(2.0 / 1.15, abs=0.05)
    assert analyze_clip(y, sr)["pitch_median_hz"] == pytest.approx(220, rel=0.03)  # no "chipmunk"


def test_build_video_tracks_subtitles_and_untouched_picture(ffmpeg_tools, tmp_path: Path):
    video = make_test_clip(ffmpeg_tools, tmp_path / "clip.mp4", seconds=3.0)
    dub = tmp_path / "dub.wav"
    sf.write(str(dub), np.stack([_tone(300, 3.0)] * 2, axis=1), SR)
    vtt = tmp_path / "hi.vtt"
    vtt.write_text("WEBVTT\n\n1\n00:00:00.500 --> 00:00:01.500\nनमस्ते\n", encoding="utf-8")

    out = build_video(ffmpeg_tools, video, tmp_path / "out.mp4",
                      [AudioTrack(dub, "hin", "Hindi (AI dub)", True), AudioTrack(None, "eng", "English (original)", False)],
                      [SubtitleTrack(vtt, "hin", "Hindi")], comment="AI-generated test")
    info = streams(ffmpeg_tools, out)
    assert [s["type"] for s in info] == ["video", "audio", "audio", "subtitle"]
    # Hindi dub is the default audio; the original is a switchable second track.
    assert [(s["language"], s["default"], s["title"]) for s in info if s["type"] == "audio"] == [
        ("hin", True, "Hindi (AI dub)"), ("eng", False, "English (original)")]
    # MP4 always switches on the first subtitle track when none is marked default; viewers can turn it off.
    assert [s["language"] for s in info if s["type"] == "subtitle"] == ["hin"]
    assert video_fingerprint(ffmpeg_tools, video) == video_fingerprint(ffmpeg_tools, out)
    assert decode_errors(ffmpeg_tools, out) == []
    assert subtitle_cues(ffmpeg_tools, out, 0) == 1

    back = extract_track(ffmpeg_tools, out, 0, tmp_path / "back.wav")
    y, _ = sf.read(str(back), dtype="float32", always_2d=True)
    assert soundtrack_metrics(y, SR)["peak_dbfs"] <= -0.9  # limiter kept peaks at or below -1 dB
