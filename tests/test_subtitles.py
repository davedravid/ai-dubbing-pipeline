import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing.config import load_settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.pipeline.subtitles_job import run_subtitles
from dubbing.subtitles.vtt import MAX_LINE_CHARS, build_cues, format_timestamp, render_vtt


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00:00.000"),
    (12.3456, "00:00:12.346"),
    (61.5, "00:01:01.500"),
    (3725.004, "01:02:05.004"),
    (-0.2, "00:00:00.000"),
])
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_render_basic_vtt():
    vtt = render_vtt([{"start": 1.0, "end": 2.5, "text": " Hello there. "}])
    assert vtt == "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\nHello there.\n"


def test_special_characters_are_escaped():
    vtt = render_vtt([{"start": 0, "end": 1, "text": "Tom & Jerry <3"}])
    assert "Tom &amp; Jerry &lt;3" in vtt


def test_empty_segments_are_skipped():
    cues = build_cues([{"start": 0, "end": 1, "text": "  "}, {"start": 1, "end": 2, "text": "Hi"}])
    assert len(cues) == 1


def test_long_segment_is_wrapped_and_split_in_time_order():
    text = " ".join(["word"] * 60)  # ~300 characters: needs more than one cue
    cues = build_cues([{"start": 10.0, "end": 20.0, "text": text}])
    assert len(cues) > 1
    assert cues[0][0] == 10.0 and cues[-1][1] == 20.0
    for (s1, e1, lines), (s2, _, _) in zip(cues, cues[1:]):
        assert s1 < e1 == s2
    for _, _, lines in cues:
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= MAX_LINE_CHARS for line in lines)
    # No words lost when splitting.
    assert sum(" ".join(l).count("word") for _, _, l in cues) == 60


def _fake_transcript(data_dir: Path, job_id="job_x", lang="en"):
    path = data_dir / "transcripts" / f"{job_id}_{lang}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "language": lang,
        "segments": [{"start": 0.5, "end": 1.5, "text": "One."}, {"start": 2, "end": 3, "text": "Two."}],
    }), encoding="utf-8")


def test_run_subtitles_writes_file_and_refuses_overwrite(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _fake_transcript(tmp_path)
    out = run_subtitles("job_x", settings)
    assert out == tmp_path / "subtitles" / "job_x_en.vtt"
    assert out.read_text(encoding="utf-8").startswith("WEBVTT\n")
    assert out.read_text(encoding="utf-8").count(" --> ") == 2

    with pytest.raises(OutputExistsError):
        run_subtitles("job_x", settings)
    run_subtitles("job_x", settings, overwrite=True)


def test_subtitles_command_line(tmp_path: Path):
    """Runs the real `subtitles` command, using a config whose data_dir is a temp folder."""
    from dubbing.cli import main
    from dubbing.config import DEFAULT_CONFIG_PATH

    cfg = tmp_path / "config.yaml"
    cfg.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
                   .replace("data_dir: data", f"data_dir: '{tmp_path / 'data'}'"), encoding="utf-8")
    _fake_transcript(tmp_path / "data")
    assert main(["subtitles", "--job-id", "job_x", "--config", str(cfg)]) == 0
    assert (tmp_path / "data" / "subtitles" / "job_x_en.vtt").is_file()
    assert main(["subtitles", "--job-id", "../bad", "--config", str(cfg)]) == 1


def test_run_subtitles_without_transcript(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    with pytest.raises(PipelineError, match="no transcript"):
        run_subtitles("job_missing", settings)
