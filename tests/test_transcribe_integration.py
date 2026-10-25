"""Real end-to-end check on a real English video.

Set DUBBING_TEST_VIDEO to the file's path, then run:  pytest -m integration
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing.config import load_settings
from dubbing.errors import OutputExistsError
from dubbing.pipeline.transcribe_job import run_transcription

pytestmark = pytest.mark.integration

VIDEO = os.environ.get("DUBBING_TEST_VIDEO")


@pytest.fixture(scope="module")
def transcript(tmp_path_factory, ffmpeg_tools):
    if not VIDEO or not Path(VIDEO).is_file():
        pytest.skip("Set DUBBING_TEST_VIDEO to an English video file to run this test")
    data_dir = tmp_path_factory.mktemp("data")
    settings = replace(load_settings(), data_dir=data_dir)
    out = run_transcription(Path(VIDEO), "itest", settings)
    return settings, out, json.loads(out.read_text(encoding="utf-8"))


def test_detects_english(transcript):
    _, out, data = transcript
    assert data["language"] == "en"
    assert out.name == "itest_en.json"


def test_segments_are_present_and_well_formed(transcript):
    _, _, data = transcript
    segs = data["segments"]
    assert data["segment_count"] == len(segs) > 0
    duration = data["source"]["duration_s"]
    previous_start = -1.0
    for seg in segs:
        assert seg["text"], f"segment {seg['id']} has empty text"
        assert 0 <= seg["start"] < seg["end"] <= duration + 1.0
        assert seg["start"] >= previous_start, "segments are out of order"
        previous_start = seg["start"]
    assert [s["id"] for s in segs] == list(range(len(segs)))


def test_metadata_recorded(transcript):
    _, _, data = transcript
    assert len(data["source"]["sha256"]) == 64
    assert data["asr"]["engine"] == "faster-whisper"
    assert set(data["timings"]) == {"probe_s", "extract_audio_s", "transcribe_s"}


def test_existing_transcript_is_not_overwritten(transcript):
    settings, out, _ = transcript
    before = out.read_bytes()
    with pytest.raises(OutputExistsError):
        run_transcription(Path(VIDEO), "itest", settings)
    assert out.read_bytes() == before
