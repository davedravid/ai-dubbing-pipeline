"""Runs the real XTTS v2 model. Run with:  pytest -m integration tests/test_speech_integration.py

The cloning test needs job_001's extracted audio (data/temp/job_001_source24k.wav)."""

from pathlib import Path

import numpy as np
import pytest

from dubbing.audio.clips import cut, read_mono, write_wav
from dubbing.config import PROJECT_ROOT, load_settings
from dubbing.pipeline.speech_job import _BackChecker
from dubbing.qa.text_metrics import char_error_rate
from dubbing.tts.text_norm import normalize_for_speech
from dubbing.tts.xtts import XttsSynthesizer

pytestmark = pytest.mark.integration

TEXT = normalize_for_speech("आपको फिर से देखकर अच्छा लगा। कमरे में 12 लोग हैं।", "hi")
SOURCE = PROJECT_ROOT / "data" / "temp" / "job_001_source24k.wav"


@pytest.fixture(scope="module")
def synth():
    return XttsSynthesizer(load_settings())


@pytest.fixture(scope="module")
def checker():
    return _BackChecker(load_settings())


def _check_speech(y, sr, checker):
    assert sr == 24000
    duration = len(y) / sr
    assert 1.5 < duration < 12, duration
    assert np.max(np.abs(y)) > 0.05, "audio is (nearly) silent"
    cer = char_error_rate(TEXT, checker.heard(y, sr, "hi"))
    assert cer < 0.35, f"back-check failed, CER {cer}"


def test_builtin_voice_speaks_hindi_with_numbers(synth, checker):
    y = synth.speak(TEXT, "hi", synth.builtin_voice(load_settings().tts.standard_speaker))
    _check_speech(y, synth.sample_rate, checker)


def test_cloned_voice_speaks_hindi(synth, checker, tmp_path: Path):
    if not SOURCE.is_file():
        pytest.skip("Run 'analyze' or 'speak' for job_001 first to create the source audio")
    src, sr = read_mono(SOURCE)
    ref = write_wav(tmp_path / "ref.wav", cut(src, sr, 22.5, 26.5), sr)  # a stretch of speech from the clip
    y = synth.speak(TEXT, "hi", synth.voice_from_audio([ref], "test reference"))
    _check_speech(y, synth.sample_rate, checker)
