from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dubbing.analysis.prosody import analyze_clip, compare
from dubbing.audio.clips import place_on_timeline, read_mono, write_wav
from dubbing.config import DEFAULT_CONFIG_PATH, load_settings
from dubbing.errors import ConfigError
from dubbing.qa.text_metrics import char_error_rate
from dubbing.tts.references import plan_references, reference_quality
from dubbing.tts.text_norm import hindi_number_words, normalize_for_speech

SR = 24000


# ---- Hindi numbers --------------------------------------------------------

@pytest.mark.parametrize("n, words", [
    (0, "शून्य"), (12, "बारह"), (45, "पैंतालीस"), (99, "निन्यानवे"),
    (100, "एक सौ"), (105, "एक सौ पाँच"), (1995, "एक हज़ार नौ सौ पचानवे"),
    (250000, "दो लाख पचास हज़ार"), (30000000, "तीन करोड़"),
])
def test_hindi_number_words(n, words):
    assert hindi_number_words(n) == words


def test_normalize_hindi_text():
    assert normalize_for_speech("12.", "hi") == "बारह।"
    assert normalize_for_speech("कमरा ३ में 1,200 लोग", "hi") == "कमरा तीन में एक हज़ार दो सौ लोग"
    assert normalize_for_speech("2.5 घंटे", "hi") == "दो दशमलव पाँच घंटे"
    assert normalize_for_speech("धन्यवाद, श्री. ट्विस्टल.", "hi") == "धन्यवाद, श्री ट्विस्टल।"
    assert normalize_for_speech("Room 12", "en") == "Room 12"  # other languages untouched


# ---- splitting long text for XTTS ----------------------------------------

def test_split_text_keeps_short_text_whole():
    from dubbing.tts.xtts import split_text

    assert split_text("नमस्ते।", 250) == ["नमस्ते।"]
    assert split_text("   ", 250) == []


def test_split_text_prefers_sentence_then_comma_breaks():
    from dubbing.tts.xtts import split_text

    text = "पहला वाक्य यहाँ है। दूसरा वाक्य, जिसमें अल्पविराम है, थोड़ा लंबा है।"
    pieces = split_text(text, 30)
    assert all(len(p) <= 30 for p in pieces)
    assert pieces[0] == "पहला वाक्य यहाँ है।"
    assert " ".join(pieces).split() == text.split()  # no words lost


# ---- back-check text comparison ------------------------------------------

def test_char_error_rate():
    assert char_error_rate("नमस्ते दुनिया।", "नमस्ते, दुनिया") == 0.0  # punctuation/spaces ignored
    assert char_error_rate("गाँव", "गांव") == 0.0                    # chandrabindu vs anusvara
    assert 0 < char_error_rate("नमस्ते", "नमस") < 1
    assert char_error_rate("नमस्ते", "") == 1.0


# ---- reference choice -----------------------------------------------------

def _units(spans):
    return [{"id": i, "start": s, "end": e, "text": "x"} for i, (s, e) in enumerate(spans)]


def test_long_lines_use_themselves_short_lines_borrow():
    units = _units([(0, 5), (5.5, 6.0), (30, 31)])
    plan = plan_references(units, min_ref_s=3.0, max_borrow_s=10.0)
    assert (plan[0].ref_unit_id, plan[0].borrowed) == (0, False)
    assert (plan[1].ref_unit_id, plan[1].borrowed) == (0, True)
    assert plan[2].ref_unit_id is None  # nothing long enough within 10 s: built-in voice


# ---- delivery measurements ------------------------------------------------

def _tone(freq, seconds=1.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pitch_of_known_tone():
    m = analyze_clip(_tone(220.0), SR)
    assert m["pitch_median_hz"] == pytest.approx(220.0, rel=0.03)
    assert m["pitch_range_st"] < 0.5  # a steady tone is "flat"


def test_compare_reports_pitch_and_loudness_difference():
    low, high = analyze_clip(_tone(220.0, amp=0.1), SR), analyze_clip(_tone(440.0, amp=0.3), SR)
    diff = compare(low, high)
    assert diff["pitch_shift_st"] == pytest.approx(12.0, abs=0.5)   # one octave up
    assert diff["loudness_diff_db"] == pytest.approx(9.5, abs=1.0)  # 3x amplitude


def test_reference_quality_flags_silence_and_clipping():
    silent = np.zeros(SR * 3, dtype=np.float32)
    silent[:SR // 4] = _tone(200, 0.25)
    assert "mostly_silence" in reference_quality(silent, SR)["warnings"]
    loud = np.clip(_tone(200, 3, amp=2.0), -1, 1)
    assert "clipped" in reference_quality(loud, SR)["warnings"]


def test_quiet_reference_is_boosted_but_loud_one_is_left_alone():
    from dubbing.tts.references import normalize_peak

    quiet, gain = normalize_peak(_tone(200, amp=0.05))
    assert gain == 24.0  # would need +25 dB; capped at +24 dB
    assert np.max(np.abs(quiet)) == pytest.approx(0.05 * 10 ** (24 / 20), rel=0.01)
    loud = _tone(200, amp=0.95)
    same, gain = normalize_peak(loud)
    assert gain == 0.0 and np.array_equal(same, loud)


# ---- take checks ----------------------------------------------------------

def test_invented_speech_detection():
    from dubbing.languages import get_language
    from dubbing.pipeline.speech_job import is_invented_speech

    hi = get_language("hi")
    assert is_invented_speech(7.3, 1.0, "क्या यह पर्याप्त है?", hi)      # 20 chars can't need 7 s
    assert not is_invented_speech(2.0, 1.0, "क्या यह पर्याप्त है?", hi)
    assert not is_invented_speech(1.2, 1.0, "जे।", hi)                   # short floor of 1.5 s


def test_only_failed_lines_are_redone():
    from dubbing.pipeline.speech_job import _needs_redo

    assert _needs_redo(None)
    assert _needs_redo({"status": "generated", "flags": ["low_intelligibility"]})
    assert _needs_redo({"status": "generated", "flags": ["invented_speech"]})
    assert not _needs_redo({"status": "generated", "flags": ["overruns_slot", "retried"]})


# ---- audio helpers --------------------------------------------------------

def test_timeline_places_clips_and_reports_overlap():
    a, b = _tone(200, 1.0), _tone(300, 1.0)
    track, overlaps = place_on_timeline([(0.0, a), (0.5, b)], SR, total_s=3.0)
    assert len(track) == SR * 3 + 1
    assert overlaps == [(0.5, pytest.approx(1.0, abs=0.01))]
    _, none = place_on_timeline([(0.0, a), (1.5, b)], SR, total_s=3.0)
    assert none == []


def test_wav_round_trip_with_ai_label(tmp_path: Path):
    import soundfile as sf

    path = write_wav(tmp_path / "x.wav", _tone(200), SR, comment="AI-generated speech")
    y, sr = read_mono(path)
    assert sr == SR and len(y) == SR
    assert "AI-generated" in sf.info(str(path)).extra_info or sf.SoundFile(str(path)).comment == "AI-generated speech"


# ---- config ---------------------------------------------------------------

def test_tts_config_defaults():
    tts = load_settings().tts
    assert tts.voice_mode == "clone" and tts.accept_coqui_license is True
    assert 1.0 <= tts.max_speed <= 2.0


def test_bad_voice_mode_rejected(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("voice_mode: clone", "voice_mode: robot"),
                   encoding="utf-8")
    with pytest.raises(ConfigError, match="voice_mode"):
        load_settings(cfg)
