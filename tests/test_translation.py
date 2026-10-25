import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing.config import load_settings
from dubbing.errors import OutputExistsError, PipelineError
from dubbing.languages import get_language, script_fraction
from dubbing.pipeline.translate_job import run_translation
from dubbing.translator.checks import check_translation, timing_facts
from dubbing.translator.units import build_units

HI = get_language("hi")


# ---- language table -------------------------------------------------------

def test_language_codes_map_across_models():
    assert HI.nllb == "hin_Deva" and HI.xtts == "hi" and HI.mms == "hin"
    assert get_language("EN").nllb == "eng_Latn"
    assert get_language("te").xtts is None  # XTTS v2 cannot speak Telugu


def test_unsupported_language_has_helpful_error():
    with pytest.raises(PipelineError, match="not in the supported list"):
        get_language("xx")


def test_script_fraction():
    assert script_fraction("नमस्ते दुनिया", "Devanagari") == 1.0
    assert script_fraction("hello", "Devanagari") == 0.0
    assert script_fraction("123 !!", "Latin") == 0.0


# ---- per-segment checks ---------------------------------------------------

EN = get_language("en")


def _check(src, tgt, duration=2.0, available=2.0, src_cps=14.0):
    return check_translation(src, tgt, EN, HI, duration, available, src_cps)


def test_good_translation_has_no_flags():
    flags, est = _check("Good morning.", "सुप्रभात।", 1.5, 2.0)
    assert flags == []
    assert est["est_target_duration_s"] > 0


@pytest.mark.parametrize("target, flag", [
    ("", "empty_translation"),
    ("Good morning.", "wrong_script"),
    ("नमस्ते नमस्ते नमस्ते नमस्ते", "repetition"),
    ("नम", "unusual_length"),
])
def test_bad_translations_are_flagged(target, flag):
    flags, _ = _check("Good morning, everyone here.", target)
    assert flag in flags


def test_numbers_only_is_not_called_wrong_script():
    flags, _ = _check("12.", "12.")
    assert "numbers_only" in flags
    assert "wrong_script" not in flags and "unchanged" not in flags


def test_long_translation_in_short_slot_may_not_fit():
    long_hi = "यह एक बहुत लंबा वाक्य है जिसे बोलने में काफी समय लगेगा"
    flags, est = _check("It is a long sentence.", long_hi, 0.8, 0.9)
    assert "may_not_fit" in flags
    assert est["est_speedup_needed"] > 1.25


def test_fast_speaker_gets_more_room():
    hi = "आपको फिर से देखकर अच्छा लगा"
    _, normal = _check("Good to see you again.", hi, src_cps=14.0)
    _, fast = _check("Good to see you again.", hi, src_cps=28.0)
    assert fast["est_target_duration_s"] < normal["est_target_duration_s"]
    assert fast["speaker_pace"] == 1.8  # 28/14 = 2.0, clamped to the maximum


# ---- sentence units -------------------------------------------------------

def test_split_sentence_is_joined_into_one_unit():
    segs = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "I was sitting there trying"},
        {"id": 1, "start": 2.1, "end": 4.0, "text": "to come up with a story."},
        {"id": 2, "start": 4.2, "end": 5.0, "text": "Really?"},
    ]
    units = build_units(segs)
    assert [u["source_segment_ids"] for u in units] == [[0, 1], [2]]
    assert units[0]["text"] == "I was sitting there trying to come up with a story."
    assert (units[0]["start"], units[0]["end"]) == (0.0, 4.0)


def test_long_pause_splits_units_even_mid_sentence():
    segs = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "Well"},
        {"id": 1, "start": 3.0, "end": 4.0, "text": "maybe not."},
    ]
    assert [u["source_segment_ids"] for u in build_units(segs)] == [[0], [1]]


def test_timing_facts():
    segs = [
        {"start": 1.0, "end": 2.0, "text": "one two", "words": [{}, {}]},
        {"start": 3.5, "end": 4.0, "text": "three"},
    ]
    f = timing_facts(segs, media_duration=10.0)
    assert f[0]["pause_before_s"] == 1.0 and f[0]["pause_after_s"] == 1.5
    assert f[0]["available_s"] == 2.5 and f[0]["source_words_per_s"] == 2.0
    assert f[1]["available_s"] == 6.5  # until the end of the video


# ---- pipeline bookkeeping (stand-in translator, test-only) ----------------

class StandInTranslator:
    """Test double: returns fixed Hindi text so no model is needed."""

    def __init__(self, drop_one=False):
        self.drop_one = drop_one

    def translate(self, texts, src_nllb, tgt_nllb):
        assert (src_nllb, tgt_nllb) == ("eng_Latn", "hin_Deva")
        out = [f"अनुवाद {i}" for i in range(len(texts))]
        return out[:-1] if self.drop_one else out

    def info(self):
        return {"engine": "test-stand-in"}


def _write_transcript(data_dir: Path, segments):
    path = data_dir / "transcripts" / "job_t_en.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"language": "en", "source": {"duration_s": 20.0}, "segments": segments}),
                    encoding="utf-8")


SEGMENTS = [
    {"id": 0, "start": 0.5, "end": 2.0, "text": "Hello there."},
    {"id": 1, "start": 2.5, "end": 3.0, "text": ""},
    {"id": 2, "start": 4.5, "end": 6.0, "text": "How are you today?"},
]


def test_every_segment_is_kept_in_order(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _write_transcript(tmp_path, SEGMENTS)
    [out] = run_translation("job_t", settings, ["hi"], translator_factory=StandInTranslator)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert out.name == "job_t_hi.json" and data["language"] == "hi"
    assert [s["source_segment_ids"] for s in data["segments"]] == [[0], [1], [2]]
    assert data["segments"][1]["status"] == "skipped_empty_source"
    summary = data["summary"]
    assert (summary["unit_count"], summary["translated"], summary["skipped"]) == (3, 2, 1)
    assert data["segments"][0]["source_text"] == "Hello there."
    assert data["segments"][0]["text"].startswith("अनुवाद")
    assert {"duration_s", "available_s", "source_words_per_s", "est_speedup_needed"} <= set(data["segments"][0])


def test_missing_results_save_nothing(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _write_transcript(tmp_path, SEGMENTS)
    with pytest.raises(PipelineError, match="nothing was saved"):
        run_translation("job_t", settings, ["hi"], translator_factory=lambda: StandInTranslator(drop_one=True))
    assert not (tmp_path / "translations" / "job_t_hi.json").exists()


def test_existing_translation_not_overwritten(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _write_transcript(tmp_path, SEGMENTS)
    run_translation("job_t", settings, ["hi"], translator_factory=StandInTranslator)
    with pytest.raises(OutputExistsError):
        run_translation("job_t", settings, ["hi"], translator_factory=StandInTranslator)


def test_same_source_and_target_rejected(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _write_transcript(tmp_path, SEGMENTS)
    with pytest.raises(PipelineError, match="already in English"):
        run_translation("job_t", settings, ["en"], translator_factory=StandInTranslator)


def test_translation_config_loads():
    tr = load_settings().translation
    assert tr.model == "facebook/nllb-200-distilled-600M" and tr.batch_size >= 1
