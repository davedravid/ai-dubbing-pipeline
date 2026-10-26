"""Tests for Phase B: translating the dubbing script."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing.config import load_settings
from dubbing.errors import OutputExistsError
from dubbing.pipeline.translate_script_job import run_translate_script
from dubbing.translator.choose import choose


def test_choose_prefers_meaning_fit_and_no_english():
    src = "Tesseract belongs on Asgard."
    options = ["टेसेरैक्ट असगार्ड पर belongs.", "टेसेरैक्ट असगार्ड का है.", "यदि केवल टेसेरैक्ट असगार्ड का है और बहुत कुछ."]
    backs = {options[0]: "Tesseract belongs on Asgard.", options[1]: "Tesseract belongs to Asgard.",
             options[2]: "If only the Tesseract belongs to Asgard and much more."}
    chosen, scored = choose(options, src, backs, slot_s=1.5, chars_per_second=13, pace=1.0)
    assert chosen["text"] == options[1]          # English word and formal/long options lose
    assert [o["text"] for o in scored][0] == options[1] and len(scored) == 3


def test_runaway_repetition_is_not_chosen():
    src = "No, no, no."
    options = ["नो। " * 13, "नहीं, नहीं, नहीं।"]
    backs = {options[0]: "No. no. no. no. no. no. no. no.", options[1]: "No, no, no."}
    chosen, _ = choose(options, src, backs, slot_s=0.8, chars_per_second=13, pace=1.0)
    assert chosen["text"] == options[1]


def test_abbreviation_does_not_end_the_sentence():
    from dubbing.translator.units import build_units

    segs = [{"id": 0, "start": 0, "end": 1.8, "text": "I'd like to know why S.H.I.E.L.D.", "speaker": "S2"},
            {"id": 1, "start": 1.9, "end": 4.6, "text": "is using the Tesseract.", "speaker": "S2"},
            {"id": 2, "start": 5.0, "end": 6.0, "text": "Would you escort Dr.", "speaker": "S3"},
            {"id": 3, "start": 6.1, "end": 7.0, "text": "Banner back to his room?", "speaker": "S3"},
            {"id": 4, "start": 7.5, "end": 8.0, "text": "Done.", "speaker": "S3"}]
    assert [u["source_segment_ids"] for u in build_units(segs)] == [[0, 1], [2, 3], [4]]


def test_translator_is_chosen_by_language():
    from dubbing.languages import get_language
    from dubbing.pipeline.translate_script_job import engine_for

    s = load_settings()
    assert engine_for(s, get_language("hi")) == "indictrans2"
    assert engine_for(s, get_language("es")) == "nllb"
    with pytest.raises(Exception, match="only translates into Indian languages"):
        engine_for(replace(s, translation=replace(s.translation, engine="indictrans2")), get_language("es"))


def test_spanish_skips_hindi_rules_and_keeps_latin_letters():
    from dubbing.translator.spoken import make_spoken

    r = make_spoken("Estaba en Calcuta", "female", "tum", {"Calcutta": "कलकत्ता"}, english="I was in Calcutta.",
                    lang="es")
    assert r.text == "Estaba en Calcuta."      # no Hindi names, no danda, end punctuation kept
    src = "Tesseract belongs on Asgard."
    opts = ["El Tesseract pertenece a Asgard.", "Tesseract es de Asgard y mucho, mucho, mucho, mucho más."]
    chosen, _ = choose(opts, src, {o: src for o in opts}, slot_s=2.0, chars_per_second=15, pace=1.0,
                       penalize_latin=False)
    assert chosen["text"] == opts[0] and chosen["latin_words"] == 0


class StandIn:
    """Test double: 'translates' by returning fixed options, and back-translates to the English."""

    def __init__(self, english_by_hindi=None):
        self.english_by_hindi = english_by_hindi or {}

    def translate_candidates(self, texts, src, tgt, n):
        return [[f"मैं वहाँ था ({i})", f"मैं वहाँ गया ({i})"] for i, _ in enumerate(texts)]

    def translate(self, texts, src, tgt):
        return [self.english_by_hindi.get(t, "I was there.") for t in texts]

    def info(self):
        return {"engine": "test-stand-in"}


def _script(data_dir: Path):
    lines = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "I was there.", "speaker": "S1", "speaker_uncertain": False,
         "words": [{"word": w, "start": 0, "end": 1} for w in "I was there.".split()], "flags": []},
        {"id": 1, "start": 2.0, "end": 3.0, "text": "I was there.", "speaker": "S2", "speaker_uncertain": False,
         "words": [{"word": w, "start": 2, "end": 3} for w in "I was there.".split()], "flags": ["unsure_words:1"]},
    ]
    script = {"job_id": "job_t", "language": "en", "source": {"duration_s": 5.0, "path": str(data_dir / "v.mp4")},
              "speakers": {"S1": {"gender": "female"}, "S2": {"gender": "male"}}, "lines": lines}
    path = data_dir / "scripts" / "job_t_script.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(script), encoding="utf-8")


def test_translate_script_uses_gender_and_keeps_every_line(tmp_path: Path):
    settings = replace(load_settings(), data_dir=tmp_path)
    _script(tmp_path)
    out = run_translate_script("job_t", settings, "hi", translator=StandIn(), referee=StandIn())
    data = json.loads(out.read_text(encoding="utf-8"))
    segs = data["segments"]
    assert [s["source_segment_ids"] for s in segs] == [[0], [1]]          # one sentence per speaker line
    assert segs[0]["text"].startswith("मैं वहाँ थी") and segs[0]["speaker_gender"] == "female"
    assert segs[1]["text"].startswith("मैं वहाँ था")                       # male speaker unchanged
    assert "english_unsure" in segs[1]["flags"]
    assert all(s["options"] and s["back_translation"] for s in segs)
    assert (tmp_path / "translations" / "job_t_hi_review.html").is_file()
    with pytest.raises(OutputExistsError):
        run_translate_script("job_t", settings, "hi", translator=StandIn(), referee=StandIn())
