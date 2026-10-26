"""Tests for the source dubbing script (Phase A)."""

from pathlib import Path

import numpy as np
import pytest

from dubbing.script.clone_refs import choose_samples
from dubbing.script.delivery import add_styles
from dubbing.script.events import kind_of_sound, overlaps_from_turns
from dubbing.script.glossary import apply_glossary, clean_text, fix_acronyms, load_glossary
from dubbing.script.speakers import (apply_merges, build_profiles, gender_from_pitch, load_overrides,
                                     write_overrides_template)
from dubbing.script.timing import Envelope, breath_groups, pauses

SR = 16000


def _tone(seconds, amp=0.3, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _w(word, start, end, p=0.9):
    return {"word": word, "start": start, "end": end, "probability": p}


# ---- glossary and text clean-up ---------------------------------------------

def test_spaced_acronyms_are_joined():
    assert fix_acronyms("I'd like to know why S .H .I .E .L .D. is here") == "I'd like to know why S.H.I.E.L.D. is here"
    assert fix_acronyms("This is M .O., isn't it?") == "This is M.O., isn't it?"
    assert fix_acronyms("I was in Calcutta.") == "I was in Calcutta."


def test_glossary_spelling_is_applied():
    terms = ["Tesseract", "S.H.I.E.L.D.", "Dr. Banner"]
    assert apply_glossary("the tesseract and shield", terms) == "the Tesseract and shield"
    assert clean_text("why S .H .I .E .L .D.  uses the TESSERACT", terms) == "why S.H.I.E.L.D. uses the Tesseract"
    assert apply_glossary("dr. banner", terms) == "Dr. Banner"
    assert apply_glossary("s.h.i.e.l.d.", terms) == "S.H.I.E.L.D."


def test_glossary_file_ignores_notes_and_duplicates(tmp_path: Path):
    f = tmp_path / "g.txt"
    f.write_text("# notes\nLoki\n\nLoki\nAsgard  # realm\n", encoding="utf-8")
    assert load_glossary(f, extra=["Thor", "Loki"]) == ["Loki", "Asgard", "Thor"]


# ---- timing -------------------------------------------------------------------

def test_line_edges_snap_to_the_real_voice():
    y = np.concatenate([np.zeros(int(1.0 * SR), np.float32), _tone(1.0), np.zeros(int(1.0 * SR), np.float32)])
    env = Envelope(y, SR)
    # Whisper says 0.85-2.2; the voice is really 1.0-2.0
    start, end = env.refine(0.85, 2.2)
    assert start == pytest.approx(1.0, abs=0.02) and end == pytest.approx(2.0, abs=0.02)


def test_start_does_not_jump_into_the_previous_persons_voice():
    # Voice runs on without a break from 0 to 3 s (person A then person B); Whisper says B starts at 1.5 s.
    env = Envelope(_tone(3.0), SR)
    start, end = env.refine(1.5, 2.9, not_before=1.4)
    assert start == 1.5  # kept: there is no silence before it, so it is not a real onset


def test_edges_never_cross_neighbouring_words():
    y = np.concatenate([np.zeros(SR, np.float32), _tone(1.0), np.zeros(SR, np.float32)])
    env = Envelope(y, SR)
    start, _ = env.refine(1.1, 1.9, not_before=1.05)  # previous line's last word ends at 1.05
    assert start >= 1.05


def test_pauses_and_breath_groups():
    words = [_w("I", 0.0, 0.2), _w("was", 0.25, 0.5), _w("there,", 0.5, 0.9), _w("then", 1.4, 1.7), _w("left.", 1.7, 2.0)]
    assert pauses(words) == [{"start": 0.9, "end": 1.4, "length_s": 0.5}]
    assert [g["text"] for g in breath_groups(words)] == ["I was there,", "then left."]


# ---- speakers ------------------------------------------------------------------

@pytest.mark.parametrize("hz, gender", [(110, "male"), (200, "female"), (158, "unknown"), (None, "unknown")])
def test_gender_from_pitch(hz, gender):
    assert gender_from_pitch(hz)[0] == gender


def _line(i, spk, start, end, pitch=None, loud=-30.0, uncertain=False):
    return {"id": i, "speaker": spk, "start": start, "end": end, "speaker_uncertain": uncertain,
            "words": [], "text": "x", "delivery": {"pitch_median_hz": pitch, "loudness_db": loud}}


def test_profiles_use_pitch_and_your_corrections_win():
    lines = [_line(0, "S1", 0, 2, 120), _line(1, "S1", 2, 3, 125), _line(2, "S2", 3, 5, 200),
             _line(3, "S3", 5, 6, 158)]
    profiles = build_profiles(lines, {"S3": {"name": "Rogers", "gender": "male"}})
    assert profiles["S1"]["gender"] == "male" and profiles["S2"]["gender"] == "female"
    assert profiles["S3"] == {**profiles["S3"], "name": "Rogers", "gender": "male", "gender_source": "your correction"}
    assert profiles["S1"]["speech_s"] == 3.0


def test_unknown_gender_is_flagged_for_checking():
    profiles = build_profiles([_line(0, "S1", 0, 2, 158)], {})
    assert profiles["S1"]["flags"] == ["check_gender"]


def test_merge_two_labels_into_one_person(tmp_path: Path):
    f = tmp_path / "s.yaml"
    f.write_text("speakers:\n  S4:\n    same_as: S2\n  S2:\n    name: Fury\n", encoding="utf-8")
    overrides = load_overrides(f)
    lines = apply_merges([_line(0, "S4", 0, 1), _line(1, "S2", 1, 2)], overrides)
    assert [l["speaker"] for l in lines] == ["S2", "S2"] and lines[0]["speaker_detected"] == "S4"


def test_corrections_template_is_created_once(tmp_path: Path):
    f = tmp_path / "speakers" / "job.yaml"
    profiles = build_profiles([_line(0, "S1", 0, 2, 120)], {})
    assert write_overrides_template(f, profiles)
    f.write_text("speakers:\n  S1:\n    name: Banner\n", encoding="utf-8")   # you edited it
    assert not write_overrides_template(f, profiles)                    # never overwritten
    assert load_overrides(f)["S1"]["name"] == "Banner"


# ---- events ---------------------------------------------------------------------

def test_overlaps_between_two_speakers():
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 1.5, "end": 3.0, "speaker": "B"},
             {"start": 3.0, "end": 4.0, "speaker": "A"}, {"start": 3.95, "end": 4.05, "speaker": "B"}]
    assert overlaps_from_turns(turns) == [{"type": "overlapping_speech", "start": 1.5, "end": 2.0}]


def test_tiny_turns_at_speaker_changes_are_not_overlaps():
    turns = [{"start": 0.2, "end": 5.3, "speaker": "A"}, {"start": 2.8, "end": 3.1, "speaker": "B"}]
    assert overlaps_from_turns(turns) == []


def test_tiny_speaker_groups_merge_into_matching_voice():
    from dubbing.diarization.speakers import _merge_small_speakers

    sr = 16000
    low = lambda s: (0.3 * np.sin(2 * np.pi * 110 * np.arange(int(sr * s)) / sr)).astype(np.float32)
    high = lambda s: (0.3 * np.sin(2 * np.pi * 210 * np.arange(int(sr * s)) / sr)).astype(np.float32)
    y = np.concatenate([low(5.0), high(5.0), high(1.0), low(1.0)])  # man 0-5, woman 5-10, tiny groups after
    lines = [{"id": 0, "start": 0, "end": 5, "speaker": "S1", "speaker_uncertain": False},
             {"id": 1, "start": 5, "end": 10, "speaker": "S2", "speaker_uncertain": False},
             {"id": 2, "start": 10, "end": 11, "speaker": "S3", "speaker_uncertain": False},
             {"id": 3, "start": 11, "end": 12, "speaker": "S4", "speaker_uncertain": False}]
    # Fingerprints: the tiny high line looks a bit more like the MAN's group, but pitch must win.
    e = lambda v: np.array(v) / np.linalg.norm(v)
    emb = {0: e([1, 0]), 1: e([0, 1]), 2: e([0.8, 0.6]), 3: e([0.9, 0.1])}
    centroids = {0: e([1, 0]), 1: e([0, 1]), 2: emb[2], 3: emb[3]}
    names = {0: "S1", 1: "S2", 2: "S3", 3: "S4"}
    out = _merge_small_speakers(lines, y, sr, emb, centroids, names)
    assert [l["speaker"] for l in out] == ["S1", "S2", "S2", "S1"]
    assert out[2]["speaker_uncertain"] and out[2]["speaker_reassigned_from"] == "S3"


@pytest.mark.parametrize("heard, kind", [("Ha ha ha", "laugh"), ("(sighs)", "sigh"), ("Hmm.", "hum"), ("", "sound")])
def test_kind_of_sound(heard, kind):
    assert kind_of_sound(heard) == kind


# ---- delivery style ---------------------------------------------------------------

def test_style_is_relative_to_the_same_speaker():
    lines = [_line(0, "S1", 0, 1, loud=-30), _line(1, "S1", 1, 2, loud=-30), _line(2, "S1", 2, 3, loud=-22),
             _line(3, "S1", 3, 4, loud=-38)]
    add_styles(lines)
    assert [l["delivery"]["style"] for l in lines] == ["normal", "normal", "raised", "soft"]


# ---- clone samples -----------------------------------------------------------------

def test_clone_samples_skip_overlaps_sounds_and_music():
    sr = 16000
    vocals = _tone(40.0, sr=sr)
    background = np.zeros_like(vocals)
    background[int(30 * sr):int(34 * sr)] = 0.3   # loud music under the line at 30-34 s
    lines = [_line(0, "S1", 0, 6), _line(1, "S1", 10, 14), _line(2, "S1", 20, 24), _line(3, "S1", 30, 34),
             _line(4, "S1", 36, 37), _line(5, "S1", 37.5, 39.5, uncertain=True)]
    events = [{"type": "overlapping_speech", "start": 11.0, "end": 12.0}, {"type": "laugh", "start": 21, "end": 22}]
    out = choose_samples(lines, events, vocals, background, sr)["S1"]
    assert [s["line_id"] for s in out["samples"]] == [0]
    assert out["rejected"] == {"speaker_uncertain": 1, "too_short": 1, "overlapping_speech": 1,
                               "sound_inside": 1, "music_under_voice": 1}
    assert out["grade"] == "weak" and out["clean_seconds"] == 6.0
