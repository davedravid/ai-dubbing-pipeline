"""Tests for the fixes made after the first viewing of the dubbed video."""

from pathlib import Path

import numpy as np
import pytest

from dubbing.audio.nonverbal import candidate_regions, classify_regions, is_nonverbal_text, restore_mask
from dubbing.diarization.speakers import label_segments
from dubbing.qa.text_metrics import chrf
from dubbing.translator.units import build_units

SR = 16000


def _tone(seconds, amp=0.3, freq=200, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---- speakers ----------------------------------------------------------------

TURNS = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_07"},
         {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_02"}]


def test_segments_get_the_speaker_who_talks_most():
    segs = [{"id": 0, "start": 0.2, "end": 1.8, "text": "a", "words": [{"start": 0.2, "end": 1.8}]},
            {"id": 1, "start": 1.5, "end": 4.5, "text": "b", "words": [{"start": 1.5, "end": 4.5}]}]
    out = label_segments(segs, TURNS)
    assert [s["speaker"] for s in out] == ["S1", "S2"]   # renamed in order of appearance
    assert out[0]["speaker_confidence"] == 1.0 and not out[0]["speaker_uncertain"]
    assert out[1]["speaker_confidence"] == pytest.approx(2.5 / 3.0, abs=0.01)


def test_mixed_segment_is_uncertain_and_silent_one_has_no_speaker():
    segs = [{"id": 0, "start": 1.0, "end": 3.0, "text": "a"},   # half S1, half S2
            {"id": 1, "start": 9.0, "end": 9.5, "text": "b"}]   # nobody speaking
    out = label_segments(segs, TURNS)
    assert out[0]["speaker_uncertain"] and out[0]["speaker_confidence"] == 0.5
    assert out[1]["speaker"] is None and out[1]["speaker_uncertain"]


def test_sentences_never_join_two_speakers():
    segs = [{"id": 0, "start": 0.0, "end": 1.0, "text": "I was going", "speaker": "S1", "speaker_uncertain": False},
            {"id": 1, "start": 1.1, "end": 2.0, "text": "to say", "speaker": "S2", "speaker_uncertain": False}]
    units = build_units(segs)
    assert [u["source_segment_ids"] for u in units] == [[0], [1]]
    assert [u["speaker"] for u in units] == ["S1", "S2"]


def test_results_with_numpy_values_can_be_saved(tmp_path: Path):
    import json

    from dubbing.pipeline.transcribe_job import write_json

    write_json(tmp_path / "x.json", {"flag": np.bool_(True), "n": np.float64(1.5), "a": np.arange(3)})
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == {"flag": True, "n": 1.5, "a": [0, 1, 2]}


# ---- recovering skipped speech ------------------------------------------------

class _StandInWhisper:
    """Test double for faster-whisper: 'hears' a fixed line in any clip it is given."""

    class _Word:
        def __init__(self, word, start, end, probability=0.9):
            self.word, self.start, self.end, self.probability = word, start, end, probability

    class _Seg:
        def __init__(self, words):
            self.words, self.avg_logprob, self.no_speech_prob = words, -0.2, 0.1

    def __init__(self, text):
        self.text, self.calls = text, 0

    def transcribe(self, clip, **kwargs):
        self.calls += 1
        if not self.text:
            return iter([]), None
        words = [self._Word(w, 0.35 + i * 0.3, 0.6 + i * 0.3) for i, w in enumerate(self.text.split())]
        return iter([self._Seg(words)]), None


def test_skipped_speech_is_recovered_with_shifted_timings():
    from dubbing.asr.gapfill import fill_gaps

    y = np.concatenate([_tone(1.0), _tone(1.0), np.zeros(SR, np.float32), _tone(1.0)])  # voice 0-2 s and 3-4 s
    segs = [{"id": 0, "start": 3.0, "end": 4.0, "text": "later",
             "words": [{"word": "later", "start": 3.0, "end": 4.0}]}]
    merged, report = fill_gaps(_StandInWhisper("Good morning"), y, None, segs, "en")
    assert [s["text"] for s in merged] == ["Good morning", "later"]
    assert merged[0]["recovered"] and merged[0]["start"] == pytest.approx(0.35, abs=0.05)
    assert [s["id"] for s in merged] == [0, 1]
    assert report[0]["source"] == "vocals"


def test_laugh_in_a_gap_is_not_added_as_a_line():
    from dubbing.asr.gapfill import fill_gaps

    y = _tone(2.0)
    merged, report = fill_gaps(_StandInWhisper("Ha ha"), y, None, [], "en")
    assert merged == [] and report[0]["recovered"] == []


# ---- splitting long transcript segments ----------------------------------------

def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_long_segment_split_at_sentence_ends_and_pauses():
    from dubbing.asr.resegment import split_segments

    seg = {"id": 0, "start": 0.0, "end": 6.0, "text": "...", "avg_logprob": -0.2, "words": [
        _w("Hello.", 0.0, 0.5), _w("How", 0.6, 0.8), _w("are", 0.8, 1.0), _w("you", 1.0, 1.2),
        _w("today", 2.5, 3.0), _w("sir?", 3.0, 3.4)]}
    out = split_segments([seg], max_s=10, pause_s=0.6)
    assert [s["text"] for s in out] == ["Hello.", "How are you", "today sir?"]
    assert [s["id"] for s in out] == [0, 1, 2]
    assert out[1]["start"] == 0.6 and out[1]["end"] == 1.2 and out[1]["avg_logprob"] == -0.2
    assert sum(len(s["words"]) for s in out) == 6  # no word lost


def test_over_long_sentence_is_cut_at_a_comma():
    from dubbing.asr.resegment import split_segments

    words = [_w(f"w{i}" + ("," if i == 7 else ""), i * 1.0, i * 1.0 + 0.9) for i in range(14)]
    out = split_segments([{"id": 0, "start": 0, "end": 14, "text": "", "words": words}], max_s=10, pause_s=0.6)
    assert len(out) == 2 and out[0]["text"].endswith("w7,")
    assert all(s["end"] - s["start"] <= 10 for s in out)


# ---- meaning check of shorter wordings ----------------------------------------

def test_chrf_scores_similar_sentences_higher():
    src = "I have been looking for a job for a long time."
    assert chrf(src, src) == 100.0
    close = chrf(src, "I've been looking for work for a long time.")
    far = chrf(src, "The weather is nice today.")
    assert close > 60 > far


# ---- laughs and other non-word sounds -------------------------------------------

@pytest.mark.parametrize("text, nonverbal", [
    ("", True), ("Ha ha ha!", True), ("(laughs)", True), ("Hmm.", True), ("[Laughter]", True),
    ("Okay, sure.", False), ("Thank you", False)])
def test_nonverbal_text(text, nonverbal):
    assert is_nonverbal_text(text) is nonverbal


def test_regions_outside_words_and_dub_are_found():
    y = np.concatenate([_tone(1.0), np.zeros(SR // 2, np.float32), _tone(1.0), _tone(1.0)])
    # words at 0-1 s; dub plays at 2.5-3.5 s; so only 1.5-2.5 s remains (a "laugh")
    regions = candidate_regions(y, SR, word_spans=[(0.0, 1.0)], busy_spans=[(2.5, 3.5)])
    assert len(regions) == 1
    a, b = regions[0]
    assert 1.5 <= a <= 1.65 and 2.45 <= b <= 2.5


def test_words_heard_are_reported_not_kept():
    y = _tone(3.0)
    replies = iter(["Ha ha", "Wait a second"])
    keep, speech = classify_regions([(0.0, 1.0), (1.5, 2.5)], y, SR, lambda clip, sr: next(replies))
    assert [k["start"] for k in keep] == [0.0]
    assert [s["heard"] for s in speech] == ["Wait a second"]


def test_restore_mask_fades_in_and_out():
    m = restore_mask([{"start": 1.0, "end": 2.0}], 3 * SR, SR)
    assert m[int(1.5 * SR)] == pytest.approx(1.0)
    assert m[int(0.5 * SR)] == 0.0 and m[int(2.5 * SR)] == 0.0
    assert 0 < m[int(1.0 * SR)] < 1  # gentle edge, no click


# ---- timing: start when the actor starts ----------------------------------------

def test_voice_onset_finds_real_start():
    from dubbing.pipeline.mix_job import voice_onset

    y = np.concatenate([np.zeros(int(0.4 * SR), np.float32), _tone(1.0)])
    # transcript says 0.1 s; the voice really starts at 0.4 s
    assert voice_onset(y, SR, start=0.1, end=1.4) == pytest.approx(0.4, abs=0.02)


def test_voice_onset_never_much_earlier_than_transcript():
    from dubbing.pipeline.mix_job import voice_onset

    y = _tone(2.0)  # voice everywhere
    assert voice_onset(y, SR, start=1.0, end=1.8) == pytest.approx(0.9, abs=0.02)


# ---- choosing between wordings ------------------------------------------------

def test_clear_and_fitting_take_wins():
    from dubbing.pipeline.speech_job import _better

    long_clear = {"passed": True, "fits": False, "y": np.zeros(300)}
    short_clear = {"passed": True, "fits": True, "y": np.zeros(200)}
    short_unclear = {"passed": False, "fits": True, "y": np.zeros(100)}
    assert _better(short_clear, long_clear)
    assert _better(long_clear, short_unclear)  # clarity beats fitting


# ---- per-speaker voice samples -------------------------------------------------

def test_speaker_references_use_only_confident_long_lines(tmp_path: Path):
    from dubbing.tts.references import build_speaker_references

    sr = 24000
    y = _tone(40.0, sr=sr)
    units = [
        {"id": 0, "start": 0.0, "end": 6.0, "speaker": "S1", "speaker_uncertain": False},
        {"id": 1, "start": 7.0, "end": 8.0, "speaker": "S1", "speaker_uncertain": False},   # too short
        {"id": 2, "start": 9.0, "end": 14.0, "speaker": "S1", "speaker_uncertain": True},   # uncertain
        {"id": 3, "start": 15.0, "end": 20.0, "speaker": "S2", "speaker_uncertain": False},
    ]
    refs = build_speaker_references(units, y, sr, tmp_path)
    assert [c["unit_id"] for c in refs["S1"]["clips"]] == [0]
    assert [c["unit_id"] for c in refs["S2"]["clips"]] == [3]
    assert all(Path(c["path"]).is_file() for r in refs.values() for c in r["clips"])
