"""Tests for Phase A2: the story bible (rules, combining with the AI, corrections)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing.config import load_settings
from dubbing.context import corrections as corr
from dubbing.context.llm import _check
from dubbing.context.merge import addressee_gender, merge_lines, pair_registers, suggest_names
from dubbing.context.rules import analyse, vocatives

NAMES = ["Stark", "Romanoff", "Banner", "Rogers"]


def _l(i, spk, text):
    return {"id": i, "speaker": spk, "text": text, "start": float(i)}


def test_vocatives_at_start_and_end():
    assert vocatives("Remind me again how you made your fortune, Stark?", NAMES) == ["Stark"]
    assert vocatives("Agent Romanoff, would you escort him?", NAMES + ["Agent"]) == ["Romanoff"]
    assert vocatives("I was in Calcutta.", NAMES) == []


def test_rules_back_and_forth_plural_and_name_votes():
    lines = [_l(0, "S1", "Loki is manipulating you."), _l(1, "S2", "And you've been doing what?"),
             _l(2, "S1", "Are you boys really that naive?"),
             _l(3, "S3", "Remind me how you made your fortune, Stark?"), _l(4, "S6", "Because I'm a genius.")]
    r = analyse(lines, NAMES)
    assert r["lines"][0]["addressee"] == ["S2"] and r["lines"][0]["confidence"] == "high"   # A-B-A
    assert r["lines"][2]["addressee"] == ["group"] and r["lines"][2]["plural"]
    assert r["name_votes"]["S6"]["Stark"] > r["name_votes"].get("S1", {}).get("Stark", 0)   # Stark answers next


def test_names_agree_or_are_marked():
    speakers = {"S1": {"gender": "female"}, "S6": {"gender": "male"}, "S2": {"gender": "male"}}
    names = suggest_names({"S6": {"Stark": 1.0}, "S2": {"Banner": 0.6}},
                          {"S6": {"name": "Tony Stark"}, "S2": {"name": "Rogers"}}, speakers)
    assert names["S6"]["suggested_name"] == "Tony Stark" and names["S6"]["confidence"] == "high"
    assert names["S2"]["sources_agree"] is False and names["S2"]["confidence"] == "low"
    assert names["S1"]["suggested_name"] is None


def test_merge_name_in_line_wins_and_disagreement_is_shown():
    speakers = {"S1": {"gender": "female"}, "S2": {"gender": "male"}, "S3": {"gender": "male"}}
    names = {"S1": {"suggested_name": "Romanoff"}, "S2": {"suggested_name": "Banner"}, "S3": {}}
    lines = [_l(0, "S3", "Agent Romanoff, would you escort him?"), _l(1, "S1", "Fine."), _l(2, "S2", "Why?")]
    rules = {0: {"addressee": ["S1"], "confidence": "low", "vocatives": ["Romanoff"]},
             1: {"addressee": ["S3"], "confidence": "high"}, 2: {"addressee": ["S1"], "confidence": "low"}}
    ai = {0: {"addressee": ["S2"], "confidence": "high"}, 1: {"addressee": ["S3"], "confidence": "medium"},
          2: {"addressee": ["S3"], "confidence": "high"}}
    out = merge_lines(lines, rules, ai, names, speakers)
    assert out[0]["addressee"] == ["S1"] and out[0]["reason"] == "name said in the line"
    assert out[1]["sources_agree"] is True and out[1]["confidence"] == "high"
    assert out[2]["addressee"] == ["S3"] and out[2]["sources_agree"] is False and out[2]["confidence"] == "medium"


def test_confident_rules_beat_the_ai():
    speakers = {"S1": {"gender": "female"}, "S2": {"gender": "male"}, "S3": {"gender": "male"}}
    lines = [_l(0, "S1", "Loki is manipulating you."), _l(1, "S2", "Says who?")]
    rules = {0: {"addressee": ["S2"], "confidence": "high", "evidence": "back-and-forth"}}
    ai = {0: {"addressee": ["S3"], "confidence": "high"}}
    out = merge_lines(lines, rules, ai, {}, speakers)
    assert out[0]["addressee"] == ["S2"] and out[0]["sources_agree"] is False


def test_ai_name_given_to_several_speakers_is_ignored():
    speakers = {"S4": {}, "S5": {}, "S6": {}}
    names = suggest_names({"S5": {"Rogers": 1.0}}, {"S4": {"name": "Fury"}, "S5": {"name": "Fury"},
                                                   "S6": {"name": "Fury"}}, speakers)
    assert names["S5"]["suggested_name"] == "Rogers" and names["S4"]["suggested_name"] is None


def test_only_real_hearing_fixes_are_kept():
    from dubbing.context.merge import usable_corrections

    originals = {1: "What are you?", 2: "We're a time bomb.", 3: "You gonna stop me?",
                 5: "You didn't come here because I bat my eyelash, is that you?"}
    suggestions = {1: {"suggested": "We're a time bomb."},                 # copied another line
                   3: {"suggested": "You gonna stop me?"},                 # unchanged
                   2: {"suggested": "A completely different sentence here."},  # rewritten
                   5: {"suggested": "You didn't come here because I bat my eyelashes at you?"}}
    assert list(usable_corrections(suggestions, originals)) == [5]


def test_listener_gender_and_pair_formality():
    speakers = {"S1": {"gender": "female"}, "S2": {"gender": "male"}, "S3": {"gender": "male"}}
    assert addressee_gender(["S1"], speakers, []) == "female"
    assert addressee_gender(["group"], speakers, ["S2", "S3"]) == "male_or_mixed_group"
    assert addressee_gender(["group"], speakers, ["S1"]) == "female_group"
    pairs = pair_registers([{"a": "S3", "b": "S2", "relationship": "boss and doctor", "register": "formal"}],
                           speakers, "familiar")
    assert pairs["S3>S2"]["register"] == "formal" and pairs["S2>S3"]["register"] == "formal"
    assert pairs["_default"]["register"] == "familiar"


def test_corrections_keep_your_decisions(tmp_path: Path):
    f = tmp_path / "context" / "job_corrections.yaml"
    corr.update(f, {5: {"suggested": "bat my eyelashes at you", "reason": "mis-heard"}}, {5: "bat my eyelash, is that you"})
    f.write_text(f.read_text(encoding="utf-8").replace("accept: null", "accept: yes"), encoding="utf-8")  # you accepted
    merged = corr.update(f, {5: {"suggested": "something else", "reason": "x"}, 9: {"suggested": "B", "reason": "y"}},
                         {5: "bat my eyelash, is that you", 9: "A"})
    assert merged[5]["accept"] is True and merged[5]["suggested"] == "bat my eyelashes at you"   # kept
    assert merged[9]["accept"] is None                                                       # new, waiting
    merged = corr.update(f, {}, {5: "bat my eyelash, is that you", 9: "A"})
    assert 5 in merged and 9 not in merged   # undecided and no longer suggested: dropped; decision kept


def test_ai_answer_check_finds_missing_parts():
    assert _check({"characters": {}, "lines": [{"id": 1}], "relationships": [], "scene": {}, "corrections": []}, {1}) == []
    assert "1 line ids missing" in _check({"characters": {}, "lines": [], "relationships": [], "scene": {},
                                           "corrections": []}, {1})


def test_story_phase_end_to_end_with_stand_in_ai(tmp_path: Path):
    from dubbing.pipeline.story_job import run_story

    settings = replace(load_settings(), data_dir=tmp_path / "data")
    lines = [dict(_l(0, "S1", "Loki is manipulating you."), end=1.0), dict(_l(1, "S2", "Says who?"), end=2.0)]
    script = {"job_id": "job_t", "source": {"path": str(tmp_path / "v.mp4"), "duration_s": 3},
              "speakers": {"S1": {"gender": "female"}, "S2": {"gender": "male"}}, "lines": lines, "clone_samples": {}}
    p = tmp_path / "data" / "scripts" / "job_t_script.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(script), encoding="utf-8")
    fake_ai = lambda ls, spk, t: {"characters": {"S1": {"name": "Romanoff", "evidence": "addressed by name"}},
                                  "lines": {0: {"addressee": ["S2"], "confidence": "high"}},
                                  "relationships": [], "scenes": [{"summary": "an argument", "tone": "tense"}],
                                  "terms": ["Loki"], "corrections": {}, "sections": []}
    out = run_story("job_t", settings, ai_reader=fake_ai)
    story = json.loads(out.read_text(encoding="utf-8"))
    assert story["lines"]["0"]["addressee"] == ["S2"] and story["lines"]["0"]["addressee_gender"] == "male"
    assert story["characters"]["S1"]["suggested_name"] == "Romanoff"
    assert (tmp_path / "data" / "context" / "job_t_story_review.html").is_file()
