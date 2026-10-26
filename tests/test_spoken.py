"""Tests for the spoken-Hindi layer (Phase B)."""

from pathlib import Path

import pytest

from dubbing.script.glossary import load_glossary, load_name_spellings
from dubbing.translator.spoken import formality_score, make_spoken


def test_formal_words_become_everyday_words():
    r = make_spoken("यदि तुम केवल सहायता करो, परंतु जल्दी।", "male")
    assert r.text == "अगर तुम सिर्फ़ मदद करो, लेकिन जल्दी।"
    assert "everyday word: यदि → अगर" in r.changes


def test_waiting_as_a_verb_is_not_changed():
    assert make_spoken("मैं प्रतीक्षा कर रहा हूँ।", "male").text == "मैं प्रतीक्षा कर रहा हूँ।"
    assert make_spoken("प्रतीक्षा करो, क्या?", "male").text == "रुको, क्या?"


@pytest.mark.parametrize("hindi, female", [
    ("मैं कलकत्ता में था।", "मैं कलकत्ता में थी।"),
    ("मैं नहीं जा रहा हूँ।", "मैं नहीं जा रही हूँ।"),
    ("मैं यह कर सकता हूँ।", "मैं यह कर सकती हूँ।"),
    ("मैं तुम्हें बताऊँगा।", "मैं तुम्हें बताऊँगी।"),
])
def test_female_speaker_verbs(hindi, female):
    assert make_spoken(hindi, "female").text == female
    assert make_spoken(hindi, "male").text == hindi  # unchanged for a man


def test_maine_clause_is_left_alone_and_only_her_clause_changes():
    # "मैंने" takes the object's gender; "वह गया" is about someone else.
    assert make_spoken("मैंने खाना खाया।", "female").text == "मैंने खाना खाया।"
    assert make_spoken("वह गया, मैं रहा।", "female").text == "वह गया, मैं रही।"


def test_aap_becomes_tum_with_matching_verb():
    r = make_spoken("आप क्या कर रहे हैं?", "male", register="tum")
    assert r.text == "तुम क्या कर रहे हो?"
    assert make_spoken("आपका काम।", "male", register="tum").text == "तुम्हारा काम।"
    assert make_spoken("आप क्या कर रहे हैं?", "male", register="aap").text == "आप क्या कर रहे हैं?"


def test_tum_fixes_found_in_the_first_comparison():
    # reflexive "अपने आप" (oneself) is not the respectful "you"
    assert make_spoken("आप अपने आप को दूर करना चाहते हैं?", "male").text == "तुम अपने आप को दूर करना चाहते हो?"
    # spaced form, and "तुम को" -> "तुम्हें"
    assert make_spoken("लोकी आप को हेरफेर कर रहा है।", "female").text == "लोकी तुम्हें हेरफेर कर रहा है।"
    # verb not at the end of the clause
    assert make_spoken("आप अच्छी तरह से जानते हैं कि क्यों।", "male").text == "तुम अच्छी तरह से जानते हो कि क्यों।"
    # polite command -> familiar command
    assert make_spoken("रुकिए, रुकिए।", "female").text == "रुको, रुको।"
    assert make_spoken("रुकिए।", "female", register="aap").text == "रुकिए।"


def test_fixes_found_in_the_second_comparison():
    # any -ता verb for a female speaker, but never words like पता
    assert make_spoken("मैं अपनी पलकें बजाता हूँ।", "female").text == "मैं अपनी पलकें बजाती हूँ।"
    assert make_spoken("मुझे पता है, मैं कलकत्ता में था।", "female").text == "मुझे पता है, मैं कलकत्ता में थी।"
    # formal phrases
    assert make_spoken("मुझे खेद है।", "male").text == "माफ़ करना।"
    assert make_spoken("यही कारण है कि मैं यहाँ हूँ।", "male").text == "इसीलिए मैं यहाँ हूँ।"
    # a question stays a question
    assert make_spoken("तुम सोचना चाहते हो डॉक्टर", "male", english="Doctor?").text == "तुम सोचना चाहते हो डॉक्टर?"
    assert make_spoken("रुको", "male", english="Wait!").text == "रुको!"
    # spelling variants
    assert make_spoken("एसगार्ड से", "male", variants={"एसगार्ड": "असगार्ड"}).text == "असगार्ड से"


def test_runaway_repeats_and_abbreviation_dots():
    r = make_spoken("नो। " * 13, "male", english="No, no, no.", max_repeat=3)
    assert r.text == "नहीं। नहीं। नहीं।"
    assert make_spoken("ओह, हाँ, हाँ, हाँ, हाँ यार", "male", max_repeat=1).text == "ओह, हाँ, यार"
    assert make_spoken("क्या तुम डॉ. बैनर को ले जाओगे?", "male").text == "क्या तुम डॉक्टर बैनर को ले जाओगे?"


@pytest.mark.parametrize("hindi, to_woman", [
    ("तुम क्या कर रहे हो?", "तुम क्या कर रही हो?"),
    ("क्या तुम उसे ले जाओगे?", "क्या तुम उसे ले जाओगी?"),
    ("तुम यह कर सकते हो।", "तुम यह कर सकती हो।"),
    ("तुम अकेले नहीं जा रहे हो।", "तुम अकेले नहीं जा रही हो।"),
])
def test_verbs_when_speaking_to_a_woman(hindi, to_woman):
    assert make_spoken(hindi, "male", listener_gender="female").text == to_woman
    assert make_spoken(hindi, "male", listener_gender="male").text == hindi
    assert make_spoken(hindi, "male", listener_gender="male_or_mixed_group").text == hindi


def test_group_and_other_subjects_are_left_alone():
    assert make_spoken("तुम लोग क्या कर रहे हो?", "male", listener_gender="female").text == "तुम लोग क्या कर रहे हो?"
    assert make_spoken("वे जा रहे थे।", "male", listener_gender="female").text == "वे जा रहे थे।"


def test_dotted_names_are_protected_before_translating():
    from dubbing.translator.spoken import protect_names

    assert protect_names("why S.H.I.E.L.D. is here", {"S.H.I.E.L.D.": "शील्ड"}) == "why SHIELD is here"
    assert protect_names("this is M.O.", {"M.O.": "तरीका"}) == "this is MO"


def test_names_in_english_letters_get_hindi_spelling():
    names = {"Tesseract": "टेसेरैक्ट", "Asgard": "असगार्ड", "S.H.I.E.L.D.": "शील्ड"}
    r = make_spoken("Tesseract Asgard का है. S.H.I.E.L.D. क्यों?", "male", names=names)
    assert r.text == "टेसेरैक्ट असगार्ड का है। शील्ड क्यों?"


def test_formality_score_counts_formal_words():
    assert formality_score("यदि केवल") == 2 and formality_score("अगर सिर्फ़") == 0


def test_glossary_with_hindi_spellings(tmp_path: Path):
    from dubbing.script.glossary import load_name_variants

    f = tmp_path / "g.txt"
    f.write_text("# notes\nTesseract = टेसेरैक्ट\nEarth\nS.H.I.E.L.D. = शील्ड | एस.एच.\n", encoding="utf-8")
    assert load_glossary(f) == ["Tesseract", "Earth", "S.H.I.E.L.D."]
    assert load_name_spellings(f) == {"Tesseract": "टेसेरैक्ट", "S.H.I.E.L.D.": "शील्ड", "SHIELD": "शील्ड"}
    assert load_name_variants(f) == {"एस.एच.": "शील्ड"}
