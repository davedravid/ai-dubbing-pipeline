"""Runs the real NLLB model on a few sentences. Run with:  pytest -m integration"""

import pytest

from dubbing.config import load_settings
from dubbing.languages import get_language, script_fraction
from dubbing.translator.nllb import NllbTranslator

pytestmark = pytest.mark.integration

SENTENCES = [
    "Good morning.",
    "I have been looking for a job for a long time.",
    "What would you say if a man walked in here without a shirt?",
]


@pytest.fixture(scope="module")
def translator():
    return NllbTranslator(load_settings().translation)


def test_english_to_hindi(translator):
    hi = get_language("hi")
    out = translator.translate(SENTENCES, "eng_Latn", hi.nllb)
    assert len(out) == len(SENTENCES)
    for text in out:
        assert text.strip()
        assert script_fraction(text, "Devanagari") > 0.9, text


def test_order_is_preserved_after_length_sorting(translator):
    # Input sorted longest-first; the translator sorts internally and must restore order.
    out = translator.translate(list(reversed(SENTENCES)), "eng_Latn", "hin_Deva")
    assert len(out[0]) > len(out[-1])
