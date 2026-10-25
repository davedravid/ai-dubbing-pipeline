"""Prepare text for speech: write numbers as words.

XTTS converts digits to words with the num2words library, which has no Hindi, so a
Hindi line containing "12" crashes it. Numbers are spelled out here first.
"""

from __future__ import annotations

import re

# Hindi has an irregular word for every number from 0 to 99.
_HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_HI_0_99) == 100

# Indian place values: crore (10^7), lakh (10^5), thousand, hundred.
_HI_SCALES = ((10_000_000, "करोड़"), (100_000, "लाख"), (1000, "हज़ार"), (100, "सौ"))

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


def hindi_number_words(n: int) -> str:
    if n < 0:
        return "माइनस " + hindi_number_words(-n)
    if n < 100:
        return _HI_0_99[n]
    words = []
    for value, name in _HI_SCALES:
        if n >= value:
            count, n = divmod(n, value)
            words += [hindi_number_words(count), name]
    if n:
        words.append(_HI_0_99[n])
    return " ".join(words)


def _hindi_number_token(token: str) -> str:
    whole, _, frac = token.replace(",", "").partition(".")
    text = hindi_number_words(int(whole))
    if frac:
        text += " दशमलव " + " ".join(_HI_0_99[int(d)] for d in frac)
    return text


# Hindi abbreviations the translator writes with a full stop ("श्री." = "Mr.").
_HI_ABBREVIATIONS = re.compile(r"(श्री|श्रीमती|डॉ|सु)\.")
# A Latin full stop at a sentence end; Hindi uses the danda "।" instead.
_LATIN_FULL_STOP = re.compile(r"\.(?=\s|$)")


def normalize_for_speech(text: str, lang: str) -> str:
    """Return text ready for TTS in `lang`. Only Hindi needs help today."""
    if lang != "hi":
        return text
    text = text.translate(_DEVANAGARI_DIGITS)
    text = _NUMBER.sub(lambda m: _hindi_number_token(m.group()), text)
    text = _HI_ABBREVIATIONS.sub(r"\1", text)
    return _LATIN_FULL_STOP.sub("।", text)
