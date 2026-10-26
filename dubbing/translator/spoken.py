"""Turn translated Hindi into everyday spoken Hindi for dubbing.

Translation models write formal, written Hindi and do not know who is speaking.
This layer, applied to any translator's output:
  1. swaps formal words for the ones people actually say (यदि -> अगर),
  2. makes verbs agree with a female speaker talking about herself (मैं गया -> मैं गई),
  3. uses तुम instead of आप between peers (optional, per scene),
  4. writes names in Hindi letters (Tesseract -> टेसेरैक्ट) so the voice can say them.
Every change is recorded, so a reviewer can see exactly what was altered and why.
The word lists are deliberately small and conservative; extend them as needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Formal / written word -> everyday spoken word. Longest first when applied.
COLLOQUIAL: dict[str, str] = {
    "प्रतीक्षा करो": "रुको", "प्रतीक्षा करें": "रुकिए", "प्रतीक्षा कीजिए": "रुकिए",
    "क्षमा करें": "माफ़ कीजिए", "क्षमा करना": "माफ़ करना", "क्षमा": "माफ़ी",
    "कृपया": "ज़रा", "धन्यवाद": "शुक्रिया", "प्रसन्नता": "खुशी",
    "यदि": "अगर", "परंतु": "लेकिन", "परन्तु": "लेकिन", "किंतु": "लेकिन", "किन्तु": "लेकिन",
    "अतः": "इसलिए", "केवल": "सिर्फ़", "वास्तव में": "सच में", "अत्यधिक": "बहुत ज़्यादा",
    "तत्काल": "फ़ौरन", "शीघ्र": "जल्दी", "उपयोग": "इस्तेमाल", "सहायता": "मदद",
    "प्रयास": "कोशिश", "आवश्यकता": "ज़रूरत", "आवश्यक": "ज़रूरी", "प्रश्न": "सवाल",
    "उत्तर": "जवाब", "निर्णय": "फ़ैसला", "स्मरण": "याद", "पर्याप्त": "काफ़ी",
    "नष्ट": "तबाह", "हेतु": "के लिए", "द्वारा": "से", "अथवा": "या", "तथा": "और",
    "सम्भव": "मुमकिन", "संभव": "मुमकिन", "असंभव": "नामुमकिन",
    "मुझे खेद है": "माफ़ करना", "खेद": "अफ़सोस", "यही कारण है कि": "इसीलिए", "यही कारण है": "इसीलिए",
    "याद दिलाएं": "याद दिलाओ", "याद दिलाएँ": "याद दिलाओ", "बताएं": "बताओ", "बताएँ": "बताओ",
}

# Words ending in -ता that are not verbs, so must never be made feminine.
NOT_VERBS = {"पता", "कलकत्ता", "रिश्ता", "दाता", "नेता", "देवता", "जनता", "क्षमता", "सत्ता", "रास्ता",
             "नाता", "पत्ता", "कुत्ता", "जूता", "छाता", "बत्ता", "सविता", "कविता", "माता", "पिता", "भ्राता"}

# Masculine -> feminine first-person forms (for "मैं ..." clauses of a female speaker).
FEMININE: dict[str, str] = {
    "रहा": "रही", "था": "थी", "गया": "गई", "सकता": "सकती", "चाहता": "चाहती", "करता": "करती",
    "जाता": "जाती", "होता": "होती", "आया": "आई", "लगा": "लगी", "पाया": "पाई", "चुका": "चुकी",
    "बैठा": "बैठी", "हुआ": "हुई", "देता": "देती", "लेता": "लेती", "जानता": "जानती",
    "सोचता": "सोचती", "समझता": "समझती", "कहता": "कहती", "रखता": "रखती", "मानता": "मानती",
    "करूँगा": "करूँगी", "करूंगा": "करूंगी", "जाऊँगा": "जाऊँगी", "जाऊंगा": "जाऊंगी",
    "रहूँगा": "रहूँगी", "रहूंगा": "रहूंगी", "दूँगा": "दूँगी", "दूंगा": "दूंगी",
    "लूँगा": "लूँगी", "लूंगा": "लूंगी", "बताऊँगा": "बताऊँगी", "बताऊंगा": "बताऊंगी",
    "आऊँगा": "आऊँगी", "आऊंगा": "आऊंगी", "सकूँगा": "सकूँगी", "सकूंगा": "सकूंगी",
}

# आप (respectful) -> तुम (familiar) forms, for scenes between peers. Written joined or with a space.
TUM: dict[str, str] = {
    "आपका": "तुम्हारा", "आपकी": "तुम्हारी", "आपके": "तुम्हारे", "आपको": "तुम्हें", "आपसे": "तुमसे",
    "आपने": "तुमने", "आपमें": "तुममें", "आपपर": "तुमपर",
    "आप का": "तुम्हारा", "आप की": "तुम्हारी", "आप के": "तुम्हारे", "आप को": "तुम्हें", "आप से": "तुमसे",
    "आप ने": "तुमने", "आप में": "तुममें", "आप पर": "तुमपर", "तुम को": "तुम्हें",
    "आप": "तुम",
}
# Polite commands -> familiar commands (only in तुम scenes): रुकिए -> रुको.
TUM_COMMANDS: dict[str, str] = {
    "रुकिए": "रुको", "रुकिये": "रुको", "सुनिए": "सुनो", "देखिए": "देखो", "चलिए": "चलो", "बैठिए": "बैठो",
    "बताइए": "बताओ", "बताइये": "बताओ", "जाइए": "जाओ", "आइए": "आओ", "कीजिए": "करो", "करिए": "करो",
    "दीजिए": "दो", "लीजिए": "लो", "सोचिए": "सोचो", "समझिए": "समझो", "रहिए": "रहो",
}
# "अपने आप" means "oneself", not the respectful "you": never touched.
_REFLEXIVE = "अपने आप"
_REFLEXIVE_HOLD = ""
CLAUSE_SPLIT = re.compile(r"([,;:।.?!]+)")


@dataclass
class SpokenResult:
    text: str
    changes: list[str] = field(default_factory=list)


def _replace_words(text: str, mapping: dict[str, str], label: str, changes: list[str]) -> str:
    for src in sorted(mapping, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\wऀ-ॣ०-ॿ]){re.escape(src)}(?![\wऀ-ॣ०-ॿ])")
        if pattern.search(text):
            text = pattern.sub(mapping[src], text)
            changes.append(f"{label}: {src} → {mapping[src]}")
    return text


def feminine_first_person(text: str, changes: list[str]) -> str:
    """In each clause where the speaker talks about herself ("मैं ..."), make verbs feminine.
    "मैंने" clauses are left alone: there the verb agrees with the object, not the speaker."""
    parts = CLAUSE_SPLIT.split(text)
    for i, clause in enumerate(parts):
        if re.search(r"(?<![ऀ-ॣ०-ॿ])मैं(?![ऀ-ॣ०-ॿ])", clause):
            clause = _replace_words(clause, FEMININE, "speaker is female", changes)

            def fem(m: re.Match) -> str:  # any other "-ता" verb: बजाता -> बजाती
                word = m.group(0)
                if word in NOT_VERBS:
                    return word
                changes.append(f"speaker is female: {word} → {word[:-1]}ी")
                return word[:-1] + "ी"
            clause = re.sub(r"(?<![ऀ-ॣ०-ॿ])[ऀ-ॣ०-ॿ]+ता(?![ऀ-ॣ०-ॿ])", fem, clause)
            parts[i] = clause
    return "".join(parts)


def to_tum(text: str, changes: list[str]) -> str:
    """आप -> तुम (with the matching verb, हैं -> हो) and polite commands -> familiar ones."""
    text = text.replace(_REFLEXIVE, _REFLEXIVE_HOLD)
    parts = CLAUSE_SPLIT.split(text)
    for i, clause in enumerate(parts):
        if re.search(r"(?<![ऀ-ॣ०-ॿ])(आप|तुम को)", clause):
            clause = _replace_words(clause, TUM, "तुम between peers", changes)
        subject = re.search(r"(?<![ऀ-ॣ०-ॿ])तुम(?![ऀ-ॣ०-ॿ])", clause)
        if subject:
            # the verb that goes with तुम: "तुम ... जानते हैं कि" -> "तुम ... जानते हो कि"
            rest = clause[subject.end():]
            fixed = re.sub(r"(?<![ऀ-ॣ०-ॿ])हैं(?![ऀ-ॣ०-ॿ])", "हो", rest, count=1)
            if fixed != rest:
                changes.append("तुम between peers: हैं → हो")
            clause = clause[:subject.end()] + fixed
        clause = _replace_words(clause, TUM_COMMANDS, "familiar command", changes)
        parts[i] = clause
    return "".join(parts).replace(_REFLEXIVE_HOLD, _REFLEXIVE)


def apply_names(text: str, names: dict[str, str], changes: list[str]) -> str:
    """Names/terms left in English letters -> their Hindi spelling from the glossary."""
    for en in sorted(names, key=len, reverse=True):
        loose = re.escape(en).replace(r"\.", r"\.?\s*").replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){loose}(?!\w)", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(names[en], text)
            changes.append(f"name: {en} → {names[en]}")
    return text


def apply_variants(text: str, variants: dict[str, str], changes: list[str]) -> str:
    """Other Hindi spellings of a name -> the glossary's spelling (एसगार्ड -> असगार्ड)."""
    return _replace_words(text, variants, "name spelling", changes) if variants else text


def match_end_punctuation(hindi: str, english: str | None, changes: list[str]) -> str:
    """A question stays a question (the voice needs the '?' to sound like one)."""
    if not english:
        return hindi
    end = english.rstrip()[-1:] if english.strip() else ""
    want = {"?": "?", "!": "!", ".": "।"}.get(end)
    stripped = hindi.rstrip(" ।.?!,")
    have = hindi.rstrip()[-1:] if hindi.strip() else ""
    if want and have != want:
        changes.append(f"ending: {want} (as in the English)")
        return stripped + want
    return hindi


def make_spoken(text: str, speaker_gender: str | None, register: str = "tum",
                names: dict[str, str] | None = None, english: str | None = None,
                variants: dict[str, str] | None = None) -> SpokenResult:
    changes: list[str] = []
    out = apply_names(text, names or {}, changes)
    out = apply_variants(out, variants or {}, changes)
    out = _replace_words(out, COLLOQUIAL, "everyday word", changes)
    if speaker_gender == "female":
        out = feminine_first_person(out, changes)
    if register == "tum":
        out = to_tum(out, changes)
    out = re.sub(r"\.(?=\s|$)", "।", " ".join(out.split()))  # Hindi full stop
    out = match_end_punctuation(out, english, changes)
    return SpokenResult(out, changes)


def protect_names(english: str, names: dict[str, str]) -> str:
    """Before translating: write dotted names without dots (S.H.I.E.L.D. -> SHIELD) so the
    translator treats them as one word instead of spelling out the letters."""
    for term in sorted(names, key=len, reverse=True):
        if "." in term:
            loose = re.escape(term).replace(r"\.", r"\.?\s*")
            english = re.sub(rf"(?<!\w){loose}(?!\w)", term.replace(".", ""), english, flags=re.IGNORECASE)
    return english


# Words that show formal, written Hindi (fewer = more conversational); used to rank options.
def formality_score(text: str) -> int:
    return sum(len(re.findall(rf"(?<![ऀ-ॣ०-ॿ]){re.escape(w)}(?![ऀ-ॣ०-ॿ])", text))
               for w in COLLOQUIAL)
