# Limitations

What does not work well yet, with the numbers we measured, and what would fix each one.
Measured on a Windows laptop (AMD Ryzen AI 7 350, 23 GB RAM, **no NVIDIA graphics card**), on two
4-minute film clips: a slow job interview (*The Pursuit of Happyness*) and a fast six-person
argument (*The Avengers*, "We're a Time Bomb").

## 1. It needs a GPU to be practical

Everything runs on the processor (CPU), because the AI tools we use need an NVIDIA graphics card
(CUDA) to run fast, and this PC has an AMD one.

| Step (4-minute clip) | Time on this PC | Notes |
|---|---|---|
| Separate voices from music | ~2 min | |
| Transcribe + recover skipped speech + speakers | ~10 min | |
| Story bible (local AI model) | ~11–14 min | |
| Translate (6 options per line + meaning check) | ~9 min | |
| **Generate speech (114 lines)** | **~3.9 hours** | Slowest step: retries on lines that are too long |
| Mix and build the video | ~1 min | |

**What would fix it:** a PC or cloud machine with an NVIDIA GPU (12 GB or more of graphics memory).
Speech generation, transcription and translation typically run **10–20× faster** on such a card,
which would bring a 4-minute clip from hours down to minutes. The code already uses a GPU
automatically when one is present (`device: auto` in config.yaml), but **this has not been tested**.

## 2. The Hindi speech is too long for fast scenes (the biggest quality problem)

In the fast Avengers argument, the Hindi speech came out **1.36× longer** than the time available
(346 s of speech for 255 s of time). 96 of 114 lines are longer than their slot; 40 are more than
twice as long. The mix can only tighten a line by up to 15% without it sounding rushed, so many
lines overlap the next one.

**Why:** not the translation (Hindi is only ~7% longer than the English) but the **voice model's
speed**: XTTS speaks Hindi at about **7.8 characters per second**; the actors speak English at about
**22**. My earlier planning estimate (13 characters per second) was also too optimistic.

**What would fix it:**
- **Condense lines** to fit, the way professional dubbing adapts scripts (needs a stronger AI model; see 3).
- Allow faster speech (sounds rushed) and let lines run into later pauses (small sync drift).
- Use the measured speed when choosing wordings (not done yet).

## 3. The local AI model is too small for real "understanding"

The story bible (Phase A2) uses **Qwen 2.5 7B** running on this PC. On the Avengers scene it:
- got **3 of 7 character names wrong**, and once gave the same name to three people,
- marked almost every pair "formal", even people shouting at each other,
- suggested unusable English "corrections" (fixed by filtering) and missed the real mishearing,
- disagreed with the reliable turn-taking rules on 73 of 122 lines.

The **rules** (names said in a line, "you people", back-and-forth) do most of the useful work.

**What would fix it:** a bigger model. Options: a 14B local model (~9 GB, about twice as slow on
this PC, quality gain uncertain), or a **large online model** (much better at names, relationships,
formality, spotting mishearings, and condensing lines for timing), which **costs money per video and
sends the dialogue text online**: a business decision.

## 4. Translation

- **IndicTrans2 (200M)** is more natural than NLLB but still sometimes literal or stiff
  ("टेसेरैक्ट असगार्ड से संबंधित है" where people would say "असगार्ड का है").
- **It can get the meaning wrong**: "You want to know my secret, Agent Romanoff?" became
  "my *secret agent* Romanoff"; "we are not alone" once became "we are alone". The meaning check
  (translating back) catches some of these, not all, so **a person must review the translation**.
- **Who is being spoken to is not used yet.** The story bible now knows the listener (and their
  gender), but translation does not use it yet, so "speaking to a woman" verb forms (जाओगी) can be wrong.
- **Formality is one setting for the whole scene** (तुम by default), not per pair of people yet.
- Names written in capitals ("SHIELD MONITORS") can be translated as a word ("ढाल").
- Translations take mistakes from the English transcript with them.

## 5. Voices

- **Cloning quality depends on clean samples.** Film audio rarely has a clean voice: speakers with
  little clean speech get weak clones (Avengers: one speaker had 4.7 s, another none).
- **One-word lines are unreliable**: XTTS sometimes adds sounds after the word (cut off
  automatically), and the clarity check (Whisper listening back) is itself unreliable under ~1 second.
- When a cloned line cannot be understood, a **built-in voice** is used for clarity (27 of 114 lines
  on the Avengers clip); it does not sound like the actor.
- **Emotion is copied from the original line**, not controlled: it can be flatter than the original.
- **No lip-sync.** Close-ups will look dubbed.

## 6. Who is speaking

- Voices are grouped by "voice fingerprints" per line. This works well (pitch confirmed the groups)
  but **short lines (under ~0.6 s) are unreliable** and are marked "uncertain" (37 of 122 lines).
- **Gender is estimated from voice pitch.** It can be wrong for deep female or high male voices,
  shouting, or children, so uncertain cases are marked and you can correct them.
- When people talk over each other, the line is marked "overlapping speech", but only one voice is dubbed.

## 7. Transcription

- **Whisper `medium`** (the larger model was ruled out for speed) mis-hears some words, especially
  in crosstalk ("anchor fees", "bat my eyelash, is that you?"). Mis-heard lines are flagged but not fixed
  automatically: a person (or a stronger AI, see 3) has to correct them.

## 8. Mix and video

- Lines that are too long still **overlap** (see 2).
- MP4 switches on the first subtitle track by default in some players.
- Laughs and sighs are restored only where no dubbed speech plays.

## 9. Licences and consent

- **XTTS v2 and NLLB-200 are licensed for non-commercial use only.** A commercial product would need
  replacements (or licensed alternatives) for the voice model and the meaning-check model.
- Cloning an actor's voice needs their **written consent** and rights to the content. The test clips
  are commercial films, used only for local testing; outputs are never shared and are labelled AI-generated.

## 10. Windows and setup

- Several workarounds were needed: FFmpeg DLLs for the voice model, a **separate Python environment**
  for IndicTrans2 (it needs an older library version), Windows' 260-character path limit, and three
  Hugging Face model pages whose terms must be accepted.
- The project folder is on OneDrive; large files (`data/`, models, Python environments) are kept out
  of Git, and models and environments are outside OneDrive.

## 11. Not built or not tested yet

- Phase D (mix) has **not been run on the new Avengers speech**; it was verified on the earlier flow.
- Using the story bible in translation (listener's gender, formality per pair).
- A single command for the whole flow, and an easy-to-use screen (pick a video and language).
- Re-use of finished work exists for some steps, not all.
- Only **English → Hindi**; other target languages are not supported in this version.
- Running on a GPU.
