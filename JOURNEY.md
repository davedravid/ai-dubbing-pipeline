# The journey: hurdles and how they were solved

This project was built by **Dravid Albert** (product manager, project owner) with an AI coding
assistant (Claude) writing the code. This page explains, in plain words, the problems we hit and how
each one was solved: what the owner spotted and decided, and what was fixed in the code.

## The big turning points (decided by the project owner)

**1. "Build it one phase at a time, and let me approve each phase."**
The first version ran the whole chain (transcribe → translate → speak → mix) in one go. When the
owner watched the result, the problems were tangled together and hard to trace. The owner reset
the approach: tune one phase, look at its output on a review page, approve it, then move on. Every
phase now ends with a review page, and nothing moves forward without approval.

**2. "The English transcript is not subtitles for viewers. It is the script the dubbing is built on."**
The first plan for English subtitles followed rules for viewers (reading speed, 42 characters per line).
The owner pointed out that the text is the **source script for dubbing**, so what matters is different:
exact words, exact timing, the right speaker, gender, pauses, laughs, and clean voice samples for
cloning. That changed Phase A into a "dubbing script".

**3. "The system should read the whole script first, to understand the context."**
Translating line by line, without knowing who is talking to whom, gives wrong grammar in Hindi
(the verb changes with the speaker's and the listener's gender, and तुम vs आप depends on the relationship).
The owner proposed reading the whole script first and using voice pitch to tell male and female apart,
including "a man speaking to a woman". This became **Phase A2, the story bible**, built with
**rules plus a local AI model**, with every suggested correction waiting for the owner's approval.

**4. "Use conversational Hindi."**
The first translator (NLLB) wrote formal, textbook Hindi ("प्रतीक्षा, क्या?" for "Wait, what?").
Following the owner's direction, we compared NLLB with **IndicTrans2** side by side on 20 hard lines.
The owner chose IndicTrans2 (more natural, and MIT-licensed), and a **spoken-Hindi layer** was added:
everyday words, female verb forms, तुम between peers, names in Hindi letters.

**5. "Try it on a harder video."**
The owner brought in a fast six-person argument (*The Avengers*). It exposed problems the slower
job-interview clip never showed: speaker detection merging people, and Hindi speech too slow for fast dialogue.

**6. "Try Spanish."**
Testing Spanish showed that the translation itself was rarely too long. The real bottleneck is how
slowly the voice model speaks Hindi. (The Spanish flow was then removed; this version is Hindi only.)

## Problems the owner heard in the first dubbed video, and the fixes

| What the owner heard | What was really wrong | What we did |
|---|---|---|
| **Lines overlapping** | Hindi was longer than the time the actor took | Gentle fixes in the mix (use pauses, play up to 15% faster without changing the pitch); shorter alternative wordings checked by translating them back. Still the main open problem (see LIMITATIONS) |
| **Laughs were missing** | Separating voices from music took laughs out with the voices | Laughs, sighs and breaths are put back from the original, after checking they are not words |
| **Speech started before the actor** | A "start early" rule, and Whisper's line edges were padded | Lines now start where the actor's voice actually starts, measured on the clean voice track |
| **Actor talking, no audio** | The transcription skipped whole lines (music confused it) | Transcribe the clean voices; then re-listen to every audible stretch with no words; recovered the opening greetings |
| **Same person, different voices** | No speaker detection; short lines borrowed a neighbour's voice | Speakers are detected, and each person gets one voice built from their best clean samples |
| **Default voice too strong** | A deep built-in voice, and a loudness limit set too low | Backup voice now matches the speaker's gender and sound; loudness correction allowed up to 30 dB |
| **Female voice at the start wrong, timing slightly off** | Speaker grouping and edge detection were still rough | Better speaker grouping (voice fingerprints, checked with pitch) and stricter start/end detection |

## Technical hurdles along the way, and how they were solved

**Setup on Windows**
- Python and FFmpeg were not installed. Both were installed with Windows' own installer (winget).
- The voice model needed FFmpeg's DLL files, which the normal FFmpeg download does not include,
  so we added the "shared" FFmpeg build and the code finds it automatically.
- Windows' 260-character path limit broke a file save, so files are kept at short paths.
- IndicTrans2 needs an older library version than the voice model, so it runs in its own small
  Python environment, and the pipeline talks to it through a file.

**Speech-to-text**
- On clean voices, Whisper returned a few very long lines (one of 26 seconds), so we split them into
  sentences using the timing of each word.
- Whisper skipped short greetings and replies, so we added "re-listen to silent-looking stretches".

**Speaker detection**
- On the fast Avengers argument, the standard tool (pyannote) put three different people (one of
  them the only woman) under one label. We switched to comparing a "voice fingerprint" for every
  line, and checked it against voice pitch: the woman's lines formed their own group.
- Very short lines created fake extra "speakers" (23 instead of about 7). Tiny groups are now
  merged into the closest real voice **with a matching pitch**, so a woman's line never joins a man's voice.
- pyannote sends usage statistics by default; the pipeline switches that off.

**Translation**
- Sentences were split at "S.H.I.E.L.D." and "Dr." because of the dots; abbreviations no longer end a sentence.
- The translator sometimes repeated a word 13 times ("नो। नो। नो। …"); such options are now
  penalised, and if all options do it, the extra repeats are removed.
- Names were spelled differently each time (एसगार्ड / असगार्ड), so the glossary lists spelling variants.
- Hindi numbers crashed the voice model ("12"), so numbers are written out as words (बारह) first.
- Our own rule that changes आप to तुम broke the phrase "अपने आप" (oneself); it is now protected.

**Voices**
- On one-word lines, the voice model sometimes kept "talking" nonsense afterwards. Such takes are
  cut at the first pause, re-made with a new random seed, or switched to a clear built-in voice.
- To check clarity without ears, Whisper listens to each dubbed line and compares it with the text.

**Understanding the story**
- The local AI model (7B) got names and formality wrong on a busy scene, so the reliable rules
  now take priority, the AI fills only the gaps, and every disagreement is shown for review.

**Measuring instead of guessing**
- The planning estimate of how fast Hindi is spoken (13 characters per second) turned out to be
  wrong: we measured 7.8. That measurement explained the timing problem and changed the plan.

## How the work was checked

- 160+ automatic tests run in seconds after every change; real-model tests run on real clips.
- Every phase writes a review page (click a time to play that moment; listen to every dubbed line).
- Results were checked against measurements (pitch, loudness, timing, back-translation), not assumed.
- Every working version is saved in Git and uploaded to a private GitHub repository.
