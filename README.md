# AI Dubbing Pipeline

Local tool that transcribes a video, translates it, and produces a dubbed version.
Current focus: **English → Hindi**.

## Status

| Phase | Feature | Status |
|---|---|---|
| 2 | Video → audio extraction → timestamped transcript (JSON) | **Verified 2026-10-25**: 16 unit tests + 4 integration tests pass on a 4-min English clip (84 segments, English detected, about 1× real time on CPU) |
| 3 | English → Hindi translation (NLLB-200 600M) with completeness checks and timing budget | **Verified 2026-10-25**: job_001's 84 segments → 69 sentence units, all translated, 100% Devanagari, 12 flagged for review; 47 unit + 2 model tests pass. Translation *quality* not yet reviewed by a native speaker |
| 5 | Voice/music separation (Demucs), timing fixes, loudness matching, dubbed MP4 with switchable audio and subtitles | **Built and checked 2026-10-26**: job_001_hi.mp4 passes all 7 checks (plays, picture bit-identical, sound length within 0.005 s, loudness within 0.02 LU, no distortion, all 69 lines, all 161 captions); 0 words of original dialogue leaked into the background. First viewing found overlaps, missing laughs, early starts, missed speech, changing voices and a too-strong default voice. |
| 5b | Fixes from that viewing: transcription from separated voices + gap recovery, sentence splitting, pyannote speakers (one voice per person, each line's own delivery), shorter wordings checked by back-translation, laughs kept, lines start at the actor's voice, loudness cap 30 dB, closest built-in backup voice | **Run as job_002, 2026-10-26**: overlaps 39 → 8 lines (47.8 s → 6.7 s); early starts 11 → 0; 21 laughs/sounds kept; 24/33 lines in their speaker's voice; built-in voice 26 → 8 lines. **Worse:** 5 lines fail the clarity check (job_001: 2). **Open:** 3 very short lines (≤0.55 s) still not dubbed (gap recovery ignores stretches under 0.5 s). Not yet watched by a person |
| 4 | Delivery measurements + Hindi speech with XTTS v2 voice cloning, back-check, retries, invented-speech cut | **Verified 2026-10-26**: all 69 job_001 lines generated; median back-check error 12.5%; 0 lines with invented speech; 2 one-word lines still flagged (units 21 "जे।" and 61 "ठीक है।"; Whisper is unreliable on clips under 1 s, so these need a listen). Voices: 13 cloned from their own line, 30 from a nearby line, 26 built-in. 44 lines longer than their slot (Phase 5). 73 fast + 2 speech-model tests pass. **Not yet reviewed by ear** |
| 5 | Mixing, video output, subtitles | Not started |
| 6 | Multiple speakers | Not started |
| 7 | Vocal separation, caching, QA reports | Not started |
| 8 | Easy file picker, full docs | Not started |

## One-time setup (Windows)

Already done on this PC on 2026-10-25. To repeat on another PC:

1. **Python 3.11:** `winget install --id Python.Python.3.11 --exact --scope user`
2. **FFmpeg:** `winget install --id Gyan.FFmpeg --exact`, then open a **new** terminal window.
   For speech (Phase 4) also install FFmpeg's DLLs: `winget install --id Gyan.FFmpeg.Shared --exact`.
3. **Virtual environment** (a private package folder, kept outside OneDrive):
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv "$env:USERPROFILE\.venvs\dubbing"
   & "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

AI models download automatically on first use to `C:\Users\<you>\.cache\huggingface`
(Whisper `medium` ≈ 1.5 GB, `large-v3` ≈ 3 GB, NLLB ≈ 2.5 GB, XTTS v2 ≈ 1.9 GB).

## Running

Open PowerShell **in this folder** (in File Explorer: right-click the folder → *Open in Terminal*).

```powershell
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing transcribe "C:\path\to\video.mp4" --job-id job_001
```

Options:

- `--job-id NAME`: name used for this run's files (letters, numbers, `_`, `-`).
- `--preset fast|quality`: `fast` uses Whisper `medium`, `quality` uses `large-v3` (much slower on CPU).
- `--language en`: skip auto-detection.
- `--overwrite`: replace this job's existing transcript (otherwise the tool refuses).
- `--verbose`: more detail on screen.

Default settings live in [config.yaml](config.yaml).

Translate a finished transcript (first run downloads NLLB, about 2.5 GB):

```powershell
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing translate --job-id job_001 --to hi
```

Make subtitles (`--lang hi` uses the Hindi translation; without `--lang`, the transcript):

```powershell
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing subtitles --job-id job_001 --lang hi
```

This writes `data/subtitles/job_001_<lang>.vtt` (max 2 lines of 42 characters per caption).

### How translation works

1. Transcript segments are grouped into **sentence units**. Whisper often splits one sentence
   across segments, and translating halves separately breaks the grammar. Each unit lists its
   `source_segment_ids`, and the run stops without saving if any segment is missing or duplicated.
2. Each unit is translated with NLLB and checked. Flags to review:

| Flag | Meaning |
|---|---|
| `empty_translation` | The model returned nothing |
| `wrong_script` | Output is mostly not in the target script (probably untranslated) |
| `unchanged` | Output is identical to the input |
| `unusual_length` | Translation is under 0.4× or over 3× the source length |
| `repetition` | The same word 4+ times in a row (a known model failure) |
| `numbers_only` | Only digits, e.g. "12." Fine for subtitles; speech will need it in words |
| `may_not_fit` | Estimated Hindi speech is over 1.25× the time available before the next line. This is an **estimate** (from the speaker's own pace); Phase 4 measures real durations |

Each unit also records timing facts for dubbing: `duration_s`, `available_s` (time until the
next line), `pause_before_s`, `pause_after_s`, `source_words_per_s`, `speaker_pace`, `est_speedup_needed`.

### Measuring the original delivery

```powershell
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing analyze --job-id job_001
```

Writes `data/features/job_001_en.json`: for every sentence unit, pitch (median, range in
semitones), loudness, share of the slot that is voiced, words per second and pauses. These are
**measurements, not emotion labels**, and film music/effects can distort them.

### Generating dubbed speech

```powershell
# Quick test: first 5 lines with a built-in voice
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing speak --job-id job_001 --lang hi --voice standard --limit 5
# Full run with voice cloning (about 30-40 min for 4 minutes of video on this PC's CPU)
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing speak --job-id job_001 --lang hi --voice clone
```

For each sentence unit:

1. **Voice:** `clone` copies the voice *and delivery* from that line's own original audio.
   Lines under `tts.min_reference_s` (3 s) borrow the nearest long-enough line within 10 s;
   until speaker detection (Phase 6) that may be a different person, so they are flagged.
2. **Text:** numbers are written as Hindi words (XTTS cannot read Hindi digits) and Latin full stops become "।".
3. **Fit:** if the speech is longer than the time until the next line, it is generated again faster (up to `tts.max_speed`).
4. **Back-check:** Whisper transcribes the dub; if it heard something too different
   (character error rate > 0.35), another take is generated with a new random seed (3 takes at most) and the best is kept.
5. **Delivery comparison:** pitch, pitch range and loudness of the dub vs. the original.

Every WAV is labelled **AI-generated** in its metadata.

Remake only the lines that failed the checks (keeps all other lines):

```powershell
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing speak --job-id job_001 --lang hi --redo-flagged
```

**Known limits:** XTTS is unreliable on one-word lines (it may add sounds after the word; the
pipeline cuts those off at the first pause), and the Whisper back-check is unreliable on clips
under a second, so very short lines should be checked by ear. Until speaker detection (Phase 6),
lines under 3 s borrow a nearby line's voice, which may belong to another character.

**Fallback voice (Meta MMS):** not used for Hindi, because XTTS v2 supports Hindi and MMS cannot
clone voices. MMS would only matter for languages XTTS lacks (e.g. Telugu, Odia); asking for those
today gives a clear "not built yet" error.

**What makes a good voice reference:** at least 3 seconds of one person speaking clearly, without
music, effects or other voices over it. Film audio rarely is, which is why vocal separation
(Demucs) is planned next to the mixing work.

| Speech flag | Meaning |
|---|---|
| `reference_borrowed` | Voice taken from a nearby line (may be another speaker) |
| `builtin_voice_fallback` | No usable reference nearby; built-in voice used |
| `reference_mostly_silence` / `_clipped` / `_very_quiet` | Weak reference audio |
| `overruns_slot` | Still longer than its slot at maximum speed (Phase 5 must handle it) |
| `retried` | First take failed the back-check |
| `invented_speech` | XTTS kept talking after the real words, and it could not be cut away |
| `tail_cut` | Invented speech after the words was cut off at the first pause (check the ending) |
| `clarity_fallback_builtin` | The cloned voice could not be understood, so the built-in voice was used |
| `low_intelligibility` | Even the best take failed the back-check: listen to it |
| `flatter_than_original` | Pitch range over 4 semitones narrower than the original (less expressive) |

### Making the dubbed video

```powershell
# 1) Split the original sound into voices and music/effects (about 2 min for 4 min of video)
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing separate --job-id job_001
# 2) Mix the Hindi voices onto the music/effects and build the video
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m dubbing mix --job-id job_001 --lang hi
```

(`mix` runs `separate` by itself if it has not been run.)

What `mix` does:

1. **Keeps the music and effects:** Demucs separates the original voices from everything else; the
   dubbed voices go on top of the "everything else" track. A leak check listens to that track with
   Whisper and reports any original dialogue still audible in it.
2. **Fixes overlaps, gentlest first:** use the pause after a line → start up to 0.3 s early → play it
   up to 15% faster with the voice's pitch unchanged. Words are never cut and lines never start
   later than the original. Lines that still do not fit are flagged `needs_shorter_translation`.
3. **Loudness:** each dubbed line is set to the loudness of the original line it replaces; the music
   dips by 2 dB under speech; the whole soundtrack is matched to the original's loudness (LUFS);
   a limiter keeps peaks at or below -1 dB.
4. **Builds the video:** the original picture is copied exactly (not re-encoded). Tracks: Hindi dub
   (default), original English (switchable), Hindi and English subtitles (switchable). MP4 turns
   on the first subtitle track automatically in some players; viewers can switch it off.
5. **Checks** (in `data/qa_reports/<job>_<lang>_mix_qa.json`): plays without errors, picture
   identical to the original, sound the same length as the picture, loudness within 1 LU of the
   original, no distortion, every line present, all subtitle captions present. It also lists the
   times worth a listen.

### Speaker detection setup (one time)

Needs a free Hugging Face account. Accept the terms on **all three** pages (same account):
https://huggingface.co/pyannote/speaker-diarization-3.1,
https://huggingface.co/pyannote/segmentation-3.0 and
https://huggingface.co/pyannote/speaker-diarization-community-1 (pyannote 4 fetches one file
from it). Create a "Read" token at https://huggingface.co/settings/tokens and save it with
`setx HF_TOKEN "hf_..."` (never put it in code or config). pyannote's usage statistics are
switched off by the pipeline.

## Where files go

| File | What it is |
|---|---|
| `data/transcripts/<job>_<lang>.json` | The transcript: segments with start/end seconds, text, word timings |
| `data/translations/<job>_<lang>.json` | Sentence units: English, Hindi, timing facts, review flags |
| `data/subtitles/<job>_<lang>.vtt` | Subtitles for any video player |
| `data/features/<job>_<src>.json` | Delivery measurements of the original lines |
| `data/references/<job>/unit_NNN.wav` | Original audio cut out as voice-cloning references |
| `data/tts/<job>_<lang>/unit_NNN.wav` | One dubbed WAV per sentence unit |
| `data/tts/<job>_<lang>.json` | Speech manifest: voice used, speed, takes, back-check, delivery comparison, flags |
| `data/outputs/<job>_<lang>_speech_preview.wav` | All dubbed lines placed on the video's timeline (speech only, not yet mixed) |
| `data/outputs/<job>_<lang>.mp4` | **The dubbed video** |
| `data/outputs/<job>_<lang>.wav` | The final dubbed soundtrack (music/effects + dubbed voices) |
| `data/qa_reports/<job>_<lang>_mix_qa.json` | Checks on the dubbed video, timing fixes, loudness, what to listen to |
| `data/temp/<job>_vocals.wav`, `_background.wav` | Original voices and music/effects, separated by Demucs |
| `data/temp/<job>_asr16k.wav`, `_source24k.wav` | Audio extracted for recognition / analysis (safe to delete) |
| `data/logs/<job>.log` | Detailed log of the run |

## Tests

```powershell
# Fast checks (seconds)
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m pytest -m "not integration"

# Real model on a real English video (minutes)
$env:DUBBING_TEST_VIDEO = "C:\path\to\english_video.mp4"
& "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m pytest -m integration
```

## Troubleshooting

- **"FFmpeg was not found"**: close and reopen the terminal after installing FFmpeg, or set
  `paths.ffmpeg` in `config.yaml` to the full path of `ffmpeg.exe`.
- **"Python was not found; run without arguments to install from the Microsoft Store"**: that is a
  Windows shortcut, not real Python. Use the full venv path shown above.
- **Slow**: this PC has no NVIDIA GPU, so everything runs on the CPU. Use short clips and the `fast` preset.
- **"already has a transcript"**: use a new `--job-id` or add `--overwrite`.
- **"From Pytorch 2.9, the torchcodec library is required"** or **"FFmpeg's shared libraries (DLLs) were not found"**:
  install `torchcodec==0.17.0` and `winget install --id Gyan.FFmpeg.Shared --exact`.
- **"System error" when saving a file**: Windows' 260-character path limit (long paths are off on
  this PC). Keep the project and data folders at short paths.

## Licensing and consent

Whisper/faster-whisper and Demucs are MIT-licensed. NLLB-200 and MMS are CC-BY-NC 4.0, and XTTS v2 is under the
Coqui Public Model License; all three are for **non-commercial use only**. The CPML was accepted on
2026-10-25 for internal testing (`tts.accept_coqui_license` in config.yaml).

Voice cloning of the test clip's actors is for **local internal testing only**: cloned audio is not
shared or published, and every generated file is labelled AI-generated. Anything beyond testing
needs voices with the speaker's written consent.
