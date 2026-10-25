# AI Dubbing Pipeline

Local tool that transcribes a video, translates it, and produces a dubbed version.
Current focus: **English → Hindi**.

## Status

| Phase | Feature | Status |
|---|---|---|
| 2 | Video → audio extraction → timestamped transcript (JSON) | **Verified 2026-10-25**: 16 unit tests + 4 integration tests pass on a 4-min English clip (84 segments, English detected, about 1× real time on CPU) |
| 3 | English → Hindi translation (NLLB-200 600M) with completeness checks and timing budget | **Verified 2026-10-25**: job_001's 84 segments → 69 sentence units, all translated, 100% Devanagari, 12 flagged for review; 47 unit + 2 model tests pass. Translation *quality* not yet reviewed by a native speaker |
| 5 (early) | WebVTT subtitles (English and Hindi) | **Verified 2026-10-25**: FFmpeg reads all cues of both files. Embedding into video: not started |
| 4 | Dubbed speech (XTTS v2) | Not started |
| 5 | Mixing, video output, subtitles | Not started |
| 6 | Multiple speakers | Not started |
| 7 | Vocal separation, caching, QA reports | Not started |
| 8 | Easy file picker, full docs | Not started |

## One-time setup (Windows)

Already done on this PC on 2026-10-25. To repeat on another PC:

1. **Python 3.11:** `winget install --id Python.Python.3.11 --exact --scope user`
2. **FFmpeg:** `winget install --id Gyan.FFmpeg --exact`, then open a **new** terminal window.
3. **Virtual environment** (a private package folder, kept outside OneDrive):
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv "$env:USERPROFILE\.venvs\dubbing"
   & "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

AI models download automatically on first use to `C:\Users\<you>\.cache\huggingface`
(Whisper `medium` ≈ 1.5 GB, `large-v3` ≈ 3 GB).

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

## Where files go

| File | What it is |
|---|---|
| `data/transcripts/<job>_<lang>.json` | The transcript: segments with start/end seconds, text, word timings |
| `data/translations/<job>_<lang>.json` | Sentence units: English, Hindi, timing facts, review flags |
| `data/subtitles/<job>_<lang>.vtt` | Subtitles for any video player |
| `data/temp/<job>_asr16k.wav` | Audio extracted for speech recognition (safe to delete) |
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

## Licensing note

Whisper/faster-whisper are MIT-licensed. Models planned for later phases (NLLB-200, XTTS v2, MMS)
are licensed for **non-commercial use only**. Only clone voices with the speaker's written consent.
