# AI Dubbing Pipeline

A local tool that dubs an **English video into Hindi**: it writes down what is said and by whom,
understands the scene, translates into conversational Hindi, speaks each line in the original
actor's cloned voice, and mixes it back with the original music and effects.

Everything runs on this PC. The work is done **one phase at a time**: each phase ends with a
review page, and the next phase starts only after a person approves.

- **[LIMITATIONS.md](LIMITATIONS.md)**: what does not work well yet, and what it would take (GPU, bigger models, licences).
- **[JOURNEY.md](JOURNEY.md)**: the problems we hit while building this and how each was solved, in plain words.

## The flow

```
Video ─► A. Dubbing script ─► A2. Story bible ─► B. Translation ─► C. Speech ─► D. Mix + video
         who/what/when         who→whom, tone     Hindi, spoken     cloned voices   music kept
         [review page]         [review page]      [review page]     [review page]   [QA report]
```

| Phase | What it does | Status (on the test clips) |
|---|---|---|
| **A. Dubbing script** | Separates voices from music (Demucs), transcribes the clean voices (Whisper `medium`), recovers skipped speech, splits sentences, groups lines by voice fingerprint, estimates gender from pitch, snaps line edges to the real voice, records pauses, laughs, overlaps, delivery, and picks the best clean voice samples per speaker | **Approved** on the Avengers clip (122 lines, 7 speakers, 99% of audible voice covered) |
| **A2. Story bible** | Reads the whole script with rules + a local AI model: who talks to whom, the listener's gender, formality per pair, scene and tone, names, suggested English corrections (only applied if you accept them) | **Built, under review.** Rules work well for clear exchanges; the local 7B AI model is weak on names and formality (see LIMITATIONS) |
| **B. Translation** | IndicTrans2 gives 6 wordings per sentence; meaning checked by translating back (NLLB); the best is chosen for meaning, time, everyday words and no English left; a spoken-Hindi layer fixes female verb forms, तुम/आप, names in Hindi letters, punctuation | **Approved** on the Avengers clip (114 sentences, 0 English left, 0 wrong female verbs). Not yet using A2's who-talks-to-whom |
| **C. Speech** | XTTS v2 speaks each line in that person's cloned voice (with the line's own delivery when long enough), shorter wordings for tight lines, Whisper back-check, retries, babble cut, gender-matched backup voice | **Built.** Voices consistent per speaker (cloned pitch within 0.4 semitones of the originals). **Main problem: the Hindi speech is 1.36× longer than the time available** |
| **D. Mix + video** | Music and effects kept, laughs restored, lines start with the actor, gentle tightening, loudness matched, original picture untouched, Hindi + English audio and subtitles | **Verified** on the job-interview clip with the earlier flow (all checks pass). Not yet run on the Avengers speech |

## One-time setup (Windows)

All of this is already done on this PC. To repeat it on another PC:

1. **Python 3.11:** `winget install --id Python.Python.3.11 --exact --scope user`
2. **FFmpeg** (program + its DLLs, which the voice model needs):
   `winget install --id Gyan.FFmpeg --exact` and `winget install --id Gyan.FFmpeg.Shared --exact`,
   then open a **new** terminal window.
3. **Main Python environment** (kept outside OneDrive):
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv "$env:USERPROFILE\.venvs\dubbing"
   & "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe" -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
4. **IndicTrans2's own environment** (it needs an older Transformers than the voice model):
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv "$env:USERPROFILE\.venvs\indictrans"
   & "$env:USERPROFILE\.venvs\indictrans\Scripts\python.exe" -m pip install torch==2.14.1 transformers==4.44.2 sentencepiece==0.2.2 --extra-index-url https://download.pytorch.org/whl/cpu
   ```
5. **Hugging Face access** (free account): accept the terms on these pages with one account:
   [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
   [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
   [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
   [indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M).
   Create a "Read" token at https://huggingface.co/settings/tokens and save it with
   `setx HF_TOKEN "hf_..."`. Never put the token in code or config.
6. **Local AI model for the story bible:** `winget install --id Ollama.Ollama`, then
   `ollama pull qwen2.5:7b-instruct` (4.7 GB). Ollama runs on this PC only.

Models download on first use (about 15 GB in total) to `C:\Users\<you>\.cache` and `C:\Users\<you>\.ollama`.
pyannote's usage statistics are switched off by the pipeline.

## Running the Hindi flow

Open PowerShell in this folder (File Explorer: right-click the folder → *Open in Terminal*) and use
the main environment's Python. Shorthand used below: `$py = "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe"`.

```powershell
$py = "$env:USERPROFILE\.venvs\dubbing\Scripts\python.exe"

# A. Dubbing script (about 10-15 min for 4 min of video)
& $py -m dubbing transcribe "C:\path\to\video.mp4" --job-id job_003
& $py -m dubbing script --job-id job_003
#   review: data/scripts/job_003_review.html   (optional: names/genders in speakers/job_003.yaml)

# A2. Story bible (about 10-15 min; needs Ollama running)
& $py -m dubbing story --job-id job_003
#   review: data/context/job_003_story_review.html
#   accept/reject English corrections in context/job_003_corrections.yaml

# B. Translation into Hindi (about 10 min)
& $py -m dubbing translate --job-id job_003 --to hi
#   review: data/translations/job_003_hi_review.html

# C. Speech (about 1-4 hours on this PC's CPU; see LIMITATIONS)
& $py -m dubbing speak --job-id job_003 --lang hi
#   review: data/tts/job_003_hi_review.html   (remake only failed lines: add --redo-flagged)

# D. Mix and dubbed video (a few minutes)
& $py -m dubbing mix --job-id job_003 --lang hi
#   result: data/outputs/job_003_hi.mp4, checks: data/qa_reports/job_003_hi_mix_qa.json
```

Every command refuses to overwrite earlier results unless you add `--overwrite`.
Settings (models, thresholds, formality default, voices) live in [config.yaml](config.yaml).

### Files you edit

| File | What for |
|---|---|
| `glossaries/<job>.txt` | Names and terms, one per line, optionally with their Hindi spelling: `Tesseract = टेसेरैक्ट \| टेसराक्ट` (variants after `\|` are changed to the first) |
| `speakers/<job>.yaml` | Your corrections to speakers: `name`, `gender`, `same_as` (merge two labels). Created automatically, never overwritten |
| `context/<job>_corrections.yaml` | Suggested English corrections: set `accept: yes` or `no`. Only accepted ones are used. Kept on this PC only (contains dialogue) |

## Where results go

| File | What it is |
|---|---|
| `data/scripts/<job>_script.json`, `_review.html` | Phase A dubbing script and its review page |
| `data/scripts/<job>_speakers/S1.wav` … | Each speaker's chosen clean voice samples, to listen and confirm |
| `data/context/<job>_story.json`, `_story_review.html` | Phase A2 story bible and its review page |
| `data/translations/<job>_hi.json`, `_hi_review.html` | Phase B translation and its review page |
| `data/tts/<job>_hi/unit_NNN.wav`, `data/tts/<job>_hi_review.html` | Phase C dubbed lines and their review page |
| `data/outputs/<job>_hi.mp4` | **The dubbed video** (Hindi + original audio, Hindi + English subtitles) |
| `data/qa_reports/<job>_hi_mix_qa.json` | Checks on the final video, and the times worth a listen |
| `data/temp/…` | Separated voices/music and other working files (safe to delete) |

`data/` stays on this PC: it is never added to Git, because it contains the films' audio and dialogue.

## Tests

```powershell
& $py -m pytest -m "not integration"        # 160+ fast checks, a few seconds
$env:DUBBING_TEST_VIDEO = "C:\path\to\english_video.mp4"
& $py -m pytest -m integration              # real models on a real video (minutes)
```

## Troubleshooting

- **"FFmpeg was not found"**: open a new terminal after installing FFmpeg, or set `paths.ffmpeg` in config.yaml.
- **"Python was not found; … Microsoft Store"**: a Windows shortcut, not real Python. Use the full venv path above.
- **"torchcodec is required" / "FFmpeg's shared libraries (DLLs) were not found"**: install `Gyan.FFmpeg.Shared`.
- **"403 … restricted"** from Hugging Face: accept the terms on the model's page with the same account as your token.
- **"Ollama is not running"**: start the Ollama app (or run `ollama serve`).
- **"System error" when saving a file**: Windows' 260-character path limit. Keep folders at short paths.
- **Very slow**: no NVIDIA GPU on this PC, so everything runs on the CPU. See LIMITATIONS.md.

## Licensing and consent

Whisper, Demucs, pyannote code, IndicTrans2 (MIT), Qwen 2.5 (Apache 2.0) and Ollama allow commercial use.
**XTTS v2 (Coqui Public Model License) and NLLB-200 (CC-BY-NC 4.0) are non-commercial only**: XTTS
speaks every line and NLLB checks meanings, so the pipeline as a whole is for **non-commercial testing**.

Voice cloning of the test clips' actors is for **local internal testing only**: cloned audio is not
shared or published, and every generated file is labelled AI-generated. Anything beyond testing
needs voices with the speakers' written consent and rights to the content.
