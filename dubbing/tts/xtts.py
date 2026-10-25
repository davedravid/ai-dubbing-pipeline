"""Coqui XTTS v2: multilingual speech generation with voice cloning."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from dubbing.config import Settings
from dubbing.errors import PipelineError
from dubbing.media.ffmpeg import enable_ffmpeg_dlls

log = logging.getLogger(__name__)

LICENSE_NOTE = "XTTS v2, Coqui Public Model License: non-commercial use only"


class Voice:
    """What XTTS needs to speak in a particular voice ("conditioning")."""

    def __init__(self, gpt_cond_latent: Any, speaker_embedding: Any, description: str):
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding
        self.description = description


class XttsSynthesizer:
    def __init__(self, settings: Settings):
        cfg = settings.tts
        if not cfg.accept_coqui_license:
            raise PipelineError(
                "XTTS v2 is disabled: tts.accept_coqui_license is false in config.yaml.\n"
                "XTTS is licensed for non-commercial use only (https://coqui.ai/cpml)."
            )
        enable_ffmpeg_dlls(settings.ffmpeg_dll_dir)
        # Records the licence acceptance from config.yaml so Coqui does not prompt.
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        import torch
        from TTS.api import TTS

        self._torch = torch
        self.cfg = cfg
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info("Loading XTTS v2 on %s. First use downloads it (~1.9 GB).", device)
        started = time.perf_counter()
        try:
            self._api = TTS(cfg.model).to(device)
        except Exception as exc:
            raise PipelineError(f"Could not load XTTS model '{cfg.model}': {exc}") from exc
        self.model = self._api.synthesizer.tts_model
        self.sample_rate = int(self._api.synthesizer.output_sample_rate)
        self.languages = set(getattr(self._api, "languages", None) or [])
        log.info("XTTS ready in %.1fs", time.perf_counter() - started)

    # ---- voices -------------------------------------------------------------

    def builtin_speakers(self) -> list[str]:
        return list(self.model.speaker_manager.speakers.keys())

    def builtin_voice(self, name: str) -> Voice:
        speakers = self.model.speaker_manager.speakers
        if name not in speakers:
            raise PipelineError(
                f"XTTS has no built-in voice called '{name}'. Available: {', '.join(sorted(speakers))}"
            )
        s = speakers[name]
        return Voice(s["gpt_cond_latent"], s["speaker_embedding"], f"built-in voice '{name}'")

    def voice_from_audio(self, reference_wavs: list[Path], description: str) -> Voice:
        """Build a cloned voice from one or more reference WAV files."""
        latent, embedding = self.model.get_conditioning_latents(audio_path=[str(p) for p in reference_wavs])
        return Voice(latent, embedding, description)

    # ---- speech -------------------------------------------------------------

    def speak(self, text: str, xtts_lang: str, voice: Voice, speed: float = 1.0, seed_offset: int = 0) -> np.ndarray:
        """Generate speech. A different `seed_offset` gives a different take of the same line."""
        if self.languages and xtts_lang not in self.languages:
            raise PipelineError(f"XTTS cannot speak language '{xtts_lang}'.")
        limit = self.model.tokenizer.char_limits.get(xtts_lang, 250)
        pieces = []
        for chunk in split_text(text, limit):
            self._torch.manual_seed(self.cfg.seed + seed_offset)
            out = self.model.inference(
                chunk,
                xtts_lang,
                voice.gpt_cond_latent,
                voice.speaker_embedding,
                temperature=self.cfg.temperature,
                speed=speed,
                # XTTS's own splitter needs the large spaCy package; split_text() does it instead.
                enable_text_splitting=False,
            )
            wav = out["wav"]
            if hasattr(wav, "cpu"):
                wav = wav.cpu().numpy()
            pieces.append(np.asarray(wav, dtype=np.float32).reshape(-1))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


_SENTENCE_BREAK = re.compile(r"(?<=[.?!।])\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def split_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters, preferring sentence ends,
    then commas, then spaces. Short text comes back as one piece."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    for pattern in (_SENTENCE_BREAK, _CLAUSE_BREAK, re.compile(r"\s+")):
        parts = [p for p in pattern.split(text) if p]
        if len(parts) > 1:
            pieces, current = [], ""
            for part in parts:
                candidate = f"{current} {part}".strip()
                if len(candidate) <= limit:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)
                    current = part
            if current:
                pieces.append(current)
            return [q for p in pieces for q in split_text(p, limit)]
    # One unbroken run of characters longer than the limit: cut it.
    return [text[i:i + limit] for i in range(0, len(text), limit)]
