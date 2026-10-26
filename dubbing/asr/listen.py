"""Whisper used as a 'listener': what does it hear in a short clip?"""

from __future__ import annotations

import numpy as np

from dubbing.audio.clips import resample
from dubbing.config import Settings


class Listener:
    def __init__(self, settings: Settings, lang: str):
        from faster_whisper import WhisperModel

        from dubbing.asr.transcriber import resolve_device

        device, compute_type = resolve_device(settings.device)
        self.model = WhisperModel(settings.asr.model, device=device, compute_type=compute_type)
        self.lang = lang

    def heard(self, clip: np.ndarray, sr: int) -> str:
        if len(clip) < sr * 0.2:
            return ""
        segments, _ = self.model.transcribe(resample(clip, sr, 16000), language=self.lang, beam_size=5,
                                            vad_filter=False)
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
