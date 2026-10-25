"""Translate text with Meta's NLLB-200 model via Hugging Face Transformers."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from dubbing.config import TranslationSettings
from dubbing.errors import PipelineError

log = logging.getLogger(__name__)


class Translator(Protocol):
    """Anything with this method can be used by the translation stage."""

    def translate(self, texts: list[str], src_nllb: str, tgt_nllb: str) -> list[str]: ...

    def info(self) -> dict[str, Any]: ...


class NllbTranslator:
    def __init__(self, settings: TranslationSettings, device: str = "auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.settings = settings
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info("Loading translation model '%s' on %s. First use downloads it (~2.5 GB).",
                 settings.model, device)
        started = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.model).to(device).eval()
        except Exception as exc:
            raise PipelineError(
                f"Could not load translation model '{settings.model}': {exc}\n"
                "Check your internet connection for the first download, and free disk space."
            ) from exc
        log.info("Model ready in %.1fs", time.perf_counter() - started)

    def info(self) -> dict[str, Any]:
        return {
            "engine": "transformers",
            "model": self.settings.model,
            "device": self.device,
            "num_beams": self.settings.num_beams,
            "batch_size": self.settings.batch_size,
        }

    def translate(self, texts: list[str], src_nllb: str, tgt_nllb: str) -> list[str]:
        """Translate a list of texts and return results in the same order."""
        import torch

        if not texts:
            return []
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
        if tgt_token_id is None or tgt_token_id == self.tokenizer.unk_token_id:
            raise PipelineError(f"The translation model does not know the language code '{tgt_nllb}'.")
        self.tokenizer.src_lang = src_nllb

        # Similar-length texts in one batch waste less padding, so sort first.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: list[str | None] = [None] * len(texts)
        batch_size = self.settings.batch_size
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [texts[i] for i in idx]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_token_id,
                    num_beams=self.settings.num_beams,
                    max_new_tokens=self.settings.max_new_tokens,
                    max_length=None,  # the model's stored default conflicts with max_new_tokens
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for i, text in zip(idx, decoded):
                results[i] = text.strip()
            log.info("  translated %d/%d", min(start + batch_size, len(order)), len(order))

        if any(r is None for r in results):
            raise PipelineError("Translation returned fewer results than inputs.")
        return results  # type: ignore[return-value]
