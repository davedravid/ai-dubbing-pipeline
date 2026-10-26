"""Translate with AI4Bharat's IndicTrans2 (MIT licence), built for Indian languages.

IndicTrans2's code needs Transformers 4, but the rest of the pipeline needs version 5,
so it runs in its own small Python environment (C:\\Users\\<you>\\.venvs\\indictrans).
This class sends it the sentences through a JSON file and reads the translations back.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dubbing.errors import PipelineError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
DEFAULT_PYTHON = Path.home() / ".venvs" / "indictrans" / "Scripts" / "python.exe"
WORKER = Path(__file__).with_name("indictrans_worker.py")


class IndicTransTranslator:
    def __init__(self, model: str = DEFAULT_MODEL, python: Path | None = None, num_beams: int = 5,
                 batch_size: int = 8, max_new_tokens: int = 256):
        self.model_name = model
        self.python = Path(python) if python else DEFAULT_PYTHON
        self.num_beams, self.batch_size, self.max_new_tokens = num_beams, batch_size, max_new_tokens
        if not self.python.is_file():
            raise PipelineError(
                f"IndicTrans2's separate Python environment was not found at {self.python}.\n"
                "Create it with:  python -m venv %USERPROFILE%\\.venvs\\indictrans  then install "
                "torch, transformers==4.57.6 and sentencepiece into it (see README)."
            )

    def info(self) -> dict[str, Any]:
        return {"engine": "indictrans2", "model": self.model_name, "num_beams": self.num_beams,
                "runs_in": str(self.python)}

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        return [c[0] for c in self.translate_candidates(texts, src, tgt, 1)]

    def translate_candidates(self, texts: list[str], src: str, tgt: str, n: int) -> list[list[str]]:
        """Up to `n` translations per text, best first. Codes are FLORES (eng_Latn, hin_Deva), as for NLLB."""
        if not texts:
            return []
        with tempfile.TemporaryDirectory() as tmp:
            req, res = Path(tmp) / "request.json", Path(tmp) / "result.json"
            req.write_text(json.dumps({"model": self.model_name, "texts": texts, "src": src, "tgt": tgt, "n": n,
                                       "num_beams": self.num_beams, "batch_size": self.batch_size,
                                       "max_new_tokens": self.max_new_tokens}, ensure_ascii=False), encoding="utf-8")
            log.info("Translating %d sentences with IndicTrans2 (separate environment)...", len(texts))
            done = subprocess.run([str(self.python), "-X", "utf8", str(WORKER), str(req), str(res)],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            if done.returncode != 0 or not res.is_file():
                tail = "\n".join(done.stderr.strip().splitlines()[-6:])
                raise PipelineError(f"IndicTrans2 failed:\n{tail}")
            result = json.loads(res.read_text(encoding="utf-8"))
        log.info("IndicTrans2 finished in %ss", result["seconds"])
        if len(result["candidates"]) != len(texts):
            raise PipelineError("IndicTrans2 returned a different number of results than sentences sent.")
        return result["candidates"]
