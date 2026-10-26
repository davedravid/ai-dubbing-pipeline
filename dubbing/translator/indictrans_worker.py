"""IndicTrans2 worker: runs in its OWN Python environment (Transformers 4.x).

IndicTrans2's code needs Transformers 4, while the rest of the pipeline uses version 5
(needed by the voice model). So the pipeline runs this file with the separate
environment's Python, passing a JSON request file and receiving a JSON result file.
It imports nothing from the pipeline on purpose.

Request:  {"model": ..., "texts": [...], "src": "eng_Latn", "tgt": "hin_Deva", "n": 6,
           "num_beams": 5, "batch_size": 8, "max_new_tokens": 256}
Result:   {"candidates": [[...], ...], "seconds": 12.3}
"""

import json
import os
import re
import sys
import time


def _install_onnx_stand_in():
    import importlib.util
    import types

    if "transformers.onnx" in sys.modules or importlib.util.find_spec("transformers.onnx") is not None:
        return
    onnx, utils = types.ModuleType("transformers.onnx"), types.ModuleType("transformers.onnx.utils")

    class OnnxConfig:
        def __init__(self, *a, **k):
            pass

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        pass

    onnx.OnnxConfig, onnx.OnnxSeq2SeqConfigWithPast = OnnxConfig, OnnxSeq2SeqConfigWithPast
    utils.compute_effective_axis_dimension = lambda dimension, fixed_dimension, num_token_to_add=0: dimension
    onnx.utils = utils
    sys.modules["transformers.onnx"], sys.modules["transformers.onnx.utils"] = onnx, utils


def _prepare(text):
    return " ".join(text.replace("’", "'").replace("“", '"').replace("”", '"').split())


def _tidy(text):
    return " ".join(re.sub(r"\s+([,.?!।])", r"\1", text).split())


def main(request_path, result_path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    req = json.load(open(request_path, encoding="utf-8"))
    _install_onnx_stand_in()
    started = time.perf_counter()
    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(req["model"], trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(req["model"], trust_remote_code=True, token=token).eval()
    n, beams = req["n"], max(req.get("num_beams", 5), req["n"])
    out = []
    texts = req["texts"]
    for start in range(0, len(texts), req.get("batch_size", 8)):
        batch = [f'{req["src"]} {req["tgt"]} {_prepare(t)}' for t in texts[start:start + req.get("batch_size", 8)]]
        inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt")
        with torch.inference_mode():
            kwargs = dict(num_beams=beams, num_return_sequences=n, max_length=req.get("max_new_tokens", 256))
            try:
                # The speed-up ("cache") works with the Transformers version IndicTrans2 was built for...
                gen = model.generate(**inputs, use_cache=True, **kwargs)
            except (AttributeError, TypeError):
                # ...and newer versions changed its format, so fall back: same output, several times slower.
                gen = model.generate(**inputs, use_cache=False, **kwargs)
        with tok.as_target_tokenizer():
            decoded = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for k in range(len(batch)):
            unique = []
            for text in decoded[k * n:(k + 1) * n]:
                text = _tidy(text)
                if text and text not in unique:
                    unique.append(text)
            out.append(unique or [""])
    json.dump({"candidates": out, "seconds": round(time.perf_counter() - started, 1)},
              open(result_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
