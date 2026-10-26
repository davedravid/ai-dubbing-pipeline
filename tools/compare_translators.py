"""Compare translators on chosen lines of a dubbing script, with the spoken-Hindi layer.

Usage (from the project folder):
  python tools\\compare_translators.py job_003 0,1,3,5,7,10
Writes data/translations/<job>_compare.html and .json.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dubbing.config import load_settings  # noqa: E402
from dubbing.errors import PipelineError  # noqa: E402
from dubbing.paths import JobPaths  # noqa: E402
from dubbing.qa.text_metrics import chrf  # noqa: E402
from dubbing.script.glossary import load_name_spellings, load_name_variants  # noqa: E402
from dubbing.translator.spoken import formality_score, make_spoken, protect_names  # noqa: E402
from dubbing.translator.units import build_units  # noqa: E402

HINDI_CHARS_PER_S = 13.0
N_OPTIONS = 6


def pick(sources, candidates, backs, slots, paces):
    """Best option per sentence: keeps meaning, fits the slot, not formal."""
    chosen = []
    for k, cands in enumerate(candidates):
        best = None
        for text in cands:
            meaning = chrf(sources[k], backs[text])
            est = len(text) / (HINDI_CHARS_PER_S * paces[k])
            over = max(0.0, est / max(slots[k], 0.3) - 1.25)
            latin = len(re.findall(r"[A-Za-z]{2,}", text))  # English words left in a Hindi line
            score = meaning - 5 * formality_score(text) - 30 * over - 25 * latin
            if best is None or score > best["score"]:
                best = {"text": text, "score": round(score, 1), "meaning": meaning, "est_s": round(est, 2),
                        "first_option": text == cands[0]}
        chosen.append(best)
    return chosen


def main(job_id: str, line_ids: list[int]) -> None:
    settings = load_settings()
    paths = JobPaths(settings.data_dir, job_id)
    script = json.loads(paths.script.read_text(encoding="utf-8"))
    names = load_name_spellings(paths.glossary)
    lines = script["lines"]
    units = build_units(lines)
    picked = [u for u in units if set(u["source_segment_ids"]) & set(line_ids)]
    by_id = {l["id"]: l for l in lines}
    sources = [u["text"] for u in picked]
    gender = [script["speakers"].get(u.get("speaker") or "", {}).get("gender") for u in picked]
    starts = [u["start"] for u in units]
    slots, paces = [], []
    for u in picked:
        nxt = next((s for s in starts if s > u["start"] + 0.01), script["source"]["duration_s"])
        slots.append(max(u["end"], nxt) - u["start"])
        wps = sum(len(by_id[i]["words"]) for i in u["source_segment_ids"]) / max(u["end"] - u["start"], 0.3)
        paces.append(min(max(wps / 2.8, 0.7), 1.8))

    from dubbing.translator.nllb import NllbTranslator

    nllb = NllbTranslator(settings.translation, "cpu")
    engines = {"A: NLLB-200 600M": nllb}
    try:
        from dubbing.translator.indictrans import IndicTransTranslator

        engines["B: IndicTrans2 200M"] = IndicTransTranslator()
    except PipelineError as exc:
        print(f"IndicTrans2 not available: {exc}")

    variants = load_name_variants(paths.glossary)
    to_translate = [protect_names(s, names) for s in sources]  # S.H.I.E.L.D. -> SHIELD
    results = {}
    for name, eng in engines.items():
        cands = eng.translate_candidates(to_translate, "eng_Latn", "hin_Deva", N_OPTIONS)
        flat = sorted({t for c in cands for t in c})
        backs = dict(zip(flat, nllb.translate(flat, "hin_Deva", "eng_Latn")))  # same referee for both
        chosen = pick(sources, cands, backs, slots, paces)
        rows = []
        for k, ch in enumerate(chosen):
            spoken = make_spoken(ch["text"], gender[k], "tum", names, english=sources[k], variants=variants)
            rows.append({**ch, "spoken": spoken.text, "changes": spoken.changes, "back": backs[ch["text"]],
                         "options": cands[k]})
        results[name] = rows
        print(f"{name}: done")

    out = {"job_id": job_id, "lines": [{"unit_ids": u["source_segment_ids"], "start": u["start"],
                                         "speaker": u.get("speaker"), "gender": gender[k], "english": sources[k],
                                         "slot_s": round(slots[k], 2)} for k, u in enumerate(picked)],
           "results": results}
    json_path = paths.translations_dir / f"{job_id}_compare.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_page(out, paths.translations_dir / f"{job_id}_compare.html")
    print(f"Wrote {json_path} and the comparison page.")


def write_page(out, path: Path) -> None:
    names = list(out["results"])
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    rows = []
    for k, line in enumerate(out["lines"]):
        cells = []
        for n in names:
            r = out["results"][n][k]
            changes = "".join(f"<li>{html.escape(c)}</li>" for c in r["changes"])
            cells.append(
                f'<td><div class="hi">{html.escape(r["spoken"])}</div>'
                f'<div class="muted">translator wrote: {html.escape(r["text"])}</div>'
                f'{"<ul class=ch>" + changes + "</ul>" if changes else ""}'
                f'<div class="muted">back to English: “{html.escape(r["back"])}” · meaning {r["meaning"]:.0f}/100'
                f' · needs ≈{r["est_s"]} s of {line["slot_s"]} s</div></td>')
        g = {"male": "♂", "female": "♀"}.get(line["gender"] or "", "?")
        rows.append(f'<tr><td class="t">{int(line["start"] // 60)}:{line["start"] % 60:05.2f}<br>'
                    f'{html.escape(str(line["speaker"]))} {g}</td><td class="en">{html.escape(line["english"])}</td>'
                    f'{"".join(cells)}</tr>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Translator Comparison</title><style>
:root {{ --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6b6862; --line:#e4e1da; --card:#fff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --fg:#ecebe8; --muted:#a19e97; --line:#2c2d30; --card:#1f2023; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,Segoe UI,sans-serif }}
main {{ max-width:1300px; margin:0 auto; padding:16px }} .muted {{ color:var(--muted); font-size:13px }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }}
th, td {{ text-align:left; vertical-align:top; padding:8px; border-bottom:1px solid var(--line) }}
th {{ font-size:13px; color:var(--muted) }} .hi {{ font-size:18px; margin-bottom:4px }} .en {{ max-width:260px }}
.t {{ white-space:nowrap; font-variant-numeric:tabular-nums }} ul.ch {{ margin:2px 0; padding-left:18px; font-size:12px; color:var(--muted) }}
.wrap {{ overflow-x:auto }}</style></head><body><main>
<h1>Translator comparison: {html.escape(out["job_id"])}</h1>
<p class="muted">Both columns use the same spoken-Hindi layer (everyday words, female verb forms, तुम, names in Hindi letters).
Big text = what would be spoken. "Meaning" = how close the back-translation is to the English (0-100; same referee for both).</p>
<div class="wrap"><table><thead><tr><th>Time / speaker</th><th>English</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></main></body></html>"""
    path.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")])
