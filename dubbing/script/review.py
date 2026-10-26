"""A local review page for the dubbing script (opened from your PC, never published:
it plays the original video, which is copyrighted)."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

LOW_CONFIDENCE = 0.5

_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6b6862; --line:#e4e1da; --card:#ffffff;
        --warn:#b4540a; --warn-bg:#fdf0e3; --accent:#2f5d8a; --event:#f1efe9; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#ecebe8; --muted:#a19e97; --line:#2c2d30;
        --card:#1f2023; --warn:#f2a65a; --warn-bg:#3a2a1b; --accent:#8ab4e0; --event:#24252a; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif }
main { max-width:1180px; margin:0 auto; padding:16px }
h1 { font-size:22px; margin:8px 0 4px } h2 { font-size:17px; margin:22px 0 8px }
.muted { color:var(--muted) } .top { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px }
@media (max-width:800px) { .top { grid-template-columns:1fr } }
video { width:100%; border-radius:8px; background:#000; position:sticky; top:8px }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px }
.card b { font-size:16px } .card audio { width:100%; margin-top:6px }
.grade-good { color:#2e7d32 } .grade-ok { color:var(--accent) } .grade-weak { color:var(--warn) }
.stats { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 }
.stat { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }
th, td { text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line) }
th { position:sticky; top:0; background:var(--card); font-size:13px; color:var(--muted) }
td.t { white-space:nowrap; font-variant-numeric:tabular-nums } td.t a { color:var(--accent); cursor:pointer }
tr.event td { background:var(--event); color:var(--muted); font-style:italic }
.low { background:var(--warn-bg); color:var(--warn); border-radius:3px; padding:0 2px }
.flag { display:inline-block; font-size:12px; background:var(--warn-bg); color:var(--warn);
        border-radius:4px; padding:0 5px; margin:1px 2px 1px 0 }
.pause { color:var(--muted) } .spk { font-weight:600; white-space:nowrap } .tbl { overflow-x:auto }
"""


def _t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def _text_html(line: dict[str, Any]) -> str:
    words = line.get("words") or []
    if not words:
        return html.escape(line["text"])
    stressed = {w.lower() for w in line["delivery"].get("emphasis", [])}
    parts = []
    for i, w in enumerate(words):
        word = html.escape(w["word"])
        if w["word"].strip(".,!?;:'\"").lower() in stressed:
            word = f"<b>{word}</b>"
        if (w.get("probability") or 1.0) < LOW_CONFIDENCE:
            word = f'<span class="low" title="Whisper is unsure ({w["probability"]:.2f})">{word}</span>'
        parts.append(word)
        if i + 1 < len(words) and words[i + 1]["start"] - w["end"] >= 0.25:
            parts.append('<span class="pause" title="pause">‖</span>')
    return " ".join(parts)


def write_review(script: dict[str, Any], out_path: Path, video_path: Path) -> Path:
    speakers = script["speakers"]
    name = lambda s: (speakers.get(s or "", {}).get("name") or s or "?")
    rows = [(l["start"], "line", l) for l in script["lines"]] + [(e["start"], "event", e) for e in script["events"]]
    body = []
    for start, kind, x in sorted(rows, key=lambda r: r[0]):
        if kind == "event":
            label = {"overlapping_speech": "two or more people talking at once",
                     "untranscribed_speech": f"speech not in the script: “{x.get('heard', '')}”"}.get(
                x["type"], f"[{x['type']}]" + (f" heard as “{x['heard']}”" if x.get("heard") else ""))
            body.append(f'<tr class="event"><td class="t"><a data-t="{x["start"]}">{_t(x["start"])}</a></td>'
                        f'<td></td><td>{html.escape(label)} · {x["end"] - x["start"]:.2f} s</td><td></td><td></td></tr>')
            continue
        d = x["delivery"]
        notes = [d.get("style", ""), d.get("kind", "")]
        if d.get("words_per_s"):
            notes.append(f'{d["words_per_s"]} w/s')
        if d.get("pitch_median_hz"):
            notes.append(f'{d["pitch_median_hz"]:.0f} Hz')
        flags = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in x.get("flags", []))
        spk = speakers.get(x.get("speaker") or "", {})
        g = {"male": "♂", "female": "♀"}.get(spk.get("gender", ""), "?")
        body.append(
            f'<tr><td class="t"><a data-t="{x["start"]}">{_t(x["start"])}</a><br>'
            f'<span class="muted">{x["end"] - x["start"]:.2f} s</span></td>'
            f'<td class="spk">{html.escape(name(x.get("speaker")))} {g}'
            f'{"<br><span class=muted>uncertain</span>" if x.get("speaker_uncertain") else ""}</td>'
            f'<td>{_text_html(x)}</td><td class="muted">{html.escape(" · ".join(n for n in notes if n))}</td>'
            f'<td>{flags}</td></tr>')

    cards = []
    for spk, p in speakers.items():
        c = script["clone_samples"].get(spk, {})
        audio = ""
        if c.get("preview"):
            rel = os.path.relpath(c["preview"], out_path.parent).replace(os.sep, "/")
            audio = f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'
        flags = "".join(f'<span class="flag">{f}</span>' for f in p.get("flags", []))
        cards.append(
            f'<div class="card"><b>{html.escape(p.get("name") or spk)}</b> <span class="muted">{spk}</span><br>'
            f'{p["gender"]} <span class="muted">({p["gender_source"]}, {p.get("median_pitch_hz") or "?"} Hz)</span><br>'
            f'{p["lines"]} lines · {p["speech_s"]} s speech · {p["uncertain_lines"]} uncertain<br>'
            f'clone samples: <span class="grade-{c.get("grade", "weak")}">{c.get("grade", "none")}</span> '
            f'<span class="muted">({c.get("clean_seconds", 0)} s clean)</span> {flags}{audio}</div>')

    s = script["summary"]
    stats = "".join(f'<div class="stat"><span class="muted">{html.escape(k)}</span><br><b>{html.escape(str(v))}</b></div>'
                    for k, v in s.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dubbing Script Review</title>
<style>{_CSS}</style></head><body><main>
<h1>Dubbing script review: {html.escape(script["job_id"])}</h1>
<div class="muted">Click a time to play that moment. Orange = Whisper unsure · <b>bold</b> = stressed word · ‖ = pause.
Correct speakers in <code>{html.escape(str(script["speaker_corrections_file"]))}</code>.</div>
<div class="stats">{stats}</div>
<div class="top"><div><video id="v" controls preload="metadata" src="{html.escape(video_path.resolve().as_uri())}"></video></div>
<div><h2 style="margin-top:0">Speakers</h2><div class="cards">{"".join(cards)}</div></div></div>
<h2>Lines and sounds</h2><div class="tbl"><table><thead><tr><th>Time</th><th>Speaker</th><th>Text</th>
<th>Delivery</th><th>Flags</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
</main><script>
const v = document.getElementById('v');
document.querySelectorAll('a[data-t]').forEach(a => a.addEventListener('click', () => {{
  v.currentTime = Math.max(0, parseFloat(a.dataset.t) - 0.3); v.play(); }}));
</script></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
