"""Local review page for the story bible (Phase A2)."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6b6862; --line:#e4e1da; --card:#fff; --warn:#b4540a;
        --warn-bg:#fdf0e3; --accent:#2f5d8a; --ok:#2e7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#ecebe8; --muted:#a19e97; --line:#2c2d30;
        --card:#1f2023; --warn:#f2a65a; --warn-bg:#3a2a1b; --accent:#8ab4e0; --ok:#7bc47f; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif }
main { max-width:1250px; margin:0 auto; padding:16px } h1 { font-size:22px; margin:8px 0 4px } h2 { font-size:17px; margin:22px 0 8px }
.muted { color:var(--muted); font-size:13px } .stats { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 }
.stat { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px } .card audio { width:100%; margin-top:6px }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }
th, td { text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line) }
th { position:sticky; top:0; background:var(--card); font-size:13px; color:var(--muted) }
td.t { white-space:nowrap; font-variant-numeric:tabular-nums } td.t a { color:var(--accent); cursor:pointer }
.flag { display:inline-block; font-size:12px; background:var(--warn-bg); color:var(--warn); border-radius:4px; padding:0 5px }
.ok { color:var(--ok) } .top { display:grid; grid-template-columns:minmax(0,420px) minmax(0,1fr); gap:16px; align-items:start }
@media (max-width:860px) { .top { grid-template-columns:1fr } } video { width:100%; border-radius:8px; background:#000 }
.tbl { overflow-x:auto } code { font-size:13px }
"""
G = {"male": "♂", "female": "♀"}


def _t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def _name(chars: dict[str, Any], spk: str) -> str:
    c = chars.get(spk) or {}
    n = c.get("final_name") or c.get("suggested_name")
    return f"{spk}" + (f" ({n})" if n else "")


def write_story_review(story: dict[str, Any], script: dict[str, Any], out_path: Path, video: Path) -> Path:
    chars = story["characters"]
    base = out_path.parent
    cards = []
    for spk, c in chars.items():
        sample = (script.get("clone_samples", {}).get(spk) or {}).get("preview")
        audio = (f'<audio controls preload="none" src="{html.escape(os.path.relpath(sample, base).replace(os.sep, "/"))}"></audio>'
                 if sample else "")
        name = c.get("final_name") or c.get("suggested_name") or "—"
        src = "your name" if c.get("final_name") else (
            f'rules: {c.get("rules_say") or "—"} · AI: {c.get("ai_says") or "—"}')
        badge = "" if c.get("sources_agree") is not False else ' <span class="flag">rules and AI disagree</span>'
        cards.append(f'<div class="card"><b>{html.escape(spk)}: {html.escape(name)}</b> {G.get(c.get("gender"), "?")}{badge}<br>'
                     f'<span class="muted">{html.escape(src)}<br>{html.escape(c.get("ai_evidence") or "")}</span>{audio}</div>')
    pair_rows = "".join(
        f'<tr><td>{html.escape(_name(chars, k.split(">")[0]))} → {html.escape(_name(chars, k.split(">")[1]))}</td>'
        f'<td>{v["register"]}</td><td class="muted">{html.escape(v.get("relationship", ""))}</td></tr>'
        for k, v in story["pairs"].items() if k != "_default")
    scenes = "".join(f'<li><b>lines {s.get("from")}–{s.get("to")}:</b> {html.escape(s.get("summary", ""))} '
                     f'<span class="muted">tone: {html.escape(s.get("tone", ""))}</span></li>' for s in story["scenes"])
    rows = []
    for lid, v in story["lines"].items():
        to = ", ".join(_name(chars, a) if a != "group" else "a group" for a in v["addressee"]) or "unclear"
        agree = ('<span class="ok">agree</span>' if v["sources_agree"] else
                 '<span class="flag">disagree</span>' if v["sources_agree"] is False else "")
        rows.append(f'<tr><td class="t"><a data-t="{v["start"]}">{_t(v["start"])}</a></td>'
                    f'<td>{html.escape(_name(chars, v["speaker"] or "?"))} {G.get(v.get("speaker_gender"), "?")}</td>'
                    f'<td>→ {html.escape(to)} <span class="muted">({html.escape(str(v["addressee_gender"]))})</span></td>'
                    f'<td>{html.escape(v["text"])}</td><td>{v["confidence"]} {agree}<br>'
                    f'<span class="muted">{html.escape(v["reason"])}</span></td></tr>')
    corr_rows = "".join(
        f'<tr><td>{lid}</td><td>{html.escape(c["original"])}</td><td><b>{html.escape(c["suggested"])}</b></td>'
        f'<td class="muted">{html.escape(c["reason"])}</td>'
        f'<td>{"accepted" if c["accept"] else "rejected" if c["accept"] is False else "<span class=flag>waiting for you</span>"}</td></tr>'
        for lid, c in story["corrections"].items())
    stats = "".join(f'<div class="stat"><span class="muted">{html.escape(k)}</span><br><b>{v}</b></div>'
                    for k, v in story["summary"].items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Story Bible Review</title>
<style>{_CSS}</style></head><body><main>
<h1>Story bible: {html.escape(story["job_id"])}</h1>
<div class="muted">Read by rules + a local AI model ({html.escape(story["ai_model"])}). Correct names in
<code>speakers/{html.escape(story["job_id"])}.yaml</code>; accept or reject English corrections in
<code>{html.escape(story["corrections_file"])}</code>.</div>
<div class="stats">{stats}</div>
<div class="top"><video id="v" controls preload="metadata" src="{html.escape(video.resolve().as_uri())}"></video>
<div><h2 style="margin-top:0">Scene</h2><ul>{scenes or "<li>—</li>"}</ul></div></div>
<h2>Characters</h2><div class="cards">{"".join(cards)}</div>
<h2>Formality between people</h2><div class="tbl"><table><thead><tr><th>Who → to whom</th><th>Register</th><th>Relationship</th></tr></thead>
<tbody>{pair_rows or '<tr><td colspan=3 class=muted>none found</td></tr>'}<tr><td class="muted">anyone else</td>
<td>{story["pairs"]["_default"]["register"]}</td><td class="muted">default</td></tr></tbody></table></div>
<h2>Suggested English corrections</h2><div class="tbl"><table><thead><tr><th>Line</th><th>Now</th><th>Suggested</th><th>Why</th><th>Decision</th></tr></thead>
<tbody>{corr_rows or '<tr><td colspan=5 class=muted>none</td></tr>'}</tbody></table></div>
<h2>Who talks to whom</h2><div class="tbl"><table><thead><tr><th>Time</th><th>Speaker</th><th>Talking to</th><th>Line</th>
<th>Confidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
</main><script>
const v = document.getElementById('v');
document.querySelectorAll('a[data-t]').forEach(a => a.addEventListener('click', () => {{
  v.currentTime = Math.max(0, parseFloat(a.dataset.t) - 0.3); v.play(); }}));
</script></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
