"""A local review page for a translation (opened from your PC; plays the original video)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6b6862; --line:#e4e1da; --card:#fff; --warn:#b4540a;
        --warn-bg:#fdf0e3; --accent:#2f5d8a; --ok:#2e7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#ecebe8; --muted:#a19e97; --line:#2c2d30;
        --card:#1f2023; --warn:#f2a65a; --warn-bg:#3a2a1b; --accent:#8ab4e0; --ok:#7bc47f; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif }
main { max-width:1300px; margin:0 auto; padding:16px } h1 { font-size:22px; margin:8px 0 4px }
.muted { color:var(--muted); font-size:13px } .stats { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 }
.stat { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px }
.top { display:grid; grid-template-columns:minmax(0,460px) minmax(0,1fr); gap:16px; align-items:start }
@media (max-width:860px) { .top { grid-template-columns:1fr } }
video { width:100%; border-radius:8px; background:#000; position:sticky; top:8px }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }
th, td { text-align:left; vertical-align:top; padding:7px 8px; border-bottom:1px solid var(--line) }
th { position:sticky; top:0; background:var(--card); font-size:13px; color:var(--muted) }
td.t { white-space:nowrap; font-variant-numeric:tabular-nums } td.t a { color:var(--accent); cursor:pointer }
.hi { font-size:18px } .flag { display:inline-block; font-size:12px; background:var(--warn-bg); color:var(--warn);
  border-radius:4px; padding:0 5px; margin:1px 2px 1px 0 } .fit-ok { color:var(--ok) } .fit-bad { color:var(--warn) }
details summary { cursor:pointer; color:var(--muted); font-size:13px } ul { margin:4px 0; padding-left:18px; font-size:13px }
.tbl { overflow-x:auto }
"""

FLAG_HELP = {
    "may_not_fit": "Hindi may be too long for the time the actor took",
    "check_meaning": "back-translation is far from the English: check the meaning",
    "speaker_uncertain": "speaker (so gender) may be wrong",
    "listener_maybe_female": "probably said to a woman, but not sure: check the 'you' verb forms",
    "english_unsure": "Whisper was unsure of some English words",
    "wrong_script": "words not in Hindi script", "repetition": "a word repeats", "unusual_length": "odd length",
}


def _t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def write_translation_review(result: dict[str, Any], out_path: Path, video_path: Path) -> Path:
    rows = []
    for u in result["segments"]:
        g = {"male": "♂", "female": "♀"}.get(u.get("speaker_gender") or "", "?")
        to = ", ".join("a group" if a == "group" else a for a in (u.get("listener") or []))
        lg = {"male": "♂", "female": "♀"}.get(u.get("listener_gender") or "", "")
        listener_html = (f'<br><span class="muted">→ {html.escape(to)} {lg} ({html.escape(str(u.get("listener_confidence")))})</span>'
                         if to else "")
        fit = (f'<span class="{"fit-bad" if "may_not_fit" in u["flags"] else "fit-ok"}">'
               f'≈{u["est_speech_s"]} s of {u["available_s"]} s</span>')
        flags = "".join(f'<span class="flag" title="{html.escape(FLAG_HELP.get(f, f))}">{html.escape(f)}</span>'
                        for f in u["flags"])
        changes = "".join(f"<li>{html.escape(c)}</li>" for c in u["changes"])
        options = "".join(
            f'<li>{html.escape(o["text"])} <span class="muted">“{html.escape(o["back_translation"])}” · '
            f'meaning {o["meaning"]:.0f} · ≈{o["est_s"]} s · score {o["score"]}</span></li>'
            for o in u["options"])
        rows.append(
            f'<tr><td class="t"><a data-t="{u["start"]}">{_t(u["start"])}</a><br>'
            f'<span class="muted">{u["end"] - u["start"]:.2f} s</span></td>'
            f'<td>{html.escape(str(u.get("speaker")))} {g}{listener_html}</td>'
            f'<td>{html.escape(u["source_text"])}</td>'
            f'<td><div class="hi">{html.escape(u["text"])}</div>'
            f'<div class="muted">back to English: “{html.escape(u["back_translation"])}” · meaning {u["meaning_score"]:.0f}/100</div>'
            f'{"<details><summary>spoken-Hindi changes (" + str(len(u["changes"])) + ")</summary><ul>" + changes + "</ul></details>" if changes else ""}'
            f'<details><summary>all {len(u["options"])} wordings</summary><ul>{options}</ul></details></td>'
            f'<td>{fit}</td><td>{flags}</td></tr>')
    s = result["summary"]
    stats = "".join(f'<div class="stat"><span class="muted">{html.escape(k)}</span><br><b>{html.escape(str(v))}</b></div>'
                    for k, v in s.items())
    tr = result["translator"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Translation Review</title>
<style>{_CSS}</style></head><body><main>
<h1>Translation review: {html.escape(result["job_id"])} → {html.escape(result["language"])}</h1>
<div class="muted">Translator: {html.escape(tr.get("model", tr.get("engine", "")))} · register: {html.escape(result["register"])}
 · meaning checked by translating back with NLLB · click a time to play that moment.</div>
<div class="stats">{stats}</div>
<div class="top"><video id="v" controls preload="metadata" src="{html.escape(video_path.resolve().as_uri())}"></video>
<div class="muted"><b>How to read a row:</b> big text = the Hindi that will be spoken. “Back to English” is what that Hindi means
when translated back; it should match the English. The time column compares how long the Hindi needs with the time
the actor took (green fits, orange may be too long). Open “all wordings” to see the alternatives and why this one won.</div></div>
<h2 style="font-size:17px">Lines</h2><div class="tbl"><table><thead><tr><th>Time</th><th>Speaker</th><th>English</th>
<th>Hindi</th><th>Time needed</th><th>Flags</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
</main><script>
const v = document.getElementById('v');
document.querySelectorAll('a[data-t]').forEach(a => a.addEventListener('click', () => {{
  v.currentTime = Math.max(0, parseFloat(a.dataset.t) - 0.3); v.play(); }}));
</script></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
