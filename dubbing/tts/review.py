"""A local review page for generated speech: listen to every dubbed line next to the original
(opened from your PC; never published, since it plays the original video and cloned voices)."""

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
main { max-width:1300px; margin:0 auto; padding:16px } h1 { font-size:22px; margin:8px 0 4px }
.muted { color:var(--muted); font-size:13px } .stats { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 }
.stat { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px }
.top { display:grid; grid-template-columns:minmax(0,460px) minmax(0,1fr); gap:16px; align-items:start }
@media (max-width:860px) { .top { grid-template-columns:1fr } }
video { width:100%; border-radius:8px; background:#000 } audio { width:220px; height:32px }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }
th, td { text-align:left; vertical-align:top; padding:7px 8px; border-bottom:1px solid var(--line) }
th { position:sticky; top:0; background:var(--card); font-size:13px; color:var(--muted) }
td.t { white-space:nowrap; font-variant-numeric:tabular-nums } td.t a { color:var(--accent); cursor:pointer }
.hi { font-size:17px } .flag { display:inline-block; font-size:12px; background:var(--warn-bg); color:var(--warn);
  border-radius:4px; padding:0 5px; margin:1px 2px 1px 0 } .ok { color:var(--ok) } .bad { color:var(--warn) }
.tbl { overflow-x:auto }
"""

FLAG_HELP = {
    "low_intelligibility": "Whisper could not understand the dub well: listen",
    "overruns_slot": "longer than the time before the next line (the mix will tighten it)",
    "clarity_fallback_builtin": "the cloned voice was unclear, so a built-in voice was used",
    "no_clean_voice_sample": "this speaker had no clean sample, so a built-in voice of the right gender",
    "speaker_uncertain": "speaker label uncertain: the voice may be the wrong person's",
    "shorter_wording": "a shorter wording was used so the line fits",
    "tail_cut": "extra sounds after the words were cut off: check the ending",
    "flatter_than_original": "less expressive than the original",
}


def _t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def _rel(path: str, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def write_speech_review(manifest: dict[str, Any], out_path: Path, video_path: Path) -> Path:
    base = out_path.parent
    rows = []
    for u in manifest["units"]:
        if u.get("status") != "generated":
            continue
        g = {"male": "♂", "female": "♀"}.get(u.get("speaker_gender") or "", "?")
        fits = u["duration_s"] <= u["available_s"] + 0.05
        heard = (u.get("back_check") or {}).get("heard", "")
        cer = (u.get("back_check") or {}).get("char_error_rate")
        flags = "".join(f'<span class="flag" title="{html.escape(FLAG_HELP.get(f, f))}">{html.escape(f)}</span>'
                        for f in u.get("flags", []) if f not in ("retried",))
        rows.append(
            f'<tr><td class="t"><a data-t="{u["start"]}">{_t(u["start"])}</a></td>'
            f'<td>{html.escape(str(u.get("speaker")))} {g}<br><span class="muted">{html.escape(u.get("voice", ""))}</span></td>'
            f'<td><span class="muted">{html.escape(u.get("source_text") or "")}</span>'
            f'<div class="hi">{html.escape(u["text"])}</div>'
            f'<div class="muted">Whisper heard: “{html.escape(heard)}”'
            f'{f" · {100 * (1 - min(cer, 1)):.0f}% match" if cer is not None else ""}</div></td>'
            f'<td><audio controls preload="none" src="{html.escape(_rel(u["path"], base))}"></audio></td>'
            f'<td class="{"ok" if fits else "bad"}">{u["duration_s"]:.2f} s<br>'
            f'<span class="muted">of {u["available_s"]:.2f} s</span></td><td>{flags}</td></tr>')
    s = manifest["summary"]
    shown = {"lines": s.get("generated"), "total speech (s)": s.get("total_speech_s"),
             "median Whisper error": s.get("median_char_error_rate"),
             **{k: v for k, v in (s.get("flag_counts") or {}).items() if k in FLAG_HELP}}
    stats = "".join(f'<div class="stat"><span class="muted">{html.escape(str(k))}</span><br><b>{html.escape(str(v))}</b></div>'
                    for k, v in shown.items())
    preview = manifest.get("speech_preview")
    preview_html = (f'<div><b>All dubbed lines on the timeline</b> (speech only, no music yet; '
                    f'overlaps are fixed in the mix):<br><audio controls preload="none" style="width:100%" '
                    f'src="{html.escape(_rel(preview, base))}"></audio></div>') if preview else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Speech Review</title>
<style>{_CSS}</style></head><body><main>
<h1>Speech review: {html.escape(manifest["job_id"])} → {html.escape(manifest["language"])}</h1>
<div class="muted">AI-generated speech (local review only). Click a time to see the original moment; press ▶ to hear the Hindi line.</div>
<div class="stats">{stats}</div>
<div class="top"><video id="v" controls preload="metadata" src="{html.escape(video_path.resolve().as_uri())}"></video>
<div>{preview_html}<p class="muted"><b>What to listen for:</b> is it the same person's voice as the original? Does the
emotion match? Is every word clear? Green time = fits before the next line; orange = longer (the mix step tightens it).</p></div></div>
<h2 style="font-size:17px">Lines</h2><div class="tbl"><table><thead><tr><th>Time</th><th>Speaker / voice</th>
<th>English → Hindi</th><th>Listen</th><th>Length</th><th>Flags</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
</main><script>
const v = document.getElementById('v');
document.querySelectorAll('a[data-t]').forEach(a => a.addEventListener('click', () => {{
  v.currentTime = Math.max(0, parseFloat(a.dataset.t) - 0.3); v.play(); }}));
</script></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
