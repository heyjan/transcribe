"""Render transcript JSON into per-language HTML transcripts."""

from __future__ import annotations

import html
import json
from pathlib import Path

def ui_for(lang: str) -> dict:
    """Chrome labels; languages we have no wording for borrow the English set."""
    return UI.get(lang, UI["en"])


UI = {
    "en": {
        "transcript": "Transcript",
        "blocks": "blocks",
        "duration": "Duration",
        "model": "Model",
        "generated": "Generated",
        "source": "Source",
        "flag": {
            "repetition": "possible repetition loop — check the audio",
            "untranslated": "left untranslated — this is the source text",
            "unpunctuated": "no punctuation — the model skipped it here",
        },
        "empty": "[no speech detected]",
        "index": "All transcripts",
        "lang_label": "English",
        "toc": "Contents",
    },
    "de": {
        "transcript": "Transkript",
        "blocks": "Blöcke",
        "duration": "Dauer",
        "model": "Modell",
        "generated": "Erstellt",
        "source": "Quelle",
        "flag": {
            "repetition": "mögliche Wiederholungsschleife — Audio prüfen",
            "untranslated": "nicht übersetzt — dies ist der Originaltext",
            "unpunctuated": "ohne Satzzeichen — vom Modell hier ausgelassen",
        },
        "empty": "[keine Sprache erkannt]",
        "index": "Alle Transkripte",
        "lang_label": "Deutsch",
        "toc": "Inhalt",
    },
}

CSS = """
:root {
  --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862; --rule: #e3ded6;
  --card: #ffffff; --accent: #7a5c2e; --chip: #f0ebe2; --flag: #9a5b1a;
  --flag-bg: #fdf3e6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16151a; --fg: #e8e5df; --muted: #9a958c; --rule: #2e2c33;
    --card: #1d1c22; --accent: #d7b57a; --chip: #26242b; --flag: #e0a95f;
    --flag-bg: #2a2119;
  }
}
:root[data-theme="dark"] {
  --bg: #16151a; --fg: #e8e5df; --muted: #9a958c; --rule: #2e2c33;
  --card: #1d1c22; --accent: #d7b57a; --chip: #26242b; --flag: #e0a95f;
  --flag-bg: #2a2119;
}
:root[data-theme="light"] {
  --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862; --rule: #e3ded6;
  --card: #ffffff; --accent: #7a5c2e; --chip: #f0ebe2; --flag: #9a5b1a;
  --flag-bg: #fdf3e6;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.wrap { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }
header { border-bottom: 1px solid var(--rule); padding-bottom: 1.5rem; margin-bottom: .5rem; }
.eyebrow {
  font-size: .75rem; letter-spacing: .09em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 .5rem;
}
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .9rem; font-weight: 620; }
.meta { display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; font-size: .8rem; color: var(--muted); }
.meta b { font-weight: 550; color: var(--fg); }
.nav { margin: 1.1rem 0 0; display: flex; gap: .6rem; flex-wrap: wrap; }
.nav a {
  font-size: .8rem; text-decoration: none; color: var(--fg);
  border: 1px solid var(--rule); border-radius: 999px; padding: .3rem .8rem;
  background: var(--card);
}
.nav a:hover { border-color: var(--accent); color: var(--accent); }
.player { position: sticky; top: 0; z-index: 5; padding: .9rem 0; background: var(--bg); }
.player audio { width: 100%; height: 34px; }
.block {
  display: grid; grid-template-columns: 4.6rem 1fr; gap: 1rem;
  padding: .85rem 0; border-bottom: 1px solid var(--rule); scroll-margin-top: 5rem;
}
.block:last-child { border-bottom: 0; }
.t {
  appearance: none; border: 0; cursor: pointer; align-self: start;
  font: 500 .74rem/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--muted); background: var(--chip); border-radius: 5px;
  padding: .18rem .42rem; margin-top: .18rem; text-align: center;
}
.t:hover { color: var(--accent); }
.block.playing { background: color-mix(in srgb, var(--accent) 8%, transparent); }
.block.playing .t { color: var(--accent); }
p.s { margin: 0; }
.block.flagged p.s {
  border-left: 3px solid var(--flag); background: var(--flag-bg);
  padding: .5rem .75rem; border-radius: 0 5px 5px 0;
}
.warn { font-size: .74rem; color: var(--flag); margin: .4rem 0 0; }
.empty { color: var(--muted); font-style: italic; }
footer { margin-top: 2.5rem; font-size: .76rem; color: var(--muted); }
footer a { color: inherit; }
ul.idx { list-style: none; padding: 0; margin: 1.5rem 0 0; }
ul.idx li { border-bottom: 1px solid var(--rule); padding: .8rem 0; }
ul.idx a { color: var(--fg); text-decoration: none; font-weight: 550; }
ul.idx a:hover { color: var(--accent); }
ul.idx .sub { font-size: .78rem; color: var(--muted); margin-top: .2rem; }
@media (max-width: 34rem) {
  .block { grid-template-columns: 1fr; gap: .3rem; }
  .t { justify-self: start; }
}
@media print {
  .player, .nav { display: none; }
  body { background: #fff; color: #000; }
  .block { break-inside: avoid; }
}
"""

SEEK_JS = """
(function () {
  var audio = document.getElementById('player');
  var blocks = Array.prototype.slice.call(document.querySelectorAll('.block'));
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.t');
    if (!btn || !audio) return;
    audio.currentTime = parseFloat(btn.dataset.start);
    audio.play();
  });
  if (!audio) return;
  var current = null;
  audio.addEventListener('timeupdate', function () {
    var t = audio.currentTime;
    var hit = null;
    for (var i = 0; i < blocks.length; i++) {
      if (t >= parseFloat(blocks[i].dataset.start) && t < parseFloat(blocks[i].dataset.end)) {
        hit = blocks[i];
        break;
      }
    }
    if (hit === current) return;
    if (current) current.classList.remove('playing');
    if (hit) hit.classList.add('playing');
    current = hit;
  });
})();
"""


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s"


def _page(title: str, lang: str, body: str, script: str = "") -> str:
    tail = f"<script>{script}</script>" if script else ""
    return (
        f'<!doctype html>\n<html lang="{html.escape(lang)}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n"
        f'<body>\n<div class="wrap">\n{body}\n</div>\n{tail}\n</body>\n</html>\n'
    )


def fmt_srt(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def render_srt(doc: dict, lang: str) -> str:
    """One cue per block; silent blocks are left out rather than shown empty."""
    cues = []
    for b in doc["blocks"]:
        text = (b["text"].get(lang) or "").strip()
        if text:
            cues.append(f"{len(cues) + 1}\n{fmt_srt(b['start'])} --> {fmt_srt(b['end'])}\n{text}")
    return "\n\n".join(cues) + "\n"


def render_txt(doc: dict, lang: str) -> str:
    texts = [(b["text"].get(lang) or "").strip() for b in doc["blocks"]]
    return "\n\n".join(t for t in texts if t) + "\n"


def render_transcript(doc: dict, lang: str, sibling: tuple[str, str] | None,
                      index_href: str | None) -> str:
    ui = ui_for(lang)
    src = doc["source"]
    title = src["title"]
    # Languages can come from different models (e.g. Granite ASR + Voxtral
    # translation), so credit whichever one actually produced this page.
    model_name = doc["model"].get("providers", {}).get(lang, doc["model"]["name"])
    parts: list[str] = []

    parts.append("<header>")
    if src.get("module"):
        parts.append(f'<p class="eyebrow">{html.escape(src["module"])}</p>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        '<div class="meta">'
        f'<span>{ui["duration"]} <b>{fmt_duration(src["duration"])}</b></span>'
        f'<span>{len(doc["blocks"])} {ui["blocks"]}</span>'
        f'<span>{ui["model"]} <b>{html.escape(model_name)}</b></span>'
        f'<span>{ui["generated"]} <b>{doc["generated_at"][:10]}</b></span>'
        "</div>"
    )
    nav = []
    if sibling:
        href, other_lang = sibling
        nav.append(f'<a href="{html.escape(href)}">{ui_for(other_lang)["lang_label"]}</a>')
    if index_href:
        nav.append(f'<a href="{html.escape(index_href)}">{ui["index"]}</a>')
    if nav:
        parts.append(f'<div class="nav">{"".join(nav)}</div>')
    parts.append("</header>")

    if doc.get("audio", {}).get("mp3"):
        mp3 = html.escape(doc["audio"]["mp3"])
        parts.append(
            f'<div class="player"><audio id="player" controls preload="metadata" '
            f'src="{mp3}"></audio></div>'
        )

    parts.append("<main>")
    for b in doc["blocks"]:
        text = (b["text"].get(lang) or "").strip()
        reasons = [r.strip() for r in (b.get("flags", {}).get(lang) or "").split(",") if r.strip()]
        cls = "block flagged" if reasons else "block"
        parts.append(
            f'<section class="{cls}" id="t{b["index"]}" '
            f'data-start="{b["start"]:.2f}" data-end="{b["end"]:.2f}">'
        )
        parts.append(
            f'<button class="t" data-start="{b["start"]:.2f}" '
            f'title="{fmt_ts(b["start"])} – {fmt_ts(b["end"])}">'
            f'{fmt_ts(b["start"])}</button>'
        )
        if text:
            parts.append(f'<div><p class="s">{html.escape(text)}</p>')
        else:
            parts.append(f'<div><p class="s empty">{ui["empty"]}</p>')
        for reason in reasons:
            label = ui["flag"].get(reason, reason)
            parts.append(f'<p class="warn">⚠ {html.escape(label)}</p>')
        parts.append("</div></section>")
    parts.append("</main>")

    if src.get("url"):
        shown = f'<a href="{html.escape(src["url"])}">{html.escape(src["file"])}</a>'
    else:
        shown = html.escape(src["file"])
    parts.append(
        f'<footer>{ui["source"]}: {shown}<br>'
        f'{html.escape(model_name)} · '
        f'{html.escape(doc["model"]["task"][lang])}</footer>'
    )

    return _page(f"{title} — {ui['transcript']} ({ui['lang_label']})",
                 lang, "\n".join(parts), SEEK_JS)


def render_index(docs: list[dict], lang: str, collection: str,
                 languages: list[str]) -> str:
    """Index of every transcript that exists in `lang`; `languages` are the
    sibling indexes to link to."""
    ui = ui_for(lang)
    docs = [d for d in docs if lang in d.get("html", {})]
    rows = []
    for d in docs:
        src = d["source"]
        href = html.escape(d["html"][lang])
        sub = f'{fmt_duration(src["duration"])} · {len(d["blocks"])} {ui["blocks"]}'
        if src.get("module"):
            sub = f'{html.escape(src["module"])} · {sub}'
        rows.append(
            f'<li><a href="{href}">{html.escape(src["title"])}</a>'
            f'<div class="sub">{sub}</div></li>'
        )
    total = sum(d["source"]["duration"] for d in docs)
    nav = "".join(
        f'<a href="index.{l}.html">{ui_for(l)["lang_label"]}</a>'
        for l in languages if l != lang
    )
    body = (
        "<header>"
        f'<p class="eyebrow">{ui["index"]}</p>'
        f"<h1>{html.escape(collection)}</h1>"
        f'<div class="meta"><span>{len(docs)} × {ui["transcript"]}</span>'
        f'<span>{ui["duration"]} <b>{fmt_duration(total)}</b></span></div>'
        + (f'<div class="nav">{nav}</div>' if nav else "")
        + "</header>"
        f'<ul class="idx">{"".join(rows)}</ul>'
    )
    return _page(f"{collection} — {ui['index']}", lang, body)


def write_outputs(doc: dict, out_dir: Path, stem: str, languages: list[str],
                  index_href: str | None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written: dict[str, str] = {}
    for lang in languages:
        others = [l for l in languages if l != lang]
        sibling = (f"{stem}.{others[0]}.html", others[0]) if others else None
        name = f"{stem}.{lang}.html"
        idx = index_href.replace(".LANG.", f".{lang}.") if index_href else None
        (out_dir / name).write_text(
            render_transcript(doc, lang, sibling, idx), encoding="utf-8"
        )
        (out_dir / f"{stem}.{lang}.srt").write_text(render_srt(doc, lang), encoding="utf-8")
        (out_dir / f"{stem}.{lang}.txt").write_text(render_txt(doc, lang), encoding="utf-8")
        written[lang] = name
    return written
