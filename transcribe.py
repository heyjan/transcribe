#!/usr/bin/env python
"""Video -> mp3 -> sentence-level bilingual transcript (Granite Speech 5.0 2B + Voxtral).

Inputs are local video files/directories or YouTube URLs (videos or
playlists). Per video: get 16 kHz mono audio, cut it at natural pauses, then
transcribe each block with Granite and translate it into German from the
audio with Voxtral Small (OpenRouter), or Granite with --translator granite. For a
single-language YouTube run, the video's own captions can stand in for the
model. Emits JSON plus HTML, SRT and TXT per language.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import audio as A  # noqa: E402
from pipeline import captions as C  # noqa: E402
from pipeline import render as R  # noqa: E402
from pipeline import sources as S  # noqa: E402
from pipeline.models import GRANITE_MODEL  # noqa: E402
from compare_voxtral import AST_MODEL  # noqa: E402
from pipeline.asr import (  # noqa: E402
    ASR_PROMPT, detect_repetition, detect_unpunctuated, detect_untranslated, load_wav,
)

# Voxtral's prompt is German-only; other target languages stay on Granite.
VOXTRAL_LANGS = {"de"}
# Measured on the course: $0.0089 per audio minute for EN+DE, ~3/4 of it DE.
VOXTRAL_USD_PER_MIN = 0.0089 * 0.75

# -55 dBFS RMS: below this a block holds no speech, and asking an LLM-based
# recogniser to transcribe silence is how you get hallucinated loops.
SILENCE_RMS = 10 ** (-55 / 20)


def block_flags(texts: dict[str, str], source_lang: str) -> dict[str, str]:
    """Quality flags for one block, keyed by language.

    Kept as a pure function over the texts so it can be re-applied to a cached
    JSON without spending GPU time again.
    """
    flags: dict[str, str] = {}
    source = texts.get(source_lang, "")
    for lang, txt in texts.items():
        reasons = []
        if detect_repetition(txt):
            reasons.append("repetition")
        if lang != source_lang and detect_untranslated(source, txt):
            reasons.append("untranslated")
        if detect_unpunctuated(txt):
            reasons.append("unpunctuated")
        if reasons:
            flags[lang] = ", ".join(reasons)
    return flags


def reflag(doc: dict) -> dict:
    for b in doc["blocks"]:
        b["flags"] = block_flags(b["text"], doc["model"]["source_lang"])
    return doc


def source_meta(src: S.Source, duration: float) -> dict:
    meta = {
        "kind": "youtube" if src.is_youtube else "local",
        "file": src.file,
        "path": str(src.media),
        "title": src.title,
        "module": src.module,
        "duration": round(duration, 2),
    }
    if src.is_youtube:
        meta["url"] = src.url
        meta["video_id"] = src.info.get("id")
    return meta


def fetch_media(src: S.Source, out_dir: Path, args) -> None:
    if src.is_youtube and src.media is None:
        dest = out_dir / "sources"
        print("  yt-dlp  -> " + ("reusing downloaded audio" if S.downloaded_audio(src, dest)
                                 else "downloading audio"), flush=True)
        src.media = S.download_audio(src, dest, args.js)


def caption_track(src: S.Source, args) -> tuple[str, str] | None:
    """The caption track to use instead of the model, if any.

    Only for single-language runs: caption cues are timed independently of
    our pause-aligned blocks, so they can't be paired block-for-block with a
    translation made from the audio.
    """
    if not src.is_youtube or args.captions == "never" or len(args.languages) != 1:
        return None
    track = S.pick_caption_track(src.info, args.languages[0])
    if track is None and args.captions == "only":
        raise RuntimeError(f"no usable {args.languages[0]} captions on this video")
    return track


def from_captions(src: S.Source, track: tuple[str, str], out_dir: Path, args) -> dict:
    kind, key = track
    lang = args.languages[0]
    print(f"  captions-> YouTube {kind} captions ({key})", flush=True)
    with tempfile.TemporaryDirectory(prefix="transcribe-vtt-") as tmp:
        vtt = S.download_captions(src, kind, key, Path(tmp), args.js)
        cues = C.vtt_to_cues(vtt)
    if not cues:
        raise RuntimeError(f"{key} caption track is empty")

    fetch_media(src, out_dir, args)
    duration = src.duration or A.probe_duration(src.media)
    mp3_path = out_dir / "audio" / f"{src.stem}.mp3"
    A.encode_mp3(src.media, mp3_path, args.mp3_bitrate)

    raw = C.cues_to_blocks(cues, args.target, args.max_len, args.min_len)
    blocks = []
    for i, b in enumerate(raw):
        texts = {lang: b["text"]}
        blocks.append({
            "index": i,
            "start": round(b["start"], 2),
            "end": round(b["end"], 2),
            "duration": round(b["end"] - b["start"], 2),
            "text": texts,
            "flags": block_flags(texts, lang),
        })
    print(f"  segment -> {len(cues)} cues packed into {len(blocks)} blocks", flush=True)

    name = "YouTube captions" if kind == "manual" else "YouTube auto-captions"
    return {
        "schema": "transcribe/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": args.collection,
        "source": source_meta(src, duration),
        "audio": {"mp3": f"audio/{mp3_path.name}"},
        "model": {
            "name": name,
            "source_lang": lang,
            "languages": [lang],
            "task": {lang: f"captions ({kind}, {key})"},
        },
        "segmentation": {
            "method": "caption cues packed to target length at sentence ends",
            "cues": len(cues),
            "target": args.target, "max_len": args.max_len, "min_len": args.min_len,
        },
        "stats": {"seconds_audio": round(duration, 2), "seconds_compute": 0.0,
                  "blocks": len(blocks)},
        "blocks": blocks,
    }


def translator_for(lang: str, args) -> str:
    return "voxtral" if args.translator == "voxtral" and lang in VOXTRAL_LANGS else "granite"


def from_audio(src: S.Source, out_dir: Path, engine, api, args) -> dict:
    import numpy as np

    fetch_media(src, out_dir, args)
    duration = A.probe_duration(src.media)
    wav_path = out_dir / "audio" / f"{src.stem}.wav"
    mp3_path = out_dir / "audio" / f"{src.stem}.mp3"

    print(f"  ffmpeg  -> 16 kHz mono wav + mp3 ({R.fmt_duration(duration)})", flush=True)
    A.extract_audio(src.media, wav_path, mp3_path, args.mp3_bitrate)

    blocks, seg_meta = A.segment(
        wav_path, duration, args.target, args.max_len, args.min_len
    )
    print(
        f"  segment -> {len(blocks)} blocks "
        f"(silences {seg_meta['silences_found']} @ {seg_meta['noise_db']} dB, "
        f"{seg_meta['hard_cuts']} hard cuts)",
        flush=True,
    )

    wav, sr = load_wav(wav_path)
    remote = [l for l in args.languages
              if l != args.source_lang and translator_for(l, args) == "voxtral"]
    out_blocks = []
    t0 = time.time()
    for b in blocks:
        lo, hi = int(b.start * sr), min(int(b.end * sr), len(wav))
        chunk = wav[lo:hi]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

        texts: dict[str, str] = {}
        if rms >= SILENCE_RMS:
            for lang in args.languages:
                if lang == args.source_lang:
                    texts[lang] = engine.transcribe(chunk, b.duration)
                elif lang in remote:
                    texts[lang] = ""  # filled by Voxtral below
                else:
                    texts[lang] = engine.translate(chunk, b.duration, lang)
        else:
            texts = {lang: "" for lang in args.languages}

        out_blocks.append({
            "index": b.index,
            "start": round(b.start, 2),
            "end": round(b.end, 2),
            "duration": round(b.duration, 2),
            "rms": round(rms, 5),
            "text": texts,
        })
        done = b.index + 1
        print(
            f"\r  asr     -> block {done}/{len(blocks)} "
            f"({b.end:.0f}s / {duration:.0f}s)",
            end="", file=sys.stderr, flush=True,
        )
    elapsed = time.time() - t0
    gpu_passes = len(args.languages) - len(remote)
    print(
        f"\r  asr     -> {len(blocks)} blocks in {elapsed:.1f}s "
        f"(RTFx {duration / max(elapsed, 1e-6):.1f}, "
        f"{gpu_passes} GPU pass{'es' if gpu_passes > 1 else ''})      ",
        file=sys.stderr, flush=True,
    )

    if remote:
        from compare_voxtral import wav_b64

        # Silent blocks, and blocks Granite heard nothing in, aren't sent.
        todo = [b for b in out_blocks if b["text"][args.source_lang].strip()]

        def one(b: dict) -> str:
            chunk = wav[int(b["start"] * sr):int(b["end"] * sr)]
            return api.translate_de(wav_b64(chunk, sr))

        t1, before = time.time(), api.spend
        for lang in remote:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for n, (b, text) in enumerate(zip(todo, pool.map(one, todo)), 1):
                    b["text"][lang] = text
                    print(f"\r  voxtral -> {lang} block {n}/{len(todo)}",
                          end="", file=sys.stderr, flush=True)
        print(f"\r  voxtral -> {len(todo)} blocks {', '.join(remote)} in "
              f"{time.time() - t1:.1f}s (${api.spend - before:.3f})      ",
              file=sys.stderr, flush=True)

    for b in out_blocks:
        b["flags"] = block_flags(b["text"], args.source_lang)

    if not args.keep_wav:
        wav_path.unlink(missing_ok=True)

    task = {
        lang: ("ASR" if lang == args.source_lang
               else f"AST {args.source_lang}→{lang}")
        for lang in args.languages
    }
    providers = {lang: (AST_MODEL if lang in remote else engine.model_name)
                 for lang in args.languages}
    return {
        "schema": "transcribe/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": args.collection,
        "source": source_meta(src, duration),
        "audio": {"mp3": f"audio/{mp3_path.name}"},
        "model": {
            "name": engine.model_name,
            "revision": engine.revision,
            "source_lang": args.source_lang,
            "languages": args.languages,
            "task": task,
            "providers": providers,
            "asr_prompt": ASR_PROMPT,
        },
        "segmentation": seg_meta,
        "stats": {"seconds_audio": round(duration, 2),
                  "seconds_compute": round(elapsed, 1),
                  "blocks": len(blocks)},
        "blocks": out_blocks,
    }


def transcript_docs(out_dir: Path) -> list[dict]:
    return [
        json.loads(q.read_text(encoding="utf-8"))
        for q in sorted(out_dir.glob("*.json"))
        if ".compare" not in q.name
    ]


def write_indexes(out_dir: Path, collection: str | None) -> list[str]:
    """Rebuild the index pages from every transcript in `out_dir`, not just
    this run's, so adding one video never drops the others from the index."""
    docs = [d for d in transcript_docs(out_dir) if "html" in d]
    if not docs:
        return []
    title = (collection
             or next((d["collection"] for d in reversed(docs) if d.get("collection")), None)
             or S.prettify(out_dir.resolve().name))
    languages = sorted({l for d in docs for l in d["html"]},
                       key=lambda l: (l != docs[0]["model"]["source_lang"], l))
    for lang in languages:
        (out_dir / f"index.{lang}.html").write_text(
            R.render_index(docs, lang, title, languages), encoding="utf-8"
        )
    return [f"index.{l}.html" for l in languages]


def render_all(out_dir: Path, collection: str | None) -> int:
    """Rebuild every page in `out_dir` from its JSON: no inputs, no network, no GPU."""
    docs = transcript_docs(out_dir)
    if not docs:
        print(f"no transcript JSON in {out_dir}", file=sys.stderr)
        return 1
    for q in sorted(out_dir.glob("*.json")):
        if ".compare" in q.name:
            continue
        doc = reflag(json.loads(q.read_text(encoding="utf-8")))
        doc["html"] = R.write_outputs(doc, out_dir, q.stem, doc["model"]["languages"],
                                      "index.LANG.html")
        q.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rendered {len(docs)} transcripts; "
          f"index -> {', '.join(write_indexes(out_dir, collection))}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*",
                   help="video files, directories, or YouTube URLs / video IDs / playlists "
                        "(may be omitted with --render-only)")
    p.add_argument("-o", "--out", type=Path, default=Path("out"))
    p.add_argument("--model", default=GRANITE_MODEL, help=f"Granite checkpoint (default {GRANITE_MODEL})")
    p.add_argument("--revision", default=None,
                   help="Hub commit/tag; default: the pinned commit for the default model, "
                        "else main")
    p.add_argument("--languages", default="en,de",
                   help="output languages, comma separated (default en,de)")
    p.add_argument("--source-lang", default="en",
                   help="spoken language; others are produced by translation")
    p.add_argument("--translator", choices=("voxtral", "granite"), default="voxtral",
                   help="German leg: Voxtral Small via OpenRouter (default, paid) "
                        "or Granite on the GPU (free, noticeably worse)")
    p.add_argument("--workers", type=int, default=4, help="parallel Voxtral requests")
    p.add_argument("--max-spend", type=float, default=1.00,
                   help="stop starting new videos past this many USD (default 1.00)")
    p.add_argument("--captions", choices=("prefer", "only", "never"), default="prefer",
                   help="YouTube only, single-language runs only: use the video's "
                        "captions instead of the model (default prefer)")
    p.add_argument("--js-runtime", default="auto",
                   help="JavaScript runtime for yt-dlp: auto, none, or NAME[:PATH]")
    p.add_argument("--collection", default=None, help="title for the index page")
    p.add_argument("--target", type=float, default=24.0, help="target block seconds")
    p.add_argument("--max-len", type=float, default=30.0)
    p.add_argument("--min-len", type=float, default=6.0)
    p.add_argument("--mp3-bitrate", default="96k")
    p.add_argument("--keep-wav", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only first N videos")
    p.add_argument("--force", action="store_true", help="redo videos already done")
    p.add_argument("--render-only", action="store_true",
                   help="rebuild HTML from existing JSON; never loads the model")
    args = p.parse_args()

    args.languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    if args.source_lang not in args.languages:
        args.languages.insert(0, args.source_lang)
    if args.captions == "only" and len(args.languages) != 1:
        p.error("--captions only needs exactly one language, e.g. "
                "--languages en --source-lang en")

    if not args.inputs:
        if not args.render_only:
            p.error("give at least one input (or --render-only to rebuild OUT)")
        return render_all(args.out, args.collection)

    refs = [i for i in args.inputs if S.is_youtube_ref(i)]
    paths = [Path(i) for i in args.inputs if not S.is_youtube_ref(i)]
    for missing in (q for q in paths if not q.exists()):
        print(f"not found: {missing}", file=sys.stderr)
        return 1

    args.js = S.js_runtime_options(args.js_runtime)
    sources = S.collect_local(paths)
    playlist = None
    if refs:
        print(f"yt-dlp  : resolving {len(refs)} YouTube input(s)", flush=True)
        try:
            yt_sources, playlist = S.collect_youtube(refs, args.js)
        except RuntimeError as exc:
            print(f"yt-dlp failed: {exc}", file=sys.stderr)
            return 1
        sources += yt_sources
    if not sources:
        print("no video files found", file=sys.stderr)
        return 1
    if args.limit:
        sources = sources[: args.limit]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    args.collection = args.collection or playlist or (
        S.local_collection(paths) if paths else None
    )

    total = sum(s.duration if s.duration is not None else A.probe_duration(s.media)
                for s in sources)
    remote = [l for l in args.languages
              if l != args.source_lang and translator_for(l, args) == "voxtral"]
    api = None
    if remote and not args.render_only:
        from compare_voxtral import OpenRouter, load_key
        api = OpenRouter(load_key())  # fail now, not after an hour of GPU work

    print(f"model   : {args.model}" + (f" + {AST_MODEL} for {', '.join(remote)}"
                                       if remote else ""))
    print(f"videos  : {len(sources)}  ({total / 60:.1f} min)")
    print(f"output  : {out_dir.resolve()}")
    print(f"langs   : {', '.join(args.languages)} "
          f"(source {args.source_lang}, rest via speech translation)")
    if api:
        print(f"budget  : ~${total / 60 * VOXTRAL_USD_PER_MIN:.2f} for Voxtral if nothing "
              f"is cached, new videos stop past ${args.max_spend:.2f}")
    print()

    t0 = time.time()
    engine = None

    def get_engine():
        # Loaded on first use: a captions-only or fully cached run never
        # touches the GPU.
        nonlocal engine
        if engine is None:
            from pipeline.asr import GraniteSpeech
            t = time.time()
            engine = GraniteSpeech(args.model, args.revision)
            print(f"  model   -> loaded on {engine.device} in {time.time() - t:.1f}s", flush=True)
        return engine

    failed = 0
    for n, src in enumerate(sources, 1):
        print(f"[{n}/{len(sources)}] {src.title}", flush=True)
        json_path = out_dir / f"{src.stem}.json"

        if json_path.exists() and (args.render_only or not args.force):
            print("  skip    -> already transcribed (--force to redo)")
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        elif args.render_only:
            print("  skip    -> no JSON yet, and --render-only was given")
            continue
        elif api and api.spend > args.max_spend:
            print(f"  !! stopped: Voxtral spend ${api.spend:.2f} is past --max-spend",
                  file=sys.stderr, flush=True)
            failed += 1
            continue
        else:
            try:
                track = caption_track(src, args)
                doc = (from_captions(src, track, out_dir, args) if track
                       else from_audio(src, out_dir, get_engine(), api, args))
            except RuntimeError as exc:
                print(f"  !! failed: {exc}\n", file=sys.stderr, flush=True)
                failed += 1
                continue

        reflag(doc)
        langs = doc["model"]["languages"]
        doc["html"] = R.write_outputs(doc, out_dir, src.stem, langs, "index.LANG.html")
        json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  write   -> {src.stem}.json + .html/.srt/.txt for "
              f"{', '.join(langs)}\n", flush=True)

    indexes = write_indexes(out_dir, args.collection)
    if indexes:
        print(f"index   -> {', '.join(indexes)}")
    print(f"done in {time.time() - t0:.0f}s"
          + (f", Voxtral ${api.spend:.3f}" if api else "")
          + (f" ({failed} failed)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
