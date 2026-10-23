#!/usr/bin/env python
"""Replace the German leg of existing transcripts with Voxtral Small 24B.

English stays as Granite produced it -- the two models agreed word-for-word on
every block of the sample, so there is nothing to gain by redoing it. Only the
translation leg moves, and the Granite German is kept alongside under
`text_prev` so the swap stays auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_voxtral import AST_MODEL, OpenRouter, load_key, wav_b64  # noqa: E402
from pipeline import audio as A  # noqa: E402
from pipeline import render as R  # noqa: E402
from transcribe import block_flags, write_indexes  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, nargs="?", default=Path("out"))
    p.add_argument("--lang", default="de")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-spend", type=float, default=1.00,
                   help="abort before exceeding this many USD (default 1.00)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="redo transcripts whose German already came from Voxtral")
    args = p.parse_args()

    def pending(q: Path) -> bool:
        d = json.loads(q.read_text(encoding="utf-8"))
        if args.lang not in d["model"]["languages"]:
            return False  # e.g. a single-language captions transcript
        done = d["model"].get("providers", {}).get(args.lang) == AST_MODEL
        return args.force or not done

    found = sorted(q for q in args.out_dir.glob("*.json") if ".compare" not in q.name)
    if not found:
        sys.exit(f"no transcript JSON in {args.out_dir}")
    docs = [q for q in found if pending(q)]
    if not docs:
        print(f"nothing to do: all {len(found)} transcripts already have Voxtral "
              f"{args.lang} or are single-language (--force to redo)")
        return 0

    total_blocks = total_secs = 0
    for q in docs:
        d = json.loads(q.read_text(encoding="utf-8"))
        total_blocks += len(d["blocks"])
        total_secs += d["source"]["duration"]
    est = total_secs / 60 * 0.0089 * 0.75  # DE leg only, ~3/4 of measured rate
    print(f"files  : {len(docs)}")
    print(f"blocks : {total_blocks} ({total_secs / 60:.1f} min audio)")
    print(f"model  : {AST_MODEL} -> {args.lang}")
    print(f"budget : ~${est:.2f} estimated, hard stop at ${args.max_spend:.2f}\n")
    if args.dry_run:
        print("dry run: nothing sent.")
        return 0

    api = OpenRouter(load_key())
    scratch = args.out_dir / "_wav"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0

    for q in docs:
        doc = json.loads(q.read_text(encoding="utf-8"))
        video = Path(doc["source"]["path"])
        if not video.exists():
            print(f"  !! source missing, skipping: {video}")
            continue

        wav = scratch / (q.stem + ".wav")
        A.extract_audio(video, wav, None)
        samples, sr = sf.read(str(wav), dtype="float32", always_2d=False)

        def one(b: dict) -> tuple[dict, str]:
            if not (b["text"].get(doc["model"]["source_lang"]) or "").strip():
                return b, ""  # silent block: nothing to translate
            chunk = samples[int(b["start"] * sr):int(b["end"] * sr)]
            return b, api.translate_de(wav_b64(chunk, sr))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for b, de in pool.map(one, doc["blocks"]):
                b.setdefault("text_prev", {})[args.lang] = b["text"].get(args.lang, "")
                b["text"][args.lang] = de
                done += 1
                print(f"\r  {q.stem[:48]:50s} {done}/{total_blocks} "
                      f"${api.spend:.3f}", end="", file=sys.stderr, flush=True)

        if api.spend > args.max_spend:
            print(f"\n!! spend ${api.spend:.2f} exceeded --max-spend; stopping",
                  file=sys.stderr)
            return 2

        for b in doc["blocks"]:
            b["flags"] = block_flags(b["text"], doc["model"]["source_lang"])
        doc["model"]["providers"] = {
            doc["model"]["source_lang"]: doc["model"]["name"],
            args.lang: AST_MODEL,
        }
        # The model itself is shown from `providers`; don't repeat it here.
        doc["model"]["task"][args.lang] = f"AST {doc['model']['source_lang']}→{args.lang}"
        q.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        doc["html"] = R.write_outputs(
            doc, args.out_dir, q.stem, doc["model"]["languages"], "index.LANG.html"
        )
        q.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        wav.unlink(missing_ok=True)

    write_indexes(args.out_dir, None)
    all_docs = [json.loads(q.read_text(encoding="utf-8")) for q in docs]
    scratch.rmdir()
    print(f"\n\n{done} blocks in {time.time() - t0:.0f}s | spent ${api.spend:.3f}")
    flagged = sum(1 for d in all_docs for b in d["blocks"] if b.get("flags"))
    print(f"flagged blocks now: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
