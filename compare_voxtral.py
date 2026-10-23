#!/usr/bin/env python
"""Compare Granite Speech 4.1 2B against Voxtral (via OpenRouter) block by block.

Runs on the exact block spans already in the Granite JSON, so segmentation is
held constant and only the models differ.

  EN: mistralai/voxtral-mini-transcribe  (transcription API, ASR only)
  DE: mistralai/voxtral-small-24b-2507   (chat API, audio -> German)

mini-transcribe returns the *spoken* language, so it cannot produce German from
English audio; Voxtral Small does the translation leg.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import audio as A  # noqa: E402

API = "https://openrouter.ai/api/v1"
ASR_MODEL = "mistralai/voxtral-mini-transcribe"
AST_MODEL = "mistralai/voxtral-small-24b-2507"
DE_PROMPT = (
    "Translate the speech in this audio into German. "
    "Reply with the German translation only, with proper punctuation and "
    "capitalization. Do not add commentary."
)


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".openrouter_key"
        if path.exists():
            key = path.read_text().strip()
    if not key:
        sys.exit(
            "No API key. Either:\n"
            "  echo 'sk-or-v1-...' > ~/.openrouter_key && chmod 600 ~/.openrouter_key\n"
            "or export OPENROUTER_API_KEY before running."
        )
    if not key.startswith("sk-or-"):
        sys.exit("That does not look like an OpenRouter key (expected sk-or-...).")
    return key


def wav_b64(samples, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OpenRouter:
    def __init__(self, key: str, timeout: int = 180):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "granite-vs-voxtral",
        })
        self.timeout = timeout
        self.spend = 0.0

    def _post(self, path: str, payload: dict, tries: int = 4) -> dict:
        last = None
        for n in range(tries):
            try:
                r = self.s.post(f"{API}{path}", data=json.dumps(payload),
                                timeout=self.timeout)
            except requests.RequestException as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    return r.json()
                # 429/5xx are worth retrying; 4xx otherwise is a real error.
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code < 500 and r.status_code != 429:
                    raise RuntimeError(last)
            time.sleep(2 ** n)
        raise RuntimeError(f"failed after {tries} tries -- {last}")

    def transcribe(self, b64: str) -> str:
        out = self._post("/audio/transcriptions", {
            "model": ASR_MODEL,
            "input_audio": {"data": b64, "format": "wav"},
        })
        self._bill(out)
        return (out.get("text") or "").strip()

    def translate_de(self, b64: str) -> str:
        out = self._post("/chat/completions", {
            "model": AST_MODEL,
            # Mistral rejects temperature 0 unless top_p is explicitly 1
            # ("top_p must be 1 when using greedy sampling", code 3054).
            "temperature": 0,
            "top_p": 1,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": DE_PROMPT},
                {"type": "input_audio",
                 "input_audio": {"data": b64, "format": "wav"}},
            ]}],
        })
        self._bill(out)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def _bill(self, out: dict) -> None:
        usage = out.get("usage") or {}
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.spend += float(cost)


def norm(t: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[^\W\d_]+", t, re.UNICODE)]


def wer(ref: str, hyp: str) -> float:
    """Word error rate of hyp against ref. Neither is ground truth here --
    this measures how far the two systems disagree, not who is right."""
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    sm = difflib.SequenceMatcher(None, r, h, autojunk=False)
    edits = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"
    )
    return edits / len(r)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_file", type=Path, help="an existing Granite transcript JSON")
    p.add_argument("-n", "--blocks", type=int, default=8,
                   help="how many blocks to compare (default 8)")
    p.add_argument("--offset", type=int, default=0, help="start at this block index")
    p.add_argument("-o", "--out", type=Path, default=Path("out/compare"))
    p.add_argument("--skip-de", action="store_true", help="English comparison only")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be sent, call nothing")
    args = p.parse_args()

    doc = json.loads(args.json_file.read_text(encoding="utf-8"))
    blocks = doc["blocks"][args.offset:args.offset + args.blocks]
    video = Path(doc["source"]["path"])
    if not video.exists():
        sys.exit(f"source video is gone: {video}")

    secs = sum(b["duration"] for b in blocks)
    print(f"source : {doc['source']['title']}")
    print(f"blocks : {len(blocks)} ({secs:.0f}s of audio)")
    print(f"models : {ASR_MODEL} (en)"
          + ("" if args.skip_de else f" + {AST_MODEL} (de)"))
    if args.dry_run:
        print("\ndry run: nothing sent.")
        return 0

    key = load_key()
    api = OpenRouter(key)

    tmp = args.out / "_slice.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    full = args.out / "_full.wav"
    if not full.exists():
        print("extracting 16 kHz mono audio ...")
        A.extract_audio(video, full, None)
    samples, sr = sf.read(str(full), dtype="float32", always_2d=False)

    rows = []
    t0 = time.time()
    for n, b in enumerate(blocks, 1):
        chunk = samples[int(b["start"] * sr):int(b["end"] * sr)]
        b64 = wav_b64(chunk, sr)
        en = api.transcribe(b64)
        de = "" if args.skip_de else api.translate_de(b64)
        rows.append({
            "index": b["index"], "start": b["start"], "end": b["end"],
            "granite": {"en": b["text"]["en"], "de": b["text"]["de"]},
            "voxtral": {"en": en, "de": de},
            "en_disagreement": round(wer(b["text"]["en"], en), 3),
        })
        print(f"\r  block {n}/{len(blocks)}  spend ${api.spend:.4f}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    tmp.unlink(missing_ok=True)
    result = {
        "source": doc["source"],
        "models": {"granite": doc["model"]["name"],
                   "voxtral_en": ASR_MODEL, "voxtral_de": AST_MODEL},
        "audio_seconds": round(secs, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "cost_usd": round(api.spend, 4),
        "mean_en_disagreement": round(
            sum(r["en_disagreement"] for r in rows) / max(len(rows), 1), 3),
        "blocks": rows,
    }
    out_json = args.out / f"{args.json_file.stem}.compare.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\naudio {secs:.0f}s | wall {result['wall_seconds']:.0f}s "
          f"| cost ${result['cost_usd']:.4f}")
    print(f"mean EN disagreement (Granite vs Voxtral): "
          f"{result['mean_en_disagreement']:.1%}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
