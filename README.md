# Video → bilingual transcript pipeline

Turns **local video files** and **YouTube videos** into sentence-level
transcripts in **English and German**: JSON, plus an HTML page (click-to-seek
audio player), an SRT and a plain-text file per language.

## Models and where they run

Two models, one on this machine and one hosted:

| | English | German |
|---|---|---|
| **Model** | [`ibm-granite/granite-speech-5.0-2b`](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) (IBM, 2B, Apache 2.0, released 2026-10-20), pinned to commit `c075cf19` | [`mistralai/voxtral-small-24b-2507`](https://openrouter.ai/mistralai/voxtral-small-24b-2507) (Mistral, 24B, Apache 2.0) |
| **Task** | speech → English text (ASR) | speech → German text, translated **from the audio** (AST) |
| **Runs on** | **locally**, on the RTX 5080 through Hugging Face `transformers`; `make model` downloads the weights once (4.6 GB, public, no token) into `~/.cache/huggingface` | **hosted by Mistral**, reached through **[OpenRouter](https://openrouter.ai)** (an API router; Mistral is the only provider it lists for this model) |
| **Needs** | ~4.6 GB VRAM | an OpenRouter key in `~/.openrouter_key` |
| **Cost** | free | $0.0001 per audio second: **≈ $0.36 per hour** of video |

**So yes, partly local.** Downloading, audio extraction, segmentation, English
transcription and all rendering happen on this machine. Only the German leg
goes out: each ~24 s audio block is uploaded to OpenRouter, which passes it to
Mistral. Keep that in mind for audio you would not upload anywhere.

Two other modes:

- **Fully local, free:** `DE=granite` translates to German with Granite on the
  GPU instead. It's noticeably worse (see [Benchmarks](#german-granite-50-vs-voxtral)) and is
  meant as a fallback for offline or private material.
- **No model at all:** `make youtube-captions` uses a YouTube video's own caption
  track, one language at a time.

## Benchmarks

Measured on this machine (RTX 5080 16 GB, WSL2). RTFx = audio duration ÷
processing time, so higher is faster.

### German: Granite 5.0 vs Voxtral

**Material:** the first 10 minutes (600 s) of Harvard Business Review's
interview [*How McKinsey Plans to Survive AI (and Reinvent
Consulting)*](https://www.youtube.com/watch?v=hSpem_oGAf0): two speakers,
conversational, business jargon. Both runs used the same file, so they got
the same 26 pause-cut blocks and identical English (26/26 blocks). Only the
German differs.

| | Granite 5.0 (`DE=granite`) | Voxtral Small (**default**) |
|---|---|---|
| GPU time | 133.8 s (2 passes, RTFx 4.5) | 59.0 s (1 pass, RTFx 10.2) |
| translation time | included above | 22.3 s (4 parallel requests) |
| end to end | 148 s | **95 s** |
| cost | $0 | $0.060 |
| blocks flagged untranslated / looping | 0 | 0 |
| meaning errors, invented words, wrong terms ¹ | **~17** | **~4** |
| "AI" (should be "KI" in German) | 7× "AI", 3× "IA" | 10× "KI", 0× "AI" |
| form of address | switches du/Sie, even within one sentence | consistent Sie |

¹ Counted by reading all 26 blocks side by side against the English: one
careful reviewer, not an automatic score.

Typical Granite errors, all from this clip:

| English | Granite 5.0 | Voxtral |
|---|---|---|
| an outcomes-based model | ein … **ausgewogenes** Modell (*balanced*) | ein … ergebnisorientiertes Modell |
| rewire the organization to realize the value | die Organisation neu **anstecken**, um den Wert zu **erahnen** (*infect*, *guess*) | die Organisation neu verdrahten, um den Wert zu realisieren |
| a fast follower | ein schneller **Nachfolger** (*successor*) | schnell nachziehen |
| origination, credit scoring, collection | **Herkunft**, Kreditbewertung, **Sammlung** | Kreditvergabe, Bonitätsprüfung, Inkasso |
| truly novel co-creation | neuere **Koinzession** (not a word) | neuartige Co-Kreation |
| a client-facing side | eine **Klientenfassende** Seite (not a word) | eine kundenorientierte Seite |
| a boatload | eine **Bootsladung** (literal) | eine Menge |

Voxtral's mistakes on the same clip: "what does your org look like" became
"wie sieht **York** aus" (a mishearing), "research tank" became
"Forschungstank" (the English transcript has the same mishearing), it added
half a sentence at one block's end, and "disrupted" became a literal
"gestört" (Granite did the same).

**Verdict.** Granite 5.0 is a real step up from 4.1: no more copied-through
English or loops, and it reads fluently at a glance. But it gets the meaning
wrong often enough that you can't trust it unread, and the flags don't catch
these errors. Voxtral is better on every axis here except price, and at $0.06
per 10 minutes it stays the default. `DE=granite` is for audio that must not
leave the machine.

To reproduce (needs the OpenRouter key; about $0.06):

```bash
yt-dlp -f bestaudio -o full.webm 'https://www.youtube.com/watch?v=hSpem_oGAf0'
mkdir -p src && ffmpeg -i full.webm -t 600 -vn -c:a copy src/first10min.webm
make local SRC=src OUT=bench/granite DE=granite
make local SRC=src OUT=bench/voxtral DE=voxtral
```

### Speed and cost, all runs

| material | pipeline | GPU time | RTFx | Voxtral time | cost |
|---|---|---|---|---|---|
| HBR interview, 10 min, 26 blocks | Granite 5.0 EN + Voxtral DE (**default**) | 59.0 s | **10.2** | 22.3 s | $0.060 |
| same | Granite 5.0 EN + Granite 5.0 DE (`DE=granite`) | 133.8 s | 4.5 | – | $0 |
| career interview, 6.3 min, 17 blocks | Granite 5.0 EN + Voxtral DE (with punctuation retry) | 47.6 s | 7.9 | 11.7 s | $0.037 |
| same | Granite 5.0 EN + Voxtral DE, before the retry existed | 38.2 s | 9.9 | 11.7 s | $0.037 |
| same | Granite 4.1 EN + Voxtral DE | 40.0 s | 9.4 | 13.5 s | $0.037 |
| same | Granite 4.1 EN + Granite 4.1 DE | 93.2 s | 4.0 | – | $0 |
| same | YouTube captions (`youtube-captions`) | – | – | – | $0, 19 s total incl. download |
| narrated training videos, 6 videos, 35.3 min, 89 blocks | Granite 4.1 EN + Granite 4.1 DE | 381.7 s | 5.5 | – | $0 |
| same | Voxtral DE added afterwards (`make german`) | – | – | – | ≈ $0.21 |

Voxtral costs about $0.006 per minute of audio. Running it for German halves
the GPU work, because Granite makes one pass per block instead of two. Model
load is ~3 s once the weights are cached. Speech-to-text runs sequentially,
block by block (no batching), with about 4.6 GB of the 16 GB VRAM in use.

### English: Granite 5.0 vs 4.1

Same 17 blocks of the 6.3-min career interview:

| | 4.1 | 5.0 |
|---|---|---|
| words that differ between the two | 24 of 1 269 (1.9 %) | |
| 6 of those spot-checked against YouTube's captions | right on 3 ("well", "it'd", no "that's true") | right on 3 (recovered "Ken", "you could use some", "can we", which 4.1 dropped) |
| blocks with **no punctuation at all** | 0 | 4, of which a retry fixes 2; the other 2 are flagged `unpunctuated` |
| GPU time | 40.0 s | 38.2 s (47.6 s with retries) |

So the words are a draw and 5.0 is worse on punctuation. The failures are
flaky: the same block can come back punctuated on one run and bare on the
next. On the 10-min HBR interview, 3 of 26 English blocks stayed unpunctuated
after the retry and are flagged.

**Longer blocks don't help.** 5.0 accepts segments of up to ~2 minutes (IBM's
recommendation), so 60 s and 100 s blocks were tried on the career interview:
about as fast (RTFx 11.2 / 9.1), still with unpunctuated blocks, and the 60 s
run **silently dropped a whole 28-word passage**. Default blocks stay at ~24 s.

### Earlier: German with Granite 4.1

| material | Granite 4.1 German | Voxtral German |
|---|---|---|
| narrated training videos (89 blocks) | 2 flagged, plus unflagged meaning errors | **0** |
| career interview (17 blocks) | **8 flagged** untranslated, plus a 4× repetition loop | **0** |

On 13 aligned blocks of the training videos, Granite's and Voxtral's
**English** agreed word-for-word on every block. That is why English stays on
the free local model.

## Quick start

```bash
make setup                                    # once: .venv + GPU check

make local SRC="/mnt/c/Users/<you>/Downloads/<course-dir>"
make youtube URL='https://www.youtube.com/watch?v=9IErvdJJHs4'
make youtube URL='...' DE=granite             # fully local, free, worse German

explorer.exe "$(wslpath -w out/youtube/index.en.html)"   # open in the Windows browser
```

`make` on its own prints every target. Windows paths are reachable from WSL
under `/mnt/c/...`. **Always single-quote YouTube URLs**: an unquoted `&` in
`...watch?v=X&list=Y` makes the shell cut the command in two.

## Make targets

| target | does | default output |
|---|---|---|
| `make local SRC=<file-or-dir>` | Granite EN + Voxtral DE for one video or a whole folder tree | `out/` |
| `make youtube URL=<url...>` | the same, from the audio of one or more YouTube videos, or a playlist | `out/youtube/` |
| `make youtube-captions URL=<url...>` | the video's own captions, one language, no GPU | `out/youtube/` |
| `make german [OUT=dir]` | give transcripts made with `DE=granite` Voxtral German afterwards | in place |
| `make german-dry [OUT=dir]` | price `make german` without sending anything | – |
| `make render [OUT=dir]` | rebuild HTML/SRT/TXT and indexes from the JSON (no GPU, no network) | in place |
| `make setup` | create `.venv` from `requirements.lock.txt`, then `make check` | – |
| `make check` | print `sm_120 True` if torch sees the 5080 | – |
| `make update-ytdlp` | upgrade yt-dlp (do this first when YouTube downloads start failing) | – |

Variables, all optional:

| variable | default | meaning |
|---|---|---|
| `OUT` | `out` | output dir for `local`, `german`, `render` |
| `YOUTUBE_OUT` | `out/youtube` | output dir for the two YouTube targets |
| `LANGS` | `en,de` | output languages |
| `SOURCE_LANG` | `en` | the spoken language; the others are translated |
| `CAPTION_LANG` | `en` | language for `youtube-captions` |
| `DE` | `voxtral` | German translator: `voxtral` (hosted, paid) or `granite` (local, free) |
| `MAX_SPEND` | `1.00` | USD; no new video is started once Voxtral spend passes this |
| `LIMIT` | – | only the first N videos (good for a trial run) |
| `FORCE` | – | `FORCE=1` redoes videos that already have JSON |
| `COLLECTION` | derived | title of the index page |
| `ARGS` | – | anything else for `transcribe.py`, e.g. `ARGS='--target 20'` |

Several URLs go space-separated inside one pair of quotes:
`make youtube URL='https://youtu.be/AAA https://youtu.be/BBB'`. A bare video ID
(`URL=9IErvdJJHs4`) works too.

Re-running is cheap: anything already transcribed is skipped, and a downloaded
YouTube audio file is reused. Each index page is rebuilt from **every** JSON in
its output folder, so adding one video never drops the others.

## Why it is built this way

**German comes from the audio, not from the English text.** Both Voxtral Small
and Granite translate speech directly. So each block gets one transcription
pass (Granite) and one translation pass (Voxtral) over the *same audio*.
Mistakes in the English transcript don't get copied into the German, and the
two languages stay aligned block-for-block because they cover the same time
span.

**Audio is cut into ~24 s blocks.** Granite embeds audio at 10 Hz into a 1B LLM
(600 tokens per minute), so a whole lecture never fits in one pass; IBM
recommends segments of up to ~2 minutes for 5.0. We stay well below that on
purpose. Each block becomes one timestamp, one SRT cue and one translation
request, and measured on 5.0, longer blocks dropped text (see
[Benchmarks](#benchmarks)).

**Blocks are cut at pauses, never at fixed intervals.** `ffmpeg silencedetect`
finds the pauses, and the planner picks whichever pause sits closest to the
target length. A mid-sentence cut costs you a little in transcription and a lot
in translation, since the translator loses the clause it needs. On the test
material this yields **zero hard cuts**: every boundary is a real pause.

**Timestamps come free from the block boundaries.** The model has no
timestamp output, but
since we control where the cuts land we know each block's span exactly. That is
enough to drive click-to-seek and SRT, and makes the output usable as a RAG
corpus later.

**MP3 is an output, not an intermediate.** The model wants 16 kHz mono; feeding
it MP3 would mean a pointless lossy round-trip. Both files are decoded straight
from the source, and the MP3 becomes the HTML player's audio, so click-to-seek
always works and never depends on the original video's path.

## YouTube

URLs are handed to [yt-dlp](https://github.com/yt-dlp/yt-dlp), which downloads
the best **audio-only** stream as is (usually Opus `.webm`) into
`<out>/sources/`. From there it is exactly the local pipeline. The download is
kept because it *is* the source: re-runs and `voxtral_de.py` re-read it.
Playlists are enumerated first and then processed one video at a time; the
playlist title becomes the index title.

**When captions are used, and when not.** `make youtube` always transcribes
from the audio. `make youtube-captions` uses the video's own caption track
instead: much faster, no GPU, but only **one language** at a time. The
restriction is deliberate. Caption cues are timed independently of our
pause-aligned blocks, so a caption-based English cannot be paired
block-for-block with a German translated from the audio.

Which caption tracks count:

- **Uploaded subtitles** in the requested language: always.
- **Auto-captions**: only in the video's *spoken* language (`en-orig` for an
  English video). YouTube also offers auto-captions in every other language,
  but those are machine translations of its own auto-captions. For German
  from an English video, translating from the audio is the better source, so
  `youtube-captions CAPTION_LANG=de` on an English video fails on purpose. Use
  `make youtube` for that.

Captions are packed into blocks of the same ~24 s length as the audio path, so
both kinds of transcript look alike. Auto-captions are lower-case run-on text
with filler words (`uh`, doubled `I I`), so for anything you will read
closely, `make youtube LANGS=en` (Granite, punctuated) is better than the
captions.

`transcribe.py --captions prefer` (the CLI default) uses captions for a
single-language YouTube run whenever a usable track exists, and the model
otherwise.

**JavaScript runtime.** YouTube extraction needs one, and yt-dlp only enables
`deno` by default, failing with `HTTP Error 403: Forbidden` without it. Any of
`deno`, `node`, `bun` or `quickjs` on `PATH` is detected and passed in
(`--js-runtime` overrides; `node` from nvm is what this machine uses). If
downloads start failing anyway, `make update-ytdlp`, since YouTube changes
often.

Captions can be missing because a creator disabled them, auto-captioning does
not support the audio, or the video is age/region/login-restricted. Audio
transcription only needs yt-dlp to reach the audio.

## Setup

```bash
make setup
```

which is:

```bash
uv venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.lock.txt   # exact tested versions
.venv/bin/python -m pipeline.models                          # = make model
.venv/bin/python -c "import torch; print(torch.cuda.get_arch_list()[-1], torch.cuda.is_available())"
# -> sm_120 True
```

### On a new machine

```bash
git clone <repo> transcribe && cd transcribe
make setup                                   # venv + 4.6 GB model download + GPU check
echo 'sk-or-v1-...' > ~/.openrouter_key && chmod 600 ~/.openrouter_key
```

Needs Linux x86_64 (WSL2 is fine), [`uv`](https://docs.astral.sh/uv/) **≥ 0.13.4**
(`make setup` refuses older ones, see below),
`make`, `ffmpeg`/`ffprobe`, an NVIDIA driver new enough for **CUDA 13**
(the torch wheels are cu130), and for YouTube a JavaScript runtime such as
`node` (see above). Tested from a clean copy with empty `uv` and Hugging Face
caches: `make setup` succeeded, downloading ~4.7 GB of packages (torch and the
CUDA 13 libraries) plus 4.4 GB of model weights. It took 46 min, and the time
is almost entirely download speed. Transcription afterwards ran with
`HF_HUB_OFFLINE=1`, so nothing else is fetched at run time (YouTube and
Voxtral of course still need the network).

**How the model download stays reproducible.**

- The model is **public and ungated**: no Hugging Face account or token. Set
  `HF_TOKEN` only to avoid anonymous rate limits.
- It is **pinned to a commit** (`GRANITE_REVISION` in `pipeline/models.py`),
  not `main`. The repo was still being updated after release, so an unpinned
  install would get different weights from the ones benchmarked here. To
  upgrade, change the commit, `make model`, and re-run a known video.
- Only the files transformers loads are fetched (configs, tokenizer,
  `model.safetensors`). The repo's 411 MB `out_llm.safetensors`, sample WAV
  and images are skipped; the model was verified to load without them.
- Weights go to the standard cache (`~/.cache/huggingface`, or `$HF_HOME`), so
  several checkouts share one copy. Never commit them to git.

**Certificates.** uv ships its own CA roots, and releases before 0.13.4 (and
certifi before 2026.10.19) still trust the emSign CAs that were withdrawn after
the eMudhra private-key incident of 2026-10-20. `make setup` checks the uv
version (`make uv-check`), and `requirements.txt` sets a minimum certifi. Also
update the OS store (`sudo apt update && sudo apt install --only-upgrade
ca-certificates` on Ubuntu), which curl, apt and Debian's pip use.

`requirements.txt` lists the direct dependencies; `requirements.lock.txt` is
the full, exact environment (`make lock` regenerates it from `.venv`).

The RTX 5080 is `sm_120` (Blackwell). Current PyPI torch already ships CUDA 13
wheels that include `sm_120`, so plain PyPI is all you need.

**Do not add `--extra-index-url .../cu128`.** The older advice to force cu128
wheels for Blackwell is now counter-productive: combined with
`--index-strategy unsafe-best-match` it resolves torch from PyPI (cu130) but
torchaudio from the cu128 index, and the mismatch fails at import with
`OSError: libcudart.so.12: cannot open shared object file`. If you hit that,
`uv pip install --reinstall-package torchaudio torchaudio` repairs it.

Key versions: torch 2.13.0+cu130, torchaudio 2.11.0+cu130, transformers
5.14.1 (Granite Speech needs ≥ 4.52.1), huggingface-hub 1.26.1, yt-dlp
2026.10.21. Note transformers 5.x renamed `torch_dtype` to `dtype`; the
loader handles both.

## Direct CLI use

The Makefile only wraps `transcribe.py`; everything is available directly:

```bash
.venv/bin/python transcribe.py "/mnt/c/.../course-dir" -o out
.venv/bin/python transcribe.py 'https://www.youtube.com/watch?v=...' -o out/youtube
.venv/bin/python transcribe.py --render-only -o out          # rebuild pages only
```

Local paths and URLs can be mixed in one call.

| flag | default | meaning |
|---|---|---|
| `--languages` | `en,de` | output languages, comma separated |
| `--source-lang` | `en` | the spoken language; the rest are translated |
| `--translator` | `voxtral` | German leg: `voxtral` (OpenRouter, paid) or `granite` (local GPU, free). Languages other than German always use Granite |
| `--workers` | `4` | parallel Voxtral requests |
| `--max-spend` | `1.00` | USD; no new video is started past this. A video already in progress finishes |
| `--captions` | `prefer` | `prefer` / `only` / `never`: YouTube captions for single-language runs |
| `--js-runtime` | `auto` | yt-dlp JavaScript runtime: `auto`, `none`, or `NAME[:PATH]` |
| `--collection` | derived | index page title |
| `--target` / `--max-len` / `--min-len` | `24` / `30` / `6` | block length in seconds |
| `--limit N` | – | only the first N videos |
| `--force` | – | re-transcribe videos that already have JSON |
| `--keep-wav` | – | keep the 16 kHz WAV instead of deleting it |
| `--mp3-bitrate` | `96k` | mono speech; 96k is transparent enough |
| `--render-only` | – | rebuild pages from existing JSON; never loads the model. With no inputs it rebuilds everything in `-o` |

The model is loaded on first use, so a captions-only or fully cached run never
touches the GPU. The OpenRouter key is checked at startup, so a missing key
fails right away instead of after the GPU work.

## Output

```
out/
  <stem>.json               transcript data (the source of truth)
  <stem>.en.html            English transcript with audio player
  <stem>.de.html            German transcript
  <stem>.en.srt / .de.srt   subtitles, one cue per block
  <stem>.en.txt / .de.txt   plain text, one paragraph per block
  index.en.html             index of every transcript in this folder
  index.de.html
  audio/<stem>.mp3          playback copy, referenced by the HTML player
  sources/<stem>.webm       YouTube only: the downloaded audio
```

Stems are `module__lesson` for local folders (mirroring the directory tree) and
`<title>-<videoID>` for YouTube.

JSON shape:

```jsonc
{
  "collection":   "Index title",
  "source":       { "kind": "local|youtube", "file", "path", "title", "module",
                    "duration", "url", "video_id" },   // url/video_id: YouTube only
  "model":        { "name", "source_lang", "languages", "task", "asr_prompt",
                    "providers" },   // providers: set once Voxtral did a leg
  "segmentation": { "method", "noise_db", "silences_found", "hard_cuts", ... },
  "stats":        { "seconds_audio", "seconds_compute", "blocks" },
  "blocks": [
    { "index": 0, "start": 0.0, "end": 22.23, "duration": 22.23,
      "rms": 0.041,           // audio path only
      "text":  { "en": "...", "de": "..." },
      "text_prev": { "de": "..." },   // Granite's German, after a Voxtral pass
      "flags": { }            // e.g. {"de": "repetition"}
    }
  ]
}
```

## Quality guards

Four failure modes are specific to LLM-based ASR/AST, and all four are
handled. Everything is **flagged, not rewritten**: silently "repairing" model
output is how bad transcripts become invisible ones.

- **Silence hallucination.** A block quieter than −55 dBFS RMS is never sent to
  the model; it is recorded as empty. Asking an LLM to transcribe silence is
  how you get invented sentences.
- **Repetition loops.** Output where a 1–10 word window repeats 4+ times in a
  row is flagged. Collapsing it automatically would destroy legitimately
  repeated phrasing. Loops of a longer sentence slip through: on a
  conversational YouTube test, Granite's German repeated a 13-word sentence 4×
  without being flagged.
- **Untranslated passthrough.** Granite sometimes answers the AST prompt with
  the transcript instead, usually on blocks that open with a speaker
  introducing themselves by name. Detected by word *sequence* similarity
  between source and target.

  Sequence, not vocabulary overlap: overlap only tells you both sides mention
  Anthropic and OpenAI, and it false-positives at 0.63 on a short brand-heavy
  sentence that is a perfectly good translation. On narrated training videos, sequence
  similarity gives passthrough ≥ 0.98, real translations ≤ 0.21, and that
  brand-heavy sentence 0.56, so the 0.6 threshold has margin on both sides.

- **Missing punctuation.** Granite 5.0 sometimes ignores the "with proper
  punctuation and capitalization" part of the prompt and returns a bare word
  stream. Such a block (15+ words, under 1 mark per 100 words; normal speech
  here has 8–25) is retried once with `... Write complete sentences.` added,
  and the better-punctuated result is kept. If it's still bare it is flagged
  `unpunctuated`. On the career interview this took 4 bare blocks to 2 flagged.

Flags live in each block's `flags` map (`{"de": "untranslated"}`) and render as
a marked, ⚠-annotated block in the HTML, so a reviewer can jump straight to the
blocks worth re-checking.

## Prompts

Taken verbatim from the model card's "preferred prompt by task" table. Granite silently falls back to plain transcription on an
unrecognised prompt rather than erroring, so improvising here degrades quality
invisibly. The one exception is the punctuation retry above, which only adds a
sentence and only runs on blocks that already failed.

- ASR: `transcribe the speech with proper punctuation and capitalization.`
- AST: `translate the speech to German with proper punctuation and capitalization.`

Keyword biasing is also available (`transcribe the speech to text. Keywords: ...`,
up to 2 000 keywords) and is worth wiring in for product names or
jargon. IBM reports meaningful F1 gains on keyword recall. Granite also accepts a
language hint (`The speech is in English. Transcribe it exactly; do not
translate. ...`); in a test it made the punctuation problem worse, so it is
not used.

## German via Voxtral

Voxtral needs an OpenRouter key in `OPENROUTER_API_KEY` or `~/.openrouter_key`:

```bash
echo 'sk-or-v1-...' > ~/.openrouter_key && chmod 600 ~/.openrouter_key
```

Each run prints a cost estimate up front and the actual spend at the end.
`MAX_SPEND` (default $1.00) stops new videos from starting once it is passed.
Silent blocks aren't sent. For why German isn't on Granite, see
[German: Granite 5.0 vs Voxtral](#german-granite-50-vs-voxtral).

Transcripts made with `DE=granite` can get Voxtral German afterwards with
`make german OUT=<dir>` (`make german-dry` prices it first). It replaces only
the German and keeps Granite's version under `text_prev`. Transcripts that
already have Voxtral German are skipped.

Voxtral Small is Apache 2.0 and would fit the 16 GB card at 4-bit (`IQ4_XS`,
12.76 GB) under llama.cpp, which would make German local too. That isn't set
up; `DE=granite` is the local option until then.

## Code layout

```
Makefile                 entry points (make help)
requirements.txt         direct dependencies
requirements.lock.txt    full exact environment that make setup installs
transcribe.py            CLI: inputs -> sources -> captions or Granite -> outputs
voxtral_de.py            upgrade older Granite-German transcripts to Voxtral
compare_voxtral.py       OpenRouter client + Granite vs Voxtral A/B on existing blocks
pipeline/
  sources.py             local file discovery; yt-dlp metadata, audio, caption tracks
  captions.py            WebVTT parsing (rolling-caption dedupe) and block packing
  audio.py               ffmpeg extraction, silence detection, pause-aligned blocks
  models.py              which Granite checkpoint (pinned commit) + its download
  asr.py                 Granite Speech wrapper + repetition/passthrough/punctuation detectors
  render.py              HTML, SRT, TXT and index pages
```
