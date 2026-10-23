"""Where the audio comes from: local video files, or YouTube via yt-dlp."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
JS_RUNTIMES = ("deno", "node", "bun", "quickjs")
_YT_ID = re.compile(r"[\w-]{11}")


@dataclass
class Source:
    stem: str          # output file stem, unique within an out dir
    title: str
    module: str        # course module, or the YouTube channel
    file: str          # what the footer shows as the source
    media: Path | None = None  # local video, or downloaded audio once fetched
    duration: float | None = None
    url: str | None = None
    info: dict = field(default_factory=dict)  # yt-dlp metadata (YouTube only)

    @property
    def is_youtube(self) -> bool:
        return self.url is not None


def clean_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
    return value[:120] or "transcript"


def prettify(name: str) -> str:
    """`01-From_chatbots_to_AI-enabled_workflows` -> `From chatbots to AI-enabled workflows`."""
    name = re.sub(r"^\d+[-_. ]+", "", name)
    return name.replace("_", " ").strip()


def is_youtube_ref(value: str) -> bool:
    """A URL, or a bare 11-character video ID that isn't also a local path."""
    if "://" in value or value.startswith(("www.", "youtube.com", "youtu.be", "m.youtube.com")):
        return True
    return bool(_YT_ID.fullmatch(value)) and not Path(value).exists()


# --------------------------------------------------------------------- local

def collect_local(inputs: list[Path]) -> list[Source]:
    """Video files given directly or found under directories.

    Stems keep the path below the first directory (`module__lesson`) so a
    course's lessons sort and group the way they do on disk.
    """
    videos: list[Path] = []
    for item in inputs:
        if item.is_dir():
            videos.extend(p for p in sorted(item.rglob("*")) if p.suffix.lower() in VIDEO_EXTS)
        elif item.suffix.lower() in VIDEO_EXTS:
            videos.append(item)
    if not inputs:
        return []
    root = inputs[0] if inputs[0].is_dir() else inputs[0].parent

    sources = []
    for video in videos:
        rel = video.relative_to(root) if root in video.parents else Path(video.name)
        stem = re.sub(r"[^\w.-]+", "_", rel.with_suffix("").as_posix().replace("/", "__"))
        sources.append(Source(
            stem=stem,
            title=prettify(rel.stem),
            module=prettify(rel.parent.name) if rel.parent != Path(".") else "",
            file=rel.as_posix(),
            media=video,
        ))
    return sources


def local_collection(inputs: list[Path]) -> str:
    root = inputs[0] if inputs[0].is_dir() else inputs[0].parent
    return prettify(root.resolve().name)


# ------------------------------------------------------------------- youtube

def js_runtime_options(choice: str) -> list[str]:
    """yt-dlp arguments selecting a JavaScript runtime.

    YouTube extraction needs one, but yt-dlp only enables deno by default and
    fails with `HTTP Error 403: Forbidden` without it, so an installed runtime
    is detected and passed by explicit path.
    """
    if choice == "none":
        return []
    if choice != "auto":
        return ["--js-runtimes", choice]
    for name in JS_RUNTIMES:
        path = shutil.which(name)
        if path:
            return ["--js-runtimes", f"{name}:{path}"]
    return []


def _ytdlp(args: list[str], js: list[str]) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", *js, *args],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        lines = (cp.stderr or cp.stdout).strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"yt-dlp exited {cp.returncode}")
    return cp


def _youtube_url(ref: str) -> str:
    return ref if "://" in ref else (
        f"https://{ref}" if "." in ref else f"https://www.youtube.com/watch?v={ref}"
    )


def collect_youtube(refs: list[str], js: list[str]) -> tuple[list[Source], str | None]:
    """Resolve URLs (videos or playlists) to Sources, plus a playlist title.

    Playlists are listed flat first, so a 200-video playlist costs one request
    to enumerate rather than 200 full extractions up front.
    """
    sources: list[Source] = []
    playlist_title = None
    for ref in refs:
        url = _youtube_url(ref)
        listing = json.loads(_ytdlp(["--flat-playlist", "--dump-single-json", url], js).stdout)
        if listing.get("_type") == "playlist":
            playlist_title = playlist_title or listing.get("title")
            urls = [e.get("url") or _youtube_url(e["id"]) for e in listing.get("entries") or []]
        else:
            urls = [url]
        for video_url in urls:
            info = json.loads(_ytdlp(["--dump-single-json", video_url], js).stdout)
            sources.append(Source(
                stem=clean_name(f"{info.get('title', 'video')}-{info.get('id', '')}"),
                title=info.get("title") or info.get("id", "video"),
                module=info.get("channel") or info.get("uploader") or "",
                file=info.get("webpage_url") or video_url,
                duration=info.get("duration"),
                url=info.get("webpage_url") or video_url,
                info=info,
            ))
    return sources, playlist_title


def downloaded_audio(src: Source, dest: Path) -> Path | None:
    found = sorted(p for p in dest.glob(f"{src.stem}.*") if not p.name.endswith(".part"))
    return found[0] if found else None


def download_audio(src: Source, dest: Path, js: list[str]) -> Path:
    """Fetch the best audio-only stream as is; ffmpeg decodes it later.

    Kept on disk (not a temp dir) because it is this video's source: re-runs
    and `voxtral_de.py` re-read it from `source.path`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if existing := downloaded_audio(src, dest):
        return existing
    _ytdlp(["-f", "bestaudio/best", "-o", str(dest / f"{src.stem}.%(ext)s"), src.url], js)
    if found := downloaded_audio(src, dest):
        return found
    raise RuntimeError("yt-dlp reported success but wrote no audio file")


def pick_caption_track(info: dict, lang: str) -> tuple[str, str] | None:
    """Choose a caption track for `lang` as (kind, track key), or None.

    Uploaded subtitles in that language are always fine. Automatic captions
    only count in the video's *spoken* language: YouTube also offers every
    other language as a machine translation of its auto-captions, and those
    are worse than translating from the audio.
    """
    def matches(key: str) -> bool:
        return key == lang or key.startswith(f"{lang}-")

    manual = [k for k in (info.get("subtitles") or {}) if matches(k) and k != "live_chat"]
    if manual:
        return "manual", (lang if lang in manual else manual[0])

    auto = info.get("automatic_captions") or {}
    if f"{lang}-orig" in auto:
        return "auto", f"{lang}-orig"
    spoken = (info.get("language") or "").split("-")[0]
    if spoken == lang and lang in auto:
        return "auto", lang
    return None


def download_captions(src: Source, kind: str, key: str, workdir: Path,
                      js: list[str]) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    flag = "--write-subs" if kind == "manual" else "--write-auto-subs"
    _ytdlp([
        "--skip-download", flag, "--sub-format", "vtt",
        "--sub-langs", re.escape(key),
        "-o", str(workdir / f"{src.stem}.%(ext)s"), src.url,
    ], js)
    found = sorted(workdir.glob(f"{src.stem}*.vtt"))
    if not found:
        raise RuntimeError(f"no {key} captions were written")
    return found[0]
