"""Audio extraction and silence-aware segmentation via ffmpeg."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Granite's speech encoder consumes 16 kHz mono; anything else gets resampled
# internally, so we hand it exactly that and skip a lossy round-trip.
MODEL_SR = 16000


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(path: Path) -> float:
    cp = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ])
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{cp.stderr.strip()}")
    return float(cp.stdout.strip())


def extract_audio(video: Path, wav_out: Path, mp3_out: Path | None,
                  mp3_bitrate: str = "96k") -> None:
    """Write the model-input WAV and, optionally, an mp3 keep-copy.

    Both come from the video in one pass each rather than mp3 -> wav, so the
    transcript never sees MP3 compression artifacts.
    """
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cp = _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(MODEL_SR),
        "-c:a", "pcm_s16le", str(wav_out),
    ])
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg wav extraction failed:\n{cp.stderr.strip()}")

    if mp3_out is not None:
        encode_mp3(video, mp3_out, mp3_bitrate)


def encode_mp3(media: Path, mp3_out: Path, bitrate: str = "96k") -> None:
    """The HTML player's audio: mono speech, so 96k is plenty."""
    mp3_out.parent.mkdir(parents=True, exist_ok=True)
    cp = _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(media),
        "-vn", "-ac", "1", "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3_out),
    ])
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 encode failed:\n{cp.stderr.strip()}")


_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(wav: Path, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    """Return [(start, end)] silence intervals as reported by ffmpeg."""
    cp = _run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(wav),
        "-af", f"silencedetect=n={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ])
    # silencedetect reports on stderr regardless of exit status.
    starts, ends = [], []
    for line in cp.stderr.splitlines():
        if m := _SIL_START.search(line):
            starts.append(float(m.group(1)))
        if m := _SIL_END.search(line):
            ends.append(float(m.group(1)))
    # A trailing silence that runs to EOF has no silence_end; drop it, the
    # segmenter clamps to duration anyway.
    return [(s, e) for s, e in zip(starts, ends) if e > s]


@dataclass
class Block:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_blocks(
    duration: float,
    silences: list[tuple[float, float]],
    target: float,
    max_len: float,
    min_len: float,
) -> list[Block]:
    """Greedily cut the timeline at pauses, keeping blocks near `target`.

    Cutting mid-sentence hurts translation far more than transcription, so we
    only ever hard-cut when a stretch of speech genuinely has no pause inside
    the allowed window.
    """
    # Prefer to cut in the middle of a pause so neither side loses a word onset.
    candidates = [min(s + (e - s) / 2.0, duration) for s, e in silences]
    candidates = [c for c in candidates if 0.0 < c < duration]

    blocks: list[Block] = []
    cursor = 0.0
    idx = 0
    while duration - cursor > 0.05:
        remaining = duration - cursor
        if remaining <= max_len:
            blocks.append(Block(idx, cursor, duration))
            break

        window = [c for c in candidates if cursor + min_len <= c <= cursor + max_len]
        if window:
            # Whichever pause sits closest to the target length.
            cut = min(window, key=lambda c: abs((c - cursor) - target))
        else:
            cut = cursor + max_len  # unbroken speech: hard cut
        blocks.append(Block(idx, cursor, cut))
        cursor = cut
        idx += 1
    return blocks


def segment(
    wav: Path,
    duration: float,
    target: float = 24.0,
    max_len: float = 30.0,
    min_len: float = 6.0,
) -> tuple[list[Block], dict]:
    """Segment with progressively looser silence thresholds.

    Studio-narrated content (our case) pauses briefly and quietly; noisier
    sources need a higher threshold before any pause registers at all. We take
    the first setting that yields enough cut points to actually drive the
    planner.
    """
    wanted = max(1, int(duration // target))
    attempts = [(-32.0, 0.30), (-28.0, 0.25), (-24.0, 0.20), (-20.0, 0.15)]
    chosen, silences = attempts[-1], []
    for noise_db, min_dur in attempts:
        found = detect_silences(wav, noise_db, min_dur)
        if len(found) >= wanted:
            chosen, silences = (noise_db, min_dur), found
            break
        if len(found) > len(silences):
            chosen, silences = (noise_db, min_dur), found

    blocks = plan_blocks(duration, silences, target, max_len, min_len)
    meta = {
        "method": "ffmpeg silencedetect + greedy pause-aligned packing",
        "noise_db": chosen[0],
        "min_silence": chosen[1],
        "silences_found": len(silences),
        "target": target,
        "max_len": max_len,
        "min_len": min_len,
        "hard_cuts": sum(
            1 for b in blocks
            if b.duration >= max_len - 1e-6 and b.end < duration - 1e-6
        ),
    }
    return blocks, meta
