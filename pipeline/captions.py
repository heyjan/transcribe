"""YouTube caption tracks (WebVTT) -> transcript blocks."""

from __future__ import annotations

import html
import re
from pathlib import Path

_CUE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")


def _seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def vtt_to_cues(path: Path) -> list[dict]:
    """Parse YouTube VTT while removing rolling-caption duplicates.

    Auto-captions show each line twice: once as it is typed out, then again as
    the top line of the next cue. Only the new text of each cue is kept.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cues: list[dict] = []
    i = 0
    previous = ""
    while i < len(lines):
        match = _CUE.search(lines[i])
        if not match:
            i += 1
            continue
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            line = html.unescape(re.sub(r"<[^>]+>", "", lines[i])).strip()
            if line:
                text_lines.append(line)
            i += 1
        text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
        if text and text != previous:
            if previous and text.startswith(previous):
                text = text[len(previous):].strip()
            if text and (not cues or text != cues[-1]["text"]):
                cues.append({
                    "start": _seconds(match.group("start")),
                    "end": _seconds(match.group("end")),
                    "text": text,
                })
            previous = " ".join(text_lines).strip()
    return cues


def cues_to_blocks(cues: list[dict], target: float, max_len: float,
                   min_len: float) -> list[dict]:
    """Pack caption cues into blocks of roughly `target` seconds.

    Same length budget as the audio path, so both kinds of transcript read
    alike. A block closes at a sentence end once it is long enough, at a
    pause between cues, or before it would overrun `max_len`.
    """
    blocks: list[dict] = []
    current: list[dict] = []

    def close() -> None:
        if current:
            blocks.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(c["text"] for c in current),
            })
            current.clear()

    for cue in cues:
        if current:
            span = cue["end"] - current[0]["start"]
            length = current[-1]["end"] - current[0]["start"]
            gap = cue["start"] - current[-1]["end"]
            if span > max_len or (length >= min_len and gap > 1.0):
                close()
        current.append(cue)
        length = current[-1]["end"] - current[0]["start"]
        if length >= target and _SENTENCE_END.search(cue["text"]):
            close()
    close()

    # Rolling cues overlap their successor; clip so spans never overlap.
    for a, b in zip(blocks, blocks[1:]):
        a["end"] = min(a["end"], b["start"])
    return blocks
