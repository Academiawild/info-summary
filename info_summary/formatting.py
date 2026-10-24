"""Shared readable transcript paragraphs and time labels for every exporter."""
from __future__ import annotations

import re

from .models import TranscriptSegment, VisualNote


FRAME_REFERENCE_RE = re.compile(r"\[\[frame:([1-9][0-9]{0,8})\]\]")


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def readable_summary(text: str, notes: list[VisualNote]) -> str:
    """Keep image references readable in exports without an inline reader."""
    def label(match: re.Match) -> str:
        index = int(match[1]) - 1
        return (f"（画面 {format_timestamp(notes[index].timestamp)}）"
                if index < len(notes) else "（对应画面未取得）")
    return FRAME_REFERENCE_RE.sub(label, text)


def readable_original(segments: list[TranscriptSegment]) -> str:
    paragraphs: list[str] = []
    current = ""
    last_end = 0.0
    for segment in segments:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if not text:
            continue
        if current and (segment.start - last_end >= 1.2 or len(current) + len(text) > 280):
            paragraphs.append(current.strip())
            current = ""
        separator = " " if current and current[-1:].isascii() and text[:1].isascii() else ""
        current += separator + text
        last_end = segment.end
    if current:
        paragraphs.append(current.strip())
    return "\n\n".join(paragraphs)
