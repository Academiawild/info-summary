"""Subtitle selection and parsing without website access or download state."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import webvtt

from ..models import (
    IGNORED_SUBTITLE_LANGS, LogFn, PREFERRED_SUB_LANGS,
    SUBTITLE_FORMAT_PRIORITY, TranscriptSegment,
)


TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
MAX_SUBTITLE_LANGUAGES = 3


def _pick_language(subtitles: dict[str, Any]) -> str:
    if not subtitles:
        return ""

    available = [language for language in subtitles.keys() if language not in IGNORED_SUBTITLE_LANGS]
    if not available:
        return ""

    for language in PREFERRED_SUB_LANGS:
        if language in available:
            return language
    for language in available:
        if language.startswith("zh"):
            return language
    for language in available:
        if language.startswith("en"):
            return language
    return available[0]


def subtitle_languages(subtitles: dict[str, Any]) -> list[str]:
    available = {key: entries for key, entries in subtitles.items()
                 if key not in IGNORED_SUBTITLE_LANGS and entries}
    languages = []
    while available and len(languages) < MAX_SUBTITLE_LANGUAGES:
        language = _pick_language(available)
        languages.append(language)
        available.pop(language)
    return languages


def subtitle_entry_priority(entry: dict[str, Any]) -> tuple[int, int]:
    ext = (entry.get("ext") or "").lower()
    return (SUBTITLE_FORMAT_PRIORITY.get(ext, 99), len(ext))


def parse_subtitle_file(subtitle_path: Path, log: LogFn | None = None) -> list[TranscriptSegment]:
    log = log or (lambda _: None)
    suffix = subtitle_path.suffix.lower()
    if suffix in {".vtt", ".srt"}:
        try:
            captions = (
                webvtt.from_srt(str(subtitle_path))
                if suffix == ".srt"
                else webvtt.read(str(subtitle_path))
            )
            segments = [
                TranscriptSegment(
                    start=_parse_timestamp(caption.start),
                    end=_parse_timestamp(caption.end),
                    text=_clean_subtitle_text(caption.text),
                )
                for caption in captions
                if _clean_subtitle_text(caption.text)
            ]
            return _deduplicate_segments(segments)
        except Exception as exc:
            log(f"结构化字幕解析失败，尝试兼容解析：{exc}")

    content = subtitle_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    segments: list[TranscriptSegment] = []
    current_start = 0.0
    current_end = 0.0
    current_lines: list[str] = []
    has_timestamp = suffix == ".txt"

    for raw_line in content:
        line = raw_line.strip("\ufeff").strip()
        time_match = TIME_RANGE_RE.search(line)
        if time_match:
            has_timestamp = True
            if current_lines:
                _flush_segment(segments, current_start, current_end, current_lines)
                current_lines = []
            current_start = _parse_timestamp(time_match.group("start"))
            current_end = _parse_timestamp(time_match.group("end"))
            continue

        if not line or line.isdigit() or line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            if current_lines:
                _flush_segment(segments, current_start, current_end, current_lines)
                current_lines = []
            continue

        if has_timestamp:
            current_lines.append(line)

    if current_lines:
        _flush_segment(segments, current_start, current_end, current_lines)

    return _deduplicate_segments(segments)


def _deduplicate_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    deduped: list[TranscriptSegment] = []
    for segment in segments:
        if not deduped:
            deduped.append(segment)
            continue
        previous = deduped[-1]
        if segment.text == previous.text:
            continue
        overlaps = segment.start <= previous.end + 0.5
        if overlaps and segment.text.startswith(previous.text):
            deduped[-1] = TranscriptSegment(previous.start, segment.end, segment.text)
            continue
        if overlaps and previous.text.startswith(segment.text):
            continue
        deduped.append(segment)
    return deduped


def _clean_subtitle_text(text: str) -> str:
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    cleaned = TAG_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _flush_segment(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
    lines: list[str],
) -> None:
    text = _clean_subtitle_text("\n".join(lines))
    if text:
        segments.append(TranscriptSegment(start=start, end=end, text=text))


def _parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
