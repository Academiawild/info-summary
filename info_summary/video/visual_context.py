"""Keep model context bounded without dropping timed transcript or note text."""
from __future__ import annotations

import json

from ..models import TranscriptSegment, VisualNote


_OMITTED = "…（已省略）"


def _positive_limit(limit: int) -> int:
    if type(limit) is not int or limit <= 0:
        raise ValueError("上下文字数必须是正整数。")
    return limit


def split_timed_segments(segments: list[TranscriptSegment], limit: int) -> list[str]:
    """Return complete, contiguous text pieces, each with its original times.

    A timestamp label alone may exceed an unusually small limit; in that case
    retain that label and at least one text character rather than lose content.
    """
    _positive_limit(limit)
    result = []
    for segment in segments:
        label = f"[{segment.start}-{segment.end}] "
        length = max(1, limit - len(label))
        if not segment.text:
            result.append(label)
        else:
            result.extend(label + segment.text[index:index + length]
                          for index in range(0, len(segment.text), length))
    return result


def excerpt(text: str, limit: int) -> str:
    """Return at most limit characters, explicitly marking omitted text."""
    if type(limit) is not int or limit < 0:
        raise ValueError("摘录字数不能小于零。")
    if len(text) <= limit:
        return text
    if limit == 0:
        return ""
    if limit < len(_OMITTED):
        return "…"
    return text[:limit - len(_OMITTED)] + _OMITTED


def nearby(
    segments: list[TranscriptSegment], start: float, end: float | None = None,
    limit: int = 1200,
) -> str:
    """Return bounded subtitles overlapping the interval or eight seconds nearby."""
    end = start if end is None else end
    rows = [f"[{segment.start}-{segment.end}] {segment.text}" for segment in segments
            if segment.end >= start - 8 and segment.start <= end + 8]
    return excerpt("\n".join(rows), limit)


def _json_size(rows: list[dict]) -> int:
    # Match json.dumps' default spacing, including escaping of newlines/quotes.
    return len(json.dumps(rows, ensure_ascii=False))


def review_groups(
    notes: list[VisualNote], segments: list[TranscriptSegment], limit: int,
) -> list[list[dict]]:
    """Group rows of {time, description, transcript} by their actual JSON size.

    Every description character survives, in its original order and at its
    original time. Split descriptions carry nearby subtitles on their first
    row only; identical nearby excerpts are included once across this call.
    The minimum useful group budget is 200 characters.
    """
    budget = max(200, _positive_limit(limit))
    context_limit = min(1200, budget // 4)
    groups: list[list[dict]] = []
    current: list[dict] = []
    used_contexts: set[str] = set()
    for note in notes:
        context = nearby(segments, note.timestamp, note.end, limit=context_limit)
        if context in used_contexts:
            context = ""
        elif context:
            used_contexts.add(context)
        offset = 0
        first = True
        while first or offset < len(note.description):
            transcript = context if first else ""
            row = {"time": note.timestamp, "description": "", "transcript": transcript}
            # Control characters can expand sixfold when encoded as JSON.
            # Keep room for at least one complete description character.
            probe = note.description[offset:offset + 1]
            while _json_size([dict(row, description=probe)]) > budget and row["transcript"]:
                row["transcript"] = excerpt(row["transcript"], len(row["transcript"]) // 2)
            remaining = len(note.description) - offset
            low, high = 0, min(remaining, budget)
            while low < high:
                middle = (low + high + 1) // 2
                candidate = dict(row, description=note.description[offset:offset + middle])
                if _json_size([candidate]) <= budget:
                    low = middle
                else:
                    high = middle - 1
            if remaining and low == 0:
                raise ValueError("画面时间标签过长，无法放入上下文。")
            row["description"] = note.description[offset:offset + low]
            if current and _json_size(current + [row]) > budget:
                groups.append(current)
                current = []
            current.append(row)
            offset += low
            first = False
    if current:
        groups.append(current)
    return groups
