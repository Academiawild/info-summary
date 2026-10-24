"""A portable, offline reader for the material already produced by a task."""
from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .models import ProcessingArtifact, ProcessingRequest, ProcessingStep, VisualNote
from .formatting import FRAME_REFERENCE_RE, format_timestamp


REPORT_FILENAME = "阅读报告.html"
_ASSETS = Path(__file__).with_name("report_assets")
_NO_ORIGINAL = "没有可用的字幕或语音原文。"


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _inline(text: str) -> str:
    # Escape first. The only markup interpreted is emphasis and inline code;
    # model responses and transcripts can never supply HTML or executable URLs.
    escaped = _escape(text)
    pieces = re.split(r"(`[^`\n]+`)", escaped)
    return "".join(
        "<code>" + part[1:-1] + "</code>" if part.startswith("`") and part.endswith("`")
        else re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", part)
        for part in pieces
    )


def _text_blocks(text: str) -> str:
    """Render ordinary paragraphs, headings and lists without accepting HTML."""
    blocks: list[str] = []
    paragraph: list[str] = []
    list_kind = ""

    def flush() -> None:
        if paragraph:
            blocks.append('<p data-search>' + "<br>".join(_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            blocks.append(f"</{list_kind}>")
            list_kind = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush()
            close_list()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        bullet = re.match(r"^[-*+]\s+(.+)$", line)
        numbered = re.match(r"^\d+[.)、]\s+(.+)$", line)
        if heading:
            flush()
            close_list()
            level = min(4, len(heading[1]) + 2)
            blocks.append(f'<h{level} data-search>{_inline(heading[2])}</h{level}>')
        elif bullet or numbered:
            flush()
            kind = "ul" if bullet else "ol"
            if kind != list_kind:
                close_list()
                blocks.append(f"<{kind}>")
                list_kind = kind
            item = (bullet or numbered)[1]
            blocks.append(f'<li data-search>{_inline(item)}</li>')
        else:
            close_list()
            paragraph.append(line)
    flush()
    close_list()
    return "\n".join(blocks)


def _original(artifact: ProcessingArtifact) -> str:
    rows = []
    for segment in artifact.segments:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if text:
            label = f"{format_timestamp(segment.start)} - {format_timestamp(segment.end)}"
            rows.append(
                '<div class="transcript-row"><span class="time-label" data-search>' + label
                + '</span><p data-search>' + _escape(text) + "</p></div>"
            )
    return "\n".join(rows) if rows else _text_blocks(artifact.original_text.strip() or _NO_ORIGINAL)


def _source(source: str) -> str:
    try:
        parsed = urlsplit(source)
        is_link = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
    except ValueError:
        is_link = False
    if is_link:
        return f'<a href="{_escape(source)}" target="_blank" rel="noopener noreferrer">{_escape(source)}</a>'
    return _escape(source)


def _integrated_content(
    artifact: ProcessingArtifact,
    images: list[tuple[VisualNote, Path | None]],
    figure: Callable[[int, bool], str],
) -> tuple[str, list[tuple[str, str]]]:
    """Place only explicitly referenced evidence beside its explanation.

    Old summaries without references remain available, but are never matched
    to pictures by keyword similarity or by their position in a bullet list.
    """
    topics: list[tuple[str, str]] = []
    used: set[int] = set()

    def narrative(text: str) -> str:
        def heading(match: re.Match) -> str:
            key = f"topic-{len(topics) + 1}"
            title = html.unescape(re.sub(r"<[^>]*>", "", match[2]))
            topics.append((key, title))
            return f'<h{match[1]} id="{key}" data-search>{match[2]}</h{match[1]}>'
        return re.sub(r"<h([34]) data-search>(.*?)</h\1>", heading, _text_blocks(text))

    def referenced(index: int) -> str:
        if index >= len(images):
            warning = f"图文讲解引用的第 {index + 1} 张画面未取得，已保留文字内容。"
            if warning not in artifact.warnings:
                artifact.warnings.append(warning)
            return '<p class="missing" data-search>对应画面未取得，以下保留文字说明。</p>'
        if index in used:
            label = format_timestamp(images[index][0].timestamp)
            return f'<p data-search><a href="#frame-{index + 1}">参见 {label} 的画面</a></p>'
        used.add(index)
        return figure(index, True)

    text = artifact.summary_text.strip()
    matches = list(FRAME_REFERENCE_RE.finditer(text))
    body: list[str] = []
    if matches:
        cursor = 0
        for match in matches:
            body.extend((narrative(text[cursor:match.start()]), referenced(int(match[1]) - 1)))
            cursor = match.end()
        body.append(narrative(text[cursor:]))
    else:
        # Only bracketed video times are accepted as legacy evidence links.
        # A bare clock time in a news quote is not a picture association.
        for paragraph in re.split(r"\n\s*\n", text):
            body.append(narrative(paragraph))
            times = re.findall(r"\[(?:画面\s*)?((?:\d{2}:)?\d{2}:\d{2})\]", paragraph)
            for index, (note, _) in enumerate(images):
                if index not in used and format_timestamp(note.timestamp) in times:
                    used.add(index)
                    body.append(figure(index, False))
        if not used:
            # With no trustworthy paragraph links, the saved picture notes are
            # the primary chronological explanation. Keep the old prose intact
            # as a folded reference rather than pretending it was illustrated.
            body.clear()
            topics.clear()
            first = re.split(r"\n\s*\n", text, maxsplit=1)[0]
            lead = first if first and len(first) <= 300 and not re.match(r"[#*+-]|\d+[.)、]", first) else ""
            body.append(narrative(lead or ("本次未取得完整总结，请查看处理提示后继续。" if not text else "")))
            for index in range(len(images)):
                body.append(figure(index, False))
                used.add(index)
            if text and text != lead:
                body.append('<details class="supporting-notes"><summary>原有文字总结（保留供核对）</summary>'
                            + _text_blocks(text) + '</details>')
    remaining = [index for index in range(len(images)) if index not in used]
    if remaining:
        body.append('<details class="supporting-notes"><summary>补充画面与核对说明</summary>')
        body.extend(figure(index, False) for index in remaining)
        body.append('</details>')
    if artifact.visual_overview.strip():
        body.append('<details class="supporting-notes"><summary>画面覆盖与未确定内容</summary>'
                    + _text_blocks(artifact.visual_overview) + '</details>')
    return "\n".join(body), topics


def render_report(
    artifact: ProcessingArtifact,
    request: ProcessingRequest,
    images: list[tuple[VisualNote, Path | None]],
) -> str:
    """Return a complete document; images and reader assets are embedded."""
    sections: list[tuple[str, str, str]] = []
    image_data: dict[Path, str | None] = {}
    def figure(index: int, compact: bool = False) -> str:
        note, image_path = images[index]
        label = format_timestamp(note.timestamp)
        if note.end is not None:
            label += " - " + format_timestamp(note.end)
        visual = [f'<figure id="frame-{index + 1}"><figcaption data-search>{label}</figcaption>']
        if image_path is not None and image_path not in image_data:
            try:
                content = image_path.read_bytes()
                if not content:
                    raise ValueError("empty image")
                mime = "image/png" if content.startswith(b"\x89PNG") else "image/jpeg"
                image_data[image_path] = f"data:{mime};base64," + base64.b64encode(content).decode("ascii")
            except (OSError, ValueError):
                image_data[image_path] = None
                artifact.warnings.append(f"{label} 的截图无法嵌入阅读报告，已保留文字解读。")
        if image_path is not None and image_data.get(image_path):
            visual.append(
                f'<button class="image-expand" type="button" aria-label="放大 {label} 的画面">'
                f'<img src="{image_data[image_path]}" alt="{label} 的视频画面" loading="lazy">'
                '</button><span class="image-hint">点击图片放大</span>'
            )
        else:
            visual.append('<p class="missing" data-search>截图未能保存，以下保留已经取得的文字解读。</p>')
        description = _text_blocks(note.description.strip() or "此画面尚未取得可靠解读。")
        if compact:
            description = ('<details class="frame-detail"><summary>查看画面核对说明</summary>'
                           + description + '</details>')
        visual.extend([description, "</figure>"])
        return "\n".join(visual)

    topics: list[tuple[str, str]] = []
    if request.has_step(ProcessingStep.SUMMARY):
        if request.include_visual:
            body, topics = _integrated_content(artifact, images, figure)
            sections.append(("summary", "图文讲解", body))
        else:
            sections.append(("summary", "内容总结", _text_blocks(
                artifact.summary_text.strip() or "本次未取得完整总结，请查看处理提示后继续。"
            )))
    elif request.include_visual:
        visual = [_text_blocks(artifact.visual_overview.strip() or "带时间的截图与解读，用于补充理解视频画面。")]
        visual.extend(figure(index) for index in range(len(images)))
        if not images:
            visual.append('<p class="missing" data-search>未取得可用截图，请查看处理提示后继续。</p>')
        sections.append(("visual", "画面笔记", "\n".join(visual)))
    if request.has_step(ProcessingStep.POLISH):
        sections.append(("polish", "润色稿", _text_blocks(
            artifact.polished_text.strip() or "本次未取得完整润色稿，请查看处理提示后继续。"
        )))
    if request.has_step(ProcessingStep.ORIGINAL):
        sections.append(("original", "原文稿", _original(artifact)))
    warnings = list(dict.fromkeys(artifact.warnings))
    notice = ""
    if warnings:
        notice = '<aside class="notice" aria-label="处理提示"><strong>部分完成 · 已保留可用内容</strong><ul>'
        notice += "".join(f'<li data-search>{_escape(warning)}</li>' for warning in warnings)
        notice += "</ul></aside>"
    navigation = "".join(
        f'<a href="#{key}">{title}</a>' + (
            "".join(f'<a class="topic-link" href="#{topic}">{_escape(label)}</a>' for topic, label in topics)
            if key == "summary" else ""
        ) for key, title, _ in sections
    )
    content = "\n".join(
        f'<details class="report-section" id="{key}"' + (" open" if index == 0 or key != "original" else "")
        + f'><summary><h2>{title}</h2><span class="fold-hint">展开 / 收起</span></summary>'
        + f'<div class="section-body">{body}</div></details>'
        for index, (key, title, body) in enumerate(sections)
    )
    css = (_ASSETS / "report.css").read_text(encoding="utf-8")
    script = (_ASSETS / "report.js").read_text(encoding="utf-8")
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    state = "部分完成" if warnings else "已完成"
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{_escape(artifact.title)} · 阅读报告</title><style>{css}</style></head>
<body><header class="reader-bar"><a class="brand" href="#top">信息整理 <span>离线阅读报告</span></a>
<div class="search" role="search"><label class="sr-only" for="search-input">全文搜索</label>
<input id="search-input" type="search" placeholder="搜索全文…" autocomplete="off">
<output id="search-count" aria-live="polite">输入关键词</output>
<button id="search-prev" type="button" aria-label="上一处匹配" disabled>↑</button>
<button id="search-next" type="button" aria-label="下一处匹配" disabled>↓</button></div>
<button id="print-report" class="print-button" type="button">打印 / 保存 PDF</button></header>
<div class="reader-layout"><nav class="contents" aria-label="报告目录"><span>本篇内容</span>{navigation}
<p>可离线阅读<br>⌘F / Ctrl+F 搜索全文</p></nav>
<main id="top"><header class="report-heading"><p class="eyebrow">阅读报告 · {state}</p>
<h1 data-search>{_escape(artifact.title)}</h1><p class="source" data-search>来源：{_source(artifact.source)}</p>
<p class="created">生成于 {created} · 内容与截图已保存在本文件中</p></header>
{notice}{content}<footer>信息整理工具 · 图文讲解与原文均可查阅核对。</footer></main></div>
<dialog id="image-viewer" aria-label="放大查看截图"><div class="viewer-toolbar"><span id="viewer-caption"></span>
<button id="close-viewer" type="button">关闭 ×</button></div><img id="viewer-image" alt=""></dialog>
<script>{script}</script></body></html>'''
