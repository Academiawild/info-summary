from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .models import AUDIO_FILE_EXTENSIONS, SourceType


SUPPORTED_VIDEO_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "b23.tv",
    "douyin.com",
    "iesdouyin.com",
    "ixigua.com",
    "vimeo.com",
}

_SUPPORTED_HOST = r"(?:[a-z0-9-]+\.)*(?:" + "|".join(
    re.escape(domain) for domain in sorted(SUPPORTED_VIDEO_DOMAINS)
) + r")"
# 分享文案中的中文和包裹标点是链接边界，查询参数中的标点保持原样。
# 英文逗号等只在其后紧接另一条链接时分隔，避免截断正常参数。
_URL_CHARACTERS = (
    r"(?![,;|](?:[a-z][a-z0-9+.-]*://|" + _SUPPORTED_HOST + r"(?![a-z0-9_.-])))"
    r'''[^\s<>"'，。！？；：、（）【】《》「」『』“”‘’\u3400-\u9fff]'''
)
_URL_CANDIDATES = re.compile(
    r"(?<![a-z0-9_@])(?:[a-z][a-z0-9+.-]*://(?:" + _URL_CHARACTERS + r")*)"
    r"|(?<![a-z0-9_@./:-])" + _SUPPORTED_HOST
    + r"(?![a-z0-9_.-])(?:" + _URL_CHARACTERS + r")*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceIssue:
    line_number: int
    text: str
    message: str


@dataclass
class SourceBatch:
    sources: list[tuple[SourceType, str]]
    issues: list[SourceIssue]


def strip_wrapped_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def normalized_hostname(url: str) -> str:
    try:
        hostname = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return hostname.removeprefix("www.")


def domain_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


def is_supported_video_url(value: str) -> bool:
    text = strip_wrapped_quotes(value)
    try:
        parsed = urlparse(text)
        # 读取 port 也会验证端口格式，避免把错误链接送到下载流程。
        parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or any(character.isspace() or ord(character) < 32 for character in text)
    ):
        return False
    hostname = normalized_hostname(text)
    return any(domain_matches(hostname, domain) for domain in SUPPORTED_VIDEO_DOMAINS)


def is_youtube_url(value: str) -> bool:
    hostname = normalized_hostname(value)
    return domain_matches(hostname, "youtube.com") or domain_matches(hostname, "youtu.be")


def resolve_local_path(value: str, project_dir: Path) -> Path | None:
    text = strip_wrapped_quotes(value)
    if not text:
        return None
    try:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (project_dir / path).resolve()
        return path if path.exists() else None
    except (OSError, ValueError, RuntimeError):
        # 长分享文案、无效字符和无法展开的路径都属于输入问题。
        return None


def _source_from_path(path: Path) -> tuple[SourceType, str]:
    if path.is_file():
        if path.suffix.lower() not in AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"当前文件不是支持的音频格式：{path}")
        return SourceType.AUDIO_FILE, str(path)
    if path.is_dir():
        return SourceType.AUDIO_FOLDER, str(path)
    raise ValueError(f"无法识别输入类型：{path}")


def _decode_file_url(text: str) -> str:
    try:
        parsed = urlparse(text)
    except ValueError as exc:
        raise ValueError("file:// 路径格式不正确，请重新选择本机音频文件或文件夹。") from exc
    try:
        if parsed.netloc.lower() not in {"", "localhost"}:
            raise ValueError("只支持本机的 file:// 路径，请选择本机音频文件或文件夹。")
        if (
            not parsed.path.startswith("/")
            or parsed.query
            or parsed.fragment
            or re.search(r"%(?![0-9a-fA-F]{2})", parsed.path)
        ):
            raise ValueError("file:// 路径格式不正确，请重新选择本机音频文件或文件夹。")
        path = unquote(parsed.path, errors="strict")
        if "\0" in path:
            raise ValueError("file:// 路径含有无效字符，请重新选择音频文件或文件夹。")
        return path
    except UnicodeDecodeError as exc:
        raise ValueError("file:// 路径编码不正确，请重新选择音频文件或文件夹。") from exc


def _clean_url_candidate(text: str) -> str:
    # 包裹链接的括号可以去掉；参数本身使用的括号和标点不能去掉。
    unwrapped = text.rstrip(".,;!")
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        while unwrapped.endswith(closing) and unwrapped.count(closing) > unwrapped.count(opening):
            unwrapped = unwrapped[:-1]
            text = unwrapped
    if "?" not in text and "#" not in text:
        text = text.rstrip(".,;!")
    return text if "://" in text else "https://" + text


def detect_source(value: str, project_dir: Path) -> tuple[SourceType, str]:
    text = strip_wrapped_quotes(value)
    if not text:
        raise ValueError("请填写视频链接、音频文件路径或音频文件夹路径。")
    # 文件名也可能包含网站域名或分享文字，已有本地路径始终优先。
    path = resolve_local_path(text, project_dir)
    if path is not None:
        return _source_from_path(path)
    if text.lower().startswith("file:"):
        local_text = _decode_file_url(text)
        path = resolve_local_path(local_text, project_dir)
        if path is None:
            raise ValueError(f"找不到输入路径：{local_text}")
        return _source_from_path(path)

    candidates = [_clean_url_candidate(match.group()) for match in _URL_CANDIDATES.finditer(text)]
    video_urls = [url for url in candidates if is_supported_video_url(url)]
    if len(video_urls) > 1:
        raise ValueError("请每次只粘贴一个视频链接；多个音频可以选择文件夹处理。")
    if video_urls:
        return SourceType.VIDEO_URL, video_urls[0]
    if candidates or re.search(r"(?:https?:|https?//|www\.)", text, re.IGNORECASE):
        raise ValueError("无法识别视频链接，请检查格式；当前支持 YouTube、哔哩哔哩、抖音、西瓜视频和 Vimeo。")
    if len(text) > 240 or "\n" in text:
        raise ValueError("未找到可用的视频链接或音频路径，请检查后重试。")
    raise ValueError(f"找不到输入路径：{text}")


def parse_sources(value: str, project_dir: Path) -> SourceBatch:
    """Collect usable inputs and report problems without discarding the batch."""
    text = value.strip()
    if not text:
        return SourceBatch([], [SourceIssue(1, "", "请填写视频链接、音频文件路径或音频文件夹路径。")])
    first_line_number = next(number for number, line in enumerate(value.splitlines(), 1) if line.strip())
    # A real filename can contain line breaks, domains, or other share-like text.
    path = resolve_local_path(text, project_dir)
    if path is not None:
        try:
            return SourceBatch([_source_from_path(path)], [])
        except ValueError as exc:
            return SourceBatch([], [SourceIssue(first_line_number, text, str(exc))])

    sources: list[tuple[SourceType, str]] = []
    issues: list[SourceIssue] = []
    seen: set[tuple[SourceType, str]] = set()

    def add(source: tuple[SourceType, str]) -> None:
        kind, location = source
        identity = (kind, str(Path(location).resolve()) if kind != SourceType.VIDEO_URL else location)
        if identity not in seen:
            seen.add(identity)
            sources.append(source)

    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = strip_wrapped_quotes(raw_line)
        if not line:
            continue
        try:
            path = resolve_local_path(line, project_dir)
            if path is not None:
                add(_source_from_path(path))
                continue
            matches = list(_URL_CANDIDATES.finditer(line))
            # A file URI may contain spaces or Chinese characters after decoding.
            # Only parse the complete line as a URI when no other URL shares it.
            if line.lower().startswith("file:") and not any(
                not match.group().lower().startswith("file:") for match in matches
            ) and len(matches) <= 1:
                add(detect_source(line, project_dir))
                continue

            if matches:
                remaining = strip_wrapped_quotes(_URL_CANDIDATES.sub("", line))
                malformed_link = re.search(r"(?:https?:|https?//|ftp:|www\.)", remaining, re.IGNORECASE)
                if malformed_link:
                    issues.append(SourceIssue(line_number, remaining, "无法识别视频链接，请检查链接格式和视频网站。"))
                has_file_url = any(match.group().lower().startswith("file:") for match in matches)
                mixed_path = (
                    has_file_url
                    or remaining.lower().startswith("file:")
                    or (not malformed_link and _looks_like_path(remaining))
                    or re.search(r"(?:^|\s)(?:~?/(?!/)|\.{1,2}/|[a-zA-Z]:[\\/])\S+", remaining)
                )
                if mixed_path:
                    issues.append(SourceIssue(
                        line_number, line,
                        "视频链接和音频路径混在同一行，请将每个音频文件或文件夹路径单独放在一行。",
                    ))
                for match in matches:
                    url = _clean_url_candidate(match.group())
                    if url.lower().startswith("file:"):
                        continue
                    if not is_supported_video_url(url):
                        issues.append(SourceIssue(
                            line_number, match.group(),
                            f"无法识别视频链接：{url}；请使用支持的视频网站链接。",
                        ))
                    else:
                        add((SourceType.VIDEO_URL, url))
                continue
            if re.search(r"(?:https?:|https?//|ftp:|www\.)", line, re.IGNORECASE):
                raise ValueError("无法识别视频链接，请检查链接格式和视频网站。")
            if _looks_like_path(line):
                raise ValueError(f"找不到输入路径：{line}")
            # Headings and share descriptions may sit on their own line.
        except ValueError as exc:
            issues.append(SourceIssue(line_number, line, str(exc)))

    if not sources and not issues:
        # Keep the familiar actionable error for an entirely invalid input.
        try:
            add(detect_source(text, project_dir))
        except ValueError as exc:
            issues.append(SourceIssue(first_line_number, text, str(exc)))
    return SourceBatch(sources, issues)


def detect_sources(value: str, project_dir: Path) -> list[tuple[SourceType, str]]:
    """Keep the strict interface for callers that require every input to be valid."""
    batch = parse_sources(value, project_dir)
    if batch.issues:
        issue = batch.issues[0]
        raise ValueError(f"第 {issue.line_number} 行：{issue.message}")
    return batch.sources


def _looks_like_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~/", "\\")) or re.match(r"^[a-zA-Z]:[\\/]", value):
        return True
    if not any(character.isspace() for character in value) and ("/" in value or "\\" in value):
        return True
    common_extensions = AUDIO_FILE_EXTENSIONS | {".txt", ".pdf", ".doc", ".docx", ".csv", ".mp4", ".mov", ".zip", ".7z"}
    if Path(value).suffix.lower() in common_extensions:
        return True
    # A description ending in a domain or version number is not a file path.
    return bool(
        not any(character.isspace() for character in value)
        and ":" not in value and "：" not in value
        and re.search(r"\.[a-zA-Z][a-zA-Z0-9]{0,7}$", value)
    )


def list_audio_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_FILE_EXTENSIONS
    )
