from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


LogFn = Callable[[str], None]

AUDIO_FILE_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".m4b", ".amr"}
SUBTITLE_EXTENSIONS = {".vtt", ".srt", ".txt"}
SUBTITLE_FORMAT_PRIORITY = {"vtt": 0, "srt": 1, "srv3": 2, "srv2": 3, "srv1": 4, "ttml": 5}
SUBTITLE_TIMEOUT_SECONDS = 120
PREFERRED_SUB_LANGS = ["zh-CN", "zh-Hans", "zh-Hant", "zh-TW", "zh-HK", "zh", "en", "en-US"]
IGNORED_SUBTITLE_LANGS = {"live_chat"}


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DownloadOptions:
    cookies_from_browser: str = ""
    cookies_file: str = ""
    js_runtime: str = "deno"
    js_runtime_path: str = ""


class SourceType(StrEnum):
    VIDEO_URL = "video_url"
    AUDIO_FILE = "audio_file"
    AUDIO_FOLDER = "audio_folder"


class ProcessingStep(StrEnum):
    ORIGINAL = "original"
    POLISH = "polish"
    SUMMARY = "summary"


class VideoQuality(StrEnum):
    BEST = "best"
    HD_1080 = "1080"
    HD_720 = "720"


class TextProviderKind(StrEnum):
    OPENCODE = "opencode"
    OLLAMA = "ollama"


class OutputLanguage(StrEnum):
    SIMPLIFIED_CHINESE = "zh-CN"
    FOLLOW_SOURCE = "source"
    ENGLISH = "en"


@dataclass(frozen=True)
class ProcessingRequest:
    source: str
    source_type: SourceType
    output_dir: str = ""
    steps: tuple[ProcessingStep, ...] = (
        ProcessingStep.ORIGINAL,
        ProcessingStep.POLISH,
        ProcessingStep.SUMMARY,
    )
    export_formats: tuple[str, ...] = ("docx", "md")
    model_size: str = "turbo"
    language_hint: str = ""
    device_mode: str = "cpu"
    compute_type: str = ""
    batch_size: int = 8
    text_provider: TextProviderKind = TextProviderKind.OPENCODE
    text_model_name: str = ""
    text_model_base_url: str = ""
    text_model_chunk_chars: int = 3500
    output_language: OutputLanguage = OutputLanguage.SIMPLIFIED_CHINESE
    cookies_from_browser: str = ""
    cookies_file: str = ""
    js_runtime: str = "deno"
    js_runtime_path: str = ""
    include_visual: bool = False
    visual_focus: str = ""

    def has_step(self, step: ProcessingStep) -> bool:
        return step in self.steps

    def source_fingerprint(self, project_dir: Path) -> str:
        if self.source_type == SourceType.VIDEO_URL:
            parsed = urlsplit(self.source.strip())
            normalized = urlunsplit(
                (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, "")
            )
            payload = f"url:{normalized}"
        else:
            path = Path(self.source).expanduser()
            if not path.is_absolute():
                path = (project_dir / path).resolve()
            else:
                path = path.resolve()
            stat = path.stat() if path.exists() else None
            payload = f"file:{path}:{stat.st_size if stat else 'missing'}:{stat.st_mtime_ns if stat else 'missing'}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def result_key(self, project_dir: Path) -> str:
        settings = asdict(self)
        settings.pop("source", None)
        settings.pop("output_dir", None)
        # Adding an opt-in feature must not invalidate existing text-only results.
        if not self.include_visual and not self.visual_focus.strip():
            settings.pop("include_visual", None)
            settings.pop("visual_focus", None)
        else:
            settings["include_visual"] = True
            settings["visual_focus"] = self.visual_focus.strip()
        settings["source_fingerprint"] = self.source_fingerprint(project_dir)
        encoded = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    progress: float
    message: str


@dataclass(frozen=True)
class VisualNote:
    timestamp: float
    image_path: Path
    description: str = ""
    end: float | None = None


@dataclass
class ProcessingArtifact:
    title: str
    source: str
    method: str
    language: str
    segments: list[TranscriptSegment]
    original_text: str
    polished_text: str = ""
    summary_text: str = ""
    warnings: list[str] = field(default_factory=list)
    visual_notes: list[VisualNote] = field(default_factory=list)
    visual_overview: str = ""


@dataclass
class ProcessingResult:
    title: str
    source: str
    method: str
    language: str
    result_dir: Path
    output_files: dict[str, Path]
    warnings: list[str] = field(default_factory=list)

    def to_index_data(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "method": self.method,
            "language": self.language,
            "result_dir": str(self.result_dir),
            "output_files": {key: str(path) for key, path in self.output_files.items()},
            "warnings": self.warnings,
        }

    @classmethod
    def from_index_data(cls, data: dict[str, Any]) -> ProcessingResult:
        return cls(
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            method=str(data.get("method", "")),
            language=str(data.get("language", "")),
            result_dir=Path(str(data["result_dir"])),
            output_files={key: Path(value) for key, value in dict(data.get("output_files") or {}).items()},
            warnings=list(data.get("warnings") or []),
        )


@dataclass(frozen=True)
class VideoDownloadRequest:
    url: str
    output_dir: str = ""
    quality: VideoQuality = VideoQuality.BEST
    cookies_from_browser: str = ""
    cookies_file: str = ""
    js_runtime: str = "deno"
    js_runtime_path: str = ""

    def result_key(self) -> str:
        parsed = urlsplit(self.url.strip())
        normalized = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, "")
        )
        payload = json.dumps(
            {"url": normalized, "quality": self.quality.value},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class VideoDownloadResult:
    title: str
    source: str
    quality: VideoQuality
    result_dir: Path
    video_path: Path

    def to_index_data(self) -> dict[str, str]:
        return {
            "title": self.title,
            "source": self.source,
            "quality": self.quality.value,
            "result_dir": str(self.result_dir),
            "video_path": str(self.video_path),
        }

    @classmethod
    def from_index_data(cls, data: dict[str, Any]) -> VideoDownloadResult:
        return cls(
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            quality=VideoQuality(str(data.get("quality", VideoQuality.BEST.value))),
            result_dir=Path(str(data["result_dir"])),
            video_path=Path(str(data["video_path"])),
        )
