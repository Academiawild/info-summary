from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import (
    OutputLanguage,
    ProcessingRequest,
    ProcessingStep,
    SourceType,
    TextProviderKind,
    VideoDownloadRequest,
    VideoQuality,
)
from .providers import validate_api_url
from .settings import AppSettings, default_output_dir
from .sources import detect_source


@dataclass(frozen=True)
class RequestRequirements:
    include_visual: bool
    needs_model: bool
    missing_model_fields: tuple[str, ...]
    export_formats: tuple[str, ...]


def request_requirements(settings: AppSettings) -> RequestRequirements:
    """Describe a request without touching files, the network, or saved settings."""
    visual = settings.include_visual or bool(settings.visual_focus.strip())
    needs_model = settings.include_polish or settings.include_summary or visual
    missing: tuple[str, ...] = ()
    if needs_model:
        if settings.text_provider == "opencode":
            missing = tuple(label for label, value in (
                ("接口地址", settings.opencode_base_url),
                ("模型名称", settings.opencode_model_name),
                ("API 密钥", settings.opencode_api_key),
            ) if not value.strip())
        elif not settings.ollama_model_name.strip():
            missing = ("本机模型名称",)
    formats = tuple(name for name, enabled in (
        ("docx", settings.export_docx), ("md", settings.export_markdown),
        ("html", settings.export_html),
    ) if enabled)
    return RequestRequirements(visual, needs_model, missing, formats)


def build_processing_request(settings: AppSettings, project_dir: Path) -> ProcessingRequest:
    source_type, source = detect_source(settings.last_source, project_dir)
    requirements = request_requirements(settings)
    visual_focus = settings.visual_focus.strip()
    include_visual = requirements.include_visual
    if include_visual and source_type != SourceType.VIDEO_URL:
        raise ValueError("“结合画面”目前只支持视频链接；音频文件或文件夹请关闭“结合画面”后处理。")
    steps = [ProcessingStep.ORIGINAL]
    if settings.include_polish:
        steps.append(ProcessingStep.POLISH)
    if settings.include_summary:
        steps.append(ProcessingStep.SUMMARY)
    formats = requirements.export_formats
    if not formats:
        raise ValueError("请至少选择阅读报告、Word 或 Markdown 中的一种格式。")
    provider = TextProviderKind(settings.text_provider)
    online = provider == TextProviderKind.OPENCODE
    model = settings.opencode_model_name if online else settings.ollama_model_name
    base_url = settings.opencode_base_url if online else settings.ollama_base_url
    if requirements.needs_model:
        suggestion = ("请点击“模型设置”填写；结合画面需要支持图片的模型，也可以先关闭“结合画面”。"
                      if include_visual else "请点击“模型设置”填写；服务资料尚未确定时，可以先选择“只提取原文”（命令行使用 --preset original）。")
        if online:
            missing = requirements.missing_model_fields
            if missing:
                raise ValueError(f"OpenCode API 尚未配置：缺少{'、'.join(missing)}。\n{suggestion}")
            base_url = validate_api_url(base_url)
        elif not model.strip():
            raise ValueError("尚未选择本机模型，请点击“模型设置 → 读取模型”；结合画面需要支持图片的模型，也可以先关闭“结合画面”。" if include_visual else "尚未选择本机模型，请点击“模型设置 → 读取模型”；也可以选择 OpenCode API，或先选择“只提取原文”（命令行使用 --preset original）。")
    return ProcessingRequest(
        source=source, source_type=source_type, output_dir=settings.output_dir,
        steps=tuple(steps), export_formats=formats, model_size=settings.model_size,
        language_hint=settings.language_hint, device_mode=settings.device_mode,
        compute_type=settings.compute_type, batch_size=settings.batch_size,
        text_provider=provider, text_model_name=model,
        text_model_base_url=base_url,
        text_model_chunk_chars=settings.text_model_chunk_chars,
        output_language=OutputLanguage(settings.output_language),
        cookies_from_browser=settings.cookies_from_browser, cookies_file=settings.cookies_file,
        js_runtime=settings.js_runtime, js_runtime_path=settings.js_runtime_path,
        include_visual=include_visual, visual_focus=visual_focus,
    )


def build_download_request(settings: AppSettings, source: str) -> VideoDownloadRequest:
    return VideoDownloadRequest(
        url=source,
        output_dir=settings.output_dir,
        quality=VideoQuality(settings.video_quality),
        cookies_from_browser=settings.cookies_from_browser,
        cookies_file=settings.cookies_file,
        js_runtime=settings.js_runtime,
        js_runtime_path=settings.js_runtime_path,
    )


def validate_output(settings: AppSettings, project_dir: Path) -> None:
    path = Path(settings.output_dir).expanduser() if settings.output_dir else default_output_dir()
    if not path.is_absolute():
        path = project_dir / path
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise ValueError("保存位置无法写入，请点击“选择位置”，换一个文件夹后重试（命令行使用 --output 指定保存位置）。") from exc
    settings.output_dir = str(path.resolve())
