from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .audio import AudioTranscriber
from .cancellation import JobCancelled, check_cancelled
from .checkpoints import CheckpointStore
from .exporters import ResultExporter
from .formatting import readable_original
from .models import (
    DownloadOptions,
    ProcessingArtifact,
    ProcessingRequest,
    ProcessingResult,
    ProcessingStep,
    ProgressEvent,
    SourceType,
    TextProviderKind,
    TranscriptSegment,
)
from .providers import OllamaProvider, OpenCodeProvider, TextProvider
from .output_paths import resolve_output_dir, sanitize_transcript_title
from .settings import default_output_dir
from .sources import is_youtube_url
from .text_pipeline import ResumableTextProcessor
from .video import VideoExtractor
from .video.visual_analyzer import VisualAnalyzer
from .video.visual_media import VisualMedia


ProgressCallback = Callable[[ProgressEvent], None]


class ProcessingPipeline:
    def __init__(
        self,
        project_dir: Path,
        log: Callable[[str], None] | None = None,
        audio_transcriber: AudioTranscriber | None = None,
        video_extractor: VideoExtractor | None = None,
        checkpoint_dir: Path | None = None,
        visual_media=None,
    ) -> None:
        self.project_dir = project_dir
        self.log = log or (lambda _: None)
        self.audio_transcriber = audio_transcriber or AudioTranscriber(project_dir, log=self.log)
        self.video_extractor = video_extractor or VideoExtractor(log=self.log)
        self.text_processor = ResumableTextProcessor()
        self.exporter = ResultExporter()
        self.checkpoints = CheckpointStore(project_dir, checkpoint_dir, log=self.log)
        self.visual_analyzer = VisualAnalyzer(visual_media or VisualMedia(log=self.log), self.log)

    def run(
        self,
        request: ProcessingRequest,
        *,
        api_key: str = "",
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        force_restart: bool = False,
    ) -> ProcessingResult:
        if request.visual_focus.strip() and not request.include_visual:
            request = replace(request, include_visual=True)
        if request.include_visual and request.source_type != SourceType.VIDEO_URL:
            raise ValueError("结合画面目前只支持视频链接；音频请关闭此选项。")
        work_dir = self.project_dir / "temp" / "jobs" / uuid.uuid4().hex
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._check_cancel(cancel_event)
            persistent = not force_restart or self.checkpoints.clear_source(request)
            original = None if force_restart else self.checkpoints.load_original(request)
            original_warning = ""
            if original is None:
                try:
                    original = self._obtain_original(request, work_dir, progress, cancel_event)
                except JobCancelled:
                    raise
                except (OSError, RuntimeError) as exc:
                    if not request.include_visual:
                        raise
                    self._check_cancel(cancel_event)
                    original = ("video", [], "visual_only", "unknown")
                    original_warning = f"未能取得字幕或语音原文，将继续分析画面：{exc}"
                # Save complete transcripts before honoring a cancellation that
                # arrived as the final segment finished. Partial Whisper output
                # is deliberately never treated as a complete transcript.
                if persistent and original[1]:
                    self.checkpoints.save_original(request, original)
            else:
                self._emit("original", 0.50, "已恢复上次取得的原文，继续后续处理...", progress)
            title, segments, method, language = original
            self._check_cancel(cancel_event)
            original_text = readable_original(segments)
            artifact = ProcessingArtifact(
                title=title,
                source=request.source,
                method=method,
                language=language,
                segments=segments,
                original_text=original_text,
            )
            if request.include_visual and not segments:
                artifact.warnings.append(original_warning or "没有取得可用字幕或语音原文，以下内容仅依据画面。")
            if request.has_step(ProcessingStep.POLISH) or request.has_step(ProcessingStep.SUMMARY) or request.include_visual:
                text_dir = self._text_checkpoint_dir(request, work_dir, persistent)
                provider = self._provider_for(request, api_key)
                # Generation is authoritative: a service may accept a manually
                # entered model while not exposing a model-list endpoint. Cached
                # text can also be re-exported without a network probe.
                self._check_cancel(cancel_event)
                if request.has_step(ProcessingStep.POLISH) and original_text:
                    artifact.polished_text, warnings = self._polish(
                        request, provider, segments, original_text,
                        text_dir, progress, cancel_event,
                    )
                    self._check_cancel(cancel_event)
                    artifact.warnings.extend(warnings)
                if request.include_visual:
                    visual_dir = self.checkpoints.source_dir(request) / "visual" if persistent else work_dir / "visual"

                    def summarize_visual(notes: str) -> tuple[str, list[str]]:
                        source = artifact.polished_text or original_text
                        if notes.strip():
                            source += "\n\n以下是带时间的画面证据，来源与讲述原文不同，请在总结中保留画面时间，不把画面推测写成原话：\n" + notes
                        if not source.strip():
                            return "", []
                        # Keep vision summaries separate from the existing text-only caches.
                        return self._summarize(
                            request, provider, source, text_dir / "visual" / "inline_v1", progress, cancel_event,
                            inline_visual=True,
                        )

                    self._emit("visual", 0.60, "正在结合画面整理内容...", progress)
                    visual = self.visual_analyzer.run(
                        request, provider, visual_dir, segments,
                        options=self._download_options(request),
                        summarize=summarize_visual if request.has_step(ProcessingStep.SUMMARY) else None,
                        cancel_event=cancel_event,
                    )
                    artifact.visual_notes = visual.notes
                    artifact.visual_overview = visual.overview
                    artifact.summary_text = visual.summary
                    artifact.warnings.extend(visual.warnings)
                    if request.has_step(ProcessingStep.SUMMARY) and not artifact.summary_text and original_text:
                        artifact.summary_text, warnings = summarize_visual(visual.text())
                        artifact.warnings.extend(warnings)
                elif request.has_step(ProcessingStep.SUMMARY):
                    summary_source = artifact.polished_text or original_text
                    artifact.summary_text, warnings = self._summarize(
                        request, provider, summary_source,
                        text_dir, progress, cancel_event,
                    )
                    self._check_cancel(cancel_event)
                    artifact.warnings.extend(warnings)
            self._emit("export", 0.92, "正在生成结果文件...", progress)
            self._check_cancel(cancel_event)
            output_root = resolve_output_dir(request.output_dir, self.project_dir) or default_output_dir()
            result = self.exporter.export(
                artifact, request, output_root, sanitize_transcript_title(title)
            )
            self._emit("completed", 1.0, "处理完成", progress)
            return result
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _text_checkpoint_dir(
        self, request: ProcessingRequest, work_dir: Path, persistent: bool,
    ) -> Path:
        if persistent:
            prompts = [
                self.text_processor.polish_prompt(request.output_language),
                self.text_processor.partial_summary_prompt(request.output_language),
                self.text_processor.merge_summary_prompt(request.output_language),
                self.text_processor.final_summary_prompt(request.output_language),
            ]
            try:
                directory = self.checkpoints.text_dir(request, prompts)
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                return directory
            except OSError:
                self.log("文字续做记录暂时无法保存，本次仍会继续生成结果。")
        return work_dir / "text"

    def _obtain_original(
        self,
        request: ProcessingRequest,
        work_dir: Path,
        progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> tuple[str, list[TranscriptSegment], str, str]:
        self._emit("source", 0.04, "正在读取来源...", progress)
        self._check_cancel(cancel_event)
        if request.source_type == SourceType.VIDEO_URL:
            options = self._download_options(request)
            if is_youtube_url(request.source):
                self.video_extractor.check_youtube_runtime(options)
            info, active_options = self.video_extractor.extract_info_with_cookie_fallback(
                request.source, options, cancel_event=cancel_event,
            )
            title = sanitize_transcript_title(str(info.get("title") or info.get("id") or "video"))
            self._emit("subtitle", 0.16, "正在查找视频字幕...", progress)
            segments, language, method = self.video_extractor.try_extract_subtitles(
                info, work_dir, active_options, sanitize_transcript_title,
                cancel_event=cancel_event,
            )
            if segments:
                self._emit("original", 0.50, "已直接取得视频字幕", progress)
                return title, segments, method, language or "unknown"
            if not self.video_extractor.has_usable_audio_format(info):
                if request.include_visual:
                    return title, [], "visual_only", language or "unknown"
                raise RuntimeError("视频既没有可用字幕，也没有可转写的音频。")
            self._emit("audio", 0.22, "未取得可用字幕，正在准备音频...", progress)
            self.audio_transcriber.check_audio_dependencies()
            downloaded = self.video_extractor.download_audio(
                request.source, info, work_dir, active_options, fallback_options=options,
                cancel_event=cancel_event,
            )
            audio_path = self.audio_transcriber.prepare_audio(downloaded, work_dir, cancel_event=cancel_event)
            method = "audio_transcript"
        else:
            source_path = Path(request.source).expanduser()
            if not source_path.is_absolute():
                source_path = self.project_dir / source_path
            source_path = source_path.resolve()
            if not source_path.exists() or not source_path.is_file():
                raise RuntimeError(f"未找到音频文件：{source_path}")
            title = sanitize_transcript_title(source_path.stem or "audio")
            self.audio_transcriber.check_audio_dependencies()
            audio_path = self.audio_transcriber.prepare_audio(source_path, work_dir, cancel_event=cancel_event)
            method = "audio_file_transcript"
        self._check_cancel(cancel_event)
        self._emit("transcribe", 0.32, "正在使用 Whisper 转写音频...", progress)
        segments = self.audio_transcriber.transcribe_audio(
            audio_path, request.model_size, request.language_hint,
            request.device_mode, request.compute_type, request.batch_size,
            progress=lambda value, message: self._emit("transcribe", 0.32 + value * 0.18, message, progress),
            cancel_event=cancel_event,
        )
        if not segments:
            if request.include_visual:
                return title, [], "visual_only", self.audio_transcriber.last_detected_language or "unknown"
            raise RuntimeError("没有识别到可用文字。")
        language = self.audio_transcriber.last_detected_language or request.language_hint or "auto"
        self._emit("original", 0.50, "原文获取完成", progress)
        return title, segments, method, language

    def _polish(
        self, request: ProcessingRequest, provider: TextProvider,
        segments: list[TranscriptSegment], original_text: str,
        checkpoint_dir: Path, progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> tuple[str, list[str]]:
        self._emit("polish", 0.54, "正在润色原文...", progress)
        polished, _, warnings = self.text_processor.polish(
            segments, original_text, provider, request.text_model_name,
            request.output_language, request.text_model_chunk_chars,
            checkpoint_dir, progress=lambda value, message: self._emit(
                "polish", 0.54 + value * 0.20, message, progress
            ), cancelled=lambda: bool(cancel_event and cancel_event.is_set()),
        )
        return polished, warnings

    def _summarize(
        self, request: ProcessingRequest, provider: TextProvider,
        source_text: str, checkpoint_dir: Path,
        progress: ProgressCallback | None, cancel_event: threading.Event | None,
        *, inline_visual: bool = False,
    ) -> tuple[str, list[str]]:
        self._emit("summary", 0.76, "正在生成内容总结...", progress)
        groups = self.text_processor.chunk_strings(
            [part for part in source_text.split("\n\n") if part.strip()],
            request.text_model_chunk_chars,
        )
        chunks = ["\n\n".join(group) for group in groups]
        return self.text_processor.summarize(
            chunks, provider, request.text_model_name, request.output_language,
            request.text_model_chunk_chars, checkpoint_dir, progress=lambda value, message: self._emit(
                "summary", 0.76 + value * 0.14, message, progress
            ), cancelled=lambda: bool(cancel_event and cancel_event.is_set()), inline_visual=inline_visual,
        )

    def _provider_for(self, request: ProcessingRequest, api_key: str) -> TextProvider:
        if request.text_provider == TextProviderKind.OPENCODE:
            return OpenCodeProvider(api_key, request.text_model_base_url)
        return OllamaProvider(request.text_model_base_url)

    def _download_options(self, request: ProcessingRequest) -> DownloadOptions:
        return DownloadOptions(
            request.cookies_from_browser.strip(), request.cookies_file.strip(),
            request.js_runtime.strip(), request.js_runtime_path.strip(),
        )

    def _check_cancel(self, cancel_event: threading.Event | None) -> None:
        check_cancelled(cancel_event.is_set if cancel_event else None)

    def _emit(self, stage: str, value: float, message: str, progress: ProgressCallback | None) -> None:
        self.log(message)
        if progress:
            progress(ProgressEvent(stage, max(0.0, min(1.0, value)), message))
