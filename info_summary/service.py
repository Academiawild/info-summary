from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from .models import ProcessingRequest, ProcessingResult, SourceType
from .pipeline import ProcessingPipeline, ProgressCallback
from .result_index import ResultIndex
from .sources import list_audio_files


class ProcessingService:
    """The single public entry for URLs, audio files, and audio folders."""

    def __init__(
        self,
        project_dir: Path,
        log=None,
        result_index: ResultIndex | None = None,
        pipeline: ProcessingPipeline | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.index = result_index or ResultIndex()
        self.pipeline = pipeline or ProcessingPipeline(project_dir, log=log)

    def expand_requests(self, request: ProcessingRequest) -> list[ProcessingRequest]:
        if request.source_type != SourceType.AUDIO_FOLDER:
            return [request]
        folder = Path(request.source).expanduser()
        if not folder.is_absolute():
            folder = self.project_dir / folder
        folder = folder.resolve()
        if not folder.exists() or not folder.is_dir():
            raise RuntimeError(f"未找到音频文件夹：{folder}")
        files = list_audio_files(folder)
        if not files:
            raise RuntimeError(f"文件夹中没有支持的音频文件：{folder}")
        return [replace(request, source=str(path), source_type=SourceType.AUDIO_FILE) for path in files]

    def find_existing(self, request: ProcessingRequest) -> ProcessingResult | None:
        return self.index.find(request, self.project_dir)

    def run(
        self,
        request: ProcessingRequest,
        *,
        api_key: str = "",
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        force_restart: bool = False,
    ) -> ProcessingResult:
        result = self.pipeline.run(
            request, api_key=api_key, progress=progress, cancel_event=cancel_event,
            force_restart=force_restart,
        )
        if not result.warnings:
            try:
                self.index.register(request, self.project_dir, result)
            except OSError:
                result.warnings.append("结果文件已保存，但未能记录到历史记录。下次处理相同内容时可能需要重新生成。")
        return result
