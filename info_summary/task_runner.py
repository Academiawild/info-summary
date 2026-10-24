"""Run one desktop queue without reading or updating any Tk widgets."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import ProcessingRequest, VideoDownloadRequest
from .task_queue import TaskQueue
from .cancellation import JobCancelled


@dataclass(frozen=True)
class TaskEvent:
    kind: str
    payload: object


class TaskRunner:
    def __init__(self, project_dir: Path, emit: Callable[[TaskEvent], None],
                 cancel_event: threading.Event) -> None:
        self.project_dir = project_dir
        self.emit = emit
        self.cancel_event = cancel_event

    def _send(self, kind: str, payload: object) -> None:
        self.emit(TaskEvent(kind, payload))

    def run(self, work: TaskQueue, api_key: str) -> None:
        processing_service = download_service = None

        def stopped() -> None:
            for item in work.stop_waiting():
                self._send("task_state", (item.index, "已停止", "已保存可续做的进度，点击“继续未完成”接着处理"))

        while True:
            if self.cancel_event.is_set():
                stopped()
                break
            item = work.claim()
            if item is None:
                break
            kind, request, index = item.kind, item.request, item.index
            try:
                label = request.url if isinstance(request, VideoDownloadRequest) else request.source
                self._send("task_state", (index, "处理中", "正在准备"))
                self._send("log", f"任务 {index + 1}：{label}")

                def report(value, message):
                    counts = work.counts()
                    finished = counts["completed"] + counts["partial"] + counts["failed"]
                    self._send("queue_progress", (index, finished, counts["total"], value, message))

                if kind == "download" and isinstance(request, VideoDownloadRequest):
                    if download_service is None:
                        from .video import VideoDownloadService
                        download_service = VideoDownloadService(log=lambda text: self._send("log", text))
                    result = download_service.run(request, progress=report, cancel_event=self.cancel_event)
                elif kind == "process" and isinstance(request, ProcessingRequest):
                    if processing_service is None:
                        from .service import ProcessingService
                        processing_service = ProcessingService(self.project_dir, log=lambda text: self._send("log", text))
                    result = processing_service.run(request, api_key=api_key,
                                                    progress=lambda event: report(event.progress, event.message),
                                                    cancel_event=self.cancel_event, force_restart=item.force_restart)
                    item.force_restart = False
                else:
                    raise RuntimeError("无法识别的任务类型。")
                self._send("result", result)
                self._send("task_result", (index, result.result_dir))
                warnings = getattr(result, "warnings", [])
                status = "部分完成" if warnings else "已完成"
                work.finish(item, status)
                self._send("task_state", (index, status, "\n".join(warnings) if warnings else f"已保存：{result.title}"))
            except JobCancelled:
                stopped()
                break
            except Exception as exc:
                if self.cancel_event.is_set():
                    stopped()
                    break
                work.finish(item, "失败")
                self._send("task_state", (index, "失败", str(exc)))
                self._send("task_error", f"任务 {index + 1}：{exc}")
        self._send("idle", work.counts())
