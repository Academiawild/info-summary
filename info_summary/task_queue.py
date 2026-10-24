from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

from .models import (OutputLanguage, ProcessingRequest, ProcessingStep, SourceType,
                     TextProviderKind, VideoDownloadRequest, VideoQuality)
from .settings import default_app_data_dir
from .storage import read_json, write_json_atomic


Request = ProcessingRequest | VideoDownloadRequest


def task_identity(kind: str, request: Request) -> tuple[str, str]:
    return kind, request.url if isinstance(request, VideoDownloadRequest) else request.source


class UnfinishedTasks:
    """Keep unfinished sources across batches and restarts; never store API keys."""

    def __init__(self, path: Path | None = None, log: Callable[[str], None] | None = None):
        self.path = path or default_app_data_dir() / "unfinished_tasks.json"
        self.log = log or (lambda _text: None)
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str], tuple[str, Request]] = {}
        data = read_json(self.path, list)
        migrated = 0
        for item in data if isinstance(data, list) else []:
            try:
                kind, request = self._decode(item)
            except (ValueError, TypeError, KeyError):
                continue
            self._items[task_identity(kind, request)] = (kind, request)
            if kind == "process" and item["request"].get("text_provider") == "poe":
                migrated += 1
        if migrated:
            self.log(
                f"已保留 {migrated} 个旧 Poe 未完成任务，并切换到 OpenCode API。"
                "请先在模型设置中填写新服务地址、密钥和模型，再点击“继续未完成”；已提取的原文会保留。"
            )

    @staticmethod
    def _decode(item: dict) -> tuple[str, Request]:
        if not isinstance(item, dict) or not isinstance(item.get("request"), dict):
            raise ValueError("无效任务记录")
        kind = item["kind"]
        cls = ProcessingRequest if kind == "process" else VideoDownloadRequest
        if kind not in {"process", "download"}:
            raise ValueError("无效任务类型")
        values = {key: value for key, value in item["request"].items()
                  if key in {field.name for field in fields(cls)}}
        if kind == "process":
            if values.get("text_provider") == "poe":
                values.update(text_provider="opencode", text_model_name="", text_model_base_url="")
            values["source_type"] = SourceType(values["source_type"])
            values["steps"] = tuple(ProcessingStep(value) for value in values["steps"])
            values["export_formats"] = tuple(values["export_formats"])
            values["text_provider"] = TextProviderKind(values["text_provider"])
            values["output_language"] = OutputLanguage(values["output_language"])
        else:
            values["quality"] = VideoQuality(values["quality"])
        request = cls(**values)
        # A damaged record must not poison the remaining queue.
        for field in fields(request):
            value = getattr(request, field.name)
            if field.name == "include_visual":
                if type(value) is not bool:
                    raise ValueError("无效画面设置")
            elif field.name in {"batch_size", "text_model_chunk_chars"}:
                if type(value) is not int or value <= 0:
                    raise ValueError("无效任务设置")
            elif field.name in {"steps", "export_formats"}:
                if not value or any(not isinstance(part, str) for part in value):
                    raise ValueError("无效输出设置")
            elif not isinstance(value, str) or "\0" in value:
                raise ValueError("无效任务内容")
            elif isinstance(value, str):
                value.encode("utf-8")
        if isinstance(request, ProcessingRequest) and not set(request.export_formats) <= {"docx", "md", "html"}:
            raise ValueError("无效输出格式")
        if not task_identity(kind, request)[1].strip():
            raise ValueError("空任务")
        return kind, request

    def pending(self) -> list[tuple[str, Request]]:
        with self._lock:
            return list(self._items.values())

    def add(self, items: list[tuple[str, Request]]) -> None:
        with self._lock:
            for kind, request in items:
                self._items[task_identity(kind, request)] = (kind, request)
            self._save()

    def discard(self, kind: str, request: Request) -> None:
        with self._lock:
            identity = task_identity(kind, request)
            if self._items.get(identity) == (kind, request):
                self._items.pop(identity, None)
                self._save()

    def _save(self) -> None:
        try:
            write_json_atomic(self.path, [{"kind": kind, "request": asdict(request)}
                                         for kind, request in self._items.values()])
        except (OSError, UnicodeError, ValueError):
            self.log("未能保存续做列表；本次仍可处理，关闭前请保留来源清单。")


@dataclass
class QueuedTask:
    index: int
    kind: str
    request: Request
    status: str = "等待"
    force_restart: bool = False


class TaskQueue:
    """Stable row numbers and atomic claim/remove operations for one worker."""

    def __init__(self, store: UnfinishedTasks | None = None):
        self.store = store
        self._lock = threading.RLock()
        self.items: list[QueuedTask] = []

    def add(self, pending: list[tuple[str, Request]], *, force_restart: bool = False) -> list[QueuedTask]:
        with self._lock:
            seen = {task_identity(item.kind, item.request) for item in self.items if item.status != "已移除"}
            added = []
            for kind, request in pending:
                identity = task_identity(kind, request)
                if identity in seen:
                    continue
                item = QueuedTask(len(self.items), kind, request, force_restart=force_restart)
                self.items.append(item)
                added.append(item)
                seen.add(identity)
            if self.store and added:
                self.store.add([(item.kind, item.request) for item in added])
            return added

    def claim(self) -> QueuedTask | None:
        with self._lock:
            for item in self.items:
                if item.status == "等待":
                    item.status = "处理中"
                    return item
            return None

    def remove(self, index: int) -> bool:
        with self._lock:
            if not 0 <= index < len(self.items) or self.items[index].status not in {"等待", "已停止"}:
                return False
            item = self.items[index]
            item.status = "已移除"
            if self.store:
                self.store.discard(item.kind, item.request)
            return True

    def finish(self, item: QueuedTask, status: str) -> None:
        with self._lock:
            item.status = status
            if status == "已完成" and self.store:
                self.store.discard(item.kind, item.request)

    def stop_waiting(self) -> list[QueuedTask]:
        with self._lock:
            stopped = [item for item in self.items if item.status in {"等待", "处理中"}]
            for item in stopped:
                item.status = "已停止"
            return stopped

    def has_waiting(self) -> bool:
        with self._lock:
            return any(item.status == "等待" for item in self.items)

    def counts(self) -> dict:
        with self._lock:
            active = [item for item in self.items if item.status != "已移除"]
            return {
                "completed": sum(item.status == "已完成" for item in active),
                "partial": sum(item.status == "部分完成" for item in active),
                "failed": sum(item.status == "失败" for item in active),
                "cancelled": any(item.status == "已停止" for item in active),
                "total": len(active),
                "retry_pending": [(item.kind, item.request) for item in active
                                  if item.status not in {"已完成"}],
            }
