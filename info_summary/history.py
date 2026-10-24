from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import ProcessingResult, VideoDownloadResult
from .settings import default_app_data_dir
from .storage import read_json, update_json


MAX_HISTORY_ENTRIES = 100


class ResultHistory:
    """Recent saved folders, including useful results with unfinished work."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_app_data_dir() / "recent_results.json"

    def record(self, result: ProcessingResult | VideoDownloadResult) -> None:
        folder = str(result.result_dir.expanduser().resolve())
        entry = {
            "title": result.title or result.result_dir.name,
            "source": result.source,
            "result_dir": folder,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "部分完成" if getattr(result, "warnings", []) else "已完成",
            "kind": "download" if isinstance(result, VideoDownloadResult) else "process",
        }
        def prepend(entries: list) -> None:
            previous = [item for item in self._valid_entries(entries, MAX_HISTORY_ENTRIES)
                        if item["result_dir"] != folder]
            entries[:] = [entry, *previous][:MAX_HISTORY_ENTRIES]

        update_json(self.path, list, prepend)

    def recent(self, limit: int = 50) -> list[dict]:
        return self._valid_entries(read_json(self.path, list), limit)

    def _valid_entries(self, values: list, limit: int) -> list[dict]:
        if limit <= 0:
            return []
        entries: list[dict] = []
        seen: set[str] = set()
        for item in values:
            entry = self._valid_entry(item)
            if entry is None or entry["result_dir"] in seen:
                continue
            seen.add(entry["result_dir"])
            entries.append(entry)
            if len(entries) >= min(limit, MAX_HISTORY_ENTRIES):
                break
        return entries

    def _valid_entry(self, item: object) -> dict | None:
        fields = ("title", "source", "result_dir", "created_at", "status", "kind")
        if not isinstance(item, dict) or any(not isinstance(item.get(field), str) for field in fields):
            return None
        if item["status"] not in {"已完成", "部分完成"} or item["kind"] not in {"process", "download"}:
            return None
        try:
            datetime.fromisoformat(item["created_at"])
            folder = Path(item["result_dir"])
            if not folder.is_absolute() or not folder.is_dir():
                return None
        except (OSError, ValueError):
            return None
        return {field: item[field] for field in fields}

