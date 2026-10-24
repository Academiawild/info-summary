"""Per-run visual response and frame-context persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..storage import read_json, write_json_atomic


def digest(value: object) -> str:
    # Keep the exact encoding used by existing visual cache filenames.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class VisualCache:
    def __init__(self, directory: Path, warnings: list[str]) -> None:
        self.calls_dir = directory / "calls"
        self.context_path = directory / "frame_contexts.json"
        self.contexts = read_json(self.context_path)
        self.warnings = warnings

    def load(self, identity: dict) -> str | None:
        key = digest(identity)
        data = read_json(self.calls_dir / (key + ".json"))
        value = data.get("output")
        if data.get("input") != key or not isinstance(value, str) or not value.strip():
            return None
        try:
            value.encode("utf-8")
        except UnicodeError:
            return None
        return value

    def save(self, identity: dict, output: str) -> None:
        key = digest(identity)
        try:
            write_json_atomic(self.calls_dir / (key + ".json"), {"input": key, "output": output})
        except (OSError, UnicodeError, ValueError):
            self._warn("画面解读已取得，但本段续做进度未能保存；再次处理时可能需要重新调用模型。")

    def context(self, key: str) -> dict:
        context = self.contexts.get(key, {})
        if (not isinstance(context, dict) or not isinstance(context.get("focus", ""), str)
                or not isinstance(context.get("questions", []), list)
                or not all(isinstance(part, str) for part in context.get("questions", []))):
            return {}
        try:
            context.get("focus", "").encode("utf-8")
            for question in context.get("questions", []):
                question.encode("utf-8")
        except UnicodeError:
            return {}
        return context

    def save_context(self, key: str, context: dict) -> None:
        self.contexts[key] = context
        try:
            write_json_atomic(self.context_path, self.contexts)
        except (OSError, UnicodeError, ValueError):
            self._warn("画面重点进度暂时未能保存，下次可能需要重新解读相关截图。")

    def _warn(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)
