"""Completed text and video results, with shared safe metadata storage."""
from __future__ import annotations

from pathlib import Path

from .models import ProcessingRequest, ProcessingResult, VideoDownloadRequest, VideoDownloadResult
from .settings import default_app_data_dir
from .storage import read_json, update_json


def _discard_stale(path: Path, key: str, previous: dict) -> None:
    def discard(entries: dict) -> None:
        # A different process may already have replaced this stale result.
        if entries.get(key) == previous:
            entries.pop(key, None)

    try:
        update_json(path, dict, discard)
    except OSError:
        pass  # A stale cache must not prevent the user from retrying.


class ResultIndex:
    """Complete processing results; incomplete or missing files are never reused."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_app_data_dir() / "results.json"

    def find(self, request: ProcessingRequest, project_dir: Path) -> ProcessingResult | None:
        key = request.result_key(project_dir)
        data = self._read().get(key)
        if not isinstance(data, dict):
            return None
        try:
            result = ProcessingResult.from_index_data(data)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            document_formats = [name for name in request.export_formats if name in {"docx", "md"}]
            expected_files = {
                f"{step.value}.{format_name}"
                for step in request.steps for format_name in document_formats
            }
            if "html" in request.export_formats:
                expected_files.add("report.html")
            visual_files = {name for name in result.output_files if name.startswith("image.")}
            if request.include_visual:
                expected_files.update(f"visual.{name}" for name in document_formats)
                expected_files.update(visual_files)
            complete = (
                not result.warnings and result.result_dir.is_dir() and bool(expected_files)
                and (not request.include_visual or bool(visual_files))
                and expected_files.issubset(result.output_files)
                and all(result.output_files[name].is_file() for name in expected_files)
            )
        except (OSError, KeyError, TypeError, ValueError):
            complete = False
        if not complete:
            _discard_stale(self.path, key, data)
            return None
        return result

    def register(self, request: ProcessingRequest, project_dir: Path, result: ProcessingResult) -> None:
        if result.warnings:
            return
        key, payload = request.result_key(project_dir), result.to_index_data()
        update_json(self.path, dict, lambda entries: entries.update({key: payload}))

    def _read(self) -> dict[str, dict]:
        return read_json(self.path)


class VideoDownloadIndex:
    """Completed video downloads; stored in the existing video_downloads.json."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_app_data_dir() / "video_downloads.json"

    def find(self, request: VideoDownloadRequest) -> VideoDownloadResult | None:
        key = request.result_key()
        data = self._read().get(key)
        if not isinstance(data, dict):
            return None
        try:
            result = VideoDownloadResult.from_index_data(data)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            complete = result.result_dir.is_dir() and result.video_path.is_file()
        except (OSError, KeyError, TypeError, ValueError):
            complete = False
        if not complete:
            _discard_stale(self.path, key, data)
            return None
        return result

    def register(self, request: VideoDownloadRequest, result: VideoDownloadResult) -> None:
        key, payload = request.result_key(), result.to_index_data()
        update_json(self.path, dict, lambda entries: entries.update({key: payload}))

    def _read(self) -> dict[str, dict]:
        return read_json(self.path)
