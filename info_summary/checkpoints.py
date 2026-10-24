from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .models import OutputLanguage, ProcessingRequest, TranscriptSegment
from .settings import default_app_data_dir
from .storage import read_json, write_json_atomic


Original = tuple[str, list[TranscriptSegment], str, str]
CHECKPOINT_VERSION = 1


def _digest(data: object) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CheckpointStore:
    """Keep completed text outside disposable audio/download working folders."""

    def __init__(
        self,
        project_dir: Path,
        root_dir: Path | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.root_dir = root_dir if root_dir is not None else default_app_data_dir() / "checkpoints"
        self.log = log or (lambda _: None)

    def source_dir(self, request: ProcessingRequest) -> Path:
        return self.root_dir / request.source_fingerprint(self.project_dir)

    def original_dir(self, request: ProcessingRequest) -> Path:
        # Export choices and text models cannot change an already obtained
        # transcript. Keep only the settings used to obtain that transcript.
        options = {
            "version": CHECKPOINT_VERSION,
            "model": request.model_size,
            "language": request.language_hint,
            "device": request.device_mode,
            "compute": request.compute_type,
            "batch_size": request.batch_size,
        }
        return self.source_dir(request) / _digest(options)

    def text_dir(self, request: ProcessingRequest, prompts: list[str]) -> Path:
        options = {
            "version": CHECKPOINT_VERSION,
            "provider": request.text_provider.value,
            "base_url": request.text_model_base_url.rstrip("/"),
            "model": request.text_model_name,
            "language": request.output_language.value,
            "chunk_chars": request.text_model_chunk_chars,
            "prompts": prompts,
        }
        # Only hashes become path names; credentials and provider options are
        # never written into the checkpoint payload.
        return self.original_dir(request) / "text" / _digest(options)

    def load_original(self, request: ProcessingRequest) -> Original | None:
        try:
            path = self.original_dir(request) / "original.json"
            data = read_json(path)
            if data.get("version") != CHECKPOINT_VERSION:
                return None
            title, method, language = (data.get(name) for name in ("title", "method", "language"))
            if not all(isinstance(value, str) and value.strip() for value in (title, method, language)):
                return None
            for value in (title, method, language):
                value.encode("utf-8")
            rows = data.get("segments")
            if not isinstance(rows, list) or not rows:
                return None
            segments: list[TranscriptSegment] = []
            for row in rows:
                if not isinstance(row, dict):
                    return None
                start, end, text = (row.get(name) for name in ("start", "end", "text"))
                if (
                    type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end < start or not isinstance(text, str) or not text.strip()
                ):
                    return None
                text.encode("utf-8")
                segments.append(TranscriptSegment(float(start), float(end), text))
            return title, segments, method, language
        except (OSError, UnicodeError, ValueError, OverflowError, RecursionError):
            return None

    def save_original(self, request: ProcessingRequest, original: Original) -> bool:
        title, segments, method, language = original
        payload = {
            "version": CHECKPOINT_VERSION,
            "title": title,
            "method": method,
            "language": language,
            "segments": [asdict(segment) for segment in segments],
        }
        try:
            path = self.original_dir(request) / "original.json"
            write_json_atomic(path, payload)
            return True
        except (OSError, UnicodeError, ValueError):
            self.log("未能保存续做记录，本次仍会继续生成结果。")
            return False

    def clear_source(self, request: ProcessingRequest) -> bool:
        try:
            directory = self.source_dir(request)
            if directory.exists():
                shutil.rmtree(directory)
            return True
        except OSError:
            self.log("旧续做记录未能清理；本次会从头处理，不使用旧记录。")
            return False


class TextCheckpointStore:
    """Text response caching; the existing filenames and identity stay unchanged."""

    @staticmethod
    def cache_key(input_text: str, model_name: str, output_language: OutputLanguage) -> str:
        payload = f"{model_name}\0{output_language.value}\0{input_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(
        self, path: Path, input_text: str, model_name: str, output_language: OutputLanguage,
    ) -> str | None:
        data = read_json(path)
        if data.get("key") != self.cache_key(input_text, model_name, output_language):
            return None
        output = data.get("output")
        if not isinstance(output, str):
            return None
        try:
            output.encode("utf-8")
        except UnicodeError:
            return None
        return output.strip() or None

    def save(
        self, path: Path, input_text: str, model_name: str,
        output_language: OutputLanguage, output: str,
    ) -> bool:
        payload = {"key": self.cache_key(input_text, model_name, output_language), "output": output}
        try:
            write_json_atomic(path, payload)
            return True
        except (OSError, UnicodeError, ValueError):
            return False
