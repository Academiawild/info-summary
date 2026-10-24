from __future__ import annotations

import shutil
from threading import Event
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
from faster_whisper import BatchedInferencePipeline, WhisperModel

from ..cancellation import check_cancelled
from ..media_commands import run_media_command
from ..output_paths import resolve_output_dir, sanitize_transcript_title
from ..models import (
    LogFn,
    TranscriptSegment,
)


class AudioTranscriber:
    def __init__(self, project_dir: Path, log: LogFn | None = None) -> None:
        self.project_dir = project_dir
        self.log = log or (lambda _: None)
        self.last_detected_language = ""
        self._pipeline_key: tuple[str, str, str] | None = None
        self._batched_model: BatchedInferencePipeline | None = None

    def check_audio_dependencies(self) -> None:
        if not self.ffmpeg_command():
            raise RuntimeError("未检测到 ffmpeg，请先确认 ffmpeg 已加入 PATH。")

    def ffmpeg_command(self) -> str:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which("ffmpeg") or ""

    def sanitize_filename(self, name: str) -> str:
        return sanitize_transcript_title(name)

    def resolve_output_dir(self, output_dir: str) -> Path | None:
        return resolve_output_dir(output_dir, self.project_dir)

    def prepare_audio(
        self, audio_path: Path, work_dir: Path, *, cancel_event: Event | None = None,
    ) -> Path:
        output_path = work_dir / "normalized_audio.wav"
        self.log("正在用 ffmpeg 提取并标准化无损音频...")
        self.run_command(
            [
                self.ffmpeg_command(),
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            cancel_event=cancel_event,
        )
        return output_path

    def transcribe_audio(
        self,
        mp3_path: Path,
        model_size: str,
        language_hint: str,
        device_mode: str,
        compute_type: str,
        batch_size: int,
        *,
        progress: Callable[[float, str], None] | None = None,
        cancel_event: Event | None = None,
    ) -> list[TranscriptSegment]:
        check_cancelled(cancel_event.is_set if cancel_event else None)
        self.last_detected_language = ""
        device = self.resolve_device(device_mode)
        resolved_compute_type = self.resolve_compute_type(device=device, compute_type=compute_type)

        segments_iter, info = self.run_faster_whisper(
            mp3_path=mp3_path,
            model_size=model_size,
            language_hint=language_hint,
            device=device,
            compute_type=resolved_compute_type,
            batch_size=batch_size,
        )
        segments = self.collect_segments(
            segments_iter, duration=getattr(info, "duration", 0.0),
            progress=progress, cancel_event=cancel_event,
        )

        if not segments:
            raise RuntimeError("转写完成，但没有拿到有效文本。")

        detected_language = getattr(info, "language", "") or language_hint or "auto"
        self.last_detected_language = detected_language
        self.log(f"转写完成，检测语言：{detected_language}")
        return segments

    def collect_segments(
        self, segments_iter, *, duration: float = 0.0,
        progress: Callable[[float, str], None] | None = None,
        cancel_event: Event | None = None,
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        cancelled = cancel_event.is_set if cancel_event else None
        iterator = iter(segments_iter)
        recognized_seconds = 0.0
        total_seconds = max(0.0, float(duration or 0.0))
        try:
            while True:
                # Whisper performs inference inside next(); avoid starting a
                # further batch after the user has already pressed Stop.
                check_cancelled(cancelled)
                try:
                    segment = next(iterator)
                except StopIteration:
                    break
                check_cancelled(cancelled)
                text = (segment.text or "").strip()
                if text:
                    segments.append(TranscriptSegment(
                        start=float(segment.start), end=float(segment.end), text=text,
                    ))
                recognized_seconds = max(recognized_seconds, float(segment.end))
                if progress:
                    fraction = min(1.0, recognized_seconds / total_seconds) if total_seconds else 0.0
                    message = f"已识别到第 {recognized_seconds / 60:.1f} 分钟"
                    if total_seconds:
                        message += f" / 共 {total_seconds / 60:.1f} 分钟"
                    progress(fraction, message)
            check_cancelled(cancelled)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return segments

    def run_faster_whisper(
        self,
        mp3_path: Path,
        model_size: str,
        language_hint: str,
        device: str,
        compute_type: str,
        batch_size: int,
    ):
        pipeline_key = (model_size, device, compute_type)
        if self._pipeline_key != pipeline_key or self._batched_model is None:
            self.log(f"正在加载本地模型：{model_size}（device={device}, compute_type={compute_type}）")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._batched_model = BatchedInferencePipeline(model=model)
            self._pipeline_key = pipeline_key
        else:
            self.log(f"继续使用已加载模型：{model_size}（device={device}, compute_type={compute_type}）")

        self.log(f"开始批处理离线音频转写（batch_size={batch_size}）...")
        return self._batched_model.transcribe(
            str(mp3_path),
            language=language_hint or None,
            batch_size=batch_size,
            beam_size=5,
            word_timestamps=False,
        )

    def run_command(self, command: list[str], *, cancel_event: Event | None = None) -> None:
        returncode, stdout, stderr = run_media_command(command, cancel_event=cancel_event)
        if returncode:
            # Match the former text-mode subprocess decoding and newline handling.
            def decode(value: bytes) -> str:
                return value.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n").strip()

            message = decode(stderr) or decode(stdout)
            raise RuntimeError(message or f"命令执行失败：{' '.join(command)}")

    def resolve_device(self, device_mode: str) -> str:
        normalized = (device_mode or "cpu").strip().lower()
        if normalized not in {"auto", "cpu"}:
            raise ValueError("当前仅支持 CPU 转写，请把 device_mode 设置为 cpu 或 auto。")
        return "cpu"

    def resolve_compute_type(self, device: str, compute_type: str) -> str:
        normalized = (compute_type or "").strip().lower()
        if not normalized:
            return "int8"
        if device == "cpu" and normalized not in {"int8", "float32"}:
            raise ValueError("CPU 模式下 compute_type 只支持 int8、float32。")
        return normalized
