"""Subtitle and audio acquisition using the shared website client."""
from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable, TypeVar
from urllib.error import URLError

from yt_dlp import YoutubeDL
from yt_dlp.networking.exceptions import RequestError
from yt_dlp.utils import DownloadError

from ..cancellation import JobCancelled, check_cancelled
from ..models import DownloadOptions, LogFn, SUBTITLE_EXTENSIONS, SUBTITLE_TIMEOUT_SECONDS, TranscriptSegment
from .client import VideoClient
from .subtitles import parse_subtitle_file, subtitle_entry_priority, subtitle_languages


RetryResult = TypeVar("RetryResult")


class VideoExtractor:
    def __init__(self, log: LogFn | None = None, client: VideoClient | None = None) -> None:
        self.log = log or (lambda _: None)
        self.client = client or VideoClient(log=self.log)

    def check_youtube_runtime(self, download_options: DownloadOptions) -> None:
        self.client.check_youtube_runtime(download_options)

    def extract_info_with_cookie_fallback(
        self, url: str, download_options: DownloadOptions, *, cancel_event: Event | None = None,
    ) -> tuple[dict[str, Any], DownloadOptions]:
        return self.client.extract_info_with_cookie_fallback(
            url, download_options, cancel_event=cancel_event,
            reader=lambda location, options: self.extract_info(location, options, cancel_event=cancel_event),
        )

    def with_download_retries(
        self, operation: Callable[[], RetryResult], label: str, *, cancel_event: Event | None = None,
    ) -> RetryResult:
        return self.client.with_retries(operation, label, cancel_event=cancel_event)

    def extract_info(
        self, url: str, download_options: DownloadOptions, *, cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        return self.client.extract_info(url, download_options, cancel_event=cancel_event)

    def should_retry_with_cookies(
        self, url: str, active: DownloadOptions, configured: DownloadOptions | None, error: DownloadError,
    ) -> bool:
        return self.client.should_retry_with_cookies(url, active, configured, error)

    def try_extract_subtitles(
        self,
        info: dict[str, Any],
        work_dir: Path,
        download_options: DownloadOptions,
        sanitize_filename: Callable[[str], str],
        *,
        cancel_event: Event | None = None,
    ) -> tuple[list[TranscriptSegment], str, str]:
        check_cancelled(cancel_event.is_set if cancel_event else None)
        found_subtitles = False
        for automatic, key, method in (
            (False, "subtitles", "embedded_subtitle"),
            (True, "automatic_captions", "auto_subtitle"),
        ):
            languages = subtitle_languages(info.get(key) or {})
            found_subtitles = found_subtitles or bool(languages)
            for language in languages:
                kind = "自动字幕" if automatic else "人工字幕"
                self.log(f"正在尝试{kind}：{language}")
                subtitle_file = self._download_subtitle_file(
                    info=info, work_dir=work_dir, language=language, automatic=automatic,
                    download_options=download_options, sanitize_filename=sanitize_filename,
                    cancel_event=cancel_event,
                )
                if subtitle_file:
                    segments = parse_subtitle_file(subtitle_file, self.log)
                    check_cancelled(cancel_event.is_set if cancel_event else None)
                    if segments:
                        return segments, language, method

        if found_subtitles:
            self.log("发现字幕信息，但未能取得可用字幕，将尝试音频转写。")
        return [], "", ""

    def has_usable_audio_format(self, info: dict[str, Any]) -> bool:
        for fmt in info.get("formats") or []:
            acodec = str(fmt.get("acodec") or "").lower()
            vcodec = str(fmt.get("vcodec") or "").lower()
            protocol = str(fmt.get("protocol") or "").lower()
            ext = str(fmt.get("ext") or "").lower()
            if acodec and acodec != "none":
                return True
            if vcodec == "none" and protocol != "mhtml":
                return True
            if ext in {"m4a", "webm", "mp3"} and protocol != "mhtml":
                return True
        return False

    def download_audio(
        self,
        url: str,
        info: dict[str, Any],
        work_dir: Path,
        download_options: DownloadOptions,
        fallback_options: DownloadOptions | None = None,
        *,
        cancel_event: Event | None = None,
    ) -> Path:
        check_cancelled(cancel_event.is_set if cancel_event else None)
        audio_dir = work_dir / "audio"
        video_id = info.get("id") or "source"
        active = download_options
        target_dir = audio_dir

        def download() -> Path:
            nonlocal active, target_dir
            try:
                return self._download_audio_once(
                    url, video_id, target_dir, active, cancel_event=cancel_event,
                )
            except DownloadError as exc:
                check_cancelled(cancel_event.is_set if cancel_event else None)
                if not self.should_retry_with_cookies(url, active, fallback_options, exc):
                    raise
                active = fallback_options
                target_dir = audio_dir / "with_cookies"
                self.log("音频下载被拒绝，改用已配置的登录信息重试...")
                return self._download_audio_once(
                    url, video_id, target_dir, active, cancel_event=cancel_event,
                )

        try:
            return self.with_download_retries(download, "下载音频", cancel_event=cancel_event)
        except DownloadError as exc:
            check_cancelled(cancel_event.is_set if cancel_event else None)
            raise self.client.rewrite_error(exc) from exc

    def _download_audio_once(
        self, url: str, video_id: str, audio_dir: Path, options: DownloadOptions,
        *, cancel_event: Event | None = None,
    ) -> Path:
        check_cancelled(cancel_event.is_set if cancel_event else None)
        audio_dir.mkdir(parents=True, exist_ok=True)
        ydl_options = self.client.build_options(
            skip_download=False, download_options=options, cancel_event=cancel_event,
        )
        ydl_options.update({
            # Prefer AAC/M4A for transcription: some YouTube WebM audio streams
            # currently return 403 even when the equivalent M4A is accessible.
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": str(audio_dir / f"{video_id}.%(ext)s"),
            "noplaylist": True,
        })
        with YoutubeDL(ydl_options) as ydl:
            ydl.download([url])
        check_cancelled(cancel_event.is_set if cancel_event else None)
        files = sorted(
            (item for item in audio_dir.iterdir() if item.is_file()
             and item.suffix.lower() not in {".part", ".ytdl", ".temp"} and item.stat().st_size > 0),
            key=lambda item: item.stat().st_mtime, reverse=True,
        )
        if not files:
            raise RuntimeError("音频下载失败，未找到下载文件。")
        return files[0]

    def _download_subtitle_file(
        self,
        info: dict[str, Any],
        work_dir: Path,
        language: str,
        automatic: bool,
        download_options: DownloadOptions,
        sanitize_filename: Callable[[str], str],
        *,
        cancel_event: Event | None = None,
    ) -> Path | None:
        cancelled = cancel_event.is_set if cancel_event else None
        check_cancelled(cancelled)
        subtitle_dir = work_dir / "subtitle"
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        subtitle_map = info.get("automatic_captions") if automatic else info.get("subtitles")
        subtitle_entries = (subtitle_map or {}).get(language) or []
        if not subtitle_entries:
            return None

        preferred_entries = sorted(subtitle_entries, key=subtitle_entry_priority)
        title = sanitize_filename(info.get("title") or info.get("id") or "subtitle")
        video_id = info.get("id") or "video"

        request_options = self.client.build_options(
            skip_download=True, download_options=download_options, cancel_event=cancel_event,
        )
        request_options["socket_timeout"] = SUBTITLE_TIMEOUT_SECONDS
        with YoutubeDL(request_options) as ydl:
            for entry in preferred_entries:
                check_cancelled(cancelled)
                subtitle_url = entry.get("url")
                if not subtitle_url:
                    continue
                ext = (entry.get("ext") or "vtt").lower()
                if f".{ext}" not in SUBTITLE_EXTENSIONS:
                    continue
                target_path = subtitle_dir / f"{title} [{video_id}].{language}.{ext}"

                def download() -> Path:
                    with ydl.urlopen(subtitle_url) as response:
                        with target_path.open("wb") as target:
                            while True:
                                check_cancelled(cancelled)
                                chunk = response.read(64 * 1024)
                                check_cancelled(cancelled)
                                if not chunk:
                                    break
                                target.write(chunk)
                    return target_path

                try:
                    self.with_download_retries(download, "下载字幕", cancel_event=cancel_event)
                    if parse_subtitle_file(target_path, self.log):
                        return target_path
                    self.log("此字幕格式没有可用文字，尝试其他字幕。")
                except JobCancelled:
                    target_path.unlink(missing_ok=True)
                    raise
                except (DownloadError, RequestError, URLError, TimeoutError, ConnectionError, RuntimeError):
                    check_cancelled(cancelled)
                    self.log("此字幕请求未成功，尝试其他字幕。")
                target_path.unlink(missing_ok=True)

        return None
