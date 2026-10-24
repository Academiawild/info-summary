from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from ..cancellation import JobCancelled, check_cancelled
from ..models import (
    DownloadOptions,
    VideoDownloadRequest,
    VideoDownloadResult,
    VideoQuality,
)
from ..output_paths import create_result_dir, sanitize_video_title as sanitize_filename
from ..result_index import VideoDownloadIndex
from ..settings import default_output_dir
from ..sources import is_youtube_url
from .client import VideoClient


DownloadProgress = Callable[[float, str], None]

QUALITY_FORMATS = {
    VideoQuality.BEST: "bestvideo*+bestaudio/best",
    VideoQuality.HD_1080: "bestvideo*[height<=1080]+bestaudio/best[height<=1080]",
    VideoQuality.HD_720: "bestvideo*[height<=720]+bestaudio/best[height<=720]",
}

VIDEO_ONLY_FORMATS = {
    VideoQuality.BEST: "bestvideo*/best",
    VideoQuality.HD_1080: "bestvideo*[height<=1080]/bestvideo*/best",
    VideoQuality.HD_720: "bestvideo*[height<=720]/bestvideo*/best",
}


class VideoDownloader:
    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.log = log or (lambda _: None)
        self.client = VideoClient(log=self.log)

    def download(
        self,
        request: VideoDownloadRequest,
        *,
        progress: DownloadProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VideoDownloadResult:
        self._check_cancel(cancel_event)
        options = DownloadOptions(
            cookies_from_browser=request.cookies_from_browser.strip(),
            cookies_file=request.cookies_file.strip(),
            js_runtime=request.js_runtime.strip(),
            js_runtime_path=request.js_runtime_path.strip(),
        )
        if is_youtube_url(request.url):
            self.client.check_youtube_runtime(options)
        self._emit(progress, 0.03, "正在读取视频信息...")
        info, active_options = self.client.extract_info_with_cookie_fallback(
            request.url, options, cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        title = sanitize_filename(str(info.get("title") or info.get("id") or "video"))
        output_root = Path(request.output_dir).expanduser() if request.output_dir else default_output_dir()
        if not output_root.is_absolute():
            output_root = output_root.resolve()
        result_dir = create_result_dir(output_root, title)

        try:
            video_path = self.download_to_directory(
                request.url, result_dir, options, quality=request.quality,
                active_options=active_options, progress=progress, cancel_event=cancel_event,
            )
        except Exception:
            shutil.rmtree(result_dir, ignore_errors=True)
            raise
        return VideoDownloadResult(title, request.url, request.quality, result_dir, video_path)

    def download_to_directory(
        self,
        url: str,
        directory: Path,
        options: DownloadOptions,
        *,
        quality: VideoQuality = VideoQuality.BEST,
        video_only: bool = False,
        active_options: DownloadOptions | None = None,
        progress: DownloadProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Shared download operation; callers own the directory and its lifetime."""
        self._check_cancel(cancel_event)
        if active_options is None:
            if is_youtube_url(url):
                self.client.check_youtube_runtime(options)
            _, active_options = self.client.extract_info_with_cookie_fallback(
                url, options, cancel_event=cancel_event,
            )
        directory.mkdir(parents=True, exist_ok=True)

        def progress_hook(data: dict) -> None:
            self._check_cancel(cancel_event)
            status = str(data.get("status") or "")
            if status == "downloading":
                total = int(data.get("total_bytes") or data.get("total_bytes_estimate") or 0)
                downloaded = int(data.get("downloaded_bytes") or 0)
                fraction = downloaded / total if total else 0.0
                speed = str(data.get("_speed_str") or "").strip()
                eta = str(data.get("_eta_str") or "").strip()
                details = "正在下载视频"
                if total:
                    details += f" {fraction * 100:.0f}%"
                if speed:
                    details += f" · {speed}"
                if eta:
                    details += f" · 剩余 {eta}"
                self._emit(progress, 0.08 + fraction * 0.84, details)
            elif status == "finished":
                self._emit(progress, 0.94, "下载完成，正在合并视频和声音...")

        def attempt_download(active: DownloadOptions) -> None:
            ydl_options = self.client.build_options(
                skip_download=False, download_options=active, cancel_event=cancel_event,
            )
            ydl_options.update(
                {
                    "format": (VIDEO_ONLY_FORMATS if video_only else QUALITY_FORMATS)[quality],
                    "outtmpl": str(directory / "视频.%(ext)s"),
                    "merge_output_format": "mp4",
                    "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                    "progress_hooks": [progress_hook],
                    "noplaylist": True,
                    # The command-line entry can run without activating .venv;
                    # use the same bundled executable as audio/frame processing.
                    "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
                }
            )
            with YoutubeDL(ydl_options) as ydl:
                ydl.download([url])

        active = active_options

        def download_with_cookie_fallback() -> None:
            nonlocal active
            try:
                attempt_download(active)
            except DownloadError as exc:
                self._check_cancel(cancel_event)
                if not self.client.should_retry_with_cookies(url, active, options, exc):
                    raise
                self.log("视频下载被拒绝，正在使用已配置的登录信息重试一次...")
                # Only remove this operation's files. The caller may also keep
                # persistent frame/checkpoint files in its directory.
                for path in directory.glob("视频.*"):
                    if path.is_file():
                        path.unlink()
                active = options
                attempt_download(active)

        try:
            self._emit(progress, 0.08, "开始下载视频...")
            self.client.with_retries(
                download_with_cookie_fallback, "下载视频", cancel_event=cancel_event,
            )
            self._check_cancel(cancel_event)
            video_path = self._find_video(directory)
        except Exception as exc:
            self._check_cancel(cancel_event)
            if isinstance(exc, JobCancelled):
                raise
            if isinstance(exc, DownloadError):
                raise self.client.rewrite_error(exc) from exc
            raise
        self._emit(progress, 1.0, "视频下载完成")
        return video_path

    def _find_video(self, result_dir: Path) -> Path:
        path = result_dir / "视频.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("下载结束后没有找到完整的 MP4 视频，请查看运行记录后重试。")
        return path

    def _check_cancel(self, cancel_event: threading.Event | None) -> None:
        check_cancelled(cancel_event.is_set if cancel_event else None)

    def _emit(self, progress: DownloadProgress | None, value: float, message: str) -> None:
        self.log(message)
        if progress:
            progress(max(0.0, min(1.0, value)), message)


class VideoDownloadService:
    def __init__(
        self,
        downloader: VideoDownloader | None = None,
        index: VideoDownloadIndex | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.log = log or (lambda _: None)
        self.downloader = downloader or VideoDownloader(log=log)
        self.index = index or VideoDownloadIndex()

    def find_existing(self, request: VideoDownloadRequest) -> VideoDownloadResult | None:
        return self.index.find(request)

    def run(
        self,
        request: VideoDownloadRequest,
        *,
        progress: DownloadProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VideoDownloadResult:
        result = self.downloader.download(request, progress=progress, cancel_event=cancel_event)
        try:
            self.index.register(request, result)
        except OSError:
            self.log("视频已保存，但未能记录复用信息。下次可以通过“最近结果”打开。")
        return result
