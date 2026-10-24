"""Shared website access, cancellation, retry, and error rules for video operations."""
from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path
from threading import Event
from typing import Any, Callable, TypeVar
from urllib.error import URLError

from yt_dlp import YoutubeDL
from yt_dlp.networking.exceptions import RequestError
from yt_dlp.utils import DownloadError

from ..cancellation import JobCancelled, check_cancelled
from ..macos import find_runtime
from ..models import DownloadOptions, LogFn
from ..sources import is_youtube_url


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RETRY_DELAYS = (5, 15)
ACCESS_RETRY_DELAYS = (15, 30)
RetryResult = TypeVar("RetryResult")


class VideoClient:
    def __init__(self, log: LogFn | None = None) -> None:
        self.log = log or (lambda _: None)

    def check_youtube_runtime(self, download_options: DownloadOptions) -> None:
        if not importlib.util.find_spec("yt_dlp_ejs"):
            raise RuntimeError("视频组件不完整。请关闭软件，在项目文件夹运行 ./scripts/setup_app.sh 后重试。")

        runtime = download_options.js_runtime or "deno"
        runtime_path = download_options.js_runtime_path.strip()

        if runtime_path:
            if not Path(runtime_path).exists():
                raise RuntimeError(f"未找到指定的 JavaScript runtime：{runtime_path}")
            return

        if find_runtime(runtime) is None:
            raise RuntimeError(
                f"处理 YouTube 需要 {runtime}。请先安装 Deno（https://deno.com/），"
                "或在“更多设置 → 网站访问设置”中选择已安装的 Node 并填写路径。"
                "安装后重新打开软件。"
            )

    def extract_info_with_cookie_fallback(
        self,
        url: str,
        download_options: DownloadOptions,
        *,
        reader: Callable[[str, DownloadOptions], dict[str, Any]] | None = None,
        cancel_event: Event | None = None,
    ) -> tuple[dict[str, Any], DownloadOptions]:
        check_cancelled(cancel_event.is_set if cancel_event else None)

        def lookup(active_options: DownloadOptions) -> dict[str, Any]:
            if reader is not None:
                return reader(url, active_options)
            return self.extract_info(url, active_options, cancel_event=cancel_event)

        active = download_options
        if self._should_try_without_cookies(url, download_options):
            active = DownloadOptions(
                js_runtime=download_options.js_runtime,
                js_runtime_path=download_options.js_runtime_path,
            )
            self.log("先尝试不使用 cookies 读取公开视频...")

        def read_info() -> dict[str, Any]:
            nonlocal active
            try:
                return lookup(active)
            except JobCancelled:
                raise
            except RuntimeError as exc:
                check_cancelled(cancel_event.is_set if cancel_event else None)
                if active == download_options or not self._should_fallback_to_cookies(
                    self._error_message(exc), download_options,
                ):
                    raise
                active = download_options
                self.log("网站限制了访问，改用已配置的登录信息重试...")
                return lookup(active)

        info = self.with_retries(read_info, "读取视频信息", cancel_event=cancel_event)
        return info, active

    def with_retries(
        self, operation: Callable[[], RetryResult], label: str, *, cancel_event: Event | None = None,
    ) -> RetryResult:
        """Retry transient failures, with one shared policy for reads and downloads."""
        cancelled = cancel_event.is_set if cancel_event else None
        for attempt in range(len(RETRY_DELAYS) + 1):
            check_cancelled(cancelled)
            try:
                result = operation()
                check_cancelled(cancelled)
                return result
            except JobCancelled:
                raise
            except (DownloadError, RequestError, URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
                check_cancelled(cancelled)
                if not self._is_temporary_error(exc):
                    raise
                if attempt == len(RETRY_DELAYS):
                    detail = self.rewrite_error(exc) if isinstance(exc, DownloadError) else exc
                    raise RuntimeError(f"{label}已自动尝试 {attempt + 1} 次，仍未成功。{detail}") from exc
                message = self._error_message(exc)
                restricted = "not a bot" in message or "429" in message
                delay = (ACCESS_RETRY_DELAYS if restricted else RETRY_DELAYS)[attempt]
                reason = "网站暂时限制了访问" if restricted else "网络请求暂时失败"
                self.log(f"{label}：{reason}，等待 {delay} 秒后自动重试（{attempt + 2}/{len(RETRY_DELAYS) + 1}）...")
                if cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
                check_cancelled(cancelled)
        raise AssertionError("unreachable")

    @staticmethod
    def _error_message(exc: Exception) -> str:
        messages = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            messages.append(ANSI_ESCAPE_RE.sub("", str(current)).lower())
            current = current.__cause__
        return " ".join(messages)

    def _is_temporary_error(self, exc: Exception) -> bool:
        message = self._error_message(exc)
        # Missing/private/paid/age-restricted content will not become accessible
        # by repeating a request. Check these before any network-error markers.
        if any(marker in message for marker in (
            "private video", "video has been removed", "video unavailable", "copyright",
            "members-only", "members on level", "age-restricted", "confirm your age",
        )):
            return False
        status = getattr(exc, "status", getattr(exc, "code", None))
        if status is not None:
            return status in {403, 408, 429, 500, 502, 503, 504}
        return isinstance(exc, (TimeoutError, ConnectionError)) or any(marker in message for marker in (
            "not a bot", "http error 403", "http error 408", "http error 429",
            "http error 500", "http error 502", "http error 503", "http error 504",
            "timed out", "timeout", "connection reset", "connection refused",
            "remote end closed", "temporary failure in name resolution", "temporarily unavailable",
        ))

    def extract_info(
        self, url: str, download_options: DownloadOptions, *, cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        cancelled = cancel_event.is_set if cancel_event else None
        check_cancelled(cancelled)
        options = self.build_options(
            skip_download=True, download_options=download_options, cancel_event=cancel_event,
        )
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False, process=False)
            check_cancelled(cancelled)
            return info
        except DownloadError as exc:
            check_cancelled(cancelled)
            raise self.rewrite_error(exc) from exc

    def should_retry_with_cookies(
        self, url: str, active: DownloadOptions, configured: DownloadOptions | None,
        error: DownloadError,
    ) -> bool:
        return bool(
            configured and is_youtube_url(url)
            and (configured.cookies_file or configured.cookies_from_browser)
            and not (active.cookies_file or active.cookies_from_browser)
            and self._should_fallback_to_cookies(self._error_message(error), configured)
        )

    def _should_try_without_cookies(self, url: str, download_options: DownloadOptions) -> bool:
        if not is_youtube_url(url):
            return False
        return bool(download_options.cookies_from_browser or download_options.cookies_file)

    def _should_fallback_to_cookies(self, error_message: str, download_options: DownloadOptions) -> bool:
        if not (download_options.cookies_from_browser or download_options.cookies_file):
            return False

        markers = [
            "需要登录态 Cookie",
            "Sign in to confirm",
            "This content isn't available",
            "age-restricted",
            "members-only",
            "not a bot",
            "HTTP Error 403",
            "confirm your age",
        ]
        normalized = error_message.lower()
        return any(marker.lower() in normalized for marker in markers)

    def build_options(
        self, skip_download: bool, download_options: DownloadOptions, *, cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": skip_download,
            "ignoreconfig": True,
            "socket_timeout": 30,
            "retries": 2,
            "fragment_retries": 2,
            "extractor_retries": 0,
        }
        if cancel_event is not None:
            def check_progress(_status) -> None:
                check_cancelled(cancel_event.is_set)

            class CancellationLogger:
                # Extraction has no byte progress hook. Its normal log calls
                # still provide safe stopping points between HTTP requests.
                debug = staticmethod(check_progress)
                warning = staticmethod(check_progress)
                error = staticmethod(check_progress)

            options.update({
                "progress_hooks": [check_progress],
                "postprocessor_hooks": [check_progress],
                "logger": CancellationLogger(),
            })

        if download_options.cookies_from_browser:
            options["cookiesfrombrowser"] = (download_options.cookies_from_browser,)
        if download_options.cookies_file:
            options["cookiefile"] = download_options.cookies_file
        if download_options.js_runtime:
            runtime_config: dict[str, str] = {}
            if download_options.js_runtime_path:
                runtime_config["path"] = download_options.js_runtime_path
            options["js_runtimes"] = {download_options.js_runtime: runtime_config}

        return options

    def rewrite_error(self, exc: DownloadError) -> RuntimeError:
        message = ANSI_ESCAPE_RE.sub("", str(exc))
        if "Sign in to confirm you’re not a bot" in message or "Sign in to confirm you're not a bot" in message:
            return RuntimeError(
                "YouTube 要求验证访问请求，可能是临时访问限制。"
                "请稍后点击“继续未完成”；若浏览器也要求登录验证，完成验证后在更多设置中选择有效的 cookies.txt。"
            )
        if "age-restricted" in message.lower() or "confirm your age" in message.lower():
            return RuntimeError("该视频有年龄限制，请使用有访问权限的账号，并在更多设置中选择其 cookies.txt。")
        if "HTTP Error 429" in message:
            return RuntimeError("网站暂时限制了访问频率，请稍后点击“继续未完成”。")
        if "Could not copy Chrome cookie database" in message or "failed to load cookies" in message:
            return RuntimeError(
                "无法读取浏览器 Cookie 数据库。请先完全关闭浏览器后重试，"
                "如果仍失败，建议直接使用手动导出的 cookies.txt。"
            )
        if "Failed to decrypt with DPAPI" in message:
            return RuntimeError(
                "浏览器 Cookie 已读到，但系统没有成功解密。"
                "建议不要继续直读浏览器，改为在软件设置中选择手动导出的 cookies.txt。"
            )
        if "members on level" in message or "members-only" in message:
            return RuntimeError(
                "该视频是频道会员专享内容，当前 cookies 对应账号没有访问权限。"
            )
        if "Requested format is not available" in message:
            return RuntimeError(
                "当前视频没有可用的音频或视频格式可供 yt-dlp 提取，通常是 YouTube 提取受限导致的。"
            )
        if "HTTP Error 403" in message:
            return RuntimeError(
                "网站拒绝下载音频或视频（HTTP 403）。请确认该内容在浏览器中可以播放；"
                "若仍失败，可在更多设置中配置网站 cookies 后重试。"
            )
        return RuntimeError(message)
