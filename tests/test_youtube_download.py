from __future__ import annotations

import tempfile
import io
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from yt_dlp.utils import DownloadError

from info_summary.models import DownloadOptions, VideoDownloadRequest, VideoDownloadResult, VideoQuality
from info_summary.video.video_downloader import VideoDownloader, VideoDownloadService, VideoDownloadIndex
from info_summary.video.video_extractor import VideoExtractor
from info_summary.cancellation import JobCancelled
from info_summary.models import ProcessingRequest, ProcessingStep, SourceType
from info_summary.pipeline import ProcessingPipeline


class YoutubeDownloadRecoveryTests(unittest.TestCase):
    URL = "https://www.youtube.com/watch?v=example1234"

    def test_temporary_bot_check_recovers_without_cookies(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        info = {"id": "example1234", "subtitles": {"zh-TW": []}}
        ydl.extract_info.side_effect = [
            DownloadError("Sign in to confirm you’re not a bot"), info,
        ]
        log = MagicMock()
        with patch("info_summary.video.client.YoutubeDL", return_value=ydl) as factory, patch(
            "info_summary.video.client.time.sleep"
        ) as sleep:
            result, active = VideoExtractor(log=log).extract_info_with_cookie_fallback(self.URL, DownloadOptions())
        self.assertEqual(result, info)
        self.assertFalse(active.cookies_file)
        self.assertEqual(ydl.extract_info.call_count, 2)
        self.assertTrue(all("cookiefile" not in call.args[0] for call in factory.call_args_list))
        sleep.assert_called_once_with(15)
        self.assertIn("自动重试", log.call_args.args[0])

    def test_persistent_bot_check_stops_after_three_attempts_with_clear_message(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.extract_info.side_effect = DownloadError("Sign in to confirm you're not a bot")
        with patch("info_summary.video.client.YoutubeDL", return_value=ydl), patch(
            "info_summary.video.client.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(RuntimeError, "已自动尝试 3 次") as raised:
                VideoExtractor().extract_info_with_cookie_fallback(self.URL, DownloadOptions())
        self.assertEqual(ydl.extract_info.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15, 30])
        self.assertIn("可能是临时访问限制", str(raised.exception))
        self.assertNotIn("需要登录态 Cookie", str(raised.exception))

    def test_private_removed_and_age_restricted_content_is_not_retried(self):
        for message in ("Private video", "Video has been removed", "Video unavailable", "Sign in to confirm your age"):
            with self.subTest(message=message):
                ydl = MagicMock()
                ydl.__enter__.return_value = ydl
                ydl.extract_info.side_effect = DownloadError(message)
                with patch("info_summary.video.client.YoutubeDL", return_value=ydl), patch(
                    "info_summary.video.client.time.sleep"
                ) as sleep:
                    with self.assertRaises(RuntimeError):
                        VideoExtractor().extract_info_with_cookie_fallback(self.URL, DownloadOptions())
                self.assertEqual(ydl.extract_info.call_count, 1)
                sleep.assert_not_called()

    def test_configured_cookies_are_tried_once_then_retained_during_retry(self):
        seen = []
        responses = iter([DownloadError("Sign in to confirm you're not a bot"),
                          DownloadError("HTTP Error 503: Service Unavailable"), {"id": "example1234"}])

        class FakeYDL:
            def __init__(self, options):
                seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, *_args, **_kwargs):
                response = next(responses)
                if isinstance(response, Exception):
                    raise response
                return response

        configured = DownloadOptions(cookies_file="/example/cookies.txt")
        with patch("info_summary.video.client.YoutubeDL", FakeYDL), patch(
            "info_summary.video.client.time.sleep"
        ) as sleep:
            info, active = VideoExtractor().extract_info_with_cookie_fallback(self.URL, configured)
        self.assertEqual(info["id"], "example1234")
        self.assertEqual(active, configured)
        self.assertNotIn("cookiefile", seen[0])
        self.assertEqual([entry["cookiefile"] for entry in seen[1:]], [configured.cookies_file] * 2)
        sleep.assert_called_once_with(5)

    def test_cancelling_retry_wait_does_not_start_another_request(self):
        cancel = threading.Event()
        operation = MagicMock(side_effect=DownloadError("HTTP Error 503"))
        with patch.object(cancel, "wait", side_effect=lambda _delay: cancel.set()):
            with self.assertRaises(JobCancelled):
                VideoExtractor().with_download_retries(operation, "下载", cancel_event=cancel)
        self.assertEqual(operation.call_count, 1)

    def test_rate_limit_waits_longer_but_local_file_errors_are_not_retried(self):
        operation = MagicMock(side_effect=[DownloadError("HTTP Error 429"), "ok"])
        with patch("info_summary.video.client.time.sleep") as sleep:
            self.assertEqual(VideoExtractor().with_download_retries(operation, "读取"), "ok")
        sleep.assert_called_once_with(15)
        operation = MagicMock(side_effect=PermissionError("只读目录"))
        with patch("info_summary.video.client.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                VideoExtractor().with_download_retries(operation, "下载")
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    def test_audio_recovers_from_timeout_and_ignores_incomplete_files(self):
        calls = []

        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                calls.append(self.options)
                if len(calls) == 1:
                    raise DownloadError("Connection timed out")
                target = Path(self.options["outtmpl"].replace("%(ext)s", "m4a"))
                target.write_bytes(b"audio")
                target.with_suffix(".m4a.part").write_bytes(b"partial")

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_extractor.YoutubeDL", FakeYDL
        ), patch("info_summary.video.client.time.sleep"):
            result = VideoExtractor().download_audio(self.URL, {"id": "example1234"}, Path(temp), DownloadOptions())
            self.assertEqual(result.suffix, ".m4a")
            self.assertEqual(result.read_bytes(), b"audio")
        self.assertEqual(len(calls), 2)

    def test_video_recovers_from_temporary_server_error(self):
        calls = []

        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                calls.append(self.options)
                if len(calls) == 1:
                    raise DownloadError("HTTP Error 503: Service Unavailable")
                Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_downloader.YoutubeDL", FakeYDL
        ), patch("info_summary.video.client.time.sleep"):
            downloader = VideoDownloader()
            downloader.client.check_youtube_runtime = MagicMock()
            downloader.client.extract_info_with_cookie_fallback = MagicMock(
                return_value=({"title": "Demo"}, DownloadOptions())
            )
            result = downloader.download(VideoDownloadRequest(self.URL, output_dir=temp))
            self.assertEqual(result.video_path.read_bytes(), b"video")
        self.assertEqual(len(calls), 2)

    def test_unmerged_or_empty_video_is_not_reported_as_successful(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "视频.f137.mp4").write_bytes(b"video without sound")
            (root / "视频.mp4.part").write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "完整的 MP4"):
                VideoDownloader()._find_video(root)
            (root / "视频.mp4").touch()
            with self.assertRaises(RuntimeError):
                VideoDownloader()._find_video(root)

    def test_audio_retries_403_with_configured_cookies(self) -> None:
        seen: list[dict] = []

        class FakeYDL:
            def __init__(self, options):
                self.options = options
                seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                if "cookiefile" not in self.options:
                    raise DownloadError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
                Path(self.options["outtmpl"].replace("%(ext)s", "m4a")).write_bytes(b"audio")

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_extractor.YoutubeDL", FakeYDL
        ):
            result = VideoExtractor().download_audio(
                self.URL, {"id": "example1234"}, Path(temp), DownloadOptions(),
                fallback_options=DownloadOptions(cookies_file="/example/cookies.txt"),
            )
        self.assertEqual(result.name, "example1234.m4a")
        self.assertEqual(len(seen), 2)
        self.assertIn("bestaudio[ext=m4a]", seen[0]["format"])
        self.assertNotIn("cookiefile", seen[0])
        self.assertEqual(seen[1]["cookiefile"], "/example/cookies.txt")

    def test_video_retries_403_with_configured_cookies(self) -> None:
        seen: list[dict] = []

        class FakeYDL:
            def __init__(self, options):
                self.options = options
                seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                if "cookiefile" not in self.options:
                    raise DownloadError("ERROR: HTTP Error 403: Forbidden")
                Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_downloader.YoutubeDL", FakeYDL
        ):
            downloader = VideoDownloader()
            downloader.client.check_youtube_runtime = MagicMock()
            downloader.client.extract_info_with_cookie_fallback = MagicMock(
                return_value=({"title": "Demo", "id": "example1234"}, DownloadOptions())
            )
            result = downloader.download(VideoDownloadRequest(
                self.URL, output_dir=temp, cookies_file="/example/cookies.txt"
            ))
            self.assertTrue(result.video_path.is_file())
        self.assertEqual(len(seen), 2)
        self.assertNotIn("cookiefile", seen[0])
        self.assertEqual(seen[1]["cookiefile"], "/example/cookies.txt")

    def test_403_error_is_clear_and_has_no_terminal_color_codes(self) -> None:
        error = VideoExtractor().client.rewrite_error(
            DownloadError("\x1b[0;31mERROR:\x1b[0m HTTP Error 403: Forbidden")
        )
        self.assertIn("HTTP 403", str(error))
        self.assertIn("cookies", str(error))
        self.assertNotIn("\x1b", str(error))

    def test_saved_video_is_still_successful_when_index_cannot_be_written(self):
        request = VideoDownloadRequest("https://vimeo.com/1")
        result = VideoDownloadResult("saved", request.url, VideoQuality.BEST, Path("saved"), Path("saved/video.mp4"))
        downloader, index, log = MagicMock(), MagicMock(), MagicMock()
        downloader.download.return_value = result
        index.register.side_effect = PermissionError("read only")
        self.assertIs(VideoDownloadService(downloader, index, log).run(request), result)
        log.assert_called_once()

    def test_malformed_or_missing_cached_video_allows_redownload(self):
        request = VideoDownloadRequest("https://vimeo.com/1")
        index = VideoDownloadIndex()
        with patch.object(index, "_read", return_value={request.result_key(): {"quality": "unknown"}}):
            self.assertIsNone(index.find(request))
        data = VideoDownloadResult("gone", request.url, VideoQuality.BEST, Path("missing-dir"), Path("missing-dir/video.mp4")).to_index_data()
        with patch.object(index, "_read", return_value={request.result_key(): data}), patch(
            "info_summary.result_index.update_json", side_effect=PermissionError,
        ):
            self.assertIsNone(index.find(request))


class SubtitleRecoveryTests(unittest.TestCase):
    VTT = b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello world\n\n"

    def extract(self, info, ydl, root):
        ydl.__enter__.return_value = ydl
        with patch("info_summary.video.video_extractor.YoutubeDL", return_value=ydl), patch(
            "info_summary.video.client.time.sleep"
        ):
            return VideoExtractor().try_extract_subtitles(info, root, DownloadOptions(), lambda text: text)

    def test_subtitle_request_recovers_from_server_error_before_audio_fallback(self):
        info = {"title": "Demo", "subtitles": {"zh-TW": [{"url": "https://example.com/sub.vtt", "ext": "vtt"}]}}
        ydl = MagicMock()
        ydl.urlopen.side_effect = [HTTPError("https://example.com/sub.vtt", 503, "unavailable", {}, None), io.BytesIO(self.VTT)]
        with tempfile.TemporaryDirectory() as temp:
            segments, language, method = self.extract(info, ydl, Path(temp))
        self.assertEqual(segments[0].text, "hello world")
        self.assertEqual((language, method), ("zh-TW", "embedded_subtitle"))
        self.assertEqual(ydl.urlopen.call_count, 2)

    def test_empty_or_invalid_subtitle_tries_another_format(self):
        for payload in (b"WEBVTT\n\n", b"<html>Access denied</html>"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                info = {"title": "Demo", "subtitles": {"zh": [
                    {"url": "https://example.com/sub.vtt", "ext": "vtt"},
                    {"url": "https://example.com/sub.srt", "ext": "srt"},
                ]}}
                ydl = MagicMock()
                ydl.urlopen.side_effect = [io.BytesIO(payload), io.BytesIO(b"1\n00:00:00,000 --> 00:00:02,000\nvalid\n\n")]
                segments, language, method = self.extract(info, ydl, Path(temp))
                self.assertEqual(segments[0].text, "valid")
                self.assertEqual((language, method), ("zh", "embedded_subtitle"))
                self.assertEqual(ydl.urlopen.call_count, 2)
                self.assertFalse(any(Path(temp).rglob("*.vtt")))

    def test_missing_preferred_language_tries_another_language_or_auto_captions(self):
        for backup in ("manual", "auto"):
            with self.subTest(backup=backup), tempfile.TemporaryDirectory() as temp:
                info = {"title": "Demo", "subtitles": {"zh": [{"url": "https://example.com/zh.vtt", "ext": "vtt"}]}}
                key = "subtitles" if backup == "manual" else "automatic_captions"
                info.setdefault(key, {})["en"] = [{"url": "https://example.com/en.vtt", "ext": "vtt"}]
                ydl = MagicMock()
                ydl.urlopen.side_effect = [HTTPError("https://example.com/zh.vtt", 404, "not found", {}, None), io.BytesIO(self.VTT)]
                segments, language, method = self.extract(info, ydl, Path(temp))
                self.assertEqual(segments[0].text, "hello world")
                self.assertEqual(language, "en")
                self.assertEqual(method, "embedded_subtitle" if backup == "manual" else "auto_subtitle")
                self.assertEqual(ydl.urlopen.call_count, 2)

    def test_all_empty_subtitles_return_no_text_instead_of_false_success(self):
        info = {"title": "Demo", "subtitles": {"zh": [{"url": "https://example.com/sub.vtt", "ext": "vtt"}]}}
        ydl = MagicMock()
        ydl.urlopen.return_value = io.BytesIO(b"WEBVTT\n\n")
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(self.extract(info, ydl, Path(temp)), ([], "", ""))
            self.assertEqual(list((Path(temp) / "subtitle").iterdir()), [])

    def test_partial_subtitle_download_is_replaced_on_retry(self):
        class BrokenResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise TimeoutError("timed out")
                return super().read(size)

        info = {"title": "Demo", "subtitles": {"zh": [{"url": "https://example.com/sub.vtt", "ext": "vtt"}]}}
        ydl = MagicMock()
        ydl.urlopen.side_effect = [BrokenResponse(b"partial garbage"), io.BytesIO(self.VTT)]
        with tempfile.TemporaryDirectory() as temp:
            segments, _language, _method = self.extract(info, ydl, Path(temp))
            self.assertEqual(segments[0].text, "hello world")
            self.assertEqual(next((Path(temp) / "subtitle").iterdir()).read_bytes(), self.VTT)

    def test_complete_pipeline_recovers_bot_check_and_saves_subtitles_without_whisper(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.extract_info.side_effect = [
            DownloadError("Sign in to confirm you're not a bot"),
            {"id": "example1234", "title": "Recovered", "subtitles": {
                "zh": [{"url": "https://example.com/sub.vtt", "ext": "vtt"}],
            }},
        ]
        ydl.urlopen.return_value = io.BytesIO(self.VTT)
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_extractor.YoutubeDL", return_value=ydl
        ), patch("info_summary.video.client.YoutubeDL", return_value=ydl), patch("info_summary.video.client.time.sleep"):
            root = Path(temp)
            pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints")
            pipeline.video_extractor.check_youtube_runtime = MagicMock()
            pipeline.audio_transcriber.transcribe_audio = MagicMock(side_effect=AssertionError("不应识别声音"))
            result = pipeline.run(ProcessingRequest(
                YoutubeDownloadRecoveryTests.URL, SourceType.VIDEO_URL, output_dir=str(root / "out"),
                steps=(ProcessingStep.ORIGINAL,), export_formats=("md",),
            ))
            self.assertEqual(result.method, "embedded_subtitle")
            self.assertIn("hello world", result.output_files["original.md"].read_text(encoding="utf-8"))
            self.assertEqual(list((root / "temp" / "jobs").iterdir()), [])
            self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
