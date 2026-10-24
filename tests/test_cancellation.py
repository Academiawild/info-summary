from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yt_dlp.utils import DownloadError

from info_summary.audio.audio_transcriber import AudioTranscriber
from info_summary.cancellation import JobCancelled
from info_summary.checkpoints import TextCheckpointStore
from info_summary.models import DownloadOptions, OutputLanguage, TranscriptSegment
from info_summary.providers import ProviderError, generate_with_retry
from info_summary.text_pipeline import ResumableTextProcessor
from info_summary.video.video_extractor import VideoExtractor


class AudioCancellationTests(unittest.TestCase):
    def test_progress_reports_recognized_minutes_against_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcriber = AudioTranscriber(Path(temp))
            segments = iter([
                SimpleNamespace(start=0, end=30, text="第一段"),
                SimpleNamespace(start=30, end=90, text="第二段"),
            ])
            transcriber.run_faster_whisper = MagicMock(return_value=(
                segments, SimpleNamespace(duration=120, language="zh"),
            ))
            updates = []
            result = transcriber.transcribe_audio(
                Path("audio.wav"), "tiny", "", "cpu", "int8", 4,
                progress=lambda fraction, message: updates.append((fraction, message)),
            )
        self.assertEqual([value for value, _ in updates], [0.25, 0.75])
        self.assertIn("第 1.5 分钟", updates[-1][1])
        self.assertIn("共 2.0 分钟", updates[-1][1])
        self.assertEqual([item.text for item in result], ["第一段", "第二段"])

    def test_cancel_does_not_request_another_whisper_batch_and_closes_iterator(self) -> None:
        cancel = threading.Event()
        read_segments, closed = [], []

        def segments():
            try:
                for index in range(3):
                    read_segments.append(index)
                    yield SimpleNamespace(start=index, end=index + 1, text="内容")
            finally:
                closed.append(True)

        with tempfile.TemporaryDirectory() as temp:
            transcriber = AudioTranscriber(Path(temp))
            with self.assertRaises(JobCancelled):
                transcriber.collect_segments(
                    segments(), duration=3, cancel_event=cancel,
                    progress=lambda *_args: cancel.set(),
                )
        self.assertEqual(read_segments, [0])
        self.assertEqual(closed, [True])

    def test_cancellation_during_whisper_batch_is_observed_before_progress(self) -> None:
        cancel = threading.Event()

        def segments():
            cancel.set()
            yield SimpleNamespace(start=0, end=1, text="内容")

        progress = MagicMock()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(JobCancelled):
                AudioTranscriber(Path(temp)).collect_segments(
                    segments(), progress=progress, cancel_event=cancel,
                )
        progress.assert_not_called()

    def test_ffmpeg_cancellation_stops_and_reaps_the_process(self) -> None:
        cancel = threading.Event()
        processes = []
        original_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            cancel.set()
            return process

        started = time.monotonic()
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.media_commands.subprocess.Popen", side_effect=launch,
        ):
            with self.assertRaises(JobCancelled):
                AudioTranscriber(Path(temp)).run_command(
                    [sys.executable, "-c", "import time; time.sleep(30)"], cancel_event=cancel,
                )
        self.assertLess(time.monotonic() - started, 4)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stderr.closed)

    def test_already_cancelled_conversion_does_not_start_process(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.media_commands.subprocess.Popen",
        ) as start:
            with self.assertRaises(JobCancelled):
                AudioTranscriber(Path(temp)).run_command(["ffmpeg"], cancel_event=cancel)
        start.assert_not_called()

    def test_cancel_immediately_after_process_exit_also_closes_pipes(self) -> None:
        cancel = threading.Event()
        processes = []
        original_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            process.wait(timeout=2)
            cancel.set()
            return process

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.media_commands.subprocess.Popen", side_effect=launch,
        ):
            with self.assertRaises(JobCancelled):
                AudioTranscriber(Path(temp)).run_command(
                    [sys.executable, "-c", "pass"], cancel_event=cancel,
                )
        self.assertEqual(processes[0].returncode, 0)
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stderr.closed)


class TextCancellationTests(unittest.TestCase):
    def test_failed_request_after_cancellation_is_not_retried(self) -> None:
        cancel = threading.Event()
        provider = MagicMock()

        def generate(*_args):
            cancel.set()
            raise ProviderError("网络暂时不可用")

        provider.generate.side_effect = generate
        with patch("info_summary.providers.time.sleep") as sleep:
            with self.assertRaises(JobCancelled):
                generate_with_retry(provider, "model", "system", "text", cancelled=cancel.is_set)
        self.assertEqual(provider.generate.call_count, 1)
        sleep.assert_not_called()

    def test_retry_wait_can_be_cancelled_without_second_request(self) -> None:
        cancel = threading.Event()
        provider = MagicMock()
        provider.generate.side_effect = ProviderError("网络暂时不可用")
        with patch("info_summary.providers.time.sleep", side_effect=lambda _delay: cancel.set()) as sleep:
            with self.assertRaises(JobCancelled):
                generate_with_retry(provider, "model", "system", "text", cancelled=cancel.is_set)
        self.assertEqual(provider.generate.call_count, 1)
        sleep.assert_called_once_with(0.1)

    def test_successful_polish_is_saved_before_obeying_cancel(self) -> None:
        cancel = threading.Event()
        provider = MagicMock()

        def generate(*_args):
            cancel.set()
            return "整理好的文字"

        provider.generate.side_effect = generate
        with tempfile.TemporaryDirectory() as temp:
            processor = ResumableTextProcessor()
            args = ([TranscriptSegment(0, 1, "原文")], "原文", provider, "model",
                    OutputLanguage.FOLLOW_SOURCE, 100, Path(temp))
            with self.assertRaises(JobCancelled):
                processor.polish(*args, cancelled=cancel.is_set)
            self.assertTrue((Path(temp) / "polish_0000.json").exists())
            cancel.clear()
            provider.generate.reset_mock()
            output, _parts, warnings = ResumableTextProcessor().polish(*args, cancelled=cancel.is_set)
        self.assertEqual(output, "整理好的文字")
        self.assertEqual(warnings, [])
        provider.generate.assert_not_called()

    def test_successful_final_summary_is_saved_before_obeying_cancel(self) -> None:
        cancel = threading.Event()
        provider = MagicMock()
        outputs = iter(["分段摘要", "整篇总结"])

        def generate(*_args):
            output = next(outputs)
            if output == "整篇总结":
                cancel.set()
            return output

        provider.generate.side_effect = generate
        with tempfile.TemporaryDirectory() as temp:
            args = (["原文"], provider, "model", OutputLanguage.FOLLOW_SOURCE, 100, Path(temp), None)
            with self.assertRaises(JobCancelled):
                ResumableTextProcessor().summarize(*args, cancelled=cancel.is_set)
            self.assertTrue((Path(temp) / "summary_final.json").exists())
            cancel.clear()
            provider.generate.reset_mock()
            output, warnings = ResumableTextProcessor().summarize(*args, cancelled=cancel.is_set)
        self.assertEqual((output, warnings), ("整篇总结", []))
        provider.generate.assert_not_called()


class CheckpointRobustnessTests(unittest.TestCase):
    def test_corrupt_cache_does_not_replace_original_with_invalid_text(self) -> None:
        store = TextCheckpointStore()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "polish.json"
            key = store.cache_key("原文", "model", OutputLanguage.FOLLOW_SOURCE)
            bodies = [b"[]", b"null", b"{", b"\xff",
                      json.dumps({"key": key, "output": ["错误"]}).encode(),
                      json.dumps({"key": key, "output": "\ud800"}).encode()]
            for body in bodies:
                with self.subTest(body=body):
                    path.write_bytes(body)
                    self.assertIsNone(store.load(path, "原文", "model", OutputLanguage.FOLLOW_SOURCE))

    def test_unwritable_checkpoint_directory_does_not_discard_generated_text(self) -> None:
        provider = MagicMock()
        provider.generate.return_value = "整理好的文字"
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.storage.Path.mkdir", side_effect=PermissionError("只读"),
        ):
            output, _parts, warnings = ResumableTextProcessor().polish(
                [TranscriptSegment(0, 1, "原文")], "原文", provider, "model",
                OutputLanguage.FOLLOW_SOURCE, 100, Path(temp) / "checkpoints",
            )
        self.assertEqual(output, "整理好的文字")
        self.assertTrue(any("进度未能保存" in warning for warning in warnings))

    def test_failed_atomic_replace_keeps_previous_checkpoint_and_removes_temp_file(self) -> None:
        store = TextCheckpointStore()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text("之前的缓存", encoding="utf-8")
            with patch("info_summary.storage.os.replace", side_effect=PermissionError):
                saved = store.save(path, "原文", "model", OutputLanguage.FOLLOW_SOURCE, "总结")
            self.assertFalse(saved)
            self.assertEqual(path.read_text(encoding="utf-8"), "之前的缓存")
            self.assertEqual(list(Path(temp).iterdir()), [path])


class VideoCancellationTests(unittest.TestCase):
    def test_cancel_during_information_lookup_does_not_try_cookies(self) -> None:
        cancel = threading.Event()
        extractor = VideoExtractor()

        def extract(*_args, **_kwargs):
            cancel.set()
            raise RuntimeError("Sign in to confirm")

        extractor.extract_info = MagicMock(side_effect=extract)
        with self.assertRaises(JobCancelled):
            extractor.extract_info_with_cookie_fallback(
                "https://www.youtube.com/watch?v=123", DownloadOptions(cookies_file="cookies.txt"),
                cancel_event=cancel,
            )
        self.assertEqual(extractor.extract_info.call_count, 1)

    def test_download_progress_hook_raises_cancellation(self) -> None:
        cancel = threading.Event()

        class Download:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                cancel.set()
                self.options["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 100})

        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_extractor.YoutubeDL", Download,
        ):
            with self.assertRaises(JobCancelled):
                VideoExtractor().download_audio(
                    "https://vimeo.com/123", {"id": "123"}, Path(temp), DownloadOptions(), cancel_event=cancel,
                )

    def test_cancelled_audio_error_is_not_retried_with_cookies(self) -> None:
        cancel = threading.Event()
        extractor = VideoExtractor()

        def download(*_args, **_kwargs):
            cancel.set()
            raise DownloadError("HTTP Error 403")

        extractor._download_audio_once = MagicMock(side_effect=download)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(JobCancelled):
                extractor.download_audio(
                    "https://www.youtube.com/watch?v=123", {"id": "123"}, Path(temp), DownloadOptions(),
                    fallback_options=DownloadOptions(cookies_file="cookies.txt"), cancel_event=cancel,
                )
        self.assertEqual(extractor._download_audio_once.call_count, 1)

    def test_cancel_during_subtitle_read_is_not_swallowed_as_missing_subtitles(self) -> None:
        cancel = threading.Event()

        class Response(io.BytesIO):
            def read(self, size=-1):
                cancel.set()
                return super().read(size)

        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.urlopen.return_value = Response(b"WEBVTT\n")
        info = {"title": "Demo", "subtitles": {"en": [{"url": "https://example.com/sub.vtt", "ext": "vtt"}]}}
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.video.video_extractor.YoutubeDL", return_value=ydl,
        ):
            with self.assertRaises(JobCancelled):
                VideoExtractor().try_extract_subtitles(
                    info, Path(temp), DownloadOptions(), lambda value: value, cancel_event=cancel,
                )
            self.assertEqual(list((Path(temp) / "subtitle").iterdir()), [])
        self.assertEqual(ydl.urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
