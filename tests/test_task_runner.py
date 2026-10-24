from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary.models import (ProcessingRequest, ProcessingResult, SourceType,
                                 VideoDownloadRequest, VideoDownloadResult, VideoQuality)
from info_summary.task_queue import TaskQueue
from info_summary.task_runner import TaskEvent, TaskRunner


class TaskRunnerTests(unittest.TestCase):
    def test_mixed_queue_emits_typed_events_without_a_window(self):
        work = TaskQueue()
        text = ProcessingRequest("audio.wav", SourceType.AUDIO_FILE)
        video = VideoDownloadRequest("https://vimeo.com/1")
        work.add([("process", text), ("download", video)])
        processed = ProcessingResult("文字", text.source, "audio", "zh", Path("text"), {})
        downloaded = VideoDownloadResult("视频", video.url, VideoQuality.BEST, Path("video"), Path("video/a.mp4"))
        processing, download = MagicMock(), MagicMock()
        processing.run.return_value = processed
        download.run.return_value = downloaded
        events = []
        runner = TaskRunner(Path.cwd(), events.append, threading.Event())
        with patch("info_summary.service.ProcessingService", return_value=processing), patch("info_summary.video.VideoDownloadService", return_value=download):
            runner.run(work, "key")
        self.assertTrue(all(isinstance(event, TaskEvent) for event in events))
        self.assertEqual([event.payload for event in events if event.kind == "result"], [processed, downloaded])
        self.assertEqual(events[-1].kind, "idle")
        self.assertEqual(events[-1].payload["completed"], 2)
        self.assertEqual(processing.run.call_args.kwargs["api_key"], "key")
        self.assertEqual(download.run.call_args.args[0], video)

    def test_cancellation_before_claim_keeps_every_item_for_retry(self):
        work = TaskQueue()
        work.add([("process", ProcessingRequest("a.wav", SourceType.AUDIO_FILE))])
        cancellation = threading.Event()
        cancellation.set()
        events = []
        with patch("info_summary.service.ProcessingService") as factory:
            TaskRunner(Path.cwd(), events.append, cancellation).run(work, "")
        factory.assert_not_called()
        self.assertTrue(events[-1].payload["cancelled"])
        self.assertEqual(events[-1].payload["failed"], 0)
        self.assertEqual(len(events[-1].payload["retry_pending"]), 1)
