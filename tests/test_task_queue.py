from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from info_summary.models import (
    OutputLanguage,
    ProcessingRequest,
    ProcessingStep,
    SourceType,
    TextProviderKind,
    VideoDownloadRequest,
    VideoQuality,
)
from info_summary.task_queue import TaskQueue, UnfinishedTasks


class TaskQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "app-data" / "unfinished_tasks.json"
        self.logs: list[str] = []
        self.store = UnfinishedTasks(self.path, self.logs.append)

    def request(self, name: str, **settings) -> ProcessingRequest:
        return ProcessingRequest(
            source=f"https://youtu.be/{name}", source_type=SourceType.VIDEO_URL,
            output_dir=str(self.root / "results"), **settings,
        )

    def test_append_deduplicates_existing_and_new_tasks_without_interrupting_current_task(self) -> None:
        queue = TaskQueue(self.store)
        first, second, third = (self.request(name) for name in ("first", "second", "third"))
        queue.add([("process", first), ("process", second)])
        current = queue.claim()
        added = queue.add([
            ("process", replace(first, text_model_name="other-model")),
            ("process", third), ("process", third),
            ("download", VideoDownloadRequest(first.source, str(self.root))),
        ])
        self.assertEqual([item.index for item in added], [2, 3])
        self.assertEqual(current.status, "处理中")
        self.assertIs(queue.items[0], current)
        self.assertEqual(queue.claim().request, second)
        self.assertEqual(queue.claim().request, third)
        self.assertEqual(queue.claim().kind, "download")
        self.assertIsNone(queue.claim())
        self.assertEqual(len(self.store.pending()), 4)

    def test_only_waiting_or_stopped_tasks_can_be_removed_and_row_numbers_stay_stable(self) -> None:
        queue = TaskQueue(self.store)
        first, second = self.request("first"), self.request("second")
        queue.add([("process", first), ("process", second)])
        current = queue.claim()
        self.assertFalse(queue.remove(current.index))
        self.assertTrue(queue.remove(1))
        self.assertFalse(queue.remove(1))
        self.assertFalse(queue.remove(-1))
        self.assertFalse(queue.remove(99))
        self.assertIsNone(queue.claim())
        self.assertEqual(self.store.pending(), [("process", first)])
        readded = queue.add([("process", second)], force_restart=True)
        self.assertEqual(readded[0].index, 2)
        self.assertTrue(readded[0].force_restart)
        stopped = queue.stop_waiting()
        self.assertEqual([item.index for item in stopped], [0, 2])
        self.assertTrue(queue.remove(2))
        self.assertEqual(self.store.pending(), [("process", first)])

    def test_claim_and_remove_cannot_both_take_the_same_waiting_task(self) -> None:
        def after_barrier(barrier, operation):
            barrier.wait(timeout=5)
            return operation()

        with ThreadPoolExecutor(max_workers=2) as workers:
            for _ in range(20):
                queue = TaskQueue()
                queue.add([("process", self.request("race"))])
                barrier = threading.Barrier(2)
                claiming = workers.submit(after_barrier, barrier, queue.claim)
                removing = workers.submit(after_barrier, barrier, lambda: queue.remove(0))
                claimed, removed = claiming.result(timeout=5), removing.result(timeout=5)
                self.assertNotEqual(claimed is not None, removed)
                self.assertEqual(queue.items[0].status, "已移除" if removed else "处理中")

    def test_success_is_removed_from_disk_but_failed_partial_and_stopped_tasks_remain(self) -> None:
        queue = TaskQueue(self.store)
        requests = [self.request(name) for name in ("success", "failure", "partial", "stopped")]
        queue.add([("process", request) for request in requests])
        for status in ("已完成", "失败", "部分完成"):
            queue.finish(queue.claim(), status)
        queue.stop_waiting()
        expected = [("process", request) for request in requests[1:]]
        self.assertEqual(self.store.pending(), expected)
        self.assertEqual(UnfinishedTasks(self.path).pending(), expected)
        self.assertFalse(queue.has_waiting())
        self.assertEqual(queue.counts(), {
            "completed": 1, "partial": 1, "failed": 1, "cancelled": True,
            "total": 4, "retry_pending": expected,
        })

    def test_starting_another_batch_does_not_discard_earlier_unfinished_tasks(self) -> None:
        earlier, later = self.request("earlier"), self.request("later")
        first_queue = TaskQueue(self.store)
        first_queue.add([("process", earlier)])
        first_queue.finish(first_queue.claim(), "失败")
        second_queue = TaskQueue(self.store)
        second_queue.add([("process", later)])
        self.assertEqual(UnfinishedTasks(self.path).pending(), [("process", earlier), ("process", later)])
        second_queue.finish(second_queue.claim(), "已完成")
        self.assertEqual(UnfinishedTasks(self.path).pending(), [("process", earlier)])

    def test_restart_preserves_processing_and_download_settings(self) -> None:
        process = self.request(
            "process", steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY),
            export_formats=("md",), output_language=OutputLanguage.ENGLISH,
            text_provider=TextProviderKind.OPENCODE, text_model_name="summary-model",
            text_model_base_url="https://example.test/v1", batch_size=4,
            include_visual=True, visual_focus="重点说明 05:20 的图表",
        )
        download = VideoDownloadRequest(
            "https://b23.tv/download", str(self.root / "downloads"), VideoQuality.HD_720,
        )
        expected = [("process", process), ("download", download)]
        self.store.add(expected)
        restored = UnfinishedTasks(self.path)
        self.assertEqual(restored.pending(), expected)
        restarted_queue = TaskQueue(restored)
        restarted_queue.add(restored.pending())
        self.assertEqual(restarted_queue.claim().request, process)
        self.assertEqual(restarted_queue.claim().request, download)

    def test_corrupt_records_do_not_hide_valid_records(self) -> None:
        process = self.request("valid")
        valid = {"kind": "process", "request": asdict(process)}
        broken_fields = [
            {"source": ["not", "text"]}, {"source": ""}, {"source": "bad\0source"},
            {"source": "bad\ud800source"}, {"text_model_name": "bad\ud800model"},
            {"source_type": "unknown"}, {"steps": ["unknown"]}, {"steps": None},
            {"batch_size": True}, {"batch_size": 0}, {"text_model_chunk_chars": "3500"},
            {"export_formats": ["exe"]}, {"export_formats": []}, {"text_provider": "unknown"},
            {"include_visual": "true"}, {"include_visual": 1}, {"visual_focus": ["bad"]},
        ]
        records = [None, "bad record", {}, {"kind": [], "request": {}},
                   {"kind": "unknown", "request": {}}, {"kind": "process", "request": {}}]
        records.extend({"kind": "process", "request": dict(valid["request"], **changes)}
                       for changes in broken_fields)
        records.insert(2, valid)
        download = VideoDownloadRequest("https://b23.tv/valid-download")
        records.append({"kind": "download", "request": asdict(download)})
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(records), encoding="utf-8")
        self.assertEqual(UnfinishedTasks(self.path).pending(), [("process", process), ("download", download)])

    def test_damaged_queue_file_does_not_prevent_new_tasks(self) -> None:
        self.path.parent.mkdir(parents=True)
        for contents in (b"not json", b"{", b'{}', b'null', b'\xff\xfe'):
            with self.subTest(contents=contents):
                self.path.write_bytes(contents)
                restored = UnfinishedTasks(self.path)
                self.assertEqual(restored.pending(), [])
                expected = [("process", self.request("new"))]
                restored.add(expected)
                self.assertEqual(UnfinishedTasks(self.path).pending(), expected)

    def test_html_and_existing_export_formats_survive_restart_without_conversion(self) -> None:
        old = self.request("old", export_formats=("md",))
        new = self.request("new", export_formats=("docx", "html"))
        expected = [("process", old), ("process", new)]
        self.store.add(expected)
        before = self.path.read_bytes()
        self.assertEqual(UnfinishedTasks(self.path).pending(), expected)
        self.assertEqual(self.path.read_bytes(), before)

    def test_write_failure_keeps_memory_queue_usable_and_previous_file_intact(self) -> None:
        previous, new = self.request("previous"), self.request("new")
        self.store.add([("process", previous)])
        saved = self.path.read_bytes()
        queue = TaskQueue(self.store)
        with patch("info_summary.storage.os.fsync", side_effect=OSError("disk full")):
            queue.add([("process", new)])
            self.assertEqual(queue.claim().request, new)
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(self.store.pending(), [("process", previous), ("process", new)])
        self.assertEqual(list(self.path.parent.glob(".*.tmp")), [])
        self.assertTrue(any("本次仍可处理" in line for line in self.logs))
        queue.finish(queue.items[0], "已完成")
        self.assertEqual(UnfinishedTasks(self.path).pending(), [("process", previous)])

    def test_credentials_are_not_written_to_task_file(self) -> None:
        secret = "test-online-secret-that-must-not-be-persisted"
        process = self.request("online", text_provider=TextProviderKind.OPENCODE)
        with patch.dict("os.environ", {"OPENCODE_API_KEY": secret}):
            self.store.add([("process", process)])
        self.assertNotIn(secret, self.path.read_text(encoding="utf-8"))
        records = json.loads(self.path.read_text(encoding="utf-8"))
        records[0]["request"]["api_key"] = secret
        records[0]["api_key"] = secret
        self.path.write_text(json.dumps(records), encoding="utf-8")
        restored = UnfinishedTasks(self.path)
        restored.add([])
        contents = self.path.read_text(encoding="utf-8")
        self.assertNotIn(secret, contents)
        self.assertNotIn("api_key", contents)
        self.assertEqual(restored.pending(), [("process", process)])

    def test_legacy_poe_tasks_keep_sources_but_clear_old_service_configuration(self) -> None:
        original = self.request("legacy-pending", text_model_name="old/model", text_model_base_url="https://api.poe.com/v1")
        data = asdict(original)
        data["text_provider"] = "poe"
        data["poe_api_key"] = "old-service-only-key"
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps([{"kind": "process", "request": data}]), encoding="utf-8")
        logs = []
        restored = UnfinishedTasks(self.path, logs.append)
        expected = replace(original, text_provider=TextProviderKind.OPENCODE, text_model_name="", text_model_base_url="")
        self.assertEqual(restored.pending(), [("process", expected)])
        self.assertEqual(original.source_fingerprint(self.root), expected.source_fingerprint(self.root))
        self.assertEqual(len(logs), 1)
        self.assertIn("已保留 1 个旧 Poe 未完成任务", logs[0])
        self.assertIn("继续未完成", logs[0])
        self.assertNotIn("old-service-only-key", logs[0])
        restored.add([])
        saved = self.path.read_text(encoding="utf-8")
        self.assertNotIn("poe", saved)
        self.assertNotIn("old/model", saved)
        self.assertNotIn("old-service-only-key", saved)

    def test_legacy_local_tasks_keep_their_model_and_address(self) -> None:
        local = self.request("local", text_provider=TextProviderKind.OLLAMA,
                             text_model_name="local/model", text_model_base_url="http://localhost:2222")
        self.store.add([("process", local)])
        logs = []
        self.assertEqual(UnfinishedTasks(self.path, logs.append).pending(), [("process", local)])
        self.assertEqual(logs, [])


if __name__ == "__main__":
    unittest.main()
