from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from info_summary.history import ResultHistory
from info_summary.models import ProcessingResult, VideoDownloadResult, VideoQuality


class ResultHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "app-data" / "recent_results.json"
        self.history = ResultHistory(self.path)

    def result(self, name: str, *, partial: bool = False) -> ProcessingResult:
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        return ProcessingResult(name, "https://youtu.be/example", "subtitle", "zh", folder, {},
                                ["总结未完成"] if partial else [])

    def test_records_both_kinds_and_preserves_partial_results_in_latest_first_order(self) -> None:
        complete = self.result("complete")
        partial = self.result("partial", partial=True)
        video_folder = self.root / "video"
        video_folder.mkdir()
        video = VideoDownloadResult("video", "https://youtu.be/video", VideoQuality.BEST,
                                    video_folder, video_folder / "视频.mp4")
        for result in (complete, partial, video):
            self.history.record(result)
        entries = ResultHistory(self.path).recent()
        self.assertEqual([entry["title"] for entry in entries], ["video", "partial", "complete"])
        self.assertEqual([entry["kind"] for entry in entries], ["download", "process", "process"])
        self.assertEqual([entry["status"] for entry in entries], ["已完成", "部分完成", "已完成"])
        self.assertEqual(set(entries[0]), {"title", "source", "result_dir", "created_at", "status", "kind"})
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_same_folder_moves_to_top_without_duplicating_and_limit_is_applied(self) -> None:
        one = self.result("one")
        self.history.record(one)
        self.history.record(self.result("two"))
        self.history.record(one)
        self.assertEqual([entry["title"] for entry in self.history.recent()], ["one", "two"])
        self.assertEqual(len(self.history.recent(1)), 1)
        self.assertEqual(self.history.recent(0), [])

    def test_oldest_entries_are_limited_to_one_hundred(self) -> None:
        for index in range(103):
            self.history.record(self.result(f"item-{index}"))
        self.assertEqual(len(json.loads(self.path.read_text())), 100)
        entries = self.history.recent(1000)
        self.assertEqual(len(entries), 100)
        self.assertEqual((entries[0]["title"], entries[-1]["title"]), ("item-102", "item-3"))

    def test_missing_directories_and_malformed_entries_are_ignored(self) -> None:
        self.history.record(self.result("keep"))
        self.history.record(self.result("deleted"))
        (self.root / "deleted").rmdir()
        entries = json.loads(self.path.read_text())
        valid = entries[-1]
        entries.extend([None, 7, {}, dict(valid, created_at="bad date"), dict(valid, kind="bad"),
                        dict(valid, result_dir="relative"), dict(valid, title=[]), valid])
        self.path.write_text(json.dumps(entries), encoding="utf-8")
        self.assertEqual([entry["title"] for entry in self.history.recent()], ["keep"])

    def test_damaged_or_missing_json_does_not_break_history_and_new_results_repair_it(self) -> None:
        self.assertEqual(self.history.recent(), [])
        self.path.parent.mkdir()
        for data in (b"{broken", b"null", b'{}', b"\xff\xfe"):
            with self.subTest(data=data):
                self.path.write_bytes(data)
                self.assertEqual(self.history.recent(), [])
        self.history.record(self.result("repaired"))
        self.assertEqual(self.history.recent()[0]["title"], "repaired")

    def test_failed_atomic_replace_preserves_previous_history_and_cleans_temporary_file(self) -> None:
        self.history.record(self.result("keep"))
        before = self.path.read_bytes()
        with patch("info_summary.storage.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.history.record(self.result("new"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
