from __future__ import annotations

import hashlib
import json
import multiprocessing
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from info_summary.checkpoints import TextCheckpointStore
from info_summary.history import ResultHistory
from info_summary.models import (
    OutputLanguage, ProcessingRequest, ProcessingResult, ProcessingStep, SourceType,
    VideoDownloadRequest, VideoDownloadResult, VideoQuality,
)
from info_summary.result_index import ResultIndex, VideoDownloadIndex
from info_summary.storage import read_json, update_json, write_json_atomic
from info_summary.video.visual_cache import VisualCache, digest


def _increment_worker(path: str, barrier) -> None:
    barrier.wait(timeout=15)
    for _ in range(6):
        def increment(values: dict) -> None:
            previous = values.get("count", 0)
            time.sleep(0.005)  # Force overlapping read/modify operations without a lock.
            values["count"] = previous + 1
        update_json(Path(path), dict, increment)


def _record_worker(root_name: str, number: int, barrier) -> None:
    root = Path(root_name)
    processing_index = ResultIndex(root / "results.json")
    video_index = VideoDownloadIndex(root / "video_downloads.json")
    history = ResultHistory(root / "recent_results.json")
    barrier.wait(timeout=15)
    for item in range(6):
        title = f"{number}-{item}"
        folder = root / title
        folder.mkdir()
        report = folder / "阅读报告.html"
        report.write_text("<p>完整结果</p>", encoding="utf-8")
        request = ProcessingRequest(
            f"https://vimeo.com/{number}{item}", SourceType.VIDEO_URL,
            steps=(ProcessingStep.ORIGINAL,), export_formats=("html",),
        )
        result = ProcessingResult(title, request.source, "subtitle", "zh", folder, {"report.html": report})
        processing_index.register(request, root, result)
        video = folder / "video.mp4"
        video.write_bytes(b"video")
        video_request = VideoDownloadRequest(request.source)
        video_result = VideoDownloadResult(title, request.source, VideoQuality.BEST, folder, video)
        video_index.register(video_request, video_result)
        history.record(result)


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "state.json"

    def test_reader_ignores_damage_and_wrong_top_level_types(self) -> None:
        self.assertEqual(read_json(self.path), {})
        for data in (b"{", b"\xff", b"[]", b"null", b"42", b'"value"', b'{"bad":NaN}', b"[" * 2000 + b"]" * 2000):
            with self.subTest(data=data[:20]):
                self.path.write_bytes(data)
                self.assertEqual(read_json(self.path), {})
        self.path.write_text('[{"title":"旧结果"}]', encoding="utf-8")
        self.assertEqual(read_json(self.path, list), [{"title": "旧结果"}])
        self.path.unlink()
        first = read_json(self.path)
        first["changed"] = True
        self.assertEqual(read_json(self.path), {})

    def test_writer_creates_private_file_before_content_and_preserves_unicode(self) -> None:
        original_dump = json.dump
        modes = []

        def inspect(payload, stream, **kwargs):
            modes.append(stat.S_IMODE(Path(stream.name).stat().st_mode))
            self.assertEqual(stream.tell(), 0)
            return original_dump(payload, stream, **kwargs)

        with patch("info_summary.storage.json.dump", side_effect=inspect):
            write_json_atomic(self.path, {"正文": "中文结果"})
        self.assertEqual(modes, [0o600])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(read_json(self.path), {"正文": "中文结果"})
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_failure_at_each_write_stage_keeps_old_file_and_cleans_temporary_files(self) -> None:
        write_json_atomic(self.path, {"previous": True})
        previous = self.path.read_bytes()
        for target in ("tempfile.NamedTemporaryFile", "json.dump", "os.fsync", "os.replace"):
            with self.subTest(stage=target), patch("info_summary.storage." + target, side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_json_atomic(self.path, {"next": True})
            self.assertEqual(self.path.read_bytes(), previous)
            self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_unserializable_payload_does_not_replace_existing_data(self) -> None:
        write_json_atomic(self.path, {"previous": True})
        previous = self.path.read_bytes()
        for value in (object(), float("nan"), "\ud800"):
            with self.subTest(value=type(value).__name__), self.assertRaises((TypeError, ValueError, UnicodeError)):
                write_json_atomic(self.path, {"invalid": value})
            self.assertEqual(self.path.read_bytes(), previous)
            self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def _run_processes(self, target, arguments: list[tuple]) -> None:
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(len(arguments) + 1)
        processes = [context.Process(target=target, args=(*args, barrier)) for args in arguments]
        try:
            for process in processes:
                process.start()
            barrier.wait(timeout=15)
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

    def test_update_lock_covers_complete_read_modify_write_between_processes(self) -> None:
        self._run_processes(_increment_worker, [(str(self.path),)] * 3)
        self.assertEqual(read_json(self.path), {"count": 18})
        self.assertTrue(self.path.with_name("state.json.lock").is_file())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_two_indexes_and_history_keep_every_concurrent_registration(self) -> None:
        self._run_processes(_record_worker, [(str(self.root), number) for number in range(3)])
        self.assertEqual(len(read_json(self.root / "results.json")), 18)
        self.assertEqual(len(read_json(self.root / "video_downloads.json")), 18)
        self.assertEqual(len(ResultHistory(self.root / "recent_results.json").recent()), 18)


class ResultIndexCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "results.json"
        self.index = ResultIndex(self.path)
        self.folder = self.root / "result"
        self.folder.mkdir()
        self.report = self.folder / "阅读报告.html"
        self.report.write_text("<p>报告</p>", encoding="utf-8")
        self.request = ProcessingRequest(
            "https://vimeo.com/42", SourceType.VIDEO_URL,
            steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY), export_formats=("html",),
        )
        self.result = ProcessingResult("结果", self.request.source, "subtitle", "zh", self.folder,
                                       {"report.html": self.report})

    def test_html_uses_one_report_instead_of_a_file_for_each_step(self) -> None:
        self.index.register(self.request, self.root, self.result)
        self.assertEqual(self.index.find(self.request, self.root), self.result)
        self.report.unlink()
        self.assertIsNone(self.index.find(self.request, self.root))

    def test_visual_html_still_requires_exported_original_images(self) -> None:
        from dataclasses import replace
        request = replace(self.request, include_visual=True)
        image = self.folder / "frame.jpg"
        image.write_bytes(b"image")
        self.result.output_files["image.frame.jpg"] = image
        self.index.register(request, self.root, self.result)
        self.assertIsNotNone(self.index.find(request, self.root))
        image.unlink()
        self.assertIsNone(self.index.find(request, self.root))
        self.assertTrue(self.report.is_file())

    def test_html_and_word_require_both_selected_formats(self) -> None:
        from dataclasses import replace
        request = replace(self.request, export_formats=("docx", "html"))
        for name in ("original.docx", "summary.docx"):
            path = self.folder / name
            path.touch()
            self.result.output_files[name] = path
        self.index.register(request, self.root, self.result)
        self.assertIsNotNone(self.index.find(request, self.root))
        self.result.output_files["summary.docx"].unlink()
        self.assertIsNone(self.index.find(request, self.root))

    def test_invalid_utf8_or_json_indexes_do_not_block_processing(self) -> None:
        video_index = VideoDownloadIndex(self.path)
        for data in (b"\xff", b"{", b"[]", b"[" * 2000 + b"]" * 2000):
            with self.subTest(data=data[:20]):
                self.path.write_bytes(data)
                self.assertIsNone(self.index.find(self.request, self.root))
                self.assertIsNone(video_index.find(VideoDownloadRequest(self.request.source)))

    def test_stale_cleanup_cannot_remove_a_newer_registration(self) -> None:
        self.index.register(self.request, self.root, self.result)
        old = read_json(self.path)
        replacement = ProcessingResult("new", self.request.source, "subtitle", "zh", self.folder,
                                       {"report.html": self.report})
        self.index.register(self.request, self.root, replacement)
        old_data = next(iter(old.values()))
        old_data["output_files"] = {"report.html": str(self.folder / "missing.html")}
        with patch.object(self.index, "_read", return_value=old):
            self.assertIsNone(self.index.find(self.request, self.root))
        self.assertEqual(self.index.find(self.request, self.root).title, "new")


class CacheCompatibilityTests(unittest.TestCase):
    def test_text_store_reads_existing_key_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polish_0000.json"
            key = hashlib.sha256("model\0source\0原文".encode("utf-8")).hexdigest()
            path.write_text(json.dumps({"key": key, "output": "已保存的整理稿"}), encoding="utf-8")
            store = TextCheckpointStore()
            self.assertEqual(store.cache_key("原文", "model", OutputLanguage.FOLLOW_SOURCE), key)
            self.assertEqual(store.load(path, "原文", "model", OutputLanguage.FOLLOW_SOURCE), "已保存的整理稿")

    def test_visual_store_reads_existing_filenames_contexts_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"version": 1, "model": "vision", "prompt": "旧重点"}
            key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            self.assertEqual(digest(identity), key)
            (root / "calls").mkdir()
            (root / "calls" / (key + ".json")).write_text(json.dumps({"input": key, "output": "旧解读"}), encoding="utf-8")
            (root / "frame_contexts.json").write_text(json.dumps({"frame": {"focus": "旧重点", "questions": ["图表"]}}), encoding="utf-8")
            cache = VisualCache(root, [])
            self.assertEqual(cache.load(identity), "旧解读")
            self.assertEqual(cache.context("frame"), {"focus": "旧重点", "questions": ["图表"]})

    def test_visual_runs_keep_contexts_and_warnings_in_their_own_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_warnings, second_warnings = [], []
            first = VisualCache(root / "first", first_warnings)
            second = VisualCache(root / "second", second_warnings)
            first.save_context("frame", {"focus": "first", "questions": []})
            self.assertEqual(second.context("frame"), {})
            with patch("info_summary.video.visual_cache.write_json_atomic", side_effect=OSError("disk full")):
                first.save({"call": 1}, "完成的响应")
            self.assertEqual(len(first_warnings), 1)
            self.assertEqual(second_warnings, [])


if __name__ == "__main__":
    unittest.main()
