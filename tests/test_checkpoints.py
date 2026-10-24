from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary.checkpoints import CheckpointStore
from info_summary.models import (
    OutputLanguage, ProcessingRequest, ProcessingStep, SourceType, TextProviderKind,
    TranscriptSegment,
)
from info_summary.pipeline import ProcessingPipeline
from info_summary.providers import ProviderError
from info_summary.service import ProcessingService
from info_summary.storage import write_json_atomic
from info_summary.text_pipeline import JobCancelled


class CheckpointStoreTests(unittest.TestCase):
    def test_keys_reuse_only_compatible_work_and_never_save_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = ProcessingRequest("https://vimeo.com/123", SourceType.VIDEO_URL)
            store = CheckpointStore(root, root / "checkpoints")
            original = store.original_dir(request)
            text = store.text_dir(request, ["prompt v1"])
            for changes in (
                {"output_dir": "elsewhere"}, {"export_formats": ("docx",)},
                {"steps": (ProcessingStep.SUMMARY,)}, {"cookies_file": "private-cookie-path"},
            ):
                changed = replace(request, **changes)
                self.assertEqual(store.original_dir(changed), original)
                self.assertEqual(store.text_dir(changed, ["prompt v1"]), text)
            for changes in (
                {"text_provider": TextProviderKind.OLLAMA}, {"text_model_name": "other-model"},
                {"text_model_base_url": "http://another-model-host"},
                {"output_language": OutputLanguage.ENGLISH}, {"text_model_chunk_chars": 1234},
            ):
                changed = replace(request, **changes)
                self.assertEqual(store.original_dir(changed), original)
                self.assertNotEqual(store.text_dir(changed, ["prompt v1"]), text)
            self.assertNotEqual(store.text_dir(request, ["prompt v2"]), text)
            self.assertNotEqual(store.original_dir(replace(request, model_size="small")), original)
            self.assertNotEqual(store.original_dir(replace(request, language_hint="en")), original)
            secret_request = replace(request, cookies_file="secret-cookie-path", cookies_from_browser="secret-profile")
            store.save_original(secret_request, ("title", [TranscriptSegment(0, 1, "内容")], "subtitle", "zh"))
            content = (original / "original.json").read_text(encoding="utf-8")
            self.assertNotIn("secret", content)
            self.assertEqual((original / "original.json").stat().st_mode & 0o777, 0o600)

    def test_local_checkpoint_paths_remain_compatible_and_online_uses_configured_address(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore(root, root / "checkpoints")
            request = ProcessingRequest(
                "https://vimeo.com/123", SourceType.VIDEO_URL,
                text_provider=TextProviderKind.OLLAMA, text_model_name="local-model",
                text_model_base_url="http://127.0.0.1:11434/",
            )
            old_options = {
                "version": 1, "provider": "ollama", "base_url": "http://127.0.0.1:11434",
                "model": "local-model", "language": "zh-CN", "chunk_chars": 3500,
                "prompts": ["prompt"],
            }
            old_digest = hashlib.sha256(json.dumps(
                old_options, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            self.assertEqual(store.text_dir(request, ["prompt"]), store.original_dir(request) / "text" / old_digest)
            online = replace(request, text_provider=TextProviderKind.OPENCODE, text_model_base_url="https://one.test/v1")
            moved = replace(online, text_model_base_url="https://two.test/v1")
            self.assertEqual(store.original_dir(online), store.original_dir(request))
            self.assertEqual(store.original_dir(moved), store.original_dir(request))
            self.assertNotEqual(store.text_dir(online, ["prompt"]), store.text_dir(request, ["prompt"]))
            self.assertNotEqual(store.text_dir(online, ["prompt"]), store.text_dir(moved, ["prompt"]))
            self.assertEqual(store.text_dir(online, ["prompt"]), store.text_dir(replace(online, text_model_base_url="https://one.test/v1/"), ["prompt"]))

    def test_changed_local_file_does_not_reuse_old_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio.wav"
            audio.write_bytes(b"one")
            store = CheckpointStore(root, root / "checkpoints")
            request = ProcessingRequest(str(audio), SourceType.AUDIO_FILE)
            store.save_original(request, ("title", [TranscriptSegment(0, 1, "旧内容")], "audio", "zh"))
            self.assertIsNotNone(store.load_original(request))
            audio.write_bytes(b"different audio")
            self.assertIsNone(store.load_original(request))

    def test_corrupt_original_checkpoints_are_safely_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore(root, root / "checkpoints")
            request = ProcessingRequest("https://vimeo.com/123", SourceType.VIDEO_URL)
            path = store.original_dir(request) / "original.json"
            path.parent.mkdir(parents=True)
            valid = {"version": 1, "title": "title", "method": "subtitle", "language": "zh"}
            for content in (
                b"{broken", b"[]", b"\xff", b"null",
                json.dumps({**valid, "segments": [None]}).encode(),
                json.dumps({**valid, "segments": [{"start": 1, "end": 0, "text": "bad"}]}).encode(),
                json.dumps({**valid, "segments": [{"start": 0, "end": 1, "text": []}]}).encode(),
                json.dumps({**valid, "segments": [{"start": 0, "end": 1, "text": "\ud800"}]}).encode(),
            ):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    self.assertIsNone(store.load_original(request))

    def test_failed_atomic_save_preserves_previous_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore(root, root / "checkpoints")
            request = ProcessingRequest("https://vimeo.com/123", SourceType.VIDEO_URL)
            original = ("title", [TranscriptSegment(0, 1, "原文")], "subtitle", "zh")
            self.assertTrue(store.save_original(request, original))
            with patch("info_summary.storage.os.fsync", side_effect=OSError("disk full")):
                self.assertFalse(store.save_original(request, ("new", original[1], "subtitle", "zh")))
            self.assertEqual(store.load_original(request), original)
            self.assertEqual(list(store.original_dir(request).glob("*.tmp")), [])


class PipelineResumeTests(unittest.TestCase):
    def make_pipeline(self, root, *, segments=None, provider=None):
        pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints")
        segments = segments or [TranscriptSegment(0, 1, "已完成的原始内容。")]

        def original(_request, work_dir, *_args):
            (work_dir / "large-audio.wav").write_bytes(b"temporary audio")
            return "title", segments, "subtitle", "zh"

        pipeline._obtain_original = MagicMock(side_effect=original)
        provider = provider or MagicMock()
        provider.health_check.return_value = (True, "ok")
        if provider.generate.side_effect is None:
            provider.generate.return_value = "生成好的文字"
        pipeline._provider_for = MagicMock(return_value=provider)
        request = ProcessingRequest(
            "https://vimeo.com/123", SourceType.VIDEO_URL,
            output_dir=str(root / "output"), export_formats=("md",), text_model_name="fake",
        )
        return pipeline, provider, request

    def test_resume_after_failure_only_retries_unfinished_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, provider, request = self.make_pipeline(root)

            def generate(_model, prompt, _text):
                if "整篇内容的总结" in prompt:
                    raise ProviderError("final failed", retryable=False)
                return "已生成文字"

            provider.generate.side_effect = generate
            self.assertTrue(first.run(request).warnings)
            self.assertEqual(provider.generate.call_count, 3)
            self.assertEqual(list((root / "temp" / "jobs").iterdir()), [])
            second, resumed_provider, _ = self.make_pipeline(root)
            result = second.run(request)
            self.assertEqual(result.warnings, [])
            second._obtain_original.assert_not_called()
            self.assertEqual(resumed_provider.generate.call_count, 1)
            self.assertIn("整篇内容的总结", resumed_provider.generate.call_args.args[1])

    def test_cancellation_keeps_complete_chunks_for_next_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            segments = [TranscriptSegment(i * 2, i * 2 + 1, text * 8) for i, text in enumerate("甲乙丙")]
            first, provider, request = self.make_pipeline(root, segments=segments)
            request = replace(request, steps=(ProcessingStep.ORIGINAL, ProcessingStep.POLISH), text_model_chunk_chars=10)
            cancelled = threading.Event()

            def generate(*_args):
                cancelled.set()
                return "完成分块"

            provider.generate.side_effect = generate
            with self.assertRaises(JobCancelled):
                first.run(request, cancel_event=cancelled)
            self.assertEqual(list((root / "temp" / "jobs").iterdir()), [])
            self.assertFalse((root / "output").exists())
            second, resumed_provider, _ = self.make_pipeline(root, segments=segments)
            result = second.run(request)
            self.assertEqual(result.warnings, [])
            second._obtain_original.assert_not_called()
            self.assertEqual(resumed_provider.generate.call_count, 2)

    def test_incomplete_whisper_output_is_never_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline, _, request = self.make_pipeline(root)
            pipeline._obtain_original.side_effect = JobCancelled("任务已取消。")
            with self.assertRaises(JobCancelled):
                pipeline.run(request)
            self.assertIsNone(pipeline.checkpoints.load_original(request))
            self.assertEqual(list((root / "temp" / "jobs").iterdir()), [])

    def test_add_text_steps_and_switch_model_reuses_original_but_not_wrong_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline, provider, request = self.make_pipeline(root)
            pipeline.run(replace(request, steps=(ProcessingStep.ORIGINAL,)))
            provider.generate.assert_not_called()
            pipeline.run(request)
            self.assertEqual(pipeline._obtain_original.call_count, 1)
            self.assertEqual(provider.generate.call_count, 3)
            provider.generate.reset_mock()
            pipeline.run(replace(request, text_model_name="different-model"))
            self.assertEqual(pipeline._obtain_original.call_count, 1)
            self.assertEqual(provider.generate.call_count, 3)

    def test_export_failure_preserves_transcript_and_text_for_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _, request = self.make_pipeline(root)
            with patch.object(first.exporter, "export", side_effect=OSError("output is not writable")):
                with self.assertRaises(OSError):
                    first.run(request)
            second, provider, _ = self.make_pipeline(root)
            result = second.run(replace(request, output_dir=str(root / "writable-output")))
            self.assertTrue(result.output_files["summary.md"].is_file())
            second._obtain_original.assert_not_called()
            provider.generate.assert_not_called()

    def test_reexport_and_changed_steps_reuse_text_even_when_provider_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _, request = self.make_pipeline(root)
            first.run(request, api_key="private-key-must-not-be-saved")
            second, provider, _ = self.make_pipeline(root)
            provider.health_check.return_value = (False, "模型暂时离线")
            provider.generate.side_effect = ProviderError("模型暂时离线", retryable=False)
            result = second.run(replace(request, output_dir=str(root / "new-output"), export_formats=("docx",)))
            self.assertEqual(result.warnings, [])
            self.assertTrue(all(path.is_file() for path in result.output_files.values()))
            second._obtain_original.assert_not_called()
            provider.generate.assert_not_called()
            provider.health_check.assert_not_called()
            all_checkpoints = "".join(path.read_text() for path in (root / "checkpoints").rglob("*.json"))
            self.assertNotIn("private-key-must-not-be-saved", all_checkpoints)
            original_only = replace(request, steps=(ProcessingStep.ORIGINAL,))
            second.run(original_only)
            second._obtain_original.assert_not_called()

    def test_forced_restart_regenerates_all_work_and_service_forwards_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline, provider, request = self.make_pipeline(root)
            pipeline.run(request)
            provider.generate.reset_mock()
            pipeline.run(request, force_restart=True)
            self.assertEqual(pipeline._obtain_original.call_count, 2)
            self.assertEqual(provider.generate.call_count, 3)
            service_pipeline = MagicMock()
            ProcessingService(root, pipeline=service_pipeline, result_index=MagicMock()).run(request, force_restart=True)
            self.assertTrue(service_pipeline.run.call_args.kwargs["force_restart"])

    def test_checkpoint_write_failure_still_exports_successful_work(self):
        def fail_original_only(path, payload):
            if path.name == "original.json":
                raise OSError("disk full")
            write_json_atomic(path, payload)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline, _, request = self.make_pipeline(root)
            with patch("info_summary.checkpoints.write_json_atomic", side_effect=fail_original_only):
                result = pipeline.run(request)
            self.assertTrue(result.output_files["summary.md"].is_file())
            self.assertEqual(result.warnings, [])

    def test_failed_cache_clear_still_forces_new_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline, provider, request = self.make_pipeline(root)
            pipeline.run(request)
            provider.generate.reset_mock()
            with patch.object(pipeline.checkpoints, "clear_source", return_value=False):
                pipeline.run(request, force_restart=True)
            self.assertEqual(pipeline._obtain_original.call_count, 2)
            self.assertEqual(provider.generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
