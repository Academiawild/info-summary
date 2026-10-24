from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary.models import (
    OutputLanguage,
    ProcessingRequest,
    ProcessingResult,
    ProcessingStep,
    SourceType,
    TranscriptSegment,
)
from info_summary.pipeline import ProcessingPipeline
from info_summary.providers import ProviderError
from info_summary.result_index import ResultIndex
from info_summary.service import ProcessingService
from info_summary.text_pipeline import JobCancelled, ResumableTextProcessor


class ResultReliabilityTests(unittest.TestCase):
    def make_result(self, root: Path, *, summary: bool = True):
        audio = root / "audio.wav"
        audio.write_bytes(b"audio")
        request = ProcessingRequest(
            str(audio), SourceType.AUDIO_FILE,
            steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY),
            export_formats=("md",),
        )
        result_dir = root / "result"
        result_dir.mkdir()
        files = {"original.md": result_dir / "original.md"}
        if summary:
            files["summary.md"] = result_dir / "summary.md"
        for path in files.values():
            path.write_text("已保存内容", encoding="utf-8")
        result = ProcessingResult("title", str(audio), "audio", "zh", result_dir, files)
        return request, result, ResultIndex(root / "results.json")

    def test_missing_requested_output_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, result, index = self.make_result(root, summary=False)
            index.register(request, root, result)
            self.assertIsNone(index.find(request, root))

    def test_deleting_one_output_or_replacing_it_with_a_folder_requires_retry(self) -> None:
        for replace_with_folder in (False, True):
            with self.subTest(folder=replace_with_folder), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                request, result, index = self.make_result(root)
                index.register(request, root, result)
                result.output_files["summary.md"].unlink()
                if replace_with_folder:
                    result.output_files["summary.md"].mkdir()
                self.assertIsNone(index.find(request, root))

    def test_warning_results_are_not_registered_or_reused_from_old_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, result, index = self.make_result(root)
            result.warnings.append("部分摘要生成失败")
            index.register(request, root, result)
            self.assertFalse(index.path.exists())
            index.path.write_text(
                json.dumps({request.result_key(root): result.to_index_data()}), encoding="utf-8"
            )
            self.assertIsNone(index.find(request, root))

    def test_failed_cache_cleanup_does_not_block_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, result, index = self.make_result(root, summary=False)
            index.register(request, root, result)
            with patch("info_summary.result_index.update_json", side_effect=OSError("read only")):
                self.assertIsNone(index.find(request, root))

    def test_index_write_failure_preserves_saved_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, result, _ = self.make_result(root)
            pipeline = MagicMock()
            pipeline.run.return_value = result
            index = MagicMock()
            index.register.side_effect = OSError("read only")
            service = ProcessingService(root, result_index=index, pipeline=pipeline)
            returned = service.run(request)
            self.assertIs(returned, result)
            self.assertTrue(all(path.is_file() for path in returned.output_files.values()))
            self.assertIn("结果文件已保存", returned.warnings[0])

    def test_service_does_not_register_partial_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request, result, _ = self.make_result(root)
            result.warnings.append("模型暂时不可用")
            pipeline = MagicMock()
            pipeline.run.return_value = result
            index = MagicMock()
            returned = ProcessingService(root, result_index=index, pipeline=pipeline).run(request)
            self.assertIs(returned, result)
            index.register.assert_not_called()


class CancellationBoundaryTests(unittest.TestCase):
    def test_cancel_during_each_last_stage_never_exports(self) -> None:
        for stage in ("provider", "polish", "summary", "export"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                event = threading.Event()
                pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints")
                pipeline._obtain_original = MagicMock(
                    return_value=("title", [TranscriptSegment(0, 1, "原文")], "audio", "zh")
                )
                provider = MagicMock()

                def create_provider(*_args):
                    if stage == "provider":
                        event.set()
                    return provider

                def finish(text_stage):
                    if stage == text_stage:
                        event.set()
                    return "文字内容", []

                pipeline._provider_for = MagicMock(side_effect=create_provider)
                pipeline._polish = MagicMock(side_effect=lambda *args: finish("polish"))
                pipeline._summarize = MagicMock(side_effect=lambda *args: finish("summary"))
                pipeline.exporter.export = MagicMock()

                def progress(update):
                    if stage == "export" and update.stage == "export":
                        event.set()

                with self.assertRaises(JobCancelled):
                    pipeline.run(
                        ProcessingRequest("https://vimeo.com/1", SourceType.VIDEO_URL),
                        cancel_event=event, progress=progress,
                    )
                pipeline.exporter.export.assert_not_called()
                self.assertEqual(list((root / "temp" / "jobs").iterdir()), [])

    def test_cancel_during_partial_summary_does_not_start_final_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            event = threading.Event()
            provider = MagicMock()

            def generate(*args):
                event.set()
                return "已生成的分段摘要"

            provider.generate.side_effect = generate
            with self.assertRaises(JobCancelled):
                ResumableTextProcessor().summarize(
                    ["原始内容"], provider, "fake", OutputLanguage.FOLLOW_SOURCE,
                    100, Path(temp), None, event.is_set,
                )
            self.assertEqual(provider.generate.call_count, 1)

    def test_provider_cancellation_is_not_treated_as_a_text_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            provider = MagicMock()
            provider.generate.side_effect = JobCancelled("任务已取消。")
            with self.assertRaises(JobCancelled):
                ResumableTextProcessor().summarize(
                    ["原始内容"], provider, "fake", OutputLanguage.FOLLOW_SOURCE,
                    100, Path(temp), None, None,
                )
            self.assertEqual(provider.generate.call_count, 1)


class LongTextReliabilityTests(unittest.TestCase):
    def test_long_sentences_stay_within_limit_without_losing_text_or_spaces(self):
        processor = ResumableTextProcessor()
        for text in (
            "甲" * 1000 + "。短句。",
            "First sentence. " + "a long sentence " * 90 + ".\n\nLast sentence.",
            "没有标点" * 250,
        ):
            with self.subTest(text=text[:20]):
                groups = processor.chunk_strings([text], 100)
                self.assertTrue(all(len("\n\n".join(group)) <= 100 for group in groups))
                self.assertEqual("".join(part for group in groups for part in group), text)

    def test_long_transcript_segments_are_split_without_omissions(self):
        text = "甲" * 1000 + "。短句。"
        chunks = ResumableTextProcessor().chunk_segments([TranscriptSegment(0, 60, text)], 100)
        self.assertTrue(all(len(chunk.fallback_text) <= 100 for chunk in chunks))
        self.assertEqual("".join(chunk.fallback_text for chunk in chunks), text)

    def test_invalid_chunk_limits_fail_before_processing(self):
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                ResumableTextProcessor().chunk_strings(["原文"], limit)


class SummaryConvergenceTests(unittest.TestCase):
    def test_unchanged_or_failed_merges_preserve_summaries_and_stop(self) -> None:
        for fail_merge in (False, True):
            with self.subTest(fail=fail_merge), tempfile.TemporaryDirectory() as temp:
                provider = MagicMock()

                def generate(model, prompt, text):
                    if "阶段摘要" in prompt and fail_merge:
                        raise ProviderError("模型不可用", retryable=False)
                    return text

                provider.generate.side_effect = generate
                summary, warnings = ResumableTextProcessor().summarize(
                    ["甲" * 80, "乙" * 80], provider, "fake", OutputLanguage.FOLLOW_SOURCE,
                    100, Path(temp), None, None,
                )
                self.assertIn("甲" * 80, summary)
                self.assertIn("乙" * 80, summary)
                self.assertTrue(any("未能缩短" in warning for warning in warnings))
                self.assertEqual(provider.generate.call_count, 4)

    def test_slowly_shrinking_merges_stop_at_bounded_round_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            provider = MagicMock()

            def generate(model, prompt, text):
                return text[:-1] if "阶段摘要" in prompt else text

            provider.generate.side_effect = generate
            with patch("info_summary.text_pipeline.MAX_SUMMARY_MERGE_ROUNDS", 2):
                summary, warnings = ResumableTextProcessor().summarize(
                    ["甲" * 80, "乙" * 80], provider, "fake", OutputLanguage.FOLLOW_SOURCE,
                    100, Path(temp), None, None,
                )
            self.assertIn("甲" * 78, summary)
            self.assertIn("乙" * 78, summary)
            self.assertTrue(any("达到处理上限" in warning for warning in warnings))
            self.assertEqual(provider.generate.call_count, 6)


class InlineVisualSummaryTests(unittest.TestCase):
    def test_each_visual_summary_stage_preserves_exact_evidence_references(self):
        processor = ResumableTextProcessor()
        for method in (processor.partial_summary_prompt, processor.merge_summary_prompt, processor.final_summary_prompt):
            with self.subTest(stage=method.__name__):
                prompt = method(OutputLanguage.FOLLOW_SOURCE, inline_visual=True)
                self.assertIn("[[frame:1]]", prompt)
                self.assertIn("原样保留", prompt)
                self.assertIn("只能引用本次输入里存在的标记", prompt)
                self.assertIn("自然段", prompt)
        self.assertIn("不要分成文字总结和画面笔记两块", processor.final_summary_prompt(
            OutputLanguage.FOLLOW_SOURCE, inline_visual=True,
        ))

    def test_plain_summary_prompt_strings_remain_compatible_with_old_cache_identity(self):
        processor = ResumableTextProcessor()
        language_rule = "跟随原文的主要语言输出，不主动翻译。"
        expected = (
            "请总结下面这段文字，提炼 3 到 5 条关键要点，覆盖其中的核心信息和结论。"
            "只依据原文，不补充原文没有的内容，不重复啰嗦。"
            "每条要点单独一行、以“- ”开头，一句话讲清一个点。"
            "保留英文术语、产品名、代码和缩写。" + language_rule,
            "请把下面这些分段摘要合并成更紧凑的阶段摘要：删除重复，保留全部关键事实与结论，不新增内容。"
            "保持要点形式，每条单独一行、以“- ”开头。" + language_rule,
            "请基于下面这些分段摘要，写出整篇内容的总结。格式："
            "先用一句话概括全文主旨；然后另起一段，按重要性列出 4 到 8 条要点，"
            "每条单独一行、以“- ”开头。"
            "要求：准确、清晰、不编造、不重复，保留英文术语、产品名、代码和缩写。" + language_rule,
        )
        actual = tuple(method(OutputLanguage.FOLLOW_SOURCE) for method in (
            processor.partial_summary_prompt, processor.merge_summary_prompt, processor.final_summary_prompt,
        ))
        self.assertEqual(actual, expected)

    def test_inline_generation_uses_existing_stages_and_reuses_their_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = MagicMock()
            # Two ordinary partial calls, two existing merges, then one final:
            # adding image placement never adds a separate generation step.
            provider.generate.side_effect = [
                "甲" * 35 + " [[frame:1]]",
                "乙" * 35 + " [[frame:2]]",
                "观点一\n[[frame:1]]\n第一幅解释。",
                "观点二\n[[frame:2]]\n第二幅解释。",
                "## 主题\n结合说明。\n[[frame:1]]\n画面支持该观点。\n[[frame:2]]\n下一幅补充依据。",
            ]
            processor = ResumableTextProcessor()
            args = (["讲述与 [[frame:1]] 画面一", "讲述与 [[frame:2]] 画面二"], provider,
                    "fake", OutputLanguage.FOLLOW_SOURCE, 80, Path(directory), None, None)
            summary, warnings = processor.summarize(*args, inline_visual=True)
            self.assertFalse(warnings)
            self.assertIn("[[frame:1]]", summary)
            self.assertIn("[[frame:2]]", summary)
            self.assertEqual(provider.generate.call_count, 5)
            prompts = [call.args[1] for call in provider.generate.call_args_list]
            self.assertIn("分段摘要", prompts[0])
            self.assertIn("阶段摘要", prompts[2])
            self.assertIn("一篇连贯的图文解读", prompts[4])
            repeated, warnings = processor.summarize(*args, inline_visual=True)
            self.assertEqual(repeated, summary)
            self.assertFalse(warnings)
            self.assertEqual(provider.generate.call_count, 5)


if __name__ == "__main__":
    unittest.main()
