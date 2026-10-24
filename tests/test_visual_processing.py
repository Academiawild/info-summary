from __future__ import annotations

import base64
import hashlib
import json
import struct
import tempfile
import threading
import unittest
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from info_summary.cancellation import JobCancelled
from info_summary.models import DownloadOptions, ProcessingRequest, ProcessingStep, SourceType, TranscriptSegment, VisualNote
from info_summary.pipeline import ProcessingPipeline
from info_summary.providers import OllamaProvider, OpenCodeProvider, ProviderError
from info_summary.video.visual_analyzer import VisualAnalysis, VisualAnalyzer


def image_bytes(color: int) -> bytes:
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes([color, 50, 100]) * 2) * 2)) + chunk(b"IEND", b""))


class Media:
    def __init__(self):
        self.refinements = []
        self.repeated_picture = False

    def prepare(self, url, directory, options, cancel_event=None):
        directory.mkdir(parents=True, exist_ok=True)
        video = directory / "video.mp4"
        video.write_bytes(b"test video")
        return video, 60.0

    def extract_frame(self, video, timestamp, directory, high_resolution=False, cancel_event=None):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{timestamp:.3f}{'_hd' if high_resolution else ''}.png"
        color = 0 if self.repeated_picture else (int(timestamp * 3) + (1 if high_resolution else 0)) % 256
        path.write_bytes(image_bytes(color))
        return path

    def initial_frames(self, video, duration, directory, cancel_event=None):
        return [(0.0, self.extract_frame(video, 0.0, directory))]

    def available_frames(self, video, directory, cancel_event=None):
        return [(float(path.stem.split("_")[0]), path) for path in directory.glob("*.png")]

    def refine_frames(self, video, start, end, directory, kind="sequence", cancel_event=None):
        self.refinements.append((start, end, kind))
        at = (start + end) / 2
        return [(at, self.extract_frame(video, at, directory, high_resolution=kind == "detail"))]


def gap(start=5.0, end=6.0, kind="sequence"):
    return {"start": start, "end": end, "kind": kind, "reason": "关键操作步骤缺少依据", "material": True}


def review(sufficient=True, requests=None, new=True):
    return {"sufficient": sufficient, "new_information": new, "overview": "主要画面已记录。", "requests": requests or []}


class Provider:
    def __init__(self, reviews=None, targets=None):
        self.reviews = list(reviews or [review()])
        self.targets = targets or []
        self.calls = []
        self.images = []
        self.image_error = None
        self.cancel_after_image = None

    def generate(self, model, system, prompt):
        self.calls.append((system, prompt))
        if "定位值得查看" in system:
            return json.dumps({"targets": self.targets})
        if "检查视频的画面笔记" in system:
            return json.dumps(self.reviews.pop(0) if len(self.reviews) > 1 else self.reviews[0])
        return "综合总结：[画面 00:00] 图表显示了原文未提到的增长。"

    def generate_vision(self, model, system, prompt, images):
        self.images.append((model, prompt, images))
        if self.image_error:
            raise self.image_error
        if self.cancel_after_image:
            self.cancel_after_image.set()
        return "图表显示了原文未提到的增长。截图：" + images[0].stem


class VisualAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = Media()
        self.request = ProcessingRequest("https://example.test/video", SourceType.VIDEO_URL,
                                         include_visual=True, text_model_name="vision", steps=(ProcessingStep.ORIGINAL,))
        self.segments = [TranscriptSegment(0, 12, "看这张图，接下来点这里。")]

    def run_visual(self, provider, request=None, **kwargs):
        return VisualAnalyzer(self.media).run(request or self.request, provider, self.root, self.segments,
                                              options=DownloadOptions(), **kwargs)

    def test_picture_references_are_unique_even_at_the_same_display_time(self):
        result = VisualAnalysis(notes=[
            VisualNote(5.1, self.root / "ordinary.png", "普通截图。"),
            VisualNote(5.1, self.root / "hd.png", "高清截图。"),
            VisualNote(5.2, self.root / "empty.png", ""),
            VisualNote(5.3, self.root / "next.png", "另一截图。"),
        ])
        self.assertEqual(result.text(),
                         "[[frame:1]] [画面 00:05] 普通截图。\n\n"
                         "[[frame:2]] [画面 00:05] 高清截图。\n\n"
                         "[[frame:4]] [画面 00:05] 另一截图。")

    def test_refined_summary_reference_order_matches_the_exported_notes(self):
        provider = Provider([review(False, [gap(kind="detail")]), review()])
        drafts = []
        result = self.run_visual(provider, summarize=lambda text: (drafts.append(text) or text, []))
        self.assertEqual(result.summary, result.text())
        self.assertIn("[[frame:1]] [画面 00:00]", drafts[0])
        self.assertIn("[[frame:2]] [画面 00:05]", result.summary)
        self.assertEqual(len(drafts), 2)

    def test_failed_refinement_cannot_keep_references_to_a_different_note_order(self):
        provider = Provider([review(False, [gap(kind="detail")])])
        analyzer = VisualAnalyzer(self.media)
        describe = analyzer._describe

        def describe_then_fail(*args, **kwargs):
            if len(args[1].notes):
                # Simulate an error after one new earlier frame was recorded;
                # its insertion would shift all existing numbered references.
                args[1].notes.append(VisualNote(0, self.root / "earlier.png", "新增证据。"))
                raise RuntimeError("补看中断")
            return describe(*args, **kwargs)

        with patch.object(analyzer, "_describe", side_effect=describe_then_fail):
            result = analyzer.run(self.request, provider, self.root, self.segments, options=DownloadOptions(),
                                  summarize=lambda text: ("观点\n[[frame:1]]\n原先截图。", []))
        self.assertEqual(result.summary, "")
        self.assertEqual(result.notes[0].description, "新增证据。")
        self.assertIn("补看中断", "".join(result.warnings))

    def test_sufficient_summary_never_refines_even_if_model_also_proposes_frames(self):
        provider = Provider([review(True, [gap()])])
        drafts = []
        result = self.run_visual(provider, summarize=lambda text: (drafts.append(text) or "总结够用了", []))
        self.assertEqual(result.summary, "总结够用了")
        self.assertEqual(self.media.refinements, [])
        self.assertEqual(len(drafts), 1)
        self.assertFalse(result.warnings)

    def test_concrete_gap_refines_then_rebuilds_summary(self):
        provider = Provider([review(False, [gap(kind="detail")]), review()])
        drafts = []
        result = self.run_visual(provider, summarize=lambda text: (drafts.append(text) or text, []))
        self.assertEqual(self.media.refinements, [(5.0, 6.0, "detail")])
        self.assertEqual(len(drafts), 2)
        self.assertEqual(len(result.notes), 2)
        self.assertIn("00:05", result.summary)
        self.assertFalse(result.warnings)

    def test_no_fixed_refinement_round_limit(self):
        provider = Provider([review(False, [gap(at, at + 1)]) for at in (5, 10, 15, 20)] + [review()])
        result = self.run_visual(provider)
        self.assertEqual(len(self.media.refinements), 4)
        self.assertEqual(len(result.notes), 5)
        self.assertFalse(result.warnings)

    def test_no_new_information_stops_without_another_refinement(self):
        provider = Provider([review(False, [gap()]), review(False, [gap(8, 9)], new=False)])
        result = self.run_visual(provider)
        self.assertEqual(len(self.media.refinements), 1)
        self.assertIn("没有新增有效信息", "".join(result.warnings))

    def test_identical_image_at_new_time_is_not_new_evidence(self):
        self.media.repeated_picture = True
        provider = Provider([review(False, [gap()])])
        result = self.run_visual(provider)
        self.assertEqual(len(provider.images), 1)
        self.assertEqual(len(self.media.refinements), 1)
        self.assertIn("没有新增有效证据", "".join(result.warnings))

    def test_repeated_and_invalid_requests_do_not_execute_again(self):
        requests = [gap(), gap(), gap(-1, 1), gap(1, 2, "shell"), gap(True, 2), gap(float("nan"), 3), gap(10**400, 10**400)]
        provider = Provider([review(False, requests)])
        result = self.run_visual(provider)
        self.assertEqual(self.media.refinements, [(5.0, 6.0, "sequence")])
        self.assertTrue(result.warnings)

    def test_successful_image_calls_resume_without_repeating(self):
        provider = Provider([review(False, [gap()]), review()])
        first = self.run_visual(provider)
        count = len(provider.images)
        second = self.run_visual(provider)
        self.assertEqual(len(provider.images), count)
        self.assertEqual({note.image_path for note in first.notes}, {note.image_path for note in second.notes})

    def test_new_focus_retains_previously_refined_unrelated_evidence(self):
        provider = Provider([review(False, [gap()]), review()])
        first = self.run_visual(provider)
        count = len(provider.images)
        provider.targets = [{"start": 40, "end": 42, "reason": "请补充这一段"}]
        second = self.run_visual(provider, replace(self.request, visual_focus="请补充 00:40 这一段"))
        self.assertEqual(len(provider.images), count + 1)
        self.assertEqual(len(second.notes), len(first.notes) + 1)

    def test_model_change_reinterprets_images(self):
        provider = Provider()
        self.run_visual(provider)
        self.run_visual(provider, replace(self.request, text_model_name="other-model"))
        self.assertEqual([call[0] for call in provider.images], ["vision", "other-model"])

    def test_cancel_after_response_saves_completed_image_for_resume(self):
        event = threading.Event()
        provider = Provider()
        provider.cancel_after_image = event
        with self.assertRaises(JobCancelled):
            self.run_visual(provider, cancel_event=event)
        event.clear()
        provider.cancel_after_image = None
        self.run_visual(provider, cancel_event=event)
        self.assertEqual(len(provider.images), 1)

    def test_unsupported_images_preserve_screenshot_and_original_summary(self):
        provider = Provider()
        provider.image_error = ProviderError("model does not support images", retryable=False)
        result = self.run_visual(provider, summarize=lambda text: ("原文总结", []))
        self.assertEqual(len(provider.images), 1)
        self.assertTrue(result.notes[0].image_path.exists())
        self.assertTrue(result.warnings)
        self.assertEqual(result.summary, "原文总结")

    def test_malformed_structured_response_is_not_cached(self):
        provider = Provider()
        original_generate = provider.generate
        provider.generate = lambda model, system, prompt: "{}" if "检查视频" in system else original_generate(model, system, prompt)
        first = self.run_visual(provider)
        self.assertTrue(first.warnings)
        provider.generate = original_generate
        second = self.run_visual(provider)
        self.assertFalse(second.warnings)
        self.assertEqual(len(provider.images), 1)

    def test_text_targets_add_picture_with_local_question(self):
        provider = Provider(targets=[{"start": 20, "end": 22, "reason": "读清图表"}])
        result = self.run_visual(provider, replace(self.request, visual_focus="读清图表"))
        self.assertEqual(len(result.notes), 2)
        self.assertTrue(any("读清图表" in call[1] for call in provider.images))

    def test_static_slide_dedup_preserves_the_users_target_question(self):
        self.media.repeated_picture = True
        provider = Provider(targets=[{"start": 20, "end": 22, "reason": "只读清纵轴单位"}])
        result = self.run_visual(provider, replace(self.request, visual_focus="只读清纵轴单位"))
        self.assertEqual(len(provider.images), 1)
        self.assertIn("只读清纵轴单位", provider.images[0][1])
        self.assertEqual(result.notes[0].timestamp, 21.0)

    def test_partial_frame_extraction_keeps_already_saved_pictures(self):
        original_initial = self.media.initial_frames
        def incomplete(*args, **kwargs):
            original_initial(*args, **kwargs)
            raise RuntimeError("failed next frame")
        self.media.initial_frames = incomplete
        result = self.run_visual(Provider())
        self.assertEqual(len(result.notes), 1)
        self.assertIn("全程画面提取未完成", "".join(result.warnings))

    def test_failed_image_retry_does_not_repeat_successful_image(self):
        provider = Provider(targets=[{"start": 20, "end": 22, "reason": "读清图表"}])
        original = provider.generate_vision
        def fail_once(model, system, prompt, images):
            if json.loads(prompt)["time"] == 0 and len(provider.images) == 1:
                provider.images.append((model, prompt, images))
                raise ProviderError("temporary failure", retryable=False)
            return original(model, system, prompt, images)
        provider.generate_vision = fail_once
        first = self.run_visual(provider)
        self.assertTrue(first.warnings)
        second = self.run_visual(provider)
        self.assertFalse(second.warnings)
        self.assertEqual(len(provider.images), 3)
        self.assertEqual(sum(json.loads(call[1])["time"] == 21 for call in provider.images), 1)

    def test_failed_checkpoint_write_keeps_generated_notes(self):
        with patch("info_summary.storage.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
            result = self.run_visual(Provider())
        self.assertEqual(len(result.notes), 1)
        self.assertIn("图表显示", result.notes[0].description)
        self.assertIn("进度未能保存", "".join(result.warnings))

    def test_long_transcript_and_note_do_not_create_unbounded_requests(self):
        self.segments = [TranscriptSegment(0, 50, "长原文。" * 2000)]
        provider = Provider()
        original = provider.generate_vision
        def long_description(*args):
            original(*args)
            return "长画面说明。" * 1800
        provider.generate_vision = long_description
        result = self.run_visual(provider, summarize=lambda text: ("长总结。" * 2000, []))
        self.assertFalse(result.warnings)
        self.assertEqual(result.notes[0].description, "长画面说明。" * 1800)
        self.assertTrue(all(len(prompt) <= self.request.text_model_chunk_chars for _, prompt in provider.calls))
        self.assertTrue(all(len(prompt) <= self.request.text_model_chunk_chars for _, prompt, _ in provider.images))

    def test_new_focus_without_narrowed_time_still_reaches_the_image_model(self):
        provider = Provider()
        self.run_visual(provider, replace(self.request, visual_focus="读清纵轴单位"))
        self.assertIn("读清纵轴单位", provider.images[0][1])


class VisualProviderTests(unittest.TestCase):
    def test_online_and_ollama_send_real_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(image_bytes(1))
            for provider, response in (
                (OpenCodeProvider("test-key", "https://example.test/v1"), {"choices": [{"message": {"content": "画面"}}]}),
                (OllamaProvider(), {"message": {"content": "画面"}}),
            ):
                with self.subTest(provider=type(provider).__name__), patch("info_summary.providers._read_json", return_value=response) as read:
                    self.assertEqual(provider.generate_vision("vision", "系统", "说明", [image]), "画面")
                    data = json.loads(read.call_args.args[0].data)
                    message = data["messages"][1]
                    if isinstance(provider, OllamaProvider):
                        encoded = message["images"][0]
                    else:
                        encoded = message["content"][1]["image_url"]["url"].split(",", 1)[1]
                    self.assertEqual(base64.b64decode(encoded), image.read_bytes())


class VisualPipelineTests(unittest.TestCase):
    def test_inline_visual_summary_keeps_legacy_cache_and_adds_no_display_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints", visual_media=Media())
            request = ProcessingRequest(
                "https://example.test/video", SourceType.VIDEO_URL, output_dir=str(root / "out"),
                text_model_name="vision", include_visual=True,
                steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY), export_formats=("md",),
            )
            text_dir = pipeline._text_checkpoint_dir(request, root / "work", True)
            legacy_file = text_dir / "visual" / "summary_final.json"
            legacy_file.parent.mkdir(parents=True)
            legacy_file.write_text('{"legacy":true}', encoding="utf-8")
            provider = Provider()
            with patch.object(pipeline, "_obtain_original", return_value=(
                "示例", [TranscriptSegment(0, 2, "这是讲述原话。")], "subtitle", "zh",
            )), patch.object(pipeline, "_provider_for", return_value=provider), patch.object(
                pipeline, "_summarize", wraps=pipeline._summarize,
            ) as summarize:
                pipeline.run(request)
            self.assertEqual(legacy_file.read_text(encoding="utf-8"), '{"legacy":true}')
            self.assertTrue((legacy_file.parent / "inline_v1" / "summary_final.json").is_file())
            self.assertEqual(summarize.call_count, 1)
            self.assertTrue(summarize.call_args.kwargs["inline_visual"])
            self.assertIn("[[frame:1]]", summarize.call_args.args[2])
            summary_calls = [(system, prompt) for system, prompt in provider.calls
                             if "定位值得查看" not in system and "检查视频的画面笔记" not in system]
            self.assertEqual(len(summary_calls), 2)
            self.assertTrue(all("原样保留" in system for system, _prompt in summary_calls))

    def test_visual_summary_export_and_plain_original_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints", visual_media=Media())
            request = ProcessingRequest("https://example.test/video", SourceType.VIDEO_URL, output_dir=str(root / "out"),
                                        text_model_name="vision", include_visual=True,
                                        steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY), export_formats=("md", "docx"))
            provider = Provider()
            with patch.object(pipeline, "_obtain_original", return_value=("示例", [TranscriptSegment(0, 2, "这是讲述原话。")], "subtitle", "zh")), patch.object(pipeline, "_provider_for", return_value=provider):
                result = pipeline.run(request)
            self.assertFalse(result.warnings)
            original = result.output_files["original.md"].read_text()
            self.assertNotIn("图表显示", original)
            self.assertIn("图表显示", result.output_files["visual.md"].read_text())
            self.assertIn("画面", result.output_files["summary.md"].read_text())
            self.assertTrue(any(name.startswith("image.") for name in result.output_files))

    def test_no_transcript_still_processes_visuals_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints", visual_media=Media())
            request = ProcessingRequest("https://example.test/video", SourceType.VIDEO_URL, output_dir=str(root / "out"),
                                        include_visual=True, text_model_name="vision", steps=(ProcessingStep.ORIGINAL,), export_formats=("md",))
            with patch.object(pipeline, "_obtain_original", side_effect=RuntimeError("没有音频")), patch.object(pipeline, "_provider_for", return_value=Provider()):
                result = pipeline.run(request)
            self.assertTrue(result.warnings)
            self.assertIn("没有可用", result.output_files["original.md"].read_text())
            self.assertIn("图表显示", result.output_files["visual.md"].read_text())

    def test_disabled_option_preserves_old_result_key(self):
        request = ProcessingRequest("https://example.test/video", SourceType.VIDEO_URL)
        data = asdict(request)
        for key in ("source", "output_dir", "include_visual", "visual_focus"):
            data.pop(key)
        data["source_fingerprint"] = request.source_fingerprint(Path.cwd())
        expected = hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(request.result_key(Path.cwd()), expected)


if __name__ == "__main__":
    unittest.main()
