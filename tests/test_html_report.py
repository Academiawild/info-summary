from __future__ import annotations

import base64
import shutil
import struct
import tempfile
import unittest
import zlib
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary.exporters import ResultExporter
from info_summary.html_report import REPORT_FILENAME, render_report
from info_summary.models import (
    ProcessingArtifact, ProcessingRequest, ProcessingStep, SourceType, TranscriptSegment, VisualNote,
)
from info_summary.pipeline import ProcessingPipeline
from info_summary.requests import build_processing_request, request_requirements
from info_summary.result_index import ResultIndex
from info_summary.service import ProcessingService
from info_summary.settings import AppSettings


def write_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + b"\x28\x63\x4f" * 2) * 2)) + chunk(b"IEND", b""))


class Tags(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class HtmlReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.image = self.root / "frame.png"
        write_png(self.image)
        self.request = ProcessingRequest(
            "https://vimeo.com/1", SourceType.VIDEO_URL,
            steps=(ProcessingStep.ORIGINAL, ProcessingStep.POLISH, ProcessingStep.SUMMARY),
            export_formats=("html",), include_visual=True,
        )
        self.artifact = ProcessingArtifact(
            title="图表阅读示例", source=self.request.source, method="embedded_subtitle", language="zh",
            segments=[TranscriptSegment(0, 2, "原文中的关键词。"), TranscriptSegment(5, 7, "第二段原文。")],
            original_text="原文中的关键词。\n\n第二段原文。", polished_text="整理后的关键词。",
            summary_text="## 主要结论\n\n**关键词**带来清晰的查阅体验。\n\n[[frame:1]]\n\n保留原文便于核对。",
            visual_notes=[VisualNote(5, self.image, "画面中的关键词与图表。", end=7)],
            visual_overview="图表解读依据截图。",
        )

    def export(self, request=None):
        return ResultExporter().export(self.artifact, request or self.request, self.root / "output", "演示")

    def test_one_portable_report_contains_ordered_sections_and_original_assets(self):
        original_bytes = self.image.read_bytes()
        result = self.export()
        self.assertEqual(set(result.output_files), {"report.html", "image.frame_0001.png"})
        self.assertEqual(result.output_files["report.html"].name, REPORT_FILENAME)
        source = result.output_files["report.html"].read_text(encoding="utf-8")
        offsets = [source.index(f'class="report-section" id="{name}"') for name in ("summary", "polish", "original")]
        self.assertEqual(offsets, sorted(offsets))
        self.assertNotIn('class="report-section" id="visual"', source)
        self.assertIn("图文讲解", source)
        self.assertIn("00:00 - 00:02", source)
        self.assertTrue(any(attrs.get("class") == "time-label" and "data-search" in attrs
                            for tag, attrs in Tags(source).tags if tag == "span"))
        self.assertIn('id="search-count"', source)
        self.assertIn('id="search-next"', source)
        self.assertIn('id="image-viewer"', source)
        self.assertIn("@media print", source)
        self.assertEqual(result.warnings, [])
        embedded = [attrs["src"] for tag, attrs in Tags(source).tags if tag == "img" and "src" in attrs]
        self.assertEqual(len(embedded), 1)
        self.assertEqual(base64.b64decode(embedded[0].split(",", 1)[1]), original_bytes)
        self.assertFalse(any(tag == "script" and "src" in attrs for tag, attrs in Tags(source).tags))
        self.assertFalse(any(tag == "link" for tag, _ in Tags(source).tags))
        moved = self.root / "portable.html"
        shutil.copyfile(result.output_files["report.html"], moved)
        shutil.rmtree(result.result_dir)
        self.image.unlink()
        self.assertEqual(moved.read_text(encoding="utf-8"), source)

    def test_picture_is_between_its_claim_and_explanation_in_one_article(self):
        source = self.export().output_files["report.html"].read_text()
        self.assertLess(source.index("带来清晰的查阅体验"), source.index('<figure id="frame-1"'))
        self.assertLess(source.index('<figure id="frame-1"'), source.index("保留原文便于核对。"))
        self.assertIn('href="#topic-1"', source)
        self.assertIn('id="topic-1"', source)
        self.assertNotIn("[[frame:", source)

    def test_same_time_references_are_unique_and_duplicates_do_not_repeat_pictures(self):
        other = self.root / "other.png"
        write_png(other)
        self.artifact.visual_notes.append(VisualNote(5, other, "第二张图的依据。"))
        self.artifact.summary_text = "第一个观点。\n\n[[frame:1]]\n\n第二个观点。\n\n[[frame:2]]\n\n再次引用。[[frame:1]]"
        source = self.export().output_files["report.html"].read_text()
        self.assertEqual(source.count('<figure id="frame-1"'), 1)
        self.assertEqual(source.count('<figure id="frame-2"'), 1)
        self.assertIn('href="#frame-1"', source)
        self.assertLess(source.index("第二个观点。"), source.index('<figure id="frame-2"'))
        self.assertIn("第二张图的依据。", source)

    def test_unknown_reference_keeps_prose_and_marks_partial_without_arbitrary_images(self):
        self.artifact.summary_text = "已有观点。\n\n[[frame:99]]\n\n后续解释。"
        result = self.export()
        source = result.output_files["report.html"].read_text()
        self.assertIn("已有观点。", source)
        self.assertIn("后续解释。", source)
        self.assertIn("对应画面未取得", source)
        self.assertIn("补充画面与核对说明", source)
        self.assertNotIn('id="frame-99"', source)
        self.assertTrue(result.warnings)

    def test_legacy_time_reference_places_the_correct_picture_next_to_its_paragraph(self):
        self.artifact.summary_text = "开头主旨。\n\n[画面 00:05] 这张图解释增长。\n\n下一个话题。"
        source = self.export().output_files["report.html"].read_text()
        self.assertLess(source.index("这张图解释增长。"), source.index('<figure id="frame-1"'))
        self.assertLess(source.index('<figure id="frame-1"'), source.index("下一个话题。"))
        self.assertNotIn("原有文字总结", source)

    def test_untimed_legacy_summary_is_preserved_without_guessing_picture_links(self):
        self.artifact.summary_text = "- 旧总结第一点。\n- 旧总结第二点。"
        source = self.export().output_files["report.html"].read_text()
        self.assertLess(source.index('<figure id="frame-1"'), source.index("原有文字总结"))
        self.assertIn("旧总结第一点。", source)
        self.assertIn("画面中的关键词与图表。", source)

    def test_unused_evidence_stays_searchable_in_folded_supporting_material(self):
        other = self.root / "other.png"
        write_png(other)
        self.artifact.visual_notes.append(VisualNote(12, other, "另一个主题的画面依据。"))
        source = self.export().output_files["report.html"].read_text()
        self.assertLess(source.index("补充画面与核对说明"), source.index('<figure id="frame-2"'))
        self.assertIn("另一个主题的画面依据。", source)
        self.assertIn('<p data-search>另一个主题的画面依据。</p>', source)

    def test_visual_without_summary_keeps_selected_notes_and_omits_summary(self):
        request = replace(self.request, steps=(ProcessingStep.ORIGINAL,))
        source = self.export(request).output_files["report.html"].read_text()
        self.assertIn('id="visual"', source)
        self.assertNotIn('id="summary"', source)
        self.assertIn("画面中的关键词与图表。", source)

    def test_transcripts_titles_model_content_and_sources_cannot_supply_markup(self):
        payload = '</script><img src=x onerror="alert(1)"><script>alert(2)</script>'
        self.artifact.title = payload
        self.artifact.source = "javascript:alert(1)"
        self.artifact.summary_text = payload
        self.artifact.segments = [TranscriptSegment(0, 1, payload)]
        self.artifact.visual_notes[0] = VisualNote(5, self.image, payload)
        self.artifact.warnings = [payload]
        result = self.export()
        source = result.output_files["report.html"].read_text(encoding="utf-8")
        tags = Tags(source).tags
        self.assertEqual(sum(tag == "script" for tag, _ in tags), 1)
        self.assertNotIn('<img src=x', source)
        self.assertIn("&lt;/script&gt;", source)
        self.assertFalse(any(name.startswith("on") for _, attrs in tags for name in attrs))
        self.assertFalse(any(attrs.get("href", "").startswith("javascript:") for _, attrs in tags))

    def test_original_only_report_never_creates_a_model_or_includes_unselected_text(self):
        request = replace(self.request, include_visual=False, steps=(ProcessingStep.ORIGINAL,))
        pipeline = ProcessingPipeline(self.root, checkpoint_dir=self.root / "checkpoints")
        pipeline._obtain_original = MagicMock(return_value=("标题", self.artifact.segments, "subtitle", "zh"))
        pipeline._provider_for = MagicMock(side_effect=AssertionError("model must not be called"))
        result = pipeline.run(replace(request, output_dir=str(self.root / "output")))
        source = result.output_files["report.html"].read_text(encoding="utf-8")
        self.assertIn('id="original" open', source)
        self.assertNotIn('id="summary"', source)
        self.assertNotIn('id="visual"', source)
        pipeline._provider_for.assert_not_called()

    def test_html_combines_with_existing_formats_without_changing_their_keys(self):
        result = self.export(replace(self.request, export_formats=("docx", "md", "html")))
        self.assertEqual(set(result.output_files), {
            "original.docx", "original.md", "polish.docx", "polish.md", "summary.docx", "summary.md",
            "visual.docx", "visual.md", "report.html", "image.frame_0001.png",
        })
        self.assertTrue(all(path.is_file() for path in result.output_files.values()))

    def test_report_is_reused_only_when_report_and_promised_images_are_complete(self):
        result = self.export()
        index = ResultIndex(self.root / "results.json")
        index.register(self.request, self.root, result)
        self.assertEqual(index.find(self.request, self.root).output_files, result.output_files)
        result.output_files["image.frame_0001.png"].unlink()
        self.assertIsNone(index.find(self.request, self.root))
        self.assertTrue(result.output_files["report.html"].is_file())
        second = self.export()
        index.register(self.request, self.root, second)
        second.output_files["report.html"].unlink()
        self.assertIsNone(index.find(self.request, self.root))

    def test_missing_picture_or_selected_text_is_partial_and_preserves_available_content(self):
        self.image.unlink()
        self.artifact.summary_text = ""
        result = self.export()
        source = result.output_files["report.html"].read_text(encoding="utf-8")
        self.assertTrue(result.warnings)
        self.assertIn("部分完成", source)
        self.assertIn("画面中的关键词与图表。", source)
        self.assertIn("截图未能保存", source)
        self.assertIn("原文中的关键词", source)
        index = ResultIndex(self.root / "results.json")
        index.register(self.request, self.root, result)
        self.assertIsNone(index.find(self.request, self.root))

    def test_html_failure_preserves_other_files_and_reports_partial_completion(self):
        with patch("info_summary.exporters.render_report", side_effect=OSError("reader asset missing")):
            result = self.export(replace(self.request, export_formats=("md", "html")))
        self.assertNotIn("report.html", result.output_files)
        self.assertTrue(result.output_files["original.md"].is_file())
        self.assertTrue(any("阅读报告未能生成" in warning for warning in result.warnings))

    def test_failed_report_replace_keeps_old_file_and_cleans_temporary_copy(self):
        destination = self.root / "report.html"
        destination.write_text("old report", encoding="utf-8")
        with patch("info_summary.exporters.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                ResultExporter()._write_text_atomic(destination, "new report")
        self.assertEqual(destination.read_text(encoding="utf-8"), "old report")
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_form_and_request_use_the_same_format_and_model_rules(self):
        settings = AppSettings(last_source=self.request.source)
        requirements = request_requirements(settings)
        self.assertEqual(requirements.export_formats, ("html",))
        self.assertFalse(requirements.needs_model)
        self.assertEqual(build_processing_request(settings, self.root).export_formats, ("html",))
        settings.visual_focus = "图表"
        requirements = request_requirements(settings)
        self.assertTrue(requirements.include_visual)
        self.assertTrue(requirements.needs_model)
        self.assertIn("API 密钥", requirements.missing_model_fields)

    def test_service_resolves_relative_audio_folder_against_project(self):
        folder = self.root / "audio"
        folder.mkdir()
        audio = folder / "test.wav"
        audio.touch()
        request = ProcessingRequest("audio", SourceType.AUDIO_FOLDER, export_formats=("html",))
        service = ProcessingService(self.root, pipeline=MagicMock(), result_index=MagicMock())
        with patch("pathlib.Path.cwd", return_value=Path("/")):
            expanded = service.expand_requests(request)
        self.assertEqual([item.source for item in expanded], [str(audio)])


if __name__ == "__main__":
    unittest.main()
