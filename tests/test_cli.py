from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary import cli
from info_summary.models import (
    ProcessingResult,
    ProcessingStep,
    SourceType,
    TextProviderKind,
    VideoDownloadResult,
    VideoQuality,
)
from info_summary.service import ProcessingService
from info_summary.settings import AppSettings
from info_summary.text_pipeline import JobCancelled


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.audio = self.root / "sample.wav"
        self.audio.write_bytes(b"offline test fixture")
        self.settings = AppSettings(output_dir=str(self.output))
        self.store = MagicMock()
        self.store.load.return_value = self.settings
        self.history = MagicMock()
        self.service = MagicMock()
        self.service.find_existing.return_value = None
        self.service.expand_requests.side_effect = (
            lambda request: ProcessingService.expand_requests(self.service, request)
        )
        self.service.run.side_effect = self.make_result
        self.downloader = MagicMock()
        self.downloader.find_existing.return_value = None
        self.downloader.run.side_effect = self.make_download
        for target, value in (
            ("info_summary.cli.SettingsStore", self.store),
            ("info_summary.cli.ResultHistory", self.history),
            ("info_summary.service.ProcessingService", self.service),
            ("info_summary.video.VideoDownloadService", self.downloader),
        ):
            self.enterContext(patch(target, return_value=value))
        self.enterContext(patch.dict(os.environ, {"INFO_SUMMARY_API_KEY": ""}))

    def make_result(self, request, **kwargs):
        folder = Path(request.output_dir) / Path(request.source).stem
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for step in request.steps:
            for format_name in request.export_formats:
                if format_name == "html":
                    continue
                path = folder / f"{step.value}.{format_name}"
                path.write_text("offline result", encoding="utf-8")
                files[f"{step.value}.{format_name}"] = path
        if "html" in request.export_formats:
            path = folder / "阅读报告.html"
            path.write_text("<html>offline result</html>", encoding="utf-8")
            files["report.html"] = path
        return ProcessingResult(
            "测试结果", request.source, "audio_file_transcript", "zh", folder, files,
        )

    def make_download(self, request, **kwargs):
        folder = Path(request.output_dir) / "downloaded"
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "视频.mp4"
        video.write_bytes(b"offline video fixture")
        return VideoDownloadResult("测试视频", request.url, request.quality, folder, video)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = cli.main(list(args), project_dir=self.project)
        # loads 拒绝额外输出，确保调用者可以直接读取一个完整 JSON。
        payload = json.loads(stdout.getvalue())
        for name in ("status", "results", "errors"):
            self.assertIn(name, payload)
        self.store.save.assert_not_called()
        return exit_code, payload, stderr.getvalue()

    def configure_online(self):
        self.settings.opencode_model_name = "saved-model"
        self.settings.opencode_base_url = "https://example.test/v1"
        self.settings.opencode_api_key = "saved-private-key"

    def test_original_word_defaults_do_not_follow_gui_generation_choices(self):
        self.settings.include_summary = True
        self.settings.include_polish = True
        self.settings.include_visual = True
        self.settings.visual_focus = "重点看图表"
        self.settings.export_docx = False
        self.settings.export_markdown = True
        code, payload, _ = self.invoke("process", str(self.audio))
        request = self.service.run.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(request.steps, (ProcessingStep.ORIGINAL,))
        self.assertEqual(request.export_formats, ("docx",))
        self.assertFalse(request.include_visual)
        self.assertEqual(request.visual_focus, "")
        self.assertTrue(self.settings.include_visual)
        self.assertEqual(self.settings.visual_focus, "重点看图表")
        self.assertEqual(payload["results"][0]["source"], str(self.audio))
        self.assertTrue(Path(payload["results"][0]["result_dir"]).is_absolute())

    def test_visual_requires_model_even_with_original_preset(self):
        code, payload, _ = self.invoke("process", "https://vimeo.com/1", "--visual")
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "invalid")
        self.assertIn("API 密钥", payload["errors"][0]["message"])
        self.assertIn("结合画面", payload["errors"][0]["message"])
        self.service.run.assert_not_called()

    def test_html_is_explicit_and_does_not_call_a_model_or_change_saved_formats(self):
        self.settings.export_docx = True
        self.settings.export_markdown = True
        code, payload, _ = self.invoke("process", str(self.audio), "--format", "html")
        request = self.service.run.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(request.export_formats, ("html",))
        self.assertEqual(request.steps, (ProcessingStep.ORIGINAL,))
        self.assertEqual(set(payload["results"][0]["output_files"]), {"report.html"})
        self.assertTrue(self.settings.export_docx)
        self.assertTrue(self.settings.export_markdown)

    def test_both_keeps_its_existing_meaning_when_gui_html_is_enabled(self):
        self.settings.export_html = True
        code, payload, _ = self.invoke("process", str(self.audio), "--format", "both")
        self.assertEqual(code, 0)
        self.assertEqual(self.service.run.call_args.args[0].export_formats, ("docx", "md"))
        self.assertNotIn("report.html", payload["results"][0]["output_files"])

    def test_visual_focus_enables_visual_without_changing_preset_or_preferences(self):
        self.configure_online()
        with patch.dict(os.environ, {"INFO_SUMMARY_API_KEY": "test-key"}):
            code, payload, _ = self.invoke(
                "process", "https://vimeo.com/1", "--visual-focus", "  05:20 附近的图表  ",
            )
        self.assertEqual(code, 0)
        request = self.service.run.call_args.args[0]
        self.assertTrue(request.include_visual)
        self.assertEqual(request.visual_focus, "05:20 附近的图表")
        self.assertEqual(request.steps, (ProcessingStep.ORIGINAL,))
        self.assertFalse(self.settings.include_visual)
        self.assertEqual(self.settings.visual_focus, "")

    def test_visual_skips_audio_without_losing_valid_video(self):
        self.configure_online()
        with patch.dict(os.environ, {"INFO_SUMMARY_API_KEY": "test-key"}):
            code, payload, _ = self.invoke("process", str(self.audio), "https://vimeo.com/1", "--visual")
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "partial")
        self.assertIn("只支持视频链接", payload["errors"][0]["message"])
        self.assertEqual(self.service.run.call_args.args[0].source, "https://vimeo.com/1")
        self.assertEqual(self.service.run.call_count, 1)

    def test_visual_audio_only_is_an_input_error(self):
        code, payload, _ = self.invoke("process", str(self.audio), "--visual")
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "invalid")
        self.assertIn("只支持视频链接", payload["errors"][0]["message"])
        self.service.run.assert_not_called()

    def test_explicit_model_options_and_environment_key_override_saved_settings(self):
        self.configure_online()
        with patch.dict(os.environ, {"INFO_SUMMARY_API_KEY": "environment-private-key"}):
            code, payload, _ = self.invoke(
                "process", str(self.audio), "--preset", "all", "--format", "both",
                "--provider", "opencode", "--model", "chosen-model",
                "--base-url", "https://chosen.example/v1",
            )
        call = self.service.run.call_args
        request = call.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(set(request.steps), set(ProcessingStep))
        self.assertEqual(request.export_formats, ("docx", "md"))
        self.assertEqual(request.text_model_name, "chosen-model")
        self.assertEqual(request.text_model_base_url, "https://chosen.example/v1")
        self.assertEqual(call.kwargs["api_key"], "environment-private-key")
        self.assertNotIn("private-key", json.dumps(payload))
        self.assertEqual(self.settings.opencode_model_name, "saved-model")

    def test_ollama_summary_uses_saved_local_model_without_online_key(self):
        self.settings.text_provider = "ollama"
        self.settings.ollama_model_name = "local-model"
        code, _, _ = self.invoke("process", str(self.audio), "--preset", "summary", "--format", "md")
        request = self.service.run.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(request.steps, (ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY))
        self.assertEqual(request.text_provider, TextProviderKind.OLLAMA)
        self.assertEqual(request.text_model_name, "local-model")
        self.assertEqual(request.export_formats, ("md",))

    def test_missing_model_configuration_reports_json_without_starting_processing(self):
        code, payload, _ = self.invoke("process", str(self.audio), "--preset", "summary")
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "invalid")
        self.assertTrue(payload["errors"])
        self.service.run.assert_not_called()

    def test_invalid_argument_and_invalid_source_report_machine_readable_errors(self):
        for args in (
            ("process", str(self.audio), "--format", "pdf"),
            ("process", str(self.root / "missing.wav")),
            ("download", str(self.audio)),
        ):
            with self.subTest(args=args):
                code, payload, _ = self.invoke(*args)
                self.assertEqual(code, 2)
                self.assertEqual(payload["status"], "invalid")
                self.assertTrue(payload["errors"])
        self.service.run.assert_not_called()
        self.downloader.run.assert_not_called()

    def test_runtime_failure_does_not_drop_later_valid_input(self):
        second = self.root / "second.wav"
        second.write_bytes(b"second offline fixture")
        self.service.run.side_effect = [RuntimeError("识别失败"), self.make_result_for(second)]
        code, payload, _ = self.invoke("process", str(self.audio), str(second))
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(self.service.run.call_count, 2)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["errors"][0]["source"], str(self.audio))
        self.assertIn("识别失败", payload["errors"][0]["message"])

    def make_result_for(self, source):
        from info_summary.models import ProcessingRequest

        return self.make_result(ProcessingRequest(
            str(source), SourceType.AUDIO_FILE, output_dir=str(self.output),
            steps=(ProcessingStep.ORIGINAL,), export_formats=("docx",),
        ))

    def test_warning_keeps_existing_files_and_reports_partial_completion(self):
        result = self.make_result_for(self.audio)
        result.warnings = ["部分总结未生成"]
        self.service.run.side_effect = None
        self.service.run.return_value = result
        code, payload, _ = self.invoke("process", str(self.audio))
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["results"][0]["status"], "partial")
        self.assertEqual(payload["results"][0]["warnings"], result.warnings)
        self.assertTrue(payload["results"][0]["output_files"])

    def test_complete_result_is_reused_only_in_the_selected_output_location(self):
        result = self.make_result_for(self.audio)
        self.service.find_existing.return_value = result
        code, payload, _ = self.invoke("process", str(self.audio))
        self.assertEqual(code, 0)
        self.assertEqual(payload["results"][0]["status"], "reused")
        self.service.run.assert_not_called()
        other_output = self.root / "other-output"
        code, _, _ = self.invoke("process", str(self.audio), "--output", str(other_output))
        self.assertEqual(code, 0)
        self.assertEqual(self.service.run.call_args.args[0].output_dir, str(other_output))

    def test_force_skips_result_lookup_and_restarts_processing(self):
        code, _, _ = self.invoke("process", str(self.audio), "--force")
        self.assertEqual(code, 0)
        self.service.find_existing.assert_not_called()
        self.assertTrue(self.service.run.call_args.kwargs["force_restart"])

    def test_relative_source_and_output_are_based_on_callers_current_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            code, _, _ = self.invoke("process", "sample.wav", "--output", "relative-output")
        finally:
            os.chdir(previous)
        request = self.service.run.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(request.source, str(self.audio))
        self.assertEqual(request.output_dir, str(self.root / "relative-output"))

    def test_folder_expansion_deduplicates_audio_also_supplied_individually(self):
        folder = self.root / "audio-folder"
        folder.mkdir()
        first = folder / "first.wav"
        second = folder / "second.mp3"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        (folder / "ignore.txt").write_text("not audio", encoding="utf-8")
        code, payload, _ = self.invoke("process", str(folder), str(first))
        self.assertEqual(code, 0)
        sources = [call.args[0].source for call in self.service.run.call_args_list]
        self.assertEqual(sources, [str(first), str(second)])
        self.assertEqual(len(payload["results"]), 2)

    def test_business_prints_and_logs_do_not_pollute_json_output(self):
        def noisy_run(request, **kwargs):
            print("第三方组件输出")
            return self.make_result(request)

        self.service.run.side_effect = noisy_run
        code, _, stderr = self.invoke("process", str(self.audio))
        self.assertEqual(code, 0)
        self.assertIn("第三方组件输出", stderr)

    def test_cancellation_stops_batch_and_returns_json_with_exit_130(self):
        second = self.root / "second.wav"
        second.write_bytes(b"second")
        self.service.run.side_effect = JobCancelled("任务已取消。")
        code, payload, _ = self.invoke("process", str(self.audio), str(second))
        self.assertEqual(code, 130)
        self.assertEqual(payload["status"], "cancelled")
        self.assertEqual(self.service.run.call_count, 1)

    def test_download_uses_saved_site_options_and_requested_quality(self):
        self.settings.cookies_file = str(self.root / "cookies.txt")
        self.settings.js_runtime = "node"
        self.settings.js_runtime_path = "/custom/node"
        code, payload, _ = self.invoke("download", "https://youtu.be/offline-test", "--quality", "720")
        request = self.downloader.run.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(request.quality, VideoQuality.HD_720)
        self.assertEqual(request.cookies_file, self.settings.cookies_file)
        self.assertEqual(request.js_runtime, "node")
        self.assertEqual(request.js_runtime_path, "/custom/node")
        self.assertTrue(payload["results"][0]["output_files"])

    def test_last_video_download_cancel_is_not_reported_as_failure(self):
        def cancelled_download(request, **kwargs):
            # 下载器当前使用 RuntimeError 报告取消，而不是 JobCancelled。
            kwargs["cancel_event"].set()
            raise RuntimeError("视频下载已取消。")

        self.downloader.run.side_effect = cancelled_download
        code, payload, _ = self.invoke("download", "https://youtu.be/offline-test")
        self.assertEqual(code, 130)
        self.assertEqual(payload["status"], "cancelled")


class ModuleEntryTests(unittest.TestCase):
    def test_no_arguments_keeps_desktop_entry(self):
        from info_summary import __main__ as entry

        with patch("info_summary.app.launch_desktop_app") as launch:
            entry.main([])
        launch.assert_called_once()

    def test_help_does_not_require_tkinter(self):
        project = Path(__file__).resolve().parents[1]
        script = """
import importlib.abc
import runpy
import sys

class NoTkinter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tkinter' or fullname.startswith('tkinter.'):
            raise RuntimeError('CLI help must not import tkinter')

sys.meta_path.insert(0, NoTkinter())
sys.argv = ['info_summary', '--help']
runpy.run_module('info_summary', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project, capture_output=True,
            text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("process", result.stdout)
        self.assertIn("download", result.stdout)


class OfflineIntegrationTests(unittest.TestCase):
    def test_real_checkpoint_export_reuse_and_changed_output_directory(self):
        from docx import Document

        from info_summary.checkpoints import CheckpointStore
        from info_summary.models import TranscriptSegment
        from info_summary.requests import build_processing_request

        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sandbox_project = root / "project"
            # 真实入口会在项目下创建临时目录，因此连程序副本也放进隔离目录。
            shutil.copytree(
                project / "info_summary", sandbox_project / "info_summary",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            audio = root / "offline.wav"
            audio.write_bytes(b"not real audio: must use complete checkpoint")
            output = root / "first-output"
            alternate_output = root / "second-output"
            data = root / "private-data"
            settings = AppSettings(
                last_source=str(audio), output_dir=str(output),
                export_docx=True, export_markdown=True,
            )
            request = build_processing_request(settings, sandbox_project)
            checkpoints = CheckpointStore(
                sandbox_project, root_dir=data / "InfoSummary" / "checkpoints",
            )
            self.assertTrue(checkpoints.save_original(request, (
                "离线集成结果", [TranscriptSegment(0.0, 2.5, "这是离线保存的完整原文。")],
                "audio_file_transcript", "zh",
            )))
            environment = dict(os.environ)
            environment.update({
                "XDG_DATA_HOME": str(data), "INFO_SUMMARY_API_KEY": "",
                "HF_HOME": str(root / "model-cache"), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            })
            script = """
import importlib.abc
import runpy
import sys

class NoTkinter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tkinter' or fullname.startswith('tkinter.'):
            raise RuntimeError('Offline CLI must not import tkinter')

network_attempts = []
def deny_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        network_attempts.append(event)
        raise RuntimeError('Offline integration test forbids network access')

sys.addaudithook(deny_network)
sys.meta_path.insert(0, NoTkinter())
sys.argv = ['info_summary', *sys.argv[1:]]
try:
    runpy.run_module('info_summary', run_name='__main__')
finally:
    if network_attempts:
        raise RuntimeError('CLI attempted network access: ' + str(network_attempts))
"""

            def invoke(output_directory, format_name="both"):
                result = subprocess.run(
                    [sys.executable, "-c", script, "process", str(audio),
                     "--format", format_name, "--output", str(output_directory)],
                    cwd=sandbox_project, env=environment, capture_output=True,
                    text=True, timeout=30, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["status"], "completed")
                self.assertEqual(payload["errors"], [])
                self.assertEqual(len(payload["results"]), 1)
                item = payload["results"][0]
                self.assertEqual(Path(item["result_dir"]).parent, output_directory)
                expected = {"report.html"} if format_name == "html" else {"original.docx", "original.md"}
                self.assertEqual(set(item["output_files"]), expected)
                for filename in item["output_files"].values():
                    self.assertTrue(Path(filename).is_file())
                return item

            first = invoke(output)
            self.assertEqual(first["status"], "completed")
            document = Document(first["output_files"]["original.docx"])
            self.assertIn("这是离线保存的完整原文。", "\n".join(p.text for p in document.paragraphs))
            markdown = Path(first["output_files"]["original.md"]).read_text(encoding="utf-8")
            self.assertIn("这是离线保存的完整原文。", markdown)
            self.assertIn("[00:00 - 00:02]", markdown)

            reused = invoke(output)
            self.assertEqual(reused["status"], "reused")
            self.assertEqual(reused["output_files"], first["output_files"])

            relocated = invoke(alternate_output)
            self.assertEqual(relocated["status"], "completed")
            self.assertNotEqual(relocated["result_dir"], first["result_dir"])
            html_result = invoke(root / "html-output", "html")
            html_source = Path(html_result["output_files"]["report.html"]).read_text(encoding="utf-8")
            self.assertIn("这是离线保存的完整原文。", html_source)
            self.assertIn('id="search-input"', html_source)
            self.assertEqual(invoke(root / "html-output", "html")["status"], "reused")
            self.assertTrue((data / "InfoSummary" / "results.json").is_file())
            self.assertTrue((data / "InfoSummary" / "recent_results.json").is_file())
            self.assertFalse((data / "InfoSummary" / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
