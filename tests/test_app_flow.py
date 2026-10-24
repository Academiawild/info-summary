from __future__ import annotations

import queue
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from info_summary.app import InfoSummaryApp
from info_summary.models import (ProcessingRequest, ProcessingResult, ProcessingStep, SourceType,
                                 TextProviderKind, VideoDownloadRequest, VideoDownloadResult, VideoQuality)
from info_summary.models import TranscriptSegment
from info_summary.history import ResultHistory
from info_summary.pipeline import ProcessingPipeline
from info_summary.providers import ProviderError
from info_summary.result_index import ResultIndex
from info_summary.service import ProcessingService
from info_summary.settings import AppSettings
from info_summary.task_queue import TaskQueue, UnfinishedTasks
from info_summary.text_pipeline import JobCancelled


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_app() -> InfoSummaryApp:
    app = InfoSummaryApp.__new__(InfoSummaryApp)
    app.root = MagicMock()
    app.project_dir = Path.cwd()
    app.settings = AppSettings()
    app.settings_store = MagicMock()
    app.history = MagicMock()
    app.view = MagicMock()
    app.view.progress = Value(0)
    app.view.status = Value("")
    app.view.log_visible = False
    app.events = queue.Queue()
    app.cancel_event = threading.Event()
    app.busy = True
    app.close_when_idle = False
    app.result_dirs = []
    app.reused_result_dirs = []
    app.task_results = {}
    app.retry_pending = []
    app.task_queue = None
    app.unfinished = MagicMock()
    app.unfinished.pending.return_value = []
    app.input_issues = []
    app.force_restart = False
    return app


def result(name="done", warnings=None) -> ProcessingResult:
    return ProcessingResult(name, "audio.wav", "audio_file_transcript", "zh", Path(name), {}, warnings or [])


def request(name="audio.wav") -> ProcessingRequest:
    return ProcessingRequest(name, SourceType.AUDIO_FILE, steps=(ProcessingStep.ORIGINAL,), export_formats=("md",))


class BatchFlowTests(unittest.TestCase):
    def test_failed_item_does_not_interrupt_the_rest_of_batch(self):
        app = make_app()
        service = MagicMock()
        service.run.side_effect = [RuntimeError("bad audio"), result()]
        with patch("info_summary.service.ProcessingService", return_value=service):
            app._worker_loop((("process", request("first.wav")), ("process", request("second.wav"))), "")
        with patch("info_summary.app.messagebox.showerror") as dialog:
            app._poll_events()
        self.assertEqual(service.run.call_count, 2)
        self.assertIn("成功 1 项", app.view.status.get())
        self.assertIn("失败 1 项", app.view.status.get())
        self.assertEqual(app.result_dirs, [Path("done")])
        self.assertFalse(app.busy)
        app.view.toggle_log.assert_called_once()
        dialog.assert_not_called()

    def test_cancel_is_not_failure_and_does_not_start_next_item(self):
        app = make_app()
        service = MagicMock()
        service.run.side_effect = JobCancelled("cancelled")
        with patch("info_summary.service.ProcessingService", return_value=service):
            app._worker_loop((("process", request()), ("process", request("next.wav"))), "")
        app._poll_events()
        self.assertEqual(service.run.call_count, 1)
        self.assertIn("已停止", app.view.status.get())
        self.assertIn("未完成 2 项", app.view.status.get())
        self.assertNotIn("失败", app.view.status.get())
        self.assertNotEqual(app.view.progress.get(), 100)

    def test_partial_result_keeps_folder_and_explains_missing_work(self):
        app = make_app()
        service = MagicMock()
        service.run.return_value = result(warnings=["总结未生成"])
        with patch("info_summary.service.ProcessingService", return_value=service):
            app._worker_loop((("process", request()),), "")
        app._poll_events()
        self.assertIn("部分完成 1 项", app.view.status.get())
        app.view.append_log.assert_any_call("需要留意：总结未生成")
        self.assertEqual(app.result_dirs, [Path("done")])

    def test_progress_is_for_whole_batch_and_cancel_status_stays_visible(self):
        app = make_app()
        app.events.put(("queue_progress", (1, 1, 4, 0.5, "正在转写")))
        app._poll_events()
        self.assertEqual(app.view.progress.get(), 37.5)
        self.assertIn("已完成 1/4", app.view.status.get())
        app.view.update_task.assert_called_with(1, "处理中", "正在转写")
        app._cancel()
        status = app.view.status.get()
        app.view.update_task.reset_mock()
        app.events.put(("queue_progress", (1, 1, 4, 0.8, "正在润色")))
        app._poll_events()
        self.assertEqual(app.view.progress.get(), 45)
        self.assertEqual(app.view.status.get(), status)
        app.view.update_task.assert_not_called()

    def test_no_second_worker_can_start_while_running(self):
        app = make_app()
        with patch.object(app, "_settings_from_view") as collect:
            app._start()
        collect.assert_not_called()

    def test_secrets_are_redacted_from_failures_and_warnings(self):
        app = make_app()
        app.settings.opencode_api_key = "private-key-1234"
        app.events.put(("task_error", "bad private-key-1234"))
        app.events.put(("result", result(warnings=["private-key-1234 unavailable"])))
        app._poll_events()
        self.assertTrue(app.view.append_log.called)
        self.assertNotIn("private-key-1234", str(app.view.append_log.call_args_list))


class DuplicateFlowTests(unittest.TestCase):
    def test_one_reuse_decision_for_batch_only_queues_new_items(self):
        app = make_app()
        items = [request("one.wav"), request("two.wav"), request("three.wav")]
        service = MagicMock()
        service.expand_requests.return_value = items
        service.find_existing.side_effect = [result("one"), result("two"), None]
        app.view.choose_existing.return_value = "existing"
        with patch.object(app, "_build_requests", return_value=[items[0]]), patch("info_summary.service.ProcessingService", return_value=service):
            accepted = app._prepare_processing(AppSettings())
        self.assertEqual(accepted, [("process", items[2])])
        app.view.choose_existing.assert_called_once()
        self.assertEqual(app.reused_result_dirs, [Path("one"), Path("two")])

    def test_cancel_duplicate_dialog_queues_nothing_even_if_new_item_came_first(self):
        app = make_app()
        items = [request("new.wav"), request("duplicate.wav")]
        service = MagicMock()
        service.expand_requests.return_value = items
        service.find_existing.side_effect = [None, result()]
        app.view.choose_existing.return_value = "cancel"
        with patch.object(app, "_build_requests", return_value=[items[0]]), patch("info_summary.service.ProcessingService", return_value=service):
            self.assertEqual(app._prepare_processing(AppSettings()), [])

    def test_start_clears_old_state_and_keeps_reused_results(self):
        app = make_app()
        app.busy = False
        app.result_dirs = [Path("old")]
        app.view.progress.set(100)
        def prepare(_settings):
            app.reused_result_dirs = [Path("reused")]
            return [("process", request())]
        with patch.object(app, "_settings_from_view", return_value=AppSettings()), patch.object(app, "_validate_output"), patch.object(app, "_prepare_processing", side_effect=prepare), patch("info_summary.app.threading.Thread") as thread:
            app._start()
        self.assertEqual(app.result_dirs, [Path("reused")])
        self.assertEqual(app.view.progress.get(), 0)
        app.view.clear_log.assert_called_once()
        thread.return_value.start.assert_called_once()


class SettingsFlowTests(unittest.TestCase):
    def populate_view(self, app):
        mapping = {
            "source": "last_source", "provider": "text_provider", "ollama_model": "ollama_model_name",
            "opencode_model": "opencode_model_name", "opencode_url": "opencode_base_url", "ollama_url": "ollama_base_url", "opencode_key": "opencode_api_key",
            "chunk_chars": "text_model_chunk_chars", "cookies_browser": "cookies_from_browser",
        }
        for field in AppSettings.__dataclass_fields__:
            view_field = next((k for k, v in mapping.items() if v == field), field)
            setattr(app.view, view_field, Value(getattr(app.settings, field)))

    def test_hidden_ai_setting_cannot_block_original_only_task(self):
        app = make_app()
        self.populate_view(app)
        app.view.include_polish.set(False)
        app.view.include_summary.set(False)
        app.view.chunk_chars.set("invalid")
        settings = app._settings_from_view()
        self.assertEqual(settings.text_model_chunk_chars, app.settings.text_model_chunk_chars)

    def test_visual_focus_is_loaded_and_collected_without_changing_preset(self):
        app = make_app()
        app.settings = AppSettings(include_visual=True, visual_focus="看 05:20 的图表")
        self.populate_view(app)
        app._load_settings()
        self.assertTrue(app.view.include_visual.get())
        self.assertEqual(app.view.visual_focus.get(), "看 05:20 的图表")
        settings = app._settings_from_view()
        self.assertTrue(settings.include_visual)
        self.assertEqual(settings.visual_focus, "看 05:20 的图表")
        self.assertFalse(settings.include_summary)

    def test_visual_original_validates_model_and_audio_is_rejected_clearly(self):
        app = make_app()
        settings = AppSettings(last_source="https://vimeo.com/1", include_visual=True)
        with self.assertRaisesRegex(ValueError, "API 密钥"):
            app._build_request(settings)
        settings.text_provider = "ollama"
        settings.ollama_model_name = "vision-model"
        request = app._build_request(settings)
        self.assertTrue(request.include_visual)
        self.assertEqual(request.steps, (ProcessingStep.ORIGINAL,))
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "sample.wav"
            audio.touch()
            settings.last_source = str(audio)
            with self.assertRaisesRegex(ValueError, "只支持视频链接"):
                app._build_request(settings)

    def test_visual_focus_implicitly_enables_visual_and_model_validation(self):
        app = make_app()
        self.populate_view(app)
        app.view.visual_focus.set("看图表")
        settings = app._settings_from_view()
        self.assertTrue(settings.include_visual)
        settings.last_source = "https://vimeo.com/1"
        with self.assertRaisesRegex(ValueError, "支持图片"):
            app._build_request(settings)

    def test_download_preserves_saved_transcription_settings(self):
        app = make_app()
        app.settings.batch_size = 3
        app.settings.text_model_chunk_chars = 2000
        self.populate_view(app)
        app.view.task_mode.set("download")
        app.view.batch_size.set("invalid")
        app.view.chunk_chars.set("invalid")
        settings = app._settings_from_view()
        self.assertEqual(settings.batch_size, 3)
        self.assertEqual(settings.text_model_chunk_chars, 2000)

    def test_missing_opencode_key_is_caught_before_expensive_processing(self):
        app = make_app()
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "audio.wav"
            audio.touch()
            settings = AppSettings(last_source=str(audio), text_provider="opencode", opencode_model_name="test",
                                   opencode_base_url="https://service.example/v1", include_summary=True)
            with self.assertRaisesRegex(ValueError, "API 密钥"):
                app._build_request(settings)
            settings.include_summary = settings.include_polish = False
            self.assertEqual(app._build_request(settings).steps, (ProcessingStep.ORIGINAL,))

    def test_default_extracts_original_without_credentials_and_summary_has_clear_fix(self):
        app = make_app()
        settings = AppSettings(last_source="https://vimeo.com/11")
        built = app._build_request(settings)
        self.assertEqual(built.steps, (ProcessingStep.ORIGINAL,))
        self.assertEqual(built.export_formats, ("html",))
        self.assertEqual(built.text_provider, TextProviderKind.OPENCODE)
        settings.include_summary = True
        with self.assertRaisesRegex(ValueError, "接口地址、模型名称、API 密钥"):
            app._build_request(settings)

    def test_startup_simplifies_unconfigured_presets_and_preserves_configured_choices(self):
        for provider, configured in (("opencode", False), ("opencode", True), ("ollama", False), ("ollama", True)):
            with self.subTest(provider=provider, configured=configured):
                app = make_app()
                app.settings = AppSettings(
                    text_provider=provider, include_polish=True, include_summary=True,
                    output_dir="/tmp/saved-results", export_markdown=True,
                    opencode_base_url="https://service.example/v1", opencode_model_name="online-model",
                    opencode_api_key="test-key" if configured else "",
                    ollama_model_name="local-model" if configured else "",
                )
                self.populate_view(app)
                app._load_settings()
                self.assertEqual(app.view.include_polish.get(), configured)
                self.assertEqual(app.view.include_summary.get(), configured)
                self.assertEqual(app.view.output_dir.get(), "/tmp/saved-results")
                self.assertTrue(app.view.export_markdown.get())
                self.assertEqual(app.view.opencode_model.get(), "online-model")

    def test_model_check_uses_actual_generation_and_surfaces_generation_failures(self):
        for error in (None, ProviderError("没有生成权限", retryable=False)):
            with self.subTest(error=error):
                app = make_app()
                self.populate_view(app)
                app.view.opencode_model.set("test-model")
                provider = MagicMock()
                provider.generate.return_value = "连接成功。"
                provider.generate.side_effect = error
                with patch.object(app, "_current_provider", return_value=provider), patch(
                    "info_summary.app.threading.Thread",
                ) as thread:
                    app._check_provider()
                    thread.return_value.start.assert_called_once()
                    thread.call_args.kwargs["target"]()
                provider.generate.assert_called_once_with("test-model", "请用中文简短回答。", "请回复：连接成功。")
                provider.health_check.assert_not_called()
                provider.list_models.assert_not_called()
                self.assertEqual(provider.timeout_seconds, 30)
                event, message = app.events.get_nowait()
                self.assertEqual(event, "provider_error" if error else "provider")
                self.assertIn("没有生成权限" if error else "可以生成文字", message)

    def test_model_check_requests_a_model_before_sending_a_request(self):
        app = make_app()
        self.populate_view(app)
        with patch.object(app, "_current_provider") as current, patch(
            "info_summary.app.messagebox.showinfo",
        ) as dialog, patch("info_summary.app.threading.Thread") as thread:
            app._check_provider()
        current.return_value.generate.assert_not_called()
        thread.assert_not_called()
        self.assertIn("模型名称", dialog.call_args.args[1])

    def test_switching_provider_uses_its_own_address_model_and_credentials(self):
        app = make_app()
        app.settings = AppSettings(
            last_source="https://vimeo.com/11", opencode_base_url="https://service.example/v1",
            opencode_model_name="online-model", opencode_api_key="online-only-key",
            ollama_base_url="http://127.0.0.1:11435", ollama_model_name="local-model",
        )
        self.populate_view(app)
        online_request = app._build_request(app._settings_from_view())
        self.assertEqual(online_request.text_model_base_url, "https://service.example/v1")
        self.assertEqual(online_request.text_model_name, "online-model")
        online = app._current_provider()
        self.assertEqual(online.api_key, "online-only-key")
        self.assertEqual(online.base_url, online_request.text_model_base_url)
        app.view.provider.set("ollama")
        local_settings = app._settings_from_view()
        local_request = app._build_request(local_settings)
        self.assertEqual(local_request.text_model_base_url, "http://127.0.0.1:11435")
        self.assertEqual(local_request.text_model_name, "local-model")
        self.assertEqual(local_request.text_provider, TextProviderKind.OLLAMA)
        self.assertEqual(local_settings.opencode_base_url, "https://service.example/v1")
        local = app._current_provider()
        self.assertFalse(hasattr(local, "api_key"))
        self.assertEqual(local.base_url, local_request.text_model_base_url)

    def test_incomplete_online_options_do_not_block_optional_local_processing(self):
        app = make_app()
        settings = AppSettings(last_source="https://vimeo.com/11", text_provider="ollama", ollama_model_name="installed-model")
        self.assertEqual(app._build_request(settings).text_model_name, "installed-model")


class FlexibleQueueTests(unittest.TestCase):
    def test_visual_skips_audio_but_keeps_valid_video_in_batch(self):
        app = make_app()
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "audio.wav"
            audio.touch()
            settings = AppSettings(
                last_source=f"{audio}\nhttps://vimeo.com/11", include_visual=True,
                text_provider="ollama", ollama_model_name="vision-model",
            )
            with patch("info_summary.service.ProcessingService.find_existing", return_value=None):
                pending = app._prepare_processing(settings)
            self.assertEqual([item.source for _, item in pending], ["https://vimeo.com/11"])
            self.assertTrue(pending[0][1].include_visual)
            self.assertEqual(len(app.input_issues), 1)
            self.assertIn("只支持视频链接", app.input_issues[0].message)

    def test_bad_link_and_empty_folder_do_not_block_valid_sources(self):
        app = make_app()
        app.busy = False
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "empty"
            folder.mkdir()
            settings = AppSettings(last_source=f"https://vimeo.com/11\nhttps://bad.example/2\n{folder}", include_polish=False, include_summary=False)
            with patch("info_summary.service.ProcessingService.find_existing", return_value=None):
                pending = app._prepare_processing(settings)
            self.assertEqual([item.source for _, item in pending], ["https://vimeo.com/11"])
            self.assertEqual(len(app.input_issues), 2)
            self.assertTrue(any("文件夹中没有" in issue.message for issue in app.input_issues))

    def test_download_skips_local_audio_but_keeps_video_links(self):
        app = make_app()
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "audio.wav"
            audio.touch()
            settings = AppSettings(last_source=f"{audio}\nhttps://vimeo.com/11", task_mode="download")
            with patch("info_summary.video.VideoDownloadService.find_existing", return_value=None):
                pending = app._prepare_video_download(settings)
            self.assertEqual([item.url for _, item in pending], ["https://vimeo.com/11"])
            self.assertEqual(len(app.input_issues), 1)

    def test_append_uses_frozen_settings_and_never_starts_second_worker(self):
        app = make_app()
        app.settings.include_polish = app.settings.include_summary = False
        app.view.source = Value("https://vimeo.com/12\nhttps://vimeo.com/12\nhttps://bad.example/2")
        app.task_queue = TaskQueue()
        app.task_queue.add([("process", ProcessingRequest("https://vimeo.com/11", SourceType.VIDEO_URL))])
        with patch("info_summary.service.ProcessingService.find_existing", return_value=None), patch("info_summary.app.threading.Thread") as worker, patch.object(app, "_settings_from_view") as edited_settings:
            app._start()
        worker.assert_not_called()
        edited_settings.assert_not_called()
        self.assertEqual(len(app.task_queue.items), 2)
        self.assertEqual(app.task_queue.items[1].request.steps, (ProcessingStep.ORIGINAL,))
        self.assertEqual(app.view.source.get(), "https://bad.example/2")
        app.view.append_tasks.assert_called_once_with(["https://vimeo.com/12"])

    def test_append_after_worker_empty_event_is_processed_in_next_pass(self):
        app = make_app()
        first, second = request("first.wav"), request("second.wav")
        app.task_queue = TaskQueue()
        app.task_queue.add([("process", first)])
        active = app.task_queue.claim()
        app.task_queue.finish(active, "已完成")
        old_summary = app.task_queue.counts()
        app.task_queue.add([("process", second)])
        with patch("info_summary.app.threading.Thread") as thread:
            app._finish_batch(old_summary)
        self.assertTrue(app.busy)
        thread.return_value.start.assert_called_once()
        service = MagicMock()
        service.run.return_value = result()
        with patch("info_summary.service.ProcessingService", return_value=service):
            app._worker_loop(app.task_queue, "")
        app._poll_events()
        self.assertEqual(service.run.call_args.args[0], second)
        self.assertFalse(app.busy)
        self.assertIn("成功 2 项", app.view.status.get())

    def test_cancel_at_empty_queue_boundary_stops_late_appends(self):
        app = make_app()
        app.task_queue = TaskQueue()
        old_summary = app.task_queue.counts()
        app.task_queue.add([("process", request("late.wav"))])
        app.cancel_event.set()
        with patch("info_summary.app.threading.Thread") as thread:
            app._finish_batch(old_summary)
        thread.assert_not_called()
        self.assertEqual(app.task_queue.items[0].status, "已停止")
        self.assertIn("未完成 1 项", app.view.status.get())

    def test_restart_recovers_incomplete_queue_without_starting_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unfinished.json"
            saved = UnfinishedTasks(path)
            saved.add([("process", request("unfinished.wav"))])
            app = make_app()
            app.busy = False
            app.unfinished = UnfinishedTasks(path)
            with patch("info_summary.app.threading.Thread") as thread:
                app._restore_unfinished()
            thread.assert_not_called()
            self.assertEqual(app.retry_pending, [("process", request("unfinished.wav"))])
            self.assertIn("已找回 1 个", app.view.status.get())
            app.view.selected_task_index.return_value = 0
            app._remove_selected_task()
            self.assertEqual(UnfinishedTasks(path).pending(), [])


class BatchUsabilityTests(unittest.TestCase):
    def test_result_view_opens_report_then_legacy_document_or_video(self):
        app = make_app()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = folder / "阅读报告.html"
            summary = folder / "内容总结.docx"
            video = folder / "视频.mp4"
            for path in (report, summary, video):
                path.write_bytes(b"result")
            with patch("info_summary.app.open_path") as opened:
                app._open_saved_result(folder)
                opened.assert_called_with(report)
                report.unlink()
                app._open_saved_result(folder)
                opened.assert_called_with(summary)
                summary.unlink()
                app._open_saved_result(folder)
                opened.assert_called_with(video)

    def test_closed_settings_dialog_ignores_late_model_and_error_responses(self):
        app = make_app()
        target = MagicMock()
        target.closed = True
        app.events.put(("settings_provider", (target, "models", ("opencode", ["late-model"]))))
        app.events.put(("settings_provider", (target, "provider_error", "late error")))
        with patch("info_summary.app.messagebox.showerror") as error, patch.object(app, "_choose_model") as choose:
            app._poll_events()
        choose.assert_not_called()
        error.assert_not_called()
        target.message.set.assert_not_called()

    def test_provider_checks_use_draft_without_committing_to_main_form(self):
        app = make_app()
        target = MagicMock()
        target.closed = False
        target.provider.get.return_value = "opencode"
        target.opencode_key.get.return_value = "draft-secret"
        target.opencode_url.get.return_value = "https://draft.example/v1"
        target.opencode_model.get.return_value = "draft-model"
        provider = MagicMock()
        def immediate_thread(*, target, daemon):
            return MagicMock(start=target)
        with patch("info_summary.app.OpenCodeProvider", return_value=provider) as factory, patch("info_summary.app.threading.Thread", side_effect=immediate_thread):
            app._check_provider(target)
        factory.assert_called_once_with("draft-secret", "https://draft.example/v1")
        self.assertEqual(provider.generate.call_args.args[0], "draft-model")
        app._poll_events()
        target.message.set.assert_any_call("测试成功，当前模型可以生成文字。")
        app.settings_store.save.assert_not_called()
        app.view.opencode_model.set.assert_not_called()

    def test_multiple_video_links_use_one_worker_queue_and_deduplicate(self):
        app = make_app()
        settings = AppSettings(last_source="https://vimeo.com/1\nhttps://vimeo.com/2\nhttps://vimeo.com/1",
                               include_polish=False, include_summary=False)
        service = MagicMock()
        service.expand_requests.side_effect = lambda item: [item]
        service.find_existing.return_value = None
        with patch("info_summary.service.ProcessingService", return_value=service):
            pending = app._prepare_processing(settings)
        self.assertEqual([item.source for _, item in pending], ["https://vimeo.com/1", "https://vimeo.com/2"])

    def test_overlapping_folder_and_file_are_only_processed_once(self):
        app = make_app()
        first, second = request("one.wav"), request("two.wav")
        service = MagicMock()
        service.expand_requests.side_effect = [[first, second], [first]]
        service.find_existing.return_value = None
        with patch.object(app, "_build_requests", return_value=[first, first]), patch("info_summary.service.ProcessingService", return_value=service):
            pending = app._prepare_processing(AppSettings())
        self.assertEqual(pending, [("process", first), ("process", second)])

    def test_download_batch_reuses_existing_and_queues_only_new_urls(self):
        app = make_app()
        old = VideoDownloadResult("first", "https://vimeo.com/1", VideoQuality.BEST, Path("first"), Path("first/video.mp4"))
        service = MagicMock()
        service.find_existing.side_effect = [old, None]
        app.view.choose_existing.return_value = "existing"
        with patch("info_summary.video.VideoDownloadService", return_value=service):
            pending = app._prepare_video_download(AppSettings(last_source="https://vimeo.com/1\nhttps://vimeo.com/2", task_mode="download"))
        self.assertEqual([item.url for _, item in pending], ["https://vimeo.com/2"])
        self.assertEqual(app.reused_result_dirs, [Path("first")])
        app.view.choose_existing.assert_called_once()

    def test_retry_queue_keeps_failed_partial_and_stopped_without_successes(self):
        app = make_app()
        requests = [request(f"{i}.wav") for i in range(5)]
        service = MagicMock()
        service.run.side_effect = [result("success"), RuntimeError("失去网络连接"), result("partial", ["缺少总结"]), JobCancelled()]
        with patch("info_summary.service.ProcessingService", return_value=service):
            app._worker_loop(tuple(("process", item) for item in requests), "")
        app._poll_events()
        self.assertEqual([item for _, item in app.retry_pending], requests[1:])
        self.assertEqual(app.task_results, {0: Path("success"), 2: Path("partial")})
        self.assertEqual(app.history.record.call_count, 2)
        app.view.set_retry_available.assert_called_with(4)
        app.view.update_task.assert_any_call(4, "已停止", "已保存可续做的进度，点击“继续未完成”接着处理")

    def test_retry_uses_current_settings_and_preserves_saved_results(self):
        app = make_app()
        app.busy = False
        app.retry_pending = [("process", ProcessingRequest("https://vimeo.com/1", SourceType.VIDEO_URL))]
        app.result_dirs = [Path("already-saved")]
        settings = AppSettings(last_source="unrelated input", include_summary=False, include_polish=False,
                               text_provider="ollama", ollama_base_url="http://localhost:2222")
        with patch.object(app, "_settings_from_view", return_value=settings), patch.object(app, "_validate_output"), patch("info_summary.app.threading.Thread"):
            app._retry_incomplete()
        updated = app.task_queue.items[0].request
        self.assertEqual(updated.source, "https://vimeo.com/1")
        self.assertEqual(updated.steps, (ProcessingStep.ORIGINAL,))
        self.assertEqual(updated.text_model_base_url, "http://localhost:2222")
        self.assertEqual(app.result_dirs, [Path("already-saved")])

    def test_history_write_failure_does_not_lose_completed_result(self):
        app = make_app()
        app.history.record.side_effect = OSError("disk full")
        app.events.put(("result", result("saved")))
        app._poll_events()
        self.assertEqual(app.result_dirs, [Path("saved")])
        app.view.open_button.configure.assert_called_with(state="normal")

    def test_unwritable_output_is_caught_before_any_worker_starts(self):
        app = make_app()
        app.busy = False
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "not-a-folder"
            file.touch()
            with patch.object(app, "_settings_from_view", return_value=AppSettings(output_dir=str(file))), patch.object(app, "_prepare_processing") as prepare, patch("info_summary.app.messagebox.showerror") as error:
                app._start()
            prepare.assert_not_called()
            self.assertIn("保存位置无法写入", error.call_args.args[1])

    def test_edited_unsaved_secret_is_also_hidden(self):
        app = make_app()
        app.view.opencode_key = Value("new-private-key")
        self.assertNotIn("new-private-key", app._safe_text("error: new-private-key"))


class DesktopIntegrationTests(unittest.TestCase):
    def test_live_append_remove_and_bad_input_share_one_real_window(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        root.withdraw()
        release, started = threading.Event(), threading.Event()
        app = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                settings = AppSettings(output_dir=str(project / "output"), include_polish=False, include_summary=False)
                store = MagicMock()
                store.load.return_value = settings
                unfinished = UnfinishedTasks(project / "unfinished.json")
                with patch("info_summary.app.SettingsStore", return_value=store), patch("info_summary.app.ResultHistory", return_value=ResultHistory(project / "recent.json")), patch("info_summary.app.UnfinishedTasks", return_value=unfinished):
                    app = InfoSummaryApp(root, project)
                pipeline = ProcessingPipeline(project, checkpoint_dir=project / "checkpoints")
                service = ProcessingService(project, pipeline=pipeline, result_index=ResultIndex(project / "index.json"))
                attempts = []

                def original(request, *_args):
                    attempts.append(request.source)
                    if len(attempts) == 1:
                        started.set()
                        if not release.wait(5):
                            raise RuntimeError("test release timed out")
                    return "title", [TranscriptSegment(0, 2, "有效原文。")], "subtitle", "zh"

                with patch("info_summary.service.ProcessingService", return_value=service), patch.object(pipeline, "_obtain_original", side_effect=original):
                    app.view.source.set("https://vimeo.com/1\nhttps://vimeo.com/2")
                    app._start()
                    self.assertTrue(started.wait(3))
                    app.view.source.set("https://bad.example/1\nhttps://vimeo.com/3")
                    app._start()
                    self.assertEqual(app.view.start_button.cget("text"), "添加到队列")
                    self.assertEqual(app.view.source.get(), "https://bad.example/1")
                    self.assertIn("无法识别视频链接", app.view.input_issues_text.get("1.0", "end"))
                    self.assertEqual(len(app.view.task_tree.get_children()), 3)
                    app.view.task_tree.selection_set("1")
                    root.update()
                    self.assertNotIn("disabled", app.view.remove_task_button.state())
                    app.view.remove_task_button.invoke()
                    self.assertEqual(app.view.task_tree.set("1", "status"), "已移除")
                    release.set()
                    deadline = time.monotonic() + 8
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    self.assertFalse(app.busy)
                self.assertEqual(attempts, ["https://vimeo.com/1", "https://vimeo.com/3"])
                self.assertEqual(app.view.task_tree.set("0", "status"), "已完成")
                self.assertEqual(app.view.task_tree.set("2", "status"), "已完成")
                self.assertIn("已完成 2/2", app.view.tasks_heading.get())
                self.assertEqual(UnfinishedTasks(project / "unfinished.json").pending(), [])
        finally:
            release.set()
            if app is not None and app.worker is not None:
                app.worker.join(timeout=3)
            for callback in root.tk.call("after", "info"):
                root.after_cancel(callback)
            root.destroy()

    def test_real_window_worker_exports_history_and_retry_work_together(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                settings = AppSettings(output_dir=str(project / "output"), include_polish=False, include_summary=False)
                history = ResultHistory(project / "recent.json")
                store = MagicMock()
                store.load.return_value = settings
                with patch("info_summary.app.SettingsStore", return_value=store), patch("info_summary.app.ResultHistory", return_value=history), patch("info_summary.app.UnfinishedTasks", return_value=UnfinishedTasks(project / "unfinished.json")):
                    app = InfoSummaryApp(root, project)
                pipeline = ProcessingPipeline(project, checkpoint_dir=project / "checkpoints")
                service = ProcessingService(project, pipeline=pipeline, result_index=ResultIndex(project / "index.json"))
                attempts = []

                def original(request, *_args):
                    attempts.append(request.source)
                    if len(attempts) == 1:
                        raise RuntimeError("用于验证的网络中断")
                    return request.source.rsplit("/", 1)[-1], [TranscriptSegment(0, 2, "这是保存到文件的原文。")], "subtitle", "zh"

                def wait_for_worker():
                    deadline = time.monotonic() + 8
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    self.assertFalse(app.busy, "后台任务没有正常结束")

                with patch("info_summary.service.ProcessingService", return_value=service), patch.object(pipeline, "_obtain_original", side_effect=original):
                    app.view.source.set("https://vimeo.com/1\nhttps://vimeo.com/2")
                    app._start()
                    wait_for_worker()
                    self.assertEqual(app.view.task_tree.set("0", "status"), "失败")
                    self.assertEqual(app.view.task_tree.set("1", "status"), "已完成")
                    self.assertEqual(len(history.recent()), 1)
                    self.assertEqual(len(app.retry_pending), 1)
                    app._retry_incomplete()
                    wait_for_worker()
                self.assertEqual(attempts, ["https://vimeo.com/1", "https://vimeo.com/2", "https://vimeo.com/1"])
                self.assertEqual(app.view.task_tree.set("0", "status"), "已完成")
                self.assertEqual(len(app.result_dirs), 2)
                self.assertEqual(len(history.recent()), 2)
                reports = list((project / "output").rglob("阅读报告.html"))
                self.assertEqual(len(reports), 2)
                self.assertIn("这是保存到文件的原文", reports[0].read_text(encoding="utf-8"))
                self.assertEqual(list((project / "output").rglob("原文稿.docx")), [])
                self.assertEqual(list((project / "output").rglob("原文稿.md")), [])
                self.assertFalse(app.retry_pending)
        finally:
            for callback in root.tk.call("after", "info"):
                root.after_cancel(callback)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
