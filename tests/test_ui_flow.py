from __future__ import annotations

import tkinter as tk
import unittest
from tkinter import ttk
from unittest.mock import patch

from info_summary.ui import TaskView


class InterfaceFlowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.view = TaskView(self.root, {name: lambda: None for name in (
            "choose_audio", "choose_folder", "choose_output", "choose_cookies",
            "check_provider", "load_models", "start_ollama", "start", "cancel", "open_result",
            "paste_source", "import_sources", "clear_source", "show_history",
            "retry_incomplete", "open_selected_result",
        )})

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def widgets(self, widget):
        yield widget
        for child in widget.winfo_children():
            yield from self.widgets(child)

    def test_presets_and_preview_match_output_files(self):
        view = self.view
        view.select_preset(False, True)
        self.assertFalse(view.include_polish.get())
        self.assertTrue(view.include_summary.get())
        self.assertIn("原文稿 + 内容总结", view.output_preview.get())
        self.assertNotIn("润色稿", view.output_preview.get())
        view.select_preset(True, False)
        self.assertIn("原文稿 + 润色稿", view.output_preview.get())
        view.select_preset(False, False)
        self.assertIn("无需文字模型或 API 密钥", view.model_hint.get())
        view.preset_choice.set("文稿和总结")
        view.preset_combo.event_generate("<<ComboboxSelected>>")
        self.assertIn("内容总结", view.output_preview.get())
        self.assertIn("润色稿", view.output_preview.get())

    def test_start_requires_source_and_only_text_mode_requires_format(self):
        view = self.view
        self.assertIn("disabled", view.start_button.state())
        view.source.set("https://vimeo.com/1")
        self.assertNotIn("disabled", view.start_button.state())
        view.export_html.set(False)
        view.export_docx.set(False)
        view.export_markdown.set(False)
        self.assertIn("disabled", view.start_button.state())
        view.select_mode("download")
        self.assertNotIn("disabled", view.start_button.state())
        self.assertIn("MP4", view.output_preview.get())
        self.assertEqual(view.process_steps_card.winfo_manager(), "")
        view.select_mode("process")
        self.assertIn("disabled", view.start_button.state())

    def test_visual_is_optional_requires_model_and_is_independent_from_presets(self):
        view = self.view
        self.assertFalse(view.include_visual.get())
        self.assertEqual(view.visual_focus.get(), "")
        view.source.set("https://vimeo.com/1")
        view.visual_button.invoke()
        self.assertTrue(view.include_visual.get())
        self.assertEqual(view.visual_button.cget("style"), "Selected.TButton")
        self.assertIn("已开启", view.visual_button.cget("text"))
        self.assertIn("画面笔记和截图", view.output_preview.get())
        self.assertIn("disabled", view.start_button.state())
        view.opencode_url.set("https://service.example/v1")
        view.opencode_model.set("vision-model")
        view.opencode_key.set("test-secret")
        self.assertNotIn("disabled", view.start_button.state())
        self.assertIn("原文和截图", view.model_hint.get())
        self.assertIn("支持图片", view.model_hint.get())
        view.select_preset(False, True)
        self.assertTrue(view.include_visual.get())
        view.select_preset(False, False)
        self.assertTrue(view.include_visual.get())
        view.visual_button.invoke()
        self.assertFalse(view.include_visual.get())
        self.assertEqual(view.visual_button.cget("style"), "Secondary.TButton")
        self.assertIn("已关闭", view.visual_button.cget("text"))
        self.assertNotIn("画面笔记", view.output_preview.get())

    def test_visual_focus_enables_visual_and_controls_freeze_while_running(self):
        view = self.view
        view.visual_focus.set("把 05:20 的图表讲细一点")
        self.assertTrue(view.include_visual.get())
        view.set_busy(True)
        self.assertIn("disabled", view.visual_button.state())
        self.assertIn("disabled", view.visual_focus_entry.state())
        view.set_busy(False)
        self.assertNotIn("disabled", view.visual_button.state())
        self.assertNotIn("disabled", view.visual_focus_entry.state())
        self.assertEqual(view.visual_focus.get(), "把 05:20 的图表讲细一点")
        view.visual_button.invoke()
        self.assertEqual(view.visual_focus.get(), "")

    def test_running_freezes_settings_but_keeps_logs_and_results_available(self):
        view = self.view
        view.source.set("https://vimeo.com/1")
        view.open_button.configure(state="normal")
        view.ollama_model.set("local-model")
        view.provider.set("ollama")
        view.select_preset(True, True)
        combos = [widget for widget in self.widgets(view.form) if isinstance(widget, ttk.Combobox)]
        original_states = [str(widget.cget("state")) for widget in combos]
        view.set_busy(True)
        view.select_mode("download")
        view.select_preset(False, False)
        self.assertEqual(view.task_mode.get(), "process")
        self.assertTrue(view.include_polish.get())
        self.assertTrue(view.include_summary.get())
        self.assertEqual(str(view.source_entry.cget("state")), "normal")
        self.assertNotIn("disabled", view.log_button.state())
        self.assertNotIn("disabled", view.open_button.state())
        view.toggle_log()
        self.assertTrue(view.log_visible)
        view.set_busy(False)
        self.assertEqual([str(widget.cget("state")) for widget in combos], original_states)
        self.assertNotIn("disabled", view.start_button.state())
        self.assertIn("disabled", view.cancel_button.state())

    def test_multiline_source_tracks_typing_programmatic_updates_and_busy_state(self):
        view = self.view
        sources = "https://vimeo.com/1\nhttps://vimeo.com/2"
        view.source.set(sources)
        self.assertEqual(view.source_entry.get("1.0", "end-1c"), sources)
        view.source_entry.insert("end", "\n/Users/me/录音.m4a")
        self.root.update()
        self.assertEqual(view.source.get(), sources + "\n/Users/me/录音.m4a")
        view.set_busy(True)
        view.source.set("https://vimeo.com/3")
        self.assertEqual(view.source_entry.get("1.0", "end-1c"), "https://vimeo.com/3")
        self.assertEqual(str(view.source_entry.cget("state")), "normal")
        view.source_entry.insert("end", "\nhttps://vimeo.com/4")
        self.root.update()
        self.assertEqual(view.source.get(), "https://vimeo.com/3\nhttps://vimeo.com/4")
        view.set_busy(False)
        self.assertEqual(str(view.source_entry.cget("state")), "normal")

    def test_batch_tasks_keep_results_and_error_details_available_while_running(self):
        view = self.view
        self.assertEqual(view.tasks_card.winfo_manager(), "")
        view.set_tasks(["视频一", "录音二", "视频三"])
        self.assertEqual(view.tasks_card.winfo_manager(), "pack")
        self.assertEqual(view.selected_task_index(), 0)
        self.assertEqual(view.task_tree.set("0", "status"), "等待")
        view.update_task(0, "已完成", "已保存原文与总结")
        view.update_task(1, "失败", "网站要求登录，请在网站访问设置中选择 cookies 文件。")
        view.set_retry_available(1)
        self.assertNotIn("disabled", view.retry_button.state())
        view.set_busy(True)
        self.assertIn("disabled", view.retry_button.state())
        self.assertNotIn("disabled", view.tasks_toggle.state())
        self.assertNotIn("disabled", view.open_task_button.state())
        self.assertNotIn("disabled", view.history_button.state())
        view.task_tree.selection_set("1")
        self.root.update()
        self.assertEqual(view.selected_task_index(), 1)
        self.assertIn("网站要求登录", view.task_detail.get("1.0", "end-1c"))
        self.assertIn("disabled", view.open_task_button.state())
        view.toggle_tasks()
        self.assertEqual(view.tasks_body.winfo_manager(), "")
        view.set_busy(False)
        self.assertNotIn("disabled", view.retry_button.state())
        self.assertIn("（1）", view.retry_button.cget("text"))
        view.set_tasks([])
        self.assertEqual(view.tasks_card.winfo_manager(), "")
        self.assertIsNone(view.selected_task_index())

    def test_model_settings_shortcut_reveals_ai_options(self):
        view = self.view
        view.show_ai_settings()
        dialog = view.settings_dialog
        self.assertFalse(dialog.closed)
        self.assertEqual(dialog.tabs.select(), str(dialog.pages["模型设置"]))
        self.assertEqual(tuple(dialog.pages), ("模型设置", "语音识别", "网站访问"))
        self.assertIsNot(dialog.opencode_key, view.opencode_key)
        dialog.close()

    def test_settings_draft_is_applied_only_on_save(self):
        view = self.view
        saved = []
        view.callbacks["save_settings"] = saved.append
        view.open_settings()
        first = view.settings_dialog
        first.opencode_model.set("cancelled-model")
        first.opencode_key.set("cancelled-secret")
        self.assertEqual(view.opencode_model.get(), "")
        first.close()
        self.assertEqual(view.opencode_key.get(), "")
        self.assertEqual(saved, [])
        view.open_settings()
        second = view.settings_dialog
        self.assertEqual(second.opencode_model.get(), "")
        second.opencode_model.set("saved-model")
        second.opencode_url.set("https://service.example/v1")
        second.opencode_key.set("saved-secret")
        second.save()
        self.assertTrue(second.closed)
        self.assertEqual(view.opencode_model.get(), "saved-model")
        self.assertEqual(view.opencode_key.get(), "saved-secret")
        self.assertEqual(len(saved), 1)

    def test_failed_settings_save_keeps_draft_without_changing_current_values(self):
        view = self.view
        def failed_save(_settings):
            raise OSError("无法写入")
        view.callbacks["save_settings"] = failed_save
        view.open_settings()
        dialog = view.settings_dialog
        dialog.opencode_model.set("draft-model")
        with patch("info_summary.ui.dialogs.messagebox.showerror") as error:
            dialog.save()
        error.assert_called_once()
        self.assertFalse(dialog.closed)
        self.assertEqual(dialog.opencode_model.get(), "draft-model")
        self.assertEqual(view.opencode_model.get(), "")
        dialog.close()

    def test_small_settings_window_keeps_save_cancel_visible_and_content_scrollable(self):
        self.root.deiconify()
        self.view.open_settings()
        dialog = self.view.settings_dialog
        dialog.window.geometry("520x460")
        for category in dialog.pages:
            dialog.tabs.select(dialog.pages[category])
            self.root.update()
            canvas = dialog.canvases[dialog.tabs.select()]
            self.assertGreater(canvas.winfo_height(), 100)
            for button in self.widgets(dialog.window):
                if isinstance(button, ttk.Button) and button.cget("text") in ("保存", "取消"):
                    self.assertTrue(button.winfo_ismapped())
                    self.assertGreaterEqual(button.winfo_rooty(), dialog.window.winfo_rooty())
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), dialog.window.winfo_rooty() + 460)
        dialog.tabs.select(dialog.pages["模型设置"])
        self.root.update()
        canvas = dialog.canvases[dialog.tabs.select()]
        canvas.yview_moveto(1)
        self.assertGreater(canvas.yview()[0], 0)
        dialog.close()

    def test_friendly_choices_load_and_save_original_setting_values(self):
        self.view.open_settings()
        dialog = self.view.settings_dialog
        for name, stored, label, first_value in (
            ("output_language", "source", "跟随原文", "zh-CN"),
            ("device_mode", "auto", "自动选择", "cpu"),
            ("compute_type", "int8", "省内存（int8）", ""),
            ("js_runtime", "node", "Node.js", "deno"),
        ):
            with self.subTest(setting=name):
                variable = getattr(dialog, name)
                variable.set(stored)
                combo = next(widget for widget in self.widgets(dialog.window)
                             if isinstance(widget, ttk.Combobox) and label in widget.cget("values"))
                self.assertEqual(combo.get(), label)
                combo.current(0)
                combo.event_generate("<<ComboboxSelected>>")
                self.assertEqual(variable.get(), first_value)
        dialog.close()

    def test_model_requirements_are_clear_before_starting_and_do_not_block_downloads(self):
        view = self.view
        view.source.set("https://vimeo.com/1")
        view.select_preset(False, True)
        self.assertIn("disabled", view.start_button.state())
        self.assertIn("接口地址、模型名称、API 密钥", view.start_hint.get())
        view.opencode_model.set("test-model")
        view.opencode_url.set("https://api.example.test/v1")
        self.assertIn("API 密钥", view.start_hint.get())
        self.assertNotIn("模型名称", view.start_hint.get())
        view.opencode_key.set("test-secret")
        self.assertNotIn("disabled", view.start_button.state())
        self.assertEqual(view.start_hint.get(), "")
        view.provider.set("ollama")
        self.assertIn("disabled", view.start_button.state())
        self.assertIn("本机模型名称", view.start_hint.get())
        view.ollama_model.set("local-model")
        self.assertNotIn("disabled", view.start_button.state())
        view.ollama_model.set("")
        view.select_mode("download")
        self.assertNotIn("disabled", view.start_button.state())
        view.select_mode("process")
        view.select_preset(False, False)
        self.assertNotIn("disabled", view.start_button.state())

    def test_editing_source_hides_previous_input_warnings(self):
        view = self.view
        view.source.set("incorrect input")
        view.show_input_issues(["之前的内容无法识别"])
        self.assertEqual(view.input_issues_box.winfo_manager(), "pack")
        view.source_entry.insert("end", " edited")
        self.root.update()
        self.assertEqual(view.input_issues_box.winfo_manager(), "")
        view.show_input_issues(["之前的内容无法识别"])
        view.source.set("https://vimeo.com/1")
        self.assertEqual(view.input_issues_box.winfo_manager(), "")

    def test_only_chosen_provider_fields_are_shown(self):
        self.view.open_settings()
        dialog = self.view.settings_dialog
        dialog.provider_buttons["opencode"].invoke()
        self.assertEqual(dialog.ollama_fields.winfo_manager(), "")
        self.assertEqual(dialog.opencode_fields.winfo_manager(), "pack")
        self.assertEqual(dialog.start_ollama_button.winfo_manager(), "")
        dialog.provider_buttons["ollama"].invoke()
        self.assertEqual(dialog.provider.get(), "ollama")
        self.assertEqual(self.view.provider.get(), "opencode")
        self.assertEqual(dialog.opencode_fields.winfo_manager(), "")
        self.assertEqual(dialog.ollama_fields.winfo_manager(), "pack")
        self.assertEqual(dialog.start_ollama_button.winfo_manager(), "pack")
        dialog.close()

    def test_online_default_reports_missing_configuration_without_opening_settings(self):
        view = self.view
        self.assertIsNone(view.settings_dialog)
        self.assertEqual(view.provider.get(), "opencode")
        self.assertFalse(view.include_polish.get())
        self.assertFalse(view.include_summary.get())
        self.assertTrue(view.export_html.get())
        self.assertFalse(view.export_docx.get())
        self.assertFalse(view.export_markdown.get())
        self.assertIn("无需文字模型或 API 密钥", view.model_hint.get())
        self.assertEqual(view.ai_settings_button.winfo_manager(), "")
        view.select_preset(False, True)
        self.assertEqual(view.ai_settings_button.winfo_manager(), "pack")
        self.assertIn("尚未配置，点击模型设置", view.model_hint.get())
        view.opencode_model.set("test-model")
        view.opencode_url.set("https://api.example.test/v1")
        self.assertIn("尚未配置", view.model_hint.get())
        view.opencode_key.set("test-secret")
        self.assertNotIn("尚未配置", view.model_hint.get())
        self.assertIn("可能产生费用", view.model_hint.get())
        self.assertNotIn("test-secret", view.model_hint.get())
        view.open_settings()
        dialog = view.settings_dialog
        key_field = next(widget for widget in self.widgets(dialog.opencode_fields)
                         if isinstance(widget, ttk.Entry)
                         and str(widget.cget("textvariable")) == str(dialog.opencode_key))
        self.assertEqual(key_field.cget("show"), "*")
        dialog.close()
        view.opencode_key.set("")
        self.assertIn("尚未配置", view.model_hint.get())

    def test_switching_providers_preserves_separate_settings_and_is_locked_while_running(self):
        view = self.view
        view.select_preset(False, True)
        view.opencode_model.set("online-model")
        view.opencode_url.set("https://api.example.test/v1")
        view.opencode_key.set("test-secret")
        view.ollama_model.set("local-model")
        view.ollama_url.set("http://localhost:11434")
        view.select_provider("ollama")
        self.assertIn("local-model", view.model_hint.get())
        view.set_busy(True)
        view.select_provider("opencode")
        view.open_settings()
        self.assertIsNone(view.settings_dialog)
        self.assertEqual(view.provider.get(), "ollama")
        view.set_busy(False)
        view.select_provider("opencode")
        self.assertIn("online-model", view.model_hint.get())
        self.assertEqual(view.opencode_key.get(), "test-secret")
        self.assertEqual(view.ollama_model.get(), "local-model")
        self.assertEqual(view.ollama_url.get(), "http://localhost:11434")

    def test_more_settings_keeps_formats_available_after_switching_modes(self):
        view = self.view
        self.assertEqual(view.formats.master, view.output_card)
        self.assertEqual(view.formats.winfo_manager(), "pack")
        view.export_docx.set(True)
        view.export_markdown.set(True)
        view.select_mode("download")
        self.assertEqual(view.formats.winfo_manager(), "")
        view.select_mode("process")
        self.assertEqual(view.formats.winfo_manager(), "pack")
        self.assertIn("阅读报告、Word、Markdown", view.output_preview.get())

    def test_visible_controls_fit_small_window_and_expanded_settings(self):
        view = self.view
        self.root.geometry("720x520")
        self.root.deiconify()
        view.source.set("https://vimeo.com/1")
        view.set_tasks(["较长的任务名称 " * 12, "第二个视频"])
        view.set_retry_available(1)
        view.visual_focus.set("把 05:20 的图表讲细一点")
        for mode in ("process", "download"):
            view.select_mode(mode)
            self.root.update()
            for widget in self.widgets(view.form):
                if isinstance(widget, (ttk.Button, ttk.Label, ttk.Entry)) and widget.winfo_ismapped():
                    with self.subTest(mode=mode, widget=str(widget)):
                        self.assertLessEqual(widget.winfo_x() + widget.winfo_width(), widget.master.winfo_width() + 1)
                        self.assertGreater(widget.winfo_width(), 5)
            for button in (view.start_button, view.cancel_button, view.log_button, view.open_button,
                           view.open_folder_button, view.history_button, view.settings_button):
                with self.subTest(mode=mode, button=button.cget("text")):
                    self.assertTrue(button.winfo_ismapped())
                    self.assertGreaterEqual(button.winfo_rootx(), self.root.winfo_rootx())
                    self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), self.root.winfo_rootx() + 720)
                    self.assertGreaterEqual(button.winfo_rooty(), self.root.winfo_rooty())
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), self.root.winfo_rooty() + 520)

    def test_small_window_keeps_bottom_actions_visible_with_tasks_and_open_log(self):
        view = self.view
        self.root.geometry("720x520")
        self.root.deiconify()
        view.source.set("https://vimeo.com/1")
        view.set_tasks(["任务一", "任务二"])
        view.update_task(0, "失败", "请检查网站访问设置，然后继续未完成任务。")
        view.status.set("部分完成 · 成功 1 项，失败 1 项，请查看运行记录")
        view.set_retry_available(1)
        view.toggle_log()
        self.root.update()
        self.assertGreater(view.content_canvas.winfo_height(), 80)
        for button in (view.start_button, view.cancel_button, view.log_button,
                       view.history_button, view.open_button):
            with self.subTest(button=button.cget("text")):
                self.assertTrue(button.winfo_ismapped())
                self.assertGreaterEqual(button.winfo_rootx(), self.root.winfo_rootx())
                self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), self.root.winfo_rootx() + 720)
                self.assertGreaterEqual(button.winfo_rooty(), self.root.winfo_rooty())
                self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), self.root.winfo_rooty() + 520)


if __name__ == "__main__":
    unittest.main()
