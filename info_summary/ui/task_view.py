from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..requests import request_requirements
from ..settings import AppSettings
from .dialogs import SettingsDialog
from .form_state import FormState
from .task_list import TaskList
from .theme import COLORS, configure_theme


PRESETS = {
    "只提取原文": (False, False), "快速看重点": (False, True),
    "整理成文稿": (True, False), "文稿和总结": (True, True),
}


class TaskView:
    """The daily workflow. Detailed preferences live in SettingsDialog."""

    def __init__(self, root: tk.Tk, callbacks: dict) -> None:
        self.root, self.callbacks = root, callbacks
        self._settings = AppSettings()
        self.state = FormState(root, self._settings)
        self._busy = self._stopping = self._syncing_source = False
        self._saved_control_states = {}
        self.settings_dialog = None
        self.progress = tk.DoubleVar(root, value=0)
        self.status = tk.StringVar(root, value="准备就绪")
        self.output_preview = tk.StringVar(root)
        self.model_hint = tk.StringVar(root)
        self.start_hint = tk.StringVar(root)
        self.source_label = tk.StringVar(root)
        self.source_hint = tk.StringVar(root)
        self.preset_choice = tk.StringVar(root, value="只提取原文")
        configure_theme(root)
        self._build()
        self.source.trace_add("write", self._sync_source_text)
        for name in ("source", "include_polish", "include_summary", "include_visual", "export_html",
                     "export_docx", "export_markdown", "provider", "ollama_model", "ollama_url",
                     "opencode_model", "opencode_url", "opencode_key", "video_quality"):
            getattr(self, name).trace_add("write", self._on_options_changed)
        self.visual_focus.trace_add("write", self._on_visual_focus_changed)
        self._on_options_changed()

    def __getattr__(self, name):
        state = self.__dict__.get("state")
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        tasks = self.__dict__.get("task_list")
        if tasks is not None and hasattr(tasks, name):
            return getattr(tasks, name)
        raise AttributeError(name)

    def load_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self.state.load(settings)
        self.refresh_mode()

    def collect_settings(self, previous: AppSettings) -> AppSettings:
        return self.state.collect(previous)

    def _build(self) -> None:
        root = self.root
        root.title("信息整理工具")
        width = min(900, int(root.winfo_screenwidth() * .86))
        height = min(740, int(root.winfo_screenheight() * .8))
        root.geometry(f"{width}x{height}")
        root.minsize(min(720, width), min(520, height))
        shell = ttk.Frame(root, style="Page.TFrame")
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell, padding=(20, 12), style="Page.TFrame")
        header.pack(fill="x")
        self.mode_buttons = {}
        for mode, label in (("process", "整理文字"), ("download", "下载视频")):
            button = ttk.Button(header, text=label, command=lambda value=mode: self.select_mode(value), style="Secondary.TButton")
            button.pack(side="left", padx=(0, 8))
            self.mode_buttons[mode] = button
        self.settings_button = ttk.Button(header, text="设置", command=self.open_settings, style="Compact.TButton")
        self.settings_button.pack(side="right")
        self.history_button = ttk.Button(header, text="最近结果", command=self.callbacks.get("show_history"), style="Compact.TButton")
        self.history_button.pack(side="right", padx=(0, 8))
        self._action_card(shell)
        self.content_canvas = canvas = tk.Canvas(shell, bg=COLORS["page"], highlightthickness=0, yscrollincrement=1)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.form = content = ttk.Frame(canvas, padding=(20, 0, 12, 16), style="Page.TFrame")
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        self._is_macos = root.tk.call("tk", "windowingsystem") == "aqua"
        self._scroll_bind_tag = "InfoSummaryFormScroll"
        root.bind_class(self._scroll_bind_tag, "<MouseWheel>", self._scroll_content)
        try:
            root.bind_class(self._scroll_bind_tag, "<TouchpadScroll>", self._scroll_touchpad)
        except tk.TclError:
            pass
        self._source_card(content)
        self._steps_card(content)
        self._download_card(content)
        self._output_card(content)
        self.task_list = TaskList(self, content)
        self.log_frame = ttk.Frame(content, padding=12, style="Panel.TFrame")
        self.log_text = tk.Text(self.log_frame, height=7, wrap="word", state="disabled", bg=COLORS["log"],
                                fg=COLORS["text"], relief="flat", font=("Helvetica Neue", 11))
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_visible = False
        self.refresh_mode()
        self._enable_content_scrolling(content)

    def _card(self, parent, title: str) -> ttk.Frame:
        card = ttk.Frame(parent, padding=14, style="Panel.TFrame")
        card.pack(fill="x", pady=(0, 10))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return card

    def _source_card(self, parent) -> None:
        self.source_card = card = self._card(parent, "添加内容")
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        self.source_entry = tk.Text(row, height=4, width=1, wrap="word", undo=True, bg=COLORS["card"],
                                    fg=COLORS["text"], relief="flat", highlightthickness=1,
                                    highlightbackground=COLORS["line"], font=("Helvetica Neue", 12), padx=8, pady=6)
        self.source_entry.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(row, command=self.source_entry.yview)
        scroll.pack(side="right", fill="y")
        self.source_entry.configure(yscrollcommand=scroll.set)
        self.source_entry.bind("<<Modified>>", self._source_text_changed)
        self.source_entry.edit_modified(False)
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="粘贴", command=self.callbacks.get("paste_source"), style="Compact.TButton").pack(side="left", padx=(0, 8))
        self.add_menu_button = ttk.Menubutton(actions, text="添加…", style="Compact.TButton")
        self.add_menu_button.pack(side="left")
        self.add_menu = tk.Menu(self.add_menu_button, tearoff=False)
        for label, name in (("选择音频（可多选）", "choose_audio"), ("选择文件夹", "choose_folder"), ("导入链接清单", "import_sources")):
            self.add_menu.add_command(label=label, command=self.callbacks.get(name))
        self.add_menu_button.configure(menu=self.add_menu)
        ttk.Button(actions, text="清空", command=self.callbacks.get("clear_source"), style="Compact.TButton").pack(side="right")
        hint = ttk.Label(card, textvariable=self.source_hint, style="Hint.TLabel", wraplength=600)
        hint.pack(anchor="w", pady=(6, 0))
        card.bind("<Configure>", lambda event: hint.configure(wraplength=max(180, event.width - 28)), add="+")
        self.input_issues_box = ttk.Frame(card, style="Card.TFrame")
        self.input_issues_text = tk.Text(self.input_issues_box, height=3, width=1, wrap="word", state="disabled", bg=COLORS["card"], fg=COLORS["warning"], relief="flat")
        self.input_issues_text.pack(side="left", fill="x", expand=True)
        issue_scroll = ttk.Scrollbar(self.input_issues_box, command=self.input_issues_text.yview)
        issue_scroll.pack(side="right", fill="y")
        self.input_issues_text.configure(yscrollcommand=issue_scroll.set)

    def _steps_card(self, parent) -> None:
        self.process_steps_card = card = self._card(parent, "选择结果")
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        self.preset_combo = ttk.Combobox(row, textvariable=self.preset_choice, values=tuple(PRESETS), state="readonly", width=18)
        self.preset_combo.pack(side="left", padx=(0, 10))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_preset(*PRESETS[self.preset_choice.get()]))
        self.visual_button = ttk.Button(row, text="结合画面  已关闭", command=self._toggle_visual, style="Secondary.TButton")
        self.visual_button.pack(side="left")
        self.visual_focus_row = ttk.Frame(card, style="Card.TFrame")
        ttk.Label(self.visual_focus_row, text="重点说明（可选）", style="Hint.TLabel").pack(anchor="w", pady=(8, 4))
        self.visual_focus_entry = ttk.Entry(self.visual_focus_row, textvariable=self.visual_focus)
        self.visual_focus_entry.pack(fill="x")
        ttk.Label(self.visual_focus_row, text="仅支持视频链接，例如：把 05:20 附近的图表讲细一点。", style="Hint.TLabel", wraplength=580).pack(anchor="w", pady=4)
        self.model_row = ttk.Frame(card, style="Card.TFrame")
        self.model_row.pack(fill="x", pady=(8, 0))
        self.ai_settings_button = ttk.Button(self.model_row, text="模型设置", command=self.show_ai_settings, style="Compact.TButton")
        label = ttk.Label(self.model_row, textvariable=self.model_hint, style="Hint.TLabel", wraplength=500)
        label.pack(side="left", fill="x", expand=True)
        self.model_row.bind("<Configure>", lambda event: label.configure(wraplength=max(160, event.width - 100)), add="+")

    def _download_card(self, parent) -> None:
        self.download_card = self._card(parent, "视频清晰度")
        self.quality_buttons = {}
        row = ttk.Frame(self.download_card, style="Card.TFrame")
        row.pack(fill="x")
        for value, label in (("best", "最佳质量"), ("1080", "最高 1080p"), ("720", "最高 720p")):
            button = ttk.Button(row, text=label, command=lambda selected=value: self.select_quality(selected), style="Secondary.TButton")
            button.pack(side="left", padx=(0, 8))
            self.quality_buttons[value] = button

    def _output_card(self, parent) -> None:
        self.output_card = card = self._card(parent, "保存结果")
        self.formats = ttk.Frame(card, style="Card.TFrame")
        self.formats.pack(fill="x", pady=(0, 8))
        self.format_buttons = {}
        for name, label in (("export_html", "阅读报告"), ("export_docx", "Word"), ("export_markdown", "Markdown")):
            button = ttk.Button(self.formats, text=label, style="Secondary.TButton",
                                command=lambda field=name: self._toggle_format(field))
            button.pack(side="left", padx=(0, 12))
            self.format_buttons[name] = button
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="保存到", style="Hint.TLabel").pack(side="left", padx=(0, 8))
        self.output_entry = ttk.Entry(row, textvariable=self.output_dir)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="更改位置", command=self.callbacks.get("choose_output"), style="Compact.TButton").pack(side="left")
        preview = ttk.Label(card, textvariable=self.output_preview, style="Preview.TLabel", wraplength=620)
        preview.pack(anchor="w", pady=(8, 0))
        card.bind("<Configure>", lambda event: preview.configure(wraplength=max(180, event.width - 28)), add="+")

    def _action_card(self, parent) -> None:
        footer = ttk.Frame(parent, padding=(20, 10, 20, 12), style="Card.TFrame")
        footer.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(footer, textvariable=self.status, style="CardText.TLabel", wraplength=640)
        self.status_label.pack(anchor="w")
        self.start_hint_label = ttk.Label(footer, textvariable=self.start_hint, style="Hint.TLabel", wraplength=640)
        self.start_hint_label.pack(anchor="w", pady=(2, 0))
        footer.bind("<Configure>", lambda event: [label.configure(wraplength=max(200, event.width - 40)) for label in (self.status_label, self.start_hint_label)], add="+")
        ttk.Progressbar(footer, variable=self.progress, maximum=100, style="Blue.Horizontal.TProgressbar").pack(fill="x", pady=(7, 8))
        buttons = ttk.Frame(footer, style="Card.TFrame")
        buttons.pack(fill="x")
        self.log_button = ttk.Button(buttons, text="运行记录", command=self.toggle_log, style="Compact.TButton")
        self.log_button.pack(side="left")
        self.open_folder_button = ttk.Button(buttons, text="打开文件夹", command=self.callbacks.get("open_folder"), state="disabled", style="Compact.TButton")
        self.open_folder_button.pack(side="right")
        self.open_button = ttk.Button(buttons, text="查看结果", command=self.callbacks.get("open_result"), state="disabled", style="Soft.TButton")
        self.open_button.pack(side="right", padx=(0, 8))
        self.cancel_button = ttk.Button(buttons, text="停止", command=self.callbacks.get("cancel"), state="disabled", style="Compact.TButton")
        self.cancel_button.pack(side="right", padx=(0, 8))
        self.start_button = ttk.Button(buttons, text="开始处理", command=self._start_if_ready, style="Primary.TButton")
        self.start_button.pack(side="right", padx=(0, 8))

    def select_mode(self, mode: str) -> None:
        if not self._busy:
            self.task_mode.set(mode)
            self.refresh_mode()

    def refresh_mode(self) -> None:
        downloading = self.task_mode.get() == "download"
        for value, button in self.mode_buttons.items():
            button.configure(style="Selected.TButton" if value == self.task_mode.get() else "Secondary.TButton")
        self.source_hint.set("粘贴视频链接或分享文案，可一次添加多条" if downloading else "粘贴视频链接或分享文案；音频路径每行一个")
        self.source_label.set("视频链接 / 分享文案" if downloading else "视频链接 / 分享文案 / 音频路径")
        self.process_steps_card.pack_forget()
        self.download_card.pack_forget()
        target = self.download_card if downloading else self.process_steps_card
        target.pack(fill="x", pady=(0, 10), before=self.output_card)
        if downloading:
            self.formats.pack_forget()
        else:
            self.formats.pack(fill="x", pady=(0, 8), before=self.output_entry.master)
        for index in (0, 1):
            self.add_menu.entryconfigure(index, state="disabled" if downloading else "normal")
        self.refresh_quality()
        self._refresh_request_preview()

    def select_preset(self, polish: bool, summary: bool) -> None:
        if not self._busy:
            self.include_polish.set(polish)
            self.include_summary.set(summary)

    def _toggle_visual(self) -> None:
        if not self._busy:
            self.include_visual.set(not self.include_visual.get())
            if not self.include_visual.get():
                self.visual_focus.set("")

    def _on_visual_focus_changed(self, *_args) -> None:
        if self.visual_focus.get().strip():
            self.include_visual.set(True)
        self._on_options_changed()

    def select_quality(self, quality: str) -> None:
        if not self._busy:
            self.video_quality.set(quality)
            self.refresh_quality()

    def refresh_quality(self) -> None:
        for value, button in self.quality_buttons.items():
            button.configure(style="Selected.TButton" if value == self.video_quality.get() else "Secondary.TButton")

    def select_provider(self, provider: str) -> None:
        if not self._busy:
            self.provider.set(provider)

    def _refresh_choice_widgets(self) -> None:
        selected = self.include_polish.get(), self.include_summary.get()
        self.preset_choice.set(next(label for label, flags in PRESETS.items() if flags == selected))
        visual = self.include_visual.get()
        self.visual_button.configure(style="Selected.TButton" if visual else "Secondary.TButton", text=f"结合画面  {'已开启' if visual else '已关闭'}")
        if visual:
            self.visual_focus_row.pack(fill="x", before=self.model_row)
        else:
            self.visual_focus_row.pack_forget()
        for name, button in self.format_buttons.items():
            button.configure(style="Selected.TButton" if getattr(self, name).get() else "Secondary.TButton")

    def _toggle_format(self, name: str) -> None:
        if not self._busy:
            variable = getattr(self, name)
            variable.set(not variable.get())

    def _on_options_changed(self, *_args) -> None:
        self._refresh_choice_widgets()
        self._refresh_request_preview()

    def _refresh_request_preview(self) -> None:
        downloading = self.task_mode.get() == "download"
        settings = self.state.collect(self._settings, validate=False)
        requirements = request_requirements(settings)
        missing = requirements.missing_model_fields if not downloading else ()
        if downloading:
            quality = {"best": "最佳质量", "1080": "最高 1080p", "720": "最高 720p"}.get(self.video_quality.get(), "最佳质量")
            self.output_preview.set(f"将保存：带声音的 MP4 视频 · {quality}")
        else:
            outputs = ["原文稿"]
            if settings.include_polish:
                outputs.append("润色稿")
            if settings.include_summary:
                outputs.append("内容总结")
            if requirements.include_visual:
                outputs.append("画面笔记和截图")
            formats = [label for name, label in (("html", "阅读报告"), ("docx", "Word"), ("md", "Markdown")) if name in requirements.export_formats]
            self.output_preview.set(f"将生成：{' + '.join(outputs)} · {'、'.join(formats) if formats else '请至少选择一种文件格式'}")
        if not downloading and requirements.needs_model:
            self.ai_settings_button.pack(side="right", padx=(8, 0))
            if settings.text_provider == "ollama":
                self.model_hint.set(f"本机 Ollama · {settings.ollama_model_name or '尚未选择模型'}，需启动本机模型服务。" + ("结合画面需要支持图片的模型。" if requirements.include_visual else ""))
            elif missing:
                self.model_hint.set(f"OpenCode API · 尚未配置，点击模型设置填写{'、'.join(missing)}。")
            else:
                content = "原文和截图" if requirements.include_visual else "原文"
                self.model_hint.set(f"OpenCode API · {settings.opencode_model_name}，联网发送{content}处理，可能产生费用。" + ("需要支持图片的模型。" if requirements.include_visual else ""))
        else:
            self.ai_settings_button.pack_forget()
            self.model_hint.set("只提取原文：无需文字模型或 API 密钥。")
        if self._busy:
            hint = "正在停止；已有进度会保留" if self._stopping else "可继续添加内容到队尾，使用本批次设置"
        elif not self.source.get().strip():
            hint = "请先粘贴视频链接" if downloading else "请先添加视频链接或音频"
        elif not downloading and not requirements.export_formats:
            hint = "请至少选择阅读报告、Word 或 Markdown 一种格式"
        elif missing:
            hint = f"请点击“模型设置”填写{'、'.join(missing)}"
        else:
            hint = ""
        self.start_hint.set(hint)
        ready = bool(self.source.get().strip()) and not self._stopping and (self._busy or downloading or (bool(requirements.export_formats) and not missing))
        self.start_button.configure(text="添加到队列" if self._busy else ("开始下载" if downloading else "开始处理"), state="normal" if ready else "disabled")

    def _start_if_ready(self) -> None:
        if "disabled" not in self.start_button.state():
            self.callbacks["start"]()

    def set_stopping(self, stopping: bool) -> None:
        self._stopping = stopping
        self._refresh_request_preview()

    def set_busy(self, busy: bool) -> None:
        if busy and not self._busy:
            self._saved_control_states = {}
            pending = [self.process_steps_card, self.download_card, self.output_card]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Checkbutton)):
                    self._saved_control_states[widget] = str(widget.cget("state"))
                    widget.configure(state="disabled")
        elif not busy and self._busy:
            for widget, state in self._saved_control_states.items():
                if widget.winfo_exists():
                    widget.configure(state=state)
            self._saved_control_states.clear()
        self._busy, self._stopping = busy, False
        for widget in [*self.mode_buttons.values(), self.settings_button]:
            widget.configure(state="disabled" if busy else "normal")
        self.task_list.set_busy(busy)
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self._refresh_request_preview()

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def toggle_log(self) -> None:
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_frame.pack(fill="x", pady=(0, 10))
            self.root.update_idletasks()
            self.content_canvas.yview_moveto(1)
        else:
            self.log_frame.pack_forget()
        self.log_button.configure(text="收起记录" if self.log_visible else "运行记录")

    def open_settings(self, category: str = "模型设置") -> None:
        if self._busy:
            return
        if self.settings_dialog is not None and not self.settings_dialog.closed:
            self.settings_dialog.tabs.select(self.settings_dialog.pages[category])
            self.settings_dialog.window.lift()
            return
        settings = self.state.collect(self._settings, validate=False)
        self.settings_dialog = SettingsDialog(self.root, settings, self.callbacks, self._save_dialog_settings, category=category)
        self.settings_dialog.window.grab_set()

    def _save_dialog_settings(self, settings: AppSettings) -> None:
        callback = self.callbacks.get("save_settings")
        if callback:
            callback(settings)
        self.load_settings(settings)

    def show_ai_settings(self) -> None:
        self.open_settings("模型设置")

    def toggle_advanced(self) -> None:
        self.open_settings()

    def refresh_advanced_visibility(self) -> None:
        self._refresh_request_preview()

    def refresh_ai_visibility(self) -> None:
        self._refresh_request_preview()

    def _enable_content_scrolling(self, widget) -> None:
        """Handle touchpad gestures before Tk's individual control handlers."""
        tags = list(widget.bindtags())
        if self._scroll_bind_tag not in tags:
            # Put page scrolling before the widget's own bindings.  Otherwise
            # entries and comboboxes can consume the touchpad gesture first.
            tags.insert(0, self._scroll_bind_tag)
            widget.bindtags(tuple(tags))
        for child in widget.winfo_children():
            self._enable_content_scrolling(child)

    def _bind_window_drag(self, widget) -> None:
        """The heading is also a drag handle; the native title bar still works."""
        widget.bind("<ButtonPress-1>", self._start_window_drag)
        widget.bind("<B1-Motion>", self._drag_window)
        for child in widget.winfo_children():
            self._bind_window_drag(child)

    def _start_window_drag(self, event) -> None:
        self._drag_origin = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y())

    def _drag_window(self, event) -> None:
        mouse_x, mouse_y, window_x, window_y = self._drag_origin
        x = window_x + event.x_root - mouse_x
        y = window_y + event.y_root - mouse_y
        self.root.geometry(f"{x:+d}{y:+d}")

    def _scroll_content(self, event) -> None:
        canvas = self.content_canvas
        if self._can_scroll_inside(event.widget, event.delta):
            return
        # Touchpads on macOS report small deltas (often only 1).  Moving a few
        # pixels immediately for each event feels responsive without jumping.
        scale = 3 if self._is_macos else 40 / 120
        pixels = int(round(-event.delta * scale))
        if pixels:
            canvas.yview_scroll(pixels, "units")
        return "break"

    def _scroll_touchpad(self, event) -> None:
        canvas = self.content_canvas
        _delta_x, delta_y = self.root.tk.call("tk::PreciseScrollDeltas", event.delta)
        if self._can_scroll_inside(event.widget, delta_y):
            return
        pixels = int(round(float(self.root.tk.call("tk::ScaleNum", -delta_y))))
        if pixels:
            canvas.yview_scroll(pixels, "units")
        return "break"

    @staticmethod
    def _can_scroll_inside(widget, delta: float) -> bool:
        """Keep long source text and task lists scrollable within the page."""
        if not isinstance(widget, (tk.Text, ttk.Treeview)):
            return False
        first, last = widget.yview()
        return (delta > 0 and first > 0) or (delta < 0 and last < 1)

    def show_input_issues(self, messages: list[str]) -> None:
        self.input_issues_text.configure(state="normal")
        self.input_issues_text.delete("1.0", "end")
        self.input_issues_text.insert("1.0", "\n\n".join(messages))
        self.input_issues_text.configure(state="disabled")
        if messages:
            self.input_issues_box.pack(fill="x", pady=(8, 0))
        else:
            self.input_issues_box.pack_forget()

    def _source_text_changed(self, _event=None) -> None:
        if not self.source_entry.edit_modified():
            return
        self.source_entry.edit_modified(False)
        if self._syncing_source:
            return
        self._syncing_source = True
        try:
            self.source.set(self.source_entry.get("1.0", "end-1c"))
        finally:
            self._syncing_source = False

    def _sync_source_text(self, *_args) -> None:
        # Input warnings describe the previous contents, not the edited text.
        self.show_input_issues([])
        if self._syncing_source:
            return
        value = self.source.get()
        if self.source_entry.get("1.0", "end-1c") == value:
            return
        self._syncing_source = True
        original_state = self.source_entry.cget("state")
        try:
            self.source_entry.configure(state="normal")
            self.source_entry.delete("1.0", "end")
            self.source_entry.insert("1.0", value)
            self.source_entry.edit_modified(False)
        finally:
            self.source_entry.configure(state=original_state)
            self._syncing_source = False

    def choose_existing(self, title: str, message: str,
                        existing_label: str = "打开已有结果",
                        regenerate_label: str = "重新生成") -> str:
        """Show explicit reuse / regenerate / cancel actions for one task or a batch."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=22, style="Card.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text=message, wraplength=500, justify="left",
                  style="CardText.TLabel").pack(anchor="w", pady=(12, 20))
        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.pack(fill="x")
        choice = "cancel"

        def finish(value: str) -> None:
            nonlocal choice
            choice = value
            dialog.destroy()

        existing = ttk.Button(actions, text=existing_label, style="Primary.TButton",
                              command=lambda: finish("existing"))
        existing.pack(side="right")
        ttk.Button(actions, text=regenerate_label, style="Secondary.TButton",
                   command=lambda: finish("regenerate")).pack(side="right", padx=(0, 8))
        ttk.Button(actions, text="取消", style="Secondary.TButton",
                   command=lambda: finish("cancel")).pack(side="left", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish("cancel"))
        dialog.bind("<Escape>", lambda _event: finish("cancel"))
        dialog.bind("<Return>", lambda _event: finish("existing") if dialog.focus_get() == existing else None)
        dialog.update_idletasks()
        width, height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        dialog.grab_set()
        existing.focus_set()
        self.root.wait_window(dialog)
        return choice
