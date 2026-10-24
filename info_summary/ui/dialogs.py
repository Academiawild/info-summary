"""Small desktop dialogs, including isolated drafts for saved settings."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..settings import AppSettings
from .form_state import FormState


class ListChoiceDialog(simpledialog.Dialog):
    def __init__(self, parent, title: str, choices: list[str], *, action: str = "打开") -> None:
        self.choices, self.action = choices, action
        super().__init__(parent, title)

    def body(self, master):
        frame = ttk.Frame(master)
        frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(frame, width=68, height=max(1, min(10, len(self.choices))), exportselection=False)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.listbox.pack(fill="both", expand=True)
        for choice in self.choices:
            self.listbox.insert("end", choice)
        self.listbox.selection_set(0)
        self.listbox.bind("<Double-Button-1>", self.ok)
        return self.listbox

    def buttonbox(self):
        box = ttk.Frame(self, padding=10)
        box.pack(fill="x")
        ttk.Button(box, text=self.action, command=self.ok).pack(side="right")
        ttk.Button(box, text="取消", command=self.cancel).pack(side="right", padx=(0, 8))
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def validate(self) -> bool:
        return bool(self.listbox.curselection())

    def apply(self) -> None:
        self.result = self.listbox.curselection()[0]


class SettingsDialog:
    def __init__(self, parent, settings: AppSettings, callbacks: dict, on_save,
                 *, category: str = "模型设置") -> None:
        self.previous = settings
        self.callbacks = callbacks
        self.on_save = on_save
        self.closed = False
        self.window = tk.Toplevel(parent)
        self.window.title("设置")
        height = min(610, int(parent.winfo_screenheight() * .8))
        self.window.geometry(f"640x{height}")
        self.window.minsize(520, 460)
        self.window.transient(parent)
        self.state = FormState(self.window, settings)
        self.message = tk.StringVar(self.window, value="修改后点击保存；取消不会改变当前设置。")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        actions = ttk.Frame(self.window, padding=12)
        actions.pack(side="bottom", fill="x")
        ttk.Button(actions, text="保存", command=self.save, style="Primary.TButton").pack(side="right")
        ttk.Button(actions, text="取消", command=self.close).pack(side="right", padx=8)
        message_label = ttk.Label(actions, textvariable=self.message, wraplength=350)
        message_label.pack(side="left", fill="x", expand=True)
        actions.bind("<Configure>", lambda event: message_label.configure(wraplength=max(140, event.width - 230)))
        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)
        self.pages, self.page_content, self.canvases = {}, {}, {}
        for name in ("模型设置", "语音识别", "网站访问"):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            self.pages[name] = frame
            canvas = tk.Canvas(frame, highlightthickness=0, yscrollincrement=1)
            scroll = ttk.Scrollbar(frame, command=canvas.yview)
            scroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            canvas.configure(yscrollcommand=scroll.set)
            content = ttk.Frame(canvas, padding=16)
            window = canvas.create_window((0, 0), window=content, anchor="nw")
            content.bind("<Configure>", lambda _event, view=canvas: view.configure(scrollregion=view.bbox("all")))
            canvas.bind("<Configure>", lambda event, view=canvas, item=window: view.itemconfigure(item, width=event.width))
            self.canvases[str(frame)], self.page_content[name] = canvas, content
        self._model_page(self.page_content["模型设置"])
        self._fields(self.page_content["语音识别"], (
            ("语音模型", "model_size", ("turbo", "tiny", "base", "small", "medium", "large-v3")),
            ("语言提示（可留空）", "language_hint", ()),
            ("运行设备", "device_mode", {"cpu": "CPU（通用）", "auto": "自动选择"}),
            ("计算方式", "compute_type", {"": "默认（推荐）", "int8": "省内存（int8）", "float32": "标准精度（float32）"}),
            ("每批识别片段数", "batch_size", ()),
        ))
        website = self.page_content["网站访问"]
        self._fields(website, (
            ("cookies.txt", "cookies_file", ()), ("浏览器 cookies", "cookies_browser", ()),
            ("YouTube 运行工具", "js_runtime", {"deno": "Deno（推荐）", "node": "Node.js"}),
            ("工具路径（可留空）", "js_runtime_path", ()),
        ))
        ttk.Button(website, text="选择 cookies.txt", command=self.choose_cookies).pack(anchor="w", pady=12)
        self.tabs.select(self.pages[category])
        self._scroll_tag = f"InfoSummarySettingsScroll{id(self)}"
        self.window.bind_class(self._scroll_tag, "<MouseWheel>", self._scroll)
        try:
            self.window.bind_class(self._scroll_tag, "<TouchpadScroll>", self._scroll_touchpad)
        except tk.TclError:
            pass
        for content in self.page_content.values():
            self._bind_scroll(content)

    def _bind_scroll(self, widget) -> None:
        widget.bindtags((self._scroll_tag, *widget.bindtags()))
        for child in widget.winfo_children():
            self._bind_scroll(child)

    def _scroll(self, event) -> str:
        macos = self.window.tk.call("tk", "windowingsystem") == "aqua"
        pixels = int(round(-event.delta * (3 if macos else 40 / 120)))
        self.canvases[self.tabs.select()].yview_scroll(pixels, "units")
        return "break"

    def _scroll_touchpad(self, event) -> str:
        _, delta = self.window.tk.call("tk::PreciseScrollDeltas", event.delta)
        pixels = int(round(float(self.window.tk.call("tk::ScaleNum", -delta))))
        self.canvases[self.tabs.select()].yview_scroll(pixels, "units")
        return "break"

    def __getattr__(self, name):
        state = self.__dict__.get("state")
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        raise AttributeError(name)

    def _fields(self, parent, fields) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        for row, (label, name, choices) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=7)
            variable = getattr(self.state, name)
            if isinstance(choices, dict):
                display = tk.StringVar(self.window, value=choices.get(variable.get(), variable.get()))
                variable.trace_add("write", lambda *_args, var=variable, text=display, labels=choices:
                                   text.set(labels.get(var.get(), var.get())))
                widget = ttk.Combobox(grid, textvariable=display, values=tuple(choices.values()), state="readonly")
                stored = {label: value for value, label in choices.items()}
                widget.bind("<<ComboboxSelected>>", lambda event, var=variable, values=stored: var.set(values[event.widget.get()]))
            elif choices:
                widget = ttk.Combobox(grid, textvariable=variable, values=choices)
            else:
                widget = ttk.Entry(grid, textvariable=variable, show="*" if name == "opencode_key" else "")
            widget.grid(row=row, column=1, sticky="ew", pady=7)

    def _model_page(self, parent) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 10))
        self.provider_buttons = {}
        for value, label in (("opencode", "OpenCode API"), ("ollama", "本机 Ollama")):
            button = ttk.Button(row, text=label, command=lambda selected=value: self.provider.set(selected))
            button.pack(side="left", padx=(0, 8))
            self.provider_buttons[value] = button
        self.provider_container = ttk.Frame(parent)
        self.provider_container.pack(fill="x")
        self.opencode_fields = ttk.Frame(self.provider_container)
        self._fields(self.opencode_fields, (("接口地址", "opencode_url", ()), ("模型名称", "opencode_model", ()), ("API 密钥", "opencode_key", ())))
        self.ollama_fields = ttk.Frame(self.provider_container)
        self._fields(self.ollama_fields, (("本机地址", "ollama_url", ()), ("模型名称", "ollama_model", ())))
        self._fields(parent, (
            ("解读、润色和总结语言", "output_language", {"zh-CN": "简体中文", "source": "跟随原文", "en": "英文"}),
            ("每段文字字数", "chunk_chars", ()),
        ))
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=12)
        for label, action in (("测试生成", "check_provider"), ("读取模型", "load_models"), ("启动 Ollama", "start_ollama")):
            button = ttk.Button(actions, text=label, command=lambda name=action: self.invoke(name))
            button.pack(side="left", padx=(0, 8))
            if action == "start_ollama":
                self.start_ollama_button = button
        hint = ttk.Label(parent, text="在线模型会发送文字，结合画面还会发送截图，可能产生费用。\n画面功能需要支持图片的模型；只提取原文无需填写模型。", wraplength=540)
        hint.pack(anchor="w", pady=4)
        parent.bind("<Configure>", lambda event: hint.configure(wraplength=max(180, event.width - 32)), add="+")
        self.provider.trace_add("write", lambda *_args: self.refresh_provider())
        self.refresh_provider()

    def refresh_provider(self) -> None:
        self.opencode_fields.pack_forget()
        self.ollama_fields.pack_forget()
        target = self.ollama_fields if self.provider.get() == "ollama" else self.opencode_fields
        target.pack(fill="x")
        for name, button in self.provider_buttons.items():
            button.configure(style="Selected.TButton" if name == self.provider.get() else "Secondary.TButton")
        if self.provider.get() == "ollama":
            self.start_ollama_button.pack(side="left", padx=(0, 8))
        else:
            self.start_ollama_button.pack_forget()

    def invoke(self, name: str) -> None:
        callback = self.callbacks.get(name)
        if callback and not self.closed:
            callback(self)

    def choose_cookies(self) -> None:
        path = filedialog.askopenfilename(parent=self.window, filetypes=[("cookies.txt", "*.txt")])
        if path:
            self.cookies_file.set(path)

    def save(self) -> None:
        try:
            settings = self.state.collect(self.previous)
            self.on_save(settings)
        except (ValueError, OSError) as exc:
            messagebox.showerror("无法保存设置", str(exc), parent=self.window)
            return
        self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.window.unbind_class(self._scroll_tag, "<MouseWheel>")
            try:
                self.window.unbind_class(self._scroll_tag, "<TouchpadScroll>")
            except tk.TclError:
                pass
            self.window.destroy()
