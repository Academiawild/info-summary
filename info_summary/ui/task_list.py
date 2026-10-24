"""Task table presentation; task ownership stays in TaskQueue."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from .theme import COLORS


class TaskList:
    def __init__(self, view, parent) -> None:
        self.view = view
        self.root = view.root
        self.callbacks = view.callbacks
        self._busy = False
        self._retry_count = 0
        self._task_labels: list[str] = []
        self._task_details: list[str] = []
        self.tasks_visible = True
        self.tasks_heading = tk.StringVar(self.root, value="任务列表")
        self._tasks_card(parent)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.set_retry_available(self._retry_count)

    def _tasks_card(self, parent) -> None:
        self.tasks_card = ttk.Frame(parent, padding=16, style="Panel.TFrame")
        heading = ttk.Frame(self.tasks_card, style="Card.TFrame")
        heading.pack(fill="x")
        self.tasks_toggle = ttk.Button(heading, textvariable=self.tasks_heading,
                                       command=self.toggle_tasks, style="Soft.TButton")
        self.tasks_toggle.pack(side="left")
        self.retry_button = ttk.Button(heading, text="继续未完成", state="disabled",
                                       command=self.callbacks.get("retry_incomplete"), style="Compact.TButton")
        self.retry_button.pack(side="right")
        retry_hint = ttk.Label(self.tasks_card, text="继续未完成会复用已保存的原文和文字进度；运行时可添加内容到队尾，也可移除等待中的任务。",
                               style="Hint.TLabel", wraplength=580, justify="left")
        retry_hint.pack(fill="x", pady=(8, 0))
        self.tasks_card.bind("<Configure>", lambda event: retry_hint.configure(wraplength=max(220, event.width - 32)), add="+")
        self.tasks_body = ttk.Frame(self.tasks_card, style="Card.TFrame")
        self.tasks_body.pack(fill="x", pady=(10, 0))
        table = ttk.Frame(self.tasks_body, style="Card.TFrame")
        table.pack(fill="x")
        self.task_tree = ttk.Treeview(table, columns=("status",), show="tree headings",
                                      height=4, selectmode="browse", style="Tasks.Treeview")
        self.task_tree.heading("#0", text="内容")
        self.task_tree.heading("status", text="状态")
        self.task_tree.column("#0", width=380, minwidth=120, stretch=True)
        self.task_tree.column("status", width=94, minwidth=80, stretch=False, anchor="center")
        self.task_tree.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.task_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.task_tree.configure(yscrollcommand=scrollbar.set)
        for status, color in (("等待", COLORS["muted"]), ("处理中", COLORS["blue"]),
                               ("已完成", COLORS["success"]), ("部分完成", COLORS["warning"]),
                               ("失败", COLORS["warning"]), ("已停止", COLORS["muted"]),
                               ("已移除", COLORS["muted"])):
            self.task_tree.tag_configure(status, foreground=color)
        self.task_tree.bind("<<TreeviewSelect>>", self._show_task_detail)
        self.task_tree.bind("<Double-1>", self._open_task_result)
        self.task_tree.bind("<Return>", self._open_task_result)
        detail = ttk.Frame(self.tasks_body, style="Card.TFrame")
        detail.pack(fill="x", pady=(8, 0))
        self.task_detail = tk.Text(detail, height=2, width=1, wrap="word", state="disabled",
                                   bg=COLORS["card"], fg=COLORS["muted"], relief="flat",
                                   font=("Helvetica Neue", 11), highlightthickness=0)
        self.task_detail.pack(side="left", fill="x", expand=True, padx=(0, 8))
        detail_scroll = ttk.Scrollbar(detail, orient="vertical", command=self.task_detail.yview)
        detail_scroll.pack(side="left", fill="y")
        self.task_detail.configure(yscrollcommand=detail_scroll.set)
        self.open_task_button = ttk.Button(detail, text="打开结果", command=self.callbacks.get("open_selected_result"),
                                           state="disabled", style="Compact.TButton")
        self.open_task_button.pack(side="right", padx=(8, 0))
        self.remove_task_button = ttk.Button(detail, text="移除待办", command=self.callbacks.get("remove_selected_task"),
                                             state="disabled", style="Compact.TButton")
        self.remove_task_button.pack(side="right")

    def set_tasks(self, labels: list[str]) -> None:
        self.task_tree.delete(*self.task_tree.get_children())
        self._task_labels = list(labels)
        self._task_details = [""] * len(labels)
        self.set_retry_available(0)
        for index, label in enumerate(labels):
            self.task_tree.insert("", "end", iid=str(index), text=f"{index + 1}. {label}", values=("等待",), tags=("等待",))
        self.tasks_visible = True
        self.tasks_body.pack(fill="x", pady=(10, 0))
        if labels:
            self.tasks_card.pack(fill="x", pady=(0, 12), after=self.view.output_card)
            self.task_tree.selection_set("0")
        else:
            self.tasks_card.pack_forget()
        self._refresh_tasks_heading()
        self._show_task_detail()
        if labels:
            self.root.update_idletasks()
            self.view.content_canvas.yview_moveto(max(0, self.tasks_card.winfo_y() - 8) / max(1, self.view.form.winfo_height()))

    def update_task(self, index: int, status: str, detail: str = "") -> None:
        if not 0 <= index < len(self._task_labels):
            return
        self._task_details[index] = detail
        self.task_tree.item(str(index), values=(status,), tags=(status,))
        self._refresh_tasks_heading()
        if self.selected_task_index() == index:
            self._show_task_detail()

    def append_tasks(self, labels: list[str]) -> None:
        for label in labels:
            index = len(self._task_labels)
            self._task_labels.append(label)
            self._task_details.append("")
            self.task_tree.insert("", "end", iid=str(index), text=f"{index + 1}. {label}", values=("等待",), tags=("等待",))
        self._refresh_tasks_heading()

    def selected_task_index(self) -> int | None:
        selected = self.task_tree.selection()
        return int(selected[0]) if selected else None

    def set_retry_available(self, count: int) -> None:
        self._retry_count = max(0, count)
        self.retry_button.configure(
            text=f"继续未完成（{self._retry_count}）" if self._retry_count else "继续未完成",
            state="normal" if self._retry_count and not self._busy else "disabled",
        )

    def toggle_tasks(self) -> None:
        self.tasks_visible = not self.tasks_visible
        if self.tasks_visible:
            self.tasks_body.pack(fill="x", pady=(10, 0))
        else:
            self.tasks_body.pack_forget()
        self._refresh_tasks_heading()

    def _refresh_tasks_heading(self) -> None:
        statuses = [self.task_tree.set(item, "status") for item in self.task_tree.get_children()]
        completed = statuses.count("已完成")
        total = sum(status != "已移除" for status in statuses)
        self.tasks_heading.set(f"任务列表 · 已完成 {completed}/{total}  {'⌃' if self.tasks_visible else '⌄'}")

    def _show_task_detail(self, _event=None) -> None:
        index = self.selected_task_index()
        self.task_detail.configure(state="normal")
        self.task_detail.delete("1.0", "end")
        if index is not None:
            detail = self._task_details[index] or "选择任务查看进度；完成后可双击打开结果。"
            self.task_detail.insert("1.0", f"{self._task_labels[index]}\n{detail}")
        self.task_detail.configure(state="disabled")
        status = self.task_tree.set(str(index), "status") if index is not None else ""
        self.open_task_button.configure(state="normal" if status in ("已完成", "部分完成") else "disabled")
        self.remove_task_button.configure(state="normal" if status in ("等待", "已停止") else "disabled")

    def _open_task_result(self, event=None) -> None:
        if event is not None and event.type == tk.EventType.ButtonPress:
            item = self.task_tree.identify_row(event.y)
            if not item:
                return
            self.task_tree.selection_set(item)
            self._show_task_detail()
        if "disabled" not in self.open_task_button.state():
            callback = self.callbacks.get("open_selected_result")
            if callback:
                callback()

