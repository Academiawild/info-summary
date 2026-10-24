from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox

from .macos import find_ollama, open_path, require_macos, start_background_process
from .history import ResultHistory
from .models import (
    ProcessingRequest,
    ProcessingResult,
    SourceType,
    VideoDownloadRequest,
    VideoDownloadResult,
)
from .providers import OllamaProvider, OpenCodeProvider, ProviderError
from .requests import build_download_request, build_processing_request, validate_output
from .settings import AppSettings, SettingsStore, redact_secret
from .sources import SourceIssue, parse_sources
from .task_queue import TaskQueue, UnfinishedTasks
from .task_runner import TaskEvent, TaskRunner
from .ui import TaskView
from .ui.dialogs import ListChoiceDialog as _ListChoiceDialog
from .ui.form_state import collect_settings, load_settings


class InfoSummaryApp:
    def __init__(self, root: tk.Tk, project_dir: Path) -> None:
        self.root = root
        self.project_dir = project_dir
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.history = ResultHistory()
        self.events: queue.Queue[TaskEvent | tuple[str, object]] = queue.Queue()
        self.unfinished = UnfinishedTasks(log=lambda text: self.events.put(("log", text)))
        self.task_queue: TaskQueue | None = None
        self.input_issues: list[SourceIssue] = []
        self.force_restart = False
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.result_dirs: list[Path] = []
        self.reused_result_dirs: list[Path] = []
        self.retry_pending: list[tuple[str, object]] = []
        self.task_results: dict[int, Path] = {}
        self.busy = False
        self.close_when_idle = False
        callbacks = {
            "choose_audio": self._choose_audio,
            "choose_folder": self._choose_folder,
            "choose_output": self._choose_output,
            "check_provider": self._check_provider,
            "load_models": self._load_models,
            "start_ollama": self._start_ollama,
            "start": self._start,
            "cancel": self._cancel,
            "open_result": self._open_result,
            "open_folder": self._open_result_folder,
            "save_settings": self._save_settings,
            "paste_source": self._paste_source,
            "import_sources": self._import_sources,
            "clear_source": self._clear_source,
            "show_history": self._show_history,
            "retry_incomplete": self._retry_incomplete,
            "open_selected_result": self._open_selected_result,
            "remove_selected_task": self._remove_selected_task,
        }
        self.view = TaskView(root, callbacks)
        self._load_settings()
        self._restore_unfinished()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll_events)

    def _load_settings(self) -> None:
        s, v = self.settings, self.view
        # An older unconfigured installation should also open ready to extract
        # text, rather than requiring online credentials for its saved preset.
        configured = (
            all(value.strip() for value in (s.opencode_base_url, s.opencode_model_name, s.opencode_api_key))
            if s.text_provider == "opencode" else bool(s.ollama_model_name.strip())
        )
        if not configured:
            s.include_polish = s.include_summary = False
        load_settings(v, s)
        if isinstance(v, TaskView):
            v._settings = s
        v._refresh_choice_widgets()
        v.refresh_quality()
        v.refresh_mode()

    def _settings_from_view(self) -> AppSettings:
        return collect_settings(self.view, self.settings)

    def _save_settings(self, settings: AppSettings) -> None:
        self.settings_store.save(settings)
        self.settings = settings

    def _build_request(self, settings: AppSettings) -> ProcessingRequest:
        return build_processing_request(settings, self.project_dir)

    def _build_requests(self, settings: AppSettings) -> list[ProcessingRequest]:
        batch = parse_sources(settings.last_source, self.project_dir)
        self.input_issues.extend(batch.issues)
        requests = []
        for source_type, source in batch.sources:
            if (settings.include_visual or settings.visual_focus.strip()) and source_type != SourceType.VIDEO_URL:
                self.input_issues.append(SourceIssue(0, source, "“结合画面”目前只支持视频链接；音频文件或文件夹请关闭“结合画面”后处理。"))
                continue
            requests.append(self._build_request(replace(settings, last_source=source)))
        return requests

    def _validate_output(self, settings: AppSettings) -> None:
        validate_output(settings, self.project_dir)

    def _start(self) -> None:
        if self.busy:
            self._add_to_queue()
            return
        self.reused_result_dirs = []
        self.input_issues = []
        self.force_restart = False
        try:
            settings = self._settings_from_view()
            self._validate_output(settings)
            if settings.task_mode == "download":
                pending = self._prepare_video_download(settings)
            else:
                pending = self._prepare_processing(settings)
        except (ValueError, RuntimeError, OSError) as exc:
            messagebox.showerror("无法开始", str(exc))
            return
        if not pending:
            self._show_input_issues()
            return
        self._begin_batch(pending, settings)
        self._show_input_issues(clear_accepted=True)

    def _begin_batch(self, pending: list[tuple[str, object]], settings: AppSettings,
                     *, keep_results: bool = False) -> None:
        self.settings = settings
        save_warning = ""
        try:
            self.settings_store.save(settings)
        except OSError:
            save_warning = "偏好设置未能保存，本次任务仍会按当前选择运行。"
        self.retry_pending = []
        self.task_results = {}
        self.task_queue = TaskQueue(self.unfinished)
        self.task_queue.add(pending, force_restart=self.force_restart)
        self.force_restart = False
        self.view.set_tasks([self._request_label(request) for _, request in pending])
        self.view.set_retry_available(0)
        self.cancel_event.clear()
        self.busy = True
        self.view.set_busy(True)
        self.view.clear_log()
        if not keep_results:
            self.result_dirs = list(self.reused_result_dirs)
        self.view.open_button.configure(state="normal" if self.result_dirs else "disabled")
        self.view.open_folder_button.configure(state="normal" if self.result_dirs else "disabled")
        self.view.progress.set(0)
        self.view.status.set("准备下载视频" if settings.task_mode == "download" else f"准备处理 {len(pending)} 个任务")
        if save_warning:
            self.view.append_log(save_warning)
        if self.reused_result_dirs:
            self.view.append_log(f"已复用 {len(self.reused_result_dirs)} 项完整结果。")
        self.worker = threading.Thread(target=self._worker_loop, args=(self.task_queue, settings.opencode_api_key), daemon=True)
        self.worker.start()

    def _restore_unfinished(self) -> None:
        self.retry_pending = self.unfinished.pending()
        if not self.retry_pending:
            return
        self.task_queue = TaskQueue(self.unfinished)
        self.task_queue.add(self.retry_pending)
        self.task_queue.stop_waiting()
        self.view.set_tasks([self._request_label(request) for _, request in self.retry_pending])
        for index in range(len(self.retry_pending)):
            self.view.update_task(index, "已停止", "上次未完成；点击“继续未完成”可接着处理，或移除这一项。")
        self.view.set_retry_available(len(self.retry_pending))
        self.view.status.set(f"已找回 {len(self.retry_pending)} 个未完成任务，可继续处理")

    def _show_input_issues(self, *, clear_accepted: bool = False) -> None:
        if clear_accepted:
            self.view.source.set("\n".join(dict.fromkeys(issue.text for issue in self.input_issues)))
        self.view.show_input_issues([
            f"{'第 ' + str(issue.line_number) + ' 行：' if issue.line_number else ''}{issue.text}\n{issue.message}"
            for issue in self.input_issues
        ])
        if self.input_issues:
            self._log(f"{len(self.input_issues)} 处输入需要修改；已识别的有效内容可以正常处理。")

    def _add_to_queue(self) -> None:
        if self.cancel_event.is_set() or self.task_queue is None:
            return
        self.input_issues = []
        # The current batch owns its options; edits are for sources only.
        settings = replace(self.settings, last_source=self.view.source.get().strip())
        try:
            pending = (self._prepare_video_download(settings) if settings.task_mode == "download"
                       else self._prepare_processing(settings))
        except (ValueError, RuntimeError, OSError) as exc:
            messagebox.showerror("无法添加", self._safe_text(str(exc)))
            return
        added = self.task_queue.add(pending)
        self.view.append_tasks([self._request_label(item.request) for item in added])
        self._show_input_issues(clear_accepted=True)
        self._log(f"已追加 {len(added)} 项到队尾。重复内容会自动合并。")

    def _remove_selected_task(self) -> None:
        index = self.view.selected_task_index()
        if index is None or self.task_queue is None:
            return
        if self.task_queue.remove(index):
            self.view.update_task(index, "已移除", "已从待办中移除，不会处理这一项。")
            self.retry_pending = self.unfinished.pending()
            self.view.set_retry_available(len(self.retry_pending))
        else:
            self._log("这一项已开始或已结束，无法作为待办移除。")

    @staticmethod
    def _request_label(request: object) -> str:
        return request.url if isinstance(request, VideoDownloadRequest) else request.source

    def _retry_incomplete(self) -> None:
        if self.busy or not self.retry_pending:
            return
        try:
            settings = self._settings_from_view()
            self._validate_output(settings)
            pending = []
            for kind, request in self.retry_pending:
                if kind != settings.task_mode:
                    # Other task modes retain their original options. The
                    # visible options only apply to the mode being edited.
                    pending.append((kind, request))
                    continue
                if kind == "process":
                    try:
                        updated = self._build_request(replace(settings, last_source=request.source))
                    except (ValueError, OSError) as exc:
                        self._log(f"暂时不能继续 {request.source}：{exc}")
                        if not self.view.log_visible:
                            self.view.toggle_log()
                        continue
                else:
                    updated = self._download_request(settings, request.url)
                pending.append((kind, updated))
        except (ValueError, RuntimeError, OSError) as exc:
            messagebox.showerror("暂时无法继续", self._safe_text(str(exc)))
            return
        if pending:
            self.force_restart = False
            self._begin_batch(pending, settings, keep_results=True)

    def _prepare_processing(self, settings: AppSettings) -> list[tuple[str, object]]:
        requests = self._build_requests(settings)
        from .service import ProcessingService
        service = ProcessingService(self.project_dir, log=lambda text: None)
        items = []
        seen = set()
        for request in requests:
            try:
                expanded = service.expand_requests(request)
            except (OSError, RuntimeError, ValueError) as exc:
                self.input_issues.append(SourceIssue(0, request.source, str(exc)))
                continue
            for item in expanded:
                identity = (item.source_type, item.source)
                if identity not in seen:
                    seen.add(identity)
                    items.append(item)
        checked = [(item, service.find_existing(item)) for item in items]
        return self._resolve_existing(checked, "process")

    def _resolve_existing(self, checked: list[tuple[object, object]], kind: str) -> list[tuple[str, object]]:
        duplicates = [result for _, result in checked if result is not None]
        if not duplicates:
            return [(kind, item) for item, _ in checked]
        if self.busy and self.task_queue is not None:
            # Appending never interrupts the running task with a decision dialog.
            for item, result in checked:
                if result is not None:
                    self._record_history(result)
                    self.unfinished.discard(kind, item)
                    if result.result_dir not in self.result_dirs:
                        self.result_dirs.append(result.result_dir)
                    self._log(f"已有完整结果：{result.title}")
            return [(kind, item) for item, result in checked if result is None]
        new_count = len(checked) - len(duplicates)
        message = f"共 {len(checked)} 项，其中 {len(duplicates)} 项已有完整结果。"
        if new_count:
            message += f"\n复用后只处理剩余 {new_count} 项；重新生成则处理全部项目。"
        else:
            message += "\n可以打开已保存的结果，或重新生成一份。"
        answer = self.view.choose_existing(
            "发现已有结果", message,
            existing_label="复用并继续" if new_count else "打开已有结果",
            regenerate_label="全部重新生成",
        )
        if answer == "cancel":
            return []
        if answer == "existing":
            for item, result in checked:
                if result is not None:
                    self.unfinished.discard(kind, item)
            for result in duplicates:
                self._record_history(result)
            self.result_dirs = [result.result_dir for result in duplicates]
            self.reused_result_dirs = list(self.result_dirs)
            self.view.open_button.configure(state="normal")
            self.view.open_folder_button.configure(state="normal")
            if not new_count:
                self.retry_pending = self.unfinished.pending()
                self.view.set_tasks([result.title for result in duplicates])
                self.view.set_retry_available(len(self.retry_pending))
                self.task_results = {}
                for index, result in enumerate(duplicates):
                    self.task_results[index] = result.result_dir
                    self.view.update_task(index, "已完成", "已复用电脑上的结果")
                self.view.status.set(f"已找到 {len(duplicates)} 项已有结果")
                self._open_result()
            return [(kind, item) for item, result in checked if result is None]
        self.force_restart = True
        return [(kind, item) for item, _ in checked]

    def _prepare_video_download(self, settings: AppSettings) -> list[tuple[str, object]]:
        batch = parse_sources(settings.last_source, self.project_dir)
        self.input_issues.extend(batch.issues)
        sources = []
        for source_type, source in batch.sources:
            if source_type.value == "video_url":
                sources.append((source_type, source))
            else:
                self.input_issues.append(SourceIssue(0, source, "这是本机音频；请在“整理文字”模式处理。"))
        from .video import VideoDownloadService
        service = VideoDownloadService(log=lambda text: None)
        requests = [self._download_request(settings, source) for _, source in sources]
        return self._resolve_existing([(item, service.find_existing(item)) for item in requests], "download")

    @staticmethod
    def _download_request(settings: AppSettings, source: str) -> VideoDownloadRequest:
        return build_download_request(settings, source)

    def _worker_loop(self, pending, api_key: str) -> None:
        work = pending if isinstance(pending, TaskQueue) else TaskQueue()
        if not isinstance(pending, TaskQueue):
            work.add(list(pending))
        TaskRunner(self.project_dir, self.events.put, self.cancel_event).run(work, api_key)

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
                event_type, payload = (event.kind, event.payload) if isinstance(event, TaskEvent) else event
            except queue.Empty:
                break
            if event_type == "log":
                self._log(str(payload))
            elif event_type == "task_state":
                index, status, detail = payload
                self.view.update_task(index, status, self._safe_text(str(detail)))
            elif event_type == "task_result":
                index, path = payload
                self.task_results[index] = path
            elif event_type == "queue_progress":
                index, finished, total, value, message = payload
                self.view.progress.set((finished + max(0.0, min(1.0, value))) / max(1, total) * 100)
                if not self.cancel_event.is_set():
                    self.view.status.set(f"已完成 {finished}/{total} · {message}")
                    if value < 1.0:
                        self.view.update_task(index, "处理中", self._safe_text(str(message)))
            elif event_type == "result" and isinstance(payload, (ProcessingResult, VideoDownloadResult)):
                if payload.result_dir not in self.result_dirs:
                    self.result_dirs.append(payload.result_dir)
                self.view.open_button.configure(state="normal")
                self.view.open_folder_button.configure(state="normal")
                self._log(f"结果文件夹：{payload.result_dir}")
                for warning in getattr(payload, "warnings", []):
                    self._log(f"需要留意：{warning}")
                self._record_history(payload)
            elif event_type == "task_error":
                self._log(f"失败：{payload}")
            elif event_type == "settings_provider":
                self._handle_settings_response(payload)
            elif event_type == "provider_error":
                self._log(f"模型服务：{payload}")
                messagebox.showerror("模型服务", self._safe_text(str(payload)))
            elif event_type == "provider":
                messagebox.showinfo("模型服务", self._safe_text(str(payload)))
                self._log(str(payload))
            elif event_type == "models":
                provider_kind, models = payload
                self._choose_model(models, provider_kind)
            elif event_type == "idle":
                self._finish_batch(payload)
                if self.close_when_idle:
                    self.root.destroy()
                    return
        self.root.after(100, self._poll_events)

    def _safe_text(self, text: str) -> str:
        # Also redact a key edited in the form before the settings are saved.
        for key in (self.settings.opencode_api_key, self.view.opencode_key.get()):
            if isinstance(key, str) and key:
                text = text.replace(key, redact_secret(key))
        return text

    def _record_history(self, result: ProcessingResult | VideoDownloadResult) -> None:
        try:
            self.history.record(result)
        except OSError:
            self._log("结果已经保存，但最近结果记录未能更新。请通过下方按钮打开结果。")

    def _log(self, text: str) -> None:
        self.view.append_log(self._safe_text(text))

    def _finish_batch(self, summary: dict) -> None:
        if self.task_queue is not None:
            if self.cancel_event.is_set():
                for item in self.task_queue.stop_waiting():
                    self.view.update_task(item.index, "已停止", "已保存可续做的进度，点击“继续未完成”接着处理")
            summary = self.task_queue.counts()
        # An append can arrive after the worker saw an empty queue but before
        # this UI event. Start another pass on the same queue, keeping row IDs.
        if self.task_queue is not None and self.task_queue.has_waiting() and not self.cancel_event.is_set():
            self.worker = threading.Thread(target=self._worker_loop,
                                           args=(self.task_queue, self.settings.opencode_api_key), daemon=True)
            self.worker.start()
            return
        self.busy = False
        self.view.set_busy(False)
        self.retry_pending = (self.unfinished.pending() if self.task_queue is not None
                              else summary.get("retry_pending", []))
        self.view.set_retry_available(len(self.retry_pending))
        completed, partial, failed = summary["completed"], summary["partial"], summary["failed"]
        counts = [f"成功 {completed} 项"]
        if partial:
            counts.append(f"部分完成 {partial} 项")
        if failed:
            counts.append(f"失败 {failed} 项")
        if self.reused_result_dirs:
            counts.append(f"复用 {len(self.reused_result_dirs)} 项")
        if summary["cancelled"]:
            remaining = summary["total"] - completed - partial - failed
            status = f"已停止 · {'，'.join(counts)}，未完成 {remaining} 项"
        elif partial or failed:
            status = f"{'部分完成' if completed or partial else '处理失败'} · {'，'.join(counts)}，请查看运行记录"
            self.view.progress.set(100)
        else:
            status = f"已完成 · {'，'.join(counts)}，可打开结果"
            self.view.progress.set(100)
        self.view.status.set(status)
        self._log(status)
        if (partial or failed) and not self.view.log_visible:
            self.view.toggle_log()

    def _cancel(self) -> None:
        if not self.busy:
            return
        self.cancel_event.set()
        self.view.set_stopping(True)
        self.view.cancel_button.configure(state="disabled")
        self.view.status.set("正在停止…保存已有进度，在当前处理间隙退出")

    def _close(self) -> None:
        if not self.busy:
            try:
                self.settings_store.save(self._settings_from_view())
            except (OSError, ValueError, tk.TclError):
                pass
            self.root.destroy()
            return
        if messagebox.askokcancel("停止并退出", "任务仍在运行。停止后退出？\n已经生成的结果会保留，当前操作结束后窗口将自动关闭。"):
            self.close_when_idle = True
            self._cancel()

    def _selected_result_folder(self) -> Path | None:
        index = self.view.selected_task_index()
        selected = self.task_results.get(index)
        if selected is not None and selected.is_dir():
            return selected
        available = [path for path in self.result_dirs if path.is_dir()]
        if not available:
            messagebox.showinfo("结果位置", "结果文件夹已被移动或删除，请重新生成。")
            self.view.open_button.configure(state="disabled")
            self.view.open_folder_button.configure(state="disabled")
            return None
        if len(available) == 1:
            return available[0]
        dialog = _ListChoiceDialog(self.root, "选择结果", [str(path) for path in available])
        return available[dialog.result] if dialog.result is not None else None

    def _open_result(self) -> None:
        selected = self._selected_result_folder()
        if selected is not None:
            self._open_saved_result(selected)

    def _open_result_folder(self) -> None:
        selected = self._selected_result_folder()
        if selected is not None:
            self._open_folder(selected)

    def _open_saved_result(self, folder: Path) -> None:
        try:
            if not folder.is_dir():
                messagebox.showinfo("结果位置", "这个文件夹已被移动或删除，请重新生成。")
                return
            candidates = [folder / name for name in (
                "阅读报告.html", "内容总结.docx", "内容总结.md", "润色稿.docx", "润色稿.md",
                "原文稿.docx", "原文稿.md", "画面笔记.docx", "画面笔记.md",
            )]
            selected = next((path for path in candidates if path.is_file()), None)
            if selected is None:
                selected = next((path for path in sorted(folder.glob("*.mp4")) if path.is_file()), folder)
            open_path(selected)
        except OSError as exc:
            messagebox.showerror("无法打开结果", str(exc))

    def _open_folder(self, path: Path) -> None:
        try:
            if not path.is_dir():
                messagebox.showinfo("结果位置", "这个文件夹已被移动或删除，请重新生成。")
                return
            open_path(path)
        except OSError as exc:
            messagebox.showerror("无法打开结果", str(exc))

    def _open_selected_result(self) -> None:
        index = self.view.selected_task_index()
        path = self.task_results.get(index)
        if path is not None:
            self._open_saved_result(path)

    def _show_history(self) -> None:
        entries = self.history.recent()
        if not entries:
            messagebox.showinfo("最近结果", "还没有可打开的最近结果。\n本次及以后保存的结果会记录在这里，重新打开软件也能找到。")
            return
        choices = [f"{item['created_at'][:16].replace('T', ' ')}  ·  {item['title']}  ·  {item['status']}"
                   for item in entries]
        dialog = _ListChoiceDialog(self.root, "最近结果", choices)
        if dialog.result is not None:
            self._open_saved_result(Path(entries[dialog.result]["result_dir"]))

    def _current_provider(self, target=None):
        values = target if target is not None else self.view
        if values.provider.get() == "opencode":
            return OpenCodeProvider(values.opencode_key.get().strip(), values.opencode_url.get().strip())
        return OllamaProvider(values.ollama_url.get().strip() or "http://127.0.0.1:11434")

    def _provider_response(self, target, kind: str, payload: object) -> None:
        self.events.put(("settings_provider", (target, kind, payload)) if target is not None else (kind, payload))

    def _handle_settings_response(self, response) -> None:
        target, kind, payload = response
        if target.closed:
            return
        if kind == "models":
            provider_kind, models = payload
            self._choose_model(models, provider_kind, target)
        else:
            target.message.set(str(payload))
            if kind == "provider_error":
                messagebox.showerror("模型服务", str(payload), parent=target.window)

    def _check_provider(self, target=None) -> None:
        values = target if target is not None else self.view
        provider = self._current_provider(target)
        model = (values.opencode_model.get() if values.provider.get() == "opencode" else values.ollama_model.get()).strip()
        if not model:
            messagebox.showinfo("测试生成", "请先填写模型名称，或点击“读取模型”选择一个模型。")
            return
        provider.timeout_seconds = 30
        key = values.opencode_key.get().strip()
        if target is not None:
            target.message.set("正在测试文字生成，请稍候……")
        else:
            self._log("正在测试文字生成，请稍候……")
        def worker() -> None:
            try:
                provider.generate(model, "请用中文简短回答。", "请回复：连接成功。")
                self._provider_response(target, "provider", "测试成功，当前模型可以生成文字。")
            except (ProviderError, OSError, ValueError) as exc:
                text = str(exc).replace(key, redact_secret(key)) if key else str(exc)
                self._provider_response(target, "provider_error", text)
        threading.Thread(target=worker, daemon=True).start()

    def _load_models(self, target=None) -> None:
        values = target if target is not None else self.view
        provider = self._current_provider(target)
        provider_kind, key = values.provider.get(), values.opencode_key.get().strip()
        if target is not None:
            target.message.set("正在读取模型列表……")
        def worker() -> None:
            try:
                self._provider_response(target, "models", (provider_kind, provider.list_models()))
            except (ProviderError, OSError, ValueError) as exc:
                text = str(exc).replace(key, redact_secret(key)) if key else str(exc)
                self._provider_response(target, "provider_error", f"未能读取模型列表：{text}\n如果服务提供了模型名称，可以直接填写，再点击“测试生成”。")
        threading.Thread(target=worker, daemon=True).start()

    def _choose_model(self, models: list[str], provider_kind: str, target=None) -> None:
        if target is not None and target.closed:
            return
        if not models:
            messagebox.showinfo("模型列表", "没有发现可用模型。在线接口可以填写服务提供的模型名称；本机处理需要先在 Ollama 中安装模型。")
            return
        if self.busy:
            self._log("任务运行中，模型列表已读取；请在任务结束后重新选择模型。")
            return
        parent = target.window if target is not None else self.root
        dialog = _ListChoiceDialog(parent, "选择模型", models, action="使用此模型")
        if target is not None and not target.closed:
            target.window.grab_set()
        if dialog.result is not None and (target is None or not target.closed):
            values = target if target is not None else self.view
            variable = values.opencode_model if provider_kind == "opencode" else values.ollama_model
            variable.set(models[dialog.result])
            if target is not None:
                target.message.set("模型已选择，点击保存后生效。")

    def _start_ollama(self, target=None) -> None:
        executable = find_ollama()
        if not executable:
            messagebox.showerror("未找到 Ollama", "本机处理需要先安装 Ollama 和文字模型。也可以选择 OpenCode API，或先选择“只提取原文”。")
            return
        try:
            start_background_process([str(executable), "serve"])
        except OSError as exc:
            messagebox.showerror("未能启动 Ollama", str(exc))
            return
        if target is not None and not target.closed:
            target.message.set("已请求启动 Ollama。")
        else:
            self.view.append_log("已请求启动 Ollama。")

    def _choose_audio(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("音频文件", "*.mp3 *.m4a *.wav *.aac *.flac *.ogg *.opus *.wma *.m4b *.amr"), ("所有文件", "*.*")])
        if paths:
            self._append_sources("\n".join(paths))

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self._append_sources(path)

    def _append_sources(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        current = self.view.source.get().strip()
        self.view.source.set(current + "\n" + text if current else text)

    def _paste_source(self) -> None:
        try:
            self._append_sources(self.root.clipboard_get())
        except tk.TclError:
            messagebox.showinfo("粘贴内容", "剪贴板里没有文字，请先复制链接或分享文案。")

    def _clear_source(self) -> None:
        self.view.source.set("")
        self.view.show_input_issues([])

    def _import_sources(self) -> None:
        filename = filedialog.askopenfilename(title="选择链接清单", filetypes=[("文字清单", "*.txt"), ("所有文件", "*.*")])
        if not filename:
            return
        try:
            path = Path(filename)
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("清单超过 2 MB，请拆成较小的文字文件后导入。")
            content = path.read_text(encoding="utf-8-sig")
            self._append_sources(content)
        except UnicodeError:
            messagebox.showerror("无法导入清单", "请将清单存为 UTF-8 编码的 .txt 文件，也可以直接复制内容到输入框。")
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法导入清单", str(exc))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.view.output_dir.set(path)

def launch_desktop_app(project_dir: Path) -> None:
    require_macos()
    root = tk.Tk()
    InfoSummaryApp(root, project_dir)
    root.mainloop()
