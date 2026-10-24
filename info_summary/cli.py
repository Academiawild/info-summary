"""Non-interactive commands sharing the desktop application's services."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path

from .history import ResultHistory
from .models import ProcessingRequest, ProcessingResult, SourceType
from .requests import build_download_request, build_processing_request, validate_output
from .settings import AppSettings, SettingsStore
from .sources import detect_source


class CommandError(ValueError):
    pass


class CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CommandError(message)


def _parser() -> argparse.ArgumentParser:
    parser = CommandParser(
        prog="python -m info_summary",
        description="不带参数打开窗口；以下命令无需窗口，执行后输出 JSON 结果。",
        epilog="普通日志写入 stderr；--help 显示帮助。不会修改窗口里保存的设置。",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("process", "从视频链接、音频文件或文件夹整理文字"),
                            ("download", "下载带声音的 MP4 视频")):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument("sources", nargs="+", metavar="SOURCE", help="链接或本机路径；多个内容分别传入，含空格时加引号")
        command.add_argument("--output", metavar="PATH", help="保存位置；默认沿用窗口中保存的位置，未改过则为桌面")
        command.add_argument("--force", action="store_true", help="重新生成；默认复用同一保存位置的完整结果")
        if name == "process":
            command.add_argument("--preset", choices=("original", "summary", "polish", "all"), default="original",
                                 help="original 原文（默认）；summary 原文和总结；polish 原文和润色稿；all 全部")
            command.add_argument("--format", choices=("html", "docx", "md", "both"), default="docx", help="html 离线阅读报告；docx Word（默认）；md Markdown；both Word 和 Markdown")
            command.add_argument("--visual", action="store_true", help="结合视频画面生成画面笔记；需要支持图片的模型，仅支持视频链接")
            command.add_argument("--visual-focus", default="", metavar="TEXT", help="想重点看的画面或时间；填写后自动开启结合画面，保留已有文稿")
            command.add_argument("--provider", choices=("opencode", "ollama"), help="模型来源；默认沿用保存的设置")
            command.add_argument("--model", help="本次使用的模型名称；结合画面时需支持图片")
            command.add_argument("--base-url", help="本次使用的模型基础接口地址")
        else:
            command.add_argument("--quality", choices=("best", "1080", "720"), default="best", help="视频清晰度上限，默认 best")
    return parser


def _settings(args: argparse.Namespace) -> AppSettings:
    settings = replace(SettingsStore().load(), task_mode=args.command)
    if args.output is not None:
        settings.output_dir = args.output
    settings.opencode_api_key = os.environ.get("INFO_SUMMARY_API_KEY", settings.opencode_api_key).strip()
    if args.command == "process":
        settings.include_polish = args.preset in {"polish", "all"}
        settings.include_summary = args.preset in {"summary", "all"}
        settings.visual_focus = args.visual_focus.strip()
        settings.include_visual = args.visual or bool(settings.visual_focus)
        settings.export_docx = args.format in {"docx", "both"}
        settings.export_markdown = args.format in {"md", "both"}
        settings.export_html = args.format == "html"
        if args.provider:
            settings.text_provider = args.provider
        online = settings.text_provider == "opencode"
        if args.model is not None:
            if online:
                settings.opencode_model_name = args.model.strip()
            else:
                settings.ollama_model_name = args.model.strip()
        if args.base_url is not None:
            if online:
                settings.opencode_base_url = args.base_url.strip()
            else:
                settings.ollama_base_url = args.base_url.strip()
    else:
        settings.video_quality = args.quality
    return settings


@contextmanager
def _cancellation():
    event = threading.Event()
    previous = {}

    def stop(_signum, _frame):
        event.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
    try:
        yield event
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _error(source: str | None, message: str) -> dict:
    return {"source": source, "message": message}


def _redact(value, secret: str):
    if isinstance(value, str):
        return value.replace(secret, "[已隐藏]") if secret else value
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _result(result, *, reused: bool) -> dict:
    warnings = list(getattr(result, "warnings", []))
    files = (result.output_files if isinstance(result, ProcessingResult)
             else {"video": result.video_path})
    return {
        "source": result.source,
        "status": "partial" if warnings else "reused" if reused else "completed",
        "title": result.title,
        "result_dir": str(result.result_dir.resolve()),
        "output_files": {name: str(path.resolve()) for name, path in files.items()},
        "warnings": warnings,
    }


def _run(args, settings: AppSettings, project_dir: Path, cancelled: threading.Event) -> tuple[dict, int]:
    # CLI paths are relative to the caller, even when imported from another directory.
    caller_dir = Path.cwd()
    requests = []
    errors = []
    for source in args.sources:
        try:
            source_type, location = detect_source(source, caller_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(_error(source, str(exc)))
            continue
        if args.command == "download":
            if source_type != SourceType.VIDEO_URL:
                errors.append(_error(source, "下载视频只接受视频链接；音频请使用 process 命令。"))
                continue
            request = build_download_request(settings, location)
        else:
            if settings.include_visual and source_type != SourceType.VIDEO_URL:
                errors.append(_error(source, "“结合画面”目前只支持视频链接；音频请去掉 --visual 和 --visual-focus 后处理。"))
                continue
            # Configuration errors apply to the entire invocation, before any download.
            request = build_processing_request(replace(settings, last_source=location), caller_dir)
        if request not in requests:
            requests.append(request)
    if not requests:
        return {"status": "invalid", "results": [], "errors": errors}, 2
    validate_output(settings, caller_dir)
    requests = [replace(request, output_dir=settings.output_dir) for request in requests]

    def log(message):
        print(_redact(str(message), settings.opencode_api_key), file=sys.stderr)

    if args.command == "process":
        from .service import ProcessingService
        service = ProcessingService(project_dir, log=log)
    else:
        from .video import VideoDownloadService
        service = VideoDownloadService(log=log)
    expanded = []
    for request in requests:
        try:
            items = service.expand_requests(request) if args.command == "process" else [request]
            for item in items:
                if item not in expanded:
                    expanded.append(item)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(_error(request.source, str(exc)))
    if not expanded:
        return {"status": "invalid", "results": [], "errors": errors}, 2

    from .cancellation import JobCancelled
    history = ResultHistory()
    results = []
    for index, request in enumerate(expanded):
        source = request.source if isinstance(request, ProcessingRequest) else request.url
        try:
            if cancelled.is_set():
                raise JobCancelled("任务已取消；再次运行相同命令可复用已保存的文字进度。")
            log(f"{index + 1}/{len(expanded)}：{source}")
            existing = None if args.force else service.find_existing(request)
            # Never ignore a requested output location because an old result exists elsewhere.
            if existing is not None and existing.result_dir.resolve().parent != Path(settings.output_dir):
                existing = None
            if existing is not None:
                result = existing
            elif args.command == "process":
                result = service.run(
                    request, api_key=settings.opencode_api_key, cancel_event=cancelled,
                    force_restart=args.force,
                )
            else:
                result = service.run(request, cancel_event=cancelled)
            results.append(_result(result, reused=existing is not None))
            try:
                history.record(result)
            except OSError:
                log("结果已保存，最近结果记录暂时无法写入；请使用 JSON 中的文件位置。")
        except Exception as exc:
            # Video downloads also use RuntimeError when their cancel flag is set.
            if isinstance(exc, JobCancelled) or cancelled.is_set():
                for pending in expanded[index:]:
                    location = pending.source if isinstance(pending, ProcessingRequest) else pending.url
                    errors.append(_error(location, str(exc)))
                return {"status": "cancelled", "results": results, "errors": errors}, 130
            errors.append(_error(source, str(exc) or type(exc).__name__))
    incomplete = bool(errors) or any(item["warnings"] for item in results)
    status = ("partial" if results else "failed") if incomplete else "completed"
    return {"status": status, "results": results, "errors": errors}, 1 if incomplete else 0


def main(argv: list[str] | None = None, project_dir: Path | None = None) -> int:
    secret = ""
    try:
        args = _parser().parse_args(argv)
        # Keep third-party prints away from the single machine-readable result.
        with redirect_stdout(sys.stderr), _cancellation() as cancelled:
            settings = _settings(args)
            secret = settings.opencode_api_key
            report, code = _run(args, settings, project_dir or Path(__file__).resolve().parents[1], cancelled)
    except (CommandError, ValueError) as exc:
        report, code = {"status": "invalid", "results": [], "errors": [_error(None, str(exc))]}, 2
    except KeyboardInterrupt:
        report, code = {"status": "cancelled", "results": [], "errors": [_error(None, "任务已取消。")]}, 130
    except ImportError as exc:
        report, code = {"status": "failed", "results": [], "errors": [
            _error(None, f"缺少运行组件，请先运行 ./scripts/setup_app.sh：{exc}")
        ]}, 1
    except Exception as exc:
        report, code = {"status": "failed", "results": [], "errors": [_error(None, str(exc) or type(exc).__name__)]}, 1
    print(json.dumps(_redact(report, secret), ensure_ascii=False))
    return code
