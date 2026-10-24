from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .macos import app_data_dir
from .storage import read_json, write_json_atomic


APP_NAME = "InfoSummary"


def default_app_data_dir() -> Path:
    return app_data_dir(APP_NAME)


def default_output_dir() -> Path:
    desktop = Path.home() / "Desktop"
    return desktop if desktop.exists() else Path.home()


@dataclass
class AppSettings:
    output_dir: str = field(default_factory=lambda: str(default_output_dir()))
    opencode_api_key: str = ""
    last_source: str = ""
    model_size: str = "turbo"
    language_hint: str = ""
    device_mode: str = "cpu"
    compute_type: str = ""
    batch_size: int = 8
    text_provider: str = "opencode"
    opencode_model_name: str = ""
    opencode_base_url: str = ""
    ollama_model_name: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    text_model_chunk_chars: int = 3500
    output_language: str = "zh-CN"
    cookies_from_browser: str = ""
    cookies_file: str = ""
    js_runtime: str = "deno"
    js_runtime_path: str = ""
    include_polish: bool = False
    include_summary: bool = False
    include_visual: bool = False
    visual_focus: str = ""
    export_html: bool = True
    export_docx: bool = False
    export_markdown: bool = False
    task_mode: str = "process"
    video_quality: str = "best"


class SettingsStore:
    def __init__(self, app_data_dir: Path | None = None) -> None:
        self.app_data_dir = app_data_dir or default_app_data_dir()
        self.path = self.app_data_dir / "settings.json"

    def load(self) -> AppSettings:
        settings = AppSettings()
        data = read_json(self.path)

        legacy_fields = {"poe_api_key", "poe_model_name", "text_model_base_url"}
        online_fields = {"opencode_api_key", "opencode_model_name", "opencode_base_url"}
        if legacy_fields.intersection(data) and not online_fields.intersection(data):
            # The old online credentials belong to a different service. Keep
            # local preferences available without sending its key or model to
            # the new provider, whose connection details are not configured yet.
            data = dict(data)
            data["text_provider"] = "opencode"
            if "text_model_base_url" in data:
                data.setdefault("ollama_base_url", data["text_model_base_url"])

        choices = {
            "text_provider": {"ollama", "opencode"},
            "output_language": {"zh-CN", "source", "en"},
            "task_mode": {"process", "download"},
            "video_quality": {"best", "1080", "720"},
            "device_mode": {"cpu", "auto"},
            "compute_type": {"", "int8", "float32"},
        }
        required_text = {"output_dir", "model_size", "ollama_base_url", "js_runtime"}
        # Reject only the damaged fields so a bad option does not discard the
        # user's working output folder, model choices, or credentials.
        for name, default in asdict(settings).items():
            value = data.get(name, default)
            if type(value) is not type(default):
                continue
            if isinstance(value, str):
                if "\x00" in value or (name in required_text and not value.strip()):
                    continue
                try:
                    value.encode("utf-8")
                except UnicodeError:
                    continue
            if type(value) is int and value <= 0:
                continue
            if name in choices and value not in choices[name]:
                continue
            setattr(settings, name, value)
        return settings

    def save(self, settings: AppSettings) -> None:
        write_json_atomic(self.path, asdict(settings))


def redact_secret(value: str, visible_suffix: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible_suffix:
        return "*" * len(value)
    return "*" * max(8, len(value) - visible_suffix) + value[-visible_suffix:]
