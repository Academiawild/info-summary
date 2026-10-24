"""Desktop form values and their mapping to saved settings."""
from __future__ import annotations

import tkinter as tk
from dataclasses import fields, replace

from ..settings import AppSettings


FIELD_NAMES = {
    "last_source": "source", "text_provider": "provider", "ollama_model_name": "ollama_model",
    "opencode_model_name": "opencode_model", "opencode_base_url": "opencode_url",
    "ollama_base_url": "ollama_url", "opencode_api_key": "opencode_key",
    "text_model_chunk_chars": "chunk_chars", "cookies_from_browser": "cookies_browser",
}


def positive_int(value: str, name: str) -> int:
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}必须是整数。") from exc
    if number <= 0:
        raise ValueError(f"{name}必须大于 0。")
    return number


def load_settings(target, settings: AppSettings) -> None:
    for field in fields(settings):
        getattr(target, FIELD_NAMES.get(field.name, field.name)).set(getattr(settings, field.name))


def collect_settings(target, previous: AppSettings, *, validate: bool = True) -> AppSettings:
    values = {}
    for field in fields(previous):
        value = getattr(target, FIELD_NAMES.get(field.name, field.name)).get()
        values[field.name] = value.strip() if isinstance(value, str) else value
    downloading = values["task_mode"] == "download"
    values["include_visual"] = bool(values["include_visual"] or values["visual_focus"])
    needs_ai = not downloading and (values["include_polish"] or values["include_summary"] or values["include_visual"])
    for name, label, needed in (("batch_size", "批大小", not downloading),
                                ("text_model_chunk_chars", "分块字符数", needs_ai)):
        if validate and needed:
            values[name] = positive_int(values[name], label)
        else:
            try:
                values[name] = positive_int(values[name], label) if needed else getattr(previous, name)
            except ValueError:
                values[name] = getattr(previous, name)
    for name, default in (("model_size", "turbo"), ("device_mode", "cpu"),
                          ("text_provider", "opencode"), ("ollama_base_url", "http://127.0.0.1:11434"),
                          ("output_language", "zh-CN"), ("js_runtime", "deno")):
        values[name] = values[name] or default
    return replace(previous, **values)


class FormState:
    def __init__(self, master: tk.Misc, settings: AppSettings | None = None) -> None:
        defaults = settings or AppSettings()
        for field in fields(defaults):
            value = getattr(defaults, field.name)
            variable = tk.BooleanVar(master, value=value) if type(value) is bool else tk.StringVar(master, value=value)
            setattr(self, FIELD_NAMES.get(field.name, field.name), variable)

    def load(self, settings: AppSettings) -> None:
        load_settings(self, settings)

    def collect(self, previous: AppSettings, *, validate: bool = True) -> AppSettings:
        return collect_settings(self, previous, validate=validate)
