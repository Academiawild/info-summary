from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("当前项目只支持 macOS。")


def app_data_dir(app_name: str = "InfoSummary") -> Path:
    custom = os.environ.get("XDG_DATA_HOME")
    base = Path(custom) if custom else Path.home() / "Library" / "Application Support"
    return base / app_name


def open_path(path: Path) -> None:
    subprocess.Popen(["open", str(path)])


def find_ollama() -> Path | None:
    found = shutil.which("ollama")
    candidates = [
        Path(found) if found else None,
        Path("/opt/homebrew/bin/ollama"),
        Path("/usr/local/bin/ollama"),
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    ]
    return next((path for path in candidates if path and path.exists()), None)


def find_runtime(name: str) -> str | None:
    return shutil.which(name)


def start_background_process(command: Sequence[str]) -> None:
    subprocess.Popen(list(command), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
