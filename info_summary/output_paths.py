"""Output folder creation and the existing transcript/video title policies."""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path


def create_result_dir(output_root: Path, title: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        suffix = "" if counter == 1 else f"_{counter}"
        candidate = output_root / f"{title}_{stamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            # Another operation may have created the same name since our last try.
            counter += 1
        else:
            return candidate


def resolve_output_dir(value: str, project_dir: Path) -> Path | None:
    path = value.strip()
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (project_dir / resolved).resolve()
    return resolved


def sanitize_transcript_title(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:80] or f"video_{int(time.time())}"


def sanitize_video_title(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" ._")
    return (cleaned or "video")[:100]
