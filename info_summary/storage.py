"""Small, shared JSON file operations; callers own validation and warnings."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar


Container = TypeVar("Container", dict, list)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def read_json(path: Path, default_factory: Callable[[], Container] = dict) -> Container:
    """Ignore damaged files and return a fresh container of the expected type."""
    fallback = default_factory()
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, UnicodeError, RecursionError):
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


def write_json_atomic(path: Path, payload: object) -> None:
    """Replace a file only after its private, uniquely named copy is complete."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        # NamedTemporaryFile creates mode 0600 before any data is written.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix="." + path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def update_json(
    path: Path, default_factory: Callable[[], Container], mutate: Callable[[Container], None],
) -> Container:
    """Serialize metadata read/modify/write across threads and local processes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        value = read_json(path, default_factory)
        mutate(value)
        write_json_atomic(path, value)
        return value
    finally:
        # Keep the lock file: unlinking it would allow simultaneous locks on
        # different inodes while another process is still waiting on this one.
        os.close(descriptor)
