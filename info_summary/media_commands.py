"""Run local media tools with one cancellation and process-cleanup policy."""
from __future__ import annotations

import subprocess
from threading import Event

from .cancellation import check_cancelled


def run_media_command(
    command: list[str], *, cancel_event: Event | None = None,
) -> tuple[int, bytes, bytes]:
    """Return raw output; callers own decoding and user-facing error messages."""
    cancelled = cancel_event.is_set if cancel_event else None
    check_cancelled(cancelled)
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        while True:
            check_cancelled(cancelled)
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        check_cancelled(cancelled)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        # Cancellation may arrive after exit but before pipes are collected.
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    return process.returncode, stdout, stderr
