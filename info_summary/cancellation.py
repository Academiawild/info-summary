from __future__ import annotations

from typing import Callable


class JobCancelled(RuntimeError):
    """用户主动停止任务；不得当成下载或模型失败重试。"""


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise JobCancelled("任务已取消。")
