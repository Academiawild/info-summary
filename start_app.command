#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PATH="$PROJECT_DIR/.venv/bin:$HOME/.local/bin:$HOME/.deno/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

show_failure() {
    echo "未能打开工具，请根据上方提示修复后重试。" >&2
    if [[ -t 0 ]]; then
        read -r -p "按回车键关闭这个窗口……" || true
    fi
}
trap show_failure ERR

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "当前项目只支持 macOS。"
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "macOS 启动脚本检查通过。"
    exit 0
fi

REQUIREMENTS_STAMP=".venv/.requirements.sha256"
REQUIREMENTS_DIGEST="$(shasum -a 256 < requirements.txt)"
REQUIREMENTS_DIGEST="${REQUIREMENTS_DIGEST%% *}"

if [[ ! -x ".venv/bin/python" ]] \
    || [[ ! -f "$REQUIREMENTS_STAMP" ]] \
    || [[ "$(cat "$REQUIREMENTS_STAMP")" != "$REQUIREMENTS_DIGEST" ]] \
    || ! ".venv/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
    || ! ".venv/bin/python" -c 'import tkinter; tkinter.Tcl()' 2>/dev/null \
    || ! ".venv/bin/python" -c 'import importlib.util; import sys; sys.exit(0 if all(importlib.util.find_spec(name) for name in ("yt_dlp", "docx", "faster_whisper", "imageio_ffmpeg", "opencc", "webvtt", "httpx")) else 1)' 2>/dev/null; then
    echo "正在准备运行环境……"
    ./scripts/setup_app.sh
fi

".venv/bin/python" -m info_summary
