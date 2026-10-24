#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Finder may launch the app without the paths configured by a terminal shell.
export PATH="$HOME/.local/bin:$HOME/.deno/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "当前安装脚本只支持 macOS。"
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "macOS 安装脚本检查通过。"
    exit 0
fi

trap 'echo "安装没有完成。请检查上方提示，解决后再次双击 start_app.command。" >&2' ERR

PYTHON_BIN=""
# Reuse an existing working environment, including a Python not on PATH.
for candidate in .venv/bin/python python3.13 python3.12 python3.11 python3.14 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
            && "$candidate" -c 'import tkinter; tkinter.Tcl()' 2>/dev/null; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "未找到带 tkinter 界面组件的 Python 3.11 或更新版本。"
    echo "请从 https://www.python.org/downloads/macos/ 安装 Python 3.13，然后重新打开本工具。"
    echo "如果已用 Homebrew 安装 Python，请为同一版本补装 python-tk。"
    exit 1
fi

if [[ "$PYTHON_BIN" != ".venv/bin/python" ]]; then
    if [[ -d ".venv" ]]; then
        echo "正在修复项目自己的 Python 环境；设置和已生成文件会保留。"
        "$PYTHON_BIN" -m venv --clear .venv
    else
        echo "正在创建项目自己的 Python 环境……"
        "$PYTHON_BIN" -m venv .venv
    fi
fi

REQUIREMENTS_STAMP=".venv/.requirements.sha256"
REQUIREMENTS_DIGEST="$(shasum -a 256 < requirements.txt)"
REQUIREMENTS_DIGEST="${REQUIREMENTS_DIGEST%% *}"
# A failed repair must be retried the next time the app opens.
rm -f "$REQUIREMENTS_STAMP"

echo "正在安装工具所需组件，首次运行可能需要几分钟……"
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

FFMPEG_EXE="$(".venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG_EXE" ".venv/bin/ffmpeg"
printf '%s\n' "$REQUIREMENTS_DIGEST" > "$REQUIREMENTS_STAMP"

if ! command -v deno >/dev/null 2>&1; then
    echo "提示：处理 YouTube 链接需要 Deno（https://deno.com/），或在更多设置中选择已安装的 Node。"
fi
echo "提示：润色和总结默认使用 OpenCode API，请在模型设置中填写接口地址、模型名称和密钥。"
echo "只提取原文、下载视频不需要文字模型；Ollama 仅在选择本机处理时自行安装。"
echo "安装完成。以后双击 start_app.command 即可打开。"
