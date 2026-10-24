from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / ".venv/bin").mkdir(parents=True)
        shutil.copy2(PROJECT_DIR / "start_app.command", self.root / "start_app.command")
        shutil.copy2(PROJECT_DIR / "scripts/setup_app.sh", self.root / "scripts/setup_app.sh")
        (self.root / "requirements.txt").write_text("", encoding="utf-8")
        self.requirements_stamp = self.root / ".venv/.requirements.sha256"
        self.requirements_stamp.write_text(self.requirements_digest() + "\n", encoding="utf-8")
        self.log = self.root / "commands.log"
        self.shell_env = self.root / "test-shell-env"
        # Keep every check local: never discover or invoke the test machine's
        # Python, package manager, model runner, or JavaScript runtime.
        self.shell_env.write_text('''
uname() { echo Darwin; }
command() {
    if [[ "$1" == "-v" && "$2" != ".venv/bin/python" ]]; then
        return 1
    fi
    builtin command "$@"
}
brew() { echo "unexpected-brew" >> "$COMMAND_LOG"; return 99; }
''', encoding="utf-8")
        python = self.root / ".venv/bin/python"
        python.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$COMMAND_LOG"
case "$*" in
    *sys.version_info*) [[ "${FAKE_OLD_PYTHON:-0}" != "1" ]] ;;
    *import\\ tkinter*) [[ "${FAKE_NO_TK:-0}" != "1" ]] ;;
    *importlib.util*) [[ "${FAKE_MISSING_DEPS:-0}" != "1" ]] ;;
    *get_ffmpeg_exe*)
        [[ "${FAKE_FFMPEG_EXIT:-0}" == "0" ]] || exit "$FAKE_FFMPEG_EXIT"
        echo "/tmp/fake-ffmpeg" ;;
    "-m info_summary") exit "${FAKE_APP_EXIT:-0}" ;;
    "-m pip install --upgrade pip") exit "${FAKE_PIP_UPGRADE_EXIT:-0}" ;;
    "-m pip install -r requirements.txt") exit "${FAKE_PIP_INSTALL_EXIT:-0}" ;;
    *) exit 98 ;;
esac
''', encoding="utf-8")
        python.chmod(0o755)

    def requirements_digest(self) -> str:
        return hashlib.sha256((self.root / "requirements.txt").read_bytes()).hexdigest()

    def run_script(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(self.root / script), *args], cwd=self.root.parent,
            env={**os.environ, "BASH_ENV": str(self.shell_env), "COMMAND_LOG": str(self.log), **env},
            text=True, capture_output=True, timeout=10,
        )

    def test_setup_succeeds_without_ollama_deno_or_homebrew(self) -> None:
        result = self.run_script("scripts/setup_app.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("安装完成", result.stdout)
        self.assertIn("Deno", result.stdout)
        self.assertIn("-m pip install -r requirements.txt", self.log.read_text())
        self.assertNotIn("unexpected-brew", self.log.read_text())
        self.assertTrue((self.root / ".venv/bin/ffmpeg").is_symlink())
        self.assertEqual(self.requirements_stamp.read_text().strip(), self.requirements_digest())

    def test_setup_rejects_python_before_311_or_without_tkinter(self) -> None:
        for env in ({"FAKE_OLD_PYTHON": "1"}, {"FAKE_NO_TK": "1"}):
            with self.subTest(env=env):
                result = self.run_script("scripts/setup_app.sh", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tkinter", result.stdout)
                self.assertIn("3.11", result.stdout)
                self.assertNotIn("-m pip install", self.log.read_text())

    def test_healthy_launch_does_not_reinstall_dependencies(self) -> None:
        result = self.run_script(
            "start_app.command", FAKE_PIP_UPGRADE_EXIT="23", FAKE_PIP_INSTALL_EXIT="23",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("-m info_summary", self.log.read_text())
        self.assertNotIn("-m pip", self.log.read_text())

    def test_unmarked_environment_prepares_once_then_launches_offline(self) -> None:
        self.requirements_stamp.unlink()
        result = self.run_script("start_app.command")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.log.read_text()
        self.assertLess(commands.index("-m pip install"), commands.index("-m info_summary"))
        self.assertEqual(self.requirements_stamp.read_text().strip(), self.requirements_digest())

        self.log.write_text("", encoding="utf-8")
        result = self.run_script(
            "start_app.command", FAKE_PIP_UPGRADE_EXIT="23", FAKE_PIP_INSTALL_EXIT="23",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("-m info_summary", self.log.read_text())
        self.assertNotIn("-m pip", self.log.read_text())

    def test_changed_requirements_install_before_opening(self) -> None:
        (self.root / "requirements.txt").write_text("example==2.0\n", encoding="utf-8")
        result = self.run_script("start_app.command")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.log.read_text()
        self.assertLess(commands.index("-m pip install"), commands.index("-m info_summary"))
        self.assertEqual(self.requirements_stamp.read_text().strip(), self.requirements_digest())

    def test_failed_setup_clears_success_marker_and_next_launch_retries(self) -> None:
        for failed_step in ("FAKE_PIP_UPGRADE_EXIT", "FAKE_PIP_INSTALL_EXIT", "FAKE_FFMPEG_EXIT"):
            with self.subTest(failed_step=failed_step):
                self.requirements_stamp.write_text(self.requirements_digest() + "\n", encoding="utf-8")
                self.log.write_text("", encoding="utf-8")
                result = self.run_script("scripts/setup_app.sh", **{failed_step: "23"})
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertFalse(self.requirements_stamp.exists())
                self.assertNotIn("安装完成", result.stdout)

                self.log.write_text("", encoding="utf-8")
                result = self.run_script("start_app.command")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands = self.log.read_text()
                self.assertLess(commands.index("-m pip install"), commands.index("-m info_summary"))
                self.assertEqual(self.requirements_stamp.read_text().strip(), self.requirements_digest())

    def test_failed_automatic_setup_does_not_open_app(self) -> None:
        self.requirements_stamp.unlink()
        result = self.run_script("start_app.command", FAKE_PIP_INSTALL_EXIT="23")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertFalse(self.requirements_stamp.exists())
        self.assertNotIn("-m info_summary", self.log.read_text())

    def test_incomplete_environment_runs_setup_before_opening(self) -> None:
        result = self.run_script("start_app.command", FAKE_MISSING_DEPS="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.log.read_text()
        self.assertLess(commands.index("-m pip install"), commands.index("-m info_summary"))

    def test_app_error_keeps_exit_status_and_shows_recovery_message(self) -> None:
        result = self.run_script("start_app.command", FAKE_APP_EXIT="7")
        self.assertEqual(result.returncode, 7)
        self.assertIn("未能打开工具", result.stderr)

    def test_check_mode_does_not_install_or_launch(self) -> None:
        for script in ("start_app.command", "scripts/setup_app.sh"):
            with self.subTest(script=script):
                result = self.run_script(script, "--check")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
