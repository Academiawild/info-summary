from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from info_summary.audio.audio_transcriber import AudioTranscriber
from info_summary.media_commands import run_media_command
from info_summary.video.visual_media import VisualMedia


class MediaCommandTests(unittest.TestCase):
    def test_returns_exit_code_and_unchanged_output_bytes(self) -> None:
        result = run_media_command([
            sys.executable, "-c",
            "import os; os.write(1, bytes([0, 255, 13, 10])); "
            "os.write(2, bytes([254, 10])); raise SystemExit(7)",
        ])
        self.assertEqual(result, (7, b"\x00\xff\r\n", b"\xfe\n"))

    def test_audio_keeps_its_decoding_and_newline_error_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch(
            "info_summary.audio.audio_transcriber.run_media_command",
            return_value=(1, b"fallback", b"bad\xff\r\nline\r"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                AudioTranscriber(Path(temp)).run_command(["ffmpeg"])
        self.assertEqual(str(raised.exception), "bad\nline")

    def test_visual_keeps_its_error_decoding_limit_and_allow_failure(self) -> None:
        stderr = b"old output " * 200 + b"\xfffinal"
        with patch(
            "info_summary.video.visual_media.run_media_command",
            return_value=(1, b"out", stderr),
        ):
            media = VisualMedia()
            with self.assertRaises(RuntimeError) as raised:
                media._run(["ffmpeg"])
            self.assertEqual(str(raised.exception), "视频画面处理失败：" + stderr.decode("utf-8", errors="replace")[-1000:])
            self.assertEqual(media._run(["ffmpeg"], allow_failure=True), (b"out", stderr))


if __name__ == "__main__":
    unittest.main()
