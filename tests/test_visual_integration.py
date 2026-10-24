"""Exercise real ffmpeg, HTTP image payloads, checkpoints and document export locally."""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
import zipfile
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import imageio_ffmpeg

from info_summary.models import ProcessingRequest, ProcessingStep, SourceType, TranscriptSegment
from info_summary.pipeline import ProcessingPipeline
from info_summary.video.visual_media import VisualMedia


class VisualIntegrationTests(unittest.TestCase):
    def test_real_frames_http_and_word_export_then_focused_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "slides.mp4"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=320x180:r=4:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=4:d=1",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(video),
            ], check=True, capture_output=True)
            payloads = []

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    payloads.append(payload)
                    system = payload["messages"][0]["content"]
                    user = payload["messages"][1]["content"]
                    if isinstance(user, list):
                        output = "截图中有演示页面，作为画面证据保存。"
                    elif "定位值得查看" in system:
                        output = json.dumps({"targets": []})
                    elif "检查视频的画面笔记" in system:
                        output = json.dumps({"sufficient": True, "new_information": True, "overview": "页面已说明。", "requests": []})
                    else:
                        output = "根据讲述和画面整理：[画面 00:00] 展示了演示页面。"
                    body = json.dumps({"choices": [{"message": {"content": output}}]}, ensure_ascii=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                media = VisualMedia()
                media.prepare = lambda *args, **kwargs: (video, 2.0)
                media.refine_frames = Mock(side_effect=AssertionError("总结够用时不应补图"))
                pipeline = ProcessingPipeline(root, checkpoint_dir=root / "checkpoints", visual_media=media)
                request = ProcessingRequest(
                    "https://vimeo.com/123456", SourceType.VIDEO_URL, output_dir=str(root / "out"),
                    include_visual=True, text_model_name="local-contract-test",
                    text_model_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    steps=(ProcessingStep.ORIGINAL, ProcessingStep.SUMMARY), export_formats=("docx", "md"),
                )
                with patch.object(pipeline, "_obtain_original", return_value=("两页演示", [TranscriptSegment(0, 1, "请看页面。")], "subtitle", "zh")) as original:
                    first = pipeline.run(request, api_key="local-test-key")
                    image_calls = sum(isinstance(item["messages"][1]["content"], list) for item in payloads)
                    focused = replace(request, visual_focus="确认页面内容是否清楚")
                    second = pipeline.run(focused, api_key="local-test-key")
                    focused_image_calls = sum(isinstance(item["messages"][1]["content"], list) for item in payloads)
                    pipeline.run(focused, api_key="local-test-key")
                self.assertFalse(first.warnings)
                self.assertFalse(second.warnings)
                self.assertGreaterEqual(image_calls, 2)
                self.assertEqual(sum(isinstance(item["messages"][1]["content"], list) for item in payloads), focused_image_calls)
                self.assertEqual(original.call_count, 1)
                self.assertNotEqual(first.result_dir, second.result_dir)
                media.refine_frames.assert_not_called()
                with zipfile.ZipFile(first.output_files["visual.docx"]) as document:
                    self.assertTrue(any(name.startswith("word/media/") for name in document.namelist()))
                self.assertIn("images/", first.output_files["visual.md"].read_text())
                self.assertNotIn("截图中", first.output_files["original.md"].read_text())
                self.assertTrue(all(path.exists() for path in first.output_files.values()))
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
