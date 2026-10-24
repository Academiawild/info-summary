from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from docx.image.image import Image

from info_summary.cancellation import JobCancelled
from info_summary.models import DownloadOptions, VideoQuality
from info_summary.video.video_downloader import VideoDownloader
from info_summary.video.visual_media import VisualMedia


class VisualMediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.media = VisualMedia()

    def video(self, name="source.mp4", colors=("red",), seconds=2, size="320x180", boxes=None):
        target = self.root / name
        command = [self.media._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
        for index, color in enumerate(colors):
            source = f"color=c={color}:s={size}:r=10:d={seconds}"
            if boxes:
                source += ",drawbox=" + boxes[index]
            command += ["-f", "lavfi", "-i", source]
        command += ["-filter_complex",
                    "".join(f"[{index}:v]" for index in range(len(colors)))
                    + f"concat=n={len(colors)}:v=1:a=0[v]", "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)]
        subprocess.run(command, check=True, capture_output=True)
        return target

    def sample(self, *patches):
        frame = bytearray([255] * (48 * 27 * 3))
        for left, top, width, height, shade in patches:
            for row in range(top, top + height):
                for column in range(left, left + width):
                    at = (row * 48 + column) * 3
                    frame[at:at + 3] = bytes([shade] * 3)
        return bytes(frame)

    def scan_samples(self, frames, *, representative=True):
        def decoded_samples(command, **_kwargs):
            Path(command[-1]).write_bytes(b"".join(frames))
            return b"", b""

        with patch.object(self.media, "_run", side_effect=decoded_samples):
            return self.media._scan_candidates(
                self.root / "source.mp4", 0.0, len(frames) / 2, self.root,
                representative=representative,
            )

    def test_static_slides_are_deduplicated_instead_of_filling_a_quota(self):
        video = self.video(seconds=65)
        frames = self.media.initial_frames(video, 65.0, self.root / "frames")
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0][1].is_file())

    def test_changed_scenes_across_complete_video_are_discovered(self):
        video = self.video(colors=("red", "blue", "green"), seconds=2)
        duration, _ = self.media._probe(video, None)
        self.assertAlmostEqual(duration, 6, places=2)
        frames = self.media.initial_frames(video, duration, self.root / "frames")
        self.assertEqual(len(frames), 3)
        self.assertTrue(any(4 <= timestamp <= 6 for timestamp, _ in frames))

    def test_small_local_slide_change_is_kept_despite_an_almost_identical_background(self):
        video = self.video(colors=("white", "white"), seconds=2, boxes=(
            "x=150:y=75:w=8:h=12:color=black:t=fill",
            "x=162:y=75:w=8:h=12:color=black:t=fill",
        ))
        frames = self.media.initial_frames(video, 4.0, self.root / "frames")
        self.assertEqual(len(frames), 2)
        self.assertTrue(any(timestamp >= 2.0 for timestamp, _ in frames))

    def test_local_comparison_does_not_amplify_small_pixel_noise(self):
        original = bytes([128] * (48 * 27 * 3))
        noisy = bytearray(original)
        noisy[30:33] = bytes([131, 131, 131])
        self.assertLess(self.media._difference(original, bytes(noisy)), 2.0)

    def test_changing_bottom_subtitles_do_not_trigger_one_initial_frame_per_change(self):
        frames = [self.sample((5 + index, 24, 16, 2, 0)) for index in range(12)]
        self.assertEqual(len(self.scan_samples(frames)), 1)

    def test_bottom_only_persistent_content_is_kept_by_full_picture_coverage(self):
        original = self.sample()
        changed = self.sample((8, 24, 20, 3, 0))
        timestamps = self.scan_samples([original] * 4 + [changed] * 128)
        self.assertEqual(len(timestamps), 2)
        self.assertGreaterEqual(timestamps[1], 30.0)
        self.assertLess(timestamps[1], 31.0)

    def test_isolated_cursor_and_short_edit_flash_are_ignored_only_during_initial_selection(self):
        original = self.sample()
        cursor = self.sample((18, 9, 1, 1, 0))
        self.assertEqual(len(self.scan_samples([original] * 4 + [cursor] * 4 + [original] * 4)), 1)
        flash = self.sample((15, 9, 4, 3, 0))
        frames = [original] * 4 + [flash] + [original] * 5
        self.assertEqual(len(self.scan_samples(frames)), 1)
        self.assertIn(2.25, self.scan_samples(frames, representative=False))

    def test_persistent_small_menu_is_kept_without_raising_the_local_change_threshold(self):
        original = self.sample()
        menu = self.sample((15, 9, 4, 3, 0))
        timestamps = self.scan_samples([original] * 4 + [menu] * 4 + [original] * 4)
        self.assertEqual(timestamps, [0.25, 2.25])

    def test_chart_switch_is_kept_while_a_small_pointer_keeps_moving(self):
        original = self.sample()
        frames = [original] * 4 + [self.sample(
            (10, 8, 12, 8, 200), (28 + index, 9, 1, 1, 0),
        ) for index in range(6)]
        self.assertEqual(self.scan_samples(frames), [0.25, 2.25])

    def test_high_resolution_revisit_uses_source_resolution(self):
        video = self.video(size="1920x1080", seconds=1)
        ordinary = self.media.extract_frame(video, 0.2, self.root / "frames")
        detail = self.media.refine_frames(video, 0.2, 0.2, self.root / "frames", kind="detail")
        self.assertEqual(Image.from_file(str(ordinary)).px_width, 1280)
        self.assertEqual(Image.from_file(str(detail[0][1])).px_width, 1920)

    def test_short_sequence_revisit_adds_intermediate_times(self):
        video = self.video(colors=("red", "blue"), seconds=1)
        initial = self.media.initial_frames(video, 2.0, self.root / "frames")
        refined = self.media.refine_frames(video, 0.0, 1.9, self.root / "frames")
        self.assertTrue(set(time for time, _ in refined) - set(time for time, _ in initial))
        self.assertTrue(all(0 <= time <= 1.9 for time, _ in refined))

    def test_existing_plan_and_images_resume_without_rescan(self):
        video = self.video()
        directory = self.root / "frames"
        initial = self.media.initial_frames(video, 2.0, directory)
        with patch.object(self.media, "_scan_candidates", side_effect=AssertionError("不能重复扫描")), patch.object(
            self.media, "_run", side_effect=AssertionError("不能重复抽图"),
        ):
            self.assertEqual(self.media.initial_frames(video, 2.0, directory), initial)

    def test_valid_legacy_plan_and_all_completed_revisits_resume_unchanged(self):
        video = self.video()
        directory = self.root / "frames"
        initial = self.media.initial_frames(video, 2.0, directory)
        ordinary = self.media.extract_frame(video, 1.0, directory)
        detail = self.media.extract_frame(video, 0.25, directory, high_resolution=True)
        plan_path = directory / "initial_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(plan["version"], 3)
        plan["version"] = 2
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        before = plan_path.read_bytes()
        with patch.object(self.media, "_scan_candidates", side_effect=AssertionError("不能重做有效旧计划")):
            self.assertEqual(self.media.initial_frames(video, 2.0, directory), initial)
        self.assertEqual(plan_path.read_bytes(), before)
        recovered = self.media.available_frames(video, directory)
        self.assertIn((1.0, ordinary), recovered)
        self.assertIn((0.25, detail), recovered)

    def test_missing_image_is_regenerated_from_saved_plan(self):
        video = self.video()
        directory = self.root / "frames"
        frames = self.media.initial_frames(video, 2.0, directory)
        frames[0][1].unlink()
        with patch.object(self.media, "_scan_candidates", side_effect=AssertionError("不能重复扫描")):
            repeated = self.media.initial_frames(video, 2.0, directory)
        self.assertTrue(repeated[0][1].is_file())

    def test_changed_video_does_not_reuse_images_at_the_same_timestamp(self):
        red = self.video(name="red.mp4", colors=("red",))
        blue = self.video(name="blue.mp4", colors=("blue",))
        directory = self.root / "frames"
        first_bytes = self.media.extract_frame(red, 0.25, directory).read_bytes()
        second_bytes = self.media.extract_frame(blue, 0.25, directory).read_bytes()
        self.assertNotEqual(first_bytes, second_bytes)

    def test_available_frames_recovers_successful_images_after_extraction_failure(self):
        video = self.video(colors=("red", "blue", "green"))
        directory = self.root / "frames"
        extract = self.media.extract_frame
        calls = []

        def fail_after_first(*args, **kwargs):
            calls.append(args)
            if len(calls) > 1:
                raise RuntimeError("抽图中断")
            return extract(*args, **kwargs)

        with patch.object(self.media, "extract_frame", side_effect=fail_after_first):
            with self.assertRaisesRegex(RuntimeError, "抽图中断"):
                self.media.initial_frames(video, 6.0, directory)
        recovered = self.media.available_frames(video, directory)
        self.assertEqual(len(recovered), 1)
        self.assertTrue(recovered[0][1].is_file())
        self.assertTrue((directory / "initial_plan.json").exists())

    def test_available_frames_includes_detail_revisits_and_ignores_incomplete_images(self):
        video = self.video()
        directory = self.root / "frames"
        regular = self.media.extract_frame(video, 0.25, directory)
        detail = self.media.extract_frame(video, 0.25, directory, high_resolution=True)
        (directory / "frame_000000001000.jpg").write_bytes(b"incomplete")
        recovered = self.media.available_frames(video, directory)
        self.assertEqual(recovered, [(0.25, regular), (0.25, detail)])

    def test_available_frames_does_not_load_evidence_from_another_video(self):
        red = self.video(name="red.mp4")
        blue = self.video(name="blue.mp4", colors=("blue",))
        directory = self.root / "frames"
        self.media.extract_frame(red, 0.25, directory)
        self.assertEqual(self.media.available_frames(blue, directory), [])

    def test_prepare_reuses_only_completed_download_and_probes_real_duration(self):
        source = self.video(colors=("red", "blue"), seconds=2)
        directory = self.root / "cache"

        def download(_url, target, *_args, **_kwargs):
            target.mkdir(parents=True, exist_ok=True)
            output = target / "视频.mp4"
            shutil.copyfile(source, output)
            return output

        self.media.downloader.download_to_directory = MagicMock(side_effect=download)
        first = self.media.prepare("https://example.com/video", directory, DownloadOptions())
        self.assertAlmostEqual(first[1], 4.0, places=2)
        second = self.media.prepare("https://example.com/video", directory, DownloadOptions())
        self.assertEqual(first, second)
        self.assertEqual(self.media.downloader.download_to_directory.call_count, 1)
        self.assertTrue(self.media.downloader.download_to_directory.call_args.kwargs["video_only"])
        (directory / "video.complete.json").unlink()
        self.media.prepare("https://example.com/video", directory, DownloadOptions())
        self.assertEqual(self.media.downloader.download_to_directory.call_count, 2)

    def test_visual_download_accepts_video_only_formats_without_requiring_audio(self):
        seen = []

        class FakeYDL:
            def __init__(self, options):
                self.options = options
                seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def download(self, _urls):
                Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")

        with patch("info_summary.video.video_downloader.YoutubeDL", FakeYDL):
            for video_only in (True, False):
                VideoDownloader().download_to_directory(
                    "https://example.com/video", self.root / str(video_only), DownloadOptions(),
                    quality=VideoQuality.HD_1080, video_only=video_only,
                    active_options=DownloadOptions(),
                )
        self.assertEqual(seen[0]["format"], "bestvideo*[height<=1080]/bestvideo*/best")
        self.assertNotIn("bestaudio", seen[0]["format"])
        self.assertIn("+bestaudio", seen[1]["format"])

    def test_failed_or_cancelled_prepare_does_not_write_completion_marker(self):
        for error in (RuntimeError("下载失败"), JobCancelled("任务已取消")):
            with self.subTest(error=type(error).__name__):
                self.media.downloader.download_to_directory = MagicMock(side_effect=error)
                with self.assertRaises(type(error)):
                    self.media.prepare("https://example.com/video", self.root / "cache", DownloadOptions())
                self.assertFalse((self.root / "cache" / "video.complete.json").exists())

    def test_corrupted_complete_cache_is_downloaded_again(self):
        source = self.video()
        directory = self.root / "cache"
        video_directory = directory / "video"
        video_directory.mkdir(parents=True)
        video = video_directory / "视频.mp4"
        video.write_bytes(b"corrupted video")
        (directory / "video.complete.json").write_text(json.dumps({
            "url": "https://example.com/video", "size": video.stat().st_size, "duration": 2.0,
        }), encoding="utf-8")

        def replace(_url, target, *_args, **_kwargs):
            shutil.copyfile(source, target / "视频.mp4")
            return target / "视频.mp4"

        self.media.downloader.download_to_directory = MagicMock(side_effect=replace)
        path, duration = self.media.prepare("https://example.com/video", directory, DownloadOptions())
        self.assertEqual(path.read_bytes(), source.read_bytes())
        self.assertAlmostEqual(duration, 2.0)
        self.media.downloader.download_to_directory.assert_called_once()

    def test_invalid_times_and_actions_never_run_ffmpeg(self):
        video = self.video()
        self.media._probe(video, None)
        with patch.object(self.media, "_run") as run:
            for timestamp in (-1, float("nan"), 3):
                with self.assertRaises(ValueError):
                    self.media.extract_frame(video, timestamp, self.root)
            with self.assertRaises(ValueError):
                self.media.refine_frames(video, 1, 0, self.root)
            with self.assertRaises(ValueError):
                self.media.refine_frames(video, 0, 1, self.root, kind="shell command")
        run.assert_not_called()

    def test_cancellation_stops_and_reaps_local_subprocess(self):
        cancel = threading.Event()
        processes = []
        original = subprocess.Popen

        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            processes.append(child)
            cancel.set()
            return child

        started = time.monotonic()
        with patch("info_summary.media_commands.subprocess.Popen", side_effect=launch):
            with self.assertRaises(JobCancelled):
                self.media._run([sys.executable, "-c", "import time; time.sleep(30)"], cancel_event=cancel)
        self.assertLess(time.monotonic() - started, 4)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stderr.closed)

    def test_already_cancelled_does_not_start_downloading_or_extracting(self):
        cancel = threading.Event()
        cancel.set()
        with patch("info_summary.media_commands.subprocess.Popen") as launch:
            with self.assertRaises(JobCancelled):
                self.media.extract_frame(Path("not-present.mp4"), 0, self.root, cancel_event=cancel)
            with self.assertRaises(JobCancelled):
                self.media.initial_frames(Path("not-present.mp4"), 1, self.root, cancel_event=cancel)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
