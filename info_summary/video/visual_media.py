from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from threading import Event
from typing import Callable

import imageio_ffmpeg

from ..cancellation import JobCancelled, check_cancelled
from ..media_commands import run_media_command
from ..models import DownloadOptions, VideoQuality
from ..storage import read_json, write_json_atomic
from .video_downloader import VideoDownloader


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FRAME_NAME = re.compile(r"^frame_(\d+)(?:_hd)?\.jpg$")
_SIGNATURE_SIZE = 48 * 27 * 3
_PLAN_VERSION = 3
_CONTENT_ROWS = 24
# Compare small areas as well as the complete image. A changed chart value or
# opened menu occupies too little space to survive a whole-image average.
# Each area averages 12 RGB pixels, rather than trusting a single noisy pixel.
_SIGNATURE_BLOCKS = tuple(
    tuple((row * 48 + column) * 3 + channel
          for row in range(top, top + 3)
          for column in range(left, left + 4)
          for channel in range(3))
    for top in range(0, 27, 3)
    for left in range(0, 48, 4)
)


class VisualMedia:
    """Owns downloaded video and deterministic, resumable frame extraction.

    Selection has no frame quota: stable repeated slides collapse to one image,
    while changed slides/scenes and periodic full-timeline coverage add images.
    The model decides whether these are sufficient in the higher-level pipeline.
    """

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.log = log or (lambda _: None)
        self.downloader = VideoDownloader(log=self.log)
        self._metadata: dict[tuple[str, int, int], tuple[float, float]] = {}

    def prepare(
        self, url: str, directory: Path, options: DownloadOptions,
        cancel_event: Event | None = None,
    ) -> tuple[Path, float]:
        self._check(cancel_event)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "video.complete.json"
        cached = self._read_json(marker)
        video = directory / "video" / "视频.mp4"
        if (cached.get("url") == url and video.is_file()
                and cached.get("size") == video.stat().st_size
                and self._positive_number(cached.get("duration"))):
            try:
                duration = self._probe(video, cancel_event)[0]
            except JobCancelled:
                raise
            except (RuntimeError, OSError, TypeError, ValueError):
                self.log("已保存的视频不完整，正在重新取得画面分析视频。")
            else:
                self.log("复用已下载的画面分析视频。")
                return video, duration

        # An interrupted merge or a file left without its completion marker is
        # never accepted as a completed cache by yt-dlp on the next run.
        marker.unlink(missing_ok=True)
        if video.exists():
            video.unlink()
        self.log("正在取得用于画面分析的视频...")
        video = self.downloader.download_to_directory(
            url, directory / "video", options, quality=VideoQuality.HD_1080,
            video_only=True, cancel_event=cancel_event,
        )
        duration = self._probe(video, cancel_event)[0]
        self._check(cancel_event)
        self._write_json(marker, {"url": url, "size": video.stat().st_size, "duration": duration})
        return video, duration

    def initial_frames(
        self, video_path: Path, duration: float, directory: Path,
        cancel_event: Event | None = None,
    ) -> list[tuple[float, Path]]:
        self._check(cancel_event)
        self._validate_duration(duration)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_frame_source(video_path, directory)
        fingerprint = self._fingerprint(video_path)
        plan_path = directory / "initial_plan.json"
        plan = self._read_json(plan_path)
        timestamps = plan.get("timestamps")
        # Existing plans may already have completed pictures and model calls;
        # reuse that progress instead of silently replacing its evidence set.
        if (plan.get("version") not in {2, _PLAN_VERSION} or plan.get("video") != fingerprint
                or not isinstance(timestamps, list) or not timestamps
                or any(not self._valid_time(t, duration) for t in timestamps)):
            self.log("正在检查全程画面变化，挑选有代表性的截图...")
            timestamps = self._scan_candidates(
                video_path, 0.0, duration, directory, cancel_event=cancel_event,
            )
            self._write_json(plan_path, {
                "version": _PLAN_VERSION, "video": fingerprint, "timestamps": timestamps,
            })
        frames = []
        for timestamp in timestamps:
            self._check(cancel_event)
            frames.append((float(timestamp), self.extract_frame(
                video_path, float(timestamp), directory, cancel_event=cancel_event,
            )))
        return frames

    def extract_frame(
        self, video_path: Path, timestamp: float, directory: Path,
        high_resolution: bool = False, cancel_event: Event | None = None,
    ) -> Path:
        self._check(cancel_event)
        duration, frame_interval = self._probe(video_path, cancel_event)
        if not self._valid_time(timestamp, duration):
            raise ValueError("截图时间必须在视频范围内。")
        timestamp = float(timestamp)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_frame_source(video_path, directory)
        suffix = "_hd" if high_resolution else ""
        output = directory / f"frame_{int(timestamp * 1000):012d}{suffix}.jpg"
        if self._complete_jpeg(output):
            return output
        temporary = output.with_suffix(".tmp.jpg")
        limit = 1920 if high_resolution else 1280
        scale = f"scale=w='min({limit},iw)':h='min({limit},ih)':force_original_aspect_ratio=decrease"
        seek = min(timestamp, max(0.0, duration - frame_interval))
        try:
            self._run([
                self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", f"{seek:.6f}", "-i", str(video_path), "-map", "0:v:0",
                "-frames:v", "1", "-vf", scale, "-an", "-sn", "-q:v", "2", str(temporary),
            ], cancel_event=cancel_event)
            if not self._complete_jpeg(temporary):
                raise RuntimeError("此时间没有取得完整画面，请选择稍早的位置重试。")
            self._check(cancel_event)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def refine_frames(
        self, video_path: Path, start: float, end: float, directory: Path,
        kind: str = "sequence", cancel_event: Event | None = None,
    ) -> list[tuple[float, Path]]:
        self._check(cancel_event)
        duration = self._probe(video_path, cancel_event)[0]
        if (not self._valid_time(start, duration) or not self._valid_time(end, duration)
                or end < start):
            raise ValueError("补看时间段必须在视频范围内，结束时间不能早于开始时间。")
        if kind not in {"sequence", "detail"}:
            raise ValueError("补看方式只能是连续画面或高清细节。")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_frame_source(video_path, directory)
        if kind == "detail":
            timestamp = (float(start) + float(end)) / 2
            return [(timestamp, self.extract_frame(
                video_path, timestamp, directory, high_resolution=True, cancel_event=cancel_event,
            ))]
        if end - start < 0.1:
            return [(float(start), self.extract_frame(
                video_path, float(start), directory, cancel_event=cancel_event,
            ))]
        # Inspect changes at a finer time scale, then fill gaps between existing
        # screenshots. Repeated requests can only expose new times; the caller
        # stops when the additional evidence no longer changes the answer.
        timestamps = self._scan_candidates(
            video_path, float(start), float(end), directory, sample_rate=4.0,
            coverage_interval=2.0, representative=False, cancel_event=cancel_event,
        )
        existing = sorted({int(match.group(1)) / 1000 for path in directory.glob("frame_*.jpg")
                           if (match := _FRAME_NAME.match(path.name))
                           and start <= int(match.group(1)) / 1000 <= end
                           and self._complete_jpeg(path)})
        boundaries = sorted({float(start), float(end), *existing})
        timestamps.extend((left + right) / 2 for left, right in zip(boundaries, boundaries[1:])
                          if right - left >= 0.2)
        timestamps.extend((float(start), float(end)))
        result = []
        for timestamp in sorted({min(float(end), max(float(start), round(t, 3))) for t in timestamps}):
            self._check(cancel_event)
            result.append((timestamp, self.extract_frame(
                video_path, timestamp, directory, cancel_event=cancel_event,
            )))
        return result

    def available_frames(
        self, video_path: Path, directory: Path, cancel_event: Event | None = None,
    ) -> list[tuple[float, Path]]:
        """Recover all completed frames, including earlier detail revisits."""
        self._check(cancel_event)
        directory = Path(directory)
        if not directory.is_dir():
            return []
        self._ensure_frame_source(video_path, directory)
        duration = self._probe(video_path, cancel_event)[0]
        frames = []
        for path in directory.glob("frame_*.jpg"):
            self._check(cancel_event)
            match = _FRAME_NAME.match(path.name)
            if match and self._complete_jpeg(path):
                timestamp = int(match.group(1)) / 1000
                if self._valid_time(timestamp, duration):
                    frames.append((timestamp, path))
        return sorted(frames, key=lambda item: (item[0], "_hd" in item[1].stem))

    def _scan_candidates(
        self, video_path: Path, start: float, end: float, directory: Path, *,
        sample_rate: float = 2.0, coverage_interval: float = 30.0,
        representative: bool = True,
        cancel_event: Event | None = None,
    ) -> list[float]:
        # Raw tiny RGB frames provide visual comparison without Pillow/OpenCV.
        # Decode the entire interval: subtitle timing is not involved here.
        with tempfile.TemporaryDirectory(prefix=".visual_scan_", dir=directory) as temporary:
            raw = Path(temporary) / "scan.rgb"
            self._run([
                self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", f"{start:.6f}", "-i", str(video_path), "-t", f"{end-start:.6f}",
                "-map", "0:v:0", "-vf", f"fps={sample_rate},scale=48:27",
                "-an", "-sn", "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw),
            ], cancel_event=cancel_event)
            chosen: list[tuple[float, bytes]] = []
            hashes: set[bytes] = set()
            previous: bytes | None = None
            last_coverage = start - coverage_interval
            with raw.open("rb") as stream:
                window = [stream.read(_SIGNATURE_SIZE) for _ in range(3)]
                index = 0
                while len(window[0]) == _SIGNATURE_SIZE:
                    self._check(cancel_event)
                    current, following = window[:2]
                    timestamp = min(end, start + (index + 0.5) / sample_rate)
                    coverage = timestamp - last_coverage >= coverage_interval
                    if representative:
                        # Initial notes need settled content, not every subtitle,
                        # moving crosshair, cursor, or intermediate edit state.
                        reference = chosen[-1][1] if chosen else None
                        change, overall, pixels = self._comparison(reference, current, representative=True)
                        following_changes = [self._comparison(current, frame, representative=True)
                                             for frame in window[1:] if len(frame) == _SIGNATURE_SIZE]
                        settled = len(following_changes) == 2
                        stable = settled and all(difference[0] < 2.0 for difference in following_changes)
                        # A chart/slide switch spans a substantial area. Keep it
                        # even if a small pointer continues moving on top of it.
                        broad = (overall >= 3.0 and pixels >= _CONTENT_ROWS * 48 // 12
                                 and settled and all(difference[1] < 2.0 for difference in following_changes))
                        candidate = (not chosen or overall >= 20.0
                                     or broad or (change >= 3.0 and pixels >= 4 and stable) or coverage)
                    else:
                        # Directed revisits deliberately keep the original,
                        # denser sensitivity, including subtitle/overlay detail.
                        change = self._difference(previous, current) if previous else math.inf
                        stable = not following or self._difference(current, following) < 2.0
                        candidate = not chosen or change >= 20.0 or (change >= 3.0 and stable) or coverage
                    if candidate:
                        # Periodic coverage compares the complete picture, so
                        # real information at the bottom is never masked forever.
                        cropped = representative and not coverage
                        signature = current[:_CONTENT_ROWS * 48 * 3] if cropped else current
                        signature_hash = hashlib.sha256(signature).digest()
                        duplicate = signature_hash in hashes or any(
                            self._difference(old, current, representative=cropped) < 2.0
                            for _, old in chosen[-128:]
                        )
                        if not duplicate:
                            chosen.append((timestamp, current))
                            hashes.add(hashlib.sha256(current).digest())
                            if representative:
                                hashes.add(hashlib.sha256(current[:_CONTENT_ROWS * 48 * 3]).digest())
                        if not representative or not duplicate or coverage:
                            last_coverage = timestamp
                    previous = current
                    window = window[1:] + [stream.read(_SIGNATURE_SIZE)]
                    index += 1
            # Very short clips can be shorter than one analysis sampling tick.
            return [min(end, round(time, 3)) for time, _ in chosen] or [float(start)]

    def _probe(self, video: Path, cancel_event: Event | None) -> tuple[float, float]:
        self._check(cancel_event)
        stat = video.stat()
        key = (str(video.resolve()), stat.st_size, stat.st_mtime_ns)
        if key in self._metadata:
            return self._metadata[key]
        probe = shutil.which("ffprobe")
        if probe:
            stdout, _ = self._run([
                probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=duration,avg_frame_rate:format=duration", "-of", "json", str(video),
            ], cancel_event=cancel_event)
            data = json.loads(stdout)
            streams = data.get("streams") or []
            if not streams:
                raise RuntimeError("下载文件没有可用的视频画面。")
            duration = streams[0].get("duration")
            try:
                valid_stream_duration = self._positive_number(float(duration))
            except (TypeError, ValueError):
                valid_stream_duration = False
            if not valid_stream_duration:
                duration = data.get("format", {}).get("duration")
            rate = str(streams[0].get("avg_frame_rate") or "25/1").split("/")
            fps = float(rate[0]) / float(rate[1]) if len(rate) == 2 and float(rate[1]) else 25.0
        else:
            _, stderr = self._run([
                self._ffmpeg(), "-hide_banner", "-nostdin", "-i", str(video),
            ], allow_failure=True, cancel_event=cancel_event)
            text = stderr.decode("utf-8", errors="replace")
            match = _DURATION.search(text)
            if not match or "Video:" not in text:
                raise RuntimeError("无法读取完整视频时长或视频画面，请重新下载后重试。")
            duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
            rate_match = re.search(r"\b(\d+(?:\.\d+)?) fps\b", text)
            fps = float(rate_match[1]) if rate_match else 25.0
        duration = float(duration)
        self._validate_duration(duration)
        result = (duration, 1.0 / fps if fps > 0 else 0.04)
        self._metadata[key] = result
        return result

    def _run(
        self, command: list[str], *, allow_failure: bool = False,
        cancel_event: Event | None = None,
    ) -> tuple[bytes, bytes]:
        returncode, stdout, stderr = run_media_command(command, cancel_event=cancel_event)
        output = (stdout, stderr)
        if returncode and not allow_failure:
            # Only local media commands reach here. Never echo a download URL,
            # request headers, cookies, or an executable command in this error.
            raise RuntimeError("视频画面处理失败：" + output[1].decode("utf-8", errors="replace").strip()[-1000:])
        return output

    @staticmethod
    def _ffmpeg() -> str:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            command = shutil.which("ffmpeg")
            if command:
                return command
            raise RuntimeError("没有找到视频处理组件，请运行 ./scripts/setup_app.sh 后重试。") from None

    @staticmethod
    def _check(cancel_event: Event | None) -> None:
        check_cancelled(cancel_event.is_set if cancel_event else None)

    @staticmethod
    def _comparison(
        left: bytes | None, right: bytes, *, representative: bool = False,
    ) -> tuple[float, float, int]:
        if left is None:
            return math.inf, math.inf, 0
        if representative:
            # Initial selection ignores the usual bottom subtitle strip, but
            # returned screenshots remain complete and revisits inspect it too.
            left, right = left[:_CONTENT_ROWS * 48 * 3], right[:_CONTENT_ROWS * 48 * 3]
        differences = [abs(a - b) for a, b in zip(left, right)]
        overall = sum(differences) / len(differences)
        local = max(sum(differences[index] for index in block) / len(block)
                    for block in _SIGNATURE_BLOCKS if block[-1] < len(differences))
        # A single changed pixel can be a mouse pointer. Small chart/menu edits
        # still count when several pixels change and the view settles.
        pixels = sum(
            sum(differences[index:index + 3]) >= 36
            for index in range(0, len(differences), 3)
        ) if representative else 0
        # Keep local changes meaningful while damping ordinary codec noise.
        return max(overall, local * 0.65), overall, pixels

    @classmethod
    def _difference(cls, left: bytes | None, right: bytes, *, representative: bool = False) -> float:
        return cls._comparison(left, right, representative=representative)[0]

    @staticmethod
    def _fingerprint(video: Path) -> dict:
        stat = video.stat()
        return {"path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _ensure_frame_source(self, video: Path, directory: Path) -> None:
        marker = directory / "frames_source.json"
        fingerprint = self._fingerprint(video)
        if self._read_json(marker) == fingerprint:
            return
        # Reusing a directory for a changed video must not return an old frame
        # merely because its timestamp happens to have the same filename.
        for path in directory.glob("frame_*.jpg"):
            if _FRAME_NAME.match(path.name) or path.name.endswith(".tmp.jpg"):
                path.unlink(missing_ok=True)
        (directory / "initial_plan.json").unlink(missing_ok=True)
        self._write_json(marker, fingerprint)

    @staticmethod
    def _positive_number(value: object) -> bool:
        try:
            return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        except OverflowError:
            return False

    @classmethod
    def _validate_duration(cls, duration: float) -> None:
        if not cls._positive_number(duration):
            raise ValueError("视频时长必须是有效的正数。")

    @staticmethod
    def _valid_time(value: object, duration: float) -> bool:
        try:
            return (isinstance(value, (float, int)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 <= value <= duration)
        except OverflowError:
            return False

    @staticmethod
    def _complete_jpeg(path: Path) -> bool:
        try:
            with path.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    return False
                stream.seek(-2, os.SEEK_END)
                return stream.read(2) == b"\xff\xd9"
        except (OSError, ValueError):
            return False

    @staticmethod
    def _read_json(path: Path) -> dict:
        data = read_json(path, dict)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        write_json_atomic(path, data)
