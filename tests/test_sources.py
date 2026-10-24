from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from info_summary.models import SourceType
from info_summary.sources import (
    detect_source,
    detect_sources,
    is_supported_video_url,
    is_youtube_url,
    list_audio_files,
    parse_sources,
)


class SourceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve()

    def assert_video(self, value: str, expected: str) -> None:
        self.assertEqual(detect_source(value, self.root), (SourceType.VIDEO_URL, expected))

    def test_existing_supported_video_sites(self) -> None:
        for url in (
            "https://www.youtube.com/watch?v=abc&t=10",
            "https://youtu.be/abc",
            "https://www.bilibili.com/video/BV123",
            "https://b23.tv/abc",
            "https://v.douyin.com/abc/",
            "https://www.iesdouyin.com/share/video/123",
            "https://www.ixigua.com/123",
            "https://vimeo.com/123",
        ):
            with self.subTest(url=url):
                self.assert_video(url, url)

    def test_share_text_extracts_one_url_and_preserves_query_and_fragment(self) -> None:
        url = "https://www.youtube.com/watch?v=abc&list=PL123&t=42&title=%E4%B8%AD%E6%96%87#chapter"
        for value in (
            f"分享一个视频：{url}，一起看看。",
            f"复制链接\n「{url}」\n打开看看",
            f"推荐 ({url}).",
            f"[视频]({url})",
        ):
            with self.subTest(value=value):
                self.assert_video(value, url)

    def test_chinese_share_words_directly_after_url_are_not_part_of_url(self) -> None:
        self.assert_video(
            "6.38 复制打开抖音 https://v.douyin.com/abc/?from=share&u=123复制此链接，打开App直接观看",
            "https://v.douyin.com/abc/?from=share&u=123",
        )
        self.assert_video("分享https://b23.tv/abc复制打开", "https://b23.tv/abc")
        self.assert_video("分享b23.tv/abc复制打开", "https://b23.tv/abc")

    def test_query_punctuation_and_balanced_parentheses_are_preserved(self) -> None:
        for url in (
            "https://youtu.be/abc?token=A,+!=&option=(abc)",
            "https://youtu.be/abc?token=abc!",
            "https://youtu.be/abc?token=abc.",
            "https://youtu.be/abc?token=abc;",
        ):
            with self.subTest(url=url):
                self.assert_video(url, url)

    def test_missing_protocol_is_added_for_supported_hosts(self) -> None:
        for value in (
            "www.youtube.com/watch?v=abc&list=123",
            "youtu.be/abc?t=42",
            "b23.tv/abc",
            "www.bilibili.com/video/BV123",
            "v.douyin.com/abc/",
            "www.iesdouyin.com/share/video/123",
            "www.ixigua.com/123",
            "vimeo.com/123",
        ):
            with self.subTest(value=value):
                self.assert_video(f"推荐：{value}。", "https://" + value)

    def test_multiple_video_links_are_rejected_even_when_repeated(self) -> None:
        for value in (
            "https://youtu.be/abc\nhttps://b23.tv/xyz",
            "分享 https://youtu.be/abc，https://b23.tv/xyz。",
            "www.youtube.com/watch?v=abc b23.tv/xyz",
            "https://youtu.be/abc https://youtu.be/abc",
            "https://youtu.be/abc,https://b23.tv/xyz",
            "https://youtu.be/abc;b23.tv/xyz",
            "youtu.be/abc|b23.tv/xyz",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "请每次只粘贴一个视频链接"):
                    detect_source(value, self.root)

    def test_urls_inside_another_url_are_not_extracted(self) -> None:
        with self.assertRaisesRegex(ValueError, "无法识别视频链接"):
            detect_source("https://example.test/redirect?url=https://youtu.be/abc", self.root)

    def test_similar_domains_are_not_supported(self) -> None:
        for value in (
            "https://youtube.com.evil.test/watch?v=abc",
            "https://notyoutube.com/watch?v=abc",
            "www.youtube.com.evil.test/watch?v=abc",
            "https://youtube.com@evil.test/video",
            "https://evil.test/youtu.be/abc",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_supported_video_url(value))
                with self.assertRaises(ValueError):
                    detect_source(value, self.root)

    def test_malformed_links_report_an_input_error(self) -> None:
        for value in (
            "https://[broken",
            "https://youtu.be:wrong/abc",
            "https://youtu.be:99999/abc",
            "https:///youtu.be/abc",
            "https:/youtu.be/abc",
            "ftp://youtu.be/abc",
            "https://example.test/" + "x" * 5000,
        ):
            with self.subTest(value=value[:80]):
                with self.assertRaisesRegex(ValueError, "无法识别视频链接"):
                    detect_source(value, self.root)
        self.assertFalse(is_supported_video_url("https://[broken"))
        self.assertFalse(is_youtube_url("https://[broken"))

    def test_files_paths_quotes_and_recursive_folders_still_work(self) -> None:
        folder = self.root / "音频 文件夹"
        nested = folder / "嵌套"
        nested.mkdir(parents=True)
        first = folder / "原音频.MP3"
        second = nested / "第二段.m4a"
        first.touch()
        second.touch()
        (nested / "说明.txt").touch()
        self.assertEqual(detect_source('"音频 文件夹/原音频.MP3"', self.root), (SourceType.AUDIO_FILE, str(first)))
        self.assertEqual(detect_source(str(folder), self.root), (SourceType.AUDIO_FOLDER, str(folder)))
        self.assertEqual(list_audio_files(folder), sorted([first, second]))

    def test_valid_local_path_takes_priority_over_link_extraction(self) -> None:
        folder = self.root / "youtu.be"
        folder.mkdir()
        audio = folder / "abc.mp3"
        audio.touch()
        self.assertEqual(detect_source("youtu.be/abc.mp3", self.root), (SourceType.AUDIO_FILE, str(audio)))
        self.assertEqual(detect_source("youtu.be", self.root), (SourceType.AUDIO_FOLDER, str(folder)))

    def test_file_urls_decode_spaces_chinese_and_literal_percent(self) -> None:
        folder = self.root / "音频 文件夹"
        folder.mkdir()
        audio = folder / "完成 100% #1?.mp3"
        audio.touch()
        self.assertEqual(detect_source(audio.as_uri(), self.root), (SourceType.AUDIO_FILE, str(audio)))
        self.assertEqual(detect_source(folder.as_uri(), self.root), (SourceType.AUDIO_FOLDER, str(folder)))
        localhost_url = audio.as_uri().replace("file:///", "file://localhost/", 1)
        self.assertEqual(detect_source(localhost_url, self.root), (SourceType.AUDIO_FILE, str(audio)))

    def test_remote_file_urls_are_rejected_without_resolving_the_remote_path(self) -> None:
        for value in ("file://server/audio.mp3", "file://localhost.evil.test/audio.mp3", "file://localhost:80/audio.mp3"):
            with self.subTest(value=value):
                with patch("info_summary.sources.resolve_local_path", return_value=None) as resolve:
                    with self.assertRaisesRegex(ValueError, "只支持本机"):
                        detect_source(value, self.root)
                resolve.assert_called_once_with(value, self.root)

    def test_bad_file_url_encoding_is_rejected(self) -> None:
        for value in ("file:///tmp/%FF.mp3", "file:///tmp/%ZZ.mp3", "file:///tmp/%00.mp3", "file:relative.mp3", "file:///tmp/a.mp3?download=1", "file://[broken"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "file:// 路径"):
                    detect_source(value, self.root)

    def test_missing_or_unsupported_files_have_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "找不到输入路径"):
            detect_source("missing.mp3", self.root)
        document = self.root / "notes.txt"
        document.touch()
        with self.assertRaisesRegex(ValueError, "不是支持的音频格式"):
            detect_source(str(document), self.root)
        with self.assertRaisesRegex(ValueError, "请填写"):
            detect_source("  ", self.root)

    def test_long_share_text_does_not_raise_a_filesystem_error(self) -> None:
        self.assert_video("推荐" * 5000 + " https://b23.tv/abc ", "https://b23.tv/abc")
        with self.assertRaisesRegex(ValueError, "未找到可用"):
            detect_source("分享文字" * 5000, self.root)

    def test_batch_extracts_share_links_and_deduplicates_in_original_order(self) -> None:
        value = (
            "本周想整理的视频\n"
            "分享 https://youtu.be/abc，https://b23.tv/xyz。\n"
            "https://youtu.be/abc;b23.tv/last\n"
            "复制以上链接，打开浏览器观看"
        )
        self.assertEqual(detect_sources(value, self.root), [
            (SourceType.VIDEO_URL, "https://youtu.be/abc"),
            (SourceType.VIDEO_URL, "https://b23.tv/xyz"),
            (SourceType.VIDEO_URL, "https://b23.tv/last"),
        ])

    def test_batch_accepts_each_quoted_file_folder_and_file_url(self) -> None:
        first = self.root / "第一 段.mp3"
        first.touch()
        folder = self.root / "音频 文件夹"
        folder.mkdir()
        second = self.root / "second.wav"
        second.touch()
        value = f'"{first}"\n"{folder}"\n{second.as_uri()}\n{first}'
        self.assertEqual(detect_sources(value, self.root), [
            (SourceType.AUDIO_FILE, str(first)),
            (SourceType.AUDIO_FOLDER, str(folder)),
            (SourceType.AUDIO_FILE, str(second)),
        ])

    def test_batch_keeps_real_whole_path_priority_and_deduplicates_local_aliases(self) -> None:
        audio = self.root / "line one\nyoutu.be note.mp3"
        audio.touch()
        self.assertEqual(detect_sources(str(audio), self.root), [(SourceType.AUDIO_FILE, str(audio))])
        normal = self.root / "audio.mp3"
        normal.touch()
        self.assertEqual(detect_sources(f"audio.mp3\n{normal}", self.root), [
            (SourceType.AUDIO_FILE, str(normal)),
        ])

    def test_batch_never_silently_skips_invalid_links_beside_valid_ones(self) -> None:
        for invalid in (
            "https://example.test/video",
            "https://youtu.be:wrong/abc",
            "https:/youtu.be/broken",
            "ftp://youtu.be/abc",
            "www.youtube.com.evil.test/watch?v=abc",
        ):
            for separator in ("\n", " "):
                with self.subTest(invalid=invalid, separator=separator):
                    with self.assertRaisesRegex(ValueError, "第 [12] 行：无法识别视频链接"):
                        detect_sources(f"https://youtu.be/good{separator}{invalid}", self.root)

    def test_batch_reports_missing_or_unsupported_paths_by_line(self) -> None:
        unsupported = self.root / "notes.txt"
        unsupported.touch()
        for invalid in ("missing.mp3", "missing folder/recording.wav", "./missing", "~/missing-folder", str(unsupported)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "第 2 行："):
                    detect_sources(f"https://youtu.be/good\n{invalid}", self.root)

    def test_batch_rejects_paths_on_a_link_line_instead_of_silently_dropping_them(self) -> None:
        audio = self.root / "音频 文件.mp3"
        audio.touch()
        for value in (
            f"{audio} https://youtu.be/good",
            f"https://youtu.be/good {audio}",
            "missing.wav https://youtu.be/good",
            "https://youtu.be/good 同时处理 /missing-folder",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "第 1 行：.*单独放在一行"):
                    detect_sources(value, self.root)
        self.assertEqual(detect_sources("介绍：v1.0\n来源见 OpenAI.com\n分享 https://youtu.be/good", self.root), [
            (SourceType.VIDEO_URL, "https://youtu.be/good"),
        ])

    def test_batch_rejects_empty_and_entirely_unrecognized_input(self) -> None:
        for value in ("", "   \n ", "只有说明文案，没有链接"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    detect_sources(value, self.root)

    def test_tolerant_batch_keeps_good_links_beside_every_invalid_candidate(self) -> None:
        for invalid in (
            "https://example.test/video",
            "https://youtu.be:wrong/abc",
            "https://[broken",
            "ftp://youtu.be/abc",
            "https:/youtu.be/broken",
            "www.youtube.com.evil.test/watch?v=abc",
        ):
            for separator in ("\n", " "):
                for value in (
                    f"https://youtu.be/good{separator}{invalid}",
                    f"{invalid}{separator}https://youtu.be/good",
                ):
                    with self.subTest(value=value):
                        batch = parse_sources(value, self.root)
                        self.assertEqual(batch.sources, [(SourceType.VIDEO_URL, "https://youtu.be/good")])
                        self.assertEqual(len(batch.issues), 1)
                        self.assertIn(invalid, batch.issues[0].text)
                        self.assertIn("无法识别视频链接", batch.issues[0].message)

    def test_tolerant_batch_collects_each_error_with_original_line_numbers(self) -> None:
        batch = parse_sources(
            "\n待处理资料\nhttps://youtu.be/good https://example.test/a https://youtu.be:wrong/b\n"
            "missing.mp3\nhttps://b23.tv/second\nhttps:/youtu.be/bad",
            self.root,
        )
        self.assertEqual(batch.sources, [
            (SourceType.VIDEO_URL, "https://youtu.be/good"),
            (SourceType.VIDEO_URL, "https://b23.tv/second"),
        ])
        self.assertEqual([issue.line_number for issue in batch.issues], [3, 3, 4, 6])
        self.assertEqual([issue.text for issue in batch.issues], [
            "https://example.test/a", "https://youtu.be:wrong/b", "missing.mp3", "https:/youtu.be/bad",
        ])

    def test_tolerant_batch_deduplicates_links_and_local_path_aliases(self) -> None:
        audio = self.root / "audio.mp3"
        audio.touch()
        batch = parse_sources(
            f"https://youtu.be/good,https://example.test/bad;https://b23.tv/second\n"
            f"audio.mp3\n{audio}\n{audio.as_uri()}\nhttps://youtu.be/good",
            self.root,
        )
        self.assertEqual(batch.sources, [
            (SourceType.VIDEO_URL, "https://youtu.be/good"),
            (SourceType.VIDEO_URL, "https://b23.tv/second"),
            (SourceType.AUDIO_FILE, str(audio)),
        ])
        self.assertEqual(len(batch.issues), 1)

    def test_tolerant_batch_keeps_files_when_other_paths_fail(self) -> None:
        audio = self.root / "可用音频.mp3"
        audio.touch()
        unsupported = self.root / "video.mp4"
        unsupported.touch()
        batch = parse_sources(f"missing.mp3\n{audio}\n{unsupported}\nfile:///tmp/%00.mp3", self.root)
        self.assertEqual(batch.sources, [(SourceType.AUDIO_FILE, str(audio))])
        self.assertEqual([issue.line_number for issue in batch.issues], [1, 3, 4])
        self.assertIn("不是支持的音频格式", batch.issues[1].message)
        self.assertIn("无效字符", batch.issues[2].message)

    def test_tolerant_batch_keeps_links_and_reports_paths_mixed_on_their_line(self) -> None:
        audio = self.root / "音频 文件.mp3"
        audio.touch()
        for path in (str(audio), audio.as_uri(), "missing.wav", "file:relative.mp3", "/missing-folder"):
            for value in (f"{path} https://youtu.be/good", f"https://youtu.be/good {path}"):
                with self.subTest(value=value):
                    batch = parse_sources(value, self.root)
                    self.assertEqual(batch.sources, [(SourceType.VIDEO_URL, "https://youtu.be/good")])
                    self.assertEqual(len(batch.issues), 1)
                    self.assertIn("单独放在一行", batch.issues[0].message)

    def test_tolerant_batch_never_extracts_links_hidden_in_unsupported_urls(self) -> None:
        batch = parse_sources(
            "https://example.test/redirect?url=https://youtu.be/hidden\n"
            "https://youtube.com@evil.test/video\nhttps://b23.tv/good",
            self.root,
        )
        self.assertEqual(batch.sources, [(SourceType.VIDEO_URL, "https://b23.tv/good")])
        self.assertEqual(len(batch.issues), 2)

    def test_tolerant_batch_reports_file_uri_errors_without_remote_resolution(self) -> None:
        for invalid in (
            "file://server/audio.mp3", "file://localhost:80/audio.mp3", "file://[broken",
            "file:///tmp/%ZZ.mp3", "file:///tmp/%FF.mp3",
        ):
            with self.subTest(invalid=invalid):
                with patch("info_summary.sources.resolve_local_path", return_value=None) as resolve:
                    batch = parse_sources(f"{invalid}\nhttps://youtu.be/good", self.root)
                self.assertEqual(batch.sources, [(SourceType.VIDEO_URL, "https://youtu.be/good")])
                self.assertEqual(len(batch.issues), 1)
                self.assertTrue(all(call.args[0] != "/audio.mp3" for call in resolve.call_args_list))

    def test_tolerant_batch_returns_issues_for_empty_or_entirely_invalid_input(self) -> None:
        for value in ("", " \n ", "只有说明文案，没有链接", "https://example.test/bad", "missing.mp3", "说明" * 5000):
            with self.subTest(value=value[:80]):
                batch = parse_sources(value, self.root)
                self.assertEqual(batch.sources, [])
                self.assertTrue(batch.issues)
        batch = parse_sources("只有说明文案\nhttps://youtu.be/good\n来源见 OpenAI.com", self.root)
        self.assertEqual(batch.sources, [(SourceType.VIDEO_URL, "https://youtu.be/good")])
        self.assertEqual(batch.issues, [])


if __name__ == "__main__":
    unittest.main()
