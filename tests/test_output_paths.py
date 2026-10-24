from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from info_summary.output_paths import (
    create_result_dir, resolve_output_dir, sanitize_transcript_title, sanitize_video_title,
)


class OutputPathTests(unittest.TestCase):
    def test_existing_title_policies_remain_distinct(self) -> None:
        self.assertEqual(sanitize_transcript_title('  A::B\nC  '), "A_B C")
        self.assertEqual(sanitize_video_title('  A::B\nC.  '), "A__B_C")
        self.assertEqual(sanitize_transcript_title("名" * 120), "名" * 80)
        self.assertEqual(sanitize_video_title("名" * 120), "名" * 100)
        with patch("info_summary.output_paths.time.time", return_value=123):
            self.assertEqual(sanitize_transcript_title(" \n"), "video_123")
        self.assertEqual(sanitize_video_title(" ._ "), "video")

    def test_relative_output_is_based_on_the_project_and_blank_uses_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            self.assertIsNone(resolve_output_dir(" \n", project))
            self.assertEqual(resolve_output_dir("results", project), (project / "results").resolve())
            self.assertEqual(resolve_output_dir(str(project / "absolute"), project), project / "absolute")

    def test_folder_creation_skips_existing_names_and_concurrent_creators(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch("info_summary.output_paths.datetime") as clock:
            root = Path(temp)
            clock.now.return_value.strftime.return_value = "20261017_120000"
            (root / "Demo_20261017_120000").mkdir()
            (root / "Demo_20261017_120000_2").write_text("occupied", encoding="utf-8")
            with ThreadPoolExecutor(max_workers=8) as pool:
                folders = list(pool.map(lambda _: create_result_dir(root, "Demo"), range(12)))
            self.assertEqual(len(set(folders)), 12)
            self.assertTrue(all(path.is_dir() for path in folders))
            self.assertEqual({path.name for path in folders}, {
                f"Demo_20261017_120000_{number}" for number in range(3, 15)
            })


if __name__ == "__main__":
    unittest.main()
