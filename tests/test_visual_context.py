from __future__ import annotations

import json
import unittest
from pathlib import Path

from info_summary.models import TranscriptSegment, VisualNote
from info_summary.video.visual_context import excerpt, nearby, review_groups, split_timed_segments


class VisualContextTests(unittest.TestCase):
    def test_long_transcript_preserves_every_character_and_original_times(self):
        text = '  图表说明 "quoted"\n下一行\t ' * 30
        segment = TranscriptSegment(12.34567, 87.65432, text)
        rows = split_timed_segments([segment], 55)
        label = "[12.34567-87.65432] "
        self.assertGreater(len(rows), 1)
        self.assertTrue(all(row.startswith(label) and len(row) <= 55 for row in rows))
        self.assertEqual("".join(row[len(label):] for row in rows), text)

    def test_multiple_segments_keep_their_own_times_and_order(self):
        rows = split_timed_segments([
            TranscriptSegment(0.0, 2.0, "first"), TranscriptSegment(3.0, 5.0, "second"),
        ], 100)
        self.assertEqual(rows, ["[0.0-2.0] first", "[3.0-5.0] second"])
        self.assertEqual(split_timed_segments([], 100), [])
        with self.assertRaises(ValueError):
            split_timed_segments([], 0)

    def test_excerpt_is_bounded_and_marks_omissions(self):
        self.assertEqual(excerpt("短文字", 20), "短文字")
        value = excerpt("很长的文字" * 20, 25)
        self.assertLessEqual(len(value), 25)
        self.assertIn("已省略", value)
        self.assertEqual(excerpt("长文字", 1), "…")
        self.assertEqual(excerpt("长文字", 0), "")
        with self.assertRaises(ValueError):
            excerpt("文字", -1)

    def test_nearby_filters_by_time_and_limits_long_subtitles(self):
        segments = [
            TranscriptSegment(0, 2, "远处内容"),
            TranscriptSegment(15, 18, "附近内容" * 1000),
            TranscriptSegment(200, 210, "后面的内容"),
        ]
        text = nearby(segments, 20, limit=120)
        self.assertLessEqual(len(text), 120)
        self.assertIn("[15-18] 附近内容", text)
        self.assertIn("已省略", text)
        self.assertNotIn("远处内容", text)
        self.assertNotIn("后面的内容", text)
        self.assertEqual(nearby(segments, 100), "")

    def test_review_splits_long_notes_without_dropping_text_or_filling_with_repeated_subtitles(self):
        description = '完整图表解读 "A"\n\t\x00' * 160
        segments = [TranscriptSegment(0, 10, "邻近原文" * 300)]
        notes = [VisualNote(1.0, Path("one.jpg"), description),
                 VisualNote(2.0, Path("two.jpg"), "第二张说明")]
        groups = review_groups(notes, segments, 400)
        rows = [row for group in groups for row in group]
        self.assertGreater(len(groups), 1)
        self.assertEqual("".join(row["description"] for row in rows if row["time"] == 1.0), description)
        self.assertEqual("".join(row["description"] for row in rows if row["time"] == 2.0), "第二张说明")
        self.assertEqual(sum(bool(row["transcript"]) for row in rows), 1)
        for group in groups:
            self.assertLessEqual(len(json.dumps(group, ensure_ascii=False)), 400)
            self.assertTrue(all(set(row) == {"time", "description", "transcript"} for row in group))

    def test_review_packs_small_notes_by_content_size_instead_of_fixed_picture_count(self):
        notes = [VisualNote(float(index), Path(f"{index}.jpg"), f"画面 {index}") for index in range(30)]
        groups = review_groups(notes, [], 500)
        self.assertLess(len(groups), len(notes))
        self.assertTrue(any(len(group) > 1 for group in groups))
        self.assertEqual([row["description"] for group in groups for row in group],
                         [note.description for note in notes])
        self.assertTrue(all(len(json.dumps(group, ensure_ascii=False)) <= 500 for group in groups))

    def test_small_review_budget_and_json_escaping_still_preserve_all_descriptions(self):
        description = "\x00\n\t\\\"" * 500
        notes = [VisualNote(2.0, Path("long.jpg"), description)]
        groups = review_groups(notes, [TranscriptSegment(0, 5, "\x00" * 400)], 1)
        self.assertTrue(all(len(json.dumps(group, ensure_ascii=False)) <= 200 for group in groups))
        self.assertEqual("".join(row["description"] for group in groups for row in group), description)
        self.assertEqual(review_groups([], [], 200), [])


if __name__ == "__main__":
    unittest.main()
