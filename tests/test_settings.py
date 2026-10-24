from __future__ import annotations

import json
import stat
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from info_summary.settings import AppSettings, SettingsStore


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SettingsStore(Path(self.temp.name))

    def test_missing_damaged_or_non_object_files_use_defaults(self) -> None:
        defaults = AppSettings()
        self.assertEqual(self.store.load(), defaults)
        for content in (b"{", b"\xff", b"[]", b"null", b"12", b'"text"', b"true", b'{"batch_size":' + b"9" * 5000 + b"}"):
            with self.subTest(content=content):
                self.store.path.write_bytes(content)
                self.assertEqual(self.store.load(), defaults)

    def test_invalid_field_types_do_not_discard_valid_preferences(self) -> None:
        self.store.path.write_text(json.dumps({
            "last_source": "https://example.com/video",
            "opencode_api_key": "saved-test-key",
            "include_summary": "false",
            "export_docx": 0,
            "batch_size": True,
            "text_model_chunk_chars": "3500",
            "output_dir": [],
            "ollama_model_name": None,
            "future_option": "ignored",
        }), encoding="utf-8")
        settings = self.store.load()
        expected = AppSettings(last_source="https://example.com/video", opencode_api_key="saved-test-key")
        self.assertEqual(settings, expected)

    def test_invalid_choices_empty_required_fields_and_numbers_use_defaults(self) -> None:
        invalid = {
            "batch_size": 0, "text_model_chunk_chars": -1,
            "task_mode": "broken", "video_quality": "unknown",
            "text_provider": "unknown", "output_language": "unknown",
            "device_mode": "cuda", "compute_type": "float16",
            "output_dir": "  ", "model_size": "",
            "ollama_model_name": "", "ollama_base_url": "", "js_runtime": "",
        }
        self.store.path.write_text(json.dumps(invalid), encoding="utf-8")
        self.assertEqual(self.store.load(), AppSettings())

    def test_invalid_unicode_or_null_characters_cannot_reach_the_interface(self) -> None:
        self.store.path.write_text(json.dumps({"last_source": "bad\x00path", "opencode_model_name": "\ud800"}), encoding="utf-8")
        self.assertEqual(self.store.load(), AppSettings())

    def test_valid_preferences_and_false_checkboxes_round_trip(self) -> None:
        settings = AppSettings(
            output_dir="/tmp/中文结果", task_mode="download", video_quality="720",
            text_provider="opencode", opencode_api_key="saved-test-key", opencode_model_name="custom/model",
            opencode_base_url="https://example.test/v1",
            include_summary=False, include_polish=False, export_docx=False,
            include_visual=True, visual_focus="05:20 附近的图表",
            batch_size=3, text_model_chunk_chars=2400, output_language="source",
        )
        self.store.save(settings)
        self.assertEqual(self.store.load(), settings)
        self.assertEqual(json.loads(self.store.path.read_text(encoding="utf-8")), asdict(settings))
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.assertEqual(list(self.store.app_data_dir.glob("*.tmp")), [])

    def test_defaults_use_unconfigured_online_service_without_assuming_a_local_model(self) -> None:
        settings = self.store.load()
        self.assertEqual(settings.text_provider, "opencode")
        self.assertEqual(settings.opencode_base_url, "")
        self.assertEqual(settings.opencode_api_key, "")
        self.assertEqual(settings.opencode_model_name, "")
        self.assertEqual(settings.ollama_model_name, "")
        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11434")
        self.assertFalse(settings.include_polish)
        self.assertFalse(settings.include_summary)
        self.assertFalse(settings.include_visual)
        self.assertEqual(settings.visual_focus, "")
        self.assertTrue(settings.export_html)
        self.assertFalse(settings.export_docx)
        self.assertFalse(settings.export_markdown)

    def test_legacy_settings_keep_preferences_without_reusing_poe_credentials(self) -> None:
        for provider in ("poe", "ollama"):
            with self.subTest(provider=provider):
                legacy = {
                    "text_provider": provider,
                    "poe_api_key": "old-service-only-key",
                    "poe_model_name": "old-service/model",
                    "ollama_model_name": "my-local-model",
                    "text_model_base_url": "http://localhost:2222",
                    "last_source": "https://example.test/video",
                    "output_dir": "/tmp/我的文稿",
                    "include_polish": False,
                    "export_markdown": False,
                }
                self.store.path.write_text(json.dumps(legacy), encoding="utf-8")
                settings = self.store.load()
                self.assertEqual(settings.text_provider, "opencode")
                self.assertEqual(settings.ollama_model_name, "my-local-model")
                self.assertEqual(settings.ollama_base_url, "http://localhost:2222")
                self.assertEqual(settings.opencode_api_key, "")
                self.assertEqual(settings.opencode_model_name, "")
                self.assertEqual(settings.opencode_base_url, "")
                self.assertEqual(settings.last_source, legacy["last_source"])
                self.assertEqual(settings.output_dir, legacy["output_dir"])
                self.assertFalse(settings.include_polish)
                self.assertFalse(settings.export_markdown)
                self.store.save(settings)
                saved = self.store.path.read_text(encoding="utf-8")
                self.assertNotIn("old-service", saved)
                self.assertNotIn("poe_", saved)
                self.assertNotIn("text_model_base_url", saved)

    def test_old_export_choices_gain_html_without_losing_word_or_markdown(self) -> None:
        for word, markdown in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(word=word, markdown=markdown):
                self.store.path.write_text(json.dumps({
                    "export_docx": word, "export_markdown": markdown,
                    "last_source": "https://vimeo.com/42",
                }), encoding="utf-8")
                loaded = self.store.load()
                self.assertTrue(loaded.export_html)
                self.assertEqual((loaded.export_docx, loaded.export_markdown), (word, markdown))
                self.assertEqual(loaded.last_source, "https://vimeo.com/42")
                loaded.export_html = False
                self.store.save(loaded)
                self.assertEqual(self.store.load(), loaded)
                self.assertFalse(self.store.load().export_html)

    def test_invalid_html_preference_uses_new_default_without_resetting_other_formats(self) -> None:
        self.store.path.write_text(json.dumps({
            "export_html": "false", "export_docx": True, "export_markdown": True,
        }), encoding="utf-8")
        loaded = self.store.load()
        self.assertTrue(loaded.export_html)
        self.assertTrue(loaded.export_docx)
        self.assertTrue(loaded.export_markdown)

    def test_legacy_address_alone_migrates_but_new_format_keeps_local_selection(self) -> None:
        self.store.path.write_text(json.dumps({
            "text_provider": "ollama", "text_model_base_url": "http://localhost:2222",
        }), encoding="utf-8")
        migrated = self.store.load()
        self.assertEqual(migrated.text_provider, "opencode")
        self.assertEqual(migrated.ollama_base_url, "http://localhost:2222")

        settings = AppSettings(
            text_provider="ollama", ollama_model_name="local/model",
            ollama_base_url="http://localhost:3333", opencode_api_key="new-online-key",
        )
        self.store.save(settings)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data.update(poe_api_key="obsolete-key", text_model_base_url="http://obsolete-host")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load(), settings)

    def test_failed_replacement_keeps_previous_settings_and_removes_temporary_file(self) -> None:
        original = AppSettings(last_source="previous")
        self.store.save(original)
        with patch("info_summary.storage.os.replace", side_effect=OSError("模拟保存失败")):
            with self.assertRaises(OSError):
                self.store.save(AppSettings(last_source="next"))
        self.assertEqual(self.store.load(), original)
        self.assertEqual(list(self.store.app_data_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
