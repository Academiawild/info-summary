from __future__ import annotations

import http.client
import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from info_summary.providers import (
    OllamaProvider,
    OpenCodeProvider,
    ProviderError,
    _read_json,
    generate_with_retry,
    validate_api_url,
)


class JsonResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = urllib.request.Request("http://127.0.0.1/test")

    def test_http_errors_support_ollama_and_opencode_error_shapes(self) -> None:
        for payload in ({"error": "model not found"}, {"error": {"message": "model not found"}}):
            with self.subTest(payload=payload):
                error = urllib.error.HTTPError(
                    self.request.full_url, 404, "Not Found", {},
                    io.BytesIO(json.dumps(payload).encode()),
                )
                with patch("info_summary.providers._urlopen", side_effect=error):
                    with self.assertRaises(ProviderError) as caught:
                        _read_json(self.request, timeout=1)
                self.assertIn("HTTP 404: model not found", str(caught.exception))
                self.assertFalse(caught.exception.retryable)

    def test_malformed_http_error_bodies_preserve_retry_policy(self) -> None:
        for code, retryable in ((401, False), (429, True), (503, True)):
            for body in (b"[]", b"null", b'"failure"', b'{"error": [1]}', b"not-json"):
                with self.subTest(code=code, body=body):
                    error = urllib.error.HTTPError(self.request.full_url, code, "Failure", {}, io.BytesIO(body))
                    with patch("info_summary.providers._urlopen", side_effect=error):
                        with self.assertRaises(ProviderError) as caught:
                            _read_json(self.request, timeout=1)
                    self.assertEqual(caught.exception.retryable, retryable)
                    self.assertIn(f"HTTP {code}", str(caught.exception))

    def test_failure_reading_http_error_body_uses_status_reason(self) -> None:
        body = MagicMock()
        body.read.side_effect = ConnectionResetError("reset")
        error = urllib.error.HTTPError(self.request.full_url, 503, "Unavailable", {}, body)
        with patch("info_summary.providers._urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                _read_json(self.request, timeout=1)
        self.assertEqual(str(caught.exception), "HTTP 503: Unavailable")
        self.assertTrue(caught.exception.retryable)
        body.close.assert_called_once()

    def test_connection_and_incomplete_response_errors_are_retryable(self) -> None:
        for error in (ConnectionResetError("reset"), ConnectionResetError(), TimeoutError("timeout"), OSError("network"),
                      http.client.IncompleteRead(b"partial")):
            for failure_location in ("open", "read"):
                with self.subTest(error=type(error).__name__, location=failure_location):
                    response = MagicMock()
                    response.__enter__.return_value = response
                    response.read.side_effect = error
                    with patch("info_summary.providers._urlopen", **(
                        {"side_effect": error} if failure_location == "open" else {"return_value": response}
                    )):
                        with self.assertRaises(ProviderError) as caught:
                            _read_json(self.request, timeout=1)
                    self.assertTrue(caught.exception.retryable)
                    self.assertGreater(len(str(caught.exception)), len("读取文本模型响应失败："))

    def test_invalid_encoding_json_and_non_object_roots_are_provider_errors(self) -> None:
        for body in (b"\xff", b"{", b"[]", b"null", b"12", b'"text"'):
            with self.subTest(body=body):
                with patch("info_summary.providers._urlopen", return_value=io.BytesIO(body)):
                    with self.assertRaises(ProviderError):
                        _read_json(self.request, timeout=1)

    def test_valid_object_is_returned_without_changes(self) -> None:
        body = {"message": {"content": "中文内容"}}
        with patch("info_summary.providers._urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
            self.assertEqual(_read_json(self.request, timeout=1), body)


class ProviderResponseTests(unittest.TestCase):
    def test_model_lists_skip_invalid_entries_and_keep_valid_names(self) -> None:
        cases = (
            (OllamaProvider(), "models", [{"name": " z "}, {"model": "a"}, {"name": 42, "model": "b"}], ["a", "b", "z"]),
            (OpenCodeProvider("test-key", "https://service.example/v1"), "data", [{"id": " z "}, {"id": "a"}], ["a", "z"]),
        )
        for provider, field, valid_items, expected in cases:
            with self.subTest(provider=type(provider).__name__):
                items = [None, "broken", 1, {}, {"name": []}, {"id": []}, *valid_items]
                with patch("info_summary.providers._read_json", return_value={field: items}):
                    self.assertEqual(provider.list_models(), expected)

    def test_invalid_model_collections_are_provider_errors(self) -> None:
        for provider, field in ((OllamaProvider(), "models"), (OpenCodeProvider("test-key", "https://service.example/v1"), "data")):
            for value in (None, "broken", {}, 42):
                with self.subTest(provider=type(provider).__name__, value=value):
                    with patch("info_summary.providers._read_json", return_value={field: value}):
                        with self.assertRaises(ProviderError):
                            provider.list_models()

    def test_ollama_rejects_invalid_messages_instead_of_exporting_their_string_form(self) -> None:
        for message in (None, "broken", [], {}, {"content": None}, {"content": []}, {"content": 42}, {"content": " "}):
            with self.subTest(message=message):
                with patch("info_summary.providers._read_json", return_value={"message": message}):
                    with self.assertRaises(ProviderError):
                        OllamaProvider().generate("test-model", "system", "user")

    def test_opencode_rejects_invalid_choices_and_messages(self) -> None:
        choices_cases = (None, "broken", {}, [], [None], ["broken"], [{}],
                         [{"message": "broken"}], [{"message": {"content": None}}],
                         [{"message": {"content": ["broken"]}}])
        for choices in choices_cases:
            with self.subTest(choices=choices):
                with patch("info_summary.providers._read_json", return_value={"choices": choices}):
                    with self.assertRaises(ProviderError):
                        OpenCodeProvider("test-key", "https://service.example/v1").generate("test-model", "system", "user")

    def test_valid_text_is_trimmed_and_returned(self) -> None:
        for provider, response in (
            (OllamaProvider(), {"message": {"content": " 中文内容 \n"}}),
            (OpenCodeProvider("test-key", "https://service.example/v1"), {"choices": [{"message": {"content": " 中文内容 \n"}}]}),
        ):
            with self.subTest(provider=type(provider).__name__):
                with patch("info_summary.providers._read_json", return_value=response):
                    self.assertEqual(provider.generate("test-model", "system", "user"), "中文内容")


class OpenCodeRequestTests(unittest.TestCase):
    def test_model_list_and_generation_use_configured_address_and_bearer_key(self) -> None:
        provider = OpenCodeProvider("  private-test-key  ", "https://service.example/custom/v1/")
        responses = [
            {"data": [{"id": "my-model"}]},
            {"choices": [{"message": {"content": "整理完成"}}]},
        ]
        with patch("info_summary.providers._read_json", side_effect=responses) as read:
            self.assertEqual(provider.list_models(), ["my-model"])
            self.assertEqual(provider.generate("my-model", "系统说明", "原文内容"), "整理完成")
        listing, generation = [call.args[0] for call in read.call_args_list]
        self.assertEqual(listing.full_url, "https://service.example/custom/v1/models")
        self.assertEqual(listing.get_method(), "GET")
        self.assertEqual(generation.full_url, "https://service.example/custom/v1/chat/completions")
        self.assertEqual(generation.get_method(), "POST")
        self.assertEqual(generation.get_header("Authorization"), "Bearer private-test-key")
        payload = json.loads(generation.data)
        self.assertEqual(payload["model"], "my-model")
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": "系统说明"},
            {"role": "user", "content": "原文内容"},
        ])
        self.assertFalse(payload["stream"])
        self.assertNotIn("private-test-key", generation.full_url)
        self.assertNotIn("private-test-key", generation.data.decode())

    def test_incomplete_setup_is_actionable_and_makes_no_network_request(self) -> None:
        for provider, missing in (
            (OpenCodeProvider(""), "接口地址"),
            (OpenCodeProvider("", "https://service.example/v1"), "密钥"),
        ):
            with self.subTest(missing=missing), patch("info_summary.providers._read_json") as read:
                healthy, message = provider.health_check()
                self.assertFalse(healthy)
                self.assertIn(missing, message)
                read.assert_not_called()
        with patch("info_summary.providers._read_json") as read:
            with self.assertRaisesRegex(ProviderError, "尚未选择"):
                OpenCodeProvider("test", "https://service.example/v1").generate("", "system", "user")
            read.assert_not_called()

    def test_bad_api_addresses_fail_before_sending_credentials(self) -> None:
        invalid = (
            "", "service.example/v1", "file:///tmp/api", "https://service.example:bad/v1",
            "https://user:key@service.example/v1", "https://service.example/v1?key=secret",
            "https://service.example/v1#secret", "https://service.example/v1/chat/completions",
            "http://service.example/v1", "https://service.example/my path",
        )
        for address in invalid:
            with self.subTest(address=address):
                with self.assertRaises(ValueError):
                    validate_api_url(address)
        self.assertEqual(validate_api_url(" https://service.example/v1/ "), "https://service.example/v1")
        self.assertEqual(validate_api_url("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1")

    def test_common_api_errors_explain_how_to_fix_the_problem(self) -> None:
        for code, phrase in ((401, "密钥无效"), (403, "没有调用权限"), (404, "接口或模型"), (429, "额度不足")):
            with self.subTest(code=code):
                error = ProviderError("opaque response", retryable=code == 429, status_code=code)
                with patch("info_summary.providers._read_json", side_effect=error):
                    with self.assertRaises(ProviderError) as caught:
                        OpenCodeProvider("test", "https://service.example/v1").list_models()
                self.assertIn(phrase, str(caught.exception))
                self.assertEqual(caught.exception.retryable, code == 429)

    def test_server_errors_cannot_leak_key_to_warnings_or_exported_documents(self) -> None:
        key = "private-test-key-for-redaction"
        provider = OpenCodeProvider(key, "https://service.example/v1")
        for code in (400, 500):
            error = ProviderError(f"server echoed {key}", retryable=code == 500, status_code=code)
            with self.subTest(code=code), patch("info_summary.providers._read_json", side_effect=error):
                with self.assertRaises(ProviderError) as caught:
                    provider.generate("test-model", "system", "user")
                self.assertNotIn(key, str(caught.exception))
                healthy, message = provider.health_check()
                self.assertFalse(healthy)
                self.assertNotIn(key, message)


class ProviderRetryTests(unittest.TestCase):
    def test_non_retryable_error_is_preserved_and_not_retried(self) -> None:
        error = ProviderError("model not found", retryable=False)
        provider = MagicMock()
        provider.generate.side_effect = error
        with patch("info_summary.providers.time.sleep") as sleep:
            with self.assertRaises(ProviderError) as caught:
                generate_with_retry(provider, "model", "system", "user")
        self.assertIs(caught.exception, error)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(provider.generate.call_count, 1)
        sleep.assert_not_called()

    def test_transient_failure_retries_and_can_succeed(self) -> None:
        provider = MagicMock()
        provider.generate.side_effect = [ProviderError("temporary"), "完成"]
        with patch("info_summary.providers.time.sleep") as sleep:
            self.assertEqual(generate_with_retry(provider, "model", "system", "user"), "完成")
        self.assertEqual(provider.generate.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_exhausted_retries_preserve_the_last_error(self) -> None:
        provider = MagicMock()
        error = ProviderError("still unavailable")
        provider.generate.side_effect = error
        with patch("info_summary.providers.time.sleep"):
            with self.assertRaises(ProviderError) as caught:
                generate_with_retry(provider, "model", "system", "user")
        self.assertIs(caught.exception, error)
        self.assertEqual(provider.generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
