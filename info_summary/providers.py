from __future__ import annotations

import base64
import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit

from .cancellation import check_cancelled
from .settings import redact_secret

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Loopback services (Ollama, the local test server) must never be routed through
# an HTTP(S)/SOCKS proxy configured for outbound traffic, so use an explicit
# no-proxy opener for them while leaving remote calls (OpenCode API) on the env proxy.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(request: urllib.request.Request, timeout):
    host = urlsplit(request.full_url).hostname or ""
    if host in _LOOPBACK_HOSTS:
        return _NO_PROXY_OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


class TextProvider(Protocol):
    def health_check(self) -> tuple[bool, str]: ...

    def list_models(self) -> list[str]: ...

    def generate(self, model: str, system_prompt: str, user_prompt: str) -> str: ...

    def generate_vision(self, model: str, system_prompt: str, user_prompt: str, images: list[Path]) -> str: ...


def _encoded_image(path: Path) -> tuple[str, str]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ProviderError("截图暂时无法读取，请重新处理该视频。", retryable=False) from exc
    if not content:
        raise ProviderError("截图为空，请重新提取画面。", retryable=False)
    mime = "image/png" if content.startswith(b"\x89PNG") else "image/jpeg"
    return mime, base64.b64encode(content).decode("ascii")


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _read_json(
    request: urllib.request.Request,
    timeout: int,
) -> dict:
    try:
        with _urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ProviderError("文本模型返回的数据格式不正确，请稍后重试。")
        return data
    except urllib.error.HTTPError as exc:
        try:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException):
                # The body read can be reset mid-stream (common with loopback
                # servers); fall back to the status reason so we still surface
                # a clean ProviderError instead of leaking ConnectionResetError.
                body = ""
            try:
                data = json.loads(body)
            except (ValueError, RecursionError):
                message = body
            else:
                error = data.get("error") if isinstance(data, dict) else None
                if isinstance(error, dict):
                    error = error.get("message")
                message = error.strip() if isinstance(error, str) and error.strip() else body
            message = message or exc.reason or "请求失败"
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            raise ProviderError(f"HTTP {exc.code}: {message}", retryable=retryable, status_code=exc.code) from exc
        finally:
            try:
                exc.close()
            except (OSError, http.client.HTTPException):
                pass
    except (OSError, ValueError, RecursionError, http.client.HTTPException) as exc:
        message = str(exc) or "连接已中断，请稍后重试。"
        raise ProviderError(f"读取文本模型响应失败：{message}") from exc


def _model_names(data: dict, field: str, name_fields: tuple[str, ...]) -> list[str]:
    items = data.get(field, [])
    if not isinstance(items, list):
        raise ProviderError("文本模型返回的模型列表格式不正确，请稍后重试。")
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in name_fields:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break
    return sorted(names)


def _message_text(message: object, provider: str, model: str) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderError(f"{provider} 模型 {model} 没有返回有效文本。")
    return content.strip()


@dataclass
class OllamaProvider:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 300

    def health_check(self) -> tuple[bool, str]:
        request = urllib.request.Request(self.base_url.rstrip("/") + "/api/tags")
        try:
            _read_json(request, timeout=5)
            return True, "Ollama 服务可用。"
        except ProviderError as exc:
            return False, f"Ollama 服务不可用：{exc}"

    def list_models(self) -> list[str]:
        request = urllib.request.Request(self.base_url.rstrip("/") + "/api/tags")
        data = _read_json(request, timeout=10)
        return _model_names(data, "models", ("name", "model"))

    def generate(self, model: str, system_prompt: str, user_prompt: str) -> str:
        return self._generate(model, system_prompt, {"role": "user", "content": user_prompt})

    def generate_vision(self, model: str, system_prompt: str, user_prompt: str, images: list[Path]) -> str:
        return self._generate(model, system_prompt, {
            "role": "user", "content": user_prompt,
            "images": [_encoded_image(path)[1] for path in images],
        })

    def _generate(self, model: str, system_prompt: str, user_message: dict) -> str:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        user_message,
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data = _read_json(request, timeout=self.timeout_seconds)
        return _message_text(data.get("message"), "Ollama", model)


def validate_api_url(value: str) -> str:
    """Validate a configurable API base without guessing a remote service."""
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("请在“模型设置”填写 OpenCode API 接口地址；服务资料尚未确定时，可以先选择“只提取原文”。")
    try:
        parsed = urlsplit(value)
        _ = parsed.port  # Validate malformed port values before constructing a request.
    except ValueError as exc:
        raise ValueError("接口地址格式不正确，请填写服务提供的完整基础地址。") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(char.isspace() for char in value)):
        raise ValueError("接口地址应是 http:// 或 https:// 开头的基础地址，不要在地址中填写密钥、账号或查询参数。")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("在线接口请使用 https:// 地址；本机接口可以使用 http://127.0.0.1。")
    if parsed.path.endswith(("/chat/completions", "/models", "/responses", "/messages")):
        raise ValueError("请填写服务提供的基础地址（通常以 /v1 结尾），不要包含 /chat/completions、/models 等具体接口。")
    return value


@dataclass
class OpenCodeProvider:
    """Configurable Chat Completions API; no service or model is assumed."""

    api_key: str
    base_url: str = ""
    timeout_seconds: int = 300

    def _request(self, path: str, data: dict | None = None, timeout: int | None = None) -> dict:
        try:
            base_url = validate_api_url(self.base_url)
        except ValueError as exc:
            raise ProviderError(str(exc), retryable=False) from exc
        if not self.api_key.strip():
            raise ProviderError("尚未填写 OpenCode API 密钥，请点击“模型设置”填写；也可以先选择“只提取原文”。", retryable=False)
        request = urllib.request.Request(
            base_url + path,
            data=json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None,
            headers={"Authorization": f"Bearer {self.api_key.strip()}", "Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            return _read_json(request, timeout=timeout or self.timeout_seconds)
        except ProviderError as exc:
            explanation = {
                401: "API 密钥无效或已过期，请在“模型设置”重新填写。",
                403: "当前账号没有调用权限，请检查服务权限或额度。",
                404: "未找到接口或模型，请核对接口地址和模型名称；此入口需要兼容 Chat Completions 的服务。",
                429: "请求过于频繁或额度不足，请检查服务额度后重试。",
            }.get(exc.status_code)
            if explanation:
                raise ProviderError(f"{explanation}（HTTP {exc.status_code}）", retryable=exc.retryable,
                                    status_code=exc.status_code) from exc
            message = str(exc)
            for key in (self.api_key, self.api_key.strip()):
                if key:
                    message = message.replace(key, redact_secret(key))
            raise ProviderError(message, retryable=exc.retryable, status_code=exc.status_code) from exc

    def health_check(self) -> tuple[bool, str]:
        try:
            self.list_models()
            return True, "已连接 OpenCode API，可以读取模型列表。实际生成权限会在处理时验证。"
        except ProviderError as exc:
            return False, f"OpenCode API 尚不可用：{exc}"

    def list_models(self) -> list[str]:
        data = self._request("/models", timeout=15)
        return _model_names(data, "data", ("id",))

    def generate(self, model: str, system_prompt: str, user_prompt: str) -> str:
        return self._generate(model, system_prompt, user_prompt)

    def generate_vision(self, model: str, system_prompt: str, user_prompt: str, images: list[Path]) -> str:
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for path in images:
            mime, encoded = _encoded_image(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return self._generate(model, system_prompt, content)

    def _generate(self, model: str, system_prompt: str, content: str | list[dict]) -> str:
        if not model.strip():
            raise ProviderError("尚未选择 OpenCode API 模型，请在“模型设置”读取模型列表或填写服务提供的模型名称。", retryable=False)
        data = self._request(
            "/chat/completions",
            {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "stream": False,
            },
        )
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        return _message_text(message, "OpenCode API", model)


def generate_with_retry(
    provider: TextProvider,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 2,
    sleep_seconds: float = 1.0,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    last_error: ProviderError | None = None
    for attempt in range(retries + 1):
        check_cancelled(cancelled)
        try:
            # Preserve a completed response so callers can checkpoint it before
            # honoring cancellation. Failed requests never start another retry.
            return provider.generate(model, system_prompt, user_prompt)
        except ProviderError as exc:
            check_cancelled(cancelled)
            last_error = exc
            if attempt < retries and exc.retryable:
                delay = sleep_seconds * (attempt + 1)
                if cancelled is None:
                    time.sleep(delay)
                else:
                    remaining = max(0.0, delay)
                    while remaining > 0:
                        check_cancelled(cancelled)
                        interval = min(0.1, remaining)
                        time.sleep(interval)
                        remaining -= interval
                    check_cancelled(cancelled)
            else:
                break
    if last_error is not None:
        raise last_error
    raise ProviderError("文本模型调用失败。")


def generate_vision_with_retry(
    provider: TextProvider, model: str, system_prompt: str, user_prompt: str,
    images: list[Path], cancelled: Callable[[], bool] | None = None,
) -> str:
    class VisionCall:
        def generate(self, model_name: str, system: str, user: str) -> str:
            method = getattr(provider, "generate_vision", None)
            if method is None:
                raise ProviderError("当前模型接口不支持图片，请选择支持看图的模型。", retryable=False)
            return method(model_name, system, user, images)

    return generate_with_retry(VisionCall(), model, system_prompt, user_prompt, cancelled=cancelled)
