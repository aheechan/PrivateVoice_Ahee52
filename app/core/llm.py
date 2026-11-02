"""OpenAI 兼容 LLM 客户端：流式、多 provider、按 priority 回退。

约定：
- POST {base_url}/chat/completions，stream=true，SSE；
- 三段超时：连接 3s、首 token 5s、整体 30s（可在 config 改）；
- 默认向 extra_body 注入关思考参数，provider 可覆盖；
- API Key 用 Windows DPAPI 加密后存 config.yaml，日志里一律脱敏。
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import httpx

DEFAULT_THINKING_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------------------
# DPAPI
# ---------------------------------------------------------------------------
class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _DATA_BLOB:
    buf = ctypes.create_string_buffer(data, len(data))
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def dpapi_encrypt(text: str) -> str:
    """返回 base64 密文；只能在同一个 Windows 用户下解密。"""
    if not text:
        return ""
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    blob_in = _blob(text.encode("utf-8"))
    blob_out = _DATA_BLOB()
    ok = crypt32.CryptProtectData(
        ctypes.byref(blob_in), "voicelab-llm-key", None, None, None, 0,
        ctypes.byref(blob_out))
    if not ok:
        raise OSError(ctypes.get_last_error(), "CryptProtectData 失败")
    try:
        raw = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)
    return base64.b64encode(raw).decode("ascii")


def dpapi_decrypt(token: str) -> str:
    if not token:
        return ""
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    raw = base64.b64decode(token)
    blob_in = _blob(raw)
    blob_out = _DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    if not ok:
        raise OSError(ctypes.get_last_error(), "CryptUnprotectData 失败")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(blob_out.pbData)


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "sk-***"
    return f"{key[:3]}***{key[-4:]}"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
@dataclass
class Provider:
    id: str
    name: str
    base_url: str
    model: str
    priority: int = 10
    enabled: bool = True
    timeout_ms: int = 30000
    supports_stream: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key_enc: str = ""

    @property
    def api_key(self) -> str:
        try:
            return dpapi_decrypt(self.api_key_enc)
        except Exception:
            return ""

    @property
    def masked_key(self) -> str:
        return mask_key(self.api_key)

    def url(self, path: str = "/chat/completions") -> str:
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return base + path

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "base_url": self.base_url,
            "model": self.model, "priority": self.priority, "enabled": self.enabled,
            "timeout_ms": self.timeout_ms, "supports_stream": self.supports_stream,
            "extra_body": self.extra_body, "api_key_enc": self.api_key_enc,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provider":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class ProviderStore:
    """providers 存在 config.yaml 的 llm.providers 里，按 priority 升序生效。"""

    def __init__(self, config: Any):
        self.config = config

    def list(self) -> list[Provider]:
        raw = self.config.get("llm.providers", []) or []
        return [Provider.from_dict(d) for d in raw]

    def active(self) -> list[Provider]:
        items = [p for p in self.list() if p.enabled and p.base_url]
        return sorted(items, key=lambda p: (p.priority, p.name))

    def get(self, pid: str) -> Provider | None:
        return next((p for p in self.list() if p.id == pid), None)

    def save(self, providers: list[Provider]) -> None:
        self.config.set("llm.providers", [p.as_dict() for p in providers])

    def upsert(self, provider: Provider) -> None:
        items = self.list()
        for i, p in enumerate(items):
            if p.id == provider.id:
                items[i] = provider
                break
        else:
            items.append(provider)
        self.save(items)

    def remove(self, pid: str) -> None:
        self.save([p for p in self.list() if p.id != pid])

    def move(self, pid: str, delta: int) -> None:
        items = sorted(self.list(), key=lambda p: p.priority)
        idx = next((i for i, p in enumerate(items) if p.id == pid), None)
        if idx is None:
            return
        j = max(0, min(len(items) - 1, idx + delta))
        if idx == j:
            return
        items[idx], items[j] = items[j], items[idx]
        for k, p in enumerate(items):
            p.priority = (k + 1) * 10
        self.save(items)


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
@dataclass
class LLMDelta:
    text: str = ""
    provider_id: str = ""
    model: str = ""
    first_token_ms: float | None = None
    error: str = ""


@dataclass
class ChatResult:
    text: str = ""
    provider: str = ""
    model: str = ""
    first_token_ms: float | None = None
    total_ms: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Any):
        self.config = config
        self.store = ProviderStore(config)
        self._lock = threading.Lock()

    # -- 参数 ----------------------------------------------------------
    def _timeouts(self, provider: Provider) -> tuple[float, float, float]:
        connect = self.config.get("llm.connect_timeout_ms", 3000) / 1000.0
        first = self.config.get("llm.first_token_timeout_ms", 5000) / 1000.0
        total = min(provider.timeout_ms or 30000,
                    self.config.get("llm.total_timeout_ms", 30000)) / 1000.0
        return connect, first, total

    def _payload(self, provider: Provider, messages: list[dict[str, str]],
                 temperature: float, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra: dict[str, Any] = {}
        if self.config.get("llm.disable_thinking", True):
            extra.update(self.config.get("llm.thinking_extra_body",
                                         DEFAULT_THINKING_BODY))
        extra.update(provider.extra_body or {})
        body.update(extra)
        return body

    def _headers(self, provider: Provider) -> dict[str, str]:
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream"}
        key = provider.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    # -- 单 provider 流式 ----------------------------------------------
    def stream_once(self, provider: Provider, messages: list[dict[str, str]],
                    temperature: float = 0.8, max_tokens: int = 160,
                    cancel: threading.Event | None = None) -> Iterator[LLMDelta]:
        connect_t, first_t, total_t = self._timeouts(provider)
        payload = self._payload(provider, messages, temperature, max_tokens)
        t0 = time.perf_counter()
        first_ms: float | None = None
        try:
            timeout = httpx.Timeout(connect=connect_t, read=first_t, write=10.0,
                                    pool=connect_t)
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("POST", provider.url(), json=payload,
                                   headers=self._headers(provider)) as resp:
                    if resp.status_code >= 400:
                        body = resp.read()[:300].decode("utf-8", "replace")
                        raise LLMError(f"HTTP {resp.status_code}: {body}")
                    for line in resp.iter_lines():
                        if cancel is not None and cancel.is_set():
                            return
                        if time.perf_counter() - t0 > total_t:
                            raise LLMError(f"整体超时 {total_t:.0f}s")
                        if first_ms is None and time.perf_counter() - t0 > first_t:
                            raise LLMError(f"首 token 超时 {first_t:.0f}s")
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content") or ""
                        if not text:
                            continue
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t0) * 1000.0
                        yield LLMDelta(text=text, provider_id=provider.id,
                                       model=provider.model, first_token_ms=first_ms)
        except httpx.ConnectError as exc:
            raise LLMError(f"连接失败：{exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"请求超时：{exc}") from exc

    # -- 多 provider 回退 ----------------------------------------------
    def chat_stream(self, messages: list[dict[str, str]],
                    temperature: float = 0.8, max_tokens: int = 160,
                    cancel: threading.Event | None = None,
                    on_attempt: Callable[[dict[str, Any]], None] | None = None,
                    ) -> Iterator[LLMDelta]:
        """按 priority 逐个尝试；第一个能吐出 token 的 provider 就锁定。"""
        providers = self.store.active()
        if not providers:
            raise LLMError("没有启用的 provider")
        errors: list[str] = []
        for p in providers:
            if cancel is not None and cancel.is_set():
                return
            t0 = time.perf_counter()
            started = False
            try:
                for delta in self.stream_once(p, messages, temperature, max_tokens,
                                              cancel):
                    if not started:
                        started = True
                        if on_attempt:
                            on_attempt({"provider": p.id, "ok": True,
                                        "ttft_ms": delta.first_token_ms})
                    yield delta
                if started:
                    return
                errors.append(f"{p.name}: 空回复")
            except Exception as exc:  # noqa: BLE001
                msg = f"{p.name}: {exc}"
                errors.append(msg)
                if on_attempt:
                    on_attempt({"provider": p.id, "ok": False,
                                "elapsed_ms": (time.perf_counter() - t0) * 1000,
                                "error": str(exc)[:200]})
                continue
        raise LLMError("所有 provider 都失败：" + " | ".join(errors))

    # -- 连接测试 ------------------------------------------------------
    def test(self, provider: Provider, timeout_s: float = 8.0) -> dict[str, Any]:
        messages = [{"role": "user", "content": "ping"}]
        t0 = time.perf_counter()
        text = ""
        first_ms: float | None = None
        try:
            for delta in self.stream_once(provider, messages, 0.2, 4):
                if first_ms is None:
                    first_ms = delta.first_token_ms
                text += delta.text
                if len(text) > 40:
                    break
                if time.perf_counter() - t0 > timeout_s:
                    break
            return {"ok": True, "model": provider.model,
                    "first_token_ms": round(first_ms or 0, 1),
                    "sample": text[:40],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:300],
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


__all__ = [
    "Provider", "ProviderStore", "LLMClient", "LLMDelta", "ChatResult", "LLMError",
    "dpapi_encrypt", "dpapi_decrypt", "mask_key", "DEFAULT_THINKING_BODY",
]
