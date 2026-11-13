"""LLM endpoint check — run with the packaged runtime.

Usage (from the install folder):
    runtime\\python.exe tools\\llm_check.py
    runtime\\python.exe tools\\llm_check.py http://192.168.1.10:8080/v1  model-name
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402


def try_url(base: str, model: str, key: str = "") -> tuple[bool, str]:
    base = base.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if not base.endswith("/v1"):
        base += "/v1"
    url = base + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8, "stream": False}
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return False, f"connect failed: {type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    try:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return True, f"ok, reply={text!r}"
    except Exception:
        return False, f"unexpected body: {r.text[:200]}"


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 1:
        base = args[0]
        model = args[1] if len(args) > 1 else "local"
        ok, msg = try_url(base, model)
        print(("PASS " if ok else "FAIL ") + base + " -> " + msg)
        return 0 if ok else 1

    from app.config import Config

    cfg = Config.load()
    providers = cfg.get("llm.providers", []) or []
    if not providers:
        base = str(cfg.get("llm.base_url", "") or "")
        model = str(cfg.get("llm.model", "") or "")
        if not base:
            print("No provider configured (llm.providers empty, llm.base_url empty).")
            print("Open Settings -> LLM and add an OpenAI-compatible endpoint.")
            return 2
        providers = [{"name": "config", "base_url": base, "model": model,
                      "api_key_enc": cfg.get("llm.api_key_enc", "")}]

    from app.core.llm import Provider

    rc = 0
    for raw in providers:
        p = Provider.from_dict(raw)
        ok, msg = try_url(p.base_url, p.model, p.api_key)
        print(("PASS " if ok else "FAIL ") + f"[{p.name}] {p.base_url} / {p.model}")
        print("     " + msg)
        rc = 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
