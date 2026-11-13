"""LLM 探针：连接测试 / 流式首 token / 关思考 / provider 回退 / DPAPI 往返。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.llm import (  # noqa: E402
    LLMClient, Provider, ProviderStore, dpapi_decrypt, dpapi_encrypt, mask_key,
)

MESSAGES = [
    {"role": "system", "content": "你是用户的语音陪伴角色，每次回复 1–2 句，不超过 40 字。"},
    {"role": "user", "content": "今天有点累，不想说话。"},
]


def main() -> int:
    cfg = Config.load()
    store = ProviderStore(cfg)
    client = LLMClient(cfg)
    provs = store.list()
    print("providers:", [(p.id, p.name, p.base_url, p.priority, p.enabled) for p in provs])
    if not provs:
        print("NO PROVIDER")
        return 1
    p = provs[0]

    # 1) DPAPI 往返
    token = dpapi_encrypt("sk-test-1234567890abcd")
    back = dpapi_decrypt(token)
    print(f"dpapi: roundtrip={'OK' if back.endswith('abcd') else 'FAIL'} "
          f"masked={mask_key(back)} enc_len={len(token)}")

    # 2) 连接测试
    t0 = time.perf_counter()
    res = client.test(p)
    print(f"test: ok={res.get('ok')} model={res.get('model')} "
          f"ttft={res.get('first_token_ms')}ms total={res.get('elapsed_ms')}ms "
          f"sample={res.get('sample')!r}")
    if not res.get("ok"):
        print("TEST_FAILED:", res.get("error"))
        return 1

    # 3) 流式：首 token 与全文
    ttft = None
    chunks = []
    t0 = time.perf_counter()
    for d in client.chat_stream(MESSAGES, temperature=0.8, max_tokens=160):
        if ttft is None:
            ttft = d.first_token_ms
        chunks.append(d.text)
    text = "".join(chunks)
    total = (time.perf_counter() - t0) * 1000
    print(f"stream: ttft={ttft:.0f}ms total={total:.0f}ms chars={len(text)}")
    print("reply:", text.strip()[:120])
    has_think = ("思考" in text and "我" not in text[:2]) or "```" in text
    print(f"thinking_off: {'OK' if ttft < 1500 and not has_think else 'CHECK'}")

    # 4) 回退：先放一个死端口，再放真 provider
    dead = Provider(id="dead", name="死端口", base_url="http://127.0.0.1:9/v1",
                    model="none", priority=1, enabled=True,
                    timeout_ms=3000)
    live = Provider(**{**p.as_dict(), "priority": 2})
    store.save([dead, live])
    attempts = []
    got = ""
    t0 = time.perf_counter()
    try:
        for d in client.chat_stream(MESSAGES, max_tokens=64,
                                    on_attempt=attempts.append):
            got += d.text
    except Exception as exc:  # noqa: BLE001
        print("FALLBACK_FAILED:", exc)
        store.save(provs)
        return 1
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"fallback: attempts={[(a['provider'], a['ok']) for a in attempts]} "
          f"elapsed={elapsed:.0f}ms reply={got.strip()[:40]!r}")
    store.save(provs)

    ok = (back.endswith("abcd") and res.get("ok") and ttft and ttft < 1500
          and got and any(not a["ok"] for a in attempts))
    print("LLM_PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
