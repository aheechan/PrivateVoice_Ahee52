"""验收 #5：所有 provider 都不可用时，播兜底短句且不崩。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def main() -> int:
    # 先把本地 llama-server 停掉，模拟「本地挂了、又没配云端」
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                   capture_output=True)
    time.sleep(2.0)

    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    events: list[tuple[str, str]] = []

    def on_event(ev):
        events.append((ev.kind, ev.text or ev.detail or ev.state))
        print(f"  [{ev.kind}] {(ev.text or ev.detail or ev.state)[:80]}")

    pipe = Pipeline(cfg, store, ch, on_event=on_event)
    print("warmup:", pipe.warmup())
    print("=== 发起一轮（本地 LLM 已停） ===")
    t0 = time.time()
    pipe.submit_text("你还在吗")
    while time.time() - t0 < 60:
        if pipe.state.value == "idle" and pipe._turn_thread is not None \
                and not pipe._turn_thread.is_alive():
            break
        time.sleep(0.05)
    fallback = str(cfg.get("llm.fallback_text", "我这会儿有点接不上话。"))
    replies = [t for k, t in events if k == "turn_done"]
    errs = [t for k, t in events if k == "error"]
    rows = pipe.metrics.recent(1)
    pipe.close()

    ok = bool(replies) and fallback[:6] in replies[-1] and any("LLM" in e for e in errs)
    print(f"\nfallback_expected={fallback!r}")
    print(f"replies={replies!r}")
    print(f"errors={errs!r}")
    print(f"metrics={rows[-1] if rows else None}")
    print("FALLBACK_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
