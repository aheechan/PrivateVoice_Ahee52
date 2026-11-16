"""压力测试：反复「说到一半打断 → 立刻重说」，验证引擎不会卡死。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    events: list[str] = []
    p = Pipeline(cfg, store, ch,
                 on_event=lambda e: events.append(f"{e.kind}:{(e.text or e.detail)[:40]}"))
    print("warmup:", p.warmup(), flush=True)

    fails = 0
    for i in range(8):
        text = f"第{i+1}句，测试打断之后的恢复能力。"
        p.submit_text(text)
        t0 = time.time()
        while time.time() - t0 < 40:
            if p.state.value == "speaking":
                break
            time.sleep(0.02)
        time.sleep(0.35 * (i % 3))
        if i % 2 == 1:
            p.interrupt()
            time.sleep(0.2)
        # 等这一轮收尾
        t1 = time.time()
        while time.time() - t1 < 40:
            if p.state.value == "idle" and p._turn_thread and not p._turn_thread.is_alive():
                break
            time.sleep(0.05)
        m = p.metrics.recent(1)[-1]
        err = m.get("error") or ""
        audio = m.get("audio_sec") or 0
        interrupted = (i % 2 == 1)
        ok = bool(audio) and not err
        fails += 0 if (ok or interrupted) else 1
        verdict = "OK(已打断)" if interrupted else ("OK" if ok else "FAIL")
        print(f"turn{i+1}: audio={audio} e2e={m.get('e2e_ms')} err={err!r} "
              f"{verdict}", flush=True)

    resets = [e for e in events if "重置" in e]
    print("engine resets:", resets)
    print("STRESS", "PASS" if fails == 0 else f"FAIL({fails})")
    p.close()
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
