"""验证：预热与第一轮并发，是否会让 TTS 卡死；以及路径是否有影响。

用法：
    python tools/repro_warmup_race.py            # 带保护（当前代码）
    python tools/repro_warmup_race.py --bypass   # 绕过预热门闩，模拟旧行为
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bypass", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    print("root:", root)
    cfg = Config.load(root / "config.yaml") if (root / "config.yaml").exists() else Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    p = Pipeline(cfg, store, ch)

    warm = {}
    t = threading.Thread(target=lambda: warm.update(p.warmup()), daemon=True)
    t.start()
    time.sleep(2.0)                      # 预热正在进行（ASR 阶段）
    if args.bypass:
        p._warming = False               # 模拟旧代码：不拦第一轮
        print("bypass gate -> 立刻开一轮")
    else:
        print("gate on -> 这一轮会等预热结束")

    t0 = time.time()
    p.submit_text("预热还没结束我就说话了")
    deadline = t0 + 120
    while time.time() < deadline:
        if p.state.value == "idle" and p._turn_thread and not p._turn_thread.is_alive():
            break
        time.sleep(0.05)
    m = p.metrics.recent(1)[-1] if p.metrics.recent(1) else {}
    print(f"turn done in {time.time()-t0:.1f}s | audio={m.get('audio_sec')} "
          f"e2e={m.get('e2e_ms')} err={m.get('error')}")
    print("warmup:", warm)
    ok = bool(m.get("audio_sec"))
    print("RACE_TEST", "PASS" if ok else "FAIL")
    p.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
