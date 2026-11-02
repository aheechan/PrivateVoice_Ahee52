"""复现：切换角色 / 打断之后，TTS 是否会把引擎搞挂。"""
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
    p = Pipeline(cfg, store, ch)
    print("warmup:", p.warmup(), flush=True)

    def turn(text: str, tag: str, timeout: float = 45) -> dict:
        t0 = time.time()
        p.submit_text(text)
        while time.time() - t0 < timeout:
            if p.state.value == "idle" and p._turn_thread and not p._turn_thread.is_alive():
                break
            time.sleep(0.05)
        m = p.metrics.recent(1)[-1]
        print(f"{tag}: {time.time()-t0:.1f}s e2e={m.get('e2e_ms')} "
              f"audio={m.get('audio_sec')} err={m.get('error')}", flush=True)
        return m

    turn("第一轮测试", "turn1")
    p.set_character(ch)
    time.sleep(0.5)
    turn("切换角色之后再说一句", "turn2")
    p.set_character(ch)
    time.sleep(0.5)
    turn("再切一次角色", "turn3")

    # 打断后立刻重说
    p.submit_text("这一句会被打断")
    t0 = time.time()
    while time.time() - t0 < 30:
        if p.state.value == "speaking":
            break
        time.sleep(0.02)
    time.sleep(0.5)
    print("interrupt ms:", round(p.interrupt(), 1), flush=True)
    turn("打断之后立刻再说", "turn4")
    p.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
