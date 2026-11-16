"""验收 #6：连续 20 轮后显存/内存是否持续增长。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def sample(pid_tag: str) -> dict:
    import gc

    import torch
    gc.collect()
    proc = psutil.Process()
    rss = proc.memory_info().rss / 1024 ** 2
    vram = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    return {"tag": pid_tag, "rss_mb": round(rss, 1), "vram_mb": round(vram, 1)}


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    y, sr, _ = audio_qc.process_to_48k(ROOT / "assets" / "ref.wav", cfg.audio)
    n = int(4.0 * sr)
    clip = y[:n] if y.size >= n else np.pad(y, (0, n - y.size))

    pipe = Pipeline(cfg, store, ch)
    print("warmup:", pipe.warmup())
    rows = [sample("start")]
    for i in range(20):
        pipe.submit_audio(clip, sr, t_press=time.perf_counter() - 0.3,
                          t_release=time.perf_counter())
        deadline = time.time() + 60
        while time.time() < deadline:
            if pipe.state.value == "idle" and pipe._turn_thread is not None \
                    and not pipe._turn_thread.is_alive():
                break
            time.sleep(0.05)
        if (i + 1) % 5 == 0:
            rows.append(sample(f"turn{i+1}"))
            print("sample", rows[-1], flush=True)
    pipe.close()
    time.sleep(2)
    rows.append(sample("after_close"))
    for r in rows:
        print(r)

    def growth(key: str) -> float:
        vals = [r[key] for r in rows]
        base = vals[1] if len(vals) > 1 else vals[0]   # 首轮之后的基线
        return vals[-1] - base

    rss_growth = growth("rss_mb")
    vram_growth = growth("vram_mb")
    print(f"rss_growth={rss_growth:.1f}MB vram_growth={vram_growth:.1f}MB")
    ok = rss_growth < 300 and vram_growth < 300
    print("MEMORY_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
