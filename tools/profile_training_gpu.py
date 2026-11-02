"""采样训练期间 GPU 利用率 / 显存 / 功耗，回答"2GB 占用是否合理、算力是否吃满"。"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402

SAMPLES: list[tuple[float, float, float, float]] = []


def sampler(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"], text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            SAMPLES.append((time.time(), float(parts[0]), float(parts[1]),
                            float(parts[2])))
        except Exception:
            pass
        stop.wait(0.5)


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ds_root = Path(cfg.path_of("datasets"))
    cand = [d for d in ds_root.iterdir() if (d / "clips").exists()]
    ds = cand[0] if cand else ds_root / "emily_test"
    if not ds.exists():
        print("no dataset", ds)
        return 1
    info = train.gpu_info()
    plan = cfg.plan(info.total_gb, "conservative")
    print(f"GPU {info.name} {info.total_gb:.0f}GB free {info.free_gb:.1f}GB | "
          f"plan batch={plan.batch} accum={plan.grad_accum} epochs={plan.epochs}")
    ch = store.create(f"采样{int(time.time())%10000}", persona="t", greeting="hi")
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop,), daemon=True)
    th.start()
    t0 = time.time()
    runner = train.TrainingRunner(cfg, store, ch.id, ds, plan,
                                 on_event=lambda e: None)
    try:
        runner.run()
    finally:
        stop.set()
        th.join(timeout=2)
    elapsed = time.time() - t0
    if not SAMPLES:
        print("没采到 nvidia-smi 数据")
        return 1
    utils = [s[1] for s in SAMPLES]
    mems = [s[2] for s in SAMPLES]
    pows = [s[3] for s in SAMPLES]
    busy = [u for u in utils if u > 5]
    print(f"训练耗时 {elapsed/60:.1f} min，采样 {len(SAMPLES)} 次（0.5s/次）")
    print(f"GPU 利用率：均值 {sum(utils)/len(utils):.0f}%  "
          f"有负载时均值 {sum(busy)/len(busy) if busy else 0:.0f}%  "
          f"峰值 {max(utils):.0f}%  >50% 的采样占比 {100*sum(1 for u in utils if u>50)/len(utils):.0f}%")
    print(f"显存占用：均值 {sum(mems)/len(mems):.0f} MB  峰值 {max(mems):.0f} MB  "
          f"起始 {mems[0]:.0f} MB")
    print(f"功耗：均值 {sum(pows)/len(pows):.0f} W  峰值 {max(pows):.0f} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
