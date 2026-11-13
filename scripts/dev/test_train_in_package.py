"""在分发包里跑一次完整训练，验证 run_upstream 垫片修好了 'No module named text'。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ds = Path(cfg.path_of("datasets")) / "emily_test"
    if not ds.exists():
        print("no dataset", ds)
        return 1
    name = f"测试_{time.strftime('%H%M%S')}"
    ch = store.create(name, persona="测试用", greeting="你好")
    print("char", ch.id, name)
    info = train.gpu_info()
    plan = cfg.plan(info.total_gb, "conservative")
    runner = train.TrainingRunner(cfg, store, ch.id, ds, plan,
                                 on_event=lambda e: print(
                                     f"  [{e.stage}] ep={e.epoch}/{e.total_epochs} {e.message}",
                                     flush=True))
    t0 = time.time()
    res = runner.run()
    print(f"DONE in {(time.time()-t0)/60:.1f}min", res)
    c = store.get(ch.id)
    print("voice:", c.voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
