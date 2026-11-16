"""用已有角色的切片数据训练一个 v4 LoRA 角色，并和 v3 做对比。

前提：GPT_SoVITS/pretrained_models/ 下要有 s2Gv4.pth 与 vocoder.pth。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402

SRC_NAME = "格雷丝"
NEW_NAME = "格雷丝_v4"


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    src = next(c for c in store.list() if c.name == SRC_NAME)
    src_ds = Path(cfg.path_of("datasets")) / src.id
    if not (src_ds / "clips").exists():
        print("源数据集没有 clips：", src_ds)
        return 1

    existing = [c for c in store.list() if c.name == NEW_NAME]
    if existing:
        ch = existing[0]
        print("复用已有角色", ch.id)
    else:
        ch = store.create(NEW_NAME, persona=src.persona, greeting=src.greeting)
        ch.llm = dict(src.llm)
        store.save(ch)
        print("新建角色", ch.id)

    ds = Path(cfg.path_of("datasets")) / ch.id
    ds.mkdir(parents=True, exist_ok=True)
    if not (ds / "clips").exists():
        shutil.copytree(src_ds / "clips", ds / "clips")
        for f in ("dataset.list", "metadata.json", "qc.json"):
            if (src_ds / f).exists():
                shutil.copyfile(src_ds / f, ds / f)
    if not (ds / "reference.wav").exists():
        shutil.copyfile(store.dir_of(src.id) / "reference.wav", ds / "reference.wav")

    cfg.set("training.s2_version", "v4")
    info = train.gpu_info()
    plan = cfg.plan(info.total_gb, "conservative")
    print("plan:", plan.as_dict())

    def on_event(ev):
        if ev.stage in ("prepare", "sovits", "gpt", "reference", "done", "failed"):
            print(f"  [{ev.stage}] ep={ev.epoch}/{ev.total_epochs} {ev.message}", flush=True)

    t0 = time.time()
    runner = train.TrainingRunner(cfg, store, ch.id, ds, plan, on_event=on_event)
    result = runner.run()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {json.dumps(result, ensure_ascii=False)}")

    c = store.get(ch.id)
    print("voice:", c.voice.get("sovits_ckpt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
