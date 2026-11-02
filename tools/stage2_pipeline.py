"""Stage 2 端到端流水线（无 UI）：建角色 → 质检 → 切分打标 → 训练 → 复测。

用法：
    python tools/stage2_pipeline.py --stage prep     # 只跑到切分打标
    python tools/stage2_pipeline.py --stage train    # 接着训练 + 复测
    python tools/stage2_pipeline.py --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc, segment, train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.label import Transcriber  # noqa: E402

SRC = Path(r"C:\Users\72157\OneDrive\Desktop\Grace_voice\grace voice nanpu plus orig 0327.wav")
NAME = "格雷丝"
STATE = ROOT / "datasets" / "_pipeline_state.json"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def do_prep(cfg: Config, store: CharacterStore) -> dict:
    ch = store.create(NAME, persona="沉稳温柔的播客女声，语速偏慢，句尾会轻轻收住。",
                      greeting="嗯，我在。今天想聊点什么？")
    cid = ch.id
    print(f"[pipeline] character {cid} created")

    ds = Path(cfg.path_of("datasets")) / cid
    ds.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    qc = audio_qc.run_qc(SRC, cfg)
    audio_qc.save_qc(qc, ds / "qc.json")
    print(f"[pipeline] QC {qc.status} dur={qc.duration_sec}s lufs={qc.lufs} "
          f"snr={qc.snr_db} speakers={qc.speakers} ({time.time()-t0:.1f}s)")
    for c in qc.checks:
        print(f"           {c.level:5s} {c.label}: {c.message}")
    if not qc.ok:
        raise SystemExit("质检不通过，停止")

    t0 = time.time()
    tr = Transcriber(cfg.label_cfg)
    res = segment.segment_file(SRC, cfg, ds, transcriber=tr, character_id=cid,
                               progress=lambda m: print("           ..", m, flush=True))
    print(f"[pipeline] segments={len(res.rows)} ({time.time()-t0:.1f}s)")
    for r in res.rows:
        print(f"  {r.index:02d} {r.duration:5.2f}s {r.flags} {r.text[:40]}")
    store.set_dataset(cid, sum(r.duration for r in res.rows), len(res.rows))
    state = {"character_id": cid, "dataset_dir": str(ds), "source": str(SRC)}
    save_state(state)
    return state


def do_train(cfg: Config, store: CharacterStore, state: dict) -> None:
    cid = state["character_id"]
    ds = Path(state["dataset_dir"])
    info = train.gpu_info()
    print(f"[pipeline] GPU {info.describe()} torch={info.torch}")
    plan = cfg.plan(info.total_gb, "conservative")
    print(f"[pipeline] plan {plan.as_dict()}")

    def on_event(ev: train.TrainEvent) -> None:
        print(f"  [{ev.stage}] ep={ev.epoch}/{ev.total_epochs} "
              f"eta={ev.eta_sec and round(ev.eta_sec)}s {ev.message}", flush=True)

    runner = train.TrainingRunner(cfg, store, cid, ds, plan, on_event=on_event)
    t0 = time.time()
    result = runner.run()
    print(f"[pipeline] train done in {(time.time()-t0)/60:.1f} min -> {result}")

    ch = store.get(cid)
    print("[pipeline] benchmarking ...")
    bench = train.run_benchmark(cfg, cid, ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"],
                                str(store.dir_of(cid) / ch.voice["reference"]),
                                ch.voice["reference_text"])
    if "error" in bench:
        print("[pipeline] benchmark failed:", bench["error"][:600])
    else:
        for tier, t in bench["tiers"].items():
            print(f"  {tier:6s} ttfp_p50={t['ttfp_p50_ms']:.0f}ms "
                  f"p95={t['ttfp_p95_ms']:.0f}ms rtf={t['rtf_p50']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prep", "train", "all"), default="all")
    ap.add_argument("--character", default="")
    args = ap.parse_args()

    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    state = load_state()

    if args.stage in ("prep", "all"):
        state = do_prep(cfg, store)
    elif not state:
        raise SystemExit("没有前置状态，请先跑 --stage prep")

    if args.stage == "prep":
        return 0
    do_train(cfg, store, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
