"""验收 #4：取消训练后进程退出、临时文件清理干净、可重新开始。"""
from __future__ import annotations

import shutil
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

SRC_CHAR = "格雷丝-748caf"


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    src_ds = Path(cfg.path_of("datasets")) / SRC_CHAR

    ch = store.create("取消测试", persona="用来验证取消的临时角色")
    ds = Path(cfg.path_of("datasets")) / ch.id
    shutil.copytree(src_ds / "clips", ds / "clips")
    shutil.copyfile(src_ds / "dataset.list", ds / "dataset.list")
    shutil.copyfile(src_ds / "metadata.json", ds / "metadata.json")
    store.set_dataset(ch.id, 40.0, 10)

    info = train.gpu_info()
    plan = cfg.plan(info.total_gb, "conservative")
    events = []
    runner = train.TrainingRunner(cfg, store, ch.id, ds, plan,
                                  on_event=lambda e: events.append((e.stage, e.message)))
    out = {}

    def run():
        try:
            runner.run()
            out["result"] = "unexpected-success"
        except train.TrainingCancelled:
            out["result"] = "cancelled"
        except Exception as exc:  # noqa: BLE001
            out["result"] = f"error: {exc}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # 等到真正进入 sovits 阶段再取消
    deadline = time.time() + 180
    while time.time() < deadline:
        if any(s == "sovits" and "开始" in m for s, m in events):
            break
        time.sleep(2)
    print("stages seen:", [s for s, _ in events][:8])
    time.sleep(8)            # 让它真的开始跑，确认子进程已起
    runner.cancel()
    t.join(timeout=120)

    time.sleep(3)
    leftover = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
        capture_output=True, text=True, errors="replace").stdout
    s2_left = [ln for ln in leftover.splitlines() if "s2_train" in ln or "s1_train" in ln]

    status = store.get(ch.id).status
    work_gone = not (ds / "work").exists()
    up_logs = Path(cfg.path_of("gpt_sovits")).parent / "GPT-SoVITS" / "logs" / f"vlab_{ch.id.split('-')[0]}"

    print(json_like := {
        "result": out.get("result"),
        "status_after_cancel": status,
        "work_cleaned": work_gone,
        "upstream_logs_cleaned": not up_logs.exists(),
        "leftover_trainer_processes": len(s2_left),
    })

    store.delete(ch.id)
    ok = (out.get("result") == "cancelled" and status == "draft"
          and work_gone and not up_logs.exists() and not s2_left)
    print("CANCEL_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
