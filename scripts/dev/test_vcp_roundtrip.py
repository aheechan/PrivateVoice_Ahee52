"""验收 #5 / #6：重启后角色仍在且 last_used 正确；.vcp 导出再导入可用。"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc, tts  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402


def main() -> int:
    cfg = Config.load()
    chars_root = Path(cfg.path_of("characters"))
    store = CharacterStore(chars_root)
    ready = [c for c in store.list() if c.status == "ready"]
    if not ready:
        print("FAIL: 没有 ready 角色")
        return 1
    ch = ready[0]
    print(f"角色 {ch.id} status={ch.status} ckpt={Path(ch.voice['gpt_ckpt']).name}")

    # ---- #5 重开应用：新建 store 实例，模拟进程重启 ----
    store.set_active(ch.id)
    time.sleep(1)
    store2 = CharacterStore(chars_root)
    again = store2.get(ch.id)
    ok5 = (again.id == ch.id and again.status == "ready"
           and bool(again.last_used) and store2.active_id == ch.id)
    print(f"#5 重启后：存在={again.id == ch.id} last_used={again.last_used} "
          f"active={store2.active_id} -> {'PASS' if ok5 else 'FAIL'}")

    # ---- #6 导出 / 导入 ----
    pkg = Path(cfg.path_of("results")) / f"{ch.name}.vcp"
    path = store.export_vcp(ch.id, pkg)
    print(f"#6 导出 {path.name} {path.stat().st_size/1024/1024:.1f} MB")

    with tempfile.TemporaryDirectory(prefix="vcp-chk-") as tmp:
        other = CharacterStore(Path(tmp) / "characters")
        back = other.import_vcp(path)
        ck = list((other.dir_of(back.id) / "checkpoints").glob("*"))
        ok = (back.id == ch.id and back.status == "ready" and ck
              and (other.dir_of(back.id) / "reference.wav").exists())
        print(f"#6 导入到空库：id={back.id} ckpt={len(ck)} 个 "
              f"ref={ (other.dir_of(back.id) / 'reference.wav').exists() } -> "
              f"{'PASS' if ok else 'FAIL'}")

        # 导入副本必须能真的推理（这是“可用”的唯一硬标准）
        engine = tts.make_engine(cfg)
        engine.load(back.voice["gpt_ckpt"], back.voice["sovits_ckpt"])
        out = Path(tmp) / "probe.wav"
        engine.synthesize_file("导入之后我还能说话吗？", out,
                               str(other.dir_of(back.id) / back.voice["reference"]),
                               back.voice["reference_text"])
        data, sr = audio_qc.decode(out)
        mono = audio_qc.to_mono(data)
        dur = mono.size / sr
        peak = audio_qc.true_peak_dbfs(mono)
        ok_tts = dur > 0.5 and peak < -0.5
        print(f"#6 导入后合成：{dur:.2f}s peak={peak:.1f} dBFS -> "
              f"{'PASS' if ok_tts else 'FAIL'}")

    # 覆盖式重新导入（幂等性）
    again2 = store.import_vcp(path, overwrite=True)
    ok_idem = again2.id == ch.id and again2.status == "ready"
    print(f"#6 覆盖导入 -> {'PASS' if ok_idem else 'FAIL'}")

    print("VCP_TEST", "PASS" if (ok5 and ok and ok_tts and ok_idem) else "FAIL")
    return 0 if (ok5 and ok and ok_tts and ok_idem) else 1


if __name__ == "__main__":
    raise SystemExit(main())
