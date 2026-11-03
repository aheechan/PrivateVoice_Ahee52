"""自查角色文件：参考音频是否可解码、权重是否齐全、合成是否出声。

用法（在安装目录下）：
    runtime\\python.exe tools\\check_character.py            # 检查所有角色
    runtime\\python.exe tools\\check_character.py emily      # 只查名字含 emily 的角色
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402


def check_reference(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在"
    size = path.stat().st_size
    if size < 1024:
        return False, f"文件只有 {size} 字节（几乎肯定是坏的）"
    try:
        y, sr = audio_qc.decode(str(path))
        if y.size == 0:
            return False, "解码为空"
        mono = audio_qc.to_mono(y)
        peak = float(np.abs(mono).max())
        return True, (f"{size/1024:.0f}KB / {y.shape[0]/sr:.2f}s @ {sr}Hz / "
                      f"峰值 {peak:.3f}")
    except Exception as exc:  # noqa: BLE001
        return False, f"解码失败：{exc}"


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    keyword = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    chars = [c for c in store.list() if keyword in c.name.lower()]
    if not chars:
        print("没有匹配的角色")
        return 1
    for ch in chars:
        d = store.dir_of(ch.id)
        print(f"=== {ch.name} ({ch.id}) status={ch.status}")
        for key in ("gpt_ckpt", "sovits_ckpt"):
            p = Path(str(ch.voice.get(key) or ""))
            ok = p.exists()
            print(f"  {key:12s}: {'OK ' if ok else '缺失'} {p}")
        ref = d / str(ch.voice.get("reference") or "reference.wav")
        ok, detail = check_reference(ref)
        print(f"  reference   : {'OK ' if ok else '坏 '} {detail}")
        if not ok:
            clips = sorted((d / "dataset" / "clips").glob("*.wav"))
            print(f"  可用的 dataset 片段：{len(clips)} 个"
                  + (f"（例如 {clips[0].name}）" if clips else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
