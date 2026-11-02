"""造一个「同音色、不同人设」的副本角色，用于验证切换角色时人设/历史隔离。

不重新训练，直接复用格雷丝的 checkpoint 与参考音频。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402

SRC = "格雷丝-748caf"


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    src = store.get(SRC)
    if any(c.name == "阿岚" for c in store.list()):
        print("already exists")
        return 0
    ch = store.create(
        "阿岚",
        persona="28 岁的声音设计师，语速偏快、句子更短，喜欢用反问，"
                "偶尔冷幽默，不会安慰人但会陪着你。",
        greeting="哟，又是你。今天又怎么了？",
    )
    d_src, d_dst = store.dir_of(SRC), store.dir_of(ch.id)
    (d_dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    for f in (d_src / "checkpoints").glob("*"):
        shutil.copyfile(f, d_dst / "checkpoints" / f.name)
    shutil.copyfile(d_src / "reference.wav", d_dst / "reference.wav")
    shutil.copyfile(d_src / "reference.txt", d_dst / "reference.txt")
    ref_text = (d_dst / "reference.txt").read_text(encoding="utf-8").strip()
    store.set_reference(ch.id, d_dst / "reference.wav", ref_text)
    store.set_voice(ch.id, str(d_dst / "checkpoints" / Path(src.voice["gpt_ckpt"]).name),
                    str(d_dst / "checkpoints" / Path(src.voice["sovits_ckpt"]).name),
                    "reference.wav", ref_text)
    store.set_status(ch.id, "ready")
    c = store.get(ch.id)
    c.llm = {"temperature": 2.0, "max_tokens": 120}
    store.save(c)
    print("created", ch.id, ch.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
