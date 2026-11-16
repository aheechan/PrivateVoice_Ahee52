"""定位语气词开头的两种异常：起始恒定平台（DC hold）与异常拉长（重复/延音）。

对每个文本 × 参数组合，报告：时长、每字时长、起始 300ms 的最长恒定段（非零）、
以及 >40ms 的非零恒定平台数量。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

CASES = ["嗯。", "唔，那就早点睡。", "哦，这样啊。", "嗯，我知道了。"]
PARAMS = [
    {"sample_steps": 4, "top_k": 5, "temperature": 0.6},
    {"sample_steps": 8, "top_k": 5, "temperature": 0.6},
    {"sample_steps": 4, "top_k": 15, "temperature": 0.7},
]


def holds(x: np.ndarray, min_run: int = 1920) -> tuple[int, int]:
    """返回 (最长非零恒定段长度, >40ms 的平台个数)。"""
    if x.size < 2:
        return 0, 0
    same = np.abs(np.diff(x)) < 1e-7
    best = cur = 0
    count = 0
    for i, v in enumerate(same):
        if v and abs(x[i]) > 1e-4:
            cur += 1
        else:
            if cur >= min_run:
                count += 1
            best = max(best, cur)
            cur = 0
    if cur >= min_run:
        count += 1
    return max(best, cur), count


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = next(c for c in store.list() if "v4" in c.name)
    engine = make_engine(cfg)
    engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
    ref = str(store.dir_of(ch.id) / "reference.wav")
    rt = ch.voice.get("reference_text", "")

    for text in CASES:
        for p in PARAMS:
            parts, sr = [], 48000
            t0 = time.perf_counter()
            for c in engine.synthesize_stream(text, ref, rt, **p):
                parts.append(c.samples)
                sr = c.samplerate
            el = time.perf_counter() - t0
            x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            dur = x.size / sr
            head = x[: int(sr * 0.3)]
            hold_len, hold_n = holds(head)
            all_hold, all_n = holds(x)
            print(f"{text!r:18s} steps={p['sample_steps']} top_k={p['top_k']:2d} "
                  f"T={p['temperature']:.2f} | dur={dur:5.2f}s "
                  f"每字={dur/max(1,len(text)):.2f}s rtf={el/max(dur,1e-6):.2f} "
                  f"| 起始平台={hold_len:4d}样点 全段平台={all_hold:5d}样点/{all_n}处",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
