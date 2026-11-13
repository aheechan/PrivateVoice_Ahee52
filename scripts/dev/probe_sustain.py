"""用 emily 角色测长句的"单字拖音"，对比不同语音参数。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

CHAR = Path(r"C:\Users\72157\OneDrive\Desktop\emily-b384a1")
GPT = CHAR / "checkpoints" / "vlab_emily-e12.ckpt"
SOVITS = CHAR / "checkpoints" / "vlab_emily_e12_s204_l48.pth"
REF = CHAR / "reference.wav"
REF_TXT = (CHAR / "reference.txt").read_text(encoding="utf-8").strip()

SENTENCES = [
    "我说你让我去学CAD设计，这事儿真不是我一时半会儿能搞定的，得慢慢来。",
    "今天下班路上突然下雨，我一路小跑回家，衣服全湿透了。",
    "你要是真想帮我，不如先陪我把这些图纸理一遍，再看看哪里能改。",
]

PARAMS = {
    "A 当前(0.96/k9/t0.6/s4)": dict(speed=0.96, top_k=9, temperature=0.6, sample_steps=4),
    "B top_k=15": dict(speed=0.96, top_k=15, temperature=0.6, sample_steps=4),
    "C 温度0.45": dict(speed=0.96, top_k=9, temperature=0.45, sample_steps=4),
    "D 步数8": dict(speed=0.96, top_k=9, temperature=0.6, sample_steps=8),
    "E 语速1.05": dict(speed=1.05, top_k=9, temperature=0.6, sample_steps=4),
}


def frames(x: np.ndarray, sr: int, win=0.025, hop=0.010):
    w, h = int(sr * win), int(sr * hop)
    n = max(0, 1 + (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx]


def longest_sustain(x: np.ndarray, sr: int) -> float:
    """最长"稳定拖音"：浊音段内过零率波动很小的连续时长。"""
    fr = frames(x.astype(np.float64), sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = np.percentile(db, 60)
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1)
    voiced = db > thr
    best = cur = 0
    start = None
    for i, v in enumerate(voiced):
        if v:
            if cur == 0:
                start = i
            cur += 1
            seg = zcr[start:i + 1]
            if seg.size > 5 and seg.std() > 0.06:   # 过零率变化大 -> 不是拖音
                cur = 0
                start = None
        else:
            best = max(best, cur)
            cur = 0
    best = max(best, cur)
    return best * 0.010


def main() -> int:
    cfg = Config.load()
    engine = make_engine(cfg)
    engine.load(str(GPT), str(SOVITS))
    for name, kw in PARAMS.items():
        print(f"=== {name}")
        for text in SENTENCES:
            t0 = time.perf_counter()
            parts, sr = [], 48000
            for c in engine.synthesize_stream(text, str(REF), REF_TXT, **kw):
                parts.append(np.asarray(c.samples, np.float32))
                sr = c.samplerate
            x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            dur = x.size / sr
            exp = len(text) * 0.22
            sus = longest_sustain(x, sr)
            print(f"  dur={dur:5.2f}s 期望≈{exp:4.1f}s 比值={dur/max(exp,0.1):4.2f} "
                  f"最长稳定段={sus:4.2f}s 峰值={float(np.abs(x).max()):.2f} "
                  f"耗时={time.perf_counter()-t0:4.1f}s | {text[:14]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
