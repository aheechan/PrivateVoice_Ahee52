"""对比 repetition_penalty 对"单字拖音"的影响（emily 角色，长句）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

CHAR = Path(r"C:\Users\72157\OneDrive\Desktop\emily-b384a1")
GPT = CHAR / "checkpoints" / "vlab_emily-e12.ckpt"
SOVITS = CHAR / "checkpoints" / "vlab_emily_e12_s204_l48.pth"
REF = CHAR / "reference.wav"
REF_TXT = (CHAR / "reference.txt").read_text(encoding="utf-8").strip()

TEXTS = [
    "我说你让我去学CAD设计，这事儿真不是我一时半会儿能搞定的，得慢慢来。",
    "你要是真想帮我，不如先陪我把这些图纸理一遍，再看看哪里能改。",
]


def frames(x, sr, win=0.025, hop=0.010):
    w, h = int(sr * win), int(sr * hop)
    n = max(0, 1 + (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx]


def longest_sustain(x, sr):
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
            if seg.size > 5 and seg.std() > 0.06:
                cur = 0
                start = None
        else:
            best = max(best, cur)
            cur = 0
    return max(best, cur) * 0.010


def main():
    cfg = Config.load()
    engine = make_engine(cfg)
    engine.load(str(GPT), str(SOVITS))
    for rp in (1.35, 1.6, 1.9):
        print(f"=== repetition_penalty={rp}")
        for text in TEXTS:
            kw = dict(speed=0.96, top_k=9, temperature=0.6, sample_steps=4,
                      repetition_penalty=rp)
            parts, sr = [], 48000
            for c in engine.synthesize_stream(text, str(REF), REF_TXT, **kw):
                parts.append(np.asarray(c.samples, np.float32))
                sr = c.samplerate
            x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            exp = len(text) * 0.22
            print(f"  dur={x.size/sr:5.2f}s 期望≈{exp:4.1f} 比值={x.size/sr/exp:4.2f} "
                  f"最长稳定段={longest_sustain(x, sr):4.2f}s 峰值={float(np.abs(x).max()):.2f}"
                  f" | {text[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
