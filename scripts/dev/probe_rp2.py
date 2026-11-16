"""按社区建议的组合再测：高 top_k + 高温 + 适中 rp。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"C:\Local Repositories\PrivateVoice")))
from app.config import Config  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

CHAR = Path(r"C:\Users\72157\OneDrive\Desktop\emily-b384a1")
TEXTS = [
    "我说你让我去学CAD设计，这事儿真不是我一时半会儿能搞定的，得慢慢来。",
    "你要是真想帮我，不如先陪我把这些图纸理一遍，再看看哪里能改。",
]
COMBOS = {
    "现默认 rp1.9 k9 t0.6": dict(speed=0.96, top_k=9, temperature=0.6,
                                repetition_penalty=1.9, sample_steps=4),
    "社区建议 rp1.5 k60 t0.9": dict(speed=0.96, top_k=60, temperature=0.9,
                                   repetition_penalty=1.5, sample_steps=4),
    "折中 rp1.8 k30 t0.75": dict(speed=0.96, top_k=30, temperature=0.75,
                                repetition_penalty=1.8, sample_steps=4),
}


def frames(x, sr, win=0.025, hop=0.010):
    w, h = int(sr * win), int(sr * hop)
    n = max(0, 1 + (len(x) - w) // h)
    return x[np.arange(w)[None, :] + h * np.arange(n)[:, None]]


def sustain(x, sr):
    fr = frames(x.astype(np.float64), sr)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12) + 1e-12)
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


cfg = Config.load()
engine = make_engine(cfg)
engine.load(str(CHAR / "checkpoints" / "vlab_emily-e12.ckpt"),
            str(CHAR / "checkpoints" / "vlab_emily_e12_s204_l48.pth"))
ref = str(CHAR / "reference.wav")
rt = (CHAR / "reference.txt").read_text(encoding="utf-8").strip()
for name, kw in COMBOS.items():
    print(f"=== {name}")
    for t in TEXTS:
        parts, sr = [], 48000
        for c in engine.synthesize_stream(t, ref, rt, **kw):
            parts.append(np.asarray(c.samples, np.float32))
            sr = c.samplerate
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        exp = len(t) * 0.22
        print(f"  dur={x.size/sr:5.2f}s 比值={x.size/sr/exp:4.2f} "
              f"拖音={sustain(x, sr):4.2f}s 峰值={float(np.abs(x).max()):.2f} | {t[:10]}")
