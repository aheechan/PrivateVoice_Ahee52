"""对比参考音频与历史生成音频的频段分布，判断"闷"来自素材还是模型。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def bands(path: Path) -> None:
    if not path.exists():
        print(f"--- {path.name}: 不存在")
        return
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum() + 1e-12
    out = []
    for lo, hi in ((0, 1000), (1000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)):
        if lo >= sr / 2:
            continue
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        out.append(f"{lo//1000}-{hi//1000}k:{100*spec[m].sum()/total:5.1f}%")
    print(f"--- {path.name} sr={sr} dur={len(x)/sr:.2f}s | " + "  ".join(out))


def main() -> int:
    store = ROOT / "characters"
    char = next((p for p in store.glob("*/profile.json")), None)
    cands = [
        ROOT / "assets" / "ref.wav",
        ROOT / "results" / "sample.wav",
    ]
    if char:
        d = char.parent
        cands += [d / "reference.wav", d / "dataset" / "preview" / "preview_1.wav"]
    for p in cands:
        bands(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
