"""最终对比：v3 vs v4（修复 int16 量纲之后）的峰值、削波与频段。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def report(name: str) -> None:
    p = ROOT / "results" / "compare" / name
    if not p.exists():
        print(f"{name}: missing")
        return
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    peak = float(np.abs(x).max())
    clipped = 100 * float(np.mean(np.abs(x) >= 0.999))
    spec = np.abs(np.fft.rfft(x.astype("float64"))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tot = spec.sum() + 1e-12

    def hf(lo: int, hi: int) -> float:
        return 100 * spec[(f >= lo) & (f < min(hi, sr / 2))].sum() / tot

    print(f"{name:18s} sr={sr} peak={peak:.3f} clip={clipped:5.2f}% | "
          f"4-8k={hf(4000, 8000):5.2f}% 8-12k={hf(8000, 12000):5.2f}% "
          f"12-16k={hf(12000, 16000):5.2f}% 16-24k={hf(16000, 24000):5.2f}%")


def main() -> int:
    for n in ("v3_output.wav", "v4_output.wav", "v4_upstream.wav"):
        report(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
