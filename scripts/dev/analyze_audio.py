"""分析生成音频里是否存在块边界跳变 / 静音断点。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "results" / "quality"


def analyse(path: Path) -> None:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    d = np.abs(np.diff(x))
    print(f"--- {path.name} sr={sr} dur={len(x)/sr:.2f}s peak={np.abs(x).max():.3f}")
    silent = np.abs(x) < 1e-6
    print(f"    silence ratio={silent.mean():.3f} longest silent run={_longest(silent)/sr:.3f}s")
    print(f"    diff p50={np.median(d):.5f} p99={np.percentile(d, 99):.5f} max={d.max():.5f}")
    step = int(sr * 0.25)
    bounds = np.arange(step, len(x) - 1, step)
    if len(bounds):
        jumps = np.abs(x[bounds] - x[bounds - 1])
        print(f"    chunk-boundary(0.25s) jumps n={len(bounds)} max={jumps.max():.4f} "
              f"p50={np.median(jumps):.5f}")
    # 高频能量占比（颗粒感/噪声多会偏高）
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum() + 1e-12
    for lo, hi in ((0, 4000), (4000, 8000), (8000, 12000), (12000, sr / 2)):
        m = (freqs >= lo) & (freqs < hi)
        print(f"    energy {lo}-{hi}Hz: {100*spec[m].sum()/total:.2f}%")


def _longest(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main() -> int:
    files = sorted(Q.glob("*.wav"))
    if not files:
        print("no wav in", Q)
        return 1
    for f in files:
        analyse(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
