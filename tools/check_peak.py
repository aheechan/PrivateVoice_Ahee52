"""检查生成音频的峰值与削波比例。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for name in ("v3_output.wav", "v4_upstream.wav", "base_steps4.wav"):
        p = ROOT / "results" / "compare" / name
        if not p.exists():
            print(name, "missing")
            continue
        x, sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        peak = float(np.abs(x).max())
        clipped = float(np.mean(np.abs(x) >= 0.999))
        db = 20 * np.log10(max(peak, 1e-9))
        rms = 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9))
        print(f"{name:18s} sr={sr} peak={peak:.3f} ({db:5.1f} dBFS) "
              f"rms={rms:5.1f} dBFS clipped={clipped*100:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
