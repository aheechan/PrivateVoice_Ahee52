"""定位静音段是否对齐 0.25s 的流式块边界（判断是不是 chunk 级丢音）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
CHUNK = 6000  # 0.25s

for p in sorted((ROOT / "results").glob("quality_*.npy")):
    x = np.load(str(p))
    q = np.abs(x) < 1e-4
    idx = np.where(q)[0]
    runs = []
    if idx.size:
        start = prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                runs.append((start, prev))
                start = i
            prev = i
        runs.append((start, prev))
    print(f"--- {p.name} samples={x.size} dur={x.size/SR:.2f}s")
    for a, b in runs:
        n = b - a + 1
        if n < SR * 0.08:
            continue
        print(f"    silence {n/SR*1000:6.0f}ms at {a/SR:5.2f}s "
              f"| start%chunk={a % CHUNK:4d} end%chunk={(b + 1) % CHUNK:4d}")
