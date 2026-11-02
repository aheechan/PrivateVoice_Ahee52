"""硬件门禁：不支持的卡必须拒绝，5070 Ti（sm_120）必须放行。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.core import train  # noqa: E402


@pytest.mark.parametrize("cc,vram,expect", [
    ((7, 5), 4.0, False),    # GTX 1650：算力不支持
    ((8, 6), 8.0, False),    # 3060 8G：显存不够
    ((8, 6), 12.0, True),    # 3060 12G
    ((8, 9), 24.0, True),    # 4090
    ((12, 0), 16.0, True),   # 5070 Ti
])
def test_guard(monkeypatch, cc, vram, expect):
    info = train.GpuInfo(name="test", total_gb=vram, free_gb=vram - 2,
                         compute_cap=cc, torch="x")
    monkeypatch.setattr(train, "gpu_info", lambda: info)
    ok, msg = train.check_hardware({})
    assert ok is expect, msg


def test_guard_no_gpu(monkeypatch):
    monkeypatch.setattr(train, "gpu_info", lambda: train.GpuInfo())
    ok, msg = train.check_hardware({})
    assert not ok
    assert "没有检测到" in msg
