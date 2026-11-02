"""质量契约的三个必测点：削波、静音、过短。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> Config:
    c = Config.load()
    return c


def _write(tmp_path: Path, seconds: float, sr: int = 48000,
           amplitude: float = 0.3, silence_ratio: float = 0.0) -> Path:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    # 类语音：200Hz 基频 + 谐波，带 3Hz 包络
    x = (np.sin(2 * np.pi * 200 * t) + 0.4 * np.sin(2 * np.pi * 400 * t))
    x *= (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = amplitude * x / max(1e-9, np.max(np.abs(x)))
    x = x.astype(np.float32)
    if silence_ratio > 0:
        cut = int(n * (1 - silence_ratio))
        x[cut:] = 0.0
    path = tmp_path / f"tone_{seconds:.0f}s_{sr}.wav"
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


def test_short_audio_rejected(tmp_path, cfg):
    p = _write(tmp_path, 3.0)
    r = audio_qc.run_qc(p, cfg, check_speakers=False)
    assert r.status == "error"
    dur_check = next(c for c in r.checks if c.key == "duration")
    assert dur_check.level == "error"
    assert "不足" in dur_check.message


def test_clipping_warned(tmp_path, cfg):
    p = _write(tmp_path, 50.0, amplitude=1.6)   # 明显过载
    r = audio_qc.run_qc(p, cfg, check_speakers=False)
    clip = next(c for c in r.checks if c.key == "clipping")
    assert clip.level == "warn"
    assert r.clip_ratio > 0


def test_silence_warned(tmp_path, cfg):
    p = _write(tmp_path, 50.0, silence_ratio=0.45)
    r = audio_qc.run_qc(p, cfg, check_speakers=False)
    sil = next(c for c in r.checks if c.key == "silence")
    assert sil.level == "warn"
    assert r.silence_ratio >= 0.2


def test_loudness_normalized(tmp_path, cfg):
    p = _write(tmp_path, 50.0, amplitude=0.05)
    r = audio_qc.run_qc(p, cfg, check_speakers=False)
    assert abs(r.lufs - (-23.0)) < 1.5
    assert r.peak_dbfs <= -1.0 + 0.2


def test_low_sample_rate_rejected(tmp_path, cfg):
    p = _write(tmp_path, 50.0, sr=16000)
    r = audio_qc.run_qc(p, cfg, check_speakers=False)
    sr_check = next(c for c in r.checks if c.key == "sample_rate")
    assert sr_check.level == "error"
