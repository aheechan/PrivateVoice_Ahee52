"""量一下 v4 输出的底噪：静音段电平、频谱，与参考音频/v3 对比。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def frames(x: np.ndarray, sr: int, win_ms: float = 30, hop_ms: float = 10):
    w = int(sr * win_ms / 1000)
    h = int(sr * hop_ms / 1000)
    n = max(0, 1 + (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx]


def report(path: Path, tag: str) -> None:
    if not path.exists():
        print(f"--- {tag}: 不存在")
        return
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    fr = frames(x.astype(np.float64), sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    noise_db = float(np.percentile(db, 10))
    speech_db = float(np.percentile(db, 90))
    # 取最安静的那一帧做噪声频谱
    quiet = fr[int(np.argmin(db))]
    spec = np.abs(np.fft.rfft(quiet * np.hanning(len(quiet)))) ** 2
    freqs = np.fft.rfftfreq(len(quiet), 1 / sr)
    tot = spec.sum() + 1e-12
    hf = 100 * spec[(freqs >= 4000)].sum() / tot
    print(f"--- {tag:12s} sr={sr} dur={len(x)/sr:.2f}s "
          f"noise_floor={noise_db:6.1f} dBFS  speech={speech_db:6.1f} dBFS "
          f"snr≈{speech_db-noise_db:5.1f} dB  静音帧4k+占比={hf:4.1f}%")


def main() -> int:
    cmp_dir = ROOT / "results" / "compare"
    for name, tag in (("v3_output.wav", "v3"),
                      ("v4_upstream.wav", "v4"),
                      ("base_steps4.wav", "base-v3")):
        report(cmp_dir / name, tag)
    report(ROOT / "characters" / "格雷丝-748caf" / "reference.wav", "reference")
    report(ROOT / "assets" / "ref.wav", "ref-原始")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
