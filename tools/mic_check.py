"""麦克风自检：设备 / 实际采样率 / 电平 / 底噪 / 录一句能不能识别。

在 5070 Ti 上先跑这个，再跑应用，可以少走弯路：
    runtime\\python.exe tools\\mic_check.py           # 录 5 秒
    runtime\\python.exe tools\\mic_check.py --sec 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_io  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=5.0)
    args = ap.parse_args()

    print("== 输入设备 ==")
    try:
        devs = audio_io.list_devices("input")
    except Exception as exc:  # noqa: BLE001
        print("  枚举失败：", exc)
        return 1
    if not devs:
        print("  ❌ 没有检测到任何输入设备（麦克风未插/被禁用）")
        return 1
    for d in devs:
        print(f"  {'*' if d.default else ' '} [{d.index}] {d.name} · "
              f"{d.channels}ch · {int(d.samplerate)}Hz · {d.hostapi}")
    try:
        print("  默认输入:", audio_io.default_device("input"))
    except Exception:
        pass

    cfg = Config.load()
    rec = audio_io.Recorder(device=cfg.audio.get("input_device"))
    levels: list[float] = []
    rec.on_level = levels.append
    print(f"\n== 录音 {args.sec:.0f} 秒，请正常说话 ==")
    try:
        rec.start()
    except Exception as exc:  # noqa: BLE001
        print("  ❌ 打不开麦克风：", exc)
        return 1
    print(f"  实际采样率 {rec.samplerate} Hz，声道 {rec.actual_channels}")
    t0 = time.time()
    while time.time() - t0 < args.sec:
        time.sleep(0.1)
        if int((time.time() - t0) * 5) % 5 == 0 and levels:
            print(f"  {time.time()-t0:4.1f}s  电平 {levels[-1]:6.1f} dBFS", end="\r")
    data = rec.stop()
    dur = len(data) / rec.samplerate
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = 20 * np.log10(max(float(np.sqrt(np.mean(data ** 2))), 1e-9)) if data.size else -120
    zero = float(np.mean(np.abs(data) < 1e-6)) if data.size else 1.0
    print(f"\n  录到 {dur:.2f}s，峰值 {20*np.log10(max(peak,1e-9)):.1f} dBFS，"
          f"RMS {rms:.1f} dBFS，静音占比 {zero*100:.0f}%")

    if peak < 1e-4:
        print("  ❌ 全是静音：检查麦克风是否被静音、是否选错设备、系统隐私里是否允许桌面应用用麦克风")
        return 1
    if zero > 0.9:
        print("  ⚠️ 静音太多，可能录到了错误的设备")

    print("\n== 送 ASR（走应用同一条链路）==")
    try:
        from app.core.asr import ASREngine, to_16k

        engine = ASREngine(cfg.data.get("asr", {}))
        res = engine.transcribe(to_16k(data, rec.samplerate))
        print(f"  识别：{res.text!r}  置信度 {res.confidence}  "
              f"耗时 {res.elapsed_ms:.0f}ms  错误 {res.error or '无'}")
        if not res.ok:
            print("  ⚠️ 识别为空：可能音量太低或说得太短（>1 秒、离麦 15-20cm）")
    except Exception as exc:  # noqa: BLE001
        print("  ASR 失败：", exc)

    out = ROOT / "results" / "mic_check.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_io.save_recording(data, rec.samplerate, out)
    print("\n样本已保存：", out)
    print("MIC_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
