"""采集、播放、设备枚举。

录音契约：先录 5 秒环境噪声 → 估算底噪 → 给出建议增益；录音过程中通过回调
把 dBFS 电平抛给 UI（绿 -18~-12 / 黄 -12~-6 / 红 ≥-6）。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import audio_qc


# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    samplerate: float
    hostapi: str
    default: bool = False

    def label(self) -> str:
        return f"{self.name} ({self.channels}ch @ {int(self.samplerate)}Hz)"


def _hostapi_name(idx: int) -> str:
    try:
        return sd.query_hostapis(idx)["name"]
    except Exception:
        return "?"


def list_devices(kind: str = "input") -> list[DeviceInfo]:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    try:
        default_idx = (sd.default.device[0] if kind == "input" else sd.default.device[1])
    except Exception:
        default_idx = -1
    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get(key, 0)) <= 0:
            continue
        out.append(DeviceInfo(
            index=i,
            name=str(d["name"]),
            channels=int(d[key]),
            samplerate=float(d.get("default_samplerate") or 48000),
            hostapi=_hostapi_name(int(d.get("hostapi", 0))),
            default=(i == default_idx),
        ))
    return out


def default_device(kind: str = "input") -> int | None:
    try:
        idx = sd.default.device[0] if kind == "input" else sd.default.device[1]
        return int(idx) if idx is not None and int(idx) >= 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 电平
# ---------------------------------------------------------------------------
def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    return 20.0 * np.log10(max(rms, 1e-6))


def level_color(db: float) -> str:
    """与规格一致：绿 -18~-12，黄 -12~-6，红 >-6。"""
    if db > -6.0:
        return "red"
    if db > -12.0:
        return "yellow"
    return "green"


@dataclass
class NoiseProfile:
    floor_dbfs: float
    snr_db: float
    suggested_gain_db: float
    message: str


def estimate_noise_profile(noise: np.ndarray, sr: int) -> NoiseProfile:
    floor = dbfs(noise)
    # 目标：说话峰值落在 -12 dBFS 左右；假设人声比底噪高 35 dB
    target_peak = -12.0
    assumed_speech = floor + 35.0
    gain = float(np.clip(target_peak - assumed_speech, -12.0, 24.0))
    if floor > -45.0:
        msg = f"底噪偏高（{floor:.1f} dBFS），建议先换安静环境或加 -{min(12, int(floor + 45))} dB 降噪。"
    elif floor < -65.0:
        msg = f"环境很安静（{floor:.1f} dBFS），保持当前增益即可。"
    else:
        msg = f"底噪 {floor:.1f} dBFS，可用。"
    return NoiseProfile(round(floor, 1), round(assumed_speech - floor, 1), round(gain, 1), msg)


# ---------------------------------------------------------------------------
# 录音
# ---------------------------------------------------------------------------
class Recorder:
    """后台线程录音；UI 通过 on_level / on_done 回调取数据。"""

    def __init__(self, samplerate: int | None = None, channels: int = 1,
                 device: int | None = None):
        self._requested_sr = samplerate
        self.channels = channels
        self.device = device
        self._sr = int(samplerate or 48000)
        self._ch = channels
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self.on_level: Callable[[float], None] | None = None
        self.started_at: float = 0.0

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32)
        else:
            mono = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        self._q.put(mono)

    def _resolve_device(self) -> int | None:
        """配置里的设备索引在新机器上可能失效，失效就回落到系统默认。"""
        idx = self.device
        try:
            if idx is not None:
                info = sd.query_devices(idx)
                if int(info.get("max_input_channels", 0)) > 0:
                    return int(idx)
        except Exception:
            pass
        try:
            default = sd.default.device[0]
            if default is not None and int(default) >= 0:
                return int(default)
        except Exception:
            pass
        raise RuntimeError(
            "没有检测到可用的麦克风。请插上麦克风，或在「设置 → 音频设备」里选一个输入设备。")

    def start(self) -> None:
        if self._recording:
            return
        dev = self._resolve_device()
        try:
            caps = sd.query_devices(dev if dev is not None else sd.default.device[0])
            dev_sr = int(caps.get("default_samplerate") or 48000)
            dev_ch = int(caps.get("max_input_channels") or 1)
        except Exception:
            dev_sr, dev_ch = 48000, 1
        candidates = []
        for sr in ([self._requested_sr] if self._requested_sr else []) + [dev_sr, 48000, 44100]:
            for ch in ([self.channels] if self.channels else []) + [1, 2]:
                if (sr, ch) not in candidates:
                    candidates.append((sr, ch))
        last_err: Exception | None = None
        for sr, ch in candidates:
            if ch > dev_ch:
                continue
            try:
                stream = sd.InputStream(
                    samplerate=sr, channels=ch, device=dev, dtype="float32",
                    blocksize=int(sr * 0.02), callback=self._callback)
                stream.start()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
            self._stream = stream
            self._sr, self._ch = sr, ch
            break
        if self._stream is None:
            raise RuntimeError(
                f"麦克风打不开（设备 {dev}）。可能是被其它程序占用，或系统隐私设置里"
                f"没允许桌面应用使用麦克风。原始错误：{last_err}")
        self._frames.clear()
        self._recording = True
        self.started_at = time.time()
        threading.Thread(target=self._drain, daemon=True).start()

    @property
    def samplerate(self) -> int:
        return self._sr

    @property
    def actual_channels(self) -> int:
        return self._ch

    def _drain(self) -> None:
        while self._recording:
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._frames.append(block)
            if self.on_level:
                self.on_level(dbfs(block))

    def stop(self) -> np.ndarray:
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        time.sleep(0.05)          # 让 drain 线程把尾巴收完
        with self._lock:
            data = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
            self._frames.clear()
        return data

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at if self._recording else 0.0


def record_noise_probe(seconds: float = 5.0, device: int | None = None,
                       samplerate: int = 48000,
                       on_tick: Callable[[float, float], None] | None = None) -> np.ndarray:
    """录一段环境噪声用于估算底噪。on_tick(elapsed, dbfs)。"""
    rec = Recorder(samplerate=samplerate, device=device)
    if on_tick:
        rec.on_level = lambda db: on_tick(rec.elapsed, db)
    rec.start()
    try:
        while rec.elapsed < seconds:
            time.sleep(0.05)
    finally:
        data = rec.stop()
    return data


def save_recording(data: np.ndarray, sr: int, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.asarray(data, dtype=np.float32), sr, subtype="PCM_24")
    return dest


# ---------------------------------------------------------------------------
# 播放
# ---------------------------------------------------------------------------
class Player:
    """最小播放器：支持按设备播放文件或内存里的 float32 音频。"""

    def __init__(self, device: int | None = None):
        self.device = device
        self._stream: sd.OutputStream | None = None

    def play_file(self, path: str | Path, blocking: bool = True) -> None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        self.play(data, sr, blocking=blocking)

    def play(self, data: np.ndarray, sr: int, blocking: bool = True) -> None:
        if data.ndim == 1:
            data = data[:, None]
        sd.play(data, sr, device=self.device, blocking=blocking)

    @staticmethod
    def stop() -> None:
        sd.stop()


def duration_of(path: str | Path) -> float:
    try:
        return float(audio_qc.probe(path).get("duration") or 0.0)
    except Exception:
        return 0.0


__all__ = [
    "DeviceInfo", "list_devices", "default_device", "dbfs", "level_color",
    "NoiseProfile", "estimate_noise_profile", "Recorder", "record_noise_probe",
    "save_recording", "Player", "duration_of",
]
