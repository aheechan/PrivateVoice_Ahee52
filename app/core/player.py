"""低延迟播放：sounddevice OutputStream + 环形缓冲 + 打断。

链路里只有一个重采样点：TTS 出 24k float32，这里按设备采样率（通常 48k）
用 soxr 转一次，然后进缓冲。
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from . import audio_qc


class StreamingResampler:
    """跨 chunk 连续的重采样：保留上一块尾部，避免块边界爆音。"""

    TAIL = 64

    def __init__(self, sr_in: int, sr_out: int):
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.ratio = sr_out / max(1, sr_in)
        self.tail: np.ndarray | None = None

    def reset(self) -> None:
        self.tail = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if x.size == 0:
            return x
        if self.sr_in == self.sr_out:
            return x
        had_tail = self.tail is not None and self.tail.size > 0
        buf = np.concatenate([self.tail, x]) if had_tail else x
        out = audio_qc.resample(buf, self.sr_in, self.sr_out)
        self.tail = buf[-self.TAIL:].copy() if buf.size > self.TAIL else buf.copy()
        drop = int(self.TAIL * self.ratio) if had_tail else 0
        return out[drop:] if drop < out.size else np.zeros(0, np.float32)


class Player:
    def __init__(self, device: int | None = None, samplerate: int | None = None,
                 blocksize_ms: int = 30, sentence_gap_ms: int = 80,
                 limiter_dbtp: float = -1.0):
        self.config_device = device          # 用户选择：None = 系统默认
        self.device = self._resolve_output(device)
        # 用设备自己的默认采样率，避免 Windows 再插一层系统重采样
        if samplerate is None:
            try:
                info = sd.query_devices(self.device if self.device is not None
                                        else sd.default.device[1])
                samplerate = int(info["default_samplerate"]) or 48000
            except Exception:
                samplerate = 48000
        self.samplerate = int(samplerate)
        self.blocksize_ms = max(10, int(blocksize_ms))
        self.gap_ms = max(0, int(sentence_gap_ms))
        self.blocksize = max(1, int(samplerate * self.blocksize_ms / 1000))
        self.gap = np.zeros(int(samplerate * sentence_gap_ms / 1000), np.float32)
        self.limiter_dbtp = limiter_dbtp
        self._buf: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._playing = False
        self._abort = threading.Event()
        self._resampler = StreamingResampler(24000, samplerate)
        self.t_play_start: float | None = None
        self.played_samples = 0
        self.underruns = 0
        self.dump = None          # 调试：把实际送出的音频写成原始 float32

    # -- 设备 ----------------------------------------------------------
    def set_gap_ms(self, ms: int) -> None:
        self.gap_ms = max(0, int(ms))
        self.gap = np.zeros(int(self.samplerate * self.gap_ms / 1000), np.float32)

    def _resolve_output(self, device: int | None) -> int | None:
        """device=None 表示“系统默认”：每次实时解析，避免缓存过期。"""
        if device is not None:
            try:
                sd.query_devices(device)
                return device
            except Exception:
                pass
        try:
            info = sd.query_devices(kind="output")
            idx = sd.default.device[1]
            # 若默认设备不可用，退回 query 到的设备索引
            try:
                sd.query_devices(idx)
                return idx
            except Exception:
                return info.get("index")
        except Exception:
            return None

    def set_device(self, device: int | None) -> None:
        self.config_device = device
        self.device = self._resolve_output(device)

    def _open(self) -> None:
        """优先 WASAPI：MME 在欠载时会把上一块重复播出（听起来像"某个字重复"），
        WASAPI 则是纯静音补齐。失败再退回系统默认。"""
        if self._stream is not None:
            return
        # 每次开流都重新解析“系统默认”，用户中途切换设备也能跟上
        self.device = self._resolve_output(self.config_device)
        candidates: list[dict] = []
        try:
            hostapis = sd.query_hostapis()
            wasapi = next((i for i, h in enumerate(hostapis)
                           if "WASAPI" in h["name"].upper()), None)
            if self.device is not None:
                candidates.append({"device": self.device})
            else:
                # 优先"默认输出设备"，但只在它属于 WASAPI 时保留；
                # 否则先试默认设备（可能是 MME），再试 WASAPI 设备。
                try:
                    default_out = sd.default.device[1]
                except Exception:
                    default_out = None
                if default_out is not None and int(default_out) >= 0:
                    info = sd.query_devices(int(default_out))
                    if wasapi is None or int(info["hostapi"]) == wasapi:
                        candidates.append({"device": int(default_out)})
                if wasapi is not None:
                    devs = [i for i, d in enumerate(sd.query_devices())
                            if d["max_output_channels"] > 0 and d["hostapi"] == wasapi]
                    for d in devs[:4]:
                        candidates.append({"device": d})
        except Exception:
            pass
        candidates.append({"device": self.device})
        last: Exception | None = None
        for cand in candidates:
            try:
                sr = self.samplerate
                try:
                    info = sd.query_devices(cand["device"] if cand["device"] is not None
                                            else sd.default.device[1])
                    sr = int(info["default_samplerate"]) or sr
                except Exception:
                    pass
                stream = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="float32",
                    blocksize=max(1, int(sr * self.blocksize_ms / 1000)),
                    device=cand["device"], callback=self._callback)
                stream.start()
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
            self._stream = stream
            self.samplerate = sr
            self.blocksize = max(1, int(sr * self.blocksize_ms / 1000))
            self.gap = np.zeros(int(sr * self.gap_ms / 1000), np.float32)
            try:
                info = sd.query_devices(stream.device)
                self.hostapi_name = sd.query_hostapis(info["hostapi"])["name"]
                self.device_name = info["name"]
            except Exception:
                pass
            return
        raise RuntimeError(f"打不开音频输出设备：{last}")

    def _callback(self, outdata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        chunk = np.zeros(frames, np.float32)
        got = 0
        with self._lock:
            while got < frames and self._buf:
                head = self._buf[0]
                need = frames - got
                if len(head) <= need:
                    chunk[got:got + len(head)] = head
                    got += len(head)
                    self._buf.popleft()
                else:
                    chunk[got:got + need] = head[:need]
                    self._buf[0] = head[need:]
                    got += need
            if got < frames:
                self.underruns += 1
                if not self._buf:
                    self._playing = False
        if got > 0 and self.t_play_start is None:
            self.t_play_start = time.perf_counter()
        outdata[:, 0] = chunk
        if self.dump is not None:
            try:
                self.dump.write(chunk.tobytes())
            except Exception:
                self.dump = None

    # -- 播放控制 ------------------------------------------------------
    def begin_utterance(self) -> None:
        """一轮回复开始：**先清空上一轮残留的缓冲**，再重置计时。

        以前这里只重置计时，如果上一轮被打断、或末尾的 80ms 句间静音还在队列里，
        新的一轮开口就会先放出一小段旧音频——听感就是"开头有一截延音"。
        """
        with self._lock:
            self._buf.clear()
            self._playing = False
        self.t_play_start = None
        self.played_samples = 0
        self.underruns = 0
        self._abort.clear()
        self._resampler.reset()

    def push(self, samples: np.ndarray, sr: int = 24000) -> None:
        if samples is None or len(samples) == 0:
            return
        if sr != self._resampler.sr_in:
            self._resampler = StreamingResampler(sr, self.samplerate)
        samples = self._resampler.process(np.asarray(samples, np.float32))
        if samples.size == 0:
            return
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        ceiling = 10 ** (self.limiter_dbtp / 20.0)
        if peak > ceiling:
            samples = audio_qc.limit_peak(samples, self.samplerate,
                                          self.limiter_dbtp)
        with self._lock:
            self._buf.append(np.asarray(samples, np.float32))
            self._playing = True
        self._open()

    def push_gap(self) -> None:
        with self._lock:
            self._buf.append(self.gap.copy())

    def end_utterance(self) -> None:
        """不再有新数据；调用方可以 wait_done() 等播完。"""

    def wait_done(self, timeout: float | None = None) -> bool:
        t0 = time.perf_counter()
        while True:
            with self._lock:
                empty = not self._buf
                playing = self._playing
            if empty and not playing:
                return True
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return False
            time.sleep(0.01)

    def interrupt(self) -> float:
        """打断：清空缓冲 + 停流。返回耗时（ms），用于验证 ≤150ms。"""
        t0 = time.perf_counter()
        self._abort.set()
        with self._lock:
            self._buf.clear()
            self._playing = False
        self._close()
        return (time.perf_counter() - t0) * 1000

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass

    def close(self) -> None:
        self.interrupt()

    # -- 状态 ----------------------------------------------------------
    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing or bool(self._buf)

    @property
    def buffered_sec(self) -> float:
        with self._lock:
            return sum(len(x) for x in self._buf) / self.samplerate


__all__ = ["Player"]
