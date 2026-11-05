"""离线整句 ASR：FunASR Paraformer-zh。

输入固定为 16k 单声道 float32；进模型前先做首尾静音修剪（保留 100ms 余量），
避免把按住热键前后的空白也算进识别时间。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import audio_qc


@dataclass
class ASRResult:
    text: str = ""
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    speech_sec: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


def trim_silence(y: np.ndarray, sr: int, floor_db: float = -45.0,
                 pad_ms: int = 100) -> np.ndarray:
    """按帧能量找首尾语音边界，两端各留 pad_ms。"""
    if y.size == 0:
        return y
    frame = max(1, int(sr * 0.02))
    hop = frame
    n = max(1, (y.size - frame) // hop + 1)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    idx = idx[idx.max(axis=1) < y.size]
    if idx.size == 0:
        return y
    frames = y[idx]
    rms = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    voiced = np.where(rms > floor_db)[0]
    if voiced.size == 0:
        return y
    start = max(0, idx[voiced[0], 0] - int(sr * pad_ms / 1000))
    end = min(y.size, idx[voiced[-1], -1] + int(sr * pad_ms / 1000))
    return y[start:end]


class ASREngine:
    """懒加载 + 复用；首次加载约 1–2 秒。"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self.warmed = False

    def _load(self) -> None:
        if self._model is not None:
            return
        if str(self.cfg.get("model", "")).lower().startswith("whisper"):
            from faster_whisper import WhisperModel

            self._backend = "whisper"
            self._model = WhisperModel(
                self.cfg.get("whisper_size", "small"),
                device=self.cfg.get("device", "cuda"),
                compute_type="int8_float16",
                download_root=self.cfg.get("whisper_root") or None,
            )
            return
        self._backend = "funasr"
        from funasr import AutoModel
        self._model = AutoModel(
            model=self.cfg.get("model", "paraformer-zh"),
            device=self.cfg.get("device", "cuda"),
            disable_update=True,
        )

    def _transcribe_whisper(self, samples: np.ndarray, sr: int) -> ASRResult:
        import time as _t

        t0 = _t.perf_counter()
        try:
            import soundfile as _sf  # noqa: F401
            segments, info = self._model.transcribe(
                samples.astype("float32"), language=None, beam_size=5,
                vad_filter=True)
            text = "".join(seg.text for seg in segments).strip()
            return ASRResult(text=text, confidence=0.9,
                             elapsed_ms=(_t.perf_counter() - t0) * 1000,
                             error="")
        except Exception as exc:  # noqa: BLE001
            return ASRResult(text="", confidence=0.0,
                             elapsed_ms=(_t.perf_counter() - t0) * 1000,
                             error=str(exc))

    def unload(self) -> None:
        """释放显存（给训练腾地方）。下次调用会自动重新加载。"""
        with self._lock:
            self._model = None
            self.warmed = False
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def warmup(self) -> float:
        """跑一段 1s 静音，把 CUDA kernel 预热掉。"""
        t0 = time.perf_counter()
        self.transcribe(np.zeros(16000, np.float32))
        self.warmed = True
        return (time.perf_counter() - t0) * 1000

    def transcribe(self, y16: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        if y16.dtype != np.float32:
            y16 = y16.astype(np.float32)
        y16 = trim_silence(y16, 16000,
                           float(self.cfg.get("silence_floor_db", -45.0)),
                           int(self.cfg.get("trim_head_ms", 100)))
        speech_sec = y16.size / 16000.0
        if speech_sec < 0.2:
            return ASRResult(text="", confidence=0.0, speech_sec=speech_sec,
                             elapsed_ms=(time.perf_counter() - t0) * 1000,
                             error="too_short")
        with self._lock:
            try:
                self._load()
                out = self._model.generate(input=y16, batch_size_s=300)
            except Exception as exc:  # noqa: BLE001
                return ASRResult(text="", confidence=0.0, speech_sec=speech_sec,
                                 elapsed_ms=(time.perf_counter() - t0) * 1000,
                                 error=f"{type(exc).__name__}: {exc}")
        text = ""
        if isinstance(out, list) and out:
            text = str(out[0].get("text") or "")
        text = text.replace(" ", "").strip()
        # funasr 1.0.27 的 paraformer 不给逐 token 概率，这里用长度/字符构成做保守估计
        conf = self._heuristic_conf(text)
        return ASRResult(
            text=text, confidence=conf, speech_sec=speech_sec,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _heuristic_conf(text: str) -> float:
        if not text:
            return 0.0
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        ratio = cjk / max(1, len(text))
        if len(text) < 2:
            return 0.3
        return round(min(1.0, 0.4 + 0.6 * ratio), 3)

    @property
    def model(self):
        self._load()
        return self._model


def to_16k(samples: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000:
        return samples.astype(np.float32, copy=False)
    return audio_qc.resample(samples, sr_in, 16000)


__all__ = ["ASREngine", "ASRResult", "trim_silence", "to_16k"]
