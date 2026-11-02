"""TTS 引擎接口（本阶段只服务「试听」）。

对话管线属于 Stage 3，这里只定义抽象与 AquaTTS 适配器，不引入 LLM。
"""
from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import audio_qc


@dataclass
class AudioChunk:
    samples: np.ndarray
    samplerate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(self.samplerate or 1)


class TTSEngine(ABC):
    """所有引擎的最小契约。"""

    name: str = "base"

    @abstractmethod
    def load(self, gpt_ckpt: str, sovits_ckpt: str) -> None: ...

    @abstractmethod
    def synthesize_stream(self, text: str, reference: str, reference_text: str,
                          **kwargs: Any) -> Iterator[AudioChunk]: ...

    def synthesize(self, text: str, reference: str, reference_text: str,
                   **kwargs: Any) -> tuple[np.ndarray, int]:
        sr = 24000
        parts: list[np.ndarray] = []
        for chunk in self.synthesize_stream(text, reference, reference_text, **kwargs):
            sr = chunk.samplerate
            parts.append(np.asarray(chunk.samples, dtype=np.float32))
        if not parts:
            return np.zeros(0, np.float32), sr
        return np.concatenate(parts), sr

    def synthesize_file(self, text: str, dest: str | Path, reference: str,
                        reference_text: str, **kwargs: Any) -> Path:
        samples, sr = self.synthesize(text, reference, reference_text, **kwargs)
        return audio_qc.save_wav(dest, samples, sr, subtype="PCM_16")

    @property
    def loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class AquaTTSEngine(TTSEngine):
    """GPT-SoVITS v3（经由 Aqua-TTS 的流式封装）。

    Stage 1 结论直接落在这里：
    - `infer_stream` 产出 `(sample_rate, chunk, text)` 三元组，必须按三元组解包；
    - BigVGAN 自定义 CUDA 核在本机编不出来，默认关掉，避免每次启动触发编译。
    """

    name = "gpt-sovits-v3/aqua"

    def __init__(self, cfg: dict[str, Any], gpt_sovits_home: str | Path):
        self.cfg = cfg
        self.home = Path(gpt_sovits_home)
        self._infer = None
        self._loaded = False
        self._ckpt: tuple[str, str] = ("", "")
        self.load_seconds: float = 0.0
        self.last_ttfp_ms: float | None = None

    # -- 加载 ----------------------------------------------------------
    def _prepare_env(self) -> None:
        os.environ.setdefault("GPT_SOVITS_HOME", str(self.home))
        os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")
        if self.cfg.get("disable_bigvgan_kernel", True):
            os.environ["BIGVGAN_USE_CUDA_KERNEL"] = "0"
        os.environ.setdefault("PYTHONUTF8", "1")

    def _detect_version(self, sovits_ckpt: str) -> str:
        """先看 checkpoint 头，判断是 v3 还是 v4；失败再退回路径启发。"""
        repo = Path(self.home)
        for p in (str(repo), str(repo / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast

            _sym, model, _lora = get_sovits_version_from_path_fast(str(sovits_ckpt))
            return str(model)
        except Exception:
            name = str(sovits_ckpt).lower()
            return "v4" if ("_v4" in name or "v4" in Path(name).name) else ""

    def load(self, gpt_ckpt: str, sovits_ckpt: str, force: bool = False) -> None:
        if self._loaded and not force and self._ckpt == (str(gpt_ckpt), str(sovits_ckpt)):
            return
        self._prepare_env()
        # 上游 G2PW 按进程 CWD 解析 GPT_SoVITS/text/G2PWModel
        os.chdir(self.home)
        version = self._detect_version(str(sovits_ckpt))
        if version == "v4":
            # Aqua 0.3.0 只支持到 v3；v4 直接走上游自带推理栈
            self._load_upstream_v4(gpt_ckpt, sovits_ckpt)
            return
        try:
            from aquatts import TTSInferencer

            t0 = time.perf_counter()
            self._infer = TTSInferencer(
                device=self.cfg.get("device", "cuda"),
                gpt_path=str(gpt_ckpt),
                sovits_path=str(sovits_ckpt),
            )
            self.load_seconds = time.perf_counter() - t0
            self._ckpt = (str(gpt_ckpt), str(sovits_ckpt))
            self._loaded = True
            self._backend = "aqua"
            return
        except Exception as exc:  # noqa: BLE001
            if "v4" not in str(exc):
                raise
            self._load_upstream_v4(gpt_ckpt, sovits_ckpt)

    def _load_upstream_v4(self, gpt_ckpt: str, sovits_ckpt: str) -> None:
        import soundfile as sf
        import torch
        import torchaudio

        # torchaudio 2.11 的 load 依赖 torchcodec（未安装），进程内用 soundfile 顶上
        if getattr(torchaudio, "_voicelab_patched", False) is False:
            def _load(path, **kw):  # noqa: ANN001
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T.copy()), int(sr)

            torchaudio.load = _load  # type: ignore[assignment]
            torchaudio._voicelab_patched = True  # type: ignore[attr-defined]

        repo = Path(self.home)
        for p in (str(repo), str(repo / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        pm = repo / "GPT_SoVITS" / "pretrained_models"
        # fast_langdetect 的模型目录必须存在，否则上游语言分段会报错
        (pm / "fast_langdetect").mkdir(parents=True, exist_ok=True)

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        config = {"custom": {
            "device": self.cfg.get("device", "cuda"),
            "is_half": True,
            "version": "v4",
            "t2s_weights_path": str(gpt_ckpt),
            "vits_weights_path": str(pm / "gsv-v4-pretrained" / "s2Gv4.pth"),
            "bert_base_path": str(pm / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(pm / "chinese-hubert-base"),
        }}
        t0 = time.perf_counter()
        self._infer = TTS(TTS_Config(config))
        self._infer.init_t2s_weights(str(gpt_ckpt))
        self._infer.init_vits_weights(str(sovits_ckpt))
        self.load_seconds = time.perf_counter() - t0
        self._ckpt = (str(gpt_ckpt), str(sovits_ckpt))
        self._loaded = True
        self._backend = "upstream-v4"

    def unload(self) -> None:
        self._infer = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # -- 推理 ----------------------------------------------------------
    def synthesize_stream(self, text: str, reference: str, reference_text: str,
                          **kwargs: Any) -> Iterator[AudioChunk]:
        if not self._loaded:
            raise RuntimeError("TTS 引擎尚未加载")
        params: dict[str, Any] = {
            "text": text,
            "ref_audio_path": str(reference),
            "prompt_text": reference_text,
            "text_language": self.cfg.get("text_language", "中文"),
            "prompt_language": self.cfg.get("prompt_language", "中文"),
            "how_to_cut": self.cfg.get("how_to_cut", "按标点符号切"),
            "top_k": self.cfg.get("top_k", 5),
            "top_p": self.cfg.get("top_p", 1.0),
            "temperature": self.cfg.get("temperature", 0.6),
            "speed": self.cfg.get("speed", 1.1),
            "sample_steps": self.cfg.get("sample_steps", 4),
            "enable_cuda_graph": self.cfg.get("enable_cuda_graph", True),
            "enable_static_kv": self.cfg.get("enable_static_kv", True),
            "chunk_size_seconds": self.cfg.get("chunk_size_seconds", 0.25),
        }
        params.update(kwargs)
        if getattr(self, "_backend", "aqua") == "upstream-v4":
            yield from self._stream_upstream(text, reference, reference_text, **kwargs)
            return
        t0 = time.perf_counter()
        first = True
        for item in self._infer.infer_stream(**params):
            # 三元组 (sr, chunk, text)
            if isinstance(item, tuple) and len(item) == 3:
                sr, chunk, _seg = item
            elif isinstance(item, tuple) and len(item) == 2:
                sr, chunk = item
            else:
                continue
            if chunk is None:
                continue
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            if first:
                self.last_ttfp_ms = (time.perf_counter() - t0) * 1000.0
                first = False
            yield AudioChunk(samples=arr, samplerate=int(sr or 24000))

    def _stream_upstream(self, text: str, reference: str, reference_text: str,
                         **kwargs: Any) -> Iterator[AudioChunk]:
        """v4：走上游 TTS.run()，cut5 切分下首包约 1.1s、RTF≈0.6。"""
        inputs = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": str(reference),
            "prompt_text": reference_text,
            "prompt_lang": "zh",
            "top_k": self.cfg.get("top_k", 5),
            "top_p": self.cfg.get("top_p", 1.0),
            "temperature": self.cfg.get("temperature", 0.6),
            "text_split_method": self.cfg.get("v4_split_method", "cut5"),
            "batch_size": 1,
            "speed_factor": self.cfg.get("speed", 1.1),
            "streaming_mode": True,
            "return_fragment": True,
            "fragment_interval": self.cfg.get("fragment_interval", 0.3),
            "seed": 42,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": self.cfg.get("sample_steps", 4),
            "super_sampling": False,
        }
        inputs.update({k: v for k, v in kwargs.items() if k in inputs})
        t0 = time.perf_counter()
        first = True
        for chunk in self._infer.run(inputs):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                sr, audio = chunk
            else:
                sr, audio = 48000, chunk
            if audio is None:
                continue
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            # v4 上游返回的是 int16 量纲的 float（峰值到 1e4 量级），
            # 不归一化会直接把播放端怼到 0 dBFS 削波（听起来就是"底噪/糊"）。
            peak = float(np.max(np.abs(arr)))
            if peak > 2.0:
                arr = arr / 32768.0
            if first:
                self.last_ttfp_ms = (time.perf_counter() - t0) * 1000.0
                first = False
            yield AudioChunk(samples=arr, samplerate=int(sr or 48000))

    # -- 诊断 ----------------------------------------------------------
    def backend_info(self) -> dict[str, Any]:
        if not self._infer:
            return {}
        decoder = getattr(getattr(self._infer, "t2s_model", None), "model", None)
        return {
            "bigvgan_kernel_mode": getattr(self._infer, "bigvgan_kernel_mode", None),
            "flash_attn_active": bool(getattr(decoder, "use_flash_attn_kvcache", False)),
            "kv_cache_layout": getattr(decoder, "kv_cache_layout", None),
            "bucketed_graphs": len(getattr(decoder, "bucket_graphs", {}) or {}),
        }


def make_engine(config: Any) -> TTSEngine:
    return AquaTTSEngine(config.data.get("tts", {}), config.path_of("gpt_sovits"))


def preview_sentences(config: Any) -> list[str]:
    return list(config.data.get("tts", {}).get("preview_texts", []))


def synthesize_preview(engine: TTSEngine, config: Any, character: dict[str, Any],
                       out_dir: str | Path, progress=None) -> list[Path]:
    """三句固定文本合成，供向导第 6 步试听。"""
    voice = character.get("voice", {})
    gpt = voice.get("gpt_ckpt") or ""
    sovits = voice.get("sovits_ckpt") or ""
    char_dir = Path(character.get("_dir", ""))
    ref = char_dir / (voice.get("reference") or "reference.wav")
    ref_text = voice.get("reference_text") or ""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine.load(gpt, sovits)
    outs: list[Path] = []
    for i, text in enumerate(preview_sentences(config), start=1):
        if progress:
            progress(i, len(preview_sentences(config)), text)
        dest = out_dir / f"preview_{i}.wav"
        engine.synthesize_file(text, dest, str(ref), ref_text)
        outs.append(dest)
    return outs


__all__ = ["AudioChunk", "TTSEngine", "AquaTTSEngine", "make_engine",
           "preview_sentences", "synthesize_preview"]
