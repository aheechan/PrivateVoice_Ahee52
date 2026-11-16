"""用上游 GPT-SoVITS 推理栈跑 v4（Aqua 0.3.0 不支持 v4）。

torchaudio 2.11 的 load 依赖 torchcodec（未安装），这里在进程内用 soundfile 顶上，
不改上游源码。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
REPO = ROOT / "GPT-SoVITS"
for p in (str(REPO), str(REPO / "GPT_SoVITS")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _load_with_soundfile(path, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), int(sr)


torchaudio.load = _load_with_soundfile

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = REPO / "GPT_SoVITS" / "pretrained_models"
TEXT = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"
REF = ROOT / "characters" / "格雷丝-748caf" / "reference.wav"
REF_TXT = (ROOT / "characters" / "格雷丝-748caf" / "reference.txt").read_text(
    encoding="utf-8").strip()
LORA = REPO / "SoVITS_weights_v4" / "vlab_格雷丝_e8_s200_l32.pth"
GPT = REPO / "GPT_weights_v3" / "vlab_格雷丝-e8.ckpt"


def synth(tag: str, version: str, base_sovits: Path, lora: Path | None) -> None:
    cfg = {"custom": {
        "device": "cuda", "is_half": True, "version": version,
        "t2s_weights_path": str(PM / "s1v3.ckpt"),
        "vits_weights_path": str(base_sovits),
        "bert_base_path": str(PM / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(PM / "chinese-hubert-base"),
    }}
    t0 = time.perf_counter()
    tts = TTS(TTS_Config(cfg))
    load_s = time.perf_counter() - t0
    tts.init_t2s_weights(str(GPT))
    if lora is not None:
        tts.init_vits_weights(str(lora))

    inputs = {
        "text": TEXT, "text_lang": "zh",
        "ref_audio_path": str(REF), "prompt_text": REF_TXT, "prompt_lang": "zh",
        "top_k": 5, "top_p": 1.0, "temperature": 0.6, "text_split_method": "cut5",
        "batch_size": 1, "speed_factor": 1.1, "streaming_mode": True,
        "return_fragment": True, "fragment_interval": 0.3,
        "seed": 42, "parallel_infer": True, "repetition_penalty": 1.35,
        "sample_steps": 4, "super_sampling": False,
    }
    parts, sr = [], 48000
    first = None
    t0 = time.perf_counter()
    for chunk in tts.run(inputs):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        if isinstance(chunk, tuple) and len(chunk) == 2:
            sr, audio = chunk
        else:
            audio = chunk
        if audio is None:
            continue
        parts.append(np.asarray(audio, np.float32).reshape(-1))
    el = time.perf_counter() - t0
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    dur = x.size / sr
    out = ROOT / "results" / "compare" / f"{tag}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), x, sr, subtype="PCM_16")
    print(f"[{tag}] sr={sr} dur={dur:.2f}s load={load_s:.1f}s synth={el:.2f}s "
          f"rtf={el/dur:.2f} ttfp={first:.0f}ms chunks={len(parts)} -> {out}")


def main() -> int:
    synth("v4_upstream", "v4", PM / "gsv-v4-pretrained" / "s2Gv4.pth", LORA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
