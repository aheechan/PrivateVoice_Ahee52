"""用上游 GPT-SoVITS 自带推理栈跑 v4（Aqua 0.3.0 不支持 v4）。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
REPO = ROOT / "GPT-SoVITS"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "GPT_SoVITS"))
os.chdir(REPO)
os.environ.setdefault("PYTHONPATH", f"{REPO};{REPO / 'GPT_SoVITS'}")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = REPO / "GPT_SoVITS" / "pretrained_models"
TEXT = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"
REF = r"C:\Local Repositories\PrivateVoice\characters\格雷丝-748caf\reference.wav"
REF_TXT = Path(r"C:\Local Repositories\PrivateVoice\characters\格雷丝-748caf\reference.txt").read_text(
    encoding="utf-8").strip()
LORA = REPO / "SoVITS_weights_v4" / "vlab_格雷丝_e8_s200_l32.pth"
GPT = REPO / "GPT_weights_v3" / "vlab_格雷丝-e8.ckpt"


def main() -> int:
    cfg = {
        "custom": {
            "device": "cuda", "is_half": True, "version": "v4",
            "t2s_weights_path": str(PM / "s1v3.ckpt"),
            "vits_weights_path": str(PM / "s2Gv4.pth"),
            "bert_base_path": str(PM / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(PM / "chinese-hubert-base"),
        }
    }
    t0 = time.perf_counter()
    tts = TTS(TTS_Config(cfg))
    print(f"load {time.perf_counter()-t0:.1f}s")
    tts.init_t2s_weights(str(GPT))
    tts.init_vits_weights(str(LORA))

    inputs = {
        "text": TEXT, "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": REF_TXT, "prompt_lang": "zh",
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
    print(f"sr={sr} dur={dur:.2f}s synth={el:.2f}s rtf={el/dur:.2f} "
          f"ttfp={first:.0f}ms chunks={len(parts)}")
    out = ROOT / "results" / "compare" / "v4_upstream.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), x, sr, subtype="PCM_16")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
