"""Render results/sample.wav for listening review (default settings)."""
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["GPT_SOVITS_HOME"] = r"C:\VoiceLab\GPT-SoVITS"
os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")
os.chdir(os.environ["GPT_SOVITS_HOME"])

import numpy as np
import soundfile as sf
import torch

from aquatts import TTSInferencer

PM = r"C:\VoiceLab\GPT-SoVITS\GPT_SoVITS\pretrained_models"
REF = r"C:\VoiceLab\assets\ref.wav"
REFTXT = open(r"C:\VoiceLab\assets\ref.txt", encoding="utf-8").read().strip()

LINES = [
    "今天有点累",
    "今天有点累不过还好，晚上想吃点热乎的汤面吧",
    "今天有点累不过还好，下午开了两个会脑子有点转不动，晚上想吃点热的然后早点睡觉，明天再慢慢处理剩下的所有事情",
]

tts = TTSInferencer(device="cuda", gpt_path=PM + r"\s1v3.ckpt", sovits_path=PM + r"\s2Gv3.pth")
print("kernel:", getattr(tts, "bigvgan_kernel_mode", None), flush=True)

sr_out = 24000
pieces = []
for line in LINES:
    chunks = []
    sr = 24000
    for out_sr, chunk, _seg in tts.infer_stream(
        text=line,
        ref_audio_path=REF,
        prompt_text=REFTXT,
        text_language="中文",
        prompt_language="中文",
        how_to_cut="按标点符号切",
        top_k=5, top_p=1, temperature=0.6,
        speed=1.1, sample_steps=4,
        enable_cuda_graph=True, enable_static_kv=True,
        chunk_size_seconds=0.25,
    ):
        sr = out_sr or sr
        if chunk is not None and len(chunk):
            chunks.append(chunk)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sr_out = int(sr)
    pieces.append(audio)
    pieces.append(np.zeros(int(sr_out * 0.5), dtype=np.float32))
    print("rendered %.2fs: %s" % (len(audio) / sr, line[:16]), flush=True)

full = np.concatenate(pieces[:-1])
sf.write(r"C:\VoiceLab\results\sample.wav", full, sr_out, subtype="PCM_16")
print("wrote sample.wav  %.2fs @ %d Hz" % (len(full) / sr_out, sr_out))
