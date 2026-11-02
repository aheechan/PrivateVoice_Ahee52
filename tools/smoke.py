"""Smoke test: load v3 pipeline through Aqua and synthesize one short line."""
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["GPT_SOVITS_HOME"] = r"C:\VoiceLab\GPT-SoVITS"
os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")
os.environ.setdefault("ENABLE_CUDA_GRAPH_PRECAPTURE", "1")
if len(sys.argv) > 1 and sys.argv[1] == "--nokernel":
    os.environ["BIGVGAN_USE_CUDA_KERNEL"] = "0"

# upstream G2PW reads GPT_SoVITS/text/G2PWModel relative to CWD
os.chdir(os.environ["GPT_SOVITS_HOME"])

import numpy as np
import soundfile as sf
import torch

from aquatts import TTSInferencer

PM = r"C:\VoiceLab\GPT-SoVITS\GPT_SoVITS\pretrained_models"
REF = r"C:\VoiceLab\assets\ref.wav"
REFTXT = open(r"C:\VoiceLab\assets\ref.txt", encoding="utf-8").read().strip()

print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
t0 = time.perf_counter()
tts = TTSInferencer(device="cuda", gpt_path=PM + r"\s1v3.ckpt", sovits_path=PM + r"\s2Gv3.pth")
print("loaded in %.1fs" % (time.perf_counter() - t0), flush=True)
print("bigvgan_kernel_mode:", getattr(tts, "bigvgan_kernel_mode", None), flush=True)
d = tts.t2s_model.model
print("flash_attn_kvcache:", getattr(d, "use_flash_attn_kvcache", None))
print("kv_cache_layout:", getattr(d, "kv_cache_layout", None))
print("bucket_graphs:", len(getattr(d, "bucket_graphs", {}) or {}))
print("vram allocated %.2f GB reserved %.2f GB"
      % (torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3))

parts = []
sr = None
t0 = time.perf_counter()
first = None
for out_sr, chunk, seg in tts.infer_stream(
    text="今天有点累",
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
    sr = out_sr
    if chunk is None or len(chunk) == 0:
        continue
    if first is None:
        first = time.perf_counter() - t0
        print("first chunk ttfp=%.1f ms" % (first * 1000), flush=True)
    parts.append(chunk)
total = time.perf_counter() - t0
audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
dur = len(audio) / float(sr or 24000)
print("total=%.1f ms audio=%.2fs rtf=%.2f chunks=%d peak=%.2f GB"
      % (total * 1000, dur, total / dur if dur else 0, len(parts),
         torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)
os.makedirs(r"C:\VoiceLab\results", exist_ok=True)
sf.write(r"C:\VoiceLab\results\smoke.wav", audio, int(sr or 24000))
print("wrote results/smoke.wav")
