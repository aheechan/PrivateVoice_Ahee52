"""Stage 1 TTFP spike: GPT-SoVITS v3 + Aqua-TTS 0.3.0 on RTX 3090.

Measures first-packet latency (TTFP), total generation time, audio duration,
RTF and peak VRAM for three text lengths (5 / 20 / 50 Chinese chars).

Two groups:
  --group default    auto: let Aqua use the custom BigVGAN CUDA kernel if it can
  --group nokernel   force BIGVGAN_USE_CUDA_KERNEL=0 -> pure PyTorch fallback

Measurement settings mirror benchmarks/aqua_ttfp.py so numbers are comparable
with the Aqua README (top_k=5, top_p=1, temperature=0.6, speed=1.1,
sample_steps=4, how_to_cut="按标点符号切", chunk_size_seconds=0.25).

Usage:
    set GPT_SOVITS_HOME=C:\\Local Repositories\\PrivateVoice\\GPT-SoVITS
    python benchmark.py --group default
    python benchmark.py --group nokernel
    python benchmark.py --merge          # combine both groups into results/spike.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
GPT_SOVITS = ROOT / "GPT-SoVITS"
PM = GPT_SOVITS / "GPT_SoVITS" / "pretrained_models"
GPT_CKPT = PM / "s1v3.ckpt"
SOVITS_CKPT = PM / "s2Gv3.pth"
REF = ROOT / "assets" / "ref.wav"
REF_TXT = ROOT / "assets" / "ref.txt"
REF_TEXT_STR = REF_TXT.read_text(encoding="utf-8").strip()
RESULTS = ROOT / "results"

SEED = 20260926

# 5 / 20 / 50 Chinese characters (punctuation excluded from the count below)
TEXTS = {
    "short": "今天有点累",
    "medium": "今天有点累不过还好，晚上想吃点热乎的汤面吧",
    "long": ("今天有点累不过还好，下午开了两个会脑子有点转不动，"
             "晚上想吃点热的然后早点睡觉，明天再慢慢处理剩下的所有事情"),
}

# aqua_ttfp.py protocol
TOP_K, TOP_P, TEMPERATURE = 5, 1, 0.6
SPEED, SAMPLE_STEPS = 1.1, 4
HOW_TO_CUT = "按标点符号切"
CHUNK_SECONDS = 0.25


def cjk_len(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def pct(values, q):
    """Nearest-rank percentile (same convention as Aqua's reports)."""
    xs = sorted(values)
    idx = max(0, math.ceil(q * len(xs)) - 1)
    return xs[idx]


def run_once(tts, text):
    """One streamed synthesis. Returns timings for that single request."""
    t0 = time.perf_counter()
    first = None
    samples = 0
    sr = None
    chunks = 0
    for out_sr, chunk, _seg in tts.infer_stream(
        text=text,
        ref_audio_path=str(REF),
        prompt_text=REF_TEXT_STR,
        text_language="中文",
        prompt_language="中文",
        how_to_cut=HOW_TO_CUT,
        top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE,
        speed=SPEED, sample_steps=SAMPLE_STEPS,
        enable_cuda_graph=True,
        enable_static_kv=True,
        chunk_size_seconds=CHUNK_SECONDS,
    ):
        if out_sr is not None:
            sr = out_sr
        if chunk is None or len(chunk) == 0:
            continue
        if first is None:
            first = time.perf_counter() - t0
        chunks += 1
        samples += len(chunk)
    total = time.perf_counter() - t0

    audio_sec = samples / float(sr or 24000)
    return {
        "ttfp_ms": (first if first is not None else total) * 1000.0,
        "total_ms": total * 1000.0,
        "audio_sec": audio_sec,
        "rtf": (total / audio_sec) if audio_sec > 0 else None,
        "chunks": chunks,
        "sample_rate": sr,
    }


def bench_tier(tts, torch, name, text, warmup, runs):
    for _ in range(warmup):
        run_once(tts, text)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows = [run_once(tts, text) for _ in range(runs)]

    ttfp = [r["ttfp_ms"] for r in rows]
    total = [r["total_ms"] for r in rows]
    rtf = [r["rtf"] for r in rows if r["rtf"]]
    return {
        "chars_cjk": cjk_len(text),
        "chars_total": len(text),
        "text": text,
        "runs": runs,
        "warmup": warmup,
        "ttfp_p50_ms": round(statistics.median(ttfp), 2),
        "ttfp_p95_ms": round(pct(ttfp, 0.95), 2),
        "ttfp_min_ms": round(min(ttfp), 2),
        "ttfp_max_ms": round(max(ttfp), 2),
        "total_p50_ms": round(statistics.median(total), 2),
        "total_p95_ms": round(pct(total, 0.95), 2),
        "audio_sec_p50": round(statistics.median(r["audio_sec"] for r in rows), 3),
        "rtf_p50": round(statistics.median(rtf), 4) if rtf else None,
        "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "raw": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("default", "nokernel"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merged = {}
        for g in ("default", "nokernel"):
            p = RESULTS / ("spike_%s.json" % g)
            if p.exists():
                merged[g] = json.loads(p.read_text(encoding="utf-8"))
        out = RESULTS / "spike.json"
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print("merged ->", out)
        return

    if not args.group:
        ap.error("--group or --merge required")
    if not os.environ.get("GPT_SOVITS_HOME"):
        sys.exit("GPT_SOVITS_HOME must point to " + str(GPT_SOVITS))
    for p in (GPT_CKPT, SOVITS_CKPT, REF):
        if not p.exists():
            sys.exit("missing required file: " + str(p))
    if not REF_TXT.exists():
        sys.exit("missing ref transcript: " + str(REF_TXT))

    # Group B: disable the custom BigVGAN CUDA kernel BEFORE the package loads it,
    # so no nvcc/JIT compile is ever attempted.
    if args.group == "nokernel":
        os.environ["BIGVGAN_USE_CUDA_KERNEL"] = "0"
    os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

    # Upstream G2PW resolves GPT_SoVITS/text/G2PWModel relative to the process CWD.
    os.chdir(GPT_SOVITS)

    import torch
    from aquatts import TTSInferencer

    torch.manual_seed(SEED)
    random.seed(SEED)
    assert torch.cuda.is_available(), "CUDA not available"

    print("[bench] device:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
    print("[bench] torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("[bench] group:", args.group, "| BIGVGAN_USE_CUDA_KERNEL=",
          os.environ.get("BIGVGAN_USE_CUDA_KERNEL"))

    t_load = time.perf_counter()
    tts = TTSInferencer(
        device="cuda",
        gpt_path=str(GPT_CKPT),
        sovits_path=str(SOVITS_CKPT),
    )
    load_s = time.perf_counter() - t_load

    kernel_mode = getattr(tts, "bigvgan_kernel_mode", None)
    decoder = tts.t2s_model.model
    backend = {
        "bigvgan_kernel_mode": kernel_mode,
        "bigvgan_use_cuda_kernel_attr": bool(
            getattr(getattr(tts, "bigvgan_model", None), "use_cuda_kernel", False)),
        "flash_attn_active": bool(getattr(decoder, "use_flash_attn_kvcache", False)),
        "kv_cache_layout": getattr(decoder, "kv_cache_layout", None),
        "bucketed_graphs": len(getattr(decoder, "bucket_graphs", {}) or {}),
    }
    print("[bench] backend:", json.dumps(backend, ensure_ascii=False))
    if args.group == "nokernel":
        assert kernel_mode == "pytorch", (
            "expected PyTorch fallback, got %r" % (kernel_mode,))

    out = {
        "group": args.group,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "gpt_checkpoint": str(GPT_CKPT),
        "sovits_checkpoint": str(SOVITS_CKPT),
        "reference_audio": str(REF),
        "reference_text": REF_TXT.read_text(encoding="utf-8").strip(),
        "protocol": {
            "how_to_cut": HOW_TO_CUT, "top_k": TOP_K, "top_p": TOP_P,
            "temperature": TEMPERATURE, "speed": SPEED,
            "sample_steps": SAMPLE_STEPS, "chunk_size_seconds": CHUNK_SECONDS,
            "enable_cuda_graph": True, "enable_static_kv": True,
        },
        "load_seconds": round(load_s, 2),
        "backend": backend,
        "tiers": {},
    }

    for name, text in TEXTS.items():
        print("[bench] tier %-6s %2d chars ..." % (name, cjk_len(text)), flush=True)
        out["tiers"][name] = bench_tier(tts, torch, name, text, args.warmup, args.runs)
        t = out["tiers"][name]
        print("        ttfp p50=%.1f ms  p95=%.1f ms  rtf=%.2f  vram=%.2f GB"
              % (t["ttfp_p50_ms"], t["ttfp_p95_ms"], t["rtf_p50"], t["vram_peak_alloc_gb"]),
              flush=True)

    dest = RESULTS / ("spike_%s.json" % args.group)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[bench] wrote", dest)


if __name__ == "__main__":
    main()
