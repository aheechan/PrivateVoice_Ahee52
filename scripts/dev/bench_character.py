"""用某个角色的 checkpoint 复跑 Stage 1 的 TTFP 协议，产出 character_ttfp.json。

复用 benchmark.py 的 run_once / bench_tier，保证与 Stage 1 的数字可比。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--sovits", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-text", default="")
    ap.add_argument("--reference-text-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ref = Path(args.reference)
    if not ref.exists():
        sys.exit(f"参考音频不存在: {ref}")
    text = args.reference_text
    if not text and args.reference_text_file:
        text = Path(args.reference_text_file).read_text(encoding="utf-8")
    if not text:
        txt = ref.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        sys.exit("缺少参考文本")

    os.environ["GPT_SOVITS_HOME"] = str(ROOT / "GPT-SoVITS")
    os.environ["BIGVGAN_USE_CUDA_KERNEL"] = "0"
    os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.chdir(ROOT / "GPT-SoVITS")

    B.REF = ref
    B.REF_TEXT_STR = text

    import torch
    from aquatts import TTSInferencer

    torch.manual_seed(B.SEED)
    if not torch.cuda.is_available():
        sys.exit("CUDA 不可用")

    t0 = time.perf_counter()
    tts = TTSInferencer(device="cuda", gpt_path=args.gpt, sovits_path=args.sovits)
    load_s = time.perf_counter() - t0
    backend = {
        "bigvgan_kernel_mode": getattr(tts, "bigvgan_kernel_mode", None),
        "flash_attn_active": bool(getattr(
            getattr(getattr(tts, "t2s_model", None), "model", None),
            "use_flash_attn_kvcache", False)),
        "bucketed_graphs": len(getattr(
            getattr(getattr(tts, "t2s_model", None), "model", None),
            "bucket_graphs", {}) or {}),
    }

    out = {
        "character_id": args.character,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "python": sys.version.split()[0],
        "gpt_checkpoint": str(Path(args.gpt)),
        "sovits_checkpoint": str(Path(args.sovits)),
        "reference_audio": str(ref),
        "reference_text": text,
        "protocol": {
            "how_to_cut": B.HOW_TO_CUT, "top_k": B.TOP_K, "top_p": B.TOP_P,
            "temperature": B.TEMPERATURE, "speed": B.SPEED,
            "sample_steps": B.SAMPLE_STEPS, "chunk_size_seconds": B.CHUNK_SECONDS,
            "enable_cuda_graph": True, "enable_static_kv": True,
        },
        "load_seconds": round(load_s, 2),
        "backend": backend,
        "tiers": {},
    }

    for tier in ("short", "medium", "long"):
        print(f"[bench] {tier} ...", flush=True)
        out["tiers"][tier] = B.bench_tier(
            tts, torch, tier, B.TEXTS[tier], args.warmup, args.runs)
        t = out["tiers"][tier]
        print("        ttfp p50=%.1f p95=%.1f rtf=%.2f vram=%.2fGB"
              % (t["ttfp_p50_ms"], t["ttfp_p95_ms"], t["rtf_p50"],
                 t["vram_peak_alloc_gb"]), flush=True)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[bench] wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
