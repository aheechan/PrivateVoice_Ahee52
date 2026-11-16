"""分发包内的收尾验证：默认参数、语气词时长、释放/加载模型。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402


def synth_len(pipe: Pipeline, text: str) -> float:
    ch = pipe.character
    ref = str(pipe.store.dir_of(ch.id) / "reference.wav")
    parts, sr = [], 48000
    for c in pipe._ensure_tts().synthesize_stream(
            text, ref, ch.voice.get("reference_text", "")):
        parts.append(c.samples)
        sr = c.samplerate
    return (np.concatenate(parts).size / sr) if parts else 0.0


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = next(c for c in store.list() if c.status == "ready")
    print("cfg top_k:", cfg.get("tts.top_k"), "temp:", cfg.get("tts.temperature"),
          "version:", cfg.get("training.s2_version"))

    pipe = Pipeline(cfg, store, ch)
    t0 = time.time()
    pipe.warm_tts()
    print(f"warm_tts {time.time()-t0:.1f}s")
    for text in ("嗯。", "唔，那就早点睡。", "今天有点累，早点睡吧。"):
        d = synth_len(pipe, text)
        print(f"  {text!r}: {d:.2f}s ({d/max(1,len(text)):.2f}s/字)")

    try:
        import torch
        free_before = torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:
        free_before = -1
    info = pipe.release_models()
    try:
        import torch
        free_after = torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:
        free_after = -1
    print(f"release: free {free_before:.2f} -> {free_after:.2f} GB "
          f"(释放 {free_after-free_before:+.2f} GB) info={info}")

    t0 = time.time()
    pipe.load_models()
    print(f"reload {time.time()-t0:.1f}s, tts_ready={pipe.is_warm}")
    d = synth_len(pipe, "嗯，我知道了。")
    print(f"  after reload: {d:.2f}s")
    pipe.close()
    print("PKG_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
