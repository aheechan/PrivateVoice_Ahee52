"""A/B：官方底模 zero-shot vs 我们的 LoRA 微调；以及 sample_steps 4 vs 8。

同一段参考音频、同一段文本，只有模型/步数不同，便于直接对比。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

TEXT = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"
OUT = ROOT / "results" / "compare"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    d = store.dir_of(ch.id)
    ref = str(d / "reference.wav")
    ref_text = ch.voice.get("reference_text", "")

    pm = Path(cfg.path_of("gpt_sovits")) / "GPT_SoVITS" / "pretrained_models"
    base_gpt = str(pm / "s1v3.ckpt")
    base_sovits = str(pm / "s2Gv3.pth")
    fine_gpt = ch.voice["gpt_ckpt"]
    fine_sovits = ch.voice["sovits_ckpt"]

    engine = make_engine(cfg)
    for tag, gpt, sov in (("base", base_gpt, base_sovits),
                          ("finetuned", fine_gpt, fine_sovits)):
        engine.load(gpt, sov)
        for steps in (4, 8):
            parts = []
            t0 = time.perf_counter()
            for c in engine.synthesize_stream(TEXT, ref, ref_text,
                                              sample_steps=steps):
                parts.append(np.asarray(c.samples, np.float32))
            el = time.perf_counter() - t0
            x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            dur = x.size / 24000
            name = f"{tag}_steps{steps}.wav"
            audio_qc.save_wav(OUT / name, x, 24000, subtype="PCM_16")
            print(f"{name:26s} audio={dur:5.2f}s synth={el:5.2f}s rtf={el/dur:.2f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
