"""判断"断断续续"发生在句内（模型）还是句间（播放调度）。

1) 单句合成（不含标点）→ 统计句内 >80ms 的静音段
2) 整段合成 → 统计每句内部的静音、以及句与句之间的静音
3) 打印每句的合成耗时与音频时长，算 RTF
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

SINGLE = "今天有点累不过还好"
MULTI = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"


def silences(x: np.ndarray, sr: int, min_ms: int = 80) -> list[float]:
    quiet = np.abs(x) < 1e-4
    runs, cur = [], 0
    for v in quiet:
        if v:
            cur += 1
        elif cur:
            if cur / sr * 1000 >= min_ms:
                runs.append(cur / sr * 1000)
            cur = 0
    if cur / sr * 1000 >= min_ms:
        runs.append(cur / sr * 1000)
    return runs


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    engine = make_engine(cfg)
    engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
    ref = str(store.dir_of(ch.id) / (ch.voice.get("reference") or "reference.wav"))
    rtext = ch.voice.get("reference_text", "")

    for label, text in (("单句", SINGLE), ("多句", MULTI)):
        parts = []
        t0 = time.perf_counter()
        for c in engine.synthesize_stream(text, ref, rtext):
            parts.append(np.asarray(c.samples, np.float32))
        el = time.perf_counter() - t0
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        dur = len(x) / 24000
        sil = silences(x, 24000)
        print(f"[{label}] dur={dur:.2f}s synth={el:.2f}s rtf={el/dur:.2f} "
              f"chunks={len(parts)} 句内静音段={[round(s) for s in sil]}ms")
        np.save(str(ROOT / "results" / f"quality_{label}.npy"), x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
