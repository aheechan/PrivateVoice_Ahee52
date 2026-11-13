"""v3 vs v4：同一段文本、同一个人的音色，比较采样率、频段与听感样本。"""
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


def bands(x: np.ndarray, sr: int) -> str:
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum() + 1e-12
    parts = []
    for lo, hi in ((0, 4000), (4000, 8000), (8000, 12000), (12000, 16000)):
        if lo >= sr / 2:
            continue
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        parts.append(f"{lo//1000}-{hi//1000}k:{100*spec[m].sum()/total:5.1f}%")
    return "  ".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    chars = {c.name: c for c in store.list() if c.status == "ready"}
    targets = []
    if "格雷丝" in chars:
        targets.append(("v3", chars["格雷丝"]))
    v4 = next((c for c in chars.values() if "v4" in c.name.lower()), None)
    if v4:
        targets.append(("v4", v4))
    if not targets:
        print("没有可用角色")
        return 1

    engine = make_engine(cfg)
    for tag, ch in targets:
        d = store.dir_of(ch.id)
        ref = str(d / (ch.voice.get("reference") or "reference.wav"))
        engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
        parts, sr_out = [], 24000
        t0 = time.perf_counter()
        for c in engine.synthesize_stream(TEXT, ref, ch.voice.get("reference_text", "")):
            parts.append(np.asarray(c.samples, np.float32))
            sr_out = int(c.samplerate or sr_out)
        el = time.perf_counter() - t0
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        dur = x.size / sr_out
        path = OUT / f"{tag}_output.wav"
        audio_qc.save_wav(path, x, sr_out, subtype="PCM_16")
        print(f"[{tag}] {ch.name} sr={sr_out} dur={dur:.2f}s synth={el:.2f}s "
              f"rtf={el/dur:.2f} | {bands(x, sr_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
