"""检查 v4 在「嗯」这类语气词开头的音频有没有起始卡顿/重复/延音。

对每条样本看：
  - 起始 300ms 是否有样点重复段（zero-order hold，听感是"卡住"）
  - 是否有 >60ms 的完全静音（起始停顿）
  - 样点跳变是否异常（咔哒）
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

TEXTS = [
    "嗯。",
    "嗯，我知道了。",
    "嗯…今天有点累吧？",
    "哦，这样啊。",
    "唉，别想太多了。",
    "唔，那就早点睡。",
]
OUT = ROOT / "results" / "um_test"


def repeated_runs(x: np.ndarray, min_run: int = 6) -> int:
    """连续完全相同样点的最长段（zero-order hold 的典型特征）。"""
    if x.size < min_run:
        return 0
    same = np.abs(np.diff(x)) < 1e-7
    best = cur = 0
    for v in same:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def silent_head_ms(x: np.ndarray, sr: int, thresh: float = 1e-4) -> float:
    idx = np.where(np.abs(x) > thresh)[0]
    if idx.size == 0:
        return x.size / sr * 1000
    return float(idx[0]) / sr * 1000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = next(c for c in store.list() if "v4" in c.name)
    engine = make_engine(cfg)
    engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
    ref = str(store.dir_of(ch.id) / (ch.voice.get("reference") or "reference.wav"))
    rt = ch.voice.get("reference_text", "")

    for i, text in enumerate(TEXTS, 1):
        parts, sr = [], 48000
        t0 = time.perf_counter()
        first = None
        for c in engine.synthesize_stream(text, ref, rt):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            parts.append(c.samples)
            sr = c.samplerate
        el = time.perf_counter() - t0
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        head = x[: int(sr * 0.35)]
        runs = repeated_runs(head)
        jump = float(np.max(np.abs(np.diff(head)))) if head.size > 2 else 0.0
        p99 = float(np.percentile(np.abs(np.diff(x)), 99)) if x.size > 10 else 0
        name = f"{i:02d}_{text[:6].replace('，', '_').replace('。', '')}.wav"
        audio_qc.save_wav(OUT / name, x, sr, subtype="PCM_16")
        print(f"[{i}] {text!r:20s} dur={x.size/sr:5.2f}s ttfp={first:6.0f}ms "
              f"rtf={el/(x.size/sr):.2f} | 起始静音={silent_head_ms(x, sr):5.1f}ms "
              f"重复样点={runs:3d} 跳变={jump:.3f}(p99={p99:.3f})")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
