"""音质诊断：区分「模型/训练」还是「我们播放链路的拼接与重采样」。

输出三份音频让人耳对比，同时算客观指标：
  1) model_24k.wav      引擎原始输出（24k，直接拼接，不动）
  2) good_48k.wav       把整段一次性 soxr 重采样到 48k（参考）
  3) stream_48k.wav     按我们的 StreamingResampler 逐块重采样（实际播放路径）
另外跑一轮真实播放，统计 buffer underrun（欠载=断断续续的直接原因）。
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
from app.core.player import StreamingResampler  # noqa: E402
from app.core.tts import make_engine  # noqa: E402

TEXT = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"
OUT = ROOT / "results" / "quality"


def boundary_jumps(x: np.ndarray, bounds: list[int]) -> float:
    """块边界处的样点跳变，和整体典型跳变比较。"""
    if not bounds:
        return 0.0
    d = np.abs(np.diff(x))
    typical = float(np.median(d)) + 1e-9
    jumps = []
    for b in bounds:
        if 1 <= b < len(x):
            jumps.append(abs(float(x[b]) - float(x[b - 1])))
    if not jumps:
        return 0.0
    return float(np.max(jumps) / typical)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    engine = make_engine(cfg)
    engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
    ref = str(store.dir_of(ch.id) / (ch.voice.get("reference") or "reference.wav"))

    chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    for c in engine.synthesize_stream(TEXT, ref, ch.voice.get("reference_text", "")):
        chunks.append(np.asarray(c.samples, np.float32))
    elapsed = time.perf_counter() - t0
    raw24 = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    print(f"chunks={len(chunks)} samples={raw24.size} "
          f"dur={raw24.size/24000:.2f}s elapsed={elapsed:.2f}s "
          f"rtf={elapsed/(raw24.size/24000):.2f}")

    audio_qc.save_wav(OUT / "model_24k.wav", raw24, 24000, subtype="PCM_16")

    # 参考：整段一次性重采样
    good48 = audio_qc.resample(raw24, 24000, 48000)
    audio_qc.save_wav(OUT / "good_48k.wav", good48, 48000, subtype="PCM_16")

    # 实际路径：逐块流式重采样
    rs = StreamingResampler(24000, 48000)
    parts, bounds = [], []
    total = 0
    for c in chunks:
        y = rs.process(c)
        parts.append(y)
        total += y.size
        bounds.append(total)
    stream48 = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    audio_qc.save_wav(OUT / "stream_48k.wav", stream48, 48000, subtype="PCM_16")

    n = min(good48.size, stream48.size)
    diff = np.abs(good48[:n] - stream48[:n])
    print(f"stream vs one-shot: max={diff.max():.4f} mean={diff.mean():.5f} "
          f"p99={np.percentile(diff, 99):.4f}")
    print(f"boundary jump ratio (stream) = {boundary_jumps(stream48, bounds):.1f}x "
          f"(1.0 = 与普通样点跳变一样)")

    # 频谱平坦度：颗粒感/噪声往往表现为高频平坦度升高
    def flatness(x: np.ndarray) -> float:
        if x.size < 4096:
            return 0.0
        spec = np.abs(np.fft.rfft(x[: 1 << 15] * np.hanning(1 << 15))) + 1e-12
        return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    print(f"spectral flatness 24k={flatness(raw24):.5f} "
          f"good48={flatness(good48):.5f} stream48={flatness(stream48):.5f}")
    print("wrote", OUT)

    # 真实播放：统计欠载
    from app.core.pipeline import Pipeline
    ev = []
    pipe = Pipeline(cfg, store, ch, on_event=lambda e: ev.append((e.kind, e.text or e.detail)))
    pipe.warm_tts()
    pipe.submit_text(TEXT)
    t0 = time.time()
    last = None
    while time.time() - t0 < 40:
        if pipe.state.value == "idle" and pipe._turn_thread and not pipe._turn_thread.is_alive():
            break
        cur = pipe.player.underruns
        if last is not None and cur != last:
            print(f"  underrun +{cur-last} at buffered={pipe.player.buffered_sec:.2f}s")
        last = cur
        time.sleep(0.05)
    print(f"total underruns={pipe.player.underruns} "
          f"audio={pipe.metrics.recent(1)[-1].get('audio_sec')}s")
    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
