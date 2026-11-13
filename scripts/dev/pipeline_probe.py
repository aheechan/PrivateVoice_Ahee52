"""Stage 3 链路探针：不开 UI，直接驱动 Pipeline 跑 N 轮并统计延迟。

用法：
    python tools/pipeline_probe.py --turns 2      # 冒烟
    python tools/pipeline_probe.py --turns 20     # 出 logs/turns.jsonl
    python tools/pipeline_probe.py --interrupt    # 打断测试
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402

SRC = ROOT / "assets" / "ref.wav"


def make_clip(seconds: float = 4.0) -> np.ndarray:
    y, sr, _ = audio_qc.process_to_48k(SRC, Config.load().audio)
    n = int(seconds * sr)
    clip = y[:n]
    if clip.size < n:
        clip = np.pad(clip, (0, n - clip.size))
    return clip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--interrupt", action="store_true")
    ap.add_argument("--text", default="")
    args = ap.parse_args()

    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    chars = [c for c in store.list() if c.status == "ready"]
    if not chars:
        print("NO_READY_CHARACTER")
        return 1
    ch = chars[0]

    states: list[str] = []

    def on_event(ev):
        if ev.kind == "state":
            states.append(ev.state)
            print(f"  [state] {ev.state} {ev.detail}")
        elif ev.kind == "user_text":
            print(f"  [asr] {ev.text}")
        elif ev.kind == "turn_done":
            print(f"  [reply] {ev.text[:60]}")
        elif ev.kind == "error":
            print(f"  [error] {ev.text}")

    pipe = Pipeline(cfg, store, ch, on_event=on_event)
    print("warmup:", pipe.warmup())

    if args.interrupt:
        return run_interrupt(pipe, make_clip(args.seconds))

    clip = make_clip(args.seconds)
    ok = 0
    for i in range(args.turns):
        print(f"\n=== turn {i+1}/{args.turns} ===")
        t0 = time.perf_counter()
        if args.text:
            pipe.submit_text(args.text)
        else:
            tp = time.perf_counter()
            pipe.submit_audio(clip, 48000, t_press=tp - 0.4, t_release=tp)
        deadline = time.time() + 40
        while time.time() < deadline:
            if pipe.state.value == "idle" and pipe._turn_thread is not None \
                    and not pipe._turn_thread.is_alive():
                break
            time.sleep(0.05)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  round_wall={dt:.0f}ms")
        ok += 1
        time.sleep(0.4)

    s = pipe.metrics.summary(20)
    print("\nsummary:", s)
    rows = pipe.metrics.recent(20)
    for r in rows[-5:]:
        print(f"  asr={r.get('asr_ms')} llm={r.get('llm_ttft_ms')} "
              f"tts={r.get('tts_ttfp_ms')} e2e={r.get('e2e_ms')} "
              f"len={r.get('text_len')}/{r.get('reply_len')} prov={r.get('provider')}")
    pipe.close()
    print("PIPELINE_PROBE", "PASS" if ok == args.turns else "FAIL")
    return 0 if ok == args.turns else 1


def run_interrupt(pipe: Pipeline, clip: np.ndarray) -> int:
    """播放中途打断，测停止耗时与状态回落。"""
    print("\n=== interrupt test ===")
    pipe.submit_audio(clip, 48000)
    t0 = time.time()
    while time.time() - t0 < 20:
        if pipe.state.value == "speaking" and pipe.player.playing:
            break
        time.sleep(0.02)
    else:
        print("never reached speaking")
        return 1
    time.sleep(0.3)
    ms = pipe.interrupt()
    print(f"interrupt stop_ms={ms:.1f} state={pipe.state.value} "
          f"buffered={pipe.player.buffered_sec:.2f}s playing={pipe.player.playing}")
    ok = ms <= 150 and pipe.state.value == "idle" and not pipe.player.playing
    pipe.close()
    print("INTERRUPT_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
