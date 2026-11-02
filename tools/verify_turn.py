"""验证：4 句回复、以及打断后新一轮开头没有残留音频。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "results" / "verify"


def run_turn(pipe: Pipeline, text: str, timeout: float = 90.0) -> dict:
    pipe.submit_text(text)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pipe.state.value == "idle" and pipe._turn_thread and \
                not pipe._turn_thread.is_alive():
            break
        time.sleep(0.05)
    m = pipe.metrics.recent(1)
    return m[-1] if m else {}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = next(c for c in store.list() if "v4" in c.name)
    pipe = Pipeline(cfg, store, ch)
    pipe.warm_tts()

    m1 = run_turn(pipe, "为什么人越累越睡不着？说说你的看法。")
    reply = ""
    try:
        ctx = pipe.history.context(budget_chars=100000)
        for msg in reversed(ctx):
            if msg.get("role") == "assistant":
                reply = msg.get("content", "")
                break
    except Exception as exc:
        reply = f"(取不到) {exc}"
    print(f"[长回复] {reply}")
    print(f"  字数={len(reply)} 句子数={reply.count('。') + reply.count('！') + reply.count('？')} "
          f"e2e={m1.get('e2e_ms')}ms audio={m1.get('audio_sec')}s")

    # 打断场景：说一半就取消，然后立刻开始新一轮
    dump = (OUT / "after_interrupt.raw").open("wb")
    pipe.player.dump = dump
    pipe.submit_text("给我讲讲你小时候最喜欢的一个夏天。")
    t0 = time.time()
    while time.time() - t0 < 6:
        if pipe.state.value == "speaking":
            break
        time.sleep(0.05)
    time.sleep(1.0)
    pipe.interrupt()
    time.sleep(0.6)
    m2 = run_turn(pipe, "嗯，那你现在想干嘛？")
    dump.close()
    pipe.player.dump = None
    print(f"[打断后] e2e={m2.get('e2e_ms')}ms")

    raw = np.frombuffer((OUT / "after_interrupt.raw").read_bytes(), dtype=np.float32)
    if raw.size:
        sf.write(str(OUT / "after_interrupt.wav"), raw, pipe.player.samplerate,
                 subtype="PCM_16")
        head = raw[: int(pipe.player.samplerate * 0.25)]
        print(f"  旁录时长 {raw.size/pipe.player.samplerate:.2f}s，"
              f"开头 250ms 峰值 {np.abs(head).max():.3f}，"
              f"其中前 50ms 峰值 {np.abs(raw[:int(pipe.player.samplerate*0.05)]).max():.3f}")
    pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
