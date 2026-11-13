"""把实际播放的音频旁录下来，检查有没有断裂/静音插空。"""
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

TEXT = "今天有点累，不过见到你就好多了。晚上想吃点热乎的汤面，然后早点睡。"
OUT = ROOT / "results" / "quality"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ch = [c for c in store.list() if c.status == "ready"][0]
    pipe = Pipeline(cfg, store, ch)
    pipe.warm_tts()

    dump = (OUT / "playback_48k.raw").open("wb")
    pipe.player.dump = dump
    pipe.submit_text(TEXT)
    t0 = time.time()
    while time.time() - t0 < 40:
        if pipe.state.value == "idle" and pipe._turn_thread and not pipe._turn_thread.is_alive():
            break
        time.sleep(0.05)
    dump.close()
    pipe.player.dump = None
    underruns = pipe.player.underruns
    pipe.close()

    raw = np.frombuffer((OUT / "playback_48k.raw").read_bytes(), dtype=np.float32)
    sf.write(str(OUT / "playback_48k.wav"), raw, 48000, subtype="PCM_16")
    print(f"captured {len(raw)/48000:.2f}s | underruns={underruns}")

    silence = np.abs(raw) < 1e-6
    runs, cur = [], 0
    for v in silence:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    long_runs = [r for r in runs if r > 48000 * 0.04]     # >40ms 的空洞
    print(f"silence ratio={silence.mean():.3f} | gaps>40ms={len(long_runs)} "
          f"| longest={max(runs)/48000:.3f}s" if runs else "no silence")
    if long_runs:
        print("  gap lengths(ms):", [round(r / 48, 1) for r in long_runs][:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
