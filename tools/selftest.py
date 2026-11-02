"""打包自检：验证分发包在本机能否跑通「ASR → LLM → 切句 → TTS → 播放」。

用法（在分发包根目录）：
    runtime\\python.exe tools\\selftest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def main() -> int:
    report: dict = {"root": str(ROOT), "steps": [], "ok": False}

    def step(name: str, ok: bool, detail="") -> None:
        report["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:300]})
        print(("  [ok]   " if ok else "  [fail] ") + name + (f" · {detail}" if detail else ""))

    print("== 1. 运行时 ==")
    import torch
    print(f"  python {sys.version.split()[0]} · torch {torch.__version__} · "
          f"cuda {torch.version.cuda} available={torch.cuda.is_available()}")
    step("torch/cuda", torch.cuda.is_available())

    from app.config import Config
    from app.core import train
    from app.core.characters import CharacterStore

    cfg = Config.load()
    cfg.bootstrap()
    print("== 2. 硬件门禁 ==")
    ok, msg = train.check_hardware(cfg)
    print("  " + msg)
    report["gpu"] = msg
    step("hardware", ok, msg)
    if not ok:
        report["ok"] = False
        _save(report)
        return 1
    info = train.gpu_info()
    print(f"  {info.name} · {info.total_gb:.1f}GB · 可用 {info.free_gb:.1f}GB · "
          f"sm_{info.compute_cap[0]}{info.compute_cap[1]}")

    print("== 3. 依赖导入 ==")
    try:
        import httpx, sounddevice, soxr, funasr  # noqa: F401
        from PySide6.QtWidgets import QApplication  # noqa: F401
        step("imports", True)
    except Exception as exc:  # noqa: BLE001
        step("imports", False, exc)
        _save(report)
        return 1

    print("== 4. 角色与模型 ==")
    store = CharacterStore(cfg.path_of("characters"))
    ready = [c for c in store.list() if c.status == "ready"]
    step("character", bool(ready), f"{len(ready)} 个 ready")
    if not ready:
        _save(report)
        return 1
    ch = ready[0]
    gpt = Path(ch.voice.get("gpt_ckpt", ""))
    sov = Path(ch.voice.get("sovits_ckpt", ""))
    print(f"  {ch.name}: gpt={gpt.name} exists={gpt.exists()} | "
          f"sovits={sov.name} exists={sov.exists()}")
    step("checkpoints", gpt.exists() and sov.exists())

    print("== 5. 链路（注入 4s 真实语音）==")
    from app.core import audio_qc
    from app.core.pipeline import Pipeline

    events: list[tuple[str, str]] = []
    pipe = Pipeline(cfg, store, ch,
                    on_event=lambda e: events.append((e.kind, e.text or e.detail)))
    t0 = time.time()
    warm = pipe.warmup()
    print(f"  warmup {time.time()-t0:.1f}s {warm}")
    step("warmup", "tts_error" not in warm and "asr_error" not in warm, warm)

    y, sr, _ = audio_qc.process_to_48k(ROOT / "assets" / "ref.wav", cfg.audio)
    n = int(4.0 * sr)
    clip = y[:n] if y.size >= n else np.pad(y, (0, n - y.size))
    pipe.submit_audio(clip, sr, t_press=time.perf_counter() - 0.3,
                      t_release=time.perf_counter())
    deadline = time.time() + 90
    while time.time() < deadline:
        if pipe.state.value == "idle" and pipe._turn_thread is not None \
                and not pipe._turn_thread.is_alive():
            break
        time.sleep(0.05)

    asr = [t for k, t in events if k == "user_text"]
    reply = [t for k, t in events if k == "turn_done"]
    errs = [t for k, t in events if k == "error"]
    rows = pipe.metrics.recent(1)
    m = rows[-1] if rows else {}
    print(f"  ASR: {asr[:1]}")
    print(f"  reply: {reply[:1]}")
    print(f"  errors: {errs[:3]}")
    print(f"  e2e={m.get('e2e_ms')}ms asr={m.get('asr_ms')}ms "
          f"llm={m.get('llm_ttft_ms')}ms tts={m.get('tts_ttfp_ms')}ms "
          f"audio={m.get('audio_sec')}s")
    step("asr", bool(asr))
    step("llm_reply", bool(reply))
    step("audio_played", float(m.get("audio_sec") or 0) > 0.3)
    step("metrics", bool(m.get("e2e_ms")))

    print("== 6. 打断 ==")
    pipe.submit_text("再说一句我听听")
    t0 = time.time()
    while time.time() - t0 < 20:
        if pipe.state.value == "speaking":
            break
        time.sleep(0.02)
    time.sleep(0.2)
    ms = pipe.interrupt()
    step("interrupt", ms <= 150 and not pipe.player.playing, f"{ms:.0f}ms")

    pipe.close()
    report["metrics"] = m
    report["ok"] = all(s["ok"] for s in report["steps"])
    _save(report)
    print("\nSELFTEST", "PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


def _save(report: dict) -> None:
    out = ROOT / "results" / "selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", out)


if __name__ == "__main__":
    raise SystemExit(main())
