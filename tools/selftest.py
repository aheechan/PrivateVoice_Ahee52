"""Package self-test: verifies ASR -> LLM -> sentence split -> TTS -> playback.

Run from the install folder:
    runtime\\python.exe tools\\selftest.py
    runtime\\python.exe tools\\selftest.py --skip-llm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def step(report: dict, name: str, ok: bool, detail: str = "") -> None:
    report["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(("  [ok]   " if ok else "  [fail] ") + name + (f" | {detail}" if detail else ""))


def find_reference(store) -> Path | None:
    """Reference audio lives inside the default character, not in assets/."""
    try:
        for ch in store.list():
            ref = store.dir_of(ch.id) / "reference.wav"
            if ref.exists():
                return ref
    except Exception:
        pass
    for name in ("default_voice.wav", "ref.wav"):
        p = ROOT / "assets" / name
        if p.exists():
            return p
    return None


def main() -> int:
    report: dict = {"root": str(ROOT), "steps": [], "ok": False}
    skip_llm = "--skip-llm" in sys.argv

    print("== 1. Runtime ==")
    import torch

    print(f"  python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda} available={torch.cuda.is_available()}")
    step(report, "torch/cuda", torch.cuda.is_available())

    from app.config import Config
    from app.core import train
    from app.core.characters import CharacterStore

    cfg = Config.load()
    cfg.bootstrap()

    print("== 2. Hardware ==")
    ok, msg = train.check_hardware(cfg)
    print("  " + msg)
    step(report, "hardware", ok)

    print("== 3. Model paths ==")
    gpt = Path(cfg.path_of("gpt_sovits"))
    step(report, "gpt-sovits", (gpt / "GPT_SoVITS").exists(), str(gpt))
    pm = gpt / "GPT_SoVITS" / "pretrained_models"
    step(report, "s1v3.ckpt", (pm / "s1v3.ckpt").exists())
    step(report, "s2Gv4.pth", (pm / "gsv-v4-pretrained" / "s2Gv4.pth").exists())

    print("== 4. Characters ==")
    store = CharacterStore(cfg.path_of("characters"))
    chars = store.list()
    step(report, "characters", bool(chars), f"{len(chars)} found")
    ref = find_reference(store)
    step(report, "reference audio", ref is not None, str(ref or "missing"))
    if ref is None:
        print("\nRESULT: FAIL (no reference audio; train a character first)")
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "selftest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print("== 5. Reference decode ==")
    from app.core import audio_qc

    try:
        y, sr = audio_qc.decode(ref)
        step(report, "decode reference", len(y) > 0, f"{len(y)/sr:.1f}s @ {sr} Hz")
    except Exception as exc:  # noqa: BLE001
        step(report, "decode reference", False, str(exc))
        print("\nRESULT: FAIL (cannot decode reference audio)")
        return 1

    print("== 6. TTS ==")
    from app.core.tts import make_engine

    try:
        eng = make_engine(cfg)
        ch = chars[0]
        eng.load(ch.voice.get("gpt_ckpt"), ch.voice.get("sovits_ckpt"))
        text = "Hello, this is the package self-test."
        parts, rate = [], 48000
        for chunk in eng.synthesize_stream(text, str(ref),
                                           ch.voice.get("reference_text", "")):
            parts.append(np.asarray(chunk.samples, np.float32))
            rate = chunk.samplerate
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        step(report, "tts", peak > 0.02,
             f"peak={peak:.2f}, {audio.size/rate:.1f}s")
    except Exception as exc:  # noqa: BLE001
        step(report, "tts", False, f"{type(exc).__name__}: {exc}")

    print("== 7. Playback ==")
    try:
        import sounddevice as sd

        info = sd.query_devices(kind="output")
        print(f"  default output: {info['name']} (index {sd.default.device[1]})")
        step(report, "output device", True, info["name"])
    except Exception as exc:  # noqa: BLE001
        step(report, "output device", False, str(exc))

    if skip_llm:
        print("== 8. LLM (skipped) ==")
    else:
        print("== 8. LLM ==")
        from app.core.llm import ProviderStore, try_provider

        active = ProviderStore(cfg).active()
        if not active:
            step(report, "llm", False, "no provider configured")
        else:
            for prov in active:
                ok, detail = try_provider(prov)
                print(f"  [{prov.name}] {prov.base_url} -> {detail}")
                step(report, f"llm:{prov.name}", ok, detail)
                if ok:
                    break

    print("== 9. ASR ==")
    try:
        from app.core.asr import ASREngine

        eng = ASREngine(cfg.data.get("asr", {}))
        warm = eng.warmup()
        step(report, "asr warmup", True,
             f"{warm:.0f} ms | model={cfg.get('asr.model')}")
    except Exception as exc:  # noqa: BLE001
        step(report, "asr warmup", False, f"{type(exc).__name__}: {exc}")

    report["ok"] = all(s["ok"] for s in report["steps"])
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "selftest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRESULT: " + ("PASS" if report["ok"] else "FAIL"))
    print("report: " + str(ROOT / "logs" / "selftest.json"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
