"""验收辅助：合三句试听，量化检查可懂度/爆音/音色接近度。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.config import Config  # noqa: E402
from app.core import audio_qc, tts  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402


def speaker_similarity(a: np.ndarray, b: np.ndarray, sr: int = 24000) -> float | None:
    try:
        model = audio_qc._speaker_model()
        a16 = audio_qc.resample(a, sr, 16000)
        b16 = audio_qc.resample(b, sr, 16000)
        out = model.generate(input=[a16.astype(np.float32), b16.astype(np.float32)])
        vecs = []
        for item in out:
            v = np.asarray(item["spk_embedding"].detach().cpu().numpy(),
                           dtype=np.float64).reshape(-1)
            vecs.append(v / (np.linalg.norm(v) or 1.0))
        return float(np.dot(vecs[0], vecs[1]))
    except Exception as exc:
        print("  similarity failed:", exc)
        return None


def main() -> int:
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    cid = sys.argv[1] if len(sys.argv) > 1 else store.list()[0].id
    ch = store.get(cid)
    d = store.dir_of(cid)
    ref_wav, ref_sr = audio_qc.decode(d / ch.voice["reference"])
    ref = audio_qc.to_mono(ref_wav)

    engine = tts.make_engine(cfg)
    t0 = time.time()
    engine.load(ch.voice["gpt_ckpt"], ch.voice["sovits_ckpt"])
    print(f"load {engine.load_seconds:.1f}s (total {time.time()-t0:.1f}s)")

    out_dir = Path(cfg.path_of("datasets")) / cid / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for i, text in enumerate(tts.preview_sentences(cfg), start=1):
        dest = out_dir / f"preview_{i}.wav"
        t1 = time.time()
        engine.synthesize_file(text, dest, str(d / ch.voice["reference"]),
                               ch.voice["reference_text"])
        samples, sr = audio_qc.decode(dest)
        mono = audio_qc.to_mono(samples)
        dur = mono.size / sr
        peak = audio_qc.true_peak_dbfs(mono)
        lufs = audio_qc.integrated_lufs(mono, sr)
        sil = audio_qc.silence_ratio(mono, sr)
        sim = speaker_similarity(mono, ref, sr)
        row = {
            "index": i, "text": text, "seconds": round(dur, 2),
            "peak_dbfs": round(peak, 2), "lufs": round(lufs, 2),
            "silence_ratio": round(sil, 3),
            "speaker_similarity": None if sim is None else round(sim, 4),
            "elapsed_sec": round(time.time() - t1, 2),
            "file": str(dest),
        }
        report.append(row)
        print(f"  {i}. {dur:5.2f}s peak={peak:6.1f} lufs={lufs:6.1f} "
              f"sil={sil:.2f} sim={row['speaker_similarity']} {text}")

    summary = {
        "character_id": cid,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference": str(d / ch.voice["reference"]),
        "clips": report,
        "mean_similarity": round(float(np.mean(
            [r["speaker_similarity"] for r in report if r["speaker_similarity"]])), 4)
        if any(r["speaker_similarity"] for r in report) else None,
    }
    dest = Path(cfg.path_of("results")) / "character_preview.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
