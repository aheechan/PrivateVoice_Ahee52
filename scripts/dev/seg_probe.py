"""探针 3：在真实素材上跑两遍切分 + 打标，检查片段分布。"""
import sys
import time

sys.path.insert(0, r"C:\Local Repositories\PrivateVoice")

from app.config import Config  # noqa: E402
from app.core import segment  # noqa: E402
from app.core.label import Transcriber  # noqa: E402

WAV = r"C:\Users\72157\OneDrive\Desktop\Grace_voice\grace voice nanpu plus orig 0327.wav"

cfg = Config.load()
tr = Transcriber(cfg.label_cfg)
t0 = time.time()
res = segment.segment_file(WAV, cfg, r"C:\Local Repositories\PrivateVoice\datasets\_probe",
                           transcriber=tr, progress=lambda m: print("   ..", m, flush=True))
print("[seg] elapsed %.1fs" % (time.time() - t0))
print("[seg] clips:", len(res.rows))
print("[seg] warnings:", res.warnings)
for r in res.rows:
    print(f"  {r.index:02d} {r.clip} {r.duration:5.2f}s peak={r.peak_dbfs:6.1f} snr={r.snr_db:5.1f} "
          f"flags={r.flags} text={r.text[:46]}")
print("[seg] dataset.list:")
print(res.dataset_list.read_text(encoding="utf-8")[:1200])
