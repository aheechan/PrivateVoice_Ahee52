"""用现有素材验证：参考校验 + 自适应 rank + 3 句自检。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Local Repositories\PrivateVoice")
from app.config import Config
from app.core import train
from app.core.characters import CharacterStore

cfg = Config.load()
store = CharacterStore(cfg.path_of("characters"))
ds = Path(cfg.path_of("datasets")) / "格雷丝-v4-7c0e9d"
ch = store.create(f"验证{int(time.time()) % 10000}", persona="t", greeting="hi")
plan = cfg.plan(train.gpu_info().total_gb, "balanced")
events = []
runner = train.TrainingRunner(cfg, store, ch.id, ds, plan,
                             on_event=lambda e: events.append((e.stage, e.level, e.message)))
runner.run()
for e in events:
    if e[1] in ("warn", "error") or "rank" in e[2] or "参考音频" in e[2] or "自检" in e[2]:
        print("ev:", e)
print("VOICE:", store.get(ch.id).voice.get("sovits_ckpt"))
