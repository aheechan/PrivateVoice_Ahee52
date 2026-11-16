"""核对三个更新包是否都含 beta-3 的最终改动。"""
import pathlib

ROOT = pathlib.Path(r"C:\Local Repositories\PrivateVoice\dist")
TARGETS = ["Ahee52_PrivateVoice", "update_beta3", "update_beta2", "update_beta2_hotfix5"]

MARKERS = {
    "wizard QC 放行": ("app/ui/wizard.py", 'return bool(self.wiz.state.get("qc"))'),
    "QC 文案": ("app/ui/wizard.py", "仍可继续"),
    "原子写": ("app/core/audio_qc.py", "def save_wav_atomic"),
    "自适应 rank": ("app/core/train.py", "_auto_tune_rank"),
    "基础音色兜底": ("app/core/pipeline.py", "_base_engine"),
    "英文空格修复": ("app/core/pipeline.py", "英文单词之间的空格必须保留"),
}

for name in TARGETS:
    base = ROOT / name
    if not base.exists():
        print(f"--- {name}: 不存在")
        continue
    print(f"--- {name}")
    for label, (rel, needle) in MARKERS.items():
        f = base / rel
        ok = f.exists() and needle in f.read_text(encoding="utf-8", errors="ignore")
        print(f"    {'OK ' if ok else '缺 '} {label}")
    chars = sorted(p.name for p in (base / "characters").iterdir()
                   if p.is_dir()) if (base / "characters").exists() else []
    print(f"    characters: {chars}")
    has_grace = any("格雷丝" in c for c in chars)
    print(f"    {'OK ' if not has_grace else '缺 '} 已移除格雷丝")
    dv = base / "assets" / "default_voice.wav"
    print(f"    {'OK ' if dv.exists() else '缺 '} 默认音色参考音频")
