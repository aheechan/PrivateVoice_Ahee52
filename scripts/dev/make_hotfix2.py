import pathlib
import shutil

ROOT = pathlib.Path(r"C:\Local Repositories\PrivateVoice")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
HK = ROOT / "dist" / "update_beta2_hotfix2"

if HK.exists():
    shutil.rmtree(HK)
HK.mkdir(parents=True)

shutil.copytree(PKG / "app", HK / "app",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """beta-2 热修 2（只替换 app 目录）
============================

修复：
1) TTS 静音重试时 name 'cancel' is not defined（崩溃，导致完全没有声音）
2) 静音重试现在会同时换解码参数（seed / sample_steps / top_k / temperature）
3) 训练完自检改为连合 3 句，明确告诉你有几句出声；
   3 句里不足 2 句出声会直接判定"该角色大概率不可用，建议换素材重训"

安装：
1. 退出程序（托盘图标 -> 退出）
2. 把 app 目录整个复制到安装目录覆盖
   （默认 C:\\Program Files\\Ahee52_PrivateVoice\\）
3. 重新启动

注意：之前训练出来的 emily 大概率是坏的，需要按 beta-2 重训一次。
训练日志里看这行："自检：3 句全部出声（峰值 0.5x / 0.4x / 0.6x）"。
"""
(HK / "安装说明.txt").write_text(readme, encoding="utf-8")

size = sum(f.stat().st_size for f in HK.rglob("*") if f.is_file()) / 1e6
files = sum(1 for f in HK.rglob("*") if f.is_file())
print(f"hotfix2: {size:.2f} MB / {files} files")
