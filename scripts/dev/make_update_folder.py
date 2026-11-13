"""生成 beta-2 的增量 update 文件夹：只放这次真正改过/新增的内容。"""
import shutil
from pathlib import Path

ROOT = Path(r"C:\Local Repositories\PrivateVoice")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
OUT = ROOT / "dist" / "update_beta2"
SP = PKG / "runtime" / "Lib" / "site-packages"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

items = [
    ("app", PKG / "app", True),
    ("assets", PKG / "assets", True),
    ("nltk_data", PKG / "nltk_data", True),
    ("tools", PKG / "tools", True),
    ("characters", PKG / "characters", True),
]
for name, src, whole in items:
    if src.exists():
        shutil.copytree(src, OUT / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

# 新增的语言运行时（eunjeon / 日语词典）
for rel in ("eunjeon", "pyopenjtalk"):
    src = SP / rel
    if src.exists():
        shutil.copytree(src, OUT / "runtime" / "Lib" / "site-packages" / rel,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """Ahee52_PrivateVoice beta-2 增量更新包
=====================================

适用：已安装 beta-1（Ahee52_PrivateVoice-ASR-cn-beta）的机器。

安装方法（复制粘贴即可）：
1. 关闭正在运行的程序（托盘图标 → 退出）
2. 把本文件夹里的 app、assets、nltk_data、tools 四个目录
   整个复制到安装目录（默认 %ProgramFiles%\\Ahee52_PrivateVoice\\），覆盖同名目录
3. 如果 runtime 目录存在，同样把 runtime 里的内容覆盖过去
   （只有新增的多语言组件 eunjeon / pyopenjtalk 词典，约 250MB）
4. 重新启动程序

本次 beta-2 更新内容：
- 英文/日文/韩文/粤语 合成自动切换（新增语言词典，含 nltk 英文数据）
- 修英文回复缺少空格与标点（朗读连成一片）
- 修偶发"整句没声音"：换随机种子重试 3 次，仍失败则跳过并提示
- 训练后自检：合成一句验证音量，过低会警告
- 参考音频自动响度归一化，素材偏轻也能训出正常音量
- 联网检索：维基（合规 UA）+ Bing + 百度兜底；搜索前先播"稍等，让我查一下。"
- 回复长度自适应：简单 1-2 句、复杂 4-6 句、新闻/科普可更长
- 语速回到 1.0（去掉拉伸瑕疵），句间/语言切换处加短停顿
"""
(OUT / "安装说明.txt").write_text(readme, encoding="utf-8")

size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
files = sum(1 for f in OUT.rglob("*") if f.is_file())
print(f"update_beta2: {size:.1f} MB / {files} files -> {OUT}")
