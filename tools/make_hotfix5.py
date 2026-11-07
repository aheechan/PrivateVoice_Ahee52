import pathlib
import shutil

ROOT = pathlib.Path(r"C:\Local Repositories\PrivateVoice")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
HK = ROOT / "dist" / "update_beta2_hotfix5"

if HK.exists():
    shutil.rmtree(HK)
HK.mkdir(parents=True)
for name in ("app", "assets", "tools", "characters"):
    shutil.copytree(PKG / name, HK / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """beta-2 热修 5 / 等同 beta-3 代码（app + assets + tools + characters）
================================

1) 参考音频改为原子写（先写 .tmp，校验能读回来再换名）
   —— 避免"写入过程中被读到半截文件"，这正是 emily 那种
      'Error opening reference.wav: System error' 的典型成因。
2) 新增角色自查工具：tools\\check_character.py
   用法（在安装目录下）：
       runtime\\python.exe tools\\check_character.py
       runtime\\python.exe tools\\check_character.py emily
   会打印每个角色的权重是否齐全、reference.wav 是否可解码/多大/多长/峰值，
   以及可用的 dataset 片段数量。
3) 参考音频校验与自动修复（训练收尾 + 推理前）沿用上一版。

安装：
1. 退出程序
2. 把 app、assets、tools 三个目录复制到 D:\\Ahee52_PrivateVoice\\ 覆盖
3. 重新启动
"""
(HK / "安装说明.txt").write_text(readme, encoding="utf-8")
size = sum(f.stat().st_size for f in HK.rglob("*") if f.is_file()) / 1e6
print(f"hotfix5: {size:.2f} MB")
