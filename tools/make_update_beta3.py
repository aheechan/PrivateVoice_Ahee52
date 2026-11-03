"""beta-3 增量包：app + assets + tools + characters（默认音色）。"""
import pathlib
import shutil

ROOT = pathlib.Path(r"C:\VoiceLab")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
HK = ROOT / "dist" / "update_beta3"

if HK.exists():
    shutil.rmtree(HK)
HK.mkdir(parents=True)
for name in ("app", "assets", "tools", "characters"):
    shutil.copytree(PKG / name, HK / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """Ahee52_PrivateVoice beta-3 增量更新
====================================

这一版的改动：

1) 质检不再拦人
   第 3 步质检所有项目只做"提示"，不会出现"不通过"、也不会阻止进入下一步。
   台词从"有不合格项，修好再继续"改为"有提示项，仍可继续（不影响训练流程）"。

2) 内置默认基础音色
   删除了格雷丝（grace）的克隆数据与角色，新增一个开箱可用的角色「默认音色」。
   它用 s1v3 + s2Gv4 底模 + 一段合成参考音频（assets/default_voice.wav），
   不属于任何真人音色，第一次启动就能直接对话。

3) 参考音频原子写 + 校验（沿用 beta-2 热修）
4) 自适应 LoRA rank（低沉/闷素材自动 32→48）
5) 角色自查工具 tools/check_character.py

安装（在已安装 beta-2 的机器上）：
1. 退出程序
2. 把 app、assets、tools、characters 四个目录复制到安装目录覆盖
   （D:\\Ahee52_PrivateVoice\\）
3. 重新启动。角色库里应该只剩「默认音色」

注意：本次会覆盖 characters 目录，你之前训练的角色会被移除；
      如果还想保留，请先把 characters 目录备份。
"""
(HK / "安装说明.txt").write_text(readme, encoding="utf-8")
size = sum(f.stat().st_size for f in HK.rglob("*") if f.is_file()) / 1e6
files = sum(1 for f in HK.rglob("*") if f.is_file())
print(f"update_beta3: {size:.2f} MB / {files} files")
