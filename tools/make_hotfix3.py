import pathlib
import shutil

ROOT = pathlib.Path(r"C:\Local Repositories\PrivateVoice")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
HK = ROOT / "dist" / "update_beta2_hotfix3"

if HK.exists():
    shutil.rmtree(HK)
HK.mkdir(parents=True)

for name in ("app", "assets"):
    shutil.copytree(PKG / name, HK / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """beta-2 热修 3（app + assets）
============================

修复 / 调整：
1) 角色音色失效时的兜底：连续合成不出声音时，本回合自动改用"基础音色"（s1v3 + s2Gv4，
   不带 LoRA），并提示"角色音色异常，本回合临时改用基础音色"——不会再整段没声音。
2) 回复长度收紧：默认 1-2 句、≤40 字；只有明确要求解释才 3-4 句；
   新闻/科普最多 8 句。max_tokens 默认 180、联网 512。
3) 训练完成弹窗去掉"释放模型/加载模型"提示，改为"点确定后自动重启"。

安装：
1. 退出程序
2. 把 app 和 assets 两个目录复制到安装目录覆盖
   （你的机器是 D:\\Ahee52_PrivateVoice\\）
3. 重新启动

说明：已经训坏的 emily 不会因为这次更新变好，需要重训；
但即使不重训，现在对话也会用基础音色把话说出来，而不是静音。
"""
(HK / "安装说明.txt").write_text(readme, encoding="utf-8")

size = sum(f.stat().st_size for f in HK.rglob("*") if f.is_file()) / 1e6
files = sum(1 for f in HK.rglob("*") if f.is_file())
print(f"hotfix3: {size:.2f} MB / {files} files")
