import pathlib
import shutil

ROOT = pathlib.Path(r"C:\Local Repositories\PrivateVoice")
PKG = ROOT / "dist" / "Ahee52_PrivateVoice"
HK = ROOT / "dist" / "update_beta2_hotfix4"

if HK.exists():
    shutil.rmtree(HK)
HK.mkdir(parents=True)
for name in ("app", "assets"):
    shutil.copytree(PKG / name, HK / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

readme = """beta-2 热修 4（app + assets）
============================

针对"训练出来没声音"的参数与自愈改动：

1) 自适应 LoRA rank
   - 默认 32；检测到素材偏闷（4kHz 以上 <2%）或基频偏低（F0 <130Hz，男低音）
     自动提升到 48，训练日志会打印：
     检测到低沉/偏闷素材（F0≈xxxHz，4k+ 占比 x.x%），LoRA rank 从 32 提升到 48
   - 实测同一素材：rank32 自检峰值 0.66/0.51/0.63，rank48 提升到 0.80/0.56/0.67

2) 参考音频完整性校验与自动修复
   训练收尾会解码一次 reference.wav；失败则自动用数据集原片段重建。
   推理前也会校验，损坏就用角色 dataset/clips 里的第一段重建，
   避免"参考音频打不开 -> 整段静音"。

3) 训练自检固定 3 句，明确报告每句峰值；<2 句出声判角色不可用。

4) 角色音色异常时，对话自动改用基础音色兜底（不会全哑）。

安装：
1. 退出程序
2. 把 app、assets 两个目录复制到 D:\\Ahee52_PrivateVoice\\ 覆盖
3. 重新启动，重新训练角色
"""
(HK / "安装说明.txt").write_text(readme, encoding="utf-8")
size = sum(f.stat().st_size for f in HK.rglob("*") if f.is_file()) / 1e6
print(f"hotfix4: {size:.2f} MB")
