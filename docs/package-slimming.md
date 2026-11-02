# 安装包瘦身记录（v2）

> 2026-11-02，目标：去掉冗余但不影响功能

## 1. 包的构成（瘦身前 12.62 GB）

| 目录 | 大小 | 说明 |
|---|---|---|
| `runtime\` | 5.79 GB | 嵌入式 Python + 依赖；torch 单独就 4.19 GB |
| `GPT-SoVITS\` | 4.55 GB | 代码 + v3/v4 底模 + 训练产物 |
| `models_cache\` | 2.11 GB | ASR / 标点 / 说话人模型（离线用） |
| `characters\` | 0.45 GB | 两个角色（格雷丝 v3、格雷丝_v4） |
| 其余 | <1 MB | app、assets、bat |

`GPT-SoVITS` 内部：`GPT_SoVITS\pretrained_models` 2.84 GB（s2Gv3 733MB、gsv-v4 788MB、
roberta 621MB、bigvgan-24k 429MB、hubert 180MB、s1v3 148MB、fast_langdetect 125MB），
权重目录 0.88 GB（9 个 epoch 文件），代码 ~0.1 GB。

## 2. 已删除（安全，共 ~716 MB）

| 项目 | 大小 | 为什么安全 |
|---|---|---|
| `SoVITS_weights_v3\*_e6_*`、`*_e7_*` | 144 MB | 角色只引用 `e8`；e6/e7 是训练过程中的历史快照 |
| `SoVITS_weights_v4\*_e6_*`、`*_e7_*` | 144 MB | 同上 |
| `GPT_weights_v3\*-e6.ckpt`、`*-e7.ckpt` | 296 MB | 同上，角色只用 `-e8.ckpt` |
| `**\__pycache__\`（约 4900 个目录） | 120 MB | Python 首次运行会重建 |
| 合计 | **~716 MB** | 12.62 → 11.92 GB（跑完自检后 __pycache__ 重建，最终安装包 12.04 GB） |

删除后跑分发包自检：**SELFTEST PASS**（v3 角色走 Aqua、v4 角色走 upstream-v4 都能出声）。

## 3. 保留但"看起来冗余"的东西（删了会坏）

| 项目 | 大小 | 为什么不能删 |
|---|---|---|
| `torch\` | 4.19 GB | v3/v4 推理与训练都靠它，含 CUDA 库 |
| `s2Gv3.pth` | 733 MB | v3 角色（格雷丝）的底模；v3 LoRA 必须配它 |
| `models--nvidia--bigvgan_v2_24khz_100band_256x` | 429 MB | v3 的 24k 声码器，v3 角色出声必需 |
| `SoVITS_weights_v3`（e8） | 72 MB | 格雷丝 v3 的 LoRA |
| `s1v3.ckpt` | 148 MB | **v4 也用**（GPT/T2S 部分仍用 s1v3） |
| `gsv-v4-pretrained`（s2Gv4 + vocoder） | 788 MB | v4 底模与 48k 声码器 |
| `chinese-roberta-wwm-ext-large` / `chinese-hubert-base` | 801 MB | 文本前端与语义特征，v3/v4 共用 |
| `fast_langdetect\lid.176.bin` | 125 MB | v4 上游推理的语言分段要用 |
| `models_cache` | 2.11 GB | 离线 ASR/标点/说话人模型 |
| `characters\<id>\checkpoints` | 0.44 GB | `.vcp` 导出/搬运角色靠它（与权重目录是两份） |
| `Aqua-TTS` | 20 MB | v3 角色的推理运行时 |

## 4. 还可以再砍（选项，需你确认）

**方案 A：彻底去掉 v3 支持**（≈ -1.46 GB → 10.46 GB）

- 删 `s2Gv3.pth`（733MB）、`bigvgan-24k`（429MB）、`SoVITS_weights_v3`（72MB）、
  角色 `格雷丝`（224MB）
- 代价：旧 v3 角色和"A/B 对比"没了；`Aqua-TTS` 也可一并删（但只省 20MB）

**方案 B：删 webui 专用依赖**（≈ -110 MB）

- `gradio` 52MB、`ctranslate2` 61MB（faster-whisper 用，GPT-SoVITS 的 ASR 工具才需要）
- 风险：上游脚本若在运行时 import，会报 ModuleNotFound；需要逐个验证

**方案 C：删 site-packages 里的 `tests`**（≈ -60 MB）

- 少数包会 import 自己的 tests，风险低但非零

我的建议：**保持现状**。11.92 GB 里 4.19 GB 是 torch、2.11 GB 是离线模型，剩下的才是可动的；
方案 A 省 1.46 GB 但丢掉 v3 兼容，看你还要不要留"老角色对比"。
