# 安装包瘦身记录（v2，最终：仅 v4）

> 2026-11-02。目标：去掉旧 v3 角色相关内容，但**不能碰 v4 链路共用的模型**。

## 1. 结果

| 阶段 | 体积 |
|---|---|
| 瘦身前（v3+v4 双角色） | 12.62 GB |
| 删旧 epoch + `__pycache__` | 12.04 GB |
| **删 v3 专属（本轮）** | **10.59 GB** |

## 2. 这一轮删了什么（1.44 GB）

| 项目 | 大小 | 为什么 v4 不需要它 |
|---|---|---|
| `characters\格雷丝-748caf\`（v3 角色） | 224 MB | 只保留 `格雷丝_v4` |
| `GPT-SoVITS\SoVITS_weights_v3\` | 72 MB | v3 LoRA，v4 用 `SoVITS_weights_v4` |
| `pretrained_models\s2Gv3.pth` | 733 MB | v3 底模；v4 用 `gsv-v4-pretrained/s2Gv4.pth` |
| `pretrained_models\models--nvidia--bigvgan_v2_24khz_100band_256x\` | 429 MB | v3 的 24k 声码器；v4 用 `vocoder.pth`（48k） |
| `Aqua-TTS\`（aquatts） | 20 MB | Aqua 只服务 v3 |

**关键改动**：删掉 `Aqua-TTS` 后，`app/core/tts.py` 不能再"先试 Aqua、失败再回退"
（那样会直接 `ImportError`）。新增 `_detect_version()`：先读 checkpoint 头判断版本，
`v4` 直接走 `_load_upstream_v4()`，完全不碰 `aquatts`。

## 3. 明确保留（名字带 v3，但 v4 在用）

| 项目 | 大小 | 为什么不能删 |
|---|---|---|
| `s1v3.ckpt` | 148 MB | **v4 的 GPT/T2S 部分仍用它** |
| `GPT_weights_v3\vlab_格雷丝-e8.ckpt` | 148 MB | 目录名带 v3，但这就是 v4 角色的 GPT 权重 |
| `chinese-roberta-wwm-ext-large` | 621 MB | 文本前端，v4 用 |
| `chinese-hubert-base` | 180 MB | 语义特征，v4 用 |
| `gsv-v4-pretrained\{s2Gv4.pth, vocoder.pth}` | 788 MB | v4 底模 + 48k 声码器 |
| `fast_langdetect\lid.176.bin` | 125 MB | v4 上游推理的语言分段 |
| `models_cache\` | 2.11 GB | ASR / 标点 / 说话人（离线） |
| `runtime\` | 5.79 GB | 其中 torch 4.19 GB |

> `GPT_weights_v3` 这个目录名容易误解：它只是 Stage 2 训练时按 v3 命名的产物目录，
> 里面放的就是 v4 角色用的那个 `.ckpt`。没有改名（要同步 profile 与打包脚本，收益为零）。

## 4. 验证

删除后在包内跑 `tools\selftest.py`：

```
[ok] character · 1 个 ready
[ok] checkpoints
e2e=4503.9ms  asr=358.1ms  llm=741.9ms  tts=1403.3ms  audio=3.812s
SELFTEST PASS
```

即：只靠 v4 角色 + `s2Gv4` + `s1v3` + roberta/hubert + vocoder，链路完整。
