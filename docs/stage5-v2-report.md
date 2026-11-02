# Stage 5（v2 版本）变更报告：v4 语音 + 情绪档位 + 深色主题

> 日期：2026-11-02
> 安装包：`dist\installer\VoiceCompanion_v2_Setup.exe`（12.6 GB，分卷）
> 包内模型：GPT-SoVITS **v4**（`s2Gv4.pth` + `vocoder.pth`）+ 我们重训的 v4 LoRA

---

## 1. 为什么换 v4

官方 README 的 V4 说明原文：

> Version 4 fixes the issue of **metallic artifacts in Version 3 caused by non-integer multiple
> upsampling**, and natively outputs **48k audio to prevent muffled sound** (whereas Version 3 only
> natively outputs 24k audio).

实测频段（同一段文本、同一个人的音色）：

| 版本 | 采样率 | 0-4k | 4-8k | 8-12k | 12-16k | 16-24k |
|---|---|---|---|---|---|---|
| v3 | 24 kHz | 99.6% | 0.3% | 0.2% | — | — |
| **v4** | **48 kHz** | 89.0% | **5.5%** | **2.8%** | 1.2% | 1.6% |

4kHz 以上能量从 0.5% 提到 11%，这就是"闷/颗粒感"的直接来源。

## 2. v4 的工程接入

- Aqua-TTS 0.3.0（PyPI 最新）**不支持 v4**：`checkpoints.py` 里硬编码
  `SUPPORTED_MODELS = {v1, v2, v2Pro, v2ProPlus, v3}`，加载 v4 权重直接抛
  `Unsupported SoVITS architecture 'v4'`。
- 做法：`app/core/tts.py` 里 `AquaTTSEngine.load()` 捕获该异常后自动切到
  **upstream-v4** 后端——直接用 GPT-SoVITS 自带的 `TTS_infer_pack.TTS.run()`，
  不改上游源码。
- 需要补两个东西：
  1. `torchaudio.load` 在 torchaudio 2.11 依赖 `torchcodec`（未安装）→ 进程内用
     soundfile 打补丁；
  2. `pretrained_models/fast_langdetect/lid.176.bin`（125MB，上游语言分段要用）。
- 参数：`text_split_method=cut5`、`fragment_interval=0.3`、`sample_steps=4`。

实测（3090，预热后）：**首包 580 ms、RTF 0.48、输出 48 kHz**。
（第一次调用有 ~5s CUDA Graph 预热，已被启动预热吸收。）

## 3. LLM 改动

- `assets/prompts/companion_system.txt` 增加两节：
  - **反重复**：同一问题不要复读，要换角度/换开头，可以承认"刚说过啦"；
  - **思考深度**：先判断意图（求安慰/求建议/闲聊/试探）再回答，先结论后一句理由。
- 新增**情绪档位**（`llm.styles`）：温柔 / 活泼 / 冷静 / 毒舌 / 低落。
  每档带 `temperature`、`top_p`、`repeat_penalty` 和一句"本轮情绪"指令，
  主界面下拉即切，下一轮生效。
- LLM 请求现在支持 `top_p / top_k / min_p / repeat_penalty / presence_penalty /
  frequency_penalty`；`llm.enable_thinking=true` 可打开思考（会更慢）。

## 4. 界面（深色 + 紫）

- `app/ui/theme.py`：深色渐变底（`#0b0d16 → #161226`）、紫色文字（`#b388ff / #d9cdf5`）、
  紫色渐变主按钮、圆角输入框/表格/滚动条。
- 主界面新增：情绪下拉、**清理缓存**按钮。
- 角色管理新增 **「上传素材/训练…」** 按钮：选中角色 → 直接进向导第 2 步（上传音频），
  跳过"基本信息"；训练完成弹窗提示。
- 截图：`results/screenshots/14_main_dark.png`、`15_manager_dark.png`。

## 5. 缓存目录与一键清理

安装在 `D:\VoiceCompanion_v1`（旧）或 `%ProgramFiles%\VoiceCompanion_v2`（新）时，
程序会写这些位置：

| 路径 | 内容 | 能否删 |
|---|---|---|
| `logs\app.log*` | 运行日志（5MB 轮转） | ✅ 可删 |
| `logs\turns.jsonl` | 每轮延迟埋点 | ✅ 可删 |
| `datasets\<角色id>\work\` | 训练中间产物（wav24k、train.list、配置） | ✅ 可删 |
| `datasets\_probe` 等 `_` 开头目录 | 调试残留 | ✅ 可删 |
| `results\quality\`、`results\compare\` | 音质诊断 / A-B 音频 | ✅ 可删 |
| `%TEMP%\vlab-*` | 解码临时 wav | ✅ 可删 |
| `characters\<id>\` | 角色本体（profile/ckpt/reference/dataset） | ❌ 删了角色就没了 |
| `models_cache\` | 内置 ASR/标点/说话人模型 2GB | ❌ 删了要重新下载 |
| `GPT-SoVITS\pretrained_models\` | v3/v4 底模（~3GB） | ❌ 删了不能推理 |

主界面「清理缓存」按钮 = `core/housekeeping.py`：**白名单**删除上面 ✅ 的项，
删前弹窗列出各目录占用，删后报告释放了多少 MB；永不触碰角色与模型。

## 6. 已知问题

- **v4 首包比 v3 慢**（580ms vs 313ms），因为上游栈没有 Aqua 的静态 KV/CUDA Graph 优化。
- **v4 角色目前只有 `格雷丝_v4` 一个**（用同一批素材重训），旧 `格雷丝` 仍是 v3。
- `fast_langdetect` 模型 125MB 已进包；如果换机器仍报缺模型，是路径问题不是下载问题。
- 5070 Ti 仍未实机验证。
