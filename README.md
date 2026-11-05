# PrivateVoice Ahee52 (Beta)

**中文** | [English](#english)

> 本地运行的私人语音陪伴程序：中文/英文界面、本地大模型、GPT-SoVITS v4 语音克隆、可选联网检索。
> Local-first voice companion with bilingual UI, local LLM, GPT-SoVITS v4 voice cloning and optional web search.

---

## 中文

### 这是什么

PrivateVoice Ahee52 是一个 **完全本地运行** 的语音陪伴程序：

- 按住热键说话 → 本地 ASR（FunASR paraformer，可选 Whisper）→ 本地 LLM → GPT-SoVITS v4 克隆音色朗读
- 全中文 / 全英文界面，可随时切换（设置或主界面按钮）
- 训练自己的音色：上传 45 秒–5 分钟素材，程序自动质检、切分、训练 LoRA
- 可选联网检索（Bing / 百度 / 维基百科），问新闻、查资料时会先回应“稍等，让我查一下”

### 本仓库包含什么

本仓库是 **程序骨架与 UI**，**不包含模型权重**（体积过大）。首次运行按你的选择下载：

| 模型 | 用途 | 大小 |
|---|---|---|
| GPT-SoVITS v4（s1v3 / s2Gv4 / vocoder） | 语音合成 | ~940 MB |
| chinese-roberta / chinese-hubert | 文本与语义前端 | ~800 MB |
| FunASR paraformer-zh（中文 ASR，默认） | 语音识别 | ~840 MB |
| faster-whisper small（英文/多语 ASR，可选） | 语音识别 | ~480 MB |

首次启动向导里有 **ASR 模型选择**（中文 paraformer / English whisper），点「下载缺失模型」即可。

### 快速开始（从源码）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 上游项目不随本仓库分发，需自行 clone 到同级目录
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git clone https://github.com/Lucas1479/Aqua-TTS.git

python app/main.py
```

首次启动：选语言 → 检查显卡 → 选 ASR 并下载 → 配置 LLM（任意 OpenAI 兼容接口）。

### 目录结构

```
app/
  main.py           入口、控制器、托盘、热键
  config.py         配置与训练档位
  i18n.py           中英文文案
  core/             ASR / LLM / TTS / 播放 / 训练 / 质检 / 检索 / 模型下载
  ui/               主界面、角色库、设置、诊断、首启向导、主题
tools/              辅助脚本（自检、麦克风自检、角色检查）
assets/prompts/     系统提示词
installer.iss       Inno Setup 打包脚本
```

### 使用要点

- **热键**：默认 `shift+a`，可改任意组合（也支持单独按 Shift）
- **训练音色**：角色库 →「上传素材/训练…」；安静环境、离麦 15–20cm、关闭降噪
- **显存**：16GB 可训（默认「均衡」档），不足时自动降 batch
- **缓存**：`results/tts` 保留最近 10 分钟合成音频，可一键清理

### 已知限制

- 仅 Windows 验证（CUDA 11.8+，建议 12GB+ 显存）
- 极端音色（很闷/很低）仍可能偶发拖音：角色页把「重复惩罚」调到 1.9–2.1
- 联网检索为网页解析，目标站点改版可能失效

### 致谢与许可

本项目基于以下开源项目，**详见 [LICENSE](LICENSE)**：

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — MIT
- [Aqua-TTS](https://github.com/Lucas1479/Aqua-TTS) — MIT
- [FunASR](https://github.com/modelscope/FunASR) — MIT
- [PySide6](https://www.qt.io/qt-for-python) — LGPLv3
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — MIT
- 模型权重各自遵循发布方许可

---

## English

### What this is

PrivateVoice Ahee52 is a **fully local** voice companion:

- Hold a hotkey → local ASR (FunASR paraformer, or Whisper) → local LLM → GPT-SoVITS v4 cloned voice
- Full **Chinese / English UI**, switchable anytime
- Train your own voice from 45 s – 5 min of audio; QC, segmentation and LoRA training are automated
- Optional web lookup (Bing / Baidu / Wikipedia) — replies “let me check” before searching

### What this repo contains

**Application skeleton and UI only** — no model weights. The first-run wizard downloads what you pick:

| Model | Purpose | Size |
|---|---|---|
| GPT-SoVITS v4 (s1v3 / s2Gv4 / vocoder) | speech synthesis | ~940 MB |
| chinese-roberta / chinese-hubert | text & semantic front-end | ~800 MB |
| FunASR paraformer-zh (Chinese ASR, default) | speech recognition | ~840 MB |
| faster-whisper small (English, optional) | speech recognition | ~480 MB |

### Quick start (from source)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git clone https://github.com/Lucas1479/Aqua-TTS.git

python app/main.py
```

### Highlights

- Push-to-talk with any global hotkey (a single modifier such as Shift works)
- Character editor exposes voice/pace parameters (speed, sentence gap, repetition penalty, top-k, sampling steps) with inline explanations
- Training ends with a 3-sentence self-test; silent models are reported instead of silently shipped
- Cache cleanup: `results/tts` keeps the last 10 minutes of audio

### Limitations

- Windows-only so far (CUDA 11.8+, 12 GB+ VRAM recommended)
- Very dark or low-pitched source audio can still drag vowels — raise “repetition penalty” to 1.9–2.1
- Web lookup is HTML scraping; sites may change

### Credits & license

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — MIT
- [Aqua-TTS](https://github.com/Lucas1479/Aqua-TTS) — MIT
- [FunASR](https://github.com/modelscope/FunASR) — MIT
- [PySide6](https://www.qt.io/qt-for-python) — LGPLv3
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — MIT
- Model weights keep their own upstream licenses

---

**本地整合包版本 / Integrated build:** `PrivateVoice_Ahee52 1.0.0`
