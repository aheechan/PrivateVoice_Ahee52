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

## 安装

### A. 普通用户：下载安装包（推荐）

**不需要装 Python，也不需要编译任何东西。**

1. 打开本仓库右侧 **Releases** 页面，下载最新版本的 4 个文件：

   ```
   PrivateVoice_Ahee52-1.0.0-Setup.exe
   PrivateVoice_Ahee52-1.0.0-Setup-1.bin
   PrivateVoice_Ahee52-1.0.0-Setup-2.bin
   PrivateVoice_Ahee52-1.0.0-Setup-3.bin
   ```

   > ⚠️ 四个文件必须放在**同一个文件夹**里，`.exe` 会自己去找同目录的 `.bin`。

2. 双击 `PrivateVoice_Ahee52-1.0.0-Setup.exe`
   - 若 Windows 弹出「已保护你的电脑」，点 **更多信息 → 仍要运行**（安装包未做代码签名）
   - 默认装到 `C:\Program Files\PrivateVoice_Ahee52`，约需 12 GB

3. 装完桌面有 `PrivateVoice_Ahee52` 快捷方式；首次启动向导：选语言 → 显卡自检 → 模型检查（已内置）→ 填 LLM 地址

4. 「对话模型」填任意 OpenAI 兼容接口，例如本地 llama.cpp：

   ```
   Base URL: http://127.0.0.1:8082/v1
   模型名:   Qwen3.5-9B
   ```

   点「测试连接」通过后「开始使用」。

5. 按住 `shift+a` 说话（也可在主界面打字）。

**卸载**：设置 → 应用 → `PrivateVoice_Ahee52`。训练出的角色在安装目录 `characters\` 下，卸载前请备份。

### B. 开发者：从源码运行

前提：Windows 10/11、Python 3.12、NVIDIA 显卡（CUDA 11.8+ 驱动）、约 15 GB 磁盘。

```bash
git clone <本仓库>
cd PrivateVoice_Ahee52

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 上游推理框架不随本仓库分发，clone 到仓库同级目录
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git clone https://github.com/Lucas1479/Aqua-TTS.git

python app/main.py
```

首次启动点 **「下载缺失模型」**（约 2.6 GB，需联网）：

- GPT-SoVITS v4 底模 + 48k 声码器
- chinese-roberta / chinese-hubert 文本前端
- ASR 二选一：**中文 paraformer**（默认）或 **English whisper**

也可手动下载，清单见 `app/core/models.py`。

### 首次启动会做什么

| 步骤 | 说明 |
|---|---|
| 选语言 | 中文 / English，之后可在主界面或托盘切换 |
| 显卡自检 | 需 RTX 20 系以上；显存 <8GB 推理会紧张 |
| 模型检查 | 安装包已内置；源码用户在此下载 |
| LLM 配置 | 任意 OpenAI 兼容接口，不填也能启动但不能对话 |

### 常见问题

- **装完打不开/闪退**：看 `安装目录\logs\app.log`，多为显卡驱动或杀软拦截
- **没有麦克风**：主界面有文字输入框，功能完整
- **提示已在运行**：程序是单实例的，检查右下角托盘
- **训练显存不足**：训练前会自动卸载语音模型；仍失败请换素材或降档
- **模型下载慢**：源码用户可手动下载到 `GPT-SoVITS/GPT_SoVITS/pretrained_models/`

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

## Installation

### A. End users — download the installer (recommended)

**No Python, no build tools needed.**

1. Open the **Releases** page and download all four files:

   ```
   PrivateVoice_Ahee52-1.0.0-Setup.exe
   PrivateVoice_Ahee52-1.0.0-Setup-1.bin
   PrivateVoice_Ahee52-1.0.0-Setup-2.bin
   PrivateVoice_Ahee52-1.0.0-Setup-3.bin
   ```

   > ⚠️ Keep all four in **the same folder** — the `.exe` looks for the `.bin` files next to it.

2. Run `PrivateVoice_Ahee52-1.0.0-Setup.exe`
   - If SmartScreen appears, choose **More info → Run anyway** (unsigned installer)
   - Defaults to `C:\Program Files\PrivateVoice_Ahee52`, needs ~12 GB

3. Launch `PrivateVoice_Ahee52`; wizard: language → GPU → models (bundled) → LLM endpoint

4. Fill in any OpenAI-compatible endpoint, e.g. a local llama.cpp server:

   ```
   Base URL: http://127.0.0.1:8082/v1
   Model:    Qwen3.5-9B
   ```

   “Test connection”, then “Start”.

5. Hold `shift+a` to talk, or type in the main window.

**Uninstall**: Settings → Apps → `PrivateVoice_Ahee52`. Trained characters live in `characters\` inside the install folder — back them up first.

### B. Developers — run from source

Requirements: Windows 10/11, Python 3.12, NVIDIA GPU (CUDA 11.8+), ~15 GB disk.

```bash
git clone <this repo>
cd PrivateVoice_Ahee52

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# upstream inference stacks are not vendored; clone them next to this repo
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git clone https://github.com/Lucas1479/Aqua-TTS.git

python app/main.py
```

On first run click **“Download missing models”** (~2.6 GB):

- GPT-SoVITS v4 base + 48 kHz vocoder
- chinese-roberta / chinese-hubert front-ends
- ASR: **Chinese paraformer** (default) or **English whisper**

Manual download paths are listed in `app/core/models.py`.

### First run overview

| Step | Notes |
|---|---|
| Language | Chinese / English; switchable later from the main window or tray |
| GPU check | RTX 20-series or newer; <8 GB VRAM is tight |
| Models | Bundled in the installer; downloaded for source users |
| LLM | Any OpenAI-compatible endpoint; app starts without it but cannot chat |

### Troubleshooting

- **Won't start / crashes**: check `logs\app.log`; usually GPU driver or antivirus
- **No microphone**: the main window has a text input
- **“Already running”**: single-instance app — check the system tray
- **Training OOM**: voice models are unloaded automatically; if it persists, use a lighter profile
- **Slow model download**: fetch from HuggingFace into `GPT-SoVITS/GPT_SoVITS/pretrained_models/`

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
