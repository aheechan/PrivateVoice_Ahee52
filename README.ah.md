# PrivateVoice_Ahee52

[English](README.md) | **中文**

**完全本地运行的中文情感陪伴器：本地 ASR、本地大模型、GPT-SoVITS v4 音色克隆，可选联网检索。**

按住热键说话，它用克隆出来的声音回答你。音频不出本机。

> 作者：**Ahee52**
> 状态：**1.0.3**（首个稳定版）
> 整合包：**PrivateVoice_Ahee52-1.0.3**，1 个安装器 + 7 个分卷，共 10.85 GB
> 安装目录：`%ProgramFiles%\PrivateVoice_Ahee52`
> 配置与日志：`<安装目录>\config.yaml`、`<安装目录>\logs\app.log`
> 模型位置（安装版）：`<安装目录>\GPT-SoVITS\GPT_SoVITS\pretrained_models`
> 模型位置（源码版）：`<仓库>\models_cache` 与 `<仓库>\GPT-SoVITS\...\pretrained_models`

---

## ⚠️ 使用前必读

**这是一个情感陪伴器，不是医疗器械，也不是紧急求助工具。**
它不能替代家人、朋友或专业心理咨询。如果你或身边的人正处于危机中，
请联系当地紧急服务或专业机构。

**用他人的声音训练音色，必须先取得对方明确的、可撤回的授权。**
本程序的语音克隆能力仅用于你**本人**的声音，或你已取得**书面同意**的声音。

明确禁止：

- 冒充他人（亲友、公众人物、客服、官方机构）
- 欺诈、骚扰、色情或误导性内容
- 在对方不知情的情况下克隆其声音

使用者需自行遵守当地法律；作者不对滥用行为承担责任。

---

## 为什么做这个

最初只是为了让自己在安静的夜里，有个能说说话的声音。

做完之后发现，它可能对更多人有用：

- **独居老人**——有人应答、能提醒吃药、能听他们讲过去的事
- **长期卧床或行动不便者**——不用起身、不用打字，说一句就有回应
- **异地独居的人**——加班回家后，至少屋里有个声音
- **视障用户、语言学习者**——纯语音交互，不依赖屏幕

如果你愿意把它改造到真实场景里——养老院、社区关怀、无障碍辅助——**非常欢迎**。
这是这个项目最想被用到的方向。

---

## 这个仓库是什么

一个能跑起来的项目，不只是一个安装包下载页。

```text
app/               管线、ASR / LLM / TTS / 播放、训练、界面、多语言文案
tools/             辅助脚本（麦克风自检、角色检查、打包）
assets/prompts/    系统提示词
docs/              各阶段报告与研究笔记
tests/             单元测试
installer.iss      Inno Setup 打包脚本
```

| 包含 | 不包含 |
|---|---|
| 完整应用代码与中英双语界面 | 模型权重（约 2.6 GB，见下方下载表） |
| 训练、质检、切分工具 | 大语言模型（需自备 OpenAI 兼容接口） |
| 打包脚本 | 角色数据与训练产物 |
| | 上游 GPT-SoVITS 源码（需自行 clone） |

---

## 两种使用方式

| 路线 | 适合谁 | 需要什么 |
|---|---|---|
| **安装包** | 普通用户 | 从 Releases 下载 8 个文件（1 个 `.exe` + 7 个 `.bin`） |
| **源码** | 开发者 / 想改代码的人 | Python 3.12、NVIDIA 显卡、约 20 GB 磁盘、下载模型 |

整合包已内置默认音色与全部模型，装完即可对话，不需要 Python。

---

## 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| 操作系统 | Windows 10 x64（21H2+） | Windows 11 x64 |
| 显卡 | NVIDIA RTX 2060 / 6 GB 显存 | **RTX 3090 / 4070 Ti / 5070 Ti，16–24 GB** |
| 驱动 | 支持 CUDA 12.8（≥ 570.x） | 最新 Game Ready / Studio |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 15 GB 空闲 | 30 GB（多角色 + 训练中间产物） |
| 声卡 | 任意麦克风 | 带监听回路的 USB 声卡或耳机 |
| 网络 | 对话本身不需要 | 仅首次下模型、联网检索时需要 |

说明：

- **推理**约需 6–8 GB 显存；**训练**在 24 GB 上最稳，16 GB 会自动选择「均衡」档并降低 batch。
- 不支持纯 CPU 推理（约 30–60 秒/句，且 v4 依赖 GPU 算子）。
- 显存不足时训练会明确提示，不会静默失败。

---

## 从源码复现

### 0. 前置条件

- Windows 10/11 x64
- Python **3.12**（整合包用的是 3.12 嵌入式运行时）
- Git、NVIDIA 驱动
- 约 20 GB 空闲磁盘

### 1. 取代码与上游

```bash
git clone <本仓库地址> PrivateVoice
cd PrivateVoice

# 上游推理框架不随本仓库分发
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
```

> `Aqua-TTS` 只在遗留的 v3 引擎（`app/core/tts.py` 的 `AquaTTSEngine`）里用到，
> 默认的 v4 管线不依赖它，可以不装。

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate

# torch 需匹配你的 CUDA 版本，整合包用的是 cu128
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

### 3. 下载模型

所有路径都**相对项目根目录**，程序按这些位置加载：

```text
PrivateVoice\
├─ GPT-SoVITS\
│  └─ GPT_SoVITS\pretrained_models\
│     ├─ s1v3.ckpt                                       148 MB
│     ├─ gsv-v4-pretrained\s2Gv4.pth                     733 MB
│     ├─ gsv-v4-pretrained\vocoder.pth                    55 MB
│     ├─ chinese-roberta-wwm-ext-large\model.safetensors 621 MB
│     └─ chinese-hubert-base\pytorch_model.bin           181 MB
├─ models_cache\
│  ├─ modelscope\models\         ← 中文 ASR（FunASR 自动落盘）
│  └─ whisper\faster-whisper-small\    ← 英文 / 多语 ASR（可选）
└─ nltk_data\                    ← cmudict / 词性数据（英文朗读需要）
```

#### 3.1 自动下载（推荐）

启动程序，在首启向导里选语言 → 显卡自检 → **选择 ASR 模型** → 点「下载缺失模型」。
下载清单与目标路径定义在 [`app/core/models.py`](app/core/models.py)，界面会显示进度。

#### 3.2 手动下载

| 模型 | 用途 | 大小 | 来源 | 放到 |
|---|---|---|---|---|
| `s1v3.ckpt` | GPT / T2S 主干 | 148 MB | [HF lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) | `GPT-SoVITS/GPT_SoVITS/pretrained_models/` |
| `s2Gv4.pth` | v4 SoVITS | 733 MB | 同上仓库 `gsv-v4-pretrained/` | `…/pretrained_models/gsv-v4-pretrained/` |
| `vocoder.pth` | 48 kHz 声码器 | 55 MB | 同上仓库 | 同上目录 |
| `chinese-roberta-wwm-ext-large` | 文本前端 | 621 MB | [HF hfl](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | `…/pretrained_models/chinese-roberta-wwm-ext-large/` |
| `chinese-hubert-base` | 语义特征 | 181 MB | [HF TencentGameMate](https://huggingface.co/TencentGameMate/chinese-hubert-base) | `…/pretrained_models/chinese-hubert-base/` |
| `paraformer-zh` + `ct-punc` + `campplus` | **中文 ASR（默认）** | ≈1.2 GB | [ModelScope iic](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) | 交给 FunASR 落到 `models_cache/modelscope/models/` |
| `faster-whisper-small` | **英文 / 多语 ASR（可选）** | 484 MB | [HF Systran](https://huggingface.co/Systran/faster-whisper-small) | `models_cache/whisper/faster-whisper-small/` |

### 4. 运行

```bash
python app/main.py
```

首启向导依次询问：语言 → 显卡 → 模型 → LLM 接口。任意 OpenAI 兼容接口都可以，
例如本地 llama.cpp：

```yaml
llm:
  base_url: "http://127.0.0.1:8082/v1"
  model: "Qwen3.5-9B"
  api_key: ""        # 本地服务通常留空
```

### 5. 验证

```bash
python tools/mic_check.py            # 麦克风与播放设备
python tools/check_character.py 默认音色   # 角色文件与权重是否齐全
```

日志：`logs/app.log`。

---

## 中英文 ASR 怎么选

程序内置两条 ASR 路线，**选一条即可**：中文用户选 paraformer，英文或多语用户选 whisper。

| | 中文 paraformer-zh | English whisper-small |
|---|---|---|
| 优势 | 中文识别最准、标点自然、带说话人校验 | 英文 / 日文 / 韩文可用，支持混说 |
| 体积 | ≈1.2 GB | ≈484 MB |
| 显存 | ≈0.8 GB | ≈1 GB |
| 切换方式 | 向导里选「中文 paraformer」 | 选「English / 多语 whisper」 |

切换后在 `config.yaml` 里体现为：

```yaml
asr:
  model: "paraformer-zh"      # 或 "whisper"
  whisper_size: "small"
```

两条都装也没问题，程序只加载被选中的那个。如果一句话里中英混说，选 whisper。

---

## 功能概览

- **按住说话**：默认 `shift+a`，可改成任意组合键（也支持单按 Shift）
- **全本地链路**：ASR → LLM → GPT-SoVITS v4，音频不上传
- **中 / 英 / 日 / 韩 / 粤** 朗读，按句子自动判定语言
- **角色训练**：45 秒–5 分钟素材 → 质检 → 切分 → LoRA → 三句试听
- **音色与节奏面板**：语速、句间停顿、音色温度、top_k、采样步数、重复惩罚，每项都有说明
- **可选联网检索**：Bing / 百度 / 维基百科，检索前先说「稍等，让我查一下」
- **中英双语界面**：首次启动选择，主界面或托盘随时切换

---

## 常见问题

**装完没声音？**
先看 `logs/app.log` 有没有 `TTS 失败`；确认角色目录 `checkpoints/` 下有两个权重文件。

**训练报显存不足？**
训练前程序会卸载语音模型；仍失败就换更干净、更长的素材，或用「保守」档。

**提示已在运行？**
单实例程序，检查右下角托盘。

**能从别的机器拷贝角色吗？**
可以，把整个角色文件夹（`profile.json`、`reference.wav/.txt`、`checkpoints/`、`dataset/`）
一起拷进 `characters/`，索引会自动补齐。权重文件名不要改。

**声音拖尾 / 单字拖音？**
角色页把「重复惩罚」调到 1.9–2.1；仍不行就把「语速」降到 0.92。

---

## 关于声音授权（再次强调）

- 训练前请确认：这是**你的声音**，或你持有**书面授权**
- 授权应写明用途、范围、期限，并允许对方随时撤回
- 不要把合成语音用于冒充、欺诈、骚扰、色情或误导性内容
- 如果用于产品对外服务，请自行完成当地法律要求的告知与合规

---

## 许可

本项目代码采用 **MIT** 许可，详见 [LICENSE](LICENSE)。

依赖与权重遵循各自许可：GPT-SoVITS（MIT）、FunASR（MIT）、faster-whisper（MIT）、
CTranslate2（MIT）、PySide6（LGPLv3，动态链接）、ModelScope（Apache-2.0），
以及 chinese-roberta / chinese-hubert / paraformer / whisper 权重发布方的条款。

---

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) —— 音色克隆与合成
- [FunASR](https://github.com/modelscope/FunASR) —— 中文语音识别
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) —— 多语识别
- [PySide6](https://www.qt.io/qt-for-python) —— 界面
- [HuggingFace](https://huggingface.co) / [ModelScope](https://modelscope.cn) —— 模型托管

---

## 最后

这个项目最初只是想在夜里有人说句话。
如果它能陪一位独居的老人、帮一个不方便打字的人、让加班回家的人不那么安静，
那它就比「一个语音合成 Demo」更有意义。

**愿世界和平，让陪伴无处不在。**
