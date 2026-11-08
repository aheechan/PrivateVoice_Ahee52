# PrivateVoice_Ahee52

**中文** | [English](#english)

> 一个完全本地运行的中文情感陪伴器。
> A fully local emotional-companion app for Chinese-speaking users.

---

## ⚠️ 使用前必读 / Read this first

**这是一个情感陪伴器，不是医疗器械，也不是紧急求助工具。**
它不能替代家人、朋友或专业心理咨询；如果你或身边的人正处于危机中，请联系当地紧急服务或专业机构。

**用他人的声音训练音色，必须先取得对方明确的、可撤回的授权。**
本程序内置的语音克隆能力仅用于你**本人**的声音，或已获得书面同意的声音。以下行为明确禁止：

- 冒充他人（亲友、公众人物、客服、官方机构）
- 制作欺诈、骚扰、色情或误导性内容
- 在对方不知情的情况下克隆其声音

本项目不对使用者的合规性负责；由此产生的法律责任由使用者承担。

> **This is a companionship app — not a medical device, not an emergency service.**
> It does not replace family, friends or professional care. In a crisis, contact local emergency services.
>
> **Cloning someone's voice requires their explicit, revocable consent.**
> Use it only for your own voice, or for a voice you have written permission to use.
> Impersonation, fraud, harassment, sexual or misleading content are strictly prohibited.

---

## 为什么做这个 / Why this exists

最初只是为了让自己在独处的夜里，有个能说说话的声音。

做完之后发现，它可能对更多人有用：

- **独居老人**：有人应答、能提醒吃药、能听他们讲过去的事
- **长期卧床 / 行动不便者**：不用起身、不用打字，说一句就有回应
- **异地独居的年轻人**：加班回家后，至少屋里有个声音
- **语言学习者 / 视障用户**：纯语音交互，不依赖屏幕

如果你愿意把它改造到真实场景里——养老院、社区关怀、无障碍辅助——**非常欢迎**。
这是这个项目最想被用到的方向。

> It started as a voice to keep me company at night.
> Then it seemed useful for more people: elderly living alone, people with limited mobility,
> anyone far from home, or users who simply can't use a screen.
> If you can adapt it for elder care, community support or accessibility — please do.

---

## 本仓库包含什么 / What's in this repo

| 包含 / Included | 不包含 / Not included |
|---|---|
| 完整应用代码（`app/`） | 模型权重（约 2.6 GB，见下方下载表） |
| 中英双语界面 | 大语言模型（需自备 OpenAI 兼容接口） |
| 训练 / 质检 / 切分工具 | 角色数据与训练产物 |
| Inno Setup 打包脚本 | 上游 GPT-SoVITS 源码（需自行 clone） |

> 想直接安装使用，不用看后面的复现步骤：去 **Releases** 下载整合包即可，
> 内含默认音色与全部模型，安装后即可对话。

---

## 硬件要求 / Hardware requirements

| 项目 | 最低 / Minimum | 推荐 / Recommended |
|---|---|---|
| 操作系统 | Windows 10 x64 (21H2+) | Windows 11 x64 |
| 显卡 | NVIDIA RTX 2060 / 6 GB 显存 | **RTX 3090 / 4070 Ti / 5070 Ti 及以上，16–24 GB** |
| 驱动 | 支持 CUDA 12.8（≥ 570.x） | 最新 Game Ready / Studio 驱动 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 15 GB 空闲 | 30 GB（含多角色与训练中间产物） |
| 声卡 | 任意可录音设备 | 带监听回路的 USB 声卡 / 耳机 |
| 其他 | 无需联网即可对话 | 联网仅用于首次下模型、联网检索 |

说明 / Notes：

- **推理**约需 6–8 GB 显存；**训练**在 24 GB 上最稳，16 GB 会用「均衡」档并自动降 batch。
- 纯 CPU 推理不支持（延迟约 30–60 秒/句，且 v4 依赖 GPU 算子）。
- 显存不足时训练会提示换素材或降档，不会静默失败。

---

## 从源码复现 / Reproduce from source

### 0. 前置

- Windows 10/11 x64
- Python **3.12**（打包版用的是 3.12 嵌入式运行时）
- Git、NVIDIA 驱动
- 约 20 GB 磁盘

### 1. 取代码与上游

```bash
git clone <本仓库地址> PrivateVoice
cd PrivateVoice

# 上游推理框架不随本仓库分发
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
```

> `Aqua-TTS` 只在遗留的 v3 引擎（`app/core/tts.py` 的 `AquaTTSEngine`）里用到；
> 默认的 v4 管线不依赖它，可不装。

### 2. 装依赖

```bash
python -m venv .venv
.venv\Scripts\activate

# torch 需按你的 CUDA 版本安装，打包版用的是 cu128
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

### 3. 下载模型（关键步骤）

所有路径都是**相对项目根目录**的，程序按这些位置加载：

```
PrivateVoice\
├─ GPT-SoVITS\
│  └─ GPT_SoVITS\pretrained_models\
│     ├─ s1v3.ckpt                                   ← GPT 主干 (148 MB)
│     ├─ gsv-v4-pretrained\s2Gv4.pth                 ← v4 SoVITS 底模 (733 MB)
│     ├─ gsv-v4-pretrained\vocoder.pth               ← 48 kHz 声码器 (55 MB)
│     ├─ chinese-roberta-wwm-ext-large\model.safetensors  ← 文本前端 (621 MB)
│     └─ chinese-hubert-base\pytorch_model.bin       ← 语义特征 (181 MB)
├─ models_cache\
│  ├─ modelscope\models\                             ← FunASR 自动下载（中文 ASR）
│  │  ├─ iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\
│  │  ├─ iic--punc_ct-transformer_zh-cn-common-vocab272727-pytorch\   ← ct-punc 标点
│  │  └─ iic--speech_campplus_sv_zh-cn_16k-common\                    ← 说话人向量
│  └─ whisper\faster-whisper-small\                   ← 英文 ASR（可选）
└─ nltk_data\                                        ← cmudict / 词性数据（英文朗读需要）
```

#### 3.1 自动下载（推荐）

启动程序，首次向导里选语言 → 显卡自检 → **选择 ASR 模型** → 点「下载缺失模型」。
下载清单与目标路径定义在 [`app/core/models.py`](app/core/models.py)，会显示进度。

#### 3.2 手动下载

| 模型 | 用途 | 大小 | 来源 | 放到 |
|---|---|---|---|---|
| `s1v3.ckpt` | GPT/T2S 主干 | 148 MB | [HF lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) | `GPT-SoVITS/GPT_SoVITS/pretrained_models/` |
| `s2Gv4.pth` | v4 SoVITS | 733 MB | 同上 `gsv-v4-pretrained/` | `…/pretrained_models/gsv-v4-pretrained/` |
| `vocoder.pth` | 48 kHz 声码器 | 55 MB | 同上 | 同上 |
| `chinese-roberta-wwm-ext-large` | 文本前端 | 621 MB | [HF hfl](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | `…/pretrained_models/chinese-roberta-wwm-ext-large/` |
| `chinese-hubert-base` | 语义特征 | 181 MB | [HF TencentGameMate](https://huggingface.co/TencentGameMate/chinese-hubert-base) | `…/pretrained_models/chinese-hubert-base/` |
| `paraformer-zh` + `ct-punc` + `campplus` | **中文 ASR**（默认） | ≈ 1.2 GB | [ModelScope iic](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) | 交给 FunASR 自动落盘到 `models_cache/modelscope/models/` |
| `faster-whisper-small` | **英文 / 多语 ASR**（可选） | 484 MB | [HF Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | `models_cache/whisper/faster-whisper-small/` |

#### 3.3 中英文 ASR 怎么选

程序内置两条 ASR 路线，**按需选一条即可**（中文用户选 paraformer，英文或多语用户选 whisper）：

| | 中文 paraformer-zh | English whisper-small |
|---|---|---|
| 优势 | 中文识别最准、标点自然、带说话人校验 | 英文/日文/韩文可用，多语种 |
| 体积 | ≈ 1.2 GB | ≈ 484 MB |
| 显存 | 约 0.8 GB | 约 1 GB |
| 切换方式 | 首次向导里选「中文 paraformer」 | 选「English / 多语 whisper」 |

切换后在 `config.yaml` 里体现为：

```yaml
asr:
  model: "paraformer-zh"      # 或 "whisper"
  whisper_size: "small"       # whisper 路线使用
```

> 两条路线都装也没问题，程序只加载被选中的那个。
> 如果你希望**一句话里中英混说**也能识别，选 whisper。

### 4. 运行

```bash
python app/main.py
```

首次启动会打开向导：语言 → 显卡自检 → 模型检查 → LLM 配置。
LLM 填任意 OpenAI 兼容接口，例如本地 llama.cpp：

```yaml
llm:
  base_url: "http://127.0.0.1:8082/v1"
  model: "Qwen3.5-9B"
  api_key: ""        # 本地服务通常留空
```

### 5. 验证是否装好

```bash
# 显卡与依赖
python tools/mic_check.py          # 麦克风与播放设备

# 角色目录结构、权重是否齐全
python tools/check_character.py 默认音色

# 只跑 TTS，检查能否出声
python -c "from app.config import Config; from app.core.tts import make_engine; \
e=make_engine(Config.load()); print('TTS ok')"
```

日志在 `logs/app.log`；出问题先看这里。

---

## 功能概览 / Features

- **按住说话**：默认 `shift+a`，可改成任意组合键（也支持单按 Shift）
- **本地 ASR → 本地 LLM → GPT-SoVITS v4**，全链路不上传音频
- **中英日韩粤** 混合朗读，按句子自动判定语言
- **角色训练**：上传 45 秒–5 分钟素材，自动质检 → 切分 → LoRA 训练 → 三句试听
- **角色参数页**：语速、句间停顿、音色温度、top_k、采样步数、重复惩罚，均带说明
- **可选联网检索**：Bing / 百度 / 维基百科，检索前先说「稍等，让我查一下」
- **中英双语界面**：首次启动选择，主界面 / 托盘随时切换

---

## 常见问题 / FAQ

**装完没声音？**
检查 `logs/app.log` 有没有 `TTS 失败`；先确认 `characters/` 下角色的 `checkpoints/` 里有两个权重文件。

**训练报显存不足？**
训练前程序会卸载语音模型；仍失败就换更干净、更长的素材，或用「保守」档。

**启动提示已在运行？**
单实例程序，看右下角托盘。

**能从别的机器拷贝角色吗？**
可以，整个角色文件夹（`profile.json` + `reference.wav/.txt` + `checkpoints/` + `dataset/`）一起拷进 `characters/`，
程序会自动补索引；权重文件名不要改。

**声音拖尾 / 单字拖音？**
角色页把「重复惩罚」调到 1.9–2.1；再不行把「语速」降到 0.92。

---

## 关于声音授权（再次强调）/ On voice consent (again)

- 训练前请确认：这是**你的声音**，或你有**书面授权**使用它
- 授权应包含：用途、范围、期限，并允许对方随时撤回
- 不要把合成语音用于冒充、诈骗、骚扰、色情或误导性内容
- 如果用于产品/对外服务，请自行完成当地法律要求的告知与合规

> Only clone voices you own or have written permission to use.
> Never use synthesized speech for impersonation, fraud, harassment or sexual content.

---

## 许可 / License

本项目代码采用 **MIT** 许可，详见 [LICENSE](LICENSE)。

依赖与模型遵循各自许可：GPT-SoVITS (MIT)、FunASR (MIT)、faster-whisper (MIT)、
CTranslate2 (MIT)、PySide6 (LGPLv3，动态链接)、ModelScope (Apache-2.0)、
chinese-roberta / chinese-hubert / paraformer / whisper 权重按其发布方许可使用。

---

## 致谢 / Credits

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 语音克隆与合成
- [FunASR](https://github.com/modelscope/FunASR) — 中文语音识别
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 多语识别
- [PySide6](https://www.qt.io/qt-for-python) — 界面
- [HuggingFace](https://huggingface.co) / [ModelScope](https://modelscope.cn) — 模型托管

---

## 最后 / One last thing

这个项目最初只是想在夜里有人说句话。
如果它能被用来陪一位独居的老人、帮一个不方便打字的人、或者让某个加班回家的人不那么安静，
那它就比「一个语音合成 Demo」更有意义。

**愿世界和平，让陪伴无处不在。**

> It began as a voice in the night.
> If it can keep an elderly person company, help someone who can't type, or make a quiet
> apartment feel less empty — it has done more than a TTS demo ever could.
>
> **May there be peace, and may companionship be everywhere.**

---

## English

### ⚠️ Before you start

- **This is a companionship app**, not a medical device and not an emergency service.
- **Voice cloning requires explicit, revocable consent** from the voice owner.
  Impersonation, fraud, harassment, sexual or misleading content are prohibited.

### Why it exists

Built as a voice to keep the author company at night. It may also help people living alone,
users with limited mobility, or anyone who cannot use a screen.
Adapting it for elder care, community support or accessibility is warmly encouraged.

### What's in this repo

Application code, bilingual UI, training/QC tooling and the installer script.
**No model weights** (≈2.6 GB) and no LLM. Upstream GPT-SoVITS is not vendored.

For end users: grab the packaged build from **Releases** — it ships with models and a default voice.

### Hardware

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 x64 | Windows 11 x64 |
| GPU | RTX 2060 / 6 GB VRAM | RTX 3090 / 4070 Ti / 5070 Ti, 16–24 GB |
| Driver | CUDA 12.8 capable (≥570.x) | latest |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB free | 30 GB |

Inference uses ~6–8 GB VRAM; training is comfortable on 24 GB and auto-downshifts on 16 GB.
CPU-only inference is not supported.

### Reproduce from source

```bash
git clone <this repo> PrivateVoice
cd PrivateVoice
git clone https://github.com/RVC-Boss/GPT-SoVITS.git

python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python app/main.py
```

On first run, pick a language, run the GPU check, **choose an ASR model**
(Chinese paraformer *or* English whisper), then click **“Download missing models”**.

#### Model locations (relative to project root)

```
GPT-SoVITS/GPT_SoVITS/pretrained_models/
├── s1v3.ckpt                                  148 MB
├── gsv-v4-pretrained/s2Gv4.pth                733 MB
├── gsv-v4-pretrained/vocoder.pth               55 MB
├── chinese-roberta-wwm-ext-large/model.safetensors   621 MB
└── chinese-hubert-base/pytorch_model.bin      181 MB

models_cache/
├── modelscope/models/       ← FunASR auto-download (Chinese ASR: paraformer + ct-punc + campplus)
└── whisper/faster-whisper-small/   ← English/multilingual ASR (optional)
```

| Model | Purpose | Size | Source |
|---|---|---|---|
| `s1v3.ckpt` / `s2Gv4.pth` / `vocoder.pth` | GPT + v4 SoVITS + 48 kHz vocoder | 936 MB | [HF lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) |
| `chinese-roberta-wwm-ext-large` | text front-end | 621 MB | [HF hfl](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) |
| `chinese-hubert-base` | semantic features | 181 MB | [HF TencentGameMate](https://huggingface.co/TencentGameMate/chinese-hubert-base) |
| `paraformer-zh` (+punc+sv) | **Chinese ASR (default)** | ≈1.2 GB | [ModelScope iic](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |
| `faster-whisper-small` | **English / multilingual ASR** | 484 MB | [HF Systran](https://huggingface.co/Systran/faster-whisper-small) |

#### Chinese vs English ASR

Pick one in the wizard (or set `asr.model` in `config.yaml`):

- `paraformer-zh` — most accurate for Chinese, natural punctuation, speaker check
- `whisper` (+ `whisper_size: small`) — English/Japanese/Korean, handles mixed-language speech

Both can be installed; only the selected one is loaded.

### Run & verify

```bash
python app/main.py
python tools/mic_check.py
python tools/check_character.py 默认音色
```

Logs: `logs/app.log`.

### Features

Push-to-talk hotkey · fully local pipeline · zh/en/ja/ko/yue speech · character training with
QC and self-test · per-character voice/pace parameters · optional web lookup · bilingual UI.

### License

MIT for this project's code (see [LICENSE](LICENSE)); dependencies and model weights keep their
own licenses (GPT-SoVITS MIT, FunASR MIT, faster-whisper MIT, PySide6 LGPLv3, ModelScope Apache-2.0).

### Closing

Built to make a quiet room feel less quiet. If it can keep someone company, it has done its job.

**May there be peace, and may companionship be everywhere.**
