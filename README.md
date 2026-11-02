# VoiceLab — 本地低延迟角色语音（Stage 1）

在 Windows + RTX 3090 上跑通 **GPT-SoVITS v3 + Aqua-TTS 0.3.0** 的流式推理，并测出真实首包延迟。
本阶段只做运行时基线：**不含 UI、不含训练、不接 LLM**。

## 实测结论（先看这个）

| 文本档 | 字数 | TTFP p50 | TTFP p95 | 总耗时 p50 | 音频时长 | RTF |
|---|---|---|---|---|---|---|
| short | 5 | **355 ms** | 716 ms | 1283 ms | 1.30 s | 0.90 |
| medium | 20 | **394 ms** | 500 ms | 3765 ms | 4.59 s | 0.82 |
| long | 50 | **414 ms** | 818 ms | 8420 ms | 10.20 s | 0.83 |

- 峰值显存 **2.17–2.19 GB**（allocated）/ 2.32 GB（reserved）
- 模型加载 7.2 s，CUDA Graph 预捕获 6 个 bucket（128/256/448/512/768/1024）
- 逐题回答见 [`conclusion.md`](conclusion.md)，原始数据见 [`results/spike.json`](results/spike.json)

> **关键限制**：本机 **没能编出 BigVGAN 的自定义 CUDA 核**（nvcc 12.8 与 MSVC 14.51 不兼容，
> `cudafe++` 报 `0xC0000005`），Aqua 自动回退到 PyTorch 实现。所以"默认"和"强制回退"
> 两组实际跑的是同一条路径，表里数字是**回退路径**的延迟。详见下文「已知问题」。

## 版本锁定

| 组件 | 版本 / commit |
|---|---|
| GPT-SoVITS | `08d627c3338173c3229286d8787060d6559fe0f8`（2026-04-30，含 cuda graph 支持） |
| Aqua-TTS | `eb8e0d59a9ca317b40d8182b04c12ba01a489f8d`（tag `v0.3.0`） |
| PyTorch | `2.11.0+cu128`（torchaudio 同版本） |
| CUDA runtime | 12.8（驱动 610.62，cuDNN 9.19） |
| Python | 3.12.10（venv at `C:\VoiceLab\.venv`） |

`requirements.txt` 为全量锁版本；`logs/pip_freeze.txt` 为安装后的原始快照。

## 目录结构

```
C:\VoiceLab\
├── .venv\                      uv 虚拟环境（Python 3.12.10）
├── GPT-SoVITS\                 上游仓库，pin 08d627c
│   └── GPT_SoVITS\pretrained_models\
│       ├── s1v3.ckpt           GPT / T2S 权重
│       ├── s2Gv3.pth           SoVITS v3 权重
│       ├── chinese-hubert-base\
│       ├── chinese-roberta-wwm-ext-large\
│       └── models--nvidia--bigvgan_v2_24khz_100band_256x\
├── Aqua-TTS\                   pin v0.3.0（editable 安装）
├── assets\
│   ├── ref.wav                 9.12s 参考音频（48k / 单声道 / 16bit）
│   ├── ref.txt                 参考音频逐字文稿
│   └── prompts\                ← Stage 2/3 用，本阶段按指示跳过
├── benchmark.py                TTFP 基准（可复现）
├── requirements.txt
├── results\
│   ├── spike.json              两组 × 三档原始测量
│   ├── spike_default.json
│   ├── spike_nokernel.json
│   └── sample.wav              16.1s 试听音频（三档串接）
├── tools\                      下载/构建/冒烟脚本
├── logs\
│   ├── setup.log               全流程命令与报错（由各步日志合并）
│   ├── weights.sha256          每个权重的 URL / 大小 / SHA256
│   └── *.log                   分步原始日志
└── conclusion.md
```

## 复现步骤

```powershell
# 1) 环境
uv venv C:\VoiceLab\.venv --python 3.12
uv pip install --python C:\VoiceLab\.venv\Scripts\python.exe `
  torch==2.11.0+cu128 torchaudio==2.11.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128

# 2) 上游仓库（git 走本地代理，见 logs/setup.log）
git clone https://github.com/RVC-Boss/GPT-SoVITS.git C:\VoiceLab\GPT-SoVITS
git -C C:\VoiceLab\GPT-SoVITS checkout 08d627c3338173c3229286d8787060d6559fe0f8
git clone https://github.com/Lucas1479/Aqua-TTS.git C:\VoiceLab\Aqua-TTS
git -C C:\VoiceLab\Aqua-TTS checkout v0.3.0

# 3) 依赖与包
uv pip install --python C:\VoiceLab\.venv\Scripts\python.exe -r requirements-gptsovits.txt
uv pip install --python C:\VoiceLab\.venv\Scripts\python.exe -e "C:\VoiceLab\Aqua-TTS[runtime]"
uv pip install --python C:\VoiceLab\.venv\Scripts\python.exe ninja   # 编译 BigVGAN 核必需

# 4) 权重（hf-mirror / ModelScope，含 SHA256 校验）
python tools\download_weights.py

# 5) 跑基准（两组）
$env:GPT_SOVITS_HOME = "C:\VoiceLab\GPT-SoVITS"
python benchmark.py --group default   --runs 20 --warmup 5
python benchmark.py --group nokernel  --runs 20 --warmup 5
python benchmark.py --merge
```

测量口径对齐上游 `benchmarks/aqua_ttfp.py`：`top_k=5, top_p=1, temperature=0.6,
speed=1.1, sample_steps=4, how_to_cut=按标点符号切, chunk_size_seconds=0.25`，
CUDA Graph 与 static KV 开启；每档 5 次预热 + 20 次测量，TTFP 取**第一个非空音频块**。

## 已知问题

### 1. BigVGAN 自定义 CUDA 核编译失败（未解决，已量化影响）

```
nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
```

- 原因：本机唯一可用 nvcc 是 CUDA **12.8**，而 MSVC 是 **14.51**（VS 2026）；CUDA 12.8 不支持该编译器版本。
  机器上的 CUDA 12.6 是残缺安装（只有 `include/`、`lib/`，没有 `bin/nvcc.exe`），无法替换。
- 后果：`tts.bigvgan_kernel_mode == "pytorch"`，A/B 两组等价，`--group default` 仅多花约 19 s 在失败编译上。
- 想要真 CUDA 核：需装 **VS 2022 Build Tools（v143 / MSVC 14.3x）**。当前会话是非管理员，
  VS Installer 需要提权，故未执行。装好后把 `MSVC_VCVARS64_PATH` 指向其 `vcvars64.bat` 即可复用。

### 2. `vcvars64.bat` 路径未加引号（已绕过）

Aqua 用 `GetShortPathNameW` 取 8.3 短名；E: 盘未启用 8.3，回退成带空格的裸路径，
`cmd /c call E:\Program Files\...` 直接失败。绕过方式（不改模型代码）：

```powershell
cmd /c mklink /J C:\VoiceLab\tools\vs "E:\Program Files\Microsoft Visual Studio\18\Community"
$env:MSVC_VCVARS64_PATH = "C:\VoiceLab\tools\vs\VC\Auxiliary\Build\vcvars64.bat"
```

### 3. G2PW 词典缺失

上游把 `GPT_SoVITS/text/G2PWModel` 列入 gitignore，中文前端需要它。
已从 ModelScope `XXXXRT/GPT-SoVITS-Pretrained` 的 `G2PWModel.zip` 补齐（588 MB，SHA256 见
`logs/weights.sha256`）。**该路径按进程 CWD 解析**，所以 `benchmark.py` 会先 `chdir` 到 GPT-SoVITS 根目录。

## 素材说明

- `assets/ref.wav` 取自桌面 `Grace_voice\grace voice nanpu plus orig 0327.wav`（48.07 s 原始录音），
  按既有打标 `asr_opt\slicer_opt.list` 的偏移（32 kHz 采样点 186880→478720）在 **48k 原始音轨**上无损回切，
  得到 5.84–14.96 s 的连读段，共 **9.12 s**，转单声道 48k/16bit。
- `assets/ref.txt` 直接使用打标原文拼接，未二次 ASR，保证逐字一致。
- 原录音总长 48 s，落在 45–60 s 区间，可直接用于 Stage 2 微调。

## 角色提示词层（Stage 3，本阶段未落盘）

按本轮指示，`assets/prompts/companion_system.txt` 与 `persona.example.yaml` **本阶段跳过**
（附件 A 原文未提供，不自行杜撰），仅在此记录对接时必须遵守的约束：

- 每次回复 **1–2 句**，**不超过 40 字**，不说教，不主动提及自己是 AI
- 调用时**关闭 thinking 模式**，`max_tokens` 设 **120–200**
- 推理服务已由 `D:\llama.cpp\启动_llama_v2.bat`（选项 3）提供，**Stage 1 未接任何 LLM 接口**

## 不在本阶段范围

托盘/UI、角色系统、LLM 接入、打包、模型训练、v2 降级。

---

# Stage 2：训练链路 + 角色系统 + 托盘骨架

Stage 2 在 Stage 1 的运行时之上，做通了「上传 45–90 秒录音 → 质检 → 两遍切分 → 四层打标 →
本地训练 v3 音色 → 角色落库 → 托盘里试听」这条闭环。**不含 LLM、不含正式对话、不含打包。**

详细数据、偏离与已知问题见 [docs/stage2-report.md](docs/stage2-report.md)，
训练 CLI 的源码考证见 [docs/research/training-cli.md](docs/research/training-cli.md)。

## 运行

`powershell
C:\VoiceLab\.venv\Scripts\python.exe app\main.py
`

启动后常驻系统托盘：左键点图标打开角色管理，右键有完整菜单。
单实例；关掉窗口只会最小化到托盘；开机自启在设置页开关（写 HKCU 的 Run 项）。

## 目录

`
app/
  main.py            入口：单实例、托盘、日志、硬件门禁（RTX30/40/50 且显存≥12GB）
  tray.py            托盘 5 态图标 + 菜单
  config.py          config.yaml 加载 + 档位派生（PROFILE_MATRIX，无散落魔法数字）
  core/
    audio_io.py      设备枚举、录音(5s 底噪采样)、播放、电平
    audio_qc.py      解码→48k→-23LUFS；时长/采样率/削波/SNR/静音/多人声 六项检查
    segment.py       第一遍 Silero VAD，第二遍 ASR 词边界吸附；产出 clips/dataset.list/metadata.json
    label.py         FunASR Paraformer-zh + CT-Punc；数字/日期/百分比归一化；黄红 flag
    train.py         子进程训练、JSON 进度、OOM 重试、取消、挑参考音频、跑 benchmark
    characters.py    角色库 CRUD、状态机、.vcp 导入导出
    tts.py           TTSEngine 抽象 + AquaTTS 适配器（只服务试听）
  ui/
    wizard.py        6 步向导：基本信息/上传录制/质检/切分打标/训练/试听
    char_manager.py  角色管理（导出导入 .vcp、删除、设为当前）
    settings.py      档位、音频设备、热键（仅保存）、日志级别、开机自启
    diagnostics.py   GPU/驱动/torch/CUDA/commit/最近 TTFP + 导出诊断包
characters/          角色数据根（index.json + <id>/profile.json + checkpoints + reference）
datasets/            临时数据（切分产物、qc.json、work/）
config.yaml          运行配置（首次启动自动生成）
`

## 档位

| 参数 | 保守(默认) | 标准 | 激进 |
|---|---|---|---|
| epochs | 8 | 10 | 12 |
| lr_scale | 0.6 | 1.0 | 1.0 |
| batch 24G / 16G / 12G | 4 / 2 / 1 | 6 / 3 / 1 | 8 / 4 / 2 |
| grad_accum | 4 | 2 | 1 |
| max_clip_sec | 6 | 7 | 8 |

片段上限还会按显存夹一次（12G→5s、16G→6s、24G→8s）。

## 示例角色

characters/格雷丝-748caf/：用桌面 Grace_voice 的 48.07 秒真实人声，
conservative 档训练 8 epoch（SoVITS 170s + GPT 90s），
产出 lab_格雷丝_e8_s200_l32.pth 与 lab_格雷丝-e8.ckpt。
试听三句平均音色相似度 0.664，TTFP p50 313–332 ms（见 
esults/character_ttfp.json）。

## Stage 2 已知问题（摘要）

1. **GPT(s1) 训练必须走 	ools/s1shim/ 的运行时垫片**：上游硬编码 DDP+gloo，
   Windows 单卡直接 0xC0000005；DataLoader 子进程同样崩。垫片不改上游源码。
2. **onnxruntime-gpu 是 CUDA 13 版**，与 torch 的 CUDA 12.8 不匹配，日志会刷
   Failed to load cublasLt64_13.dll，自动回退 CPU，不影响结果。
3. **PySide6 不含 WebEngine**（显式卸载，规格要求）。
4. **热键只保存不注册**，对话留到 Stage 3。

---

# Stage 4：打包

分发包在 dist\情感陪伴器_v1\（11.51 GB / 49,986 文件），安装器在
dist\installer\（受 Windows 单 exe 4.2GB 限制，分卷为 Setup.exe + .bin）。
构建脚本是仓库根目录的 installer.iss（Inno Setup 6，Compression=none + 分卷）。

## 分发包结构

`
情感陪伴器_v1\
  runtime\        嵌入式 Python 3.12.10 + site-packages（torch/PySide6/funasr/...）
  app\            应用代码
  GPT-SoVITS\     v3 代码 + pretrained_models + G2PWModel
  Aqua-TTS\       aquatts 源码（不依赖 editable 安装）
  characters\     示例角色（格雷丝、阿岚，路径已相对化）
  assets\ logs\ results\
  启动.bat  自检.bat  使用说明.txt
`

## 打包要点（详见 docs/stage4-report.md）

1. python312._pth **必须含 import site**，否则 setuptools 的 distutils 补丁不生效，
   funasr 会静默少注册 11 个模型（paraformer-zh 直接不可用）。
2. 删掉 __editable__.aqua_tts-0.3.0.pth 与同名前缀 finder（写死了 C:\VoiceLab 绝对路径），
   改由 _pth 加 ..\Aqua-TTS。
3. 不要把 GPT-SoVITS / GPT_SoVITS 加进 _pth（里面的 config.py、module\、	ext\
   会污染 sys.path），TTS 引擎加载时按需插入即可。
4. 角色的 gpt_ckpt / sovits_ckpt 走 CharacterStore._relocate()：路径不存在时回落到
   <角色目录>\checkpoints\<文件名>，这样整包搬到 5070 Ti 也能用。

## 验证

`powershell
cd dist\情感陪伴器_v1
runtime\python.exe tools\selftest.py     # 期望最后一行 SELFTEST PASS
`

3090 上实测：运行时/CUDA、门禁、依赖、角色权重、ASR→LLM→TTS→播放（e2e 2421ms，
音频 4.75s）、打断 11ms 全部通过。
GPU 拒卡逻辑有单测覆盖（	ests/test_gpu_guard.py，含 5070 Ti sm_120 放行）。

> 5070 Ti 未实机验证；包内已带 ModelScope 缓存（ASR/标点/说话人），换机首次运行不需要联网。
