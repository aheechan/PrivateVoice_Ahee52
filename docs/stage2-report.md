# Stage 2 报告：训练链路 + 角色系统 + 托盘骨架

> 完成日期：2026-11-02
> 环境：RTX 3090 24GB（sm_86）· 驱动 610.62 · torch 2.11.0+cu128 · Python 3.12.10
> 基线对照：`docs/stage1-baseline.md`

---

## 1. 完成项

| # | 规格条目 | 状态 | 位置 |
|---|---|---|---|
| 1 | 工程结构（app/ 全模块） | ✅ 完成 | `app/{main,tray,config}.py`、`app/core/*`、`app/ui/*` |
| 2 | config.yaml + 档位派生，无魔法数字 | ✅ 完成 | `app/config.py:PROFILE_MATRIX` / `config.yaml` |
| 3 | 音频导入与质量契约（6 项检查 + 中文建议） | ✅ 完成 | `app/core/audio_qc.py` |
| 4 | 录音：5s 底噪采样 + 建议增益 + 三色电平条 | ✅ 完成 | `app/core/audio_io.py`、`ui/wizard.py:AudioPage` |
| 5 | 两遍切分（Silero + 词边界吸附） | ✅ 完成 | `app/core/segment.py` |
| 6 | 打标四层（Paraformer+CTPunc / 归一化 / flag / 复核表） | ✅ 完成 | `app/core/label.py`、`ui/wizard.py:SegmentPage` |
| 7 | 训练子进程封装（预检/OOM 重试/取消/JSON 进度/选参考） | ✅ 完成 | `app/core/train.py` |
| 8 | 角色库 CRUD + 状态机 + .vcp 导入导出 | ✅ 完成 | `app/core/characters.py` |
| 9 | 托盘（5 态图标、菜单、单实例、最小化、自启） | ✅ 完成 | `app/tray.py`、`app/main.py` |
| 10 | 6 步向导（可回退） | ✅ 完成 | `app/ui/wizard.py` |
| 11 | 设置页 / 诊断页 | ✅ 完成 | `app/ui/settings.py`、`app/ui/diagnostics.py` |
| 12 | TTS 抽象接口 + Aqua 适配器（只服务试听） | ✅ 完成 | `app/core/tts.py` |
| 13 | pytest 单测（不需要 GPU） | ✅ 21 条全过 | `tests/`、`docs/pytest.log` |
| 14 | 示例角色（真实素材训练完成） | ✅ 完成 | `characters/格雷丝-748caf/` |
| 15 | results/character_ttfp.json | ✅ 完成 | `results/character_ttfp.json` |

---

## 2. 实测数据

### 2.1 素材与质检

素材：`C:\Users\72157\OneDrive\Desktop\Grace_voice\grace voice nanpu plus orig 0327.wav`
（48 kHz / 双声道 / 16 bit / **48.07 秒**，按需求方 2026-11-02 的调整，时长门槛由 60s 改为 **45–90s**）

| 检查 | 结果 | 实测值 |
|---|---|---|
| 总时长 | ✅ 通过 | 48.07 s（落在 45–90 s） |
| 采样率 | ✅ 通过 | 48000 Hz |
| 削波 | ✅ 通过 | ≥0.99 的采样点 0.0000% |
| 信噪比 | ✅ 通过 | 60.8 dB |
| 静音占比 | ⚠️ 警告 | 25.2%（>20%） |
| 多人声 | ✅ 通过 | 1 个说话人（campplus，两两余弦 0.59–0.88） |

处理链验证：归一化后 **-23.0 LUFS**、真峰值 **-3.81 dBTP**，符合契约。

### 2.2 切分与打标

| 指标 | 目标 | 实测 |
|---|---|---|
| 片段数 | 12–20（原定 60–90s 素材） | **10**（48s 素材，见 §4.2） |
| 平均段长 | 4–5 s | **4.18 s** |
| 最短 / 最长 | ≥2 s / ≤7 s | 2.40 s / 6.64 s |
| 总时长 | — | 41.8 s |
| 分布警告 | 静音 <20% | 切分后无空片段 |

10 段明细：

| # | 时长 | flag | 文本（归一化后） |
|---|---|---|---|
| 01 | 3.97s | — | 我再过一遍最后的吧，因为你后面可能不是很好垫画面，是吗？ |
| 02 | 6.64s | — | 伊丽莎白斯坦顿是吗？对是什么？那是广，会不会反？这个是连读的是吗？对，那那。 |
| 03 | 2.56s | — | 应该怎么读？玛丽沃斯通克。 |
| 04 | 4.86s | snr | 他创办的什么东西？这个英文我不会念了，快帮我看一下。 |
| 05 | 2.50s | rate_slow | 啊，哦，他创办的。 |
| 06 | 4.36s | snr | 如果你喜欢今天的内容，请分享给你身边需要的朋友，让。 |
| 07 | 4.84s | — | 我们一起把这些美妙的沟通艺术传播开来。 |
| 08 | 2.40s | — | 信用卡米拉了，应该没有什么大问题哦。 |
| 09 | 3.56s | — | 这是另外一个开头了，是吗？要报姓名哦，可以直接开始啦。 |
| 10 | 6.12s | — | 你刚才看是放手上面还是放下来歪哦，来喽，保一条吧，直接开始就可以了啊。 |

flag 只打标不删除，全部 10 段都进入训练（用户在复核表里没有标丢弃）。

### 2.3 训练

计划：**conservative** 档，24GB 桶 → epochs 8 / batch 4 / grad_accum 4 / lr_scale 0.6 / 片段≤6s（素材最长 6.64s，未触发截断）。

| 阶段 | 耗时 | 产物 |
|---|---|---|
| 1A 音素+BERT | 16.4 s | `2-name2text.txt` |
| 1B HuBERT+32k | ~20 s | `4-cnhubert/`、`5-wav32k/` |
| 1C 语义 token | ~25 s | `6-name2semantic.tsv` |
| SoVITS v3 LoRA（8 ep × 25 step） | **170.2 s** | `SoVITS_weights_v3/vlab_格雷丝_e8_s200_l32.pth`（72.05 MB） |
| GPT s1（8 ep × 25 step） | **~90 s** | `GPT_weights_v3/vlab_格雷丝-e8.ckpt`（148.12 MB） |
| **合计** | **4.2 min** | 保留最近 3 个 epoch 的权重 |

- 参考音频自动挑选：`0010.wav`（6.12 s，无 flag，SNR 最高）→ `characters/格雷丝-748caf/reference.wav`
- 没有触发 OOM 重试（可用显存 22.8 GB，估算需求 (5+0.9×4)×1.3 = 11.2 GB）

### 2.4 试听与 TTFP

三句固定文本（不接 LLM），`results/character_preview.json`：

| # | 时长 | 峰值 | LUFS | 与参考音色的余弦相似度 |
|---|---|---|---|---|
| 1 | 2.38 s | -7.8 dBTP | -25.5 | 0.710 |
| 2 | 1.97 s | -5.4 dBTP | -25.7 | 0.696 |
| 3 | 2.17 s | -8.6 dBTP | -25.3 | 0.585 |

平均相似度 **0.664**（campplus，同一人常见区间 0.6–0.8）。三句全部可懂、无爆音（峰值均低于 -5 dBTP）。

TTFP（`results/character_ttfp.json`，Stage 1 同一协议，20 次 + 5 次预热）：

| 档位 | TTFP p50 | p95 | RTF | 峰值显存 | 加载 |
|---|---|---|---|---|---|
| short(5) | **313.35 ms** | 341.14 ms | 0.769 | 2.164 GB | 7.84 s |
| medium(20) | **332.37 ms** | 359.46 ms | 0.752 | 2.167 GB | — |
| long(50) | **331.48 ms** | 345.02 ms | 0.767 | 2.167 GB | — |

> 与 Stage 1 基线（355/394/414 ms）相比更快，主要因为参考音频从 9.12 s 换成了 6.12 s 且音色不同；
> **不要**把这次数字当成"训练让推理变快"的证据，只能说明新角色在这台机器上依然满足实时要求。

### 2.5 角色包

`results/格雷丝.vcp` = **203.5 MB**（profile + 2 个 checkpoint + reference + dataset 标注）。
导出→空库导入→合成 1.51 s 音频，全链路通过。

---

## 3. 手工验收（3090 上实测）

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 75s 真实人声走完"新建→试听"不崩 | ✅ | 用 48.07s 素材走完；`tools/stage2_pipeline.py --stage all` 全绿 |
| 2 | conservative 档完成训练并产出 checkpoint + reference.wav | ✅ | `vlab_格雷丝_e8_s200_l32.pth`、`vlab_格雷丝-e8.ckpt`、`reference.wav` |
| 3 | 试听可懂、音色接近、无爆音 | ✅（客观指标） | 3 句全出、峰值 -5.4~-8.6 dBTP、相似度 0.59–0.71（**主观听感需人工确认**） |
| 4 | 取消后进程退出、临时文件清干净、可重新开始 | ✅ | `tools/test_cancel.py` → `CANCEL_TEST PASS` |
| 5 | 重开应用角色还在、last_used 正确 | ✅ | `tools/test_vcp_roundtrip.py` → `#5 ... PASS` |
| 6 | 导出 .vcp 再导入，角色可用 | ✅ | 同上 → `VCP_TEST PASS`（含导入后重新合成） |
| 7 | 训练期间托盘显示进度且无法启动对话 | ✅（结构保证） | 托盘 tooltip 走 `training` 态 + 进度文案；阶段内**没有对话入口**，`set_busy` 会禁用"新建角色" |

第 7 条的实测记录：`app.log` 确认 `托盘就绪`；`TrayIcon.set_state("training","ep 3/8")` 之后 tooltip 为 `VoiceLab — 训练中：ep 3/8`（`tools/test_tray.py`）。

---

## 4. 未完成项与偏离

### 4.1 按规格"明确不做"的部分

LLM 接入、正式对话、免提 VAD、打包/安装器、自动更新、8GB 卡适配、v2Pro、断点续训、Web UI —— **均未实现，符合预期**。
热键（`ui.hotkey`）只保存到配置，不注册全局热键（Stage 3）。

### 4.2 与原始规格的偏差（都需你确认）

1. **素材时长门槛 60s → 45s**（按你 2026-11-02 的口头调整）。副作用：48s 素材只切出 **10 段**，
   低于"12–20 段"的目标。代码里这是**软检查**（只写警告，不阻断），目标区间配置在 `segment.target_min_clips`。
2. **多人声阈值取 0.62**（campplus 余弦）。本素材同人两两得分最低 0.59，说明阈值偏严，
   若换素材出现误判，调 `audio_qc.estimate_speakers(threshold=...)`。
3. **超长段拆分允许 7.2s 轻微越界**：宁可超一点，也不切在词中间（`_split_long`）。
4. **示例角色没有头像**（向导里可选，训练产物里留空）。

### 4.3 已知问题 / 环境坑

1. **s1（GPT）训练在本机会崩，已用运行时垫片绕过，未改上游源码**：
   - `s1_train.py` 硬编码 `DDPStrategy(process_group_backend="gloo")`，单卡也走 DDP，Windows 下
     gloo `all_reduce` 直接 0xC0000005；
   - 同时 DataLoader 子进程在首个 batch 前也 0xC0000005。
   - 解法：`tools/s1shim/sitecustomize.py`（经 PYTHONPATH 注入）把 DDP 降级为单卡、`num_workers=0`、
     并在 `dist.get_world_size()/get_rank()/all_reduce()` 未初始化时返回单卡语义。
     **换机器/换 torch 版本时要重新验证这个垫片是否仍然必要。**
2. **onnxruntime-gpu 1.31.1 是按 CUDA 13 编的**，而 torch 是 CUDA 12.8。日志里每次都会打印
   `Failed to load cublasLt64_13.dll ...`，随后回退 CPU。不影响 ASR/VAD，只是慢 + 日志噪音。
3. **PySide6 装了 Addons、卸了 WebEngine**：`requirements.txt` 只锁 `PySide6/-Essentials/-Addons`，
   不含 `PySide6-WebEngine`（规格要求不引入 Qt WebEngine）。
4. **silero-vad 6.2.3 的 `get_speech_timestamps` 需要 16k 输入**，代码里先重采样再送 VAD，
   边界时间是秒，回映射到 48k 后才切。
5. **funasr 1.0.27 的 `sentence_timestamp=True` 不返回标点**，也不返回 `sentence_info`（返回扁平
   `timestamp` + 空格分隔字符）。所以标点是**单独跑 ct-punc** 再贴回字符 token 的。
6. **pyloudnorm 0.2.0 改了参数名**：`Meter(block_size=...)`，旧文档写的 `block=` 会 TypeError。
7. 训练日志里 `expandable_segments not supported on this platform` 是 Windows 的正常提示（配置仍写入）。

---

## 5. 关键实现决策（供 review）

| 决策 | 理由 |
|---|---|
| 训练走 `s2_train_v3_lora.py`（不是 `s2_train.py`） | v3 的 LoRA 训练入口，见 `docs/research/training-cli.md` |
| 不跑 `2-get-sv.py` | v3 的 loader 不读 `7-sv_cn`，只有 v2Pro 读 |
| epoch/batch 只写配置文件 | 两个脚本都没有对应命令行参数 |
| 训练全程 `cwd=GPT-SoVITS` + `PYTHONPATH=repo;repo/GPT_SoVITS` | 否则 prepare 脚本 `ModuleNotFoundError: text/tools` |
| 切分产物 24k/24bit，训练时上游自己重采样到 32k | v3 用 "32k 谱 + 24k mel"，输入不必是 24k |
| 试听在 UI 进程内加载 TTS | 训练已结束，显存占用 2.2GB，可接受；对话阶段（Stage 3）再考虑常驻服务 |

---

## 6. 复现命令

```powershell
# 单测（不需要 GPU）
C:\VoiceLab\.venv\Scripts\python.exe -m pytest tests -v

# 端到端（建角色→质检→切分→训练→复测）
C:\VoiceLab\.venv\Scripts\python.exe tools\stage2_pipeline.py --stage prep
C:\VoiceLab\.venv\Scripts\python.exe tools\stage2_pipeline.py --stage train

# 试听 + 音色相似度
C:\VoiceLab\.venv\Scripts\python.exe tools\verify_preview.py

# 取消 / .vcp / 持久化 校验
C:\VoiceLab\.venv\Scripts\python.exe tools\test_cancel.py
C:\VoiceLab\.venv\Scripts\python.exe tools\test_vcp_roundtrip.py

# 启动应用
C:\VoiceLab\.venv\Scripts\python.exe app\main.py
```
