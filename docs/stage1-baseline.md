# Stage 1 基线（VoiceLab 运行时实测记录）

> 来源：`results/spike.json`（default / nokernel 两组各 20 次运行）、`conclusion.md`、`README.md`。
> 本文件是 Stage 2 的对照基线，Stage 2 训练出的角色 checkpoint 必须用同一套协议复测并写入 `results/character_ttfp.json`。

## 1. 版本锁定

| 组件 | 版本 / commit |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24 GB，compute capability 8.6（sm_86） |
| 驱动 / CUDA | 610.62 / CUDA 12.8，cuDNN 9.19 |
| Python | 3.12.10（`C:\VoiceLab\.venv`） |
| PyTorch | `2.11.0+cu128`（torchaudio 同版本） |
| GPT-SoVITS | `08d627c3338173c3229286d8787060d6559fe0f8`（2026-04-30，含 cuda graph 支持） |
| Aqua-TTS | `eb8e0d59a9ca317b40d8182b04c12ba01a489f8d`（tag `v0.3.0`） |
| 权重 | `s1v3.ckpt` 155,284,856 B；`s2Gv3.pth` 769,035,145 B |

## 2. TTFP 实测（v3，3090，`nokernel` 组为有效基线）

| 文本档 | 字数 | TTFP p50 | TTFP p95 | TTFP min/max | 总耗时 p50 | 音频时长 p50 | RTF p50 |
|---|---|---|---|---|---|---|---|
| short | 5 | **354.59 ms** | 715.71 ms | 307.14 / 735.32 | 1283.29 ms | 1.303 s | 0.8985 |
| medium | 20 | **393.54 ms** | 499.88 ms | 362.25 / 650.68 | 3765.85 ms | 4.586 s | 0.8247 |
| long | 50 | **414.28 ms** | 818.40 ms | 345.81 / 1159.73 | 8420.80 ms | 10.198 s | 0.8267 |

- 20 次运行，5 次预热后统计（`runs=20`, `warmup=5`）。
- `default` 组数字与 `nokernel` 基本相同（short p50 354.59 vs 353.85 ms），因为两组实际跑的是同一条 PyTorch 路径，差异只在加载时间。

## 3. 显存与加载

| 指标 | short | medium | long |
|---|---|---|---|
| 峰值 allocated | 2.173 GB | 2.178 GB | 2.186 GB |
| 峰值 reserved | 2.32 GB | 2.324 GB | 2.324 GB |

- 模型加载：**7.17 s**（有效组）；26.64 s 那次是失败的 BigVGAN 内核编译拖慢，不是正常值。
- backend 实测：`bigvgan_kernel_mode = pytorch`、`bigvgan_use_cuda_kernel_attr = false`、`flash_attn_active = false`、`kv_cache_layout = bucket`、`bucketed_graphs = 6`（128/256/448/512/768/1024）。

## 4. 推理协议（Stage 2 试听与复测必须沿用）

```json
{"how_to_cut":"按标点符号切","top_k":5,"top_p":1,"temperature":0.6,
 "speed":1.1,"sample_steps":4,"chunk_size_seconds":0.25,
 "enable_cuda_graph":true,"enable_static_kv":true}
```

- 参考音频：`C:\VoiceLab\assets\ref.wav`（9.12 s，48k 单声道 16bit），参考文本在 `assets/ref.txt`。
- 输出采样率 24 kHz，流式 chunk 约 0.25 s。

## 5. Stage 1 遗留问题（Stage 2 必须避开）

1. **BigVGAN 自定义 CUDA 核从未编出来**：nvcc 12.8 与 MSVC 14.51（VS 2026）不兼容，`cudafe++` 报 `0xC0000005`。结论只能写"核从未跑起来"，不能写"关掉无影响"。
2. **无 VS Build Tools 也能出声**：`nokernel` 组不触发编译，三档全部正常，见 `results/sample.wav`。
3. `README.md` 同时写了 "v3 (2025-04-01)" 与 "validated against 08d627c"，后者实为 2026-04-30，**按 commit 执行**。
4. `vcvars64.bat` 路径未加引号，E: 盘无 8.3 短名时 MSVC 环境导入失败；当时用 junction + `MSVC_VCVARS64_PATH` 绕过，未改模型代码。
5. `ninja` 未声明为依赖但编译内核必需。
6. **`infer_stream` 返回 `(sr, chunk, text)`**——"兼容两种返回"的写法会把 `text` 误当音频。Stage 2 的 TTS 适配器必须按三元组解包。
7. `assets/ref.wav` 是 9.12 s 单段参考，**不满足 Stage 2 训练所需的 60–90 s 素材**。

## 6. Stage 2 复测口径

- 同一批三档文本（5/20/50 字）、同一协议参数、20 次运行 + 5 次预热。
- 角色 checkpoint 训练完成后跑一次，结果写 `results/character_ttfp.json`，字段与 `spike.json` 的 `tiers.*` 对齐，外加 `character_id`、`gpt_ckpt`、`sovits_ckpt`、`trained_at`。
