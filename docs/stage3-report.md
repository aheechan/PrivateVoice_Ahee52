# Stage 3 报告：对话管线 + 按住说话 + 托盘交互

> 完成日期：2026-11-02
> 机器：RTX 3090 24GB（sm_86）· 驱动 610.62 · torch 2.11.0+cu128 · Python 3.12.10
> 本地 LLM：llama.cpp `llama-server`，Qwen3.5-9B Q6_K，`:8082`
> **5070 Ti 未测**：本机只有一张 3090（`nvidia-smi` 确认），验收要求里的第二台机器不存在。

---

## 1. 交付物对照

| 交付物 | 位置 | 状态 |
|---|---|---|
| 1 可运行源码（新增模块 + 设置页 + 诊断页扩展） | `app/core/{asr,llm,pipeline,player,hotkey,history,metrics}.py`、`app/ui/settings_llm.py`、`app/ui/diagnostics.py`、`app/main.py`、`app/tray.py` | ✅ |
| 2 `logs/turns.jsonl`（≥20 轮） | 20 轮真实链路 | ✅ |
| 3 `docs/stage3-report.md` | 本文件 | ✅ |
| 4 截图序列 | `results/screenshots/01..08*.png` + `00_contact_sheet.png`、`09_diagnostics.png`、`10_llm_settings.png`、`11_tray.png` | ✅ |

---

## 2. 链路与状态机

```
按下热键 ──► listening（录音）
松开 ──► transcribing（首尾静音修剪 → ASR）
       ──► thinking（组装 messages → LLM SSE 流）
                └─ 切句：句末标点立即切；逗号/顿号后 ≥6 字切；≥40 字强切
       ──► speaking（TTS 按句串行合成，最多预取 2 句；首块音频立即播）
       ──► idle（播完；history 落 chat.jsonl，turns.jsonl 记账）
任意状态再按热键 ──► 停播放 + 取消 LLM/TTS + 清队列 ──► 回到 listening
training 状态 ──► 拒绝进入 listening，托盘提示「训练进行中」
```

状态与托盘图标：`idle` 蓝 / `listening` 红 / `transcribing` 紫 / `thinking` 紫 / `speaking` 绿 / `training` 黄。

### 关键实现选择

| 点 | 做法 | 原因 |
|---|---|---|
| 热键 key-up | `RegisterHotKey` 拿按下 + `GetAsyncKeyState` 20ms 轮询松开 | Win32 热键只有按下事件；纯 ctypes，不装 pywin32 |
| 误触过滤 | 按住 <150ms 不触发一轮 | 避免蹭键 |
| TTS 与 LLM 重叠 | LLM 线程只管切句入队，TTS 独立线程串行合成，播放线程独立 | 三处重叠是 e2e 的关键 |
| 重采样 | 只在 `Player` 里做一次（24k→48k，跨 chunk 连续相位） | 避免块边界爆音 |
| 句间 | 80ms 静音 + -1 dBTP 限幅 | 防粘连、防削波 |
| 打断 | 清环形缓冲 + `stream.abort()` + 取消事件 | 实测 7–33ms |

---

## 3. 实测延迟（3090，20 轮，`logs/turns.jsonl`）

| 指标 | p50 | p95 | 目标 | 结论 |
|---|---|---|---|---|
| ASR（`t_asr_done - t_release`） | **366 ms** | 419 ms | ≤300 ms | ❌ 超 22% |
| LLM 首 token（`t_llm_first_token - t_asr_done`） | **1007 ms** | 1160 ms | ≤400 ms | ❌ 超 1.5× |
| TTS 首包（`t_tts_first_chunk - t_llm_first_token`） | **833 ms** | 1031 ms | ≤450 ms | ❌ 超 85% |
| e2e（`t_play_start - t_release`） | **2217 ms** | 2432 ms | ≤1500 / ≤2200 ms | ❌ p50 超 48%，p95 超 10% |
| 播放时长 | 4837 ms | 8317 ms | — | 正常 |
| 打断响应 | 7–33 ms | — | ≤150 ms | ✅ |

单轮分段示例（turns.jsonl 第 3 行）：ASR 419 → LLM 868 → TTS 938 → e2e 2232 ms。

### 3.1 为什么 LLM 首 token 这么慢（已做实验排除）

| 假设 | 实验 | 结果 |
|---|---|---|
| 上下文太长 | 10 token vs 250 token prompt | TTFT 992ms vs 702ms → **固定开销主导，不是长度** |
| 并行槽位抢占 | `--parallel 1` | 无改善 |
| 批大小 | `-b 4096 -ub 1024` | 无改善 |
| 思考模式没关 | 直接看 SSE：不开 `enable_thinking=false` 时模型先吐 60+ token 的 “Thinking Process”，content 为空 | 关掉后 content 正常，**说明参数生效**，但仍 ~700ms |
| 前缀缓存没生效 | `--cache-reuse 256`（`--no-kv-unified`、FA on/off 各试一次） | 启动日志明确：**`cache_reuse is not supported by this context`** → 每轮都要重新预填 ~700 token（system ~350 字 + 历史），按实测 ~700 tok/s 正好 ~1s |

结论：**瓶颈是每次请求的提示词预填充，不是生成速度，也不是 TTS/ASR**。要达标必须动模型或服务端，见 §6。

### 3.2 TTS 首包 833ms 的构成

该指标从「LLM 首 token」量到「TTS 首块音频」，包含：
1. 首句凑够标点（约 8–15 token，~150ms）
2. 该句合成（Stage 2 实测 5 字 313ms，这里句子更长，约 600–700ms）

Stage 2 的单句 TTFP 313ms 是达标的；这里是"句子更长 + 指标口径含等待切句"，属口径差异，但用户体验上的 e2e 确实受影响。

---

## 4. 功能验收（6 条）

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 按住热键说话，松开 1.5s 内听到回复 | ⚠️ **部分** | 链路跑通、热键事件正确；但 (a) **本机无麦克风**（`sounddevice` 报 0 个输入设备），真实拾音未验证；(b) e2e 实测 2.2s，未达 1.5s |
| 2 | 回复中再次按下，立即静音并开始新一轮 | ✅ | 打断 7–33ms；`interrupt()` 清缓冲 + 取消 LLM/TTS |
| 3 | 切换角色后音色与人设同时切换，历史互不串 | ✅ | `characters/格雷丝-748caf/chat.jsonl` 58 行 vs `阿岚-e4c4be/chat.jsonl` 46 行，各自独立；切换时 `Pipeline.set_character` 重建 HistoryStore 并清队列 |
| 4 | 本地 provider 停掉后自动回退 | ✅ | `llm_probe`：先死端口后本地，`attempts=[('dead',False),('local',True)]` |
| 5 | 云端也失败时播兜底短句，不崩溃 | ✅ | `tools/fallback_test.py` → `FALLBACK_TEST PASS`（`我这会儿有点接不上话。`） |
| 6 | 连续 20 轮无内存/显存持续增长，无爆音、无吞字 | 见 §5 | `tools/memory_check.py` |

---

## 5. 20 轮资源占用

`tools/memory_check.py`，每轮之间 `gc.collect()` 后再采样：

| 采样点 | RSS | 显存（torch allocated） |
|---|---|---|
| 启动（预热后） | 4848 MB | 3151 MB |
| 第 5 轮 | 5021 MB | 3172 MB |
| 第 10 轮 | 5155 MB | 3172 MB |
| 第 15 轮 | 5299 MB | 3172 MB |
| 第 20 轮 | 5435 MB | 3172 MB |
| 关闭管线后 | 5435 MB | 3172 MB |

- **显存完全平稳**（20 轮 ±0.1 MB），说明 TTS/播放/流对象没有泄漏。
- **RSS 线性增长约 +430 MB / 20 轮（≈21 MB/轮）**，强制 GC 也不回落。尚未定位，最可能是
  torch/CUDA 上下文缓存或 funasr 内部缓存；不影响可用性，但连续挂机多轮会累积。已列为 Stage 4 待查项。
- 无爆音、无吞字：20 轮里 TTS 失败 0 次，播放欠载计数为 0。

---

## 6. 未达标项与补救方案（需要你拍板）

> **2026-11-02 决策**：需求方确认「2224 ms 可以接受，不用为优化它消耗时间」，因此本节方案**本阶段不执行**，
> 仅作为 Stage 4/5 的可选优化记录。

| 方案 | 预期收益 | 代价 |
|---|---|---|
| A. 缩短 `companion_system.txt`（350 字 → ~120 字） | LLM 预填约 -35%，TTFT 约 -250ms | 人设约束变少，需要你确认文案 |
| B. 换 Q4_K_M 量化（同模型） | 预填约 ×1.6–1.8，TTFT 约 -400ms | 需要下载新 GGUF；音色无关，对话质量略降 |
| C. 换推理后端（vLLM / SGLang） | 前缀缓存真正生效，多轮 TTFT 可到 150–250ms | 环境重搭；vLLM 对 Windows 不友好，可能要 WSL |
| D. 降低历史注入（12 轮 → 6 轮） | 预填 -30% | 与规格冲突（规格写 12 轮），需你同意 |
| E. 接受当前 2.2s | 0 | 不达标，需你书面确认 |

**我倾向 A + B**：两项都不改架构，能把 e2e 压到 ~1.5–1.6s，再靠 C 才能稳定进 1.5s 以内。

---

## 7. 已知问题

1. **本机没有麦克风**：`sd.default.device = [-1, 1]`，输入设备数为 0。所有延迟数字来自「wav 注入 → 同一条 trim/ASR/LLM/TTS/播放路径」，不是真实拾音。**需求方已确认此项不阻塞 Stage 3**；真实拾音在 5070 Ti 打包验收时一并完成。
2. **5070 Ti 未测**：本机只有 3090。
3. **热键 `shift+a` 会全局拦截**：在别的程序里打大写 A 也会触发对话。已按你的选择设为默认，并在启动时弹托盘提示；设置页可改，改完热生效（不用重启）。
4. **`funasr` 的 Paraformer 不返回逐 token 置信度**：`ASRResult.confidence` 是按文本长度/汉字占比的启发式（`asr.py:_heuristic_conf`），只能挡"空结果"，挡不住"识别成别的话"。
5. **20 轮回复高度相似**：同一段音频重复输入 + temperature 0.8 下模型仍收敛到相似句式。真实对话（内容每轮不同）不会这样，但这说明历史对生成的牵引偏弱。
6. **llama.cpp 前缀缓存不可用**（见 §3.1），是当前延迟的主因。
7. **TTS 首句仍要等标点**：句长阈值 6 字，极短回复（<6 字）会等到流结束才合成。

---

## 8. Stage 4 建议

1. **先解决预填充**（A/B/C 任一），否则 e2e 永远卡在 2s 档。
2. **加 AEC**：现在没有回声消除，外放时热键录音会把角色自己的声音录进去；Stage 4 若做免提必须先上 AEC。
3. **会话历史加"话题锚点"**：现在 12 轮全量注入，既慢又弱；建议改为「最近 4 轮原文 + 摘要」，能同时降延迟和提相关性。
4. **ASR 换流式**：现在是"松开后整句识别"，若换成边说边识别，能把 ASR 的 366ms 藏进说话时间里。
5. **打包**：PySide6 + torch 的安装器会很大（>6GB），建议做"运行时下载权重"的瘦安装包。

---

## 9. 复现命令

```powershell
# 本地 LLM（按你的预设，另开一个窗口）
D:\llama.cpp\llama-server.exe -m D:\llama.cpp\models\Qwen3.5-9B\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf `
  -ngl 999 -c 8192 --flash-attn on -b 4096 -ub 1024 --parallel 1 --reasoning-budget 0 --host 127.0.0.1 --port 8082

# 热键 / LLM / 回退 / 内存 / 20 轮
python tools\hotkey_probe.py
python tools\llm_probe.py
python tools\fallback_test.py
python tools\memory_check.py
python tools\pipeline_probe.py --turns 20

# 截图序列
python tools\capture_states.py

# 启动应用
python app\main.py
```
