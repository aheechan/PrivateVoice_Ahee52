# Release Notes — PrivateVoice_Ahee52 1.0.6

**版本 / Version:** `1.0.6`
**日期 / Date:** 2026-11-21
**类型 / Type:** 体验优化（对话节奏、托盘图标、ASR 热切换、情绪预设）

---

## 中文

### 1.0.6 改了什么

| # | 问题 | 修复 |
|---|---|---|
| 1 | 普通对话也冒“稍等，让我查一下” | 该句只在**联网检索**时出现；普通对话不再预设填充语，只有首 token 超过 900ms 时**偶尔**“嗯……”一声，多数时候直接安静地等 |
| 2 | 托盘图标不是紫色对话框 | 托盘改用品牌图标 `assets/app_256.png`，仅叠加状态色角标（待机 / 听着 / 思考 / 说话 / 训练） |
| 3 | 切换 ASR 要重启程序 | 改为**热加载**：保存设置后后台重建识别引擎并预热，托盘提示“正在加载…”，完成后提示“可以开始对话”，无需重启 |
| 4 | 回复普遍过于谄媚 | system prompt 增加表达底线（不奉承、不逐句夸、不复述原话、允许不同意见）；情绪档位的提示词同步收紧，语气更克制 |

### 对话节奏细节

- **检索时**：随机说一句「稍等，让我查一下。」/「嗯，我查一下。」/「我看看最新的。」，趁这句合成的时间去搜索
- **普通对话**：不预设任何填充语；首 token 若超过 900ms，有 45% 概率轻声「嗯……」，否则保持沉默
- 三个参数都可在 `config.yaml` 调整：`chat.filler_delay_ms` / `chat.filler_probability` / `chat.fillers`

### 安装包（8 个文件，每个 < 2 GB）

| 文件 | SHA256 |
|---|---|
| `PrivateVoice_Ahee52-1.0.6-Setup.exe` | `7edb480f20d11d6b63a14d49f1b5b1a3b3937935c5f1d01edf703b83f36aa4cd` |
| `-1.bin` | `84e1892f0370b527dc7cb0571deaf4f61476577bc14ea9cf21caaeedfceac9cd` |
| `-2.bin` | `8c68bea12937b1f6272e7bf0dd00c9b8bb38842a124c71fad8ff9bdc3d33b6be` |
| `-3.bin` | `9e7e6228374144d63fa2c5e0af733a6aadde3c38d5691534dd706f00a23f6133` |
| `-4.bin` | `ed67443d43d8e8ba238986afd52534e45153d0806deb43a3b8bbc6164d34746b` |
| `-5.bin` | `69924596f89f5368f80bff4cfa8d31c386093d12aa474991a4b873cd970e2578` |
| `-6.bin` | `8b5ff0d55fac9f718ffbc9a893c400d1ca5ef1a30f62affa8b436154ce1e15de` |
| `-7.bin` | `6da94a91c55b3c3cb82adb54ba0bef8d0f396ee6f9197dbfd34636aea7d4b404` |

### 增量升级包

`PrivateVoice_Ahee52-1.0.6-update.zip`（1.24 MB）

退出程序 → 解压 → 覆盖 `app`/`assets`/`tools`/`runtime`/`README.txt`/`app.ico` 到安装目录 → 重启。
不动 `models_cache\`、`GPT-SoVITS\`、`characters\`、`config.yaml`。

### 已知限制

- 热切换 ASR 时，第一次识别可能需要等预热完成（托盘会显示进度）
- whisper 首次加载约 1 GB 显存，显存不足回退 CPU
- 安装向导为英文

---

## English

### Fixed in 1.0.6

1. “Let me check” now appears **only** for web lookups. Normal turns have no canned filler; if the first
   token takes longer than 900 ms there is a 45% chance of a soft “mm…”, otherwise it simply stays quiet.
2. Tray icon uses the brand chat-wave artwork (`assets/app_256.png`) with a small state badge.
3. Switching ASR no longer restarts the app: the engine is rebuilt and warmed up in the background,
   with tray feedback (“loading…” → “ready to talk”).
4. Replies are less sycophantic: the system prompt now forbids flattery, per-sentence praise and
   parroting, and allows disagreement. Mood hints were tightened accordingly.

### Package

Eight files, hashes above. Incremental update zip ≈ 1.24 MB.
