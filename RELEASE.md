# Release Notes — PrivateVoice_Ahee52 1.0.5

**版本 / Version:** `1.0.5`
**日期 / Date:** 2026-11-13
**类型 / Type:** 修复版（1.0.4 基础上修 4 个验收问题）
**安装目录 / Install dir:** `%ProgramFiles%\PrivateVoice_Ahee52`

---

## 中文

### 1.0.5 修了什么

| # | 问题 | 修复 |
|---|---|---|
| 1 | 设置里的快捷键无法改回 Shift+A | 打开设置时**暂停全局热键**，录制不再被全局拦截；关闭设置自动恢复 |
| 2 | whisper 多语言识别“没听清” | VAD 过滤失败时自动**去掉 VAD 重试**，并放宽 `no_speech_threshold` / `log_prob_threshold` |
| 3 | 提示语不跟随语言 | “没听清，再说一次”按界面语言输出英文；识别语言为 en/ja/ko 时回复同语言 |
| 4 | 大模型管理无法新增/编辑 | 修复 `tz(超时 (ms))` 语法错误导致的 `NameError`，Provider 对话框可正常打开与保存 |
| 5 | 设置 / 大模型页面英文不完整 | 两个页面在打开时统一翻译（标题、按钮、标签、表头、ASR 分组、Provider 对话框） |
| 6 | 英文识别无空格/标点 | whisper 结果统一补空格、标点后空格、首字母大写、句末句号 |
| 7 | 安装目录脚本中文名 | 改为 `Launch.bat` / `SelfTest.bat` / `MicCheck.bat` / `README.txt`（英文） |
| 8 | 图标仍是旧的 | 包根与 `assets` 全部换成 chat-wave 图标 |
| 10 | 训练报错（gradio 导入失败） | `huggingface_hub` 被 faster-whisper 升到 2.4，移除了 `HfFolder`；已降级到 0.26.2 并随包分发 |
| 11 | 切换 ASR 后识别失败 | 保存设置后**自动重启**程序（1.2 秒后），不再依赖手动重启 |
| 9 | 英文输入仍回中文 | 语言指令移到 system prompt 末尾（中英双语强约束），user 轮再追加 `(Please answer in English only.)`，历史仍存原文 |

### 实测

```
快捷键：Shift+A 录制为 shift+a，解析为 shift+A
设置窗口：打开成功；大模型 Provider 对话框可新增/编辑
whisper 中文参考音频（5.5s）：识别成功，带标点
英文文本：'hello,it is good  to see you' -> 'Hello, it is good to see you.'
语言判定：en / zh / ja / ko
启动：硬件检查通过 -> 热键注册 -> 进程持续运行
```

### 安装包（8 个文件，每个 < 2 GB）

| 文件 | 大小 | SHA256 |
|---|---|---|
| `PrivateVoice_Ahee52-1.0.5-Setup.exe` | 6.6 MB | `5d7864eb6615b3f106fe036e6edc0c0621b498543c1bc16098b038bd5e40ed06` |
| `-1.bin` | 1.76 GB | `479543827342ff1c585390f9ff9a543f2ec6adfa288afe272a2d97f3c2acd580` |
| `-2.bin` | 1.77 GB | `0c2c211583d96b8b73c029e8bd00e3da00a94d70ecf6c41ae827123cbe7cb937` |
| `-3.bin` | 1.77 GB | `91e6a28824e76517bde975bb3158b302d1eba1eef25c6bc75f12e8759d6b708d` |
| `-4.bin` | 1.77 GB | `43e5c03980abf1527c43a11e08e06ccf2a87cf4ee8f04270852d42ff6463c5e0` |
| `-5.bin` | 1.77 GB | `267607db33cf9465b795166f6ec2cb975c3576ddf70412a9f4034ce0138166d3` |
| `-6.bin` | 1.77 GB | `773cb578cc1c5b30eae31c310b63e14e31b6c892beacb2a57620a6f7f1a9e9da` |
| `-7.bin` | 0.80 GB | `30f92f6670679b783a9244229c8fd18accd2ab41420d295c55b996aff94b1269` |

### 增量升级包

`PrivateVoice_Ahee52-1.0.5-update.zip`（约 0.8 MB，只含 `app/ assets/ tools/ README.txt app.ico`）

用法：退出程序 → 解压 → 覆盖到 `D:\PrivateVoice_Ahee52\` → 重启。
不覆盖 `models_cache\`、`GPT-SoVITS\`、`characters\`、`config.yaml`。

### 已知限制

- whisper 只带 `small` 模型（461 MB）；大模型更准但更慢，见 README 的可选升级说明
- 中英文 ASR 互斥，切换后自动重载（3–5 秒）
- 安装向导为英文

---

## English

### Fixed in 1.0.5

1. Hotkey could not be captured while the global hotkey was active — settings now suspends it.
2. Whisper "didn't catch that" — retries without VAD and relaxes speech thresholds.
3. Prompts now follow the UI language; replies follow the detected language (en/ja/ko).
4. LLM provider dialog crashed with `NameError` (`tz(超时 (ms))`) — fixed; add/edit works.
5. Settings and LLM pages are fully translated in English mode.
6. English ASR output gets spaces, punctuation and capitalization.
7. Bundled scripts renamed to `Launch.bat`, `SelfTest.bat`, `MicCheck.bat`, `README.txt`.
8. chat-wave icon applied everywhere, including the package root `app.ico`.

### Package

Eight files, SHA256 above; total ≈ 11.5 GB. Incremental update zip ≈ 0.8 MB.

### Known limitations

- Whisper ships `small` only (461 MB); larger variants are more accurate but slower
- Chinese/English ASR are mutually exclusive; switching reloads (~3–5 s)
- Installer UI is English
