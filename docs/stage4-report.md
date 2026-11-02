# Stage 4 报告：打包（嵌入式 Python + 安装器 + 首启向导）

> 完成日期：2026-11-02
> 产物：`C:\VoiceLab\dist\情感陪伴器_v1\`（免安装版，**11.51 GB / 49,986 文件**）
> 安装器：`C:\VoiceLab\dist\installer\情感陪伴器_v1_Setup.exe`（分卷，共 11.51 GB）
> 实机验证：3090 上完成理论运行测试（`SELFTEST PASS`）+ **离线 ASR 验证**
> **5070 Ti 未实机验证**：本机无此卡；已用 `GpuInfo` 注入方式验证 sm_120 / 16GB 能通过门禁。

---

## 1. 包结构

```
情感陪伴器_v1\
├── runtime\                 嵌入式 Python 3.12.10 + 5.6GB site-packages
│   ├── python.exe / pythonw.exe
│   ├── python312._pth       路径表（含 import site，见 §3 坑 1）
│   └── Lib\site-packages\   torch 2.11+cu128 / PySide6 / funasr / sounddevice …
├── app\                     应用代码（Stage 2 + Stage 3 全部模块）
├── GPT-SoVITS\              v3 推理/训练代码 + pretrained_models（s1v3/s2Gv3/hubert/roberta）+ G2PWModel
├── Aqua-TTS\                aquatts 源码（不再依赖 editable 安装）
├── characters\              示例角色：格雷丝、阿岚（路径已改为相对）
├── models_cache\            内置 ModelScope 缓存（ASR/标点/说话人，2.06 GB，离线可用）
├── assets\                  参考音频 + 提示词
├── logs\ / datasets\ / results\
├── 启动.bat                 启动（pythonw，无控制台）
├── 自检.bat                 一键自检
├── 使用说明.txt
└── app.ico
```

安装器会把上面整个目录复制到 `%ProgramFiles%\情感陪伴器_v1`，创建开始菜单与桌面快捷方式
（快捷方式直接指向 `runtime\pythonw.exe app\main.py`，不需要额外 launcher exe），
并在安装完成后默认勾选“立即自检”。

---

## 2. 理论运行测试（3090，打包产物内跑）

`自检.bat`（= `runtime\python.exe tools\selftest.py`）的输出：

| 步骤 | 结果 |
|---|---|
| 运行时 | python 3.12.10 · torch 2.11.0+cu128 · cuda 12.8 available=True ✅ |
| 显卡门禁 | RTX 3090 · 24GB · sm_86 → 通过 ✅ |
| 依赖导入 | httpx / sounddevice / soxr / funasr / PySide6 ✅ |
| 角色与权重 | 2 个 ready 角色，gpt/sovits ckpt 均存在（相对路径回落生效） ✅ |
| 预热 | ASR 6.7s（首次加载）+ TTS 加载 13.1s + 文本前端预热 8.5s ✅ |
| 链路（注入 4s 真实语音） | ASR 命中、LLM 回复、TTS 出声 **4.75s 音频** ✅ |
| 延迟 | e2e **2421ms**（ASR 358 / LLM 928 / TTS 1118） |
| 打断 | **11ms** ✅ |
| 结论 | **SELFTEST PASS**，报告写 `results\selftest.json` |

另外验证了首启向导在包内可用：`gpu_ok=True`、`model_ok=True`（模型齐全），
`ui.first_run_done=False`（首次启动确实会弹向导）。

### GPU 拒卡逻辑

用注入 `GpuInfo` 的方式验证（不依赖真实硬件）：

| 模拟显卡 | 结果 |
|---|---|
| GTX 1650 / 4GB / sm_75 | ❌ 拒绝：「计算能力 sm_75，只支持 RTX 30/40/50 系」 |
| RTX 3060 12GB / sm_86 | ✅ 通过 |
| **RTX 5070 Ti / 16GB / sm_120** | ✅ **通过**（目标机对照） |

---

## 3. 打包过程中踩到的坑（都已修）

0. **离线模型**：把 `%USERPROFILE%\.cache\modelscope` 整包复制到 `models_cache\modelscope`，
   并在 `app/config.py` 顶层（早于 funasr/modelscope import）设 `MODELSCOPE_CACHE` /
   `MODELSCOPE_CACHE_MODEL` / `MODELSCOPE_MODEL_CACHE`。
   **验证方式**：把用户缓存目录改名藏起来再跑 ASR，模型路径仍指向包内
   `models_cache\modelscope\models\...\model.pt`，识别输出正常 → 证明真离线可用。
   未打包 Whisper large-v3 的 HF 缓存（2.88GB）：它只是 ASR 的降级备选，
   需要时把 `%USERPROFILE%\.cache\huggingface` 放进 `models_cache\huggingface` 即可（`HF_HOME` 已接）。
1. **嵌入式 Python 默认不处理 `.pth`**：`python312._pth` 里必须加一行 `import site`，
   否则 `setuptools` 的 distutils 补丁不生效，`funasr` 里 11 个模型模块导入失败
   （注册表从 51 个掉到 40 个，`paraformer-zh` 找不到类 → `TypeError: 'NoneType' object is not callable`）。
   这个错误很隐蔽：注册是 lazy 的、异常被吞掉。
2. **editable 安装的 aquatts 写死了绝对路径**：`__editable__.aqua_tts-0.3.0.pth` 里映射
   `C:\VoiceLab\Aqua-TTS\aquatts`，换机必炸。做法是删掉这两个文件，改由 `_pth` 加
   `..\Aqua-TTS` 走正常路径查找。
3. **不能把 `GPT-SoVITS` 和 `GPT_SoVITS` 塞进 `_pth`**：这两个目录里有 `config.py`、
   `module\`、`text\` 等通用名，会污染 sys.path。改成 TTS 引擎在加载时按需插入。
4. **角色的绝对路径**：`profile.json` 里的 `gpt_ckpt` 是 `C:\VoiceLab\...`，换机失效。
   新增 `CharacterStore._relocate()`：路径不存在时回落到 `<角色目录>\checkpoints\<文件名>`。
   包内示例角色已改成相对路径，用于验证这条回落。
5. **G2PW 按 CWD 解析**：TTS 引擎加载时会 `chdir` 到 `GPT-SoVITS`，包里必须保留该目录结构。
6. **安装器体积**：9.45GB 源，Inno Setup 用 `Compression=none`（压缩要几十分钟），
   构建时间从 ~40 分钟降到几分钟；代价是 Setup.exe 也接近 9.5GB。

---

## 4. 安装后"启动没反应"的排查与修复（2026-11-02）

用户在 `D:\情感陪伴器_v1` 安装后双击启动，看不到任何反应。实际有三个叠加问题：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 首次启动完全没有窗口 | `main.py` 里写成了 `wiz.exec() != wiz.Accepted`，`QDialog` 的枚举在类上，实例没有该属性 → 抛 `AttributeError` 并被 `except` 吞掉，向导直接跳过 | 改成 `QDialog.DialogCode.Accepted`，并记录返回值 |
| 2 | 之后再点也没反应 | 第一次其实**已经启动了**（托盘进程 PID 在跑，只是没窗口）；第二次启动命中单实例检测，**静默 return 0** | 第二次启动改为弹提示「已经在运行了，图标在托盘」，并说明怎么找 |
| 3 | 看不到托盘图标 | Windows 默认把新图标折叠进 `^` 溢出区；且托盘程序本身没有主窗口 | 启动时弹气泡提示；**手动启动额外打开「角色管理」窗口**，开机自启则加 `--tray-only` 只留托盘 |

顺带修掉的两个隐患：

4. **包里带了一份 config.yaml**（是我自检时生成的），里面写死了 `C:\VoiceLab\dist\...` 的绝对路径。
   装到 D: 后程序按那份配置去找模型，在别的机器上会全部找不到。
   现在**发布包里不再带 config.yaml**，首次运行按程序所在目录生成；同时加了
   `Config._heal_paths()`：只要 `paths.gpt_sovits` 不存在，就把所有路径重置为本机默认值。
5. **单实例提示**：以前第二次启动静默退出，用户完全不知道发生了什么。

验证（在 `D:\情感陪伴器_v1` 实测）：修完后启动 → 日志出现
`硬件检查通过 → 托盘就绪 → 窗口标题=角色管理`，进程存活；首启向导的
`_finish()` 也能正常写回 `first_run_done=true`，配置路径自愈为 `D:\情感陪伴器_v1\...`。

---

## 5. 主界面与角色编辑（2026-11-02 追加）

用户反馈"打开只有角色管理表格"，补了三块：

### 5.1 主界面 `app/ui/main_window.py`

- 头像（无图时用名字首字）、角色名、音色/素材信息
- 状态灯：`待机 / 听着 / 识别中 / 思考中 / 说话中`，与托盘同色
- **文字输入框 + 发送**（在说话按钮上方）：输入回车 = `pipeline.submit_text()`，
  跳过录音与 ASR，直接走 LLM→切句→TTS→播放。**没有麦克风的机器靠它跑通整条链路**
- **按住说话按钮**（鼠标按下 = `pipeline.press()`，松开 = `release()`，与热键同一条路径）
- 最近一轮对话（读 `chat.jsonl`），出错红字提示
- **预热按钮**：手动跑 `pipeline.warmup()`，显示 `ASR x.xs · TTS 加载 y.yy s · 试合成 z.zz s`
- 快捷入口：设置 / 大模型 / 诊断 / 新建角色
- **关闭 = 收进托盘**，不退出；手动启动默认打开它（`--tray-only` 时只留托盘）

### 5.2 角色管理加编辑栏

左侧列表 + 右侧表单：名字、开场白、**人设**、LLM 温度 / max_tokens、头像，
底部只读显示状态与 ckpt 文件名。保存后写回 `profile.json`，并让 `pipeline`
重新加载角色对象（下一轮对话生效）。

### 5.3 修掉"人设没进 LLM"的 bug

原来 `Pipeline._messages()` 只拼了 `companion_system.txt` 模板，**角色的 persona 根本没发出去**，
LLM 一直按通用模板说话。现在 system prompt 结构：

```
<通用说话规则 template>
【你的设定：格雷丝】
<profile.json 里的 persona>
以上设定优先于前面的通用规则…
你的开场白是：「…」（只在第一次见面时用）
```

验证：`_messages()` 第一条消息里能搜到 persona 与 greeting 原文。

### 5.4 其他

- 删掉演示角色「阿岚」（复制格雷丝权重的假数据集，显示 0 段容易误解）。
- `CharacterStore` 读 `index.json` / `profile.json` 改用 `utf-8-sig`：
  用记事本另存会带 BOM，之前会把整个角色库读成空列表。
- 表格里没有数据集的角色显示 `—` 而不是 `0`。

截图：`results/screenshots/12_main_window.png`、`13_char_manager.png`。

---

### 5.5 语音合成偶发卡死（2026-11-02 排查）

现象：托盘气泡「出了点问题 · 回复合成失败」。

日志定位到这一行之后停住：

```
15:00:58.487  tts_inference: 执行GPT推理...
15:01:15.636  WARNING voicelab.controller: pipeline error: 回复合成失败
```

即 **T2S（GPT）推理进去后 17 秒没返回**，被我们当时 12 秒的首包看门狗判为失败。
没有异常栈，说明是卡在 CUDA kernel / graph replay 里，不是抛错。

改动（都为了"卡住能自愈、且不再误报"）：

1. **打断时显式 `gen.close()`**：之前中断是直接 `return` 抛弃生成器，上游的
   `finally`（释放 CUDA Graph / KV）要等 GC 才执行，期间引擎状态是脏的。
   压测「说到一半打断 → 立刻重说」8 轮，改前偶发下一句卡死，改后 8/8 通过。
2. **预热与正常合成串行**：`warmup()` 的试合成原来不持 `_tts_lock`，启动那 30 秒里
   如果用户马上说话，会两个线程同时用同一个引擎。现在也拿锁。
3. **首包看门狗 12s → 25s**，并且超时后**重置引擎**（unload，下次用到重新加载），
   错误文案改成「语音合成超时，已重置引擎，请再说一次」。
4. **取消不再误报超时**：打断时 `_t_tts` 为 0 会被当成超时并重置引擎，现在取消走静默返回。
5. 新增日志：`TTS 引擎已加载/复用`、`TTS 完成：… N 块 / X.Xs`、`TTS 被取消：…`，
   下次再卡能直接从 app.log 看出卡在第几句、第几块。

**根因仍未 100% 确认**：只在用户那次（切角色后第一句）出现过一次，本地反复压测没复现。
现在的行为是"最多 25 秒后自愈"，不会再永久卡住。

---

### 5.6 安装目录改成纯 ASCII（2026-11-02）

原来的默认目录是 `%ProgramFiles%\情感陪伴器_v1`，用户指出中文路径在以往项目里会出问题。处理：

- 安装目录改为 **`{autopf}\VoiceCompanion_v1`**，安装包名 `VoiceCompanion_v1_Setup.exe`；
  **显示名仍然是「情感陪伴器_v1」**（开始菜单、窗口标题、托盘提示不受影响）。
- 便携目录也重命名为 `dist\VoiceCompanion_v1`。
- `main.py` 启动时检查 `ROOT` 是否含非 ASCII 字符，含则写一条 warning 日志，
  便于以后有人手工搬到中文目录时快速定位。
- 现场验证：把已安装的 `D:\情感陪伴器_v1` 直接改名成 `D:\VoiceCompanion_v1`，
  旧 `config.yaml` 里的绝对路径失效 → `Config._heal_paths()` 自动重置为本机默认值，
  应用正常启动。

**结论：中文路径不是这次"合成卡死"的根因**——在纯 ASCII 的 `C:\VoiceLab` 上
用同一套代码复现出了完全一样的卡死（见 5.5 的并发问题）。但既然历史经验有风险，
安装路径统一 ASCII，配置与代码都不再依赖非 ASCII 路径。

实测（`D:\VoiceCompanion_v1`，打包运行时）：

| 项目 | 结果 |
|---|---|
| 预热（ASR+TTS+试合成） | 30.5 s |
| 文字输入一轮 | e2e 2513 ms，音频 5.44 s，无错误 |
| 界面转录 | 「你：你好，我在测试文字输入」/「格雷丝：嗯，行，打字也行…」 |
| 8 轮「打断→重说」压测 | 8/8 通过，零引擎重置 |
| `GUI_HEADLESS` | PASS |

---

### 5.7 预热完成提示

- 预热期间：主界面输入框/「按住说话」禁用，灰字显示「正在预热模型（约 30 秒），完成后自动恢复输入…」
- 预热完成：**托盘气泡「预热完成 · 可以按住热键说话了」**（4 秒）+ 窗口灰字变成
  `预热完成：ASR x.xs · TTS 加载 y.ys · 试合成 z.zs`，输入框与按钮恢复可用
- 预热失败：托盘气泡「预热失败」+ 具体原因（6 秒）
- 开机自启（`--tray-only`）没有窗口时，也靠这个气泡告知

---

## 6. 已知问题

1. **5070 Ti 未实机跑过**：只做了门禁与 sm_120 逻辑验证。真实安装后的验收按你的计划在那台机器完成。
2. **包内已带 ASR 缓存**（ModelScope 2.06GB），换机首次运行**不需要联网**；
   只有 Whisper 降级路径（large-v3，2.88GB）没带，见 §3 第 0 条。
3. **包内没带 LLM**：llama.cpp 与 GGUF 由用户自备，默认连 `127.0.0.1:8082`。
4. **安装器未做代码签名**：Windows SmartScreen 会提示“未知发布者”。
5. **X 光片：`dist\` 不进 git**（9.5GB），仓库里只保留 `installer.iss` 与打包脚本。

---

## 7. 在 5070 Ti 上的验收步骤（建议）

```powershell
# 1. 安装
情感陪伴器_v1_Setup.exe        # 装到 %ProgramFiles%\情感陪伴器_v1

# 2. 启动本地 LLM（或先不启，验回退）
D:\llama.cpp\llama-server.exe -m <Qwen3.5-9B>.gguf -ngl 999 -c 8192 --flash-attn on `
    --parallel 1 --reasoning-budget 0 --host 127.0.0.1 --port 8082

# 3. 自检（最直观）
自检.bat                        # 期望最后一行 SELFTEST PASS

# 4. 手动
启动.bat                        # 首启向导 → 托盘 → 按住 shift+a 说话
```

判据：`SELFTEST PASS` + 真实说一句话能听到克隆音色回复 + 回复中按住热键能立刻打断。
