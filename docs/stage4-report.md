# Stage 4 报告：打包（嵌入式 Python + 安装器 + 首启向导）

> 完成日期：2026-11-02
> 产物：`C:\VoiceLab\dist\情感陪伴器_v1\`（免安装版，9.45 GB / 49,841 文件）
> 安装器：`C:\VoiceLab\dist\installer\情感陪伴器_v1_Setup.exe`
> 实机验证：3090 上完成理论运行测试（`SELFTEST PASS`）
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

## 4. 已知问题

1. **5070 Ti 未实机跑过**：只做了门禁与 sm_120 逻辑验证。真实安装后的验收按你的计划在那台机器完成。
2. **包内没带 ASR/标点模型缓存**：funasr 的 `paraformer-zh`、`ct-punc`、campplus 仍在
   `%USERPROFILE%\.cache\modelscope`。**换机首次运行需要联网下载约 1.1GB**；
   若目标机离线，需要把该缓存目录一并拷过去（可后续加进安装器）。
3. **包内没带 LLM**：llama.cpp 与 GGUF 由用户自备，默认连 `127.0.0.1:8082`。
4. **安装器未做代码签名**：Windows SmartScreen 会提示“未知发布者”。
5. **X 光片：`dist\` 不进 git**（9.5GB），仓库里只保留 `installer.iss` 与打包脚本。

---

## 5. 在 5070 Ti 上的验收步骤（建议）

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
