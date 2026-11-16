# 安装注意事项 / Installation notes

适用于 **PrivateVoice_Ahee52 1.0.5** 整合包（1 个安装器 + 7 个分卷 + 1 个增量包）。

---

## 一、下载

| 文件 | 大小 | 说明 |
|---|---|---|
| `PrivateVoice_Ahee52-1.0.5-Setup.exe` | 6.6 MB | 安装器，双击这个 |
| `PrivateVoice_Ahee52-1.0.5-Setup-1.bin` ~ `-7.bin` | 各 ≤ 1.77 GB | 分卷，必须一起下载 |
| `PrivateVoice_Ahee52-1.0.5-update.zip` | 1.23 MB | 已装 1.0.4 的用户用这个 |

- **8 个安装文件必须放在同一个文件夹**，`.exe` 会在同目录查找 `.bin`，缺一个都会中断
- **不要改名**：文件名参与分卷校验，改名后安装器找不到
- **不要用下载工具的多线程合并**：请逐个下载，避免分卷损坏
- 下载完成后建议核对 SHA256（见文末）

## 二、安装前

1. **完全退出旧版**：托盘图标右键 → 退出；确认任务管理器里没有 `pythonw.exe`
   > 程序运行时相关文件被占用，安装器会报“文件被占用”或覆盖失败
2. **备份角色数据**：把 `D:\PrivateVoice_Ahee52\characters\` 整个复制一份
   > 覆盖安装不会删除角色，但卸载会
3. **预留磁盘空间**：安装约 **12 GB**，加上模型缓存与训练中间产物建议留 **20 GB**
4. **以管理员身份运行**（安装器默认请求提权），杀毒软件如拦截请放行
5. **安装路径用纯英文**，例如 `D:\PrivateVoice_Ahee52`
   > 含中文或空格的路径可能导致部分 CUDA / 音频组件加载失败

## 三、安装过程

- 双击 `Setup.exe` → 若出现「已保护你的电脑」，点 **更多信息 → 仍要运行**
  （安装器未做代码签名，SmartScreen 会默认拦截）
- 安装向导是英文的，一路 **Next** 即可，路径可在第一步修改
- 安装器会自动刷新 Windows 图标缓存；若桌面图标仍是旧的，手动执行：
  ```
  ie4uinit.exe -show
  ```
  或在任务管理器里重启「Windows 资源管理器」

## 四、首次启动

1. **选择语言**：中文 / English（之后可在主界面或托盘切换）
2. **显卡自检**：需要 NVIDIA 显卡与可用驱动
3. **模型检查**：整合包已内置全部模型，无需下载
4. **配置对话模型**：填任意 OpenAI 兼容接口，例如本地 llama.cpp / Ollama / vLLM
   ```
   Base URL: http://127.0.0.1:8080/v1
   模型名:   Qwen3.5-9B
   API Key:  本地服务通常留空
   ```
   可先点「测试连接」，通过后点「开始使用」
5. **授权麦克风权限**（Windows 设置 → 隐私 → 麦克风）
6. 按住 `shift+a` 说话，或在主界面输入文字

## 五、硬件与系统

| 项目 | 最低 | 推荐 |
|---|---|---|
| 系统 | Windows 10 x64（21H2+） | Windows 11 x64 |
| 显卡 | NVIDIA RTX 20 系及以上、8 GB+ 显存（更小显存未测试） | RTX 30 系及以上、8 GB+（已验证流畅） |
| 驱动 | 支持 CUDA 12.8（≥ 570.x） | 最新 Game Ready / Studio |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 15 GB 空闲 | 30 GB |

- 推理在 8 GB 显存以上比较从容；训练建议 16 GB 以上
- **不支持纯 CPU 推理**（每句 30–60 秒，且 v4 依赖 GPU 算子）
- 集成显卡 / AMD 显卡 / macOS / Linux 均未验证

## 六、从 1.0.4 升级

用增量包即可，不必重装 11 GB：

1. 退出程序
2. 解压 `PrivateVoice_Ahee52-1.0.5-update.zip`
3. 把里面的 `app`、`assets`、`tools`、`runtime`、`README.txt`、`app.ico`
   覆盖到安装目录（资源管理器会合并目录，选择“替换目标中的文件”）
4. 重新启动

**不会改动**：`models_cache\`、`GPT-SoVITS\`、`characters\`、`config.yaml`

## 七、安装后检查清单

```
[ ] 桌面快捷方式图标是新版 chat-wave
[ ] 启动后托盘出现图标，右键菜单可用
[ ] 设置 → 录音设备 能看到麦克风
[ ] 按住 shift+a 有电平反应
[ ] 第一句话能听到声音
[ ] logs\app.log 中出现「热键 shift+A 已注册」
```

自检脚本（英文输出）：

```
D:\PrivateVoice_Ahee52\SelfTest.bat
```

结果写入 `logs\selftest.json`。

## 八、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 双击没反应、日志停在“硬件检查通过” | 显卡驱动或运行库问题；看 `logs\app.log`，现在崩溃会写堆栈并弹窗 |
| 提示“已经在运行” | 单实例程序，检查右下角托盘（可能被折叠进 `^`） |
| 安装到一半报文件被占用 | 旧版程序没退出，结束 `pythonw.exe` 后重试 |
| 没有声音 | 设置 → 播放设备 选对输出；确认角色 `checkpoints\` 下有两个权重文件 |
| 切换语音识别模型后识别失败 | 1.0.5 已改为自动重启（约 1.2 秒）；若未重启，手动退出再开 |
| 英文识别没有空格/标点 | 1.0.5 已修复；确认用的是 whisper 模型 |
| 训练报显存不足 | 训练前程序会卸载语音模型；仍失败换更干净的素材或「保守」档 |
| 训练报 `HfFolder` 导入错误 | `huggingface_hub` 版本过新，1.0.5 运行时已固定 0.26.2；确认覆盖了 `runtime` 目录 |
| 杀软报毒 / 拦截 | 未签名的本地程序常见误报；将安装目录加入白名单 |
| 桌面图标还是旧的 | 执行 `ie4uinit.exe -show` 或重启资源管理器；仍不行则删除图标缓存后重建快捷方式 |

## 九、数据与隐私

- 音频、识别、合成全部在本机完成，**不上传任何音频**
- 对话模型接口由你自己填写；如果填的是云端地址，对话文本会发到该地址
- 联网检索（Bing / 百度 / 维基百科）默认关闭，开启后仅发送检索关键词
- API Key 使用 Windows DPAPI 加密后存入 `config.yaml`，界面只显示尾 4 位
- 训练产生的音色权重保存在 `characters\<角色>\checkpoints\`，属于敏感数据，请自行保管

## 十、卸载与回滚

- 卸载：设置 → 应用 → `PrivateVoice_Ahee52` → 卸载
- 卸载会删除程序与内置模型，**不会**保留 `characters\`，请先备份
- 回滚到 1.0.4：重新运行 1.0.4 安装包覆盖安装即可，角色数据不受影响

## 十一、声音授权

训练音色前请确认：这是**你本人的声音**，或你已取得对方的**书面授权**（含用途、范围、期限，并允许随时撤回）。

禁止用于冒充他人、欺诈、骚扰、色情或误导性内容。

## 十二、SHA256 校验

```
5d7864eb6615b3f106fe036e6edc0c0621b498543c1bc16098b038bd5e40ed06  Setup.exe
479543827342ff1c585390f9ff9a543f2ec6adfa288afe272a2d97f3c2acd580  Setup-1.bin
0c2c211583d96b8b73c029e8bd00e3da00a94d70ecf6c41ae827123cbe7cb937  Setup-2.bin
91e6a28824e76517bde975bb3158b302d1eba1eef25c6bc75f12e8759d6b708d  Setup-3.bin
43e5c03980abf1527c43a11e08e06ccf2a87cf4ee8f04270852d42ff6463c5e0  Setup-4.bin
267607db33cf9465b795166f6ec2cb975c3576ddf70412a9f4034ce0138166d3  Setup-5.bin
773cb578cc1c5b30eae31c310b63e14e31b6c892beacb2a57620a6f7f1a9e9da  Setup-6.bin
30f92f6670679b783a9244229c8fd18accd2ab41420d295c55b996aff94b1269  Setup-7.bin
```

Windows 校验命令：

```powershell
cd 安装包所在目录
Get-FileHash .\PrivateVoice_Ahee52-1.0.5-Setup.exe -Algorithm SHA256
```

## 十三、遇到问题怎么反馈

1. 打开 `安装目录\logs\app.log`，取最后 30 行
2. 附上 Windows 版本、显卡型号、显存
3. 说明当时在做什么（首次启动 / 切换 ASR / 训练 / 播放）
4. 发到 Issues，标题写清现象

---

## English summary

- Download all 8 installer files into **one folder**; do not rename the `.bin` slices.
- The installer is unsigned: SmartScreen → **More info → Run anyway**.
- Quit the running app first, back up `characters\`, keep the install path ASCII.
- Requires Windows 10/11 x64, NVIDIA RTX 20-series or newer with **8 GB+ VRAM**
  (30-series and above with 8 GB+ verified smooth). CPU-only inference is not supported.
- Models are bundled; on first run pick a language, run the GPU check and point the app at any
  OpenAI-compatible LLM endpoint (llama.cpp, Ollama, vLLM…).
- Upgrading from 1.0.4: extract the update zip over the install folder — models, characters and
  `config.yaml` are untouched.
- Audio never leaves the machine. Voice cloning requires consent from the voice owner.
