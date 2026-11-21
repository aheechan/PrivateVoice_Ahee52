# 升级文档 — PrivateVoice_Ahee52 1.0.5 → 1.0.6

**适用对象**：已安装 1.0.5（或更早 1.0.x）的用户
**升级方式**：覆盖式增量包，**不需要重装 11.5 GB，不需要重新下载模型**
**升级包**：`PrivateVoice_Ahee52-1.0.6-update.zip`（1.24 MB）
**SHA256**：`9302ebcd9bd24cf26122cba34d2a3bc9d80d95af3960e680f4f8dd6e131b55e9`

---

## 一、这个版本改了什么

| # | 之前 | 1.0.6 |
|---|---|---|
| 1 | 普通聊天也常冒「稍等，让我查一下」 | 这句**只在联网检索时**出现；普通对话不再预设填充语，只有首 token 超过 900 ms 时才偶尔「嗯……」，多数时候安静等待 |
| 2 | 托盘图标是旧的圆点 | 换成品牌紫色对话框图标，右下角叠加状态色点（待机/听着/思考/说话/训练） |
| 3 | 切换 ASR 模型要重启程序 | 改为**后台热加载 + 预热**，托盘显示「正在加载…」→「可以开始对话」，不用重启 |
| 4 | 回复普遍过于谄媚 | 收紧情绪提示词，并新增表达底线：不奉承、不逐句夸、不复述原话、允许不同意见 |

### 你可以立刻感受到的变化

- 问天气、问新闻 → 会先说一句「稍等，让我查一下」再检索
- 普通闲聊 → 开头更安静，不再每次都接一句客套话
- 设置里换 whisper / paraformer → 几秒后就能直接说话，不用重启
- 回答语气更平实，少一些「你真的很棒」这类奉承

---

## 二、升级前准备（约 2 分钟）

1. **退出程序**：托盘图标右键 → 退出
2. 确认任务管理器里没有残留：
   ```
   任务管理器 → 详细信息 → 查找 pythonw.exe
   有就结束任务
   ```
3. **备份角色数据**（可选但建议）：
   ```
   把 安装目录\characters\  整个复制一份
   ```
4. 确认安装目录，默认是：
   ```
   C:\Program Files\PrivateVoice_Ahee52
   或你自定义的 D:\PrivateVoice_Ahee52
   ```

---

## 三、升级步骤

1. 下载 `PrivateVoice_Ahee52-1.0.6-update.zip`
2. 解压到一个临时文件夹（例如桌面新建 `update`）
3. 打开解压后的目录，应该看到：
   ```
   app\  assets\  tools\  runtime\  README.txt  app.ico
   ```
4. 全选这 6 项 → 复制
5. 粘贴到安装目录（`D:\PrivateVoice_Ahee52\`）
6. 弹出「目标包含同名文件」时，选 **替换目标中的文件**
   > 资源管理器会合并目录，不会删除模型和角色
7. 重新启动程序（桌面快捷方式 / `Launch.bat`）

### 升级不会动的东西

```
models_cache\        模型缓存（约 1.5 GB）
GPT-SoVITS\          语音合成模型与权重
characters\          你的角色、训练结果、参考音频
config.yaml          你的设置（LLM 地址、热键、设备）
logs\                历史日志
```

> 唯一会变的 `runtime\` 只是补一个 `huggingface_hub` 补丁（修复训练报错），不影响模型。

---

## 四、升级后验收清单

按顺序试一遍，全部通过即升级成功：

```
[ ] 程序能正常启动，托盘出现紫色对话框图标
[ ] 主界面标题栏图标是新的
[ ] 说一句话 → 有回复、有声音
[ ] 问一句「今天有什么新闻」→ 先听到「稍等，让我查一下」再回答
[ ] 普通闲聊 → 开头没有固定的「让我查一下」
[ ] 设置 → 语音识别 → 切到 whisper → 保存
      → 托盘提示「正在加载识别模型…」→「可以开始对话」
      → 直接说英文，不需要重启程序
[ ] 看 logs\app.log 出现「ASR 热加载完成」
```

自检脚本（可选）：

```
安装目录\SelfTest.bat
```

结果写入 `logs\selftest.json`。

---

## 五、常见问题

| 现象 | 处理 |
|---|---|
| 覆盖时提示文件被占用 | 程序没退干净，结束所有 `pythonw.exe` 后重试 |
| 启动后还是旧图标 | Windows 图标缓存：执行 `ie4uinit.exe -show`，或重启资源管理器 |
| 切换 ASR 后没反应 | 看托盘是否提示「加载识别模型…」；若 30 秒仍无提示，看 `logs\app.log` 的 `ASR 热加载失败` |
| 切换后第一次识别很慢 | 正常，正在预热模型（whisper 约 3–5 秒） |
| 训练报 `HfFolder` 错误 | 说明 `runtime` 目录没覆盖成功，重新复制 `runtime\` |
| 普通对话还是总说「让我查一下」 | 旧 `app\` 没覆盖成功，检查安装目录 `app\core\pipeline.py` 修改时间 |
| 想回滚 | 用 1.0.5 的 update zip 同样方式覆盖回去即可，角色数据不受影响 |

---

## 六、校验下载

```powershell
Get-FileHash .\PrivateVoice_Ahee52-1.0.6-update.zip -Algorithm SHA256
```

应得到：

```
9302ebcd9bd24cf26122cba34d2a3bc9d80d95af3960e680f4f8dd6e131b55e9
```

---

## 七、完整安装（新用户 / 想重装）

如果你不是升级而是全新安装，请下载 8 个文件：

```
PrivateVoice_Ahee52-1.0.6-Setup.exe
PrivateVoice_Ahee52-1.0.6-Setup-1.bin ~ -7.bin
```

放在同一文件夹，双击 `.exe` 安装。详见 `docs/INSTALL.md`。

| 文件 | SHA256 |
|---|---|
| Setup.exe | `7edb480f20d11d6b63a14d49f1b5b1a3b3937935c5f1d01edf703b83f36aa4cd` |
| -1.bin | `84e1892f0370b527dc7cb0571deaf4f61476577bc14ea9cf21caaeedfceac9cd` |
| -2.bin | `8c68bea12937b1f6272e7bf0dd00c9b8bb38842a124c71fad8ff9bdc3d33b6be` |
| -3.bin | `9e7e6228374144d63fa2c5e0af733a6aadde3c38d5691534dd706f00a23f6133` |
| -4.bin | `ed67443d43d8e8ba238986afd52534e45153d0806deb43a3b8bbc6164d34746b` |
| -5.bin | `69924596f89f5368f80bff4cfa8d31c386093d12aa474991a4b873cd970e2578` |
| -6.bin | `8b5ff0d55fac9f718ffbc9a893c400d1ca5ef1a30f62affa8b436154ce1e15de` |
| -7.bin | `6da94a91c55b3c3cb82adb54ba0bef8d0f396ee6f9197dbfd34636aea7d4b404` |

---

## English summary

Upgrading from 1.0.5 to 1.0.6 is an **in-place file overlay** — no reinstall, no model download.

1. Quit the app (check Task Manager for `pythonw.exe`).
2. Back up `characters\` (optional).
3. Extract `PrivateVoice_Ahee52-1.0.6-update.zip` and copy `app\ assets\ tools\ runtime\ README.txt app.ico`
   into the install folder, choosing **Replace files in the destination**.
4. Start the app again.

What's new: “let me check” only for web lookups; a short “mm…” only when the first token is slow;
brand chat-wave tray icon; ASR switches reload and warm up in the background (no restart);
less sycophantic replies.

Verify: tray icon is the purple chat bubble, normal chat no longer opens with “let me check”,
switching ASR shows “loading…” then “ready to talk” and English recognition works immediately.

Models, characters and `config.yaml` are never touched.
