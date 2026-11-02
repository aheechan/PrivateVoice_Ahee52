# 本轮修复记录（v2.1）

> 2026-11-02，来自 5070 Ti 实机反馈

## 1. 训练失败：「文本/音素提取失败（exit=1）」

**日志证据**（`train_emily-e12c3f.log`）：

```
File "...\prepare_datasets\1-get-text.py", line 20
    from text.cleaner import clean_text
ModuleNotFoundError: No module named 'text'
```

**根因**：分发包的 `runtime\python312._pth` 存在时，**Python 会完全忽略
`PYTHONPATH`**。训练器原本靠 `_env()` 设置 `PYTHONPATH=repo;repo\GPT_SoVITS`，
在开发 venv 里有效，在打包环境里无效 → 上游脚本里的 `from text.cleaner ...` 找不到。

**修复**：
- 新增 `tools/run_upstream.py`：运行时把 `repo` 与 `repo\GPT_SoVITS` 插进 `sys.path`，
  再 `runpy.run_path()` 执行上游脚本；`TrainingRunner._wrap_upstream()` 自动把
  `python -s 上游脚本.py` 改写成经它启动。
- 顺带发现包里**根本没带 `tools/s1shim`**（GPT 训练垫片），所以第二个错是
  `prefetch_factor ... num_workers > 0`。已补进包，并通过环境变量 `VOICELAB_SHIM`
  让 `run_upstream` 显式加载 `sitecustomize`（同样是因为 `._pth` 让 PYTHONPATH 失效）。

**验证**（在分发包里真跑一遍完整训练）：

```
[prep_text] → [prep_hubert] → [prep_semantic] → [sovits] ep 8/8 → [gpt] ep 7/7
DONE in 3.8min
voice: SoVITS_weights_v4/vlab_测试_e8_s200_l32.pth + GPT_weights_v3/vlab_测试-e8.ckpt
```

## 2. 首启向导误报「缺少 s2Gv3.pth」，且点退出仍启动主程序

- 检查清单还是 v3 的：`s1v3.ckpt / s2Gv3.pth / hubert / roberta / G2PWModel`。
  包内已删 s2Gv3（只留 v4），必然误报。
- 现在检查：`s1v3.ckpt`、`gsv-v4-pretrained/s2Gv4.pth`、`gsv-v4-pretrained/vocoder.pth`、
  `fast_langdetect/lid.176.bin`、`chinese-hubert-base`、`chinese-roberta-wwm-ext-large`、`G2PWModel`。
- 点「退出」现在会真正退出（`controller.shutdown()` + `app.quit()`），不再留托盘和主窗口。
- **确认**：缺 `s2Gv3.pth` 不影响运行——v4 链路只用 `s2Gv4.pth` + `vocoder.pth`；
  实测删掉 s2Gv3 后自检与对话都正常。

附带修掉一个隐藏坑：包里的 `config.yaml` 写死了 `training.s2_version: v3`，
覆盖了代码默认值，导致训练去找 `s2Gv3.pth`。现在：
- 分发包不再带 `config.yaml`；
- `Config._heal_training()` 在「有 v4 底模、没有 v3 底模」且配置写着 v3 时自动切 v4。

## 3. 「清理缓存」→「释放模型 / 加载模型」

- 主界面按钮改为 **释放模型** ⇄ **加载模型**：卸载 ASR + TTS，`torch.cuda.empty_cache()`，
  状态栏显示释放后可用显存；再点一次重新加载（约 18-22s）。
- 实测（3090）：释放后可用显存 20.76 → 22.69 GB（+1.93 GB），重新加载 18.2s 后可正常合成。
- 本地 LLM（llama.cpp）是**独立进程**，程序内只能提示用其启动脚本停止，无法直接卸载。
- 训练完成弹窗补了一句：若训练前释放过模型，结束后请重启或点「加载模型」。
- `housekeeping.py` 保留（日志/中间产物清理逻辑还在），只是不再挂主界面按钮。

## 4. 语气词开头的「卡顿延音」

**结论：是生成侧，不是播放缓冲。**

诊断（`tools/probe_um.py`、`probe_um_params.py`）：v4 在 `top_k=5` 这种贪婪采样下，
遇到「嗯 / 哦 / 唔」会把元音拖长，甚至整句退化：

| 文本 | top_k=5 | top_k=15 / T=0.7 |
|---|---|---|
| 唔，那就早点睡。 | **11.47s**（1.43s/字） | **2.18s**（0.27s/字） |
| 嗯。 | 1.78s（0.89s/字） | 1.48s（0.74s/字） |
| 哦，这样啊。 | 1.62s | 1.02s |

同时排除了播放链路：起始 300ms 内没有"恒定平台"（zero-order hold 只有个位数样点），
说明不是缓冲区欠载造成的重复。

**修复**：
- 默认采样改为 `top_k=15`、`top_p=0.95`、`temperature=0.7`；
- 加异常时长保护：单句音频超过 `0.55s/字 + 1s` 时截断并记日志，避免一句话卡 10 秒。

## 5. 未验证/风险

- 5070 Ti + 真实麦克风仍未由我实测（本机无输入设备）。
- 释放模型后如果立刻按热键，会走"重新加载+等待"路径，首轮会慢。
- `GPT_weights_v3` 目录名仍带 v3（实际是 v4 角色用的 GPT 权重），未改名。
