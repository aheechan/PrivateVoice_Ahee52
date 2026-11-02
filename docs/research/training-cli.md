# GPT-SoVITS v3 微调训练命令行考证

- 考证对象：`C:\VoiceLab\GPT-SoVITS`，commit `08d627c3338173c3229286d8787060d6559fe0f8`
- Python：`C:\VoiceLab\.venv\Scripts\python.exe`（torch 2.11.0+cu128，funasr 1.0.27，peft 可用）
- 预训练：`GPT_SoVITS/pretrained_models/s2Gv3.pth`（769,035,145 B）、`s1v3.ckpt`、`chinese-hubert-base\`、`chinese-roberta-wwm-ext-large\`
- 结论均以本仓库源码为准；凡与网上 v2 教程冲突处，以本文引用的行号为准。

---

## 0. 先纠正四个会直接导致失败的前提

| 常见说法 | 本 commit 的真实情况 | 源码证据 |
|---|---|---|
| v3 用 `s2_train.py` 训练 | **v3 走 `GPT_SoVITS/s2_train_v3_lora.py`**（LoRA 训练）。`s2_train.py` 只服务 v1/v2/v2Pro/v2ProPlus | `webui.py:541-544` |
| v3 必须跑 `2-get-sv.py` | **不需要**。只有版本名里含 `"Pro"` 才跑；v3 的 loader 完全不读 `7-sv_cn` | `webui.py:1151-1167`、`webui.py:910-926`、`module/data_utils.py:32-35,120-121`、`module/data_utils.py:288-306` |
| 会有 `train.list` / `val.list` | **本 commit 没有这两个文件**。训练只读 `2-name2text.txt`、`4-cnhubert/`、`5-wav32k/`、`6-name2semantic.tsv`、可选 `3-bert/` | 全仓库 grep；`module/data_utils.py:26-35,288-306`、`AR/data/dataset.py:63-82` |
| s2 的 epoch 用 `-e` 传 | **`s2_train*.py` 没有 `-e`**，只认 `-c/--config`、`-p/--pretrain`、`-rs/--resume_step`；epoch 写在 JSON 的 `train.epochs` | `utils.py:189-220`；实测 `--help` 输出 |
| `7-sv_cn.npy` 是单个文件 | 实际是目录 `7-sv_cn/`，里面是 `<wav名>.pt`，每个 20480 维 | `prepare_datasets/2-get-sv.py:42,88-96` |

> 本 commit 的 WebUI 版本单选框里没有 `v3`（只有 v1/v2/v4/v2Pro/v2ProPlus，`webui.py:1495-1500`），但 v3 的代码路径、预训练权重和 `s2_train_v3_lora.py` 都在。用本文的命令行可以直接按 v3 跑。

---

## 1. WebUI 里真实拼出的命令字符串

`python_exec` 等于 `sys.executable`（`config.py:138`），在本环境就是 `C:\VoiceLab\.venv\Scripts\python.exe`。WebUI 的工作目录是仓库根 `C:\VoiceLab\GPT-SoVITS`（`webui.py:5-6`）。

### 1.1 数据准备三段

WebUI **不传任何命令行参数**，全部通过环境变量驱动：

```text
"<python_exec>" -s GPT_SoVITS/prepare_datasets/1-get-text.py
"<python_exec>" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
"<python_exec>" -s GPT_SoVITS/prepare_datasets/2-get-sv.py          # 仅 Pro 版本
"<python_exec>" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
```

对应源码：

- `webui.py:808`（1a）
- `webui.py:898`（1b）
- `webui.py:920`（1b 的 Pro 分支，`if "Pro" in version` 在 `webui.py:910`）
- `webui.py:992`（1c）
- 一键三连版本：`webui.py:1092`、`1139`、`1161`、`1201`

环境变量在 `os.environ.update(config)` 之后由子进程继承：

- 1A：`inp_text`、`inp_wav_dir`、`exp_name`、`opt_dir`、`bert_pretrained_dir`、`i_part`、`all_parts`、`_CUDA_VISIBLE_DEVICES`、`is_half`（`webui.py:789-807`）
- 1B：`inp_text`、`inp_wav_dir`、`exp_name`、`opt_dir`、`cnhubert_base_dir`、`sv_path`、`i_part`、`all_parts`、`_CUDA_VISIBLE_DEVICES`（`webui.py:878-897`）
- 1C：`inp_text`、`exp_name`、`opt_dir`、`pretrained_s2G`、`s2config_path`、`i_part`、`all_parts`、`_CUDA_VISIBLE_DEVICES`（`webui.py:973-991`）

每个分片 `i_part` 从 0 到 `all_parts-1`，脚本内部用 `lines[i_part::all_parts]` 取模切分（`1-get-text.py:127`、`2-get-hubert-wav32k.py:111`、`3-get-semantic.py:106`）。

### 1.2 SoVITS（s2）

```text
# v1 / v2 / v2Pro / v2ProPlus
"<python_exec>" -s GPT_SoVITS/s2_train.py --config "<TEMP>/tmp_s2.json"

# v3 / v4
"<python_exec>" -s GPT_SoVITS/s2_train_v3_lora.py --config "<TEMP>/tmp_s2.json"
```

源码：`webui.py:541-544`；`tmp_config_path = "%s/tmp_s2.json" % tmp` 在 `webui.py:538`，`tmp` 是仓库根下的 `TEMP`（`webui.py:21-22`）。

### 1.3 GPT（s1）

```text
"<python_exec>" -s GPT_SoVITS/s1_train.py --config_file "<TEMP>/tmp_s1.yaml"
```

源码：`webui.py:636`；临时 YAML 路径在 `webui.py:632`。

> 这三条训练命令都只有 `--config` / `--config_file`，**没有** `-e`、`--exp_name`、`-i`、`-x`、`--model_dir`、`batch_size` 等参数。网上 v2 教程里的那些参数在本 commit 的训练脚本里不存在。

---

## 2. 启动前提（比参数更容易踩）

### 2.1 工作目录必须是仓库根

所有路径都是相对 cwd 的：

- 预训练：`GPT_SoVITS/pretrained_models/s2Gv3.pth`（`config.py:15`、`webui.py:507-526`）
- 实验目录：`logs/<exp>`（`config.py:137`、`webui.py:515`）
- 最终权重目录：`SoVITS_weights_v3`、`GPT_weights_v3`（`config.py:60-75`）
- 临时配置：`TEMP/tmp_s2.json`（`webui.py:21,538`）

所以启动前：

```powershell
Set-Location C:\VoiceLab\GPT-SoVITS
```

不要 `cd GPT_SoVITS` 再跑；那样 `GPT_SoVITS/pretrained_models/...` 会变成 `GPT_SoVITS/GPT_SoVITS/...`，v3 训练脚本还会**静默地**从随机初始化开始（见 §8 第 4 条）。

### 2.2 必须给子进程设置 PYTHONPATH

数据准备脚本按“仓库根 + `GPT_SoVITS`”两处找包：

- `1-get-text.py` 需要 `text`（在 `GPT_SoVITS/text`）和 `tools`（在仓库根）
- `2-get-hubert-wav32k.py` 需要 `feature_extractor`（在 `GPT_SoVITS/feature_extractor`）和 `tools`
- `3-get-semantic.py` 需要 `utils`（在 `GPT_SoVITS`）和 `tools`
- 训练脚本需要 `utils/module/text/AR`（`GPT_SoVITS`）和 `tools`（仓库根）

脚本自身只把 `sys.path[0]` 设为**脚本所在目录**（`prepare_datasets`），不会自动加入仓库根，也不会自动加入 `GPT_SoVITS`。实测：

```text
# cwd=仓库根，不设 PYTHONPATH
python -s GPT_SoVITS\prepare_datasets\1-get-text.py --help
→ ModuleNotFoundError: No module named 'text'

# PYTHONPATH 只加 GPT_SoVITS
→ ModuleNotFoundError: No module named 'tools'

# PYTHONPATH 同时加仓库根和 GPT_SoVITS
→ 导入通过（随后因缺环境变量退出）
```

因此统一设置：

```powershell
$env:PYTHONPATH = "C:\VoiceLab\GPT-SoVITS;C:\VoiceLab\GPT-SoVITS\GPT_SoVITS"
```

> `webui.py` 自己在进程内 `sys.path.insert(0, now_dir)`（`webui.py:6`），但这个修改**不会**传给 `Popen` 子进程；子进程只继承环境变量。若从已设置 `PYTHONPATH` 的 shell 启动 WebUI，子进程才会正常。这是本机 venv 下实测的结论。

### 2.3 预先创建权重目录

`SoVITS_weights_v3/` 和 `GPT_weights_v3/` 不会由训练脚本创建；WebUI 是在启动时统一建的（`webui.py:200-201`）。直接跑命令行必须先建：

```powershell
New-Item -ItemType Directory -Force C:\VoiceLab\GPT-SoVITS\SoVITS_weights_v3, C:\VoiceLab\GPT-SoVITS\GPT_weights_v3 | Out-Null
```

`savee()` 的落盘在 `try/except` 里（`process_ckpt.py:41-60`），目录不存在时只会把 traceback 写进日志，训练本身继续跑，最后“看起来训练完了但没有权重文件”。`s1` 的 `my_save` 则会在 `shutil.move` 处直接抛异常。

---

## 3. 每个步骤的输入、命令、输出

### 3.1 1A：文本 → 音素 + BERT 特征

**输入**：`dataset.list`（格式见 §5）+ 已切好的 wav 目录。

**环境变量与命令**：

```powershell
$repo = "C:\VoiceLab\GPT-SoVITS"
$py   = "C:\VoiceLab\.venv\Scripts\python.exe"
$exp  = "myvoice"

$env:PYTHONPATH           = "$repo;$repo\GPT_SoVITS"
$env:inp_text             = "C:\VoiceLab\data\myvoice\dataset.list"
$env:inp_wav_dir          = "C:\VoiceLab\data\myvoice\wav24k"
$env:exp_name             = $exp
$env:opt_dir              = "logs\$exp"
$env:bert_pretrained_dir  = "$repo\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"
$env:i_part               = "0"
$env:all_parts            = "1"
$env:_CUDA_VISIBLE_DEVICES = "0"
$env:is_half              = "True"
$env:version              = "v3"

Set-Location $repo
& $py -s GPT_SoVITS\prepare_datasets\1-get-text.py
```

**后置产物**：

- `logs\<exp>\2-name2text-0.txt`（分片文件）
- `logs\<exp>\3-bert\<wav名>.pt`（仅 `language == "zh"` 时生成，`1-get-text.py:94-98`）

**必须合并/改名**：训练只认 `logs\<exp>\2-name2text.txt`（`module/data_utils.py:26`、`AR/data/dataset.py:63`）。单分片直接改名：

```powershell
Move-Item "logs\$exp\2-name2text-0.txt" "logs\$exp\2-name2text.txt"
```

多分片时按 `webui.py:819-827` 的逻辑把各分片按行拼起来，去掉临时文件。注意不要用会写 BOM 的 `Set-Content`；BOM 会污染第一行的文件名。

**注意**：`line.split("|")` 没有 `maxsplit`（`1-get-text.py:129`），所以每行必须**恰好** 4 段，文本里不能出现 `|`。

### 3.2 1B：SSL/HuBERT 特征 + 32k wav

**前置**：1A 的 `2-name2text.txt` 不是本步必需；本步只读 `inp_text` 和 wav（`2-get-hubert-wav32k.py:108-123`）。但 WebUI 的顺序是先 1A 后 1B。

**环境变量与命令**：

```powershell
$env:inp_text            = "C:\VoiceLab\data\myvoice\dataset.list"
$env:inp_wav_dir         = "C:\VoiceLab\data\myvoice\wav24k"
$env:exp_name            = $exp
$env:opt_dir             = "logs\$exp"
$env:cnhubert_base_dir   = "$repo\GPT_SoVITS\pretrained_models\chinese-hubert-base"
$env:is_half             = "True"
$env:i_part              = "0"
$env:all_parts           = "1"
$env:_CUDA_VISIBLE_DEVICES = "0"

& $py -s GPT_SoVITS\prepare_datasets\2-get-hubert-wav32k.py
```

**后置产物**：

- `logs\<exp>\4-cnhubert\<wav名>.pt`（768 维 SSL 特征）
- `logs\<exp>\5-wav32k\<wav名>`（**32 kHz 单声道 wav 内容**）

**v3 不跑 `2-get-sv.py`**。只有 `v2Pro` / `v2ProPlus` 才跑：

```text
webui.py:1151   if "Pro" in version:
webui.py:1161       cmd = "... 2-get-sv.py"
```

`2-get-sv.py` 的产物是 `logs\<exp>\7-sv_cn\<wav名>.pt`（`2-get-sv.py:42,88-96`），它读的是 `5-wav32k`（`2-get-sv.py:91-93`，断言采样率 32000），所以如果真要跑 Pro，必须在 1B 之后。

### 3.3 1C：语义 token

**前置**：`4-cnhubert/` 必须已经存在；`3-get-semantic.py:90-93` 找不到 `<name>.pt` 就跳过该条。

**环境变量与命令**：

```powershell
$env:inp_text        = "C:\VoiceLab\data\myvoice\dataset.list"
$env:exp_name        = $exp
$env:opt_dir         = "logs\$exp"
$env:pretrained_s2G  = "$repo\GPT_SoVITS\pretrained_models\s2Gv3.pth"
$env:s2config_path   = "$repo\GPT_SoVITS\configs\s2.json"
$env:is_half         = "True"
$env:i_part          = "0"
$env:all_parts       = "1"
$env:_CUDA_VISIBLE_DEVICES = "0"

& $py -s GPT_SoVITS\prepare_datasets\3-get-semantic.py
```

**后置产物**：`logs\<exp>\6-name2semantic-0.tsv`，每行 `wav名<TAB>语义id空格串`（`3-get-semantic.py:100`）。

**必须加表头并合并**：训练用 `pandas.read_csv(..., delimiter="\t")`，默认第一行为表头（`AR/data/dataset.py:57-61`）；WebUI 合并时写入的表头固定为 `item_name\tsemantic_audio`（`webui.py:1003,1213`）。单分片：

```powershell
& $py -c "from pathlib import Path; d=Path('logs/$exp'); body=(d/'6-name2semantic-0.tsv').read_text('utf-8'); (d/'6-name2semantic.tsv').write_text('item_name\tsemantic_audio\n'+body.strip('\n')+'\n', encoding='utf-8'); (d/'6-name2semantic-0.tsv').unlink()"
```

**版本判定是“猜”的**：`3-get-semantic.py:18-28` 按 `pretrained_s2G` 文件大小判断 v1/v2/v3；`s2Gv3.pth` 为 769,035,145 B（>700 MiB），会被判成 v3。若误传 v2 权重，它会按 v2 建模型，语义 token 与 v3 训练不匹配。

### 3.4 2：SoVITS（v3 LoRA）

**前置**：`logs\<exp>\2-name2text.txt`、`4-cnhubert/`、`5-wav32k/`（`module/data_utils.py:288-293`）。不需要 `6-name2semantic.tsv`，也不需要 `3-bert/`。

**命令**：

```powershell
Set-Location $repo
& $py -s GPT_SoVITS\s2_train_v3_lora.py --config "C:\VoiceLab\work\myvoice\s2_v3.json"
```

配置模板见 §4.1。`--config` 指向**你生成的完整 JSON**；仓库里的 `GPT_SoVITS/configs/s2.json` 只是模板（WebUI 会读它再覆盖字段，写 `TEMP/tmp_s2.json`）。

**后置产物**：

- 训练状态：`logs\<exp>\logs_s2_v3_lora_<lora_rank>\G_*.pth`（`s2_train_v3_lora.py:142-143,344-360`）
- TensorBoard：`logs\<exp>\events.out.tfevents.*`、`logs\<exp>\train.log`（`s2_train_v3_lora.py:78-82`）
- 最终权重：`SoVITS_weights_v3\<exp>_e<epoch>_s<global_step>_l<lora_rank>.pth`（`s2_train_v3_lora.py:372-386`）

### 3.5 3：GPT（s1）

**前置**：`2-name2text.txt`、`6-name2semantic.tsv`；`3-bert/` 可选（缺失时 `bert_feature=None`，`AR/data/dataset.py:223-231`）。

**命令**：

```powershell
$env:_CUDA_VISIBLE_DEVICES = "0"
$env:hz = "25hz"
Set-Location $repo
& $py -s GPT_SoVITS\s1_train.py --config_file "C:\VoiceLab\work\myvoice\s1_v3.yaml"
```

配置模板见 §4.2。`_CUDA_VISIBLE_DEVICES` / `hz` 是 WebUI 在 `webui.py:630-631` 设的；`s1_train.py:4-5` 只把 `_CUDA_VISIBLE_DEVICES` 映射到 `CUDA_VISIBLE_DEVICES`。

**后置产物**：

- 训练状态：`logs\<exp>\logs_s1_v3\ckpt\*.ckpt`（`s1_train.py:88-92,106`）
- 最终权重：`GPT_weights_v3\<exp>-e<epoch>.ckpt`（`s1_train.py:73-81`）

---

## 4. 配置注入机制与模板

### 4.1 s2：JSON（v3 用 `s2.json` 模板，没有 `s2v3.json`）

WebUI 的注入过程（`webui.py:507-540`）：

```python
config_file = "GPT_SoVITS/configs/s2.json"       # v3 走这里
data["train"]["batch_size"]       = batch_size
data["train"]["epochs"]           = total_epoch
data["train"]["text_low_lr_rate"] = text_low_lr_rate
data["train"]["pretrained_s2G"]   = pretrained_s2G
data["train"]["pretrained_s2D"]   = pretrained_s2D
data["train"]["if_save_latest"]   = if_save_latest
data["train"]["if_save_every_weights"] = if_save_every_weights
data["train"]["save_every_epoch"] = save_every_epoch
data["train"]["gpu_numbers"]      = gpu_numbers1Ba
data["train"]["grad_ckpt"]        = if_grad_ckpt
data["train"]["lora_rank"]        = lora_rank
data["model"]["version"]          = version
data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir   # logs/<exp>
data["save_weight_dir"] = SoVITS_weight_version2root[version]  # SoVITS_weights_v3
data["name"] = exp_name
data["version"] = version
```

注意：`s2_train_v3_lora.py` **不使用** `pretrained_s2D`、`text_low_lr_rate`，也没有判别器；v3 只需要 `pretrained_s2G`。本机也没有 `s2Dv3.pth`，WebUI 的预训练检查专门跳过了它（`webui.py:177`）。

可直接用的最小 JSON（`C:\VoiceLab\work\myvoice\s2_v3.json`）：

```json
{
  "train": {
    "log_interval": 100,
    "eval_interval": 500,
    "seed": 1234,
    "epochs": 2,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "batch_size": 4,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 20480,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "text_low_lr_rate": 0.4,
    "grad_ckpt": false,
    "pretrained_s2G": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    "if_save_latest": true,
    "if_save_every_weights": true,
    "save_every_epoch": 1,
    "gpu_numbers": "0",
    "lora_rank": 32
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 32000,
    "filter_length": 2048,
    "hop_length": 640,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 300,
    "cleaned_text": true,
    "exp_dir": "logs/myvoice"
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 512,
    "semantic_frame_rate": "25hz",
    "freeze_quantizer": true,
    "version": "v3"
  },
  "s2_ckpt_dir": "logs/myvoice",
  "save_weight_dir": "SoVITS_weights_v3",
  "name": "myvoice",
  "version": "v3"
}
```

字段含义（只列训练脚本真正读的）：

| 字段 | 作用 | 源码 |
|---|---|---|
| `train.epochs` | 总 epoch（v3 默认 UI 是 2） | `s2_train_v3_lora.py:229` |
| `train.batch_size` | 每卡 batch | `s2_train_v3_lora.py:99-101` |
| `train.save_every_epoch` | 每 N epoch 存一次训练状态/权重 | `s2_train_v3_lora.py:344` |
| `train.if_save_latest` | true 时训练状态固定写 `G_233333333333.pth`，否则写 `G_<step>.pth` | `s2_train_v3_lora.py:345-360` |
| `train.if_save_every_weights` | 是否导出可推理的 `SoVITS_weights_v3/*.pth` | `s2_train_v3_lora.py:361` |
| `train.lora_rank` | LoRA rank，同时进文件名 `_l<rank>` 和权重的 `lora_rank` 键 | `s2_train_v3_lora.py:142-150,379` |
| `train.gpu_numbers` | 字符串，`"0"` 或 `"0-1"`；脚本自己设 `CUDA_VISIBLE_DEVICES` | `s2_train_v3_lora.py:8-9` |
| `train.grad_ckpt` | v3 梯度检查点，省显存 | `s2_train_v3_lora.py:319` |
| `train.pretrained_s2G` | v3 底模；路径不存在会**静默从头训练** | `s2_train_v3_lora.py:195-208` |
| `data.exp_dir` | 日志/训练状态根目录 | `s2_train_v3_lora.py:78,142` |
| `s2_ckpt_dir` | `utils.get_hparams` 写 `config.json` 和 TensorBoard 的目录 | `utils.py:222-232`、`s2_train_v3_lora.py:81-82` |
| `save_weight_dir` | 最终权重目录；必须已存在 | `process_ckpt.py:41-57` |
| `name` | 最终权重文件名前缀 | `s2_train_v3_lora.py:379` |
| `model.version` | 决定走 V3 还是 V4 的 loader/collate | `s2_train_v3_lora.py:96-97` |

> `GPT_SoVITS/configs/s2.json:89` 的 `s2_ckpt_dir` 是 `"logs/s2/big2k1"`，这是仓库自带的示例默认值；WebUI 会覆盖成 `logs/<exp>`。直接拿原文件跑而不覆盖，权重和日志会落到 `logs/s2/big2k1`。

### 4.2 s1：YAML（v3 用 `s1longer-v2.yaml` 模板）

WebUI 的注入过程（`webui.py:604-634`）：

```python
with open("GPT_SoVITS/configs/s1longer.yaml" if version == "v1"
          else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
    data = yaml.load(...)
data["train"]["batch_size"]           = batch_size
data["train"]["epochs"]               = total_epoch          # 默认 15
data["pretrained_s1"]                 = pretrained_s1        # s1v3.ckpt
data["train"]["save_every_n_epoch"]   = save_every_epoch     # 默认 5
data["train"]["if_save_every_weights"] = if_save_every_weights
data["train"]["if_save_latest"]       = if_save_latest
data["train"]["if_dpo"]               = if_dpo
data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]  # GPT_weights_v3
data["train"]["exp_name"]             = exp_name
data["train_semantic_path"] = "logs/<exp>/6-name2semantic.tsv"
data["train_phoneme_path"]  = "logs/<exp>/2-name2text.txt"
data["output_dir"]          = "logs/<exp>/logs_s1_v3"
```

可直接用的最小 YAML（`C:\VoiceLab\work\myvoice\s1_v3.yaml`）：

```yaml
train:
  seed: 1234
  epochs: 15
  batch_size: 4
  save_every_n_epoch: 5
  precision: 16-mixed
  gradient_clip: 1.0
  if_save_every_weights: true
  if_save_latest: true
  if_dpo: false
  half_weights_save_dir: GPT_weights_v3
  exp_name: myvoice
optimizer:
  lr: 0.01
  lr_init: 0.00001
  lr_end: 0.0001
  warmup_steps: 2000
  decay_steps: 40000
data:
  max_eval_sample: 8
  max_sec: 54
  num_workers: 2
  pad_val: 1024
model:
  vocab_size: 1025
  phoneme_vocab_size: 732
  embedding_dim: 512
  hidden_dim: 512
  head: 16
  linear_units: 2048
  n_layer: 24
  dropout: 0
  EOS: 1024
  random_bert: 0
inference:
  top_k: 15
pretrained_s1: GPT_SoVITS/pretrained_models/s1v3.ckpt
train_semantic_path: logs/myvoice/6-name2semantic.tsv
train_phoneme_path: logs/myvoice/2-name2text.txt
output_dir: logs/myvoice/logs_s1_v3
```

与 `s1.yaml` 的差别（`configs/s1.yaml` vs `configs/s1longer-v2.yaml`）：

| 字段 | s1.yaml（v1 旧） | s1longer-v2.yaml（v3 用） |
|---|---|---|
| `train.epochs` | 300 | 20（WebUI 再覆盖为 15） |
| `train.precision` | `16` | `16-mixed` |
| `train.gradient_accumulation` | 4 | 无 |
| `data.num_workers` | 1 | 4 |
| `model.phoneme_vocab_size` | 512 | **732** |
| `model.n_layer` | 12 | **24** |
| `inference.top_k` | 5 | 15 |

v3 的 GPT 底模 `s1v3.ckpt` 与 `s1longer-v2.yaml` 的 24 层/732 音素表匹配；不要拿 `s1.yaml` 去配 v3。

字段含义（只列训练脚本真正读的）：

| 字段 | 作用 | 源码 |
|---|---|---|
| `train.epochs` | `Trainer(max_epochs=...)` | `s1_train.py:111-112` |
| `train.batch_size` | DataModule 的 batch（还会被 `len(dataset)//4` 夹一次） | `AR/data/data_module.py:46-55` |
| `train.save_every_n_epoch` | `ModelCheckpoint(every_n_epochs=...)` | `s1_train.py:95-107` |
| `train.precision` | `16-mixed` / `32` | `s1_train.py:123` |
| `train.if_save_latest` | 是否只保留最新 ckpt | `s1_train.py:50-61` |
| `train.if_save_every_weights` | 是否导出 `GPT_weights_v3/<exp>-e<N>.ckpt` | `s1_train.py:62-81` |
| `train.half_weights_save_dir` | 最终权重目录 | `s1_train.py:73-81` |
| `train.exp_name` | 最终权重文件名前缀 | `s1_train.py:73-81` |
| `pretrained_s1` | GPT 底模；`config.get("pretrained_s1")` | `AR/models/t2s_lightning_module.py:24-35` |
| `train_semantic_path` | `6-name2semantic.tsv` | `s1_train.py:132-138` |
| `train_phoneme_path` | `2-name2text.txt` | 同上 |
| `output_dir` | 训练状态和 TensorBoard 根 | `s1_train.py:88-108` |

### 4.3 路径落点总表

| 逻辑名 | 实际路径（cwd=仓库根） | 由谁产生/消费 |
|---|---|---|
| 输入标注 | 任意 `*.list` | 1A/1B/1C 读 |
| `2-name2text.txt` | `logs/<exp>/2-name2text.txt` | 1A 产；s2/s1 读 |
| `3-bert/` | `logs/<exp>/3-bert/<wav>.pt` | 1A 产；s1 可选读 |
| `4-cnhubert/` | `logs/<exp>/4-cnhubert/<wav>.pt` | 1B 产；1C、s2 读 |
| `5-wav32k/` | `logs/<exp>/5-wav32k/<wav>` | 1B 产；s2 读；2-get-sv 读 |
| `6-name2semantic.tsv` | `logs/<exp>/6-name2semantic.tsv` | 1C 产；s1 读 |
| `7-sv_cn/` | `logs/<exp>/7-sv_cn/<wav>.pt` | 仅 Pro：2-get-sv 产；s2_train 的 Pro 分支读 |
| s2 配置副本 | `logs/<exp>/config.json` | `utils.get_hparams` 写 |
| s2 训练状态 | `logs/<exp>/logs_s2_v3_lora_<rank>/` | v3 LoRA 训练写 |
| s1 训练状态 | `logs/<exp>/logs_s1_v3/`（ckpt 在其下 `ckpt/`） | s1 训练写 |
| s2 最终权重 | `SoVITS_weights_v3/<exp>_e<ep>_s<step>_l<rank>.pth` | 可推理 |
| s1 最终权重 | `GPT_weights_v3/<exp>-e<ep>.ckpt` | 可推理 |

> 没有 `logs/s2/<exp>/` 这种目录，也没有 `logs/s1/` 这种全局目录。`s2.json` 里的 `logs/s2/big2k1` 只是示例默认值。

---

## 5. 数据集格式

### 5.1 输入 `.list`（唯一的数据清单）

每行 4 列，**竖线 `|` 分隔**，列序固定：

```text
wav文件名或路径|角色名|语言标签|文本
001.wav|myvoice|ZH|你好，这是测试。
002.wav|myvoice|ZH|今天天气不错。
```

源码：

- `1-get-text.py:129`：`wav_name, spk_name, language, text = line.split("|")`
- `2-get-hubert-wav32k.py:114`：同样的 4 段解包
- `3-get-semantic.py:110`：同样的 4 段解包

要点：

- `spk_name` 被解析但**从不使用**；本项目实际是单说话者训练，不要指望它做多说话人。
- `language` 支持 `ZH/zh`、`JP/jp/JA/ja`、`EN/en/En`、`KO/Ko/ko`、`yue/YUE/Yue`（`1-get-text.py:110-126`）。其它值会打印 warning 并跳过（`1-get-text.py:134`）。
- `wav_name` 可以写绝对路径；但一旦设置 `inp_wav_dir`，三个脚本都会只取 `os.path.basename(wav_name)` 再拼目录（`1-get-text.py:89-90`、`2-get-hubert-wav32k.py:116-122`、`3-get-semantic.py:111-114`）。
- 如果 `inp_wav_dir` 留空，`wav_name` 必须本身是可访问的完整路径。
- 文本里不能含 `|`，否则解包失败（该行被跳过或抛异常）。

### 5.2 `2-name2text.txt`

4 列，**制表符 `\t` 分隔**，无表头：

```text
001.wav<TAB>n i3 h ao3 ...<TAB>[2, 1, ...]<TAB>你好，这是测试。
```

- 第 1 列：wav 的 basename（含扩展名，如 `001.wav`）
- 第 2 列：空格分隔的音素
- 第 3 列：`word2ph` 的 Python list 字符串（s2 只取第 2 列；s1 存下来但当前不参与计算）
- 第 4 列：归一化后的文本

源码：`1-get-text.py:140-143`；消费方 `module/data_utils.py:41-48`（要求 `len(tmp)==4`，否则跳过）和 `AR/data/dataset.py:75-82`。

### 5.3 `6-name2semantic.tsv`

制表符分隔，**有表头**，两列：

```text
item_name	semantic_audio
001.wav	12 345 6789 ...
```

源码：表头由 WebUI 合并时写死（`webui.py:1003,1213`），行由 `3-get-semantic.py:100` 生成；消费方 `AR/data/dataset.py:57-61,130-142`。

### 5.4 wav 采样率要求

- 输入 wav：**没有强制 24k**。`2-get-hubert-wav32k.py:82` 用 `load_audio(wav_path, 32000)` 通过 ffmpeg 解码并重采样到 32k；所以 24k 切好的 wav 可以直接用。
- 中间产物 `5-wav32k/`：**32 kHz、单声道、16-bit PCM**（`2-get-hubert-wav32k.py:100-104`）。
- v3 训练：从 `5-wav32k` 以 32k 读入，再在 `TextAudioSpeakerLoaderV3.get_audio` 里重采样到 **24 kHz** 计算 100 维 mel（`module/data_utils.py:396-425`），同时保留 32k 的线性谱。因此 v3 是“32k 谱 + 24k mel”，不是把输入 wav 限定为 24k。
- v3 推理输出是 24k（`TTS_infer_pack/TTS.py:1618`）。
- `2-get-sv.py` 只接受 `5-wav32k` 且断言 `sr0 == 32000`（`2-get-sv.py:91-93`）——但 v3 不跑它。

### 5.5 关于 `train.list` / `val.list`

本 commit 的训练代码**不读** `train.list`、`val.list`，也没有 validation loader：

- `s2_train.py:129-133` 的 eval 代码被注释；
- `s1_train.py:116` 设置 `limit_val_batches=0`；
- 全仓库没有生成 `train.list`/`val.list` 的代码。

如果封装层需要这两个名字，它们只是你外部数据组织的概念，不要把它们传给训练脚本。

---

## 6. 最小可行调用序列（可直接粘贴）

假设：

- 实验名 `myvoice`
- wav 目录 `C:\VoiceLab\data\myvoice\wav24k`
- 清单 `C:\VoiceLab\data\myvoice\dataset.list`
- 工作目录 `C:\VoiceLab\work\myvoice`

### 6.1 一次设置

```powershell
$repo = "C:\VoiceLab\GPT-SoVITS"
$py   = "C:\VoiceLab\.venv\Scripts\python.exe"
$work = "C:\VoiceLab\work\myvoice"
$exp  = "myvoice"
$data = "C:\VoiceLab\data\myvoice"

New-Item -ItemType Directory -Force $work, "$repo\logs\$exp", "$repo\SoVITS_weights_v3", "$repo\GPT_weights_v3" | Out-Null
$env:PYTHONPATH = "$repo;$repo\GPT_SoVITS"
$env:is_half = "True"
$env:_CUDA_VISIBLE_DEVICES = "0"
$env:i_part = "0"
$env:all_parts = "1"
Set-Location $repo
```

### 6.2 1A

```powershell
$env:inp_text            = "$data\dataset.list"
$env:inp_wav_dir         = "$data\wav24k"
$env:exp_name            = $exp
$env:opt_dir             = "logs\$exp"
$env:bert_pretrained_dir = "$repo\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"
$env:version             = "v3"

& $py -s GPT_SoVITS\prepare_datasets\1-get-text.py
Move-Item "logs\$exp\2-name2text-0.txt" "logs\$exp\2-name2text.txt"
```

### 6.3 1B（v3 到此为止，不跑 2-get-sv）

```powershell
$env:cnhubert_base_dir = "$repo\GPT_SoVITS\pretrained_models\chinese-hubert-base"
& $py -s GPT_SoVITS\prepare_datasets\2-get-hubert-wav32k.py
```

### 6.4 1C

```powershell
$env:pretrained_s2G = "$repo\GPT_SoVITS\pretrained_models\s2Gv3.pth"
$env:s2config_path  = "$repo\GPT_SoVITS\configs\s2.json"

& $py -s GPT_SoVITS\prepare_datasets\3-get-semantic.py
& $py -c "from pathlib import Path; d=Path('logs/$exp'); body=(d/'6-name2semantic-0.tsv').read_text('utf-8'); (d/'6-name2semantic.tsv').write_text('item_name\tsemantic_audio\n'+body.strip('\n')+'\n', encoding='utf-8'); (d/'6-name2semantic-0.tsv').unlink()"
```

### 6.5 s2（v3 LoRA）

把 §4.1 的 JSON 写入 `$work\s2_v3.json`（把 `logs/myvoice` 改成 `logs/$exp` 如果需要），然后：

```powershell
Set-Location $repo
& $py -s GPT_SoVITS\s2_train_v3_lora.py --config "$work\s2_v3.json"
```

### 6.6 s1（GPT）

把 §4.2 的 YAML 写入 `$work\s1_v3.yaml`，然后：

```powershell
$env:_CUDA_VISIBLE_DEVICES = "0"
$env:hz = "25hz"
Set-Location $repo
& $py -s GPT_SoVITS\s1_train.py --config_file "$work\s1_v3.yaml"
```

### 6.7 最小参数解释

- s2：**必须**有 `train.epochs`、`train.batch_size`、`train.gpu_numbers`、`train.lora_rank`、`train.pretrained_s2G`、`data.exp_dir`、`s2_ckpt_dir`、`save_weight_dir`、`name`、`model.version`。epoch 只从 JSON 的 `train.epochs` 来，没有 `-e`。
- s1：**必须**有 `train.epochs`、`train.batch_size`、`train.save_every_n_epoch`、`train.precision`、`train.if_save_every_weights`、`train.if_save_latest`、`train.half_weights_save_dir`、`train.exp_name`、`pretrained_s1`、`train_semantic_path`、`train_phoneme_path`、`output_dir`。batch/epoch/save 频率都只从 YAML 来，命令行只有 `-c/--config_file`。

---

## 7. 产物文件名规律与推理对接

### 7.1 s2（v3 LoRA）

```
SoVITS_weights_v3\<exp>_e<epoch>_s<global_step>_l<lora_rank>.pth
例：SoVITS_weights_v3\myvoice_e2_s174_l32.pth
```

源码：文件名由 `hps.name + "_e%s_s%s_l%s" % (epoch, global_step, lora_rank)` 拼出（`s2_train_v3_lora.py:379`），目录由 `hps.save_weight_dir` 决定（`process_ckpt.py:51-53`）。

这个文件**不是完整底模**，而是训练脚本筛选出的可训练参数（v3 下主要是 LoRA 增量；底模冻结部分不在其中，加载时要靠 `s2Gv3.pth` 补齐）：

- `process_ckpt.savee` 只保存 `requires_grad` 的参数（`s2_train_v3_lora.py:366-371`）；
- 文件头被写成 ASCII 的 `b"03"`（v3 LoRA），`head2version[b"03"] = ["v2","v3",True]`（`process_ckpt.py:22-27,72-80`）；
- 顶层带 `lora_rank` 键（`process_ckpt.py:51-53`）。

因此推理时必须同时有 `GPT_SoVITS/pretrained_models/s2Gv3.pth`：

- 源码 `TTS_infer_pack/TTS.py:498-502` 会检查 `default_configs["v3"]["vits_weights_path"]`，缺了直接 `FileNotFoundError`；
- 加载顺序是“先底模，再 LoRA，再 merge”（`TTS.py:561-581`）。

### 7.2 s1

```
GPT_weights_v3\<exp>-e<epoch>.ckpt
例：GPT_weights_v3\myvoice-e15.ckpt
```

源码：`s1_train.py:73-81`，注意是**连字符** `-e`，不是下划线；epoch 是 `trainer.current_epoch + 1`。

内部训练 ckpt 在 `logs\<exp>\logs_s1_v3\ckpt\*.ckpt`，是 Lightning 的 `state_dict` 结构，**不能**直接给推理用；推理要的是 `GPT_weights_v3` 下带 `["weight"]` + `["config"]` 的小模型（`TTS.py:599-603`）。

### 7.3 对接 `TTSInferencer(gpt_path=..., sovits_path=...)`

GPT-SoVITS 源码本身没有 `TTSInferencer` 这个类（只有 `TTS_infer_pack/TTS.py` 的 `TTS` / `TTS_Config`）。你工作区的封装层 `C:\VoiceLab\Aqua-TTS\aquatts\inferencer.py:153-188` 定义了 `TTSInferencer(gpt_path=..., sovits_path=...)`，其默认值正好印证了命名规律：

```python
default_gpt_path    = .../GPT_weights_v3/xxx-e15.ckpt
default_sovits_path = .../SoVITS_weights_v3/xxx_e2_s174_l32.pth
default_sovits_pretrain_path = .../GPT_SoVITS/pretrained_models/s2Gv3.pth
```

（`inferencer.py:262-264`）

所以映射是：

```python
TTSInferencer(
    gpt_path    = r"C:\VoiceLab\GPT-SoVITS\GPT_weights_v3\myvoice-e15.ckpt",
    sovits_path = r"C:\VoiceLab\GPT-SoVITS\SoVITS_weights_v3\myvoice_e2_s174_l32.pth",
    # v3 还需要底模；封装层有默认，显式写更稳：
    sovits_pretrain_path = r"C:\VoiceLab\GPT-SoVITS\GPT_SoVITS\pretrained_models\s2Gv3.pth",
    device="cuda",
)
```

对应到原生 `TTS_Config` 字段就是 `t2s_weights_path`（GPT）和 `vits_weights_path`（SoVITS）（`TTS.py:237-245`）。

---

## 8. 坑（逐条附源码行号）

1. **v3 不是 `s2_train.py`**
   `webui.py:541-544`：只有 `v1/v2/v2Pro/v2ProPlus` 走 `s2_train.py`，`else`（v3/v4）走 `s2_train_v3_lora.py`。`s2_train.py` 导入的是 `SynthesizerTrn`/`MultiPeriodDiscriminator`（`s2_train.py:35-38`），没有 v3 的 CFM/LoRA 逻辑。

2. **v3 不需要、也不应该跑 `2-get-sv.py`**
   只有版本字符串含 `"Pro"` 才跑（`webui.py:1151`、`910`）。`7-sv_cn` 只在 `is_v2Pro` 时被断言和读取（`module/data_utils.py:32-39,120-121`）；v3 的 `TextAudioSpeakerLoaderV3` 只取 `2/4/5`（`module/data_utils.py:288-306`）。本机 `GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` 不存在，跑 Pro 分支会直接失败；v3 不受影响。

3. **没有 `train.list`/`val.list`**
   训练不读它们；s2 没有 validation loader（`s2_train.py:129-133` 注释），s1 `limit_val_batches=0`（`s1_train.py:116`）。不要按老教程去造这两个文件。

4. **`pretrained_s2G` 路径错会静默从零训练**
   `s2_train_v3_lora.py:195-208`：只有 `path != "" and path != None and os.path.exists(path)` 才加载底模；否则直接随机初始化继续训练，不报错。相对路径是相对 **cwd**，所以必须从仓库根跑，或写绝对路径。

5. **`save_weight_dir` 不存在会“训练成功但没有权重”**
   `process_ckpt.py:41-60` 整个保存逻辑包在 `try/except` 里，失败只返回 traceback 字符串，`s2_train_v3_lora.py:372-387` 只把它写进 logger。必须先 `mkdir SoVITS_weights_v3`。

6. **s1 的 `half_weights_save_dir` 不存在会直接崩**
   `s1_train.py:73-81` 调 `process_ckpt.my_save`，后者 `shutil.move(tmp, dir/name)`（`process_ckpt.py:12-17`）不做 `makedirs`；目录缺失会在保存时抛异常。必须先 `mkdir GPT_weights_v3`。

7. **`logs/` 本身不用预建，但每个实验目录由代码建**
   `utils.get_logger` 会 `os.makedirs(model_dir)`（`utils.py:309-316`），`utils.get_hparams` 会建 `s2_ckpt_dir`（`utils.py:228-229`）。真正需要预建的是 §8.5/8.6 的两个权重目录。

8. **数据准备脚本不接受 `--help`，也不接受位置参数**
   它们只读 `os.environ`（`1-get-text.py:5-13`、`2-get-hubert-wav32k.py:6-16`、`2-get-sv.py:6-15`、`3-get-semantic.py:3-11`）。传 `--help` 不会打印帮助，而是直接执行，然后在缺环境变量时抛 `TypeError`（实测 `os.makedirs(None)`）。所有参数必须用环境变量传。

9. **不设 `PYTHONPATH` 会 `ModuleNotFoundError`**
   实测：从仓库根直接跑 `1-get-text.py` 报 `No module named 'text'`；只加 `GPT_SoVITS` 报 `No module named 'tools'`。必须同时加仓库根和 `GPT_SoVITS`（§2.2）。`webui.py:6` 的 `sys.path.insert` 不会传给 `Popen` 子进程。

10. **必须在仓库根跑，不能在 `GPT_SoVITS` 目录跑**
    所有预训练/权重/日志路径都是相对 cwd（`config.py:15,137`、`webui.py:515,535,609,625-627`）。在 `GPT_SoVITS` 下跑，`GPT_SoVITS/pretrained_models/...` 会变成两层 `GPT_SoVITS`，且 `--config` 里的相对路径也全错。`python -m` 不是本项目的用法；正确的形式是 `python -s GPT_SoVITS/xxx.py` + cwd=仓库根 + `PYTHONPATH`。

11. **`2-get-hubert-wav32k.py` 会静默丢样本**
    - 振幅 `max > 2.2` 直接 `return`（`2-get-hubert-wav32k.py:83-86`）；
    - SSL 出现 NaN 会进 `nan_fails` 并跳过（`:96-99`）；
    - 第一遍 half 失败后会用 float 重试（`:127-134`）。
    如果最后样本数明显变少，先看这个脚本的 stdout，不要先怀疑 list。

12. **`5-wav32k` 里的文件名保留原扩展名**
    `wavfile.write("%s/%s" % (wav32dir, wav_name), ...)`（`2-get-hubert-wav32k.py:100-104`），`wav_name` 是 list 里的 basename。如果源是 `.mp3`，`5-wav32k/xxx.mp3` 里其实是 32k WAV 内容。**不要重命名**，因为 `2-name2text`、`4-cnhubert`、`6-tsv` 都用同一个 basename 做交集（`module/data_utils.py:294-306`）。

13. **`3-get-semantic` 的版本是按文件大小猜的**
    `3-get-semantic.py:18-28`：`<82978KB` → v1，`<100MB` → v2，`<103520KB` → v1，`<700MB` → v2，否则 v3。`s2Gv3.pth` 769,035,145 B 会走 v3；如果换成 v2 底模，会静默按 v2 提取，token 与 v3 训练不匹配。

14. **单卡也可能起 `torch.distributed`**
    - `s2_train.py:53-68`：**即使 1 张卡也 `mp.spawn`**，并在 `run()` 里无条件 `dist.init_process_group`（`:80-85`）；Windows 用 gloo。所以它必须作为脚本在 `__main__` 下跑，不能 import 调用；Windows 下 `num_workers=5` + `persistent_workers=True` 也容易出问题（`:119-128`）。
    - `s2_train_v3_lora.py:53-72`：`n_gpus <= 1` 时直接 `run(0, n_gpus, hps)`，**不初始化 distributed**；Windows 单卡 `worker_count=0`（`:127`）。这是 v3 在 Windows 上更稳的原因之一。
    - `s1_train.py:111-128`：只要 CUDA 可用就用 `DDPStrategy`、`devices=-1`，Windows 后端 gloo；单卡也会走 DDP 启动流程。`_CUDA_VISIBLE_DEVICES` 必须在启动前设好。

15. **`s2_train_v3_lora.py` 自己改 `CUDA_VISIBLE_DEVICES`**
    `s2_train_v3_lora.py:8-9` 在导入 torch 前执行 `os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")`。所以 `gpu_numbers` 必须是字符串（`"0"`），不能是数字 `0`，否则 `.replace` 报错。多卡写 `"0-1"`。

16. **s1 只认 `_CUDA_VISIBLE_DEVICES`，不认 `CUDA_VISIBLE_DEVICES` 的映射**
    `s1_train.py:4-5` 只在存在 `_CUDA_VISIBLE_DEVICES` 时把它赋给 `CUDA_VISIBLE_DEVICES`。直接设 `CUDA_VISIBLE_DEVICES` 也能被 PyTorch 识别，但如果你同时想复用 WebUI 的变量名，用带下划线的那个。

17. **s2 的“最终权重”和“训练状态”不是一回事**
    `logs\<exp>\logs_s2_v3_lora_32\G_*.pth` 是含 optimizer 的训练状态（`utils.save_checkpoint`，`utils.py:75-90`），不能直接给推理；推理要 `SoVITS_weights_v3\*.pth`（`process_ckpt.savee` 产物，带 `weight`/`config`/`lora_rank`）。s1 同理：`logs\<exp>\logs_s1_v3\ckpt\*.ckpt` ≠ `GPT_weights_v3\*-e*.ckpt`。

18. **v3 LoRA 权重不能脱离底模单独加载**
    `TTS.py:498-502` 在检测到 `if_lora_v3` 后会找 `default_configs["v3"]["vits_weights_path"]`（即 `GPT_SoVITS/pretrained_models/s2Gv3.pth`）；找不到直接抛 `FileNotFoundError`。移动底模会让所有权重失效。

19. **s1 的 `output_dir` 不是 `logs/s1`**
    实际是 `logs\<exp>\logs_s1_v3`（`webui.py:627`）。老教程里的 `logs/s1` 在本 commit 不存在。

20. **WebUI 单选框没有 v3**
    `webui.py:1495-1500` 的 choices 是 `["v1","v2","v4","v2Pro","v2ProPlus"]`；但 `webui.py:4` 的默认 `version` 是 `v2Pro`，且 v3 的整条代码路径（`s2_train_v3_lora.py`、`s1longer-v2.yaml`、`s2Gv3.pth`）都在。要用 v3 就走本文的命令行，不要被 UI 选项限制。

21. **硬编码绝对路径基本没有，但“相对 cwd”到处都是**
    全仓库非注释的 `/data/docker/...` 只有 `AR/data/dataset.py:301` 的 `if __name__ == "__main__"` 测试块，不参与训练。真正会坑人的是相对路径：`GPT_SoVITS/pretrained_models/s2Gv3.pth`（`config.py:15`、`TTS.py:242`）、`GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x`（`TTS.py:625`，用 `now_dir` 拼）、`sv_path`（`webui.py:865`）。统一从仓库根跑即可。

22. **`s2_train_v3_lora.py` 有自动续训**
    启动时先找 `save_root` 下的 `G_*.pth`（`s2_train_v3_lora.py:177-189`），找到就接着训；找不到才加载 `pretrained_s2G`。要“从头重训”必须先删 `logs\<exp>\logs_s2_v3_lora_<rank>`。s1 同理，`s1_train.py:140-147` 会从 `output_dir/ckpt` 里挑最新 ckpt 续训，重训要清空该目录。

23. **`2-name2text.txt` 的 4 列少一列就会被静默跳过**
    `module/data_utils.py:44-48` 和 `AR/data/dataset.py:78-82` 都要求 `len(tmp) == 4`，不满足就 `continue`。表现是“数据明明有，训练集却是 0/很少”。合并分片时不要引入空行、BOM 或额外制表符。

24. **`6-name2semantic.tsv` 没有表头会被当表头吃掉第一条**
    `pandas.read_csv(delimiter="\t")` 默认 `header=0`（`AR/data/dataset.py:57-61`）。必须写 `item_name\tsemantic_audio` 表头；WebUI 合并时就是这么做的（`webui.py:1003,1213`）。

25. **`SoVITS_weights_v3\*.pth` 不是普通 zip，不能直接 `torch.load`**
    `process_ckpt.my_save2` 把文件前两字节替换成 ASCII 的 `b"03"`（v3 LoRA；`process_ckpt.py:30-38,51-53`）。直接 `torch.load` 会报 `UnpicklingError: invalid load key`。仓库用 `load_sovits_new` 先补 `PK` 再 `torch.load`（`process_ckpt.py:129-138`）；封装层也应走这个函数。`GPT_weights_v3\*.ckpt` 是普通 `torch.save`，可直接 `torch.load`。

---

## 9. 给封装者的最小调用序列（只含命令与环境变量）

```text
# 常量
REPO   = C:\VoiceLab\GPT-SoVITS
PY     = C:\VoiceLab\.venv\Scripts\python.exe
EXP    = <实验名>
LIST   = <dataset.list 绝对路径>
WAVDIR = <切好的 wav 目录绝对路径>
WORK   = <配置/临时文件目录>

# 全局环境
cwd = REPO
PYTHONPATH = REPO;REPO\GPT_SoVITS
is_half = True
version = v3
i_part = 0
all_parts = 1
_CUDA_VISIBLE_DEVICES = <GPU号>

mkdir REPO\logs\EXP
mkdir REPO\SoVITS_weights_v3
mkdir REPO\GPT_weights_v3

# 1A
inp_text = LIST
inp_wav_dir = WAVDIR
exp_name = EXP
opt_dir = logs\EXP
bert_pretrained_dir = REPO\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large
run PY -s GPT_SoVITS\prepare_datasets\1-get-text.py
merge/rename logs\EXP\2-name2text-*.txt -> logs\EXP\2-name2text.txt

# 1B（v3 不跑 2-get-sv.py）
cnhubert_base_dir = REPO\GPT_SoVITS\pretrained_models\chinese-hubert-base
run PY -s GPT_SoVITS\prepare_datasets\2-get-hubert-wav32k.py

# 1C
pretrained_s2G = REPO\GPT_SoVITS\pretrained_models\s2Gv3.pth
s2config_path = REPO\GPT_SoVITS\configs\s2.json
run PY -s GPT_SoVITS\prepare_datasets\3-get-semantic.py
merge logs\EXP\6-name2semantic-*.tsv -> logs\EXP\6-name2semantic.tsv，首行 item_name\tsemantic_audio

# s2
write WORK\s2_v3.json from template （train.epochs / batch_size / lora_rank / gpu_numbers ...）
run PY -s GPT_SoVITS\s2_train_v3_lora.py --config WORK\s2_v3.json

# s1
_CUDA_VISIBLE_DEVICES = <GPU号>
hz = 25hz
write WORK\s1_v3.yaml from template （train.epochs / batch_size / save_every_n_epoch ...）
run PY -s GPT_SoVITS\s1_train.py --config_file WORK\s1_v3.yaml

# 产物
sovits = REPO\SoVITS_weights_v3\EXP_e<ep>_s<step>_l<rank>.pth
gpt    = REPO\GPT_weights_v3\EXP-e<ep>.ckpt
TTSInferencer(gpt_path=gpt, sovits_path=sovits,
              sovits_pretrain_path=REPO\GPT_SoVITS\pretrained_models\s2Gv3.pth)
```

---

## 10. 源码证据索引

| 主题 | 文件:行 |
|---|---|
| WebUI 全局 version / cwd / TEMP | `webui.py:4-6,21-22` |
| 权重目录创建 | `webui.py:200-201` |
| s2 配置注入 | `webui.py:507-540` |
| v3 训练脚本选择 | `webui.py:541-544` |
| s1 配置注入 / 命令 | `webui.py:604-636` |
| 1a/1b/1c 命令与环境 | `webui.py:780-1011` |
| 一键三连与 Pro 分支 | `webui.py:1046-1225`（Pro 在 `1151-1167`） |
| 版本单选框 | `webui.py:1495-1500` |
| s1 默认 15 epoch / save 5 | `webui.py:1797-1812` |
| 权重根目录映射 | `config.py:44-75` |
| `exp_root` / `python_exec` | `config.py:137-138` |
| v3 预训练路径 | `config.py:15,24` |
| 数据准备 1A 环境变量 | `prepare_datasets/1-get-text.py:5-16` |
| 1A 分片/4 列解析/输出 | `1-get-text.py:127-143` |
| 1B 32k 重采样/输出 | `prepare_datasets/2-get-hubert-wav32k.py:82,100-104` |
| 2-get-sv 输出 | `prepare_datasets/2-get-sv.py:42,88-96` |
| 1C 版本判定/输出 | `prepare_datasets/3-get-semantic.py:18-28,58,100` |
| s2/s1 参数解析 | `utils.py:189-233`、`s1_train.py:152-169` |
| s2 的 v2Pro/sv 分支 | `s2_train.py:90,332-384` |
| s2 单卡也 spawn | `s2_train.py:53-68,80-85` |
| s2 保存/命名 | `s2_train.py:514-576` |
| v3 loader（无 sv） | `module/data_utils.py:279-306` |
| v2Pro loader（有 sv） | `module/data_utils.py:32-39,120-132` |
| v3 24k mel | `module/data_utils.py:362-370,396-425` |
| v3 LoRA 训练/保存 | `s2_train_v3_lora.py:8-9,58-72,96-98,142-150,177-211,344-387` |
| s1 权重命名 | `s1_train.py:73-81` |
| s1 Lightning 配置 | `s1_train.py:86-128` |
| savee / 版本头 | `process_ckpt.py:22-27,30-38,41-60,72-80,129-138` |
| 推理加载/LoRA 检测 | `TTS_infer_pack/TTS.py:493-581` |
| v3 推理默认路径 | `TTS_infer_pack/TTS.py:237-245` |
| v3 输出 24k | `TTS_infer_pack/TTS.py:1618` |
| 封装层默认文件名 | `C:\VoiceLab\Aqua-TTS\aquatts\inferencer.py:262-264` |
