# Models

This repository ships **no model weights**. Everything below is downloaded on first run
(wizard → “Download missing models”) or placed manually at the paths shown.

## Where models live

```
<install>\
├─ GPT-SoVITS\GPT_SoVITS\pretrained_models\      ← speech synthesis
│  ├─ s1v3.ckpt                                 148 MB
│  ├─ gsv-v4-pretrained\s2Gv4.pth               733 MB
│  ├─ gsv-v4-pretrained\vocoder.pth              55 MB
│  ├─ chinese-roberta-wwm-ext-large\pytorch_model.bin   621 MB
│  └─ chinese-hubert-base\pytorch_model.bin     180 MB
├─ models_cache\
│  ├─ modelscope\models\        ← Chinese ASR (FunASR auto-download)
│  └─ whisper\faster-whisper-small\   ← English / multilingual ASR (461 MB)
└─ nltk_data\                   ← cmudict / POS data for English speech
```

For source users the same layout applies relative to the repository root.

## Chinese ASR — default

| Model | Size | Source |
|---|---|---|
| paraformer-zh (seaco) | ~840 MB | ModelScope `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| ct-punc | ~290 MB | ModelScope `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` |
| campplus (speaker) | ~30 MB | ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` |

FunASR downloads these into `models_cache/modelscope/models/` on first recognition.

## English / multilingual ASR — optional

| Model | Size | Source |
|---|---|---|
| faster-whisper small (CT2) | 461 MB | HuggingFace `Systran/faster-whisper-small` |

Place `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt` in
`models_cache/whisper/faster-whisper-small/`.

Switch between Chinese and English ASR in **Settings → Speech recognition**; the app restarts
automatically so the new engine loads cleanly.

## Upgrading ASR later

| Option | Size | Notes |
|---|---|---|
| SenseVoice-Small (FunASR) | ~500 MB | zh/en/ja/ko/yue, faster and more accurate than whisper-small — recommended |
| faster-whisper large-v3-turbo | ~1.6 GB | clearly better English/multilingual, still fast on GPU |
| Whisper large-v3 / Qwen3-ASR | 3–6 GB | best accuracy, highest latency |

Drop the model into `models_cache/whisper/<name>` and set `asr.whisper_size` accordingly.
