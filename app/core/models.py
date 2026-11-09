"""模型清单与下载（GitHub 版不含模型，首次运行按需下载）。

提供两类 ASR：
  zh: FunASR paraformer-zh（默认，中文最准）
  en: faster-whisper small（英文/多语，需用户选择）
其余模型（GPT-SoVITS v4 底模、声码器、文本前端、语言判别）自动下载。
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

HF = "https://huggingface.co"

Progress = Callable[[str, float], None]   # (说明, 0..1)


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    dest: str          # 相对安装根目录
    size_hint_mb: int = 0


COMMON: list[Asset] = [
    Asset("s1v3.ckpt",
          f"{HF}/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt",
          "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt", 148),
    Asset("s2Gv4.pth",
          f"{HF}/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/s2Gv4.pth",
          "GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth", 733),
    Asset("vocoder.pth",
          f"{HF}/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/vocoder.pth",
          "GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth", 55),
    Asset("chinese-roberta-wwm-ext-large/pytorch_model.bin",
          f"{HF}/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
          "GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin", 621),
    Asset("chinese-hubert-base/pytorch_model.bin",
          f"{HF}/TencentGameMate/chinese-hubert-base/resolve/main/pytorch_model.bin",
          "GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin", 181),
]

# 中文 ASR 由 FunASR 在首次识别时自行下载到 models_cache/modelscope/models/，
# 这里不再重复下载（单个 model.pt 无法被 AutoModel 使用）。
ASR_ZH: list[Asset] = []

ASR_EN: list[Asset] = [
    Asset("whisper-small-ct2",
          f"{HF}/Systran/faster-whisper-small/resolve/main/model.bin",
          "models_cache/whisper/faster-whisper-small/model.bin", 484),
]


def assets_for(kind: str = "zh") -> list[Asset]:
    return COMMON + (ASR_EN if str(kind).lower().startswith("en") else ASR_ZH)


def missing(root: Path, kind: str = "zh") -> list[Asset]:
    return [a for a in assets_for(kind) if not (root / a.dest).exists()]


def _download(url: str, dest: Path, on_progress: Callable[[int, int], None] | None = None,
              timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    os.replace(tmp, dest)


def download_all(root: Path, kind: str = "zh",
                 progress: Progress | None = None) -> list[str]:
    """下载缺失模型；返回已下载的文件名列表。"""
    root = Path(root)
    todo = missing(root, kind)
    got: list[str] = []
    if not todo:
        return got
    for i, asset in enumerate(todo):
        base = i / len(todo)

        def on_prog(done: int, total: int, _a=asset, _b=base) -> None:
            if progress:
                frac = _b + (done / total / len(todo) if total else 0)
                progress(f"{_a.name}  {done/1e6:.0f}/{total/1e6:.0f} MB",
                         min(0.99, frac))

        if progress:
            progress(f"下载 {asset.name}（约 {asset.size_hint_mb} MB）", base)
        _download(asset.url, root / asset.dest, on_prog)
        got.append(asset.name)
    if progress:
        progress("全部完成", 1.0)
    return got


__all__ = ["Asset", "assets_for", "missing", "download_all", "COMMON",
           "ASR_ZH", "ASR_EN"]
