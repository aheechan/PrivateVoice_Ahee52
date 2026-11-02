"""缓存 / 临时文件清理。

分发包里哪些东西可以删、哪些绝对不能删：

可删（清理按钮只动这些）：
  logs\\*.log            运行日志（含 app.log 轮转）
  logs\\turns.jsonl      每轮延迟埋点
  datasets\\*\\work\\      训练中间产物（wav24k、train.list、s2/s1 配置）
  datasets\\_probe 等临时目录
  results\\quality\\      音质诊断产物
  results\\compare\\      A/B 对比音频
  %TEMP%\\vlab-*         解码临时 wav

不能删：
  models_cache\\         内置的 ASR/标点/说话人模型（2GB，删了要重新下载）
  characters\\<id>\\      profile、checkpoints、reference、dataset（角色本体）
  runtime\\ + app\\      程序本体
  GPT-SoVITS\\pretrained_models\\  v3/v4 底模
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CleanResult:
    removed: list[str] = field(default_factory=list)
    freed_mb: float = 0.0
    errors: list[str] = field(default_factory=list)


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def targets(config: Any) -> list[Path]:
    """清理目标（白名单，绝不递归删父目录）。"""
    root = Path(config.path_of("logs")).parent
    out = [
        root / "results" / "quality",
        root / "results" / "compare",
        root / "logs",
        root / "datasets",
    ]
    tmp = Path(tempfile.gettempdir())
    out += list(tmp.glob("vlab-*"))
    return out


def cache_report(config: Any) -> list[tuple[str, str, float]]:
    """(路径, 说明, MB) 用于设置页展示。"""
    root = Path(config.path_of("logs")).parent
    tmp = Path(tempfile.gettempdir())
    rows = [
        (str(root / "logs"), "日志：app.log / turns.jsonl", _size(root / "logs") / 1e6),
        (str(root / "datasets"), "训练中间产物（wav24k、配置）",
         _size(root / "datasets") / 1e6),
        (str(root / "results" / "quality"), "音质诊断音频",
         _size(root / "results" / "quality") / 1e6),
        (str(root / "results" / "compare"), "A/B 对比音频",
         _size(root / "results" / "compare") / 1e6),
        (str(root / "models_cache"), "内置模型缓存（不要删）",
         _size(root / "models_cache") / 1e6),
        (str(root / "characters"), "角色数据（不要删）",
         _size(root / "characters") / 1e6),
        (str(tmp), "系统临时目录里的 vlab-*",
         sum(_size(p) for p in tmp.glob("vlab-*")) / 1e6),
    ]
    return rows


def clean(config: Any, keep_logs: bool = False) -> CleanResult:
    res = CleanResult()
    root = Path(config.path_of("logs")).parent

    def drop(path: Path, label: str, keep_dir: bool = False) -> None:
        if not path.exists():
            return
        size = _size(path)
        try:
            if path.is_dir():
                if keep_dir:
                    for child in path.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            res.removed.append(label)
            res.freed_mb += size / 1e6
        except OSError as exc:
            res.errors.append(f"{label}: {exc}")

    # 日志：只删文件，保留目录（进程还在往里写）
    if not keep_logs:
        for f in (root / "logs").glob("*.log"):
            drop(f, f"logs/{f.name}")
        for f in (root / "logs").glob("turns.jsonl*"):
            drop(f, f"logs/{f.name}")
    drop(root / "results" / "quality", "results/quality")
    drop(root / "results" / "compare", "results/compare")
    # datasets：只删各角色下的 work 与临时探针目录，保留结构
    ds = root / "datasets"
    if ds.exists():
        for child in ds.iterdir():
            if child.is_dir() and (child / "work").exists():
                drop(child / "work", f"datasets/{child.name}/work")
            elif child.name.startswith("_") or child.name in ("_probe",):
                drop(child, f"datasets/{child.name}")
    for p in Path(tempfile.gettempdir()).glob("vlab-*"):
        drop(p, f"temp/{p.name}")
    res.freed_mb = round(res.freed_mb, 1)
    return res


__all__ = ["CleanResult", "clean", "cache_report", "targets"]
