"""s1（GPT）训练的运行时垫片 —— 通过 PYTHONPATH 注入，不改上游源码。

本机（Windows + torch 2.11 + Lightning）跑 GPT-SoVITS 的 s1_train.py 会撞两个坑：

1. `AR/data/data_module.py` 无条件传 `prefetch_factor`，PyTorch 要求
   `num_workers > 0` 才能设它；而 DataLoader 子进程在本机会在首个 batch 前
   以 0xC0000005 崩溃。→ 把 DataLoader 换成子类，num_workers=0 时清掉
   prefetch_factor / persistent_workers，改为进程内读数据。
2. `s1_train.py` 硬编码 `DDPStrategy(process_group_backend="gloo")` +
   `devices=-1`，单卡也会走 DDP，gloo 的 all_reduce 在 Windows 上直接访问违例。
   → 单卡时把 strategy 换成 "auto"、devices 固定 1。
"""
from __future__ import annotations

_dataloader_done = False
_trainer_done = False


def _patch_dataloader() -> None:
    global _dataloader_done
    if _dataloader_done:
        return
    try:
        import torch.utils.data as _data
    except Exception:
        return

    base = _data.DataLoader

    class DataLoader(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            if int(kwargs.get("num_workers", 0) or 0) == 0:
                kwargs["prefetch_factor"] = None
                kwargs.pop("persistent_workers", None)
            super().__init__(*args, **kwargs)

    DataLoader.__name__ = "DataLoader"
    DataLoader.__qualname__ = "DataLoader"
    DataLoader.__module__ = "torch.utils.data"
    _data.DataLoader = DataLoader
    _dataloader_done = True


def _patch_trainer() -> None:
    global _trainer_done
    if _trainer_done:
        return
    try:
        import platform

        if platform.system() != "Windows":
            _trainer_done = True
            return
        import pytorch_lightning as pl
    except Exception:
        return

    orig = pl.Trainer.__init__

    def __init__(self, *args, **kwargs):
        strategy = kwargs.get("strategy")
        is_ddp = (isinstance(strategy, str) and strategy.lower().startswith("ddp")) or \
                 ("DDP" in type(strategy).__name__)
        if is_ddp:
            kwargs["strategy"] = "auto"
            kwargs["devices"] = 1
        orig(self, *args, **kwargs)

    pl.Trainer.__init__ = __init__  # type: ignore[method-assign]
    _trainer_done = True


def _patch_dist() -> None:
    """bucket_sampler 里 `dist.get_world_size()` 假设 DDP 已初始化，单卡会炸。"""
    try:
        import torch.distributed as dist
    except Exception:
        return
    if getattr(dist, "_vlab_patched", False):
        return

    orig_ws = dist.get_world_size
    orig_rank = dist.get_rank
    orig_ar = dist.all_reduce

    def get_world_size(group=None):
        try:
            return orig_ws(group)
        except Exception:
            return 1

    def get_rank(group=None):
        try:
            return orig_rank(group)
        except Exception:
            return 0

    def all_reduce(tensor, *args, **kwargs):
        try:
            return orig_ar(tensor, *args, **kwargs)
        except Exception:
            return tensor

    dist.get_world_size = get_world_size  # type: ignore[assignment]
    dist.get_rank = get_rank  # type: ignore[assignment]
    dist.all_reduce = all_reduce  # type: ignore[assignment]
    dist._vlab_patched = True  # type: ignore[attr-defined]


_patch_dataloader()
_patch_trainer()
_patch_dist()
