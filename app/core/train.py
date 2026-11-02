"""训练子进程封装。

原则：
- 训练一律在独立子进程里跑（`-s` 方式，cwd=GPT-SoVITS 仓库根）；
- 子进程通过 JSON 行上报进度：{"stage","epoch","loss","eta"}；
- CUDA OOM 自动 batch 减半重跑，最多 3 次；支持取消并清理临时目录；
- 训练完成后自动挑参考音频，并跑一次 benchmark 出 character_ttfp.json。

训练 CLI 的考证见 docs/research/training-cli.md（v3 走 s2_train_v3_lora.py，
不需要 2-get-sv.py，epoch 只写在配置文件的 train.epochs 里）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .characters import CharacterStore

PY = sys.executable
REPO_DIRNAME = "GPT-SoVITS"
_EPOCH_RE = re.compile(r"[Ee]poch[ \t:]*(\d+)")
_LOSS_RE = re.compile(r"loss[=: \t]+([0-9]+\.?[0-9]*)")
_OOM_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA error: out of memory", re.I)


class TrainingError(RuntimeError):
    pass


class TrainingCancelled(TrainingError):
    pass


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
@dataclass
class GpuInfo:
    name: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0
    compute_cap: tuple[int, int] = (0, 0)
    driver: str = ""
    torch: str = ""
    cuda: str = ""

    @property
    def supported(self) -> bool:
        return self.compute_cap in {(8, 6), (8, 9), (12, 0)} and self.total_gb >= 11.0

    def describe(self) -> str:
        return (f"{self.name} · {self.total_gb:.0f}GB · 可用 {self.free_gb:.1f}GB · "
                f"sm_{self.compute_cap[0]}{self.compute_cap[1]}")


def gpu_info() -> GpuInfo:
    import torch
    info = GpuInfo(torch=torch.__version__, cuda=str(torch.version.cuda or ""))
    if not torch.cuda.is_available():
        return info
    info.name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    info.free_gb = free / 1024 ** 3
    info.total_gb = total / 1024 ** 3
    info.compute_cap = tuple(torch.cuda.get_device_capability(0))  # type: ignore[assignment]
    return info


def check_hardware(cfg: dict[str, Any]) -> tuple[bool, str]:
    """启动门禁：只支持 30/40/50 系且显存 ≥12GB。"""
    info = gpu_info()
    if not info.name:
        return False, "没有检测到可用的 NVIDIA 显卡，本程序需要 RTX 30/40/50 系列。"
    sm = info.compute_cap
    allowed = {(8, 6), (8, 9), (12, 0)}
    if sm not in allowed:
        return False, (f"当前显卡 {info.name} 计算能力为 sm_{sm[0]}{sm[1]}，"
                       f"只支持 RTX 30/40/50 系（sm_86/89/120）。")
    if info.total_gb < 11.0:
        return False, (f"当前显卡只有 {info.total_gb:.1f} GB 显存，"
                       f"训练至少需要 12 GB。")
    return True, info.describe()


def estimate_vram_need_gb(plan: Any) -> float:
    """按 batch 粗估训练显存需求（含优化器/激活的经验值）。"""
    return 5.0 + 0.9 * float(getattr(plan, "batch", 4))


def preflight(plan: Any) -> None:
    """训练前预检：可用显存 < 需求 × 1.3 直接拒绝。"""
    import torch
    if not torch.cuda.is_available():
        raise TrainingError("CUDA 不可用，无法训练。")
    free, _total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024 ** 3
    need = estimate_vram_need_gb(plan) * 1.3
    if free_gb < need:
        raise TrainingError(
            f"可用显存 {free_gb:.1f} GB，本次训练预计需要 {need:.1f} GB。"
            f"请关闭浏览器、微信、游戏等占用显存的程序后重试。"
        )


# ---------------------------------------------------------------------------
# 配置生成
# ---------------------------------------------------------------------------
S2_TEMPLATE: dict[str, Any] = {
    "train": {
        "log_interval": 100, "eval_interval": 500, "seed": 1234, "epochs": 8,
        "learning_rate": 0.0001, "betas": [0.8, 0.99], "eps": 1e-09, "batch_size": 4,
        "fp16_run": True, "lr_decay": 0.999875, "segment_size": 20480,
        "init_lr_ratio": 1, "warmup_epochs": 0, "c_mel": 45, "c_kl": 1.0,
        "text_low_lr_rate": 0.4, "grad_ckpt": False,
        "pretrained_s2G": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
        "if_save_latest": True, "if_save_every_weights": True,
        "save_every_epoch": 1, "gpu_numbers": "0", "lora_rank": 32,
    },
    "data": {
        "max_wav_value": 32768.0, "sampling_rate": 32000, "filter_length": 2048,
        "hop_length": 640, "win_length": 2048, "n_mel_channels": 128,
        "mel_fmin": 0.0, "mel_fmax": None, "add_blank": True, "n_speakers": 300,
        "cleaned_text": True, "exp_dir": "logs/exp",
    },
    "model": {
        "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768,
        "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0.1,
        "resblock": "1", "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2, 2], "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 8, 2, 2], "n_layers_q": 3,
        "use_spectral_norm": False, "gin_channels": 512,
        "semantic_frame_rate": "25hz", "freeze_quantizer": True, "version": "v3",
    },
    "s2_ckpt_dir": "logs/exp",
    "save_weight_dir": "SoVITS_weights_v3",
    "name": "exp",
    "version": "v3",
}

S1_TEMPLATE: dict[str, Any] = {
    "train": {
        "seed": 1234, "epochs": 8, "batch_size": 4, "save_every_n_epoch": 1,
        "precision": "16-mixed", "gradient_clip": 1.0,
        "if_save_every_weights": True, "if_save_latest": False, "if_dpo": False,
        "half_weights_save_dir": "GPT_weights_v3", "exp_name": "exp",
    },
    "optimizer": {"lr": 0.01, "lr_init": 0.00001, "lr_end": 0.0001,
                  "warmup_steps": 2000, "decay_steps": 40000},
    "data": {"max_eval_sample": 8, "max_sec": 54, "num_workers": 2, "pad_val": 1024},
    "model": {
        "vocab_size": 1025, "phoneme_vocab_size": 732, "embedding_dim": 512,
        "hidden_dim": 512, "head": 16, "linear_units": 2048, "n_layer": 24,
        "dropout": 0, "EOS": 1024, "random_bert": 0,
    },
    "inference": {"top_k": 15},
    "pretrained_s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "train_semantic_path": "logs/exp/6-name2semantic.tsv",
    "train_phoneme_path": "logs/exp/2-name2text.txt",
    "output_dir": "logs/exp/logs_s1_v3",
}


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def make_s2_config(work: Path, exp: str, plan: Any, project_root: Path,
                   lora_rank: int = 32, grad_ckpt: bool = False) -> Path:
    import copy
    cfg = copy.deepcopy(S2_TEMPLATE)
    tr = cfg["train"]
    tr["epochs"] = int(plan.epochs)
    tr["batch_size"] = int(plan.batch)
    tr["grad_ckpt"] = bool(grad_ckpt)
    tr["lora_rank"] = int(lora_rank)
    tr["learning_rate"] = round(0.0001 * float(plan.lr_scale), 8)
    tr["pretrained_s2G"] = str(
        project_root / REPO_DIRNAME / "GPT_SoVITS" / "pretrained_models" / "s2Gv3.pth")
    cfg["data"]["exp_dir"] = f"logs/{exp}"
    cfg["s2_ckpt_dir"] = f"logs/{exp}"
    cfg["name"] = exp
    return _write_json(work / "s2_v3.json", cfg)


def make_s1_config(work: Path, exp: str, plan: Any, project_root: Path,
                   semantic: str, phoneme: str) -> Path:
    import copy
    cfg = copy.deepcopy(S1_TEMPLATE)
    tr = cfg["train"]
    tr["epochs"] = int(plan.epochs)
    tr["batch_size"] = int(plan.batch)
    tr["save_every_n_epoch"] = 1
    tr["exp_name"] = exp
    # Windows + torch 2.11：DataLoader 子进程会在首个 batch 前崩，改进程内读数据
    cfg["data"]["num_workers"] = 0
    cfg["pretrained_s1"] = str(
        project_root / REPO_DIRNAME / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt")
    cfg["train_semantic_path"] = semantic
    cfg["train_phoneme_path"] = phoneme
    cfg["output_dir"] = f"logs/{exp}/logs_s1_v3"
    return _write_yaml(work / "s1_v3.yaml", cfg)


# ---------------------------------------------------------------------------
# 参考音频挑选
# ---------------------------------------------------------------------------
def pick_reference(dataset_dir: str | Path, min_sec: float = 5.0,
                   max_sec: float = 10.0) -> tuple[Path, str] | None:
    """从通过质检的片段里挑 5–10 秒、SNR 最高、无 flag 的一段。"""
    meta_path = Path(dataset_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clips = [c for c in meta.get("clips", [])
             if c.get("status", "good") != "discard" and not c.get("flags")]
    if not clips:
        clips = [c for c in meta.get("clips", []) if c.get("status", "good") != "discard"]
    if not clips:
        return None
    in_range = [c for c in clips if min_sec <= float(c["duration"]) <= max_sec]
    pool = in_range or sorted(clips, key=lambda c: -abs(float(c["duration"]) - 7.0))[:1]
    best = max(pool, key=lambda c: float(c.get("snr", 0.0)))
    path = Path(dataset_dir) / best["file"]
    return path, str(best.get("text", ""))


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
@dataclass
class TrainEvent:
    stage: str
    epoch: int = 0
    total_epochs: int = 0
    loss: float | None = None
    eta_sec: float | None = None
    message: str = ""
    level: str = "info"

    def as_json(self) -> str:
        return json.dumps({
            "stage": self.stage, "epoch": self.epoch,
            "total_epochs": self.total_epochs, "loss": self.loss,
            "eta": None if self.eta_sec is None else round(self.eta_sec),
            "message": self.message, "level": self.level,
        }, ensure_ascii=False)


class TrainingRunner:
    """一次角色训练。所有阻塞操作都在调用它的线程里做（UI 侧用 QThread 包）。"""

    def __init__(self, config: Any, store: CharacterStore, character_id: str,
                 dataset_dir: str | Path, plan: Any,
                 on_event: Callable[[TrainEvent], None] | None = None):
        self.config = config
        self.store = store
        self.cid = character_id
        self.dataset_dir = Path(dataset_dir)
        self.plan = plan
        self.on_event = on_event
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self.project_root = Path(config.get("paths.gpt_sovits")).parent
        self.repo = Path(config.get("paths.gpt_sovits"))
        self.exp = f"vlab_{character_id.split('-')[0]}"
        self.work = Path(config.path_of("datasets")) / character_id / "work"
        self.log_path = Path(config.path_of("logs")) / f"train_{character_id}.log"
        self.result: dict[str, Any] = {}

    # -- 事件 ----------------------------------------------------------
    def emit(self, ev: TrainEvent) -> None:
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.with_suffix(".events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(ev.as_json() + "\n")
        except Exception:
            pass

    # -- 取消 ----------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        self._kill_current()

    def _kill_current(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- 环境 ----------------------------------------------------------
    def _env(self, extra: dict[str, str] | None = None,
             shim: bool = False) -> dict[str, str]:
        env = os.environ.copy()
        parts: list[str] = []
        if shim:
            # s1（Lightning/DDP）在本机需要运行时垫片，见 tools/s1shim/sitecustomize.py
            parts.append(str(self.project_root / "tools" / "s1shim"))
        parts += [str(self.repo), str(self.repo / "GPT_SoVITS")]
        env["PYTHONPATH"] = ";".join(parts)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["is_half"] = "True"
        env["_CUDA_VISIBLE_DEVICES"] = "0"
        env["i_part"] = "0"
        env["all_parts"] = "1"
        env["hz"] = "25hz"
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(
            self.config.training.get("cuda_alloc_conf", "expandable_segments:True"))
        if extra:
            env.update(extra)
        return env

    def _prep_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """1A/1B/1C 三个脚本都读同一组 inp_*/exp_name/opt_dir，缺一个就 NoneType。"""
        base = {
            "inp_text": str(self.work / "train.list"),
            "inp_wav_dir": str(self.work / "wav24k"),
            "exp_name": self.exp,
            "opt_dir": f"logs/{self.exp}",
            "version": "v3",
        }
        if extra:
            base.update(extra)
        return self._env(base)

    # -- 子进程执行 ----------------------------------------------------
    def _run(self, cmd: list[str], stage: str, env: dict[str, str],
             total_epochs: int = 0, quiet: bool = False) -> int:
        if self.cancelled:
            raise TrainingCancelled("训练已取消")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("a", encoding="utf-8", errors="replace")
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} [{stage}] "
                  f"{' '.join(cmd)}\n")
        log.flush()
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd, cwd=str(self.repo), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, creationflags=creation,
        )
        self._proc = proc
        t0 = time.time()
        epoch = 0
        loss: float | None = None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if not line.strip():
                    continue
                m = _EPOCH_RE.search(line)
                if m:
                    new_epoch = int(m.group(1))
                    if new_epoch != epoch:
                        epoch = new_epoch
                        eta = None
                        if total_epochs > 0 and epoch > 0:
                            elapsed = time.time() - t0
                            eta = elapsed / epoch * max(1, total_epochs - epoch)
                        self.emit(TrainEvent(stage=stage, epoch=epoch,
                                             total_epochs=total_epochs,
                                             eta_sec=eta,
                                             message=line.strip()[:200]))
                lm = _LOSS_RE.search(line)
                if lm:
                    try:
                        loss = float(lm.group(1))
                    except ValueError:
                        pass
                if self.cancelled:
                    self._kill_current()
                    raise TrainingCancelled("训练已取消")
        finally:
            proc.wait()
            log.write(f"===== [{stage}] exit={proc.returncode} "
                      f"{time.time() - t0:.1f}s\n")
            log.close()
            self._proc = None
        if self.cancelled:
            # 取消时子进程是被 taskkill 干掉的，退出码没有意义
            raise TrainingCancelled("训练已取消")
        return int(proc.returncode or 0)

    # -- 主流程 --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        store = self.store
        store.set_status(self.cid, "training")
        try:
            self._prepare()
            self._prep_text()
            self._prep_hubert()
            self._prep_semantic()
            s2 = self._train_s2()
            s1 = self._train_s1()
            ref = self._finalize(s2, s1)
            self.result.update({"sovits": str(s2), "gpt": str(s1),
                                "reference": str(ref or "")})
            store.set_status(self.cid, "ready")
            self.emit(TrainEvent(stage="done", message="训练完成", level="ok"))
            return self.result
        except TrainingCancelled:
            store.set_status(self.cid, "draft")
            self.emit(TrainEvent(stage="cancelled", message="已取消，临时文件已清理",
                                 level="warn"))
            self._cleanup_temp()
            raise
        except Exception as exc:
            store.set_status(self.cid, "failed")
            self.emit(TrainEvent(stage="failed", message=str(exc), level="error"))
            self._cleanup_temp()          # 只清工作目录，保留训练日志供排查
            raise

    # -- 准备 ----------------------------------------------------------
    def _prepare(self) -> None:
        preflight(self.plan)
        # 每次重跑都从干净的上游目录开始，避免上次失败的半成品干扰
        stale = self.repo / "logs" / self.exp
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
        self.work.mkdir(parents=True, exist_ok=True)
        wav_dir = self.work / "wav24k"
        wav_dir.mkdir(parents=True, exist_ok=True)
        for wav in sorted((self.dataset_dir / "clips").glob("*.wav")):
            shutil.copyfile(wav, wav_dir / wav.name)
        # 写训练用清单：绝对 wav 路径 | 说话人 | ZH | 文本（4 列竖线）
        src = self.dataset_dir / "dataset.list"
        lines = []
        if src.exists():
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) != 4:
                    continue
                rel, spk, lang, text = parts
                wav = (self.dataset_dir / rel).resolve()
                lines.append(f"{wav}|{spk}|{lang}|{text}")
        (self.work / "train.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for d in (self.repo / "logs" / self.exp, self.repo / "SoVITS_weights_v3",
                  self.repo / "GPT_weights_v3"):
            d.mkdir(parents=True, exist_ok=True)
        self.emit(TrainEvent(stage="prepare",
                             message=f"工作目录就绪，{len(lines)} 条样本", level="ok"))

    def _prep_text(self) -> None:
        env = self._prep_env({
            "bert_pretrained_dir": str(self.repo / "GPT_SoVITS" / "pretrained_models"
                                       / "chinese-roberta-wwm-ext-large"),
        })
        self.emit(TrainEvent(stage="prep_text", message="提取音素与 BERT 特征…"))
        code = self._run([PY, "-s", r"GPT_SoVITS\prepare_datasets\1-get-text.py"],
                         "prep_text", env)
        self._expect(code, "文本/音素提取")
        src = self.repo / "logs" / self.exp / "2-name2text-0.txt"
        dst = self.repo / "logs" / self.exp / "2-name2text.txt"
        if src.exists():
            src.replace(dst)
        if not dst.exists():
            raise TrainingError("音素文件 2-name2text.txt 未生成，请查看日志")

    def _prep_hubert(self) -> None:
        env = self._prep_env({
            "cnhubert_base_dir": str(self.repo / "GPT_SoVITS" / "pretrained_models"
                                     / "chinese-hubert-base"),
        })
        self.emit(TrainEvent(stage="prep_hubert", message="提取 HuBERT 特征与 32k 音频…"))
        code = self._run([PY, "-s", r"GPT_SoVITS\prepare_datasets\2-get-hubert-wav32k.py"],
                         "prep_hubert", env)
        self._expect(code, "HuBERT 特征提取")

    def _prep_semantic(self) -> None:
        env = self._prep_env({
            "pretrained_s2G": str(self.repo / "GPT_SoVITS" / "pretrained_models"
                                  / "s2Gv3.pth"),
            "s2config_path": str(self.repo / "GPT_SoVITS" / "configs" / "s2.json"),
        })
        self.emit(TrainEvent(stage="prep_semantic", message="提取语义 token…"))
        code = self._run([PY, "-s", r"GPT_SoVITS\prepare_datasets\3-get-semantic.py"],
                         "prep_semantic", env)
        self._expect(code, "语义 token 提取")
        d = self.repo / "logs" / self.exp
        part = d / "6-name2semantic-0.tsv"
        merged = d / "6-name2semantic.tsv"
        if part.exists():
            body = part.read_text(encoding="utf-8")
            merged.write_text("item_name\tsemantic_audio\n" + body.strip("\n") + "\n",
                              encoding="utf-8")
            part.unlink()
        if not merged.exists():
            raise TrainingError("语义文件 6-name2semantic.tsv 未生成，请查看日志")

    # -- 训练 ----------------------------------------------------------
    def _train_s2(self) -> Path:
        attempt = 0
        batch = self.plan.batch
        while True:
            attempt += 1
            s2_cfg = make_s2_config(self.work, self.exp, self.plan, self.project_root,
                                    grad_ckpt=(self.plan.vram_bucket <= 16))
            # 覆盖 batch（config 里已写，这里保持一致）
            data = json.loads(s2_cfg.read_text(encoding="utf-8"))
            data["train"]["batch_size"] = batch
            s2_cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.emit(TrainEvent(stage="sovits", epoch=0, total_epochs=self.plan.epochs,
                                 message=f"SoVITS 训练开始（batch={batch}，第 {attempt} 次尝试）"))
            code = self._run([PY, "-s", r"GPT_SoVITS\s2_train_v3_lora.py",
                              "--config", str(s2_cfg)],
                             "sovits", self._env(), total_epochs=self.plan.epochs)
            if code == 0:
                break
            if self._last_log_has_oom() and attempt <= int(self.config.training.get("oom_retries", 3)):
                batch = max(1, batch // 2)
                self.emit(TrainEvent(stage="sovits", level="warn",
                                     message=f"显存不足，batch 减半到 {batch} 后重试"))
                self._cleanup_temp(keep_data=True)
                continue
            raise TrainingError(f"SoVITS 训练失败（exit={code}），日志见 {self.log_path}")
        weight = self._latest(self.repo / "SoVITS_weights_v3", f"{self.exp}_*.pth")
        if not weight:
            raise TrainingError("SoVITS 训练结束但没有产出权重文件（请检查 "
                                "SoVITS_weights_v3 目录是否存在）")
        self._prune(weight.parent, f"{self.exp}_*.pth")
        return weight

    def _train_s1(self) -> Path:
        d = self.repo / "logs" / self.exp
        cfg = make_s1_config(self.work, self.exp, self.plan, self.project_root,
                             semantic=f"logs/{self.exp}/6-name2semantic.tsv",
                             phoneme=f"logs/{self.exp}/2-name2text.txt")
        self.emit(TrainEvent(stage="gpt", epoch=0, total_epochs=self.plan.epochs,
                             message="GPT 训练开始"))
        code = self._run([PY, "-s", r"GPT_SoVITS\s1_train.py", "--config_file", str(cfg)],
                         "gpt", self._env(shim=True), total_epochs=self.plan.epochs)
        if code != 0:
            raise TrainingError(f"GPT 训练失败（exit={code}），日志见 {self.log_path}")
        weight = self._latest(self.repo / "GPT_weights_v3", f"{self.exp}-e*.ckpt")
        if not weight:
            raise TrainingError("GPT 训练结束但没有产出权重文件")
        self._prune(weight.parent, f"{self.exp}-e*.ckpt")
        return weight

    def _finalize(self, s2: Path, s1: Path) -> Path | None:
        # 角色目录里留一份数据集，方便审计与 .vcp 打包（spec: characters/<id>/dataset/）
        chardir = self.store.dir_of(self.cid)
        target = chardir / "dataset"
        try:
            shutil.copytree(self.dataset_dir, target, dirs_exist_ok=True)
        except OSError:
            pass
        picked = pick_reference(self.dataset_dir)
        ref_path = None
        if picked:
            src, text = picked
            chardir = self.store.dir_of(self.cid)
            ref_path = chardir / "reference.wav"
            shutil.copyfile(src, ref_path)
            (chardir / "reference.txt").write_text(text.strip() + "\n", encoding="utf-8")
            self.store.set_reference(self.cid, ref_path, text)
            self.emit(TrainEvent(stage="reference",
                                 message=f"参考音频已选定：{src.name}（{text[:12]}…）"))
        self.store.set_voice(self.cid, str(s1), str(s2),
                             "reference.wav",
                             (self.store.dir_of(self.cid) / "reference.txt").read_text(
                                 encoding="utf-8").strip() if ref_path else "")
        # 拷一份 checkpoint 进角色目录，便于 .vcp 导出
        ck = self.store.dir_of(self.cid) / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        for p in (s1, s2):
            if p.exists():
                shutil.copyfile(p, ck / p.name)
        self._cleanup_temp()
        return ref_path

    # -- 工具 ----------------------------------------------------------
    @staticmethod
    def _expect(code: int, what: str) -> None:
        if code != 0:
            raise TrainingError(f"{what}失败（exit={code}），请查看训练日志")

    @staticmethod
    def _latest(directory: Path, pattern: str) -> Path | None:
        if not directory.exists():
            return None
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _prune(self, directory: Path, pattern: str) -> None:
        keep = int(self.config.training.get("save_recent", 3))
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def _last_log_has_oom(self) -> bool:
        try:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            return bool(_OOM_RE.search(tail))
        except Exception:
            return False

    def _cleanup_temp(self, keep_data: bool = False) -> None:
        """清训练中间产物：上游 logs/<exp>、工作目录（wav24k/list）。"""
        for path in (self.repo / "logs" / self.exp,):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        if not keep_data and self.work.exists():
            shutil.rmtree(self.work, ignore_errors=True)


def run_benchmark(config: Any, cid: str, gpt: str, sovits: str,
                  reference: str, reference_text: str) -> dict[str, Any]:
    """训练完成后跑一次 TTFP 基准，写 results/character_ttfp.json。"""
    import json as _json
    tool = Path(config.path_of("gpt_sovits")).parent / "tools" / "bench_character.py"
    out = Path(config.path_of("results")) / "character_ttfp.json"
    cmd = [PY, str(tool), "--character", cid, "--gpt", gpt, "--sovits", sovits,
           "--reference", reference, "--reference-text", reference_text,
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=str(Path(config.path_of("gpt_sovits")).parent),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "")[-2000:]}
    try:
        return _json.loads(out.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "结果文件解析失败"}


__all__ = [
    "GpuInfo", "gpu_info", "check_hardware", "preflight", "estimate_vram_need_gb",
    "TrainingRunner", "TrainEvent", "TrainingError", "TrainingCancelled",
    "make_s2_config", "make_s1_config", "pick_reference", "run_benchmark",
]
