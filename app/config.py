"""配置加载与档位派生。

原则：可调数字只在 `PROFILE_MATRIX` / `config.yaml` 出现一次，业务代码只读
`Config` 与 `TrainingPlan`，不散落魔法数字。
"""
from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

ProfileName = Literal["conservative", "balanced", "aggressive"]

# ---------------------------------------------------------------------------
# 档位表（规格表逐字落地）
# ---------------------------------------------------------------------------
PROFILE_MATRIX: dict[str, dict[str, Any]] = {
    "conservative": {
        "label": "保守",
        "epochs": 8,
        "lr_scale": 0.6,
        "batch": {24: 4, 16: 2, 12: 1},
        "grad_accum": 4,
        "max_clip_sec": 6,
        "note": "默认档。训练最慢但不炸显存，小素材不容易过拟合。",
    },
    "balanced": {
        "label": "标准",
        "epochs": 10,
        "lr_scale": 1.0,
        "batch": {24: 6, 16: 3, 12: 2},
        "grad_accum": 2,
        "max_clip_sec": 7,
        "note": "速度与质量折中，素材 ≥75s 时推荐。",
    },
    "aggressive": {
        "label": "激进",
        "epochs": 12,
        "lr_scale": 1.0,
        "batch": {24: 8, 16: 4, 12: 2},
        "grad_accum": 1,
        "max_clip_sec": 8,
        "note": "最快。显存吃满，素材偏短时容易过拟合、音色发飘。",
    },
}

#: 显存 (GB) 下界 -> bucket 名义显存
VRAM_BUCKETS: tuple[tuple[float, int], ...] = ((22.0, 24), (14.0, 16), (11.0, 12))
#: 各 bucket 的片段时长上限（秒），与档位表的 max_clip_sec 取小
VRAM_CLIP_CAP: dict[int, int] = {24: 8, 16: 6, 12: 5}
#: 低于此显存直接拒绝启动训练
MIN_VRAM_GB = 11.0
#: 预检安全系数：可用显存 < 需求 × 该系数则提示关闭其他程序
VRAM_SAFETY_FACTOR = 1.3

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "profile": "conservative",
    "paths": {
        "models": str(ROOT / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"),
        "gpt_sovits": str(ROOT / "GPT-SoVITS"),
        "characters": str(ROOT / "characters"),
        "logs": str(ROOT / "logs"),
        "datasets": str(ROOT / "datasets"),
        "assets": str(ROOT / "assets"),
        "results": str(ROOT / "results"),
        "model_cache": str(ROOT / "models"),
    },
    "training": {
        "save_recent": 3,          # 每 epoch 保存，只留最近 N 个
        "oom_retries": 3,          # CUDA OOM 时 batch 减半重跑次数上限
        "fp16": True,
        "cuda_alloc_conf": "expandable_segments:True",
        "pretrained_s1": "s1v3.ckpt",
        "pretrained_s2": "s2Gv3.pth",
    },
    "audio": {
        "input_device": None,
        "output_device": None,
        "target_sr": 48000,
        "clip_sr": 24000,
        "target_lufs": -23.0,
        "peak_dbtp": -1.0,
        "hp_hz": 60.0,
        "noise_probe_sec": 5.0,
        "level_green": [-18.0, -12.0],
        "level_yellow": [-12.0, -6.0],
    },
    "qc": {
        # 需求方 2026-11-02 把素材时长定义调整为 45–90s
        "min_duration_sec": 45.0,
        "pass_max_sec": 90.0,
        "truncate_sec": 90.0,
        "min_sample_rate": 44100,
        "clip_level": 0.99,
        "clip_ratio_warn": 0.001,
        "min_snr_db": 30.0,
        "max_silence_ratio": 0.20,
        "allow_multi_speaker_reject": True,
    },
    "segment": {
        "vad_threshold": 0.5,
        "min_silence_ms": 300,
        "min_speech_sec": 2.0,
        "max_speech_sec": 7.0,
        "pad_head_ms": 120,
        "pad_tail_ms": 180,
        "snap_pause_ms": 150,
        "snap_pad_ms": 100,
        "merge_below_sec": 2.0,
        "target_min_clips": 8,
        "target_max_clips": 20,
        "fade_ms": 10,
    },
    "label": {
        "asr_primary": "paraformer-zh",
        "asr_fallback": "whisper-large-v3",
        "punc_model": "ct-punc",
        "lang": "zh",
        "rate_slow": 2.5,        # 字/秒
        "rate_fast": 7.5,
        "logprob_warn": -0.6,
        "peak_dbfs_red": -0.5,
        "dup_similarity": 0.9,
    },
    "tts": {
        # Stage 1 实测协议，试听与复测必须一致（见 docs/stage1-baseline.md）
        "how_to_cut": "按标点符号切",
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 0.6,
        "speed": 1.1,
        "sample_steps": 4,
        "chunk_size_seconds": 0.25,
        "enable_cuda_graph": True,
        "enable_static_kv": True,
        "text_language": "中文",
        "prompt_language": "中文",
        "device": "cuda",
        "disable_bigvgan_kernel": True,
        "preview_texts": [
            "今天有点累，不过见到你就好多了。",
            "你还记得我们上次说的那件事吗？",
            "别急着走，再陪我坐一会儿吧。",
        ],
    },
    # ---------------- Stage 3：对话链路 ----------------
    "asr": {
        "model": "paraformer-zh",
        "device": "cuda",
        "trim_head_ms": 100,
        "trim_tail_ms": 100,
        "silence_floor_db": -45.0,
        "min_chars": 1,
        "max_sec": 30.0,
        "warmup": True,
    },
    "hotkey": {
        # 需求方指定 shift+a。注意：RegisterHotKey 会全局截走 Shift+A，
        # 在别的程序里打大写 A 也会触发；设置页可随时改。
        "spec": "shift+a",
        "poll_ms": 20,
        "min_hold_ms": 150,
    },
    "llm": {
        "history_turns": 12,
        "token_budget": 1500,
        "connect_timeout_ms": 3000,
        "first_token_timeout_ms": 5000,
        "total_timeout_ms": 30000,
        "disable_thinking": True,
        # 关闭思考模式：默认注入这段到 extra_body，每个 provider 可覆盖
        "thinking_extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "fallback_text": "我这会儿有点接不上话。",
        "providers": [
            {
                "id": "local-qwen35-9b",
                "name": "本地 Qwen3.5-9B",
                "base_url": "http://127.0.0.1:8082/v1",
                "api_key_enc": "",
                "model": "Qwen3.5-9B",
                "priority": 10,
                "enabled": True,
                "timeout_ms": 30000,
                "extra_body": {},
                "supports_stream": True,
            },
        ],
    },
    "player": {
        "blocksize_ms": 30,
        "sentence_gap_ms": 80,
        "limiter_dbtp": -1.0,
        "buffer_chunks": 64,
    },
    "pipeline": {
        "min_sentence_chars": 6,
        "max_sentence_chars": 40,
        "prefetch_max": 2,
        "max_tts_fail_streak": 2,
    },
    "ui": {
        "hotkey": "ctrl+alt+space",        "language": "zh",
        "autostart": False,
        "log_level": "INFO",
        "minimize_to_tray": True,
    },
    "gpu": {
        "supported_sm": [86, 89, 120],
        "min_vram_gb": 12,
        "min_compute_cap": [8, 6],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def vram_bucket(vram_gb: float) -> int | None:
    """24 / 16 / 12 三档；不足 11 GB 返回 None（不支持）。"""
    for lower, bucket in VRAM_BUCKETS:
        if vram_gb >= lower:
            return bucket
    return None


@dataclass
class TrainingPlan:
    profile: str
    vram_bucket: int
    epochs: int
    batch: int
    grad_accum: int
    lr_scale: float
    max_clip_sec: int

    @property
    def effective_batch(self) -> int:
        return self.batch * self.grad_accum

    @property
    def label(self) -> str:
        return PROFILE_MATRIX[self.profile]["label"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "profile_label": self.label,
            "vram_bucket": self.vram_bucket,
            "epochs": self.epochs,
            "batch": self.batch,
            "grad_accum": self.grad_accum,
            "lr_scale": self.lr_scale,
            "max_clip_sec": self.max_clip_sec,
            "effective_batch": self.effective_batch,
        }


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    path: Path = CONFIG_PATH

    # -- 载入 / 保存 ---------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else CONFIG_PATH
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls(data=_deep_merge(DEFAULT_CONFIG, raw), path=path)
        cfg.ensure_dirs()
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # 只写用户可见字段，避免把派生值固化
        path.write_text(
            yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    # -- 读 ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    # -- 派生 ----------------------------------------------------------
    @property
    def profile(self) -> str:
        p = self.get("profile", "conservative")
        return p if p in PROFILE_MATRIX else "conservative"

    def path_of(self, key: str) -> Path:
        return Path(self.get(f"paths.{key}"))

    def ensure_dirs(self) -> None:
        for key in ("characters", "logs", "datasets", "results", "model_cache"):
            try:
                self.path_of(key).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def plan(self, vram_gb: float, profile: str | None = None) -> TrainingPlan:
        """按档位 + 实际显存派生训练计划。"""
        name = profile or self.profile
        if name not in PROFILE_MATRIX:
            raise ValueError(f"未知档位: {name}")
        m = PROFILE_MATRIX[name]
        bucket = vram_bucket(vram_gb)
        if bucket is None:
            raise RuntimeError(
                f"可用显存 {vram_gb:.1f} GB 低于最低要求 {MIN_VRAM_GB:.0f} GB"
            )
        clip = min(int(m["max_clip_sec"]), VRAM_CLIP_CAP[bucket])
        return TrainingPlan(
            profile=name,
            vram_bucket=bucket,
            epochs=int(m["epochs"]),
            batch=int(m["batch"][bucket]),
            grad_accum=int(m["grad_accum"]),
            lr_scale=float(m["lr_scale"]),
            max_clip_sec=clip,
        )

    # -- 便捷访问 ------------------------------------------------------
    @property
    def qc(self) -> dict[str, Any]:
        return self.data["qc"]

    @property
    def audio(self) -> dict[str, Any]:
        return self.data["audio"]

    @property
    def segment(self) -> dict[str, Any]:
        return self.data["segment"]

    @property
    def label_cfg(self) -> dict[str, Any]:
        return self.data["label"]

    @property
    def ui(self) -> dict[str, Any]:
        return self.data["ui"]

    @property
    def training(self) -> dict[str, Any]:
        return self.data["training"]

    # -- 首次运行 ------------------------------------------------------
    def bootstrap(self) -> None:
        """确保 config.yaml 存在；不覆盖用户已有文件。"""
        if not self.path.exists():
            self.save()
        self.ensure_dirs()
        prompt = self.path_of("assets") / "prompts" / "persona.example.yaml"
        if not prompt.exists():
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(PERSONA_EXAMPLE, encoding="utf-8")


PERSONA_EXAMPLE = """\
# 角色人设示例。新建角色向导会把这里的内容写进 characters/<id>/profile.json。
name: 小满
avatar: avatar.png
# 人设描述：写"她是谁、怎么说话、边界在哪"，不写剧情。
persona: |
  25 岁的插画师，说话轻、句子短，习惯用"嗯"开头。
  会关心对方有没有好好吃饭，但不唠叨，不追问隐私。
  遇到不知道的事会直接说不知道，不编。
greeting: 嗯……你来啦。今天过得怎么样？
llm:
  temperature: 0.8
  max_tokens: 256
voice:
  mode: finetune
  engine: gpt-sovits-v3
system_prompt_ref: prompts/companion_system.txt
"""


__all__ = [
    "ROOT",
    "CONFIG_PATH",
    "Config",
    "TrainingPlan",
    "PROFILE_MATRIX",
    "VRAM_BUCKETS",
    "VRAM_CLIP_CAP",
    "MIN_VRAM_GB",
    "VRAM_SAFETY_FACTOR",
    "vram_bucket",
    "DEFAULT_CONFIG",
]
