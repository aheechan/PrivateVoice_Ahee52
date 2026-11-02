"""音频导入与质量契约。

流程固定为：
    解码 float32 → 下混单声道 → soxr_vhq 重采样 48k → 去直流 → 60Hz 高通
    → 响度归一到 -23 LUFS（峰值限 -1 dBTP）

结论落 `qc.json`；不合格必须给中文原因和可执行的修改建议。
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soxr
import soundfile as sf
from scipy import signal

SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".flac"}
FRAME_SEC = 0.025
HOP_SEC = 0.010


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Check:
    key: str
    label: str
    level: str          # ok | warn | error | skip
    value: Any = None
    message: str = ""
    advice: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QCResult:
    source: str
    ok: bool
    status: str                      # ok | warn | error
    checks: list[Check] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    lufs: float = 0.0
    peak_dbfs: float = 0.0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    clip_ratio: float = 0.0
    speakers: int | None = None
    speaker_scores: list[float] = field(default_factory=list)
    truncated: bool = False
    processed_sec: float = 0.0

    def messages(self, level: str | None = None) -> list[str]:
        return [
            c.message for c in self.checks
            if c.message and (level is None or c.level == level)
        ]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.as_dict() for c in self.checks]
        return d


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------
def probe(path: str | Path) -> dict[str, Any]:
    """只读元信息，不做解码。"""
    path = Path(path)
    info: dict[str, Any] = {"path": str(path), "name": path.name, "suffix": path.suffix.lower()}
    try:
        si = sf.info(str(path))
        info.update(
            samplerate=si.samplerate,
            channels=si.channels,
            frames=si.frames,
            duration=round(si.frames / si.samplerate, 3),
            subtype=si.subtype,
            format=si.format,
        )
        return info
    except Exception:
        pass
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError(f"无法读取音频信息：{path}")
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,duration,codec_name",
         "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    sr = int(st.get("sample_rate") or 0)
    dur = float(st.get("duration") or 0.0)
    info.update(
        samplerate=sr,
        channels=int(st.get("channels") or 1),
        duration=round(dur, 3),
        frames=int(dur * sr) if sr else 0,
        subtype=st.get("codec_name"),
        format="ffprobe",
    )
    return info


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("缺少 ffmpeg，无法解码 m4a/mp3")
    with tempfile.TemporaryDirectory(prefix="vlab-dec-") as tmp:
        wav = Path(tmp) / "decoded.wav"
        cmd = [exe, "-v", "error", "-y", "-i", str(path),
               "-map", "0:a:0", "-c:a", "pcm_f32le", str(wav)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError(f"ffmpeg 解码失败：{r.stderr.strip()[:300]}")
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return data, sr


def decode(path: str | Path) -> tuple[np.ndarray, int]:
    """返回 (n, 2) float32 立体声棋盘（单声道会复制两列）与原始采样率。"""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise RuntimeError(
            f"不支持的格式 {path.suffix}，仅支持 {', '.join(sorted(SUPPORTED_EXT))}"
        )
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        data, sr = _decode_with_ffmpeg(path)
    if data.size == 0:
        raise RuntimeError("音频为空（0 个采样点）")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return data.astype(np.float32, copy=False), int(sr)


# ---------------------------------------------------------------------------
# DSP 处理
# ---------------------------------------------------------------------------
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    y = soxr.resample(x.astype(np.float32, copy=False), sr_in, sr_out, quality="VHQ")
    return np.asarray(y, dtype=np.float32)


def remove_dc(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))).astype(np.float32, copy=False)


def highpass(x: np.ndarray, sr: int, hz: float = 60.0, order: int = 2) -> np.ndarray:
    if x.size < 32:
        return x
    sos = signal.butter(order, hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32, copy=False)


def _db(x: np.ndarray | float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def true_peak_dbfs(x: np.ndarray) -> float:
    """4x 线性插值近似真峰值（对 -1 dBTP 判定足够）。"""
    if x.size == 0:
        return -np.inf
    if x.size < 8:
        return _db(np.max(np.abs(x)))
    up = signal.resample_poly(x, 4, 1)
    return _db(np.max(np.abs(up)))


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    import pyloudnorm
    if x.size < sr * 0.4:
        # 太短，用 RMS 近似（BS.1770 至少要 400ms 块）
        return _db(np.sqrt(np.mean(np.square(x)))) - 0.691
    meter = pyloudnorm.Meter(sr, block_size=0.400, overlap=0.75)
    try:
        return float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return _db(np.sqrt(np.mean(np.square(x)))) - 0.691


def limit_peak(x: np.ndarray, sr: int, ceiling_dbfs: float = -1.0,
               lookahead_ms: float = 5.0) -> np.ndarray:
    """前瞻限幅：只压超过 ceiling 的峰，不改变整体响度观感。"""
    ceiling = 10 ** (ceiling_dbfs / 20.0)
    if x.size == 0:
        return x
    la = max(1, int(sr * lookahead_ms / 1000.0))
    absx = np.abs(x)
    # 滑动最大值（前瞻窗口）
    padded = np.pad(absx, (la, la), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * la + 1)
    env = windows.max(axis=1)
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
    # 平滑增益，避免抽吸感
    k = max(1, int(sr * 0.002))
    kernel = np.ones(k, dtype=np.float32) / k
    gain = np.convolve(gain, kernel, mode="same").astype(np.float32)
    gain = np.minimum(gain, 1.0)
    out = (x * gain).astype(np.float32)
    # 兜底硬限幅（浮点误差）
    np.clip(out, -ceiling, ceiling, out=out)
    return out


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -23.0,
                       peak_dbtp: float = -1.0) -> tuple[np.ndarray, float, float]:
    lufs = integrated_lufs(x, sr)
    gain_db = target_lufs - lufs
    # 限制增益范围，避免把静音素材放大成噪声墙
    gain_db = float(np.clip(gain_db, -30.0, 30.0))
    y = (x * (10 ** (gain_db / 20.0))).astype(np.float32)
    if true_peak_dbfs(y) > peak_dbtp:
        y = limit_peak(y, sr, peak_dbtp)
    return y, integrated_lufs(y, sr), true_peak_dbfs(y)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def frame_rms_db(x: np.ndarray, sr: int,
                 frame_sec: float = FRAME_SEC, hop_sec: float = HOP_SEC) -> np.ndarray:
    fl = max(2, int(sr * frame_sec))
    hp = max(1, int(sr * hop_sec))
    if x.size < fl:
        return np.array([_db(np.sqrt(np.mean(np.square(x)) + 1e-12))])
    n = 1 + (x.size - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def estimate_snr(x: np.ndarray, sr: int) -> float:
    """噪声底 = 10 分位帧能量，语音 = 95 分位。"""
    d = frame_rms_db(x, sr)
    if d.size < 5:
        return 0.0
    noise = float(np.percentile(d, 10))
    speech = float(np.percentile(d, 95))
    return speech - noise


def silence_ratio(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    d = frame_rms_db(x, sr)
    if d.size == 0:
        return 1.0
    return float(np.mean(d < floor_db))


def clip_ratio(x: np.ndarray, level: float = 0.99) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= level))


# ---------------------------------------------------------------------------
# 多人声检测（best-effort：campplus 说话人向量 + 连通分量聚类）
# ---------------------------------------------------------------------------
def estimate_speakers(x: np.ndarray, sr: int, max_chunks: int = 6,
                      threshold: float = 0.62) -> tuple[int | None, list[float]]:
    """返回 (说话人数, 两两余弦相似度)。模型不可用时返回 (None, [])。

    用 funasr 的 campplus 说话人向量：取能量最高的若干 3s 窗口求 embedding，
    余弦相似度低于阈值就认为换人，最后数连通分量。
    """
    try:
        from funasr import AutoModel
    except Exception:
        return None, []

    mono = x if x.ndim == 1 else to_mono(x)
    mono16 = mono if sr == 16000 else resample(mono, sr, 16000)
    dur = mono16.size / 16000.0
    if dur < 6.0:
        return None, []

    win = int(3.0 * 16000)
    hop = int(1.0 * 16000)
    candidates: list[tuple[float, int]] = []
    for start in range(0, max(1, mono16.size - win + 1), hop):
        seg = mono16[start:start + win]
        candidates.append((float(np.sqrt(np.mean(np.square(seg)) + 1e-12)), start))
    candidates.sort(reverse=True)
    picks = sorted({s for _, s in candidates[:max_chunks]})
    if len(picks) < 2:
        return None, []

    try:
        model = _speaker_model()
        chunks = [np.ascontiguousarray(mono16[s:s + win], dtype=np.float32)
                  for s in picks]
        out = model.generate(input=chunks)
        embeds = []
        for item in out:
            emb = item.get("spk_embedding")
            if emb is None:
                return None, []
            vec = np.asarray(emb.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            n = np.linalg.norm(vec)
            embeds.append(vec / (n if n > 0 else 1.0))
        scores: list[float] = []
        parent = list(range(len(embeds)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sim = float(np.dot(embeds[i], embeds[j]))
                scores.append(round(sim, 4))
                if sim >= threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        return len({find(i) for i in range(len(embeds))}), scores
    except Exception:
        return None, []


_SPEAKER_MODEL = None


def _speaker_model():
    global _SPEAKER_MODEL
    if _SPEAKER_MODEL is None:
        from funasr import AutoModel
        _SPEAKER_MODEL = AutoModel(
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            device="cpu", disable_update=True, hub="ms",
        )
    return _SPEAKER_MODEL


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_to_48k(path: str | Path, cfg: dict[str, Any]) -> tuple[np.ndarray, int, dict]:
    """解码 + 统一到 48k 单声道 -23 LUFS。返回 (samples, sr, info)。"""
    info = probe(path)
    data, sr_in = decode(path)
    mono = to_mono(data)
    y = resample(mono, sr_in, int(cfg.get("target_sr", 48000)))
    y = remove_dc(y)
    y = highpass(y, int(cfg["target_sr"]), float(cfg.get("hp_hz", 60.0)))
    y, lufs, peak = normalize_loudness(
        y, int(cfg["target_sr"]),
        float(cfg.get("target_lufs", -23.0)),
        float(cfg.get("peak_dbtp", -1.0)),
    )
    info.update(sr_in=sr_in, lufs=round(lufs, 2), peak_dbfs=round(peak, 2))
    return y, int(cfg["target_sr"]), info


def run_qc(path: str | Path, config: Any,
           check_speakers: bool = True) -> QCResult:
    """完整质检。`config` 为 `app.config.Config`；不抛异常（解码失败除外）。"""
    path = Path(path)
    audio_cfg = getattr(config, "audio", None) or config["audio"]
    cfg = getattr(config, "qc", None) or config["qc"]
    info = probe(path)
    raw, sr_in = decode(path)
    raw_mono = to_mono(raw)

    # 削波必须在归一化之前测，否则证据被抹掉
    cr = clip_ratio(raw_mono, float(cfg.get("clip_level", 0.99)))

    y, sr, pinfo = process_to_48k(path, audio_cfg)
    info.update(pinfo)

    # 超过上限就截断
    truncate_sec = float(cfg.get("truncate_sec", 90.0))
    truncated = False
    if y.size / sr > truncate_sec:
        y = y[: int(truncate_sec * sr)]
        truncated = True

    dur = y.size / sr
    lufs = integrated_lufs(y, sr)
    peak = true_peak_dbfs(y)
    snr = estimate_snr(y, sr)
    sil = silence_ratio(y, sr)
    speakers, spk_scores = (estimate_speakers(y, sr) if check_speakers else (None, []))

    checks: list[Check] = []

    # 1) 总时长
    min_d = float(cfg.get("min_duration_sec", 45.0))
    pass_max = float(cfg.get("pass_max_sec", 90.0))
    if dur < min_d:
        checks.append(Check(
            "duration", "总时长", "error", round(dur, 1),
            f"素材只有 {dur:.1f} 秒，不足 {min_d:.0f} 秒，训练出来的音色会很不稳定。",
            f"请补录至少 {min_d:.0f} 秒，目标 45–90 秒；一句话一句话说，中间留 0.5 秒空隙。",
        ))
    elif dur > pass_max:
        checks.append(Check(
            "duration", "总时长", "warn", round(dur, 1),
            f"素材 {dur:.1f} 秒，超过建议上限 {pass_max:.0f} 秒，已截断到 "
            f"{min(dur, truncate_sec):.0f} 秒参与训练。",
            "截断是安全的；想用完整素材请自己在时间轴上剪成两段分别训练。",
        ))
    else:
        checks.append(Check(
            "duration", "总时长", "ok", round(dur, 1),
            f"时长 {dur:.1f} 秒，落在 45–90 秒区间。",
        ))

    # 2) 采样率
    min_sr = int(cfg.get("min_sample_rate", 44100))
    if int(info.get("sr_in") or sr_in) < min_sr:
        checks.append(Check(
            "sample_rate", "采样率", "error", sr_in,
            f"原始采样率 {sr_in} Hz 低于 {min_sr} Hz，高频细节已丢失，训练会得到闷、发虚的音色。",
            f"请用 {min_sr} Hz 或 48 kHz 重新导出/重录；手机录音请在设置里选“高质量/无损”。",
        ))
    else:
        checks.append(Check(
            "sample_rate", "采样率", "ok", sr_in,
            f"原始采样率 {sr_in} Hz，重采样到 48 kHz 后使用。",
        ))

    # 3) 削波
    warn_ratio = float(cfg.get("clip_ratio_warn", 0.001))
    if cr >= warn_ratio:
        checks.append(Check(
            "clipping", "削波", "warn", round(cr * 100, 3),
            f"有 {cr * 100:.3f}% 的采样点贴顶（≥0.99），爆音会被学进音色里。",
            "把麦克风增益降低 6–10 dB 重录；已录好的可在 Audacity 里降 6 dB 后导出。",
        ))
    else:
        checks.append(Check("clipping", "削波", "ok", round(cr * 100, 4),
                            f"贴顶采样点 {cr * 100:.4f}%，没有明显爆音。"))

    # 4) 信噪比
    min_snr = float(cfg.get("min_snr_db", 30.0))
    if snr < min_snr:
        checks.append(Check(
            "snr", "信噪比", "warn", round(snr, 1),
            f"信噪比约 {snr:.1f} dB，低于 {min_snr:.0f} dB，底噪会被一起学进去。",
            "换安静房间、关空调/风扇，麦克风离嘴 15–20 cm；或先用降噪工具处理再导入。",
        ))
    else:
        checks.append(Check("snr", "信噪比", "ok", round(snr, 1),
                            f"信噪比约 {snr:.1f} dB，够干净。"))

    # 5) 静音占比
    max_sil = float(cfg.get("max_silence_ratio", 0.20))
    if sil >= max_sil:
        checks.append(Check(
            "silence", "静音占比", "warn", round(sil * 100, 1),
            f"静音占 {sil * 100:.1f}%，超过 {max_sil * 100:.0f}%，切分时会产生大量空片段。",
            "剪掉开头结尾的长停顿，句与句之间留 0.3–0.5 秒即可。",
        ))
    else:
        checks.append(Check("silence", "静音占比", "warn" if sil >= max_sil else "ok",
                            round(sil * 100, 1),
                            f"静音占 {sil * 100:.1f}%。"))

    # 6) 多人声
    if speakers is None:
        checks.append(Check(
            "speakers", "多人声", "skip", None,
            "说话人分离模型不可用，本次跳过多人声检测。",
            "如需启用：确认能访问 modelscope 下载 campplus 模型。",
        ))
    elif speakers > 1:
        level = "error" if cfg.get("allow_multi_speaker_reject", True) else "warn"
        checks.append(Check(
            "speakers", "多人声", level, speakers,
            f"检测到约 {speakers} 个不同说话人，单人音色克隆只能用一个人的声音。",
            "只保留目标说话人的片段重录；两人对话请分别录。",
        ))
    else:
        checks.append(Check("speakers", "多人声", "ok", 1, "只检测到 1 个说话人。"))

    status = "error" if any(c.level == "error" for c in checks) else (
        "warn" if any(c.level == "warn" for c in checks) else "ok"
    )
    return QCResult(
        source=str(path),
        ok=status != "error",
        status=status,
        checks=checks,
        info=info,
        duration_sec=round(dur, 2),
        lufs=round(lufs, 2),
        peak_dbfs=round(peak, 2),
        snr_db=round(snr, 2),
        silence_ratio=round(sil, 4),
        clip_ratio=round(cr, 6),
        speakers=speakers,
        speaker_scores=spk_scores,
        truncated=truncated,
        processed_sec=round(dur, 2),
    )


def save_qc(result: QCResult, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["speaker_scores"] = getattr(result, "speaker_scores", [])
    payload["generated_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def load_qc(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_wav(path: str | Path, samples: np.ndarray, sr: int,
             subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr, subtype=subtype)
    return path


def fade(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), x.size // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


__all__ = [
    "Check", "QCResult", "SUPPORTED_EXT",
    "probe", "decode", "to_mono", "resample", "remove_dc", "highpass",
    "normalize_loudness", "integrated_lufs", "true_peak_dbfs", "limit_peak",
    "frame_rms_db", "estimate_snr", "silence_ratio", "clip_ratio",
    "estimate_speakers", "process_to_48k", "run_qc", "save_qc", "load_qc",
    "save_wav", "fade",
]
