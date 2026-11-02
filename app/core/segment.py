"""两遍切分。

第一遍：Silero VAD 粗切（阈值 0.5 / 最小静音 0.30s / 最小语音 2.0s / 最大语音 7.0s，
        边界前 120ms 后 180ms）。
第二遍：用 ASR 词级时间戳吸附——优先标点、其次 ≥150ms 的最长停顿，绝不切在词中间；
        切点前后各留 100ms；切完 <2s 的段并回相邻段。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

from . import audio_qc
from .label import ReviewRow, Transcript, Transcriber, WordSpan, cjk_len, compute_flags, mark_duplicates

PUNCT = set("，。！？；、…,.!?;")


@dataclass
class Region:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class SegmentResult:
    dataset_dir: Path
    clips: list[Path] = field(default_factory=list)
    rows: list[ReviewRow] = field(default_factory=list)
    transcript: Transcript | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def dataset_list(self) -> Path:
        return self.dataset_dir / "dataset.list"

    @property
    def metadata(self) -> Path:
        return self.dataset_dir / "metadata.json"


# ---------------------------------------------------------------------------
# 第一遍：VAD
# ---------------------------------------------------------------------------
_silero = None


def _silero_model():
    global _silero
    if _silero is None:
        from silero_vad import load_silero_vad
        _silero = load_silero_vad()
    return _silero


def vad_regions(y: np.ndarray, sr: int, cfg: dict[str, Any]) -> list[Region]:
    from silero_vad import get_speech_timestamps

    y16 = audio_qc.resample(y, sr, 16000) if sr != 16000 else y
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(y16)),
        _silero_model(),
        sampling_rate=16000,
        threshold=float(cfg.get("vad_threshold", 0.5)),
        min_silence_duration_ms=int(cfg.get("min_silence_ms", 300)),
        min_speech_duration_ms=int(float(cfg.get("min_speech_sec", 2.0)) * 1000),
        max_speech_duration_s=float(cfg.get("max_speech_sec", 7.0)),
        speech_pad_ms=0,
        return_seconds=True,
    )
    pad_h = float(cfg.get("pad_head_ms", 120)) / 1000.0
    pad_t = float(cfg.get("pad_tail_ms", 180)) / 1000.0
    total = y.size / sr
    regions: list[Region] = []
    for item in ts:
        s = max(0.0, float(item["start"]) - pad_h)
        e = min(total, float(item["end"]) + pad_t)
        if e - s >= 0.5:
            regions.append(Region(s, e))
    return _merge_overlaps(regions)


def _merge_overlaps(regions: list[Region]) -> list[Region]:
    if not regions:
        return []
    out = [Region(regions[0].start, regions[0].end)]
    for r in regions[1:]:
        if r.start <= out[-1].end:
            out[-1].end = max(out[-1].end, r.end)
        else:
            out.append(Region(r.start, r.end))
    return out


# ---------------------------------------------------------------------------
# 第二遍：词边界吸附
# ---------------------------------------------------------------------------
def _gaps(words: list[WordSpan]) -> list[tuple[float, int]]:
    """返回 (切点时间, 质量)。质量 2=标点后，1=停顿≥150ms，0=词间窄缝。"""
    out: list[tuple[float, int]] = []
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        gap = b.start - a.end
        if gap < 0:
            continue
        quality = 0
        if a.text and a.text[-1] in PUNCT:
            quality = 2
        elif gap >= 0.150:
            quality = 1
        elif gap >= 0.05:
            quality = 0
        else:
            continue
        out.append(((a.end + b.start) / 2.0, quality))
    return out


def _snap(t: float, gaps: list[tuple[float, int]], tol: float = 0.30,
          min_quality: int = 1) -> float | None:
    """在 t 附近找最合适的词间切点：先看标点，再看停顿长度，最后看距离。"""
    cands = [(gt, q, abs(gt - t)) for gt, q in gaps
             if q >= min_quality and abs(gt - t) <= tol]
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[1], c[2]))
    return cands[0][0]


def snap_regions(regions: list[Region], words: list[WordSpan],
                 cfg: dict[str, Any]) -> tuple[list[Region], list[str]]:
    notes: list[str] = []
    if not words:
        notes.append("没有拿到词级时间戳，第二遍吸附被跳过（只保留 VAD 边界）。")
        return regions, notes

    gaps = _gaps(words)
    if not gaps:
        notes.append("词之间没有可用停顿，第二遍吸附被跳过。")
        return regions, notes

    pad = float(cfg.get("snap_pad_ms", 100)) / 1000.0
    tol = float(cfg.get("snap_tol_ms", 300)) / 1000.0
    total = words[-1].end
    out: list[Region] = []
    cursor = 0.0
    snapped = 0
    for r in regions:
        s = _snap(r.start, gaps, tol) if r.start > 0.05 else 0.0
        e = _snap(r.end, gaps, tol) if r.end < total - 0.05 else min(total, r.end)
        s = r.start if s is None else s
        e = r.end if e is None else e
        if e - s < 0.6:
            s, e = r.start, r.end
        # 单调不重叠，避免相邻段被吸附到同一处后粘成一整段
        s = max(s, cursor + 0.02, 0.0)
        e = max(e, s + 0.5)
        s2 = max(0.0, s - pad) if not out else max(s - pad, out[-1].end + 0.02)
        e2 = min(total, e + pad)
        if e2 - s2 < 0.3:
            continue
        out.append(Region(s2, e2))
        cursor = e2
        snapped += 1

    merged = _merge_short(out, float(cfg.get("merge_below_sec", 2.0)))
    if len(merged) != len(out):
        notes.append(f"有 {len(out) - len(merged)} 个不足 2 秒的片段并回了相邻段。")
    if snapped != len(regions):
        notes.append(f"{len(regions) - snapped} 个片段边界没吸附成功，保留了 VAD 原始边界。")
    max_sec = float(cfg.get("max_speech_sec", 7.0))
    before = len(merged)
    merged = _split_long(merged, gaps, max_sec,
                         float(cfg.get("merge_below_sec", 2.0)))
    if len(merged) != before:
        notes.append(f"有 {len(merged) - before} 个超长片段被拆到 {max_sec:.0f} 秒以内。")
    return merged, notes


def _split_long(regions: list[Region], gaps: list[tuple[float, int]],
                max_sec: float, min_sec: float) -> list[Region]:
    """VAD 的 padding 会让相邻段粘成超长段，这里按词间停顿重新拆开。"""
    out: list[Region] = []
    for r in regions:
        pieces = [Region(r.start, r.end)]
        guard = 0
        while guard < 64:
            guard += 1
            idx = next((i for i, p in enumerate(pieces) if p.dur > max_sec + 0.2), None)
            if idx is None:
                break
            p = pieces[idx]
            lo, hi = p.start + min_sec, p.end - min_sec
            target = p.start + max_sec
            cands = [(gt, q) for gt, q in gaps if lo <= gt <= hi]
            if not cands:
                cut = min(p.end - 0.5, target)
            else:
                # 取离目标切点最近的那个词间隙（标点优先仅在距离相近时起作用）
                cands.sort(key=lambda c: (abs(c[0] - target), -c[1]))
                cut = cands[0][0]
            if cut <= p.start + 0.5 or cut >= p.end - 0.5:
                break
            pieces[idx:idx + 1] = [Region(p.start, cut), Region(cut, p.end)]
        out.extend(pieces)
    return out


def _merge_short(regions: list[Region], min_sec: float) -> list[Region]:
    if len(regions) <= 1:
        return regions
    work = [Region(r.start, r.end) for r in regions]
    changed = True
    while changed and len(work) > 1:
        changed = False
        for i, r in enumerate(work):
            if r.dur >= min_sec:
                continue
            # 并到间隙更小的邻居
            left = work[i - 1] if i > 0 else None
            right = work[i + 1] if i < len(work) - 1 else None
            if left and (not right or (r.start - left.end) <= (right.start - r.end)):
                left.end = r.end
                work.pop(i)
            elif right:
                right.start = r.start
                work.pop(i)
            changed = True
            break
    return work


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def segment_file(wav: str | Path, config: Any, dataset_dir: str | Path,
                 transcriber: Transcriber | None = None,
                 character_id: str = "preview",
                 progress=None) -> SegmentResult:
    """切分 → 落 clips → 打标 → 写 dataset.list / metadata.json。"""
    cfg = config.segment
    lcfg = config.label_cfg
    dataset_dir = Path(dataset_dir)
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # 统一到 48k / -23 LUFS（与分析同一套 DSP）
    y48, sr48, _ = audio_qc.process_to_48k(wav, config.audio)

    def tick(msg: str) -> None:
        if progress:
            progress(msg)

    tick("第一遍：VAD 粗切…")
    regions = vad_regions(y48, sr48, cfg)

    tick("识别中：取词级时间戳…")
    if transcriber is None:
        transcriber = Transcriber(lcfg, device="cuda")
    tr = transcriber.transcribe(wav, with_words=True)

    tick("第二遍：吸附到词边界…")
    regions, notes = snap_regions(regions, tr.words, cfg)

    # 切 24k / 24bit
    y24 = audio_qc.resample(y48, sr48, int(config.audio.get("clip_sr", 24000)))
    sr24 = int(config.audio.get("clip_sr", 24000))
    fade_ms = float(cfg.get("fade_ms", 10))

    rows: list[ReviewRow] = []
    for i, r in enumerate(regions, start=1):
        s = max(0, int(round(r.start * sr24)))
        e = min(y24.size, int(round(r.end * sr24)))
        if e - s < int(0.3 * sr24):
            continue
        chunk = audio_qc.fade(y24[s:e], sr24, fade_ms)
        name = f"{i:04d}.wav"
        path = clips_dir / name
        audio_qc.save_wav(path, chunk, sr24, subtype="PCM_24")

        dur = chunk.size / sr24
        peak = audio_qc.true_peak_dbfs(chunk)
        snr = audio_qc.estimate_snr(chunk, sr24)
        lufs = audio_qc.integrated_lufs(chunk, sr24)

        tick(f"识别片段 {i}/{len(regions)}…")
        sub = transcriber.transcribe(path, with_words=False)
        raw = (sub.text or "").strip()
        text = raw
        row = ReviewRow(
            index=i,
            clip=f"clips/{name}",
            duration=round(dur, 3),
            text_raw=raw,
            text=text,
            peak_dbfs=round(peak, 2),
            snr_db=round(snr, 2),
            lufs=round(lufs, 2),
            asr_conf=1.0 if sub.logprob is None else float(sub.logprob),
        )
        row.flags = compute_flags(row, lcfg)
        rows.append(row)

    # 归一化后的建议文本 + 重复检测
    from .label import normalize_text
    for row in rows:
        row.text = normalize_text(row.text_raw, lcfg)
    mark_duplicates(rows, float(lcfg.get("dup_similarity", 0.9)))

    result = SegmentResult(dataset_dir=dataset_dir, rows=rows,
                           transcript=tr, warnings=list(notes))
    write_dataset(result, character_id=character_id)
    _distribution_report(result, cfg, notes)
    return result


def _distribution_report(result: SegmentResult, cfg: dict[str, Any],
                         notes: list[str]) -> None:
    rows = result.rows
    if not rows:
        notes.append("没有切出任何片段，请检查素材是否有连续语音。")
        return
    durs = [r.duration for r in rows]
    avg = sum(durs) / len(durs)
    lo = int(cfg.get("target_min_clips", 8))
    hi = int(cfg.get("target_max_clips", 20))
    if not (lo <= len(rows) <= hi):
        notes.append(
            f"片段数 {len(rows)} 不在目标区间 {lo}–{hi}（素材越短段数越少，属正常）。"
        )
    if avg < 4.0 or avg > 5.0:
        notes.append(f"平均段长 {avg:.2f}s，目标是 4–5s。")
    if max(durs) > float(cfg.get("max_speech_sec", 7.0)) + 0.5:
        notes.append(f"最长片段 {max(durs):.2f}s，超过 7s 上限。")


def write_dataset(result: SegmentResult, character_id: str = "preview") -> None:
    """写 dataset.list 与 metadata.json（两者都可被人工改）。"""
    lines = []
    meta: dict[str, Any] = {"character_id": character_id, "clips": []}
    for row in result.rows:
        if row.status == "discard":
            continue
        lines.append(f"{row.clip}|{character_id}|{row.lang}|{row.text}")
        meta["clips"].append({
            "index": row.index,
            "file": row.clip,
            "duration": row.duration,
            "snr": row.snr_db,
            "peak": row.peak_dbfs,
            "lufs": row.lufs,
            "asr_conf": row.asr_conf,
            "flags": row.flags,
            "status": row.status,
            "text": row.text,
            "text_raw": row.text_raw,
        })
    meta["count"] = len(meta["clips"])
    meta["total_sec"] = round(sum(r.duration for r in result.rows if r.status != "discard"), 2)
    result.dataset_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result.metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rows(dataset_dir: str | Path) -> list[ReviewRow]:
    meta = json.loads((Path(dataset_dir) / "metadata.json").read_text(encoding="utf-8"))
    rows = []
    for c in meta.get("clips", []):
        rows.append(ReviewRow(
            index=c["index"], clip=c["file"], duration=c["duration"],
            text_raw=c.get("text_raw", c.get("text", "")), text=c.get("text", ""),
            status=c.get("status", "good"), flags=c.get("flags", []),
            peak_dbfs=c.get("peak", 0.0), snr_db=c.get("snr", 0.0),
            lufs=c.get("lufs", 0.0), asr_conf=c.get("asr_conf", 1.0),
        ))
    return rows


__all__ = [
    "Region", "SegmentResult", "vad_regions", "snap_regions", "segment_file",
    "write_dataset", "load_rows", "PUNCT",
]
