"""切分：给定真实语音，验证段数 / 时长 / 边界。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import audio_qc, segment  # noqa: E402
from app.core.label import WordSpan  # noqa: E402

SPEECH = ROOT / "assets" / "ref.wav"
needs_speech = pytest.mark.skipif(not SPEECH.exists(), reason="缺少 assets/ref.wav")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def speech48(cfg) -> tuple[np.ndarray, int]:
    y, sr, _ = audio_qc.process_to_48k(SPEECH, cfg.audio)
    return y, sr


@needs_speech
def test_vad_finds_speech(cfg, speech48):
    y, sr = speech48
    regions = segment.vad_regions(y, sr, cfg.segment)
    assert len(regions) >= 2, "9 秒真实语音至少应切出 2 段"
    for r in regions:
        assert 1.5 <= r.dur <= 7.6, f"段长 {r.dur:.2f}s 超出 2–7s"
        assert r.end > r.start
    # 单调不重叠
    for a, b in zip(regions, regions[1:]):
        assert b.start >= a.end - 1e-6


def test_snap_to_word_gaps(cfg):
    """切点必须落在词间隙上，不能切在词中间。"""
    words = []
    t = 0.0
    for i in range(30):
        words.append(WordSpan("字", t, t + 0.18))
        t += 0.18 + (0.35 if i % 5 == 4 else 0.06)
    regions = [segment.Region(0.0, 1.0), segment.Region(1.0, 2.4), segment.Region(2.4, 3.6)]
    snapped, _notes = segment.snap_regions(regions, words, cfg.segment)
    assert snapped, "至少应保留一个片段"
    for r in snapped:
        for w in words:
            assert not (w.start + 0.02 < r.start < w.end - 0.02), "起点切进词里了"
            assert not (w.start + 0.02 < r.end < w.end - 0.02), "终点切进词里了"
        assert r.dur >= 0.3


def test_long_region_split(cfg):
    """超过 7s 的段必须被拆开，且不出现碎片。"""
    gaps = [(3.4, 1), (7.1, 1), (10.5, 1)]
    regions = [segment.Region(0.0, 10.8)]
    out = segment._split_long(regions, gaps, 7.0, 2.0)
    assert len(out) == 2
    assert all(p.dur <= 7.15 for p in out)
    assert out[0].end == pytest.approx(out[1].start)


@needs_speech
def test_segment_file_end_to_end(tmp_path, cfg):
    """完整跑一遍切分（Silero + 假转写器），检查产物文件齐全。"""
    out = tmp_path / "ds"
    dur = audio_qc.probe(SPEECH)["duration"]

    class FakeTranscriber:
        def transcribe(self, path, with_words=True):
            from app.core.label import Transcript
            words = [WordSpan("字", i * 0.25, i * 0.25 + 0.2)
                     for i in range(int(dur / 0.25))]
            return Transcript(text="测试文本。", words=words, engine="fake")

    res = segment.segment_file(SPEECH, cfg, out, transcriber=FakeTranscriber(),
                               character_id="t1")
    assert res.rows, "应切出至少一个片段"
    assert (out / "clips").exists()
    assert (out / "dataset.list").exists()
    assert (out / "metadata.json").exists()
    first = res.rows[0]
    assert first.clip.endswith(".wav")
    assert (out / first.clip).exists()
    line = (out / "dataset.list").read_text(encoding="utf-8").splitlines()[0]
    parts = line.split("|")
    assert len(parts) == 4
    assert parts[1] == "t1" and parts[2] == "ZH"
    # 片段必须是 24k / 单声道
    info = sf.info(str(out / first.clip))
    assert info.samplerate == 24000
    assert info.channels == 1
