"""打标四层：识别 → 归一化 → 自动质检 flag → 人工复核数据模型。

- 识别：FunASR Paraformer-zh（主）+ CT-Transformer 标点；Whisper large-v3 仅作降级。
- 归一化：结果一律可编辑，`normalize_text` 只产出“建议文本”，不锁死。
- flag：只打标不删除，最终由用户在复核表格里决定 好/可疑/丢弃。
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class WordSpan:
    text: str
    start: float          # 秒
    end: float
    score: float = 1.0


@dataclass
class Transcript:
    text: str = ""
    words: list[WordSpan] = field(default_factory=list)
    engine: str = ""
    logprob: float | None = None
    error: str = ""

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0


@dataclass
class ReviewRow:
    """复核表格的一行，UI 直接绑这个对象。"""
    index: int
    clip: str                       # 相对路径，写进 dataset.list
    duration: float
    text_raw: str
    text: str
    status: str = "good"            # good | suspect | discard
    flags: list[str] = field(default_factory=list)
    peak_dbfs: float = 0.0
    snr_db: float = 0.0
    lufs: float = 0.0
    asr_conf: float = 1.0
    lang: str = "ZH"

    @property
    def red(self) -> bool:
        return self.status == "discard"

    @property
    def yellow(self) -> bool:
        return bool(self.flags) and self.status != "discard"


FLAG_LABELS: dict[str, str] = {
    "rate_slow": "语速过慢",
    "rate_fast": "语速过快",
    "low_conf": "识别置信度低",
    "peak": "峰值过高",
    "snr": "信噪比偏低",
    "dup": "与其它片段重复",
    "short": "片段过短",
    "long": "片段过长",
    "empty": "没有识别到文本",
    "ambiguous": "文本存疑",
}

RED_FLAGS = {"peak", "short", "long", "empty"}


# ---------------------------------------------------------------------------
# 文本归一化
# ---------------------------------------------------------------------------
_CN_DIGITS = "零一二三四五六七八九"
_UNIT_MAP: tuple[tuple[str, str], ...] = (
    ("°C", "摄氏度"), ("℃", "摄氏度"), ("%", "百分之"),
    ("km/h", "公里每小时"), ("km", "公里"), ("cm", "厘米"), ("mm", "毫米"),
    ("kg", "千克"), ("mg", "毫克"), ("ml", "毫升"), ("mL", "毫升"),
    ("GB", "吉字节"), ("MB", "兆字节"), ("Hz", "赫兹"), ("kHz", "千赫兹"),
    ("min", "分钟"), ("ms", "毫秒"),
)
_EN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'’.\-]*")
_DATE_CN = re.compile(r"(\d{2,4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_DATE_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")


def _digits_cn(s: str) -> str:
    return "".join(_CN_DIGITS[int(ch)] if ch.isdigit() else ch for ch in s)


def _int_cn(n: int) -> str:
    try:
        import cn2an
        return cn2an.an2cn(int(n), "low")
    except Exception:
        return _digits_cn(str(n))


def _num_cn(token: str) -> str:
    if "." in token:
        head, _, tail = token.partition(".")
        return _int_cn(int(head)) + "点" + _digits_cn(tail)
    if len(token) >= 5:            # 长串按位读，像号码
        return _digits_cn(token)
    return _int_cn(int(token))


def normalize_text(text: str, cfg: dict[str, Any] | None = None) -> str:
    """把书面写法转成口语读法。英文单词、语气词、标点原样保留。"""
    if not text:
        return ""
    s = text.strip()
    # 全角数字/百分号先归一半角
    s = s.translate(str.maketrans("０１２３４５６７８９％", "0123456789%"))

    # 日期：2026年11月1日 / 11月1日
    def _date_full(m: re.Match) -> str:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{_digits_cn(y)}年{_int_cn(int(mo))}月{_int_cn(int(d))}日"

    def _date_md(m: re.Match) -> str:
        mo, d = m.group(1), m.group(2)
        return f"{_int_cn(int(mo))}月{_int_cn(int(d))}日"

    s = _DATE_CN.sub(_date_full, s)
    s = _DATE_MD.sub(_date_md, s)

    # 百分比：30% -> 百分之三十
    s = _PERCENT.sub(lambda m: "百分之" + _num_cn(m.group(1)), s)

    # 单位（长的先替换）
    for unit, cn in _UNIT_MAP:
        if unit == "%":
            continue
        s = re.sub(re.escape(unit) + r"(?![A-Za-z])", cn, s)

    # 独立年份：2026 -> 二零二六
    s = _YEAR.sub(lambda m: _digits_cn(m.group(1)), s)

    # 其余数字：避开英文/编号上下文
    def _num_sub(m: re.Match) -> str:
        start, end = m.span()
        before = s[start - 1] if start > 0 else ""
        after = s[end] if end < len(s) else ""
        if before.isascii() and before.isalpha():
            return m.group(0)      # GPT4 里的 4 不动
        if after.isascii() and after.isalpha():
            return m.group(0)
        return _num_cn(m.group(0))

    s = _NUM.sub(_num_sub, s)
    # 去中文之间的空格、去标点前的空格；英文单词之间的空格保留
    s = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff，。！？；：、…])", "", s)
    s = re.sub(r"[ \t]+(?=[，。！？；：、…])", "", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 识别
# ---------------------------------------------------------------------------
PUNCT_CHARS = set("，。！？；：、…—,.!?;:")


def _attach_punct(words: list["WordSpan"], punctuated: str) -> None:
    """把标点贴回对应的字符 token，好让切分知道“哪里是句读”。"""
    if not punctuated or not words:
        return
    idx = 0
    for ch in punctuated:
        if ch in PUNCT_CHARS:
            if 0 < idx <= len(words):
                words[idx - 1].text += ch
        elif not ch.isspace():
            idx += 1
            if idx > len(words):
                return
class Transcriber:
    """懒加载 ASR，进程内复用。首次调用会下载模型。"""

    def __init__(self, cfg: dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._asr = None
        self._punc = None
        self._whisper = None
        self.engine = ""

    # -- 主引擎 --------------------------------------------------------
    def _load_primary(self):
        from funasr import AutoModel
        self._asr = AutoModel(
            model=self.cfg.get("asr_primary", "paraformer-zh"),
            device=self.device,
            disable_update=True,
        )
        self._punc = AutoModel(
            model=self.cfg.get("punc_model", "ct-punc"),
            device=self.device,
            disable_update=True,
        )
        self.engine = "funasr/paraformer-zh"

    def _load_fallback(self):
        from transformers import pipeline
        self._whisper = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=0 if self.device == "cuda" else -1,
            chunk_length_s=30,
        )
        self.engine = "transformers/whisper-large-v3"

    @property
    def model(self):
        if self._asr is None:
            self._load_primary()
        return self._asr

    # -- 对外 ----------------------------------------------------------
    def transcribe(self, wav: str | Path, with_words: bool = True) -> Transcript:
        wav = str(wav)
        try:
            return self._transcribe_primary(wav, with_words)
        except Exception as exc:  # 主引擎失败 -> 降级
            try:
                return self._transcribe_fallback(wav, with_words)
            except Exception as exc2:
                return Transcript(engine="none", error=f"{exc} | fallback: {exc2}")

    def _punctuate(self, text: str) -> str:
        """CT-Transformer 只吃纯文本，返回带标点文本。失败则原样返回。"""
        if not text or self._punc is None:
            return text
        try:
            out = self._punc.generate(input=text.replace(" ", ""))
            if isinstance(out, list) and out:
                cand = out[0]
                if isinstance(cand, dict):
                    return str(cand.get("text") or text)
                return str(cand)
        except Exception:
            pass
        return text

    def _transcribe_primary(self, wav: str, with_words: bool) -> Transcript:
        res = self.model.generate(input=wav, batch_size_s=300,
                                  sentence_timestamp=True, return_raw_text=True)
        if not res:
            return Transcript(engine=self.engine, error="空结果")
        item = res[0]
        raw = (item.get("text") or "").strip()
        words: list[WordSpan] = []
        stamps = item.get("timestamp") or []
        tokens = raw.split()
        if with_words and stamps and len(stamps) == len(tokens):
            for tok, ts in zip(tokens, stamps):
                words.append(WordSpan(
                    text=tok,
                    start=float(ts[0]) / 1000.0,
                    end=float(ts[1]) / 1000.0,
                ))
        # 标点单独跑一次（paraformer 的时间戳分支不输出标点）
        text = self._punctuate(raw)
        if with_words and words:
            _attach_punct(words, text)
        return Transcript(text=text, words=words, engine=self.engine, logprob=None)

    def _transcribe_fallback(self, wav: str, with_words: bool) -> Transcript:
        if self._whisper is None:
            self._load_fallback()
        out = self._whisper(
            wav,
            return_timestamps="word" if with_words else True,
            generate_kwargs={"language": "chinese"},
        )
        text = (out.get("text") or "").strip()
        words: list[WordSpan] = []
        for ch in out.get("chunks") or []:
            ts = ch.get("timestamp") or (0.0, 0.0)
            words.append(WordSpan(text=ch.get("text", "").strip(),
                                  start=float(ts[0] or 0), end=float(ts[1] or 0)))
        return Transcript(text=text, words=words, engine=self.engine)


# ---------------------------------------------------------------------------
# 自动质检 flag
# ---------------------------------------------------------------------------
def cjk_len(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def char_speech_rate(text: str, duration: float) -> float:
    """字/秒。用汉字数除以时长（标点不算）。"""
    if duration <= 0:
        return 0.0
    return cjk_len(text) / duration


def compute_flags(row: ReviewRow, cfg: dict[str, Any],
                  duplicate_of: int | None = None) -> list[str]:
    flags: list[str] = []
    n = cjk_len(row.text)
    if n == 0:
        flags.append("empty")
        return flags
    dur = row.duration
    if dur < 2.0:
        flags.append("short")
    if dur > 7.0:
        flags.append("long")
    rate = n / dur if dur > 0 else 0.0
    if rate < float(cfg.get("rate_slow", 2.5)):
        flags.append("rate_slow")
    elif rate > float(cfg.get("rate_fast", 7.5)):
        flags.append("rate_fast")
    if row.asr_conf < float(cfg.get("logprob_warn", -0.6)) + 1.0 and row.asr_conf < 0.6:
        flags.append("low_conf")
    if row.peak_dbfs > float(cfg.get("peak_dbfs_red", -0.5)):
        flags.append("peak")
    if 0 < row.snr_db < float(cfg.get("min_snr_db", 30.0)):
        flags.append("snr")
    if duplicate_of is not None:
        flags.append("dup")
    return flags


def mark_duplicates(rows: list[ReviewRow], threshold: float = 0.9) -> None:
    """文本相似度超阈值的后出现片段打 dup flag（不改 status）。"""
    for j in range(len(rows)):
        if not rows[j].text:
            continue
        for i in range(j):
            if not rows[i].text:
                continue
            r = difflib.SequenceMatcher(None, rows[i].text, rows[j].text).ratio()
            if r >= threshold:
                if "dup" not in rows[j].flags:
                    rows[j].flags.append("dup")
                break


def flag_level(flags: Iterable[str]) -> str:
    """把 flag 归成 红/黄/绿 三档，UI 只按这个上色。"""
    fs = set(flags)
    if fs & RED_FLAGS:
        return "red"
    if fs:
        return "yellow"
    return "green"


def describe_flags(flags: Iterable[str]) -> str:
    return "、".join(FLAG_LABELS.get(f, f) for f in flags)


__all__ = [
    "WordSpan", "Transcript", "ReviewRow", "Transcriber",
    "normalize_text", "compute_flags", "mark_duplicates", "flag_level",
    "describe_flags", "cjk_len", "char_speech_rate", "FLAG_LABELS", "RED_FLAGS",
]
