"""每轮延迟埋点：logs/turns.jsonl + 最近 N 轮聚合。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnMetrics:
    char_id: str = ""
    provider: str = ""
    text_len: int = 0
    reply_len: int = 0
    audio_sec: float = 0.0
    t_press: float = 0.0
    t_release: float = 0.0
    t_asr_done: float = 0.0
    t_llm_first_token: float = 0.0
    t_tts_first_chunk: float = 0.0
    t_play_start: float = 0.0
    t_play_end: float = 0.0
    interrupted: bool = False
    error: str = ""

    # -- 派生指标（毫秒） ----------------------------------------------
    @staticmethod
    def _ms(a: float, b: float) -> float | None:
        if not a or not b:
            return None
        return round((b - a) * 1000, 1)

    @property
    def hold_ms(self) -> float | None:
        return self._ms(self.t_press, self.t_release)

    @property
    def asr_ms(self) -> float | None:
        return self._ms(self.t_release, self.t_asr_done)

    @property
    def llm_ttft_ms(self) -> float | None:
        return self._ms(self.t_asr_done, self.t_llm_first_token)

    @property
    def tts_ttfp_ms(self) -> float | None:
        return self._ms(self.t_llm_first_token, self.t_tts_first_chunk)

    @property
    def e2e_ms(self) -> float | None:
        return self._ms(self.t_release, self.t_play_start)

    @property
    def play_ms(self) -> float | None:
        return self._ms(self.t_play_start, self.t_play_end)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({
            "hold_ms": self.hold_ms, "asr_ms": self.asr_ms,
            "llm_ttft_ms": self.llm_ttft_ms, "tts_ttfp_ms": self.tts_ttfp_ms,
            "e2e_ms": self.e2e_ms, "play_ms": self.play_ms,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        return d


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))
    return xs[idx]


class MetricsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, m: TurnMetrics) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(m.as_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def summary(self, limit: int = 20) -> dict[str, Any]:
        rows = self.recent(limit)

        def col(name: str) -> list[float]:
            return [float(r[name]) for r in rows
                    if isinstance(r.get(name), (int, float))]

        out: dict[str, Any] = {"count": len(rows)}
        for name in ("asr_ms", "llm_ttft_ms", "tts_ttfp_ms", "e2e_ms", "play_ms"):
            vals = col(name)
            out[name] = {"p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95),
                         "n": len(vals)}
        return out


__all__ = ["TurnMetrics", "MetricsStore"]
