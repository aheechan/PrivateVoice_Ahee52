"""每角色会话历史：内存最近 N 轮 + chat.jsonl 落盘。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    role: str          # user | assistant
    text: str
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "ts": round(self.ts, 3)}


class HistoryStore:
    """characters/<id>/chat.jsonl，一行一轮。"""

    def __init__(self, char_dir: str | Path, context_turns: int = 12,
                 display_turns: int = 50):
        self.char_dir = Path(char_dir)
        self.path = self.char_dir / "chat.jsonl"
        self.context_turns = max(2, int(context_turns))
        self.display_turns = max(self.context_turns, int(display_turns))
        self.turns: list[Turn] = []
        self.load()

    def load(self) -> None:
        self.turns = []
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines()[-self.display_turns:]:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.turns.append(Turn(role=d.get("role", "user"),
                                       text=d.get("text", ""),
                                       ts=float(d.get("ts", 0))))
        except Exception:
            self.turns = []

    def add(self, role: str, text: str) -> Turn:
        turn = Turn(role=role, text=text)
        self.turns.append(turn)
        try:
            self.char_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(turn.as_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass
        if len(self.turns) > self.display_turns * 2:
            self.turns = self.turns[-self.display_turns:]
        return turn

    def context(self, budget_chars: int = 1500) -> list[dict[str, str]]:
        """最近 N 轮，按字符预算从新到旧截断（1 token ≈ 1.6 汉字，粗算）。"""
        window = self.turns[-self.context_turns * 2:]
        picked: list[dict[str, str]] = []
        used = 0
        for turn in reversed(window):
            cost = len(turn.text) + 8
            if used + cost > budget_chars and picked:
                break
            used += cost
            picked.append({"role": turn.role, "content": turn.text})
        return list(reversed(picked))

    def clear(self) -> None:
        self.turns = []
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass

    def last_user(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.text
        return ""


def system_prompt(template: str, **vars: Any) -> str:
    """把 companion_system.txt 里的 {{name}} 之类替换掉，不认识的原样保留。"""
    out = template
    for key, value in vars.items():
        out = out.replace("{{" + key + "}}", str(value))
        out = out.replace("{" + key + "}", str(value))
    return out


__all__ = ["Turn", "HistoryStore", "system_prompt"]
