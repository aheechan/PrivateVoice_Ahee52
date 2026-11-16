"""生成 Stage 3 的截图序列。

做法：离屏跑真实 Pipeline，把事件按时间录下来，再从事件快照渲染画面。
这样既能拿到真实的 state/文本，又不会被截图时机和渲染延迟干扰。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 离屏 Qt 默认找不到字体，只画方块；显式指向系统字体目录
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(
    os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from app.config import Config  # noqa: E402
from app.core import audio_qc  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "results" / "screenshots"
COLORS = {"idle": "#4A90D9", "listening": "#E5534B", "transcribing": "#9B6BD6",
          "thinking": "#B07CD6", "speaking": "#3FB950", "training": "#D29922"}
LABELS = {"idle": "待机", "listening": "听着", "transcribing": "识别中",
          "thinking": "思考中", "speaking": "说话中", "training": "训练中"}


@dataclass
class Snapshot:
    state: str = "idle"
    detail: str = ""
    character: str = ""
    user: str = ""
    reply: str = ""
    note: str = ""


class Hud:
    """纯绘制，不依赖事件循环。"""

    def __init__(self, size=(760, 420)):
        self.size = size

    def render(self, snap: Snapshot) -> QPixmap:
        w, h = self.size
        pm = QPixmap(w, h)
        pm.fill(QColor("#0f1420"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.setPen(QColor("#7d8aa3"))
        p.setFont(QFont("Microsoft YaHei", 11))
        p.drawText(28, 40, "VoiceLab · Stage 3 对话链路")
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(28, 62, f"角色：{snap.character}")

        color = QColor(COLORS.get(snap.state, "#4A90D9"))
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(28, 86, 150, 44), 10, 10)
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        p.drawText(QRectF(28, 86, 150, 44), Qt.AlignCenter,
                   LABELS.get(snap.state, snap.state))
        p.setPen(QColor("#9aa7bd"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(196, 114, snap.detail[:52])

        p.setPen(QColor("#5c6b85"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(28, 168, "你说：")
        p.setPen(QColor("#e6edf7"))
        p.setFont(QFont("Microsoft YaHei", 12))
        p.drawText(QRectF(28, 180, 700, 60), Qt.TextWordWrap,
                   snap.user or "（还没说话）")

        p.setPen(QColor("#5c6b85"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(28, 262, "角色回复：")
        p.setPen(QColor("#9fe6b0"))
        p.setFont(QFont("Microsoft YaHei", 12))
        p.drawText(QRectF(28, 274, 700, 90), Qt.TextWordWrap,
                   snap.reply or "（等她开口）")

        p.setPen(QColor("#7d8aa3"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QRectF(28, 380, 710, 30), Qt.TextWordWrap, snap.note)
        p.end()
        return pm


def clip(seconds: float = 4.0):
    cfg = Config.load()
    y, sr, _ = audio_qc.process_to_48k(ROOT / "assets" / "ref.wav", cfg.audio)
    n = int(seconds * sr)
    c = y[:n]
    if c.size < n:
        c = np.pad(c, (0, n - c.size))
    return c, sr


def record_turn(pipe: Pipeline, snap: Snapshot, events: list,
                audio=None, text: str | None = None,
                interrupt_at_speaking: bool = False) -> dict:
    """跑一轮，返回 {'interrupt_ms':...}；事件累积到 events 里。"""
    t0 = time.perf_counter()
    out = {"interrupt_ms": None}

    def on_event(ev):
        now = time.perf_counter() - t0
        events.append((now, ev.kind, ev.state, ev.text, ev.detail))
        if ev.kind == "state":
            snap.state = ev.state
            snap.detail = ev.detail
        elif ev.kind == "user_text":
            snap.user = ev.text
            snap.detail = "ASR：" + ev.text[:24]
        elif ev.kind == "reply_delta":
            snap.reply = (snap.reply + ev.text)[-90:]
        elif ev.kind == "turn_done":
            snap.reply = ev.text
        elif ev.kind == "error":
            snap.detail = ev.text

    pipe.on_event = on_event
    if text is not None:
        pipe.submit_text(text)
    else:
        pipe.submit_audio(audio[0], audio[1])

    if interrupt_at_speaking:
        # 等到真的出声
        deadline = time.time() + 25
        while time.time() < deadline:
            if snap.state == "speaking" and len(snap.reply) > 4:
                break
            time.sleep(0.02)
        time.sleep(0.15)
        out["interrupt_ms"] = pipe.interrupt()
        time.sleep(0.3)
        return out

    deadline = time.time() + 45
    while time.time() < deadline:
        if pipe.state.value == "idle" and pipe._turn_thread is not None \
                and not pipe._turn_thread.is_alive():
            break
        time.sleep(0.03)
    return out


def frames_from_events(events: list, snap: Snapshot, hud: Hud) -> dict[str, Snapshot]:
    """从真实事件序列里挑出几个有代表性的时刻。"""
    picks: dict[str, Snapshot] = {}
    reply = ""
    user = ""
    for _t, kind, state, text, detail in events:
        if kind == "user_text":
            user = text
        if kind == "reply_delta":
            reply += text
        if kind == "turn_done":
            reply = text
        if kind == "state" and state == "transcribing" and "transcribing" not in picks:
            picks["transcribing"] = Snapshot("transcribing", "ASR 识别中…", snap.character,
                                             user, "", "松开热键后先做首尾静音修剪，再整句识别")
        if kind == "state" and state == "thinking" and "thinking" not in picks:
            picks["thinking"] = Snapshot("thinking", "LLM 流式生成中", snap.character,
                                         user, reply, "首句一旦凑够标点就切给 TTS")
        if kind == "state" and state == "speaking" and "speaking" not in picks:
            picks["speaking"] = Snapshot("speaking", "首块音频已出声", snap.character,
                                         user, reply, "合成与播放重叠，后续句子继续预取")
    picks.setdefault("done", Snapshot("idle", "", snap.character, user, reply,
                                      "播完回到 idle，history 已落盘 chat.jsonl"))
    if "speaking" not in picks and reply:
        picks["speaking"] = Snapshot("speaking", "首块音频已出声", snap.character,
                                     user, reply, "合成与播放重叠，后续句子继续预取")
    return picks


def contact_sheet(paths: list[Path]) -> Path:
    w, h = 380, 210
    cols = 3
    rows = max(1, (len(paths) + cols - 1) // cols)
    sheet = QPixmap(w * cols, h * rows)
    sheet.fill(QColor("#0b0f18"))
    p = QPainter(sheet)
    for i, path in enumerate(paths[:cols * rows]):
        pm = QPixmap(str(path))
        p.drawPixmap(QRectF((i % cols) * w + 4, (i // cols) * h + 4, w - 8, h - 8), pm,
                     QRectF(0, 0, pm.width(), pm.height()))
    p.end()
    out = OUT / "00_contact_sheet.png"
    sheet.save(str(out))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication([])  # noqa: F841
    cfg = Config.load()
    store = CharacterStore(cfg.path_of("characters"))
    ready = [c for c in store.list() if c.status == "ready"]
    if len(ready) < 2:
        print("need 2 ready characters")
        return 1
    by_name = {c.name: c for c in ready}
    ch1 = by_name.get("阿岚") or ready[0]
    ch2 = by_name.get("格雷丝") or next(c for c in ready if c.id != ch1.id)
    print("characters:", ch1.name, "->", ch2.name)

    hud = Hud()
    snap = Snapshot(character=ch1.name)
    pipe = Pipeline(cfg, store, ch1)
    saved: list[Path] = []

    def save(name: str, s: Snapshot) -> None:
        path = OUT / name
        hud.render(s).save(str(path))
        saved.append(path)
        print("saved", name, "|", s.state, "|", s.note[:40])

    save("01_idle.png", Snapshot("idle", "", ch1.name, "", "",
                                 "启动后常驻托盘；按 shift+a 开始说话"))
    save("02_listening.png", Snapshot("listening", "按住 shift+a", ch1.name, "", "",
                                      "按住热键进入 listening（本机无麦克风，此帧为状态渲染）"))

    events: list = []
    record_turn(pipe, snap, events, audio=clip(4.0))
    picks = frames_from_events(events, snap, hud)
    for key, fname in (("transcribing", "03_transcribing.png"),
                       ("thinking", "04_thinking.png"),
                       ("speaking", "05_speaking.png")):
        if key in picks:
            save(fname, picks[key])

    # 打断
    snap2 = Snapshot(character=ch1.name)
    ev2: list = []
    stats = record_turn(pipe, snap2, ev2, text="那你陪我说会儿话吧",
                        interrupt_at_speaking=True)
    ms = stats.get("interrupt_ms") or 0.0
    save("06_interrupted.png",
         Snapshot("idle", f"已打断（{ms:.0f} ms）", ch1.name, snap2.user, "",
                  f"回复过程中再次按下热键：播放立即停止，实测 {ms:.0f} ms，队列已清空"))

    # 切换角色 + 再说话
    pipe.set_character(ch2)
    snap3 = Snapshot(character=ch2.name)
    ev3: list = []
    record_turn(pipe, snap3, ev3, text="今天真的有点撑不住了")
    picks3 = frames_from_events(ev3, snap3, hud)
    save("07_switched.png",
         Snapshot("idle", "已切换角色", ch2.name, "", "",
                  f"切换到「{ch2.name}」：人设与历史一起换，播放队列已清空"))
    if "speaking" in picks3:
        save("08_switched_speaking.png", picks3["speaking"])

    pipe.close()
    sheet = contact_sheet(saved[:6])
    print("contact sheet:", sheet.name)
    print("CAPTURE_OK", len(saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
