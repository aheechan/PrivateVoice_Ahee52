"""把系统托盘的图标状态与右键菜单渲染成一张图，用于评审布局。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(
    os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.tray import STATE_LABELS, TrayIcon, make_icon  # noqa: E402

STATES = ["idle", "listening", "transcribing", "thinking", "speaking", "training"]
OUT = ROOT / "results" / "screenshots" / "11_tray.png"


def main() -> int:
    app = QApplication([])
    tray = TrayIcon()
    tray.set_characters([("a", "格雷丝"), ("b", "阿岚")], "a")
    tray.set_can_retry(True)

    icons = []
    for s in STATES:
        tray.set_state(s, "ep 3/8" if s == "training" else "")
        icons.append((s, make_icon(s, 96).pixmap(96, 96)))

    menu = tray.contextMenu()
    menu.adjustSize()
    app.processEvents()
    menu_pm = menu.grab()
    if menu_pm.isNull():
        menu.show()
        app.processEvents()
        menu_pm = menu.grab()
        menu.hide()

    W, H = 1040, 660
    pm = QPixmap(W, H)
    pm.fill(QColor("#0f1420"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    p.setPen(QColor("#e6edf7"))
    p.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
    p.drawText(28, 44, "VoiceLab · 系统托盘")
    p.setPen(QColor("#7d8aa3"))
    p.setFont(QFont("Microsoft YaHei", 9))
    p.drawText(28, 68, "左键点图标 = 打开角色管理；右键 = 下面这个菜单")

    # 图标状态
    p.setFont(QFont("Microsoft YaHei", 10))
    for i, (state, pix) in enumerate(icons):
        col = i % 3
        row = i // 3
        x = 40 + col * 170
        y = 100 + row * 140
        p.drawPixmap(QRectF(x, y, 72, 72), pix, QRectF(0, 0, pix.width(), pix.height()))
        p.setPen(QColor("#e6edf7"))
        p.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        p.drawText(x + 84, y + 30, STATE_LABELS[state])
        p.setPen(QColor("#7d8aa3"))
        p.setFont(QFont("Microsoft YaHei", 8))
        p.drawText(x + 84, y + 50, state)

    # 菜单
    mx, my = 600, 100
    p.setPen(QColor("#7d8aa3"))
    p.setFont(QFont("Microsoft YaHei", 9))
    p.drawText(mx, my - 12, "右键菜单")
    if not menu_pm.isNull():
        p.drawPixmap(mx, my, menu_pm)

    # 说明
    p.setPen(QColor("#9aa7bd"))
    p.setFont(QFont("Microsoft YaHei", 9))
    lines = [
        "· 当前角色 ▸ 里只列 ready 状态的角色，打勾的是当前生效的；点一下即时切换（音色与人设一起换）。",
        "· 重试上一轮：把上一句原话重发一遍，用于 provider 抖动或没听清时快速重来。",
        "· 大模型设置：provider 增删改 / 排序 / 连接测试，密钥 DPAPI 加密。",
        "· 训练中：「新建角色」与「重试上一轮」置灰，热键按下只提示“训练进行中”。",
        "· 托盘 tooltip 会带状态与进度，例如「VoiceLab — 训练中：sovits epoch 3/8」。",
    ]
    y = 420
    for line in lines:
        p.drawText(QRectF(40, y, W - 80, 24), Qt.TextSingleLine, line)
        y += 26

    p.setPen(QColor("#5c6b85"))
    p.setFont(QFont("Microsoft YaHei", 8))
    p.drawText(40, H - 24, "热键当前为 shift+a（全局拦截，可在设置里改）；提示气泡在状态切换时弹出。")
    p.end()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pm.save(str(OUT))
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
