"""验收 #7 的结构证据：托盘五态 + 训练中禁用入口 + 没有对话菜单项。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.tray import STATE_LABELS, TrayIcon, make_icon  # noqa: E402

STATES = ["idle", "recording", "thinking", "speaking", "training"]


def main() -> int:
    app = QApplication([])  # noqa: F841
    tray = TrayIcon()
    ok = True
    print(f"{'state':10s} {'icon':6s} {'label':8s} tooltip")
    for state in STATES:
        assert not make_icon(state).isNull(), f"{state} 图标为空"
        tray.set_state(state)
        print(f"{state:10s} OK     {STATE_LABELS[state]:8s} {tray.toolTip()}")
    tray.set_state("training")
    tray.set_progress("soVITS epoch 3/8")
    print("progress tooltip:", tray.toolTip())

    texts = [a.text() for a in tray.contextMenu().actions()]
    print("menu:", texts)
    has_dialog = any(("对话" in t) or ("通话" in t) for t in texts)
    print("menu 含对话入口:", has_dialog, "(应为 False)")
    ok &= not has_dialog

    tray.set_busy(True)
    new_enabled = [a.isEnabled() for a in tray.contextMenu().actions()
                   if a.text().startswith("新建角色")]
    print("训练中 新建角色 enabled:", new_enabled, "(应为 [False])")
    ok &= new_enabled == [False]

    tray.set_characters([("id1", "小满"), ("id2", "格雷丝")], "id2")
    chars = [a.text() for a in tray._char_menu.actions()]
    print("角色菜单:", chars)
    ok &= any("格雷丝" in c and "✓" in c for c in chars)

    print("TRAY_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
