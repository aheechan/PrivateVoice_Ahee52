"""在真实应用对象（MainController + 托盘 + 管线）上跑一轮文字对话。

用于验证打包产物：不需要麦克风、不需要人点界面。
    runtime\\python.exe tools\\gui_headless.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(
    os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

import app.main as M  # noqa: E402
from app.config import Config  # noqa: E402


def pump(app, seconds: float) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    app = QApplication([])
    cfg = Config.load()
    cfg.bootstrap()
    print("root:", ROOT, flush=True)

    ctrl = M.MainController(app, cfg)
    print("character:", (ctrl._current_character() or type("x", (), {"name": "-"})).name,
          flush=True)

    t0 = time.time()
    while time.time() - t0 < 120:
        app.processEvents()
        if ctrl.pipeline is not None and ctrl.pipeline.is_warm:
            break
        time.sleep(0.05)
    print(f"warm: {ctrl.pipeline.is_warm if ctrl.pipeline else None} "
          f"({time.time()-t0:.1f}s)", flush=True)

    ctrl.open_main()
    ctrl.submit_text("你好，我在测试文字输入")
    t0 = time.time()
    while time.time() - t0 < 90:
        app.processEvents()
        p = ctrl.pipeline
        if p.state.value == "idle" and p._turn_thread and not p._turn_thread.is_alive():
            break
        time.sleep(0.02)

    rows = ctrl.pipeline.metrics.recent(1)
    m = rows[-1] if rows else {}
    print(f"turn: e2e={m.get('e2e_ms')}ms audio={m.get('audio_sec')}s "
          f"err={m.get('error')!r}", flush=True)
    transcript = ctrl._main.transcript.toPlainText().replace("\n", " | ")[-120:]
    print("transcript:", transcript, flush=True)

    ok = bool(m.get("audio_sec")) and not m.get("error")
    ctrl.pipeline.close()
    if ctrl.hotkey:
        ctrl.hotkey.stop()
    print("GUI_HEADLESS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
