"""程序入口：单实例、托盘、日志、硬件门禁。

本阶段不实现对话与热键，只保证：托盘常驻、窗口关闭即最小化、训练期间托盘
显示进度且无法启动对话。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.tray import TrayIcon, make_icon  # noqa: E402

APP_NAME = "VoiceLab"
SERVER_NAME = "VoiceLab-Stage2-SingleInstance"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(config: Config) -> logging.Logger:
    log_dir = config.path_of("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(config.ui.get("log_level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=5 << 20, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(sh)
    return logging.getLogger("voicelab")


class MainController:
    """把托盘、窗口、配置、角色库串起来。"""

    def __init__(self, app: QApplication, config: Config):
        self.app = app
        self.config = config
        self.log = logging.getLogger("voicelab.controller")
        self.store = CharacterStore(config.path_of("characters"))

        self.tray = TrayIcon()
        self.tray.new_character.connect(self.open_wizard)
        self.tray.manage_characters.connect(self.open_manager)
        self.tray.open_settings.connect(self.open_settings)
        self.tray.open_diagnostics.connect(self.open_diagnostics)
        self.tray.quit_requested.connect(self.quit)
        self.tray.character_selected.connect(self.select_character)
        self.tray.show()

        self._wizard = None
        self._manager = None
        self._settings = None
        self._diagnostics = None

        self.refresh_characters()
        self.set_state("idle")

    # -- 状态 ----------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        self.tray.set_state(state, detail)

    def set_training(self, busy: bool, detail: str = "") -> None:
        self.set_state("training" if busy else "idle", detail)
        self.tray.set_busy(busy)
        if detail:
            self.tray.set_progress(detail)

    # -- 角色 ----------------------------------------------------------
    def refresh_characters(self) -> None:
        chars = self.store.list()
        pairs = [(c.id, c.name) for c in chars if c.status == "ready"]
        self.tray.set_characters(pairs, self.store.active_id)

    def select_character(self, cid: str) -> None:
        self.store.set_active(cid)
        self.refresh_characters()
        name = self.store.get(cid).name
        self.tray.notify("已切换角色", f"当前角色：{name}")

    # -- 窗口 ----------------------------------------------------------
    def open_wizard(self) -> None:
        from app.ui.wizard import NewCharacterWizard

        if self._wizard is not None:
            self._wizard.raise_()
            self._wizard.activateWindow()
            return
        self._wizard = NewCharacterWizard(self.config, self.store, parent=None)
        self._wizard.training_state.connect(self.set_training)
        self._wizard.finished.connect(self._on_wizard_done)
        self._wizard.show()

    def _on_wizard_done(self, result: int) -> None:
        if self._wizard is not None:
            self._wizard.deleteLater()
            self._wizard = None
        self.refresh_characters()

    def open_manager(self) -> None:
        from app.ui.char_manager import CharacterManager

        if self._manager is None:
            self._manager = CharacterManager(self.config, self.store)
            self._manager.changed.connect(self.refresh_characters)
            self._manager.new_requested.connect(self.open_wizard)
            self._manager.closed.connect(self._on_manager_closed)
        self._manager.refresh()
        self._manager.show()
        self._manager.raise_()

    def _on_manager_closed(self) -> None:
        self._manager = None
        self.refresh_characters()

    def open_settings(self) -> None:
        from app.ui.settings import SettingsDialog

        if self._settings is None:
            self._settings = SettingsDialog(self.config)
            self._settings.applied.connect(self._on_settings_applied)
            self._settings.closed.connect(self._on_settings_closed)
        self._settings.show()
        self._settings.raise_()

    def _on_settings_closed(self) -> None:
        self._settings = None

    def _on_settings_applied(self) -> None:
        self.config.save()
        self.log.info("设置已保存")

    def open_diagnostics(self) -> None:
        from app.ui.diagnostics import DiagnosticsDialog

        if self._diagnostics is None:
            self._diagnostics = DiagnosticsDialog(self.config)
            self._diagnostics.closed.connect(self._on_diag_closed)
        self._diagnostics.refresh()
        self._diagnostics.show()
        self._diagnostics.raise_()

    def _on_diag_closed(self) -> None:
        self._diagnostics = None

    # -- 退出 ----------------------------------------------------------
    def quit(self) -> None:
        self.log.info("退出")
        self.tray.hide()
        self.app.quit()


def _already_running() -> bool:
    sock = QLocalSocket()
    sock.connectToServer(SERVER_NAME)
    if sock.waitForConnected(300):
        sock.disconnectFromServer()
        return True
    return False


def _start_server(controller_holder: list) -> QLocalServer:
    QLocalServer.removeServer(SERVER_NAME)
    server = QLocalServer()

    def on_new():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.readyRead.connect(lambda: (conn.readAll(), _raise_windows(controller_holder)))
            conn.disconnected.connect(conn.deleteLater)

    server.newConnection.connect(on_new)
    server.listen(SERVER_NAME)
    return server


def _raise_windows(holder: list) -> None:
    ctrl = holder[0] if holder else None
    if ctrl is None:
        return
    if ctrl._wizard is not None:
        ctrl._wizard.raise_()
    elif ctrl._manager is not None:
        ctrl._manager.raise_()
    else:
        ctrl.open_manager()


def install_autostart(enabled: bool) -> None:
    """注册表 Run 项，设置页可关闭。"""
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    exe = sys.executable
    cmd = f'"{exe}" "{Path(ROOT) / "app" / "main.py"}"'
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                            winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
    except OSError:
        logging.getLogger("voicelab").warning("写入开机自启失败", exc_info=True)


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    config = Config.load()
    config.bootstrap()
    log = setup_logging(config)
    log.info("VoiceLab Stage2 启动，工作目录 %s", ROOT)

    if _already_running():
        log.info("已有实例在运行，退出")
        return 0
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, APP_NAME, "系统托盘不可用，无法启动。")
        return 1

    ok, message = train.check_hardware(config)
    if not ok:
        log.error("硬件检查失败：%s", message)
        QMessageBox.critical(None, APP_NAME, message)
        return 2
    log.info("硬件检查通过：%s", message)

    app.setWindowIcon(make_icon("idle"))

    holder: list = []
    server = _start_server(holder)
    controller = MainController(app, config)
    holder.append(controller)

    if config.ui.get("autostart"):
        install_autostart(True)

    log.info("托盘就绪")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
