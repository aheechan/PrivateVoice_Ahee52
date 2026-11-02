"""程序入口：单实例、托盘、全局热键、对话管线、日志、硬件门禁。

Stage 3 在 Stage 2 的托盘骨架上接通了
「按住热键 → 录音 → ASR → LLM → 切句 → TTS → 播放」，并支持随时打断。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.core import train  # noqa: E402
from app.core.characters import CharacterStore  # noqa: E402
from app.core.hotkey import HotkeyListener  # noqa: E402
from app.core.pipeline import Pipeline, PipelineEvent  # noqa: E402
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


class Bridge(QObject):
    """把工作线程里的事件搬回 GUI 线程。"""

    stateChanged = Signal(str, str)
    userText = Signal(str)
    replyDelta = Signal(str)
    info = Signal(str)
    error = Signal(str)
    turnDone = Signal(str)
    pressed = Signal()
    released = Signal()
    hotkeyError = Signal(str)
    warmupFinished = Signal(object)


class MainController(QObject):
    def __init__(self, app: QApplication, config: Config, parent=None):
        super().__init__(parent)
        self.app = app
        self.config = config
        self.log = logging.getLogger("voicelab.controller")
        self.store = CharacterStore(config.path_of("characters"))

        self.bridge = Bridge()
        self.bridge.pressed.connect(self._on_press)
        self.bridge.released.connect(self._on_release)
        self.bridge.stateChanged.connect(self._on_state)
        self.bridge.warmupFinished.connect(self._on_warmup_finished)
        self.bridge.error.connect(self._on_error)
        self.bridge.info.connect(lambda t: self.log.info("pipeline: %s", t))

        self.tray = TrayIcon()
        self.tray.new_character.connect(self.open_wizard)
        self.tray.manage_characters.connect(self.open_manager)
        self.tray.show_main.connect(self.open_main)
        self.tray.open_settings.connect(self.open_settings)
        self.tray.open_llm_settings.connect(self.open_llm_settings)
        self.tray.open_diagnostics.connect(self.open_diagnostics)
        self.tray.retry_last.connect(self.retry_last)
        self.tray.quit_requested.connect(self.quit)
        self.tray.character_selected.connect(self.select_character)
        self.tray.show()

        self._wizard = None
        self._manager = None
        self._main = None
        self._settings = None
        self._llm_settings = None
        self._diagnostics = None
        self.training = False
        self.pipeline: Pipeline | None = None
        self.hotkey: HotkeyListener | None = None

        self.refresh_characters()
        self._ensure_pipeline()
        self._start_hotkey()
        self.set_state("idle")
        threading.Thread(target=self._warmup, daemon=True, name="vlab-warmup").start()

    # -- 角色 ----------------------------------------------------------
    def _current_character(self):
        ready = [c for c in self.store.list() if c.status == "ready"]
        if not ready:
            return None
        cid = self.store.active_id
        return next((c for c in ready if c.id == cid), ready[0])

    def refresh_characters(self) -> None:
        chars = self.store.list()
        pairs = [(c.id, c.name) for c in chars if c.status == "ready"]
        self.tray.set_characters(pairs, self.store.active_id)
        if self._main is not None:
            self._main.set_character(self._current_character())

    def _ensure_pipeline(self) -> Pipeline | None:
        ch = self._current_character()
        if ch is None:
            return None
        if self.pipeline is None:
            self.pipeline = Pipeline(self.config, self.store, ch,
                                     on_event=self._on_pipeline_event)
            self.store.set_active(ch.id)
        return self.pipeline

    def select_character(self, cid: str) -> None:
        try:
            ch = self.store.get(cid)
        except KeyError:
            return
        self.store.set_active(cid)
        if self.pipeline is not None:
            self.pipeline.set_character(ch)
        if self._main is not None:
            self._main.set_character(ch)
        self.refresh_characters()
        self.tray.notify("已切换角色", f"当前角色：{ch.name}")
        self.set_state("idle", ch.name)

    # -- 热键 ----------------------------------------------------------
    def _start_hotkey(self) -> None:
        spec = str(self.config.get("hotkey.spec", "shift+a"))
        try:
            self.hotkey = HotkeyListener(
                spec,
                on_press=self.bridge.pressed.emit,
                on_release=self.bridge.released.emit,
                poll_ms=int(self.config.get("hotkey.poll_ms", 20)),
                min_hold_ms=int(self.config.get("hotkey.min_hold_ms", 150)),
                on_error=self.bridge.hotkeyError.emit,
            )
            self.hotkey.start()
            self.bridge.hotkeyError.connect(self._on_hotkey_error)
            if self.hotkey.registered:
                self.log.info("热键 %s 已注册", self.hotkey.spec.label)
                if self.hotkey.spec.mods in (0, 4) and self.hotkey.spec.vk < 0x5B:
                    self.tray.notify(
                        "热键提示",
                        f"当前热键 {self.hotkey.spec.label} 会全局拦截该按键；"
                        f"在其它程序里输入同样的组合也会触发对话。可在设置里更改。", 6000)
            else:
                self._on_hotkey_error("热键注册超时")
        except Exception as exc:  # noqa: BLE001
            self._on_hotkey_error(str(exc))

    def _on_hotkey_error(self, message: str) -> None:
        self.log.error("热键错误：%s", message)
        self.tray.notify("热键不可用", message, 8000)

    def _on_press(self) -> None:
        if self.training:
            self.tray.notify("训练进行中", "等训练结束再说话")
            return
        pipe = self._ensure_pipeline()
        if pipe is None:
            self.tray.notify("还没有角色", "先新建一个角色再来找我")
            return
        pipe.press()

    def _on_release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.release()

    # -- pipeline 事件 --------------------------------------------------
    def _on_pipeline_event(self, ev: PipelineEvent) -> None:
        if ev.kind == "state":
            self.bridge.stateChanged.emit(ev.state, ev.detail)
        elif ev.kind == "user_text":
            self.bridge.userText.emit(ev.text)
        elif ev.kind == "reply_delta":
            self.bridge.replyDelta.emit(ev.text)
        elif ev.kind == "error":
            self.bridge.error.emit(ev.text)
        elif ev.kind == "info":
            self.bridge.info.emit(ev.text)
        elif ev.kind == "turn_done":
            self.bridge.turnDone.emit(ev.text)

    def _on_state(self, state: str, detail: str) -> None:
        self.tray.set_state(state, detail)
        if state == "idle":
            ch = self._current_character()
            if ch is not None and self.pipeline is not None:
                last = self.pipeline.history.last_user()
                self.tray.set_can_retry(bool(last))

    def _on_error(self, text: str) -> None:
        self.log.warning("pipeline error: %s", text)
        self.tray.notify("出了点问题", text, 5000)

    # -- 预热 ----------------------------------------------------------
    def _warmup(self) -> None:
        pipe = self.pipeline
        if pipe is None:
            self.bridge.warmupFinished.emit({})
            return
        try:
            info = pipe.warmup()
            self.log.info("预热完成：%s", info)
            self.bridge.warmupFinished.emit(info)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("预热失败：%s", exc)
            self.bridge.warmupFinished.emit({"error": str(exc)})

    def _on_warmup_finished(self, info: dict) -> None:
        if self._main is None:
            return
        self._main.set_warm(True)
        if info and "tts_error" not in info:
            self._main.set_warmup_status(
                f"预热完成：ASR {info.get('asr_ms', 0)/1000:.1f}s · "
                f"TTS 加载 {info.get('tts_load_s', 0)}s · "
                f"试合成 {info.get('tts_prime_s', 0)}s")
        elif info.get("error"):
            self._main.set_warmup_status(f"预热失败：{info['error']}")

    # -- 菜单动作 -------------------------------------------------------
    def retry_last(self) -> None:
        pipe = self._ensure_pipeline()
        if pipe is None:
            return
        last = pipe.history.last_user()
        if not last:
            self.tray.notify("没有上一轮", "先说一句话吧")
            return
        pipe.submit_text(last)

    def set_training(self, busy: bool, detail: str = "") -> None:
        self.training = busy
        self.tray.set_busy(busy)
        if self.pipeline is not None:
            self.pipeline.set_training(busy)
        if busy:
            self.tray.set_state("training", detail)
            self.tray.set_progress(detail)
        else:
            self.tray.set_state("idle")

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
        self._ensure_pipeline()
        if self._main is not None:
            self._main.set_character(self._current_character())

    def open_main(self) -> None:
        from app.ui.main_window import MainWindow

        if self._main is None:
            self._main = MainWindow(self.config, self.store)
            self._main.request_characters.connect(self.open_manager)
            self._main.request_settings.connect(self.open_settings)
            self._main.request_llm.connect(self.open_llm_settings)
            self._main.request_diagnostics.connect(self.open_diagnostics)
            self._main.request_new_character.connect(self.open_wizard)
            self._main.request_warmup.connect(self.start_warmup)
            self._main.text_submitted.connect(self.submit_text)
            self._main.talk_pressed.connect(self._on_press)
            self._main.talk_released.connect(self._on_release)
            self.bridge.stateChanged.connect(self._main.set_state)
            self.bridge.userText.connect(self._main.show_user)
            self.bridge.turnDone.connect(self._main.show_reply)
            self.bridge.error.connect(self._main.show_error)
        self._main.set_character(self._current_character())
        pipe = self.pipeline
        self._main.set_warm(bool(pipe is not None and pipe.is_warm))
        self._main.show()
        self._main.raise_()
        self._main.activateWindow()
        self.log.info("主界面已打开：%s", self._main.windowTitle())

    def submit_text(self, text: str) -> None:
        """主界面文字输入：跳过录音与 ASR，直接走 LLM→TTS→播放。"""
        text = (text or "").strip()
        if not text:
            return
        if self.training:
            self.tray.notify("训练进行中", "等训练结束再说话")
            return
        pipe = self._ensure_pipeline()
        if pipe is None:
            self.tray.notify("还没有角色", "先新建一个角色")
            return
        if self._main is not None:
            self._main.set_state("thinking", "文字输入")
        pipe.submit_text(text)

    def start_warmup(self) -> None:
        """手动预热：把 ASR / TTS 提前加载进显存。"""
        pipe = self._ensure_pipeline()
        if pipe is None:
            self._notify_warmup("还没有可用角色，先新建一个")
            return
        if getattr(self, "_warming", False):
            return
        self._warming = True
        self._notify_warmup("预热中…（加载 ASR/TTS，约 20–40 秒）")

        def job():
            try:
                info = pipe.warmup()
                self._notify_warmup(
                    f"预热完成：ASR {info.get('asr_ms', 0)/1000:.1f}s · "
                    f"TTS 加载 {info.get('tts_load_s', 0)}s · 试合成 {info.get('tts_prime_s', 0)}s")
            except Exception as exc:  # noqa: BLE001
                self._notify_warmup(f"预热失败：{exc}")
            finally:
                self._warming = False

        threading.Thread(target=job, daemon=True, name="vlab-warmup-manual").start()

    def _notify_warmup(self, text: str) -> None:
        self.log.info(text)
        if self._main is not None:
            self._main.set_warmup_status(text)

    def open_manager(self) -> None:
        from app.ui.char_manager import CharacterManager

        if self._manager is None:
            self._manager = CharacterManager(self.config, self.store)
            self._manager.changed.connect(self._on_characters_changed)
            self._manager.new_requested.connect(self.open_wizard)
            self._manager.closed.connect(self._on_manager_closed)
        self._manager.refresh()
        self._manager.show()
        self._manager.raise_()

    def _on_characters_changed(self) -> None:
        """角色被编辑/删除/切换后，让 pipeline 和主界面都拿到最新对象。"""
        self.refresh_characters()
        ch = self._current_character()
        if ch is not None and self.pipeline is not None:
            self.pipeline.set_character(ch)

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
        self.log.info("设置已保存，热键重新注册")
        if self.hotkey is not None:
            self.hotkey.stop()
        self._start_hotkey()
        if self.pipeline is not None:
            self.pipeline.player.set_device(self.config.audio.get("output_device"))
            self.pipeline.recorder.device = self.config.audio.get("input_device")

    def open_llm_settings(self) -> None:
        from app.ui.settings_llm import SettingsLLM

        if self._llm_settings is None:
            self._llm_settings = SettingsLLM(self.config)
            self._llm_settings.closed.connect(self._on_llm_closed)
        self._llm_settings.refresh()
        self._llm_settings.show()
        self._llm_settings.raise_()

    def _on_llm_closed(self) -> None:
        self._llm_settings = None

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

    # -- 状态 ----------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        self.tray.set_state(state, detail)

    def quit(self) -> None:
        self.log.info("退出")
        try:
            if self.hotkey is not None:
                self.hotkey.stop()
            if self.pipeline is not None:
                self.pipeline.close()
        finally:
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
        ctrl.open_main()


def install_autostart(enabled: bool) -> None:
    """注册表 Run 项，设置页可关闭。"""
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    exe = sys.executable
    cmd = f'"{exe}" "{Path(ROOT) / "app" / "main.py"}" --tray-only'
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
    log.info("VoiceLab Stage3 启动，工作目录 %s", ROOT)

    if _already_running():
        log.info("已有实例在运行，提示用户看托盘")
        QMessageBox.information(
            None, APP_NAME,
            "情感陪伴器已经在运行了。\n\n"
            "它常驻在系统托盘（任务栏右下角，可能被折叠进“^”里）。\n"
            "右键托盘图标可以打开「角色管理」「设置」「诊断」。")
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

    # 安装路径含中文时给个明确提醒：部分 CUDA / 音频组件对非 ASCII 路径不稳
    if any(ord(ch) > 127 for ch in str(ROOT)):
        log.warning("安装路径包含非 ASCII 字符：%s", ROOT)

    app.setWindowIcon(make_icon("idle"))

    holder: list = []
    server = _start_server(holder)
    controller = MainController(app, config)
    holder.append(controller)

    if config.ui.get("autostart"):
        install_autostart(True)

    # 首启向导：模型目录 / provider / 热键确认
    if not config.get("ui.first_run_done", False):
        try:
            from PySide6.QtWidgets import QDialog
            from app.ui.first_run import FirstRunWizard
            wiz = FirstRunWizard(config)
            code = wiz.exec()
            log.info("首启向导返回 %s", code)
            if code != QDialog.DialogCode.Accepted:
                log.info("首启向导被取消，继续在托盘运行")
            controller.refresh_characters()
        except Exception as exc:  # noqa: BLE001
            log.warning("首启向导失败（不阻塞启动）：%s", exc, exc_info=True)

    # 让用户知道它去哪儿了（Windows 默认会把新图标塞进溢出区）
    controller.tray.notify(
        "情感陪伴器_v1 已启动",
        "图标在系统托盘（任务栏右下角，可能被折叠）。右键图标有菜单；"
        "按住 shift+a 说话。", 8000)

    # 手动启动时直接打开主界面，避免"点了没反应"的错觉；开机自启只留托盘
    if "--tray-only" not in sys.argv:
        try:
            controller.open_main()
        except Exception as exc:  # noqa: BLE001
            log.warning("打开主界面失败：%s", exc)

    log.info("托盘就绪")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
