"""设置页：档位、音频设备、说话热键、日志级别、开机自启。"""
from __future__ import annotations

from app import i18n
from app.i18n_tz import tz

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from app.config import PROFILE_MATRIX, Config
from app.core import audio_io

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def cfg_spec(config) -> str:
    return str(config.get("hotkey.spec", config.ui.get("hotkey", "shift+a")))

_QT_KEY_NAMES = {
    Qt.Key_Space: "space", Qt.Key_Return: "enter", Qt.Key_Enter: "enter",
    Qt.Key_Tab: "tab", Qt.Key_Escape: "esc", Qt.Key_Backspace: "backspace",
    Qt.Key_Up: "up", Qt.Key_Down: "down", Qt.Key_Left: "left",
    Qt.Key_Right: "right", Qt.Key_Insert: "insert", Qt.Key_Delete: "delete",
    Qt.Key_Home: "home", Qt.Key_End: "end", Qt.Key_PageUp: "pageup",
    Qt.Key_PageDown: "pagedown",
}
_MODIFIER_KEYS = {
    Qt.Key_Shift: "shift", Qt.Key_Control: "ctrl", Qt.Key_Alt: "alt",
    Qt.Key_Meta: "win",
}


class HotkeyEdit(QPushButton):
    """点一下进入“按键捕获”，直接按想用的组合键（也支持单独按 Shift）。"""

    captured = Signal(str)

    def __init__(self, spec: str = "shift+a", parent=None):
        super().__init__(parent)
        self.spec = spec
        self._capturing = False
        self._pending_mod = ""
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumWidth(220)
        self._refresh()
        self.clicked.connect(self.start_capture)

    def _refresh(self) -> None:
        self.setText(f"{self.spec.upper()}　({tz('点击更换')})" if self.spec
                     else tz("点击后按键"))

    def set_spec(self, spec: str) -> None:
        self.spec = (spec or "").strip().lower()
        self._refresh()

    def start_capture(self) -> None:
        self._capturing = True
        self.setText(tz("请按下要用的键…（Esc 取消）"))
        self.setFocus(Qt.OtherFocusReason)
        self.grabKeyboard()

    def _finish(self, cancelled: bool = False) -> None:
        self._capturing = False
        self._pending_mod = ""
        self.releaseKeyboard()
        self._refresh()
        if not cancelled:
            self.captured.emit(self.spec)

    def _commit_pending_mod(self) -> None:
        spec = getattr(self, "_pending_mod", "")
        if not spec:
            return
        self.spec = spec
        self._finish()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._capturing:
            return super().keyPressEvent(event)
        key = event.key()
        if key == Qt.Key_Escape:
            self._finish(cancelled=True)
            return
        mods = event.modifiers()
        names: list[str] = []
        if mods & Qt.ControlModifier:
            names.append("ctrl")
        if mods & Qt.AltModifier:
            names.append("alt")
        if mods & Qt.ShiftModifier:
            names.append("shift")
        if mods & Qt.MetaModifier:
            names.append("win")
        if key in _MODIFIER_KEYS:
            # 修饰键可能是组合键的前半段：先挂起，400ms 内没有主键再当作单键
            self._pending_mod = _MODIFIER_KEYS[key]
            if getattr(self, "_mod_timer", None) is None:
                from PySide6.QtCore import QTimer

                self._mod_timer = QTimer(self)
                self._mod_timer.setSingleShot(True)
                self._mod_timer.timeout.connect(self._commit_pending_mod)
            self._mod_timer.start(400)
            return
        if getattr(self, "_mod_timer", None) is not None:
            self._mod_timer.stop()
        self._pending_mod = ""
        if Qt.Key_A <= key <= Qt.Key_Z:
            main = chr(key).lower()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            main = chr(key)
        elif Qt.Key_F1 <= key <= Qt.Key_F24:
            main = f"f{key - Qt.Key_F1 + 1}"
        elif key in _QT_KEY_NAMES:
            main = _QT_KEY_NAMES[key]
        else:
            return
        if not names:
            return
        self.spec = "+".join([*names, main])
        self._finish()


class SettingsDialog(QDialog):
    applied = Signal()
    closed = Signal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tz("设置"))
        self.setMinimumWidth(520)
        self.config = config
        self._build()
        self._load()

    # -- UI ------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        gb_train = QGroupBox(tz("训练档位"))
        form1 = QFormLayout(gb_train)
        self.profile_box = QComboBox()
        for key, meta in PROFILE_MATRIX.items():
            self.profile_box.addItem(f"{tz(meta['label'])}（epochs={meta['epochs']}，"
                                     f"grad_accum={meta['grad_accum']}）", key)
        form1.addRow(tz("默认档位"), self.profile_box)
        self.profile_hint = QLabel("")
        self.profile_hint.setWordWrap(True)
        self.profile_hint.setStyleSheet("color:#666;")
        form1.addRow("", self.profile_hint)
        root.addWidget(gb_train)

        gb_audio = QGroupBox(tz("音频设备"))
        form2 = QFormLayout(gb_audio)
        self.in_box = QComboBox()
        self.out_box = QComboBox()
        for kind, box in (("input", self.in_box), ("output", self.out_box)):
            box.addItem(tz("系统默认"), None)
            try:
                for d in audio_io.list_devices(kind):
                    box.addItem(d.label(), d.index)
            except Exception:
                pass
        form2.addRow(tz("录音设备"), self.in_box)
        form2.addRow(tz("播放设备"), self.out_box)
        self.save_tts = QCheckBox(tz("把每轮合成的音频存到 results\\tts（便于检查音质）"))
        form2.addRow("", self.save_tts)
        self.btn_open_tts = QPushButton(tz("打开音频目录"))
        self.btn_open_tts.clicked.connect(self._open_tts_dir)
        form2.addRow("", self.btn_open_tts)
        root.addWidget(gb_audio)

        gb_asr = QGroupBox(tz("语音识别"))
        form_asr = QFormLayout(gb_asr)
        self.asr_box = QComboBox()
        self.asr_box.addItem(tz("中文 paraformer（默认）"), "paraformer-zh")
        self.asr_box.addItem(tz("English / 多语 whisper"), "whisper")
        cur = str(self.config.get("asr.model", "paraformer-zh"))
        self.asr_box.setCurrentIndex(0 if not cur.startswith("whisper") else 1)
        form_asr.addRow(tz("识别模型"), self.asr_box)
        self.asr_hint = QLabel("")
        self.asr_hint.setObjectName("Muted")
        self.asr_hint.setWordWrap(True)
        form_asr.addRow("", self.asr_hint)

        self.asr_lang = QComboBox()
        for label, code in ((tz("自动检测"), "auto"), ("English", "en"),
                            ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko")):
            self.asr_lang.addItem(label, code)
        _lang = str(self.config.get("asr.language", "auto"))
        _idx = self.asr_lang.findData(_lang)
        self.asr_lang.setCurrentIndex(_idx if _idx >= 0 else 0)
        form_asr.addRow(tz("识别语言"), self.asr_lang)
        form_asr.addRow("", QLabel(tz("仅 whisper 路线使用；paraformer 固定中文。")))
        self.asr_box.currentIndexChanged.connect(self._update_asr_hint)
        root.addWidget(gb_asr)
        self._update_asr_hint()

        gb_ui = QGroupBox(tz("交互与日志"))
        form3 = QFormLayout(gb_ui)
        self.hotkey = HotkeyEdit(str(cfg_spec(self.config)))
        self.hotkey_hint = QLabel(tz("点按钮后直接按组合键；也可以只按 Shift 当按住说话。"))
        self.hotkey_hint.setObjectName("Muted")
        form3.addRow(tz("说话热键"), self.hotkey)
        form3.addRow("", self.hotkey_hint)
        self.log_level = QComboBox()
        self.log_level.addItems(LOG_LEVELS)
        form3.addRow(tz("日志级别"), self.log_level)
        self.web_enabled = QCheckBox(tz("允许联网搜索（Bing，问新闻/最新信息时会上网查）"))
        form3.addRow("", self.web_enabled)
        self.autostart = QCheckBox(tz("开机自动启动（注册表 Run 项）"))
        form3.addRow("", self.autostart)
        root.addWidget(gb_ui)

        btn = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn.accepted.connect(self._save)
        btn.rejected.connect(self.reject)
        root.addWidget(btn)
        self.profile_box.currentIndexChanged.connect(self._update_hint)

    def _open_tts_dir(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        path = Path(self.config.path_of("logs")).parent / "results" / "tts"
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _update_hint(self) -> None:
        key = self.profile_box.currentData()
        meta = PROFILE_MATRIX.get(key, {})
        self.profile_hint.setText(tz(str(meta.get("note", ""))))

    # -- 数据 ----------------------------------------------------------
    def _load(self) -> None:
        cfg = self.config
        idx = self.profile_box.findData(cfg.profile)
        self.profile_box.setCurrentIndex(max(0, idx))
        self._update_hint()
        self.hotkey.set_spec(str(cfg.get("hotkey.spec",
                                         cfg.ui.get("hotkey", "shift+a"))))
        lvl = str(cfg.ui.get("log_level", "INFO")).upper()
        self.log_level.setCurrentText(lvl if lvl in LOG_LEVELS else "INFO")
        self.autostart.setChecked(bool(cfg.ui.get("autostart", False)))
        for box, key in ((self.in_box, "input_device"), (self.out_box, "output_device")):
            val = cfg.audio.get(key)
            i = box.findData(val)
            box.setCurrentIndex(max(0, i))
        self.save_tts.setChecked(bool(cfg.get("audio.save_tts", True)))
        self.web_enabled.setChecked(bool(cfg.get("web.enabled", False)))

    def _update_asr_hint(self) -> None:
        model = self.asr_box.currentData()
        app_root = Path(self.config.path_of("gpt_sovits")).parent
        whisper_dir = app_root / "models_cache" / "whisper" / "faster-whisper-small"
        has_whisper = (whisper_dir / "model.bin").exists()
        if str(model).startswith("whisper"):
            if has_whisper:
                self.asr_hint.setText(tz("英文/多语模型已就绪。切换后需重启程序生效；首次使用会占用约 1 GB 显存。"))
                self.asr_hint.setStyleSheet("color:#7dd3a0;font-size:12px;")
            else:
                self.asr_hint.setText(
                    tz("缺少 whisper 模型（models_cache\\whisper\\faster-whisper-small）。"
                       "可用安装向导里的「下载缺失模型」，或手动放入后重启。"))
                self.asr_hint.setStyleSheet("color:#e0a458;font-size:12px;")
        else:
            self.asr_hint.setText(tz("中文识别最准，标点自然，默认推荐。"))
            self.asr_hint.setStyleSheet("color:#8f83ad;font-size:12px;")

    def _save(self) -> None:
        cfg = self.config
        cfg.set("profile", self.profile_box.currentData())
        spec = (self.hotkey.spec or "shift+a").strip()
        cfg.set("hotkey.spec", spec)
        cfg.set("ui.hotkey", spec)
        cfg.set("ui.log_level", self.log_level.currentText())
        cfg.set("ui.autostart", self.autostart.isChecked())
        cfg.set("audio.input_device", self.in_box.currentData())
        cfg.set("audio.output_device", self.out_box.currentData())
        cfg.set("audio.save_tts", self.save_tts.isChecked())
        cfg.set("web.enabled", self.web_enabled.isChecked())
        cfg.set("asr.model", self.asr_box.currentData())
        cfg.set("asr.language", self.asr_lang.currentData())
        try:
            from app.main import install_autostart
            install_autostart(self.autostart.isChecked())
        except Exception:
            pass
        self.applied.emit()
        self.accept()

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["SettingsDialog"]
