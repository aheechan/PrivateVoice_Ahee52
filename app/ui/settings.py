"""设置页：档位、音频设备、热键（本阶段只保存）、日志级别、开机自启。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QVBoxLayout,
)

from app.config import PROFILE_MATRIX, Config
from app.core import audio_io

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


class SettingsDialog(QDialog):
    applied = Signal()
    closed = Signal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(520)
        self.config = config
        self._build()
        self._load()

    # -- UI ------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        gb_train = QGroupBox("训练档位")
        form1 = QFormLayout(gb_train)
        self.profile_box = QComboBox()
        for key, meta in PROFILE_MATRIX.items():
            self.profile_box.addItem(f"{meta['label']}（epochs={meta['epochs']}，"
                                     f"grad_accum={meta['grad_accum']}）", key)
        form1.addRow("默认档位", self.profile_box)
        self.profile_hint = QLabel("")
        self.profile_hint.setWordWrap(True)
        self.profile_hint.setStyleSheet("color:#666;")
        form1.addRow("", self.profile_hint)
        root.addWidget(gb_train)

        gb_audio = QGroupBox("音频设备")
        form2 = QFormLayout(gb_audio)
        self.in_box = QComboBox()
        self.out_box = QComboBox()
        for kind, box in (("input", self.in_box), ("output", self.out_box)):
            box.addItem("系统默认", None)
            try:
                for d in audio_io.list_devices(kind):
                    box.addItem(d.label(), d.index)
            except Exception:
                pass
        form2.addRow("录音设备", self.in_box)
        form2.addRow("播放设备", self.out_box)
        root.addWidget(gb_audio)

        gb_ui = QGroupBox("交互与日志")
        form3 = QFormLayout(gb_ui)
        self.hotkey = QLineEdit()
        self.hotkey.setPlaceholderText("ctrl+alt+space")
        form3.addRow("说话热键（本阶段仅保存）", self.hotkey)
        self.log_level = QComboBox()
        self.log_level.addItems(LOG_LEVELS)
        form3.addRow("日志级别", self.log_level)
        self.autostart = QCheckBox("开机自动启动（注册表 Run 项）")
        form3.addRow("", self.autostart)
        root.addWidget(gb_ui)

        btn = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn.accepted.connect(self._save)
        btn.rejected.connect(self.reject)
        root.addWidget(btn)
        self.profile_box.currentIndexChanged.connect(self._update_hint)

    def _update_hint(self) -> None:
        key = self.profile_box.currentData()
        meta = PROFILE_MATRIX.get(key, {})
        self.profile_hint.setText(meta.get("note", ""))

    # -- 数据 ----------------------------------------------------------
    def _load(self) -> None:
        cfg = self.config
        idx = self.profile_box.findData(cfg.profile)
        self.profile_box.setCurrentIndex(max(0, idx))
        self._update_hint()
        self.hotkey.setText(str(cfg.ui.get("hotkey", "ctrl+alt+space")))
        lvl = str(cfg.ui.get("log_level", "INFO")).upper()
        self.log_level.setCurrentText(lvl if lvl in LOG_LEVELS else "INFO")
        self.autostart.setChecked(bool(cfg.ui.get("autostart", False)))
        for box, key in ((self.in_box, "input_device"), (self.out_box, "output_device")):
            val = cfg.audio.get(key)
            i = box.findData(val)
            box.setCurrentIndex(max(0, i))

    def _save(self) -> None:
        cfg = self.config
        cfg.set("profile", self.profile_box.currentData())
        cfg.set("ui.hotkey", self.hotkey.text().strip() or "ctrl+alt+space")
        cfg.set("ui.log_level", self.log_level.currentText())
        cfg.set("ui.autostart", self.autostart.isChecked())
        cfg.set("audio.input_device", self.in_box.currentData())
        cfg.set("audio.output_device", self.out_box.currentData())
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
