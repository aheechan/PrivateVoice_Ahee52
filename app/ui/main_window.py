"""主界面：陪伴者面板。

托盘程序默认看到的窗口——头像、名字、状态灯、按住说话按钮、最近一轮对话。
关闭只是收进托盘，不退出。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget,
)

STATUS = {
    "idle": ("待机", "#4A90D9"),
    "listening": ("听着…", "#E5534B"),
    "transcribing": ("识别中…", "#9B6BD6"),
    "thinking": ("思考中…", "#B07CD6"),
    "speaking": ("说话中…", "#3FB950"),
    "training": ("训练中", "#D29922"),
    "recording": ("录音中…", "#E5534B"),
}


def _avatar_pixmap(path: Path | None, size: int = 72,
                   initial: str = "") -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor("#243044"))
    p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, size, size)
    if path and path.exists():
        img = QPixmap(str(path))
        if not img.isNull():
            scaled = img.scaled(size, size, Qt.KeepAspectRatioByExpanding,
                                Qt.SmoothTransformation)
            p.setClipPath(_circle_path(size))
            p.drawPixmap(0, 0, scaled)
            p.end()
            return pm
    if initial:
        p.setPen(QColor("#8fa6c8"))
        f = QFont("Microsoft YaHei", int(size * 0.38), QFont.Bold)
        p.setFont(f)
        p.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, initial[:1])
    p.end()
    return pm


def _circle_path(size: int):
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    return path


class MainWindow(QWidget):
    closed = Signal()
    request_characters = Signal()
    request_settings = Signal()
    request_diagnostics = Signal()
    request_llm = Signal()
    request_new_character = Signal()
    request_warmup = Signal()
    text_submitted = Signal(str)
    style_changed = Signal(str)
    request_clean = Signal()
    talk_pressed = Signal()
    talk_released = Signal()

    def __init__(self, config, store, parent=None):
        super().__init__(parent)
        self.config = config
        self.store = store
        self.setWindowTitle("Ahee52_PrivateVoice-ASR-cn-beta")
        self.setObjectName("Root")
        self.resize(460, 620)
        self._character = None
        self._last_state = "idle"
        self._build()

    # -- UI ------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 16)
        root.setSpacing(10)

        head = QHBoxLayout()
        self.avatar = QLabel()
        self.avatar.setFixedSize(72, 72)
        head.addWidget(self.avatar)
        col = QVBoxLayout()
        self.name = QLabel("—")
        self.name.setStyleSheet("font-size:19px;font-weight:600;")
        self.sub = QLabel("还没有角色")
        self.sub.setStyleSheet("color:#777;")
        col.addWidget(self.name)
        col.addWidget(self.sub)
        head.addLayout(col, 1)
        self.btn_chars = QPushButton("角色库")
        self.btn_chars.clicked.connect(self.request_characters.emit)
        head.addWidget(self.btn_chars)
        root.addLayout(head)

        self.status = QLabel("● 待机")
        self.status.setObjectName("Status")
        root.addWidget(self.status)

        row_style = QHBoxLayout()
        lab = QLabel("情绪")
        lab.setObjectName("Muted")
        row_style.addWidget(lab)
        self.style_box = QComboBox()
        for s in self.config.get("llm.styles", []) or []:
            self.style_box.addItem(s.get("name", s.get("id")), s.get("id"))
        idx = self.style_box.findData(self.config.get("llm.default_style", "gentle"))
        self.style_box.setCurrentIndex(max(0, idx))
        self.style_box.currentIndexChanged.connect(
            lambda _i: self.style_changed.emit(self.style_box.currentData()))
        row_style.addWidget(self.style_box)
        row_style.addStretch(1)
        root.addLayout(row_style)

        self.warm_hint = QLabel("")
        self.warm_hint.setStyleSheet("color:#888;font-size:12px;")
        self.warm_hint.setWordWrap(True)
        root.addWidget(self.warm_hint)

        # 文字输入：没有麦克风时也能完整跑通链路
        row_in = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("也可以打字：输入后回车（没麦克风时用这个）")
        self.input.setMinimumHeight(36)
        self.input.setStyleSheet(
            "QLineEdit{background:#121722;border:1px solid #2a3444;border-radius:9px;"
            "color:#e6edf7;padding:6px 10px;font-size:13px;}")
        self.input.returnPressed.connect(self._submit_text)
        self.btn_send = QPushButton("发送")
        self.btn_send.setMinimumHeight(36)
        self.btn_send.clicked.connect(self._submit_text)
        row_in.addWidget(self.input, 1)
        row_in.addWidget(self.btn_send)
        root.addLayout(row_in)

        self.talk = QPushButton("按住说话")
        self.talk.setObjectName("Primary")
        self.talk.setMinimumHeight(96)
        self.talk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.talk.pressed.connect(self._on_press)
        self.talk.released.connect(self._on_release)
        root.addWidget(self.talk)

        self.hint = QLabel(f"也可以按住 {self.config.get('hotkey.spec', 'shift+a')} 说话")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#888;font-size:12px;")
        root.addWidget(self.hint)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setStyleSheet(
            "QTextEdit{background:#121722;border:1px solid #232b3a;border-radius:10px;"
            "color:#d7dee9;font-size:13px;padding:8px;}")
        self.transcript.setPlaceholderText("这里显示最近一轮对话")
        root.addWidget(self.transcript, 1)

        row = QHBoxLayout()
        self.btn_warm = QPushButton("预热")
        self.btn_warm.setToolTip("提前把 ASR / TTS 加载进显存，第一次说话就不用等加载")
        self.btn_warm.clicked.connect(self.request_warmup.emit)
        row.addWidget(self.btn_warm)
        self.btn_clean = QPushButton("清理缓存")
        self.btn_clean.setToolTip(
            "删除临时音频与日志：\n"
            "results\\tts（每轮合成音频，默认保留 10 分钟）、results\\quality、"
            "results\\compare、训练中间产物、日志\n"
            "不会动角色、模型缓存和程序本体。")
        self.btn_clean.clicked.connect(self.request_clean.emit)
        row.addWidget(self.btn_clean)
        for text, sig in (("设置", self.request_settings),
                          ("大模型", self.request_llm),
                          ("诊断", self.request_diagnostics),
                          ("新建角色", self.request_new_character)):
            b = QPushButton(text)
            b.clicked.connect(sig.emit)
            row.addWidget(b)
        root.addLayout(row)

    def set_hotkey_label(self, label: str) -> None:
        if hasattr(self, "hint"):
            self.hint.setText(f"也可以按住 {label} 说话")

    def set_warmup_status(self, text: str) -> None:
        self.warm_hint.setText(text)

    def set_warm(self, warm: bool) -> None:
        """预热期间禁用输入，避免和预热抢 GPU（会让 v3 CUDA Graph 卡死）。"""
        self._warm = warm
        ready = bool(self._character and self._character.status == "ready")
        self.input.setEnabled(ready and warm)
        self.btn_send.setEnabled(ready and warm)
        self.talk.setEnabled(ready and warm)
        if not warm:
            self.warm_hint.setText("正在预热模型（约 30 秒），完成后自动恢复输入…")
        elif not self.warm_hint.text():
            self.warm_hint.setText("")

    # -- 数据 ----------------------------------------------------------
    def set_character(self, character) -> None:
        self._character = character
        if character is None:
            self.name.setText("—")
            self.sub.setText("还没有角色，先新建一个")
            self.avatar.setPixmap(_avatar_pixmap(None))
            self.talk.setEnabled(False)
            self.input.setEnabled(False)
            self.btn_send.setEnabled(False)
            self.transcript.clear()
            return
        self.name.setText(character.name)
        voice = "音色已就绪" if character.status == "ready" else character.status
        self.sub.setText(f"{voice} · 片段 {character.dataset.get('clips', 0)} · "
                         f"{character.dataset.get('duration_sec', 0)} 秒素材")
        self.setWindowTitle(f"Ahee52_PrivateVoice-ASR-cn-beta · {character.name}")
        av = self.store.avatar_path(character.id) if character else None
        self.avatar.setPixmap(_avatar_pixmap(av, initial=character.name))
        self.talk.setEnabled(character.status == "ready")
        self.input.setEnabled(character.status == "ready" and getattr(self, "_warm", True))
        self.btn_send.setEnabled(character.status == "ready" and getattr(self, "_warm", True))
        self._load_history(character)

    def _load_history(self, character) -> None:
        try:
            from app.core.history import HistoryStore
            h = HistoryStore(self.store.dir_of(character.id),
                             context_turns=int(self.config.get("llm.history_turns", 12)))
            turns = h.turns[-4:]
        except Exception:
            turns = []
        if not turns:
            self.transcript.setPlainText("")
        else:
            lines = []
            for t in turns:
                who = "你" if t.role == "user" else character.name
                lines.append(f"<b>{who}：</b>{t.text}")
            self.transcript.setHtml("<br>".join(lines))
        self._scroll_bottom()

    # -- 状态 ----------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        self._last_state = state
        text, color = STATUS.get(state, (state, "#888"))
        self.status.setText(f"● {text}" + (f"　{detail}" if detail else ""))
        self.status.setStyleSheet(f"color:{color};font-size:13px;")
        busy = state in ("transcribing", "thinking", "speaking")
        self.talk.setText("按住打断" if state in ("speaking", "thinking") else "按住说话")

    def show_user(self, text: str) -> None:
        self.transcript.append(f"<b>你：</b>{text}")
        self._scroll_bottom()

    def show_reply(self, text: str) -> None:
        name = self._character.name if self._character else "她"
        self.transcript.append(f"<b>{name}：</b>{text}")
        self._scroll_bottom()

    def show_error(self, text: str) -> None:
        self.transcript.append(f"<span style='color:#e06c60'>⚠ {text}</span>")
        self._scroll_bottom()

    def _scroll_bottom(self) -> None:
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    # -- 交互 ----------------------------------------------------------
    def _on_press(self) -> None:
        self.talk_pressed.emit()

    def _on_release(self) -> None:
        self.talk_released.emit()

    def _submit_text(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.text_submitted.emit(text)

    def closeEvent(self, event):  # noqa: N802
        # 关掉只是收进托盘
        event.ignore()
        self.hide()
        self.closed.emit()
