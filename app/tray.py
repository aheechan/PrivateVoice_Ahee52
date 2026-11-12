"""托盘图标与菜单。

图标用 QPainter 现画：外圈颜色表示状态，右下角角标表示训练/录音。
不依赖任何外部图片资源。
"""
from __future__ import annotations

from typing import Callable

from app import i18n

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

STATE_COLORS = {
    "idle": "#4A90D9",
    "listening": "#E5534B",
    "transcribing": "#9B6BD6",
    "thinking": "#B07CD6",
    "speaking": "#3FB950",
    "training": "#D29922",
    # 兼容 Stage 2 的旧名字
    "recording": "#E5534B",
}

STATE_LABELS = {
    "idle": "待机",
    "listening": "听着",
    "transcribing": "识别中",
    "thinking": "思考中",
    "speaking": "说话中",
    "training": "训练中",
    "recording": "录音中",
}


def _brand_pixmap(state: str, size: int) -> QPixmap | None:
    """品牌图标（气泡声波）+ 右下角状态圆点。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("app_256.png", "app_64.png"):
        p = root / "assets" / name
        if not p.exists():
            continue
        base = QPixmap(str(p)).scaled(size, size, Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        if state == "idle":
            return base
        pm = QPixmap(base)
        pt = QPainter(pm)
        pt.setRenderHint(QPainter.Antialiasing, True)
        pt.setBrush(QColor(STATE_COLORS.get(state, STATE_COLORS["idle"])))
        pt.setPen(QPen(QColor("#FFFFFF"), max(1.0, size / 32)))
        r = size * 0.32
        pt.drawEllipse(QRectF(size - r - size * 0.04, size - r - size * 0.04, r, r))
        pt.end()
        return pm
    return None


def make_icon(state: str = "idle", size: int = 64) -> QIcon:
    brand = _brand_pixmap(state, size)
    if brand is not None:
        return QIcon(brand)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    color = QColor(STATE_COLORS.get(state, STATE_COLORS["idle"]))
    p.setBrush(color)
    p.setPen(Qt.NoPen)
    p.drawEllipse(QRectF(4, 4, size - 8, size - 8))

    # 中间一条声波
    p.setPen(QPen(QColor("#FFFFFF"), max(2.0, size / 16), Qt.SolidLine, Qt.RoundCap))
    cx, cy = size / 2, size / 2
    for i, h in enumerate((0.14, 0.26, 0.34, 0.26, 0.14)):
        x = cx + (i - 2) * size * 0.12
        p.drawLine(QPointF(x, cy - size * h), QPointF(x, cy + size * h))

    if state != "idle":
        badge = QColor(STATE_COLORS[state])
        p.setBrush(badge)
        p.setPen(QPen(QColor("#FFFFFF"), 2))
        r = size * 0.30
        p.drawEllipse(QRectF(size - r - 2, size - r - 2, r, r))

    p.end()
    return QIcon(pm)


class TrayIcon(QSystemTrayIcon):
    """托盘：状态图标 + 菜单。所有动作都通过信号交给 MainController。"""

    new_character = Signal()
    manage_characters = Signal()
    show_main = Signal()
    open_settings = Signal()
    open_diagnostics = Signal()
    open_llm_settings = Signal()
    retry_last = Signal()
    quit_requested = Signal()
    character_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = "idle"
        self._progress_text = ""
        self._can_retry = False
        self.setIcon(make_icon("idle"))
        self.setToolTip("PrivateVoice_Ahee52")

        self._menu = QMenu()
        self.retranslate()
        self.setContextMenu(self._menu)

        self.activated.connect(self._on_activated)

    def retranslate(self) -> None:
        """重建菜单（语言切换后调用）。"""
        self._menu.clear()
        self._char_menu = self._menu.addMenu(i18n.t("tray.character"))
        self._char_menu.setEnabled(False)
        self._retry_action = self._menu.addAction(
            i18n.t("tray.retry"), self.retry_last.emit)
        self._retry_action.setEnabled(False)
        self._menu.addSeparator()
        self._menu.addAction(i18n.t("tray.new"), self.new_character.emit)
        self._menu.addAction(i18n.t("tray.manage"), self.manage_characters.emit)
        self._menu.addSeparator()
        self._menu.addAction(i18n.t("tray.llm"), self.open_llm_settings.emit)
        self._menu.addAction(i18n.t("tray.settings"), self.open_settings.emit)
        self._menu.addAction(i18n.t("tray.diagnostics"), self.open_diagnostics.emit)
        self._menu.addSeparator()
        self._menu.addAction(i18n.t("tray.quit"), self.quit_requested.emit)
        self.set_state(self._state)

    # -- 状态 ----------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self.setIcon(make_icon(state))
        label = i18n.t(f"state.{state}") if f"state.{state}" in i18n.STRINGS["zh"] \
            else STATE_LABELS.get(state, state)
        text = i18n.t("tray.tooltip", state=label)
        if detail:
            text += f"：{detail}"
        self.setToolTip(text)

    @property
    def state(self) -> str:
        return self._state

    def set_progress(self, text: str) -> None:
        """训练中把进度文案挂到托盘 tooltip 上。"""
        self._progress_text = text
        label = i18n.t(f"state.{self._state}") if f"state.{self._state}" in i18n.STRINGS["zh"] \
            else self._state
        base = i18n.t("tray.tooltip", state=label)
        self.setToolTip(f"{base}：{text}" if text else base)

    # -- 角色菜单 ------------------------------------------------------
    def set_characters(self, characters: list[tuple[str, str]], active_id: str) -> None:
        self._char_menu.clear()
        if not characters:
            act = QAction(i18n.t("tray.no_char"), self._char_menu)
            act.setEnabled(False)
            self._char_menu.addAction(act)
            self._char_menu.setEnabled(False)
            return
        self._char_menu.setEnabled(True)
        for cid, name in characters:
            act = QAction(name + ("  ✓" if cid == active_id else ""), self._char_menu)
            act.setCheckable(True)
            act.setChecked(cid == active_id)
            act.triggered.connect(lambda _=False, c=cid: self.character_selected.emit(c))
            self._char_menu.addAction(act)

    def set_busy(self, busy: bool) -> None:
        """训练中禁止启动对话（菜单里没有对话入口，但这里统一控制可用性）。"""
        for act in self._menu.actions():
            if act.text().startswith("新建角色"):
                act.setEnabled(not busy)
        self._retry_action.setEnabled(not busy and self._can_retry)

    def set_can_retry(self, can: bool) -> None:
        self._can_retry = bool(can)
        self._retry_action.setEnabled(self._can_retry)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self.show_main.emit()

    def notify(self, title: str, message: str, msec: int = 4000) -> None:
        self.showMessage(title, message, make_icon(self._state), msec)


__all__ = ["TrayIcon", "make_icon", "STATE_COLORS", "STATE_LABELS"]
