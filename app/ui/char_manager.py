"""角色管理：列表 + 右侧编辑栏（人设/开场白/头像/温度），以及 .vcp 导入导出。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.core.characters import CharacterStore

STATUS_TEXT = {"draft": "草稿", "training": "训练中", "ready": "可用", "failed": "失败"}


class CharacterManager(QDialog):
    changed = Signal()
    closed = Signal()
    new_requested = Signal()

    def __init__(self, config: Config, store: CharacterStore, parent=None):
        super().__init__(parent)
        self.config = config
        self.store = store
        self._ids: list[str] = []
        self._current_id: str | None = None
        self.setWindowTitle("角色管理")
        self.resize(1000, 520)
        self._build()
        self.refresh()

    # -- UI ------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel("左边选角色，右边改人设。改完点「保存」，下一个对话轮立即生效。")
        tip.setStyleSheet("color:#666;")
        root.addWidget(tip)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["名字", "状态", "片段", "最近使用"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda _i: self._set_active())
        lv.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.btn_new = QPushButton("新建角色…")
        self.btn_active = QPushButton("设为当前")
        self.btn_export = QPushButton("导出 .vcp…")
        self.btn_import = QPushButton("导入 .vcp…")
        self.btn_delete = QPushButton("删除")
        for b in (self.btn_new, self.btn_active, self.btn_export, self.btn_import,
                  self.btn_delete):
            row.addWidget(b)
        lv.addLayout(row)
        split.addWidget(left)

        split.addWidget(self._build_editor())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self.btn_new.clicked.connect(self.new_requested.emit)
        self.btn_active.clicked.connect(self._set_active)
        self.btn_export.clicked.connect(self._export)
        self.btn_import.clicked.connect(self._import)
        self.btn_delete.clicked.connect(self._delete)

    def _build_editor(self) -> QWidget:
        box = QGroupBox("当前角色")
        v = QVBoxLayout(box)

        head = QHBoxLayout()
        self.avatar = QLabel("无头像")
        self.avatar.setFixedSize(64, 64)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setStyleSheet("background:#eef1f6;border-radius:32px;color:#999;")
        head.addWidget(self.avatar)
        self.btn_avatar = QPushButton("换头像…")
        self.btn_avatar.clicked.connect(self._pick_avatar)
        head.addWidget(self.btn_avatar)
        head.addStretch(1)
        v.addLayout(head)

        form = QFormLayout()
        self.f_name = QLineEdit()
        self.f_greeting = QLineEdit()
        self.f_greeting.setPlaceholderText("第一次见面时说的话")
        form.addRow("名字", self.f_name)
        form.addRow("开场白", self.f_greeting)

        self.f_persona = QPlainTextEdit()
        self.f_persona.setPlaceholderText(
            "写她是谁、怎么说话、边界在哪。这段会作为 system prompt 的一部分发给 LLM。")
        self.f_persona.setMinimumHeight(130)
        form.addRow("人设", self.f_persona)

        self.f_temp = QDoubleSpinBox()
        self.f_temp.setRange(0.0, 2.0)
        self.f_temp.setSingleStep(0.1)
        self.f_temp.setValue(0.8)
        self.f_max_tokens = QSpinBox()
        self.f_max_tokens.setRange(80, 512)
        self.f_max_tokens.setValue(160)
        row_llm = QHBoxLayout()
        row_llm.addWidget(QLabel("温度"))
        row_llm.addWidget(self.f_temp)
        row_llm.addWidget(QLabel("max_tokens"))
        row_llm.addWidget(self.f_max_tokens)
        row_llm.addStretch(1)
        form.addRow("LLM", row_llm)
        v.addLayout(form)

        self.info = QLabel("—")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#666;font-size:12px;")
        v.addWidget(self.info)

        row = QHBoxLayout()
        self.btn_save = QPushButton("保存")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        self.btn_revert = QPushButton("还原本地修改")
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._load_into_editor)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_revert)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        self.editor = box
        return box

    # -- 数据 ----------------------------------------------------------
    def refresh(self) -> None:
        chars = self.store.list()
        self._ids = [c.id for c in chars]
        self.table.setRowCount(len(chars))
        active = self.store.active_id
        for r, c in enumerate(chars):
            name = c.name + ("（当前）" if c.id == active else "")
            clips = c.dataset.get("clips", 0)
            values = [name, STATUS_TEXT.get(c.status, c.status),
                      "—" if not clips else str(clips),
                      (c.last_used or "—")[:16].replace("T", " ")]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col == 1:
                    item.setForeground(Qt.GlobalColor.darkGreen if c.status == "ready"
                                       else Qt.GlobalColor.darkYellow)
                self.table.setItem(r, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if self._current_id in self._ids:
            self.table.selectRow(self._ids.index(self._current_id))
        elif self._ids:
            self.table.selectRow(0)
        else:
            self._load_into_editor()

    def _on_select(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._ids):
            self._current_id = self._ids[row]
            self._load_into_editor()

    def _load_into_editor(self) -> None:
        if not self._current_id:
            self.editor.setEnabled(False)
            return
        try:
            c = self.store.get(self._current_id)
        except KeyError:
            self.editor.setEnabled(False)
            return
        self.editor.setEnabled(True)
        self.f_name.setText(c.name)
        self.f_greeting.setText(c.greeting or "")
        self.f_persona.setPlainText(c.persona or "")
        llm = c.llm or {}
        self.f_temp.setValue(float(llm.get("temperature", 0.8)))
        self.f_max_tokens.setValue(int(llm.get("max_tokens", 160)))
        av = self.store.avatar_path(c.id)
        if av.exists():
            pm = QPixmap(str(av)).scaled(64, 64, Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            self.avatar.setPixmap(pm)
            self.avatar.setText("")
        else:
            self.avatar.setPixmap(QPixmap())
            self.avatar.setText("无头像")
        self.info.setText(
            f"状态：{STATUS_TEXT.get(c.status, c.status)} · "
            f"素材 {c.dataset.get('duration_sec', 0)} 秒 / {c.dataset.get('clips', 0)} 段\n"
            f"gpt：{Path(c.voice.get('gpt_ckpt', '')).name or '—'}　"
            f"sovits：{Path(c.voice.get('sovits_ckpt', '')).name or '—'}")
        self.btn_save.setEnabled(True)
        self.btn_revert.setEnabled(True)

    # -- 动作 ----------------------------------------------------------
    def _save(self) -> None:
        if not self._current_id:
            return
        name = self.f_name.text().strip()
        if not name:
            QMessageBox.warning(self, "名字不能为空", "给角色起个名字。")
            return
        if self.store.name_taken(name, exclude=self._current_id):
            QMessageBox.warning(self, "重名", f"已经有叫「{name}」的角色了。")
            return
        c = self.store.get(self._current_id)
        c.name = name
        c.greeting = self.f_greeting.text().strip()
        c.persona = self.f_persona.toPlainText().strip()
        c.llm = {"temperature": round(self.f_temp.value(), 2),
                 "max_tokens": int(self.f_max_tokens.value())}
        self.store.save(c)
        self.refresh()
        self.changed.emit()
        self.info.setText(self.info.text() + "\n已保存（下一轮对话生效）")

    def _pick_avatar(self) -> None:
        if not self._current_id:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择头像", "", "图片 (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self.store.set_avatar(self._current_id, path)
        self._load_into_editor()
        self.refresh()
        self.changed.emit()

    def _set_active(self) -> None:
        if not self._current_id:
            return
        self.store.set_active(self._current_id)
        self.refresh()
        self.changed.emit()

    def _export(self) -> None:
        if not self._current_id:
            return
        c = self.store.get(self._current_id)
        if c.status != "ready":
            QMessageBox.warning(self, "无法导出", "只有「可用」状态的角色才能导出。")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出角色包", f"{c.name}.vcp", "VoiceLab 角色包 (*.vcp)")
        if not dest:
            return
        try:
            path = self.store.export_vcp(c.id, dest)
            QMessageBox.information(self, "导出完成", f"已导出到：\n{path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))

    def _import(self) -> None:
        src, _ = QFileDialog.getOpenFileName(
            self, "导入角色包", "", "VoiceLab 角色包 (*.vcp)")
        if not src:
            return
        overwrite = QMessageBox.question(
            self, "导入", "如果同名角色已存在，是否覆盖？",
            QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        try:
            c = self.store.import_vcp(src, overwrite=overwrite)
            QMessageBox.information(self, "导入完成", f"角色「{c.name}」已就绪。")
            self.refresh()
            self.changed.emit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", str(exc))

    def _delete(self) -> None:
        if not self._current_id:
            return
        c = self.store.get(self._current_id)
        if QMessageBox.question(self, "删除角色",
                                f"确定删除「{c.name}」？这会删掉它的数据集与权重。") \
                != QMessageBox.Yes:
            return
        self.store.delete(self._current_id)
        self._current_id = None
        self.refresh()
        self.changed.emit()

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["CharacterManager"]
