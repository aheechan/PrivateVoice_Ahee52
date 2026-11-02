"""角色管理：列表、导出/导入 .vcp、删除、设为当前。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
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
        self.setWindowTitle("角色管理")
        self.resize(760, 460)
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        self.hint = QLabel("双击一行可设为当前角色。")
        self.hint.setStyleSheet("color:#666;")
        root.addWidget(self.hint)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["名字", "状态", "片段", "时长(秒)", "创建时间", "最近使用"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda _i: self._set_active())
        root.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.btn_new = QPushButton("新建角色…")
        self.btn_active = QPushButton("设为当前")
        self.btn_export = QPushButton("导出 .vcp…")
        self.btn_import = QPushButton("导入 .vcp…")
        self.btn_delete = QPushButton("删除")
        self.btn_close = QPushButton("关闭")
        for b in (self.btn_new, self.btn_active, self.btn_export, self.btn_import,
                  self.btn_delete):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.btn_close)
        root.addLayout(row)

        self.btn_new.clicked.connect(self._new)
        self.btn_active.clicked.connect(self._set_active)
        self.btn_export.clicked.connect(self._export)
        self.btn_import.clicked.connect(self._import)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_close.clicked.connect(self.close)

    # -- 数据 ----------------------------------------------------------
    def refresh(self) -> None:
        chars = self.store.list()
        self.table.setRowCount(len(chars))
        self._ids: list[str] = []
        active = self.store.active_id
        for r, c in enumerate(chars):
            self._ids.append(c.id)
            name = c.name + ("（当前）" if c.id == active else "")
            values = [name, STATUS_TEXT.get(c.status, c.status),
                      str(c.dataset.get("clips", 0)),
                      f"{float(c.dataset.get('duration_sec', 0)):.1f}",
                      c.created_at[:19].replace("T", " "),
                      (c.last_used or "—")[:19].replace("T", " ")]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col == 1:
                    item.setForeground(Qt.GlobalColor.darkGreen if c.status == "ready"
                                       else Qt.GlobalColor.darkYellow)
                self.table.setItem(r, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _selected(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(getattr(self, "_ids", [])):
            return None
        return self._ids[row]

    # -- 动作 ----------------------------------------------------------
    def _new(self) -> None:
        self.new_requested.emit()

    def _set_active(self) -> None:
        cid = self._selected()
        if not cid:
            return
        self.store.set_active(cid)
        self.refresh()
        self.changed.emit()

    def _export(self) -> None:
        cid = self._selected()
        if not cid:
            return
        c = self.store.get(cid)
        if c.status != "ready":
            QMessageBox.warning(self, "无法导出", "只有「可用」状态的角色才能导出。")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出角色包", f"{c.name}.vcp", "VoiceLab 角色包 (*.vcp)")
        if not dest:
            return
        try:
            path = self.store.export_vcp(cid, dest)
            QMessageBox.information(self, "导出完成", f"已导出到：\n{path}")
        except Exception as exc:
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
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def _delete(self) -> None:
        cid = self._selected()
        if not cid:
            return
        c = self.store.get(cid)
        if QMessageBox.question(self, "删除角色",
                                f"确定删除「{c.name}」？这会删掉它的数据集与权重。") \
                != QMessageBox.Yes:
            return
        self.store.delete(cid)
        self.refresh()
        self.changed.emit()

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["CharacterManager"]
