"""角色管理：列表 + 右侧编辑栏（人设/开场白/头像/温度），以及 .vcp 导入导出。"""
from __future__ import annotations

from app import i18n
from app.i18n_tz import tz

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


def _hint(text: str) -> "QLabel":
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("Muted")
    lab.setStyleSheet("color:#8f83ad;font-size:11px;")
    return lab


class CharacterManager(QDialog):
    changed = Signal()
    closed = Signal()
    new_requested = Signal()
    train_requested = Signal(str)

    def __init__(self, config: Config, store: CharacterStore, parent=None):
        super().__init__(parent)
        self.config = config
        self.store = store
        self._ids: list[str] = []
        self._current_id: str | None = None
        self.setWindowTitle(tz("角色管理"))
        self.resize(1000, 520)
        self._build()
        self.refresh()

    # -- UI ------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(tz("左边选角色，右边改人设。改完点「保存」，下一个对话轮立即生效。"))
        tip.setStyleSheet("color:#666;")
        root.addWidget(tip)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tz("名字"), tz("状态"), tz("片段"), tz("最近使用")])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        from PySide6.QtGui import QKeySequence, QShortcut

        self._refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        self._refresh_shortcut.activated.connect(self._refresh_list)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda _i: self._set_active())
        lv.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.btn_new = QPushButton(tz("新建角色…"))
        self.btn_train = QPushButton(tz("上传素材/训练…"))
        self.btn_train.setToolTip(tz("给选中的角色重新上传录音并训练（v3/v4 由设置决定）"))
        self.btn_active = QPushButton(tz("设为当前"))
        self.btn_export = QPushButton(tz("导出 .vcp…"))
        self.btn_import = QPushButton(tz("导入 .vcp…"))
        self.btn_delete = QPushButton(tz("删除"))
        for b in (self.btn_new, self.btn_train, self.btn_active, self.btn_export,
                  self.btn_import, self.btn_delete):
            row.addWidget(b)
        lv.addLayout(row)
        split.addWidget(left)

        split.addWidget(self._build_editor())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        btn_close = QPushButton(tz("关闭"))
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self.btn_new.clicked.connect(self.new_requested.emit)
        self.btn_train.clicked.connect(self._train)
        self.btn_active.clicked.connect(self._set_active)
        self.btn_export.clicked.connect(self._export)
        self.btn_import.clicked.connect(self._import)
        self.btn_delete.clicked.connect(self._delete)

    def _build_editor(self) -> QWidget:
        box = QGroupBox(tz("当前角色"))
        v = QVBoxLayout(box)

        head = QHBoxLayout()
        self.avatar = QLabel(tz("无头像"))
        self.avatar.setFixedSize(64, 64)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setStyleSheet("background:#eef1f6;border-radius:32px;color:#999;")
        head.addWidget(self.avatar)
        self.btn_avatar = QPushButton(tz("换头像…"))
        self.btn_avatar.clicked.connect(self._pick_avatar)
        head.addWidget(self.btn_avatar)
        head.addStretch(1)
        v.addLayout(head)

        form = QFormLayout()
        self.f_name = QLineEdit()
        self.f_greeting = QLineEdit()
        self.f_greeting.setPlaceholderText(tz("第一次见面时说的话"))
        form.addRow(tz("名字"), self.f_name)
        form.addRow(tz("开场白"), self.f_greeting)

        self.f_persona = QPlainTextEdit()
        self.f_persona.setPlaceholderText(
            tz("写她是谁、怎么说话、边界在哪。这段会作为 system prompt 的一部分发给 LLM。"))
        self.f_persona.setMinimumHeight(130)
        form.addRow(tz("人设"), self.f_persona)

        self.f_temp = QDoubleSpinBox()
        self.f_temp.setRange(0.0, 2.0)
        self.f_temp.setSingleStep(0.1)
        self.f_temp.setValue(0.8)
        self.f_max_tokens = QSpinBox()
        self.f_max_tokens.setRange(80, 512)
        self.f_max_tokens.setValue(160)
        row_llm = QHBoxLayout()
        row_llm.addWidget(QLabel(tz("温度")))
        row_llm.addWidget(self.f_temp)
        row_llm.addWidget(QLabel("max_tokens"))
        row_llm.addWidget(self.f_max_tokens)
        row_llm.addStretch(1)
        form.addRow("LLM", row_llm)
        form.addRow("", _hint(tz("温度越高越发散、越容易跑题；max_tokens 越大回复越长、越慢。")))
        v.addLayout(form)

        self.f_speed = QDoubleSpinBox()
        self.f_speed.setRange(0.7, 1.2)
        self.f_speed.setSingleStep(0.02)
        self.f_speed.setDecimals(2)
        self.f_speed.setValue(0.96)
        self.f_speed.setFixedWidth(90)
        self.f_speed.setMinimumHeight(26)
        self.f_gap = QSpinBox()
        self.f_gap.setRange(60, 400)
        self.f_gap.setSingleStep(10)
        self.f_gap.setValue(130)
        self.f_gap.setFixedWidth(90)
        self.f_gap.setMinimumHeight(26)
        self.f_vtemp = QDoubleSpinBox()
        self.f_vtemp.setRange(0.3, 1.0)
        self.f_vtemp.setSingleStep(0.05)
        self.f_vtemp.setDecimals(2)
        self.f_vtemp.setValue(0.6)
        self.f_vtemp.setFixedWidth(90)
        self.f_vtemp.setMinimumHeight(26)
        self.f_topk = QSpinBox()
        self.f_topk.setRange(1, 30)
        self.f_topk.setValue(5)
        self.f_topk.setFixedWidth(90)
        self.f_topk.setMinimumHeight(26)
        self.f_steps = QSpinBox()
        self.f_steps.setRange(4, 16)
        self.f_steps.setValue(4)
        self.f_steps.setFixedWidth(90)
        self.f_steps.setMinimumHeight(26)
        self.f_rp = QDoubleSpinBox()
        self.f_rp.setRange(1.2, 2.2)
        self.f_rp.setSingleStep(0.05)
        self.f_rp.setDecimals(2)
        self.f_rp.setValue(1.9)
        self.f_rp.setFixedWidth(90)
        self.f_rp.setMinimumHeight(26)
        self.f_interj = QDoubleSpinBox()
        self.f_interj.setRange(0.5, 3.0)
        self.f_interj.setSingleStep(0.1)
        self.f_interj.setDecimals(1)
        self.f_interj.setValue(1.2)
        self.f_interj.setFixedWidth(90)
        self.f_interj.setMinimumHeight(26)

        gb_voice = QGroupBox(tz("音色与节奏"))
        vb = QVBoxLayout(gb_voice)
        vb.setSpacing(2)
        for title, widget, hint in (
            ("语速", self.f_speed,
             "1.0 为原速；调低更从容，超过 1.05 容易糊、颗粒感重。"),
            ("句间停顿（ms）", self.f_gap,
             "两句之间留多久；太小显赶、容易拖尾，130-180 比较自然。"),
            ("音色温度", self.f_vtemp,
             "越高音色变化越多、情绪更活，但更容易拖音或跑调。"),
            ("top_k", self.f_topk,
             "采样范围；偏小更稳但容易把语气词拖长，10-15 更活泼。"),
            ("重复惩罚", self.f_rp,
             "越大越不容易拖音、复读；调到 1.8-2.0 效果明显，太大可能吞字。"),
            ("采样步数", self.f_steps,
             "v4 扩散步数；越大越细腻但更慢，4 最快，8 更稳。"),
            ("语气词上限（秒）", self.f_interj,
             "单独的“嗯/哦/唔”最多说多久，超过会被平滑收短，治单字延音。"),
        ):
            row = QHBoxLayout()
            lab = QLabel(tz(title))
            lab.setFixedWidth(110)
            row.addWidget(lab)
            row.addWidget(widget)
            row.addStretch(1)
            vb.addLayout(row)
            vb.addWidget(_hint(tz(hint)))
        v.addWidget(gb_voice)

        self.info = QLabel("—")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#666;font-size:12px;")
        for w in (self.f_name, self.f_greeting, self.f_temp, self.f_max_tokens,
                  self.f_speed, self.f_gap, self.f_vtemp, self.f_topk,
                  self.f_steps, self.f_interj, self.f_rp):
            sig = getattr(w, "valueChanged", None) or getattr(w, "textChanged", None)
            if sig is not None:
                sig.connect(self._mark_dirty)
        self.f_persona.textChanged.connect(self._mark_dirty)
        v.addWidget(self.info)

        row = QHBoxLayout()
        self.btn_save = QPushButton(tz("保存"))
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        self.btn_revert = QPushButton(tz("还原本地修改"))
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._revert)
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
            values = [name, tz(STATUS_TEXT.get(c.status, c.status)),
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

    def _refresh_list(self) -> None:
        """F5：重新读盘上的角色列表，保持当前选中项。"""
        keep = self._current_id
        self.refresh()
        if keep:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.data(Qt.UserRole) == keep:
                    self.table.selectRow(row)
                    break
        self.info.setText(tz("列表已刷新（F5）"))

    def _on_select(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._ids):
            self._current_id = self._ids[row]
            self._load_into_editor()

    def _revert(self) -> None:
        """还原本地修改：从磁盘重读该角色并刷新界面。"""
        if not self._current_id:
            return
        cid = self._current_id
        self._load_into_editor(cid)
        self.btn_save.setEnabled(False)
        self.btn_revert.setEnabled(False)
        self.info.setText(tz("已还原为磁盘上的版本"))

    def _load_into_editor(self, cid: str | None = None) -> None:
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
        voice = c.voice or {}
        self.f_speed.setValue(float(voice.get("speed", 0.96)))
        self.f_gap.setValue(int(voice.get("sentence_gap_ms", 130)))
        self.f_vtemp.setValue(float(voice.get("temperature", 0.6)))
        self.f_topk.setValue(int(voice.get("top_k", 5)))
        self.f_steps.setValue(int(voice.get("sample_steps", 4)))
        self.f_interj.setValue(float(voice.get("interjection_max_sec", 1.2)))
        self.f_rp.setValue(float(voice.get("repetition_penalty", 1.9)))
        av = self.store.avatar_path(c.id)
        if av.exists():
            pm = QPixmap(str(av)).scaled(64, 64, Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            self.avatar.setPixmap(pm)
            self.avatar.setText("")
        else:
            self.avatar.setPixmap(QPixmap())
            self.avatar.setText(tz("无头像"))
        self.info.setText(
            f"{tz('状态')}: {tz(STATUS_TEXT.get(c.status, c.status))} · "
            f"{tz('素材')} {c.dataset.get('duration_sec', 0)} s / "
            f"{c.dataset.get('clips', 0)} {tz('片段')}\n"
            f"gpt：{Path(c.voice.get('gpt_ckpt', '')).name or '—'}　"
            f"sovits：{Path(c.voice.get('sovits_ckpt', '')).name or '—'}")
        self.btn_save.setEnabled(True)
        self.btn_revert.setEnabled(True)

    # -- 动作 ----------------------------------------------------------
    def _mark_dirty(self, *_args) -> None:
        if self._current_id:
            self.btn_save.setEnabled(True)
            self.btn_revert.setEnabled(True)

    def _save(self) -> None:
        if not self._current_id:
            return
        name = self.f_name.text().strip()
        if not name:
            QMessageBox.warning(self, tz("名字不能为空"), "给角色起个名字。")
            return
        if self.store.name_taken(name, exclude=self._current_id):
            QMessageBox.warning(self, tz("重名"), f"已经有叫「{name}」的角色了。")
            return
        c = self.store.get(self._current_id)
        c.name = name
        c.greeting = self.f_greeting.text().strip()
        c.persona = self.f_persona.toPlainText().strip()
        c.llm = {"temperature": round(self.f_temp.value(), 2),
                 "max_tokens": int(self.f_max_tokens.value())}
        voice = dict(c.voice or {})
        voice.update({
            "speed": round(self.f_speed.value(), 2),
            "sentence_gap_ms": int(self.f_gap.value()),
            "temperature": round(self.f_vtemp.value(), 2),
            "top_k": int(self.f_topk.value()),
            "sample_steps": int(self.f_steps.value()),
            "interjection_max_sec": round(self.f_interj.value(), 2),
            "repetition_penalty": round(self.f_rp.value(), 2),
        })
        c.voice = voice
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

    def _train(self) -> None:
        if not self._current_id:
            return
        self.train_requested.emit(self._current_id)

    def _set_active(self) -> None:
        if not self._current_id:
            return
        self.store.set_active(self._current_id)
        self.refresh()
        self.changed.emit()

    def _export(self) -> None:
        target = cid or self._current_id
        if not target:
            return
        self._current_id = target
        c = self.store.get(target)
        if c.status != "ready":
            QMessageBox.warning(self, tz("无法导出"), "只有「可用」状态的角色才能导出。")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出角色包", f"{c.name}.vcp", "PrivateVoice 角色包 (*.vcp)")
        if not dest:
            return
        try:
            path = self.store.export_vcp(c.id, dest)
            QMessageBox.information(self, tz("导出完成"), f"已导出到：\n{path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tz("导出失败"), str(exc))

    def _import(self) -> None:
        src, _ = QFileDialog.getOpenFileName(
            self, "导入角色包", "", "PrivateVoice 角色包 (*.vcp)")
        if not src:
            return
        overwrite = QMessageBox.question(
            self, tz("导入"), "如果同名角色已存在，是否覆盖？",
            QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        try:
            c = self.store.import_vcp(src, overwrite=overwrite)
            QMessageBox.information(self, tz("导入完成"), f"角色「{c.name}」已就绪。")
            self.refresh()
            self.changed.emit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tz("导入失败"), str(exc))

    def _delete(self) -> None:
        target = cid or self._current_id
        if not target:
            return
        self._current_id = target
        c = self.store.get(target)
        if QMessageBox.question(self, tz("删除角色"),
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
