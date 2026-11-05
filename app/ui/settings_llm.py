"""LLM provider 管理：增删改、排序、连接测试、DPAPI 密钥。"""
from __future__ import annotations

from app import i18n
from app.i18n_tz import tz

import time
import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from app.config import Config
from app.core.llm import LLMClient, Provider, ProviderStore, dpapi_encrypt, mask_key


class ProviderDialog(QDialog):
    """单个 provider 的编辑框。"""

    def __init__(self, provider: Provider | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tz("编辑 Provider") if provider else "新增 Provider")
        self.resize(520, 420)
        self._provider = provider
        self._api_key: str = ""       # 明文只存在内存，落盘前加密

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(provider.name if provider else "")
        self.base_url = QLineEdit(provider.base_url if provider else "http://127.0.0.1:8082/v1")
        self.model = QLineEdit(provider.model if provider else "")
        self.priority = QSpinBox()
        self.priority.setRange(1, 999)
        self.priority.setValue(provider.priority if provider else 10)
        self.timeout = QSpinBox()
        self.timeout.setRange(1000, 120000)
        self.timeout.setSingleStep(1000)
        self.timeout.setValue(provider.timeout_ms if provider else 30000)
        self.enabled = QCheckBox(tz("启用"))
        self.enabled.setChecked(provider.enabled if provider else True)
        self.stream = QCheckBox(tz("支持流式（stream=true）"))
        self.stream.setChecked(provider.supports_stream if provider else True)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText(
            f"已保存：{mask_key(provider.api_key)}，留空则不修改" if provider and provider.api_key_enc
            else "可留空（局域网自建通常不需要）")
        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText('额外请求体（JSON），例如 {"chat_template_kwargs":{"enable_thinking":false}}')
        self.extra.setPlainText(
            "{}" if not provider or not provider.extra_body
            else __import__("json").dumps(provider.extra_body, ensure_ascii=False, indent=2))

        form.addRow(tz("名称"), self.name)
        form.addRow("Base URL", self.base_url)
        form.addRow(tz("模型名"), self.model)
        form.addRow(tz("优先级（小优先）"), self.priority)
        form.addRow(tz(超时 (ms)), self.timeout)
        form.addRow("API Key", self.key)
        form.addRow("", self.enabled)
        form.addRow("", self.stream)
        form.addRow("extra_body", self.extra)
        root.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _accept(self) -> None:
        import json
        if not self.name.text().strip() or not self.base_url.text().strip():
            QMessageBox.warning(self, tz("缺少字段"), "名称和 Base URL 必填。")
            return
        try:
            extra = json.loads(self.extra.toPlainText() or "{}")
            if not isinstance(extra, dict):
                raise ValueError
        except Exception:
            QMessageBox.warning(self, tz("JSON 无效"), "extra_body 必须是 JSON 对象。")
            return
        p = self._provider or Provider(
            id=f"p-{uuid.uuid4().hex[:8]}", name="", base_url="", model="")
        p.name = self.name.text().strip()
        p.base_url = self.base_url.text().strip()
        p.model = self.model.text().strip() or "default"
        p.priority = self.priority.value()
        p.timeout_ms = self.timeout.value()
        p.enabled = self.enabled.isChecked()
        p.supports_stream = self.stream.isChecked()
        p.extra_body = extra
        if self.key.text().strip():
            p.api_key_enc = dpapi_encrypt(self.key.text().strip())
        self._provider = p
        self.accept()

    @property
    def provider(self) -> Provider | None:
        return self._provider


class SettingsLLM(QDialog):
    closed = Signal()
    providers_changed = Signal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.store = ProviderStore(config)
        self.client = LLMClient(config)
        self.setWindowTitle(tz("大模型 Provider 管理"))
        self.resize(980, 520)
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(tz("按优先级从小到大依次尝试；失败或超时自动切下一个。"
                        "密钥用 Windows DPAPI 加密后存进 config.yaml，界面只显示尾 4 位。"))
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;")
        root.addWidget(tip)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [tz("启用"), tz("名称"), "Base URL", tz("模型"),
             tz("优先级"), "Key", tz("最近测试")])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        root.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.btn_add = QPushButton(tz("新增"))
        self.btn_edit = QPushButton(tz("编辑"))
        self.btn_del = QPushButton(tz("删除"))
        self.btn_up = QPushButton(tz("上移"))
        self.btn_down = QPushButton(tz("下移"))
        self.btn_test = QPushButton(tz("连接测试"))
        for b in (self.btn_add, self.btn_edit, self.btn_del, self.btn_up,
                  self.btn_down, self.btn_test):
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumHeight(120)
        root.addWidget(self.result)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.btn_add.clicked.connect(self._add)
        self.btn_edit.clicked.connect(self._edit)
        self.btn_del.clicked.connect(self._delete)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_test.clicked.connect(self._test)

    # -- 数据 ----------------------------------------------------------
    def refresh(self) -> None:
        provs = sorted(self.store.list(), key=lambda p: p.priority)
        self._ids = [p.id for p in provs]
        self._tests = getattr(self, "_tests", {})
        self.table.setRowCount(len(provs))
        for r, p in enumerate(provs):
            self.table.setItem(r, 0, QTableWidgetItem("✓" if p.enabled else "—"))
            self.table.setItem(r, 1, QTableWidgetItem(p.name))
            self.table.setItem(r, 2, QTableWidgetItem(p.base_url))
            self.table.setItem(r, 3, QTableWidgetItem(p.model))
            self.table.setItem(r, 4, QTableWidgetItem(str(p.priority)))
            self.table.setItem(r, 5, QTableWidgetItem(p.masked_key or "—"))
            self.table.setItem(r, 6, QTableWidgetItem(self._tests.get(p.id, "未测试")))
        self._update_buttons()

    def _selected(self) -> Provider | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(getattr(self, "_ids", [])):
            return None
        return self.store.get(self._ids[row])

    def _update_buttons(self) -> None:
        p = self._selected()
        for b in (self.btn_edit, self.btn_del, self.btn_up, self.btn_down, self.btn_test):
            b.setEnabled(p is not None)

    # -- 动作 ----------------------------------------------------------
    def _add(self) -> None:
        dlg = ProviderDialog(parent=self)
        if dlg.exec() and dlg.provider:
            self.store.upsert(dlg.provider)
            self.config.save()
            self.refresh()
            self.providers_changed.emit()

    def _edit(self) -> None:
        p = self._selected()
        if not p:
            return
        dlg = ProviderDialog(p, parent=self)
        if dlg.exec() and dlg.provider:
            self.store.upsert(dlg.provider)
            self.config.save()
            self.refresh()
            self.providers_changed.emit()

    def _delete(self) -> None:
        p = self._selected()
        if not p:
            return
        if QMessageBox.question(self, tz("删除"), f"删除 provider「{p.name}」？") \
                != QMessageBox.Yes:
            return
        self.store.remove(p.id)
        self.config.save()
        self.refresh()
        self.providers_changed.emit()

    def _move(self, delta: int) -> None:
        p = self._selected()
        if not p:
            return
        self.store.move(p.id, delta)
        self.config.save()
        self.refresh()
        self.providers_changed.emit()

    def _test(self) -> None:
        p = self._selected()
        if not p:
            return
        self.result.appendPlainText(f"测试 {p.name} …")
        QApplication = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication
        QApplication.processEvents()
        t0 = time.perf_counter()
        res = self.client.test(p)
        if res.get("ok"):
            text = (f"✓ {p.name} · 模型 {res.get('model')} · 首 token "
                    f"{res.get('first_token_ms')} ms · 总 {res.get('elapsed_ms')} ms")
        else:
            text = f"✗ {p.name} · {res.get('error')}"
        self.result.appendPlainText(f"{text}（{time.perf_counter()-t0:.1f}s）")
        self._tests[p.id] = "OK" if res.get("ok") else "失败"
        self.refresh()

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["SettingsLLM", "ProviderDialog"]
