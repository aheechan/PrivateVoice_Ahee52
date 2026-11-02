"""首启向导：自检 GPU、模型目录、LLM provider，然后写 first_run_done。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from app.config import Config
from app.core import train
from app.core.llm import LLMClient, ProviderStore


def _mark(ok: bool, text: str) -> QLabel:
    lab = QLabel(("✅ " if ok else "❌ ") + text)
    lab.setStyleSheet("color:#2d7d46;" if ok else "color:#c0392b;")
    return lab


class FirstRunWizard(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("情感陪伴器_v1 · 首次运行")
        self.resize(620, 520)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("欢迎使用 情感陪伴器_v1")
        title.setStyleSheet("font-size:20px;font-weight:600;")
        root.addWidget(title)
        sub = QLabel("这台电脑第一次运行，先做三项自检。全部通过就可以按住热键说话了。")
        sub.setStyleSheet("color:#666;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # 1 GPU
        gb1 = QGroupBox("1 · 显卡")
        v1 = QVBoxLayout(gb1)
        info = train.gpu_info()
        ok, msg = train.check_hardware(self.config)
        self.gpu_ok = ok
        v1.addWidget(_mark(ok, msg))
        if ok:
            tip = QLabel("支持 RTX 30/40/50 系（sm_86/89/120），显存 ≥12GB。")
            tip.setStyleSheet("color:#666;")
            v1.addWidget(tip)
        root.addWidget(gb1)

        # 2 模型目录
        gb2 = QGroupBox("2 · 模型文件")
        v2 = QVBoxLayout(gb2)
        row = QHBoxLayout()
        self.model_dir = QLineEdit(str(self.config.path_of("gpt_sovits")))
        btn = QPushButton("检查")
        btn.clicked.connect(self._check_models)
        row.addWidget(self.model_dir, 1)
        row.addWidget(btn)
        v2.addLayout(row)
        self.model_status = QLabel("点“检查”确认权重是否齐全")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("color:#666;")
        v2.addWidget(self.model_status)
        root.addWidget(gb2)

        # 3 LLM
        gb3 = QGroupBox("3 · 对话模型（OpenAI 兼容）")
        form = QFormLayout(gb3)
        prov = ProviderStore(self.config).active()
        first = prov[0] if prov else None
        self.base_url = QLineEdit(first.base_url if first else "http://127.0.0.1:8082/v1")
        self.model = QLineEdit(first.model if first else "Qwen3.5-9B")
        self.test_result = QLabel("未测试")
        self.test_result.setStyleSheet("color:#666;")
        btn_test = QPushButton("测试连接")
        btn_test.clicked.connect(self._test_llm)
        form.addRow("Base URL", self.base_url)
        form.addRow("模型名", self.model)
        row3 = QHBoxLayout()
        row3.addWidget(btn_test)
        row3.addWidget(self.test_result, 1)
        form.addRow("", row3)
        root.addWidget(gb3)

        hint = QLabel(f"当前热键：{self.config.get('hotkey.spec', 'shift+a')}（按住说话，"
                      f"松开结束；可在设置里改）")
        hint.setStyleSheet("color:#666;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox()
        self.btn_ok = btns.addButton("开始使用", QDialogButtonBox.AcceptRole)
        self.btn_quit = btns.addButton("退出", QDialogButtonBox.RejectRole)
        btns.accepted.connect(self._finish)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
        self._check_models()

    # -- 检查 ----------------------------------------------------------
    def _check_models(self) -> None:
        base = Path(self.model_dir.text().strip())
        pm = base / "GPT_SoVITS" / "pretrained_models"
        need = {
            "s1v3.ckpt": pm / "s1v3.ckpt",
            "s2Gv3.pth": pm / "s2Gv3.pth",
            "chinese-hubert-base": pm / "chinese-hubert-base",
            "chinese-roberta-wwm-ext-large": pm / "chinese-roberta-wwm-ext-large",
            "G2PWModel": base / "GPT_SoVITS" / "text" / "G2PWModel",
        }
        missing = [k for k, p in need.items() if not p.exists()]
        if missing:
            self.model_ok = False
            self.model_status.setText("缺少：" + "、".join(missing) +
                                      "\n请把完整模型目录拷进来，或先运行下载脚本。")
            self.model_status.setStyleSheet("color:#c0392b;")
        else:
            self.model_ok = True
            self.model_status.setText("模型齐全（v3 权重 + 文本前端）。")
            self.model_status.setStyleSheet("color:#2d7d46;")

    def _test_llm(self) -> None:
        self.test_result.setText("测试中…")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        provs = ProviderStore(self.config).active()
        if not provs:
            self.test_result.setText("没有启用的 provider（可在“大模型设置”里加）")
            return
        p = provs[0]
        p.base_url = self.base_url.text().strip() or p.base_url
        p.model = self.model.text().strip() or p.model
        res = LLMClient(self.config).test(p)
        if res.get("ok"):
            self.test_result.setText(f"✅ {res.get('model')} · 首 token "
                                     f"{res.get('first_token_ms')} ms")
            self.test_result.setStyleSheet("color:#2d7d46;")
        else:
            self.test_result.setText("❌ " + str(res.get("error"))[:80])
            self.test_result.setStyleSheet("color:#c0392b;")

    # -- 完成 ----------------------------------------------------------
    def _finish(self) -> None:
        if not getattr(self, "model_ok", False):
            QMessageBox.warning(self, "模型不齐", "模型文件不齐全，先补齐再开始。")
            return
        self.config.set("paths.gpt_sovits", self.model_dir.text().strip())
        provs = ProviderStore(self.config).list()
        if provs:
            provs[0].base_url = self.base_url.text().strip() or provs[0].base_url
            provs[0].model = self.model.text().strip() or provs[0].model
            ProviderStore(self.config).save(provs)
        self.config.set("ui.first_run_done", True)
        self.config.save()
        self.accept()


__all__ = ["FirstRunWizard"]
