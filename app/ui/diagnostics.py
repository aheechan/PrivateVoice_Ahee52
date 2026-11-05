"""诊断页：GPU/驱动/torch/CUDA、模型 commit、最近 TTFP、导出诊断包。"""
from __future__ import annotations

from app import i18n
from app.i18n_tz import tz

import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.config import Config
from app.core import train
from app.core.metrics import MetricsStore


class DiagnosticsDialog(QDialog):
    closed = Signal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle(tz("诊断"))
        self.resize(720, 560)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        gb = QGroupBox(tz("运行环境"))
        form = QFormLayout(gb)
        self.labels: dict[str, QLabel] = {}
        for key, title in (
            ("gpu", "显卡"), ("vram", "显存"), ("driver", "驱动"),
            ("torch", "PyTorch"), ("cuda", "CUDA"), ("python", "Python"),
            ("commit", "模型 commit"), ("ttfp", "最近 TTFP"),
        ):
            lab = QLabel("—")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.labels[key] = lab
            form.addRow(tz(title), lab)
        root.addWidget(gb)

        # 最近 20 轮延迟
        gb2 = QGroupBox(tz("最近 20 轮延迟"))
        v2 = QVBoxLayout(gb2)
        self.summary = QLabel(tz("还没有对话记录"))
        self.summary.setStyleSheet("color:#333;")
        v2.addWidget(self.summary)
        self.turns = QTableWidget(0, 7)
        self.turns.setHorizontalHeaderLabels(
            [tz("时间"), "ASR(ms)", tz("LLM首token(ms)"), tz("TTS首包(ms)"),
             "e2e(ms)", tz("播放(s)"), "provider"])
        self.turns.setEditTriggers(QTableWidget.NoEditTriggers)
        self.turns.horizontalHeader().setStretchLastSection(True)
        self.turns.setMinimumHeight(180)
        v2.addWidget(self.turns)
        root.addWidget(gb2)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText(tz("这里显示诊断包导出结果与最近日志"))
        root.addWidget(self.log, 1)

        row = QHBoxLayout()
        self.refresh_btn = QPushButton(tz("重新检测"))
        self.refresh_btn.clicked.connect(self.refresh)
        self.export_btn = QPushButton(tz("导出诊断包…"))
        self.export_btn.clicked.connect(self._export)
        row.addWidget(self.refresh_btn)
        row.addWidget(self.export_btn)
        row.addStretch(1)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.clicked.connect(lambda _b: self.reject())
        root.addWidget(btns)

    # -- 采集 ----------------------------------------------------------
    def refresh(self) -> None:
        info = train.gpu_info()
        self.labels["gpu"].setText(info.name or "未检测到")
        self.labels["vram"].setText(
            f"{info.total_gb:.1f} GB（当前可用 {info.free_gb:.1f} GB）" if info.name else "—")
        self.labels["driver"].setText(self._driver())
        self.labels["torch"].setText(info.torch or "—")
        self.labels["cuda"].setText(info.cuda or "—")
        self.labels["python"].setText(f"{platform.python_version()} · {sys.executable}")
        self.labels["commit"].setText(self._commit())
        self.labels["ttfp"].setText(self._ttfp())
        self._load_turns()

    def _load_turns(self) -> None:
        store = MetricsStore(Path(self.config.path_of("logs")) / "turns.jsonl")
        rows = store.recent(20)
        summary = store.summary(20)
        parts = []
        for key, name in (("asr_ms", "ASR"), ("llm_ttft_ms", "LLM"),
                          ("tts_ttfp_ms", "TTS"), ("e2e_ms", "e2e")):
            s = summary.get(key) or {}
            if s.get("n"):
                parts.append(f"{name} p50={s['p50']:.0f} / p95={s['p95']:.0f} ms")
        self.summary.setText(
            f"共 {summary.get('count', 0)} 轮 · " + " · ".join(parts)
            if parts else "还没有对话记录")
        self.turns.setRowCount(len(rows))
        for r, row in enumerate(rows):
            ts = str(row.get("ts", ""))[11:19]
            vals = [ts,
                    row.get("asr_ms"), row.get("llm_ttft_ms"),
                    row.get("tts_ttfp_ms"), row.get("e2e_ms"),
                    row.get("play_ms"), row.get("provider", "")]
            for c, v in enumerate(vals):
                if isinstance(v, (int, float)):
                    v = f"{v:.0f}"
                self.turns.setItem(r, c, QTableWidgetItem("—" if v in (None, "") else str(v)))
        self.turns.resizeColumnsToContents()
        self.turns.horizontalHeader().setStretchLastSection(True)

    def _driver(self) -> str:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or "—"
        except Exception:
            return "—"

    def _commit(self) -> str:
        repo = Path(self.config.path_of("gpt_sovits"))
        try:
            out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip()[:12] or "—"
        except Exception:
            return "未知（未找到 git 仓库）"

    def _ttfp(self) -> str:
        for name in ("character_ttfp.json", "spike.json"):
            p = Path(self.config.path_of("results")) / name
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tiers = data.get("tiers") or (data.get("nokernel") or {}).get("tiers") or {}
            parts = [f"{k}: {v.get('ttfp_p50_ms', 0):.0f}ms"
                     for k, v in tiers.items() if isinstance(v, dict)]
            if parts:
                return f"{'; '.join(parts)}（{name}）"
        return "还没有跑过基准"

    # -- 导出 ----------------------------------------------------------
    def _export(self) -> None:
        out = Path(self.config.path_of("logs")) / \
            f"diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        info = train.gpu_info()
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "gpu": info.describe(),
            "torch": info.torch,
            "cuda": info.cuda,
            "config": self.config.data,
        }
        try:
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("diagnostics.json",
                           json.dumps(payload, ensure_ascii=False, indent=2))
                logs = Path(self.config.path_of("logs"))
                for f in sorted(logs.glob("*.log"))[-3:]:
                    z.write(f, f"logs/{f.name}")
                for name in ("character_ttfp.json", "spike.json"):
                    p = Path(self.config.path_of("results")) / name
                    if p.exists():
                        z.write(p, f"results/{name}")
            self.log.appendPlainText(f"已导出：{out}")
        except Exception as exc:
            self.log.appendPlainText(f"导出失败：{exc}")

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["DiagnosticsDialog"]
