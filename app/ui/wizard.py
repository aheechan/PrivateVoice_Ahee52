"""新建角色向导（6 步）：基本信息 → 上传/录制 → 质量检查 → 切分打标 → 训练 → 试听。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QWizard, QWizardPage,
)

from app.config import Config
from app.core import audio_io, audio_qc, segment, train
from app.core.characters import CharacterStore
from app.core.label import Transcriber, describe_flags, flag_level
from app.core.tts import make_engine

SUPPORTED = sorted(audio_qc.SUPPORTED_EXT)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
class FnWorker(QThread):
    """把阻塞调用丢到线程里，结果/异常回到主线程。"""

    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    event = Signal(object)      # 训练进度（跨线程安全地回 GUI）

    def __init__(self, fn: Callable[[Callable[[str], None]], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn(self.progress.emit))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class WaveformWidget(QWidget):
    """极简波形缩略图：按 1px 列画 min/max。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self._samples: np.ndarray | None = None

    def set_audio(self, samples: np.ndarray | None) -> None:
        self._samples = samples
        self.update()

    def load_file(self, path: str | Path) -> None:
        try:
            data, _sr = audio_qc.decode(path)
            self.set_audio(audio_qc.to_mono(data))
        except Exception:
            self.set_audio(None)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#11151c"))
        p.setPen(QPen(QColor("#3fb950"), 1))
        w, h = self.width(), self.height()
        mid = h / 2
        if self._samples is None or self._samples.size == 0:
            p.setPen(QColor("#666"))
            p.drawText(self.rect(), Qt.AlignCenter, "没有音频")
            return
        x = self._samples
        step = max(1, x.size // max(1, w))
        for i in range(w):
            chunk = x[i * step:(i + 1) * step]
            if chunk.size == 0:
                break
            lo, hi = float(chunk.min()), float(chunk.max())
            p.drawLine(i, int(mid - hi * mid), i, int(mid - lo * mid))


# ---------------------------------------------------------------------------
# 向导
# ---------------------------------------------------------------------------
class NewCharacterWizard(QWizard):
    training_state = Signal(bool, str)

    def __init__(self, config: Config, store: CharacterStore, parent=None,
                 character_id: str | None = None):
        super().__init__(parent)
        self.config = config
        self.store = store
        self.state: dict[str, Any] = {}
        if character_id:
            self.state["cid"] = character_id
        self.setWindowTitle("新建角色 — 情感陪伴器" if not character_id
                            else "训练 / 重训角色 — 情感陪伴器")
        self.resize(900, 660)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setButtonText(QWizard.NextButton, "下一步")
        self.setButtonText(QWizard.BackButton, "上一步")
        self.setButtonText(QWizard.CancelButton, "取消")
        self.setButtonText(QWizard.FinishButton, "完成")

        self.p1 = BasicPage(self)
        self.p2 = AudioPage(self)
        self.p3 = QcPage(self)
        self.p4 = SegmentPage(self)
        self.p5 = TrainPage(self)
        self.p6 = PreviewPage(self)
        self._page_ids = [self.addPage(page) for page in
                          (self.p1, self.p2, self.p3, self.p4, self.p5, self.p6)]
        if character_id:
            # 从角色库进来：跳过「基本信息」，直接到上传素材
            ch = store.get(character_id)
            self.p1.name.setText(ch.name)
            self.p1.persona.setPlainText(ch.persona or "")
            self.p1.greeting.setText(ch.greeting or "")
            self.state["name"] = ch.name
            self.setStartId(self._page_ids[1])

    # -- 便捷 ----------------------------------------------------------
    @property
    def char_id(self) -> str:
        return self.state.get("cid", "")

    def dataset_dir(self) -> Path:
        d = Path(self.config.path_of("datasets")) / self.char_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def char_dir(self) -> Path:
        return self.store.dir_of(self.char_id)


# ---------------------------------------------------------------------------
# 1 基本信息
# ---------------------------------------------------------------------------
class BasicPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 1 步 · 基本信息")
        self.setSubTitle("给她一个名字、一张脸和一段人设。这些以后都能改。")

        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText("例如：小满")
        self.name.textChanged.connect(self._on_name)
        form.addRow("名字 *", self.name)

        row = QHBoxLayout()
        self.avatar_label = QLabel("未选择（可跳过，默认灰色头像）")
        self.avatar_label.setStyleSheet("color:#666;")
        btn = QPushButton("选择图片…")
        btn.clicked.connect(self._pick_avatar)
        row.addWidget(self.avatar_label, 1)
        row.addWidget(btn)
        form.addRow("头像", row)

        self.persona = QPlainTextEdit()
        self.persona.setPlaceholderText(
            "她是谁、怎么说话、边界在哪。例如：25 岁插画师，说话轻、句子短，"
            "习惯用“嗯”开头，会关心对方有没有吃饭，但不追问隐私。")
        self.persona.setMinimumHeight(120)
        form.addRow("人设描述", self.persona)

        self.greeting = QLineEdit()
        self.greeting.setPlaceholderText("她见到你时说的第一句话")
        form.addRow("开场白", self.greeting)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color:#c33;")
        form.addRow("", self.hint)

    def _on_name(self, text: str) -> None:
        taken = text.strip() and self.wiz.store.name_taken(text.strip())
        self.hint.setText(f"「{text.strip()}」已经有一个同名角色了。" if taken else "")
        self.completeChanged.emit()

    def _pick_avatar(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择头像", "", "图片 (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.state_avatar = path
            self.avatar_label.setText(Path(path).name)
            self.avatar_label.setStyleSheet("color:#333;")

    def isComplete(self) -> bool:
        name = self.name.text().strip()
        return bool(name) and not self.wiz.store.name_taken(name)

    def validatePage(self) -> bool:  # noqa: N802
        name = self.name.text().strip()
        if not name:
            return False
        if not self.wiz.char_id:
            ch = self.wiz.store.create(
                name,
                persona=self.persona.toPlainText().strip(),
                greeting=self.greeting.text().strip(),
            )
            self.wiz.state["cid"] = ch.id
            self.wiz.state["name"] = name
            if getattr(self, "state_avatar", ""):
                try:
                    self.wiz.store.set_avatar(ch.id, self.state_avatar)
                except Exception:
                    pass
        else:
            ch = self.wiz.store.get(self.wiz.char_id)
            ch.name = name
            ch.persona = self.persona.toPlainText().strip()
            ch.greeting = self.greeting.text().strip()
            self.wiz.store.save(ch)
        return True


# ---------------------------------------------------------------------------
# 2 上传 / 录制
# ---------------------------------------------------------------------------
class AudioPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 2 步 · 上传或录制声音")
        self.setSubTitle("45–90 秒单人说话，安静房间，麦克风离嘴 15–20 厘米。")

        root = QVBoxLayout(self)
        self.drop = QLabel("把 wav / mp3 / m4a / flac 拖到这里，或点下面的按钮")
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setMinimumHeight(70)
        self.drop.setStyleSheet(
            "border:2px dashed #888; border-radius:8px; color:#666; padding:12px;")
        self.drop.setAcceptDrops(True)
        self.drop.dragEnterEvent = self._drag_enter       # type: ignore[assignment]
        self.drop.dropEvent = self._drop                  # type: ignore[assignment]
        root.addWidget(self.drop)

        row = QHBoxLayout()
        self.btn_file = QPushButton("选择文件…")
        self.btn_rec = QPushButton("开始录制")
        self.btn_file.clicked.connect(self._pick)
        self.btn_rec.clicked.connect(self._toggle_record)
        row.addWidget(self.btn_file)
        row.addWidget(self.btn_rec)
        row.addStretch(1)
        root.addLayout(row)

        self.meter = QProgressBar()
        self.meter.setRange(-60, 0)
        self.meter.setValue(-60)
        self.meter.setFormat("电平 %v dBFS")
        root.addWidget(self.meter)

        self.wave = WaveformWidget()
        root.addWidget(self.wave, 1)

        self.info = QLabel("还没有选择音频")
        self.info.setStyleSheet("color:#333;")
        root.addWidget(self.info)

        self._recorder: audio_io.Recorder | None = None
        self._rec_started = 0.0
        self._timer = None

    # 拖放
    def _drag_enter(self, event):  # noqa: ANN001
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop(self, event):  # noqa: ANN001
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in SUPPORTED:
                self._set_source(path)
                return

    def _pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "", "音频 (*.wav *.mp3 *.m4a *.flac)")
        if path:
            self._set_source(path)

    def _set_source(self, path: str) -> None:
        # 换了音频，之前跑出来的质检/切分结果全部作废
        self.wiz.state.pop("qc", None)
        self.wiz.state.pop("seg", None)
        self.wiz.state["source"] = path
        self.wave.load_file(path)
        try:
            info = audio_qc.probe(path)
            self.info.setText(
                f"{Path(path).name} · {info['duration']:.1f} 秒 · "
                f"{info['samplerate']} Hz · {info['channels']} 声道 · {info.get('format','')}")
            dur = float(info["duration"])
            if dur < self.wiz.config.qc["min_duration_sec"]:
                self.info.setText(self.info.text() +
                                  f"  ⚠ 不足 {self.wiz.config.qc['min_duration_sec']:.0f} 秒")
        except Exception as exc:
            self.info.setText(f"读取失败：{exc}")
        self.completeChanged.emit()

    # 录制
    def _toggle_record(self) -> None:
        from PySide6.QtCore import QTimer

        if self._recorder is None:
            device = self.wiz.config.audio.get("input_device")
            self._recorder = audio_io.Recorder(device=device)
            self._recorder.on_level = self._on_level
            self._recorder.start()
            self._rec_started = time.time()
            self.btn_rec.setText("停止并保存")
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(200)
        else:
            rec = self._recorder
            sr = int(getattr(rec, "samplerate", 48000))
            data = rec.stop()
            self._recorder = None
            if self._timer:
                self._timer.stop()
            self.btn_rec.setText("开始录制")
            dest = self.wiz.dataset_dir() / "raw.wav"
            audio_io.save_recording(data, sr, dest)
            self._set_source(str(dest))

    def _tick(self) -> None:
        if self._recorder:
            self.info.setText(f"录制中… {self._recorder.elapsed:.1f} 秒")

    def _on_level(self, db: float) -> None:
        self.meter.setValue(int(max(-60, min(0, db))))
        color = {"green": "#2d7d46", "yellow": "#b58900", "red": "#c0392b"}[
            audio_io.level_color(db)]
        self.meter.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}")

    def isComplete(self) -> bool:
        src = self.wiz.state.get("source", "")
        return bool(src) and Path(src).exists()


# ---------------------------------------------------------------------------
# 3 质量检查
# ---------------------------------------------------------------------------
class QcPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 3 步 · 质量检查")
        self.setSubTitle("不合格的项会给出具体原因和修改建议。")
        root = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["项目", "结果", "说明"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_rerun = QPushButton("重新检测")
        self.btn_rerun.clicked.connect(self._run)
        row.addWidget(self.btn_rerun)
        self.status = QLabel("等待检测…")
        row.addWidget(self.status, 1)
        root.addLayout(row)
        self._worker = None

    def initializePage(self) -> None:  # noqa: N802
        if "qc" not in self.wiz.state:
            self._run()
        else:
            self._fill(self.wiz.state["qc"])

    def _run(self) -> None:
        src = self.wiz.state.get("source")
        if not src:
            self.status.setText("没有音频")
            return
        self.status.setText("检测中…")
        self.btn_rerun.setEnabled(False)

        def job(_progress):
            return audio_qc.run_qc(src, self.wiz.config)

        self._worker = FnWorker(job, self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result: audio_qc.QCResult) -> None:
        self.wiz.state["qc"] = result
        audio_qc.save_qc(result, self.wiz.dataset_dir() / "qc.json")
        self._fill(result)
        self.btn_rerun.setEnabled(True)
        self.completeChanged.emit()

    def _on_fail(self, msg: str) -> None:
        self.status.setText(f"检测失败：{msg}")
        self.btn_rerun.setEnabled(True)

    def _fill(self, result: audio_qc.QCResult) -> None:
        self.table.setRowCount(len(result.checks))
        for r, c in enumerate(result.checks):
            self.table.setItem(r, 0, QTableWidgetItem(c.label))
            level_text = {"ok": "通过", "warn": "警告", "error": "不通过",
                          "skip": "跳过"}[c.level]
            item = QTableWidgetItem(level_text)
            item.setForeground(QColor({"ok": "#2d7d46", "warn": "#b58900",
                                       "error": "#c0392b", "skip": "#888"}[c.level]))
            self.table.setItem(r, 1, item)
            self.table.setItem(r, 2, QTableWidgetItem(c.message))
            if c.advice:
                self.table.item(r, 2).setToolTip(c.advice)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        summary = {"ok": "全部通过", "warn": "有警告，可以继续",
                   "error": "有不合格项，修好再继续"}[result.status]
        self.status.setText(
            f"{summary} · 时长 {result.duration_sec:.1f}s · 响度 {result.lufs} LUFS · "
            f"信噪比 {result.snr_db} dB")

    def isComplete(self) -> bool:
        qc = self.wiz.state.get("qc")
        return bool(qc and qc.ok)


# ---------------------------------------------------------------------------
# 4 切分与打标
# ---------------------------------------------------------------------------
class SegmentPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 4 步 · 切分与打标")
        self.setSubTitle("改了文本、标了丢弃的，点「保存修改」再继续。")
        root = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "时长", "文本（可编辑）", "标记", "试听", "提示"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        root.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_save = QPushButton("保存修改")
        self.btn_save.clicked.connect(self._save)
        self.btn_play_sel = QPushButton("试听选中")
        self.btn_play_sel.clicked.connect(self._play_selected)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_play_sel)
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        root.addLayout(row)
        self._worker = None
        self._player = audio_io.Player(device=None)

    def initializePage(self) -> None:  # noqa: N802
        if "seg" in self.wiz.state:
            return
        self.status.setText("VAD 切分 + 识别中，第一次运行要下载模型，请稍候…")

        def job(progress):
            tr = Transcriber(self.wiz.config.label_cfg)
            return segment.segment_file(
                self.wiz.state["source"], self.wiz.config, self.wiz.dataset_dir(),
                transcriber=tr, character_id=self.wiz.char_id, progress=progress)

        self._worker = FnWorker(job, self)
        self._worker.progress.connect(lambda m: self.status.setText(m))
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(lambda m: self.status.setText(f"失败：{m}"))
        self._worker.start()

    def _on_done(self, result: segment.SegmentResult) -> None:
        self.wiz.state["seg"] = result
        rows = result.rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(f"{row.index:02d}"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{row.duration:.2f}s"))
            item = QTableWidgetItem(row.text)
            self.table.setItem(r, 2, item)
            combo = QComboBox()
            combo.addItems(["好", "可疑", "丢弃"])
            combo.setCurrentIndex({"good": 0, "suspect": 1, "discard": 2}.get(row.status, 0))
            self.table.setCellWidget(r, 3, combo)
            btn = QPushButton("▶")
            btn.clicked.connect(lambda _=False, i=r: self._play(i))
            self.table.setCellWidget(r, 4, btn)
            hint = describe_flags(row.flags)
            self.table.setItem(r, 5, QTableWidgetItem(hint))
            level = flag_level(row.flags)
            self.table.item(r, 5).setForeground(QColor(
                {"red": "#c0392b", "yellow": "#b58900", "green": "#2d7d46"}[level]))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        warn = "；".join(result.warnings) if result.warnings else "分布正常"
        self.status.setText(f"{len(rows)} 个片段 · {warn}")
        self.wiz.store.set_dataset(
            self.wiz.char_id, sum(r.duration for r in rows), len(rows))
        self.completeChanged.emit()

    def _play(self, index: int) -> None:
        res: segment.SegmentResult = self.wiz.state["seg"]
        if 0 <= index < len(res.rows):
            self._player.play_file(str(res.dataset_dir / res.rows[index].clip),
                                   blocking=False)

    def _play_selected(self) -> None:
        self._play(self.table.currentRow())

    def _save(self) -> None:
        res: segment.SegmentResult = self.wiz.state.get("seg")
        if not res:
            return
        status_map = {0: "good", 1: "suspect", 2: "discard"}
        for r, row in enumerate(res.rows):
            item = self.table.item(r, 2)
            if item is not None:
                row.text = item.text()
            combo = self.table.cellWidget(r, 3)
            if isinstance(combo, QComboBox):
                row.status = status_map.get(combo.currentIndex(), "good")
        segment.write_dataset(res, character_id=self.wiz.char_id)
        kept = [r for r in res.rows if r.status != "discard"]
        self.wiz.store.set_dataset(
            self.wiz.char_id, sum(r.duration for r in kept), len(kept))
        self.status.setText(f"已保存：保留 {len(kept)} 段")

    def validatePage(self) -> bool:  # noqa: N802
        self._save()
        return True


# ---------------------------------------------------------------------------
# 5 训练
# ---------------------------------------------------------------------------
class TrainPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 5 步 · 训练")
        self.setSubTitle("训练在独立子进程里跑，可以随时取消；取消会清理临时文件。")
        root = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("训练档位"))
        self.profile = QComboBox()
        from app.config import PROFILE_MATRIX
        for key, meta in PROFILE_MATRIX.items():
            self.profile.addItem(f"{meta['label']}（{meta['epochs']} epochs）", key)
        row.addWidget(self.profile)
        self.plan_label = QLabel("")
        row.addWidget(self.plan_label, 1)
        root.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        # 显存提示：语音模型常驻会占 2-4GB，训练期间容易 OOM，先让用户手动释放
        self.gpu_tip = QLabel("")
        self.gpu_tip.setWordWrap(True)
        self.gpu_tip.setObjectName("Muted")
        root.addWidget(self.gpu_tip)

        self.status = QLabel("尚未开始")
        root.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        root.addWidget(self.log)

        row2 = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_cancel = QPushButton("取消训练")
        self.btn_cancel.setEnabled(False)
        self.btn_detail = QPushButton("展开日志")
        row2.addWidget(self.btn_start)
        row2.addWidget(self.btn_cancel)
        row2.addWidget(self.btn_detail)
        row2.addStretch(1)
        root.addLayout(row2)

        self.btn_start.clicked.connect(self._start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_detail.clicked.connect(self._toggle_log)
        self._runner: train.TrainingRunner | None = None
        self._thread: QThread | None = None
        self._done = False

    def initializePage(self) -> None:  # noqa: N802
        info = train.gpu_info()
        try:
            plan = self.wiz.config.plan(info.total_gb, self.profile.currentData())
            self.plan_label.setText(
                f"{info.name} · {info.total_gb:.0f}GB · 可用 {info.free_gb:.1f}GB · "
                f"batch={plan.batch} · 片段≤{plan.max_clip_sec}s · "
                f"有效 batch={plan.effective_batch}")
        except Exception as exc:
            self.plan_label.setText(str(exc))
        if not self._done and self._thread is None:
            self.progress.setValue(0)
        self._update_gpu_tip(info)
        self.completeChanged.emit()

    def _update_gpu_tip(self, info) -> None:
        """第 5 步的显存提醒：建议先回主界面点「释放模型」。"""
        try:
            need = 8 if info.total_gb >= 20 else 6
            hint = (f"当前可用显存 {info.free_gb:.1f} GB（训练建议 ≥ {need} GB）。\n"
                    "如果主界面已经加载了语音模型，建议先切回主界面点「释放模型」腾出显存，"
                    "再回来开始训练；训练完成后点「加载模型」或重启软件即可恢复。")
            warn = info.free_gb < need
        except Exception:
            hint = ("训练前建议先在主界面点「释放模型」腾出显存，训练完成后再点「加载模型」。")
            warn = False
        self.gpu_tip.setText(("⚠️ " if warn else "") + hint)
        self.gpu_tip.setStyleSheet("color:#e0a34a;" if warn else "")

    def _toggle_log(self) -> None:
        visible = self.log.isVisible()
        self.log.setVisible(not visible)
        self.btn_detail.setText("展开日志" if visible else "收起日志")

    def _start(self) -> None:
        if self._thread is not None:
            return
        info = train.gpu_info()
        try:
            plan = self.wiz.config.plan(info.total_gb, self.profile.currentData())
        except Exception as exc:
            QMessageBox.warning(self, "无法训练", str(exc))
            return
        seg = self.wiz.state.get("seg")
        if seg is None:
            QMessageBox.warning(self, "无法训练", "还没有切分好的数据集。")
            return
        self.wiz.state["plan"] = plan
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("启动中…")
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.wiz.training_state.emit(True, "准备数据…")

        runner = train.TrainingRunner(
            self.wiz.config, self.wiz.store, self.wiz.char_id,
            seg.dataset_dir, plan, on_event=None)
        self._runner = runner

        def job(_progress):
            return runner.run()

        thread = FnWorker(job, self)
        # 训练回调发生在工作线程，必须经信号排队回 GUI 线程再碰控件
        runner.on_event = thread.event.emit
        thread.event.connect(self._on_event)
        thread.done.connect(self._on_done)
        thread.failed.connect(self._on_failed)
        self._thread = thread
        thread.start()

    def _on_event(self, ev: train.TrainEvent) -> None:
        # 子进程回调在训练线程里，用队列式更新（Qt 信号跨线程更安全，这里直接
        # 调用 setText 在 Windows 上实测可用，但保险起见只做原子赋值）
        msg = f"[{ev.stage}] {ev.epoch}/{ev.total_epochs} {ev.message}"
        self.status.setText(msg[:160])
        self.log.appendPlainText(msg)
        if ev.total_epochs:
            pct = int(100 * ev.epoch / max(1, ev.total_epochs))
            self.progress.setValue(min(99, pct))
        self.wiz.training_state.emit(True, msg[:80])

    def _on_done(self, result: dict) -> None:
        self._done = True
        self.progress.setValue(100)
        self.status.setText("训练完成，正在生成试听…")
        self.btn_cancel.setEnabled(False)
        self.wiz.training_state.emit(False, "")
        self.completeChanged.emit()
        self._thread = None
        QMessageBox.information(
            self, "训练完成",
            "训练完成，模型已保存。\n\n"
            "点「下一步」进入试听，听三句固定文本；不满意可以返回上一步换档位重训。\n\n"
            "提示：如果训练前在主界面点过「释放模型」腾显存，"
            "训练结束后请重启软件，或点「加载模型」重新加载。")

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"训练失败：{msg}")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.wiz.training_state.emit(False, "")
        self._thread = None
        QMessageBox.critical(self, "训练失败", msg)

    def _cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self.status.setText("正在取消…")

    def isComplete(self) -> bool:
        return self._done


# ---------------------------------------------------------------------------
# 6 试听
# ---------------------------------------------------------------------------
class PreviewPage(QWizardPage):
    def __init__(self, wiz: NewCharacterWizard):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle("第 6 步 · 试听")
        self.setSubTitle("三句固定文本，不接 LLM。不满意可以「上一步」回去重训。")
        root = QVBoxLayout(self)
        self.rows: list[tuple[QLabel, QPushButton]] = []
        for i in range(3):
            row = QHBoxLayout()
            lab = QLabel(f"第 {i+1} 句")
            btn = QPushButton("▶ 播放")
            btn.setEnabled(False)
            row.addWidget(lab, 1)
            row.addWidget(btn)
            root.addLayout(row)
            self.rows.append((lab, btn))
        row = QHBoxLayout()
        self.btn_ab = QPushButton("A/B 对比原声")
        self.btn_ab.setEnabled(False)
        row.addWidget(self.btn_ab)
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        root.addLayout(row)
        root.addStretch(1)
        self.btn_ab.clicked.connect(self._ab)
        self._player = audio_io.Player(device=None)
        self._files: list[Path] = []
        self._engine = None
        self._thread = None

    def initializePage(self) -> None:  # noqa: N802
        if self._files:
            return
        ch = self.wiz.store.get(self.wiz.char_id)
        if ch.status != "ready":
            self.status.setText("角色还没训练完，无法试听。")
            return
        self.status.setText("加载 TTS 引擎并合成，第一次约需 20–40 秒…")
        out_dir = self.wiz.dataset_dir() / "preview"

        def job(_progress):
            from app.core import tts as tts_mod
            engine = tts_mod.make_engine(self.wiz.config)
            char = ch.to_dict()
            char["_dir"] = str(self.wiz.char_dir())
            files = tts_mod.synthesize_preview(engine, self.wiz.config, char, out_dir)
            self._engine = engine
            return files

        self._thread = FnWorker(job, self)
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(lambda m: self.status.setText(f"合成失败：{m}"))
        self._thread.start()

    def _on_done(self, files: list[Path]) -> None:
        self._files = list(files)
        texts = self.wiz.config.data.get("tts", {}).get("preview_texts", [])
        for i, (lab, btn) in enumerate(self.rows):
            if i < len(files):
                text = texts[i] if i < len(texts) else ""
                lab.setText(f"{i+1}. {text}")
                btn.setEnabled(True)
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(
                    lambda _=False, p=files[i]: self._player.play_file(str(p), blocking=False))
        self.status.setText(f"已生成 {len(files)} 句，文件名在 {Path(files[0]).parent}")
        self.btn_ab.setEnabled(True)

    def _ab(self) -> None:
        ch = self.wiz.store.get(self.wiz.char_id)
        ref = self.wiz.char_dir() / (ch.voice.get("reference") or "reference.wav")
        if ref.exists():
            self._player.play_file(str(ref), blocking=False)
            self.status.setText("正在播放原声参考（再点一次试听句可切回）")


__all__ = ["NewCharacterWizard"]
