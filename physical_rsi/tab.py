"""PhysicalRSI 标签页。"""

from __future__ import annotations

import os
import threading
from typing import Optional

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from physical_rsi.loop import LoopConfig, run_loop
from physical_rsi.tasks import TASKS

_PAGE = "https://mmlab.hk/research/PhysicalRSI"
# 与 show_camera_topics 全局主题 token 对齐
_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_GREEN = "#50fa7b"
_RED = "#ff8888"
_BG = "#1e1f22"
_BG_LOG = "#1a1b1e"
_BORDER = "#3d4048"
_MONO = "Monospace"


class _LoopThread(QThread):
    line = pyqtSignal(str)
    harness_ready = pyqtSignal(str)
    finished_ok = pyqtSignal(str)

    def __init__(self, cfg: LoopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            final = run_loop(
                self.cfg,
                log=self.line.emit,
                on_harness=self.harness_ready.emit,
                stop=self.stop_event,
            )
        except Exception as exc:
            self.line.emit(f"循环失败: {type(exc).__name__}: {exc}")
            final = self.cfg.harness
        self.finished_ok.emit(final)

    def request_stop(self) -> None:
        self.stop_event.set()


class PhysicalRsiTab(QWidget):
    """System 2 改写 harness，用桌面任务的分数做选择。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("physicalRsiTab")
        self.setStyleSheet(
            f"#physicalRsiTab {{ background-color: {_BG}; }}"
            "#physicalRsiTab QLabel { background: transparent; }"
        )
        self._loading = False
        self._worker: Optional[_LoopThread] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "PhysicalRSI 1.0：System 2 读懂任务并改写 System 1 harness，"
            "用未见布局上的分数选择，再把胜出的 harness 继承到下一轮。"
            "默认用内置改写跑通页面里的五类技能结构；勾选模型后改为调用 OpenAI 兼容接口。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("任务"))
        self.task_combo = QComboBox()
        for spec in TASKS.values():
            self.task_combo.addItem(f"{spec.title} · {spec.key}", spec.key)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        row.addWidget(self.task_combo)

        row.addWidget(QLabel("轮数"))
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 12)
        self.budget_spin.setValue(5)
        row.addWidget(self.budget_spin)

        row.addWidget(QLabel("seeds"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 16)
        self.seed_spin.setValue(4)
        self.seed_spin.setToolTip("每一代在多少个未见布局上比较分数")
        row.addWidget(self.seed_spin)

        self.model_check = QCheckBox("用模型改写")
        self.model_check.setToolTip("System 2 调用 /chat/completions；失败时仍保留内置改写")
        row.addWidget(self.model_check)
        row.addWidget(QLabel("API"))
        self.api_edit = QLineEdit(
            os.environ.get("PHYSICAL_RSI_API_BASE", "http://127.0.0.1:8080/v1")
        )
        self.api_edit.setFont(QFont(_MONO, 9))
        self.api_edit.setMinimumWidth(180)
        row.addWidget(self.api_edit, 1)
        row.addWidget(QLabel("模型"))
        self.model_edit = QLineEdit(os.environ.get("PHYSICAL_RSI_MODEL", "qwen"))
        self.model_edit.setFont(QFont(_MONO, 9))
        self.model_edit.setFixedWidth(120)
        row.addWidget(self.model_edit)
        row.addWidget(QLabel("候选"))
        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(1, 4)
        self.variant_spin.setValue(2)
        self.variant_spin.setToolTip("模型每次改写采样几条")
        row.addWidget(self.variant_spin)
        root.addLayout(row)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.status_label = QLabel("PhysicalRSI: 空闲")
        self.status_label.setFont(QFont(_MONO, 9))
        buttons.addWidget(self.status_label, 1)
        self.run_btn = QPushButton("跑一轮")
        self.run_btn.clicked.connect(lambda: self._start(1))
        buttons.addWidget(self.run_btn)
        self.budget_btn = QPushButton("跑满轮数")
        self.budget_btn.clicked.connect(lambda: self._start(self.budget_spin.value()))
        buttons.addWidget(self.budget_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        buttons.addWidget(self.stop_btn)
        self.reset_btn = QPushButton("重置 H0")
        self.reset_btn.clicked.connect(self._reset_harness)
        buttons.addWidget(self.reset_btn)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        buttons.addWidget(self.clear_btn)
        self.page_btn = QPushButton("论文页")
        self.page_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_PAGE)))
        buttons.addWidget(self.page_btn)
        root.addLayout(buttons)

        split = QSplitter(Qt.Horizontal)
        self.harness_edit = QPlainTextEdit()
        self.harness_edit.setFont(QFont(_MONO, 10))
        self.harness_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.harness_edit.setPlaceholderText("System 1 harness：def act(o, g, m, skills, tools)")
        self.harness_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        split.addWidget(self.harness_edit)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setPlaceholderText("理解、执行、改写、选择的记录…")
        self.log_edit.setStyleSheet(
            f"QTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        split.addWidget(self.log_edit)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self._set_harness(self._spec().initial)
        self._set_idle(True)

    def _spec(self):
        key = str(self.task_combo.currentData() or "make_kong")
        return TASKS[key]

    def _set_harness(self, text: str) -> None:
        self._loading = True
        self.harness_edit.setPlainText(text)
        self._loading = False

    def _on_task_changed(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_harness(self._spec().initial)
        self.status_label.setText(f"PhysicalRSI: {self._spec().title}")

    def _reset_harness(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_harness(self._spec().initial)
        self.status_label.setText("PhysicalRSI: 已重置为 H0")

    def _cfg(self, budget: int) -> LoopConfig:
        return LoopConfig(
            task=self._spec().key,
            harness=self.harness_edit.toPlainText(),
            budget=budget,
            eval_seeds=self.seed_spin.value(),
            use_model=self.model_check.isChecked(),
            api_base=self.api_edit.text().strip(),
            model=self.model_edit.text().strip() or "qwen",
            n_variants=self.variant_spin.value(),
        )

    def _start(self, budget: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self.log_edit.append(f"开始 {self._spec().title}，budget={budget}")
        worker = _LoopThread(self._cfg(budget))
        worker.line.connect(self.log_edit.append)
        worker.harness_ready.connect(self._set_harness)
        worker.finished_ok.connect(self._on_finished)
        self._worker = worker
        self._set_idle(False)
        self.status_label.setText("PhysicalRSI: 运行中")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.status_label.setText("PhysicalRSI: 正在停止")

    def _on_finished(self, _final: str) -> None:
        self._set_idle(True)
        self.status_label.setText("PhysicalRSI: 完成")
        self.status_label.setStyleSheet(f"color: {_TEXT};")

    def _set_idle(self, idle: bool) -> None:
        self.run_btn.setEnabled(idle)
        self.budget_btn.setEnabled(idle)
        self.reset_btn.setEnabled(idle)
        self.task_combo.setEnabled(idle)
        self.stop_btn.setEnabled(not idle)
