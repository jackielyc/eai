"""编程冲 SOTA 标签页。"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from prog_sota.catalog import (
    BACKENDS,
    ActionPrim,
    TASKS,
    get_task,
    list_image_candidates,
    tasks_for_backend,
)
from prog_sota.compare import compare_to_board
from prog_sota.induce import induce_from_text
from prog_sota.loop import LoopConfig, run_loop
from prog_sota.program import actions_to_source, summarize_actions

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
    program_ready = pyqtSignal(str)
    actions_ready = pyqtSignal(object)
    finished_ok = pyqtSignal(str, float)

    def __init__(self, cfg: LoopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            final, stats = run_loop(
                self.cfg,
                log=self.line.emit,
                on_program=self.program_ready.emit,
                on_actions=self.actions_ready.emit,
                stop=self.stop_event,
            )
        except Exception as exc:  # noqa: BLE001
            self.line.emit(f"循环失败: {type(exc).__name__}: {exc}")
            final = self.cfg.program
            stats_sr = 0.0
        else:
            stats_sr = float(stats.sr)
        self.finished_ok.emit(final, stats_sr)

    def request_stop(self) -> None:
        self.stop_event.set()


class ProgSotaTab(QWidget):
    """文本/图像 → 手臂动作归纳 → 编程实现 → 抬高 SR → 对照业内榜单。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("progSotaTab")
        self.setStyleSheet(
            f"#progSotaTab {{ background-color: {_BG}; }}"
            "#progSotaTab QLabel { background: transparent; }"
        )
        self._loading = False
        self._worker: Optional[_LoopThread] = None
        self._actions: Tuple[ActionPrim, ...] = ()
        self._images: List[str] = []
        self._last_sr = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "编程冲 SOTA：选取仿真后端与任务，用文本 + 图像归纳手/臂动作，"
            "编译成可编程策略并迭代抬高成功率，再与 eai/sim_eval_leaderboards.json "
            "业内榜单对照。本地 harness 用于快速闭环；官方协议请到「仿真评测」页复现。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("后端"))
        self.backend_combo = QComboBox()
        for key, title in BACKENDS:
            self.backend_combo.addItem(title, key)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo)

        row.addWidget(QLabel("任务"))
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(220)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        row.addWidget(self.task_combo, 1)

        row.addWidget(QLabel("轮数"))
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 12)
        self.budget_spin.setValue(5)
        row.addWidget(self.budget_spin)

        row.addWidget(QLabel("seeds"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 16)
        self.seed_spin.setValue(4)
        row.addWidget(self.seed_spin)

        self.model_check = QCheckBox("用模型归纳")
        self.model_check.setToolTip("调用 OpenAI 兼容 /chat/completions；失败回退规则归纳")
        row.addWidget(self.model_check)
        row.addWidget(QLabel("API"))
        self.api_edit = QLineEdit(
            os.environ.get("PROG_SOTA_API_BASE", "http://127.0.0.1:8080/v1")
        )
        self.api_edit.setFont(QFont(_MONO, 9))
        self.api_edit.setMinimumWidth(160)
        row.addWidget(self.api_edit, 1)
        row.addWidget(QLabel("模型"))
        self.model_edit = QLineEdit(os.environ.get("PROG_SOTA_MODEL", "qwen"))
        self.model_edit.setFont(QFont(_MONO, 9))
        self.model_edit.setFixedWidth(100)
        row.addWidget(self.model_edit)
        root.addLayout(row)

        mid = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_l.addWidget(QLabel("任务文本"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFont(QFont(_MONO, 10))
        self.text_edit.setPlaceholderText("任务语言描述 / 额外约束…")
        self.text_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        left_l.addWidget(self.text_edit, 2)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("图像输入"))
        self.add_img_btn = QPushButton("添加…")
        self.add_img_btn.clicked.connect(self._add_images)
        img_row.addWidget(self.add_img_btn)
        self.scan_img_btn = QPushButton("扫描 images/")
        self.scan_img_btn.clicked.connect(self._scan_images)
        img_row.addWidget(self.scan_img_btn)
        self.clear_img_btn = QPushButton("清空")
        self.clear_img_btn.clicked.connect(self._clear_images)
        img_row.addWidget(self.clear_img_btn)
        img_row.addStretch(1)
        left_l.addLayout(img_row)

        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(90)
        self.image_list.currentTextChanged.connect(self._preview_image)
        left_l.addWidget(self.image_list)

        self.image_preview = QLabel("无预览")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(120)
        self.image_preview.setStyleSheet(
            f"QLabel {{ background-color: {_BG_LOG}; border: 1px solid {_BORDER}; "
            f"border-radius: 4px; color: {_MUTED}; }}"
        )
        left_l.addWidget(self.image_preview, 1)
        mid.addWidget(left)

        center = QWidget()
        cen_l = QVBoxLayout(center)
        cen_l.setContentsMargins(0, 0, 0, 0)
        cen_l.setSpacing(4)
        cen_l.addWidget(QLabel("手 / 臂动作归纳"))
        self.actions_edit = QPlainTextEdit()
        self.actions_edit.setReadOnly(True)
        self.actions_edit.setFont(QFont(_MONO, 10))
        self.actions_edit.setPlaceholderText("归纳结果…")
        self.actions_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        cen_l.addWidget(self.actions_edit, 1)
        cen_l.addWidget(QLabel("编程策略 act(o,g,m,skills,tools)"))
        self.program_edit = QPlainTextEdit()
        self.program_edit.setFont(QFont(_MONO, 10))
        self.program_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.program_edit.setPlaceholderText("def act(...)")
        self.program_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        cen_l.addWidget(self.program_edit, 2)
        mid.addWidget(center)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addWidget(QLabel("运行日志 / 榜单对照"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setPlaceholderText("归纳、编程、评测、榜单对照…")
        self.log_edit.setStyleSheet(
            f"QTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        right_l.addWidget(self.log_edit, 1)
        mid.addWidget(right)

        mid.setStretchFactor(0, 2)
        mid.setStretchFactor(1, 3)
        mid.setStretchFactor(2, 3)
        root.addWidget(mid, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.status_label = QLabel("编程冲SOTA: 空闲")
        self.status_label.setFont(QFont(_MONO, 9))
        buttons.addWidget(self.status_label, 1)
        self.induce_btn = QPushButton("归纳动作")
        self.induce_btn.clicked.connect(self._induce_only)
        buttons.addWidget(self.induce_btn)
        self.run_btn = QPushButton("跑一轮")
        self.run_btn.clicked.connect(lambda: self._start(1))
        buttons.addWidget(self.run_btn)
        self.budget_btn = QPushButton("优化满轮")
        self.budget_btn.clicked.connect(lambda: self._start(self.budget_spin.value()))
        buttons.addWidget(self.budget_btn)
        self.compare_btn = QPushButton("对照榜单")
        self.compare_btn.clicked.connect(self._compare_only)
        buttons.addWidget(self.compare_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        buttons.addWidget(self.stop_btn)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        buttons.addWidget(self.clear_btn)
        self.board_btn = QPushButton("打开来源")
        self.board_btn.clicked.connect(self._open_board_source)
        buttons.addWidget(self.board_btn)
        root.addLayout(buttons)

        self._reload_tasks()
        self._set_idle(True)

    def _card(self):
        key = str(self.task_combo.currentData() or "")
        return get_task(key) or next(iter(TASKS.values()))

    def _on_backend_changed(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._reload_tasks()

    def _reload_tasks(self) -> None:
        backend = str(self.backend_combo.currentData() or "libero")
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for card in tasks_for_backend(backend):
            self.task_combo.addItem(card.title, card.key)
        self.task_combo.blockSignals(False)
        if self.task_combo.count() == 0:
            return
        self._on_task_changed()

    def _on_task_changed(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        card = self._card()
        self.text_edit.setPlainText(card.text)
        self._images = []
        self.image_list.clear()
        for path in list_image_candidates(card):
            self._images.append(path)
            self.image_list.addItem(path)
        if self.image_list.count():
            self.image_list.setCurrentRow(0)
        else:
            self.image_preview.setText("无预览")
            self.image_preview.setPixmap(QPixmap())
        self._actions = induce_from_text(card, card.text, self._images)
        self.actions_edit.setPlainText(summarize_actions(self._actions))
        self.program_edit.setPlainText(actions_to_source(self._actions, card))
        self.status_label.setText(f"编程冲SOTA: {card.title}")

    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择任务图像",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        for path in paths:
            if path and path not in self._images:
                self._images.append(path)
                self.image_list.addItem(path)
        if paths:
            self.image_list.setCurrentRow(self.image_list.count() - 1)

    def _scan_images(self) -> None:
        card = self._card()
        for path in list_image_candidates(card):
            if path not in self._images:
                self._images.append(path)
                self.image_list.addItem(path)
        if self.image_list.count() and self.image_list.currentRow() < 0:
            self.image_list.setCurrentRow(0)

    def _clear_images(self) -> None:
        self._images = []
        self.image_list.clear()
        self.image_preview.setText("无预览")
        self.image_preview.setPixmap(QPixmap())

    def _preview_image(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.image_preview.setText("无预览")
            self.image_preview.setPixmap(QPixmap())
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.image_preview.setText("无法加载")
            return
        scaled = pix.scaled(
            max(160, self.image_preview.width()),
            max(120, self.image_preview.height()),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_preview.setPixmap(scaled)

    def _induce_only(self) -> None:
        card = self._card()
        self._actions = induce_from_text(
            card, self.text_edit.toPlainText(), list(self._images)
        )
        self.actions_edit.setPlainText(summarize_actions(self._actions))
        self.program_edit.setPlainText(actions_to_source(self._actions, card))
        self.log_edit.append("已归纳动作并编译程序")
        self.status_label.setText("编程冲SOTA: 已归纳")

    def _cfg(self, budget: int) -> LoopConfig:
        return LoopConfig(
            task=self._card().key,
            text=self.text_edit.toPlainText(),
            image_paths=list(self._images),
            actions=self._actions or None,
            program=self.program_edit.toPlainText(),
            budget=budget,
            eval_seeds=self.seed_spin.value(),
            use_model=self.model_check.isChecked(),
            api_base=self.api_edit.text().strip(),
            model=self.model_edit.text().strip() or "qwen",
        )

    def _start(self, budget: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        card = self._card()
        self.log_edit.append(f"开始 {card.title}，budget={budget}")
        worker = _LoopThread(self._cfg(budget))
        worker.line.connect(self.log_edit.append)
        worker.program_ready.connect(self._set_program)
        worker.actions_ready.connect(self._set_actions)
        worker.finished_ok.connect(self._on_finished)
        self._worker = worker
        self._set_idle(False)
        self.status_label.setText("编程冲SOTA: 运行中")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        worker.start()

    def _set_program(self, text: str) -> None:
        self._loading = True
        self.program_edit.setPlainText(text)
        self._loading = False

    def _set_actions(self, actions: object) -> None:
        if isinstance(actions, tuple):
            self._actions = actions  # type: ignore[assignment]
            self.actions_edit.setPlainText(summarize_actions(self._actions))

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.status_label.setText("编程冲SOTA: 正在停止")

    def _on_finished(self, _final: str, sr: float) -> None:
        self._last_sr = float(sr)
        self._set_idle(True)
        self.status_label.setText(f"编程冲SOTA: 完成  SR={sr * 100:.0f}%")
        self.status_label.setStyleSheet(f"color: {_TEXT};")

    def _compare_only(self) -> None:
        card = self._card()
        sr = self._last_sr
        if sr <= 0:
            # 用当前程序粗评
            from prog_sota.loop import evaluate_program

            stats = evaluate_program(card, self.program_edit.toPlainText(), [1, 2, 3, 4])
            sr = stats.sr
            self._last_sr = sr
        cmp = compare_to_board(card.backend, sr)
        self.log_edit.append("—— 业内榜单对照 ——")
        for line in cmp.lines:
            self.log_edit.append(line)
        self.status_label.setText(f"编程冲SOTA: 对照完成  SR={sr * 100:.0f}%")

    def _open_board_source(self) -> None:
        from sim_eval_leaderboard import board_for_backend

        board = board_for_backend(str(self.backend_combo.currentData() or "")) or {}
        url = str(board.get("source_url") or "")
        if url:
            QDesktopServices.openUrl(QUrl(url))
        else:
            self.log_edit.append("当前后端无 source_url")

    def _set_idle(self, idle: bool) -> None:
        self.run_btn.setEnabled(idle)
        self.budget_btn.setEnabled(idle)
        self.induce_btn.setEnabled(idle)
        self.compare_btn.setEnabled(idle)
        self.backend_combo.setEnabled(idle)
        self.task_combo.setEnabled(idle)
        self.stop_btn.setEnabled(not idle)
