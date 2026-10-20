"""ASPIRE 标签页：图形化配置、doctor / validate / run 与产物可视化。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal, QObject
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aspire.catalog import (
    ARXIV_URL,
    DEFAULT_REPO,
    EXAMPLE_PRESETS,
    PAGE_URL,
    ensure_host_env,
    list_validation_plans,
    probe_repo,
    resolve_python,
)
from aspire.monitor import list_recent_runs, snapshot_run
from aspire.runner import AspireRunner, RunRequest

_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_GREEN = "#50fa7b"
_RED = "#ff8888"
_YELLOW = "#e6c07b"
_BG = "#1e1f22"
_BG_LOG = "#1a1b1e"
_BORDER = "#3d4048"
_MONO = "Monospace"
_EDIT_STYLE = (
    f"QPlainTextEdit, QTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
    f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
)


class _Bridge(QObject):
    line = pyqtSignal(str)
    done = pyqtSignal(int)


class AspireTab(QWidget):
    """对接本地 ASPIRE：host 环境、bootstrap、doctor / validate / run。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("aspireTab")
        self.setStyleSheet(
            f"#aspireTab {{ background-color: {_BG}; }}"
            "#aspireTab QLabel { background: transparent; }"
        )
        self._runner = AspireRunner()
        self._bridge = _Bridge()
        self._bridge.line.connect(self._append_log)
        self._bridge.done.connect(self._on_run_done)
        self._images: List[Path] = []
        self._poll = QTimer(self)
        self._poll.setInterval(2500)
        self._poll.timeout.connect(self._refresh_snapshot)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "ASPIRE（cap-harness）：技能发现 / Code-as-Policy。"
            "host 环境在 miniconda3/envs/aspire；完整 LIBERO / Robosuite 需点 Bootstrap "
            "（uv 会把 .venv-* 落到 miniconda3/envs）。跑 episode 前请确保 SAM3 等 provider 服务已起。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        # ---- paths ----
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        path_row.addWidget(QLabel("仓库"))
        self.repo_edit = QLineEdit(os.environ.get("ASPIRE_REPO", DEFAULT_REPO))
        self.repo_edit.setFont(QFont(_MONO, 9))
        path_row.addWidget(self.repo_edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_repo)
        path_row.addWidget(browse)
        path_row.addWidget(QLabel("Python"))
        self.python_edit = QLineEdit(resolve_python(Path(self.repo_edit.text())))
        self.python_edit.setFont(QFont(_MONO, 9))
        path_row.addWidget(self.python_edit, 1)
        self.probe_btn = QPushButton("探测环境")
        self.probe_btn.clicked.connect(self._probe)
        path_row.addWidget(self.probe_btn)
        self.install_btn = QPushButton("安装 host 依赖")
        self.install_btn.setToolTip(
            "创建 miniconda3/envs/aspire (Python 3.11) 并 editable 安装 cap-harness"
        )
        self.install_btn.clicked.connect(self._install_host)
        path_row.addWidget(self.install_btn)
        root.addLayout(path_row)

        self.probe_label = QLabel("点击「探测环境」检查 miniconda3/envs/aspire 与 bootstrap 状态。")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.probe_label)

        boot_row = QHBoxLayout()
        boot_row.setSpacing(6)
        self.boot_libero_btn = QPushButton("Bootstrap LIBERO")
        self.boot_libero_btn.setToolTip("scripts/bootstrap_libero.sh → .venv-libero（bytes 在 miniconda3/envs）")
        self.boot_libero_btn.clicked.connect(lambda: self._start_cmd("bootstrap_libero"))
        boot_row.addWidget(self.boot_libero_btn)
        self.boot_robo_btn = QPushButton("Bootstrap Robosuite")
        self.boot_robo_btn.clicked.connect(lambda: self._start_cmd("bootstrap_robosuite"))
        boot_row.addWidget(self.boot_robo_btn)
        boot_row.addStretch(1)
        root.addLayout(boot_row)

        # ---- config ----
        cfg = QGroupBox("实验配置")
        form = QFormLayout(cfg)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("doctor（环境体检）", "doctor")
        self.mode_combo.addItem("validate（结构校验）", "validate")
        self.mode_combo.addItem("run（跑示例 episode）", "run")
        self.mode_combo.setCurrentIndex(2)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(QLabel("GPU"))
        self.gpu_edit = QLineEdit("0")
        self.gpu_edit.setFixedWidth(40)
        mode_row.addWidget(self.gpu_edit)
        form.addRow("模式", mode_row)

        self.example_combo = QComboBox()
        self.example_combo.setMinimumWidth(280)
        for preset in EXAMPLE_PRESETS:
            self.example_combo.addItem(preset[0], preset)
        self.example_combo.currentIndexChanged.connect(self._on_example_changed)
        form.addRow("示例", self.example_combo)

        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(280)
        form.addRow("校验计划", self.plan_combo)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("benchmark"))
        self.bench_edit = QLineEdit("libero-pro")
        self.bench_edit.setFont(QFont(_MONO, 9))
        self.bench_edit.setFixedWidth(100)
        param_row.addWidget(self.bench_edit)
        param_row.addWidget(QLabel("suite"))
        self.suite_edit = QLineEdit("libero_object_swap")
        self.suite_edit.setFont(QFont(_MONO, 9))
        param_row.addWidget(self.suite_edit, 1)
        param_row.addWidget(QLabel("task"))
        self.task_spin = QSpinBox()
        self.task_spin.setRange(0, 999)
        self.task_spin.setValue(0)
        param_row.addWidget(self.task_spin)
        param_row.addWidget(QLabel("seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 9999)
        self.seed_spin.setValue(1)
        param_row.addWidget(self.seed_spin)
        form.addRow("任务", param_row)

        prog_row = QHBoxLayout()
        self.program_edit = QLineEdit("examples/libero/object_swap_seed1.py")
        self.program_edit.setFont(QFont(_MONO, 9))
        prog_row.addWidget(self.program_edit, 1)
        browse_prog = QPushButton("…")
        browse_prog.setFixedWidth(28)
        browse_prog.clicked.connect(self._browse_program)
        prog_row.addWidget(browse_prog)
        form.addRow("program", prog_row)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit("outputs/ui")
        self.output_edit.setFont(QFont(_MONO, 9))
        out_row.addWidget(self.output_edit, 1)
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(200)
        self.history_combo.currentIndexChanged.connect(self._on_history_chosen)
        out_row.addWidget(self.history_combo)
        refresh_hist = QPushButton("刷新历史")
        refresh_hist.clicked.connect(self._reload_history)
        out_row.addWidget(refresh_hist)
        form.addRow("output", out_row)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("max-steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 20000)
        self.steps_spin.setValue(2000)
        opt_row.addWidget(self.steps_spin)
        opt_row.addWidget(QLabel("init"))
        self.init_combo = QComboBox()
        self.init_combo.addItem("saved", "saved")
        self.init_combo.addItem("seeded", "seeded")
        opt_row.addWidget(self.init_combo)
        self.no_videos_check = QCheckBox("无视频")
        opt_row.addWidget(self.no_videos_check)
        self.doctor_reset_check = QCheckBox("doctor --reset")
        opt_row.addWidget(self.doctor_reset_check)
        opt_row.addStretch(1)
        form.addRow("选项", opt_row)

        self.providers_edit = QLineEdit("")
        self.providers_edit.setPlaceholderText("doctor providers 可选，如 sam3,pyroki,curobo")
        self.providers_edit.setFont(QFont(_MONO, 9))
        form.addRow("providers", self.providers_edit)

        root.addWidget(cfg)

        # ---- actions ----
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.status_label = QLabel("ASPIRE: 空闲")
        self.status_label.setFont(QFont(_MONO, 9))
        buttons.addWidget(self.status_label, 1)
        self.start_btn = QPushButton("启动")
        self.start_btn.clicked.connect(self._start)
        buttons.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        buttons.addWidget(self.stop_btn)
        self.refresh_btn = QPushButton("刷新产物")
        self.refresh_btn.clicked.connect(self._refresh_snapshot)
        buttons.addWidget(self.refresh_btn)
        self.open_run_btn = QPushButton("打开 run")
        self.open_run_btn.clicked.connect(self._open_run_dir)
        buttons.addWidget(self.open_run_btn)
        self.page_btn = QPushButton("项目页")
        self.page_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(PAGE_URL)))
        buttons.addWidget(self.page_btn)
        self.arxiv_btn = QPushButton("论文")
        self.arxiv_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ARXIV_URL)))
        buttons.addWidget(self.arxiv_btn)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        buttons.addWidget(self.clear_btn)
        root.addLayout(buttons)

        metrics = QHBoxLayout()
        self.metric_status = QLabel("status: -")
        self.metric_status.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.metric_status)
        self.metric_success = QLabel("success: -")
        self.metric_success.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.metric_success)
        self.metric_reward = QLabel("reward: -")
        self.metric_reward.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.metric_reward)
        self.metric_steps = QLabel("steps: -")
        self.metric_steps.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.metric_steps)
        metrics.addStretch(1)
        root.addLayout(metrics)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_l.addWidget(QLabel("program / outcome"))
        self.program_view = QPlainTextEdit()
        self.program_view.setReadOnly(True)
        self.program_view.setFont(QFont(_MONO, 9))
        self.program_view.setStyleSheet(_EDIT_STYLE)
        self.program_view.setPlaceholderText("运行后显示 source/program.py 与 outcome 摘要…")
        left_l.addWidget(self.program_view, 1)
        split.addWidget(left)

        center = QWidget()
        cen_l = QVBoxLayout(center)
        cen_l.setContentsMargins(0, 0, 0, 0)
        cen_l.setSpacing(4)
        cen_l.addWidget(QLabel("keyframes / overlays"))
        img_row = QHBoxLayout()
        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(220)
        self.image_list.currentTextChanged.connect(self._preview_image)
        img_row.addWidget(self.image_list, 1)
        self.image_preview = QLabel("无关键帧")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(220, 160)
        self.image_preview.setStyleSheet(
            f"QLabel {{ background-color: {_BG_LOG}; border: 1px solid {_BORDER}; "
            f"border-radius: 4px; color: {_MUTED}; }}"
        )
        img_row.addWidget(self.image_preview, 2)
        cen_l.addLayout(img_row, 1)
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(80)
        self.video_list.itemDoubleClicked.connect(self._open_selected_video)
        cen_l.addWidget(QLabel("videos（双击打开）"))
        cen_l.addWidget(self.video_list)
        split.addWidget(center)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addWidget(QLabel("摘要 / 日志"))
        self.summary_edit = QPlainTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setFont(QFont(_MONO, 9))
        self.summary_edit.setMaximumHeight(110)
        self.summary_edit.setStyleSheet(_EDIT_STYLE)
        right_l.addWidget(self.summary_edit)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setPlaceholderText("ASPIRE 运行日志…")
        self.log_edit.setStyleSheet(_EDIT_STYLE)
        right_l.addWidget(self.log_edit, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        self._current_run_dir: Optional[Path] = None
        self._on_example_changed()
        self._reload_plans()
        self._reload_history()
        self._on_mode_changed()
        self._set_idle(True)
        QTimer.singleShot(200, self._probe)

    def _repo(self) -> Path:
        return Path(self.repo_edit.text().strip() or DEFAULT_REPO).expanduser()

    def _browse_repo(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 ASPIRE 仓库", str(self._repo()))
        if path:
            self.repo_edit.setText(path)
            self.python_edit.setText(resolve_python(Path(path)))
            self._reload_plans()
            self._reload_history()
            self._probe()

    def _browse_program(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CaP program", str(self._repo() / "examples"), "Python (*.py)"
        )
        if path:
            repo = self._repo()
            try:
                rel = str(Path(path).resolve().relative_to(repo.resolve()))
            except ValueError:
                rel = path
            self.program_edit.setText(rel)

    def _append_log(self, line: str) -> None:
        self.log_edit.append(line)

    def _set_idle(self, idle: bool) -> None:
        self.start_btn.setEnabled(idle)
        self.install_btn.setEnabled(idle)
        self.boot_libero_btn.setEnabled(idle)
        self.boot_robo_btn.setEnabled(idle)
        self.probe_btn.setEnabled(idle)
        self.stop_btn.setEnabled(not idle)

    def _probe(self) -> None:
        result = probe_repo(self._repo())
        self.python_edit.setText(result.python)
        color = _GREEN if result.ok else _YELLOW
        self.probe_label.setText("\n".join(result.lines))
        self.probe_label.setStyleSheet(f"color: {color};")

    def _install_host(self) -> None:
        self._set_idle(False)
        self.status_label.setText("ASPIRE: 安装 host 依赖中…")
        self.log_edit.append("—— 安装 ASPIRE host → miniconda3/envs/aspire ——")

        def _work() -> None:
            ok, msg = ensure_host_env(self._repo(), log=self._bridge.line.emit)
            self._bridge.line.emit(msg)
            self._bridge.done.emit(0 if ok else 1)

        import threading

        threading.Thread(target=_work, name="aspire-install", daemon=True).start()

    def _reload_plans(self) -> None:
        self.plan_combo.blockSignals(True)
        self.plan_combo.clear()
        for title, path in list_validation_plans(self._repo()):
            try:
                rel = str(path.relative_to(self._repo()))
            except ValueError:
                rel = str(path)
            self.plan_combo.addItem(title, rel)
        self.plan_combo.blockSignals(False)

    def _reload_history(self) -> None:
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("（最近 run）", "")
        out = self._repo() / (self.output_edit.text().strip() or "outputs/ui")
        for run in list_recent_runs(out, limit=30):
            try:
                rel = str(run.relative_to(self._repo()))
            except ValueError:
                rel = str(run)
            self.history_combo.addItem(rel, str(run))
        self.history_combo.blockSignals(False)

    def _on_history_chosen(self) -> None:
        path = self.history_combo.currentData()
        if path:
            self._current_run_dir = Path(path)
            self._refresh_snapshot()

    def _on_example_changed(self) -> None:
        preset = self.example_combo.currentData()
        if not preset:
            return
        _title, bench, suite, task_id, program, steps, cw, ch = preset
        self.bench_edit.setText(bench)
        self.suite_edit.setText(suite)
        self.task_spin.setValue(int(task_id))
        self.program_edit.setText(program)
        self.steps_spin.setValue(int(steps))
        self._camera_w = int(cw)
        self._camera_h = int(ch)

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        is_run = mode == "run"
        is_val = mode == "validate"
        is_doc = mode == "doctor"
        self.example_combo.setEnabled(is_run)
        self.program_edit.setEnabled(is_run)
        self.bench_edit.setEnabled(is_run)
        self.suite_edit.setEnabled(is_run)
        self.task_spin.setEnabled(is_run)
        self.seed_spin.setEnabled(is_run)
        self.steps_spin.setEnabled(is_run)
        self.init_combo.setEnabled(is_run)
        self.no_videos_check.setEnabled(is_run)
        self.plan_combo.setEnabled(is_val)
        self.providers_edit.setEnabled(is_doc)
        self.doctor_reset_check.setEnabled(is_doc)

    def _build_request(self, command: Optional[str] = None) -> RunRequest:
        mode = command or self.mode_combo.currentData()
        cw = getattr(self, "_camera_w", 800)
        ch = getattr(self, "_camera_h", 512)
        plan = self.plan_combo.currentData() or ""
        return RunRequest(
            repo=self._repo(),
            command=str(mode),
            python=self.python_edit.text().strip(),
            gpu=self.gpu_edit.text().strip() or "0",
            doctor_providers=self.providers_edit.text().strip(),
            doctor_reset=self.doctor_reset_check.isChecked(),
            plan=str(plan),
            validate_output_dir="validation-artifacts/ui",
            benchmark=self.bench_edit.text().strip() or "libero-pro",
            suite=self.suite_edit.text().strip(),
            task_id=int(self.task_spin.value()),
            seed=int(self.seed_spin.value()),
            program=self.program_edit.text().strip(),
            output_root=self.output_edit.text().strip() or "outputs/ui",
            max_steps=int(self.steps_spin.value()),
            camera_width=cw,
            camera_height=ch,
            init_mode=str(self.init_combo.currentData() or "saved"),
            no_videos=self.no_videos_check.isChecked(),
        )

    def _start(self) -> None:
        self._start_cmd(str(self.mode_combo.currentData()))

    def _start_cmd(self, command: str) -> None:
        if self._runner.running:
            QMessageBox.warning(self, "ASPIRE", "已有任务在运行")
            return
        repo = self._repo()
        if not repo.is_dir():
            QMessageBox.warning(self, "ASPIRE", f"仓库不存在:\n{repo}")
            return
        try:
            req = self._build_request(command)
            self._set_idle(False)
            self.status_label.setText(f"ASPIRE: {command} 运行中…")
            self.status_label.setStyleSheet(f"color: {_YELLOW};")
            self.log_edit.append(f"—— 启动 {command} ——")
            if command == "run":
                out = repo / req.output_root
                self._current_run_dir = out
                self._poll.start()
            self._runner.start(
                req,
                log=self._bridge.line.emit,
                on_done=self._bridge.done.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_idle(True)
            self.status_label.setText("ASPIRE: 启动失败")
            self.status_label.setStyleSheet(f"color: {_RED};")
            QMessageBox.warning(self, "ASPIRE", str(exc))

    def _stop(self) -> None:
        self.status_label.setText("ASPIRE: 正在停止")
        self._runner.stop(log=self._bridge.line.emit)

    def _on_run_done(self, code: int) -> None:
        self._poll.stop()
        self._set_idle(True)
        ok = code == 0
        self.status_label.setText(f"ASPIRE: 完成 exit={code}")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self.python_edit.setText(resolve_python(self._repo()))
        self._reload_history()
        # 尝试定位最新 run
        out = self._repo() / (self.output_edit.text().strip() or "outputs/ui")
        runs = list_recent_runs(out, limit=1)
        if runs:
            self._current_run_dir = runs[0]
        self._refresh_snapshot()
        self._probe()

    def _latest_run_dir(self) -> Optional[Path]:
        if self._current_run_dir and (self._current_run_dir / "run.json").is_file():
            return self._current_run_dir
        if self._current_run_dir and self._current_run_dir.is_dir():
            # output root：找最新
            runs = list_recent_runs(self._current_run_dir, limit=1)
            if runs:
                return runs[0]
        out = self._repo() / (self.output_edit.text().strip() or "outputs/ui")
        runs = list_recent_runs(out, limit=1)
        return runs[0] if runs else None

    def _refresh_snapshot(self) -> None:
        run = self._latest_run_dir()
        if run is None:
            return
        self._current_run_dir = run
        snap = snapshot_run(run)
        self.metric_status.setText(f"status: {snap.status or '-'}")
        if snap.success is None:
            self.metric_success.setText("success: -")
            self.metric_success.setStyleSheet(f"color: {_MUTED};")
        else:
            self.metric_success.setText(f"success: {snap.success}")
            self.metric_success.setStyleSheet(
                f"color: {_GREEN if snap.success else _RED};"
            )
        self.metric_reward.setText(
            f"reward: {snap.reward:.3f}" if snap.reward is not None else "reward: -"
        )
        self.metric_steps.setText(
            f"steps: {snap.steps}" if snap.steps is not None else "steps: -"
        )
        self.summary_edit.setPlainText("\n".join(snap.summary_lines))
        body = snap.program_text or ""
        if snap.outcome_json:
            import json

            body = (
                body
                + ("\n\n# ---- outcome.json ----\n" if body else "")
                + json.dumps(snap.outcome_json, indent=2, ensure_ascii=False)[:8000]
            )
        self.program_view.setPlainText(body)
        self._images = snap.images
        self.image_list.clear()
        for img in snap.images:
            try:
                rel = str(img.relative_to(run))
            except ValueError:
                rel = img.name
            self.image_list.addItem(rel)
        if snap.images:
            self.image_list.setCurrentRow(0)
        self.video_list.clear()
        for vid in snap.videos:
            try:
                rel = str(vid.relative_to(run))
            except ValueError:
                rel = vid.name
            item_text = rel
            self.video_list.addItem(item_text)
            self.video_list.item(self.video_list.count() - 1).setData(Qt.UserRole, str(vid))

    def _preview_image(self, rel: str) -> None:
        if not rel or not self._current_run_dir:
            return
        path = self._current_run_dir / rel
        if not path.is_file():
            for img in self._images:
                if img.name == rel or str(img).endswith(rel):
                    path = img
                    break
        if not path.is_file():
            self.image_preview.setText("无法读取")
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.image_preview.setText(path.name)
            return
        self.image_preview.setPixmap(
            pix.scaled(
                self.image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _open_selected_video(self) -> None:
        item = self.video_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole) or ""
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_run_dir(self) -> None:
        run = self._latest_run_dir()
        if run is None:
            QMessageBox.information(self, "ASPIRE", "还没有 run 产物")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run)))
