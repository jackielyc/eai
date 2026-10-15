"""EmbodiedEvo 标签页：图形化配置、启动与 run 产物可视化。"""

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
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from embodied_evo.catalog import (
    ARXIV_URL,
    DEFAULT_REPO,
    PAGE_URL,
    ensure_host_venv,
    host_deps_ok,
    list_env_tasks,
    probe_repo,
    read_task_toml,
    resolve_python,
)
from embodied_evo.monitor import RunSnapshot, list_runs, snapshot_run
from embodied_evo.runner import EmbodiedEvoRunner, RunRequest, suggest_run_dir

_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_GREEN = "#50fa7b"
_RED = "#ff8888"
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


class EmbodiedEvoTab(QWidget):
    """对接本地 EmbodiedEvo：Playground → Freeze → Test，并可视化 Skills/Lessons/分数。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("embodiedEvoTab")
        self.setStyleSheet(
            f"#embodiedEvoTab {{ background-color: {_BG}; }}"
            "#embodiedEvoTab QLabel { background: transparent; }"
        )
        self._runner = EmbodiedEvoRunner()
        self._bridge = _Bridge()
        self._bridge.line.connect(self._append_log)
        self._bridge.done.connect(self._on_run_done)
        self._scene_bars: List[QProgressBar] = []
        self._images: List[Path] = []
        self._poll = QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(self._refresh_snapshot)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "EmbodiedEvo：完整实验需要 RoboDojo + 模型 API Key。"
            "无 Key 时选模式「smoke（无 API）」即可验证仿真器与 session IPC；"
            "完整 run/explore/test 才需要 Codex 凭证。默认本机跑 agent，不用 Docker 镜像。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        # ---- paths / probe ----
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        path_row.addWidget(QLabel("仓库"))
        self.repo_edit = QLineEdit(os.environ.get("EMBODIEDEVO_REPO", DEFAULT_REPO))
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
            "创建 miniconda3/envs/embodiedevo (Python 3.12) 并安装 gymnasium 等编排依赖"
        )
        self.install_btn.clicked.connect(self._install_host)
        path_row.addWidget(self.install_btn)
        root.addLayout(path_row)

        self.probe_label = QLabel("点击「探测环境」检查 conda / host Codex（默认不使用 Docker）。")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.probe_label)

        # ---- config ----
        cfg = QGroupBox("实验配置")
        form = QFormLayout(cfg)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("smoke（无 API，验证仿真器）", "smoke")
        self.mode_combo.addItem("run（Playground+Test）", "run")
        self.mode_combo.addItem("explore（仅 Playground）", "explore")
        self.mode_combo.addItem("test（仅 Test）", "test")
        self.mode_combo.addItem("baseline test（无 Skills/Lessons）", "baseline")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        self.recipe_check = QCheckBox("使用 public recipe")
        self.recipe_check.setChecked(True)
        mode_row.addWidget(self.recipe_check)
        form.addRow("模式", mode_row)

        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(260)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        form.addRow("env-task", self.task_combo)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("GPU"))
        self.gpu_edit = QLineEdit("0")
        self.gpu_edit.setFixedWidth(48)
        param_row.addWidget(self.gpu_edit)
        param_row.addWidget(QLabel("Test scenes"))
        self.scenes_spin = QSpinBox()
        self.scenes_spin.setRange(1, 100)
        self.scenes_spin.setValue(5)
        self.scenes_spin.setToolTip("完整评测默认 25；调试可先用较小值")
        param_row.addWidget(self.scenes_spin)
        param_row.addWidget(QLabel("PG budget"))
        self.pg_budget_spin = QSpinBox()
        self.pg_budget_spin.setRange(1, 500)
        self.pg_budget_spin.setValue(100)
        param_row.addWidget(self.pg_budget_spin)
        param_row.addWidget(QLabel("Test budget"))
        self.test_budget_spin = QSpinBox()
        self.test_budget_spin.setRange(1, 500)
        self.test_budget_spin.setValue(100)
        param_row.addWidget(self.test_budget_spin)
        param_row.addStretch(1)
        form.addRow("预算", param_row)

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit("gpt-6-astra")
        self.model_edit.setFont(QFont(_MONO, 9))
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(QLabel("reasoning"))
        self.reason_combo = QComboBox()
        for effort in ("xhigh", "high", "medium", "low", "minimal"):
            self.reason_combo.addItem(effort, effort)
        model_row.addWidget(self.reason_combo)
        form.addRow("模型", model_row)

        run_row = QHBoxLayout()
        self.run_dir_edit = QLineEdit()
        self.run_dir_edit.setFont(QFont(_MONO, 9))
        self.run_dir_edit.setPlaceholderText("runs/YYYYMMDDHHMM_<task>")
        run_row.addWidget(self.run_dir_edit, 1)
        self.new_run_btn = QPushButton("新 run-dir")
        self.new_run_btn.clicked.connect(self._suggest_run_dir)
        run_row.addWidget(self.new_run_btn)
        self.browse_run_btn = QPushButton("选择…")
        self.browse_run_btn.clicked.connect(self._browse_run_dir)
        run_row.addWidget(self.browse_run_btn)
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(180)
        self.history_combo.currentIndexChanged.connect(self._on_history_chosen)
        run_row.addWidget(self.history_combo)
        self.refresh_hist_btn = QPushButton("刷新历史")
        self.refresh_hist_btn.clicked.connect(self._reload_history)
        run_row.addWidget(self.refresh_hist_btn)
        form.addRow("run-dir", run_row)
        root.addWidget(cfg)

        # ---- actions ----
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.status_label = QLabel("EmbodiedEvo: 空闲")
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

        # ---- metrics ----
        metrics = QHBoxLayout()
        self.phase_label = QLabel("phase: -")
        self.phase_label.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.phase_label)
        self.sr_label = QLabel("SR: -")
        self.sr_label.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.sr_label)
        self.score_label = QLabel("Score: -")
        self.score_label.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.score_label)
        self.skill_count_label = QLabel("Skills: 0")
        self.skill_count_label.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.skill_count_label)
        self.lesson_count_label = QLabel("Lessons: 0")
        self.lesson_count_label.setFont(QFont(_MONO, 9))
        metrics.addWidget(self.lesson_count_label)
        metrics.addStretch(1)
        root.addLayout(metrics)

        self.sr_bar = QProgressBar()
        self.sr_bar.setRange(0, 1000)
        self.sr_bar.setValue(0)
        self.sr_bar.setFormat("Success rate —")
        root.addWidget(self.sr_bar)

        # ---- main split ----
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_l.addWidget(QLabel("Recipe / Skills / Lessons"))
        self.artifact_tabs = QTabWidget()
        self.recipe_edit = QPlainTextEdit()
        self.recipe_edit.setReadOnly(True)
        self.recipe_edit.setFont(QFont(_MONO, 9))
        self.recipe_edit.setStyleSheet(_EDIT_STYLE)
        self.artifact_tabs.addTab(self.recipe_edit, "Recipe")

        skills_page = QWidget()
        sk_l = QVBoxLayout(skills_page)
        sk_l.setContentsMargins(0, 0, 0, 0)
        self.skills_list = QListWidget()
        self.skills_list.currentItemChanged.connect(self._on_skill_selected)
        sk_l.addWidget(self.skills_list, 1)
        self.skill_view = QPlainTextEdit()
        self.skill_view.setReadOnly(True)
        self.skill_view.setFont(QFont(_MONO, 9))
        self.skill_view.setStyleSheet(_EDIT_STYLE)
        sk_l.addWidget(self.skill_view, 2)
        self.artifact_tabs.addTab(skills_page, "Skills")

        lessons_page = QWidget()
        le_l = QVBoxLayout(lessons_page)
        le_l.setContentsMargins(0, 0, 0, 0)
        self.lessons_list = QListWidget()
        self.lessons_list.currentItemChanged.connect(self._on_lesson_selected)
        le_l.addWidget(self.lessons_list, 1)
        self.lesson_view = QPlainTextEdit()
        self.lesson_view.setReadOnly(True)
        self.lesson_view.setFont(QFont(_MONO, 9))
        self.lesson_view.setStyleSheet(_EDIT_STYLE)
        le_l.addWidget(self.lesson_view, 2)
        self.artifact_tabs.addTab(lessons_page, "Lessons")
        left_l.addWidget(self.artifact_tabs, 1)
        split.addWidget(left)

        center = QWidget()
        cen_l = QVBoxLayout(center)
        cen_l.setContentsMargins(0, 0, 0, 0)
        cen_l.setSpacing(4)
        cen_l.addWidget(QLabel("solution.py / 观测图像"))
        self.solution_edit = QPlainTextEdit()
        self.solution_edit.setReadOnly(True)
        self.solution_edit.setFont(QFont(_MONO, 9))
        self.solution_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.solution_edit.setStyleSheet(_EDIT_STYLE)
        self.solution_edit.setPlaceholderText("等待 Playground / Test 写入 submission/solution.py…")
        cen_l.addWidget(self.solution_edit, 2)

        img_row = QHBoxLayout()
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(100)
        self.image_list.currentTextChanged.connect(self._preview_image)
        img_row.addWidget(self.image_list, 1)
        self.image_preview = QLabel("无观测图")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(200, 140)
        self.image_preview.setStyleSheet(
            f"QLabel {{ background-color: {_BG_LOG}; border: 1px solid {_BORDER}; "
            f"border-radius: 4px; color: {_MUTED}; }}"
        )
        img_row.addWidget(self.image_preview, 1)
        cen_l.addLayout(img_row, 1)
        split.addWidget(center)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        right_l.addWidget(QLabel("场景分数 / 日志"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: 1px solid {_BORDER}; }}")
        self.scene_host = QWidget()
        self.scene_layout = QVBoxLayout(self.scene_host)
        self.scene_layout.setContentsMargins(4, 4, 4, 4)
        self.scene_layout.setSpacing(3)
        self.scene_layout.addStretch(1)
        scroll.setWidget(self.scene_host)
        right_l.addWidget(scroll, 1)

        self.summary_edit = QPlainTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setFont(QFont(_MONO, 9))
        self.summary_edit.setMaximumHeight(90)
        self.summary_edit.setStyleSheet(_EDIT_STYLE)
        right_l.addWidget(self.summary_edit)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setPlaceholderText("EmbodiedEvo 运行日志…")
        self.log_edit.setStyleSheet(_EDIT_STYLE)
        right_l.addWidget(self.log_edit, 2)
        split.addWidget(right)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

        self._load_defaults_from_toml()
        self._reload_tasks()
        self._suggest_run_dir()
        self._on_mode_changed()
        self._reload_history()
        self._set_idle(True)
        QTimer.singleShot(200, self._probe)

    # ----- helpers -----
    def _repo(self) -> Path:
        return Path(self.repo_edit.text().strip() or DEFAULT_REPO).expanduser()

    def _load_defaults_from_toml(self) -> None:
        data = read_task_toml(self._repo())
        agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
        playground = data.get("playground") if isinstance(data.get("playground"), dict) else {}
        test = data.get("test") if isinstance(data.get("test"), dict) else {}
        sim = data.get("simulator") if isinstance(data.get("simulator"), dict) else {}
        if isinstance(agent, dict):
            if agent.get("model"):
                self.model_edit.setText(str(agent["model"]))
            effort = str(agent.get("reasoning_effort") or "")
            idx = self.reason_combo.findData(effort)
            if idx >= 0:
                self.reason_combo.setCurrentIndex(idx)
        if isinstance(playground, dict) and playground.get("budget") is not None:
            self.pg_budget_spin.setValue(int(playground["budget"]))
        if isinstance(test, dict):
            if test.get("scenes") is not None:
                # UI 默认用较小 scenes 便于调试；仍可手动调到 25
                self.scenes_spin.setValue(min(int(test["scenes"]), 5))
            if test.get("budget") is not None:
                self.test_budget_spin.setValue(int(test["budget"]))
        if isinstance(sim, dict) and sim.get("env_task"):
            self._preferred_task = str(sim["env_task"])
        else:
            self._preferred_task = "stack_bowls"

    def _reload_tasks(self) -> None:
        tasks = list_env_tasks(self._repo())
        current = str(self.task_combo.currentData() or getattr(self, "_preferred_task", ""))
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for spec in tasks:
            self.task_combo.addItem(spec.title, spec.key)
        self.task_combo.blockSignals(False)
        if not tasks:
            self.recipe_edit.setPlainText("未找到 tasks/RoboDojo/recipe/*.md")
            return
        idx = self.task_combo.findData(current)
        self.task_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._on_task_changed()

    def _on_task_changed(self) -> None:
        key = str(self.task_combo.currentData() or "")
        for spec in list_env_tasks(self._repo()):
            if spec.key == key:
                self.recipe_edit.setPlainText(spec.recipe_text or "(空 recipe)")
                break
        if not self._runner.running:
            self._suggest_run_dir()

    def _on_mode_changed(self) -> None:
        smoke = self.mode_combo.currentData() == "smoke"
        # smoke 不用模型 / recipe；其它模式保持可编辑
        self.model_edit.setEnabled(not smoke)
        self.reason_combo.setEnabled(not smoke)
        self.recipe_check.setEnabled(not smoke)
        self.scenes_spin.setEnabled(not smoke)
        self.pg_budget_spin.setEnabled(not smoke)
        self.test_budget_spin.setEnabled(not smoke)
        if smoke and not self.run_dir_edit.text().strip().endswith("_host_smoke"):
            self._suggest_run_dir()

    def _suggest_run_dir(self) -> None:
        key = str(self.task_combo.currentData() or "task")
        mode = str(self.mode_combo.currentData() or "run")
        path = suggest_run_dir(
            self._repo(),
            key,
            baseline=mode == "baseline",
            smoke=mode == "smoke",
        )
        self.run_dir_edit.setText(str(path))

    def _browse_repo(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 EmbodiedEvo 仓库", str(self._repo()))
        if path:
            self.repo_edit.setText(path)
            self.python_edit.setText(resolve_python(Path(path)))
            self._load_defaults_from_toml()
            self._reload_tasks()
            self._reload_history()
            self._probe()

    def _browse_run_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择 run 目录", str(self._repo() / "runs")
        )
        if path:
            self.run_dir_edit.setText(path)
            self._refresh_snapshot()

    def _reload_history(self) -> None:
        runs = list_runs(self._repo())
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("(历史 run…)", "")
        for path in runs:
            self.history_combo.addItem(path.name, str(path))
        self.history_combo.blockSignals(False)

    def _on_history_chosen(self) -> None:
        path = str(self.history_combo.currentData() or "")
        if path:
            self.run_dir_edit.setText(path)
            self._refresh_snapshot()

    def _probe(self) -> None:
        result = probe_repo(self._repo())
        self.python_edit.setText(result.python or self.python_edit.text())
        color = _GREEN if result.ok else _RED
        self.probe_label.setText("\n".join(result.lines))
        self.probe_label.setStyleSheet(f"color: {color};")
        self._reload_tasks()
        self._reload_history()

    def _install_host(self) -> None:
        if self._runner.running:
            return
        self.install_btn.setEnabled(False)
        self.status_label.setText("EmbodiedEvo: 安装 host 依赖中…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        self.log_edit.append("—— 安装 EmbodiedEvo host 依赖 ——")

        class _InstallThread(QObject):
            pass

        # 用简单后台线程，避免卡住 UI
        from PyQt5.QtCore import QThread

        tab = self

        class Worker(QThread):
            line = pyqtSignal(str)
            finished_ok = pyqtSignal(bool, str)

            def run(self_inner) -> None:  # noqa: N805
                ok, msg = ensure_host_venv(
                    tab._repo(), log=self_inner.line.emit
                )
                self_inner.finished_ok.emit(ok, msg)

        worker = Worker(self)
        self._install_worker = worker
        worker.line.connect(self._append_log)
        worker.finished_ok.connect(self._on_install_done)
        worker.start()

    def _on_install_done(self, ok: bool, msg: str) -> None:
        self.install_btn.setEnabled(True)
        self.log_edit.append(msg)
        self.status_label.setText(
            "EmbodiedEvo: host 依赖就绪" if ok else "EmbodiedEvo: host 依赖安装失败"
        )
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._probe()

    def _append_log(self, line: str) -> None:
        self.log_edit.append(line)

    def _request(self) -> RunRequest:
        mode = str(self.mode_combo.currentData() or "run")
        baseline = mode == "baseline"
        smoke = mode == "smoke"
        command = "smoke" if smoke else ("test" if baseline else mode)
        run_dir = Path(
            self.run_dir_edit.text().strip()
            or suggest_run_dir(
                self._repo(),
                str(self.task_combo.currentData() or "task"),
                baseline=baseline,
                smoke=smoke,
            )
        )
        return RunRequest(
            repo=self._repo(),
            command=command,
            env_task=str(self.task_combo.currentData() or "stack_bowls"),
            run_dir=run_dir,
            gpu=self.gpu_edit.text().strip() or "0",
            test_scenes=int(self.scenes_spin.value()),
            baseline=baseline,
            use_recipe=False if smoke else self.recipe_check.isChecked(),
            model=self.model_edit.text().strip() or "gpt-6-astra",
            reasoning=str(self.reason_combo.currentData() or "xhigh"),
            playground_budget=int(self.pg_budget_spin.value()),
            test_budget=int(self.test_budget_spin.value()),
            python=self.python_edit.text().strip(),
            host_client=True,
        )

    def _start(self) -> None:
        if self._runner.running:
            return
        req = self._request()
        if not req.repo.is_dir():
            QMessageBox.warning(self, "EmbodiedEvo", f"仓库不存在:\n{req.repo}")
            return
        # 强制使用 miniconda3/envs/embodiedevo，避免 ros-humble 缺 gymnasium
        conda_py = resolve_python(req.repo)
        self.python_edit.setText(conda_py)
        req.python = conda_py
        ok, msg = host_deps_ok(conda_py, req.repo)
        if not ok:
            reply = QMessageBox.question(
                self,
                "EmbodiedEvo",
                f"{msg}\n\n是否现在安装 host 依赖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._install_host()
            return
        # smoke 不调模型，跳过 Codex / Docker 检查
        if req.command != "smoke":
            probe = probe_repo(req.repo)
            if probe.isolation != "docker":
                if not probe.has_host_codex:
                    QMessageBox.warning(
                        self,
                        "EmbodiedEvo",
                        "缺少本机 Codex CLI。\n"
                        "期望 EmbodiedEvo/.cache/bin/codex，或设置 EMBODIEDEVO_CODEX_BIN。",
                    )
                    self.log_edit.append("启动中止: host Codex CLI missing")
                    return
                if not probe.has_codex_home:
                    QMessageBox.warning(
                        self,
                        "EmbodiedEvo",
                        "缺少 Codex 配置目录（config.toml + auth.json）。\n"
                        "请按 EmbodiedEvo/docs/CODEX_API.md 设置 EMBODIEDEVO_CODEX_HOME。\n"
                        "或先用模式「smoke（无 API）」验证仿真器。",
                    )
                    self.log_edit.append("启动中止: EMBODIEDEVO_CODEX_HOME missing")
                    return
            elif not probe.has_agent_image:
                QMessageBox.warning(
                    self,
                    "EmbodiedEvo",
                    "缺少 Docker 镜像 embodiedevo-agent:latest。\n"
                    "当前 isolation=docker，需要该 agent 镜像。",
                )
                self.log_edit.append(
                    "启动中止: agent image 'embodiedevo-agent:latest' is missing"
                )
                return
        if req.run_dir.exists() and any(req.run_dir.iterdir()) and req.command in {
            "explore",
            "run",
            "smoke",
        }:
            reply = QMessageBox.question(
                self,
                "EmbodiedEvo",
                f"run-dir 非空:\n{req.run_dir}\n\n需要全新目录。是否改用新目录？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._suggest_run_dir()
                req.run_dir = Path(self.run_dir_edit.text())
            else:
                return
        self.log_edit.append(f"—— 启动 {req.command} / {req.env_task} ——")
        self.run_dir_edit.setText(str(req.run_dir))
        try:
            self._runner.start(
                req,
                log=self._bridge.line.emit,
                on_done=self._bridge.done.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_edit.append(f"启动失败: {type(exc).__name__}: {exc}")
            return
        self._set_idle(False)
        self.status_label.setText("EmbodiedEvo: 运行中")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        self._poll.start()

    def _stop(self) -> None:
        self._runner.stop(log=self._bridge.line.emit)
        self.status_label.setText("EmbodiedEvo: 正在停止")

    def _on_run_done(self, code: int) -> None:
        self._poll.stop()
        self._set_idle(True)
        color = _GREEN if code == 0 else _RED
        reason = ""
        run_dir = self.run_dir_edit.text().strip()
        if run_dir:
            snap = snapshot_run(run_dir)
            reason = str(
                snap.protocol.get("playground_reason")
                or (snap.report.get("failure") if snap.report else "")
                or (
                    f"smoke passed={snap.report.get('passed')}"
                    if snap.report and "passed" in snap.report
                    else ""
                )
                or ""
            )
            if code != 0 and reason:
                self.log_edit.append(f"失败原因: {reason}")
            elif code == 0 and snap.report and snap.report.get("passed") is True:
                self.log_edit.append("smoke 通过（无 API）")
        self.status_label.setText(
            f"EmbodiedEvo: 结束 exit={code}"
            + (f" — {reason[:80]}" if reason else "")
        )
        self.status_label.setStyleSheet(f"color: {color};")
        self._refresh_snapshot()
        self._reload_history()

    def _set_idle(self, idle: bool) -> None:
        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(not idle)
        self.mode_combo.setEnabled(idle)
        self.task_combo.setEnabled(idle)

    def _open_run_dir(self) -> None:
        path = Path(self.run_dir_edit.text().strip() or str(self._repo() / "runs"))
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _refresh_snapshot(self) -> None:
        path = self.run_dir_edit.text().strip()
        if not path:
            return
        snap = snapshot_run(path)
        self._apply_snapshot(snap)

    def _apply_snapshot(self, snap: RunSnapshot) -> None:
        self.phase_label.setText(f"phase: {snap.phase or '-'}")
        if snap.success_rate is not None:
            self.sr_label.setText(f"SR: {snap.success_rate * 100:.1f}%")
            self.sr_bar.setValue(int(snap.success_rate * 1000))
            self.sr_bar.setFormat(f"Success rate {snap.success_rate * 100:.1f}%")
        else:
            self.sr_label.setText("SR: -")
            self.sr_bar.setValue(0)
            self.sr_bar.setFormat("Success rate —")
        if snap.avg_score is not None:
            self.score_label.setText(f"Score: {snap.avg_score:.2f}")
        else:
            self.score_label.setText("Score: -")
        self.skill_count_label.setText(f"Skills: {len(snap.skills)}")
        self.lesson_count_label.setText(f"Lessons: {len(snap.lessons)}")
        self.summary_edit.setPlainText("\n".join(snap.summary_lines))

        self.skills_list.clear()
        for art in snap.skills:
            item = QListWidgetItem(art.relative)
            item.setData(Qt.UserRole, str(art.path))
            self.skills_list.addItem(item)
        self.lessons_list.clear()
        for art in snap.lessons:
            item = QListWidgetItem(art.relative)
            item.setData(Qt.UserRole, str(art.path))
            self.lessons_list.addItem(item)

        if snap.solution_text:
            self.solution_edit.setPlainText(snap.solution_text)

        self._images = list(snap.images)
        self.image_list.clear()
        for img in self._images:
            self.image_list.addItem(str(img))
        if self.image_list.count():
            self.image_list.setCurrentRow(0)
        else:
            self.image_preview.setText("无观测图")
            self.image_preview.setPixmap(QPixmap())

        self._rebuild_scene_bars(snap)

    def _rebuild_scene_bars(self, snap: RunSnapshot) -> None:
        while self.scene_layout.count():
            item = self.scene_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._scene_bars = []
        if not snap.scene_scores:
            empty = QLabel("尚无 Test 场景分数（完成 Test 后显示）")
            empty.setStyleSheet(f"color: {_MUTED};")
            self.scene_layout.addWidget(empty)
            self.scene_layout.addStretch(1)
            return
        for sc in snap.scene_scores:
            row = QHBoxLayout()
            lab = QLabel(sc.label)
            lab.setFont(QFont(_MONO, 8))
            lab.setFixedWidth(140)
            row.addWidget(lab)
            bar = QProgressBar()
            bar.setRange(0, 100)
            val = int(sc.score) if sc.score is not None else (100 if sc.success else 0)
            bar.setValue(max(0, min(100, val)))
            if sc.success:
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background-color: {_GREEN}; }}"
                )
            elif not sc.valid:
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background-color: {_RED}; }}"
                )
            row.addWidget(bar, 1)
            host = QWidget()
            host.setLayout(row)
            self.scene_layout.addWidget(host)
            self._scene_bars.append(bar)
        self.scene_layout.addStretch(1)

    def _on_skill_selected(self, cur: Optional[QListWidgetItem], _prev) -> None:
        if cur is None:
            return
        path = Path(str(cur.data(Qt.UserRole) or ""))
        self.skill_view.setPlainText(_read_text(path))

    def _on_lesson_selected(self, cur: Optional[QListWidgetItem], _prev) -> None:
        if cur is None:
            return
        path = Path(str(cur.data(Qt.UserRole) or ""))
        self.lesson_view.setPlainText(_read_text(path))

    def _preview_image(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.image_preview.setText("无观测图")
            self.image_preview.setPixmap(QPixmap())
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.image_preview.setText("无法加载")
            return
        scaled = pix.scaled(
            max(180, self.image_preview.width()),
            max(120, self.image_preview.height()),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_preview.setPixmap(scaled)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"# read error: {exc}"
