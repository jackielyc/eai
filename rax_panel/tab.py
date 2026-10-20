"""RAX 标签页：准备 native 代码与 miniconda3 环境，并嵌入 Operator Dashboard。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal, QObject
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rax_panel.catalog import (
    DEFAULT_INDEX_REPO,
    DEFAULT_RUN_ROOT,
    conda_python,
    ensure_conda_env,
    ensure_native_checkout,
    has_dashboard,
    probe,
)
from rax_panel.runner import DashboardRunner

_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_GREEN = "#50fa7b"
_RED = "#ff8888"
_BG = "#1e1f22"
_BG_LOG = "#1a1b1e"
_BORDER = "#3d4048"
_MONO = "Monospace"


class _Bridge(QObject):
    line = pyqtSignal(str)
    ready = pyqtSignal(str)
    done = pyqtSignal(int)


class RaxTab(QWidget):
    """对接本地 RAX：图形化打开 robodojo-behavior/native 的操作台。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("raxTab")
        self.setStyleSheet(
            f"#raxTab {{ background-color: {_BG}; }}"
            "#raxTab QLabel { background: transparent; }"
        )
        self._runner = DashboardRunner()
        self._bridge = _Bridge()
        self._bridge.line.connect(self._append_log)
        self._bridge.ready.connect(self._on_dashboard_url)
        self._bridge.done.connect(self._on_dashboard_done)
        self._web = None
        self._dashboard_url = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "RAX（Robot Agent eXploration）：main 是索引，图形操作台在 "
            "robodojo-behavior/native。环境装在 miniconda3/envs/rax，"
            "并通过 runtime/envs/robodojo 链过去。点「启动操作台」后，页面嵌在下方。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        repo_row = QHBoxLayout()
        repo_row.setSpacing(6)
        repo_row.addWidget(QLabel("索引仓库"))
        self.repo_edit = QLineEdit(os.environ.get("RAX_REPO", str(DEFAULT_INDEX_REPO)))
        self.repo_edit.setFont(QFont(_MONO, 9))
        repo_row.addWidget(self.repo_edit, 1)
        browse_repo = QPushButton("…")
        browse_repo.setFixedWidth(28)
        browse_repo.clicked.connect(self._browse_repo)
        repo_row.addWidget(browse_repo)
        repo_row.addWidget(QLabel("运行代码"))
        self.run_root_edit = QLineEdit(
            os.environ.get("RAX_RUN_ROOT", str(DEFAULT_RUN_ROOT))
        )
        self.run_root_edit.setFont(QFont(_MONO, 9))
        repo_row.addWidget(self.run_root_edit, 1)
        browse_run = QPushButton("…")
        browse_run.setFixedWidth(28)
        browse_run.clicked.connect(self._browse_run)
        repo_row.addWidget(browse_run)
        root.addLayout(repo_row)

        py_row = QHBoxLayout()
        py_row.setSpacing(6)
        py_row.addWidget(QLabel("Python"))
        self.python_edit = QLineEdit(str(conda_python()))
        self.python_edit.setFont(QFont(_MONO, 9))
        self.python_edit.setToolTip("miniconda3/envs/rax/bin/python")
        py_row.addWidget(self.python_edit, 1)
        py_row.addWidget(QLabel("端口"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8765)
        py_row.addWidget(self.port_spin)
        root.addLayout(py_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.prepare_btn = QPushButton("准备代码")
        self.prepare_btn.setToolTip("为 robodojo-behavior/native 添加 git worktree")
        self.prepare_btn.clicked.connect(self._prepare_code)
        btn_row.addWidget(self.prepare_btn)
        self.install_btn = QPushButton("安装环境")
        self.install_btn.setToolTip("在 miniconda3/envs/rax 创建 Python 3.11 并安装操作台依赖")
        self.install_btn.clicked.connect(self._install_env)
        btn_row.addWidget(self.install_btn)
        self.probe_btn = QPushButton("探测")
        self.probe_btn.clicked.connect(self._probe)
        btn_row.addWidget(self.probe_btn)
        self.start_btn = QPushButton("启动操作台")
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.stop_btn)
        self.browser_btn = QPushButton("浏览器打开")
        self.browser_btn.clicked.connect(self._open_browser)
        btn_row.addWidget(self.browser_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.status_label = QLabel("RAX: 空闲")
        self.status_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.status_label)
        self.probe_label = QLabel("点「探测」查看代码检出和 miniconda3/envs/rax。")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.probe_label)

        splitter = QSplitter(Qt.Vertical)
        self._web_host = QWidget()
        web_layout = QVBoxLayout(self._web_host)
        web_layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(
            "Operator Dashboard 会显示在这里。\n"
            "页面可准备、启动、查看相机画面和轨迹；不要把带 token 的地址发给 agent。"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(f"color: {_MUTED};")
        web_layout.addWidget(self._placeholder, 1)
        splitter.addWidget(self._web_host)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        self.log_edit.setPlaceholderText("RAX 操作台日志…")
        splitter.addWidget(self.log_edit)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _index_repo(self) -> Path:
        return Path(self.repo_edit.text().strip() or str(DEFAULT_INDEX_REPO))

    def _run_root(self) -> Path:
        return Path(self.run_root_edit.text().strip() or str(DEFAULT_RUN_ROOT))

    def _python(self) -> Path:
        text = self.python_edit.text().strip()
        return Path(text) if text else conda_python()

    def _resolved_project(self) -> Optional[Path]:
        run_root = self._run_root()
        if has_dashboard(run_root):
            return run_root
        index_repo = self._index_repo()
        if has_dashboard(index_repo):
            return index_repo
        return None

    def _append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def _browse_repo(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 RAX 索引仓库", str(self._index_repo()))
        if path:
            self.repo_edit.setText(path)

    def _browse_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 RAX 运行代码", str(self._run_root()))
        if path:
            self.run_root_edit.setText(path)

    def _probe(self) -> None:
        lines = probe(self._index_repo(), self._run_root(), self._python())
        self.probe_label.setText("\n".join(lines))
        ready = self._resolved_project() is not None and self._python().is_file()
        self.probe_label.setStyleSheet(f"color: {_GREEN if ready else _MUTED};")

    def _prepare_code(self) -> None:
        self.prepare_btn.setEnabled(False)
        self.status_label.setText("RAX: 准备代码中…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        index_repo = self._index_repo()
        dest = self._run_root()

        class Worker(QThread):
            line = pyqtSignal(str)
            finished_ok = pyqtSignal(bool, str)

            def run(self_inner) -> None:  # noqa: N805
                ok, msg = ensure_native_checkout(
                    index_repo, dest, log=self_inner.line.emit
                )
                self_inner.finished_ok.emit(ok, msg)

        worker = Worker(self)
        self._prepare_worker = worker
        worker.line.connect(self._append_log)
        worker.finished_ok.connect(self._on_prepare_done)
        worker.start()

    def _on_prepare_done(self, ok: bool, msg: str) -> None:
        self.prepare_btn.setEnabled(True)
        self._append_log(msg)
        if ok and has_dashboard(Path(msg)):
            self.run_root_edit.setText(msg)
        self.status_label.setText("RAX: 代码就绪" if ok else "RAX: 准备代码失败")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._probe()
        if not ok:
            QMessageBox.warning(self, "RAX", msg)

    def _install_env(self) -> None:
        self.install_btn.setEnabled(False)
        self.status_label.setText("RAX: 安装环境到 miniconda3/envs/rax…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")

        class Worker(QThread):
            line = pyqtSignal(str)
            finished_ok = pyqtSignal(bool, str)

            def run(self_inner) -> None:  # noqa: N805
                ok, msg = ensure_conda_env(log=self_inner.line.emit)
                self_inner.finished_ok.emit(ok, msg)

        worker = Worker(self)
        self._install_worker = worker
        worker.line.connect(self._append_log)
        worker.finished_ok.connect(self._on_install_done)
        worker.start()

    def _on_install_done(self, ok: bool, msg: str) -> None:
        self.install_btn.setEnabled(True)
        self._append_log(msg)
        if ok:
            self.python_edit.setText(str(conda_python()))
        self.status_label.setText("RAX: 环境就绪" if ok else "RAX: 环境安装失败")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._probe()
        if not ok:
            QMessageBox.warning(self, "RAX", msg)

    def _start(self) -> None:
        if self._runner.running:
            QMessageBox.information(self, "RAX", "操作台已在运行。")
            return
        project = self._resolved_project()
        if project is None:
            QMessageBox.warning(self, "RAX", "还没有可运行代码。请先点「准备代码」。")
            return
        python = self._python()
        if not python.is_file():
            QMessageBox.warning(
                self,
                "RAX",
                f"找不到 Python:\n{python}\n请先点「安装环境」。",
            )
            return
        self.start_btn.setEnabled(False)
        self.status_label.setText("RAX: 启动操作台…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        self._dashboard_url = ""
        try:
            self._runner.start(
                project,
                python,
                int(self.port_spin.value()),
                self._bridge.line.emit,
                self._bridge.ready.emit,
                self._bridge.done.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.start_btn.setEnabled(True)
            self.status_label.setText("RAX: 启动失败")
            self.status_label.setStyleSheet(f"color: {_RED};")
            QMessageBox.warning(self, "RAX", str(exc))

    def _ensure_web(self):
        if self._web is not None:
            return self._web
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"无法嵌入页面（{exc}），请用「浏览器打开」。")
            return None
        view = QWebEngineView(self._web_host)
        view.setStyleSheet("QWebEngineView { background-color: #1a1b1e; }")
        layout = self._web_host.layout()
        self._placeholder.hide()
        layout.addWidget(view, 1)
        self._web = view
        return view

    def _on_dashboard_url(self, url: str) -> None:
        self._dashboard_url = url
        self.status_label.setText("RAX: 操作台运行中")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        view = self._ensure_web()
        if view is not None:
            view.setUrl(QUrl(url))
        else:
            QDesktopServices.openUrl(QUrl(url))

    def _on_dashboard_done(self, code: int) -> None:
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"RAX: 操作台结束 exit={code}")
        self.status_label.setStyleSheet(f"color: {_MUTED if code == 0 else _RED};")

    def _stop(self) -> None:
        self._runner.stop(self._append_log)
        self.start_btn.setEnabled(True)
        self.status_label.setText("RAX: 正在停止")
        self.status_label.setStyleSheet(f"color: {_MUTED};")

    def _open_browser(self) -> None:
        if not self._dashboard_url:
            QMessageBox.information(self, "RAX", "请先启动操作台。")
            return
        QDesktopServices.openUrl(QUrl(self._dashboard_url))
