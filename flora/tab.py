"""FLORA 标签页：五项标注能力 + miniconda3/envs/flora。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QObject, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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

from flora.catalog import (
    CAPABILITIES,
    DEFAULT_CACHE,
    DEFAULT_ROOT,
    DEFAULT_RUBRIC_JSON,
    HF_COLLECTION,
    PAGE_URL,
    PROFILES,
    WECHAT_URL,
    conda_python,
    ensure_conda_env,
    ensure_repos,
    probe,
    resolve_python,
)
from flora.runner import FloraRunner, RunRequest
from flora.viz import FloraResultPanel

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
    done = pyqtSignal(int)


class FloraTab(QWidget):
    """对接 NVIDIA FLORA：分段、描述、结果/过程/评分规则奖励。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("floraTab")
        self.setStyleSheet(
            f"#floraTab {{ background-color: {_BG}; }}"
            "#floraTab QLabel { background: transparent; }"
        )
        self._runner = FloraRunner()
        self._bridge = _Bridge()
        self._bridge.line.connect(self._append_log)
        self._bridge.done.connect(self._on_run_done)
        self._last_output = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        hint = QLabel(
            "FLORA（NVIDIA GEAR）：工业级机器人数据引擎。"
            "五项能力——时间分段、上下文动作描述、结果奖励、过程奖励、评分规则奖励。"
            "环境装在 miniconda3/envs/flora；权重默认拉到 models/Flora。"
            "官方验证约需 80GB 显存（本机可用 GPU 请看「探测」）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(hint)

        # ---- paths ----
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        path_row.addWidget(QLabel("工作目录"))
        self.root_edit = QLineEdit(os.environ.get("FLORA_ROOT", str(DEFAULT_ROOT)))
        self.root_edit.setFont(QFont(_MONO, 9))
        self.root_edit.setToolTip(
            "检出目录：其下应为 Flora-TempSeg / Flora-Captioner / Flora-Rewarder"
            "（默认 …/liyichao/models）"
        )
        path_row.addWidget(self.root_edit, 1)
        browse_root = QPushButton("…")
        browse_root.setFixedWidth(28)
        browse_root.clicked.connect(self._browse_root)
        path_row.addWidget(browse_root)
        path_row.addWidget(QLabel("Python"))
        self.python_edit = QLineEdit(resolve_python())
        self.python_edit.setFont(QFont(_MONO, 9))
        self.python_edit.setToolTip("miniconda3/envs/flora/bin/python")
        path_row.addWidget(self.python_edit, 1)
        root.addLayout(path_row)

        cache_row = QHBoxLayout()
        cache_row.setSpacing(6)
        cache_row.addWidget(QLabel("模型缓存"))
        self.cache_edit = QLineEdit(os.environ.get("FLORA_MODEL_CACHE", str(DEFAULT_CACHE)))
        self.cache_edit.setFont(QFont(_MONO, 9))
        self.cache_edit.setReadOnly(True)
        self.cache_edit.setToolTip("FLORA_MODEL_CACHE / HF 下载目录")
        cache_row.addWidget(self.cache_edit, 1)
        root.addLayout(cache_row)

        env_row = QHBoxLayout()
        env_row.setSpacing(6)
        self.probe_btn = QPushButton("探测")
        self.probe_btn.clicked.connect(self._probe)
        env_row.addWidget(self.probe_btn)
        self.install_btn = QPushButton("安装环境")
        self.install_btn.setToolTip(
            "创建 miniconda3/envs/flora（优先 conda python=3.12，失败则 venv），"
            "并安装 torch/transformers（分段·描述）与 vllm（奖励）"
        )
        self.install_btn.clicked.connect(self._install_env)
        env_row.addWidget(self.install_btn)
        self.download_btn = QPushButton("下载模型")
        self.download_btn.setToolTip(
            "huggingface snapshot：Flora-TempSeg / Captioner / Rewarder → 工作目录"
        )
        self.download_btn.clicked.connect(self._download_models)
        env_row.addWidget(self.download_btn)
        self.page_btn = QPushButton("项目页")
        self.page_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(PAGE_URL)))
        env_row.addWidget(self.page_btn)
        self.hf_btn = QPushButton("HF 套件")
        self.hf_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(HF_COLLECTION)))
        env_row.addWidget(self.hf_btn)
        self.wx_btn = QPushButton("原文")
        self.wx_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(WECHAT_URL)))
        env_row.addWidget(self.wx_btn)
        env_row.addStretch(1)
        root.addLayout(env_row)

        self.probe_label = QLabel("点「探测」查看 miniconda3/envs/flora 与模型检出。")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.probe_label)

        # ---- config ----
        cfg = QGroupBox("标注配置")
        form = QFormLayout(cfg)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        for mode_id, label, tip in CAPABILITIES:
            self.mode_combo.addItem(label, mode_id)
            idx = self.mode_combo.count() - 1
            self.mode_combo.setItemData(idx, tip, Qt.ToolTipRole)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(QLabel("GPU"))
        self.gpu_edit = QLineEdit(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        self.gpu_edit.setFixedWidth(48)
        self.gpu_edit.setToolTip("CUDA_VISIBLE_DEVICES")
        mode_row.addWidget(self.gpu_edit)
        mode_row.addWidget(QLabel("TP"))
        self.tp_spin = QSpinBox()
        self.tp_spin.setRange(1, 8)
        self.tp_spin.setValue(1)
        self.tp_spin.setToolTip("Rewarder --tensor-parallel-size（rubric 官方默认可用 2）")
        mode_row.addWidget(self.tp_spin)
        mode_row.addWidget(QLabel("samples"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 32)
        self.samples_spin.setValue(1)
        self.samples_spin.setToolTip("Rubric --num-samples（官方默认 9，取 medoid）")
        mode_row.addWidget(self.samples_spin)
        form.addRow("能力", mode_row)

        video_row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("主相机 / 第一视角视频 .mp4")
        self.video_edit.setFont(QFont(_MONO, 9))
        video_row.addWidget(self.video_edit, 1)
        browse_v = QPushButton("…")
        browse_v.setFixedWidth(28)
        browse_v.clicked.connect(lambda: self._browse_file(self.video_edit, "选择视频"))
        video_row.addWidget(browse_v)
        form.addRow("视频", video_row)

        wrist_row = QHBoxLayout()
        self.left_edit = QLineEdit()
        self.left_edit.setPlaceholderText("可选 left wrist")
        self.left_edit.setFont(QFont(_MONO, 9))
        wrist_row.addWidget(self.left_edit, 1)
        bl = QPushButton("…")
        bl.setFixedWidth(28)
        bl.clicked.connect(lambda: self._browse_file(self.left_edit, "左腕视频"))
        wrist_row.addWidget(bl)
        self.right_edit = QLineEdit()
        self.right_edit.setPlaceholderText("可选 right wrist")
        self.right_edit.setFont(QFont(_MONO, 9))
        wrist_row.addWidget(self.right_edit, 1)
        br = QPushButton("…")
        br.setFixedWidth(28)
        br.clicked.connect(lambda: self._browse_file(self.right_edit, "右腕视频"))
        wrist_row.addWidget(br)
        form.addRow("腕部", wrist_row)

        prof_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        for p in PROFILES:
            self.profile_combo.addItem(p, p)
        self.profile_combo.setToolTip("human=单视角；yam/dex/tianji=机器人（可三机位）")
        prof_row.addWidget(self.profile_combo)
        self.yam32_check = QCheckBox("yam32")
        self.yam32_check.setToolTip("YAM 原生 32 秒三视角时间窗")
        prof_row.addWidget(self.yam32_check)
        self.audit_check = QCheckBox("audit")
        self.audit_check.setToolTip("annotate 时写入 temporal/caption 审计字段")
        prof_row.addWidget(self.audit_check)
        self.rationale_check = QCheckBox("rationale")
        self.rationale_check.setToolTip("结果奖励附带一行理由")
        prof_row.addWidget(self.rationale_check)
        self.dry_check = QCheckBox("dry-run")
        self.dry_check.setToolTip("奖励模式：只读视频打 prompt，不加载模型")
        prof_row.addWidget(self.dry_check)
        self.overwrite_check = QCheckBox("overwrite")
        self.overwrite_check.setChecked(True)
        prof_row.addWidget(self.overwrite_check)
        prof_row.addStretch(1)
        form.addRow("选项", prof_row)

        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText("任务指令（annotate 可选；奖励模式必填）")
        self.task_edit.setFont(QFont(_MONO, 9))
        form.addRow("指令", self.task_edit)

        self.robot_edit = QLineEdit("bimanual robot")
        self.robot_edit.setFont(QFont(_MONO, 9))
        form.addRow("机器人", self.robot_edit)

        self.cameras_edit = QLineEdit(
            "a top (scene) camera, a left-wrist camera, a right-wrist camera"
        )
        self.cameras_edit.setFont(QFont(_MONO, 9))
        self.cameras_edit.setToolTip("评分规则：与视频顺序对应的相机描述，逗号分隔")
        form.addRow("相机描述", self.cameras_edit)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit("")
        self.output_edit.setPlaceholderText("留空 → Flora/outputs/ 下默认文件名")
        self.output_edit.setFont(QFont(_MONO, 9))
        out_row.addWidget(self.output_edit, 1)
        browse_out = QPushButton("…")
        browse_out.setFixedWidth(28)
        browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(browse_out)
        form.addRow("输出", out_row)

        root.addWidget(cfg)

        self.rubric_edit = QPlainTextEdit()
        self.rubric_edit.setPlainText(DEFAULT_RUBRIC_JSON)
        self.rubric_edit.setFont(QFont(_MONO, 9))
        self.rubric_edit.setMaximumHeight(120)
        self.rubric_edit.setPlaceholderText("评分规则 JSON（rubric 模式）")
        self.rubric_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        root.addWidget(QLabel("评分规则（仅⑤ Rubric）"))
        root.addWidget(self.rubric_edit)

        # ---- actions ----
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.status_label = QLabel("FLORA: 空闲")
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
        self.open_out_btn = QPushButton("打开产物")
        self.open_out_btn.clicked.connect(self._open_output)
        buttons.addWidget(self.open_out_btn)
        self.viz_btn = QPushButton("刷新可视化")
        self.viz_btn.setToolTip("从输出文件 + 上方视频路径加载时间轴 / 帧预览")
        self.viz_btn.clicked.connect(self._try_load_result)
        buttons.addWidget(self.viz_btn)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        buttons.addWidget(self.clear_btn)
        root.addLayout(buttons)

        splitter = QSplitter(Qt.Vertical)
        self.result_panel = FloraResultPanel()
        splitter.addWidget(self.result_panel)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO, 9))
        self.log_edit.setStyleSheet(
            f"QPlainTextEdit {{ color: {_TEXT}; background-color: {_BG_LOG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        self.log_edit.setPlaceholderText("FLORA 运行日志…")
        splitter.addWidget(self.log_edit)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self._on_mode_changed()
        # 若已有默认产物则预览一次
        QTimer.singleShot(0, self._try_load_result)

    # ---- helpers ----
    def _root(self) -> Path:
        return Path(self.root_edit.text().strip() or str(DEFAULT_ROOT))

    def _python(self) -> str:
        return resolve_python(self.python_edit.text().strip())

    def _mode(self) -> str:
        return str(self.mode_combo.currentData() or "annotate")

    def _append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def _browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 FLORA 工作目录", str(self._root()))
        if path:
            self.root_edit.setText(path)

    def _browse_file(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, edit.text().strip() or str(self._root()), "Video (*.mp4 *.avi *.mkv);;All (*)"
        )
        if path:
            edit.setText(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "输出文件",
            self.output_edit.text().strip() or str(self._root() / "outputs" / "result.json"),
            "JSON/JSONL (*.json *.jsonl);;All (*)",
        )
        if path:
            self.output_edit.setText(path)

    def _on_mode_changed(self) -> None:
        mode = self._mode()
        is_rubric = mode == "rubric"
        is_reward = mode in ("outcome", "process", "rubric")
        self.rubric_edit.setEnabled(is_rubric)
        self.robot_edit.setEnabled(is_rubric)
        self.cameras_edit.setEnabled(is_rubric)
        self.profile_combo.setEnabled(not is_reward)
        self.yam32_check.setEnabled(not is_reward)
        self.audit_check.setEnabled(mode == "annotate")
        self.rationale_check.setEnabled(mode == "outcome")
        self.dry_check.setEnabled(is_reward)
        self.tp_spin.setEnabled(is_reward)
        self.samples_spin.setEnabled(is_rubric)

    def _probe(self) -> None:
        # 同步 UI 缓存路径到环境，便于 probe 显示
        os.environ["FLORA_MODEL_CACHE"] = self.cache_edit.text().strip() or str(DEFAULT_CACHE)
        lines = probe(self._root(), Path(self._python()))
        self.probe_label.setText("\n".join(lines))
        ready = Path(self._python()).is_file()
        self.probe_label.setStyleSheet(f"color: {_GREEN if ready else _MUTED};")

    def _install_env(self) -> None:
        self.install_btn.setEnabled(False)
        self.status_label.setText("FLORA: 安装环境到 miniconda3/envs/flora…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        self.log_edit.appendPlainText("—— 安装依赖 → miniconda3/envs/flora ——")

        class Worker(QThread):
            line = pyqtSignal(str)
            finished_ok = pyqtSignal(bool, str)

            def run(self_inner) -> None:  # noqa: N805
                ok, msg = ensure_conda_env(log=self_inner.line.emit, stack="both")
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
        self.status_label.setText("FLORA: 环境就绪" if ok else "FLORA: 环境安装失败")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._probe()
        if not ok:
            QMessageBox.warning(self, "FLORA", msg[-1500:])

    def _download_models(self) -> None:
        self.download_btn.setEnabled(False)
        self.status_label.setText("FLORA: 下载模型中…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        root = self._root()
        os.environ["FLORA_MODEL_CACHE"] = self.cache_edit.text().strip() or str(DEFAULT_CACHE)

        class Worker(QThread):
            line = pyqtSignal(str)
            finished_ok = pyqtSignal(bool, str)

            def run(self_inner) -> None:  # noqa: N805
                ok, msg = ensure_repos(root=root, log=self_inner.line.emit)
                self_inner.finished_ok.emit(ok, msg)

        worker = Worker(self)
        self._dl_worker = worker
        worker.line.connect(self._append_log)
        worker.finished_ok.connect(self._on_download_done)
        worker.start()

    def _on_download_done(self, ok: bool, msg: str) -> None:
        self.download_btn.setEnabled(True)
        self._append_log(msg)
        self.status_label.setText("FLORA: 模型就绪" if ok else "FLORA: 模型下载失败")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._probe()
        if not ok:
            QMessageBox.warning(self, "FLORA", msg[-1500:])

    def _start(self) -> None:
        if self._runner.running:
            QMessageBox.warning(self, "FLORA", "已有任务在运行")
            return
        mode = self._mode()
        video = self.video_edit.text().strip()
        if not video:
            QMessageBox.warning(self, "FLORA", "请选择输入视频")
            return
        os.environ["FLORA_MODEL_CACHE"] = self.cache_edit.text().strip() or str(DEFAULT_CACHE)
        req = RunRequest(
            mode=mode,
            root=self._root(),
            python=self._python(),
            gpu=self.gpu_edit.text().strip() or "0",
            video=video,
            left_wrist=self.left_edit.text().strip(),
            right_wrist=self.right_edit.text().strip(),
            profile=str(self.profile_combo.currentData() or "human"),
            yam32=self.yam32_check.isChecked(),
            instruction=self.task_edit.text().strip(),
            task=self.task_edit.text().strip(),
            robot=self.robot_edit.text().strip(),
            cameras=self.cameras_edit.text().strip(),
            rubric_json=self.rubric_edit.toPlainText().strip(),
            output=self.output_edit.text().strip(),
            overwrite=self.overwrite_check.isChecked(),
            include_audit=self.audit_check.isChecked(),
            with_rationale=self.rationale_check.isChecked(),
            dry_run=self.dry_check.isChecked(),
            tensor_parallel_size=int(self.tp_spin.value()),
            num_samples=int(self.samples_spin.value()),
        )
        self.status_label.setText(f"FLORA: {mode} 运行中…")
        self.status_label.setStyleSheet(f"color: {_GREEN};")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_edit.appendPlainText(f"—— 启动 {mode} ——")
        try:
            self._runner.start(req, self._bridge.line.emit, self._bridge.done.emit)
        except Exception as exc:  # noqa: BLE001
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("FLORA: 启动失败")
            self.status_label.setStyleSheet(f"color: {_RED};")
            QMessageBox.warning(self, "FLORA", str(exc))

    def _stop(self) -> None:
        self.status_label.setText("FLORA: 正在停止")
        self._runner.stop(self._append_log)

    def _on_run_done(self, code: int) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        ok = code == 0
        self.status_label.setText(f"FLORA: 完成 exit={code}")
        self.status_label.setStyleSheet(f"color: {_GREEN if ok else _RED};")
        self._try_load_result()

    def _guess_output(self) -> Optional[Path]:
        text = self.output_edit.text().strip()
        if text:
            p = Path(text).expanduser()
            if not p.is_absolute():
                p = self._root() / "outputs" / p.name
            return p
        mode = self._mode()
        name = {
            "annotate": "annotations.json",
            "validate": "annotations.json",
            "outcome": "terminal.jsonl",
            "process": "dense.jsonl",
            "rubric": "rubric.jsonl",
        }.get(mode, "result.json")
        return self._root() / "outputs" / name

    def _try_load_result(self) -> None:
        path = self._guess_output()
        if path is None or not path.is_file():
            # 回退：工作目录或 models/outputs 下常见产物
            candidates = [
                self._root() / "outputs" / "annotations.json",
                Path(DEFAULT_ROOT) / "outputs" / "annotations.json",
                Path(DEFAULT_ROOT) / "outputs" / "terminal.jsonl",
                Path(DEFAULT_ROOT) / "outputs" / "dense.jsonl",
            ]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                return
        self._last_output = str(path)
        video = self.video_edit.text().strip()
        video_path = Path(video).expanduser() if video else None
        try:
            self.result_panel.load(path, video_path if video_path and video_path.is_file() else None)
            self.status_label.setText(f"FLORA: 已可视化 {path.name}")
            self.status_label.setStyleSheet(f"color: {_GREEN};")
        except Exception as exc:  # noqa: BLE001
            self.result_panel.clear()
            self.result_panel.summary_label.setText(f"读取产物失败: {exc}\n{path}")
            self._append_log(f"可视化失败: {exc}")

    def _open_output(self) -> None:
        path = Path(self._last_output) if self._last_output else self._guess_output()
        if path is None or not path.exists():
            QMessageBox.information(self, "FLORA", "还没有产物文件")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))
