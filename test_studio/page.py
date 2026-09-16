# -*- coding: utf-8 -*-
"""测试工作室页面：新 UI，推理/对话后端复用 show_camera_topics。"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from test_studio.deploy_controller import DeployView, QwenDeployController


class TestStudioPage(QWidget):
    """全新布局的测试工作室：场景图 + 部署条 + 对话 + 日志。"""

    status_message = pyqtSignal(str)
    scenario_selected = pyqtSignal(str, str)  # title, path

    def __init__(
        self,
        *,
        deploy: QwenDeployController,
        chat_panel: QWidget,
        camera_frame_provider: Optional[
            Callable[[], Optional[Tuple[str, np.ndarray]]]
        ] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        import show_camera_topics as sct

        self._sct = sct
        self._deploy = deploy
        self.chat_panel = chat_panel
        if camera_frame_provider is not None and hasattr(
            chat_panel, "set_camera_frame_provider"
        ):
            chat_panel.set_camera_frame_provider(camera_frame_provider)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title = QLabel("测试工作室")
        title.setStyleSheet(
            f"color: {sct.UI_TEXT_PRIMARY}; font-weight: bold; font-size: 14pt;"
        )
        root.addWidget(title)
        tip = QLabel(
            "独立窗口：左侧选场景图 → 右侧对话发送；上方部署推理服务。"
            "启动方式：bash test_studio/run_test_studio.sh"
        )
        tip.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
        tip.setWordWrap(True)
        root.addWidget(tip)

        # —— 部署条 ——
        deploy_box = QGroupBox("推理服务")
        deploy_layout = QVBoxLayout(deploy_box)
        deploy_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("部署位置"))
        self.target_combo = sct.ImeSafeComboBox()
        self.target_combo.addItem("本地本机", "local")
        for host_id, host_label in sct.REMOTE_QWEN_HOSTS:
            self.target_combo.addItem(host_label, f"remote:{host_id}")
        row.addWidget(self.target_combo)
        row.addWidget(QLabel("权重目录"))
        self.root_combo = sct.ImeSafeComboBox()
        self.root_combo.setMinimumWidth(140)
        row.addWidget(self.root_combo)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.refresh_btn)
        row.addWidget(QLabel("模型"))
        self.model_combo = sct.ImeSafeComboBox()
        self.model_combo.setMinimumWidth(200)
        row.addWidget(self.model_combo, 1)
        self.path_label = QLabel("")
        self.path_label.setFont(QFont(sct.UI_MONO_FAMILY, sct.UI_MONO_SIZE_SMALL))
        self.path_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
        self.path_label.setWordWrap(True)
        row.addWidget(self.path_label, 1)
        deploy_layout.addLayout(row)

        row2 = QHBoxLayout()
        self.status_label = QLabel("服务: --")
        self.status_label.setFont(QFont(sct.UI_MONO_FAMILY, sct.UI_MONO_SIZE_SMALL))
        row2.addWidget(self.status_label, 1)
        self.load_last_btn = QPushButton("上次配置")
        self.load_last_btn.setFixedWidth(72)
        self.load_last_btn.setFocusPolicy(Qt.NoFocus)
        row2.addWidget(self.load_last_btn)
        self.start_btn = QPushButton("启动推理服务")
        self.start_btn.setFocusPolicy(Qt.NoFocus)
        row2.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet(f"color: {sct.UI_ACCENT_RED};")
        self.stop_btn.setFocusPolicy(Qt.NoFocus)
        row2.addWidget(self.stop_btn)
        deploy_layout.addLayout(row2)
        root.addWidget(deploy_box)

        # —— 中部：场景 | 对话 ——
        mid = QSplitter(Qt.Horizontal)
        scenario_host = QWidget()
        scenario_layout = QVBoxLayout(scenario_host)
        scenario_layout.setContentsMargins(0, 0, 0, 0)
        scen_title = QLabel("场景图")
        scen_title.setStyleSheet(f"color: {sct.UI_TEXT_PRIMARY}; font-weight: bold;")
        scenario_layout.addWidget(scen_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)
        self._image_labels: Dict[str, sct.ScaledPixmapLabel] = {}
        self._image_paths: Dict[str, str] = {}
        self._selected_scenario = ""
        cols = 2
        for idx, name in enumerate(sct.TEST_SCENARIO_LABELS):
            cell = QFrame()
            cell.setFrameShape(QFrame.StyledPanel)
            cell_l = QVBoxLayout(cell)
            cell_l.setContentsMargins(4, 4, 4, 4)
            cell_l.setSpacing(4)
            cap = QLabel(name)
            cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)
            cap.setStyleSheet(f"color: {sct.UI_TEXT_PRIMARY};")
            cell_l.addWidget(cap)
            img = sct.ScaledPixmapLabel("暂无图像")
            path = sct.resolve_test_scenario_image_path(name)
            if path is not None:
                pix = QPixmap(path)
                if not pix.isNull():
                    img.set_source_pixmap(pix)
                    img.set_clickable(True)
                    img.setToolTip(f"点击选中「{name}」\n{path}")
                    self._image_paths[name] = path
                    img.clicked.connect(
                        lambda t=name: self._on_scenario_clicked(t)
                    )
            cell_l.addWidget(img, 1)
            self._image_labels[name] = img
            grid.addWidget(cell, idx // cols, idx % cols)
        scroll.setWidget(grid_host)
        scenario_layout.addWidget(scroll, 1)
        mid.addWidget(scenario_host)

        chat_wrap = QGroupBox("AI 对话")
        chat_l = QVBoxLayout(chat_wrap)
        chat_l.setContentsMargins(4, 6, 4, 4)
        chat_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chat_l.addWidget(chat_panel)
        mid.addWidget(chat_wrap)
        mid.setStretchFactor(0, 2)
        mid.setStretchFactor(1, 3)
        root.addWidget(mid, 1)

        # —— 日志 ——
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(sct.UI_MONO_FAMILY, sct.UI_MONO_SIZE_SMALL))
        self.log_edit.setMinimumHeight(72)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setPlaceholderText("推理服务日志…")
        self.log_edit.setStyleSheet(
            f"QTextEdit {{ color: {sct.UI_TEXT_PRIMARY}; background-color: #252525; "
            f"border: 1px solid #555; }}"
        )
        root.addWidget(self.log_edit)

        view = DeployView(
            target_combo=self.target_combo,
            root_combo=self.root_combo,
            model_combo=self.model_combo,
            refresh_btn=self.refresh_btn,
            load_last_btn=self.load_last_btn,
            start_btn=self.start_btn,
            stop_btn=self.stop_btn,
            path_label=self.path_label,
            status_label=self.status_label,
            log_edit=self.log_edit,
            show_message=lambda m: self.status_message.emit(m),
            get_chat_panel=lambda: self.chat_panel,
            focus_host=self,
        )
        self._deploy_view = view
        self._deploy.register_view(view)
        self._deploy.status_message.connect(self.status_message.emit)

    def _on_scenario_clicked(self, title: str) -> None:
        path = self._image_paths.get(title, "")
        if not path:
            self.status_message.emit(f"场景「{title}」无可用图片")
            return
        if hasattr(self.chat_panel, "set_attached_image_from_path"):
            if self.chat_panel.set_attached_image_from_path(path, display_name=title):
                self._set_selection(title)
                self.scenario_selected.emit(title, path)
                self.status_message.emit(f"已选场景图: {title}")

    def _set_selection(self, title: str) -> None:
        self._selected_scenario = title
        for name, label in self._image_labels.items():
            label.set_selected(name == title)

    def sync_scenario_from_path(self, path: str) -> None:
        if not path:
            self._set_selection("")
            return
        for name, p in self._image_paths.items():
            if os.path.abspath(p) == os.path.abspath(path):
                self._set_selection(name)
                return
