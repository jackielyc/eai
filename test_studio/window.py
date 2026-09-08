# -*- coding: utf-8 -*-
"""测试工作室独立主窗口（不嵌入相机 Viewer 的 Tab）。"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from test_studio.deploy_controller import QwenDeployController
from test_studio.page import TestStudioPage


class TestStudioWindow(QMainWindow):
    """独立前端：场景图 + 推理部署 + AI 对话。"""

    def __init__(
        self,
        *,
        llm_config=None,
        camera_frame_provider: Optional[
            Callable[[], Optional[Tuple[str, np.ndarray]]]
        ] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        import show_camera_topics as sct

        self.setWindowTitle("测试工作室")
        self.resize(1280, 860)

        self._deploy = QwenDeployController(self)
        chat = sct.ChatPanelWidget(config=llm_config, parent=self)
        if camera_frame_provider is not None:
            chat.set_camera_frame_provider(camera_frame_provider)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self._page = TestStudioPage(
            deploy=self._deploy,
            chat_panel=chat,
            camera_frame_provider=camera_frame_provider,
            parent=central,
        )
        layout.addWidget(self._page)
        self.setCentralWidget(central)

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status.showMessage("测试工作室就绪（独立窗口）")
        self._page.status_message.connect(self._status.showMessage)
        self._deploy.status_message.connect(self._status.showMessage)
        chat.status_message.connect(self._status.showMessage)
        chat.attached_image_changed.connect(self._page.sync_scenario_from_path)

        self._chat = chat

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._deploy.local_launcher.shutdown()
        except Exception:
            pass
        try:
            self._deploy.remote_launcher.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
