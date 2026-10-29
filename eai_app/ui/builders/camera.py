"""ui.builders.camera — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def build(self, build):
    camera_tab = _eai_platform_bootstrap.QWidget()
    camera_layout = _eai_platform_bootstrap.QHBoxLayout(camera_tab)
    camera_layout.setContentsMargins(8, 6, 8, 6)
    camera_layout.addWidget(_eai_platform_bootstrap.QLabel("Topic 前缀:"))
    self.prefix_edit = _eai_platform_bootstrap.QLineEdit(build.prefix)
    self.prefix_edit.setPlaceholderText("/camera")
    camera_layout.addWidget(self.prefix_edit, 1)
    self.camera_connect_btn = _eai_platform_bootstrap.QPushButton("连接相机 Topic")
    self.camera_connect_btn.setToolTip(
        "启动 ROS topic 扫描并刷新列表；默认启动不自动订阅，点此后再勾选图像。"
    )
    self.camera_connect_btn.clicked.connect(self._on_refresh_clicked)
    camera_layout.addWidget(self.camera_connect_btn)
    select_all_btn = _eai_platform_bootstrap.QPushButton("全选图像")
    select_all_btn.clicked.connect(self._select_all_images)
    camera_layout.addWidget(select_all_btn)
    clear_btn = _eai_platform_bootstrap.QPushButton("取消全选")
    clear_btn.clicked.connect(self._clear_selection)
    camera_layout.addWidget(clear_btn)
    camera_layout.addStretch()
    build.control_tabs.addTab(camera_tab, "大脑")
    _eai_platform_bootstrap._boot_tick("构建界面：回放…")
