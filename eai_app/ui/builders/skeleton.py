"""ui.builders.skeleton — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    skeleton_tab = _eai_platform_bootstrap.QWidget()
    skeleton_layout = _eai_platform_bootstrap.QVBoxLayout(skeleton_tab)
    skeleton_layout.setContentsMargins(8, 6, 8, 6)
    skeleton_layout.setSpacing(6)

    sk_row1 = _eai_platform_bootstrap.QHBoxLayout()
    sk_row1.setSpacing(6)
    sk_row1.addWidget(_eai_platform_bootstrap.QLabel("相机:"))
    self.skeleton_cam_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.skeleton_cam_combo.setMinimumWidth(220)
    self.skeleton_cam_combo.setToolTip(
        "图像来源：本机摄像头 / 网络流（VNC 场景）/ ROS 彩色 topic。\n"
        "VNC 登录时请用「网络流」+ 本机推流（点「用法」）。"
    )
    self.skeleton_cam_combo.currentIndexChanged.connect(
        self._on_skeleton_cam_source_changed
    )
    sk_row1.addWidget(self.skeleton_cam_combo, 1)
    self.skeleton_refresh_cam_btn = _eai_platform_bootstrap.QPushButton("刷新列表")
    self.skeleton_refresh_cam_btn.setToolTip("重新探测本机摄像头，并刷新 ROS 彩色 topic 列表")
    self.skeleton_refresh_cam_btn.clicked.connect(self._refresh_skeleton_camera_list)
    sk_row1.addWidget(self.skeleton_refresh_cam_btn)
    self.skeleton_net_help_btn = _eai_platform_bootstrap.QPushButton("用法")
    self.skeleton_net_help_btn.setToolTip("本机摄像头经 SSH 隧道转发到远端的步骤")
    self.skeleton_net_help_btn.clicked.connect(self._on_skeleton_network_help)
    sk_row1.addWidget(self.skeleton_net_help_btn)
    skeleton_layout.addLayout(sk_row1)

    sk_url_row = _eai_platform_bootstrap.QHBoxLayout()
    sk_url_row.setSpacing(6)
    sk_url_row.addWidget(_eai_platform_bootstrap.QLabel("网络流:"))
    self.skeleton_stream_url_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT)
    self.skeleton_stream_url_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.skeleton_stream_url_edit.setPlaceholderText(
        "http://127.0.0.1:8090/cam.mjpg  或  rtsp://..."
    )
    self.skeleton_stream_url_edit.setToolTip(_eai_configuration_chat.SKELETON_NETWORK_HELP)
    sk_url_row.addWidget(self.skeleton_stream_url_edit, 1)
    skeleton_layout.addLayout(sk_url_row)

    sk_row2 = _eai_platform_bootstrap.QHBoxLayout()
    sk_row2.setSpacing(8)
    self.skeleton_flip_check = _eai_platform_bootstrap.QCheckBox("水平翻转预览")
    self.skeleton_flip_check.setChecked(True)
    self.skeleton_flip_check.setToolTip("自拍/面对相机时建议开启，便于对照屏幕")
    sk_row2.addWidget(self.skeleton_flip_check)
    self.skeleton_mirror_map_check = _eai_platform_bootstrap.QCheckBox("镜像映射到机器人")
    self.skeleton_mirror_map_check.setChecked(True)
    self.skeleton_mirror_map_check.setToolTip(
        "开启：人物左手→机器人右手（面对机器人更自然）\n关闭：人物左手→机器人左手"
    )
    sk_row2.addWidget(self.skeleton_mirror_map_check)
    self.skeleton_ctrl_left_check = _eai_platform_bootstrap.QCheckBox("控左手")
    self.skeleton_ctrl_left_check.setChecked(True)
    sk_row2.addWidget(self.skeleton_ctrl_left_check)
    self.skeleton_ctrl_right_check = _eai_platform_bootstrap.QCheckBox("控右手")
    self.skeleton_ctrl_right_check.setChecked(True)
    sk_row2.addWidget(self.skeleton_ctrl_right_check)
    sk_row2.addStretch()
    skeleton_layout.addLayout(sk_row2)

    sk_row3 = _eai_platform_bootstrap.QHBoxLayout()
    sk_row3.setSpacing(6)
    self.skeleton_track_btn = _eai_platform_bootstrap.QPushButton("开始识别")
    self.skeleton_track_btn.setToolTip("启动 MediaPipe Hands 识别（仅预览骨架，不发送命令）")
    self.skeleton_track_btn.clicked.connect(self._on_skeleton_track_toggled)
    sk_row3.addWidget(self.skeleton_track_btn)
    self.skeleton_teleop_check = _eai_platform_bootstrap.QCheckBox("开启遥控")
    self.skeleton_teleop_check.setToolTip(
        "勾选后将骨架开合量实时发布到 /ry_hand/*/set_angles。\n请先「启用手」，并注意周围安全。"
    )
    self.skeleton_teleop_check.toggled.connect(self._on_skeleton_teleop_toggled)
    sk_row3.addWidget(self.skeleton_teleop_check)
    sk_row3.addWidget(_eai_platform_bootstrap.QLabel("平滑"))
    self.skeleton_smooth_slider = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.skeleton_smooth_slider.setRange(10, 90)
    self.skeleton_smooth_slider.setValue(35)
    self.skeleton_smooth_slider.setFixedWidth(100)
    self.skeleton_smooth_slider.setToolTip("跟踪平滑：左小右大（越大越跟手、越抖）")
    sk_row3.addWidget(self.skeleton_smooth_slider)
    self.skeleton_status_label = _eai_platform_bootstrap.QLabel("手骨架: 未启动")
    self.skeleton_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.skeleton_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
    sk_row3.addWidget(self.skeleton_status_label, 1)
    skeleton_layout.addLayout(sk_row3)

    self.skeleton_preview_label = _eai_platform_bootstrap.QLabel(
        "选本地摄像头 / 网络流 / ROS topic → 开始识别（VNC 请看「用法」）"
    )
    self.skeleton_preview_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.skeleton_preview_label.setMinimumHeight(220)
    self.skeleton_preview_label.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; color: #aaa; }}"
    )
    skeleton_layout.addWidget(self.skeleton_preview_label, 1)

    self.skeleton_joints_label = _eai_platform_bootstrap.QLabel("关节: --")
    self.skeleton_joints_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.skeleton_joints_label.setWordWrap(True)
    skeleton_layout.addWidget(self.skeleton_joints_label)

    build.control_tabs.addTab(skeleton_tab, "手骨架遥控")
