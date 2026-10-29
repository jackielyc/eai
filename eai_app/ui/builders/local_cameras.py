"""ui.builders.local cameras — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_cameras_launchers = module_ref('eai_app.adapters.cameras.launchers')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.local_cam_tab = _eai_platform_bootstrap.QWidget()
    local_cam_outer = _eai_platform_bootstrap.QVBoxLayout(build.local_cam_tab)
    local_cam_outer.setContentsMargins(8, 6, 8, 6)
    local_cam_outer.setSpacing(6)

    local_cam_hint = _eai_platform_bootstrap.QLabel(
        "本机有摄像头时可直接「打开本机监控」；Remote SSH 转发请用下方推流/隧道，点「用法」。"
    )
    local_cam_hint.setWordWrap(True)
    local_cam_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    local_cam_outer.addWidget(local_cam_hint)

    _local_cam_group_ss = (
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"margin-top: 6px; padding-top: 4px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY}; }}"
    )

    # 0) 本机直连监控
    monitor_group = _eai_platform_bootstrap.QGroupBox("本机直连监控")
    monitor_group.setStyleSheet(_local_cam_group_ss)
    monitor_layout = _eai_platform_bootstrap.QVBoxLayout(monitor_group)
    monitor_layout.setContentsMargins(8, 10, 8, 8)
    monitor_layout.setSpacing(6)
    monitor_row = _eai_platform_bootstrap.QHBoxLayout()
    monitor_row.setSpacing(6)
    monitor_row.addWidget(_eai_platform_bootstrap.QLabel("设备"))
    self.local_cam_device_combo = _eai_platform_bootstrap.QComboBox()
    self.local_cam_device_combo.setMinimumWidth(280)
    self.local_cam_device_combo.setToolTip(
        "Linux: /dev/videoN（RealSense 彩色多为 video4）；也可在右侧填 index"
    )
    monitor_row.addWidget(self.local_cam_device_combo, 1)
    # 默认不扫描 /dev/video*（v4l2-ctl / OpenCV 探测会卡住启动）
    self.local_cam_device_edit = _eai_platform_bootstrap.QLineEdit("0")
    self.local_cam_device_edit.setFixedWidth(72)
    self.local_cam_device_edit.setToolTip(
        "摄像头 index（macOS 一般为 0）或 /dev/videoN；与左侧列表同步"
    )
    monitor_row.addWidget(self.local_cam_device_edit)
    self.local_cam_refresh_dev_btn = _eai_platform_bootstrap.QPushButton("刷新设备")
    self.local_cam_refresh_dev_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_refresh_dev_btn.setToolTip("重新扫描 /dev/video*")
    self.local_cam_refresh_dev_btn.clicked.connect(self._on_local_cam_refresh_devices)
    monitor_row.addWidget(self.local_cam_refresh_dev_btn)
    self.local_cam_monitor_btn = _eai_platform_bootstrap.QPushButton("打开本机监控")
    self.local_cam_monitor_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_monitor_btn.setToolTip("直接打开本机 V4L2 设备预览（无需推流）")
    self.local_cam_monitor_btn.clicked.connect(self._on_local_cam_monitor_toggled)
    monitor_row.addWidget(self.local_cam_monitor_btn)
    self.local_cam_rotate_btn = _eai_platform_bootstrap.QPushButton("旋转 180°")
    self.local_cam_rotate_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_rotate_btn.setToolTip("顺时针旋转预览画面：180° → 270° → 0° → 90° → 180°")
    self.local_cam_rotate_btn.clicked.connect(self._on_local_cam_rotate_clicked)
    monitor_row.addWidget(self.local_cam_rotate_btn)
    monitor_layout.addLayout(monitor_row)
    self.local_cam_monitor_status = _eai_platform_bootstrap.QLabel("监控: 未开启")
    self.local_cam_monitor_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    monitor_layout.addWidget(self.local_cam_monitor_status)
    local_cam_outer.addWidget(monitor_group)
    self._local_cam_preview_mode = ""  # "device" | "url" | ""
    self._local_cam_rotate_deg = 180  # 默认 180；可点按钮切 0 / 90 / 180 / 270
    self._local_cam_device_combo_syncing = False
    self.local_cam_device_combo.currentIndexChanged.connect(
        self._on_local_cam_device_combo_changed
    )
    self.local_cam_device_edit.editingFinished.connect(
        self._on_local_cam_device_edit_finished
    )
    self._local_cam_device_combo_syncing = True
    self.local_cam_device_combo.addItem("点「刷新设备」扫描 /dev/video*", "0")
    self._local_cam_device_combo_syncing = False

    # 1) 推流
    mjpeg_group = _eai_platform_bootstrap.QGroupBox("1. 本机推流 (MJPEG) — Remote SSH 转发用")
    mjpeg_group.setStyleSheet(_local_cam_group_ss)
    mjpeg_layout = _eai_platform_bootstrap.QVBoxLayout(mjpeg_group)
    mjpeg_layout.setContentsMargins(8, 10, 8, 8)
    mjpeg_layout.setSpacing(6)
    mjpeg_row = _eai_platform_bootstrap.QHBoxLayout()
    mjpeg_row.setSpacing(6)
    mjpeg_row.addWidget(_eai_platform_bootstrap.QLabel("端口"))
    self.local_cam_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.local_cam_port_spin.setRange(1024, 65535)
    self.local_cam_port_spin.setValue(_eai_configuration_chat.LOCAL_WEBCAM_DEFAULT_PORT)
    mjpeg_row.addWidget(self.local_cam_port_spin)
    self.local_cam_mjpeg_btn = _eai_platform_bootstrap.QPushButton("启动推流")
    self.local_cam_mjpeg_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_mjpeg_btn.setToolTip(
        "使用上方「设备」推流，等价于: python3 examples/local_webcam_mjpeg_server.py"
    )
    self.local_cam_mjpeg_btn.clicked.connect(self._on_local_cam_mjpeg_toggled)
    mjpeg_row.addWidget(self.local_cam_mjpeg_btn)
    self.local_cam_mjpeg_status = _eai_platform_bootstrap.QLabel("推流: 未启动")
    self.local_cam_mjpeg_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_mjpeg_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    mjpeg_row.addWidget(self.local_cam_mjpeg_status, 1)
    mjpeg_layout.addLayout(mjpeg_row)
    local_cam_outer.addWidget(mjpeg_group)

    # 2) 隧道
    tunnel_group = _eai_platform_bootstrap.QGroupBox("2. SSH 反向隧道 — Remote SSH 转发用")
    tunnel_group.setStyleSheet(_local_cam_group_ss)
    tunnel_layout = _eai_platform_bootstrap.QVBoxLayout(tunnel_group)
    tunnel_layout.setContentsMargins(8, 10, 8, 8)
    tunnel_layout.setSpacing(6)
    tunnel_row = _eai_platform_bootstrap.QHBoxLayout()
    tunnel_row.setSpacing(6)
    tunnel_row.addWidget(_eai_platform_bootstrap.QLabel("远端"))
    default_ssh = (
        _eai_platform_bootstrap.os.environ.get("EAI_SSH_REVERSE_TARGET", "").strip()
        or _eai_configuration_chat.LOCAL_WEBCAM_SSH_TARGET_DEFAULT
    )
    self.local_cam_ssh_target_edit = _eai_platform_bootstrap.QLineEdit(default_ssh)
    self.local_cam_ssh_target_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_ssh_target_edit.setPlaceholderText("root@42.192.34.154")
    self.local_cam_ssh_target_edit.setToolTip(
        "有摄像头的本机执行时填写 eai 所在远端。也可设环境变量 EAI_SSH_REVERSE_TARGET。"
    )
    tunnel_row.addWidget(self.local_cam_ssh_target_edit, 1)
    tunnel_row.addWidget(_eai_platform_bootstrap.QLabel("远端端口"))
    self.local_cam_remote_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.local_cam_remote_port_spin.setRange(1024, 65535)
    self.local_cam_remote_port_spin.setValue(_eai_configuration_chat.LOCAL_WEBCAM_DEFAULT_PORT)
    tunnel_row.addWidget(self.local_cam_remote_port_spin)
    self.local_cam_tunnel_btn = _eai_platform_bootstrap.QPushButton("启动隧道")
    self.local_cam_tunnel_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_tunnel_btn.setToolTip(
        "等价于: ssh -N -R <远端端口>:127.0.0.1:<本机端口> -p 10693 root@42.192.34.154"
    )
    self.local_cam_tunnel_btn.clicked.connect(self._on_local_cam_tunnel_toggled)
    tunnel_row.addWidget(self.local_cam_tunnel_btn)
    self.local_cam_tunnel_status = _eai_platform_bootstrap.QLabel("隧道: 未启动")
    self.local_cam_tunnel_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_tunnel_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    tunnel_row.addWidget(self.local_cam_tunnel_status, 1)
    tunnel_layout.addLayout(tunnel_row)
    tunnel_extra_row = _eai_platform_bootstrap.QHBoxLayout()
    tunnel_extra_row.setSpacing(6)
    tunnel_extra_row.addWidget(_eai_platform_bootstrap.QLabel("额外 ssh 参数"))
    self.local_cam_ssh_extra_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_chat.LOCAL_WEBCAM_SSH_EXTRA_DEFAULT)
    self.local_cam_ssh_extra_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    tunnel_extra_row.addWidget(self.local_cam_ssh_extra_edit, 1)
    tunnel_layout.addLayout(tunnel_extra_row)
    local_cam_outer.addWidget(tunnel_group)

    # 预览 + 助手
    preview_row = _eai_platform_bootstrap.QHBoxLayout()
    preview_row.setSpacing(6)
    self.local_cam_url_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT)
    self.local_cam_url_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_url_edit.setPlaceholderText("http://127.0.0.1:8090/cam.mjpg")
    preview_row.addWidget(self.local_cam_url_edit, 1)
    self.local_cam_preview_btn = _eai_platform_bootstrap.QPushButton("打开预览")
    self.local_cam_preview_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_preview_btn.clicked.connect(self._on_local_cam_preview_toggled)
    preview_row.addWidget(self.local_cam_preview_btn)
    self.local_cam_help_btn = _eai_platform_bootstrap.QPushButton("用法")
    self.local_cam_help_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_help_btn.clicked.connect(self._on_local_cam_tab_help)
    preview_row.addWidget(self.local_cam_help_btn)
    self.local_cam_mac_helper_btn = _eai_platform_bootstrap.QPushButton("复制 Mac 助手命令")
    self.local_cam_mac_helper_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.local_cam_mac_helper_btn.setToolTip(
        "Remote SSH 时在 Mac 本机终端粘贴运行（推流+隧道独立窗口）"
    )
    self.local_cam_mac_helper_btn.clicked.connect(self._on_local_cam_copy_mac_helper)
    preview_row.addWidget(self.local_cam_mac_helper_btn)
    local_cam_outer.addLayout(preview_row)

    self.local_cam_preview_status = _eai_platform_bootstrap.QLabel("预览: 未开启")
    self.local_cam_preview_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    local_cam_outer.addWidget(self.local_cam_preview_status)

    self.local_cam_preview_label = _eai_platform_bootstrap.QLabel(
        "点「打开本机监控」直接看本机摄像头，或推流后点「打开预览」"
    )
    self.local_cam_preview_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.local_cam_preview_label.setMinimumHeight(200)
    self.local_cam_preview_label.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
    )
    local_cam_outer.addWidget(self.local_cam_preview_label, 1)

    self.local_cam_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.local_cam_log_edit.setReadOnly(True)
    self.local_cam_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.local_cam_log_edit.setPlaceholderText("推流 / 隧道日志…")
    self.local_cam_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.local_cam_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        local_cam_outer,
        self.local_cam_log_edit,
        default_h=140,
        min_h=60,
        max_h=600,
        label="拖动调整本地摄像头日志高度",
    )

    self._local_cam_mjpeg_launcher = _eai_adapters_cameras_launchers.LocalWebcamMjpegLauncher(self)
    self._local_cam_mjpeg_launcher.log_line.connect(self._append_local_cam_log)
    self._local_cam_mjpeg_launcher.status_message.connect(self._on_local_cam_mjpeg_status)
    self._local_cam_mjpeg_launcher.running_changed.connect(self._on_local_cam_mjpeg_running)
    self._local_cam_tunnel_launcher = _eai_adapters_cameras_launchers.SshReverseTunnelLauncher(self)
    self._local_cam_tunnel_launcher.log_line.connect(self._append_local_cam_log)
    self._local_cam_tunnel_launcher.status_message.connect(self._on_local_cam_tunnel_status)
    self._local_cam_tunnel_launcher.running_changed.connect(self._on_local_cam_tunnel_running)
    self._local_cam_preview_cap = None
    self._local_cam_preview_active = False
    self._local_cam_preview_busy = False
    self._local_cam_preview_timer = _eai_platform_bootstrap.QTimer(self)
    self._local_cam_preview_timer.setInterval(66)
    self._local_cam_preview_timer.timeout.connect(self._on_local_cam_preview_tick)
    self.local_cam_port_spin.valueChanged.connect(self._on_local_cam_port_changed)
