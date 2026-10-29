"""ui.builders.real evaluation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.real_inner = _eai_platform_bootstrap.QWidget()
    real_outer = _eai_platform_bootstrap.QVBoxLayout(build.real_inner)
    real_outer.setContentsMargins(8, 6, 8, 6)
    real_outer.setSpacing(6)

    self.real_eval_hint_label = _eai_platform_bootstrap.QLabel("")
    self.real_eval_hint_label.setWordWrap(True)
    self.real_eval_hint_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    real_outer.addWidget(self.real_eval_hint_label)

    # 可拖动的真机后端选择条（顺序写入 eai/real_robot_backends_order.json）
    # 隐藏 radio 仍保留，供 viewer_ui_state / 既有 isChecked 逻辑使用
    real_backend_row = _eai_platform_bootstrap.QHBoxLayout()
    real_backend_row.setSpacing(8)
    real_backend_row.addWidget(_eai_platform_bootstrap.QLabel("机器人后端"))
    self.real_robot_backend_tab_bar = _eai_platform_bootstrap.QTabBar()
    self.real_robot_backend_tab_bar.setObjectName("realRobotBackendTabs")
    self.real_robot_backend_tab_bar.setMovable(True)
    self.real_robot_backend_tab_bar.setExpanding(False)
    self.real_robot_backend_tab_bar.setDocumentMode(True)
    self.real_robot_backend_tab_bar.setUsesScrollButtons(True)
    self.real_robot_backend_tab_bar.setDrawBase(False)
    self.real_robot_backend_tab_bar.setElideMode(_eai_platform_bootstrap.Qt.ElideRight)
    self.real_robot_backend_tab_bar.setToolTip(
        "可拖动调整真机后端顺序；下次启动会按此顺序打开"
        "（写入 eai/real_robot_backends_order.json）"
    )
    self.real_robot_a2d_radio = _eai_platform_bootstrap.QRadioButton("A2D")
    self.real_robot_r1_radio = _eai_platform_bootstrap.QRadioButton("R1")
    self.real_robot_franka_radio = _eai_platform_bootstrap.QRadioButton("Franka")
    self.real_robot_a2d_radio.setChecked(True)
    self.real_robot_a2d_radio.setToolTip(
        "A2D 真机：a2d-tele / A2D_SCRIPTS_DIR，现场 /camera/*_color + HAL 臂手栈。"
    )
    self.real_robot_r1_radio.setToolTip(
        "R1 真机：r1-tele / R1_SCRIPTS_DIR（可设 R1_WORKSPACE_DIR）。"
        " 与 A2D 工作区独立；栈脚本就绪后可直接启动。"
    )
    self.real_robot_franka_radio.setToolTip(
        "Franka 真机：franka-tele / FRANKA_SCRIPTS_DIR（可设 FRANKA_WORKSPACE_DIR）。"
        " 与 A2D/R1 工作区独立；栈脚本就绪后可直接启动。"
    )
    self._real_robot_backend_group = _eai_platform_bootstrap.QButtonGroup(self)
    self._real_robot_backend_group.addButton(self.real_robot_a2d_radio, 0)
    self._real_robot_backend_group.addButton(self.real_robot_r1_radio, 1)
    self._real_robot_backend_group.addButton(self.real_robot_franka_radio, 2)
    self._real_robot_radio_by_key = {
        key: getattr(self, attr)
        for key, _label, attr in _eai_configuration_policies.REAL_ROBOT_SELECTOR_SPECS
    }
    self._real_robot_tabs_guard = False
    self._real_robot_tabs_order_guard = False
    for build.radio in self._real_robot_radio_by_key.values():
        build.radio.setVisible(False)
        build.radio.toggled.connect(self._on_real_robot_backend_changed)
    self.real_robot_backend_tab_bar.currentChanged.connect(
        self._on_real_robot_backend_tab_changed
    )
    self.real_robot_backend_tab_bar.tabMoved.connect(
        self._on_real_robot_backend_tab_moved
    )
    self._real_robot_tabs_guard = True
    self._real_robot_tabs_order_guard = True
    try:
        tip_by_key = {
            _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D: self.real_robot_a2d_radio.toolTip(),
            _eai_configuration_policies.REAL_ROBOT_BACKEND_R1: self.real_robot_r1_radio.toolTip(),
            _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA: self.real_robot_franka_radio.toolTip(),
        }
        build.spec_by_key = {
            k: (label, attr) for k, label, attr in _eai_configuration_policies.REAL_ROBOT_SELECTOR_SPECS
        }
        for build.key in _eai_infrastructure_ui_storage.preferred_real_robot_backends_order():
            build.pair = build.spec_by_key.get(build.key)
            if build.pair is None:
                continue
            build.label, build._attr = build.pair
            build.idx = self.real_robot_backend_tab_bar.addTab(build.label)
            self.real_robot_backend_tab_bar.setTabData(build.idx, build.key)
            build.tip = tip_by_key.get(build.key, "")
            if build.tip:
                self.real_robot_backend_tab_bar.setTabToolTip(build.idx, build.tip)
    finally:
        self._real_robot_tabs_guard = False
        self._real_robot_tabs_order_guard = False
    self._sync_real_robot_backend_tab_from_radios()
    real_backend_row.addWidget(self.real_robot_backend_tab_bar, 1)
    real_outer.addLayout(real_backend_row)

    real_task_row = _eai_platform_bootstrap.QHBoxLayout()
    real_task_row.setSpacing(6)
    real_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.real_eval_task_edit = _eai_platform_bootstrap.QLineEdit()
    self.real_eval_task_edit.setPlaceholderText("评测任务描述（可选）")
    real_task_row.addWidget(self.real_eval_task_edit, 1)
    real_outer.addLayout(real_task_row)

    real_ctrl_row = _eai_platform_bootstrap.QHBoxLayout()
    real_ctrl_row.setSpacing(6)
    self.real_eval_status_label = _eai_platform_bootstrap.QLabel("状态: 未开始")
    self.real_eval_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    real_ctrl_row.addWidget(self.real_eval_status_label, 1)
    self.real_eval_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新就绪检查")
    self.real_eval_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_refresh_btn.clicked.connect(self._refresh_real_eval_status)
    real_ctrl_row.addWidget(self.real_eval_refresh_btn)
    self.real_eval_stack_btn = _eai_platform_bootstrap.QPushButton("启动机器人栈")
    self.real_eval_stack_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_stack_btn.setToolTip("按所选机器人后端拉起 robot stack")
    self.real_eval_stack_btn.clicked.connect(self._on_real_eval_stack_clicked)
    real_ctrl_row.addWidget(self.real_eval_stack_btn)
    self.real_eval_start_btn = _eai_platform_bootstrap.QPushButton("开始评测")
    self.real_eval_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_start_btn.setToolTip(
        "进入真机评测记录。\n"
        "先决条件：已有 /camera/*_color、手臂与手部已使能"
        "（R1 / Franka 还需栈脚本就绪）。"
    )
    self.real_eval_start_btn.clicked.connect(self._on_real_eval_start_clicked)
    real_ctrl_row.addWidget(self.real_eval_start_btn)
    self.real_eval_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.real_eval_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_stop_btn.setObjectName("dangerAction")
    self.real_eval_stop_btn.setEnabled(False)
    self.real_eval_stop_btn.clicked.connect(self._on_real_eval_stop_clicked)
    real_ctrl_row.addWidget(self.real_eval_stop_btn)
    self.real_eval_clear_log_btn = _eai_platform_bootstrap.QPushButton("清空日志")
    self.real_eval_clear_log_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_clear_log_btn.clicked.connect(self._on_real_eval_clear_log_clicked)
    real_ctrl_row.addWidget(self.real_eval_clear_log_btn)
    real_outer.addLayout(real_ctrl_row)

    self.real_eval_ready_label = _eai_platform_bootstrap.QLabel("")
    self.real_eval_ready_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.real_eval_ready_label.setWordWrap(True)
    self.real_eval_ready_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    real_outer.addWidget(self.real_eval_ready_label)

    # A2D：采集标定 / 模型·HIL / 常见问题 / 视频回放（仅 A2D 后端显示）
    self.a2d_ops_panel = self._build_a2d_ops_panel()
    real_outer.addWidget(self.a2d_ops_panel)

    real_cam_row = _eai_platform_bootstrap.QHBoxLayout()
    real_cam_row.setSpacing(6)
    real_cam_row.addWidget(_eai_platform_bootstrap.QLabel("本地摄像头"))
    self.real_eval_cam_url_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT)
    self.real_eval_cam_url_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.real_eval_cam_url_edit.setPlaceholderText(
        "http://127.0.0.1:8090/cam.mjpg  （Mac 本机推流经 SSH -R）"
    )
    self.real_eval_cam_url_edit.setToolTip(_eai_configuration_chat.REAL_EVAL_LOCAL_CAM_HELP)
    real_cam_row.addWidget(self.real_eval_cam_url_edit, 1)
    self.real_eval_cam_help_btn = _eai_platform_bootstrap.QPushButton("用法")
    self.real_eval_cam_help_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_cam_help_btn.setToolTip("Mac/本机摄像头经 SSH 隧道到远端的步骤")
    self.real_eval_cam_help_btn.clicked.connect(self._on_real_eval_local_cam_help)
    real_cam_row.addWidget(self.real_eval_cam_help_btn)
    self.real_eval_cam_preview_btn = _eai_platform_bootstrap.QPushButton("本地摄像头预览")
    self.real_eval_cam_preview_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.real_eval_cam_preview_btn.setToolTip(
        "打开下方预览：读取本机经隧道转发的 MJPEG/RTSP，或本机 /dev/video*"
    )
    self.real_eval_cam_preview_btn.clicked.connect(self._on_real_eval_local_cam_toggled)
    real_cam_row.addWidget(self.real_eval_cam_preview_btn)
    real_outer.addLayout(real_cam_row)

    self.real_eval_cam_status_label = _eai_platform_bootstrap.QLabel("本地摄像头: 未开启")
    self.real_eval_cam_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    real_outer.addWidget(self.real_eval_cam_status_label)

    self.real_eval_cam_preview_label = _eai_platform_bootstrap.QLabel(
        "点击「本地摄像头预览」显示本机画面（Remote SSH/VNC 需先推流，点「用法」）"
    )
    self.real_eval_cam_preview_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.real_eval_cam_preview_label.setMinimumHeight(200)
    self.real_eval_cam_preview_label.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
    )
    real_outer.addWidget(self.real_eval_cam_preview_label, 1)

    self.real_eval_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.real_eval_log_edit.setReadOnly(True)
    self.real_eval_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.real_eval_log_edit.setPlaceholderText("真机评测日志…")
    self.real_eval_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.real_eval_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        real_outer,
        self.real_eval_log_edit,
        default_h=180,
        min_h=80,
        max_h=900,
        label="拖动调整真机日志高度",
    )

    self._real_eval_active = False
    self._real_eval_timer = _eai_platform_bootstrap.QTimer(self)
    self._real_eval_timer.setInterval(2000)
    self._real_eval_timer.timeout.connect(self._refresh_real_eval_status)
    self._real_eval_local_cap = None
    self._real_eval_local_cam_active = False
    self._real_eval_local_cam_busy = False
    self._real_eval_local_cam_timer = _eai_platform_bootstrap.QTimer(self)
    self._real_eval_local_cam_timer.setInterval(66)  # ~15 Hz
    self._real_eval_local_cam_timer.timeout.connect(self._on_real_eval_local_cam_tick)
    self._on_real_robot_backend_changed()
