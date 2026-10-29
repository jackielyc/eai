"""ui.builders.robot services — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_adapters_robot_launchers = module_ref('eai_app.adapters.robot.launchers')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def build(self, build):
    robot_tab = _eai_platform_bootstrap.QWidget()
    robot_layout = _eai_platform_bootstrap.QVBoxLayout(robot_tab)
    robot_layout.setContentsMargins(8, 6, 8, 6)
    robot_layout.setSpacing(6)
    stack_row = _eai_platform_bootstrap.QHBoxLayout()
    self._stack_launcher = _eai_adapters_robot_launchers.RobotStackLauncher(build.node, self)
    self.robot_stack_status = _eai_platform_bootstrap.QLabel("robot: -- | stack: --")
    self.robot_stack_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_stack_status.setToolTip(
        f"robot-service 初始化 HAL；手/臂栈启动 pedal_controller/topic_router/ik/fk/arm/hand。\n"
        f"不启动 VR、扫码、数据采集、RealSense 等。\n"
        f"日志: tail -f {_eai_configuration_policies.STACK_SERVICES_LOG}"
    )
    self.robot_stack_btn = _eai_platform_bootstrap.QPushButton("启动机器人栈")
    self.robot_stack_btn.setToolTip(
        "依次启动 robot-service（约 3 分钟）与手/臂精简服务栈。\n"
        "若 HAL 已就绪则跳过 robot-service。\n"
        "仅含控制手臂与手所需节点。"
    )
    self.robot_stack_btn.clicked.connect(self._on_robot_stack_clicked)
    stack_row.addWidget(self.robot_stack_status, 1)
    stack_row.addWidget(self.robot_stack_btn)
    robot_layout.addLayout(stack_row)

    robot_conn_row = _eai_platform_bootstrap.QHBoxLayout()
    self.robot_state_connect_btn = _eai_platform_bootstrap.QPushButton("连接机器人")
    self.robot_state_connect_btn.setToolTip(
        "启动最小宿主机节点：相机 forwarder + 灵巧手 + 臂/手使能桥 + FK。\n"
        "用于看图、控手、控臂。再点一次「取消连接」会停止这些节点。"
    )
    self.robot_state_connect_btn.clicked.connect(self._on_connect_robot_state_clicked)
    self.robot_state_connect_label = _eai_platform_bootstrap.QLabel("状态: 未连接")
    self.robot_state_connect_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_state_connect_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    robot_conn_row.addWidget(self.robot_state_connect_btn)
    robot_conn_row.addWidget(self.robot_state_connect_label)
    self.robot_nodes_status_label = _eai_platform_bootstrap.QLabel("节点: （连接后显示）")
    self.robot_nodes_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_nodes_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.robot_nodes_status_label.setWordWrap(False)
    self.robot_nodes_status_label.setToolTip(
        "最小栈节点（看图/控手/控臂）：\n"
        "相机 forwarder · 灵巧手 hand_node · 臂/手使能桥 · FK · TCP bridge"
    )
    robot_conn_row.addWidget(self.robot_nodes_status_label, 1)
    self.robot_nodes_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新节点")
    self.robot_nodes_refresh_btn.setToolTip("刷新最小栈各 node 运行状态")
    self.robot_nodes_refresh_btn.clicked.connect(self._refresh_host_min_node_status)
    robot_conn_row.addWidget(self.robot_nodes_refresh_btn)
    robot_layout.addLayout(robot_conn_row)
    self._host_min_node_status_timer = _eai_platform_bootstrap.QTimer(self)
    self._host_min_node_status_timer.setInterval(2000)
    self._host_min_node_status_timer.timeout.connect(self._refresh_host_min_node_status)
    self._host_min_node_status_timer.start()
    _eai_platform_bootstrap.QTimer.singleShot(300, self._refresh_host_min_node_status)

    build.enable_row = _eai_platform_bootstrap.QHBoxLayout()
    build.enable_row.setSpacing(10)
    self.robot_arm_enable_label = _eai_platform_bootstrap.QLabel("手臂: --")
    self.robot_arm_enable_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_arm_enable_label.setToolTip(f"订阅 {_eai_configuration_policies.ARM_ENABLE_STATE_TOPIC}，1=已使能")
    self.robot_hand_enable_label = _eai_platform_bootstrap.QLabel("手: --")
    self.robot_hand_enable_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_hand_enable_label.setToolTip(f"订阅 {_eai_configuration_policies.HAND_ENABLE_STATE_TOPIC}，1=已使能")
    self.robot_control_mode_label = _eai_platform_bootstrap.QLabel("mode: --")
    self.robot_control_mode_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robot_control_mode_label.setToolTip(
        f"订阅 {_eai_configuration_policies.CONTROL_MODE_TOPIC}，回放需 mode={_eai_configuration_chat.MODEL_CONTROL_MODE}"
    )
    build.enable_row.addWidget(self.robot_arm_enable_label)
    self.robot_arm_enable_btn = _eai_platform_bootstrap.QPushButton("启用手臂")
    self.robot_arm_enable_btn.setToolTip(
        "等同踏板 F2 开启/关闭手臂控制"
    )
    self.robot_arm_enable_btn.clicked.connect(self._on_arm_enable_clicked)
    build.enable_row.addWidget(self.robot_arm_enable_btn)
    build.enable_row.addSpacing(8)
    build.enable_row.addWidget(self.robot_hand_enable_label)
    self.robot_hand_enable_btn = _eai_platform_bootstrap.QPushButton("启用手")
    self.robot_hand_enable_btn.setToolTip(
        "等同踏板 F1 开启/关闭手部控制"
    )
    self.robot_hand_enable_btn.clicked.connect(self._on_hand_enable_clicked)
    build.enable_row.addWidget(self.robot_hand_enable_btn)
    build.enable_row.addSpacing(8)
    build.enable_row.addWidget(self.robot_control_mode_label)
    self.robot_mode0_btn = _eai_platform_bootstrap.QPushButton("切 mode=0")
    self.robot_mode0_btn.setToolTip(
        f"发布 {_eai_configuration_policies.CONTROL_MODE_TOPIC}={_eai_configuration_chat.MODEL_CONTROL_MODE}（模型控制，手臂移动/回放需要）"
    )
    self.robot_mode0_btn.clicked.connect(self._on_model_mode_clicked)
    build.enable_row.addWidget(self.robot_mode0_btn)
    build.enable_row.addStretch()
    robot_layout.addLayout(build.enable_row)

    replay_row = _eai_platform_bootstrap.QHBoxLayout()
    replay_row.setSpacing(6)
    self.replay_status_label = _eai_platform_bootstrap.QLabel("回放: --")
    self.replay_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.replay_status_label.setToolTip(
        "RRD 轨迹回放状态（/rrd_replay/running_state）\n"
        "需手/臂服务栈 + control_mode=0 + 手/臂使能"
    )
    self.replay_select_btn = _eai_platform_bootstrap.QPushButton("选择路径")
    self.replay_select_btn.setToolTip(
        "选择 .rrd 回放文件。\n"
        f"默认数据目录: {_eai_adapters_robot_launch_utils.default_rrd_dataset_dir()}"
    )
    self.replay_select_btn.clicked.connect(self._on_replay_select_clicked)
    self.replay_start_btn = _eai_platform_bootstrap.QPushButton("开始")
    self.replay_start_btn.setToolTip(
        "启动 rrd_replay 并开始回放。\n"
        "先决条件：已选 .rrd + 手/臂服务栈运行中 + control_mode=0 + 手/臂已使能。"
    )
    self.replay_start_btn.clicked.connect(self._on_replay_start_clicked)
    self.replay_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.replay_stop_btn.setToolTip("停止 rrd_replay 节点与当前回放。")
    self.replay_stop_btn.setObjectName("dangerAction")
    self.replay_stop_btn.clicked.connect(self._on_replay_stop_clicked)
    replay_row.addWidget(_eai_platform_bootstrap.QLabel("次数"))
    self.replay_count_spin = _eai_platform_bootstrap.QSpinBox()
    self.replay_count_spin.setRange(1, _eai_configuration_policies.REPLAY_LOOP_COUNT_MAX)
    self.replay_count_spin.setValue(_eai_configuration_policies.REPLAY_LOOP_COUNT_DEFAULT)
    self.replay_count_spin.setFixedWidth(52)
    self.replay_count_spin.setToolTip(
        f"回放次数（1–{_eai_configuration_policies.REPLAY_LOOP_COUNT_MAX}），每次播完后自动从头再播"
    )
    replay_row.addWidget(self.replay_count_spin)
    replay_row.addWidget(self.replay_status_label, 1)
    replay_row.addWidget(self.replay_select_btn)
    replay_row.addWidget(self.replay_start_btn)
    replay_row.addWidget(self.replay_stop_btn)
    robot_layout.addLayout(replay_row)

    self._selected_rrd_path = ""
    rrd_path_row = _eai_platform_bootstrap.QHBoxLayout()
    rrd_path_row.setSpacing(6)
    rrd_path_label = _eai_platform_bootstrap.QLabel("回放文件:")
    rrd_path_label.setToolTip("当前已选择、用于回放的 .rrd 文件完整路径")
    rrd_path_row.addWidget(rrd_path_label)
    self.replay_rrd_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.replay_rrd_path_edit.setReadOnly(True)
    self.replay_rrd_path_edit.setPlaceholderText("点击「选择路径」选择 .rrd 文件")
    self.replay_rrd_path_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rrd_path_row.addWidget(self.replay_rrd_path_edit, 1)
    robot_layout.addLayout(rrd_path_row)
    build.control_tabs.addTab(robot_tab, "回放")

    self._replay_launcher = _eai_adapters_robot_launchers.RrdReplayLauncher(build.node, self)
