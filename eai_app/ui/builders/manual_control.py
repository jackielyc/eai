"""ui.builders.manual control — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    build.control_tab = _eai_platform_bootstrap.QWidget()
    build.control_layout = _eai_platform_bootstrap.QVBoxLayout(build.control_tab)
    build.control_layout.setContentsMargins(8, 6, 8, 6)
    build.control_layout.setSpacing(6)

    # 上半：手动控制（紧凑，不抢演示预览高度）
    build.manual_group = _eai_platform_bootstrap.QGroupBox("手动控制")
    build.manual_group.setStyleSheet(
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"margin-top: 6px; padding-top: 4px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY}; }}"
    )
    manual_layout = _eai_platform_bootstrap.QVBoxLayout(build.manual_group)
    manual_layout.setContentsMargins(5, 5, 5, 3)
    manual_layout.setSpacing(2)

    text_cmd_row = _eai_platform_bootstrap.QHBoxLayout()
    text_cmd_row.setSpacing(6)
    text_cmd_row.addWidget(_eai_platform_bootstrap.QLabel("文本指挥"))
    self.arm_text_cmd_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit()
    self.arm_text_cmd_edit.setPlaceholderText(_eai_platform_bootstrap.ARM_TEXT_CMD_PLACEHOLDER)
    self.arm_text_cmd_edit.setToolTip(_eai_platform_bootstrap.ARM_TEXT_CMD_HELP)
    self.arm_text_cmd_edit.returnPressed.connect(self._on_arm_text_command)
    text_cmd_row.addWidget(self.arm_text_cmd_edit, 1)
    self.arm_text_cmd_btn = _eai_platform_bootstrap.QPushButton("执行")
    self.arm_text_cmd_btn.setToolTip(_eai_platform_bootstrap.ARM_TEXT_CMD_HELP)
    self.arm_text_cmd_btn.clicked.connect(self._on_arm_text_command)
    text_cmd_row.addWidget(self.arm_text_cmd_btn)
    self.arm_home_btn = _eai_platform_bootstrap.QPushButton("归位")
    self.arm_home_btn.setToolTip(
        "两臂回到会话内首次记录的初始 EE 位姿（仿真写 gui_robot_cmd；"
        "需先有左右臂位姿回读）。"
    )
    self.arm_home_btn.clicked.connect(self._on_arm_home_clicked)
    text_cmd_row.addWidget(self.arm_home_btn)
    manual_layout.addLayout(text_cmd_row)

    from eai_app.ui.grasp.panel import GraspExecutionPanel
    self.grasp_execution_panel = GraspExecutionPanel(
        self.node,
        busy=lambda: (self.node.is_slow_motion_busy() or self._is_sim_arm_moving()
                      or self._sim_teleop_active() or bool(self._text_cmd_queue)),
        parent=self,
        preview_image=self._render_grasp_preview,
        robot_preview=self._show_grasp_preview_on_robot,
        simulation_display=self._display_grasp_simulation,
        open_perception=self._open_visual_6d_for_grasp,
        fill_extrinsic=self._fill_grasp_observation_extrinsic,
        ensure_feedback=self._ensure_host_fk,
        open_calibration=self._open_grasp_profile_calibration,
        wrench_feedback=self._grasp_wrench_feedback,
        ensure_wrench_monitor=self._ensure_grasp_wrench_monitor,
        save_task_data=self._save_grasp_task_data,
        load_task_data=self._load_grasp_task_data,
        verify_observation=self._verify_grasp_observation,
        scene_display=self._show_runtime_scene,
        reestimate_observation=self._request_grasp_reestimate,
        open_robot=self._on_mv6d_show_robot,
        assembly_live_frame=self._phone_task_live_frame,
    )


    _gap_tip = (
        f"手指间距离（cm）：0=全合，{_eai_configuration_chat.HAND_FINGER_GAP_MAX_CM:.0f}=全张；"
        f"步进 {_eai_configuration_chat.HAND_FINGER_GAP_STEP_CM} cm"
    )
    _left_gap_a = _eai_adapters_robot_geometry.hand_gap_cm_to_slider(_eai_configuration_chat.LEFT_HAND_GAP_A_DEFAULT_CM)
    _left_gap_b = _eai_adapters_robot_geometry.hand_gap_cm_to_slider(_eai_configuration_chat.LEFT_HAND_GAP_B_DEFAULT_CM)
    _right_gap_a = _eai_adapters_robot_geometry.hand_gap_cm_to_slider(_eai_perception_image_geometry.RIGHT_HAND_GAP_A_DEFAULT_CM)
    _right_gap_b = _eai_adapters_robot_geometry.hand_gap_cm_to_slider(_eai_perception_image_geometry.RIGHT_HAND_GAP_B_DEFAULT_CM)

    hand_row = _eai_platform_bootstrap.QHBoxLayout()
    hand_row.setSpacing(4)
    hand_row.addWidget(_eai_platform_bootstrap.QLabel("左手 A:"))
    # 属性名含 gap，避免旧 UI 状态把「角度%」误恢复成距离
    self.left_hand_gap_slider_a = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.left_hand_gap_slider_a.setRange(0, _eai_configuration_chat.HAND_FINGER_GAP_SLIDER_MAX)
    self.left_hand_gap_slider_a.setValue(_left_gap_a)
    self.left_hand_gap_slider_a.setMinimumWidth(72)
    self.left_hand_gap_slider_a.setMaximumWidth(140)
    self.left_hand_gap_slider_a.setToolTip(f"左手状态 A：{_gap_tip}")
    self.left_hand_gap_slider_a.valueChanged.connect(self._on_left_hand_sliders_changed)
    hand_row.addWidget(self.left_hand_gap_slider_a, 1)
    self.left_hand_label_a = _eai_platform_bootstrap.QLabel(_eai_adapters_robot_geometry.format_hand_gap_label(_left_gap_a))
    self.left_hand_label_a.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.left_hand_label_a.setMinimumWidth(52)
    hand_row.addWidget(self.left_hand_label_a)

    hand_row.addWidget(_eai_platform_bootstrap.QLabel("B:"))
    self.left_hand_gap_slider_b = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.left_hand_gap_slider_b.setRange(0, _eai_configuration_chat.HAND_FINGER_GAP_SLIDER_MAX)
    self.left_hand_gap_slider_b.setValue(_left_gap_b)
    self.left_hand_gap_slider_b.setMinimumWidth(72)
    self.left_hand_gap_slider_b.setMaximumWidth(140)
    self.left_hand_gap_slider_b.setToolTip(f"左手状态 B：{_gap_tip}")
    self.left_hand_gap_slider_b.valueChanged.connect(self._on_left_hand_sliders_changed)
    hand_row.addWidget(self.left_hand_gap_slider_b, 1)
    self.left_hand_label_b = _eai_platform_bootstrap.QLabel(_eai_adapters_robot_geometry.format_hand_gap_label(_left_gap_b))
    self.left_hand_label_b.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.left_hand_label_b.setMinimumWidth(52)
    hand_row.addWidget(self.left_hand_label_b)

    self.left_hand_toggle_btn = _eai_platform_bootstrap.QPushButton("切换")
    self.left_hand_toggle_btn.setToolTip("在游标 A 与 B 设定的两个指间距离之间切换")
    self.left_hand_toggle_btn.clicked.connect(self._on_left_hand_toggle)
    hand_row.addWidget(self.left_hand_toggle_btn)

    self.left_hand_apply_btn = _eai_platform_bootstrap.QPushButton("应用")
    self.left_hand_apply_btn.setToolTip("将主游标（当前激活状态 A 或 B）的指间距离立即发送")
    self.left_hand_apply_btn.clicked.connect(self._on_left_hand_apply_active)
    hand_row.addWidget(self.left_hand_apply_btn)
    manual_layout.addLayout(hand_row)

    right_hand_row = _eai_platform_bootstrap.QHBoxLayout()
    right_hand_row.setSpacing(4)
    right_hand_row.addWidget(_eai_platform_bootstrap.QLabel("右手 A:"))
    self.right_hand_gap_slider_a = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.right_hand_gap_slider_a.setRange(0, _eai_configuration_chat.HAND_FINGER_GAP_SLIDER_MAX)
    self.right_hand_gap_slider_a.setValue(_right_gap_a)
    self.right_hand_gap_slider_a.setMinimumWidth(72)
    self.right_hand_gap_slider_a.setMaximumWidth(140)
    self.right_hand_gap_slider_a.setToolTip(f"右手状态 A：{_gap_tip}")
    self.right_hand_gap_slider_a.valueChanged.connect(self._on_right_hand_sliders_changed)
    right_hand_row.addWidget(self.right_hand_gap_slider_a, 1)
    self.right_hand_label_a = _eai_platform_bootstrap.QLabel(_eai_adapters_robot_geometry.format_hand_gap_label(_right_gap_a))
    self.right_hand_label_a.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.right_hand_label_a.setMinimumWidth(52)
    right_hand_row.addWidget(self.right_hand_label_a)

    right_hand_row.addWidget(_eai_platform_bootstrap.QLabel("B:"))
    self.right_hand_gap_slider_b = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.right_hand_gap_slider_b.setRange(0, _eai_configuration_chat.HAND_FINGER_GAP_SLIDER_MAX)
    self.right_hand_gap_slider_b.setValue(_right_gap_b)
    self.right_hand_gap_slider_b.setMinimumWidth(72)
    self.right_hand_gap_slider_b.setMaximumWidth(140)
    self.right_hand_gap_slider_b.setToolTip(f"右手状态 B：{_gap_tip}")
    self.right_hand_gap_slider_b.valueChanged.connect(self._on_right_hand_sliders_changed)
    right_hand_row.addWidget(self.right_hand_gap_slider_b, 1)
    self.right_hand_label_b = _eai_platform_bootstrap.QLabel(_eai_adapters_robot_geometry.format_hand_gap_label(_right_gap_b))
    self.right_hand_label_b.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.right_hand_label_b.setMinimumWidth(52)
    right_hand_row.addWidget(self.right_hand_label_b)

    self.right_hand_toggle_btn = _eai_platform_bootstrap.QPushButton("切换")
    self.right_hand_toggle_btn.setToolTip("在游标 A 与 B 设定的两个指间距离之间切换")
    self.right_hand_toggle_btn.clicked.connect(self._on_right_hand_toggle)
    right_hand_row.addWidget(self.right_hand_toggle_btn)

    self.right_hand_apply_btn = _eai_platform_bootstrap.QPushButton("应用")
    self.right_hand_apply_btn.setToolTip("将主游标（当前激活状态 A 或 B）的指间距离立即发送")
    self.right_hand_apply_btn.clicked.connect(self._on_right_hand_apply_active)
    right_hand_row.addWidget(self.right_hand_apply_btn)
    manual_layout.addLayout(right_hand_row)
    self._manual_side_rows = {
        "left": hand_row,
        "right": right_hand_row,
    }

    arm_conn_row = _eai_platform_bootstrap.QHBoxLayout()
    arm_conn_row.setSpacing(6)
    self.arm_robot_state_connect_btn = _eai_platform_bootstrap.QPushButton("连接机器人")
    self.arm_robot_state_connect_btn.setToolTip(
        "启动最小宿主机节点：相机 + 手 + 臂（看图/控手/控臂）。\n"
        "再点「取消连接」停止这些节点。"
    )
    self.arm_robot_state_connect_btn.clicked.connect(self._on_connect_robot_state_clicked)
    self.arm_robot_state_connect_label = _eai_platform_bootstrap.QLabel("状态: 未连接")
    self.arm_robot_state_connect_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_robot_state_connect_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    arm_conn_row.addWidget(self.arm_robot_state_connect_btn)
    arm_conn_row.addWidget(self.arm_robot_state_connect_label)
    self.arm_nodes_status_label = _eai_platform_bootstrap.QLabel("节点: —")
    self.arm_nodes_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_nodes_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.arm_nodes_status_label.setWordWrap(False)
    self.arm_nodes_status_label.setToolTip(
        "最小栈摘要。鼠标悬停可看每个节点是否在运行。"
    )
    arm_conn_row.addWidget(self.arm_nodes_status_label, 1)

    build.enable_row = _eai_platform_bootstrap.QHBoxLayout()
    build.enable_row.setSpacing(6)
    self.hand_enable_label = _eai_platform_bootstrap.QLabel("手: --")
    self.hand_enable_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.hand_enable_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
    self.hand_enable_btn = _eai_platform_bootstrap.QPushButton("启用手")
    self.hand_enable_btn.setToolTip(
        "等同踏板 F1 开启手部控制（F1 为开关，按一次开、再按一次关）"
    )
    self.hand_enable_btn.clicked.connect(self._on_hand_enable_clicked)
    build.enable_row.addWidget(self.hand_enable_label)
    build.enable_row.addWidget(self.hand_enable_btn)
    build.enable_row.addSpacing(12)
    self.arm_enable_label = _eai_platform_bootstrap.QLabel("手臂: --")
    self.arm_enable_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_enable_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
    self.arm_enable_btn = _eai_platform_bootstrap.QPushButton("启用手臂")
    self.arm_enable_btn.setToolTip(
        "等同踏板 F2 开启手臂控制（F2 为开关，按一次开、再按一次关）"
    )
    self.arm_enable_btn.clicked.connect(self._on_arm_enable_clicked)
    build.enable_row.addWidget(self.arm_enable_label)
    build.enable_row.addWidget(self.arm_enable_btn)
    build.enable_row.addSpacing(12)
    self.control_mode_label = _eai_platform_bootstrap.QLabel("mode: --")
    self.control_mode_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
    self.control_mode_label.setToolTip(
        f"当前 {_eai_configuration_policies.CONTROL_MODE_TOPIC}，模型控制/回放需为 {_eai_configuration_chat.MODEL_CONTROL_MODE}"
    )
    build.enable_row.addWidget(self.control_mode_label)
    self.mode0_btn = _eai_platform_bootstrap.QPushButton("切 mode=0")
    self.mode0_btn.setToolTip(
        f"发布 {_eai_configuration_policies.CONTROL_MODE_TOPIC}={_eai_configuration_chat.MODEL_CONTROL_MODE}（模型控制，手臂移动/回放需要）"
    )
    self.mode0_btn.clicked.connect(self._on_model_mode_clicked)
    build.enable_row.addWidget(self.mode0_btn)
    build.enable_row.addStretch()

    arm_row1 = _eai_platform_bootstrap.QHBoxLayout()
    arm_row1.setSpacing(6)
    arm_row1.addWidget(_eai_platform_bootstrap.QLabel("移动速度"))
    self.arm_move_speed_slider = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
    self.arm_move_speed_slider.setRange(10, 100)
    default_speed_slider = _eai_adapters_robot_geometry.arm_move_speed_to_slider(_eai_configuration_chat.ARM_MOVE_SPEED_DEFAULT_RAD_S)
    self.arm_move_speed_slider.setValue(default_speed_slider)
    self.arm_move_speed_slider.setMinimumWidth(100)
    self.arm_move_speed_slider.setMaximumWidth(160)
    self.arm_move_speed_slider.setToolTip(
        f"关节空间最大角速度：{_eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S:.2f}~"
        f"{_eai_configuration_chat.ARM_MOVE_SPEED_MAX_RAD_S:.2f} rad/s（左慢右快）"
    )
    self.arm_move_speed_slider.valueChanged.connect(self._on_arm_move_speed_changed)
    arm_row1.addWidget(self.arm_move_speed_slider, 1)
    self.arm_move_speed_label = _eai_platform_bootstrap.QLabel(_eai_adapters_robot_geometry.format_arm_move_speed_label(default_speed_slider))
    self.arm_move_speed_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_move_speed_label.setMinimumWidth(72)
    arm_row1.addWidget(self.arm_move_speed_label)

    self.left_arm_move_btn = _eai_platform_bootstrap.QPushButton("左臂移动")
    self.left_arm_move_btn.setEnabled(False)
    self.left_arm_move_btn.setToolTip(
        "将左臂 TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
        "目标可为分割位姿，或相对当前位置的手动偏移。\n"
        "须先「启用手臂」；关闭使能后本按钮不可用。"
    )
    self.left_arm_move_btn.clicked.connect(
        lambda: self._on_arm_move_clicked("left")
    )

    self.right_arm_move_btn = _eai_platform_bootstrap.QPushButton("右臂移动")
    self.right_arm_move_btn.setEnabled(False)
    self.right_arm_move_btn.setToolTip(
        "将右臂 TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
        "目标可为分割位姿，或相对当前位置的手动偏移。\n"
        "须先「启用手臂」；关闭使能后本按钮不可用。"
    )
    self.right_arm_move_btn.clicked.connect(
        lambda: self._on_arm_move_clicked("right")
    )
    arm_row1.addSpacing(8)
    arm_row1.addWidget(_eai_platform_bootstrap.QLabel("间隔"))
    self.arm_repeat_interval_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.arm_repeat_interval_spin.setRange(0.0, 600.0)
    self.arm_repeat_interval_spin.setDecimals(1)
    self.arm_repeat_interval_spin.setSingleStep(0.5)
    self.arm_repeat_interval_spin.setValue(2.0)
    self.arm_repeat_interval_spin.setSuffix(" s")
    self.arm_repeat_interval_spin.setFixedWidth(88)
    self.arm_repeat_interval_spin.setToolTip(
        "每一轮两臂回到界面启动时记下的初始关节位置后，再等待的秒数，然后开始下一轮"
    )
    arm_row1.addWidget(self.arm_repeat_interval_spin)
    self.arm_repeat_btn = _eai_platform_bootstrap.QPushButton("重复执行")
    self.arm_repeat_btn.setToolTip(
        "左右臂按当前目标各移动一次，再回到界面启动时记下的初始关节位置，然后按间隔重复。\n"
        "回初始按上次下发的关节角插值，不依赖尚未更新的关节反馈。\n"
        "执行中再点一次取消。"
    )
    self.arm_repeat_btn.clicked.connect(self._on_arm_repeat_clicked)
    arm_row1.addWidget(self.arm_repeat_btn)
    # Init early: sim eval running_changed may fire via processEvents mid-build.
    self._left_arm_move_btn_idle_style = ""
    self._left_arm_move_btn_cancel_style = f"color: {_eai_configuration_policies.UI_ACCENT_RED};"
    self._right_arm_move_btn_idle_style = ""
    self._right_arm_move_btn_cancel_style = f"color: {_eai_configuration_policies.UI_ACCENT_RED};"
    arm_row1.addStretch(1)

    pose_info_box = _eai_platform_bootstrap.QHBoxLayout()
    pose_info_box.setSpacing(12)
    self.arm_pose_current_label = _eai_platform_bootstrap.QLabel("当前 TCP: --")
    self.arm_pose_current_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_pose_current_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
    self.arm_pose_current_label.setWordWrap(False)
    self.arm_pose_current_label.setMaximumHeight(24)
    self.arm_pose_target_label = _eai_platform_bootstrap.QLabel("目标 TCP: --")
    self.arm_pose_target_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arm_pose_target_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
    self.arm_pose_target_label.setWordWrap(False)
    self.arm_pose_target_label.setMaximumHeight(24)
    pose_info_box.addWidget(self.arm_pose_current_label, 1)
    pose_info_box.addWidget(self.arm_pose_target_label, 1)

    target_row = _eai_platform_bootstrap.QHBoxLayout()
    target_row.setSpacing(6)
    self.move_target_relative_radio = _eai_platform_bootstrap.QRadioButton("相对当前")
    self.move_target_segment_radio = _eai_platform_bootstrap.QRadioButton("分割位姿")
    self.move_target_relative_radio.setChecked(True)
    self.move_target_segment_radio.setEnabled(False)
    self._move_target_group = _eai_platform_bootstrap.QButtonGroup(self)
    self._move_target_group.addButton(self.move_target_relative_radio)
    self._move_target_group.addButton(self.move_target_segment_radio)
    self.move_target_relative_radio.setToolTip(
        "目标 = 当前 TCP 位置 + 偏移（base_link：X前/后，Y左/右，Z上/下）"
    )
    self.move_target_segment_radio.setToolTip(
        "使用分割、FoundationPose 或多视角 6D 得到的绝对位置作为手臂目标（base_link）。"
        "姿态保持当前 TCP。位姿还在相机系、变不到 base_link 时此项不可选。"
    )
    target_row.addWidget(self.move_target_relative_radio)
    target_row.addWidget(self.move_target_segment_radio)
    target_row.addStretch(1)
    self._move_target_group.buttonToggled.connect(self._on_move_target_params_changed)
    manual_system_row = _eai_platform_bootstrap.QHBoxLayout()
    manual_system_row.setSpacing(10)
    manual_system_row.addLayout(arm_conn_row, 2)
    manual_system_row.addLayout(build.enable_row, 2)
    manual_system_row.addLayout(arm_row1, 2)
    manual_layout.addLayout(manual_system_row)

    manual_target_info_row = _eai_platform_bootstrap.QHBoxLayout()
    manual_target_info_row.setSpacing(10)
    manual_target_info_row.addLayout(target_row)
    manual_target_info_row.addLayout(pose_info_box, 1)
    manual_layout.addLayout(manual_target_info_row)

    # 兼容旧逻辑：隐藏的 signed ΔXYZ（由六向输入框同步，供组合「臂移动」/文本指令）
    self.offset_x_label = _eai_platform_bootstrap.QLabel("ΔX")
    self.offset_y_label = _eai_platform_bootstrap.QLabel("ΔY")
    self.offset_z_label = _eai_platform_bootstrap.QLabel("ΔZ")
    self.offset_x_spin = _eai_perception_pose_estimation.make_manual_offset_spinbox()
    self.offset_y_spin = _eai_perception_pose_estimation.make_manual_offset_spinbox()
    self.offset_z_spin = _eai_perception_pose_estimation.make_manual_offset_spinbox()
    self.offset_x_spin.setValue(0.0)
    for w in (
        self.offset_x_label,
        self.offset_y_label,
        self.offset_z_label,
        self.offset_x_spin,
        self.offset_y_spin,
        self.offset_z_spin,
    ):
        w.setVisible(False)
        w.setMaximumHeight(0)

    # 每侧手和手臂共用一行；页面过宽时由手动控制 Tab 的横向滚动展示。
    self._arm_dir_spins: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QDoubleSpinBox]] = {
        "left": {},
        "right": {},
    }
    self._arm_dir_btns: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QPushButton]] = {
        "left": {},
        "right": {},
    }
    self._arm_dir_labels: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QLabel]] = {
        "left": {},
        "right": {},
    }
    self._arm_dir_row_titles: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QLabel] = {}
    self._arm_dir_extra_widgets: _eai_platform_bootstrap.List[_eai_platform_bootstrap.QWidget] = []
    for arm_side, side_label in (("left", "左臂"), ("right", "右臂")):
        dir_row = self._manual_side_rows[arm_side]
        dir_row.addSpacing(12)
        build.title = _eai_platform_bootstrap.QLabel(f"{side_label}:")
        build.title.setMinimumWidth(36)
        build.title.setToolTip(
            f"{side_label}：六个方向各自输入步长(cm)，点后方「移动」只沿该方向；"
            f"「{side_label}移动」按六向净合成偏移。"
        )
        self._arm_dir_row_titles[arm_side] = build.title
        dir_row.addWidget(build.title)
        for direction in _eai_perception_pose_estimation.ARM_DIR_ORDER:
            cn = _eai_perception_pose_estimation.ARM_DIR_LABEL_CN[direction]
            build.lab = _eai_platform_bootstrap.QLabel(cn)
            build.lab.setMinimumWidth(14)
            build.spin = _eai_perception_pose_estimation.make_positive_offset_spinbox(0.0)
            build.spin.setFixedWidth(72)
            build.spin.setToolTip(
                f"{side_label}向「{cn}」的步长（cm，base_link）"
            )
            build.spin.valueChanged.connect(self._on_arm_dir_spin_changed)
            build.btn = _eai_platform_bootstrap.QPushButton("移动")
            build.btn.setFixedWidth(40)
            build.btn.setToolTip(
                f"{side_label}只向「{cn}」移动（使用左侧输入框步长）"
            )
            build.btn.clicked.connect(
                lambda _=False, s=arm_side, d=direction: self._on_arm_direction_move(
                    s, d
                )
            )
            dir_row.addWidget(build.lab)
            dir_row.addWidget(build.spin)
            dir_row.addWidget(build.btn)
            self._arm_dir_labels[arm_side][direction] = build.lab
            self._arm_dir_spins[arm_side][direction] = build.spin
            self._arm_dir_btns[arm_side][direction] = build.btn
        move_btn = (
            self.left_arm_move_btn if arm_side == "left" else self.right_arm_move_btn
        )
        dir_row.addWidget(move_btn)
        dir_row.addStretch(1)
    self._sync_offset_spins_from_dir("left")
    self._update_move_offset_ui_visibility()
    for button in build.manual_group.findChildren(_eai_platform_bootstrap.QPushButton):
        button.setMaximumHeight(28)
    for build.spin in (build.manual_group.findChildren(_eai_platform_bootstrap.QSpinBox)
                 + build.manual_group.findChildren(_eai_platform_bootstrap.QDoubleSpinBox)):
        build.spin.setMaximumHeight(27)
    build.manual_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Maximum)
    build.manual_group.setMinimumHeight(0)
    # 勿过矮：节点摘要 + 手/臂/六向输入需完整可见；高度由 splitter 分配
    manual_layout.setSizeConstraint(_eai_platform_bootstrap.QLayout.SetMinimumSize)

    # —— 功能子面板选择条（类似仿真后端；顺序 → arm_panels_order.json）——
    build.arm_panel_row = _eai_platform_bootstrap.QHBoxLayout()
    build.arm_panel_row.setSpacing(8)
    build.arm_panel_row.addWidget(_eai_platform_bootstrap.QLabel("功能面板"))
    self.arm_panel_tab_bar = _eai_platform_bootstrap.QTabBar()
    self.arm_panel_tab_bar.setObjectName("armPanelTabs")
    self.arm_panel_tab_bar.setMovable(True)
    self.arm_panel_tab_bar.setExpanding(False)
    self.arm_panel_tab_bar.setDocumentMode(True)
    self.arm_panel_tab_bar.setUsesScrollButtons(True)
    self.arm_panel_tab_bar.setDrawBase(False)
    self.arm_panel_tab_bar.setElideMode(_eai_platform_bootstrap.Qt.ElideRight)
    self.arm_panel_tab_bar.setToolTip(
        "可拖动调整「人类演示模仿 / 多视角6D位姿」顺序；下次启动按此顺序打开"
        "（写入 eai/arm_panels_order.json）"
    )
    self.arm_panel_human_demo_radio = _eai_platform_bootstrap.QRadioButton("人类演示模仿")
    self.arm_panel_mv6d_radio = _eai_platform_bootstrap.QRadioButton("多视角6D位姿")
    self.arm_panel_human_demo_radio.setChecked(True)
    self._arm_panel_group = _eai_platform_bootstrap.QButtonGroup(self)
    self._arm_panel_group.addButton(self.arm_panel_human_demo_radio, 0)
    self._arm_panel_group.addButton(self.arm_panel_mv6d_radio, 1)
    self._arm_panel_radio_by_key = {
        key: getattr(self, attr)
        for key, _label, attr in _eai_configuration_tabs.ARM_PANEL_SELECTOR_SPECS
    }
    self._arm_panel_tabs_guard = False
    self._arm_panel_tabs_order_guard = False
    for build.radio in self._arm_panel_radio_by_key.values():
        build.radio.setVisible(False)
        build.radio.toggled.connect(self._on_arm_panel_changed)
    self.arm_panel_tab_bar.currentChanged.connect(self._on_arm_panel_tab_changed)
    self.arm_panel_tab_bar.tabMoved.connect(self._on_arm_panel_tab_moved)
    self._arm_panel_tabs_guard = True
    self._arm_panel_tabs_order_guard = True
    try:
        build.spec_by_key = {
            k: (label, attr) for k, label, attr in _eai_configuration_tabs.ARM_PANEL_SELECTOR_SPECS
        }
        for build.key in _eai_infrastructure_ui_storage.preferred_arm_panels_order():
            build.pair = build.spec_by_key.get(build.key)
            if build.pair is None:
                continue
            build.label, build._attr = build.pair
            build.idx = self.arm_panel_tab_bar.addTab(build.label)
            self.arm_panel_tab_bar.setTabData(build.idx, build.key)
    finally:
        self._arm_panel_tabs_guard = False
        self._arm_panel_tabs_order_guard = False
    self._sync_arm_panel_tab_from_radios()
    build.arm_panel_row.addWidget(self.arm_panel_tab_bar, 1)

    self._arm_panel_stack = _eai_platform_bootstrap.QStackedWidget()
    self._arm_panel_stack.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    self._arm_panel_page_by_key: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QWidget] = {}
