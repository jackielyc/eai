"""ui.builders.workspace — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_chat_panel = module_ref('eai_app.ui.chat.panel')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    workspace = _eai_platform_bootstrap.QWidget()
    workspace.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    workspace_layout = _eai_platform_bootstrap.QVBoxLayout(workspace)
    workspace_layout.setContentsMargins(0, 0, 0, 0)
    workspace_layout.setSpacing(0)

    workspace_header = _eai_platform_bootstrap.QWidget()
    workspace_header_layout = _eai_platform_bootstrap.QHBoxLayout(workspace_header)
    workspace_header_layout.setContentsMargins(6, 4, 4, 4)
    workspace_header_layout.setSpacing(6)
    self._workspace_title_label = _eai_platform_bootstrap.QLabel("Topic / 图像预览 / Live HUD / AI 对话")
    self._workspace_title_label.setStyleSheet("font-weight: bold;")
    workspace_header_layout.addWidget(self._workspace_title_label)
    workspace_header_layout.addStretch(1)
    self._workspace_toggle_btn = _eai_platform_bootstrap.QToolButton()
    self._workspace_toggle_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self._workspace_toggle_btn.setText("折叠")
    self._workspace_toggle_btn.setToolTip(
        "折叠下方工作区（Topic、图像预览、Live HUD、AI 对话）"
    )
    self._workspace_toggle_btn.clicked.connect(self._toggle_workspace_panel)
    workspace_header_layout.addWidget(self._workspace_toggle_btn)
    workspace_layout.addWidget(workspace_header)

    workspace_body = _eai_platform_bootstrap.QWidget()
    workspace_body_layout = _eai_platform_bootstrap.QVBoxLayout(workspace_body)
    workspace_body_layout.setContentsMargins(0, 0, 0, 0)
    workspace_body_layout.setSpacing(4)

    # RynnValue Live HUD controls live in 工作区; OpenCV HUD window stays separate.
    if hasattr(self, "sim_rynnvalue_hud_panel"):
        self.sim_rynnvalue_hud_panel.setParent(workspace_body)
        workspace_body_layout.addWidget(self.sim_rynnvalue_hud_panel)

    self._main_splitter = _eai_platform_bootstrap.QSplitter(_eai_platform_bootstrap.Qt.Horizontal)

    left_group = _eai_platform_bootstrap.QGroupBox("Topic 列表")
    left_layout = _eai_platform_bootstrap.QVBoxLayout(left_group)
    self.topic_list_widget = _eai_platform_bootstrap.QWidget()
    self.topic_list_layout = _eai_platform_bootstrap.QVBoxLayout(self.topic_list_widget)
    self.topic_list_layout.setAlignment(_eai_platform_bootstrap.Qt.AlignTop)

    topic_scroll = _eai_platform_bootstrap.QScrollArea()
    topic_scroll.setWidgetResizable(True)
    topic_scroll.setWidget(self.topic_list_widget)
    topic_scroll.setMinimumWidth(220)
    topic_scroll.setMaximumWidth(300)
    left_layout.addWidget(topic_scroll)
    self._main_splitter.addWidget(left_group)

    right_group = _eai_platform_bootstrap.QGroupBox("图像预览")
    right_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    right_layout = _eai_platform_bootstrap.QVBoxLayout(right_group)
    right_layout.setContentsMargins(4, 8, 4, 4)
    self.grid_widget = _eai_platform_bootstrap.QWidget()
    self.grid_widget.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    self.grid_layout = _eai_platform_bootstrap.QGridLayout(self.grid_widget)
    self.grid_layout.setContentsMargins(0, 0, 0, 0)
    self.grid_layout.setSpacing(4)
    right_layout.addWidget(self.grid_widget, stretch=1)
    self._main_splitter.addWidget(right_group)

    chat_group = _eai_platform_bootstrap.QGroupBox("AI 对话")
    chat_layout = _eai_platform_bootstrap.QVBoxLayout(chat_group)
    chat_layout.setContentsMargins(4, 6, 4, 4)
    chat_layout.setSpacing(2)
    self.chat_panel = _eai_ui_chat_panel.ChatPanelWidget(config=build.llm_config)
    self.chat_panel.set_camera_frame_provider(self._pick_chat_camera_frame)
    self.chat_panel.attached_image_changed.connect(
        self._on_chat_attached_image_changed
    )
    self.chat_panel.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    chat_layout.addWidget(self.chat_panel)
    chat_group.setMinimumWidth(380)
    chat_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Expanding)
    self._chat_group = chat_group
    self._main_splitter.addWidget(chat_group)

    self._main_splitter.setStretchFactor(0, 0)
    self._main_splitter.setStretchFactor(1, 2)
    self._main_splitter.setStretchFactor(2, 3)
    topic_w = 240
    chat_w = max(480, int(build.win_w * 0.36))
    preview_w = max(480, build.win_w - topic_w - chat_w - 40)
    self._main_splitter.setSizes([topic_w, preview_w, chat_w])
    workspace_body_layout.addWidget(self._main_splitter, stretch=1)
    workspace_layout.addWidget(workspace_body, stretch=1)

    self._workspace_panel = workspace
    self._workspace_body = workspace_body
    self._workspace_collapsed = False
    try:
        build.root_layout.removeWidget(build.control_tabs)
    except Exception:
        pass
    self._ui_vsplitter = _eai_ui_widgets_layouts._always_visible_splitter(_eai_platform_bootstrap.Qt.Vertical)
    build.control_tabs.setMinimumHeight(200)
    build.control_tabs.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    workspace.setMinimumHeight(36)
    self._ui_vsplitter.addWidget(build.control_tabs)
    self._ui_vsplitter.addWidget(workspace)
    self._ui_vsplitter.setStretchFactor(0, 3)
    self._ui_vsplitter.setStretchFactor(1, 2)
    self._ui_vsplitter_saved_sizes: _eai_platform_bootstrap.List[int] = []
    build.root_layout.addWidget(self._ui_vsplitter, 1)
    self._toggle_workspace_panel()  # 默认折叠工作区，给上方 Tab 更多空间

    # status_bar 已在空壳阶段创建
    self.status_bar.showMessage("就绪")
    self._apply_only_tabs(self._only_tabs)
    self._restore_control_tabs_order()
    self._restore_control_misc_children()
    self._refresh_control_tab_tooltips()
    self._restore_sim_backends_order()
    self._restore_real_robot_backends_order()
    self._restore_arm_panels_order()
    self._restore_viewer_ui_state()
    self._on_arm_move_speed_changed(self.arm_move_speed_slider.value())
    self._on_mv6d_methods_changed()
    # After restore: fix Pi_05 + LingBot-only ckpt (or clear incompatible leftovers).
    self.mv6d_enable_check.blockSignals(True)
    self.mv6d_enable_check.setChecked(False)
    self.mv6d_enable_check.setEnabled(False)
    self.mv6d_enable_check.setToolTip('开发缓存模式：仅点击“估计一次”且成功后更新结果')
    self.mv6d_enable_check.blockSignals(False)
    self._mv6d_timer.stop()
    _eai_platform_bootstrap.QTimer.singleShot(0, self._restore_mv6d_cache)
    self._sanitize_sim_eval_policy_ckpt_pair()
    self._sync_sim_backend_tab_from_radios()
    self._sync_real_robot_backend_tab_from_radios()
    self._sync_arm_panel_tab_from_radios()
    self._apply_arm_panel_stack()
    self._install_scroll_on_all_tabs()
    self._control_tabs_order_guard = False
    self._sim_backend_tabs_order_guard = False
    self._real_robot_tabs_order_guard = False
    self._arm_panel_tabs_order_guard = False
    outer_bar = getattr(self, "_control_tab_bar", None)
    if outer_bar is not None:
        outer_bar.tabsReordered.connect(self._on_control_tab_moved)
        outer_bar.tabNestToMisc.connect(self._nest_control_tab_into_misc)
        outer_bar.tabPopOut.connect(self._pop_control_tab_from_misc)
    misc_bar = getattr(self, "_misc_tab_bar", None)
    if misc_bar is not None:
        misc_bar.tabsReordered.connect(self._on_control_tab_moved)
        misc_bar.tabNestToMisc.connect(self._nest_control_tab_into_misc)
        misc_bar.tabPopOut.connect(self._pop_control_tab_from_misc)
    # 不在切 tab / 启动时自动打开本机摄像头：VideoCapture 探测 /dev/video* 会卡死 UI
    self._ui_frame_paused_until = 0.0
    self.control_tabs.currentChanged.connect(self._on_control_tabs_current_changed)
    try:
        self.control_tabs.tabBar().tabBarClicked.connect(self._on_control_tab_bar_clicked)
    except Exception:
        pass
    misc_inner = getattr(self, "misc_inner_tabs", None)
    if misc_inner is not None:
        misc_inner.currentChanged.connect(self._on_control_tabs_current_changed)
        try:
            misc_inner.tabBar().tabBarClicked.connect(self._on_control_tab_bar_clicked)
        except Exception:
            pass
    self.chat_panel.status_message.connect(self.status_bar.showMessage)
    self._init_qwen_deploy_controller()
    self._stack_launcher.status_message.connect(self.status_bar.showMessage)
    self._stack_launcher.stack_status_changed.connect(
        self._on_stack_status_changed, _eai_platform_bootstrap.Qt.QueuedConnection
    )
    self._replay_launcher.status_message.connect(self.status_bar.showMessage)
    self._replay_launcher.node_active_changed.connect(self._update_replay_ui)
    self._local_ai_launcher.status_message.connect(self.status_bar.showMessage)
    self._local_ai_launcher.running_changed.connect(self._update_local_ai_deploy_btn)
    self._train_launcher.log_line.connect(self._append_train_log)
    self._train_launcher.status_message.connect(self.status_bar.showMessage)
    self._train_launcher.running_changed.connect(self._update_train_ui)
    self._cad_launcher.log_line.connect(self._append_cad_log)
    self._cad_launcher.status_message.connect(self.status_bar.showMessage)
    self._sim_bridge_launcher.status_message.connect(self.status_bar.showMessage)
    self._sim_eval_launcher.status_message.connect(self.status_bar.showMessage)
    self._cad_launcher.running_changed.connect(self._update_cad_ui)
    self._cad_launcher.mesh_ready.connect(self._on_cad_mesh_ready)
    build.bridge.replay_state_changed.connect(self._update_replay_ui)

    build.bridge.topics_updated.connect(self._on_topics_updated)
    build.bridge.frame_updated.connect(self._on_frame_updated)
    build.bridge.frame_tick.connect(self._on_frame_tick)
    build.bridge.status_message.connect(self.status_bar.showMessage)
    build.bridge.frame_stats.connect(self._on_frame_stats)
    # 最小栈（相机/手/臂）改由「连接机器人」启停，启动时不自动拉起

    # File-backed LIBERO / Live HUD preview — safe only after topic_list_layout exists.
    if hasattr(self, "_rynnvalue_workspace_hud_timer"):
        self._rynnvalue_workspace_hud_timer.start()
    self._ui_timer = _eai_platform_bootstrap.QTimer(self)
    self._ui_timer.timeout.connect(self._update_waiting_hint)
    self._ui_timer.timeout.connect(self._update_robot_enable_status_ui)
    self._ui_timer.timeout.connect(self._update_prereq_gated_buttons)
    self._ui_timer.start(2000)
    self._sam3_health_timer = _eai_platform_bootstrap.QTimer(self)
    self._sam3_health_timer.timeout.connect(self._refresh_sam3_status)
    self._sam3_health_timer.start(5000)
    self._fp_health_timer = _eai_platform_bootstrap.QTimer(self)
    self._fp_health_timer.timeout.connect(self._refresh_fp_status)
    self._fp_health_timer.start(5000)
    self._on_segment_settings_changed()
    self._on_pose_settings_changed()
    self._received_topics: _eai_platform_bootstrap.Dict[str, int] = {}
    robot, base = self._stack_launcher.get_cached_status()
    self._update_robot_stack_ui(robot, base)
    self._update_replay_ui()
    self._update_prereq_gated_buttons()
    self._update_local_ai_deploy_btn()
    self._update_train_ui()
    self._update_cad_ui()
    # After full UI is ready (arm-btn styles etc.): adopt leftover Isaac/eval.
    _eai_platform_bootstrap.QTimer.singleShot(900, self._adopt_persisted_robodojo_eval)
