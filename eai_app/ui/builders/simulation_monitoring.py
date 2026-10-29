"""ui.builders.simulation monitoring — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_simulation_mujoco_launcher = module_ref('eai_app.adapters.simulation.mujoco_launcher')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    self.sim_rynnvalue_hud_panel = _eai_platform_bootstrap.QWidget()
    _rv_panel_l = _eai_platform_bootstrap.QVBoxLayout(self.sim_rynnvalue_hud_panel)
    _rv_panel_l.setContentsMargins(0, 4, 0, 0)
    _rv_panel_l.setSpacing(4)
    _rv_row1 = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    self.sim_rynnvalue_hud_check = _eai_platform_bootstrap.QCheckBox("RynnValue Live HUD")
    self.sim_rynnvalue_hud_check.setChecked(False)
    self.sim_rynnvalue_hud_check.setToolTip(
        "按秒异步调用 RynnValue reward_server，单独弹出剩余时间 HUD 窗口。\n"
        "控件在下方工作区；LIBERO / RoboTwin / MuJoCo / Molmo / Arena / Isaac 共用。\n"
        "勾选后自动探测 :8001；「部署 RynnValue」可单独启动服务（不必先勾选）。"
    )
    _rv_row1.addWidget(self.sim_rynnvalue_hud_check)
    _rv_row1.addWidget(_eai_platform_bootstrap.QLabel("HUD刷新(s)"))
    self.sim_rynnvalue_refresh_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.sim_rynnvalue_refresh_spin.setRange(0.2, 30.0)
    self.sim_rynnvalue_refresh_spin.setSingleStep(0.5)
    self.sim_rynnvalue_refresh_spin.setDecimals(1)
    self.sim_rynnvalue_refresh_spin.setValue(1.0)
    self.sim_rynnvalue_refresh_spin.setToolTip(
        "Live HUD 墙钟刷新间隔（秒），不绑定控制步频"
    )
    _rv_row1.addWidget(self.sim_rynnvalue_refresh_spin)
    _rv_panel_l.addLayout(_rv_row1)

    _rv_row2 = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    _rv_row2.addWidget(_eai_platform_bootstrap.QLabel("RynnValue URL"))
    self.sim_rynnvalue_url_edit = _eai_platform_bootstrap.QLineEdit("http://127.0.0.1:8001")
    self.sim_rynnvalue_url_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_rynnvalue_url_edit.setToolTip(
        "reward_server 地址（POST /evaluate_batch_npy）"
    )
    # _FlowLayout.addWidget only takes the widget (no stretch like QHBoxLayout).
    self.sim_rynnvalue_url_edit.setMinimumWidth(220)
    _rv_row2.addWidget(self.sim_rynnvalue_url_edit)
    _rv_row2.addWidget(_eai_platform_bootstrap.QLabel("num_frames"))
    self.sim_rynnvalue_frames_spin = _eai_platform_bootstrap.QSpinBox()
    self.sim_rynnvalue_frames_spin.setRange(2, 64)
    self.sim_rynnvalue_frames_spin.setValue(8)
    self.sim_rynnvalue_frames_spin.setToolTip("每次打分从 ring buffer 均匀采样的帧数")
    _rv_row2.addWidget(self.sim_rynnvalue_frames_spin)
    self.sim_rynnvalue_probe_btn = _eai_platform_bootstrap.QPushButton("探测")
    self.sim_rynnvalue_probe_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_rynnvalue_probe_btn.setToolTip("检查 reward_server /health 是否就绪")
    self.sim_rynnvalue_probe_btn.clicked.connect(
        self._on_sim_rynnvalue_probe_clicked
    )
    _rv_row2.addWidget(self.sim_rynnvalue_probe_btn)
    self.sim_rynnvalue_deploy_btn = _eai_platform_bootstrap.QPushButton("部署 RynnValue")
    self.sim_rynnvalue_deploy_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_rynnvalue_deploy_btn.setToolTip(
        "启动 RynnValue reward_server（模型路径沿用「RynnValue」页配置）"
    )
    self.sim_rynnvalue_deploy_btn.clicked.connect(
        self._on_sim_rynnvalue_deploy_clicked
    )
    _rv_row2.addWidget(self.sim_rynnvalue_deploy_btn)
    self.sim_rynnvalue_stop_btn = _eai_platform_bootstrap.QPushButton("停止模型")
    self.sim_rynnvalue_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_rynnvalue_stop_btn.setObjectName("dangerAction")
    self.sim_rynnvalue_stop_btn.setToolTip(
        "停止 RynnValue reward_server / 模型进程并释放 GPU（含占用 URL 端口的外部进程）"
    )
    self.sim_rynnvalue_stop_btn.clicked.connect(
        self._on_sim_rynnvalue_stop_clicked
    )
    _rv_row2.addWidget(self.sim_rynnvalue_stop_btn)
    _rv_panel_l.addLayout(_rv_row2)

    _rv_row3 = _eai_platform_bootstrap.QHBoxLayout()
    _rv_row3.setSpacing(6)
    _rv_row3.addWidget(_eai_platform_bootstrap.QLabel("RynnValue 状态"))
    self.sim_rynnvalue_status_label = _eai_platform_bootstrap.QLabel("检测中…")
    self.sim_rynnvalue_status_label.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    self.sim_rynnvalue_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.sim_rynnvalue_status_label.setWordWrap(True)
    _rv_row3.addWidget(self.sim_rynnvalue_status_label, 1)
    _rv_panel_l.addLayout(_rv_row3)

    self.sim_rynnvalue_hud_check.toggled.connect(self._on_sim_rynnvalue_hud_toggled)
    self.sim_rynnvalue_url_edit.editingFinished.connect(
        self._on_sim_rynnvalue_url_changed
    )
    self._sim_rynnvalue_status_timer = _eai_platform_bootstrap.QTimer(self)
    self._sim_rynnvalue_status_timer.setInterval(5000)
    self._sim_rynnvalue_status_timer.timeout.connect(
        self._on_sim_rynnvalue_status_tick
    )
    self._sim_rynnvalue_status_timer.start()
    _eai_platform_bootstrap.QTimer.singleShot(400, self._on_sim_rynnvalue_status_tick)
    self._update_sim_rynnvalue_hud_controls()
    # Panel is parented into 下方工作区 (not 仿真评测 tab).
    self.sim_rynnvalue_hud_panel.setVisible(True)

    self.mujoco_run_row_widget = _eai_platform_bootstrap.QWidget()
    mj_run_row = _eai_platform_bootstrap.QHBoxLayout(self.mujoco_run_row_widget)
    mj_run_row.setContentsMargins(0, 0, 0, 0)
    mj_run_row.setSpacing(6)
    self.mujoco_start_btn = _eai_platform_bootstrap.QPushButton("启动 MuJoCo")
    self.mujoco_start_btn.setToolTip(
        "启动当前所选后端（MuJoCo / MolmoSpaces / IsaacLab-Arena / LIBERO / RoboTwin）"
    )
    self.mujoco_start_btn.clicked.connect(self._on_mujoco_unified_start_clicked)
    mj_run_row.addWidget(self.mujoco_start_btn)
    self.mujoco_viewer_only_btn = _eai_platform_bootstrap.QPushButton("只开界面")
    self.mujoco_viewer_only_btn.setToolTip(
        "只启动当前后端的仿真/图形界面，不调用策略模型推理：\n"
        "· MuJoCo：官方 MJCF viewer\n"
        "· MolmoSpaces：快速演示 --viewer（首次可能要等资源索引）\n"
        "· Psi-MolmoSpaces：按任务下拉框当前项开完整场景（非裸 MJCF）\n"
        "· Arena：zero_action + Kit 窗口\n"
        "· LIBERO：开场景窗口并保持（不连 Pi）\n"
        "· RoboTwin：SAPIEN Viewer（不加载 OpenPI）\n"
        "点「停止」关闭。"
    )
    self.mujoco_viewer_only_btn.clicked.connect(
        self._on_sim_viewer_only_clicked
    )
    mj_run_row.addWidget(self.mujoco_viewer_only_btn)
    self.mujoco_pause_btn = _eai_platform_bootstrap.QPushButton("暂停")
    self.mujoco_pause_btn.setEnabled(False)
    self.mujoco_pause_btn.setCheckable(True)
    self.mujoco_pause_btn.setToolTip(
        "暂停 / 继续当前仿真任务（写入共享目录 sim_control.json）。\n"
        "MuJoCo EGL / MolmoSpaces / Arena / LIBERO / RoboTwin / 扩展评测等\n"
        "在 step 循环中轮询；暂停时保留仿真状态。官方 MuJoCo viewer 亦可用 Space。"
    )
    self.mujoco_pause_btn.clicked.connect(self._on_sim_pause_clicked)
    mj_run_row.addWidget(self.mujoco_pause_btn)
    self.mujoco_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.mujoco_stop_btn.setObjectName("dangerAction")
    self.mujoco_stop_btn.setEnabled(False)
    self.mujoco_stop_btn.clicked.connect(self._on_mujoco_stop_clicked)
    mj_run_row.addWidget(self.mujoco_stop_btn)
    self.mujoco_open_model_btn = _eai_platform_bootstrap.QPushButton("打开模型目录")
    self.mujoco_open_model_btn.setToolTip(
        "MuJoCo：打开 model/；MolmoSpaces / Arena / LIBERO / RoboTwin：打开仓库根目录"
    )
    self.mujoco_open_model_btn.clicked.connect(self._on_mujoco_open_selected_dir)
    mj_run_row.addWidget(self.mujoco_open_model_btn)
    self.mujoco_status_label = _eai_platform_bootstrap.QLabel("空闲 · MuJoCo")
    self.mujoco_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    mj_run_row.addWidget(self.mujoco_status_label, 1)
    build.sim_header_l.addWidget(self.mujoco_run_row_widget)

    self.mujoco_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.mujoco_log_edit.setReadOnly(True)
    self.mujoco_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.mujoco_log_edit.setPlaceholderText(
        "MuJoCo / MolmoSpaces / Arena / LIBERO / RoboTwin 日志…"
    )
    self.mujoco_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    _eai_ui_widgets_layouts._configure_resizable_text_edit(
        self.mujoco_log_edit,
        default_h=140,
        tooltip="拖动上下分割条可调整日志高度",
    )
    self.mujoco_log_edit.setMinimumHeight(80)
    self.mujoco_log_edit.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    self.mujoco_log_resize_bar = None
    log_pane = _eai_platform_bootstrap.QWidget()
    self._sim_log_pane = log_pane
    log_pane_l = _eai_platform_bootstrap.QVBoxLayout(log_pane)
    log_pane_l.setContentsMargins(0, 0, 0, 0)
    log_pane_l.setSpacing(2)
    self.mujoco_clear_log_btn = _eai_ui_widgets_layouts._add_clear_log_button_row(
        log_pane_l, self.mujoco_log_edit
    )
    log_pane_l.addWidget(self.mujoco_log_edit, 1)

    self._sim_body_split = _eai_ui_widgets_layouts._always_visible_splitter(_eai_platform_bootstrap.Qt.Vertical)
    self._sim_body_split.addWidget(_eai_ui_widgets_layouts._wrap_in_vscroll(build.sim_form))
    self._sim_body_split.addWidget(log_pane)
    self._sim_body_split.setStretchFactor(0, 3)
    self._sim_body_split.setStretchFactor(1, 2)
    self._sim_body_split.setSizes([420, 180])
    build.sim_outer.addWidget(self._sim_body_split, 1)

    self._mujoco_launcher = _eai_adapters_simulation_mujoco_launcher.MuJoCoViewerLauncher(self)
    self._mujoco_launcher.log_line.connect(self._append_mujoco_log)
    self._mujoco_launcher.status_message.connect(self._on_mujoco_status)
    self._mujoco_launcher.running_changed.connect(self._update_mujoco_ui)
    self._mujoco_launcher.running_changed.connect(self._update_sim_backend_mutex_ui)
    # Separate process for openpi serve_policy (survives eval stop).
    self._pi_policy_launcher = _eai_adapters_simulation_mujoco_launcher.MuJoCoViewerLauncher(self)
    self._pi_policy_launcher.enable_persist_on_exit("pi_serve")
    self._pi_policy_launcher.log_line.connect(self._append_mujoco_log)
    self._pi_policy_launcher.status_message.connect(self._on_mujoco_status)
    self._pi_policy_launcher.running_changed.connect(
        self._update_molmospaces_pi_deploy_ui
    )
    self._pi_policy_ready_timer = _eai_platform_bootstrap.QTimer(self)
    self._pi_policy_ready_timer.setInterval(1500)
    self._pi_policy_ready_timer.timeout.connect(self._update_molmospaces_pi_deploy_ui)
    self._pi_gpu_mem_cache: _eai_platform_bootstrap.Tuple[float, _eai_platform_bootstrap.Dict[str, str]] = (0.0, {})
    # Separate process for Psi Policy / Native6D / WAM servers.
    self._psi_remote_launcher = _eai_adapters_simulation_mujoco_launcher.MuJoCoViewerLauncher(self)
    self._psi_remote_launcher.enable_persist_on_exit("psi_remote")
    self._psi_remote_launcher.log_line.connect(self._append_mujoco_log)
    self._psi_remote_launcher.status_message.connect(self._on_mujoco_status)
    self._psi_remote_launcher.running_changed.connect(
        self._update_molmospaces_psi_deploy_ui
    )
    self._psi_remote_ready_timer = _eai_platform_bootstrap.QTimer(self)
    self._psi_remote_ready_timer.setInterval(1500)
    self._psi_remote_ready_timer.timeout.connect(
        self._update_molmospaces_psi_deploy_ui
    )
    self._sim_eval_launcher.running_changed.connect(self._update_sim_backend_mutex_ui)
    self._sim_bridge_launcher.running_changed.connect(self._update_sim_backend_mutex_ui)
    self._refresh_mujoco_models(
        prefer=_eai_configuration_models.PSI_R1_MODEL_XML
    )
    self._refresh_molmospaces_tasks()
    self._refresh_arena_tasks()
    if hasattr(self, "_on_libero_mode_changed"):
        self._on_libero_mode_changed()
    if hasattr(self, "_on_libero_suite_changed"):
        self._on_libero_suite_changed()
    self._on_sim_backend_changed()
    if hasattr(self, "_pi_policy_ready_timer"):
        self._pi_policy_ready_timer.start()
        self._update_molmospaces_pi_deploy_ui()
    if hasattr(self, "_psi_remote_ready_timer"):
        self._psi_remote_ready_timer.start()
        self._update_molmospaces_psi_deploy_ui()
