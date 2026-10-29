"""ui.builders.simulation isaac — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_cameras_launchers = module_ref('eai_app.adapters.cameras.launchers')
_eai_adapters_simulation_mujoco_launcher = module_ref('eai_app.adapters.simulation.mujoco_launcher')
_eai_adapters_simulation_robodojo = module_ref('eai_app.adapters.simulation.robodojo')
_eai_adapters_simulation_robodojo_discovery = module_ref('eai_app.adapters.simulation.robodojo_discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    isaac_page = _eai_platform_bootstrap.QWidget()
    isaac_outer = _eai_platform_bootstrap.QVBoxLayout(isaac_page)
    isaac_outer.setContentsMargins(0, 0, 0, 0)
    isaac_outer.setSpacing(6)

    sim_hint = _eai_platform_bootstrap.QLabel(
        "启动 RoboDojo 评测后，相关进程（Isaac / 策略 / 相机桥）会常驻运行，"
        "直到点击「停止评测」。同时自动开图像预览：左腕 / 头部 / 右腕"
        "（共享目录 .npy → ROS /camera/*_color；预览可直读 .npy）。"
    )
    sim_hint.setWordWrap(True)
    sim_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    isaac_outer.addWidget(sim_hint)

    sim_dir_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_dir_row.setSpacing(6)
    sim_dir_row.addWidget(_eai_platform_bootstrap.QLabel("共享目录"))
    self.sim_bridge_dir_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ISAAC_CAM_BRIDGE_DIR", _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT)
    )
    self.sim_bridge_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_bridge_dir_edit.setToolTip(
        "宿主机与 Isaac / Docker 都能访问的路径（勿用 /tmp）"
    )
    sim_dir_row.addWidget(self.sim_bridge_dir_edit, 1)
    self.sim_bridge_browse_btn = _eai_platform_bootstrap.QPushButton("选择…")
    self.sim_bridge_browse_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_bridge_browse_btn.clicked.connect(self._on_sim_bridge_browse_clicked)
    sim_dir_row.addWidget(self.sim_bridge_browse_btn)
    isaac_outer.addLayout(sim_dir_row)

    sim_ctrl_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_ctrl_row.setSpacing(6)
    sim_ctrl_row.addWidget(_eai_platform_bootstrap.QLabel("Hz"))
    self.sim_bridge_hz_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.sim_bridge_hz_spin.setRange(1.0, 60.0)
    self.sim_bridge_hz_spin.setSingleStep(1.0)
    self.sim_bridge_hz_spin.setDecimals(0)
    self.sim_bridge_hz_spin.setValue(30.0)
    self.sim_bridge_hz_spin.setFixedWidth(64)
    self.sim_bridge_hz_spin.setToolTip("桥轮询共享目录频率")
    sim_ctrl_row.addWidget(self.sim_bridge_hz_spin)
    self.sim_status_label = _eai_platform_bootstrap.QLabel("桥: 空闲")
    self.sim_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    sim_ctrl_row.addWidget(self.sim_status_label, 1)
    self.sim_isaac_status_label = _eai_platform_bootstrap.QLabel("Isaac Sim: 未检测")
    self.sim_isaac_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_isaac_status_label.setToolTip(
        "实时探测本机 Isaac Kit / eval_client / 持久化评测 session。\n"
        "关闭 EAI 不会停止 Isaac；再次打开可自动接管。"
    )
    sim_ctrl_row.addWidget(self.sim_isaac_status_label)
    self.sim_bridge_start_btn = _eai_platform_bootstrap.QPushButton("启动相机桥")
    self.sim_bridge_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_bridge_start_btn.setToolTip("运行 run_isaac_cam_bridge.sh（无 Humble 时自动进 Docker）")
    self.sim_bridge_start_btn.clicked.connect(self._on_sim_bridge_start_clicked)
    sim_ctrl_row.addWidget(self.sim_bridge_start_btn)
    self.sim_bridge_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.sim_bridge_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_bridge_stop_btn.setObjectName("dangerAction")
    self.sim_bridge_stop_btn.setEnabled(False)
    self.sim_bridge_stop_btn.clicked.connect(self._on_sim_bridge_stop_clicked)
    sim_ctrl_row.addWidget(self.sim_bridge_stop_btn)
    self.sim_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新帧状态")
    self.sim_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_refresh_btn.clicked.connect(self._refresh_sim_frame_status)
    sim_ctrl_row.addWidget(self.sim_refresh_btn)
    self.sim_clear_log_btn = _eai_platform_bootstrap.QPushButton("清空日志")
    self.sim_clear_log_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_clear_log_btn.clicked.connect(self._on_sim_clear_log_clicked)
    sim_ctrl_row.addWidget(self.sim_clear_log_btn)
    isaac_outer.addLayout(sim_ctrl_row)

    self.sim_frame_status_label = _eai_platform_bootstrap.QLabel("")
    self.sim_frame_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_frame_status_label.setWordWrap(True)
    self.sim_frame_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    isaac_outer.addWidget(self.sim_frame_status_label)

    self.sim_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.sim_log_edit.setReadOnly(True)
    self.sim_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_log_edit.setFixedHeight(160)
    self.sim_log_edit.setPlaceholderText("相机桥 / Isaac 评测日志…")
    self.sim_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    isaac_outer.addWidget(self.sim_log_edit)
    self.sim_log_resize_bar = None

    isaac_lower = _eai_platform_bootstrap.QWidget()
    isaac_lower_l = _eai_platform_bootstrap.QVBoxLayout(isaac_lower)
    isaac_lower_l.setContentsMargins(0, 0, 0, 0)
    isaac_lower_l.setSpacing(6)

    sim_run_sep = _eai_platform_bootstrap.QLabel("启动 RoboDojo 评测")
    sim_run_sep.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold;")
    isaac_lower_l.addWidget(sim_run_sep)

    sim_run_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_run_row.setSpacing(6)
    sim_run_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.sim_eval_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.sim_eval_task_combo.setMinimumWidth(180)
    self.sim_eval_task_combo.setEditable(True)
    self.sim_eval_task_combo.setToolTip(
        "选择或输入 RoboDojo 任务名（如 build_tower）。"
        "下拉项带场景缩略图，悬停可看任务在做什么。"
    )
    for task_name in _eai_adapters_simulation_robodojo_discovery.list_robodojo_tasks():
        self.sim_eval_task_combo.addItem(task_name, task_name)
    if self.sim_eval_task_combo.count() == 0:
        self.sim_eval_task_combo.addItem("build_tower", "build_tower")
    idx_bt = self.sim_eval_task_combo.findData("build_tower")
    if idx_bt >= 0:
        self.sim_eval_task_combo.setCurrentIndex(idx_bt)
    sim_run_row.addWidget(self.sim_eval_task_combo)
    sim_run_row.addWidget(_eai_platform_bootstrap.QLabel("策略"))
    self.sim_eval_policy_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.sim_eval_policy_combo.setMinimumWidth(160)
    self.sim_eval_policy_combo.setMaxVisibleItems(24)
    self.sim_eval_policy_combo.addItem("无", "")
    for build.label, rel in _eai_adapters_simulation_robodojo_discovery.list_robodojo_policies():
        self.sim_eval_policy_combo.addItem(build.label, rel)
    if self.sim_eval_policy_combo.count() <= 1:
        # 扫描失败时的兜底
        self.sim_eval_policy_combo.addItem(
            "demo_policy", "XPolicyLab/policy/demo_policy"
        )
        self.sim_eval_policy_combo.addItem(
            "starVLA", "XPolicyLab/policy/starVLA"
        )
        self.sim_eval_policy_combo.addItem(
            "LingBot_VA", _eai_configuration_models.LINGBOT_VA_POLICY_REL
        )
        self.sim_eval_policy_combo.addItem("VPP2", _eai_configuration_models.VPP2_POLICY_REL)
    # Ensure LingBot_VA / VPP2 always selectable even if scan missed them.
    if self.sim_eval_policy_combo.findData(_eai_configuration_models.LINGBOT_VA_POLICY_REL) < 0:
        self.sim_eval_policy_combo.addItem("LingBot_VA", _eai_configuration_models.LINGBOT_VA_POLICY_REL)
    if self.sim_eval_policy_combo.findData(_eai_configuration_models.VPP2_POLICY_REL) < 0:
        self.sim_eval_policy_combo.addItem("VPP2", _eai_configuration_models.VPP2_POLICY_REL)
    # 默认仍选 demo_policy（「无」= 仅启评测、不启策略服务）
    idx_demo = self.sim_eval_policy_combo.findData(
        "XPolicyLab/policy/demo_policy"
    )
    if idx_demo >= 0:
        self.sim_eval_policy_combo.setCurrentIndex(idx_demo)
    self.sim_eval_policy_combo.setToolTip(
        "XPolicyLab 策略目录（自动扫描带 deploy.py 的 policy/*）。\n"
        "选「无」：只启动评测（Isaac / eval client），不启动策略服务；\n"
        "评测侧用进程内 hold-pose / GUI 遥控推进，相机可预览。\n"
        "各后端「只开界面」后，「手臂/手」文本指挥 / 滑块均可遥控。\n"
        "各策略还需对应 conda 环境与 ckpt 才可真正推理。"
    )
    self.sim_eval_policy_combo.currentIndexChanged.connect(
        self._on_sim_eval_policy_changed
    )
    sim_run_row.addWidget(self.sim_eval_policy_combo)
    sim_run_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.sim_eval_ckpt_edit = _eai_platform_bootstrap.QLineEdit("demo")
    self.sim_eval_ckpt_edit.setMinimumWidth(320)
    self.sim_eval_ckpt_edit.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Fixed
    )
    self.sim_eval_ckpt_edit.setToolTip(
        "显示完整 checkpoint 目录绝对路径。\n"
        "starVLA：…/RoboDojo_cache/starvla_hf/pi_v3\n"
        "Pi_05：…/openpi/hf/pi05_droid_jointpos（含 params/）\n"
        "LingBot_VA：…/LingBot_VA/checkpoints/<run>；"
        "需 LINGBOT_VA_BASE_MODEL_PATH。\n"
        "VPP2：ckpt=joint2b_s100000；需 VPP2_BUNDLE / VPP2_WAN_ROOT。"
    )
    sim_run_row.addWidget(self.sim_eval_ckpt_edit, 1)
    sim_run_row.addWidget(_eai_platform_bootstrap.QLabel("action"))
    self.sim_eval_action_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.sim_eval_action_combo.addItem("ee", "ee")
    self.sim_eval_action_combo.addItem("joint", "joint")
    sim_run_row.addWidget(self.sim_eval_action_combo)
    self.sim_eval_use_gui_check = _eai_platform_bootstrap.QCheckBox("仿真界面")
    self.sim_eval_use_gui_check.setChecked(True)
    self.sim_eval_use_gui_check.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_use_gui_check.setToolTip(
        "勾选：Isaac Sim GUI（starVLA→run_gui_starvla_pi_v3.sh，"
        "LingBot_VA/其它→run_gui_eval.sh）。\n"
        "默认 VirtualGL→NVIDIA 硬件加速视口（ROBODOJO_USE_VGL=1）。\n"
        "不勾选：headless（无 Isaac 窗口；相机预览仍可通过桥接看到）。"
    )
    sim_run_row.addWidget(self.sim_eval_use_gui_check)
    self.sim_eval_bundled_check = _eai_platform_bootstrap.QCheckBox("一体启动")
    self.sim_eval_bundled_check.setChecked(False)
    self.sim_eval_bundled_check.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_bundled_check.setToolTip(
        "勾选：与旧 robodojo.sh eval 相同——评测进程内先起策略再起 Isaac，\n"
        "避免「先仿真后模型」叠峰 OOM；不必先点「部署启动」。\n"
        "不勾选（默认）：先单独「部署启动」，再「启动评测」只起 client。\n"
        "策略选「无」时此选项无效。"
    )
    sim_run_row.addWidget(self.sim_eval_bundled_check)
    self.sim_eval_run_status = _eai_platform_bootstrap.QLabel("评测: 空闲")
    self.sim_eval_run_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    sim_run_row.addWidget(self.sim_eval_run_status, 1)
    self.sim_eval_start_btn = _eai_platform_bootstrap.QPushButton("启动评测")
    self.sim_eval_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_start_btn.setToolTip(
        "启动前会检查是否已有 eval_client / Isaac Kit / 持久化 session：\n"
        "有则复用并按需热切换任务，不会重复拉起 Isaac。\n"
        "无运行进程时：按「仿真界面」选项新启动（EVAL_NUM=keep 常驻）。\n"
        "已由本窗口管理时：按钮为「切换任务」。\n"
        "未勾选「一体启动」且选了策略：须先「策略部署」匹配后再评测（只起 client）。\n"
        "勾选「一体启动」：评测内先起模型再起仿真（旧流程，利于避 OOM）。\n"
        "策略选「无」：只启评测客户端（零动作 / GUI 遥控）。\n"
        "关闭 EAI 不会停止 Isaac；相关进程一直保留，直到点击「停止评测」。"
    )
    self.sim_eval_start_btn.clicked.connect(
        lambda: self._start_sim_eval(viewer_only=False)
    )
    sim_run_row.addWidget(self.sim_eval_start_btn)
    self.sim_eval_viewer_only_btn = _eai_platform_bootstrap.QPushButton("只开界面")
    self.sim_eval_viewer_only_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_viewer_only_btn.setToolTip(
        "只启动当前仿真后端界面，不调用策略推理。\n"
        "可用「手臂/手」页文本指挥 / 滑块遥控；点「停止评测」或「停止」关闭。"
    )
    self.sim_eval_viewer_only_btn.clicked.connect(
        self._on_sim_viewer_only_clicked
    )
    sim_run_row.addWidget(self.sim_eval_viewer_only_btn)
    self.sim_eval_pause_btn = _eai_platform_bootstrap.QPushButton("暂停")
    self.sim_eval_pause_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_pause_btn.setEnabled(False)
    self.sim_eval_pause_btn.setCheckable(True)
    self.sim_eval_pause_btn.setToolTip(
        "暂停 / 继续当前仿真任务（写入共享目录 sim_control.json）。\n"
        "各后端在 step 循环中轮询；暂停时保留仿真状态，可随时继续。"
    )
    self.sim_eval_pause_btn.clicked.connect(self._on_sim_pause_clicked)
    sim_run_row.addWidget(self.sim_eval_pause_btn)
    self.sim_eval_stop_run_btn = _eai_platform_bootstrap.QPushButton("停止评测")
    self.sim_eval_stop_run_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_stop_run_btn.setObjectName("dangerAction")
    self.sim_eval_stop_run_btn.setEnabled(False)
    self.sim_eval_stop_run_btn.setToolTip(
        "停止全部仿真（本窗口进程 + 持久化 session + 系统中所有 eval_client / Isaac）。\n"
        "注意：仅关闭 EAI 窗口不会杀 Isaac；必须点此按钮才会真正停止。\n"
        "首次点击发 SIGTERM；约 2s 后自动 SIGKILL；仍残留时可再点一次强制清理。"
    )
    self.sim_eval_stop_run_btn.clicked.connect(self._on_sim_eval_stop_run_clicked)
    sim_run_row.addWidget(self.sim_eval_stop_run_btn)
    isaac_lower_l.addLayout(sim_run_row)
    (
        self.sim_eval_task_preview,
        self.sim_eval_task_thumb,
        _sim_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    _eai_configuration_models.stamp_eval_task_combo(self.sim_eval_task_combo)
    _eai_configuration_models.bind_eval_task_thumb(self.sim_eval_task_combo, self.sim_eval_task_thumb)
    isaac_lower_l.addWidget(self.sim_eval_task_preview)

    sim_policy_deploy_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_policy_deploy_row.setSpacing(6)
    sim_policy_deploy_row.addWidget(_eai_platform_bootstrap.QLabel("策略部署"))
    self.sim_policy_deploy_status = _eai_platform_bootstrap.QLabel("策略: 未部署")
    self.sim_policy_deploy_status.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_policy_deploy_status.setToolTip(
        f"独立起 robodojo.sh server（默认 {_eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT}:"
        f"{_eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT}），与「启动评测」解耦；\n"
        "可先部署模型，再多次启动/切换评测，避免每次重载权重。"
    )
    sim_policy_deploy_row.addWidget(self.sim_policy_deploy_status, 1)
    self.sim_policy_deploy_start_btn = _eai_platform_bootstrap.QPushButton("部署启动")
    self.sim_policy_deploy_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_policy_deploy_start_btn.setToolTip(
        "仅启动策略 WebSocket 服务（robodojo.sh server），不起 Isaac。\n"
        f"端口 {_eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT}；与 Pi/openpi 抢同一张 GPU 时请先停另一侧。"
    )
    self.sim_policy_deploy_start_btn.clicked.connect(
        self._on_sim_policy_deploy_start_clicked
    )
    sim_policy_deploy_row.addWidget(self.sim_policy_deploy_start_btn)
    self.sim_policy_deploy_stop_btn = _eai_platform_bootstrap.QPushButton("部署停止")
    self.sim_policy_deploy_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_policy_deploy_stop_btn.setObjectName("dangerAction")
    self.sim_policy_deploy_stop_btn.setEnabled(False)
    self.sim_policy_deploy_stop_btn.setToolTip("停止本窗口拉起的 RoboDojo 策略服务")
    self.sim_policy_deploy_stop_btn.clicked.connect(
        self._on_sim_policy_deploy_stop_clicked
    )
    sim_policy_deploy_row.addWidget(self.sim_policy_deploy_stop_btn)
    isaac_lower_l.addLayout(sim_policy_deploy_row)

    sim_eval_sep = _eai_platform_bootstrap.QLabel("评测结果（RoboDojo）")
    sim_eval_sep.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold;")
    isaac_lower_l.addWidget(sim_eval_sep)

    sim_eval_root_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_eval_root_row.setSpacing(6)
    sim_eval_root_row.addWidget(_eai_platform_bootstrap.QLabel("结果目录"))
    self.sim_eval_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ROBODOJO_EVAL_RESULT_ROOT", _eai_configuration_policies.ROBODOJO_EVAL_RESULT_ROOT_DEFAULT)
    )
    self.sim_eval_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_eval_root_edit.setToolTip("RoboDojo eval_result 根目录（task / policy / robot / run）")
    sim_eval_root_row.addWidget(self.sim_eval_root_edit, 1)
    self.sim_eval_root_browse_btn = _eai_platform_bootstrap.QPushButton("选择…")
    self.sim_eval_root_browse_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_root_browse_btn.clicked.connect(self._on_sim_eval_root_browse_clicked)
    sim_eval_root_row.addWidget(self.sim_eval_root_browse_btn)
    self.sim_eval_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.sim_eval_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_refresh_btn.clicked.connect(self._refresh_sim_eval_tree)
    sim_eval_root_row.addWidget(self.sim_eval_refresh_btn)
    isaac_lower_l.addLayout(sim_eval_root_row)

    sim_eval_filter_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_eval_filter_row.setSpacing(6)
    sim_eval_filter_row.addWidget(_eai_platform_bootstrap.QLabel("过滤"))
    self.sim_eval_filter_edit = _eai_platform_bootstrap.QLineEdit()
    self.sim_eval_filter_edit.setPlaceholderText("按任务名过滤…")
    self.sim_eval_filter_edit.textChanged.connect(self._on_sim_eval_filter_changed)
    sim_eval_filter_row.addWidget(self.sim_eval_filter_edit, 1)
    sim_eval_filter_row.addWidget(_eai_platform_bootstrap.QLabel("结果"))
    self.sim_eval_outcome_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.sim_eval_outcome_combo.addItem("全部", "")
    self.sim_eval_outcome_combo.addItem("success", "success")
    self.sim_eval_outcome_combo.addItem("failed", "fail")
    self.sim_eval_outcome_combo.setToolTip(
        "按视频文件名筛选：success 匹配含 success 的 mp4；"
        "failed 匹配含 fail 的 mp4（如 *_fail.mp4）"
    )
    self.sim_eval_outcome_combo.currentIndexChanged.connect(
        self._on_sim_eval_outcome_changed
    )
    sim_eval_filter_row.addWidget(self.sim_eval_outcome_combo)
    self.sim_eval_open_dir_btn = _eai_platform_bootstrap.QPushButton("打开目录")
    self.sim_eval_open_dir_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_open_dir_btn.clicked.connect(self._on_sim_eval_open_dir_clicked)
    sim_eval_filter_row.addWidget(self.sim_eval_open_dir_btn)
    self.sim_eval_open_video_btn = _eai_platform_bootstrap.QPushButton("播放视频")
    self.sim_eval_open_video_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_open_video_btn.setToolTip("窗口内播放选中 mp4（双击列表项亦可）")
    self.sim_eval_open_video_btn.clicked.connect(self._on_sim_eval_open_video_clicked)
    sim_eval_filter_row.addWidget(self.sim_eval_open_video_btn)
    self.sim_eval_merge_board_btn = _eai_platform_bootstrap.QPushButton("合并到榜单")
    self.sim_eval_merge_board_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_eval_merge_board_btn.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
    self.sim_eval_merge_board_btn.setToolTip(
        "将当前选中运行的 _result.json 合并进上方「业内评测榜单」，\n"
        "与实时/快照业内条目并排对比；刷新实时榜后本地行会保留。"
    )
    self.sim_eval_merge_board_btn.clicked.connect(
        self._on_sim_leaderboard_merge_clicked
    )
    sim_eval_filter_row.addWidget(self.sim_eval_merge_board_btn)
    isaac_lower_l.addLayout(sim_eval_filter_row)

    sim_eval_split = _eai_platform_bootstrap.QSplitter(_eai_platform_bootstrap.Qt.Horizontal)
    self.sim_eval_tree = _eai_platform_bootstrap.QTreeWidget()
    self.sim_eval_tree.setHeaderLabels(["任务 / 策略 / 机器人 / 运行"])
    self.sim_eval_tree.setMinimumWidth(280)
    self.sim_eval_tree.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_eval_tree.setStyleSheet(
        f"QTreeWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.sim_eval_tree.itemExpanded.connect(self._on_sim_eval_item_expanded)
    self.sim_eval_tree.currentItemChanged.connect(self._on_sim_eval_selection_changed)
    sim_eval_split.addWidget(self.sim_eval_tree)

    sim_eval_right = _eai_platform_bootstrap.QWidget()
    sim_eval_right_layout = _eai_platform_bootstrap.QVBoxLayout(sim_eval_right)
    sim_eval_right_layout.setContentsMargins(0, 0, 0, 0)
    sim_eval_right_layout.setSpacing(4)
    self.sim_eval_detail_edit = _eai_platform_bootstrap.QTextEdit()
    self.sim_eval_detail_edit.setReadOnly(True)
    self.sim_eval_detail_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_eval_detail_edit.setPlaceholderText("选中含 _result.json 的运行后显示摘要…")
    self.sim_eval_detail_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.sim_eval_detail_edit.setMinimumHeight(140)
    sim_eval_right_layout.addWidget(self.sim_eval_detail_edit, 1)
    self.sim_eval_files_list = _eai_platform_bootstrap.QListWidget()
    self.sim_eval_files_list.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_eval_files_list.setStyleSheet(
        f"QListWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.sim_eval_files_list.setMinimumHeight(60)
    self.sim_eval_files_list.setMaximumHeight(180)
    self.sim_eval_files_list.itemDoubleClicked.connect(
        lambda _item: self._on_sim_eval_open_video_clicked()
    )
    sim_eval_right_layout.addWidget(self.sim_eval_files_list)
    sim_eval_split.addWidget(sim_eval_right)
    sim_eval_split.setStretchFactor(0, 2)
    sim_eval_split.setStretchFactor(1, 3)
    isaac_lower_l.addWidget(sim_eval_split, 1)
    isaac_outer.addWidget(isaac_lower, 1)

    self._sim_bridge_launcher = _eai_adapters_cameras_launchers.IsaacCamBridgeLauncher(self)
    self._sim_bridge_launcher.log_line.connect(self._append_sim_log)
    self._sim_bridge_launcher.running_changed.connect(self._update_sim_bridge_ui)
    self._sim_eval_launcher = _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher(self)
    self._sim_eval_launcher.log_line.connect(self._append_sim_log)
    self._sim_eval_launcher.running_changed.connect(self._update_sim_eval_run_ui)
    self._robodojo_policy_launcher = _eai_adapters_simulation_mujoco_launcher.MuJoCoViewerLauncher(self)
    self._robodojo_policy_launcher.enable_persist_on_exit("robodojo_policy")
    self._robodojo_policy_launcher.log_line.connect(self._append_sim_log)
    self._robodojo_policy_launcher.status_message.connect(
        lambda m: self._append_sim_log(f"[策略] {m}")
    )
    self._robodojo_policy_launcher.running_changed.connect(
        self._on_sim_policy_deploy_running_changed
    )
    self._sim_policy_ready_timer = _eai_platform_bootstrap.QTimer(self)
    self._sim_policy_ready_timer.setInterval(1500)
    self._sim_policy_ready_timer.timeout.connect(self._update_sim_policy_deploy_ui)
    self._update_sim_policy_deploy_ui()
    self._sim_frame_timer = _eai_platform_bootstrap.QTimer(self)
    self._sim_frame_timer.setInterval(2000)
    self._sim_frame_timer.timeout.connect(self._refresh_sim_frame_status)
    self._sim_frame_timer.start()  # 持续刷新共享帧年龄 + Isaac Sim 状态
    # Adopt persisted Isaac/eval only after build_heavy_ui finishes (see end).
    # Scheduling here races _boot_tick processEvents before arm-btn styles exist.
    self._sim_npy_last_mtime: _eai_platform_bootstrap.Dict[str, float] = {}
    self._sim_npy_preview_timer = _eai_platform_bootstrap.QTimer(self)
    self._sim_npy_preview_timer.setInterval(50)  # ~20 Hz 直读共享帧
    self._sim_npy_preview_timer.timeout.connect(self._poll_sim_npy_preview)
    self._sim_preview_retry_timer = _eai_platform_bootstrap.QTimer(self)
    self._sim_preview_retry_timer.setInterval(1500)
    self._sim_preview_retry_timer.timeout.connect(self._retry_enable_sim_preview_topics)
    self._sim_preview_retry_left = 0
    self._rynnvalue_workspace_hud_timer = _eai_platform_bootstrap.QTimer(self)
    self._rynnvalue_workspace_hud_timer.setInterval(100)  # ~10 Hz overlay poll
    self._rynnvalue_workspace_hud_timer.timeout.connect(
        self._poll_rynnvalue_workspace_hud
    )
    self._rynnvalue_workspace_hud_mtime = 0.0
    # Started after workspace Topic/preview widgets exist (see __init__ tail).
    self._sim_eval_auto_bridge = False
    # 用户手动取消勾选的 topic：轮询 / ROS 刷新不得再强制勾回
    self._topic_user_unchecked: set[str] = set()
    self._refresh_sim_frame_status()
    self._refresh_sim_eval_tree()
    self._on_sim_eval_policy_changed()
    self.sim_backend_stack.addWidget(isaac_page)
