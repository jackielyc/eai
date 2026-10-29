"""ui.builders.simulation molmospaces — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    sp_page = _eai_platform_bootstrap.QWidget()
    sp_page_l = _eai_platform_bootstrap.QVBoxLayout(sp_page)
    sp_page_l.setContentsMargins(0, 0, 0, 0)
    sp_page_l.setSpacing(6)
    mj_spaces_row = _eai_platform_bootstrap.QHBoxLayout()
    mj_spaces_row.setSpacing(6)
    mj_spaces_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.molmospaces_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_ROOT", _eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT)
    )
    self.molmospaces_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.molmospaces_root_edit.setToolTip(
        "MolmoSpaces 仓库路径（默认 "
        f"{_eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT}）"
    )
    mj_spaces_row.addWidget(self.molmospaces_root_edit, 1)
    self.molmospaces_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.molmospaces_root_browse_btn.setFixedWidth(28)
    self.molmospaces_root_browse_btn.clicked.connect(
        self._on_molmospaces_root_browse
    )
    mj_spaces_row.addWidget(self.molmospaces_root_browse_btn)
    mj_spaces_row.addWidget(_eai_platform_bootstrap.QLabel("动作"))
    self.molmospaces_recipe_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, recipe_id in _eai_configuration_tabs.MOLMOSPACES_RUN_RECIPES:
        self.molmospaces_recipe_combo.addItem(build.label, recipe_id)
    self.molmospaces_recipe_combo.setToolTip(
        "评测所选任务：只跑任务列表里当前这一项。"
        "快速演示: VirtualGL→NVIDIA；无 GLX 时 EGL+Tk 回退"
    )
    mj_spaces_row.addWidget(self.molmospaces_recipe_combo)
    mj_spaces_row.addWidget(_eai_platform_bootstrap.QLabel("本体"))
    self.molmospaces_robot_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, robot_id in _eai_configuration_tabs.MOLMOSPACES_ROBOTS:
        self.molmospaces_robot_combo.addItem(build.label, robot_id)
    # Default Franka Droid (pipeline --robot droid)
    idx_droid = self.molmospaces_robot_combo.findData("droid")
    if idx_droid >= 0:
        self.molmospaces_robot_combo.setCurrentIndex(idx_droid)
    self.molmospaces_robot_combo.setToolTip(
        "快速演示：Droid/Franka/RUM/YAM 跑 Pick；"
        "RBY1 自动改导航 (nav_to_obj，无 parallel IK)；"
        "Psi R1 开本地 MuJoCo。"
        "评测任务以 benchmark JSON 内机器人为准。"
    )
    mj_spaces_row.addWidget(self.molmospaces_robot_combo)
    sp_page_l.addLayout(mj_spaces_row)
    mj_task_row = _eai_platform_bootstrap.QHBoxLayout()
    mj_task_row.setSpacing(6)
    mj_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.molmospaces_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.molmospaces_task_combo.setToolTip(
        "每次只评测当前选中的一个任务。"
        "下拉项带场景缩略图，悬停可看这个任务在做什么。"
        "未下载的项在启动时会提示，不会同时跑多个任务。"
    )
    mj_task_row.addWidget(self.molmospaces_task_combo, 1)
    self.molmospaces_task_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.molmospaces_task_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_task_refresh_btn.clicked.connect(
        self._refresh_molmospaces_tasks
    )
    mj_task_row.addWidget(self.molmospaces_task_refresh_btn)
    mj_task_row.addWidget(_eai_platform_bootstrap.QLabel("策略"))
    self.molmospaces_policy_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.policy_id in _eai_configuration_tabs.MOLMOSPACES_EVAL_POLICIES:
        self.molmospaces_policy_combo.addItem(build.label, build.policy_id)
    # Default: Pi（本地已下载模型路径见下方 Pi模型）
    idx_pi = self.molmospaces_policy_combo.findData(
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    if idx_pi >= 0:
        self.molmospaces_policy_combo.setCurrentIndex(idx_pi)
    self.molmospaces_policy_combo.setToolTip(
        "Pi：下方部署 openpi serve_policy（:8080）。\n"
        "Psi Policy / Native6D / WAM：下方「Psi/WAM模型」行部署对应服务。\n"
        "Psi R1 Pick Planner / Dummy 无需外部策略服务。"
    )
    self.molmospaces_policy_combo.currentIndexChanged.connect(
        self._on_molmospaces_policy_changed
    )
    mj_task_row.addWidget(self.molmospaces_policy_combo)
    mj_task_row.addWidget(_eai_platform_bootstrap.QLabel("回合"))
    self.molmospaces_episodes_spin = _eai_platform_bootstrap.QSpinBox()
    self.molmospaces_episodes_spin.setRange(0, 10000)
    self.molmospaces_episodes_spin.setValue(1)
    self.molmospaces_episodes_spin.setSpecialValueText("全部")
    self.molmospaces_episodes_spin.setToolTip(
        "该任务最多评测的回合数。0 / 全部 = benchmark 里的全部回合。"
    )
    mj_task_row.addWidget(self.molmospaces_episodes_spin)
    mj_task_row.addWidget(_eai_platform_bootstrap.QLabel("视角"))
    self.molmospaces_viewer_cam_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, cam_id in _eai_configuration_tabs.MOLMOSPACES_VIEWER_CAMS:
        self.molmospaces_viewer_cam_combo.addItem(build.label, cam_id)
    # Default: 正面第三视角（看清机器人与操作对象整体）
    self.molmospaces_viewer_cam_combo.setCurrentIndex(0)
    self.molmospaces_viewer_cam_combo.setToolTip(
        "启动前选择 Viewer 相机：\n"
        "· 正面第三视角：跟踪机身，看清机器人与操作对象整体（默认）\n"
        "· 头部第一视角：head_camera_rgb，看清手臂与操作对象"
    )
    mj_task_row.addWidget(self.molmospaces_viewer_cam_combo)
    sp_page_l.addLayout(mj_task_row)
    (
        self.molmospaces_task_preview,
        self.molmospaces_task_thumb,
        _ms_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    _eai_configuration_models.bind_eval_task_thumb(self.molmospaces_task_combo, self.molmospaces_task_thumb)
    sp_page_l.addWidget(self.molmospaces_task_preview)
    self.molmospaces_pi_row = _eai_platform_bootstrap.QWidget()
    pi_row = _eai_platform_bootstrap.QHBoxLayout(self.molmospaces_pi_row)
    pi_row.setContentsMargins(0, 0, 0, 0)
    pi_row.setSpacing(6)
    pi_row.addWidget(_eai_platform_bootstrap.QLabel("Pi模型"))
    self.molmospaces_pi_ckpt_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.MOLMOSPACES_PI_CKPT_DEFAULT)
    self.molmospaces_pi_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.molmospaces_pi_ckpt_edit.setPlaceholderText(
        "本地 ckpt，如 models/pi05_droid_jointpos"
    )
    self.molmospaces_pi_ckpt_edit.setToolTip(
        "传给 openpi serve_policy 的 --policy.dir，并覆盖 eval --checkpoint_path。"
        f"默认优先本机已下载目录（当前 {_eai_configuration_tabs.MOLMOSPACES_PI_CKPT_DEFAULT}）。"
        "若权重仍是稀疏占位，部署时会尝试补全下载。"
    )
    pi_row.addWidget(self.molmospaces_pi_ckpt_edit, 1)
    self.molmospaces_pi_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.molmospaces_pi_ckpt_browse_btn.setFixedWidth(28)
    self.molmospaces_pi_ckpt_browse_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_pi_ckpt_browse_btn.clicked.connect(
        self._on_molmospaces_pi_ckpt_browse
    )
    pi_row.addWidget(self.molmospaces_pi_ckpt_browse_btn)
    self.molmospaces_pi_start_btn = _eai_platform_bootstrap.QPushButton("部署启动")
    self.molmospaces_pi_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_pi_start_btn.setToolTip(
        "下载/校验模型并启动 openpi serve_policy（默认 :8080）"
    )
    self.molmospaces_pi_start_btn.clicked.connect(
        self._on_molmospaces_pi_start_clicked
    )
    pi_row.addWidget(self.molmospaces_pi_start_btn)
    self.molmospaces_pi_stop_btn = _eai_platform_bootstrap.QPushButton("部署停止")
    self.molmospaces_pi_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_pi_stop_btn.setToolTip("停止 :8080 上的 Pi 策略服务")
    self.molmospaces_pi_stop_btn.clicked.connect(
        self._on_molmospaces_pi_stop_clicked
    )
    pi_row.addWidget(self.molmospaces_pi_stop_btn)
    self.molmospaces_pi_probe_btn = _eai_platform_bootstrap.QPushButton("验证")
    self.molmospaces_pi_probe_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_pi_probe_btn.setToolTip(
        "探测 ws://localhost:8080 是否可调用（不自动启动）"
    )
    self.molmospaces_pi_probe_btn.clicked.connect(
        self._on_molmospaces_pi_probe_clicked
    )
    pi_row.addWidget(self.molmospaces_pi_probe_btn)
    sp_page_l.addWidget(self.molmospaces_pi_row)
    self.molmospaces_pi_status_label = _eai_platform_bootstrap.QLabel("Pi: ○ 未部署")
    self.molmospaces_pi_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.molmospaces_pi_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.molmospaces_pi_status_label.setWordWrap(True)
    self.molmospaces_pi_status_label.setToolTip(
        f"openpi serve_policy 部署状态（默认 ws://{_eai_configuration_models.PI_SERVER_HOST_DEFAULT}:"
        f"{_eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT}）"
    )
    sp_page_l.addWidget(self.molmospaces_pi_status_label)

    self.molmospaces_psi_row = _eai_platform_bootstrap.QWidget()
    psi_row = _eai_platform_bootstrap.QHBoxLayout(self.molmospaces_psi_row)
    psi_row.setContentsMargins(0, 0, 0, 0)
    psi_row.setSpacing(6)
    self.molmospaces_psi_ckpt_label = _eai_platform_bootstrap.QLabel("Psi/WAM模型")
    psi_row.addWidget(self.molmospaces_psi_ckpt_label)
    self.molmospaces_psi_ckpt_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT)
    self.molmospaces_psi_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.molmospaces_psi_ckpt_edit.setPlaceholderText(
        "默认: psi-molospaces-maobo-step=00022000-train_loss=0.018139.pt"
    )
    _ckpt_exists = bool(
        _eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT and _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT)
    )
    self.molmospaces_psi_ckpt_edit.setToolTip(
        "Psi Policy / Native6D：serve_policy_psi_policy*.py 的 --policy.dir\n"
        "WAM Psi-R1 Remote：serve_wam82_policy 的 --checkpoint\n"
        f"默认 WAM ckpt: {_eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT or '<未设置>'}\n"
        f"本地是否存在: {'是' if _ckpt_exists else '否'}\n"
        f"默认端口 Psi={_eai_adapters_models_discovery.PSI_REMOTE_POLICY_PORT_DEFAULT} / "
        f"Native6D={_eai_adapters_models_discovery.PSI_NATIVE6D_PORT_DEFAULT} / WAM={_eai_adapters_models_discovery.WAM_PSI_R1_PORT_DEFAULT}"
    )
    psi_row.addWidget(self.molmospaces_psi_ckpt_edit, 1)
    self.molmospaces_psi_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.molmospaces_psi_ckpt_browse_btn.setFixedWidth(28)
    self.molmospaces_psi_ckpt_browse_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_psi_ckpt_browse_btn.clicked.connect(
        self._on_molmospaces_psi_ckpt_browse
    )
    psi_row.addWidget(self.molmospaces_psi_ckpt_browse_btn)
    psi_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.molmospaces_psi_gpu_spin = _eai_platform_bootstrap.QSpinBox()
    self.molmospaces_psi_gpu_spin.setRange(0, 15)
    self.molmospaces_psi_gpu_spin.setValue(int(_eai_platform_bootstrap.os.environ.get("EVAL_GPU_ID") or 0))
    self.molmospaces_psi_gpu_spin.setToolTip("CUDA_VISIBLE_DEVICES（单卡）")
    psi_row.addWidget(self.molmospaces_psi_gpu_spin)
    self.molmospaces_psi_start_btn = _eai_platform_bootstrap.QPushButton("部署启动")
    self.molmospaces_psi_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_psi_start_btn.setToolTip(
        "按当前策略启动 Psi Policy / Native6D / WAM 服务"
    )
    self.molmospaces_psi_start_btn.clicked.connect(
        self._on_molmospaces_psi_start_clicked
    )
    psi_row.addWidget(self.molmospaces_psi_start_btn)
    self.molmospaces_psi_stop_btn = _eai_platform_bootstrap.QPushButton("部署停止")
    self.molmospaces_psi_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_psi_stop_btn.setToolTip("停止本窗口拉起的 Psi/WAM 策略服务")
    self.molmospaces_psi_stop_btn.clicked.connect(
        self._on_molmospaces_psi_stop_clicked
    )
    psi_row.addWidget(self.molmospaces_psi_stop_btn)
    self.molmospaces_psi_probe_btn = _eai_platform_bootstrap.QPushButton("验证")
    self.molmospaces_psi_probe_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.molmospaces_psi_probe_btn.setToolTip("探测当前策略对应端口是否可握手")
    self.molmospaces_psi_probe_btn.clicked.connect(
        self._on_molmospaces_psi_probe_clicked
    )
    psi_row.addWidget(self.molmospaces_psi_probe_btn)
    sp_page_l.addWidget(self.molmospaces_psi_row)
    self.molmospaces_psi_status_label = _eai_platform_bootstrap.QLabel("Psi/WAM: ○ 未部署")
    self.molmospaces_psi_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.molmospaces_psi_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.molmospaces_psi_status_label.setWordWrap(True)
    self.molmospaces_psi_status_label.setToolTip(
        "Psi Policy / Native6D / WAM 服务部署状态：未部署 / 部署中 / 已就绪；"
        "含 mode、端口与 checkpoint 文件名。"
    )
    sp_page_l.addWidget(self.molmospaces_psi_status_label)

    self._on_molmospaces_policy_changed()
    self._update_molmospaces_pi_deploy_ui()
    self._update_molmospaces_psi_deploy_ui()
    sp_hint = _eai_platform_bootstrap.QLabel(
        "使用 molmospaces conda 环境；与 Isaac 评测 / MuJoCo viewer 互斥。"
        "动作选「评测所选任务」时，只启动任务下拉框里当前这一项。"
        "选 Pi / Psi Policy / Native6D / WAM 时用「部署启动/停止」管理策略服务；"
        "评测前会确认可调用。"
    )
    sp_hint.setWordWrap(True)
    sp_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    sp_page_l.addWidget(sp_hint)
    sp_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(sp_page)
