"""ui.builders.robometer — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_reward_launchers = module_ref('eai_app.adapters.models.reward_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.robometer_tab = _eai_platform_bootstrap.QWidget()
    build.robometer_tab.setObjectName("robometerPolicyTab")
    rm_outer = _eai_platform_bootstrap.QVBoxLayout(build.robometer_tab)
    rm_outer.setContentsMargins(8, 6, 8, 6)
    rm_outer.setSpacing(6)
    rm_hint = _eai_platform_bootstrap.QLabel(
        "RoboMeter Policy Learning：在仓库中跑 SAC/IQL/BC、DSRL 或 Pi0 评测。"
        "「图形窗口」对 eval 等有效；train/train_async（LIBERO 离屏）会强制 egl，"
        "避免与本 GUI 的 OpenGL 冲突触发 cudnn Invalid handle。"
        "默认仓库为 robometer-policy-learning；Python 优先 "
        "miniconda3/envs/robometer-policy。"
        "额外参数写在 Hydra 一行，例: env.env_name=libero_90 env.task_id=28。"
    )
    rm_hint.setWordWrap(True)
    rm_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    rm_outer.addWidget(rm_hint)

    build.rm_path_row = _eai_platform_bootstrap.QHBoxLayout()
    build.rm_path_row.setSpacing(6)
    build.rm_path_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.robometer_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ROBOMETER_ROOT", _eai_configuration_tabs.ROBOMETER_ROOT_DEFAULT)
    )
    self.robometer_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_path_row.addWidget(self.robometer_root_edit, 1)
    self.robometer_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robometer_root_browse_btn.setFixedWidth(28)
    self.robometer_root_browse_btn.clicked.connect(
        self._on_robometer_root_browse
    )
    build.rm_path_row.addWidget(self.robometer_root_browse_btn)
    build.rm_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.robometer_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_robometer_python(self.robometer_root_edit.text())
    )
    self.robometer_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_path_row.addWidget(self.robometer_python_edit, 1)
    rm_outer.addLayout(build.rm_path_row)

    rm_job_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_job_row.setSpacing(6)
    rm_job_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.robometer_job_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.job_id, default_cfg in _eai_configuration_tabs.ROBOMETER_JOBS:
        self.robometer_job_combo.addItem(build.label, (build.job_id, default_cfg))
    self.robometer_job_combo.setToolTip(
        "选择 scripts 入口；切换后会填入对应默认 Hydra config"
    )
    self.robometer_job_combo.currentIndexChanged.connect(
        self._on_robometer_job_changed
    )
    rm_job_row.addWidget(self.robometer_job_combo)
    rm_job_row.addWidget(_eai_platform_bootstrap.QLabel("config"))
    self.robometer_config_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.robometer_config_combo.setEditable(True)
    self.robometer_config_combo.setInsertPolicy(_eai_platform_bootstrap.QComboBox.NoInsert)
    self.robometer_config_combo.setMinimumWidth(180)
    self.robometer_config_combo.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_job_row.addWidget(self.robometer_config_combo, 1)
    self.robometer_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.robometer_refresh_btn.setToolTip(
        "重新扫描 robometer_policy_learning/configs/*.yaml"
    )
    self.robometer_refresh_btn.clicked.connect(self._refresh_robometer_configs)
    rm_job_row.addWidget(self.robometer_refresh_btn)
    rm_job_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.robometer_cuda_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    self.robometer_cuda_edit.setFixedWidth(72)
    self.robometer_cuda_edit.setPlaceholderText("0")
    self.robometer_cuda_edit.setToolTip("CUDA_VISIBLE_DEVICES，留空则用系统默认")
    rm_job_row.addWidget(self.robometer_cuda_edit)
    rm_outer.addLayout(rm_job_row)

    rm_opt_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_opt_row.setSpacing(6)
    self.robometer_render_gui_check = _eai_platform_bootstrap.QCheckBox("图形窗口")
    self.robometer_render_gui_check.setChecked(
        bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip())
    )
    self.robometer_render_gui_check.setToolTip(
        "对 eval 等任务：尝试弹出 MuJoCo/robosuite 窗口（MUJOCO_GL=glfw）。\n"
        "train / train_async（LIBERO OffScreenRenderEnv）会强制 egl，"
        "避免与 Qt 父进程冲突导致 cudnn 报 Invalid handle。"
    )
    rm_opt_row.addWidget(self.robometer_render_gui_check)
    self.robometer_wandb_offline_check = _eai_platform_bootstrap.QCheckBox("离线 wandb")
    self.robometer_wandb_offline_check.setChecked(True)
    self.robometer_wandb_offline_check.setToolTip(
        "追加 logging.wandb_offline=true，避免未登录 wandb 时卡住"
    )
    rm_opt_row.addWidget(self.robometer_wandb_offline_check)
    rm_opt_row.addStretch(1)
    rm_outer.addLayout(rm_opt_row)

    build.rm_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    build.rm_ckpt_row.setSpacing(6)
    build.rm_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.robometer_ckpt_edit = _eai_platform_bootstrap.QLineEdit()
    self.robometer_ckpt_edit.setPlaceholderText(
        "可选。评测写入 policy_checkpoint=；训练写入 training.load_dir="
    )
    self.robometer_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_ckpt_row.addWidget(self.robometer_ckpt_edit, 1)
    self.robometer_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robometer_ckpt_browse_btn.setFixedWidth(28)
    self.robometer_ckpt_browse_btn.clicked.connect(
        self._on_robometer_ckpt_browse
    )
    build.rm_ckpt_row.addWidget(self.robometer_ckpt_browse_btn)
    rm_outer.addLayout(build.rm_ckpt_row)

    rm_ov_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_ov_row.setSpacing(6)
    rm_ov_row.addWidget(_eai_platform_bootstrap.QLabel("Hydra"))
    self.robometer_overrides_edit = _eai_platform_bootstrap.QLineEdit()
    self.robometer_overrides_edit.setPlaceholderText(
        "可选覆盖。例: env.env_type=libero env.env_name=libero_spatial "
        "eval.num_episodes=2"
    )
    self.robometer_overrides_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_ov_row.addWidget(self.robometer_overrides_edit, 1)
    rm_outer.addLayout(rm_ov_row)

    rm_run_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_run_row.setSpacing(6)
    self.robometer_start_btn = _eai_platform_bootstrap.QPushButton("启动")
    self.robometer_start_btn.setToolTip("启动 robometer-policy-learning")
    self.robometer_start_btn.clicked.connect(self._on_robometer_start_clicked)
    rm_run_row.addWidget(self.robometer_start_btn)
    self.robometer_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.robometer_stop_btn.setObjectName("dangerAction")
    self.robometer_stop_btn.setEnabled(False)
    self.robometer_stop_btn.clicked.connect(self._on_robometer_stop_clicked)
    rm_run_row.addWidget(self.robometer_stop_btn)
    self.robometer_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.robometer_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_run_row.addWidget(self.robometer_status_label, 1)
    rm_outer.addLayout(rm_run_row)

    rm_log_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_log_row.setSpacing(6)
    rm_log_row.addWidget(_eai_platform_bootstrap.QLabel("日志目录"))
    self.robometer_log_dir_edit = _eai_platform_bootstrap.QLineEdit()
    self.robometer_log_dir_edit.setReadOnly(True)
    self.robometer_log_dir_edit.setPlaceholderText("启动后显示 outputs/")
    self.robometer_log_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_log_row.addWidget(self.robometer_log_dir_edit, 1)
    self.robometer_open_log_btn = _eai_platform_bootstrap.QPushButton("打开")
    self.robometer_open_log_btn.setEnabled(False)
    self.robometer_open_log_btn.clicked.connect(
        self._on_robometer_open_log_clicked
    )
    rm_log_row.addWidget(self.robometer_open_log_btn)
    rm_outer.addLayout(rm_log_row)

    self.robometer_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.robometer_log_edit.setReadOnly(True)
    self.robometer_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robometer_log_edit.setMinimumHeight(180)
    self.robometer_log_edit.setPlaceholderText("RoboMeter 运行日志…")
    self.robometer_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.robometer_clear_log_btn = _eai_ui_widgets_layouts._add_clear_log_button_row(
        rm_outer, self.robometer_log_edit
    )
    rm_outer.addWidget(self.robometer_log_edit, 1)

    self._robometer_launcher = _eai_adapters_models_reward_launchers.RoboMeterPolicyLauncher(self)
    self._robometer_launcher.log_line.connect(self._append_robometer_log)
    self._robometer_launcher.status_message.connect(self._on_robometer_status)
    self._robometer_launcher.running_changed.connect(self._update_robometer_ui)
    self._robometer_launcher.log_dir_ready.connect(self._on_robometer_log_dir)
    self._on_robometer_job_changed()
