"""ui.builders.online rl — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_reward_launchers = module_ref('eai_app.adapters.models.reward_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.dojo_rl_tab = _eai_platform_bootstrap.QWidget()
    build.dojo_rl_tab.setObjectName("robodojoOnlineRlTab")
    dojo_rl_outer = _eai_platform_bootstrap.QVBoxLayout(build.dojo_rl_tab)
    dojo_rl_outer.setContentsMargins(8, 6, 8, 6)
    dojo_rl_outer.setSpacing(6)
    dojo_rl_hint = _eai_platform_bootstrap.QLabel(
        "RoboDojo · RISE 在线强化学习（基于 RLinf policy_online）："
        "在想象环境中用动力学 + 奖励模型做 policy improvement。"
        "默认 config=rl_release；开训前请填好 rollout.model_dir 与 "
        "reward_model_ckpt。单卡会自动压 placement 到 0-0。"
        "详见 RISE docs/online_training.md。"
    )
    dojo_rl_hint.setWordWrap(True)
    dojo_rl_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    dojo_rl_outer.addWidget(dojo_rl_hint)

    drl_path_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_path_row.setSpacing(6)
    drl_path_row.addWidget(_eai_platform_bootstrap.QLabel("RISE"))
    self.dojo_rl_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("RISE_ROOT", _eai_configuration_policies.RISE_ROOT_DEFAULT)
    )
    self.dojo_rl_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_path_row.addWidget(self.dojo_rl_root_edit, 1)
    self.dojo_rl_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.dojo_rl_root_browse_btn.setFixedWidth(28)
    self.dojo_rl_root_browse_btn.clicked.connect(self._on_dojo_rl_root_browse)
    drl_path_row.addWidget(self.dojo_rl_root_browse_btn)
    drl_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.dojo_rl_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_rise_python(self.dojo_rl_root_edit.text())
    )
    self.dojo_rl_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_path_row.addWidget(self.dojo_rl_python_edit, 1)
    dojo_rl_outer.addLayout(drl_path_row)

    drl_mode_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_mode_row.setSpacing(6)
    drl_mode_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.dojo_rl_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.dojo_rl_mode_combo.addItem("单机", "single")
    self.dojo_rl_mode_combo.addItem("多机 Ray", "multi")
    drl_mode_row.addWidget(self.dojo_rl_mode_combo)
    drl_mode_row.addWidget(_eai_platform_bootstrap.QLabel("config"))
    self.dojo_rl_config_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.dojo_rl_config_combo.setEditable(True)
    self.dojo_rl_config_combo.setInsertPolicy(_eai_platform_bootstrap.QComboBox.NoInsert)
    self.dojo_rl_config_combo.setMinimumWidth(180)
    self.dojo_rl_config_combo.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_mode_row.addWidget(self.dojo_rl_config_combo, 1)
    self.dojo_rl_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.dojo_rl_refresh_btn.setToolTip(
        "重新扫描 policy_online/examples/embodiment/config"
    )
    self.dojo_rl_refresh_btn.clicked.connect(self._refresh_dojo_rl_configs)
    drl_mode_row.addWidget(self.dojo_rl_refresh_btn)
    drl_mode_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.dojo_rl_cuda_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    self.dojo_rl_cuda_edit.setFixedWidth(72)
    self.dojo_rl_cuda_edit.setPlaceholderText("0")
    self.dojo_rl_cuda_edit.setToolTip("CUDA_VISIBLE_DEVICES，留空则用系统默认")
    drl_mode_row.addWidget(self.dojo_rl_cuda_edit)
    self.dojo_rl_auto_gpu_check = _eai_platform_bootstrap.QCheckBox("单卡自适应")
    self.dojo_rl_auto_gpu_check.setChecked(True)
    self.dojo_rl_auto_gpu_check.setToolTip(
        "可见 GPU=1 时自动追加 placement=0-0 并缩小 batch"
    )
    drl_mode_row.addWidget(self.dojo_rl_auto_gpu_check)
    dojo_rl_outer.addLayout(drl_mode_row)

    drl_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_ckpt_row.setSpacing(6)
    drl_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("策略 ckpt"))
    _default_policy_ckpt = _eai_adapters_models_discovery.resolve_rise_policy_ckpt()
    self.dojo_rl_model_dir_edit = _eai_platform_bootstrap.QLineEdit(_default_policy_ckpt)
    self.dojo_rl_model_dir_edit.setPlaceholderText(
        "rollout.model_dir（IL 预训练权重目录，需含 model.safetensors）"
    )
    self.dojo_rl_model_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.dojo_rl_model_dir_edit.setToolTip(
        "默认扫描本机 pi05_droid_jointpos_polaris 等目录；"
        "也可设环境变量 RISE_POLICY_CKPT"
    )
    drl_ckpt_row.addWidget(self.dojo_rl_model_dir_edit, 1)
    self.dojo_rl_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.dojo_rl_model_browse_btn.setFixedWidth(28)
    self.dojo_rl_model_browse_btn.clicked.connect(self._on_dojo_rl_model_browse)
    drl_ckpt_row.addWidget(self.dojo_rl_model_browse_btn)
    dojo_rl_outer.addLayout(drl_ckpt_row)

    drl_rew_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_rew_row.setSpacing(6)
    drl_rew_row.addWidget(_eai_platform_bootstrap.QLabel("奖励 ckpt"))
    build._default_reward_ckpt = _eai_adapters_models_discovery.resolve_rise_reward_ckpt(
        policy_ckpt=_default_policy_ckpt
    )
    self.dojo_rl_reward_ckpt_edit = _eai_platform_bootstrap.QLineEdit(build._default_reward_ckpt)
    self.dojo_rl_reward_ckpt_edit.setPlaceholderText(
        "actor.model.openpi.reward_model_ckpt（无独立 value 时回退策略 ckpt）"
    )
    self.dojo_rl_reward_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.dojo_rl_reward_ckpt_edit.setToolTip(
        "优先独立 value_release；本机未找到时回退到策略 ckpt。"
        "环境变量 RISE_REWARD_CKPT 可覆盖"
    )
    drl_rew_row.addWidget(self.dojo_rl_reward_ckpt_edit, 1)
    self.dojo_rl_reward_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.dojo_rl_reward_browse_btn.setFixedWidth(28)
    self.dojo_rl_reward_browse_btn.clicked.connect(self._on_dojo_rl_reward_browse)
    drl_rew_row.addWidget(self.dojo_rl_reward_browse_btn)
    dojo_rl_outer.addLayout(drl_rew_row)

    drl_resume_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_resume_row.setSpacing(6)
    drl_resume_row.addWidget(_eai_platform_bootstrap.QLabel("resume"))
    self.dojo_rl_resume_edit = _eai_platform_bootstrap.QLineEdit()
    self.dojo_rl_resume_edit.setPlaceholderText(
        "可选 runner.resume_dir（断点续训）"
    )
    self.dojo_rl_resume_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_resume_row.addWidget(self.dojo_rl_resume_edit, 1)
    self.dojo_rl_resume_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.dojo_rl_resume_browse_btn.setFixedWidth(28)
    self.dojo_rl_resume_browse_btn.clicked.connect(self._on_dojo_rl_resume_browse)
    drl_resume_row.addWidget(self.dojo_rl_resume_browse_btn)
    dojo_rl_outer.addLayout(drl_resume_row)

    drl_ov_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_ov_row.setSpacing(6)
    drl_ov_row.addWidget(_eai_platform_bootstrap.QLabel("Hydra"))
    self.dojo_rl_overrides_edit = _eai_platform_bootstrap.QLineEdit()
    self.dojo_rl_overrides_edit.setPlaceholderText(
        "可选覆盖。例: actor.model.openpi.use_torch_compile=False "
        "algorithm.num_group_envs=8"
    )
    self.dojo_rl_overrides_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_ov_row.addWidget(self.dojo_rl_overrides_edit, 1)
    dojo_rl_outer.addLayout(drl_ov_row)

    drl_run_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_run_row.setSpacing(6)
    self.dojo_rl_start_btn = _eai_platform_bootstrap.QPushButton("启动训练")
    self.dojo_rl_start_btn.setToolTip("启动 仿真强化学习训练")
    self.dojo_rl_start_btn.clicked.connect(self._on_dojo_rl_start_clicked)
    drl_run_row.addWidget(self.dojo_rl_start_btn)
    self.dojo_rl_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.dojo_rl_stop_btn.setObjectName("dangerAction")
    self.dojo_rl_stop_btn.setEnabled(False)
    self.dojo_rl_stop_btn.clicked.connect(self._on_dojo_rl_stop_clicked)
    drl_run_row.addWidget(self.dojo_rl_stop_btn)
    self.dojo_rl_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.dojo_rl_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_run_row.addWidget(self.dojo_rl_status_label, 1)
    dojo_rl_outer.addLayout(drl_run_row)

    drl_log_row = _eai_platform_bootstrap.QHBoxLayout()
    drl_log_row.setSpacing(6)
    drl_log_row.addWidget(_eai_platform_bootstrap.QLabel("日志目录"))
    self.dojo_rl_log_dir_edit = _eai_platform_bootstrap.QLineEdit()
    self.dojo_rl_log_dir_edit.setReadOnly(True)
    self.dojo_rl_log_dir_edit.setPlaceholderText(
        "启动后显示 runner.logger.log_path"
    )
    self.dojo_rl_log_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    drl_log_row.addWidget(self.dojo_rl_log_dir_edit, 1)
    self.dojo_rl_open_log_btn = _eai_platform_bootstrap.QPushButton("打开")
    self.dojo_rl_open_log_btn.setEnabled(False)
    self.dojo_rl_open_log_btn.clicked.connect(self._on_dojo_rl_open_log_clicked)
    drl_log_row.addWidget(self.dojo_rl_open_log_btn)
    dojo_rl_outer.addLayout(drl_log_row)

    self.dojo_rl_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.dojo_rl_log_edit.setReadOnly(True)
    self.dojo_rl_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.dojo_rl_log_edit.setMinimumHeight(180)
    self.dojo_rl_log_edit.setPlaceholderText("仿真强化学习训练 训练日志…")
    self.dojo_rl_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.dojo_rl_clear_log_btn = _eai_ui_widgets_layouts._add_clear_log_button_row(
        dojo_rl_outer, self.dojo_rl_log_edit
    )
    dojo_rl_outer.addWidget(self.dojo_rl_log_edit, 1)

    self._dojo_rl_launcher = _eai_adapters_models_reward_launchers.RoboDojoOnlineRLLauncher(self)
    self._dojo_rl_launcher.log_line.connect(self._append_dojo_rl_log)
    self._dojo_rl_launcher.status_message.connect(self._on_dojo_rl_status)
    self._dojo_rl_launcher.running_changed.connect(self._update_dojo_rl_ui)
    self._dojo_rl_launcher.log_dir_ready.connect(self._on_dojo_rl_log_dir)
    self._refresh_dojo_rl_configs(prefer="rl_release")
    if _default_policy_ckpt:
        self._append_dojo_rl_log(f"[info] 预填策略 ckpt: {_default_policy_ckpt}")
    if build._default_reward_ckpt:
        same = (
            _eai_platform_bootstrap.os.path.abspath(build._default_reward_ckpt)
            == _eai_platform_bootstrap.os.path.abspath(_default_policy_ckpt or "")
        )
        note = "（与策略相同，本机无独立 value ckpt）" if same else ""
        self._append_dojo_rl_log(
            f"[info] 预填奖励 ckpt: {build._default_reward_ckpt}{note}"
        )
