"""ui.builders.roboicl — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_policy_launchers = module_ref('eai_app.adapters.models.policy_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    roboicl_tab = _eai_platform_bootstrap.QWidget()
    roboicl_tab.setObjectName("roboiclTab")
    ricl_outer = _eai_platform_bootstrap.QVBoxLayout(roboicl_tab)
    ricl_outer.setContentsMargins(8, 6, 8, 6)
    ricl_outer.setSpacing(6)
    ricl_hint = _eai_platform_bootstrap.QLabel(
        "RoboICL：在 RoboDojo 上跑 GPT-6 Astra 的 in-context 控制。"
        "勾选「图形窗口」会去掉 Isaac --headless，弹出仿真窗口。"
        "完整 rollout 需要 ASTRA_API_KEY；dry-run / capture-only 可不填。"
        "默认仓库 "
        f"{_eai_configuration_tabs.ROBOICL_ROOT_DEFAULT}；sim/policy Python 优先 env_isaaclab。"
        "首次使用请先 submodule + fetch_assets。"
    )
    ricl_hint.setWordWrap(True)
    ricl_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    ricl_outer.addWidget(ricl_hint)

    ricl_path_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_path_row.setSpacing(6)
    ricl_path_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.roboicl_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ROBOICL_ROOT", _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT)
    )
    self.roboicl_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_path_row.addWidget(self.roboicl_root_edit, 1)
    self.roboicl_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.roboicl_root_browse_btn.setFixedWidth(28)
    self.roboicl_root_browse_btn.clicked.connect(self._on_roboicl_root_browse)
    ricl_path_row.addWidget(self.roboicl_root_browse_btn)
    ricl_outer.addLayout(ricl_path_row)

    ricl_py_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_py_row.setSpacing(6)
    ricl_py_row.addWidget(_eai_platform_bootstrap.QLabel("Policy Py"))
    self.roboicl_policy_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_roboicl_policy_python(self.roboicl_root_edit.text())
    )
    self.roboicl_policy_python_edit.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    ricl_py_row.addWidget(self.roboicl_policy_python_edit, 1)
    ricl_py_row.addWidget(_eai_platform_bootstrap.QLabel("Sim Py"))
    self.roboicl_sim_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_roboicl_sim_python(self.roboicl_root_edit.text())
    )
    self.roboicl_sim_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_py_row.addWidget(self.roboicl_sim_python_edit, 1)
    ricl_outer.addLayout(ricl_py_row)

    ricl_job_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_job_row.setSpacing(6)
    ricl_job_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.roboicl_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, mode_id in _eai_configuration_tabs.ROBOICL_MODES:
        self.roboicl_mode_combo.addItem(build.label, mode_id)
    self.roboicl_mode_combo.setToolTip(
        "dry_run=只校验配置；capture_only=起场景验相机；run=完整评测"
    )
    self.roboicl_mode_combo.currentIndexChanged.connect(
        self._on_roboicl_mode_changed
    )
    ricl_job_row.addWidget(self.roboicl_mode_combo)
    ricl_job_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.roboicl_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.roboicl_task_combo.setEditable(True)
    self.roboicl_task_combo.setInsertPolicy(_eai_platform_bootstrap.QComboBox.NoInsert)
    self.roboicl_task_combo.setMinimumWidth(180)
    self.roboicl_task_combo.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_job_row.addWidget(self.roboicl_task_combo, 1)
    self.roboicl_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.roboicl_refresh_btn.setToolTip("重新扫描 configs/tasks/*.json")
    self.roboicl_refresh_btn.clicked.connect(self._refresh_roboicl_tasks)
    ricl_job_row.addWidget(self.roboicl_refresh_btn)
    ricl_outer.addLayout(ricl_job_row)

    ricl_proto_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_proto_row.setSpacing(6)
    ricl_proto_row.addWidget(_eai_platform_bootstrap.QLabel("协议"))
    self.roboicl_protocol_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, rel_path, shots in _eai_configuration_tabs.ROBOICL_PROTOCOLS:
        self.roboicl_protocol_combo.addItem(build.label, (rel_path, shots))
    self.roboicl_protocol_combo.setToolTip(
        "切换后自动设置 shots，并填入对应 protocol json"
    )
    self.roboicl_protocol_combo.currentIndexChanged.connect(
        self._on_roboicl_protocol_changed
    )
    ricl_proto_row.addWidget(self.roboicl_protocol_combo)
    ricl_proto_row.addWidget(_eai_platform_bootstrap.QLabel("shots"))
    self.roboicl_shots_spin = _eai_platform_bootstrap.QSpinBox()
    self.roboicl_shots_spin.setRange(0, 1)
    self.roboicl_shots_spin.setValue(0)
    self.roboicl_shots_spin.setToolTip("0=zero-shot，1=one-shot")
    ricl_proto_row.addWidget(self.roboicl_shots_spin)
    ricl_proto_row.addWidget(_eai_platform_bootstrap.QLabel("seed"))
    self.roboicl_seed_spin = _eai_platform_bootstrap.QSpinBox()
    self.roboicl_seed_spin.setRange(0, 99999)
    self.roboicl_seed_spin.setValue(0)
    ricl_proto_row.addWidget(self.roboicl_seed_spin)
    ricl_proto_row.addWidget(_eai_platform_bootstrap.QLabel("layout"))
    self.roboicl_layout_spin = _eai_platform_bootstrap.QSpinBox()
    self.roboicl_layout_spin.setRange(0, 999)
    self.roboicl_layout_spin.setValue(0)
    ricl_proto_row.addWidget(self.roboicl_layout_spin)
    ricl_proto_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.roboicl_gpu_spin = _eai_platform_bootstrap.QSpinBox()
    self.roboicl_gpu_spin.setRange(0, 15)
    self.roboicl_gpu_spin.setValue(0)
    ricl_proto_row.addWidget(self.roboicl_gpu_spin)
    self.roboicl_render_gui_check = _eai_platform_bootstrap.QCheckBox("图形窗口")
    self.roboicl_render_gui_check.setChecked(
        bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip())
    )
    self.roboicl_render_gui_check.setToolTip(
        "勾选后 Isaac Sim 不以 headless 启动，弹出图形窗口。\n"
        "dry-run 不启仿真，此选项无效。"
    )
    ricl_proto_row.addWidget(self.roboicl_render_gui_check)
    ricl_proto_row.addStretch(1)
    ricl_outer.addLayout(ricl_proto_row)

    ricl_data_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_data_row.setSpacing(6)
    ricl_data_row.addWidget(_eai_platform_bootstrap.QLabel("data"))
    self.roboicl_data_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get(
            "ROBOICL_DATA_ROOT",
            _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.ROBOICL_ROOT_DEFAULT, "data"),
        )
    )
    self.roboicl_data_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_data_row.addWidget(self.roboicl_data_edit, 1)
    self.roboicl_data_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.roboicl_data_browse_btn.setFixedWidth(28)
    self.roboicl_data_browse_btn.clicked.connect(self._on_roboicl_data_browse)
    ricl_data_row.addWidget(self.roboicl_data_browse_btn)
    ricl_data_row.addWidget(_eai_platform_bootstrap.QLabel("results"))
    self.roboicl_results_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get(
            "ROBOICL_RESULTS_ROOT",
            _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.ROBOICL_ROOT_DEFAULT, "results"),
        )
    )
    self.roboicl_results_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_data_row.addWidget(self.roboicl_results_edit, 1)
    self.roboicl_results_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.roboicl_results_browse_btn.setFixedWidth(28)
    self.roboicl_results_browse_btn.clicked.connect(
        self._on_roboicl_results_browse
    )
    ricl_data_row.addWidget(self.roboicl_results_browse_btn)
    ricl_outer.addLayout(ricl_data_row)

    ricl_ref_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_ref_row.setSpacing(6)
    ricl_ref_row.addWidget(_eai_platform_bootstrap.QLabel("reference"))
    self.roboicl_reference_edit = _eai_platform_bootstrap.QLineEdit()
    self.roboicl_reference_edit.setPlaceholderText(
        "可选。one-shot 无内置 bundle 时指定 reference 目录"
    )
    self.roboicl_reference_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_ref_row.addWidget(self.roboicl_reference_edit, 1)
    self.roboicl_reference_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.roboicl_reference_browse_btn.setFixedWidth(28)
    self.roboicl_reference_browse_btn.clicked.connect(
        self._on_roboicl_reference_browse
    )
    ricl_ref_row.addWidget(self.roboicl_reference_browse_btn)
    ricl_ref_row.addWidget(_eai_platform_bootstrap.QLabel("model"))
    self.roboicl_model_edit = _eai_platform_bootstrap.QLineEdit(_eai_platform_bootstrap.os.environ.get("ASTRA_MODEL", ""))
    self.roboicl_model_edit.setPlaceholderText("可选。覆盖 ASTRA_MODEL")
    self.roboicl_model_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_ref_row.addWidget(self.roboicl_model_edit, 1)
    ricl_outer.addLayout(ricl_ref_row)

    ricl_key_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_key_row.setSpacing(6)
    ricl_key_row.addWidget(_eai_platform_bootstrap.QLabel("API Key"))
    self.roboicl_api_key_edit = _eai_platform_bootstrap.QLineEdit(_eai_platform_bootstrap.os.environ.get("ASTRA_API_KEY", ""))
    self.roboicl_api_key_edit.setEchoMode(_eai_platform_bootstrap.QLineEdit.Password)
    self.roboicl_api_key_edit.setPlaceholderText(
        "完整 rollout 需要；也可事先 export ASTRA_API_KEY"
    )
    self.roboicl_api_key_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_key_row.addWidget(self.roboicl_api_key_edit, 1)
    ricl_outer.addLayout(ricl_key_row)

    ricl_extra_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_extra_row.setSpacing(6)
    ricl_extra_row.addWidget(_eai_platform_bootstrap.QLabel("额外参数"))
    self.roboicl_extra_edit = _eai_platform_bootstrap.QLineEdit()
    self.roboicl_extra_edit.setPlaceholderText(
        "可选。透传给 roboicl.run，例: --reasoning-effort high"
    )
    self.roboicl_extra_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_extra_row.addWidget(self.roboicl_extra_edit, 1)
    ricl_outer.addLayout(ricl_extra_row)

    ricl_run_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_run_row.setSpacing(6)
    self.roboicl_start_btn = _eai_platform_bootstrap.QPushButton("启动")
    self.roboicl_start_btn.setToolTip("启动 RoboICL")
    self.roboicl_start_btn.clicked.connect(self._on_roboicl_start_clicked)
    ricl_run_row.addWidget(self.roboicl_start_btn)
    self.roboicl_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.roboicl_stop_btn.setObjectName("dangerAction")
    self.roboicl_stop_btn.setEnabled(False)
    self.roboicl_stop_btn.clicked.connect(self._on_roboicl_stop_clicked)
    ricl_run_row.addWidget(self.roboicl_stop_btn)
    self.roboicl_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.roboicl_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_run_row.addWidget(self.roboicl_status_label, 1)
    ricl_outer.addLayout(ricl_run_row)

    ricl_log_row = _eai_platform_bootstrap.QHBoxLayout()
    ricl_log_row.setSpacing(6)
    ricl_log_row.addWidget(_eai_platform_bootstrap.QLabel("结果目录"))
    self.roboicl_log_dir_edit = _eai_platform_bootstrap.QLineEdit()
    self.roboicl_log_dir_edit.setReadOnly(True)
    self.roboicl_log_dir_edit.setPlaceholderText("启动后显示 results/ 或本次 Run 目录")
    self.roboicl_log_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ricl_log_row.addWidget(self.roboicl_log_dir_edit, 1)
    self.roboicl_open_log_btn = _eai_platform_bootstrap.QPushButton("打开")
    self.roboicl_open_log_btn.setEnabled(False)
    self.roboicl_open_log_btn.clicked.connect(self._on_roboicl_open_log_clicked)
    ricl_log_row.addWidget(self.roboicl_open_log_btn)
    ricl_outer.addLayout(ricl_log_row)

    self.roboicl_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.roboicl_log_edit.setReadOnly(True)
    self.roboicl_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.roboicl_log_edit.setMinimumHeight(180)
    self.roboicl_log_edit.setPlaceholderText("RoboICL 运行日志…")
    self.roboicl_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.roboicl_clear_log_btn = _eai_ui_widgets_layouts._add_clear_log_button_row(
        ricl_outer, self.roboicl_log_edit
    )
    ricl_outer.addWidget(self.roboicl_log_edit, 1)

    self._roboicl_launcher = _eai_adapters_models_policy_launchers.RoboICLLauncher(self)
    self._roboicl_launcher.log_line.connect(self._append_roboicl_log)
    self._roboicl_launcher.status_message.connect(self._on_roboicl_status)
    self._roboicl_launcher.running_changed.connect(self._update_roboicl_ui)
    self._roboicl_launcher.log_dir_ready.connect(self._on_roboicl_log_dir)
    self._refresh_roboicl_tasks(prefer="deposit_coin")
    self._on_roboicl_protocol_changed()
    self._on_roboicl_mode_changed()

    build.control_tabs.addTab(build.sim_tab, "仿真评测")
    build.control_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(build.real_inner), "真机评测")
    build.control_tabs.addTab(build.local_cam_tab, "本地摄像头")
    build.control_tabs.addTab(build.reward_tab, "Reward评测")
    build.control_tabs.addTab(build.reward_wf_tab, "Reward训练")
    build.control_tabs.addTab(build.dojo_rl_tab, "仿真强化学习训练")
    build.control_tabs.addTab(build.robometer_tab, "RoboMeter")
    build.control_tabs.addTab(build.rynn_tab, "RynnValue")
    build.control_tabs.addTab(roboicl_tab, "RoboICL")
    build.control_tabs.addTab(build.ctx_tab, "ICL")
    _eai_platform_bootstrap._boot_tick("构建界面：收尾…")
