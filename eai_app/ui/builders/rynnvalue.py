"""ui.builders.rynnvalue — extracted application responsibilities."""
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
    build.rynn_tab = _eai_platform_bootstrap.QWidget()
    build.rynn_tab.setObjectName("rynnValueTab")
    rv_outer = _eai_platform_bootstrap.QVBoxLayout(build.rynn_tab)
    rv_outer.setContentsMargins(8, 6, 8, 6)
    rv_outer.setSpacing(6)
    rv_hint = _eai_platform_bootstrap.QLabel(
        "RynnValue：对机器人视频预测剩余完成时间，并生成 Analysis（描述 / Match / Success）。"
        "勾选「图形窗口」时，推理结束后自动播放趋势视频；评测任务会开启 save_videos。"
        "默认模型为 .../liyichao/models/RynnValue-4B；Python 优先 miniconda3/envs/rynnvalue。"
    )
    rv_hint.setWordWrap(True)
    rv_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    rv_outer.addWidget(rv_hint)

    rv_path_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_path_row.setSpacing(6)
    rv_path_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.rynnvalue_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("RYNNVALUE_ROOT", _eai_configuration_tabs.RYNNVALUE_ROOT_DEFAULT)
    )
    self.rynnvalue_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_path_row.addWidget(self.rynnvalue_root_edit, 1)
    self.rynnvalue_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.rynnvalue_root_browse_btn.setFixedWidth(28)
    self.rynnvalue_root_browse_btn.clicked.connect(
        self._on_rynnvalue_root_browse
    )
    rv_path_row.addWidget(self.rynnvalue_root_browse_btn)
    rv_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.rynnvalue_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_rynnvalue_python(self.rynnvalue_root_edit.text())
    )
    self.rynnvalue_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_path_row.addWidget(self.rynnvalue_python_edit, 1)
    rv_outer.addLayout(rv_path_row)

    rv_job_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_job_row.setSpacing(6)
    rv_job_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.rynnvalue_job_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.job_id in _eai_configuration_tabs.RYNNVALUE_JOBS:
        self.rynnvalue_job_combo.addItem(build.label, build.job_id)
    self.rynnvalue_job_combo.setToolTip(
        "infer=视频推理；reward_server=HTTP 奖励服务；其余为 robometer 评测"
    )
    self.rynnvalue_job_combo.currentIndexChanged.connect(
        self._on_rynnvalue_job_changed
    )
    rv_job_row.addWidget(self.rynnvalue_job_combo)
    rv_job_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.rynnvalue_cuda_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    self.rynnvalue_cuda_edit.setFixedWidth(72)
    self.rynnvalue_cuda_edit.setPlaceholderText("0")
    self.rynnvalue_cuda_edit.setToolTip("CUDA_VISIBLE_DEVICES")
    rv_job_row.addWidget(self.rynnvalue_cuda_edit)
    self.rynnvalue_render_gui_check = _eai_platform_bootstrap.QCheckBox("图形窗口")
    self.rynnvalue_render_gui_check.setChecked(
        bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip())
    )
    self.rynnvalue_render_gui_check.setToolTip(
        "勾选后：推理完成自动播放 Remaining Time 趋势视频；\n"
        "评测任务开启 save_videos。取消则仅写文件/日志。"
    )
    rv_job_row.addWidget(self.rynnvalue_render_gui_check)
    rv_job_row.addStretch(1)
    rv_outer.addLayout(rv_job_row)

    rv_model_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_model_row.setSpacing(6)
    rv_model_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    self.rynnvalue_model_edit = _eai_platform_bootstrap.QLineEdit(_eai_adapters_models_discovery.resolve_rynnvalue_model())
    self.rynnvalue_model_edit.setPlaceholderText(
        "本地 HF 目录（默认 .../liyichao/models/RynnValue-4B）"
    )
    self.rynnvalue_model_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_model_row.addWidget(self.rynnvalue_model_edit, 1)
    self.rynnvalue_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.rynnvalue_model_browse_btn.setFixedWidth(28)
    self.rynnvalue_model_browse_btn.clicked.connect(
        self._on_rynnvalue_model_browse
    )
    rv_model_row.addWidget(self.rynnvalue_model_browse_btn)
    rv_model_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.rynnvalue_ckpt_edit = _eai_platform_bootstrap.QLineEdit(_eai_adapters_models_discovery.resolve_rynnvalue_ckpt())
    self.rynnvalue_ckpt_edit.setPlaceholderText(
        "可选。训练 checkpoint（含 model.pt）；留空则只用上方 HF 模型"
    )
    self.rynnvalue_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_model_row.addWidget(self.rynnvalue_ckpt_edit, 1)
    self.rynnvalue_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.rynnvalue_ckpt_browse_btn.setFixedWidth(28)
    self.rynnvalue_ckpt_browse_btn.clicked.connect(
        self._on_rynnvalue_ckpt_browse
    )
    rv_model_row.addWidget(self.rynnvalue_ckpt_browse_btn)
    rv_outer.addLayout(rv_model_row)

    rv_video_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_video_row.setSpacing(6)
    rv_video_row.addWidget(_eai_platform_bootstrap.QLabel("视频"))
    _default_video = (
        _eai_configuration_tabs.RYNNVALUE_EXAMPLE_VIDEO
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.RYNNVALUE_EXAMPLE_VIDEO)
        else ""
    )
    self.rynnvalue_video_edit = _eai_platform_bootstrap.QLineEdit(_default_video)
    self.rynnvalue_video_edit.setPlaceholderText("infer 输入视频路径")
    self.rynnvalue_video_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_video_row.addWidget(self.rynnvalue_video_edit, 1)
    self.rynnvalue_video_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.rynnvalue_video_browse_btn.setFixedWidth(28)
    self.rynnvalue_video_browse_btn.clicked.connect(
        self._on_rynnvalue_video_browse
    )
    rv_video_row.addWidget(self.rynnvalue_video_browse_btn)
    rv_outer.addLayout(rv_video_row)

    rv_instr_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_instr_row.setSpacing(6)
    rv_instr_row.addWidget(_eai_platform_bootstrap.QLabel("指令"))
    self.rynnvalue_instruction_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.RYNNVALUE_DEFAULT_INSTRUCTION)
    self.rynnvalue_instruction_edit.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    rv_instr_row.addWidget(self.rynnvalue_instruction_edit, 1)
    rv_outer.addLayout(rv_instr_row)

    rv_meta_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_meta_row.setSpacing(6)
    rv_meta_row.addWidget(_eai_platform_bootstrap.QLabel("机器人"))
    self.rynnvalue_robot_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.RYNNVALUE_DEFAULT_ROBOT)
    self.rynnvalue_robot_edit.setPlaceholderText(
        "use_meta 必需，例: an ALOHA dual-arm robot"
    )
    self.rynnvalue_robot_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.rynnvalue_robot_edit.setToolTip("--robot_description")
    rv_meta_row.addWidget(self.rynnvalue_robot_edit, 1)
    rv_meta_row.addWidget(_eai_platform_bootstrap.QLabel("相机"))
    self.rynnvalue_camera_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.RYNNVALUE_DEFAULT_CAMERA)
    self.rynnvalue_camera_edit.setPlaceholderText(
        "use_meta 必需，例: the top-down camera"
    )
    self.rynnvalue_camera_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.rynnvalue_camera_edit.setToolTip("--camera_description")
    rv_meta_row.addWidget(self.rynnvalue_camera_edit, 1)
    rv_outer.addLayout(rv_meta_row)

    rv_opt_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_opt_row.setSpacing(6)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("frames"))
    self.rynnvalue_frames_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_frames_spin.setRange(1, 512)
    self.rynnvalue_frames_spin.setValue(16)
    self.rynnvalue_frames_spin.setToolTip(
        "--num_frames：每个 prefix 采样帧数；64@640px 在 44GB 上约需 28GiB 注意力矩阵"
    )
    rv_opt_row.addWidget(self.rynnvalue_frames_spin)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("steps"))
    self.rynnvalue_steps_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_steps_spin.setRange(0, 4096)
    self.rynnvalue_steps_spin.setValue(32)
    self.rynnvalue_steps_spin.setToolTip(
        "--num_steps：时间轴采样点数（0=每帧，易慢/OOM）"
    )
    rv_opt_row.addWidget(self.rynnvalue_steps_spin)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("batch"))
    self.rynnvalue_batch_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_batch_spin.setRange(1, 16)
    self.rynnvalue_batch_spin.setValue(1)
    self.rynnvalue_batch_spin.setToolTip(
        "--batch_size：默认 1；>1 在 eager 注意力下易 OOM"
    )
    rv_opt_row.addWidget(self.rynnvalue_batch_spin)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("tokens"))
    self.rynnvalue_tokens_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_tokens_spin.setRange(16, 2048)
    self.rynnvalue_tokens_spin.setValue(128)
    self.rynnvalue_tokens_spin.setToolTip("--max_new_tokens")
    rv_opt_row.addWidget(self.rynnvalue_tokens_spin)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("fps"))
    self.rynnvalue_fps_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_fps_spin.setRange(1, 60)
    self.rynnvalue_fps_spin.setValue(30)
    rv_opt_row.addWidget(self.rynnvalue_fps_spin)
    rv_opt_row.addWidget(_eai_platform_bootstrap.QLabel("port"))
    self.rynnvalue_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.rynnvalue_port_spin.setRange(1, 65535)
    self.rynnvalue_port_spin.setValue(8001)
    self.rynnvalue_port_spin.setToolTip("reward_server 端口")
    rv_opt_row.addWidget(self.rynnvalue_port_spin)
    rv_opt_row.addStretch(1)
    rv_outer.addLayout(rv_opt_row)

    rv_extra_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_extra_row.setSpacing(6)
    rv_extra_row.addWidget(_eai_platform_bootstrap.QLabel("额外参数"))
    self.rynnvalue_extra_edit = _eai_platform_bootstrap.QLineEdit()
    self.rynnvalue_extra_edit.setPlaceholderText(
        "可选。透传给脚本，例: --num_steps 32 或 Hydra: model_config.stride=2"
    )
    self.rynnvalue_extra_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_extra_row.addWidget(self.rynnvalue_extra_edit, 1)
    rv_outer.addLayout(rv_extra_row)

    rv_run_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_run_row.setSpacing(6)
    self.rynnvalue_start_btn = _eai_platform_bootstrap.QPushButton("启动")
    self.rynnvalue_start_btn.setToolTip("启动 RynnValue")
    self.rynnvalue_start_btn.clicked.connect(self._on_rynnvalue_start_clicked)
    rv_run_row.addWidget(self.rynnvalue_start_btn)
    self.rynnvalue_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.rynnvalue_stop_btn.setObjectName("dangerAction")
    self.rynnvalue_stop_btn.setEnabled(False)
    self.rynnvalue_stop_btn.clicked.connect(self._on_rynnvalue_stop_clicked)
    rv_run_row.addWidget(self.rynnvalue_stop_btn)
    self.rynnvalue_open_video_btn = _eai_platform_bootstrap.QPushButton("播放趋势视频")
    self.rynnvalue_open_video_btn.setEnabled(True)
    self.rynnvalue_open_video_btn.setToolTip(
        "播放最近一次推理的 output_with_trend.mp4；\n"
        "本会话无记录时会在输出目录 / RynnValue outputs 中查找最新文件。"
    )
    self.rynnvalue_open_video_btn.clicked.connect(
        self._on_rynnvalue_open_video_clicked
    )
    rv_run_row.addWidget(self.rynnvalue_open_video_btn)
    self.rynnvalue_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.rynnvalue_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_run_row.addWidget(self.rynnvalue_status_label, 1)
    rv_outer.addLayout(rv_run_row)

    rv_log_row = _eai_platform_bootstrap.QHBoxLayout()
    rv_log_row.setSpacing(6)
    rv_log_row.addWidget(_eai_platform_bootstrap.QLabel("输出目录"))
    self.rynnvalue_log_dir_edit = _eai_platform_bootstrap.QLineEdit()
    self.rynnvalue_log_dir_edit.setReadOnly(True)
    self.rynnvalue_log_dir_edit.setPlaceholderText("启动后显示 outputs/")
    self.rynnvalue_log_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rv_log_row.addWidget(self.rynnvalue_log_dir_edit, 1)
    self.rynnvalue_open_log_btn = _eai_platform_bootstrap.QPushButton("打开")
    self.rynnvalue_open_log_btn.setEnabled(False)
    self.rynnvalue_open_log_btn.clicked.connect(
        self._on_rynnvalue_open_log_clicked
    )
    rv_log_row.addWidget(self.rynnvalue_open_log_btn)
    rv_outer.addLayout(rv_log_row)

    self.rynnvalue_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.rynnvalue_log_edit.setReadOnly(True)
    self.rynnvalue_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.rynnvalue_log_edit.setMinimumHeight(180)
    self.rynnvalue_log_edit.setPlaceholderText("RynnValue 运行日志…")
    self.rynnvalue_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.rynnvalue_clear_log_btn = _eai_ui_widgets_layouts._add_clear_log_button_row(
        rv_outer, self.rynnvalue_log_edit
    )
    rv_outer.addWidget(self.rynnvalue_log_edit, 1)

    self._rynnvalue_last_video = ""
    self._rynnvalue_launcher = _eai_adapters_models_policy_launchers.RynnValueLauncher(self)
    self._rynnvalue_launcher.log_line.connect(self._append_rynnvalue_log)
    self._rynnvalue_launcher.status_message.connect(self._on_rynnvalue_status)
    self._rynnvalue_launcher.running_changed.connect(self._update_rynnvalue_ui)
    self._rynnvalue_launcher.running_changed.connect(
        self._on_rynnvalue_running_for_libero_hud
    )
    self._rynnvalue_launcher.log_dir_ready.connect(self._on_rynnvalue_log_dir)
    self._rynnvalue_launcher.output_video_ready.connect(
        self._on_rynnvalue_output_video
    )
    self._on_rynnvalue_job_changed()
    if hasattr(self, "_update_sim_rynnvalue_hud_controls"):
        self._update_sim_rynnvalue_hud_controls()
    # Show shared Live HUD panel when current backend supports it.
    # Live HUD controls live in 工作区; keep visible regardless of sim backend.
    if hasattr(self, "sim_rynnvalue_hud_panel"):
        self.sim_rynnvalue_hud_panel.setVisible(True)
