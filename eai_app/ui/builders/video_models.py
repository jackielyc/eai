"""ui.builders.video models — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_generation_launchers = module_ref('eai_app.adapters.models.generation_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

def build(self, build):
    video_tab = _eai_platform_bootstrap.QWidget()
    video_outer = _eai_platform_bootstrap.QVBoxLayout(video_tab)
    video_outer.setContentsMargins(8, 6, 8, 6)
    video_outer.setSpacing(6)
    video_form = _eai_platform_bootstrap.QWidget()
    video_form_l = _eai_platform_bootstrap.QVBoxLayout(video_form)
    video_form_l.setContentsMargins(0, 0, 0, 0)
    video_form_l.setSpacing(6)
    video_hint = _eai_platform_bootstrap.QLabel(
        "LingBot-Video：T2I / T2V / TI2V 生成。推荐用官方 cases 的结构化 prompt.json；"
        "TI2V 可在下方选择图像预览中的彩色画面（或点击预览图）作为首帧。"
        "首次会下载 Dense-1.3B 权重。"
    )
    video_hint.setWordWrap(True)
    video_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    video_form_l.addWidget(video_hint)

    video_row = _eai_platform_bootstrap.QHBoxLayout()
    video_row.setSpacing(6)
    video_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.lingbot_video_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_video_mode_combo.addItem("T2V 文生视频", "t2v")
    self.lingbot_video_mode_combo.addItem("T2I 文生图", "t2i")
    self.lingbot_video_mode_combo.addItem("TI2V 图生视频", "ti2v")
    self.lingbot_video_mode_combo.currentIndexChanged.connect(
        self._on_lingbot_video_mode_changed
    )
    video_row.addWidget(self.lingbot_video_mode_combo)
    video_row.addWidget(_eai_platform_bootstrap.QLabel("案例"))
    self.lingbot_video_case_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_video_case_combo.setToolTip("assets/cases 下官方结构化 prompt")
    self.lingbot_video_case_combo.currentIndexChanged.connect(
        self._on_lingbot_video_case_changed
    )
    video_row.addWidget(self.lingbot_video_case_combo, 1)
    video_row.addWidget(_eai_platform_bootstrap.QLabel("预设"))
    self.lingbot_video_preset_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_video_preset_combo.addItem("快速预览", "fast")
    self.lingbot_video_preset_combo.addItem("标准 (480p)", "standard")
    self.lingbot_video_preset_combo.addItem("自定义", "custom")
    self.lingbot_video_preset_combo.currentIndexChanged.connect(
        self._on_lingbot_video_preset_changed
    )
    video_row.addWidget(self.lingbot_video_preset_combo)
    video_form_l.addLayout(video_row)

    video_path_row = _eai_platform_bootstrap.QHBoxLayout()
    video_path_row.setSpacing(6)
    video_path_row.addWidget(_eai_platform_bootstrap.QLabel("prompt.json"))
    self.lingbot_video_prompt_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_video_prompt_edit.setPlaceholderText("结构化 JSON Prompt 路径…")
    video_path_row.addWidget(self.lingbot_video_prompt_edit, 1)
    self.lingbot_video_prompt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_video_prompt_browse_btn.setFixedWidth(28)
    self.lingbot_video_prompt_browse_btn.clicked.connect(
        self._on_lingbot_video_prompt_browse_clicked
    )
    video_path_row.addWidget(self.lingbot_video_prompt_browse_btn)
    video_path_row.addWidget(_eai_platform_bootstrap.QLabel("首帧"))
    self.lingbot_video_image_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_video_image_edit.setPlaceholderText("TI2V 首帧图（可选）")
    video_path_row.addWidget(self.lingbot_video_image_edit, 1)
    self.lingbot_video_image_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_video_image_browse_btn.setFixedWidth(28)
    self.lingbot_video_image_browse_btn.clicked.connect(
        self._on_lingbot_video_image_browse_clicked
    )
    video_path_row.addWidget(self.lingbot_video_image_browse_btn)
    self.lingbot_video_cam_btn = _eai_platform_bootstrap.QPushButton("抓取预览")
    self.lingbot_video_cam_btn.setToolTip(
        "从下方「预览图像」所选彩色画面抓一帧作 TI2V 首帧；也可点击图像预览选中。"
    )
    self.lingbot_video_cam_btn.clicked.connect(self._on_lingbot_video_cam_clicked)
    video_path_row.addWidget(self.lingbot_video_cam_btn)
    video_form_l.addLayout(video_path_row)

    video_topic_row = _eai_platform_bootstrap.QHBoxLayout()
    video_topic_row.setSpacing(6)
    video_topic_row.addWidget(_eai_platform_bootstrap.QLabel("预览图像"))
    self.lingbot_video_topic_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_video_topic_combo.setMinimumWidth(220)
    self.lingbot_video_topic_combo.setToolTip(
        "选择图像预览中的彩色 topic；点击预览图也会同步到此处（TI2V 首帧）。"
    )
    self.lingbot_video_topic_combo.currentIndexChanged.connect(
        self._on_lingbot_video_topic_changed
    )
    video_topic_row.addWidget(self.lingbot_video_topic_combo, 1)
    self.lingbot_video_topic_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.lingbot_video_topic_refresh_btn.setFixedWidth(48)
    self.lingbot_video_topic_refresh_btn.clicked.connect(
        self._refresh_lingbot_video_topic_combo
    )
    video_topic_row.addWidget(self.lingbot_video_topic_refresh_btn)
    video_form_l.addLayout(video_topic_row)

    video_model_row = _eai_platform_bootstrap.QHBoxLayout()
    video_model_row.setSpacing(6)
    video_model_row.addWidget(_eai_platform_bootstrap.QLabel("model_dir"))
    self.lingbot_video_model_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_video_model_edit.setPlaceholderText(
        f"空则自动下载 {_eai_configuration_tabs.LINGBOT_VIDEO_HF_DENSE}"
    )
    video_model_row.addWidget(self.lingbot_video_model_edit, 1)
    self.lingbot_video_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_video_model_browse_btn.setFixedWidth(28)
    self.lingbot_video_model_browse_btn.clicked.connect(
        self._on_lingbot_video_model_browse_clicked
    )
    video_model_row.addWidget(self.lingbot_video_model_browse_btn)
    video_form_l.addLayout(video_model_row)

    video_opt_row = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    video_opt_row.addWidget(_eai_platform_bootstrap.QLabel("H"))
    self.lingbot_video_h_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_video_h_spin.setRange(64, 2160)
    self.lingbot_video_h_spin.setSingleStep(16)
    video_opt_row.addWidget(self.lingbot_video_h_spin)
    video_opt_row.addWidget(_eai_platform_bootstrap.QLabel("W"))
    self.lingbot_video_w_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_video_w_spin.setRange(64, 3840)
    self.lingbot_video_w_spin.setSingleStep(16)
    video_opt_row.addWidget(self.lingbot_video_w_spin)
    video_opt_row.addWidget(_eai_platform_bootstrap.QLabel("frames"))
    self.lingbot_video_frames_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_video_frames_spin.setRange(1, 241)
    self.lingbot_video_frames_spin.setToolTip("须为 1 或 4n+1（如 9/49/121）")
    video_opt_row.addWidget(self.lingbot_video_frames_spin)
    video_opt_row.addWidget(_eai_platform_bootstrap.QLabel("steps"))
    self.lingbot_video_steps_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_video_steps_spin.setRange(1, 100)
    video_opt_row.addWidget(self.lingbot_video_steps_spin)
    video_opt_row.addWidget(_eai_platform_bootstrap.QLabel("seed"))
    self.lingbot_video_seed_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_video_seed_spin.setRange(0, 2_147_483_647)
    self.lingbot_video_seed_spin.setValue(42)
    video_opt_row.addWidget(self.lingbot_video_seed_spin)
    self.lingbot_video_refiner_check = _eai_platform_bootstrap.QCheckBox("refiner")
    self.lingbot_video_refiner_check.setToolTip("Dense 通常无 refiner；MoE 才需要")
    video_opt_row.addWidget(self.lingbot_video_refiner_check)
    video_form_l.addLayout(video_opt_row)

    video_run_row = _eai_platform_bootstrap.QHBoxLayout()
    video_run_row.setSpacing(6)
    self.lingbot_video_run_btn = _eai_platform_bootstrap.QPushButton("运行生成")
    self.lingbot_video_run_btn.clicked.connect(self._on_lingbot_video_run_clicked)
    video_run_row.addWidget(self.lingbot_video_run_btn)
    self.lingbot_video_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.lingbot_video_stop_btn.setObjectName("dangerAction")
    self.lingbot_video_stop_btn.setEnabled(False)
    self.lingbot_video_stop_btn.clicked.connect(self._on_lingbot_video_stop_clicked)
    video_run_row.addWidget(self.lingbot_video_stop_btn)
    self.lingbot_video_open_btn = _eai_platform_bootstrap.QPushButton("播放结果")
    self.lingbot_video_open_btn.setEnabled(False)
    self.lingbot_video_open_btn.setToolTip("在弹窗中播放生成的视频 / 打开图像")
    self.lingbot_video_open_btn.clicked.connect(self._on_lingbot_video_open_clicked)
    video_run_row.addWidget(self.lingbot_video_open_btn)
    self.lingbot_video_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.lingbot_video_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    video_run_row.addWidget(self.lingbot_video_status_label, 1)
    video_outer.addLayout(video_run_row)

    video_preview = _eai_platform_bootstrap.QWidget()
    video_preview_l = _eai_platform_bootstrap.QHBoxLayout(video_preview)
    video_preview_l.setContentsMargins(0, 0, 0, 0)
    video_preview_l.setSpacing(8)
    self.lingbot_video_input_label = _eai_ui_widgets_images.ScaledPixmapLabel("输入 / 首帧")
    self.lingbot_video_input_label.setMinimumHeight(96)
    video_preview_l.addWidget(self.lingbot_video_input_label, 1)

    self.lingbot_video_output_stack = _eai_platform_bootstrap.QWidget()
    out_stack_layout = _eai_platform_bootstrap.QVBoxLayout(self.lingbot_video_output_stack)
    out_stack_layout.setContentsMargins(0, 0, 0, 0)
    out_stack_layout.setSpacing(0)
    self.lingbot_video_output_label = _eai_ui_widgets_images.ScaledPixmapLabel("输出图像（T2I）")
    self.lingbot_video_output_label.setMinimumHeight(96)
    out_stack_layout.addWidget(self.lingbot_video_output_label)
    self.lingbot_video_player = _eai_ui_widgets_video.VideoPlayerWidget(self.lingbot_video_output_stack)
    self.lingbot_video_player.setMinimumHeight(96)
    self.lingbot_video_player.setVisible(False)
    out_stack_layout.addWidget(self.lingbot_video_player, 1)
    video_preview_l.addWidget(self.lingbot_video_output_stack, 2)

    self.lingbot_video_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.lingbot_video_log_edit.setReadOnly(True)
    self.lingbot_video_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_video_log_edit.setPlaceholderText("LingBot-Video 日志…")
    self.lingbot_video_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    video_log_pane = _eai_platform_bootstrap.QWidget()
    video_log_l = _eai_platform_bootstrap.QVBoxLayout(video_log_pane)
    video_log_l.setContentsMargins(0, 0, 0, 0)
    video_log_l.setSpacing(2)
    self.lingbot_video_log_edit.setMinimumHeight(72)
    self.lingbot_video_log_edit.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    self.lingbot_video_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        video_log_l,
        self.lingbot_video_log_edit,
        default_h=120,
        min_h=72,
        max_h=900,
        label="拖动调整 Video 日志高度",
        clearable=True,
    )
    self.lingbot_video_log_edit.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    self.lingbot_video_log_edit.setMaximumHeight(16777215)
    video_split = _eai_ui_widgets_layouts._always_visible_splitter(_eai_platform_bootstrap.Qt.Vertical)
    video_split.addWidget(_eai_ui_widgets_layouts._wrap_in_vscroll(video_form))
    video_split.addWidget(video_preview)
    video_split.addWidget(video_log_pane)
    video_split.setStretchFactor(0, 2)
    video_split.setStretchFactor(1, 3)
    video_split.setStretchFactor(2, 2)
    video_split.setSizes([220, 280, 140])
    video_outer.addWidget(video_split, 1)

    self._lingbot_video_launcher = _eai_adapters_models_generation_launchers.LingbotVideoLauncher(self)
    self._lingbot_video_launcher.log_line.connect(self._append_lingbot_video_log)
    self._lingbot_video_launcher.status_message.connect(self._on_lingbot_video_status)
    self._lingbot_video_launcher.running_changed.connect(self._update_lingbot_video_ui)
    self._lingbot_video_launcher.result_ready.connect(self._on_lingbot_video_result)
    self._lingbot_video_last_output = ""
    self._populate_lingbot_video_cases()
    self._on_lingbot_video_preset_changed()
    self._on_lingbot_video_mode_changed()

    build.control_tabs.addTab(video_tab, "视频生成模型")
