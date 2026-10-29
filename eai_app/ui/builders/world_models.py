"""ui.builders.world models — extracted application responsibilities."""
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
    world_tab = _eai_platform_bootstrap.QWidget()
    world_outer = _eai_platform_bootstrap.QVBoxLayout(world_tab)
    world_outer.setContentsMargins(8, 6, 8, 6)
    world_outer.setSpacing(6)
    world_form = _eai_platform_bootstrap.QWidget()
    world_form_l = _eai_platform_bootstrap.QVBoxLayout(world_form)
    world_form_l.setContentsMargins(0, 0, 0, 0)
    world_form_l.setSpacing(6)
    world_hint = _eai_platform_bootstrap.QLabel(
        "LingBot-World-V2：图像 + 相机轨迹 → 交互世界视频。默认使用本地 "
        "1.3B-causal-fast（单卡）。可在下方选择图像预览中的彩色画面（或点击预览图）"
        "作为输入；轨迹仍用所选示例 / 自定义 action 目录。"
    )
    world_hint.setWordWrap(True)
    world_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    world_form_l.addWidget(world_hint)

    world_row = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    world_row.addWidget(_eai_platform_bootstrap.QLabel("示例"))
    self.lingbot_world_example_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for ex in ("00", "01", "02", "03", "04", "05"):
        self.lingbot_world_example_combo.addItem(f"examples/{ex}", ex)
    self.lingbot_world_example_combo.addItem("自定义 / 相机", "custom")
    self.lingbot_world_example_combo.setCurrentIndex(3)  # examples/03
    self.lingbot_world_example_combo.currentIndexChanged.connect(
        self._on_lingbot_world_example_changed
    )
    world_row.addWidget(self.lingbot_world_example_combo)
    world_row.addWidget(_eai_platform_bootstrap.QLabel("size"))
    self.lingbot_world_size_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_world_size_combo.addItem("480×832", "480*832")
    self.lingbot_world_size_combo.addItem("720×1280", "720*1280")
    self.lingbot_world_size_combo.addItem("1280×720", "1280*720")
    world_row.addWidget(self.lingbot_world_size_combo)
    world_row.addWidget(_eai_platform_bootstrap.QLabel("frames"))
    self.lingbot_world_frames_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_world_frames_spin.setRange(5, 361)
    self.lingbot_world_frames_spin.setValue(17)
    self.lingbot_world_frames_spin.setToolTip("须为 4n+1，如 17/81/361")
    world_row.addWidget(self.lingbot_world_frames_spin)
    world_row.addWidget(_eai_platform_bootstrap.QLabel("seed"))
    self.lingbot_world_seed_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_world_seed_spin.setRange(0, 2_147_483_647)
    self.lingbot_world_seed_spin.setValue(42)
    world_row.addWidget(self.lingbot_world_seed_spin)
    self.lingbot_world_t5_cpu_check = _eai_platform_bootstrap.QCheckBox("t5_cpu")
    self.lingbot_world_t5_cpu_check.setChecked(True)
    world_row.addWidget(self.lingbot_world_t5_cpu_check)
    self.lingbot_world_offload_check = _eai_platform_bootstrap.QCheckBox("offload")
    self.lingbot_world_offload_check.setChecked(True)
    world_row.addWidget(self.lingbot_world_offload_check)
    world_form_l.addLayout(world_row)

    world_path_row = _eai_platform_bootstrap.QHBoxLayout()
    world_path_row.setSpacing(6)
    world_path_row.addWidget(_eai_platform_bootstrap.QLabel("图像"))
    self.lingbot_world_image_edit = _eai_platform_bootstrap.QLineEdit()
    world_path_row.addWidget(self.lingbot_world_image_edit, 1)
    self.lingbot_world_image_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_world_image_browse_btn.setFixedWidth(28)
    self.lingbot_world_image_browse_btn.clicked.connect(
        self._on_lingbot_world_image_browse_clicked
    )
    world_path_row.addWidget(self.lingbot_world_image_browse_btn)
    self.lingbot_world_cam_btn = _eai_platform_bootstrap.QPushButton("抓取预览")
    self.lingbot_world_cam_btn.setToolTip(
        "从下方「预览图像」所选彩色画面抓帧；轨迹仍用所选示例的 action_path。"
    )
    self.lingbot_world_cam_btn.clicked.connect(self._on_lingbot_world_cam_clicked)
    world_path_row.addWidget(self.lingbot_world_cam_btn)
    world_path_row.addWidget(_eai_platform_bootstrap.QLabel("action"))
    self.lingbot_world_action_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_world_action_edit.setPlaceholderText("含 poses.npy / intrinsics.npy 的目录")
    world_path_row.addWidget(self.lingbot_world_action_edit, 1)
    self.lingbot_world_action_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_world_action_browse_btn.setFixedWidth(28)
    self.lingbot_world_action_browse_btn.clicked.connect(
        self._on_lingbot_world_action_browse_clicked
    )
    world_path_row.addWidget(self.lingbot_world_action_browse_btn)
    world_form_l.addLayout(world_path_row)

    world_topic_row = _eai_platform_bootstrap.QHBoxLayout()
    world_topic_row.setSpacing(6)
    world_topic_row.addWidget(_eai_platform_bootstrap.QLabel("预览图像"))
    self.lingbot_world_topic_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_world_topic_combo.setMinimumWidth(220)
    self.lingbot_world_topic_combo.setToolTip(
        "选择图像预览中的彩色 topic；点击预览图也会同步到此处。"
    )
    self.lingbot_world_topic_combo.currentIndexChanged.connect(
        self._on_lingbot_world_topic_changed
    )
    world_topic_row.addWidget(self.lingbot_world_topic_combo, 1)
    self.lingbot_world_topic_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.lingbot_world_topic_refresh_btn.setFixedWidth(48)
    self.lingbot_world_topic_refresh_btn.clicked.connect(
        self._refresh_lingbot_world_topic_combo
    )
    world_topic_row.addWidget(self.lingbot_world_topic_refresh_btn)
    world_form_l.addLayout(world_topic_row)

    world_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    world_ckpt_row.setSpacing(6)
    world_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.lingbot_world_ckpt_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_tabs.LINGBOT_WORLD_CKPT_DEFAULT)
    world_ckpt_row.addWidget(self.lingbot_world_ckpt_edit, 1)
    self.lingbot_world_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_world_ckpt_browse_btn.setFixedWidth(28)
    self.lingbot_world_ckpt_browse_btn.clicked.connect(
        self._on_lingbot_world_ckpt_browse_clicked
    )
    world_ckpt_row.addWidget(self.lingbot_world_ckpt_browse_btn)
    world_form_l.addLayout(world_ckpt_row)

    world_prompt_row = _eai_platform_bootstrap.QHBoxLayout()
    world_prompt_row.setSpacing(6)
    world_prompt_row.addWidget(_eai_platform_bootstrap.QLabel("prompt"))
    self.lingbot_world_prompt_edit = _eai_platform_bootstrap.QTextEdit()
    _eai_ui_widgets_layouts._configure_resizable_text_edit(
        self.lingbot_world_prompt_edit, default_h=100, tooltip=""
    )
    self.lingbot_world_prompt_edit.setPlainText(_eai_configuration_tabs.LINGBOT_WORLD_PROMPT_DEFAULT)
    world_prompt_row.addWidget(self.lingbot_world_prompt_edit, 1)
    world_form_l.addLayout(world_prompt_row)
    self.lingbot_world_prompt_resize_bar = None

    world_run_row = _eai_platform_bootstrap.QHBoxLayout()
    world_run_row.setSpacing(6)
    self.lingbot_world_run_btn = _eai_platform_bootstrap.QPushButton("运行生成")
    self.lingbot_world_run_btn.clicked.connect(self._on_lingbot_world_run_clicked)
    world_run_row.addWidget(self.lingbot_world_run_btn)
    self.lingbot_world_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.lingbot_world_stop_btn.setObjectName("dangerAction")
    self.lingbot_world_stop_btn.setEnabled(False)
    self.lingbot_world_stop_btn.clicked.connect(self._on_lingbot_world_stop_clicked)
    world_run_row.addWidget(self.lingbot_world_stop_btn)
    self.lingbot_world_play_btn = _eai_platform_bootstrap.QPushButton("播放结果")
    self.lingbot_world_play_btn.setEnabled(False)
    self.lingbot_world_play_btn.clicked.connect(self._on_lingbot_world_play_clicked)
    world_run_row.addWidget(self.lingbot_world_play_btn)
    self.lingbot_world_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.lingbot_world_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    world_run_row.addWidget(self.lingbot_world_status_label, 1)
    world_outer.addLayout(world_run_row)

    world_preview = _eai_platform_bootstrap.QWidget()
    world_preview_l = _eai_platform_bootstrap.QHBoxLayout(world_preview)
    world_preview_l.setContentsMargins(0, 0, 0, 0)
    world_preview_l.setSpacing(8)
    self.lingbot_world_input_label = _eai_ui_widgets_images.ScaledPixmapLabel("输入图像")
    self.lingbot_world_input_label.setMinimumHeight(96)
    world_preview_l.addWidget(self.lingbot_world_input_label, 1)
    self.lingbot_world_player = _eai_ui_widgets_video.VideoPlayerWidget(world_preview)
    self.lingbot_world_player.setMinimumHeight(96)
    world_preview_l.addWidget(self.lingbot_world_player, 2)

    self.lingbot_world_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.lingbot_world_log_edit.setReadOnly(True)
    self.lingbot_world_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_world_log_edit.setPlaceholderText("LingBot-World 日志…")
    self.lingbot_world_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    world_log_pane = _eai_platform_bootstrap.QWidget()
    world_log_l = _eai_platform_bootstrap.QVBoxLayout(world_log_pane)
    world_log_l.setContentsMargins(0, 0, 0, 0)
    world_log_l.setSpacing(2)
    self.lingbot_world_log_edit.setMinimumHeight(72)
    self.lingbot_world_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        world_log_l,
        self.lingbot_world_log_edit,
        default_h=120,
        min_h=72,
        max_h=900,
        label="拖动调整 World 日志高度",
        clearable=True,
    )
    self.lingbot_world_log_edit.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    self.lingbot_world_log_edit.setMaximumHeight(16777215)
    world_split = _eai_ui_widgets_layouts._always_visible_splitter(_eai_platform_bootstrap.Qt.Vertical)
    world_split.addWidget(_eai_ui_widgets_layouts._wrap_in_vscroll(world_form))
    world_split.addWidget(world_preview)
    world_split.addWidget(world_log_pane)
    world_split.setStretchFactor(0, 2)
    world_split.setStretchFactor(1, 3)
    world_split.setStretchFactor(2, 2)
    world_split.setSizes([240, 280, 140])
    world_outer.addWidget(world_split, 1)

    self._lingbot_world_launcher = _eai_adapters_models_generation_launchers.LingbotWorldLauncher(self)
    self._lingbot_world_launcher.log_line.connect(self._append_lingbot_world_log)
    self._lingbot_world_launcher.status_message.connect(self._on_lingbot_world_status)
    self._lingbot_world_launcher.running_changed.connect(self._update_lingbot_world_ui)
    self._lingbot_world_launcher.result_ready.connect(self._on_lingbot_world_result)
    self._lingbot_world_last_output = ""
    self._on_lingbot_world_example_changed()
    self._refresh_lingbot_world_topic_combo()

    build.control_tabs.addTab(world_tab, "世界模型")
    _eai_platform_bootstrap._boot_tick("构建界面：BAGEL / CAD…")
