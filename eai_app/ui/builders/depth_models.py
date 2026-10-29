"""ui.builders.depth models — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_perception_launchers = module_ref('eai_app.adapters.models.perception_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    depth_tab = _eai_platform_bootstrap.QWidget()
    depth_outer = _eai_platform_bootstrap.QVBoxLayout(depth_tab)
    depth_outer.setContentsMargins(8, 6, 8, 6)
    depth_outer.setSpacing(6)
    depth_hint = _eai_platform_bootstrap.QLabel(
        "LingBot-Depth：用 RGB + 原始深度 + 内参做深度精修 / 补全。"
        "「当前相机 RGB-D」时可在下方选择图像预览中的彩色画面，"
        "或直接点击预览图选中后再运行。首次会从 Hugging Face 下载模型。"
    )
    depth_hint.setWordWrap(True)
    depth_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    depth_outer.addWidget(depth_hint)

    depth_row = _eai_platform_bootstrap.QHBoxLayout()
    depth_row.setSpacing(6)
    depth_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    self.lingbot_depth_model_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_depth_model_combo.addItem(
        "v0.5（推荐）", "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
    )
    self.lingbot_depth_model_combo.addItem(
        "v0.1", "robbyant/lingbot-depth-pretrain-vitl-14"
    )
    self.lingbot_depth_model_combo.addItem(
        "DC（稀疏深度）", "robbyant/lingbot-depth-postrain-dc-vitl14"
    )
    self.lingbot_depth_model_combo.setToolTip(
        "默认 v0.5；DC 更适合稀疏深度补全。也可在下方填写本地 model.pt。"
    )
    depth_row.addWidget(self.lingbot_depth_model_combo)
    depth_row.addWidget(_eai_platform_bootstrap.QLabel("输入"))
    self.lingbot_depth_source_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_depth_source_combo.addItem("当前相机 RGB-D", "camera")
    self.lingbot_depth_source_combo.addItem("官方示例", "example")
    self.lingbot_depth_source_combo.addItem("本地文件", "file")
    self.lingbot_depth_source_combo.setToolTip(
        "相机：抓取所选预览彩色 + 配对深度 + CameraInfo 内参\n"
        "示例：lingbot-depth/examples/<id>\n"
        "文件：指定 rgb / depth / intrinsics"
    )
    self.lingbot_depth_source_combo.currentIndexChanged.connect(
        self._on_lingbot_depth_source_changed
    )
    depth_row.addWidget(self.lingbot_depth_source_combo)
    depth_row.addWidget(_eai_platform_bootstrap.QLabel("示例"))
    self.lingbot_depth_example_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_depth_example_spin.setRange(0, 7)
    self.lingbot_depth_example_spin.setValue(0)
    self.lingbot_depth_example_spin.setToolTip("官方 examples/0..7")
    depth_row.addWidget(self.lingbot_depth_example_spin)
    depth_row.addWidget(_eai_platform_bootstrap.QLabel("设备"))
    self.lingbot_depth_device_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_depth_device_combo.addItem("auto", "auto")
    self.lingbot_depth_device_combo.addItem("cuda", "cuda")
    self.lingbot_depth_device_combo.addItem("cpu", "cpu")
    depth_row.addWidget(self.lingbot_depth_device_combo)
    depth_outer.addLayout(depth_row)

    depth_topic_row = _eai_platform_bootstrap.QHBoxLayout()
    depth_topic_row.setSpacing(6)
    depth_topic_row.addWidget(_eai_platform_bootstrap.QLabel("预览图像"))
    self.lingbot_depth_topic_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_depth_topic_combo.setMinimumWidth(220)
    self.lingbot_depth_topic_combo.setToolTip(
        "选择图像预览中的彩色 topic；点击预览图也会同步到此处。"
    )
    self.lingbot_depth_topic_combo.currentIndexChanged.connect(
        self._on_lingbot_depth_topic_changed
    )
    depth_topic_row.addWidget(self.lingbot_depth_topic_combo, 1)
    self.lingbot_depth_topic_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.lingbot_depth_topic_refresh_btn.setFixedWidth(48)
    self.lingbot_depth_topic_refresh_btn.setToolTip("重新扫描当前图像预览中的彩色画面")
    self.lingbot_depth_topic_refresh_btn.clicked.connect(
        self._refresh_lingbot_depth_topic_combo
    )
    depth_topic_row.addWidget(self.lingbot_depth_topic_refresh_btn)
    self.lingbot_depth_pair_label = _eai_platform_bootstrap.QLabel("配对深度: —")
    self.lingbot_depth_pair_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_depth_pair_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    depth_topic_row.addWidget(self.lingbot_depth_pair_label, 1)
    depth_outer.addLayout(depth_topic_row)

    depth_model_row = _eai_platform_bootstrap.QHBoxLayout()
    depth_model_row.setSpacing(6)
    depth_model_row.addWidget(_eai_platform_bootstrap.QLabel("model"))
    self.lingbot_depth_model_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_depth_model_edit.setPlaceholderText(
        "可选：覆盖上方模型（HF id 或本地 model.pt）"
    )
    depth_model_row.addWidget(self.lingbot_depth_model_edit, 1)
    self.lingbot_depth_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_depth_model_browse_btn.setFixedWidth(28)
    self.lingbot_depth_model_browse_btn.setToolTip("选择本地 model.pt")
    self.lingbot_depth_model_browse_btn.clicked.connect(
        self._on_lingbot_depth_model_browse_clicked
    )
    depth_model_row.addWidget(self.lingbot_depth_model_browse_btn)
    depth_outer.addLayout(depth_model_row)

    depth_path_row = _eai_platform_bootstrap.QHBoxLayout()
    depth_path_row.setSpacing(6)
    depth_path_row.addWidget(_eai_platform_bootstrap.QLabel("RGB"))
    self.lingbot_depth_rgb_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_depth_rgb_edit.setPlaceholderText("rgb.png / .jpg")
    depth_path_row.addWidget(self.lingbot_depth_rgb_edit, 1)
    self.lingbot_depth_rgb_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_depth_rgb_browse_btn.setFixedWidth(28)
    self.lingbot_depth_rgb_browse_btn.clicked.connect(
        self._on_lingbot_depth_rgb_browse_clicked
    )
    depth_path_row.addWidget(self.lingbot_depth_rgb_browse_btn)
    depth_path_row.addWidget(_eai_platform_bootstrap.QLabel("Depth"))
    self.lingbot_depth_depth_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_depth_depth_edit.setPlaceholderText("uint16 PNG（毫米）或浮点 NPY（米）")
    depth_path_row.addWidget(self.lingbot_depth_depth_edit, 1)
    self.lingbot_depth_depth_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_depth_depth_browse_btn.setFixedWidth(28)
    self.lingbot_depth_depth_browse_btn.clicked.connect(
        self._on_lingbot_depth_depth_browse_clicked
    )
    depth_path_row.addWidget(self.lingbot_depth_depth_browse_btn)
    depth_path_row.addWidget(_eai_platform_bootstrap.QLabel("K"))
    self.lingbot_depth_intrinsics_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_depth_intrinsics_edit.setPlaceholderText("intrinsics.txt")
    depth_path_row.addWidget(self.lingbot_depth_intrinsics_edit, 1)
    self.lingbot_depth_intrinsics_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_depth_intrinsics_browse_btn.setFixedWidth(28)
    self.lingbot_depth_intrinsics_browse_btn.clicked.connect(
        self._on_lingbot_depth_intrinsics_browse_clicked
    )
    depth_path_row.addWidget(self.lingbot_depth_intrinsics_browse_btn)
    depth_outer.addLayout(depth_path_row)

    depth_run_row = _eai_platform_bootstrap.QHBoxLayout()
    depth_run_row.setSpacing(6)
    self.lingbot_depth_run_btn = _eai_platform_bootstrap.QPushButton("运行精修")
    self.lingbot_depth_run_btn.setToolTip(
        "按「空间感知模型」页配置独立精修，并在该页预览；6D 串联请勾选「LingBot-Depth 补深度」后估计。"
    )
    self.lingbot_depth_run_btn.clicked.connect(self._on_lingbot_depth_run_clicked)
    self.lingbot_depth_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.lingbot_depth_stop_btn.setObjectName("dangerAction")
    self.lingbot_depth_stop_btn.setEnabled(False)
    self.lingbot_depth_stop_btn.clicked.connect(self._on_lingbot_depth_stop_clicked)
    depth_run_row.addWidget(self.lingbot_depth_stop_btn)
    self.lingbot_depth_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.lingbot_depth_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    depth_run_row.addWidget(self.lingbot_depth_status_label, 1)
    depth_outer.addLayout(depth_run_row)

    depth_preview = _eai_platform_bootstrap.QHBoxLayout()
    depth_preview.setSpacing(8)
    self.lingbot_depth_rgb_label = _eai_ui_widgets_images.ScaledPixmapLabel("RGB")
    self.lingbot_depth_rgb_label.setMinimumHeight(160)
    depth_preview.addWidget(self.lingbot_depth_rgb_label, 1)
    self.lingbot_depth_input_label = _eai_ui_widgets_images.ScaledPixmapLabel("输入深度")
    self.lingbot_depth_input_label.setMinimumHeight(160)
    depth_preview.addWidget(self.lingbot_depth_input_label, 1)
    self.lingbot_depth_refined_label = _eai_ui_widgets_images.ScaledPixmapLabel("精修深度")
    self.lingbot_depth_refined_label.setMinimumHeight(160)
    depth_preview.addWidget(self.lingbot_depth_refined_label, 1)
    depth_outer.addLayout(depth_preview, 1)

    self.lingbot_depth_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.lingbot_depth_log_edit.setReadOnly(True)
    self.lingbot_depth_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_depth_log_edit.setPlaceholderText("LingBot-Depth 日志…")
    self.lingbot_depth_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.lingbot_depth_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        depth_outer,
        self.lingbot_depth_log_edit,
        default_h=160,
        min_h=80,
        max_h=900,
        label="拖动调整 Depth 日志高度",
        clearable=True,
    )

    self._lingbot_depth_launcher = _eai_adapters_models_perception_launchers.LingbotDepthLauncher(self)
    self._lingbot_depth_launcher.log_line.connect(self._append_lingbot_depth_log)
    self._lingbot_depth_launcher.status_message.connect(self._on_lingbot_depth_status)
    self._lingbot_depth_launcher.running_changed.connect(self._update_lingbot_depth_ui)
    self._lingbot_depth_launcher.result_ready.connect(self._on_lingbot_depth_result)
    self._on_lingbot_depth_source_changed()

    build.control_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(depth_tab), "空间感知模型")
