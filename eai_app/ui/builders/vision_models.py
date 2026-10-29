"""ui.builders.vision models — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_perception_launchers = module_ref('eai_app.adapters.models.perception_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    vis_tab = _eai_platform_bootstrap.QWidget()
    vis_outer = _eai_platform_bootstrap.QVBoxLayout(vis_tab)
    vis_outer.setContentsMargins(8, 6, 8, 6)
    vis_outer.setSpacing(6)
    vis_hint = _eai_platform_bootstrap.QLabel(
        "LingBot-Vision：对当前相机 / 示例图 / 本地文件跑 patch-token PCA，"
        "展示密集空间特征。首次会从 Hugging Face 下载对应 backbone。"
    )
    vis_hint.setWordWrap(True)
    vis_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    vis_outer.addWidget(vis_hint)

    vis_row = _eai_platform_bootstrap.QHBoxLayout()
    vis_row.setSpacing(6)
    vis_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    self.lingbot_variant_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_variant_combo.addItem("small（演示）", "small")
    self.lingbot_variant_combo.addItem("base", "base")
    self.lingbot_variant_combo.addItem("large（推荐）", "large")
    self.lingbot_variant_combo.addItem("giant", "giant")
    self.lingbot_variant_combo.setToolTip(
        "small 最快；large 为官方推荐；giant 最强但更吃显存。\n"
        "未指定本地 ckpt 时会按 variant 下载 HF 权重。"
    )
    vis_row.addWidget(self.lingbot_variant_combo)
    vis_row.addWidget(_eai_platform_bootstrap.QLabel("输入"))
    self.lingbot_source_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_source_combo.addItem("当前相机", "camera")
    self.lingbot_source_combo.addItem("示例图", "example")
    self.lingbot_source_combo.addItem("本地文件", "file")
    self.lingbot_source_combo.setToolTip("PCA 输入图像来源")
    self.lingbot_source_combo.currentIndexChanged.connect(
        self._on_lingbot_source_changed
    )
    vis_row.addWidget(self.lingbot_source_combo)
    vis_row.addWidget(_eai_platform_bootstrap.QLabel("size"))
    self.lingbot_size_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_size_spin.setRange(224, 1024)
    self.lingbot_size_spin.setSingleStep(32)
    self.lingbot_size_spin.setValue(512)
    self.lingbot_size_spin.setToolTip("输入边长（会对齐到 patch size 16）")
    vis_row.addWidget(self.lingbot_size_spin)
    vis_outer.addLayout(vis_row)

    vis_path_row = _eai_platform_bootstrap.QHBoxLayout()
    vis_path_row.setSpacing(6)
    vis_path_row.addWidget(_eai_platform_bootstrap.QLabel("图像"))
    self.lingbot_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_path_edit.setPlaceholderText("本地图像路径…")
    self.lingbot_path_edit.setText(_eai_configuration_tabs.LINGBOT_VISION_EXAMPLE)
    vis_path_row.addWidget(self.lingbot_path_edit, 1)
    self.lingbot_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_browse_btn.setFixedWidth(28)
    self.lingbot_browse_btn.setToolTip("选择输入图像")
    self.lingbot_browse_btn.clicked.connect(self._on_lingbot_browse_clicked)
    vis_path_row.addWidget(self.lingbot_browse_btn)
    vis_path_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.lingbot_ckpt_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_ckpt_edit.setPlaceholderText("可选本地 model.pt（空则自动下载）")
    vis_path_row.addWidget(self.lingbot_ckpt_edit, 1)
    self.lingbot_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_ckpt_browse_btn.setFixedWidth(28)
    self.lingbot_ckpt_browse_btn.setToolTip("选择本地 backbone .pt")
    self.lingbot_ckpt_browse_btn.clicked.connect(self._on_lingbot_ckpt_browse_clicked)
    vis_path_row.addWidget(self.lingbot_ckpt_browse_btn)
    vis_outer.addLayout(vis_path_row)

    vis_run_row = _eai_platform_bootstrap.QHBoxLayout()
    vis_run_row.setSpacing(6)
    self.lingbot_run_btn = _eai_platform_bootstrap.QPushButton("运行 PCA")
    self.lingbot_run_btn.setToolTip(
        "启动子进程跑 LingBot-Vision，并将输入 / PCA 显示在下方。"
    )
    self.lingbot_run_btn.clicked.connect(self._on_lingbot_run_clicked)
    vis_run_row.addWidget(self.lingbot_run_btn)
    self.lingbot_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.lingbot_stop_btn.setObjectName("dangerAction")
    self.lingbot_stop_btn.setEnabled(False)
    self.lingbot_stop_btn.clicked.connect(self._on_lingbot_stop_clicked)
    vis_run_row.addWidget(self.lingbot_stop_btn)
    self.lingbot_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.lingbot_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    vis_run_row.addWidget(self.lingbot_status_label, 1)
    vis_outer.addLayout(vis_run_row)

    vis_preview = _eai_platform_bootstrap.QHBoxLayout()
    vis_preview.setSpacing(8)
    self.lingbot_input_label = _eai_ui_widgets_images.ScaledPixmapLabel("输入图像")
    self.lingbot_input_label.setMinimumHeight(180)
    vis_preview.addWidget(self.lingbot_input_label, 1)
    self.lingbot_pca_label = _eai_ui_widgets_images.ScaledPixmapLabel("Patch PCA")
    self.lingbot_pca_label.setMinimumHeight(180)
    vis_preview.addWidget(self.lingbot_pca_label, 1)
    vis_outer.addLayout(vis_preview, 1)

    self.lingbot_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.lingbot_log_edit.setReadOnly(True)
    self.lingbot_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_log_edit.setPlaceholderText("LingBot-Vision 日志…")
    self.lingbot_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.lingbot_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        vis_outer,
        self.lingbot_log_edit,
        default_h=160,
        min_h=80,
        max_h=900,
        label="拖动调整 Vision 日志高度",
        clearable=True,
    )

    self._lingbot_launcher = _eai_adapters_models_perception_launchers.LingbotVisionLauncher(self)
    self._lingbot_launcher.log_line.connect(self._append_lingbot_log)
    self._lingbot_launcher.status_message.connect(self._on_lingbot_status)
    self._lingbot_launcher.running_changed.connect(self._update_lingbot_ui)
    self._lingbot_launcher.result_ready.connect(self._on_lingbot_result)
    self._on_lingbot_source_changed()

    build.control_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(vis_tab), "视觉基础模型")
    _eai_platform_bootstrap._boot_tick("构建界面：空间/3D…")
