"""ui.builders.mapping — extracted application responsibilities."""
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

def build(self, build):
    map_tab = _eai_platform_bootstrap.QWidget()
    map_outer = _eai_platform_bootstrap.QVBoxLayout(map_tab)
    map_outer.setContentsMargins(8, 6, 8, 6)
    map_outer.setSpacing(6)
    map_hint = _eai_platform_bootstrap.QLabel(
        "LingBot-Map：流式多视角 3D 重建。运行后在浏览器打开 viser 查看点云"
        "（默认 http://127.0.0.1:8080）。首次会从 Hugging Face 下载 lingbot-map.pt。"
    )
    map_hint.setWordWrap(True)
    map_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    map_outer.addWidget(map_hint)

    map_row = _eai_platform_bootstrap.QHBoxLayout()
    map_row.setSpacing(6)
    map_row.addWidget(_eai_platform_bootstrap.QLabel("场景"))
    self.lingbot_map_scene_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_map_scene_combo.addItem("courthouse（示例）", "courthouse")
    self.lingbot_map_scene_combo.addItem("university（示例）", "university")
    self.lingbot_map_scene_combo.addItem("loop（示例）", "loop")
    self.lingbot_map_scene_combo.addItem("图像目录", "folder")
    self.lingbot_map_scene_combo.addItem("视频文件", "video")
    self.lingbot_map_scene_combo.addItem("当前相机采集", "camera")
    self.lingbot_map_scene_combo.setToolTip(
        "示例场景在 lingbot-map/example/；\n"
        "相机采集：把当前彩色帧写入缓存目录后再重建。"
    )
    self.lingbot_map_scene_combo.currentIndexChanged.connect(
        self._on_lingbot_map_scene_changed
    )
    map_row.addWidget(self.lingbot_map_scene_combo)
    map_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    self.lingbot_map_ckpt_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.lingbot_map_ckpt_combo.addItem("lingbot-map.pt（推荐）", "lingbot-map.pt")
    self.lingbot_map_ckpt_combo.addItem("lingbot-map-long.pt", "lingbot-map-long.pt")
    self.lingbot_map_ckpt_combo.setToolTip("空本地路径时从 HF robbyant/lingbot-map 下载")
    map_row.addWidget(self.lingbot_map_ckpt_combo)
    map_row.addWidget(_eai_platform_bootstrap.QLabel("port"))
    self.lingbot_map_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_map_port_spin.setRange(1024, 65535)
    self.lingbot_map_port_spin.setValue(_eai_configuration_tabs.LINGBOT_MAP_PORT_DEFAULT)
    map_row.addWidget(self.lingbot_map_port_spin)
    map_outer.addLayout(map_row)

    map_path_row = _eai_platform_bootstrap.QHBoxLayout()
    map_path_row.setSpacing(6)
    map_path_row.addWidget(_eai_platform_bootstrap.QLabel("路径"))
    self.lingbot_map_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_map_path_edit.setPlaceholderText("图像目录或视频路径…")
    self.lingbot_map_path_edit.setText(
        _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT, "example", "courthouse")
    )
    map_path_row.addWidget(self.lingbot_map_path_edit, 1)
    self.lingbot_map_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_map_browse_btn.setFixedWidth(28)
    self.lingbot_map_browse_btn.clicked.connect(self._on_lingbot_map_browse_clicked)
    map_path_row.addWidget(self.lingbot_map_browse_btn)
    map_path_row.addWidget(_eai_platform_bootstrap.QLabel("本地ckpt"))
    self.lingbot_map_model_edit = _eai_platform_bootstrap.QLineEdit()
    self.lingbot_map_model_edit.setPlaceholderText("可选本地 .pt（优先于上方 ckpt）")
    map_path_row.addWidget(self.lingbot_map_model_edit, 1)
    self.lingbot_map_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.lingbot_map_model_browse_btn.setFixedWidth(28)
    self.lingbot_map_model_browse_btn.clicked.connect(
        self._on_lingbot_map_model_browse_clicked
    )
    map_path_row.addWidget(self.lingbot_map_model_browse_btn)
    map_outer.addLayout(map_path_row)

    map_opt_row = _eai_platform_bootstrap.QHBoxLayout()
    map_opt_row.setSpacing(6)
    map_opt_row.addWidget(_eai_platform_bootstrap.QLabel("first_k"))
    self.lingbot_map_first_k_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_map_first_k_spin.setRange(0, 10000)
    self.lingbot_map_first_k_spin.setValue(48)
    self.lingbot_map_first_k_spin.setToolTip("只用前 K 帧；0=全部（大场景更慢）")
    map_opt_row.addWidget(self.lingbot_map_first_k_spin)
    map_opt_row.addWidget(_eai_platform_bootstrap.QLabel("stride"))
    self.lingbot_map_stride_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_map_stride_spin.setRange(1, 64)
    self.lingbot_map_stride_spin.setValue(1)
    map_opt_row.addWidget(self.lingbot_map_stride_spin)
    map_opt_row.addWidget(_eai_platform_bootstrap.QLabel("fps"))
    self.lingbot_map_fps_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_map_fps_spin.setRange(1, 60)
    self.lingbot_map_fps_spin.setValue(10)
    self.lingbot_map_fps_spin.setToolTip("仅视频抽帧用")
    map_opt_row.addWidget(self.lingbot_map_fps_spin)
    self.lingbot_map_mask_sky_check = _eai_platform_bootstrap.QCheckBox("mask_sky")
    self.lingbot_map_mask_sky_check.setChecked(True)
    self.lingbot_map_mask_sky_check.setToolTip("室外场景建议开启（需 onnxruntime）")
    map_opt_row.addWidget(self.lingbot_map_mask_sky_check)
    self.lingbot_map_sdpa_check = _eai_platform_bootstrap.QCheckBox("use_sdpa")
    self.lingbot_map_sdpa_check.setChecked(True)
    self.lingbot_map_sdpa_check.setToolTip("无 FlashInfer 时请保持开启")
    map_opt_row.addWidget(self.lingbot_map_sdpa_check)
    map_opt_row.addWidget(_eai_platform_bootstrap.QLabel("采集帧"))
    self.lingbot_map_capture_spin = _eai_platform_bootstrap.QSpinBox()
    self.lingbot_map_capture_spin.setRange(8, 200)
    self.lingbot_map_capture_spin.setValue(24)
    self.lingbot_map_capture_spin.setToolTip("「当前相机采集」模式下抓取的帧数")
    map_opt_row.addWidget(self.lingbot_map_capture_spin)
    map_outer.addLayout(map_opt_row)

    map_run_row = _eai_platform_bootstrap.QHBoxLayout()
    map_run_row.setSpacing(6)
    self.lingbot_map_run_btn = _eai_platform_bootstrap.QPushButton("运行重建")
    self.lingbot_map_run_btn.setToolTip("启动 demo.py；完成后用浏览器打开 viser")
    self.lingbot_map_run_btn.clicked.connect(self._on_lingbot_map_run_clicked)
    map_run_row.addWidget(self.lingbot_map_run_btn)
    self.lingbot_map_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.lingbot_map_stop_btn.setObjectName("dangerAction")
    self.lingbot_map_stop_btn.setEnabled(False)
    self.lingbot_map_stop_btn.clicked.connect(self._on_lingbot_map_stop_clicked)
    map_run_row.addWidget(self.lingbot_map_stop_btn)
    self.lingbot_map_open_btn = _eai_platform_bootstrap.QPushButton("打开 viser")
    self.lingbot_map_open_btn.setEnabled(False)
    self.lingbot_map_open_btn.clicked.connect(self._on_lingbot_map_open_clicked)
    map_run_row.addWidget(self.lingbot_map_open_btn)
    self.lingbot_map_capture_btn = _eai_platform_bootstrap.QPushButton("拍一帧")
    self.lingbot_map_capture_btn.setToolTip("相机采集模式：把当前彩色帧写入缓存目录")
    self.lingbot_map_capture_btn.clicked.connect(self._on_lingbot_map_capture_clicked)
    map_run_row.addWidget(self.lingbot_map_capture_btn)
    self.lingbot_map_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.lingbot_map_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    map_run_row.addWidget(self.lingbot_map_status_label, 1)
    map_outer.addLayout(map_run_row)

    self.lingbot_map_url_label = _eai_platform_bootstrap.QLabel("")
    self.lingbot_map_url_label.setTextInteractionFlags(_eai_platform_bootstrap.Qt.TextSelectableByMouse)
    self.lingbot_map_url_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    map_outer.addWidget(self.lingbot_map_url_label)

    self.lingbot_map_preview_label = _eai_ui_widgets_images.ScaledPixmapLabel("输入预览（首帧）")
    self.lingbot_map_preview_label.setMinimumHeight(160)
    map_outer.addWidget(self.lingbot_map_preview_label, 1)

    self.lingbot_map_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.lingbot_map_log_edit.setReadOnly(True)
    self.lingbot_map_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.lingbot_map_log_edit.setPlaceholderText("LingBot-Map 日志…")
    self.lingbot_map_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.lingbot_map_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        map_outer,
        self.lingbot_map_log_edit,
        default_h=160,
        min_h=80,
        max_h=900,
        label="拖动调整 Map 日志高度",
        clearable=True,
    )

    self._lingbot_map_launcher = _eai_adapters_models_generation_launchers.LingbotMapLauncher(self)
    self._lingbot_map_launcher.log_line.connect(self._append_lingbot_map_log)
    self._lingbot_map_launcher.status_message.connect(self._on_lingbot_map_status)
    self._lingbot_map_launcher.running_changed.connect(self._update_lingbot_map_ui)
    self._lingbot_map_launcher.viser_ready.connect(self._on_lingbot_map_viser_ready)
    self._lingbot_map_capture_count = 0
    self._on_lingbot_map_scene_changed()

    build.control_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(map_tab), "3D重建模型")
