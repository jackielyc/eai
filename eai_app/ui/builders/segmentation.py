"""ui.builders.segmentation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launchers = module_ref('eai_app.adapters.robot.launchers')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    segment_tab = _eai_platform_bootstrap.QWidget()
    segment_outer = _eai_platform_bootstrap.QVBoxLayout(segment_tab)
    segment_outer.setContentsMargins(8, 6, 8, 6)
    segment_outer.setSpacing(6)
    segment_layout = _eai_platform_bootstrap.QHBoxLayout()
    segment_layout.setSpacing(6)
    segment_layout.addWidget(_eai_platform_bootstrap.QLabel("分割"))
    self.segment_backend_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.segment_backend_combo.addItem("几何(深度+颜色)", _eai_configuration_perception.SAM3_BACKEND_GEOMETRY)
    self.segment_backend_combo.addItem("SAM3 点提示", _eai_configuration_perception.SAM3_BACKEND_POINT)
    self.segment_backend_combo.addItem("SAM3 文本", _eai_configuration_perception.SAM3_BACKEND_TEXT)
    self.segment_backend_combo.setToolTip(
        "SAM3 需单独环境安装 ultralytics 并下载 sam3.pt；"
        "推荐先启动 sam3_segment_worker.py --serve"
    )
    self.segment_backend_combo.currentIndexChanged.connect(
        self._on_segment_settings_changed
    )
    segment_layout.addWidget(self.segment_backend_combo)
    self.sam3_text_edit = _eai_platform_bootstrap.QLineEdit()
    self.sam3_text_edit.setPlaceholderText("SAM3 文本概念，如 red cup")
    self.sam3_text_edit.setMinimumWidth(160)
    self.sam3_text_edit.setToolTip("仅在「SAM3 文本」模式下使用")
    self.sam3_text_edit.textChanged.connect(self._on_segment_settings_changed)
    segment_layout.addWidget(self.sam3_text_edit, 1)
    self.sam3_http_check = _eai_platform_bootstrap.QCheckBox("HTTP")
    self.sam3_http_check.setChecked(_eai_configuration_perception.SAM3_USE_HTTP_DEFAULT)
    self.sam3_http_check.setToolTip(
        f"勾选后走 SAM3 常驻服务 ({_eai_configuration_perception.SAM3_SERVER_URL_DEFAULT})，否则每次 subprocess 调 worker"
    )
    self.sam3_http_check.toggled.connect(self._on_segment_settings_changed)
    segment_layout.addWidget(self.sam3_http_check)
    self.sam3_status_label = _eai_platform_bootstrap.QLabel("SAM3: --")
    self.sam3_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sam3_status_label.setToolTip("SAM3 HTTP 服务健康检查")
    segment_layout.addWidget(self.sam3_status_label)
    self._local_ai_launcher = _eai_adapters_robot_launchers.LocalAiLauncher(self)
    self.local_ai_deploy_btn = _eai_platform_bootstrap.QPushButton("本地部署 AI")
    self.local_ai_deploy_btn.setToolTip(
        "启动本地 SAM3 分割服务 (run_sam3.sh) 与 Ollama 对话服务。\n"
        "SAM3: 自动勾选 HTTP，分割模式切到 SAM3 点提示。\n"
        "Ollama: 自动切换 AI 对话为本地 Qwen 预设。\n"
        "Docker 内 viewer 时请在宿主机执行: bash run_sam3.sh --host 0.0.0.0"
    )
    self.local_ai_deploy_btn.clicked.connect(self._on_local_ai_deploy_clicked)
    segment_layout.addWidget(self.local_ai_deploy_btn)
    segment_outer.addLayout(segment_layout)

    sam3_invoke_row = _eai_platform_bootstrap.QHBoxLayout()
    sam3_invoke_row.setSpacing(6)
    sam3_invoke_row.addWidget(_eai_platform_bootstrap.QLabel("提示点"))
    self.sam3_u_spin = _eai_platform_bootstrap.QSpinBox()
    self.sam3_u_spin.setRange(0, 9999)
    self.sam3_u_spin.setToolTip("提示点横坐标 u（点击图像只更新此点，不自动分割）")
    sam3_invoke_row.addWidget(_eai_platform_bootstrap.QLabel("u"))
    sam3_invoke_row.addWidget(self.sam3_u_spin)
    self.sam3_v_spin = _eai_platform_bootstrap.QSpinBox()
    self.sam3_v_spin.setRange(0, 9999)
    self.sam3_v_spin.setToolTip("提示点纵坐标 v（点击图像只更新此点，不自动分割）")
    sam3_invoke_row.addWidget(_eai_platform_bootstrap.QLabel("v"))
    sam3_invoke_row.addWidget(self.sam3_v_spin)
    self.stereo_invoke_btn = _eai_platform_bootstrap.QPushButton("调用分割")
    self.stereo_invoke_btn.setEnabled(False)
    self.stereo_invoke_btn.setToolTip("请先点击彩色/深度图像选择提示点")
    self.stereo_invoke_btn.clicked.connect(self._on_stereo_invoke_clicked)
    sam3_invoke_row.addWidget(self.stereo_invoke_btn)
    self.sam3_invoke_btn = _eai_platform_bootstrap.QPushButton("调用 SAM3")
    self.sam3_invoke_btn.setEnabled(False)
    self.sam3_invoke_btn.setToolTip(
        "需有彩色图像；点提示模式下还需先点击图像选点"
    )
    self.sam3_invoke_btn.clicked.connect(self._on_sam3_invoke_clicked)
    sam3_invoke_row.addWidget(self.sam3_invoke_btn)
    sam3_invoke_row.addStretch()
    segment_outer.addLayout(sam3_invoke_row)

    self.sam3_result_edit = _eai_platform_bootstrap.QTextEdit()
    self.sam3_result_edit.setReadOnly(True)
    self.sam3_result_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    _eai_ui_widgets_layouts._configure_resizable_text_edit(self.sam3_result_edit, default_h=140, tooltip="")
    self.sam3_result_edit.setPlaceholderText("SAM3 调用结果将显示在这里…")
    self.sam3_result_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    sam3_result_row = _eai_platform_bootstrap.QHBoxLayout()
    sam3_result_row.setSpacing(8)
    sam3_result_row.addWidget(self.sam3_result_edit, 1)
    self.sam3_preview_label = _eai_platform_bootstrap.QLabel("分割预览")
    self.sam3_preview_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.sam3_preview_label.setMinimumSize(160, 140)
    self.sam3_preview_label.setMaximumSize(240, 220)
    self.sam3_preview_label.setStyleSheet(
        f"QLabel {{ color: #888; background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.sam3_preview_label.setToolTip("分割 mask 叠加预览（相机画面上也会持续绘制）")
    sam3_result_row.addWidget(self.sam3_preview_label)
    segment_outer.addLayout(sam3_result_row)
    self.sam3_result_resize_bar = None

    pose_row = _eai_platform_bootstrap.QHBoxLayout()
    pose_row.setSpacing(6)
    pose_row.addWidget(_eai_platform_bootstrap.QLabel("位姿"))
    self.pose_backend_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.pose_backend_combo.addItem("PCA(点云)", _eai_configuration_perception.POSE_BACKEND_PCA)
    self.pose_backend_combo.addItem("FoundationPose", _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE)
    self.pose_backend_combo.addItem("Any6D（无模型时为凸包近似）", _eai_configuration_perception.POSE_BACKEND_ANY6D)
    self.pose_backend_combo.setToolTip(
        "PCA：点云几何；FoundationPose：需 CAD mesh；\n"
        "Any6D：任意物体，mesh 可空（深度凸包），推荐 run_any6d.sh"
    )
    self.pose_backend_combo.currentIndexChanged.connect(self._on_pose_settings_changed)
    pose_row.addWidget(self.pose_backend_combo)
    self.fp_mesh_edit = _eai_platform_bootstrap.QLineEdit()
    self.fp_mesh_edit.setPlaceholderText("物体 mesh (.obj)；Any6D 可留空")
    self.fp_mesh_edit.setMinimumWidth(180)
    self.fp_mesh_edit.setText(_eai_configuration_models.FP_MESH_DEFAULT)
    self.fp_mesh_edit.textChanged.connect(self._on_pose_settings_changed)
    pose_row.addWidget(self.fp_mesh_edit, 1)
    self.fp_mesh_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.fp_mesh_browse_btn.setFixedWidth(28)
    self.fp_mesh_browse_btn.setToolTip("选择 mesh 文件（Any6D 可选）")
    self.fp_mesh_browse_btn.clicked.connect(self._on_fp_mesh_browse_clicked)
    pose_row.addWidget(self.fp_mesh_browse_btn)
    self.fp_http_check = _eai_platform_bootstrap.QCheckBox("HTTP")
    self.fp_http_check.setChecked(_eai_configuration_perception.FP_USE_HTTP_DEFAULT)
    self.fp_http_check.setToolTip(
        f"FP→{_eai_configuration_perception.FP_SERVER_URL_DEFAULT}；Any6D→{_eai_configuration_perception.ANY6D_SERVER_URL_DEFAULT}"
    )
    self.fp_http_check.toggled.connect(self._on_pose_settings_changed)
    pose_row.addWidget(self.fp_http_check)
    self.fp_status_label = _eai_platform_bootstrap.QLabel("位姿: --")
    self.fp_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.fp_status_label.setMinimumWidth(72)
    pose_row.addWidget(self.fp_status_label)
    self.fp_avail_detail_label = _eai_platform_bootstrap.QLabel("")
    self.fp_avail_detail_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.fp_avail_detail_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.fp_avail_detail_label.setToolTip("位姿后端可用性")
    pose_row.addWidget(self.fp_avail_detail_label, 1)
    self.fp_invoke_btn = _eai_platform_bootstrap.QPushButton("调用位姿")
    self.fp_invoke_btn.setToolTip(
        "对当前彩色+深度调用所选后端（PCA/FP/Any6D）。\n"
        "Any6D：mesh 可留空=任意物体（深度凸包）；有 mesh 更准。\n"
        "服务: bash run_foundationpose.sh 或 bash run_any6d.sh"
    )
    self.fp_invoke_btn.clicked.connect(self._on_fp_invoke_clicked)
    pose_row.addWidget(self.fp_invoke_btn)
    segment_outer.addLayout(pose_row)

    self.fp_result_edit = _eai_platform_bootstrap.QTextEdit()
    self.fp_result_edit.setReadOnly(True)
    self.fp_result_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.fp_result_edit.setPlaceholderText("FoundationPose 调用结果将显示在这里…")
    self.fp_result_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.fp_result_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        segment_outer,
        self.fp_result_edit,
        default_h=140,
        min_h=80,
        max_h=900,
        label="拖动调整 FP 结果高度",
    )

    click_row = _eai_platform_bootstrap.QHBoxLayout()
    self.click_pose_check = _eai_platform_bootstrap.QCheckBox("点击估计（仅展示）")
    self.click_pose_check.setChecked(True)
    self.click_pose_check.setToolTip("全分辨率同步 RGB-D；不会回填或自动执行机械臂目标")
    click_row.addWidget(self.click_pose_check)
    self.click_pose_calibration_edit = _eai_platform_bootstrap.QLineEdit(_eai_platform_bootstrap.os.environ.get(
        "EAI_CLICK_CALIBRATION", _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config", "click_pose_calibration.json")))
    self.click_pose_calibration_edit.setPlaceholderText("真实相机标定 JSON 路径")
    click_row.addWidget(self.click_pose_calibration_edit, 1)
    segment_outer.addLayout(click_row)
    self.click_pose_result_edit = _eai_platform_bootstrap.QTextEdit()
    self.click_pose_result_edit.setReadOnly(True)
    self.click_pose_result_edit.setMaximumHeight(220)
    self.click_pose_result_edit.setPlaceholderText("点击彩色图估计物体位姿；PCA 为几何预览，SAM3/FP/Any6D 使用上方配置。")
    segment_outer.addWidget(self.click_pose_result_edit)
    build.control_tabs.addTab(segment_tab, "分割")
