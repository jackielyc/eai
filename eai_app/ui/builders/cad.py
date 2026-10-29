"""ui.builders.cad — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_training = module_ref('eai_app.adapters.models.training')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    cad_tab = _eai_platform_bootstrap.QWidget()
    cad_outer = _eai_platform_bootstrap.QVBoxLayout(cad_tab)
    cad_outer.setContentsMargins(8, 6, 8, 6)
    cad_outer.setSpacing(6)

    cad_mode_row = _eai_platform_bootstrap.QHBoxLayout()
    cad_mode_row.setSpacing(6)
    cad_mode_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.cad_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.cad_mode_combo.addItem("多视角照片", "photos")
    self.cad_mode_combo.addItem("导入 Mesh", "mesh")
    self.cad_mode_combo.setToolTip(
        "照片：COLMAP + Poisson 重建（需本机 colmap）。\n"
        "导入：Meshroom / 扫描 / 已有 .obj/.ply 后处理。"
    )
    self.cad_mode_combo.currentIndexChanged.connect(self._on_cad_mode_changed)
    cad_mode_row.addWidget(self.cad_mode_combo)
    cad_mode_row.addWidget(_eai_platform_bootstrap.QLabel("名称"))
    self.cad_name_edit = _eai_platform_bootstrap.QLineEdit("my_object")
    self.cad_name_edit.setMinimumWidth(120)
    self.cad_name_edit.setToolTip("输出到 meshes/<名称>/reconstructed.obj")
    self.cad_name_edit.editingFinished.connect(self._sync_cad_capture_from_disk)
    cad_mode_row.addWidget(self.cad_name_edit)
    cad_mode_row.addWidget(_eai_platform_bootstrap.QLabel("边长(m)"))
    self.cad_extent_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.cad_extent_spin.setRange(0.01, 2.0)
    self.cad_extent_spin.setSingleStep(0.01)
    self.cad_extent_spin.setDecimals(3)
    self.cad_extent_spin.setValue(_eai_configuration_policies.CAD_TARGET_EXTENT_DEFAULT_M)
    self.cad_extent_spin.setFixedWidth(80)
    self.cad_extent_spin.setToolTip("后处理：mesh 最大边长缩放到该值（米）")
    cad_mode_row.addWidget(self.cad_extent_spin)
    cad_mode_row.addWidget(_eai_platform_bootstrap.QLabel("Poisson"))
    self.cad_poisson_spin = _eai_platform_bootstrap.QSpinBox()
    self.cad_poisson_spin.setRange(6, 11)
    self.cad_poisson_spin.setValue(_eai_configuration_policies.CAD_POISSON_DEPTH_DEFAULT)
    self.cad_poisson_spin.setFixedWidth(52)
    self.cad_poisson_spin.setToolTip("仅照片模式：越大越细越慢")
    cad_mode_row.addWidget(self.cad_poisson_spin)
    cad_mode_row.addStretch()
    cad_outer.addLayout(cad_mode_row)

    cad_src_row = _eai_platform_bootstrap.QHBoxLayout()
    cad_src_row.setSpacing(6)
    self.cad_src_label = _eai_platform_bootstrap.QLabel("照片目录:")
    cad_src_row.addWidget(self.cad_src_label)
    self.cad_src_edit = _eai_platform_bootstrap.QLineEdit()
    self.cad_src_edit.setPlaceholderText("选择多视角照片目录，或导入 .obj/.ply")
    self.cad_src_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    cad_src_row.addWidget(self.cad_src_edit, 1)
    self.cad_src_browse_btn = _eai_platform_bootstrap.QPushButton("选择…")
    self.cad_src_browse_btn.setToolTip("选择照片目录或 mesh 文件")
    self.cad_src_browse_btn.clicked.connect(self._on_cad_src_browse_clicked)
    cad_src_row.addWidget(self.cad_src_browse_btn)
    cad_outer.addLayout(cad_src_row)

    cad_capture_row = _eai_platform_bootstrap.QHBoxLayout()
    cad_capture_row.setSpacing(6)
    cad_capture_row.addWidget(_eai_platform_bootstrap.QLabel("采集"))
    self.cad_capture_target_spin = _eai_platform_bootstrap.QSpinBox()
    self.cad_capture_target_spin.setRange(_eai_configuration_policies.CAD_MIN_IMAGES_DEFAULT, 48)
    self.cad_capture_target_spin.setValue(_eai_configuration_policies.CAD_CAPTURE_COUNT_DEFAULT)
    self.cad_capture_target_spin.setFixedWidth(52)
    self.cad_capture_target_spin.setToolTip("目标张数；绕物体换视角后逐张拍摄")
    self.cad_capture_target_spin.valueChanged.connect(self._update_cad_capture_btn)
    cad_capture_row.addWidget(self.cad_capture_target_spin)
    self.cad_capture_btn = _eai_platform_bootstrap.QPushButton("拍一张 (0/12)")
    self.cad_capture_btn.setToolTip(
        f"固定使用头部 RGB-D：{_eai_configuration_policies.CAD_CAPTURE_TOPIC_DEFAULT} + {_eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT}\n"
        "保存彩色+深度到 meshes/<名称>/photos/，并用 TSDF 生成 CAD。\n"
        "重要：每拍一张请明显转动物体（或移动相机），相邻帧不能几乎一样。"
    )
    self.cad_capture_btn.clicked.connect(self._on_cad_capture_clicked)
    cad_capture_row.addWidget(self.cad_capture_btn)
    self.cad_capture_reset_btn = _eai_platform_bootstrap.QPushButton("清空采集")
    self.cad_capture_reset_btn.setToolTip("清空当前物体的 photos 目录与计数")
    self.cad_capture_reset_btn.clicked.connect(self._on_cad_capture_reset_clicked)
    cad_capture_row.addWidget(self.cad_capture_reset_btn)
    self.cad_auto_gen_check = _eai_platform_bootstrap.QCheckBox("满额后自动生成")
    self.cad_auto_gen_check.setChecked(True)
    self.cad_auto_gen_check.setToolTip("拍满目标张数后自动启动 RGB-D / COLMAP 重建")
    cad_capture_row.addWidget(self.cad_auto_gen_check)
    self.cad_capture_hint = _eai_platform_bootstrap.QLabel(
        "头部 RGB-D：每拍一张请转动物体，视角差太小会被拒绝"
    )
    self.cad_capture_hint.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.cad_capture_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    cad_capture_row.addWidget(self.cad_capture_hint, 1)
    cad_outer.addLayout(cad_capture_row)

    cad_action_row = _eai_platform_bootstrap.QHBoxLayout()
    cad_action_row.setSpacing(6)
    self.cad_status_label = _eai_platform_bootstrap.QLabel("CAD: 空闲")
    self.cad_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.cad_status_label.setToolTip(
        f"输出目录: {_eai_configuration_policies.CAD_MESHES_DIR}\n"
        f"Python: {_eai_configuration_policies.resolve_cad_mesh_python()}\n"
        "照片重建需安装 COLMAP"
    )
    cad_action_row.addWidget(self.cad_status_label, 1)
    self.cad_existing_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.cad_existing_combo.setMinimumWidth(160)
    self.cad_existing_combo.setToolTip("已有 meshes/<name>/reconstructed.obj")
    cad_action_row.addWidget(self.cad_existing_combo)
    self.cad_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.cad_refresh_btn.clicked.connect(self._refresh_cad_mesh_combo)
    cad_action_row.addWidget(self.cad_refresh_btn)
    self.cad_apply_fp_btn = _eai_platform_bootstrap.QPushButton("应用到 FP")
    self.cad_apply_fp_btn.setToolTip("将所选/刚生成的 mesh 填入「分割」页 FoundationPose mesh 路径")
    self.cad_apply_fp_btn.clicked.connect(self._on_cad_apply_fp_clicked)
    cad_action_row.addWidget(self.cad_apply_fp_btn)
    self.cad_start_btn = _eai_platform_bootstrap.QPushButton("生成 CAD")
    self.cad_start_btn.setToolTip("开始照片重建或 mesh 后处理")
    self.cad_start_btn.clicked.connect(self._on_cad_start_clicked)
    cad_action_row.addWidget(self.cad_start_btn)
    self.cad_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.cad_stop_btn.setObjectName("dangerAction")
    self.cad_stop_btn.setEnabled(False)
    self.cad_stop_btn.clicked.connect(self._on_cad_stop_clicked)
    cad_action_row.addWidget(self.cad_stop_btn)
    self.cad_clear_log_btn = _eai_platform_bootstrap.QPushButton("清空日志")
    self.cad_clear_log_btn.clicked.connect(self._on_cad_clear_log_clicked)
    cad_action_row.addWidget(self.cad_clear_log_btn)
    cad_outer.addLayout(cad_action_row)

    self.cad_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.cad_log_edit.setReadOnly(True)
    self.cad_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.cad_log_edit.setPlaceholderText(
        "CAD 重建日志…\n"
        "建议：≥12 张照片、相邻约 60% 重叠；哑光非透明物体更稳。"
    )
    self.cad_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.cad_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        cad_outer,
        self.cad_log_edit,
        default_h=180,
        min_h=100,
        max_h=900,
        label="拖动调整 CAD 日志高度",
    )

    self._cad_launcher = _eai_adapters_models_training.CadMeshLauncher(self)
    self._on_cad_mode_changed()
    self._refresh_cad_mesh_combo()
    self._sync_cad_capture_from_disk()
    build.control_tabs.addTab(cad_tab, "CAD")
