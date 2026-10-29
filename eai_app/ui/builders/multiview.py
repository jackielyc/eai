"""ui.builders.multiview — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    mv6d_page = _eai_platform_bootstrap.QWidget()
    mv6d_page_layout = _eai_platform_bootstrap.QVBoxLayout(mv6d_page)
    mv6d_page_layout.setContentsMargins(0, 0, 0, 0)
    mv6d_page_layout.setSpacing(6)
    mv6d_group = _eai_platform_bootstrap.QGroupBox("多视角 6D 位姿（头+腕 RGB-D）")
    mv6d_group.setStyleSheet(
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"margin-top: 8px; padding-top: 6px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE}; }}"
    )
    mv6d_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    mv6d_layout = _eai_platform_bootstrap.QVBoxLayout(mv6d_group)
    mv6d_layout.setContentsMargins(8, 10, 8, 8)
    mv6d_layout.setSpacing(4)
    mv6d_hint = _eai_platform_bootstrap.QLabel(
        "先在「头部」预览图上点击物体，再点「估计一次」。"
        "PCA、FoundationPose、Any6D 都会先用这个点做 SAM3 分割。"
        "左/右腕预览可点击放大。"
    )
    mv6d_hint.setWordWrap(True)
    mv6d_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    mv6d_layout.addWidget(mv6d_hint)

    mv6d_row1 = _eai_platform_bootstrap.QHBoxLayout()
    mv6d_row1.setSpacing(6)
    self.mv6d_enable_check = _eai_platform_bootstrap.QCheckBox("实时估计")
    self.mv6d_enable_check.setToolTip(
        "开启后按间隔用头+腕 RGB-D 估计 6D，并刷新下方数字与图像叠加。"
        "若同时勾选 LingBot-Depth，每次估计较慢，建议加大间隔。"
    )
    self.mv6d_enable_check.toggled.connect(self._on_mv6d_enable_toggled)
    mv6d_row1.addWidget(self.mv6d_enable_check)
    self.mv6d_lingbot_depth_check = _eai_platform_bootstrap.QCheckBox("LingBot-Depth 补深度")
    self.mv6d_lingbot_depth_check.setToolTip(
        "勾选后先用 LingBot-Depth 精修各视角深度，再做 SAM3 分割和 PCA/FP/Any6D 位姿估计。\n"
        "有原始深度时精修；无深度时用稀疏先验补全，尺度依赖先验。失败则停止本轮。\n"
        "模型跟随「空间感知模型」页配置；实时估计时会明显变慢。"
    )
    mv6d_row1.addWidget(self.mv6d_lingbot_depth_check)
    self.mv6d_fill_cmd_check = _eai_platform_bootstrap.QCheckBox("回填文本指挥")
    self.mv6d_fill_cmd_check.setChecked(True)
    self.mv6d_fill_cmd_check.setToolTip(
        "估计成功后把物体 XYZ（必须已在 base_link，或可由 TF 变换过去）\n"
        "写成「左/右臂移到 x=… y=… z=…」填入文本指挥；点「执行」绝对移动。\n"
        "若位姿仍在相机系（无 TF），不会回填，避免手臂越走越远。\n"
        "手臂侧跟随腕相机：左腕→左臂，右腕→右臂。实时估计不覆盖输入框。"
    )
    mv6d_row1.addWidget(self.mv6d_fill_cmd_check)
    self.mv6d_once_btn = _eai_platform_bootstrap.QPushButton("估计一次")
    self.mv6d_once_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_once_btn.setToolTip(
        "请先在头部预览图上点击选择物体。估计会先做 SAM3 分割，再按所选方法出 6D。"
    )
    self.mv6d_once_btn.clicked.connect(self._on_mv6d_estimate_once)
    mv6d_row1.addWidget(self.mv6d_once_btn)
    mv6d_row1.addWidget(self.lingbot_depth_run_btn)
    mv6d_row1.addWidget(_eai_platform_bootstrap.QLabel("腕相机"))
    self.mv6d_wrist_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    # data 用逻辑侧 "left"/"right"；实际 topic 运行时按 hand_* / left_wrist_* 解析
    self.mv6d_wrist_combo.addItem("左腕", "left")
    self.mv6d_wrist_combo.addItem("右腕", "right")
    self.mv6d_wrist_combo.setToolTip(
        "参与头+腕融合估计的腕相机；下方预览同时显示左腕与右腕。\n"
        "真机常见 /camera/hand_left_color、/camera/hand_right_color；"
        "仿真为 /camera/left_wrist_color、/camera/right_wrist_color。"
    )
    mv6d_row1.addWidget(self.mv6d_wrist_combo)
    mv6d_row1.addWidget(_eai_platform_bootstrap.QLabel("间隔"))
    self.mv6d_interval_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.mv6d_interval_spin.setRange(0.2, 5.0)
    self.mv6d_interval_spin.setSingleStep(0.1)
    self.mv6d_interval_spin.setDecimals(1)
    self.mv6d_interval_spin.setValue(0.5)
    self.mv6d_interval_spin.setSuffix(" s")
    self.mv6d_interval_spin.setFixedWidth(80)
    self.mv6d_interval_spin.valueChanged.connect(self._on_mv6d_interval_changed)
    mv6d_row1.addWidget(self.mv6d_interval_spin)
    self.mv6d_sync_seed_btn = _eai_platform_bootstrap.QPushButton("同步分割点")
    self.mv6d_sync_seed_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_sync_seed_btn.setToolTip("从「分割」页 SAM/提示点 (u,v) 同步到此处")
    self.mv6d_sync_seed_btn.clicked.connect(self._on_mv6d_sync_seed_from_segment)
    mv6d_row1.addWidget(self.mv6d_sync_seed_btn)
    self.mv6d_overlay_check = _eai_platform_bootstrap.QCheckBox("叠加到预览")
    self.mv6d_overlay_check.setChecked(True)
    self.mv6d_overlay_check.setToolTip(
        "估计完成后叠加到手臂页头/腕预览（不写主相机面板，避免每帧重绘卡顿）"
    )
    self.mv6d_overlay_check.toggled.connect(self._on_mv6d_overlay_toggled)
    mv6d_row1.addWidget(self.mv6d_overlay_check)
    self.mv6d_hand_skel_check = _eai_platform_bootstrap.QCheckBox("手骨架")
    self.mv6d_hand_skel_check.setChecked(True)
    self.mv6d_hand_skel_check.setToolTip(
        "用左右手腕 TCP 和睿研手模型，把同一套骨架投到头部和两只腕相机。\n"
        "能同时投到至少两个已标定相机时，才算联合显示。\n"
        "末节角度用开合比例，不是实测关节角。"
    )
    self.mv6d_hand_skel_check.toggled.connect(
        lambda _checked: self._refresh_mv6d_live_previews_for_pick()
    )
    mv6d_row1.addWidget(self.mv6d_hand_skel_check)
    self.mv6d_enlarge_btn = _eai_platform_bootstrap.QPushButton("放大预览")
    self.mv6d_enlarge_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_enlarge_btn.setToolTip("弹出头+腕双视角大图（估计后会随结果刷新）")
    self.mv6d_enlarge_btn.clicked.connect(self._open_mv6d_lightbox)
    mv6d_row1.addWidget(self.mv6d_enlarge_btn)
    self.mv6d_rviz_btn = _eai_platform_bootstrap.QPushButton("打开 RViz")
    self.mv6d_rviz_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_rviz_btn.setToolTip(
        "启动 rviz2，并拉起 robot_description、/joint_states→TF、map→A2D。\n"
        "与本界面相同 ROS Domain / DDS；需要 host_fk 在发布关节角。"
    )
    self.mv6d_rviz_btn.clicked.connect(self._on_mv6d_open_rviz)
    mv6d_row1.addWidget(self.mv6d_rviz_btn)
    self.mv6d_robot_btn = _eai_platform_bootstrap.QPushButton("显示机器人")
    self.mv6d_robot_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_robot_btn.setToolTip(
        "在「手臂/手」里打开机器人页，用 host_fk 的关节角画模型。\n"
        "拖动旋转，右键平移，滚轮缩放。"
    )
    self.mv6d_robot_btn.clicked.connect(self._on_mv6d_show_robot)
    mv6d_row1.addWidget(self.mv6d_robot_btn)
    mv6d_row1.addStretch(1)
    mv6d_layout.addLayout(mv6d_row1)

    method_switch_row = _eai_platform_bootstrap.QHBoxLayout()
    method_switch_row.addWidget(_eai_platform_bootstrap.QLabel("启用方法"))
    for backend, build.label, build.attr in (
        (_eai_configuration_perception.POSE_BACKEND_PCA, "PCA", "mv6d_pca_enabled_check"),
        (_eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE, "FoundationPose", "mv6d_foundationpose_enabled_check"),
        (_eai_configuration_perception.POSE_BACKEND_ANY6D, "Any6D", "mv6d_any6d_enabled_check"),
    ):
        check = _eai_platform_bootstrap.QCheckBox(build.label)
        check.setChecked(True)
        check.setToolTip(f"开启后计算 {build.label} 位姿；关闭时跳过计算并清除其机器人页结果。")
        setattr(self, build.attr, check)
        check.toggled.connect(self._on_mv6d_methods_changed)
        method_switch_row.addWidget(check)
    method_switch_row.addStretch(1)
    mv6d_layout.addLayout(method_switch_row)

    mv6d_row2 = _eai_platform_bootstrap.QHBoxLayout()
    mv6d_row2.setSpacing(6)
    mv6d_row2.addWidget(_eai_platform_bootstrap.QLabel("结果来源"))
    self.mv6d_method_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.mv6d_method_combo.addItem("PCA", _eai_configuration_perception.POSE_BACKEND_PCA)
    self.mv6d_method_combo.addItem("FoundationPose", _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE)
    self.mv6d_method_combo.addItem("Any6D", _eai_configuration_perception.POSE_BACKEND_ANY6D)
    self.mv6d_method_combo.setToolTip(
        "只计算上方已开启的方法，共用 SAM3 分割并显示到机器人页。\n"
        "这里选中的方法写入结果文本和抓取目标。\n"
        "PCA：分割区域的点云包围盒。\n"
        "FoundPose：使用位姿设置里的物体 mesh。\n"
        "Any6D：没有 mesh 时用深度凸包。"
    )
    mv6d_row2.addWidget(self.mv6d_method_combo)
    mv6d_row2.addWidget(_eai_platform_bootstrap.QLabel("头提示点 u"))
    self.mv6d_head_u_spin = _eai_platform_bootstrap.QSpinBox()
    self.mv6d_head_u_spin.setRange(0, 9999)
    self.mv6d_head_u_spin.setValue(320)
    self.mv6d_head_u_spin.setToolTip(
        "头部彩色图像上的种子点 u；请在头部预览图上点击选择，或与「分割」页同步"
    )
    self.mv6d_head_u_spin.valueChanged.connect(self._on_mv6d_seed_spin_changed)
    mv6d_row2.addWidget(self.mv6d_head_u_spin)
    mv6d_row2.addWidget(_eai_platform_bootstrap.QLabel("v"))
    self.mv6d_head_v_spin = _eai_platform_bootstrap.QSpinBox()
    self.mv6d_head_v_spin.setRange(0, 9999)
    self.mv6d_head_v_spin.setValue(240)
    self.mv6d_head_v_spin.valueChanged.connect(self._on_mv6d_seed_spin_changed)
    mv6d_row2.addWidget(self.mv6d_head_v_spin)
    self.mv6d_seed_status_label = _eai_platform_bootstrap.QLabel("未选择物体 · 请点击头部预览")
    self.mv6d_seed_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
    self.mv6d_seed_status_label.setToolTip(
        "必须先在头部预览图上点选目标物体，才会开始 6D 估计"
    )
    mv6d_row2.addWidget(self.mv6d_seed_status_label, 1)
    mv6d_row2.addWidget(_eai_platform_bootstrap.QLabel("目标"))
    self.mv6d_target_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.mv6d_target_combo.setMinimumWidth(170)
    self.mv6d_target_combo.setToolTip("点击头部图像新增目标；选择目标可查看其点位和已完成的估计结果")
    self.mv6d_target_combo.currentIndexChanged.connect(self._on_mv6d_target_changed)
    mv6d_row2.addWidget(self.mv6d_target_combo)
    self.mv6d_remove_target_btn = _eai_platform_bootstrap.QPushButton("删除目标")
    self.mv6d_remove_target_btn.clicked.connect(self._remove_mv6d_target)
    mv6d_row2.addWidget(self.mv6d_remove_target_btn)
    self.mv6d_clear_targets_btn = _eai_platform_bootstrap.QPushButton("清空目标")
    self.mv6d_clear_targets_btn.clicked.connect(self._clear_mv6d_targets)
    mv6d_row2.addWidget(self.mv6d_clear_targets_btn)
    self.mv6d_hand_skel_label = _eai_platform_bootstrap.QLabel("手骨架：未投影")
    self.mv6d_hand_skel_label.setWordWrap(True)
    self.mv6d_hand_skel_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    mv6d_row2.addWidget(self.mv6d_hand_skel_label, 1)
    mv6d_layout.addLayout(mv6d_row2)
    self.mv6d_batch_summary_label = _eai_platform_bootstrap.QLabel('目标估计：尚未开始')
    self.mv6d_batch_summary_label.setWordWrap(True)
    self.mv6d_batch_summary_label.setTextInteractionFlags(_eai_platform_bootstrap.Qt.TextSelectableByMouse)
    self.mv6d_batch_summary_label.setStyleSheet(f"color:{_eai_configuration_policies.UI_TEXT_SECONDARY};")
    mv6d_layout.addWidget(self.mv6d_batch_summary_label)

    mv6d_error_row = _eai_platform_bootstrap.QHBoxLayout()
    mv6d_error_row.setSpacing(6)
    self.mv6d_error_label = _eai_platform_bootstrap.QLabel("误差: --")
    self.mv6d_error_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.mv6d_error_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
    self.mv6d_error_label.setWordWrap(True)
    self.mv6d_error_label.setTextInteractionFlags(
        _eai_platform_bootstrap.Qt.TextSelectableByMouse | _eai_platform_bootstrap.Qt.TextSelectableByKeyboard
    )
    self.mv6d_error_label.setFocusPolicy(_eai_platform_bootstrap.Qt.ClickFocus)
    self.mv6d_error_label.setCursor(_eai_platform_bootstrap.Qt.IBeamCursor)
    mv6d_error_row.addWidget(self.mv6d_error_label, 1)
    self.mv6d_copy_btn = _eai_platform_bootstrap.QPushButton("复制结果")
    self.mv6d_copy_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_copy_btn.setToolTip("把详情里的完整估计文本复制到剪贴板")
    self.mv6d_copy_btn.clicked.connect(self._on_mv6d_copy_result)
    mv6d_error_row.addWidget(self.mv6d_copy_btn)
    mv6d_layout.addLayout(mv6d_error_row)

    mv6d_detail = _eai_platform_bootstrap.QGroupBox("详情（参考位姿 / 结果文本）")
    mv6d_detail.setToolTip("结果框内可用滚轮、滚动条或 PageUp/PageDown 查看完整文本")
    mv6d_detail.setStyleSheet(
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"margin-top: 6px; padding-top: 4px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
    )
    mv6d_detail_layout = _eai_platform_bootstrap.QVBoxLayout(mv6d_detail)
    mv6d_detail_layout.setContentsMargins(6, 8, 6, 6)
    mv6d_detail_layout.setSpacing(4)

    mv6d_gt_row = _eai_platform_bootstrap.QHBoxLayout()
    mv6d_gt_row.setSpacing(4)
    self.mv6d_gt_enable_check = _eai_platform_bootstrap.QCheckBox("对比当前目标参考位姿")
    self.mv6d_gt_enable_check.setToolTip(
        "确认下方 XYZ/RPY 是当前目标的 base_link 参考位姿后启用。差值不是 SAM3 分割误差；"
        "也可「加载仿真GT」读 object_gt_pose.json"
    )
    self.mv6d_gt_enable_check.toggled.connect(self._on_mv6d_gt_params_changed)
    mv6d_gt_row.addWidget(self.mv6d_gt_enable_check)
    for axis, build.attr, build.tip in (
        ("X", "mv6d_gt_x_spin", "参考 X (m, base_link)"),
        ("Y", "mv6d_gt_y_spin", "参考 Y (m, base_link)"),
        ("Z", "mv6d_gt_z_spin", "参考 Z (m, base_link)"),
    ):
        mv6d_gt_row.addWidget(_eai_platform_bootstrap.QLabel(axis))
        build.spin = _eai_platform_bootstrap.QDoubleSpinBox()
        build.spin.setRange(-5.0, 5.0)
        build.spin.setDecimals(4)
        build.spin.setSingleStep(0.001)
        build.spin.setSuffix(" m")
        build.spin.setFixedWidth(92)
        build.spin.setToolTip(build.tip)
        build.spin.valueChanged.connect(self._on_mv6d_gt_params_changed)
        setattr(self, build.attr, build.spin)
        mv6d_gt_row.addWidget(build.spin)
    for axis, build.attr, build.tip in (
        ("R", "mv6d_gt_roll_spin", "参考 roll (°)"),
        ("P", "mv6d_gt_pitch_spin", "参考 pitch (°)"),
        ("Y", "mv6d_gt_yaw_spin", "参考 yaw (°)"),
    ):
        mv6d_gt_row.addWidget(_eai_platform_bootstrap.QLabel(axis))
        build.spin = _eai_platform_bootstrap.QDoubleSpinBox()
        build.spin.setRange(-180.0, 180.0)
        build.spin.setDecimals(2)
        build.spin.setSingleStep(0.5)
        build.spin.setSuffix("°")
        build.spin.setFixedWidth(80)
        build.spin.setToolTip(build.tip)
        build.spin.valueChanged.connect(self._on_mv6d_gt_params_changed)
        setattr(self, build.attr, build.spin)
        mv6d_gt_row.addWidget(build.spin)
    self.mv6d_load_gt_btn = _eai_platform_bootstrap.QPushButton("加载仿真GT")
    self.mv6d_load_gt_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.mv6d_load_gt_btn.setToolTip(
        "从仿真共享目录读取 object_gt_pose.json\n"
        '{"xyz":[x,y,z],"rpy_deg":[r,p,y]} 或 quat_xyzw'
    )
    self.mv6d_load_gt_btn.clicked.connect(self._on_mv6d_load_sim_gt)
    mv6d_gt_row.addWidget(self.mv6d_load_gt_btn)
    mv6d_gt_row.addStretch(1)
    mv6d_detail_layout.addLayout(mv6d_gt_row)

    self.mv6d_result_edit = _eai_platform_bootstrap.QTextEdit()
    self.mv6d_result_edit.setReadOnly(True)
    self.mv6d_result_edit.setTextInteractionFlags(
        _eai_platform_bootstrap.Qt.TextSelectableByMouse | _eai_platform_bootstrap.Qt.TextSelectableByKeyboard
    )
    self.mv6d_result_edit.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)
    self.mv6d_result_edit.setContextMenuPolicy(_eai_platform_bootstrap.Qt.DefaultContextMenu)
    self.mv6d_result_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.mv6d_result_edit.setPlaceholderText(
        "头+腕 6D 估计结果、参考位姿差值及诊断信息将显示在这里…"
    )
    self.mv6d_result_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; padding-right: 2px; "
        f"selection-background-color: #3d5a80; selection-color: #ffffff; }}"
        f"QTextEdit QScrollBar:vertical {{ background: {_eai_configuration_policies.UI_BG_WINDOW}; width: 14px; margin: 0; }}"
        f"QTextEdit QScrollBar::handle:vertical {{ background: {_eai_configuration_policies.UI_BORDER_STRONG}; "
        f"border-radius: 4px; min-height: 28px; }}"
        f"QTextEdit QScrollBar::handle:vertical:hover {{ background: {_eai_configuration_policies.UI_ACCENT_BLUE}; }}"
        f"QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical {{ height: 0; }}"
        f"QTextEdit QScrollBar::add-page:vertical, QTextEdit QScrollBar::sub-page:vertical {{ background: transparent; }}"
    )
    self.mv6d_result_edit.setMinimumHeight(160)
    self.mv6d_result_edit.setMaximumHeight(320)
    self.mv6d_result_edit.setVerticalScrollBarPolicy(_eai_platform_bootstrap.Qt.ScrollBarAlwaysOn)
    result_bar = self.mv6d_result_edit.verticalScrollBar()
    result_bar.setEnabled(True)
    result_bar.raise_()
    mv6d_detail_layout.addWidget(self.mv6d_result_edit)
    mv6d_layout.addWidget(mv6d_detail)

    mv6d_preview_row = _eai_platform_bootstrap.QHBoxLayout()
    mv6d_preview_row.setSpacing(6)
    self.mv6d_head_preview = _eai_ui_widgets_images.ScaledPixmapLabel("头部 · 点击选择物体", fill_frame=True)
    self.mv6d_left_wrist_preview = _eai_ui_widgets_images.ScaledPixmapLabel("左腕 · 点击放大", fill_frame=True)
    self.mv6d_right_wrist_preview = _eai_ui_widgets_images.ScaledPixmapLabel("右腕 · 点击放大", fill_frame=True)
    # 兼容旧引用：单腕预览别名指向左腕
    self.mv6d_wrist_preview = self.mv6d_left_wrist_preview
    self.mv6d_head_preview.setMinimumHeight(240)
    self.mv6d_head_preview.set_pick_pixel(True)
    self.mv6d_head_preview.setToolTip(
        "在头部画面上点击要估计的物体（提示点）；选好后再点「估计一次」"
    )
    self.mv6d_head_preview.setStyleSheet(
        f"QLabel {{ background-color: #111; color: {_eai_configuration_policies.UI_TEXT_MUTED}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.mv6d_head_preview.clicked_pixel.connect(self._on_mv6d_head_preview_picked)
    mv6d_preview_row.addWidget(self.mv6d_head_preview, 1)
    for build.lab, build.tip in (
        (self.mv6d_left_wrist_preview, "左腕 6D 渲染预览（点击放大）"),
        (self.mv6d_right_wrist_preview, "右腕 6D 渲染预览（点击放大）"),
    ):
        build.lab.setMinimumHeight(240)
        build.lab.set_clickable(True)
        build.lab.setToolTip(build.tip)
        build.lab.setStyleSheet(
            f"QLabel {{ background-color: #111; color: {_eai_configuration_policies.UI_TEXT_MUTED}; "
            f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
        )
        build.lab.clicked.connect(self._open_mv6d_lightbox)
        mv6d_preview_row.addWidget(build.lab, 1)
    mv6d_layout.addLayout(mv6d_preview_row, 1)

    self._mv6d_timer = _eai_platform_bootstrap.QTimer(self)
    self._mv6d_timer.setInterval(500)
    self._mv6d_timer.timeout.connect(self._on_mv6d_timer_tick)
    self._mv6d_busy = False
    self._mv6d_object_picked = False
    self._mv6d_targets: _eai_platform_bootstrap.List[dict] = []
    self._mv6d_target_results: _eai_platform_bootstrap.Dict[str, dict] = {}
    self._mv6d_target_serial = 0
    self._mv6d_estimate_queue: _eai_platform_bootstrap.List[str] = []
    self._mv6d_active_target: _eai_platform_bootstrap.Optional[dict] = None
    self._mv6d_batch_active = False
    self._mv6d_batch_capture = None
    self._mv6d_batch_depth_cache = None
    self._mv6d_last_snap: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.MultiView6DPoseSnapshot] = None
    self._mv6d_head_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
    self._mv6d_left_wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
    self._mv6d_right_wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
    self._mv6d_wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None  # 兼容：等同左腕
    self._mv6d_live_head_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    self._mv6d_lightbox: _eai_platform_bootstrap.Optional[_eai_ui_widgets_images.Mv6dLightboxWindow] = None
    self._mv6d_bridge = _eai_perception_segmentation.Mv6dCallBridge(self)
    self._mv6d_bridge.finished.connect(self._on_mv6d_estimate_finished)
    # 不在构建时刷预览（会卡启动）；切到本面板时再刷

    mv6d_page_layout.addWidget(mv6d_group, 1)
    self._arm_panel_page_by_key["mv6d"] = mv6d_page

    # 按默认/已保存顺序把子页塞进 stacked（index 与 TabBar 无关，按 key 切换）
    for build.key in _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS:
        page = self._arm_panel_page_by_key.get(build.key)
        if page is not None:
            self._arm_panel_stack.addWidget(page)
