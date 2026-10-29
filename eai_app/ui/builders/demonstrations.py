"""ui.builders.demonstrations — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def build(self, build):
    demo_page = _eai_platform_bootstrap.QWidget()
    demo_page_layout = _eai_platform_bootstrap.QVBoxLayout(demo_page)
    demo_page_layout.setContentsMargins(0, 0, 0, 0)
    demo_page_layout.setSpacing(6)
    demo_group = _eai_platform_bootstrap.QGroupBox("人类演示模仿")
    demo_group.setStyleSheet(
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
        f"margin-top: 8px; padding-top: 6px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
        f"color: {_eai_configuration_policies.UI_ACCENT_BLUE}; }}"
    )
    demo_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    demo_layout = _eai_platform_bootstrap.QVBoxLayout(demo_group)
    demo_layout.setContentsMargins(8, 10, 8, 8)
    demo_layout.setSpacing(6)
    demo_hint = _eai_platform_bootstrap.QLabel(
        "选视频 → 分析 → 执行/场景对齐/跑策略。轨迹文本会写入上方文本指挥输入框。"
    )
    demo_hint.setWordWrap(True)
    demo_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    demo_layout.addWidget(demo_hint)

    demo_path_row = _eai_platform_bootstrap.QHBoxLayout()
    demo_path_row.setSpacing(6)
    self.human_demo_select_btn = _eai_platform_bootstrap.QPushButton("选择视频")
    self.human_demo_select_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_select_btn.setToolTip(
        "选择人类演示 mp4/avi/mkv/webm/mov。\n"
        f"分析结果缓存到 {_eai_configuration_policies.HUMAN_DEMO_CACHE_DIR}"
    )
    self.human_demo_select_btn.clicked.connect(self._on_human_demo_select_clicked)
    demo_path_row.addWidget(self.human_demo_select_btn)
    self.human_demo_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.human_demo_path_edit.setReadOnly(True)
    self.human_demo_path_edit.setPlaceholderText("人类演示视频路径…")
    self.human_demo_path_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    demo_path_row.addWidget(self.human_demo_path_edit, 1)
    demo_layout.addLayout(demo_path_row)

    demo_opt_row = _eai_platform_bootstrap.QHBoxLayout()
    demo_opt_row.setSpacing(8)
    self.human_demo_mirror_check = _eai_platform_bootstrap.QCheckBox("镜像映射")
    self.human_demo_mirror_check.setChecked(True)
    self.human_demo_mirror_check.setToolTip(
        "开启：人物左手→机器人右手（面对演示更自然）"
    )
    demo_opt_row.addWidget(self.human_demo_mirror_check)
    self.human_demo_ctrl_left_check = _eai_platform_bootstrap.QCheckBox("控左手")
    self.human_demo_ctrl_left_check.setChecked(True)
    demo_opt_row.addWidget(self.human_demo_ctrl_left_check)
    self.human_demo_ctrl_right_check = _eai_platform_bootstrap.QCheckBox("控右手")
    self.human_demo_ctrl_right_check.setChecked(True)
    demo_opt_row.addWidget(self.human_demo_ctrl_right_check)
    self.human_demo_move_arm_check = _eai_platform_bootstrap.QCheckBox("驱动手臂")
    self.human_demo_move_arm_check.setChecked(True)
    self.human_demo_move_arm_check.setToolTip(
        "按手腕相对首帧位移移动 TCP；关闭则只复现手部开合"
    )
    demo_opt_row.addWidget(self.human_demo_move_arm_check)
    demo_opt_row.addWidget(_eai_platform_bootstrap.QLabel("尺度"))
    self.human_demo_scale_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.human_demo_scale_spin.setRange(0.05, 0.80)
    self.human_demo_scale_spin.setSingleStep(0.05)
    self.human_demo_scale_spin.setDecimals(2)
    self.human_demo_scale_spin.setValue(0.30)
    self.human_demo_scale_spin.setSuffix(" m")
    self.human_demo_scale_spin.setFixedWidth(90)
    self.human_demo_scale_spin.setToolTip(
        "图像归一化位移映射到 base_link 的尺度（全图宽约对应此米数）"
    )
    demo_opt_row.addWidget(self.human_demo_scale_spin)
    demo_opt_row.addStretch(1)
    demo_layout.addLayout(demo_opt_row)

    demo_ctrl_row = _eai_platform_bootstrap.QHBoxLayout()
    demo_ctrl_row.setSpacing(6)
    self.human_demo_analyze_btn = _eai_platform_bootstrap.QPushButton("分析")
    self.human_demo_analyze_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_analyze_btn.setToolTip(
        "用 MediaPipe Hands 离线分析视频，提取双手关节与手腕轨迹；\n"
        "完成后把轨迹压缩成文本指挥填入上方文本指挥输入框"
    )
    self.human_demo_analyze_btn.clicked.connect(self._on_human_demo_analyze_clicked)
    demo_ctrl_row.addWidget(self.human_demo_analyze_btn)
    self.human_demo_run_btn = _eai_platform_bootstrap.QPushButton("执行复现")
    self.human_demo_run_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_run_btn.setEnabled(False)
    self.human_demo_run_btn.setToolTip(
        "模式A：手腕相对轨迹（含 MediaPipe world 深度）驱动手/臂。\n"
        "仿真时写 gui_robot_cmd。"
    )
    self.human_demo_run_btn.clicked.connect(self._on_human_demo_run_clicked)
    demo_ctrl_row.addWidget(self.human_demo_run_btn)
    self.human_demo_align_btn = _eai_platform_bootstrap.QPushButton("场景对齐复现")
    self.human_demo_align_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_align_btn.setEnabled(False)
    self.human_demo_align_btn.setToolTip(
        "模式B：把手腕轨迹标定到 LIBERO 物体→抽屉。\n"
        "需先开匹配任务的「只开界面」，且 state 含 objects。"
    )
    self.human_demo_align_btn.clicked.connect(
        self._on_human_demo_scene_align_clicked
    )
    demo_ctrl_row.addWidget(self.human_demo_align_btn)
    self.human_demo_policy_btn = _eai_platform_bootstrap.QPushButton("用匹配任务跑策略")
    self.human_demo_policy_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_policy_btn.setToolTip(
        "模式C：按视频匹配 LIBERO 任务，部署 Pi 后跑 pi05_libero 单任务评测。"
    )
    self.human_demo_policy_btn.clicked.connect(
        self._on_human_demo_policy_clicked
    )
    demo_ctrl_row.addWidget(self.human_demo_policy_btn)
    self.human_demo_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.human_demo_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.human_demo_stop_btn.setObjectName("dangerAction")
    self.human_demo_stop_btn.setEnabled(False)
    self.human_demo_stop_btn.clicked.connect(self._on_human_demo_stop_clicked)
    demo_ctrl_row.addWidget(self.human_demo_stop_btn)
    demo_ctrl_row.addStretch(1)
    demo_layout.addLayout(demo_ctrl_row)

    demo_status_row = _eai_platform_bootstrap.QHBoxLayout()
    demo_status_row.setSpacing(8)
    self.human_demo_status_label = _eai_platform_bootstrap.QLabel("演示: 未选择")
    self.human_demo_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
    self.human_demo_status_label.setWordWrap(True)
    demo_status_row.addWidget(self.human_demo_status_label, 1)
    self.human_demo_match_label = _eai_platform_bootstrap.QLabel("匹配任务: —")
    self.human_demo_match_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.human_demo_match_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.human_demo_match_label.setWordWrap(True)
    demo_status_row.addWidget(self.human_demo_match_label, 1)
    demo_layout.addLayout(demo_status_row)

    self.human_demo_preview_label = _eai_platform_bootstrap.QLabel("分析/执行时显示带骨架的预览")
    self.human_demo_preview_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.human_demo_preview_label.setMinimumHeight(420)
    self.human_demo_preview_label.setMinimumWidth(520)
    self.human_demo_preview_label.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    self.human_demo_preview_label.setScaledContents(False)
    self.human_demo_preview_label.setStyleSheet(
        f"QLabel {{ background-color: #111; color: {_eai_configuration_policies.UI_TEXT_MUTED}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    demo_layout.addWidget(self.human_demo_preview_label, 1)
    demo_group.setMinimumHeight(320)
    demo_page_layout.addWidget(demo_group, 1)
    self._arm_panel_page_by_key["human_demo"] = demo_page
