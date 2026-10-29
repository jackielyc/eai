"""ui.builders.reward evaluation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_reward_launchers = module_ref('eai_app.adapters.models.reward_launchers')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.reward_tab = _eai_platform_bootstrap.QWidget()
    build.reward_tab.setObjectName("rewardTab")
    reward_outer = _eai_platform_bootstrap.QVBoxLayout(build.reward_tab)
    reward_outer.setContentsMargins(8, 6, 8, 6)
    reward_outer.setSpacing(6)
    reward_hint = _eai_platform_bootstrap.QLabel(
        "RLinf 具身 Reward Model 评测：单帧打分 / A-B 偏好 / 批量 .pt 指标 / "
        "真机 teleop 在线打分。需本地 RLinf 仓库与权重；真机模式还需按 "
        "realworld_teleop.yaml 配好 Franka / SpaceMouse。"
    )
    reward_hint.setWordWrap(True)
    reward_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    reward_outer.addWidget(reward_hint)

    build.rm_path_row = _eai_platform_bootstrap.QHBoxLayout()
    build.rm_path_row.setSpacing(6)
    build.rm_path_row.addWidget(_eai_platform_bootstrap.QLabel("RLinf"))
    self.reward_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("RLINF_ROOT", _eai_configuration_tabs.RLINF_ROOT_DEFAULT)
    )
    self.reward_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_path_row.addWidget(self.reward_root_edit, 1)
    self.reward_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.reward_root_browse_btn.setFixedWidth(28)
    self.reward_root_browse_btn.clicked.connect(self._on_reward_root_browse)
    build.rm_path_row.addWidget(self.reward_root_browse_btn)
    build.rm_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.reward_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_rlinf_python(self.reward_root_edit.text())
    )
    self.reward_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_path_row.addWidget(self.reward_python_edit, 1)
    reward_outer.addLayout(build.rm_path_row)

    rm_model_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_model_row.setSpacing(6)
    rm_model_row.addWidget(_eai_platform_bootstrap.QLabel("类型"))
    self.reward_type_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_type_combo.addItem("resnet", "resnet")
    self.reward_type_combo.addItem("vlm", "vlm")
    # 默认用本地 Qwen3-VL 做零样本 VLM reward 实验
    build._default_reward_ckpt = _eai_platform_bootstrap.os.path.join(
        _eai_configuration_tabs.RLINF_REWARD_CACHE_DIR, "Qwen3-VL-4B-Instruct"
    )
    if _eai_platform_bootstrap.os.path.isdir(build._default_reward_ckpt):
        build.idx = self.reward_type_combo.findData("vlm")
        if build.idx >= 0:
            self.reward_type_combo.setCurrentIndex(build.idx)
    rm_model_row.addWidget(self.reward_type_combo)
    rm_model_row.addWidget(_eai_platform_bootstrap.QLabel("arch"))
    self.reward_arch_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for arch in ("resnet18", "resnet34", "resnet50"):
        self.reward_arch_combo.addItem(arch, arch)
    rm_model_row.addWidget(self.reward_arch_combo)
    rm_model_row.addWidget(_eai_platform_bootstrap.QLabel("size"))
    self.reward_image_size_edit = _eai_platform_bootstrap.QLineEdit("3,224,224")
    self.reward_image_size_edit.setFixedWidth(100)
    self.reward_image_size_edit.setToolTip("C,H,W 或 H,W / 单边边长")
    rm_model_row.addWidget(self.reward_image_size_edit)
    rm_model_row.addWidget(_eai_platform_bootstrap.QLabel("设备"))
    self.reward_device_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_device_combo.addItem("auto", "auto")
    self.reward_device_combo.addItem("cuda", "cuda")
    self.reward_device_combo.addItem("cpu", "cpu")
    rm_model_row.addWidget(self.reward_device_combo)
    rm_model_row.addWidget(_eai_platform_bootstrap.QLabel("阈值"))
    self.reward_threshold_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.reward_threshold_spin.setRange(0.0, 1.0)
    self.reward_threshold_spin.setSingleStep(0.05)
    self.reward_threshold_spin.setValue(0.5)
    self.reward_threshold_spin.setDecimals(2)
    self.reward_threshold_spin.setFixedWidth(72)
    self.reward_threshold_spin.setToolTip("批量二分类阈值；teleop 默认也可改")
    rm_model_row.addWidget(self.reward_threshold_spin)
    reward_outer.addLayout(rm_model_row)

    build.rm_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    build.rm_ckpt_row.setSpacing(6)
    build.rm_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt"))
    build._default_reward_ckpt = _eai_platform_bootstrap.os.path.join(
        _eai_configuration_tabs.RLINF_REWARD_CACHE_DIR, "Qwen3-VL-4B-Instruct"
    )
    self.reward_ckpt_edit = _eai_platform_bootstrap.QLineEdit(
        build._default_reward_ckpt if _eai_platform_bootstrap.os.path.isdir(build._default_reward_ckpt) else ""
    )
    self.reward_ckpt_edit.setPlaceholderText(
        "reward 权重 .pt / .pth / .safetensors 或 VLM 目录"
    )
    self.reward_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    build.rm_ckpt_row.addWidget(self.reward_ckpt_edit, 1)
    self.reward_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.reward_ckpt_browse_btn.setFixedWidth(28)
    self.reward_ckpt_browse_btn.clicked.connect(self._on_reward_ckpt_browse)
    build.rm_ckpt_row.addWidget(self.reward_ckpt_browse_btn)
    build.rm_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("task"))
    self.reward_task_edit = _eai_platform_bootstrap.QLineEdit()
    self.reward_task_edit.setPlaceholderText("VLM 任务描述（可选）")
    build.rm_ckpt_row.addWidget(self.reward_task_edit, 1)
    reward_outer.addLayout(build.rm_ckpt_row)

    rm_mode_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_mode_row.setSpacing(6)
    rm_mode_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.reward_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_mode_combo.addItem("① 单帧打分", "single")
    self.reward_mode_combo.addItem("② A/B 偏好", "pair")
    self.reward_mode_combo.addItem("③ 批量数据集", "batch")
    self.reward_mode_combo.addItem("④ 真机 teleop", "teleop")
    self.reward_mode_combo.currentIndexChanged.connect(self._on_reward_mode_changed)
    rm_mode_row.addWidget(self.reward_mode_combo)
    self.reward_run_btn = _eai_platform_bootstrap.QPushButton("运行评测")
    self.reward_run_btn.setToolTip("按当前模式启动打分或 teleop")
    self.reward_run_btn.clicked.connect(self._on_reward_run_clicked)
    rm_mode_row.addWidget(self.reward_run_btn)
    self.reward_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.reward_stop_btn.setObjectName("dangerAction")
    self.reward_stop_btn.setEnabled(False)
    self.reward_stop_btn.clicked.connect(self._on_reward_stop_clicked)
    rm_mode_row.addWidget(self.reward_stop_btn)
    self.reward_status_label = _eai_platform_bootstrap.QLabel("空闲")
    self.reward_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_mode_row.addWidget(self.reward_status_label, 1)
    reward_outer.addLayout(rm_mode_row)

    self.reward_stack = _eai_platform_bootstrap.QStackedWidget()

    # --- single ---
    rm_single = _eai_platform_bootstrap.QWidget()
    rm_single_l = _eai_platform_bootstrap.QVBoxLayout(rm_single)
    rm_single_l.setContentsMargins(0, 0, 0, 0)
    rm_single_l.setSpacing(6)
    rm_s_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_s_row.addWidget(_eai_platform_bootstrap.QLabel("输入"))
    self.reward_single_source_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_single_source_combo.addItem("当前相机", "camera")
    self.reward_single_source_combo.addItem("本地文件", "file")
    rm_s_row.addWidget(self.reward_single_source_combo)
    rm_s_row.addWidget(_eai_platform_bootstrap.QLabel("预览 topic"))
    self.reward_topic_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_topic_combo.setMinimumWidth(200)
    rm_s_row.addWidget(self.reward_topic_combo, 1)
    self.reward_topic_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.reward_topic_refresh_btn.setFixedWidth(48)
    self.reward_topic_refresh_btn.clicked.connect(self._refresh_reward_topic_combo)
    rm_s_row.addWidget(self.reward_topic_refresh_btn)
    rm_single_l.addLayout(rm_s_row)
    rm_s_path = _eai_platform_bootstrap.QHBoxLayout()
    self.reward_single_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.reward_single_path_edit.setPlaceholderText("本地图像路径…")
    rm_s_path.addWidget(self.reward_single_path_edit, 1)
    self.reward_single_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.reward_single_browse_btn.setFixedWidth(28)
    self.reward_single_browse_btn.clicked.connect(
        lambda: self._on_reward_image_browse(self.reward_single_path_edit)
    )
    rm_s_path.addWidget(self.reward_single_browse_btn)
    rm_single_l.addLayout(rm_s_path)
    rm_s_prev = _eai_platform_bootstrap.QHBoxLayout()
    self.reward_single_preview = _eai_ui_widgets_images.ScaledPixmapLabel("输入")
    self.reward_single_preview.setMinimumHeight(160)
    rm_s_prev.addWidget(self.reward_single_preview, 1)
    self.reward_score_label = _eai_platform_bootstrap.QLabel("score: —")
    self.reward_score_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.reward_score_label.setStyleSheet(
        f"font-size: 28px; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold;"
    )
    rm_s_prev.addWidget(self.reward_score_label, 1)
    rm_single_l.addLayout(rm_s_prev, 1)
    self.reward_stack.addWidget(rm_single)

    # --- pair ---
    rm_pair = _eai_platform_bootstrap.QWidget()
    rm_pair_l = _eai_platform_bootstrap.QVBoxLayout(rm_pair)
    rm_pair_l.setContentsMargins(0, 0, 0, 0)
    rm_pair_l.setSpacing(6)
    rm_p_src = _eai_platform_bootstrap.QHBoxLayout()
    rm_p_src.addWidget(_eai_platform_bootstrap.QLabel("输入"))
    self.reward_pair_source_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.reward_pair_source_combo.addItem("本地文件", "file")
    self.reward_pair_source_combo.addItem("相机抓两帧(A=当前)", "camera")
    rm_p_src.addWidget(self.reward_pair_source_combo)
    self.reward_capture_a_btn = _eai_platform_bootstrap.QPushButton("抓 A")
    self.reward_capture_a_btn.clicked.connect(
        lambda: self._on_reward_capture_slot("a")
    )
    rm_p_src.addWidget(self.reward_capture_a_btn)
    self.reward_capture_b_btn = _eai_platform_bootstrap.QPushButton("抓 B")
    self.reward_capture_b_btn.clicked.connect(
        lambda: self._on_reward_capture_slot("b")
    )
    rm_p_src.addWidget(self.reward_capture_b_btn)
    rm_p_src.addStretch(1)
    rm_pair_l.addLayout(rm_p_src)
    for side, build.attr in (("A", "reward_pair_a_edit"), ("B", "reward_pair_b_edit")):
        row = _eai_platform_bootstrap.QHBoxLayout()
        row.addWidget(_eai_platform_bootstrap.QLabel(f"图{side}"))
        edit = _eai_platform_bootstrap.QLineEdit()
        edit.setPlaceholderText(f"图像 {side} 路径…")
        setattr(self, build.attr, edit)
        row.addWidget(edit, 1)
        build.btn = _eai_platform_bootstrap.QPushButton("…")
        build.btn.setFixedWidth(28)
        build.btn.clicked.connect(lambda _=False, e=edit: self._on_reward_image_browse(e))
        row.addWidget(build.btn)
        rm_pair_l.addLayout(row)
    rm_p_prev = _eai_platform_bootstrap.QHBoxLayout()
    self.reward_pair_a_preview = _eai_ui_widgets_images.ScaledPixmapLabel("A")
    self.reward_pair_a_preview.setMinimumHeight(140)
    rm_p_prev.addWidget(self.reward_pair_a_preview, 1)
    self.reward_pair_b_preview = _eai_ui_widgets_images.ScaledPixmapLabel("B")
    self.reward_pair_b_preview.setMinimumHeight(140)
    rm_p_prev.addWidget(self.reward_pair_b_preview, 1)
    self.reward_pair_result_label = _eai_platform_bootstrap.QLabel("A/B: —")
    self.reward_pair_result_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.reward_pair_result_label.setStyleSheet(
        f"font-size: 18px; color: {_eai_configuration_policies.UI_TEXT_PRIMARY};"
    )
    rm_p_prev.addWidget(self.reward_pair_result_label, 1)
    rm_pair_l.addLayout(rm_p_prev, 1)
    self.reward_stack.addWidget(rm_pair)

    # --- batch ---
    rm_batch = _eai_platform_bootstrap.QWidget()
    rm_batch_l = _eai_platform_bootstrap.QVBoxLayout(rm_batch)
    rm_batch_l.setContentsMargins(0, 0, 0, 0)
    rm_batch_l.setSpacing(6)
    rm_b_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_b_row.addWidget(_eai_platform_bootstrap.QLabel("数据集 .pt"))
    self.reward_dataset_edit = _eai_platform_bootstrap.QLineEdit()
    self.reward_dataset_edit.setPlaceholderText(
        "RLinf processed reward 数据（含 images/labels）"
    )
    self.reward_dataset_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    rm_b_row.addWidget(self.reward_dataset_edit, 1)
    self.reward_dataset_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.reward_dataset_browse_btn.setFixedWidth(28)
    self.reward_dataset_browse_btn.clicked.connect(self._on_reward_dataset_browse)
    rm_b_row.addWidget(self.reward_dataset_browse_btn)
    rm_b_row.addWidget(_eai_platform_bootstrap.QLabel("max"))
    self.reward_max_samples_spin = _eai_platform_bootstrap.QSpinBox()
    self.reward_max_samples_spin.setRange(0, 1_000_000)
    self.reward_max_samples_spin.setValue(0)
    self.reward_max_samples_spin.setSpecialValueText("全部")
    self.reward_max_samples_spin.setToolTip("0 = 全部样本")
    rm_b_row.addWidget(self.reward_max_samples_spin)
    rm_batch_l.addLayout(rm_b_row)
    self.reward_batch_summary = _eai_platform_bootstrap.QLabel("批量结果: —")
    self.reward_batch_summary.setWordWrap(True)
    self.reward_batch_summary.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.reward_batch_summary.setStyleSheet(
        f"background-color: {_eai_configuration_policies.UI_BG_INPUT}; padding: 10px; border: 1px solid {_eai_configuration_policies.UI_BORDER};"
    )
    self.reward_batch_summary.setMinimumHeight(120)
    rm_batch_l.addWidget(self.reward_batch_summary, 1)
    self.reward_stack.addWidget(rm_batch)

    # --- teleop ---
    rm_teleop = _eai_platform_bootstrap.QWidget()
    rm_teleop_l = _eai_platform_bootstrap.QVBoxLayout(rm_teleop)
    rm_teleop_l.setContentsMargins(0, 0, 0, 0)
    rm_teleop_l.setSpacing(6)
    rm_t_row = _eai_platform_bootstrap.QHBoxLayout()
    rm_t_row.addWidget(_eai_platform_bootstrap.QLabel("config"))
    self.reward_teleop_config_edit = _eai_platform_bootstrap.QLineEdit("realworld_teleop")
    self.reward_teleop_config_edit.setFixedWidth(180)
    rm_t_row.addWidget(self.reward_teleop_config_edit)
    rm_t_row.addStretch(1)
    rm_teleop_l.addLayout(rm_t_row)
    rm_teleop_note = _eai_platform_bootstrap.QLabel(
        "将调用 RLinf examples/reward/eval_realworld_teleop.py。"
        "需集群 / Franka / SpaceMouse 已按 yaml 配置；本页只负责启停与日志。"
    )
    rm_teleop_note.setWordWrap(True)
    rm_teleop_note.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    rm_teleop_l.addWidget(rm_teleop_note)
    rm_teleop_l.addStretch(1)
    self.reward_stack.addWidget(rm_teleop)

    reward_outer.addWidget(self.reward_stack, 1)

    self.reward_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.reward_log_edit.setReadOnly(True)
    self.reward_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.reward_log_edit.setPlaceholderText("Reward 评测日志…")
    self.reward_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.reward_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        reward_outer,
        self.reward_log_edit,
        default_h=160,
        min_h=80,
        max_h=900,
        label="拖动调整 Reward 日志高度",
        clearable=True,
    )

    self._reward_launcher = _eai_adapters_models_reward_launchers.RewardModelLauncher(self)
    self._reward_launcher.log_line.connect(self._append_reward_log)
    self._reward_launcher.status_message.connect(self._on_reward_status)
    self._reward_launcher.running_changed.connect(self._update_reward_ui)
    self._reward_launcher.result_ready.connect(self._on_reward_result)
    self._on_reward_mode_changed()
    self._refresh_reward_topic_combo()
