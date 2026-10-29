"""ui.builders.training — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_training = module_ref('eai_app.adapters.models.training')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    train_tab = _eai_platform_bootstrap.QWidget()
    train_outer = _eai_platform_bootstrap.QVBoxLayout(train_tab)
    train_outer.setContentsMargins(8, 6, 8, 6)
    train_outer.setSpacing(6)

    train_path_row = _eai_platform_bootstrap.QHBoxLayout()
    train_path_row.setSpacing(6)
    train_path_row.addWidget(_eai_platform_bootstrap.QLabel("psi-policy"))
    self.train_policy_dir_edit = _eai_platform_bootstrap.QLineEdit()
    self.train_policy_dir_edit.setReadOnly(True)
    self.train_policy_dir_edit.setPlaceholderText("点击「选择路径」指定 psi-policy 仓库")
    self.train_policy_dir_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    train_path_row.addWidget(self.train_policy_dir_edit, 1)
    self.train_browse_policy_btn = _eai_platform_bootstrap.QPushButton("选择路径")
    self.train_browse_policy_btn.setToolTip(
        "选择 psi-policy 仓库根目录（需包含 psi_policy/train.py）"
    )
    self.train_browse_policy_btn.clicked.connect(self._on_train_browse_policy_clicked)
    train_path_row.addWidget(self.train_browse_policy_btn)
    train_outer.addLayout(train_path_row)

    train_row1 = _eai_platform_bootstrap.QHBoxLayout()
    train_row1.setSpacing(6)
    train_row1.addWidget(_eai_platform_bootstrap.QLabel("配置"))
    self.train_config_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.train_config_combo.setMinimumWidth(220)
    self.train_config_combo.setToolTip("psi-policy Hydra workspace 配置 (example_workspace_*)")
    psi_repo = _eai_perception_segmentation.resolve_psi_policy_dir()
    train_configs = _eai_perception_segmentation.list_psi_policy_workspace_configs(psi_repo) if psi_repo else []
    if not train_configs:
        train_configs = [_eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT]
    for cfg_name in train_configs:
        self.train_config_combo.addItem(cfg_name, cfg_name)
    default_idx = self.train_config_combo.findData(_eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT)
    if default_idx >= 0:
        self.train_config_combo.setCurrentIndex(default_idx)
    train_row1.addWidget(self.train_config_combo)
    train_row1.addWidget(_eai_platform_bootstrap.QLabel("datahouse_id"))
    self.train_datahouse_edit = _eai_platform_bootstrap.QLineEdit()
    self.train_datahouse_edit.setPlaceholderText("数据仓库 ID")
    self.train_datahouse_edit.setMinimumWidth(120)
    train_row1.addWidget(self.train_datahouse_edit, 1)
    train_row1.addWidget(_eai_platform_bootstrap.QLabel("view_id"))
    self.train_view_edit = _eai_platform_bootstrap.QLineEdit()
    self.train_view_edit.setPlaceholderText("训练视图 ID")
    self.train_view_edit.setMinimumWidth(120)
    train_row1.addWidget(self.train_view_edit, 1)
    train_outer.addLayout(train_row1)

    train_row2 = _eai_platform_bootstrap.QHBoxLayout()
    train_row2.setSpacing(6)
    train_row2.addWidget(_eai_platform_bootstrap.QLabel("epochs"))
    self.train_epochs_spin = _eai_platform_bootstrap.QSpinBox()
    self.train_epochs_spin.setRange(0, 9999)
    self.train_epochs_spin.setValue(0)
    self.train_epochs_spin.setSpecialValueText("默认")
    self.train_epochs_spin.setToolTip("0 表示使用配置文件默认值")
    self.train_epochs_spin.setFixedWidth(72)
    train_row2.addWidget(self.train_epochs_spin)
    train_row2.addWidget(_eai_platform_bootstrap.QLabel("batch"))
    self.train_batch_spin = _eai_platform_bootstrap.QSpinBox()
    self.train_batch_spin.setRange(0, 4096)
    self.train_batch_spin.setValue(0)
    self.train_batch_spin.setSpecialValueText("默认")
    self.train_batch_spin.setToolTip("0 表示使用配置文件默认值")
    self.train_batch_spin.setFixedWidth(72)
    train_row2.addWidget(self.train_batch_spin)
    train_row2.addWidget(_eai_platform_bootstrap.QLabel("WandB"))
    self.train_logging_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for mode in _eai_configuration_policies.PSI_POLICY_LOGGING_MODES:
        self.train_logging_combo.addItem(mode, mode)
    self.train_logging_combo.setCurrentIndex(
        self.train_logging_combo.findData("offline")
    )
    self.train_logging_combo.setToolTip("logging.mode：offline / online / disabled")
    train_row2.addWidget(self.train_logging_combo)
    train_row2.addWidget(_eai_platform_bootstrap.QLabel("GPU数"))
    self.train_gpu_spin = _eai_platform_bootstrap.QSpinBox()
    self.train_gpu_spin.setRange(1, 16)
    self.train_gpu_spin.setValue(1)
    self.train_gpu_spin.setToolTip(">1 时使用 accelerate launch 多卡训练")
    self.train_gpu_spin.setFixedWidth(52)
    train_row2.addWidget(self.train_gpu_spin)
    self.train_status_label = _eai_platform_bootstrap.QLabel("训练: 空闲")
    self.train_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    repo_hint = psi_repo or "(未找到 psi-policy，请设置 PSI_POLICY_DIR)"
    self.train_status_label.setToolTip(
        f"psi-policy 目录: {repo_hint}\n"
        "训练日志实时显示在下方"
    )
    train_row2.addWidget(self.train_status_label, 1)
    self.train_start_btn = _eai_platform_bootstrap.QPushButton("开始训练")
    self.train_start_btn.setToolTip("启动 psi_policy/train.py")
    self.train_start_btn.clicked.connect(self._on_train_start_clicked)
    train_row2.addWidget(self.train_start_btn)
    self.train_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.train_stop_btn.setToolTip("终止当前训练进程")
    self.train_stop_btn.setObjectName("dangerAction")
    self.train_stop_btn.setEnabled(False)
    self.train_stop_btn.clicked.connect(self._on_train_stop_clicked)
    train_row2.addWidget(self.train_stop_btn)
    self.train_clear_log_btn = _eai_platform_bootstrap.QPushButton("清空日志")
    self.train_clear_log_btn.clicked.connect(self._on_train_clear_log_clicked)
    train_row2.addWidget(self.train_clear_log_btn)
    train_outer.addLayout(train_row2)

    self.train_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.train_log_edit.setReadOnly(True)
    self.train_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.train_log_edit.setPlaceholderText("训练日志将在此实时显示…")
    self.train_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.train_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        train_outer,
        self.train_log_edit,
        default_h=200,
        min_h=80,
        max_h=900,
        label="拖动调整训练日志高度",
    )

    self._train_launcher = _eai_adapters_models_training.PsiPolicyTrainLauncher(self)
    self._update_train_path_ui()
    if self._get_psi_policy_dir() is None:
        self.train_log_edit.setPlainText(
            "未找到 psi-policy 仓库。\n"
            "请点击「选择路径」指定 psi-policy 目录。"
        )
    build.control_tabs.addTab(train_tab, "训练")
