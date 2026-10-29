"""ui.builders.simulation mujoco — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    mj_page = _eai_platform_bootstrap.QWidget()
    mj_page_l = _eai_platform_bootstrap.QVBoxLayout(mj_page)
    mj_page_l.setContentsMargins(0, 0, 0, 0)
    mj_page_l.setSpacing(6)
    mj_path_row = _eai_platform_bootstrap.QHBoxLayout()
    mj_path_row.setSpacing(6)
    mj_path_row.addWidget(_eai_platform_bootstrap.QLabel("MuJoCo"))
    self.mujoco_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("MUJOCO_ROOT", _eai_configuration_tabs.MUJOCO_ROOT_DEFAULT)
    )
    self.mujoco_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    mj_path_row.addWidget(self.mujoco_root_edit, 1)
    self.mujoco_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.mujoco_root_browse_btn.setFixedWidth(28)
    self.mujoco_root_browse_btn.clicked.connect(self._on_mujoco_root_browse)
    mj_path_row.addWidget(self.mujoco_root_browse_btn)
    mj_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.mujoco_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_discovery.resolve_mujoco_python(self.mujoco_root_edit.text())
    )
    self.mujoco_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    mj_path_row.addWidget(self.mujoco_python_edit, 1)
    self.mujoco_install_btn = _eai_platform_bootstrap.QPushButton("安装 mujoco")
    self.mujoco_install_btn.setToolTip("pip install -U mujoco（使用上方 Python）")
    self.mujoco_install_btn.clicked.connect(self._on_mujoco_install_clicked)
    mj_path_row.addWidget(self.mujoco_install_btn)
    mj_page_l.addLayout(mj_path_row)

    mj_model_row = _eai_platform_bootstrap.QHBoxLayout()
    mj_model_row.setSpacing(6)
    mj_model_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    self.mujoco_model_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.mujoco_model_combo.setEditable(True)
    self.mujoco_model_combo.setInsertPolicy(_eai_platform_bootstrap.QComboBox.NoInsert)
    self.mujoco_model_combo.setMinimumWidth(260)
    self.mujoco_model_combo.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.mujoco_model_combo.setToolTip(
        "eai/assets 本地模型（绝对路径）或相对 mujoco/model/ 的官方示例"
    )
    mj_model_row.addWidget(self.mujoco_model_combo, 1)
    self.mujoco_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.mujoco_refresh_btn.setToolTip("重新扫描 eai/assets 与 mujoco/model/**/*.xml")
    self.mujoco_refresh_btn.clicked.connect(self._refresh_mujoco_models)
    mj_model_row.addWidget(self.mujoco_refresh_btn)
    self.mujoco_mjcf_browse_btn = _eai_platform_bootstrap.QPushButton("浏览…")
    self.mujoco_mjcf_browse_btn.clicked.connect(self._on_mujoco_mjcf_browse)
    mj_model_row.addWidget(self.mujoco_mjcf_browse_btn)
    mj_model_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.mujoco_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.mujoco_mode_combo.addItem("官方 viewer (VGL)", "python")
    self.mujoco_mode_combo.addItem("自动(官方)", "auto")
    self.mujoco_mode_combo.addItem("simulate", "simulate")
    self.mujoco_mode_combo.addItem("EGL 回退", "egl")
    self.mujoco_mode_combo.setCurrentIndex(0)
    self.mujoco_mode_combo.setToolTip(
        "默认 python -m mujoco.viewer（官方左右面板）。"
        "无本机 GLX 时走 VirtualGL→NVIDIA。"
        "仅「EGL 回退」才用 Tk 自绘界面。"
    )
    mj_model_row.addWidget(self.mujoco_mode_combo)
    mj_model_row.addWidget(_eai_platform_bootstrap.QLabel("策略"))
    self.mujoco_policy_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.mujoco_policy_combo.addItem("无（仅 viewer）", "")
    self.mujoco_policy_combo.addItem(_eai_configuration_models.LINGBOT_VA_POLICY_LABEL, _eai_configuration_models.LINGBOT_VA_POLICY_ID)
    self.mujoco_policy_combo.addItem(_eai_configuration_models.VPP2_POLICY_LABEL, _eai_configuration_models.VPP2_POLICY_ID)
    self.mujoco_policy_combo.setToolTip(
        "LingBot-VA / VPP2：启动策略服务（serve_only）。"
        "原始 MJCF viewer 无 closed-loop adapter；"
        "完整评测请用 LIBERO / RoboTwin / Isaac。"
    )
    mj_model_row.addWidget(self.mujoco_policy_combo)
    mj_page_l.addLayout(mj_model_row)
    mj_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(mj_page)
