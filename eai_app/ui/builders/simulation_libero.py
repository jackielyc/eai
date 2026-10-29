"""ui.builders.simulation libero — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    libero_page = _eai_platform_bootstrap.QWidget()
    libero_page_l = _eai_platform_bootstrap.QVBoxLayout(libero_page)
    libero_page_l.setContentsMargins(0, 0, 0, 0)
    libero_page_l.setSpacing(6)
    libero_root_row = _eai_platform_bootstrap.QHBoxLayout()
    libero_root_row.setSpacing(6)
    libero_root_row.addWidget(_eai_platform_bootstrap.QLabel("RLinf"))
    self.libero_rlinf_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("RLINF_ROOT", _eai_configuration_tabs.RLINF_ROOT_DEFAULT)
    )
    self.libero_rlinf_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.libero_rlinf_edit.setToolTip(
        "RLinf 仓库（参考 toolkits/eval_scripts_openpi/libero_eval.py）"
    )
    libero_root_row.addWidget(self.libero_rlinf_edit, 1)
    self.libero_rlinf_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.libero_rlinf_browse_btn.setFixedWidth(28)
    self.libero_rlinf_browse_btn.clicked.connect(self._on_libero_rlinf_browse)
    libero_root_row.addWidget(self.libero_rlinf_browse_btn)
    libero_page_l.addLayout(libero_root_row)

    libero_task_row = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    libero_task_row.addWidget(_eai_platform_bootstrap.QLabel("套件"))
    self.libero_suite_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, suite_id in _eai_configuration_models.LIBERO_TASK_SUITES:
        self.libero_suite_combo.addItem(build.label, suite_id)
    self.libero_suite_combo.setToolTip(
        "LIBERO task suite：spatial / object / goal / 10 / 90"
    )
    self.libero_suite_combo.currentIndexChanged.connect(
        self._on_libero_suite_changed
    )
    libero_task_row.addWidget(self.libero_suite_combo)
    libero_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务序号"))
    self.libero_task_id_spin = _eai_platform_bootstrap.QSpinBox()
    self.libero_task_id_spin.setRange(0, 99)
    self.libero_task_id_spin.setValue(0)
    self.libero_task_id_spin.setToolTip(
        "suite 内任务下标；只开界面 / 单任务评测都会用到"
    )
    self.libero_task_id_spin.valueChanged.connect(self._on_libero_suite_changed)
    libero_task_row.addWidget(self.libero_task_id_spin)
    libero_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.libero_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.libero_task_combo.setMinimumWidth(280)
    self.libero_task_combo.setToolTip(
        "套件中的具体任务。下拉项带场景缩略图，悬停可看在做什么。"
    )
    self.libero_task_combo.currentIndexChanged.connect(
        self._on_libero_task_combo_changed
    )
    libero_task_row.addWidget(self.libero_task_combo)
    libero_task_row.addWidget(_eai_platform_bootstrap.QLabel("配置"))
    self.libero_config_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.cfg_id in _eai_configuration_models.LIBERO_POLICY_CONFIGS:
        self.libero_config_combo.addItem(build.label, build.cfg_id)
    self.libero_config_combo.setToolTip(
        "openpi TrainConfig（pi05_libero / pi0_libero）或 LingBot-VA（wan_va）"
    )
    self.libero_config_combo.currentIndexChanged.connect(
        self._on_libero_config_changed
    )
    libero_task_row.addWidget(self.libero_config_combo)
    libero_task_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.libero_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.libero_mode_combo.addItem("本地 ckpt（自动部署 serve）", "local")
    self.libero_mode_combo.addItem("远程 Pi（websocket）", "remote")
    self.libero_mode_combo.setToolTip(
        "local（默认）：用下方 checkpoint 自动部署 openpi serve，再用客户端评测"
        "（libero 环境无 jax，不能进程内加载 openpi）。\n"
        "remote：连已部署的 :8080 serve_policy。"
    )
    self.libero_mode_combo.currentIndexChanged.connect(self._on_libero_mode_changed)
    libero_task_row.addWidget(self.libero_mode_combo)
    libero_page_l.addLayout(libero_task_row)
    (
        self.libero_task_preview,
        self.libero_task_thumb,
        _libero_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    libero_page_l.addWidget(self.libero_task_preview)

    libero_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    libero_ckpt_row.setSpacing(6)
    libero_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("Checkpoint"))
    self.libero_ckpt_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("LIBERO_CKPT", _eai_configuration_tabs.LIBERO_CKPT_DEFAULT)
    )
    self.libero_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.libero_ckpt_edit.setToolTip(
        "默认本地 pi05_libero（HF 镜像缓存）；缺失时部署会自动下载。"
        "remote：serve_policy；local：自动改 remote 客户端评测。"
    )
    libero_ckpt_row.addWidget(self.libero_ckpt_edit, 1)
    self.libero_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.libero_ckpt_browse_btn.setFixedWidth(28)
    self.libero_ckpt_browse_btn.clicked.connect(self._on_libero_ckpt_browse)
    libero_ckpt_row.addWidget(self.libero_ckpt_browse_btn)
    self.libero_pi_start_btn = _eai_platform_bootstrap.QPushButton("部署 Pi")
    self.libero_pi_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.libero_pi_start_btn.setToolTip("用上方 config/checkpoint 启动 openpi serve_policy :8080")
    self.libero_pi_start_btn.clicked.connect(self._on_libero_pi_start_clicked)
    libero_ckpt_row.addWidget(self.libero_pi_start_btn)
    self.libero_pi_probe_btn = _eai_platform_bootstrap.QPushButton("验证")
    self.libero_pi_probe_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.libero_pi_probe_btn.clicked.connect(self._on_molmospaces_pi_probe_clicked)
    libero_ckpt_row.addWidget(self.libero_pi_probe_btn)
    self.libero_pi_stop_btn = _eai_platform_bootstrap.QPushButton("停止 Pi")
    self.libero_pi_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.libero_pi_stop_btn.setObjectName("dangerAction")
    self.libero_pi_stop_btn.setToolTip("停止 :8080 上的 OpenPI serve_policy，卸载 Pi 模型并释放 GPU")
    self.libero_pi_stop_btn.clicked.connect(self._on_libero_pi_stop_clicked)
    libero_ckpt_row.addWidget(self.libero_pi_stop_btn)
    libero_page_l.addLayout(libero_ckpt_row)

    libero_pi_status_row = _eai_platform_bootstrap.QHBoxLayout()
    libero_pi_status_row.setSpacing(6)
    libero_pi_status_row.addWidget(_eai_platform_bootstrap.QLabel("Pi 状态"))
    self.libero_pi_status_label = _eai_platform_bootstrap.QLabel("未检查")
    self.libero_pi_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.libero_pi_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.libero_pi_status_label.setWordWrap(True)
    libero_pi_status_row.addWidget(self.libero_pi_status_label, 1)
    libero_page_l.addLayout(libero_pi_status_row)

    libero_param_row = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    libero_param_row.addWidget(_eai_platform_bootstrap.QLabel("任务数"))
    self.libero_max_tasks_spin = _eai_platform_bootstrap.QSpinBox()
    self.libero_max_tasks_spin.setRange(0, 1000)
    self.libero_max_tasks_spin.setValue(1)
    self.libero_max_tasks_spin.setToolTip("0 = 跑完整个 suite；默认 1 便于冒烟")
    libero_param_row.addWidget(self.libero_max_tasks_spin)
    libero_param_row.addWidget(_eai_platform_bootstrap.QLabel("每任务 trials"))
    self.libero_trials_spin = _eai_platform_bootstrap.QSpinBox()
    self.libero_trials_spin.setRange(1, 100)
    self.libero_trials_spin.setValue(1)
    libero_param_row.addWidget(self.libero_trials_spin)
    libero_param_row.addWidget(_eai_platform_bootstrap.QLabel("action_chunk"))
    self.libero_chunk_spin = _eai_platform_bootstrap.QSpinBox()
    self.libero_chunk_spin.setRange(1, 50)
    self.libero_chunk_spin.setValue(5)
    libero_param_row.addWidget(self.libero_chunk_spin)
    libero_param_row.addWidget(_eai_platform_bootstrap.QLabel("num_steps"))
    self.libero_steps_spin = _eai_platform_bootstrap.QSpinBox()
    self.libero_steps_spin.setRange(1, 50)
    self.libero_steps_spin.setValue(10)
    self.libero_steps_spin.setToolTip("local 模式 flow denoise steps；需与训练配置一致")
    libero_param_row.addWidget(self.libero_steps_spin)
    self.libero_render_gui_check = _eai_platform_bootstrap.QCheckBox("图形窗口")
    self.libero_render_gui_check.setChecked(
        bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip())
    )
    self.libero_render_gui_check.setToolTip(
        "勾选后弹出 robosuite/MuJoCo 实时窗口（MUJOCO_GL=glfw）。\n"
        "评测结束后窗口仍保持打开，点击「停止」才关闭。\n"
        "无 DISPLAY 时可取消勾选，仅离屏渲染并保存视频。"
    )
    libero_param_row.addWidget(self.libero_render_gui_check)
    libero_page_l.addLayout(libero_param_row)

    libero_hint = _eai_platform_bootstrap.QLabel(
        "对齐 RLinf toolkits/eval_scripts_openpi/libero_eval.py。"
        "默认本地 ckpt：先「部署 Pi」（pi05_libero）再启动评测。"
        "可勾选「图形窗口」实时观看；"
        "工作区「RynnValue Live HUD」勾选后单独弹出剩余时间窗口。"
        "需 robosuite+libero（见日志安装提示）。与其它仿真后端互斥。"
    )
    libero_hint.setWordWrap(True)
    libero_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    libero_page_l.addWidget(libero_hint)
    libero_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(libero_page)
