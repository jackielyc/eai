"""ui.builders.simulation arena — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    arena_page = _eai_platform_bootstrap.QWidget()
    arena_page_l = _eai_platform_bootstrap.QVBoxLayout(arena_page)
    arena_page_l.setContentsMargins(0, 0, 0, 0)
    arena_page_l.setSpacing(6)
    arena_root_row = _eai_platform_bootstrap.QHBoxLayout()
    arena_root_row.setSpacing(6)
    arena_root_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.arena_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ISAACLAB_ARENA_ROOT", _eai_configuration_tabs.ISAACLAB_ARENA_ROOT_DEFAULT)
    )
    self.arena_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arena_root_edit.setToolTip(f"IsaacLab-Arena 仓库（默认 {_eai_configuration_tabs.ISAACLAB_ARENA_ROOT_DEFAULT}）")
    arena_root_row.addWidget(self.arena_root_edit, 1)
    self.arena_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.arena_root_browse_btn.setFixedWidth(28)
    self.arena_root_browse_btn.clicked.connect(self._on_arena_root_browse)
    arena_root_row.addWidget(self.arena_root_browse_btn)
    arena_page_l.addLayout(arena_root_row)
    arena_task_row = _eai_platform_bootstrap.QHBoxLayout()
    arena_task_row.setSpacing(6)
    arena_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.arena_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.arena_task_combo.setToolTip(
        "每次只启动当前选中的一个 environment。下拉项带场景缩略图。"
    )
    arena_task_row.addWidget(self.arena_task_combo, 1)
    self.arena_task_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.arena_task_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.arena_task_refresh_btn.clicked.connect(self._refresh_arena_tasks)
    arena_task_row.addWidget(self.arena_task_refresh_btn)
    arena_task_row.addWidget(_eai_platform_bootstrap.QLabel("策略"))
    self.arena_policy_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.policy_id in _eai_configuration_tabs.ISAACLAB_ARENA_POLICIES:
        self.arena_policy_combo.addItem(build.label, build.policy_id)
    self.arena_policy_combo.setToolTip(
        "zero_action 不需要文件。replay 需要 HDF5；rsl_rl 需要 checkpoint。"
        "缺文件时不会启动 Kit。"
    )
    self.arena_policy_combo.currentIndexChanged.connect(self._on_arena_policy_changed)
    arena_task_row.addWidget(self.arena_policy_combo)
    arena_task_row.addWidget(_eai_platform_bootstrap.QLabel("步数"))
    self.arena_steps_spin = _eai_platform_bootstrap.QSpinBox()
    self.arena_steps_spin.setRange(1, 100000)
    self.arena_steps_spin.setValue(200)
    self.arena_steps_spin.setToolTip("policy_runner --num_steps")
    arena_task_row.addWidget(self.arena_steps_spin)
    self.arena_headless_check = _eai_platform_bootstrap.QCheckBox("无界面")
    self.arena_headless_check.setChecked(not bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip()))
    self.arena_headless_check.setToolTip(
        "不勾选时打开 Omniverse Kit 窗口（--viz kit）。"
        "Isaac Lab 3 不传 --viz 会默认无界面。"
    )
    arena_task_row.addWidget(self.arena_headless_check)
    self.arena_gpu_check = _eai_platform_bootstrap.QCheckBox("GPU")
    self.arena_gpu_check.setChecked(True)
    self.arena_gpu_check.setToolTip("勾选后物理仿真与渲染使用 GPU（--device cuda）")
    arena_task_row.addWidget(self.arena_gpu_check)
    self.arena_gpu_spin = _eai_platform_bootstrap.QSpinBox()
    self.arena_gpu_spin.setRange(0, 15)
    self.arena_gpu_spin.setValue(0)
    self.arena_gpu_spin.setToolTip("nvidia-smi 上的 GPU 序号")
    self.arena_gpu_spin.setFixedWidth(52)
    arena_task_row.addWidget(self.arena_gpu_spin)
    arena_page_l.addLayout(arena_task_row)
    (
        self.arena_task_preview,
        self.arena_task_thumb,
        _arena_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    _eai_configuration_models.bind_eval_task_thumb(self.arena_task_combo, self.arena_task_thumb)
    arena_page_l.addWidget(self.arena_task_preview)
    self.arena_policy_file_row = _eai_platform_bootstrap.QWidget()
    arena_file_row = _eai_platform_bootstrap.QHBoxLayout(self.arena_policy_file_row)
    arena_file_row.setContentsMargins(0, 0, 0, 0)
    arena_file_row.setSpacing(6)
    self.arena_policy_file_label = _eai_platform_bootstrap.QLabel("回放文件")
    arena_file_row.addWidget(self.arena_policy_file_label)
    self.arena_policy_file_edit = _eai_platform_bootstrap.QLineEdit()
    self.arena_policy_file_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arena_policy_file_edit.setPlaceholderText("replay 的 HDF5，或 rsl_rl 的 .pt")
    arena_file_row.addWidget(self.arena_policy_file_edit, 1)
    self.arena_policy_file_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.arena_policy_file_browse_btn.setFixedWidth(28)
    self.arena_policy_file_browse_btn.clicked.connect(self._on_arena_policy_file_browse)
    arena_file_row.addWidget(self.arena_policy_file_browse_btn)
    self.arena_policy_file_row.setVisible(False)
    arena_page_l.addWidget(self.arena_policy_file_row)
    arena_extra_row = _eai_platform_bootstrap.QHBoxLayout()
    arena_extra_row.setSpacing(6)
    arena_extra_row.addWidget(_eai_platform_bootstrap.QLabel("额外参数"))
    self.arena_extra_edit = _eai_platform_bootstrap.QLineEdit()
    self.arena_extra_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.arena_extra_edit.setPlaceholderText("可选，写在环境名后面，如 --object cracker_box")
    arena_extra_row.addWidget(self.arena_extra_edit, 1)
    arena_page_l.addLayout(arena_extra_row)
    arena_hint = _eai_platform_bootstrap.QLabel(
        "使用 IsaacLab-Arena conda 环境。"
        "启动时只跑任务列表里当前这一项（policy_runner）。"
        "与 Isaac / MuJoCo / MolmoSpaces 互斥。"
    )
    arena_hint.setWordWrap(True)
    arena_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    arena_page_l.addWidget(arena_hint)
    arena_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(arena_page)
