"""ui.builders.simulation external — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    ext_page = _eai_platform_bootstrap.QWidget()
    ext_page_l = _eai_platform_bootstrap.QVBoxLayout(ext_page)
    ext_page_l.setContentsMargins(0, 0, 0, 0)
    ext_page_l.setSpacing(6)

    self.ext_bench_title_label = _eai_platform_bootstrap.QLabel("扩展评测")
    self.ext_bench_title_label.setStyleSheet(
        f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold;"
    )
    ext_page_l.addWidget(self.ext_bench_title_label)
    self.ext_bench_hint_label = _eai_platform_bootstrap.QLabel("")
    self.ext_bench_hint_label.setWordWrap(True)
    self.ext_bench_hint_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    ext_page_l.addWidget(self.ext_bench_hint_label)

    ext_root_row = _eai_platform_bootstrap.QHBoxLayout()
    ext_root_row.setSpacing(6)
    ext_root_row.addWidget(_eai_platform_bootstrap.QLabel("仓库根目录"))
    self.ext_bench_root_edit = _eai_platform_bootstrap.QLineEdit("")
    self.ext_bench_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.ext_bench_root_edit.setToolTip(
        "本地 clone 路径；留空则用默认候选目录。"
        "无本地仓时可用「Harness Docker」或「安装指引」。"
    )
    ext_root_row.addWidget(self.ext_bench_root_edit, 1)
    self.ext_bench_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.ext_bench_root_browse_btn.setFixedWidth(28)
    self.ext_bench_root_browse_btn.clicked.connect(self._on_ext_bench_root_browse)
    ext_root_row.addWidget(self.ext_bench_root_browse_btn)
    ext_page_l.addLayout(ext_root_row)

    ext_py_row = _eai_platform_bootstrap.QHBoxLayout()
    ext_py_row.setSpacing(6)
    ext_py_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.ext_bench_python_edit = _eai_platform_bootstrap.QLineEdit(_eai_platform_bootstrap.sys.executable)
    self.ext_bench_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ext_py_row.addWidget(self.ext_bench_python_edit, 1)
    ext_page_l.addLayout(ext_py_row)

    ext_param_row = _eai_platform_bootstrap.QHBoxLayout()
    ext_param_row.setSpacing(6)
    ext_param_row.addWidget(_eai_platform_bootstrap.QLabel("模式"))
    self.ext_bench_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.ext_bench_mode_combo.addItem("探测依赖 (probe)", "probe")
    self.ext_bench_mode_combo.addItem("安装指引 (hint)", "hint")
    self.ext_bench_mode_combo.addItem("本地评测 (eval)", "eval")
    self.ext_bench_mode_combo.addItem("只开界面 (viewer)", "viewer")
    self.ext_bench_mode_combo.addItem("Harness Docker", "harness")
    self.ext_bench_mode_combo.setCurrentIndex(0)
    self.ext_bench_mode_combo.setToolTip(
        "probe/hint：不跑仿真，只检查与打印命令。\n"
        "eval/viewer：优先本地仓库入口；找不到则回退指引。\n"
        "harness：AllenAI vla-evaluation-harness Docker 镜像。"
    )
    ext_param_row.addWidget(self.ext_bench_mode_combo)
    ext_param_row.addWidget(_eai_platform_bootstrap.QLabel("策略"))
    self.ext_bench_policy_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.ext_bench_policy_combo.addItem("默认 / 本地脚本", "")
    self.ext_bench_policy_combo.addItem(_eai_configuration_models.LINGBOT_VA_POLICY_LABEL, _eai_configuration_models.LINGBOT_VA_POLICY_ID)
    self.ext_bench_policy_combo.addItem(_eai_configuration_models.VPP2_POLICY_LABEL, _eai_configuration_models.VPP2_POLICY_ID)
    self.ext_bench_policy_combo.setToolTip(
        "选 LingBot-VA / VPP2：LIBERO* 走官方 eval；"
        "其它扩展后端为 serve_only（策略 websocket）。"
    )
    ext_param_row.addWidget(self.ext_bench_policy_combo)
    ext_param_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.ext_bench_task_edit = _eai_platform_bootstrap.QLineEdit("")
    self.ext_bench_task_edit.setPlaceholderText("可选 task / suite 名")
    self.ext_bench_task_edit.setToolTip(
        "填写后，下方缩略图会按任务名显示对应场景。"
    )
    self.ext_bench_task_edit.textChanged.connect(self._refresh_ext_task_thumb)
    self.ext_bench_task_edit.setMinimumWidth(140)
    ext_param_row.addWidget(self.ext_bench_task_edit)
    ext_param_row.addWidget(_eai_platform_bootstrap.QLabel("episodes"))
    self.ext_bench_episodes_spin = _eai_platform_bootstrap.QSpinBox()
    self.ext_bench_episodes_spin.setRange(1, 1000)
    self.ext_bench_episodes_spin.setValue(1)
    ext_param_row.addWidget(self.ext_bench_episodes_spin)
    self.ext_bench_viewer_check = _eai_platform_bootstrap.QCheckBox("图形")
    self.ext_bench_viewer_check.setChecked(True)
    self.ext_bench_viewer_check.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    ext_param_row.addWidget(self.ext_bench_viewer_check)
    ext_param_row.addStretch(1)
    ext_page_l.addLayout(ext_param_row)
    (
        self.ext_task_preview,
        self.ext_task_thumb,
        _ext_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    ext_page_l.addWidget(self.ext_task_preview)

    ext_btn_row = _eai_platform_bootstrap.QHBoxLayout()
    ext_btn_row.setSpacing(6)
    self.ext_bench_probe_btn = _eai_platform_bootstrap.QPushButton("探测")
    self.ext_bench_probe_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ext_bench_probe_btn.clicked.connect(self._on_ext_bench_probe_clicked)
    ext_btn_row.addWidget(self.ext_bench_probe_btn)
    self.ext_bench_hint_btn = _eai_platform_bootstrap.QPushButton("安装指引")
    self.ext_bench_hint_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ext_bench_hint_btn.clicked.connect(self._on_ext_bench_hint_clicked)
    ext_btn_row.addWidget(self.ext_bench_hint_btn)
    self.ext_bench_docs_btn = _eai_platform_bootstrap.QPushButton("打开文档/榜单")
    self.ext_bench_docs_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ext_bench_docs_btn.clicked.connect(self._on_ext_bench_docs_clicked)
    ext_btn_row.addWidget(self.ext_bench_docs_btn)
    ext_btn_row.addStretch(1)
    ext_page_l.addLayout(ext_btn_row)

    self.ext_bench_status_label = _eai_platform_bootstrap.QLabel("空闲 · 扩展评测")
    self.ext_bench_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.ext_bench_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.ext_bench_status_label.setWordWrap(True)
    ext_page_l.addWidget(self.ext_bench_status_label)

    ext_note = _eai_platform_bootstrap.QLabel(
        "扩展后端通过 tools/run_external_bench.py 统一启动；"
        "业内榜单来自 EmbodiedBench / AllenAI VLA Harness。"
        "策略可选 LingBot-VA（WAM）：LIBERO* 走官方 client，其余 serve_only。"
        "完整策略闭环需自行 clone 对应仓库或拉取 harness Docker。"
        "下方共享「启动」按钮与 MuJoCo 系日志区。"
    )
    ext_note.setWordWrap(True)
    ext_note.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    ext_page_l.addWidget(ext_note)
    ext_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(ext_page)
    self._refresh_ext_bench_form()

    # Shared MuJoCo / MolmoSpaces / Arena / LIBERO / RoboTwin / Extended run controls
    self.sim_backend_stack.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Maximum
    )
    build.sim_form_l.addWidget(self.sim_backend_stack)
