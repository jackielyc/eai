"""ui.builders.simulation robotwin — extracted application responsibilities."""
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
    robotwin_page = _eai_platform_bootstrap.QWidget()
    robotwin_page_l = _eai_platform_bootstrap.QVBoxLayout(robotwin_page)
    robotwin_page_l.setContentsMargins(0, 0, 0, 0)
    robotwin_page_l.setSpacing(6)

    robotwin_root_row = _eai_platform_bootstrap.QHBoxLayout()
    robotwin_root_row.setSpacing(6)
    robotwin_root_row.addWidget(_eai_platform_bootstrap.QLabel("RLinf"))
    self.robotwin_rlinf_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("RLINF_ROOT", _eai_configuration_tabs.RLINF_ROOT_DEFAULT)
    )
    self.robotwin_rlinf_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robotwin_rlinf_edit.setToolTip(
        "RLinf 仓库（examples/embodiment/eval_embodiment.sh）"
    )
    robotwin_root_row.addWidget(self.robotwin_rlinf_edit, 1)
    self.robotwin_rlinf_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robotwin_rlinf_browse_btn.setFixedWidth(28)
    self.robotwin_rlinf_browse_btn.clicked.connect(self._on_robotwin_rlinf_browse)
    robotwin_root_row.addWidget(self.robotwin_rlinf_browse_btn)
    robotwin_root_row.addWidget(_eai_platform_bootstrap.QLabel("RoboTwin"))
    self.robotwin_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ROBOTWIN_PATH", str(_eai_configuration_models.ROBOTWIN_ROOT_DEFAULT))
    )
    self.robotwin_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robotwin_root_edit.setToolTip(
        "RoboTwin 仓库（需 RLinf_support 分支，含 robotwin.envs.vector_env）"
    )
    robotwin_root_row.addWidget(self.robotwin_root_edit, 1)
    self.robotwin_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robotwin_root_browse_btn.setFixedWidth(28)
    self.robotwin_root_browse_btn.clicked.connect(self._on_robotwin_root_browse)
    robotwin_root_row.addWidget(self.robotwin_root_browse_btn)
    robotwin_page_l.addLayout(robotwin_root_row)

    robotwin_assets_row = _eai_platform_bootstrap.QHBoxLayout()
    robotwin_assets_row.setSpacing(6)
    robotwin_assets_row.addWidget(_eai_platform_bootstrap.QLabel("Assets"))
    self.robotwin_assets_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get(
            "ROBOTWIN_ASSETS",
            _eai_platform_bootstrap.os.environ.get("ASSETS_PATH", _eai_configuration_models.resolve_robotwin_assets()),
        )
    )
    self.robotwin_assets_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robotwin_assets_edit.setToolTip(
        "ASSETS_PATH：含 assets/objects 的根目录（通常即 RoboTwin 仓库根），"
        "不是内层 assets/。可用 script/_download_assets.sh 拉取。"
    )
    robotwin_assets_row.addWidget(self.robotwin_assets_edit, 1)
    self.robotwin_assets_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robotwin_assets_browse_btn.setFixedWidth(28)
    self.robotwin_assets_browse_btn.clicked.connect(self._on_robotwin_assets_browse)
    robotwin_assets_row.addWidget(self.robotwin_assets_browse_btn)
    robotwin_page_l.addLayout(robotwin_assets_row)

    robotwin_task_row = _eai_platform_bootstrap.QHBoxLayout()
    robotwin_task_row.setSpacing(6)
    robotwin_task_row.addWidget(_eai_platform_bootstrap.QLabel("任务"))
    self.robotwin_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, task_id in _eai_configuration_models.ROBOTWIN_TASKS:
        self.robotwin_task_combo.addItem(build.label, task_id)
    self.robotwin_task_combo.setToolTip(
        "RLinf 已提供 Hydra env YAML 的 RoboTwin 任务。"
        "下拉项带场景缩略图，悬停可看在做什么。"
    )
    _eai_configuration_models.stamp_eval_task_combo(self.robotwin_task_combo)
    robotwin_task_row.addWidget(self.robotwin_task_combo)
    robotwin_task_row.addWidget(_eai_platform_bootstrap.QLabel("配置"))
    self.robotwin_config_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, build.cfg_id in _eai_configuration_models.ROBOTWIN_POLICY_CONFIGS:
        self.robotwin_config_combo.addItem(build.label, build.cfg_id)
    self.robotwin_config_combo.setToolTip(
        "RLinf OpenPI dataconfig：pi05_aloha_robotwin / pi0_aloha_robotwin"
    )
    robotwin_task_row.addWidget(self.robotwin_config_combo)
    robotwin_task_row.addWidget(_eai_platform_bootstrap.QLabel("GPU"))
    self.robotwin_gpu_spin = _eai_platform_bootstrap.QSpinBox()
    self.robotwin_gpu_spin.setRange(0, 15)
    self.robotwin_gpu_spin.setValue(0)
    self.robotwin_gpu_spin.setToolTip("CUDA 设备号（冒烟默认单卡）")
    robotwin_task_row.addWidget(self.robotwin_gpu_spin)
    robotwin_task_row.addStretch(1)
    robotwin_page_l.addLayout(robotwin_task_row)
    (
        self.robotwin_task_preview,
        self.robotwin_task_thumb,
        _rt_task_cap,
    ) = _eai_configuration_models.make_eval_task_preview()
    _eai_configuration_models.bind_eval_task_thumb(self.robotwin_task_combo, self.robotwin_task_thumb)
    robotwin_page_l.addWidget(self.robotwin_task_preview)

    robotwin_ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
    robotwin_ckpt_row.setSpacing(6)
    robotwin_ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("Checkpoint"))
    self.robotwin_ckpt_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("ROBOTWIN_CKPT", _eai_configuration_tabs.ROBOTWIN_CKPT_DEFAULT)
    )
    self.robotwin_ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.robotwin_ckpt_edit.setToolTip(
        "actor.model.model_path；推荐 HF "
        f"{_eai_configuration_models.ROBOTWIN_CKPT_HF_DEFAULT} 下载到本地目录"
    )
    robotwin_ckpt_row.addWidget(self.robotwin_ckpt_edit, 1)
    self.robotwin_ckpt_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.robotwin_ckpt_browse_btn.setFixedWidth(28)
    self.robotwin_ckpt_browse_btn.clicked.connect(self._on_robotwin_ckpt_browse)
    robotwin_ckpt_row.addWidget(self.robotwin_ckpt_browse_btn)
    robotwin_page_l.addLayout(robotwin_ckpt_row)

    robotwin_param_row = _eai_ui_widgets_layouts._FlowLayout(spacing=6)
    robotwin_param_row.addWidget(_eai_platform_bootstrap.QLabel("并行环境"))
    self.robotwin_num_envs_spin = _eai_platform_bootstrap.QSpinBox()
    self.robotwin_num_envs_spin.setRange(1, 256)
    self.robotwin_num_envs_spin.setValue(1)
    self.robotwin_num_envs_spin.setToolTip("GUI 冒烟默认 1；正式评测可加大")
    robotwin_param_row.addWidget(self.robotwin_num_envs_spin)
    robotwin_param_row.addWidget(_eai_platform_bootstrap.QLabel("max_steps"))
    self.robotwin_max_steps_spin = _eai_platform_bootstrap.QSpinBox()
    self.robotwin_max_steps_spin.setRange(10, 2000)
    self.robotwin_max_steps_spin.setValue(200)
    robotwin_param_row.addWidget(self.robotwin_max_steps_spin)
    self.robotwin_render_gui_check = _eai_platform_bootstrap.QCheckBox("图形窗口")
    self.robotwin_render_gui_check.setChecked(
        bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip())
    )
    self.robotwin_render_gui_check.setToolTip(
        "勾选后弹出 SAPIEN Viewer 实时观看评测（render_freq>0）。\n"
        "评测结束后窗口会保持，需点击「停止」才退出。\n"
        "图形模式强制并行环境=1；无 DISPLAY 时可取消勾选，仅离屏并保存 mp4。"
    )
    robotwin_param_row.addWidget(self.robotwin_render_gui_check)
    robotwin_page_l.addLayout(robotwin_param_row)

    robotwin_hint = _eai_platform_bootstrap.QLabel(
        "对齐 RLinf examples/embodiment/eval_embodiment.sh（OpenPI + ALOHA）。"
        "需 RoboTwin RLinf_support + SAPIEN assets；与其它仿真后端互斥。"
        "可勾选「图形窗口」实时观看；结束后窗口保持直到点「停止」。"
        "工作区「RynnValue Live HUD」勾选后经 frame bus 单独弹出剩余时间窗口。"
        "结果仍保存到 video/eval/*.mp4。默认单卡 1 env 冒烟。"
    )
    robotwin_hint.setWordWrap(True)
    robotwin_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    robotwin_page_l.addWidget(robotwin_hint)
    robotwin_page_l.addStretch(1)
    self.sim_backend_stack.addWidget(robotwin_page)
