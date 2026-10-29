"""ui.builders.bagel — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_bagel_launchers = module_ref('eai_app.adapters.models.bagel_launchers')
_eai_adapters_models_bagel_runtime = module_ref('eai_app.adapters.models.bagel_runtime')
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.bagel_tab = _eai_platform_bootstrap.QWidget()
    self.bagel_tab = build.bagel_tab
    build.bagel_tab.setObjectName("bagelTab")
    build.bagel_tab.setStyleSheet(
        f"#bagelTab {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; }}"
        "#bagelTab QLabel { background: transparent; }"
    )
    bagel_outer = _eai_platform_bootstrap.QVBoxLayout(build.bagel_tab)
    bagel_outer.setContentsMargins(8, 6, 8, 6)
    bagel_outer.setSpacing(6)
    bagel_hint = _eai_platform_bootstrap.QLabel(
        "BAGEL：Gradio（「启动」）或推理 API（「部署推理 API」）。\n"
        "可将 sub task 选中图 + AI 对话输出点「从 sub task 导入」作为调用输入。"
    )
    bagel_hint.setWordWrap(True)
    bagel_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    bagel_outer.addWidget(bagel_hint)

    bagel_path_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_path_row.setSpacing(6)
    bagel_path_row.addWidget(_eai_platform_bootstrap.QLabel("仓库"))
    self.bagel_root_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_platform_bootstrap.os.environ.get("BAGEL_DIR", _eai_configuration_models.BAGEL_ROOT_DEFAULT)
    )
    self.bagel_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.bagel_root_edit.setToolTip("Bagel 仓库根目录（含 app.py）")
    bagel_path_row.addWidget(self.bagel_root_edit, 1)
    self.bagel_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.bagel_root_browse_btn.setFixedWidth(28)
    self.bagel_root_browse_btn.clicked.connect(self._on_bagel_root_browse_clicked)
    bagel_path_row.addWidget(self.bagel_root_browse_btn)
    bagel_path_row.addWidget(_eai_platform_bootstrap.QLabel("Python"))
    self.bagel_python_edit = _eai_platform_bootstrap.QLineEdit(
        _eai_adapters_models_bagel_runtime.resolve_bagel_python(_eai_adapters_models_bagel_runtime.resolve_bagel_root(self.bagel_root_edit.text()))
    )
    self.bagel_python_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.bagel_python_edit.setToolTip(
        "运行 app.py 的解释器。推荐 miniconda3/envs/bagel；也可设 BAGEL_PYTHON"
    )
    bagel_path_row.addWidget(self.bagel_python_edit, 1)
    self.bagel_python_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.bagel_python_browse_btn.setFixedWidth(28)
    self.bagel_python_browse_btn.clicked.connect(self._on_bagel_python_browse_clicked)
    bagel_path_row.addWidget(self.bagel_python_browse_btn)
    bagel_outer.addLayout(bagel_path_row)

    bagel_model_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_model_row.setSpacing(6)
    bagel_model_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    _default_model = (
        _eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT
        if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT)
        else _eai_configuration_models.BAGEL_MODEL_DEFAULT
    )
    self.bagel_model_edit = _eai_platform_bootstrap.QLineEdit(_default_model)
    self.bagel_model_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.bagel_model_edit.setToolTip(
        "相对仓库路径或绝对路径。默认可用 "
        ".../liyichao/models/BAGEL-7B-MoT"
    )
    bagel_model_row.addWidget(self.bagel_model_edit, 1)
    self.bagel_model_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.bagel_model_browse_btn.setFixedWidth(28)
    self.bagel_model_browse_btn.clicked.connect(self._on_bagel_model_browse_clicked)
    bagel_model_row.addWidget(self.bagel_model_browse_btn)
    self.bagel_download_btn = _eai_platform_bootstrap.QPushButton("下载模型")
    self.bagel_download_btn.setToolTip(
        "从 HuggingFace / hf-mirror 下载 ByteDance-Seed/BAGEL-7B-MoT "
        "到 .../liyichao/models/BAGEL-7B-MoT"
    )
    self.bagel_download_btn.clicked.connect(self._on_bagel_download_clicked)
    bagel_model_row.addWidget(self.bagel_download_btn)
    bagel_outer.addLayout(bagel_model_row)

    bagel_run_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_run_row.setSpacing(6)
    bagel_run_row.addWidget(_eai_platform_bootstrap.QLabel("host"))
    self.bagel_host_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT)
    self.bagel_host_edit.setFixedWidth(110)
    bagel_run_row.addWidget(self.bagel_host_edit)
    bagel_run_row.addWidget(_eai_platform_bootstrap.QLabel("port"))
    self.bagel_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.bagel_port_spin.setRange(1024, 65535)
    self.bagel_port_spin.setValue(_eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT)
    self.bagel_port_spin.setFixedWidth(72)
    bagel_run_row.addWidget(self.bagel_port_spin)
    bagel_run_row.addWidget(_eai_platform_bootstrap.QLabel("mode"))
    self.bagel_mode_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.bagel_mode_combo.addItem("1 · 全精度 (32GB+)", 1)
    self.bagel_mode_combo.addItem("2 · NF4 量化 (12~32GB)", 2)
    self.bagel_mode_combo.addItem("3 · INT8 量化", 3)
    self.bagel_mode_combo.setCurrentIndex(0)
    self.bagel_mode_combo.setToolTip(
        "对应 app.py --mode；低显存建议选 2 并勾选中文界面"
    )
    bagel_run_row.addWidget(self.bagel_mode_combo)
    self.bagel_zh_check = _eai_platform_bootstrap.QCheckBox("中文界面")
    self.bagel_zh_check.setChecked(True)
    bagel_run_row.addWidget(self.bagel_zh_check)
    self.bagel_share_check = _eai_platform_bootstrap.QCheckBox("share")
    self.bagel_share_check.setToolTip("Gradio --share 公网链接（可选）")
    bagel_run_row.addWidget(self.bagel_share_check)
    bagel_run_row.addStretch()
    self.bagel_status_label = _eai_platform_bootstrap.QLabel("Bagel: 空闲")
    self.bagel_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    bagel_run_row.addWidget(self.bagel_status_label)
    self.bagel_start_btn = _eai_platform_bootstrap.QPushButton("启动")
    self.bagel_start_btn.setToolTip("python app.py（加载模型可能较久）")
    self.bagel_start_btn.clicked.connect(self._on_bagel_start_clicked)
    bagel_run_row.addWidget(self.bagel_start_btn)
    self.bagel_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.bagel_stop_btn.setObjectName("dangerAction")
    self.bagel_stop_btn.setEnabled(False)
    self.bagel_stop_btn.clicked.connect(self._on_bagel_stop_clicked)
    bagel_run_row.addWidget(self.bagel_stop_btn)
    self.bagel_open_btn = _eai_platform_bootstrap.QPushButton("浏览器打开")
    self.bagel_open_btn.setEnabled(False)
    self.bagel_open_btn.clicked.connect(self._on_bagel_open_clicked)
    bagel_run_row.addWidget(self.bagel_open_btn)
    self.bagel_pick_image_btn = _eai_platform_bootstrap.QPushButton("选择图片")
    self.bagel_pick_image_btn.setEnabled(False)
    self.bagel_pick_image_btn.setToolTip(
        "用系统对话框选图并注入到嵌入的 Gradio（避开 WebEngine 空文件问题）"
    )
    self.bagel_pick_image_btn.clicked.connect(self._on_bagel_pick_image_clicked)
    bagel_run_row.addWidget(self.bagel_pick_image_btn)
    self.bagel_clear_log_btn = _eai_platform_bootstrap.QPushButton("清空日志")
    self.bagel_clear_log_btn.clicked.connect(self._on_bagel_clear_log_clicked)
    bagel_run_row.addWidget(self.bagel_clear_log_btn)
    bagel_outer.addLayout(bagel_run_row)

    bagel_api_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_api_row.setSpacing(6)
    bagel_api_row.addWidget(_eai_platform_bootstrap.QLabel("推理 API 端口"))
    self.bagel_api_port_spin = _eai_platform_bootstrap.QSpinBox()
    self.bagel_api_port_spin.setRange(1024, 65535)
    self.bagel_api_port_spin.setValue(_eai_configuration_models.BAGEL_API_PORT_DEFAULT)
    self.bagel_api_port_spin.setFixedWidth(72)
    self.bagel_api_port_spin.setToolTip("serve_api.py 监听端口，默认 7861（避开 Gradio 7860）")
    bagel_api_row.addWidget(self.bagel_api_port_spin)
    self.bagel_api_status_label = _eai_platform_bootstrap.QLabel("推理 API: 未部署")
    self.bagel_api_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    bagel_api_row.addWidget(self.bagel_api_status_label, 1)
    self.bagel_api_start_btn = _eai_platform_bootstrap.QPushButton("部署推理 API")
    self.bagel_api_start_btn.setToolTip(
        "只加载模型并启动 HTTP（无 Gradio）。\n"
        "若目标端口上已有就绪的 serve_api，将直接复用，不重复加载。\n"
        "随后可用下方「调用」或 AI 对话。"
    )
    self.bagel_api_start_btn.clicked.connect(self._on_bagel_api_start_clicked)
    bagel_api_row.addWidget(self.bagel_api_start_btn)
    self.bagel_api_stop_btn = _eai_platform_bootstrap.QPushButton("停止 API")
    self.bagel_api_stop_btn.setObjectName("dangerAction")
    self.bagel_api_stop_btn.setEnabled(False)
    self.bagel_api_stop_btn.setToolTip(
        "停止推理 API，并清理所有相关 Bagel 进程（含 Gradio / 残留 serve_api）。"
    )
    self.bagel_api_stop_btn.clicked.connect(self._on_bagel_api_stop_clicked)
    bagel_api_row.addWidget(self.bagel_api_stop_btn)
    bagel_outer.addLayout(bagel_api_row)

    bagel_call_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_call_row.setSpacing(6)
    bagel_call_row.addWidget(_eai_platform_bootstrap.QLabel("调用"))
    self.bagel_call_task_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.bagel_call_task_combo.addItem("文生图", "text2image")
    self.bagel_call_task_combo.addItem("图像理解", "understand")
    self.bagel_call_task_combo.addItem("图像编辑", "edit")
    self.bagel_call_task_combo.setToolTip("理解/编辑需要点「选图」")
    bagel_call_row.addWidget(self.bagel_call_task_combo)
    bagel_call_row.addWidget(_eai_platform_bootstrap.QLabel("steps"))
    self.bagel_num_timesteps_spin = _eai_platform_bootstrap.QSpinBox()
    self.bagel_num_timesteps_spin.setRange(1, 100)
    self.bagel_num_timesteps_spin.setSingleStep(1)
    self.bagel_num_timesteps_spin.setValue(_eai_configuration_models.BAGEL_NUM_TIMESTEPS_DEFAULT)
    self.bagel_num_timesteps_spin.setFixedWidth(64)
    self.bagel_num_timesteps_spin.setToolTip(
        "文生图 / 图像编辑的去噪步数 num_timesteps（1–100，步长 1；默认 25）"
    )
    bagel_call_row.addWidget(self.bagel_num_timesteps_spin)
    bagel_call_row.addWidget(_eai_platform_bootstrap.QLabel("尺寸"))
    self.bagel_image_size_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.bagel_image_size_combo.addItem("跟随输入", "input")
    self.bagel_image_size_combo.addItem("224²", "224")
    self.bagel_image_size_combo.addItem("320² psi", "320")
    self.bagel_image_size_combo.addItem("512² 快", "512")
    self.bagel_image_size_combo.addItem("768²", "768")
    self.bagel_image_size_combo.addItem("1024²", "1024")
    self.bagel_image_size_combo.addItem("1:1 1024", "1:1")
    self.bagel_image_size_combo.addItem("16:9", "16:9")
    self.bagel_image_size_combo.addItem("9:16", "9:16")
    self.bagel_image_size_combo.addItem("4:3", "4:3")
    self.bagel_image_size_combo.addItem("3:4", "3:4")
    self.bagel_image_size_combo.addItem("手动 W×H", "custom")
    self.bagel_image_size_combo.setCurrentIndex(0)  # 跟随输入
    self.bagel_image_size_combo.setToolTip(
        "输出分辨率。默认「跟随输入」：有输入图则用其大小；"
        "纯文生图无输入时回退 512²。224²/320² 适合小图；"
        "「手动 W×H」可指定宽高（16 对齐，最大 1024）。"
    )
    bagel_call_row.addWidget(self.bagel_image_size_combo)
    bagel_call_row.addWidget(_eai_platform_bootstrap.QLabel("W"))
    self.bagel_image_w_spin = _eai_ui_widgets_inputs.HoverWheelSpinBox()
    self.bagel_image_w_spin.setRange(16, 1024)
    self.bagel_image_w_spin.setSingleStep(16)
    self.bagel_image_w_spin.setValue(224)
    self.bagel_image_w_spin.setFixedWidth(64)
    self.bagel_image_w_spin.setToolTip(
        "输出宽度（像素，按 16 对齐）。滚轮或箭头可调；改动后自动切到「手动 W×H」。"
    )
    bagel_call_row.addWidget(self.bagel_image_w_spin)
    bagel_call_row.addWidget(_eai_platform_bootstrap.QLabel("H"))
    self.bagel_image_h_spin = _eai_ui_widgets_inputs.HoverWheelSpinBox()
    self.bagel_image_h_spin.setRange(16, 1024)
    self.bagel_image_h_spin.setSingleStep(16)
    self.bagel_image_h_spin.setValue(224)
    self.bagel_image_h_spin.setFixedWidth(64)
    self.bagel_image_h_spin.setToolTip(
        "输出高度（像素，按 16 对齐）。滚轮或箭头可调；改动后自动切到「手动 W×H」。"
    )
    bagel_call_row.addWidget(self.bagel_image_h_spin)
    self._bagel_wh_syncing = False
    self.bagel_image_size_combo.currentIndexChanged.connect(
        self._on_bagel_image_size_changed
    )
    self.bagel_image_w_spin.valueChanged.connect(self._on_bagel_image_wh_changed)
    self.bagel_image_h_spin.valueChanged.connect(self._on_bagel_image_wh_changed)
    self._on_bagel_image_size_changed()
    self.bagel_call_pick_btn = _eai_platform_bootstrap.QPushButton("选图")
    self.bagel_call_pick_btn.setToolTip("理解 / 编辑用输入图")
    self.bagel_call_pick_btn.clicked.connect(self._on_bagel_call_pick_clicked)
    bagel_call_row.addWidget(self.bagel_call_pick_btn)
    self.bagel_import_subtask_btn = _eai_platform_bootstrap.QPushButton("从 sub task 导入")
    self.bagel_import_subtask_btn.setToolTip(
        "导入 sub task 页选中的场景图，以及 AI 对话中最近一次模型输出，"
        "作为本页调用的输入图 / 提示词（默认切到「图像编辑」）"
    )
    self.bagel_import_subtask_btn.clicked.connect(
        self._on_bagel_import_from_sub_task_clicked
    )
    bagel_call_row.addWidget(self.bagel_import_subtask_btn)
    self.bagel_call_btn = _eai_platform_bootstrap.QPushButton("调用")
    self.bagel_call_btn.setEnabled(False)
    self.bagel_call_btn.setToolTip("向已部署的 Bagel 推理 API 发请求")
    self.bagel_call_btn.clicked.connect(self._on_bagel_call_clicked)
    bagel_call_row.addWidget(self.bagel_call_btn)
    bagel_outer.addLayout(bagel_call_row)

    bagel_prompt_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_prompt_row.setSpacing(6)
    bagel_prompt_row.addWidget(_eai_platform_bootstrap.QLabel("提示词"))
    self.bagel_call_prompt_edit = _eai_ui_widgets_inputs.ImeSafeTextEdit()
    self.bagel_call_prompt_edit.setPlaceholderText("输入提示词后点「调用」")
    self.bagel_call_prompt_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    _eai_ui_widgets_layouts._configure_resizable_text_edit(
        self.bagel_call_prompt_edit, default_h=88, tooltip=""
    )
    bagel_prompt_row.addWidget(self.bagel_call_prompt_edit, 1)
    bagel_outer.addLayout(bagel_prompt_row)
    self.bagel_call_prompt_resize_bar = None

    bagel_speed_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_speed_row.setSpacing(6)
    bagel_speed_row.addWidget(_eai_platform_bootstrap.QLabel("加速"))
    self.bagel_fast_preset_btn = _eai_platform_bootstrap.QPushButton("快速预设")
    self.bagel_fast_preset_btn.setToolTip(
        f"一键：steps={_eai_configuration_models.BAGEL_FAST_NUM_TIMESTEPS}、输出 {_eai_configuration_models.BAGEL_FAST_IMAGE_SIZE}²、"
        f"cfg_interval={_eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT}、开启 TaylorSeer"
    )
    self.bagel_fast_preset_btn.clicked.connect(self._on_bagel_fast_preset_clicked)
    bagel_speed_row.addWidget(self.bagel_fast_preset_btn)
    bagel_speed_row.addWidget(_eai_platform_bootstrap.QLabel("cfg_interval"))
    self.bagel_cfg_interval_spin = _eai_platform_bootstrap.QDoubleSpinBox()
    self.bagel_cfg_interval_spin.setRange(0.0, 1.0)
    self.bagel_cfg_interval_spin.setSingleStep(0.1)
    self.bagel_cfg_interval_spin.setDecimals(1)
    self.bagel_cfg_interval_spin.setValue(_eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT)
    self.bagel_cfg_interval_spin.setFixedWidth(64)
    self.bagel_cfg_interval_spin.setToolTip(
        "CFG 起始比例：0=全程 CFG（慢），0.4=前 40% 跳过 CFG（更快，官方 T2I 常用）"
    )
    bagel_speed_row.addWidget(self.bagel_cfg_interval_spin)
    self.bagel_taylorseer_check = _eai_platform_bootstrap.QCheckBox("TaylorSeer")
    self.bagel_taylorseer_check.setChecked(False)
    self.bagel_taylorseer_check.setToolTip(
        "开启后用 TaylorSeer 加速去噪（可能略损质量；需重启后的 serve_api 支持该字段）"
    )
    bagel_speed_row.addWidget(self.bagel_taylorseer_check)
    bagel_speed_row.addStretch(1)
    bagel_outer.addLayout(bagel_speed_row)

    bagel_hist_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_hist_row.setSpacing(6)
    self.bagel_history_title_label = _eai_platform_bootstrap.QLabel("当前: 新对话（未保存）")
    self.bagel_history_title_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.bagel_history_title_label.setWordWrap(True)
    bagel_hist_row.addWidget(self.bagel_history_title_label, 1)
    self.bagel_history_new_btn = _eai_platform_bootstrap.QPushButton("新对话")
    self.bagel_history_new_btn.setToolTip("清空当前轮次，开始新的 Bagel 会话")
    self.bagel_history_new_btn.clicked.connect(self._on_bagel_history_new_clicked)
    bagel_hist_row.addWidget(self.bagel_history_new_btn)
    self.bagel_history_save_btn = _eai_platform_bootstrap.QPushButton("保存")
    self.bagel_history_save_btn.setToolTip(
        f"保存当前会话到本地（{_eai_configuration_chat.BAGEL_HISTORY_DIR}）"
    )
    self.bagel_history_save_btn.clicked.connect(self._on_bagel_history_save_clicked)
    bagel_hist_row.addWidget(self.bagel_history_save_btn)
    self.bagel_history_btn = _eai_platform_bootstrap.QPushButton("历史")
    self.bagel_history_btn.setToolTip("浏览 / 加载 / 改名 / 删除 Bagel 历史对话")
    self.bagel_history_btn.clicked.connect(self._on_bagel_history_clicked)
    bagel_hist_row.addWidget(self.bagel_history_btn)
    self.bagel_history_del_turn_btn = _eai_platform_bootstrap.QPushButton("删本轮")
    self.bagel_history_del_turn_btn.setObjectName("dangerAction")
    self.bagel_history_del_turn_btn.setToolTip("删除左侧列表中选中的调用轮次")
    self.bagel_history_del_turn_btn.clicked.connect(
        self._on_bagel_history_delete_turn_clicked
    )
    bagel_hist_row.addWidget(self.bagel_history_del_turn_btn)
    bagel_outer.addLayout(bagel_hist_row)

    bagel_preview_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_preview_row.setSpacing(6)
    bagel_turns_col = _eai_platform_bootstrap.QVBoxLayout()
    bagel_turns_col.setContentsMargins(0, 0, 0, 0)
    bagel_turns_col.setSpacing(4)
    bagel_turns_col.addWidget(_eai_platform_bootstrap.QLabel("本会话轮次"))
    self.bagel_history_turns_list = _eai_platform_bootstrap.QListWidget()
    self.bagel_history_turns_list.setMinimumWidth(160)
    self.bagel_history_turns_list.setMaximumWidth(220)
    self.bagel_history_turns_list.setStyleSheet(
        f"QListWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.bagel_history_turns_list.itemClicked.connect(
        self._on_bagel_history_turn_clicked
    )
    bagel_turns_col.addWidget(self.bagel_history_turns_list, 1)
    bagel_preview_row.addLayout(bagel_turns_col)
    bagel_in_col = _eai_platform_bootstrap.QVBoxLayout()
    bagel_in_col.setContentsMargins(0, 0, 0, 0)
    bagel_in_col.setSpacing(4)
    self.bagel_call_input_preview = _eai_platform_bootstrap.QLabel("输入图")
    self.bagel_call_input_preview.setFixedSize(120, 90)
    self.bagel_call_input_preview.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.bagel_call_input_preview.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; color: #888; }}"
    )
    bagel_in_col.addWidget(self.bagel_call_input_preview)
    self.bagel_call_input_size_label = _eai_platform_bootstrap.QLabel("输入分辨率: —")
    self.bagel_call_input_size_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.bagel_call_input_size_label.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    self.bagel_call_input_size_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.bagel_call_input_size_label.setToolTip("输入图像的宽×高（像素）")
    bagel_in_col.addWidget(self.bagel_call_input_size_label)
    bagel_preview_row.addLayout(bagel_in_col)
    bagel_out_col = _eai_platform_bootstrap.QVBoxLayout()
    bagel_out_col.setContentsMargins(0, 0, 0, 0)
    bagel_out_col.setSpacing(4)
    self.bagel_call_output_preview = _eai_platform_bootstrap.QLabel("输出图")
    self.bagel_call_output_preview.setMinimumHeight(160)
    self.bagel_call_output_preview.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.bagel_call_output_preview.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; color: #888; }}"
    )
    self.bagel_call_output_preview.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    bagel_out_col.addWidget(self.bagel_call_output_preview, 1)
    bagel_meta_row = _eai_platform_bootstrap.QHBoxLayout()
    bagel_meta_row.setContentsMargins(0, 0, 0, 0)
    bagel_meta_row.setSpacing(10)
    self.bagel_call_output_size_label = _eai_platform_bootstrap.QLabel("输出分辨率: —")
    self.bagel_call_output_size_label.setAlignment(_eai_platform_bootstrap.Qt.AlignLeft | _eai_platform_bootstrap.Qt.AlignVCenter)
    self.bagel_call_output_size_label.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    self.bagel_call_output_size_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.bagel_call_output_size_label.setToolTip("实际输出图像的宽×高（像素）")
    bagel_meta_row.addWidget(self.bagel_call_output_size_label)
    self.bagel_call_model_size_label = _eai_platform_bootstrap.QLabel("模型输入: —")
    self.bagel_call_model_size_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.bagel_call_model_size_label.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    self.bagel_call_model_size_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.bagel_call_model_size_label.setToolTip(
        "实际送进模型的条件图尺寸（保持原图比例；仅 16 对齐，最长边 ≤1024）"
    )
    bagel_meta_row.addWidget(self.bagel_call_model_size_label)
    self.bagel_call_latency_label = _eai_platform_bootstrap.QLabel("调用耗时: —")
    self.bagel_call_latency_label.setAlignment(_eai_platform_bootstrap.Qt.AlignRight | _eai_platform_bootstrap.Qt.AlignVCenter)
    self.bagel_call_latency_label.setFont(
        _eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL)
    )
    self.bagel_call_latency_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.bagel_call_latency_label.setToolTip("Bagel 推理 API 返回的 latency_s（秒）")
    bagel_meta_row.addWidget(self.bagel_call_latency_label)
    bagel_out_col.addLayout(bagel_meta_row)
    bagel_preview_row.addLayout(bagel_out_col, 1)
    self._bagel_preview_layout = bagel_preview_row
    bagel_outer.addLayout(bagel_preview_row, 1)
    self._bagel_call_image_path = ""
    self._bagel_call_busy = False
    self._bagel_history_id = ""
    self._bagel_history_title = ""
    self._bagel_history_turns: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
    self._bagel_history_dialog = None
    self._refresh_bagel_history_title_label()
    self._refresh_bagel_history_turns_list()

    self.bagel_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.bagel_log_edit.setReadOnly(True)
    self.bagel_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.bagel_log_edit.setPlaceholderText("Bagel 启动 / 推理日志…")
    self.bagel_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.bagel_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        bagel_outer,
        self.bagel_log_edit,
        default_h=160,
        min_h=80,
        max_h=900,
        label="拖动调整 Bagel 日志高度",
    )

    # 延迟创建 QWebEngineView：启动期立刻创建会在无/弱 GLX 下 SIP segfault
    # （与 Astra 页同一策略）。真正加载 Gradio URL 时再 _ensure_bagel_web_view。
    self.bagel_web_view = None
    self.bagel_web_page = None
    self._bagel_web_container = _eai_platform_bootstrap.QWidget()
    self._bagel_web_container_layout = _eai_platform_bootstrap.QVBoxLayout(self._bagel_web_container)
    self._bagel_web_container_layout.setContentsMargins(0, 0, 0, 0)
    if _eai_platform_bootstrap.QWebEngineView is not None and self._want_control_tab("sub image"):
        self._bagel_web_placeholder = _eai_platform_bootstrap.QLabel(
            "Bagel 嵌入页按需加载：启动 Gradio 或点「浏览器打开」后创建 WebEngine。"
        )
        self._bagel_web_placeholder.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._bagel_web_placeholder.setWordWrap(True)
        self._bagel_web_placeholder.setMinimumHeight(0)
        self._bagel_web_placeholder.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._bagel_web_container_layout.addWidget(self._bagel_web_placeholder, 0)
        self._bagel_web_fallback = None
    else:
        self._bagel_web_placeholder = None
        if _eai_platform_bootstrap.QWebEngineView is None:
            bagel_msg = (
                "未安装 PyQtWebEngine：无法页内嵌入。启动后请点「浏览器打开」。"
            )
        else:
            bagel_msg = "当前未启用 sub image 页内嵌入（--tab 过滤）。"
        bagel_web_fallback = _eai_platform_bootstrap.QLabel(bagel_msg)
        bagel_web_fallback.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        bagel_web_fallback.setMinimumHeight(0)
        bagel_web_fallback.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._bagel_web_container_layout.addWidget(bagel_web_fallback, 0)
        bagel_web_fallback.hide()
        self._bagel_web_fallback = bagel_web_fallback
    bagel_outer.addWidget(self._bagel_web_container, 0)
    self._bagel_web_container.hide()

    self._bagel_launcher = _eai_adapters_models_bagel_launchers.BagelAppLauncher(self)
    self._bagel_launcher.log_line.connect(self._append_bagel_log)
    self._bagel_launcher.status_message.connect(self._on_bagel_status)
    self._bagel_launcher.running_changed.connect(self._update_bagel_run_ui)
    self._bagel_launcher.ready_url.connect(self._on_bagel_ready_url)
    self._bagel_ready_poll_timer = _eai_platform_bootstrap.QTimer(self)
    self._bagel_ready_poll_timer.setInterval(1500)
    self._bagel_ready_poll_timer.timeout.connect(self._poll_bagel_http_ready)

    self._bagel_api_launcher = _eai_adapters_models_bagel_launchers.BagelApiLauncher(self)
    self._bagel_api_launcher.log_line.connect(self._append_bagel_log)
    self._bagel_api_launcher.status_message.connect(self._on_bagel_api_status)
    self._bagel_api_launcher.running_changed.connect(self._update_bagel_api_ui)
    self._bagel_api_launcher.ready_url.connect(self._on_bagel_api_ready)
    self._bagel_api_ready_poll_timer = _eai_platform_bootstrap.QTimer(self)
    self._bagel_api_ready_poll_timer.setInterval(1500)
    self._bagel_api_ready_poll_timer.timeout.connect(self._poll_bagel_api_ready)
    self._bagel_call_bridge = _eai_adapters_models_chat_client.LlmProbeBridge()
    self._bagel_call_bridge.finished.connect(self._on_bagel_call_finished)
    self._update_bagel_api_ui()
    self._sync_bagel_gradio_embed()
    _eai_platform_bootstrap.QTimer.singleShot(700, self._adopt_persisted_bagel)
