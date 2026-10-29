"""ui.chat.panel — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_chat_history_dialogs = module_ref('eai_app.ui.chat.history_dialogs')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

from eai_app.ui.chat.parts.actions import ActionsMixin
from eai_app.ui.chat.parts.actions_append_system_line import ActionsAppendSystemLineMixin

class ChatPanelWidget(ActionsMixin, ActionsAppendSystemLineMixin, _eai_platform_bootstrap.QWidget):
    """文本/视觉对话面板：OpenAI 兼容 API（含 Hy-Embodied-VLM / RxBrain）。"""
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    attached_image_changed = _eai_platform_bootstrap.pyqtSignal(str)
    def __init__(
        self,
        config: _eai_platform_bootstrap.Optional[_eai_adapters_models_chat_client.LlmChatConfig] = None,
        parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or _eai_adapters_models_chat_client.LlmChatConfig.from_env()
        saved_settings = _eai_infrastructure_ui_storage.load_chat_user_settings()
        saved_system = str(saved_settings.get("system_prompt") or "").strip()
        if saved_system:
            self._config.system_prompt = saved_system
        saved_lang = str(saved_settings.get("system_prompt_lang") or "").strip().lower()
        if saved_lang in ("en", "zh", "en2") and not saved_system:
            self._config.system_prompt = _eai_configuration_chat.lake_orchestrator_system_prompt_for_lang(
                saved_lang
            )
        self._client = _eai_adapters_models_chat_client.LlmChatClient(self._config)
        self._messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
        self._lake_language_memory: str = _eai_configuration_chat.lake_default_previous_subtask(
            _eai_configuration_chat.lake_orchestrator_prompt_lang(self._config.system_prompt)
        )
        self._traditional_chat_mode: bool = False
        self._busy = False
        self._history_id: str = ""
        self._history_title: str = ""
        self._request_t0: _eai_platform_bootstrap.Optional[float] = None
        self._bridge = _eai_adapters_models_chat_client.LlmChatBridge()
        self._bridge.finished.connect(self._on_llm_finished)
        self._probe_bridge = _eai_adapters_models_chat_client.LlmProbeBridge()
        self._probe_bridge.finished.connect(self._on_probe_finished)
        self._camera_frame_provider: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Callable[[], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray]]]
        ] = None

        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # 多会话：每个 tab 独立对话，点「+」新开
        self._sessions: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
        self._session_index = 0
        self._session_switch_guard = False
        session_row = _eai_platform_bootstrap.QHBoxLayout()
        session_row.setSpacing(2)
        self.session_tabs = _eai_platform_bootstrap.QTabBar()
        self.session_tabs.setExpanding(False)
        self.session_tabs.setTabsClosable(True)
        self.session_tabs.setMovable(True)
        self.session_tabs.setDrawBase(False)
        self.session_tabs.setUsesScrollButtons(True)
        self.session_tabs.setElideMode(_eai_platform_bootstrap.Qt.ElideRight)
        self.session_tabs.setToolTip("切换对话；点 × 关闭当前会话（至少一个）")
        self.session_tabs.currentChanged.connect(self._on_session_tab_changed)
        self.session_tabs.tabCloseRequested.connect(self._on_session_tab_close)
        self.session_tabs.tabMoved.connect(self._on_session_tab_moved)
        session_row.addWidget(self.session_tabs, 1)
        self.new_session_btn = _eai_platform_bootstrap.QToolButton()
        self.new_session_btn.setText("+")
        self.new_session_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.new_session_btn.setToolTip("新起一个对话（可另选模型）")
        self.new_session_btn.clicked.connect(self._on_new_session_clicked)
        session_row.addWidget(self.new_session_btn)
        layout.addLayout(session_row)

        # 紧凑顶栏：仅可选「已在运行」的推理服务
        header = _eai_platform_bootstrap.QHBoxLayout()
        header.setSpacing(4)
        self.provider_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
        self.provider_combo.setToolTip(
            "仅列出已部署且正在运行的推理服务（本地/远程 Qwen、Ollama、Hy-Embodied）。\n"
            "点「探测」可刷新列表；请先在「测试」等页启动对应服务。"
        )
        self.provider_combo.currentIndexChanged.connect(self._on_running_service_changed)
        header.addWidget(self.provider_combo, 1)
        self.model_edit = _eai_platform_bootstrap.QLineEdit(self._config.model)
        self.model_edit.setPlaceholderText("模型名")
        self.model_edit.setToolTip("由上方运行中服务自动填充（只读）")
        self.model_edit.setMinimumWidth(90)
        self.model_edit.setMaximumWidth(160)
        self.model_edit.setReadOnly(True)
        self.model_edit.textChanged.connect(lambda _t: self._sync_session_tab_title())
        header.addWidget(self.model_edit)
        self.settings_toggle_btn = _eai_platform_bootstrap.QToolButton()
        self.settings_toggle_btn.setText("设置")
        self.settings_toggle_btn.setCheckable(True)
        self.settings_toggle_btn.setChecked(False)
        self.settings_toggle_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.settings_toggle_btn.setToolTip("展开/收起 API、Key 与 System prompt 设置")
        self.settings_toggle_btn.toggled.connect(self._on_settings_toggled)
        header.addWidget(self.settings_toggle_btn)
        self.probe_btn = _eai_platform_bootstrap.QPushButton("探测")
        self.probe_btn.setFixedWidth(44)
        self.probe_btn.setToolTip(
            "刷新运行中的推理服务列表，并探测当前选中服务"
        )
        self.probe_btn.clicked.connect(self._on_probe_clicked)
        header.addWidget(self.probe_btn)
        self.save_btn = _eai_platform_bootstrap.QPushButton("保存")
        self.save_btn.setFixedWidth(44)
        self.save_btn.setToolTip("保存当前对话到本地历史（eai/chat_history）")
        self.save_btn.clicked.connect(self._on_save_chat_clicked)
        header.addWidget(self.save_btn)
        self.history_btn = _eai_platform_bootstrap.QPushButton("历史")
        self.history_btn.setFixedWidth(44)
        self.history_btn.setToolTip(
            "加载 / 修改标题 / 删除已保存的对话（加载后问题进输入框，点发送再请求）"
        )
        self.history_btn.clicked.connect(self._on_history_clicked)
        header.addWidget(self.history_btn)
        self.traditional_chat_btn = _eai_platform_bootstrap.QPushButton("传统对话")
        self.traditional_chat_btn.setFixedWidth(64)
        self.traditional_chat_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.traditional_chat_btn.setToolTip(
            "切换为传统多轮对话：清空 System prompt，不再注入 Lake 编排提示词"
        )
        self.traditional_chat_btn.clicked.connect(self._on_traditional_chat_clicked)
        header.addWidget(self.traditional_chat_btn)
        self.system_lang_btn = _eai_platform_bootstrap.QPushButton("English")
        self.system_lang_btn.setFixedWidth(72)
        self.system_lang_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.system_lang_btn.setToolTip(
            "中文 → English 完整版 → EN 两行简版（All/Current）循环切换"
        )
        self.system_lang_btn.clicked.connect(self._on_toggle_system_prompt_lang)
        header.addWidget(self.system_lang_btn)
        self.clear_btn = _eai_platform_bootstrap.QPushButton("清空")
        self.clear_btn.setFixedWidth(44)
        self.clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        self.history_title_label = _eai_platform_bootstrap.QLabel("当前: 新对话（未保存）")
        self.history_title_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self.history_title_label.setWordWrap(True)
        layout.addWidget(self.history_title_label)

        self.settings_panel = _eai_platform_bootstrap.QWidget()
        settings_layout = _eai_platform_bootstrap.QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(3)
        settings_row = _eai_platform_bootstrap.QHBoxLayout()
        settings_row.addWidget(_eai_platform_bootstrap.QLabel("API"))
        self.api_base_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit(self._config.api_base)
        self.api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        self.api_base_edit.setToolTip(
            "由上方运行中服务自动填充（只读）。OpenAI 兼容：Qwen / Ollama / Hy-Embodied"
        )
        self.api_base_edit.setReadOnly(True)
        settings_row.addWidget(self.api_base_edit, stretch=1)
        settings_layout.addLayout(settings_row)
        key_row = _eai_platform_bootstrap.QHBoxLayout()
        key_row.addWidget(_eai_platform_bootstrap.QLabel("Key"))
        self.api_key_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit(self._config.api_key)
        self.api_key_edit.setEchoMode(_eai_platform_bootstrap.QLineEdit.Password)
        self.api_key_edit.setPlaceholderText(f"或设置 ${_eai_configuration_chat.LLM_API_KEY_ENV}")
        self.api_key_edit.setToolTip(
            f"API Key；Ollama/Hy-Embodied 本地可填 ollama 或 EMPTY。"
            f"也可 export {_eai_configuration_chat.LLM_API_KEY_ENV}=..."
        )
        key_row.addWidget(self.api_key_edit, stretch=1)
        settings_layout.addLayout(key_row)
        system_label = _eai_platform_bootstrap.QLabel("System")
        system_label.setToolTip("Lake / Qwen 推理时作为 system 角色发送；可编辑并保存到本地")
        settings_layout.addWidget(system_label)
        self.system_prompt_edit = _eai_ui_widgets_inputs.ImeSafeTextEdit()
        self.system_prompt_edit.setPlainText(self._config.system_prompt)
        self.system_prompt_edit.setPlaceholderText("System prompt…")
        self.system_prompt_edit.setToolTip(
            "认知编排器 system 提示词；Lake 模式下随请求发送。"
            "点「保存 System」写入 eai/config/chat_user_settings.json。"
        )
        self.system_prompt_edit.setStyleSheet(
            f"QTextEdit {{ background-color: {_eai_configuration_policies.UI_BG_LOG}; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; "
            f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; border-radius: {_eai_configuration_policies.UI_RADIUS}; }}"
        )
        self.system_prompt_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
            settings_layout,
            self.system_prompt_edit,
            default_h=160,
            min_h=72,
            max_h=600,
            label="拖动调整 System 高度",
        )
        system_btn_row = _eai_platform_bootstrap.QHBoxLayout()
        self.save_system_btn = _eai_platform_bootstrap.QPushButton("保存 System")
        self.save_system_btn.setToolTip(
            f"保存 system prompt 到 {_eai_configuration_chat.CHAT_USER_SETTINGS_PATH}"
        )
        self.save_system_btn.clicked.connect(self._on_save_system_prompt_clicked)
        system_btn_row.addWidget(self.save_system_btn)
        self.reset_system_btn = _eai_platform_bootstrap.QPushButton("恢复默认")
        self.reset_system_btn.setToolTip(
            "恢复为 Lake 训练默认 system（技能/子任务/记忆 三行版）"
        )
        self.reset_system_btn.clicked.connect(self._on_reset_system_prompt_clicked)
        system_btn_row.addWidget(self.reset_system_btn)
        system_btn_row.addStretch(1)
        settings_layout.addLayout(system_btn_row)
        self.settings_panel.setVisible(False)
        layout.addWidget(self.settings_panel)
        self._refresh_system_lang_btn()

        opt_row = _eai_platform_bootstrap.QHBoxLayout()
        opt_row.setSpacing(4)
        self.attach_camera_check = _eai_platform_bootstrap.QCheckBox("附带相机图")
        self.attach_camera_check.setChecked(True)
        self.attach_camera_check.setToolTip(
            "发送时附带当前彩色相机帧（未在测试页点选场景图时生效；"
            "Hy-Embodied / Qwen 等多模态模型需要）"
        )
        opt_row.addWidget(self.attach_camera_check)
        self.clear_image_btn = _eai_platform_bootstrap.QPushButton("清除场景图")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.setToolTip("清除已在测试页点选的场景图")
        self.clear_image_btn.clicked.connect(
            lambda _checked=False: self._on_clear_chat_image_clicked()
        )
        opt_row.addWidget(self.clear_image_btn)
        self.thinking_check = _eai_platform_bootstrap.QCheckBox("thinking")
        self.thinking_check.setToolTip(
            "Hy-Embodied-VLM：chat_template_kwargs.enable_thinking=true（慢但推理更强）"
        )
        opt_row.addWidget(self.thinking_check)
        opt_row.addStretch(1)
        self.chat_image_label = _eai_platform_bootstrap.QLabel("未选场景图")
        self.chat_image_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.chat_image_label.setWordWrap(False)
        self.chat_image_label.setMinimumWidth(0)
        self.chat_image_label.setToolTip(
            "在「测试」页点击一张场景图，或勾选附带相机图"
        )
        opt_row.addWidget(self.chat_image_label, 1)
        self.chat_image_preview = _eai_platform_bootstrap.QLabel()
        self.chat_image_preview.setFixedSize(56, 42)
        self.chat_image_preview.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.chat_image_preview.setStyleSheet(
            f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
            f"border-radius: {_eai_configuration_policies.UI_RADIUS}; color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
        )
        self.chat_image_preview.setText("预览")
        opt_row.addWidget(self.chat_image_preview)
        layout.addLayout(opt_row)

        self._chat_attach_image_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._chat_attach_image_path: str = ""

        self.history_view = _eai_platform_bootstrap.QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("对话记录将显示在这里…")
        self.history_view.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        self.history_view.setMinimumHeight(160)
        self.history_view.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.history_view.document().setDocumentMargin(8)
        self.history_view.setStyleSheet(
            f"QTextEdit {{ background-color: {_eai_configuration_policies.UI_BG_LOG}; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; "
            f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; border-radius: {_eai_configuration_policies.UI_RADIUS}; "
            f"padding: 6px 8px 12px 8px; }}"
        )
        layout.addWidget(self.history_view, stretch=1)

        input_row = _eai_platform_bootstrap.QHBoxLayout()
        input_row.setSpacing(4)
        self._input_row = input_row
        self.input_edit = _eai_ui_chat_history_dialogs.ChatInputEdit()
        self.input_edit._chat_panel = self
        self.input_edit.setPlaceholderText(_eai_configuration_chat.LAKE_CHAT_INPUT_PLACEHOLDER)
        _eai_ui_widgets_layouts._configure_resizable_text_edit(self.input_edit, default_h=100, tooltip="")
        self.input_edit.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.input_edit.document().setDocumentMargin(4)
        self.input_edit.setStyleSheet(
            f"QTextEdit {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; "
            f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; border-radius: {_eai_configuration_policies.UI_RADIUS}; "
            f"padding: 6px 8px 8px 8px; }}"
        )
        self._refresh_system_lang_btn()
        self._suppress_send_until = 0.0
        self._ime_dirty_from_history = False
        self._ime_rebuilding = False
        self._history_dialog = None
        self.input_resize_bar = None
        input_row.addWidget(self.input_edit, stretch=1)
        send_col = _eai_platform_bootstrap.QVBoxLayout()
        self.send_btn = _eai_platform_bootstrap.QPushButton("发送")
        self.send_btn.setObjectName("primaryAction")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setMinimumWidth(72)
        self.send_btn.setToolTip("调用大模型获取回复")
        self.send_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.send_btn.clicked.connect(self._on_send_clicked)
        send_col.addWidget(self.send_btn)
        send_col.addStretch()
        input_row.addLayout(send_col)
        layout.addLayout(input_row)

        self._append_system_line(
            f"模型: {self._config.model}  |  API: {self._config.api_base}"
        )
        self._append_system_line(
            "对话仅可选已部署且正在运行的推理服务；无服务时请先在「测试」页启动。"
        )
        if not self._config.api_key:
            self._append_system_line(
                f"提示: 请填写 API Key 或 export {_eai_configuration_chat.LLM_API_KEY_ENV}=your-key"
            )
        self._running_services: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]] = []
        self._extra_inference_candidates: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str, str, str]] = []
        self._provider_switch_guard = False
        self.refresh_running_providers(notify=True)

        # 初始会话 tab（在系统提示写入 history 之后再快照）
        self._session_switch_guard = True
        try:
            self.session_tabs.addTab("新对话")
            self.session_tabs.setCurrentIndex(0)
        finally:
            self._session_switch_guard = False
        self._session_index = 0
        self._sessions = [self._snapshot_current_session()]
        self._sync_session_tab_title()
