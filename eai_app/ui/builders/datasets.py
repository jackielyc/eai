"""ui.builders.datasets — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')
_eai_ui_widgets_tab_bar = module_ref('eai_app.ui.widgets.tab_bar')

def build(self, build):
    dataset_tab = _eai_platform_bootstrap.QWidget()
    dataset_tab.setObjectName("datasetTab")
    dataset_outer = _eai_platform_bootstrap.QVBoxLayout(dataset_tab)
    dataset_outer.setContentsMargins(4, 4, 4, 4)
    dataset_outer.setSpacing(4)
    self.dataset_inner_tabs = _eai_platform_bootstrap.QTabWidget()
    self.dataset_inner_tabs.setObjectName("datasetInnerTabs")

    # -- 开源梳理 --
    survey_page = _eai_platform_bootstrap.QWidget()
    survey_layout = _eai_platform_bootstrap.QVBoxLayout(survey_page)
    survey_layout.setContentsMargins(8, 6, 8, 6)
    survey_layout.setSpacing(6)
    survey_hint = _eai_platform_bootstrap.QLabel(
        "开源具身数据集梳理：选型结论、按本体下载/Mix、体积约数。"
        "内容来自 docs/embodied-open-datasets-notes.md，链接可点击打开。"
    )
    survey_hint.setWordWrap(True)
    survey_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    survey_layout.addWidget(survey_hint)
    survey_btn_row = _eai_platform_bootstrap.QHBoxLayout()
    survey_btn_row.setSpacing(6)
    self.dataset_survey_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新文档")
    self.dataset_survey_refresh_btn.setToolTip("重新加载分析 Markdown")
    self.dataset_survey_refresh_btn.clicked.connect(
        self._refresh_dataset_open_survey
    )
    survey_btn_row.addWidget(self.dataset_survey_refresh_btn)
    self.dataset_survey_open_btn = _eai_platform_bootstrap.QPushButton("打开 Markdown")
    self.dataset_survey_open_btn.setToolTip("用系统默认应用打开源文档")
    self.dataset_survey_open_btn.clicked.connect(
        self._on_dataset_open_survey_file
    )
    survey_btn_row.addWidget(self.dataset_survey_open_btn)
    self.dataset_survey_path_label = _eai_platform_bootstrap.QLabel("")
    self.dataset_survey_path_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    self.dataset_survey_path_label.setTextInteractionFlags(
        _eai_platform_bootstrap.Qt.TextSelectableByMouse
    )
    survey_btn_row.addWidget(self.dataset_survey_path_label, 1)
    survey_layout.addLayout(survey_btn_row)
    self.dataset_survey_view = _eai_platform_bootstrap.QTextBrowser()
    self.dataset_survey_view.setOpenExternalLinks(True)
    self.dataset_survey_view.setStyleSheet(
        f"QTextBrowser {{ background-color: {_eai_configuration_policies.UI_BG_LOG}; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    survey_layout.addWidget(self.dataset_survey_view, 1)
    self.dataset_inner_tabs.addTab(survey_page, "开源梳理")

    # -- 本地浏览 --
    browser_page = _eai_platform_bootstrap.QWidget()
    browser_layout = _eai_platform_bootstrap.QVBoxLayout(browser_page)
    browser_layout.setContentsMargins(8, 6, 8, 6)
    browser_layout.setSpacing(6)
    ds_hint = _eai_platform_bootstrap.QLabel(
        "浏览本地数据集目录（.rrd / 图像 / Bagel 历史 JSON 等）。"
        "可选中图像送入 sub image，或把 .rrd 设为回放文件。"
    )
    ds_hint.setWordWrap(True)
    ds_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    browser_layout.addWidget(ds_hint)

    ds_root_row = _eai_platform_bootstrap.QHBoxLayout()
    ds_root_row.setSpacing(6)
    ds_root_row.addWidget(_eai_platform_bootstrap.QLabel("预设"))
    self.dataset_preset_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.label, path in _eai_adapters_robot_launch_utils.dataset_browser_presets():
        self.dataset_preset_combo.addItem(build.label, path)
    self.dataset_preset_combo.setToolTip("切换常用数据集根目录")
    self.dataset_preset_combo.currentIndexChanged.connect(
        self._on_dataset_preset_changed
    )
    ds_root_row.addWidget(self.dataset_preset_combo)
    ds_root_row.addWidget(_eai_platform_bootstrap.QLabel("根目录"))
    _ds_default = _eai_configuration_chat.EAI_DATASETS_DIR if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_chat.EAI_DATASETS_DIR) else (
        _eai_adapters_robot_launch_utils.dataset_browser_presets()[0][1] if _eai_adapters_robot_launch_utils.dataset_browser_presets() else _eai_configuration_models.EAI_DIR
    )
    self.dataset_root_edit = _eai_platform_bootstrap.QLineEdit(_ds_default)
    self.dataset_root_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.dataset_root_edit.setToolTip("当前浏览的根目录（可手动改路径后点刷新）")
    ds_root_row.addWidget(self.dataset_root_edit, 1)
    self.dataset_root_browse_btn = _eai_platform_bootstrap.QPushButton("…")
    self.dataset_root_browse_btn.setFixedWidth(28)
    self.dataset_root_browse_btn.clicked.connect(self._on_dataset_root_browse)
    ds_root_row.addWidget(self.dataset_root_browse_btn)
    self.dataset_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.dataset_refresh_btn.clicked.connect(self._on_dataset_refresh_clicked)
    ds_root_row.addWidget(self.dataset_refresh_btn)
    self.dataset_up_btn = _eai_platform_bootstrap.QPushButton("上级")
    self.dataset_up_btn.clicked.connect(self._on_dataset_up_clicked)
    ds_root_row.addWidget(self.dataset_up_btn)
    browser_layout.addLayout(ds_root_row)

    ds_filter_row = _eai_platform_bootstrap.QHBoxLayout()
    ds_filter_row.setSpacing(6)
    ds_filter_row.addWidget(_eai_platform_bootstrap.QLabel("过滤"))
    self.dataset_filter_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit("")
    self.dataset_filter_edit.setPlaceholderText("按文件名过滤…")
    self.dataset_filter_edit.textChanged.connect(self._on_dataset_filter_changed)
    ds_filter_row.addWidget(self.dataset_filter_edit, 1)
    self.dataset_type_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.dataset_type_combo.addItem("全部", "all")
    self.dataset_type_combo.addItem("目录", "dir")
    self.dataset_type_combo.addItem("图像", "image")
    self.dataset_type_combo.addItem("视频", "video")
    self.dataset_type_combo.addItem("RRD", "rrd")
    self.dataset_type_combo.addItem("JSON", "json")
    self.dataset_type_combo.currentIndexChanged.connect(
        self._on_dataset_filter_changed
    )
    ds_filter_row.addWidget(self.dataset_type_combo)
    browser_layout.addLayout(ds_filter_row)

    ds_split = _eai_platform_bootstrap.QSplitter(_eai_platform_bootstrap.Qt.Horizontal)
    self.dataset_list = _eai_platform_bootstrap.QListWidget()
    self.dataset_list.setStyleSheet(
        f"QListWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.dataset_list.setMinimumWidth(220)
    self.dataset_list.itemSelectionChanged.connect(
        self._on_dataset_selection_changed
    )
    self.dataset_list.itemDoubleClicked.connect(self._on_dataset_item_activated)
    ds_split.addWidget(self.dataset_list)

    ds_right = _eai_platform_bootstrap.QWidget()
    ds_right_layout = _eai_platform_bootstrap.QVBoxLayout(ds_right)
    ds_right_layout.setContentsMargins(0, 0, 0, 0)
    ds_right_layout.setSpacing(6)
    self.dataset_preview = _eai_platform_bootstrap.QLabel("选择左侧条目预览")
    self.dataset_preview.setMinimumHeight(180)
    self.dataset_preview.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.dataset_preview.setStyleSheet(
        f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; color: #888; }}"
    )
    self.dataset_preview.setSizePolicy(
        _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
    )
    ds_right_layout.addWidget(self.dataset_preview, 2)
    self.dataset_detail_edit = _eai_platform_bootstrap.QTextEdit()
    self.dataset_detail_edit.setReadOnly(True)
    self.dataset_detail_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.dataset_detail_edit.setPlaceholderText("条目详情…")
    self.dataset_detail_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.dataset_detail_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        ds_right_layout,
        self.dataset_detail_edit,
        default_h=160,
        min_h=80,
        max_h=600,
        label="拖动调整详情高度",
    )
    ds_action_row = _eai_platform_bootstrap.QHBoxLayout()
    ds_action_row.setSpacing(6)
    self.dataset_open_dir_btn = _eai_platform_bootstrap.QPushButton("打开目录")
    self.dataset_open_dir_btn.clicked.connect(self._on_dataset_open_dir)
    ds_action_row.addWidget(self.dataset_open_dir_btn)
    self.dataset_copy_path_btn = _eai_platform_bootstrap.QPushButton("复制路径")
    self.dataset_copy_path_btn.clicked.connect(self._on_dataset_copy_path)
    ds_action_row.addWidget(self.dataset_copy_path_btn)
    self.dataset_to_replay_btn = _eai_platform_bootstrap.QPushButton("设为回放")
    self.dataset_to_replay_btn.setToolTip("将选中的 .rrd 设为回放页路径")
    self.dataset_to_replay_btn.clicked.connect(self._on_dataset_to_replay)
    ds_action_row.addWidget(self.dataset_to_replay_btn)
    self.dataset_to_bagel_btn = _eai_platform_bootstrap.QPushButton("送入 Bagel")
    self.dataset_to_bagel_btn.setToolTip("将选中图像设为 sub image 输入图")
    self.dataset_to_bagel_btn.clicked.connect(self._on_dataset_to_bagel)
    ds_action_row.addWidget(self.dataset_to_bagel_btn)
    ds_action_row.addStretch(1)
    ds_right_layout.addLayout(ds_action_row)
    ds_split.addWidget(ds_right)
    ds_split.setStretchFactor(0, 1)
    ds_split.setStretchFactor(1, 2)
    browser_layout.addWidget(ds_split, 1)
    self.dataset_inner_tabs.addTab(browser_page, "本地浏览")

    dataset_outer.addWidget(self.dataset_inner_tabs, 1)
    self._dataset_cwd = ""
    build.control_tabs.addTab(dataset_tab, "数据集")
    # 默认可看到开源梳理；同步预设并刷新本地列表
    self.dataset_inner_tabs.setCurrentIndex(0)
    self._refresh_dataset_open_survey()
    for i in range(self.dataset_preset_combo.count()):
        if _eai_platform_bootstrap.os.path.abspath(str(self.dataset_preset_combo.itemData(i) or "")) == _eai_platform_bootstrap.os.path.abspath(
            self.dataset_root_edit.text().strip() or ""
        ):
            self.dataset_preset_combo.setCurrentIndex(i)
            break
    self._refresh_dataset_browser()

    build.control_tabs.addTab(build.test_tab, "sub task")
    build.control_tabs.addTab(build.bagel_tab, "sub image")

    misc_page = _eai_platform_bootstrap.QWidget()
    misc_layout = _eai_platform_bootstrap.QVBoxLayout(misc_page)
    misc_layout.setContentsMargins(6, 6, 6, 6)
    misc_layout.setSpacing(6)
    self._misc_empty_hint = _eai_platform_bootstrap.QLabel(
        f"将任意顶栏标签拖到「{_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE}」上（或右键 → 移入），"
        "即可收入为本页子标签；子标签可拖回顶栏。"
    )
    self._misc_empty_hint.setWordWrap(True)
    self._misc_empty_hint.setAlignment(_eai_platform_bootstrap.Qt.AlignLeft | _eai_platform_bootstrap.Qt.AlignTop)
    misc_layout.addWidget(self._misc_empty_hint)
    self.misc_inner_tabs = _eai_platform_bootstrap.QTabWidget()
    self.misc_inner_tabs.setObjectName("miscInnerTabs")
    self.misc_inner_tabs.setDocumentMode(True)
    self.misc_inner_tabs.setMovable(False)
    self.misc_inner_tabs.setUsesScrollButtons(True)
    self._misc_tab_bar = _eai_ui_widgets_tab_bar.NestableControlTabBar("misc", self.misc_inner_tabs)
    self.misc_inner_tabs.setTabBar(self._misc_tab_bar)
    misc_layout.addWidget(self.misc_inner_tabs, 1)
    build.control_tabs.addTab(misc_page, _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
    self._misc_page = misc_page
    self._update_misc_empty_hint()

    self._refresh_real_eval_status()
    self._update_ctx_ui()

    self._hand_skeleton_detector = None
    self._skeleton_local_cap = None
    self._skeleton_tracking = False
    self._skeleton_busy = False
    self._skeleton_timer = _eai_platform_bootstrap.QTimer(self)
    self._skeleton_timer.setInterval(66)  # ~15 Hz
    self._skeleton_timer.timeout.connect(self._on_skeleton_tick)
    self._refresh_skeleton_camera_list()

    self._human_demo_path = ""
    self._human_demo_traj = None
    self._human_demo_analyzing = False
    self._human_demo_playing = False
    self._human_demo_play_idx = 0
    self._human_demo_origins: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[float, float, float]] = {}
    self._human_demo_world_origins: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[float, float, float]] = {}
    self._human_demo_play_mode: str = "wrist"  # wrist | aligned
    self._human_demo_aligned_waypoints: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Any] = []
    self._human_demo_task_match: _eai_platform_bootstrap.Any = None
    # 复现过程中各臂最近一次目标位姿（避免无手帧把目标打回起点）
    self._human_demo_last_pose: _eai_platform_bootstrap.Dict[
        str, _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
    ] = {}
    self._human_demo_tcp_base: _eai_platform_bootstrap.Dict[            str, _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
    ] = {}
    self._human_demo_video_cap = None
    self._human_demo_bridge = _eai_adapters_models_qwen_remote.HumanDemoAnalyzeBridge()
    self._human_demo_bridge.progress.connect(self._on_human_demo_analyze_progress)
    self._human_demo_bridge.finished.connect(self._on_human_demo_analyze_finished)
    self._human_demo_timer = _eai_platform_bootstrap.QTimer(self)
    self._human_demo_timer.setInterval(80)
    self._human_demo_timer.timeout.connect(self._on_human_demo_play_tick)

    self._left_arm_move_btn_idle_style = ""
    self._left_arm_move_btn_cancel_style = f"color: {_eai_configuration_policies.UI_ACCENT_RED};"
    self._right_arm_move_btn_idle_style = ""
    self._right_arm_move_btn_cancel_style = f"color: {_eai_configuration_policies.UI_ACCENT_RED};"

    # control_tabs 已在空壳阶段加入 root_layout

    build.bridge.left_hand_preset_changed.connect(self._update_left_hand_toggle_ui)
    build.bridge.right_hand_preset_changed.connect(self._update_right_hand_toggle_ui)
    build.bridge.slow_motion_progress.connect(self._on_slow_motion_progress)
    build.bridge.slow_motion_finished.connect(self._on_slow_motion_finished)
    build.bridge.robot_state_updated.connect(
        self._schedule_robot_ui_refresh, _eai_platform_bootstrap.Qt.QueuedConnection
    )
    build.bridge.arm_enable_changed.connect(self._on_arm_enable_ui_changed)
    build.bridge.hand_enable_changed.connect(self._on_hand_enable_ui_changed)
    build.bridge.control_mode_changed.connect(self._on_control_mode_ui_changed)
    self._update_left_hand_toggle_ui(self.node.is_left_hand_at_a())
    self._update_right_hand_toggle_ui(self.node.is_right_hand_at_a())
    self._update_enable_status_ui()
    self._on_arm_move_speed_changed(self.arm_move_speed_slider.value())
    self._update_arm_move_btns_ui(force=True)
