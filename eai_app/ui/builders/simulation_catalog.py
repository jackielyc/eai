"""ui.builders.simulation catalog — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    build.sim_tab = _eai_platform_bootstrap.QWidget()
    build.sim_outer = _eai_platform_bootstrap.QVBoxLayout(build.sim_tab)
    build.sim_outer.setContentsMargins(8, 6, 8, 6)
    build.sim_outer.setSpacing(6)

    sim_header = _eai_platform_bootstrap.QWidget()
    sim_header.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Maximum)
    build.sim_header_l = _eai_platform_bootstrap.QVBoxLayout(sim_header)
    build.sim_header_l.setContentsMargins(0, 0, 0, 0)
    build.sim_header_l.setSpacing(6)
    build.sim_outer.addWidget(sim_header)

    build.sim_form = _eai_platform_bootstrap.QWidget()
    build.sim_form_l = _eai_platform_bootstrap.QVBoxLayout(build.sim_form)
    build.sim_form_l.setContentsMargins(0, 0, 0, 0)
    build.sim_form_l.setSpacing(6)

    self.sim_hint_label = _eai_platform_bootstrap.QLabel("")
    self.sim_hint_label.setWordWrap(True)
    self.sim_hint_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    build.sim_form_l.addWidget(self.sim_hint_label)

    # 可拖动的后端选择条（顺序写入 eai/config/sim_backends_order.json，重启恢复）
    # 隐藏 radio 仍保留，供 viewer_ui_state / 既有 isChecked 逻辑使用
    sim_backend_row = _eai_platform_bootstrap.QHBoxLayout()
    sim_backend_row.setSpacing(8)
    sim_backend_row.addWidget(_eai_platform_bootstrap.QLabel("仿真后端"))
    self.sim_backend_tab_bar = _eai_platform_bootstrap.QTabBar()
    self.sim_backend_tab_bar.setObjectName("simBackendTabs")
    self.sim_backend_tab_bar.setMovable(True)
    self.sim_backend_tab_bar.setExpanding(False)
    self.sim_backend_tab_bar.setDocumentMode(True)
    self.sim_backend_tab_bar.setUsesScrollButtons(True)
    self.sim_backend_tab_bar.setDrawBase(False)
    self.sim_backend_tab_bar.setElideMode(_eai_platform_bootstrap.Qt.ElideRight)
    self.sim_backend_tab_bar.setToolTip(
        "可拖动调整仿真后端顺序；下次启动会按此顺序打开"
        "（写入 eai/config/sim_backends_order.json）"
    )
    self.sim_backend_isaac_radio = _eai_platform_bootstrap.QRadioButton("Isaac / RoboDojo")
    self.sim_backend_mujoco_radio = _eai_platform_bootstrap.QRadioButton("MuJoCo")
    self.sim_backend_spaces_radio = _eai_platform_bootstrap.QRadioButton("MolmoSpaces")
    self.sim_backend_psi_spaces_radio = _eai_platform_bootstrap.QRadioButton("Psi-MolmoSpaces")
    self.sim_backend_arena_radio = _eai_platform_bootstrap.QRadioButton("IsaacLab-Arena")
    self.sim_backend_libero_radio = _eai_platform_bootstrap.QRadioButton("LIBERO")
    self.sim_backend_robotwin_radio = _eai_platform_bootstrap.QRadioButton("RoboTwin")
    self.sim_backend_extended_radio = _eai_platform_bootstrap.QRadioButton("扩展评测")
    self.sim_backend_isaac_radio.setChecked(True)
    self._sim_backend_group = _eai_platform_bootstrap.QButtonGroup(self)
    self._sim_backend_group.addButton(self.sim_backend_isaac_radio, 0)
    self._sim_backend_group.addButton(self.sim_backend_mujoco_radio, 1)
    self._sim_backend_group.addButton(self.sim_backend_spaces_radio, 2)
    self._sim_backend_group.addButton(self.sim_backend_psi_spaces_radio, 7)
    self._sim_backend_group.addButton(self.sim_backend_arena_radio, 3)
    self._sim_backend_group.addButton(self.sim_backend_libero_radio, 4)
    self._sim_backend_group.addButton(self.sim_backend_robotwin_radio, 5)
    self._sim_backend_group.addButton(self.sim_backend_extended_radio, 6)
    self._sim_backend_radio_by_key = {
        key: getattr(self, attr)
        for key, _label, attr in _eai_configuration_tabs.SIM_BACKEND_SELECTOR_SPECS
    }
    self._sim_backend_tabs_guard = False
    self._sim_backend_tabs_order_guard = False
    for build.radio in self._sim_backend_radio_by_key.values():
        build.radio.setVisible(False)
        build.radio.toggled.connect(self._on_sim_backend_changed)
    self.sim_backend_tab_bar.currentChanged.connect(self._on_sim_backend_tab_changed)
    self.sim_backend_tab_bar.tabMoved.connect(self._on_sim_backend_tab_moved)
    # 填充 tabs 时挡住 currentChanged，避免把默认选中的 Isaac 冲掉
    self._sim_backend_tabs_guard = True
    self._sim_backend_tabs_order_guard = True
    try:
        build.spec_by_key = {
            k: (label, attr) for k, label, attr in _eai_configuration_tabs.SIM_BACKEND_SELECTOR_SPECS
        }
        for build.key in _eai_infrastructure_ui_storage.preferred_sim_backends_order():
            build.pair = build.spec_by_key.get(build.key)
            if build.pair is None:
                continue
            build.label, build._attr = build.pair
            build.idx = self.sim_backend_tab_bar.addTab(build.label)
            self.sim_backend_tab_bar.setTabData(build.idx, build.key)
    finally:
        self._sim_backend_tabs_guard = False
        self._sim_backend_tabs_order_guard = False
    self._sync_sim_backend_tab_from_radios()
    sim_backend_row.addWidget(self.sim_backend_tab_bar, 1)
    build.sim_header_l.addLayout(sim_backend_row)

    ext_backend_row = _eai_platform_bootstrap.QHBoxLayout()
    ext_backend_row.setSpacing(6)
    ext_backend_row.addWidget(_eai_platform_bootstrap.QLabel("扩展套件"))
    self.sim_extended_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.sim_extended_combo.setMinimumWidth(220)
    self.sim_extended_combo.setToolTip(
        "SimplerEnv / CALVIN / RoboCasa / ManiSkill / Genesis / VLA Harness 等扩展评测。\n"
        "需先选中上方「扩展评测」；榜单随套件切换；启动走 tools/run_external_bench.py。"
    )
    for build.label, build.key in _eai_configuration_models.list_extended_choices() or [
        ("SimplerEnv", "simpler"),
        ("CALVIN", "calvin"),
        ("RoboCasa", "robocasa"),
        ("ManiSkill3", "maniskill3"),
        ("VLA Harness (总榜)", "vla_harness"),
    ]:
        self.sim_extended_combo.addItem(build.label, build.key)
    idx_simpler = self.sim_extended_combo.findData("simpler")
    if idx_simpler >= 0:
        self.sim_extended_combo.setCurrentIndex(idx_simpler)
    _eai_configuration_models.stamp_eval_task_combo(self.sim_extended_combo)
    self.sim_extended_combo.currentIndexChanged.connect(
        self._on_sim_extended_combo_changed
    )
    ext_backend_row.addWidget(self.sim_extended_combo, 1)
    self.sim_extended_combo.setEnabled(False)
    build.sim_form_l.addLayout(ext_backend_row)

    # Shared industry leaderboard (switches with 仿真后端)
    self.sim_leaderboard_group = _eai_platform_bootstrap.QGroupBox("业内评测榜单")
    self.sim_leaderboard_group.setCheckable(True)
    self.sim_leaderboard_group.setChecked(True)
    self.sim_leaderboard_group.setStyleSheet(
        f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; margin-top: 8px; padding-top: 10px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}"
    )
    _lb_outer = _eai_platform_bootstrap.QVBoxLayout(self.sim_leaderboard_group)
    _lb_outer.setContentsMargins(8, 8, 8, 6)
    _lb_outer.setSpacing(4)
    _lb_meta = _eai_platform_bootstrap.QHBoxLayout()
    _lb_meta.setSpacing(6)
    self.sim_leaderboard_meta_label = _eai_platform_bootstrap.QLabel("")
    self.sim_leaderboard_meta_label.setWordWrap(True)
    self.sim_leaderboard_meta_label.setStyleSheet(
        f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; font-weight: normal;"
    )
    _lb_meta.addWidget(self.sim_leaderboard_meta_label, 1)
    self.sim_leaderboard_source_btn = _eai_platform_bootstrap.QPushButton("打开来源")
    self.sim_leaderboard_source_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_leaderboard_source_btn.setToolTip("打开当前后端官方 / 公开榜单页面")
    self.sim_leaderboard_source_btn.clicked.connect(
        self._on_sim_leaderboard_source_clicked
    )
    _lb_meta.addWidget(self.sim_leaderboard_source_btn)
    self.sim_leaderboard_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.sim_leaderboard_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_leaderboard_refresh_btn.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
    self.sim_leaderboard_refresh_btn.setToolTip(
        "从公开榜单源实时抓取当前后端结果，更新表格并写回\n"
        "eai/config/sim_eval_leaderboards.json（后台线程，不卡 UI）。\n"
        "已合并的本地评测会保留，并与实时业内条目重新排序对比。"
    )
    self.sim_leaderboard_refresh_btn.clicked.connect(
        self._on_sim_leaderboard_refresh_clicked
    )
    _lb_meta.addWidget(self.sim_leaderboard_refresh_btn)
    self.sim_leaderboard_reload_btn = _eai_platform_bootstrap.QPushButton("本地")
    self.sim_leaderboard_reload_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_leaderboard_reload_btn.setToolTip(
        "仅重新读取本地 eai/config/sim_eval_leaderboards.json（不联网）\n"
        "已合并的本地评测会与业内条目放在同一表对比"
    )
    self.sim_leaderboard_reload_btn.clicked.connect(
        self._on_sim_leaderboard_reload_clicked
    )
    _lb_meta.addWidget(self.sim_leaderboard_reload_btn)
    self.sim_leaderboard_merge_btn = _eai_platform_bootstrap.QPushButton("合并本地")
    self.sim_leaderboard_merge_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_leaderboard_merge_btn.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
    self.sim_leaderboard_merge_btn.setToolTip(
        "把当前后端的本地仿真评测结果合并进榜单：\n"
        "· Isaac：优先用下方选中的 RoboDojo `_result.json`\n"
        "· 其他后端：手动填写方法名与指标\n"
        "刷新实时榜单时会保留本地行，并与业内条目重新排序对比"
    )
    self.sim_leaderboard_merge_btn.clicked.connect(
        self._on_sim_leaderboard_merge_clicked
    )
    _lb_meta.addWidget(self.sim_leaderboard_merge_btn)
    self.sim_leaderboard_remove_local_btn = _eai_platform_bootstrap.QPushButton("移除本地")
    self.sim_leaderboard_remove_local_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.sim_leaderboard_remove_local_btn.setToolTip(
        "从榜单中移除当前选中的本地评测行（不影响业内条目）"
    )
    self.sim_leaderboard_remove_local_btn.clicked.connect(
        self._on_sim_leaderboard_remove_local_clicked
    )
    _lb_meta.addWidget(self.sim_leaderboard_remove_local_btn)
    self._sim_leaderboard_refreshing = False
    _lb_outer.addLayout(_lb_meta)
    self.sim_leaderboard_table = _eai_platform_bootstrap.QTableWidget(0, 7)
    self.sim_leaderboard_table.setHorizontalHeaderLabels(
        ["#", "来源", "方法", "指标", "说明", "机构", "日期"]
    )
    self.sim_leaderboard_table.setEditTriggers(_eai_platform_bootstrap.QAbstractItemView.NoEditTriggers)
    self.sim_leaderboard_table.setSelectionBehavior(_eai_platform_bootstrap.QAbstractItemView.SelectRows)
    self.sim_leaderboard_table.setSelectionMode(_eai_platform_bootstrap.QAbstractItemView.SingleSelection)
    self.sim_leaderboard_table.setAlternatingRowColors(True)
    self.sim_leaderboard_table.verticalHeader().setVisible(False)
    self.sim_leaderboard_table.setShowGrid(True)
    self.sim_leaderboard_table.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.sim_leaderboard_table.setStyleSheet(
        f"QTableWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; gridline-color: #444; "
        "alternate-background-color: #222; }"
        "QHeaderView::section { background-color: #2a2a2a; color: #ccc; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; padding: 3px; font-weight: bold; }}"
    )
    _hdr = self.sim_leaderboard_table.horizontalHeader()
    _hdr.setSectionResizeMode(0, _eai_platform_bootstrap.QHeaderView.ResizeToContents)
    _hdr.setSectionResizeMode(1, _eai_platform_bootstrap.QHeaderView.ResizeToContents)
    _hdr.setSectionResizeMode(2, _eai_platform_bootstrap.QHeaderView.Stretch)
    _hdr.setSectionResizeMode(3, _eai_platform_bootstrap.QHeaderView.ResizeToContents)
    _hdr.setSectionResizeMode(4, _eai_platform_bootstrap.QHeaderView.Stretch)
    _hdr.setSectionResizeMode(5, _eai_platform_bootstrap.QHeaderView.ResizeToContents)
    _hdr.setSectionResizeMode(6, _eai_platform_bootstrap.QHeaderView.ResizeToContents)
    self.sim_leaderboard_table.setMinimumHeight(72)
    self.sim_leaderboard_table.setMaximumHeight(260)
    self.sim_leaderboard_table.setFixedHeight(140)
    self.sim_leaderboard_table.setToolTip(
        "业内实时/快照 + 已合并的本地评测，同一表按指标排序对比。"
        "绿色「本地」行为本机仿真结果；切换后端会切换对应榜单。"
    )
    _lb_outer.addWidget(self.sim_leaderboard_table)
    self.sim_leaderboard_note_label = _eai_platform_bootstrap.QLabel("")
    self.sim_leaderboard_note_label.setWordWrap(True)
    self.sim_leaderboard_note_label.setStyleSheet(
        f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; font-weight: normal;"
    )
    _lb_outer.addWidget(self.sim_leaderboard_note_label)
    self.sim_leaderboard_group.toggled.connect(self._on_sim_leaderboard_toggled)
    self._sim_leaderboard_source_url = ""
    self._sim_leaderboard_row_meta: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]] = []
    build.sim_form_l.addWidget(self.sim_leaderboard_group)

    self.sim_backend_stack = _eai_platform_bootstrap.QStackedWidget()
