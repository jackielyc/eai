"""ui.builders.astra — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def build(self, build):
    astra_tab = _eai_platform_bootstrap.QWidget()
    astra_tab.setObjectName("astraTab")
    astra_tab.setStyleSheet(
        "#astraTab { background-color: #111111; }"
        "#astraTab QLabel { background: transparent; }"
    )
    astra_outer = _eai_platform_bootstrap.QVBoxLayout(astra_tab)
    astra_outer.setContentsMargins(8, 6, 8, 6)
    astra_outer.setSpacing(6)
    astra_hint = _eai_platform_bootstrap.QLabel(
        "嵌入页可切换 English / 中文解读 / Gallery，或手动输入链接后点「前往」。"
        "若页内视频无法播放，再点「浏览器打开」。"
    )
    astra_hint.setWordWrap(True)
    astra_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    astra_outer.addWidget(astra_hint)
    astra_tool_row = _eai_platform_bootstrap.QHBoxLayout()
    astra_tool_row.setSpacing(6)
    astra_tool_row.addWidget(_eai_platform_bootstrap.QLabel("页面"))
    self.astra_page_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    for build.title, url in _eai_configuration_models.ASTRA_PAGE_PRESETS:
        self.astra_page_combo.addItem(build.title, url)
    self.astra_page_combo.setToolTip("切换嵌入页面")
    self.astra_page_combo.currentIndexChanged.connect(
        self._on_astra_page_changed
    )
    astra_tool_row.addWidget(self.astra_page_combo)
    self.astra_url_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_models.ASTRA_ROBODOJO_URL)
    self.astra_url_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.astra_url_edit.setPlaceholderText("https://…")
    self.astra_url_edit.returnPressed.connect(self._on_astra_go_clicked)
    astra_tool_row.addWidget(self.astra_url_edit, 1)
    self.astra_go_btn = _eai_platform_bootstrap.QPushButton("前往")
    self.astra_go_btn.setToolTip("加载地址栏中的链接")
    self.astra_go_btn.clicked.connect(self._on_astra_go_clicked)
    astra_tool_row.addWidget(self.astra_go_btn)
    self.astra_reload_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.astra_reload_btn.setToolTip("重新加载当前页面")
    self.astra_reload_btn.clicked.connect(self._on_astra_reload_clicked)
    astra_tool_row.addWidget(self.astra_reload_btn)
    self.astra_open_btn = _eai_platform_bootstrap.QPushButton("浏览器打开")
    self.astra_open_btn.clicked.connect(self._on_astra_open_clicked)
    astra_tool_row.addWidget(self.astra_open_btn)
    astra_outer.addLayout(astra_tool_row)
    self.astra_web_view = None
    self._astra_web_container = _eai_platform_bootstrap.QWidget()
    self._astra_web_container_layout = _eai_platform_bootstrap.QVBoxLayout(self._astra_web_container)
    self._astra_web_container_layout.setContentsMargins(0, 0, 0, 0)
    if _eai_platform_bootstrap.QWebEngineView is not None:
        # 延迟创建：启动时立刻 setUrl 会在无 GLX/GL 冲突下触发 SIP segfault。
        self._astra_web_placeholder = _eai_platform_bootstrap.QLabel(
            "嵌入页按需加载：切到本页或点「前往」后创建 WebEngine。\n"
            "若启动曾因 OpenGL 崩溃，也可用「浏览器打开」。"
        )
        self._astra_web_placeholder.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._astra_web_placeholder.setWordWrap(True)
        self._astra_web_placeholder.setMinimumHeight(200)
        self._astra_web_placeholder.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._astra_web_container_layout.addWidget(self._astra_web_placeholder, 1)
    else:
        self._astra_web_placeholder = _eai_platform_bootstrap.QLabel(
            "未安装 PyQtWebEngine，无法页内嵌入。\n"
            f"请点「浏览器打开」查看：{_eai_configuration_models.ASTRA_ROBODOJO_URL}"
        )
        self._astra_web_placeholder.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._astra_web_placeholder.setWordWrap(True)
        self._astra_web_placeholder.setMinimumHeight(200)
        self._astra_web_placeholder.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._astra_web_container_layout.addWidget(self._astra_web_placeholder, 1)
        self.astra_reload_btn.setEnabled(False)
        self.astra_go_btn.setEnabled(False)
        self.astra_page_combo.setEnabled(False)
    astra_outer.addWidget(self._astra_web_container, 1)
    build.control_tabs.addTab(astra_tab, "Astra")
    if _eai_platform_bootstrap.QWebEngineView is not None and self._want_control_tab("Astra"):
        _eai_platform_bootstrap.QTimer.singleShot(1500, self._ensure_astra_web_view)

    from physical_rsi.tab import PhysicalRsiTab

    self.physical_rsi_tab = PhysicalRsiTab()
    build.control_tabs.addTab(self.physical_rsi_tab, "PhysicalRSI")

    from embodied_evo.tab import EmbodiedEvoTab

    self.embodied_evo_tab = EmbodiedEvoTab()
    build.control_tabs.addTab(self.embodied_evo_tab, "EmbodiedEvo")

    from rax_panel.tab import RaxTab

    self.rax_tab = RaxTab()
    build.control_tabs.addTab(self.rax_tab, "RAX")

    from aspire.tab import AspireTab

    self.aspire_tab = AspireTab()
    build.control_tabs.addTab(self.aspire_tab, "ASPIRE")

    from flora.tab import FloraTab

    self.flora_tab = FloraTab()
    build.control_tabs.addTab(self.flora_tab, "FLORA")

    from prog_sota.tab import ProgSotaTab

    self.prog_sota_tab = ProgSotaTab()
    build.control_tabs.addTab(self.prog_sota_tab, "编程冲SOTA")

    from system_monitor.tab import SystemMonitorTab

    self.system_monitor_tab = SystemMonitorTab()
    build.control_tabs.addTab(self.system_monitor_tab, "主机资源")
