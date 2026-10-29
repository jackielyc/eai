"""ui.window.preferences — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class PreferencesMixin:
    def _on_control_tab_changed_local_cam(self, *_args) -> None:
        """保留空实现：切 tab 不再自动打开本机摄像头。"""
        return

    def _control_tab_titles(self) -> _eai_platform_bootstrap.List[str]:
        tabs = getattr(self, "control_tabs", None)
        if tabs is None:
            return []
        return [tabs.tabText(i) for i in range(tabs.count())]

    def _control_tab_title_is_current(self, title: str) -> bool:
        tabs = getattr(self, "control_tabs", None)
        if tabs is None:
            return False
        cur = tabs.currentIndex()
        if cur < 0:
            return False
        cur_title = str(tabs.tabText(cur) or "").strip()
        if cur_title == title:
            return True
        if cur_title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            inner = getattr(self, "misc_inner_tabs", None)
            if inner is not None and inner.currentIndex() >= 0:
                return str(inner.tabText(inner.currentIndex()) or "").strip() == title
        return False

    def _on_control_tab_bar_clicked(self, *_args) -> None:
        # 在 currentChanged 之前就暂停，避免点击瞬间仍在刷图
        self._pause_ui_frames_for_interaction(2.0)

    def _on_control_tabs_current_changed(self, *_args) -> None:
        """切 tab 时暂停图像刷屏，避免「无响应」。"""
        self._pause_ui_frames_for_interaction(2.0)
        grid = getattr(self, "grid_host", None) or getattr(self, "grid_widget", None)
        if grid is not None:
            try:
                grid.setUpdatesEnabled(False)
                _eai_platform_bootstrap.QTimer.singleShot(200, lambda g=grid: g.setUpdatesEnabled(True))
            except Exception:
                pass

    def _viewer_ui_state_owners(self) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, object]]:
        """主窗口 + 对话面板 + 自定义 Tab 面板。"""
        owners: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, object]] = [("", self)]
        chat = getattr(self, "chat_panel", None)
        if chat is not None:
            owners.append(("chat_panel.", chat))
        for attr, prefix in (
            ("physical_rsi_tab", "tab:PhysicalRSI."),
            ("embodied_evo_tab", "tab:EmbodiedEvo."),
            ("rax_tab", "tab:RAX."),
            ("aspire_tab", "tab:ASPIRE."),
            ("flora_tab", "tab:FLORA."),
            ("prog_sota_tab", "tab:编程冲SOTA."),
        ):
            panel = getattr(self, attr, None)
            if panel is not None:
                owners.append((prefix, panel))
        return owners

    def _on_control_tab_moved(self, *_args) -> None:
        """顶栏或「其他」子栏拖拽排序后写盘。"""
        self._persist_control_tabs_layout()

    def _workspace_preview_topics(self) -> _eai_platform_bootstrap.List[str]:
        return [_eai_configuration_policies.LIBERO_PREVIEW_TOPIC, _eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC]

