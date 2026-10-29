"""ui.builders.foundation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def build(self, build):
    build.prefix = self._ui_prefix
    build.llm_config = self._llm_config
    build.node = self.node
    build.bridge = self.bridge
    build.control_tabs = self.control_tabs
    build.root_layout = self._root_layout
    _wx, _wy, build.win_w, _wh = _eai_perception_image_geometry.default_viewer_geometry()
    while build.control_tabs.count() > 0:
        build.control_tabs.removeTab(0)
    _eai_platform_bootstrap._boot_tick("构建界面…")
