"""ui.builders.robot workspace — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    arm_bottom = _eai_platform_bootstrap.QWidget()
    arm_bottom_l = _eai_platform_bootstrap.QVBoxLayout(arm_bottom)
    arm_bottom_l.setContentsMargins(4, 4, 4, 4)
    arm_bottom_l.setSpacing(4)
    arm_bottom_l.addLayout(build.arm_panel_row)
    arm_bottom_l.addWidget(self._arm_panel_stack, 1)
    arm_bottom_l.setSizeConstraint(_eai_platform_bootstrap.QLayout.SetMinimumSize)

    # Independent, scrollable pages: never squeeze all controls into a
    # fixed-height half of a splitter. Horizontal overflow remains reachable.
    self.arm_workspace_tabs = _eai_platform_bootstrap.QTabWidget()
    self.arm_workspace_tabs.setObjectName("armWorkspaceTabs")
    self.arm_workspace_tabs.setUsesScrollButtons(True)
    manual_page = _eai_platform_bootstrap.QWidget()
    manual_page_l = _eai_platform_bootstrap.QVBoxLayout(manual_page)
    manual_page_l.setContentsMargins(0, 0, 0, 0)
    manual_page_l.setSpacing(4)
    manual_page_l.addWidget(build.manual_group)
    manual_page_l.addWidget(arm_bottom, 1)
    self.arm_workspace_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(manual_page), "手动控制")
    self.grasp_execution_panel.layout().setSizeConstraint(_eai_platform_bootstrap.QLayout.SetMinimumSize)
    self.arm_workspace_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(self.grasp_execution_panel), "分阶段抓取")
    from eai_app.ui.calibration.cameras import CameraExtrinsicCalibPanel

    self.camera_calib_panel = CameraExtrinsicCalibPanel(
        self.node,
        get_click=self._calib_last_click,
        get_tcp=self._calib_tcp,
        parent=self,
    )
    self.arm_workspace_tabs.addTab(_eai_ui_widgets_layouts._wrap_in_vscroll(self.camera_calib_panel), "手眼标定")
    from eai_app.ui.grasp.calibration import GraspProfileCalibrationPanel

    self.grasp_profile_calib_panel = GraspProfileCalibrationPanel(
        profile_path=self.grasp_execution_panel.profile_path.text(),
        refresh_profile=self._on_grasp_profile_calibrated,
        parent=self,
    )
    self.arm_workspace_tabs.addTab(
        _eai_ui_widgets_layouts._wrap_in_vscroll(self.grasp_profile_calib_panel), "抓取标定"
    )
    self._robot_view_holder = _eai_platform_bootstrap.QWidget()
    robot_hold_layout = _eai_platform_bootstrap.QVBoxLayout(self._robot_view_holder)
    robot_hold_layout.setContentsMargins(8, 8, 8, 8)
    self._robot_view = None
    self.arm_workspace_tabs.addTab(self._robot_view_holder, "机器人")
    self.arm_workspace_tabs.currentChanged.connect(self._on_arm_workspace_tab_changed)
    build.control_layout.addWidget(self.arm_workspace_tabs, 1)
    self._apply_arm_panel_stack()
    build.control_tabs.addTab(build.control_tab, "手臂/手")
    _eai_platform_bootstrap._boot_tick("构建界面：仿真评测…")
