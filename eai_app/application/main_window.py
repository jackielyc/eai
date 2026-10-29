"""application.main window — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_ros_bridges = module_ref('eai_app.adapters.ros.bridges')
_eai_adapters_ros_camera_node = module_ref('eai_app.adapters.ros.camera_node')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_tab_bar = module_ref('eai_app.ui.widgets.tab_bar')

from eai_app.ui.window.shell import ShellMixin
from eai_app.ui.window.shell_inject_bagel_image import ShellInjectBagelImageMixin
from eai_app.ui.window.shell_start_sim_eval import ShellStartSimEvalMixin
from eai_app.ui.window.shell_build_a2d_ops_mode_page import ShellBuildA2DOpsModePageMixin
from eai_app.ui.window.shell_open_local_cam_preview_source import ShellOpenLocalCamPreviewSourceMixin
from eai_app.ui.window.shell_try_switch_chat_to_local_qwen import ShellTrySwitchChatToLocalQwenMixin
from eai_app.ui.window.shell_select_control_tab_by_title import ShellSelectControlTabByTitleMixin
from eai_app.ui.window.shell_select_preview_topic_combo import ShellSelectPreviewTopicComboMixin
from eai_app.ui.window.shell_update_mujoco_ui import ShellUpdateMujocoUiMixin
from eai_app.ui.window.shell_resolve_molmospaces_root import ShellResolveMolmospacesRootMixin
from eai_app.ui.window.shell_warmup_pi_before_rynnvalue_hud import ShellWarmupPiBeforeRynnvalueHudMixin
from eai_app.ui.window.shell_set_segment_preview import ShellSetSegmentPreviewMixin
from eai_app.ui.window.shell_force_check_essential_cameras import ShellForceCheckEssentialCamerasMixin
from eai_app.ui.window.shell_disabled_move_btn_tooltip import ShellDisabledMoveBtnTooltipMixin
from eai_app.ui.window.shell_start_mv6d_estimate import ShellStartMv6DEstimateMixin
from eai_app.ui.window.shell_collect_method_poses import ShellCollectMethodPosesMixin
from eai_app.ui.window.shell_slow_motion_finished import ShellSlowMotionFinishedMixin
from eai_app.ui.window.models_training import ModelsTrainingMixin
from eai_app.ui.window.models_bagel import ModelsBagelMixin
from eai_app.ui.window.datasets import DatasetsMixin
from eai_app.ui.window.simulation import SimulationMixin
from eai_app.ui.window.simulation_sim_eval_selected_result_json import SimulationSimEvalSelectedResultJsonMixin
from eai_app.ui.window.simulation_mujoco_start_clicked import SimulationMujocoStartClickedMixin
from eai_app.ui.window.cameras import CamerasMixin
from eai_app.ui.window.preferences import PreferencesMixin
from eai_app.ui.window.models_chat import ModelsChatMixin
from eai_app.ui.window.robot_control import RobotControlMixin
from eai_app.ui.window.demonstrations import DemonstrationsMixin
from eai_app.ui.window.models_perception import ModelsPerceptionMixin
from eai_app.ui.window.models_perception_fp_invoke_clicked import ModelsPerceptionFpInvokeClickedMixin
from eai_app.ui.window.simulation_arena import SimulationArenaMixin
from eai_app.ui.window.simulation_libero import SimulationLiberoMixin
from eai_app.ui.window.simulation_robotwin import SimulationRobotwinMixin
from eai_app.ui.window.simulation_molmospaces import SimulationMolmospacesMixin
from eai_app.ui.window.multiview import MultiviewMixin
from eai_app.ui.window.multiview_mv6d_try_wrist_view_if_visible import MultiviewMv6DTryWristViewIfVisibleMixin
from eai_app.ui.window.multiview_mv6d_estimate_finished import MultiviewMv6DEstimateFinishedMixin

class CameraTopicWindow(ShellMixin, ShellInjectBagelImageMixin, ShellStartSimEvalMixin, ShellBuildA2DOpsModePageMixin, ShellOpenLocalCamPreviewSourceMixin, ShellTrySwitchChatToLocalQwenMixin, ShellSelectControlTabByTitleMixin, ShellSelectPreviewTopicComboMixin, ShellUpdateMujocoUiMixin, ShellResolveMolmospacesRootMixin, ShellWarmupPiBeforeRynnvalueHudMixin, ShellSetSegmentPreviewMixin, ShellForceCheckEssentialCamerasMixin, ShellDisabledMoveBtnTooltipMixin, ShellStartMv6DEstimateMixin, ShellCollectMethodPosesMixin, ShellSlowMotionFinishedMixin, ModelsTrainingMixin, ModelsBagelMixin, DatasetsMixin, SimulationMixin, SimulationSimEvalSelectedResultJsonMixin, SimulationMujocoStartClickedMixin, CamerasMixin, PreferencesMixin, ModelsChatMixin, RobotControlMixin, DemonstrationsMixin, ModelsPerceptionMixin, ModelsPerceptionFpInvokeClickedMixin, SimulationArenaMixin, SimulationLiberoMixin, SimulationRobotwinMixin, SimulationMolmospacesMixin, MultiviewMixin, MultiviewMv6DTryWristViewIfVisibleMixin, MultiviewMv6DEstimateFinishedMixin, _eai_platform_bootstrap.QMainWindow):
    def __init__(
        self,
        node: _eai_adapters_ros_camera_node.CameraTopicNode,
        bridge: _eai_adapters_ros_bridges.RosBridge,
        prefix: str,
        llm_config: _eai_platform_bootstrap.Optional[_eai_adapters_models_chat_client.LlmChatConfig] = None,
        only_tabs: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.node = node
        self.bridge = bridge
        self._only_tabs = list(only_tabs) if only_tabs else []
        self.panels: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QWidget] = {}
        self.topic_checks: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.QCheckBox] = {}
        self._topic_types: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]] = {}
        self._frame_cache: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.np.ndarray] = {}
        self._last_segment_target: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.SegmentPoseTarget] = None
        self._grasp_wrench_lock = _eai_platform_bootstrap.threading.Lock()
        self._grasp_wrench_cache: _eai_platform_bootstrap.Dict[str, dict] = {}
        self._grasp_wrench_subs = []
        self._grasp_wrench_monitor_started = False
        self._click_pose_busy = False
        self._click_pose_bridge = _eai_perception_segmentation.Sam3CallBridge()
        self._click_pose_bridge.finished.connect(self._on_click_pose_finished)
        self._last_sam3_topic = ""
        self._sam3_call_busy = False
        self._sam3_call_bridge = _eai_perception_segmentation.Sam3CallBridge()
        self._sam3_call_bridge.finished.connect(self._on_sam3_call_finished)
        self._fp_call_busy = False
        self._fp_call_bridge = _eai_perception_segmentation.FpCallBridge()
        self._fp_call_bridge.finished.connect(self._on_fp_call_finished)
        self._robot_ui_last_update = 0.0
        self._pending_arm_move_goal: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = None
        self._text_cmd_queue: _eai_platform_bootstrap.List[_eai_platform_bootstrap.ArmHandTextAction] = []
        self._text_cmd_pumping = False
        self._arm_enable_wait_deadline = 0.0
        self._arm_enable_wait_timer = _eai_platform_bootstrap.QTimer(self)
        self._arm_enable_wait_timer.setInterval(200)
        self._arm_enable_wait_timer.timeout.connect(self._on_arm_enable_wait_tick)
        self._sim_arm_move: _eai_platform_bootstrap.Optional[dict] = None
        self._sim_arm_move_timer = _eai_platform_bootstrap.QTimer(self)
        self._sim_arm_move_timer.setInterval(_eai_configuration_policies.SIM_ARM_MOVE_TICK_MS)
        self._sim_arm_move_timer.timeout.connect(self._on_sim_arm_move_tick)
        # 首次读到的左右臂 EE 和当时的 16 关节，供「归位」和重复执行回到初始位姿
        self._arm_home_pose: _eai_platform_bootstrap.Dict[
            str, _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
        ] = {}
        self._arm_home_q16: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._repeat_active = False
        self._repeat_queue: _eai_platform_bootstrap.List[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = []
        self._repeat_out_goals: _eai_platform_bootstrap.List[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = []
        self._repeat_home_goals: _eai_platform_bootstrap.List[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = []
        self._repeat_home_q16: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._repeat_commanded_q16: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._repeat_cycle = 0
        self._repeat_busy_retries = 0
        self._repeat_skip_retries = 0
        self._repeat_last_goal: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = None
        self._repeat_advance_pending = False
        self._repeat_wait_timer = _eai_platform_bootstrap.QTimer(self)
        self._repeat_wait_timer.setSingleShot(True)
        self._repeat_wait_timer.timeout.connect(self._on_repeat_wait_timeout)
        self._psi_policy_dir_override: _eai_platform_bootstrap.Optional[str] = None
        self._cad_capture_dir = ""
        self._cad_capture_saved = 0
        self._cad_last_capture_gray: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None

        self.setWindowTitle("EAI · Camera Topic Viewer")
        win_x, win_y, win_w, win_h = _eai_perception_image_geometry.default_viewer_geometry()
        self.setMinimumSize(min(1280, win_w), min(820, win_h))
        self.setGeometry(win_x, win_y, win_w, win_h)

        central = _eai_platform_bootstrap.QWidget()
        central.setObjectName("viewerCentral")
        self.setCentralWidget(central)
        root_layout = _eai_platform_bootstrap.QVBoxLayout(central)
        root_layout.setContentsMargins(6, 4, 6, 4)
        root_layout.setSpacing(4)

        control_tabs = _eai_platform_bootstrap.QTabWidget()
        self.control_tabs = control_tabs
        control_tabs.setObjectName("controlTabs")
        control_tabs.setDocumentMode(True)
        control_tabs.setTabPosition(_eai_platform_bootstrap.QTabWidget.North)
        control_tabs.setMovable(False)
        control_tabs.setUsesScrollButtons(True)
        self._control_tab_bar = _eai_ui_widgets_tab_bar.NestableControlTabBar("outer", control_tabs)
        control_tabs.setTabBar(self._control_tab_bar)
        # 各 tab 悬停 tip 见 _refresh_control_tab_tooltips（含 --tab 启动命令）

        self._ui_prefix = prefix
        self._llm_config = llm_config
        self._heavy_ui_built = False
        self._root_layout = root_layout
        # 空壳：先让主窗口能 show，再显式 build_heavy_ui
        self._boot_placeholder = _eai_platform_bootstrap.QLabel("正在构建控制区…")
        self._boot_placeholder.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._boot_placeholder.setMinimumHeight(120)
        control_tabs.addTab(self._boot_placeholder, "…")
        root_layout.addWidget(control_tabs)
        self.status_bar = _eai_platform_bootstrap.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("正在构建界面…")
    def build_heavy_ui(self) -> None:
        """构建全部控制区 Tab + 工作区（在 window.show() 之后调用，先看见主窗口）。"""
        if getattr(self, "_heavy_ui_built", False):
            return
        self._heavy_ui_built = True
        from eai_app.ui.builders.context import BuildContext
        build = BuildContext()
        from eai_app.ui.builders.foundation import build as build_foundation
        build_foundation(self, build)
        from eai_app.ui.builders.camera import build as build_camera
        build_camera(self, build)
        from eai_app.ui.builders.robot_services import build as build_robot_services
        build_robot_services(self, build)
        from eai_app.ui.builders.segmentation import build as build_segmentation
        build_segmentation(self, build)
        from eai_app.ui.builders.vision_models import build as build_vision_models
        build_vision_models(self, build)
        from eai_app.ui.builders.depth_models import build as build_depth_models
        build_depth_models(self, build)
        from eai_app.ui.builders.mapping import build as build_mapping
        build_mapping(self, build)
        from eai_app.ui.builders.video_models import build as build_video_models
        build_video_models(self, build)
        from eai_app.ui.builders.world_models import build as build_world_models
        build_world_models(self, build)
        from eai_app.ui.builders.bagel import build as build_bagel
        build_bagel(self, build)
        from eai_app.ui.builders.cad import build as build_cad
        build_cad(self, build)
        from eai_app.ui.builders.training import build as build_training
        build_training(self, build)
        from eai_app.ui.builders.simulation_catalog import build as build_simulation_catalog
        build_simulation_catalog(self, build)
        from eai_app.ui.builders.simulation_isaac import build as build_simulation_isaac
        build_simulation_isaac(self, build)
        from eai_app.ui.builders.simulation_mujoco import build as build_simulation_mujoco
        build_simulation_mujoco(self, build)
        from eai_app.ui.builders.simulation_molmospaces import build as build_simulation_molmospaces
        build_simulation_molmospaces(self, build)
        from eai_app.ui.builders.simulation_arena import build as build_simulation_arena
        build_simulation_arena(self, build)
        from eai_app.ui.builders.simulation_libero import build as build_simulation_libero
        build_simulation_libero(self, build)
        from eai_app.ui.builders.simulation_robotwin import build as build_simulation_robotwin
        build_simulation_robotwin(self, build)
        from eai_app.ui.builders.simulation_external import build as build_simulation_external
        build_simulation_external(self, build)
        from eai_app.ui.builders.simulation_monitoring import build as build_simulation_monitoring
        build_simulation_monitoring(self, build)
        from eai_app.ui.builders.real_evaluation import build as build_real_evaluation
        build_real_evaluation(self, build)
        from eai_app.ui.builders.local_cameras import build as build_local_cameras
        build_local_cameras(self, build)
        from eai_app.ui.builders.manual_control import build as build_manual_control
        build_manual_control(self, build)
        from eai_app.ui.builders.demonstrations import build as build_demonstrations
        build_demonstrations(self, build)
        from eai_app.ui.builders.multiview import build as build_multiview
        build_multiview(self, build)
        from eai_app.ui.builders.robot_workspace import build as build_robot_workspace
        build_robot_workspace(self, build)
        from eai_app.ui.builders.skeleton import build as build_skeleton
        build_skeleton(self, build)
        from eai_app.ui.builders.model_tests import build as build_model_tests
        build_model_tests(self, build)
        from eai_app.ui.builders.context_learning import build as build_context_learning
        build_context_learning(self, build)
        from eai_app.ui.builders.reward_evaluation import build as build_reward_evaluation
        build_reward_evaluation(self, build)
        from eai_app.ui.builders.reward_training import build as build_reward_training
        build_reward_training(self, build)
        from eai_app.ui.builders.online_rl import build as build_online_rl
        build_online_rl(self, build)
        from eai_app.ui.builders.robometer import build as build_robometer
        build_robometer(self, build)
        from eai_app.ui.builders.rynnvalue import build as build_rynnvalue
        build_rynnvalue(self, build)
        from eai_app.ui.builders.roboicl import build as build_roboicl
        build_roboicl(self, build)
        from eai_app.ui.builders.astra import build as build_astra
        build_astra(self, build)
        from eai_app.ui.builders.humanego import build as build_humanego
        build_humanego(self, build)
        from eai_app.ui.builders.datasets import build as build_datasets
        build_datasets(self, build)
        from eai_app.ui.builders.workspace import build as build_workspace
        build_workspace(self, build)
        self._install_development_tools()


    def _install_development_tools(self):
        from eai_app.development.qt_controller import install_development_tools
        install_development_tools(self)
