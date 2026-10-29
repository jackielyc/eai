"""adapters.ros.camera node — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_ros_bridges = module_ref('eai_app.adapters.ros.bridges')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

from eai_app.adapters.ros.node_parts.feedback import FeedbackMixin
from eai_app.adapters.ros.node_parts.feedback_apply_bridged_tcp_state import FeedbackApplyBridgedTcpStateMixin
from eai_app.adapters.ros.node_parts.feedback_image_callback import FeedbackImageCallbackMixin
from eai_app.adapters.ros.node_parts.control import ControlMixin

class CameraTopicNode(FeedbackMixin, FeedbackApplyBridgedTcpStateMixin, FeedbackImageCallbackMixin, ControlMixin, _eai_platform_bootstrap.Node):
    def __init__(self, bridge: _eai_adapters_ros_bridges.RosBridge, prefix: str) -> None:
        super().__init__("camera_topic_viewer")
        self.ros_bridge = bridge
        self.prefix = prefix
        from cv_bridge import CvBridge

        self.cv_bridge = CvBridge()
        self.subscriptions_map: _eai_platform_bootstrap.Dict[str, object] = {}
        self.camera_info_subs: _eai_platform_bootstrap.Dict[str, object] = {}
        self._camera_intrinsics: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[float, float, float, float]] = {}
        self._camera_intrinsics_sizes: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[int, int]] = {}
        self.click_frames = _eai_platform_bootstrap.click_pose.FrameBuffer()
        self.enabled_topics: set[str] = set()
        self._state_lock = _eai_platform_bootstrap.threading.Lock()
        self._pending_prefix: _eai_platform_bootstrap.Optional[str] = None
        self._pending_enabled: _eai_platform_bootstrap.Optional[set[str]] = None
        self._last_camera_topics: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]] = {}
        self._last_status_message = ""
        self._frame_counts: _eai_platform_bootstrap.Dict[str, int] = {}
        self._last_frame_time: _eai_platform_bootstrap.Dict[str, float] = {}
        self._last_subscribe_time: _eai_platform_bootstrap.Dict[str, float] = {}
        self._subscribe_attempts: _eai_platform_bootstrap.Dict[str, int] = {}
        self._camera_frame_ids: _eai_platform_bootstrap.Dict[str, str] = {}
        self._last_ui_emit_time: _eai_platform_bootstrap.Dict[str, float] = {}
        self._robot_subscriptions_map: _eai_platform_bootstrap.Dict[str, object] = {}
        self._robot_subscribe_attempts: _eai_platform_bootstrap.Dict[str, int] = {}
        self._robot_resubscribe_count: _eai_platform_bootstrap.Dict[str, int] = {}
        self._robot_last_subscribe_time: _eai_platform_bootstrap.Dict[str, float] = {}
        self._robot_tcp_topic_received: _eai_platform_bootstrap.Dict[str, bool] = {}
        self._robot_tcp_give_up: _eai_platform_bootstrap.Dict[str, bool] = {}
        self._topic_graph_cache: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]]] = None
        self._topic_graph_cache_at = 0.0
        self._last_tcp_accept_time: _eai_platform_bootstrap.Dict[str, float] = {}
        self._tcp_min_interval_s = 0.1  # mink_fk~100Hz，UI 限到 10Hz 防卡顿
        self._pending_ui_frames: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.np.ndarray] = {}
        self._pending_ui_stats: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[int, float]] = {}
        self._pending_ui_lock = _eai_platform_bootstrap.threading.Lock()
        self._frame_tick_pending = False

        self._robot_lock = _eai_platform_bootstrap.threading.Lock()
        self._robot_state = _eai_adapters_robot_geometry.RobotStateSnapshot([], [], [], [], [], None, None, 0.0)
        self._last_robot_state_emit = 0.0
        from tf2_ros import Buffer as _TfBuffer

        # TF / 机器人订阅 / 相机扫描：不在启动时自动做，等 UI 按钮
        self._tf_buffer = _TfBuffer()
        self._tf_listener = None
        self._head_camera_tf = None
        self._robot_subs_ready = False
        self._topic_scan_ready = False
        self._TransformListener = None  # lazy import
        self._allow_robot_qos_probe = False
        self._pending_head_camera_tf = False
        self._robot_connect_steps = []
        self._robot_connect_step_i = 0
        self._robot_connect_label = ""
        self._robot_connect_error = ""
        self._robot_connect_running = False
        self._robot_ui_quiet_until = 0.0
        self._robot_extras_due_at = 0.0
        self._robot_extra_step_list = None
        self._robot_extra_step_i = 0
        self._robot_cb_last = {}
        self._ros_op_queue: "queue.SimpleQueue[Callable[[], None]]" = _eai_platform_bootstrap.queue.SimpleQueue()
        self._left_hand_cmd_pub = self.create_publisher(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC,
            10,
        )
        self._right_hand_cmd_pub = self.create_publisher(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC,
            10,
        )
        # 机器人本体 hand_command_node（无 Docker / 无本机 ruiyan 时生效）
        self._left_ee_cmd_pub = self.create_publisher(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.WBC_LEFT_EE_CMD_TOPIC,
            10,
        )
        self._right_ee_cmd_pub = self.create_publisher(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.WBC_RIGHT_EE_CMD_TOPIC,
            10,
        )
        self._left_ik_pub = self.create_publisher(
            _eai_platform_bootstrap.PoseStamped,
            _eai_configuration_policies.LEFT_IK_TARGET_TOPIC,
            10,
        )
        self._right_ik_pub = self.create_publisher(
            _eai_platform_bootstrap.PoseStamped,
            _eai_configuration_policies.RIGHT_IK_TARGET_TOPIC,
            10,
        )
        self._model_left_tcp_pub = self.create_publisher(
            _eai_platform_bootstrap.PoseStamped,
            _eai_configuration_policies.MODEL_LEFT_TCP_TOPIC,
            10,
        )
        self._model_right_tcp_pub = self.create_publisher(
            _eai_platform_bootstrap.PoseStamped,
            _eai_configuration_policies.MODEL_RIGHT_TCP_TOPIC,
            10,
        )
        self._control_mode_pub = self.create_publisher(
            _eai_platform_bootstrap.UInt32,
            _eai_configuration_policies.CONTROL_MODE_TOPIC,
            10,
        )
        self._wbc_joint_pub = self.create_publisher(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC,
            10,
        )
        self._ik_sync_client = self.create_client(_eai_platform_bootstrap.Trigger, "/ik/sync_joint_state")
        self._arm_set_enable_client = self.create_client(_eai_platform_bootstrap.SetBool, _eai_configuration_policies.ARM_SET_ENABLE_SERVICE)
        self._arm_reset_client = self.create_client(_eai_platform_bootstrap.SetBool, _eai_configuration_policies.ARM_RESET_TO_INITIAL_SERVICE)
        self._hand_set_enable_client = self.create_client(_eai_platform_bootstrap.SetBool, _eai_configuration_policies.HAND_SET_ENABLE_SERVICE)
        self._replay_start_client = self.create_client(_eai_platform_bootstrap.Trigger, _eai_configuration_policies.REPLAY_START_SERVICE)
        self._replay_stop_client = self.create_client(_eai_platform_bootstrap.Trigger, _eai_configuration_policies.REPLAY_STOP_SERVICE)
        self._wbc_target_joints_sub = None
        self._left_hand_at_a = True
        self._right_hand_at_a = True
        self._slow_motion_active = False
        self._slow_motion_preparing = False
        self._slow_motion_prep: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None
        self._slow_motion_phase = ""
        self._slow_motion_poses: _eai_platform_bootstrap.List[
            _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
        ] = []
        self._slow_motion_joint_poses: _eai_platform_bootstrap.List[_eai_platform_bootstrap.List[float]] = []
        self._slow_motion_step = 0
        self._slow_motion_hold_steps = 0
        self._slow_motion_timer = None
        self._slow_motion_prep_timer = None
        self._slow_motion_mode_timer = None
        self._replay_prep_mode_timer = None
        self._slow_motion_saved_control_mode: _eai_platform_bootstrap.Optional[int] = None
        self._slow_motion_moving_side = "left"
        # 移动时保持另一侧手臂的位姿（IK 双臂目标需要同时发布）
        self._slow_motion_hold_pose: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
        ] = None
        self._slow_motion_final_moving_pose: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
        ] = None
        self._slow_motion_final_joints: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._commanded_q16: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._motion_seed_override: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._slow_motion_wbc_baseline: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._slow_motion_wbc_changed = False
        self._last_wbc_joints: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]] = None
        self._arm_move_joint_speed_rad_s = _eai_configuration_chat.ARM_MOVE_SPEED_DEFAULT_RAD_S
        self._hal_arm_received_at = 0.0
        self._control_mode = _eai_adapters_models_qwen_remote.IDLE_CONTROL_MODE
        self._arm_enabled = False
        self._hand_enabled = False
        self._control_mode_received_at = 0.0
        self._arm_enable_received_at = 0.0
        self._arm_enabled_since = 0.0
        self._replay_arm_settle_sec = 3.0
        self._replay_hand_settle_sec = 3.0
        self._hand_enable_received_at = 0.0
        self._hand_enabled_since = 0.0
        self._replay_running_state = 0
        self._scan_timer = None
