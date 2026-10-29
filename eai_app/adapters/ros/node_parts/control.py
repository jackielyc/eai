"""adapters.ros.node parts.control — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ControlMixin:
    def is_arm_enabled(self) -> bool:
        return self._arm_enabled

    def set_arm_move_joint_speed(self, speed_rad_s: float) -> None:
        self._arm_move_joint_speed_rad_s = max(
            _eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S,
            min(_eai_configuration_chat.ARM_MOVE_SPEED_MAX_RAD_S, float(speed_rad_s)),
        )

    def get_arm_move_joint_speed(self) -> float:
        return self._arm_move_joint_speed_rad_s

    def get_arm_enable_label(self) -> str:
        if not self._arm_enable_received_at:
            return "手臂: 等待 /arm/enable_state ..."
        return f"手臂: {'已使能' if self._arm_enabled else '未使能（请点「启用手臂」）'}"

    def request_arm_enable(self, enable: bool = True) -> str:
        if enable and self._arm_enabled:
            return "手臂已使能"
        if not enable and not self._arm_enabled:
            return "手臂已关闭"
        # 宿主机桥：UDP + ROS 服务（无 Docker）
        _eai_adapters_robot_geometry.relay_arm_enable(bool(enable))
        self._call_set_enable_async(self._arm_set_enable_client, bool(enable), "arm")
        self.get_logger().info(f"arm/set_enable({enable}) via host bridge")
        # 本地立即门控，避免关使能后短暂窗口仍可点移动 / 发轨迹
        prev = self._arm_enabled
        self._arm_enabled = bool(enable)
        self._arm_enabled_since = _eai_platform_bootstrap.time.time() if enable else 0.0
        if self._arm_enable_received_at <= 0:
            self._arm_enable_received_at = _eai_platform_bootstrap.time.time()
        if prev != self._arm_enabled:
            self.ros_bridge.arm_enable_changed.emit(self._arm_enabled)
        if enable:
            return "已请求启用手臂（宿主机桥，请等待「已使能」）"
        return "已请求关闭手臂"

    def is_hand_enabled(self) -> bool:
        return self._hand_enabled

    def get_hand_enable_label(self) -> str:
        if not self._hand_enable_received_at:
            return "手: 等待 /hand/enable_state ..."
        return f"手: {'已使能' if self._hand_enabled else '未使能'}"

    def request_hand_enable(self, enable: bool = True) -> str:
        if enable and self._hand_enabled:
            return "手部已使能"
        if not enable and not self._hand_enabled:
            return "手部已关闭"
        _eai_adapters_robot_geometry.relay_hand_enable(bool(enable))
        self._call_set_enable_async(self._hand_set_enable_client, bool(enable), "hand")
        self.get_logger().info(f"hand/set_enable({enable}) via host bridge")
        if enable:
            return "已请求启用手部（宿主机桥）"
        return "已请求关闭手部"

    def _call_set_enable_async(self, client, enable: bool, label: str) -> None:
        def _do() -> None:
            try:
                if not client.service_is_ready():
                    if not client.wait_for_service(timeout_sec=0.4):
                        return
                req = _eai_platform_bootstrap.SetBool.Request()
                req.data = bool(enable)
                fut = client.call_async(req)

                def _done(f) -> None:
                    try:
                        res = f.result()
                        self.get_logger().info(
                            f"{label}/set_enable({enable}) -> "
                            f"{getattr(res, 'success', None)} "
                            f"{getattr(res, 'message', '')}"
                        )
                    except Exception as exc:
                        self.get_logger().warn(f"{label}/set_enable 失败: {exc}")

                fut.add_done_callback(_done)
            except Exception as exc:
                self.get_logger().warn(f"{label}/set_enable 异常: {exc}")

        self.enqueue_ros_op(_do)

    def get_arm_move_blockers(self, tf_timeout_s: float = _eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S) -> _eai_platform_bootstrap.List[str]:
        blockers: _eai_platform_bootstrap.List[str] = []
        if not self._arm_enabled:
            if self._arm_enable_received_at <= 0:
                blockers.append(
                    f"未收到 {_eai_configuration_policies.ARM_ENABLE_STATE_TOPIC}（请先点「启用手臂」）"
                )
            else:
                blockers.append("手臂未使能（请先点「启用手臂」）")
        state = self.get_robot_state()
        if state.left_tcp is None or not state.left_tcp.valid:
            blockers.append("无左臂 TCP 数据（/tele/fk/left_pose 或 /mink_fk/left_tcp_pose）")
        elif self._tcp_pose_in_ik_frame("left", timeout_s=tf_timeout_s) is None:
            frame = state.left_tcp.frame_id or "未知"
            blockers.append(
                f"左臂 TCP 在 {frame} 系，无法用于 IK（需 base_link 或 /mink_fk）"
            )
        if state.right_tcp is None or not state.right_tcp.valid:
            blockers.append("无右臂 TCP 数据")
        elif self._tcp_pose_in_ik_frame("right", timeout_s=tf_timeout_s) is None:
            frame = state.right_tcp.frame_id or "未知"
            blockers.append(
                f"右臂 TCP 在 {frame} 系，无法用于 IK（需 base_link 或 /mink_fk）"
            )
        return blockers

    def get_arm_move_current_label(self, arm_side: str = "left") -> str:
        state = self.get_robot_state()
        hand_line = _eai_perception_pose_estimation.format_left_hand_state_line(state)
        return f"当前  {hand_line}  |  {self._format_arm_tcp_current_line(arm_side)}"

    def get_arm_move_current_label_both(self) -> str:
        return (
            f"当前  左: {self._format_arm_tcp_current_line('left')}  |  "
            f"右: {self._format_arm_tcp_current_line('right')}"
        )

    def resolve_segment_move_goal(
        self,
        segment: _eai_perception_pose_estimation.SegmentPoseTarget,
        timeout_s: float = _eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S,
        arm_side: str = "left",
    ) -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.ResolvedArmMoveGoal]:
        goal = _eai_adapters_robot_geometry.transform_pose_to_base(
            segment.position_xyz,
            segment.quaternion_xyzw,
            segment.camera_frame,
            self._tf_buffer,
            base_frame=_eai_configuration_policies.IK_TARGET_FRAME,
            timeout_s=timeout_s,
        )
        if goal is None:
            return None
        xyz, quat = goal
        if segment.keep_tcp_orientation:
            tcp = self._tcp_pose_in_ik_frame(arm_side, timeout_s=timeout_s)
            if tcp is None:
                return None
            _, quat = tcp
        kind = "FoundationPose" if "foundationpose" in (segment.label or "").lower() else "分割位姿"
        orient_note = "位置绝对·姿态保持" if segment.keep_tcp_orientation else ""
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        label = f"{side_name}{kind} ({segment.label or segment.source_topic})"
        if orient_note:
            label = f"{label} [{orient_note}]"
        return _eai_perception_pose_estimation.ResolvedArmMoveGoal(
            position_xyz=xyz,
            quaternion_xyzw=quat,
            label=label,
            arm_side=arm_side,
        )

    def get_arm_move_pose_labels(
        self,
        segment: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.SegmentPoseTarget],
        arm_side: str = "left",
    ) -> _eai_platform_bootstrap.Tuple[str, str]:
        """保留兼容：仅在没有 window 侧解析时使用。"""
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        current_line = self.get_arm_move_current_label(arm_side)
        target_line = f"目标  {side_name} TCP: (请先设置)"
        if segment is not None:
            resolved = self.resolve_segment_move_goal(segment, arm_side=arm_side)
            if resolved is not None:
                target_line = (
                    f"目标  [{resolved.label}]  "
                    f"{_eai_perception_pose_estimation.format_xyz_rpy_line(side_name, resolved.position_xyz, resolved.quaternion_xyzw)}"
                )
            else:
                target_line = (
                    f"目标  {side_name} TCP: TF 不可用 "
                    f"({segment.camera_frame} -> {_eai_configuration_policies.BASE_LINK_FRAME})"
                )
        return current_line, target_line

    def _publish_ik_arm_targets(
        self,
        left_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        left_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
        right_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        right_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> None:
        if getattr(self, "_grasp_execution_reserved", False):
            return
        left_msg = self._make_pose_stamped_msg(left_xyz, left_quat)
        right_msg = self._make_pose_stamped_msg(right_xyz, right_quat)
        self._left_ik_pub.publish(left_msg)
        self._right_ik_pub.publish(right_msg)

    def _publish_dual_arm_targets(
        self,
        left_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        left_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
        right_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        right_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> None:
        if getattr(self, "_grasp_execution_reserved", False):
            return
        self._publish_ik_arm_targets(left_xyz, left_quat, right_xyz, right_quat)
        left_msg = self._make_pose_stamped_msg(left_xyz, left_quat)
        right_msg = self._make_pose_stamped_msg(right_xyz, right_quat)
        self._model_left_tcp_pub.publish(left_msg)
        self._model_right_tcp_pub.publish(right_msg)

    def _publish_control_mode(self, mode: int) -> None:
        if (getattr(self, "_grasp_execution_reserved", False) and mode != _eai_configuration_chat.MODEL_CONTROL_MODE
                and not getattr(self, "_grasp_mode_command_authorized", False)):
            return
        msg = _eai_platform_bootstrap.UInt32()
        msg.data = int(mode)
        self._control_mode_pub.publish(msg)
        _eai_adapters_robot_geometry.relay_control_mode(int(mode))
        self._control_mode = int(mode)

    def _publish_slow_motion_targets(
        self,
        moving_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        moving_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> None:
        hold_pose = self._slow_motion_hold_pose
        if hold_pose is None:
            return
        hold_xyz, hold_quat = hold_pose
        if self._slow_motion_moving_side == "right":
            self._publish_ik_arm_targets(hold_xyz, hold_quat, moving_xyz, moving_quat)
        else:
            self._publish_ik_arm_targets(moving_xyz, moving_quat, hold_xyz, hold_quat)

    def _publish_wbc_joint_target(self, positions: _eai_platform_bootstrap.List[float]) -> None:
        if getattr(self, "_grasp_execution_reserved", False) and not getattr(self, "_grasp_joint_command_authorized", False):
            return
        if not self._arm_enabled:
            self.get_logger().warn(
                "跳过关节目标：手臂未使能（请先点「启用手臂」）"
            )
            return
        msg = self._make_wbc_joint_msg(positions)
        self._wbc_joint_pub.publish(msg)
        _eai_adapters_robot_geometry.relay_wbc_joints(msg)

    def _release_ik_for_joint_move(self) -> None:
        """切到 IDLE 清除 ik_node 双臂目标，再切回 model 关节模式，避免与关节轨迹抢 /wbc/target_joints。"""
        self._publish_control_mode(_eai_adapters_models_qwen_remote.IDLE_CONTROL_MODE)
        self._burst_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)

    def _start_joint_move_from_prep(self, prep: _eai_platform_bootstrap.Dict[str, object], goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> None:
        start_joints: _eai_platform_bootstrap.List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
        goal_joints: _eai_platform_bootstrap.List[float] = prep["goal_joints"]  # type: ignore[assignment]
        duration_s = _eai_adapters_robot_geometry.compute_arm_move_duration_s(
            start_joints, goal_joints, self._arm_move_joint_speed_rad_s
        )
        max_delta = max(abs(goal_joints[i] - start_joints[i]) for i in range(16))
        start_xyz = prep.get("start_xyz")
        goal_xyz = prep.get("goal_xyz")
        cart_dist = 0.0
        if (
            isinstance(start_xyz, tuple)
            and isinstance(goal_xyz, tuple)
            and len(start_xyz) == 3
            and len(goal_xyz) == 3
        ):
            cart_dist = float(
                sum((float(goal_xyz[i]) - float(start_xyz[i])) ** 2 for i in range(3)) ** 0.5
            )
        if duration_s <= 0:
            if cart_dist > _eai_configuration_chat.ARM_MOVE_CART_NEAR_M:
                self._abort_slow_motion_prep(
                    f"IK 未求解到目标（笛卡尔距离 {cart_dist:.3f} m，关节变化 {max_delta:.5f} rad）。"
                    "目标可能不可达，或 IK 未更新 /wbc/target_joints"
                )
            else:
                self._abort_slow_motion_prep("目标关节与当前几乎相同，无需移动")
            return
        steps = max(2, int(_eai_platform_bootstrap.math.ceil(duration_s * _eai_configuration_chat.ARM_MOVE_JOINT_HZ)))
        joint_poses = _eai_adapters_robot_geometry.build_interpolated_joints(start_joints, goal_joints, steps, linear=True)
        self.get_logger().info(
            f"关节轨迹: {steps} 步 @ {_eai_configuration_chat.ARM_MOVE_JOINT_HZ:.0f}Hz / {duration_s:.1f}s, "
            f"最大关节变化 {max_delta:.3f} rad, 笛卡尔 {cart_dist:.3f} m, "
            f"发布 {_eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC}"
        )
        self._slow_motion_joint_poses = joint_poses
        self._slow_motion_final_joints = goal_joints
        self._slow_motion_preparing = False
        self._slow_motion_prep = None
        self._stop_slow_motion_prep_timer()
        self._slow_motion_active = True
        self._slow_motion_phase = "move"
        self._slow_motion_hold_steps = 0
        self._commanded_q16 = list(goal_joints)
        self._publish_wbc_joint_target(start_joints)
        self._slow_motion_step = 0
        if self._slow_motion_timer is not None:
            self.destroy_timer(self._slow_motion_timer)
        self._slow_motion_timer = self.create_timer(
            1.0 / _eai_configuration_chat.ARM_MOVE_JOINT_HZ, self._slow_motion_tick
        )
        goal_xyz_t = prep["goal_xyz"]  # type: ignore[assignment]
        gx, gy, gz = goal_xyz_t
        saved_mode = int(prep.get("saved_mode", self._control_mode))
        side_name = _eai_perception_pose_estimation.arm_side_label(goal.arm_side)
        self.get_logger().info(
            f"{side_name}移动开始 [{goal.label}]: mode {saved_mode}->"
            f"{self._control_mode if self._slow_motion_saved_control_mode is not None else saved_mode}, "
            f"{steps} 关节步 / {duration_s:.1f}s -> {_eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC}, "
            f"目标 XYZ=({gx:.3f}, {gy:.3f}, {gz:.3f}) [{_eai_configuration_policies.IK_TARGET_FRAME}]"
        )
        mode_hint = ""
        if self._slow_motion_saved_control_mode is not None:
            mode_hint = "，已临时切到 model 模式(0)"
        message = f"开始移动 [{goal.label}]（约 {duration_s:.1f}s @ {self._arm_move_joint_speed_rad_s:.2f} rad/s）{mode_hint}"
        self.ros_bridge.status_message.emit(message)
        self.ros_bridge.slow_motion_progress.emit(0.0, message)

    def request_slow_move_to_goal(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> str:
        if getattr(self, "_grasp_execution_reserved", False):
            return "无法移动: 分阶段抓取占用控制通道，请先停止抓取"
        if self.is_slow_motion_busy():
            return "已有移动进行中，请等待完成"
        blockers = self.get_arm_move_blockers()
        if blockers:
            msg = "无法移动: " + "；".join(blockers)
            self.get_logger().warn(msg)
            return msg
        moving_side = goal.arm_side if goal.arm_side in ("left", "right") else "left"
        self._slow_motion_moving_side = moving_side
        self._slow_motion_preparing = True
        override = getattr(self, "_motion_seed_override", None)
        self._motion_seed_override = None
        self._slow_motion_prep = {
            "goal": goal,
            "phase": "validate",
            "warmup_step": 0,
            "moving_side": moving_side,
        }
        if isinstance(override, list) and len(override) >= 16:
            self._slow_motion_prep["seed_q16"] = [float(v) for v in override[:16]]
        self._slow_motion_poses = []
        self._slow_motion_joint_poses = []
        self._slow_motion_step = 0
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._start_slow_motion_prep_timer()
        self.ros_bridge.slow_motion_progress.emit(0.0, "正在准备移动...")
        return "正在准备移动..."

    def request_joint_space_move(
        self,
        start_q16: _eai_platform_bootstrap.Sequence[float],
        goal_q16: _eai_platform_bootstrap.Sequence[float],
        goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal,
    ) -> str:
        """不重新求 IK，从上次下发的关节角插值回到给定关节角。"""
        if getattr(self, "_grasp_execution_reserved", False):
            return "无法移动: 分阶段抓取占用控制通道，请先停止抓取"
        if self.is_slow_motion_busy():
            return "已有移动进行中，请等待完成"
        blockers = self.get_arm_move_blockers()
        if blockers:
            msg = "无法移动: " + "；".join(blockers)
            self.get_logger().warn(msg)
            return msg
        start = [float(v) for v in list(start_q16)[:16]]
        target = [float(v) for v in list(goal_q16)[:16]]
        if len(start) < 16 or len(target) < 16:
            return "无法移动: 关节角不足"
        if max(abs(target[i] - start[i]) for i in range(16)) < 1e-4:
            return "目标关节与当前几乎相同，无需移动"
        moving_side = goal.arm_side if goal.arm_side in ("left", "right") else "left"
        self._slow_motion_moving_side = moving_side
        self._enter_model_mode_for_motion()
        prep = {
            "motion_start_joints": start,
            "goal_joints": target,
            "start_xyz": tuple(goal.position_xyz),
            "goal_xyz": tuple(goal.position_xyz),
            "saved_mode": self._control_mode,
        }
        self._start_joint_move_from_prep(prep, goal)
        if not self._slow_motion_active:
            return "目标关节与当前几乎相同，无需移动"
        return f"关节回初始 [{goal.label}]"

    def start_slow_move_to_goal(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> str:
        return self.request_slow_move_to_goal(goal)

    def start_slow_move_to_segment(
        self, target: _eai_perception_pose_estimation.SegmentPoseTarget, arm_side: str = "left"
    ) -> str:
        resolved = self.resolve_segment_move_goal(
            target, timeout_s=_eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S, arm_side=arm_side
        )
        if resolved is None:
            return (
                f"无法将分割位姿从 {target.camera_frame or '未知'} "
                f"变换到 {_eai_configuration_policies.BASE_LINK_FRAME}（请检查 TF）"
            )
        return self.start_slow_move_to_goal(resolved)

    def _publish_left_ik_target(
        self,
        xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> None:
        right_pose = self._get_right_tcp_in_base()
        if right_pose is None:
            return
        self._publish_dual_arm_targets(xyz, quat_xyzw, right_pose[0], right_pose[1])

    def _publish_hand_command(self, side: str, msg: _eai_platform_bootstrap.JointState) -> bool:
        """发到 ry_hand（本机驱动）+ wbc ee（机器人 onboard）+ UDP 桥。

        未「启用手」时直接拒绝（UI 以前直发 set_angles，关使能仍能动）。
        """
        if getattr(self, "_grasp_execution_reserved", False) and not getattr(self, "_grasp_hand_command_authorized", False):
            self.get_logger().warn("抓取占用手部控制，忽略其他开合命令")
            return False
        if not self._hand_enabled:
            self.get_logger().warn(
                "跳过手部指令：手未使能（请先点「启用手」）"
            )
            return False
        side = "right" if side == "right" else "left"
        if side == "right":
            self._right_hand_cmd_pub.publish(msg)
            self._right_ee_cmd_pub.publish(msg)
        else:
            self._left_hand_cmd_pub.publish(msg)
            self._left_ee_cmd_pub.publish(msg)
        _eai_adapters_robot_geometry.relay_hand_command(side, msg)
        return True

    def ensure_enable_state_subscriptions(self) -> None:
        """订阅 arm/hand enable_state 与 control_mode（排队到 ROS 线程）。"""
        enable_qos = self._robot_fast_qos(reliable=True)

        def _sub(topic: str, cb) -> None:
            if topic in self._robot_subscriptions_map:
                return
            sub = self.create_subscription(_eai_platform_bootstrap.UInt32, topic, cb, enable_qos)
            self._robot_subscriptions_map[topic] = sub
            self.get_logger().info(f"已订阅 {topic}")

        def _do() -> None:
            _sub(_eai_configuration_policies.ARM_ENABLE_STATE_TOPIC, self._on_arm_enable_state)
            _sub(_eai_configuration_policies.HAND_ENABLE_STATE_TOPIC, self._on_hand_enable_state)
            _sub(_eai_configuration_policies.CONTROL_MODE_TOPIC, self._on_control_mode)

        self.enqueue_ros_op(_do)

    def _on_arm_enable_state(self, msg: _eai_platform_bootstrap.UInt32) -> None:
        enabled = int(msg.data) != 0
        first_receive = self._arm_enable_received_at <= 0
        if first_receive or enabled != self._arm_enabled:
            self.get_logger().info(f"arm/enable_state={'ON' if enabled else 'OFF'}")
        self._arm_enable_received_at = _eai_platform_bootstrap.time.time()
        if first_receive or enabled != self._arm_enabled:
            self._arm_enabled = enabled
            self._arm_enabled_since = _eai_platform_bootstrap.time.time() if enabled else 0.0
            self.ros_bridge.arm_enable_changed.emit(enabled)
        else:
            self._arm_enabled = enabled

    def _on_hand_enable_state(self, msg: _eai_platform_bootstrap.UInt32) -> None:
        enabled = int(msg.data) != 0
        first_receive = self._hand_enable_received_at <= 0
        if first_receive or enabled != self._hand_enabled:
            self.get_logger().info(f"hand/enable_state={'ON' if enabled else 'OFF'}")
        self._hand_enable_received_at = _eai_platform_bootstrap.time.time()
        if first_receive or enabled != self._hand_enabled:
            self._hand_enabled = enabled
            self._hand_enabled_since = _eai_platform_bootstrap.time.time() if enabled else 0.0
            self.ros_bridge.hand_enable_changed.emit(enabled)
        else:
            self._hand_enabled = enabled

    def set_enabled_topics(self, topics: set[str]) -> None:
        with self._state_lock:
            self._pending_enabled = set(topics)

    def _qos_from_publisher_info(self, topic: str) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QoSProfile]:
        """从发布端提取 reliability/durability，构建合法 QoS（不可直接使用 endpoint QoS）。"""
        try:
            publishers = self.get_publishers_info_by_topic(topic)
            if not publishers:
                return None
            pub_qos = publishers[0].qos_profile

            reliability = pub_qos.reliability
            if reliability not in (
                _eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
                _eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT,
            ):
                reliability = _eai_platform_bootstrap.ReliabilityPolicy.RELIABLE

            durability = pub_qos.durability
            if durability not in (
                _eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
                _eai_platform_bootstrap.DurabilityPolicy.TRANSIENT_LOCAL,
            ):
                durability = _eai_platform_bootstrap.DurabilityPolicy.VOLATILE

            return _eai_platform_bootstrap.QoSProfile(
                reliability=reliability,
                durability=durability,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=30,
            )
        except Exception as exc:
            self.get_logger().debug(f"{topic}: 解析发布端 QoS 失败: {exc}")
            return None

    def _publish_frame_to_ui(self, topic: str, cv_image: _eai_platform_bootstrap.np.ndarray, now: float, source_meta=None) -> None:
        full_resolution_image = cv_image
        count = self._frame_counts.get(topic, 0) + 1
        self._frame_counts[topic] = count
        self._last_frame_time[topic] = now
        # Viewer 可通过 _ui_frame_paused_until 暂停（切 tab）；用 bridge 上的标记
        pause_until = float(getattr(self.ros_bridge, "_ui_frame_paused_until", 0.0) or 0.0)
        if now < pause_until:
            return
        if not self._should_emit_frame_to_ui(topic, now):
            return
        # 在 ROS 线程先缩图，减轻 UI 线程与 Qt 队列压力
        try:
            img = _eai_platform_bootstrap.np.asarray(cv_image)
            h0, w0 = img.shape[:2]
            edge = max(h0, w0)
            max_edge = _eai_configuration_policies.UI_PREVIEW_MAX_EDGE
            if _eai_perception_image_geometry.is_depth_topic(topic):
                max_edge = min(max_edge, 480)
            if edge > max_edge > 0:
                scale = float(max_edge) / float(edge)
                img = _eai_platform_bootstrap.cv2.resize(
                    img,
                    (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                    interpolation=_eai_platform_bootstrap.cv2.INTER_NEAREST if _eai_perception_image_geometry.is_depth_topic(topic) else _eai_platform_bootstrap.cv2.INTER_AREA,
                )
            cv_image = img
        except Exception:
            pass
        if source_meta is not None:
            stamp, frame_id, encoding = source_meta
            self.click_frames.add(topic, full_resolution_image, stamp, frame_id, encoding, displayed=cv_image)
        # 合并：同 topic 只保留最新一帧，避免 Qt 队列堆积拷贝
        with self._pending_ui_lock:
            self._pending_ui_frames[topic] = cv_image
            self._pending_ui_stats[topic] = (count, now)
            need_tick = not self._frame_tick_pending
            self._frame_tick_pending = True
        if need_tick:
            self.ros_bridge.frame_tick.emit()

