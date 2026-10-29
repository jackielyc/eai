"""adapters.ros.node parts.feedback — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class FeedbackMixin:
    def prepare_shutdown(self) -> None:
        """关闭窗口时停止 ROS 定时器与慢速移动，避免退出阻塞。"""
        if self._slow_motion_active or self._slow_motion_preparing:
            self._slow_motion_active = False
            self._slow_motion_preparing = False
            self._slow_motion_prep = None
            self._slow_motion_phase = ""
            self._slow_motion_poses = []
            self._slow_motion_joint_poses = []
            self._slow_motion_step = 0
            self._slow_motion_hold_steps = 0
            self._stop_slow_motion_timer()
            self._stop_slow_motion_prep_timer()
            self._stop_slow_motion_mode_timer()
            self._stop_replay_prep_mode_timer()
            self._restore_control_mode_if_needed()
        if self._scan_timer is not None:
            try:
                self.destroy_timer(self._scan_timer)
            except Exception:
                pass
            self._scan_timer = None
        self._stop_replay_prep_mode_timer()
        self._tf_listener = None
        self._robot_extras_due_at = 0.0

    def is_left_hand_at_a(self) -> bool:
        return self._left_hand_at_a

    def is_right_hand_at_a(self) -> bool:
        return self._right_hand_at_a

    def arm_ready_for_replay(self, settle_sec: _eai_platform_bootstrap.Optional[float] = None) -> bool:
        if not self._arm_enabled:
            return False
        settle = self._replay_arm_settle_sec if settle_sec is None else settle_sec
        since = self._arm_enabled_since if self._arm_enabled_since > 0 else self._arm_enable_received_at
        if since <= 0:
            return False
        return (_eai_platform_bootstrap.time.time() - since) >= settle

    def is_hal_arm_ready(self) -> bool:
        return self._hal_arm_received_at > 0.0

    def request_arm_reset_to_initial(self) -> str:
        """调用 /arm/reset_to_initial，把手臂（含腰/头）回到初始位姿。"""
        if getattr(self, "_grasp_execution_reserved", False):
            return "抓取占用控制通道，请先停止抓取再归位"
        _eai_adapters_robot_geometry.relay_arm_reset_to_initial(True)
        if not self._arm_reset_client.service_is_ready():
            if not self._arm_reset_client.wait_for_service(timeout_sec=0.3):
                return "已中继请求回初始位（请确认 arm_control 已运行）"
        req = _eai_platform_bootstrap.SetBool.Request()
        req.data = True
        future = self._arm_reset_client.call_async(req)

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(
                        f"arm/reset_to_initial OK: {result.message}"
                    )
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"arm/reset_to_initial 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"arm/reset_to_initial 异常: {exc}")

        future.add_done_callback(_done)
        return "已请求手臂回初始位"

    def hand_ready_for_replay(self, settle_sec: _eai_platform_bootstrap.Optional[float] = None) -> bool:
        if not self._hand_enabled:
            return False
        settle = self._replay_hand_settle_sec if settle_sec is None else settle_sec
        since = self._hand_enabled_since if self._hand_enabled_since > 0 else self._hand_enable_received_at
        if since <= 0:
            return False
        return (_eai_platform_bootstrap.time.time() - since) >= settle

    def get_control_mode_label(self) -> str:
        if not self._control_mode_received_at:
            return "mode: 等待 /control_mode ..."
        return f"mode: {self._control_mode}"

    def get_replay_state(self) -> int:
        return self._replay_running_state

    def get_replay_state_label(self) -> str:
        return _eai_configuration_policies.REPLAY_STATE_LABELS.get(
            self._replay_running_state,
            f"未知({self._replay_running_state})",
        )

    def replay_start_service_ready(self) -> bool:
        return self._replay_start_client.service_is_ready()

    def prepare_for_rrd_replay(self) -> _eai_platform_bootstrap.List[str]:
        """回放前切 model 模式并请求手/臂使能；需 pedal_controller 已运行才会收到 enable_state。"""
        self._start_replay_prep_mode_lock()
        return self.get_replay_prerequisite_warnings()

    def get_replay_prerequisite_warnings(self) -> _eai_platform_bootstrap.List[str]:
        warnings: _eai_platform_bootstrap.List[str] = []
        if not _eai_adapters_robot_services._is_stack_services_running():
            warnings.append(
                "手/臂服务栈未运行（需 pedal_controller / topic_router，请先点「启动机器人栈」）"
            )
        if not self._control_mode_received_at:
            warnings.append("未收到 /control_mode（等待 pedal_controller 或 viewer 发布）")
        elif self._control_mode != _eai_configuration_chat.MODEL_CONTROL_MODE:
            warnings.append(f"control_mode={self._control_mode}，需为 {_eai_configuration_chat.MODEL_CONTROL_MODE}")
        if not self._arm_enable_received_at:
            warnings.append("未收到 /arm/enable_state（请使能手臂）")
        elif not self._arm_enabled:
            warnings.append("手臂未使能")
        if not self._hand_enable_received_at:
            warnings.append("未收到 /hand/enable_state（请使能手部）")
        elif not self._hand_enabled:
            warnings.append("手部未使能")
        return warnings

    def request_replay_start(self) -> str:
        if not self._replay_start_client.service_is_ready():
            if not self._replay_start_client.wait_for_service(timeout_sec=0.2):
                return f"服务 {_eai_configuration_policies.REPLAY_START_SERVICE} 尚未就绪"
        future = self._replay_start_client.call_async(_eai_platform_bootstrap.Trigger.Request())

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(f"replay start OK: {result.message}")
                    self.ros_bridge.status_message.emit(
                        result.message or "回放已启动"
                    )
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"replay start 失败: {msg}")
                    self.ros_bridge.status_message.emit(f"启动回放失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"replay start 异常: {exc}")
                self.ros_bridge.status_message.emit(f"启动回放异常: {exc}")

        future.add_done_callback(_done)
        return "已请求开始回放"

    def request_replay_stop(self) -> str:
        if not self._replay_stop_client.service_is_ready():
            if not self._replay_stop_client.wait_for_service(timeout_sec=1.5):
                return f"服务 {_eai_configuration_policies.REPLAY_STOP_SERVICE} 不可用"
        future = self._replay_stop_client.call_async(_eai_platform_bootstrap.Trigger.Request())

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(f"replay stop OK: {result.message}")
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"replay stop 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"replay stop 异常: {exc}")

        future.add_done_callback(_done)
        return "已请求停止回放"

    def is_slow_motion_active(self) -> bool:
        return self._slow_motion_active

    def is_slow_motion_preparing(self) -> bool:
        return self._slow_motion_preparing

    def is_slow_motion_busy(self) -> bool:
        return self._slow_motion_active or self._slow_motion_preparing

    def _is_motion_control_locked(self) -> bool:
        return self.is_slow_motion_busy()

    def _tcp_pose_in_ik_frame(
        self,
        side: str,
        timeout_s: float = _eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
        state = self.get_robot_state()
        tcp = state.left_tcp if side == "left" else state.right_tcp
        if tcp is None or not tcp.valid:
            return None
        # mink_fk / host_fk 标 frame=map，数值即 base_link：先走别名，禁止先 lookup_transform
        #（无 TF 时 Duration(1s) 会把 Qt 主线程卡死）
        src = _eai_adapters_robot_geometry.normalize_frame_id(tcp.frame_id)
        if src == _eai_adapters_robot_geometry.normalize_frame_id(_eai_configuration_policies.IK_TARGET_FRAME) or src in _eai_configuration_policies.MINK_FK_FRAME_ALIASES:
            return tcp.xyz, tcp.quat_xyzw
        return _eai_adapters_robot_geometry.transform_pose_to_base(
            tcp.xyz,
            tcp.quat_xyzw,
            tcp.frame_id,
            self._tf_buffer,
            base_frame=_eai_configuration_policies.IK_TARGET_FRAME,
            timeout_s=timeout_s,
        )

    def _get_left_tcp_in_base(
        self,
        timeout_s: float = _eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
        return self._tcp_pose_in_ik_frame("left", timeout_s=timeout_s)

    def _get_right_tcp_in_base(
        self,
        timeout_s: float = _eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
        return self._tcp_pose_in_ik_frame("right", timeout_s=timeout_s)

    def _format_arm_tcp_current_line(self, side: str) -> str:
        state = self.get_robot_state()
        side_name = _eai_perception_pose_estimation.arm_side_label(side)
        # UI 展示路径禁止阻塞 TF
        tcp = self._tcp_pose_in_ik_frame(side, timeout_s=_eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S)
        if tcp is not None:
            xyz, quat = tcp
            return _eai_perception_pose_estimation.format_xyz_rpy_line(side_name, xyz, quat)
        raw = state.left_tcp if side == "left" else state.right_tcp
        if raw is not None and raw.valid:
            frame = raw.frame_id or _eai_configuration_policies.BASE_LINK_FRAME
            return f"{_eai_perception_pose_estimation.format_xyz_rpy_line(side_name, raw.xyz, raw.quat_xyzw)}  [{frame}]"
        return f"{side_name} TCP: (无数据)"

    def _make_pose_stamped_msg(
        self,
        xyz: _eai_platform_bootstrap.Tuple[float, float, float],
        quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> _eai_platform_bootstrap.PoseStamped:
        msg = _eai_platform_bootstrap.PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = _eai_configuration_policies.IK_TARGET_FRAME
        msg.pose.position.x = xyz[0]
        msg.pose.position.y = xyz[1]
        msg.pose.position.z = xyz[2]
        msg.pose.orientation.x = quat_xyzw[0]
        msg.pose.orientation.y = quat_xyzw[1]
        msg.pose.orientation.z = quat_xyzw[2]
        msg.pose.orientation.w = quat_xyzw[3]
        return msg

    def request_model_control_mode(self) -> str:
        """发布 control_mode=0（模型控制），便于手臂移动/回放。"""
        prev = self._control_mode
        self._burst_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)
        self._control_mode_received_at = _eai_platform_bootstrap.time.time()
        self.ros_bridge.control_mode_changed.emit(_eai_configuration_chat.MODEL_CONTROL_MODE)
        self.get_logger().info(
            f"手动切换 control_mode {prev} -> {_eai_configuration_chat.MODEL_CONTROL_MODE}"
        )
        return f"已切换 control_mode -> {_eai_configuration_chat.MODEL_CONTROL_MODE}（模型控制）"

    def _restore_control_mode_if_needed(self) -> None:
        if self._slow_motion_saved_control_mode is not None:
            self._publish_control_mode(self._slow_motion_saved_control_mode)
            self._slow_motion_saved_control_mode = None

    def _maintain_model_mode_during_motion(self) -> None:
        if self._control_mode != _eai_configuration_chat.MODEL_CONTROL_MODE:
            self._publish_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)

    def _start_slow_motion_mode_lock(self) -> None:
        self._stop_slow_motion_mode_timer()
        self._burst_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)
        self._slow_motion_mode_timer = self.create_timer(
            1.0, self._maintain_model_mode_during_motion
        )

    def _stop_slow_motion_mode_timer(self) -> None:
        if self._slow_motion_mode_timer is not None:
            self.destroy_timer(self._slow_motion_mode_timer)
            self._slow_motion_mode_timer = None

    def _maintain_replay_prep(self) -> None:
        self._burst_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)
        if self._arm_set_enable_client.service_is_ready() and not self._arm_enabled:
            self.request_arm_enable(True)
        if self._hand_set_enable_client.service_is_ready() and not self._hand_enabled:
            self.request_hand_enable(True)

    def _start_replay_prep_mode_lock(self) -> None:
        self._stop_replay_prep_mode_timer()
        self._burst_control_mode(_eai_configuration_chat.MODEL_CONTROL_MODE)
        self.request_arm_enable(True)
        self.request_hand_enable(True)
        self._replay_prep_mode_timer = self.create_timer(1.0, self._maintain_replay_prep)

    def _stop_replay_prep_mode_timer(self) -> None:
        if self._replay_prep_mode_timer is not None:
            self.destroy_timer(self._replay_prep_mode_timer)
            self._replay_prep_mode_timer = None

    def _make_wbc_joint_msg(self, positions: _eai_platform_bootstrap.List[float]) -> _eai_platform_bootstrap.JointState:
        msg = _eai_platform_bootstrap.JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = _eai_configuration_policies.IK_TARGET_FRAME
        msg.position = [float(v) for v in positions[:16]]
        return msg

    def _load_host_fk_q16(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]]:
        """从 host_fk 状态文件或 HAL UI 缓存取 16 关节种子（腰2+双臂14）。"""
        path = _eai_platform_bootstrap.os.environ.get(
            "EAI_HOST_FK_STATE",
            _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.tempfile.gettempdir(), "eai_host_fk_state.json"),
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            q = data.get("q16")
            if isinstance(q, list) and len(q) >= 16:
                age = _eai_platform_bootstrap.time.time() - float(data.get("t") or 0.0)
                if age < 5.0:
                    return [float(x) for x in q[:16]]
        except Exception:
            pass
        # 兜底：robot_state 里的腰+臂
        names = [
            "joint_lift_body",
            "joint_body_pitch",
            "joint1_l",
            "joint2_l",
            "joint3_l",
            "joint4_l",
            "joint5_l",
            "joint6_l",
            "joint7_l",
            "joint1_r",
            "joint2_r",
            "joint3_r",
            "joint4_r",
            "joint5_r",
            "joint6_r",
            "joint7_r",
        ]
        with self._robot_lock:
            d = dict(self._robot_state.waist)
            d.update(dict(self._robot_state.left_arm))
            d.update(dict(self._robot_state.right_arm))
        if not all(n in d for n in names[2:]):
            return None
        return [float(d.get(n, 0.0)) for n in names]

    def _stop_slow_motion_prep_timer(self) -> None:
        if self._slow_motion_prep_timer is not None:
            self.destroy_timer(self._slow_motion_prep_timer)
            self._slow_motion_prep_timer = None

    def _start_slow_motion_prep_timer(self) -> None:
        self._stop_slow_motion_prep_timer()
        self._slow_motion_prep_timer = self.create_timer(
            1.0 / _eai_configuration_chat.ARM_MOVE_IK_HZ, self._slow_motion_prep_tick
        )

    def _abort_slow_motion_prep(self, message: str) -> None:
        self._slow_motion_preparing = False
        self._slow_motion_prep = None
        self._slow_motion_hold_pose = None
        self._slow_motion_final_moving_pose = None
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._stop_slow_motion_prep_timer()
        self._stop_slow_motion_mode_timer()
        self._restore_control_mode_if_needed()
        self.get_logger().warn(message)
        self.ros_bridge.slow_motion_finished.emit(False, message)

    def _slow_motion_prep_tick(self) -> None:
        prep = self._slow_motion_prep
        if not self._slow_motion_preparing or prep is None:
            self._stop_slow_motion_prep_timer()
            return

        goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal = prep["goal"]  # type: ignore[assignment]
        phase = str(prep.get("phase", ""))

        if phase == "validate":
            left_pose = self._get_left_tcp_in_base(timeout_s=_eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S)
            right_pose = self._get_right_tcp_in_base(timeout_s=_eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S)
            if left_pose is None or right_pose is None:
                self._abort_slow_motion_prep("无法获取双臂 TCP（IK 目标坐标系 transform 失败）")
                return
            moving_side = goal.arm_side if goal.arm_side in ("left", "right") else "left"
            if moving_side == "right":
                start_pose, hold_pose = right_pose, left_pose
            else:
                start_pose, hold_pose = left_pose, right_pose
            prep["moving_side"] = moving_side
            prep["start_xyz"], prep["start_quat"] = start_pose
            prep["goal_xyz"] = goal.position_xyz
            prep["goal_quat"] = goal.quaternion_xyzw
            prep["hold_pose"] = hold_pose
            prep["saved_mode"] = self._control_mode
            self._slow_motion_moving_side = moving_side
            self._enter_model_mode_for_motion()
            prep["phase"] = "sync_ik"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在同步 IK 关节状态...")
            return

        if phase == "sync_ik":
            sync_future = prep.get("sync_future")
            if sync_future is None:
                if not self._ik_sync_client.wait_for_service(timeout_sec=0.0):
                    # 无 ik_node：改用宿主机 pinocchio CLIK → /wbc/arm_command
                    self.get_logger().warn(
                        "IK sync 不可用，改用宿主机 HostArmIK（无 ik_node）"
                    )
                    prep["use_host_ik"] = True
                    prep["phase"] = "host_ik"
                    self.ros_bridge.slow_motion_progress.emit(0.0, "宿主机 IK 求解中...")
                    return
                prep["sync_future"] = self._ik_sync_client.call_async(_eai_platform_bootstrap.Trigger.Request())
                prep["sync_deadline"] = _eai_platform_bootstrap.time.time() + _eai_configuration_chat.IK_SYNC_TIMEOUT_S
                return
            if not sync_future.done():  # type: ignore[union-attr]
                if _eai_platform_bootstrap.time.time() > float(prep.get("sync_deadline", 0.0)):
                    self._abort_slow_motion_prep(
                        f"IK 关节状态同步超时 ({_eai_configuration_chat.IK_SYNC_TIMEOUT_S:.1f}s)"
                    )
                return
            try:
                result = sync_future.result()  # type: ignore[union-attr]
                if not (result and result.success):
                    msg = result.message if result else "无响应"
                    self._abort_slow_motion_prep(f"IK 关节状态同步失败: {msg}")
                    return
                self.get_logger().info("IK 关节状态已同步 (/ik/sync_joint_state)")
            except Exception as exc:
                self._abort_slow_motion_prep(f"IK sync 异常: {exc}")
                return
            hold_xyz, hold_quat = prep["hold_pose"]  # type: ignore[misc]
            self._slow_motion_hold_pose = (hold_xyz, hold_quat)
            self._slow_motion_moving_side = str(prep.get("moving_side", "left"))
            self._slow_motion_final_moving_pose = (prep["goal_xyz"], prep["goal_quat"])  # type: ignore[index]
            prep["warmup_step"] = 0
            prep["phase"] = "warmup"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在预热 IK 目标...")
            return

        if phase == "host_ik":
            hold_xyz, hold_quat = prep["hold_pose"]  # type: ignore[misc]
            self._slow_motion_hold_pose = (hold_xyz, hold_quat)
            self._slow_motion_moving_side = str(prep.get("moving_side", "left"))
            goal_xyz = prep["goal_xyz"]  # type: ignore[assignment]
            goal_quat = prep["goal_quat"]  # type: ignore[assignment]
            self._slow_motion_final_moving_pose = (goal_xyz, goal_quat)  # type: ignore[arg-type]
            seed = prep.get("seed_q16")
            if not isinstance(seed, list) or len(seed) < 16:
                seed = getattr(self, "_motion_seed_override", None)
                self._motion_seed_override = None
            if not isinstance(seed, list) or len(seed) < 16:
                seed = self._load_host_fk_q16()
            if seed is None:
                self._abort_slow_motion_prep(
                    "无法读取当前关节角（请确认 host_fk 已写 eai_host_fk_state.json，或 HAL 臂关节已就绪）"
                )
                return
            try:
                from eai_app.planning.arm_ik import get_host_arm_ik

                ik = get_host_arm_ik()
                solved = ik.solve(
                    seed,
                    moving_side=str(prep.get("moving_side", "left")),
                    goal_xyz=tuple(goal_xyz),  # type: ignore[arg-type]
                    goal_xyzw=tuple(goal_quat),  # type: ignore[arg-type]
                )
            except Exception as exc:
                self._abort_slow_motion_prep(f"宿主机 IK 失败: {exc}")
                return
            if solved is None:
                info = getattr(ik, "last_info", {}) or {}
                travel = float(info.get("travel_m") or 0.0)
                pos_err = float(info.get("pos_err_m") or 0.0)
                if travel <= 0.0:
                    sx, sy, sz = prep["start_xyz"]  # type: ignore[misc]
                    gx, gy, gz = goal_xyz  # type: ignore[misc]
                    travel = ((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) ** 0.5
                self._abort_slow_motion_prep(
                    f"宿主机 IK 未收敛：目标距当前 TCP {travel:.3f} m，"
                    f"求解后位置残差 {pos_err * 1000:.0f} mm。"
                    f"目标可能在工作空间外，或请改用「相对当前」小步靠近。"
                )
                return
            prep["motion_start_joints"] = list(seed)
            prep["goal_joints"] = list(solved)
            prep["start_joints"] = list(seed)
            max_d = max(abs(solved[i] - seed[i]) for i in range(16))
            info = getattr(ik, "last_info", {}) or {}
            self.get_logger().info(
                f"宿主机 IK 成功: max|Δq|={max_d:.4f} rad，"
                f"位置残差 {float(info.get('pos_err_m', 0.0)) * 1000:.1f} mm → 关节轨迹 "
                f"(bridge 转发 /wbc/arm_command)"
            )
            self._start_joint_move_from_prep(prep, goal)
            return

        if phase == "warmup":
            warmup_step = int(prep.get("warmup_step", 0))
            start_xyz, start_quat = prep["start_xyz"], prep["start_quat"]  # type: ignore[misc]
            self._publish_slow_motion_targets(start_xyz, start_quat)
            warmup_step += 1
            prep["warmup_step"] = warmup_step
            if warmup_step >= _eai_configuration_chat.ARM_MOVE_WARMUP_STEPS:
                prep["phase"] = "check_wbc"
            return

        if phase == "check_wbc":
            if not self._slow_motion_wbc_changed:
                self._abort_slow_motion_prep(
                    f"IK 未驱动机械臂（{_eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC} 无变化）。"
                    "请确认 F2 已使能、ik_node/arm_control 已运行"
                )
                return
            if self._last_wbc_joints is None or len(self._last_wbc_joints) < 16:
                self._abort_slow_motion_prep("无法读取起始关节状态 (/wbc/target_joints)")
                return
            prep["start_joints"] = list(self._last_wbc_joints[:16])
            prep["goal_published"] = False
            prep["settle_ticks"] = 0
            prep["phase"] = "settle"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在规划关节轨迹...")
            return

        if phase == "settle":
            if not prep.get("goal_published"):
                prep["motion_start_joints"] = list(self._last_wbc_joints[:16])  # type: ignore[arg-type]
                goal_xyz, goal_quat = prep["goal_xyz"], prep["goal_quat"]  # type: ignore[misc]
                self._publish_slow_motion_targets(goal_xyz, goal_quat)
                prep["goal_published"] = True
                prep["settle_ticks"] = _eai_configuration_chat.ARM_MOVE_GOAL_SAMPLE_TICKS
                prep["best_goal_joints"] = list(self._last_wbc_joints[:16])
                prep["best_joint_delta"] = 0.0
                sx, sy, sz = prep["start_xyz"]  # type: ignore[misc]
                gx, gy, gz = goal_xyz
                cart = ((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) ** 0.5
                self.get_logger().info(
                    f"IK 目标已发布: start=({sx:.3f},{sy:.3f},{sz:.3f}) "
                    f"goal=({gx:.3f},{gy:.3f},{gz:.3f}) 距离={cart:.3f}m，"
                    f"等待最多 {_eai_configuration_chat.ARM_MOVE_GOAL_SAMPLE_TICKS} ticks"
                )
                return
            settle_ticks = int(prep.get("settle_ticks", 0)) - 1
            prep["settle_ticks"] = settle_ticks
            if self._last_wbc_joints is not None and len(self._last_wbc_joints) >= 16:
                motion_start: _eai_platform_bootstrap.List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
                cur = list(self._last_wbc_joints[:16])
                delta = max(abs(cur[i] - motion_start[i]) for i in range(16))
                if delta > float(prep.get("best_joint_delta", 0.0)):
                    prep["best_joint_delta"] = delta
                    prep["best_goal_joints"] = cur
                # 关节已明显变化则提前结束等待
                if delta >= _eai_configuration_chat.ARM_MOVE_GOAL_MIN_JOINT_DELTA and settle_ticks > 2:
                    settle_ticks = 0
                    prep["settle_ticks"] = 0
            if settle_ticks > 0:
                return
            if self._last_wbc_joints is None or len(self._last_wbc_joints) < 16:
                self._abort_slow_motion_prep("IK 未返回目标关节角")
                return
            best = prep.get("best_goal_joints")
            prep["goal_joints"] = (
                list(best) if isinstance(best, list) and len(best) >= 16
                else list(self._last_wbc_joints[:16])
            )
            prep["handoff_tick"] = 0
            prep["phase"] = "handoff"
            self.ros_bridge.slow_motion_progress.emit(0.0, "规划完成，释放 IK 控制...")
            return

        if phase == "handoff":
            handoff_tick = int(prep.get("handoff_tick", 0))
            if handoff_tick == 0:
                motion_start: _eai_platform_bootstrap.List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
                self._publish_wbc_joint_target(motion_start)
                self._release_ik_for_joint_move()
                prep["handoff_tick"] = 1
                prep["handoff_wait"] = _eai_configuration_chat.ARM_MOVE_IK_RELEASE_TICKS
                return
            wait = int(prep.get("handoff_wait", 0)) - 1
            prep["handoff_wait"] = wait
            if wait > 0:
                return
            self._start_joint_move_from_prep(prep, goal)
            return

    def _sync_ik_joint_state_blocking(self, timeout_s: float = _eai_configuration_chat.IK_SYNC_TIMEOUT_S) -> bool:
        if not self._ik_sync_client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().warn("IK sync 服务不可用，跳过 /ik/sync_joint_state")
            return False
        done = _eai_platform_bootstrap.threading.Event()
        ok_box: _eai_platform_bootstrap.List[bool] = [False]

        def _done(fut) -> None:
            try:
                result = fut.result()
                ok_box[0] = bool(result and result.success)
                if ok_box[0]:
                    self.get_logger().info("IK 关节状态已同步 (/ik/sync_joint_state)")
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"IK sync 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"IK sync 异常: {exc}")
            finally:
                done.set()

        future = self._ik_sync_client.call_async(_eai_platform_bootstrap.Trigger.Request())
        future.add_done_callback(_done)
        if not done.wait(timeout_s):
            self.get_logger().warn(f"IK sync 超时 ({timeout_s:.1f}s)")
            return False
        return ok_box[0]

    def _try_sync_ik_joint_state(self) -> None:
        self._sync_ik_joint_state_blocking()

    def _enter_model_mode_for_motion(self) -> None:
        self._slow_motion_saved_control_mode = self._control_mode
        side_name = _eai_perception_pose_estimation.arm_side_label(self._slow_motion_moving_side)
        if self._control_mode != _eai_configuration_chat.MODEL_CONTROL_MODE:
            self.get_logger().info(
                f"{side_name}移动: 切换 control_mode {self._slow_motion_saved_control_mode} -> "
                f"{_eai_configuration_chat.MODEL_CONTROL_MODE}（暂停 tracker→IK 转发，避免覆盖目标）"
            )
        self._start_slow_motion_mode_lock()

    def _burst_control_mode(self, mode: int, count: int = _eai_configuration_chat.ARM_MOVE_MODE_BURST) -> None:
        for _ in range(count):
            self._publish_control_mode(mode)

    def _stop_slow_motion_timer(self) -> None:
        if self._slow_motion_timer is not None:
            self.destroy_timer(self._slow_motion_timer)
            self._slow_motion_timer = None

    def _finish_slow_motion(self, success: bool, message: str) -> None:
        self._slow_motion_active = False
        self._slow_motion_phase = ""
        self._slow_motion_poses = []
        self._slow_motion_joint_poses = []
        self._slow_motion_step = 0
        self._slow_motion_hold_steps = 0
        self._slow_motion_hold_pose = None
        self._slow_motion_final_moving_pose = None
        self._slow_motion_final_joints = None
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._stop_slow_motion_timer()
        self._stop_slow_motion_mode_timer()
        self._restore_control_mode_if_needed()
        self.ros_bridge.slow_motion_finished.emit(success, message)

    def cancel_slow_motion(self) -> str:
        if self._slow_motion_preparing:
            self._abort_slow_motion_prep("移动准备已取消")
            return "移动准备已取消"
        if not self._slow_motion_active:
            return "当前无进行中的移动"
        progress = 0.0
        if self._slow_motion_joint_poses:
            progress = self._slow_motion_step / len(self._slow_motion_joint_poses)
        self.get_logger().info(
            f"移动已取消 ({progress * 100:.0f}%) -> {_eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC}"
        )
        self._finish_slow_motion(False, f"移动已取消 ({progress * 100:.0f}%)")
        return f"移动已取消 ({progress * 100:.0f}%)"

    def _slow_motion_tick(self) -> None:
        if not self._slow_motion_active:
            return

        if self._slow_motion_step >= len(self._slow_motion_joint_poses):
            final_joints = self._slow_motion_final_joints
            if final_joints is not None:
                self._publish_wbc_joint_target(final_joints)
            self.get_logger().info(f"移动完成 -> {_eai_configuration_policies.WBC_TARGET_JOINTS_TOPIC}")
            self._finish_slow_motion(True, "移动完成")
            return

        joints = self._slow_motion_joint_poses[self._slow_motion_step]
        self._publish_wbc_joint_target(joints)
        self._slow_motion_step += 1
        progress = self._slow_motion_step / len(self._slow_motion_joint_poses)
        self.ros_bridge.slow_motion_progress.emit(
            progress,
            f"移动中 {progress * 100:.0f}%",
        )

    def apply_left_hand_angle(self, slider: int) -> float:
        """按指间距离游标发送左手命令，返回关节 position（未使能则不发）。"""
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
        pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
        msg = _eai_adapters_robot_geometry.make_left_hand_command(pos)
        msg.header.stamp = self.get_clock().now().to_msg()
        if not self._publish_hand_command("left", msg):
            return pos
        self.get_logger().info(
            f"左手命令 -> gap={gap_cm:.1f}cm position={pos:.3f}: "
            f"{_eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC} + {_eai_configuration_policies.WBC_LEFT_EE_CMD_TOPIC}"
        )
        return pos

    def apply_right_hand_angle(self, slider: int) -> float:
        """按指间距离游标发送右手命令，返回关节 position（未使能则不发）。"""
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
        pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
        msg = _eai_adapters_robot_geometry.make_right_hand_command(pos)
        msg.header.stamp = self.get_clock().now().to_msg()
        if not self._publish_hand_command("right", msg):
            return pos
        self.get_logger().info(
            f"右手命令 -> gap={gap_cm:.1f}cm position={pos:.3f}: "
            f"{_eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC} + {_eai_configuration_policies.WBC_RIGHT_EE_CMD_TOPIC}"
        )
        return pos

    def apply_hand_joint_positions(
        self, side: str, positions: _eai_platform_bootstrap.Sequence[float]
    ) -> _eai_platform_bootstrap.List[float]:
        """按 6 关节开合量发送手部命令，返回裁剪后的 position 列表。"""
        side = "right" if side == "right" else "left"
        msg = _eai_adapters_robot_geometry.make_hand_command(side, positions)
        msg.header.stamp = self.get_clock().now().to_msg()
        if not self._publish_hand_command(side, msg):
            return [float(v) for v in msg.position]
        vals = [float(v) for v in msg.position]
        topic = (
            f"{_eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC}+{_eai_configuration_policies.WBC_RIGHT_EE_CMD_TOPIC}"
            if side == "right"
            else f"{_eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC}+{_eai_configuration_policies.WBC_LEFT_EE_CMD_TOPIC}"
        )
        self.get_logger().debug(
            f"{'右' if side == 'right' else '左'}手关节命令 {vals} -> {topic}"
        )
        return vals

    def toggle_left_hand_between(self, slider_a: int, slider_b: int) -> _eai_platform_bootstrap.Tuple[bool, float]:
        """在游标 A/B 两个状态间切换，返回 (当前是否为 A, 发送的 position)。"""
        self._left_hand_at_a = not self._left_hand_at_a
        slider = slider_a if self._left_hand_at_a else slider_b
        preset = "A" if self._left_hand_at_a else "B"
        pos = self.apply_left_hand_angle(slider)
        self.get_logger().info(f"左手切换 -> 状态{preset}")
        self.ros_bridge.left_hand_preset_changed.emit(self._left_hand_at_a)
        return self._left_hand_at_a, pos

    def toggle_right_hand_between(self, slider_a: int, slider_b: int) -> _eai_platform_bootstrap.Tuple[bool, float]:
        """在游标 A/B 两个状态间切换，返回 (当前是否为 A, 发送的 position)。"""
        self._right_hand_at_a = not self._right_hand_at_a
        slider = slider_a if self._right_hand_at_a else slider_b
        preset = "A" if self._right_hand_at_a else "B"
        pos = self.apply_right_hand_angle(slider)
        self.get_logger().info(f"右手切换 -> 状态{preset}")
        self.ros_bridge.right_hand_preset_changed.emit(self._right_hand_at_a)
        return self._right_hand_at_a, pos

    def ensure_topic_scan(self) -> None:
        """大脑页「刷新 Topic」：启动扫描定时器并立刻扫一次。

        get_topic_names_and_types 在 FastDDS 下可卡数秒，禁止在 Qt 线程同步调用；
        一律丢到 ROS spin 线程执行。
        """
        def _do() -> None:
            if self._scan_timer is None:
                self._scan_timer = self.create_timer(2.0, self._on_timer)
                self._topic_scan_ready = True
                self.get_logger().info("已启动相机 topic 扫描定时器")
            self._topic_graph_cache = None
            self._topic_graph_cache_at = 0.0
            self._on_timer()

        self.enqueue_ros_op(_do)

    def _robot_fast_qos(self, *, reliable: bool = False) -> _eai_platform_bootstrap.QoSProfile:
        return _eai_platform_bootstrap.QoSProfile(
            reliability=(
                _eai_platform_bootstrap.ReliabilityPolicy.RELIABLE if reliable else _eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT
            ),
            history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
        )

    def _create_robot_sub_fast(self, msg_type, topic: str, callback, *, reliable: bool = False):
        """单次 create_subscription，不查 publisher、不 destroy（destroy 易占 GIL 卡 UI）。"""
        if topic in self._robot_subscriptions_map:
            return self._robot_subscriptions_map[topic]
        qos = self._robot_fast_qos(reliable=reliable)
        t0 = _eai_platform_bootstrap.time.perf_counter()
        sub = self.create_subscription(msg_type, topic, callback, qos)
        dt = _eai_platform_bootstrap.time.perf_counter() - t0
        self._robot_subscriptions_map[topic] = sub
        self._robot_subscribe_attempts[topic] = 0
        self._robot_last_subscribe_time[topic] = _eai_platform_bootstrap.time.time()
        if dt > 0.05:
            try:
                self.get_logger().warn(
                    f"create_subscription 慢: {topic} 耗时 {dt:.2f}s"
                )
            except Exception:
                pass
        return sub

    def enqueue_ros_op(self, fn: _eai_platform_bootstrap.Callable[[], None]) -> None:
        """把操作丢到 ROS spin 线程执行，避免阻塞 Qt UI。"""
        try:
            self._ros_op_queue.put_nowait(fn)
        except Exception:
            pass

    def drain_ros_ops(self, *, max_ops: int = 2) -> int:
        """由 ROS spin 线程调用：每次最多执行 max_ops 个排队操作。"""
        n = 0
        while n < max_ops:
            try:
                fn = self._ros_op_queue.get_nowait()
            except _eai_platform_bootstrap.queue.Empty:
                break
            try:
                fn()
            except Exception as exc:
                try:
                    self.get_logger().error(f"ROS 排队操作失败: {exc}")
                except Exception:
                    pass
            n += 1
        return n

    def begin_robot_subscription_steps(self) -> int:
        """点击连接：本进程不订关节/TCP（GIL）；使能状态轻量订阅。

        TCP 由独立 robot_tcp_bridge 进程订阅，UI 轮询 JSON。
        """
        if self._robot_subs_ready:
            self._robot_connect_steps = []
            self._robot_connect_step_i = 0
            self._robot_connect_running = False
            self.ensure_enable_state_subscriptions()
            return 0
        self._allow_robot_qos_probe = False
        # 连接后几秒内抑制 robot_state UI 刷新，避免 TCP 一到就刷标签/按钮
        self._robot_ui_quiet_until = _eai_platform_bootstrap.time.time() + 2.0
        self._robot_extras_due_at = 0.0  # 禁止空闲补订关节
        self._robot_connect_steps = []
        self._robot_connect_step_i = 0
        self._robot_connect_running = False
        self._robot_connect_label = "已连接"
        self._robot_connect_error = ""
        self._robot_subs_ready = True
        self._pending_head_camera_tf = False
        self._robot_use_tcp_bridge = True
        self.ensure_enable_state_subscriptions()
        self.get_logger().info(
            "机器人状态已连接（TCP 走独立 bridge；使能走本进程轻量订阅）"
        )
        return 0

    def clear_robot_subscriptions(self) -> None:
        """取消连接：销毁本进程机器人相关订阅，并复位连接标志。

        TCP bridge 由 UI 侧 `_stop_robot_tcp_bridge` 停止。
        """
        self._robot_connect_running = False
        self._robot_connect_steps = []
        self._robot_connect_step_i = 0
        self._robot_connect_label = ""
        self._robot_connect_error = ""
        self._robot_subs_ready = False
        self._robot_use_tcp_bridge = False
        self._pending_head_camera_tf = False
        self._robot_extras_due_at = 0.0
        self._robot_extra_step_list = None
        self._robot_extra_step_i = 0
        self._allow_robot_qos_probe = False

        topics = list(self._robot_subscriptions_map.keys())

        def _destroy_all() -> None:
            for topic in topics:
                sub = self._robot_subscriptions_map.pop(topic, None)
                if sub is None:
                    continue
                try:
                    self.destroy_subscription(sub)
                except Exception:
                    pass
            self.get_logger().info("已取消机器人状态连接（订阅已清理）")

        if topics:
            self.enqueue_ros_op(_destroy_all)
        else:
            self.get_logger().info("已取消机器人状态连接")

        with self._robot_lock:
            self._robot_state = _eai_adapters_robot_geometry.RobotStateSnapshot(
                [], [], [], [], [], None, None, 0.0
            )
        try:
            self._touch_robot_state()
        except Exception:
            pass

