"""adapters.ros.node parts.feedback apply bridged tcp state — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class FeedbackApplyBridgedTcpStateMixin:
    def apply_bridged_tcp_state(self, data: dict) -> None:
        """从 bridge JSON 更新左右 TCP（UI 定时器调用，不碰 DDS）。"""
        if not data:
            return
        changed = False
        for side, key in (("left", "left"), ("right", "right")):
            pose = data.get(key)
            if not isinstance(pose, dict):
                continue
            xyz = pose.get("xyz") or []
            quat = pose.get("xyzw") or []
            if len(xyz) < 3 or len(quat) < 4:
                continue
            snap = _eai_adapters_robot_geometry.TcpPoseSnapshot(
                frame_id=str(pose.get("frame") or "map"),
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                quat_xyzw=(
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ),
                valid=True,
            )
            with self._robot_lock:
                if side == "left":
                    if not _eai_adapters_robot_geometry._should_accept_tcp_update(self._robot_state.left_tcp, snap):
                        continue
                    self._robot_state.left_tcp = snap
                else:
                    if not _eai_adapters_robot_geometry._should_accept_tcp_update(self._robot_state.right_tcp, snap):
                        continue
                    self._robot_state.right_tcp = snap
                changed = True
        if changed:
            self._touch_robot_state()

    def _ros_robot_connect_one_step(self) -> None:
        steps = getattr(self, "_robot_connect_steps", None) or []
        total = len(steps)
        i = int(getattr(self, "_robot_connect_step_i", 0) or 0)
        if self._robot_subs_ready or i >= total:
            self._robot_subs_ready = True
            self._robot_connect_running = False
            self._pending_head_camera_tf = False
            self._robot_connect_label = "已连接"
            return
        label, fn = steps[i]
        self._robot_connect_label = label
        try:
            fn()
        except Exception as exc:
            self._robot_connect_error = f"{label}: {exc}"
            self.get_logger().error(f"订阅步骤 {label} 失败: {exc}")
        # 主动让出 GIL，避免连续 create_subscription 把 Qt 饿死
        _eai_platform_bootstrap.time.sleep(0.08)
        self._robot_connect_step_i = i + 1
        if self._robot_connect_step_i >= total:
            self._robot_subs_ready = True
            self._robot_connect_running = False
            # 默认不自动挂 head TF（很重）；需要相机叠加时再懒挂
            self._pending_head_camera_tf = False
            self._robot_connect_label = "已连接"
            self.get_logger().info(
                "已分步订阅机器人核心状态: left/right tcp（未订关节）"
            )
            # 关节/手/使能等按需延后；默认更晚，减少点击后卡顿
            self._robot_extras_due_at = _eai_platform_bootstrap.time.time() + 3.0
        else:
            self.enqueue_ros_op(self._ros_robot_connect_one_step)

    def _robot_extra_steps(self) -> list:
        hand_qos = self._robot_fast_qos(reliable=False)
        enable_qos = self._robot_fast_qos(reliable=True)

        def _hand(topic, cb):
            if topic in self._robot_subscriptions_map:
                return
            sub = self.create_subscription(_eai_platform_bootstrap.JointState, topic, cb, hand_qos)
            self._robot_subscriptions_map[topic] = sub

        def _en(topic, cb):
            if topic in self._robot_subscriptions_map:
                return
            sub = self.create_subscription(_eai_platform_bootstrap.UInt32, topic, cb, enable_qos)
            self._robot_subscriptions_map[topic] = sub

        return [
            ("arm_joints", lambda: self._create_robot_sub_fast(
                _eai_platform_bootstrap.JointState, _eai_configuration_policies.ROBOT_ARM_TOPIC, self._on_arm_joint_state
            )),
            ("fk_joints", lambda: self._create_robot_sub_fast(
                _eai_platform_bootstrap.JointState, _eai_configuration_policies.ROBOT_FK_JOINTS_TOPIC, self._on_fk_joint_state
            )),
            ("left_hand", lambda: _hand(_eai_configuration_policies.ROBOT_LEFT_HAND_TOPIC, self._on_left_hand_state)),
            ("right_hand", lambda: _hand(_eai_configuration_policies.ROBOT_RIGHT_HAND_TOPIC, self._on_right_hand_state)),
            ("control_mode", lambda: self._create_robot_sub_fast(
                _eai_platform_bootstrap.UInt32, _eai_configuration_policies.CONTROL_MODE_TOPIC, self._on_control_mode, reliable=True
            )),
            ("arm_enable", lambda: _en(_eai_configuration_policies.ARM_ENABLE_STATE_TOPIC, self._on_arm_enable_state)),
            ("hand_enable", lambda: _en(_eai_configuration_policies.HAND_ENABLE_STATE_TOPIC, self._on_hand_enable_state)),
            ("replay", lambda: (
                None
                if _eai_configuration_policies.REPLAY_RUNNING_STATE_TOPIC in self._robot_subscriptions_map
                else self._robot_subscriptions_map.__setitem__(
                    _eai_configuration_policies.REPLAY_RUNNING_STATE_TOPIC,
                    self.create_subscription(
                        _eai_platform_bootstrap.UInt8,
                        _eai_configuration_policies.REPLAY_RUNNING_STATE_TOPIC,
                        self._on_replay_running_state,
                        self._robot_fast_qos(reliable=True),
                    ),
                )
            )),
        ]

    def maybe_pump_robot_extras(self) -> None:
        """空闲时每次最多补订 1 个次要 topic（由 ROS spin 调用）。"""
        due = float(getattr(self, "_robot_extras_due_at", 0.0) or 0.0)
        if due <= 0 or _eai_platform_bootstrap.time.time() < due:
            return
        if getattr(self, "_robot_connect_running", False):
            return
        steps = getattr(self, "_robot_extra_step_list", None)
        if steps is None:
            steps = self._robot_extra_steps()
            self._robot_extra_step_list = steps
            self._robot_extra_step_i = 0
        i = int(getattr(self, "_robot_extra_step_i", 0) or 0)
        if i >= len(steps):
            self._robot_extras_due_at = 0.0
            return
        label, fn = steps[i]
        try:
            fn()
        except Exception as exc:
            self.get_logger().warn(f"空闲补订 {label} 失败: {exc}")
        self._robot_extra_step_i = i + 1
        # 下一补订至少隔 0.4s，给 Qt 留 GIL
        self._robot_extras_due_at = _eai_platform_bootstrap.time.time() + 0.4

    def robot_connect_progress(self) -> tuple:
        """UI 轮询：返回 (done, label, index, total, error)。"""
        steps = getattr(self, "_robot_connect_steps", None) or []
        total = len(steps)
        i = int(getattr(self, "_robot_connect_step_i", 0) or 0)
        label = str(getattr(self, "_robot_connect_label", "") or "")
        err = str(getattr(self, "_robot_connect_error", "") or "")
        done = bool(self._robot_subs_ready) and not bool(
            getattr(self, "_robot_connect_running", False)
        )
        if done and total:
            i = total
            label = "已连接"
        return done, label, i, total, err

    def step_robot_subscriptions(self) -> tuple:
        """兼容：仅读进度，不在 UI 线程执行 create_subscription。"""
        return self.robot_connect_progress()

    def ensure_robot_subscriptions(self) -> str:
        """兼容旧调用：启动异步连接并忙等（尽量别从 UI 调）。"""
        if self._robot_subs_ready:
            return "机器人状态已连接"
        n = self.begin_robot_subscription_steps()
        if n <= 0:
            return "机器人状态已连接"
        deadline = _eai_platform_bootstrap.time.time() + 30.0
        while _eai_platform_bootstrap.time.time() < deadline:
            self.drain_ros_ops(max_ops=4)
            done, label, i, total, err = self.robot_connect_progress()
            if err and done:
                return f"订阅失败({label}): {err}"
            if done:
                return "已订阅机器人状态（臂/手/TCP/使能/回放）"
            _eai_platform_bootstrap.time.sleep(0.02)
        return "订阅超时"

    def ensure_head_camera_tf_lazy(self) -> None:
        """空闲时补挂 head camera TF（投递到 ROS 线程，不挡 UI）。"""
        if not getattr(self, "_pending_head_camera_tf", False):
            return
        if self._head_camera_tf is not None:
            self._pending_head_camera_tf = False
            return
        if _eai_configuration_perception.attach_head_camera_tf is None:
            self._pending_head_camera_tf = False
            return
        self._pending_head_camera_tf = False
        self.enqueue_ros_op(self._ros_attach_head_camera_tf)

    def _ros_attach_head_camera_tf(self) -> None:
        if self._head_camera_tf is not None:
            return
        if _eai_configuration_perception.attach_head_camera_tf is None:
            return
        try:
            self._head_camera_tf = _eai_configuration_perception.attach_head_camera_tf(self)
        except Exception as exc:
            self.get_logger().warn(f"延迟挂载 head camera TF 失败: {exc}")

    def robot_subscriptions_ready(self) -> bool:
        return bool(self._robot_subs_ready)

    def _setup_robot_subscriptions(self) -> None:
        hand_qos = _eai_platform_bootstrap.QoSProfile(
            reliability=_eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT,
            history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
        )
        for topic in (_eai_configuration_policies.ROBOT_ARM_TOPIC, _eai_configuration_policies.ROBOT_FK_JOINTS_TOPIC):
            self._subscribe_robot_topic(
                _eai_platform_bootstrap.JointState,
                topic,
                self._on_arm_joint_state if topic == _eai_configuration_policies.ROBOT_ARM_TOPIC else self._on_fk_joint_state,
            )
        self.create_subscription(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.ROBOT_LEFT_HAND_TOPIC,
            self._on_left_hand_state,
            hand_qos,
        )
        self.create_subscription(
            _eai_platform_bootstrap.JointState,
            _eai_configuration_policies.ROBOT_RIGHT_HAND_TOPIC,
            self._on_right_hand_state,
            hand_qos,
        )
        for side, topics in _eai_configuration_policies.ROBOT_TCP_SOURCE_TOPICS.items():
            for topic in topics:
                callback = (
                    (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                    if side == "left"
                    else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
                )
                self._subscribe_robot_topic(_eai_platform_bootstrap.PoseStamped, topic, callback)
        for topic, callback in (
            (_eai_configuration_policies.CONTROL_MODE_TOPIC, self._on_control_mode),
        ):
            self._subscribe_control_topic(_eai_platform_bootstrap.UInt32, topic, callback)
        self.create_subscription(
            _eai_platform_bootstrap.UInt32,
            _eai_configuration_policies.ARM_ENABLE_STATE_TOPIC,
            self._on_arm_enable_state,
            _eai_configuration_perception.ENABLE_STATE_QOS,
        )
        self.create_subscription(
            _eai_platform_bootstrap.UInt32,
            _eai_configuration_policies.HAND_ENABLE_STATE_TOPIC,
            self._on_hand_enable_state,
            _eai_configuration_perception.ENABLE_STATE_QOS,
        )
        self.create_subscription(
            _eai_platform_bootstrap.UInt8,
            _eai_configuration_policies.REPLAY_RUNNING_STATE_TOPIC,
            self._on_replay_running_state,
            _eai_platform_bootstrap.QoSProfile(
                reliability=_eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
            ),
        )
        self.get_logger().info(
            "已订阅机器人状态: arm/hand/tcp/control_mode/replay topics "
            f"(TCP: {_eai_configuration_policies.ROBOT_LEFT_TCP_TOPIC} / {_eai_configuration_policies.ROBOT_TELE_LEFT_TCP_TOPIC})"
        )

    def _robot_qos_candidates_for_topic(self, topic: str) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.QoSProfile]:
        reliable = _eai_platform_bootstrap.QoSProfile(
            reliability=_eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
            history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
        )
        best_effort = _eai_platform_bootstrap.QoSProfile(
            reliability=_eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT,
            history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
        )
        # 默认不查 publisher QoS：get_publishers_info_by_topic 在 FastDDS 下可卡 UI 数秒
        if not getattr(self, "_allow_robot_qos_probe", False):
            return [best_effort, reliable]
        derived = self._qos_from_publisher_info(topic)
        if derived is not None:
            return [derived, reliable, best_effort]
        return [reliable, best_effort]

    def _control_qos_candidates_for_topic(self, topic: str) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.QoSProfile]:
        candidates: _eai_platform_bootstrap.List[_eai_platform_bootstrap.QoSProfile] = []
        if getattr(self, "_allow_robot_qos_probe", False):
            derived = self._qos_from_publisher_info(topic)
            if derived is not None:
                candidates.append(derived)
        candidates.extend([
            _eai_platform_bootstrap.QoSProfile(
                reliability=_eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
            ),
            _eai_platform_bootstrap.QoSProfile(
                reliability=_eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
            ),
        ])
        return candidates

    def _robot_qos_for_topic(self, topic: str, attempt: int = 0) -> _eai_platform_bootstrap.QoSProfile:
        candidates = self._robot_qos_candidates_for_topic(topic)
        qos = candidates[min(attempt, len(candidates) - 1)]
        self.get_logger().debug(
            f"{topic}: robot QoS attempt={attempt} reliability={qos.reliability.name}"
        )
        return qos

    def _subscribe_robot_topic(
        self,
        msg_type: type,
        topic: str,
        callback,
        attempt: int = 0,
    ) -> None:
        if topic in self._robot_subscriptions_map:
            try:
                self.destroy_subscription(self._robot_subscriptions_map.pop(topic))
            except Exception:
                pass
        qos = self._robot_qos_for_topic(topic, attempt)
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self._robot_subscriptions_map[topic] = sub
        self._robot_subscribe_attempts[topic] = attempt
        self._robot_last_subscribe_time[topic] = _eai_platform_bootstrap.time.time()

    def _subscribe_control_topic(
        self,
        msg_type: type,
        topic: str,
        callback,
        attempt: int = 0,
    ) -> None:
        if topic in self._robot_subscriptions_map:
            try:
                self.destroy_subscription(self._robot_subscriptions_map.pop(topic))
            except Exception:
                pass
        candidates = self._control_qos_candidates_for_topic(topic)
        qos = candidates[min(attempt, len(candidates) - 1)]
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self._robot_subscriptions_map[topic] = sub
        self._robot_subscribe_attempts[topic] = attempt
        self._robot_last_subscribe_time[topic] = _eai_platform_bootstrap.time.time()

    def _cached_topic_graph(self, *, max_age_s: float = 2.0) -> _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]]:
        now = _eai_platform_bootstrap.time.time()
        if (
            self._topic_graph_cache is not None
            and now - self._topic_graph_cache_at < max_age_s
        ):
            return self._topic_graph_cache
        graph = dict(self.get_topic_names_and_types())
        self._topic_graph_cache = graph
        self._topic_graph_cache_at = now
        return graph

    def _side_has_tcp(self, side: str) -> bool:
        with self._robot_lock:
            tcp = self._robot_state.left_tcp if side == "left" else self._robot_state.right_tcp
            return tcp is not None and bool(tcp.valid)

    def _maybe_retry_robot_subscriptions(self) -> None:
        """重试 TCP 订阅；默认关闭自动重订，避免启动刷屏/卡顿。

        设 EAI_RETRY_TCP_SUB=1 可恢复慢速重试（仍会放弃幽灵 /tele/fk）。
        """
        if not self._robot_subs_ready:
            return
        if _eai_platform_bootstrap.os.environ.get("EAI_RETRY_TCP_SUB", "0").strip().lower() in (
            "0",
            "false",
            "off",
            "no",
            "",
        ):
            return
        now = _eai_platform_bootstrap.time.time()
        all_topics = self._cached_topic_graph()
        # mink_fk 优先；tele/fk 常有空 publisher（ik 建了端点却不发），避免无限重订卡 UI
        retry_specs = [
            (_eai_configuration_policies.ROBOT_LEFT_TCP_TOPIC, _eai_platform_bootstrap.PoseStamped, "left"),
            (_eai_configuration_policies.ROBOT_RIGHT_TCP_TOPIC, _eai_platform_bootstrap.PoseStamped, "right"),
            (_eai_configuration_policies.ROBOT_TELE_LEFT_TCP_TOPIC, _eai_platform_bootstrap.PoseStamped, "left"),
            (_eai_configuration_policies.ROBOT_TELE_RIGHT_TCP_TOPIC, _eai_platform_bootstrap.PoseStamped, "right"),
        ]
        for topic, msg_type, side in retry_specs:
            if topic not in all_topics:
                continue
            if self._robot_tcp_topic_received.get(topic, False):
                continue
            # 该侧已有 TCP（例如 mink_fk）则不必再刷 tele/fk
            if self._side_has_tcp(side):
                continue
            if self._robot_tcp_give_up.get(topic, False):
                continue
            if topic not in self._robot_subscriptions_map:
                callback = (
                    (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                    if side == "left"
                    else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
                )
                self.get_logger().info(f"{topic}: 检测到发布，补订阅 TCP")
                self._subscribe_robot_topic(msg_type, topic, callback, attempt=0)
                continue
            retries = self._robot_resubscribe_count.get(topic, 0)
            candidates = self._robot_qos_candidates_for_topic(topic)
            last_sub = self._robot_last_subscribe_time.get(topic, 0.0)
            # 整轮 QoS 试完后：mink 再慢周期试一次；tele/fk 直接放弃（幽灵端点）
            if retries >= len(candidates):
                if topic in (_eai_configuration_policies.ROBOT_TELE_LEFT_TCP_TOPIC, _eai_configuration_policies.ROBOT_TELE_RIGHT_TCP_TOPIC):
                    self._robot_tcp_give_up[topic] = True
                    self.get_logger().warn(
                        f"{topic}: 有 publisher 但无数据，停止重订阅（改用 /mink_fk）"
                    )
                    continue
                if now - last_sub < 30.0:
                    continue
                # mink_fk：最多再完整轮询 1 次后放弃，避免卡死 UI
                cycles = int(self._robot_resubscribe_count.get(f"{topic}__cycles", 0))
                if cycles >= 1:
                    self._robot_tcp_give_up[topic] = True
                    self.get_logger().warn(
                        f"{topic}: 多次未收到 TCP，停止重订阅（请确认 fk_node）"
                    )
                    continue
                self._robot_resubscribe_count[f"{topic}__cycles"] = cycles + 1
                self._robot_resubscribe_count[topic] = 0
                retries = 0
            elif now - last_sub < 5.0:
                continue
            callback = (
                (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                if side == "left"
                else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
            )
            next_attempt = retries + 1
            self.get_logger().warn(
                f"{topic}: 未收到 TCP 数据，切换 QoS 重订阅 "
                f"({next_attempt}/{len(candidates)})"
            )
            self._robot_resubscribe_count[topic] = next_attempt
            self._subscribe_robot_topic(msg_type, topic, callback, attempt=next_attempt)

    def get_robot_state(self) -> _eai_adapters_robot_geometry.RobotStateSnapshot:
        with self._robot_lock:
            return _eai_adapters_robot_geometry.RobotStateSnapshot(
                left_arm=list(self._robot_state.left_arm),
                right_arm=list(self._robot_state.right_arm),
                left_hand=list(self._robot_state.left_hand),
                right_hand=list(self._robot_state.right_hand),
                waist=list(self._robot_state.waist),
                left_tcp=self._robot_state.left_tcp,
                right_tcp=self._robot_state.right_tcp,
                updated_at=self._robot_state.updated_at,
            )

    def get_grasp_hand_feedback(self, side: str) -> dict:
        """Read measured hand positions and their receipt time atomically."""
        if side not in ('left', 'right'):
            raise ValueError('未知手臂')
        with self._robot_lock:
            values = {_eai_platform_bootstrap.canonical_hand_joint(name): float(value)
                      for name, value in getattr(self._robot_state, side+'_hand')}
            stamp = getattr(self, '_grasp_hand_state_received_at', {}).get(side, 0.)
            if any(name not in values for name in _eai_platform_bootstrap.HAND_JOINT_ORDER):
                raise ValueError('手部实际反馈不完整')
            return {'positions': [values[name] for name in _eai_platform_bootstrap.HAND_JOINT_ORDER], 'stamp': stamp}

    def get_camera_frame_id(self, topic: str) -> str:
        return self._camera_frame_ids.get(topic, "")

    def resolve_segment_camera_frame(
        self, source_topic: str, depth_topic: _eai_platform_bootstrap.Optional[str] = None
    ) -> str:
        known = list(self._camera_frame_ids.keys())
        candidates: _eai_platform_bootstrap.List[str] = []
        if depth_topic:
            candidates.append(depth_topic)
        if source_topic not in candidates:
            candidates.append(source_topic)
        if _eai_perception_image_geometry.is_depth_topic(source_topic):
            color = _eai_perception_image_geometry.find_paired_color_topic(source_topic, known)
            if color and color not in candidates:
                candidates.append(color)
        else:
            paired_depth = depth_topic or _eai_perception_image_geometry.find_paired_depth_topic(source_topic, known)
            if paired_depth and paired_depth not in candidates:
                candidates.insert(0, paired_depth)
        for topic in candidates:
            frame = self._camera_frame_ids.get(topic, "").strip()
            if frame:
                return frame
        return ""

    def get_tf_buffer(self) -> _eai_platform_bootstrap.Buffer:
        return self._tf_buffer

    def _touch_robot_state(self) -> None:
        self._robot_state.updated_at = _eai_platform_bootstrap.time.time()
        now = _eai_platform_bootstrap.time.time()
        quiet_until = float(getattr(self, "_robot_ui_quiet_until", 0.0) or 0.0)
        if quiet_until and now < quiet_until:
            return
        if now - self._last_robot_state_emit < _eai_configuration_policies.UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._last_robot_state_emit = now
        self.ros_bridge.robot_state_updated.emit()

    def _throttle_robot_cb(self, key: str, min_interval: float = 0.1) -> bool:
        """高频 joint 回调入口限频；True=应处理。避免占满 GIL 导致 Qt 假死。"""
        now = _eai_platform_bootstrap.time.time()
        last_map = getattr(self, "_robot_cb_last", None)
        if last_map is None:
            last_map = {}
            self._robot_cb_last = last_map
        if now - float(last_map.get(key, 0.0) or 0.0) < min_interval:
            return False
        last_map[key] = now
        return True

    def _on_arm_joint_state(self, msg: _eai_platform_bootstrap.JointState) -> None:
        first = self._hal_arm_received_at <= 0
        if not first and not self._throttle_robot_cb("arm", 0.1):
            return
        positions = [float(v) for v in msg.position]
        if first:
            self.get_logger().info(f"HAL 关节状态就绪: {_eai_configuration_policies.ROBOT_ARM_TOPIC}")
        self._hal_arm_received_at = _eai_platform_bootstrap.time.time()
        left, right = _eai_adapters_robot_geometry._arm_from_raw_positions(list(msg.name), positions)
        with self._robot_lock:
            if left:
                self._robot_state.left_arm = left
            if right:
                self._robot_state.right_arm = right
        self._touch_robot_state()

    def _on_fk_joint_state(self, msg: _eai_platform_bootstrap.JointState) -> None:
        if not self._throttle_robot_cb("fk", 0.1):
            return
        joints = _eai_adapters_robot_geometry._joint_dict_from_msg(msg)
        left, right, waist = _eai_adapters_robot_geometry._split_arm_joints(joints)
        with self._robot_lock:
            if left:
                self._robot_state.left_arm = left
            if right:
                self._robot_state.right_arm = right
            if waist:
                self._robot_state.waist = waist
        self._touch_robot_state()

    def _on_left_hand_state(self, msg: _eai_platform_bootstrap.JointState) -> None:
        if not self._throttle_robot_cb("left_hand", 0.1):
            return
        hand = [(n, float(msg.position[i])) for i, n in enumerate(msg.name) if i < len(msg.position)]
        with self._robot_lock:
            self._robot_state.left_hand = hand
            if not hasattr(self, '_grasp_hand_state_received_at'):
                self._grasp_hand_state_received_at = {}
            self._grasp_hand_state_received_at['left'] = _eai_platform_bootstrap.time.time()
        self._touch_robot_state()

    def _on_right_hand_state(self, msg: _eai_platform_bootstrap.JointState) -> None:
        if not self._throttle_robot_cb("right_hand", 0.1):
            return
        hand = [(n, float(msg.position[i])) for i, n in enumerate(msg.name) if i < len(msg.position)]
        with self._robot_lock:
            self._robot_state.right_hand = hand
            if not hasattr(self, '_grasp_hand_state_received_at'):
                self._grasp_hand_state_received_at = {}
            self._grasp_hand_state_received_at['right'] = _eai_platform_bootstrap.time.time()
        self._touch_robot_state()

    def _on_left_tcp_pose(self, msg: _eai_platform_bootstrap.PoseStamped, source_topic: str = _eai_configuration_policies.ROBOT_LEFT_TCP_TOPIC) -> None:
        now = _eai_platform_bootstrap.time.time()
        first = not self._robot_tcp_topic_received.get(source_topic, False)
        if (
            not first
            and now - self._last_tcp_accept_time.get("left", 0.0) < self._tcp_min_interval_s
        ):
            return
        incoming = _eai_adapters_robot_geometry._pose_from_msg(msg)
        with self._robot_lock:
            if not _eai_adapters_robot_geometry._should_accept_tcp_update(self._robot_state.left_tcp, incoming):
                return
            self._robot_state.left_tcp = incoming
            self._last_tcp_accept_time["left"] = now
            if first:
                self._robot_tcp_topic_received[source_topic] = True
                p = msg.pose.position
                frame = msg.header.frame_id or _eai_configuration_policies.BASE_LINK_FRAME
                self.get_logger().info(
                    f"首帧左臂 TCP [{source_topic}]: "
                    f"frame={frame} XYZ=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
                )
            self._touch_robot_state()

    def _on_right_tcp_pose(self, msg: _eai_platform_bootstrap.PoseStamped, source_topic: str = _eai_configuration_policies.ROBOT_RIGHT_TCP_TOPIC) -> None:
        now = _eai_platform_bootstrap.time.time()
        first = not self._robot_tcp_topic_received.get(source_topic, False)
        if (
            not first
            and now - self._last_tcp_accept_time.get("right", 0.0) < self._tcp_min_interval_s
        ):
            return
        incoming = _eai_adapters_robot_geometry._pose_from_msg(msg)
        with self._robot_lock:
            if not _eai_adapters_robot_geometry._should_accept_tcp_update(self._robot_state.right_tcp, incoming):
                return
            self._robot_state.right_tcp = incoming
            self._last_tcp_accept_time["right"] = now
            if first:
                self._robot_tcp_topic_received[source_topic] = True
                p = msg.pose.position
                frame = msg.header.frame_id or _eai_configuration_policies.BASE_LINK_FRAME
                self.get_logger().info(
                    f"首帧右臂 TCP [{source_topic}]: "
                    f"frame={frame} XYZ=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
                )
            self._touch_robot_state()

    def _on_wbc_target_joints(self, msg: _eai_platform_bootstrap.JointState) -> None:
        positions = [float(v) for v in msg.position]
        if len(positions) < 16:
            return
        pos16 = positions[:16]
        self._last_wbc_joints = pos16
        if self._slow_motion_active:
            return
        if self._slow_motion_wbc_baseline is None:
            self._slow_motion_wbc_baseline = pos16
            return
        delta = max(abs(positions[i] - self._slow_motion_wbc_baseline[i]) for i in range(16))
        if delta > 1e-4:
            self._slow_motion_wbc_changed = True

    def _on_control_mode(self, msg: _eai_platform_bootstrap.UInt32) -> None:
        new_mode = int(msg.data)
        first_receive = self._control_mode_received_at <= 0
        changed = first_receive or (
            not self._is_motion_control_locked() and new_mode != self._control_mode
        )
        if not self._is_motion_control_locked():
            self._control_mode = new_mode
        if first_receive:
            self._control_mode_received_at = _eai_platform_bootstrap.time.time()
            self.get_logger().info(f"control_mode={self._control_mode}")
        elif changed and not self._is_motion_control_locked():
            self.get_logger().info(f"control_mode={self._control_mode}")
        if changed:
            self.ros_bridge.control_mode_changed.emit(self._control_mode)

    def _on_replay_running_state(self, msg: _eai_platform_bootstrap.UInt8) -> None:
        state = int(msg.data)
        if state == self._replay_running_state:
            return
        self._replay_running_state = state
        label = _eai_configuration_policies.REPLAY_STATE_LABELS.get(state, str(state))
        self.get_logger().info(f"replay running_state={state} ({label})")
        self.ros_bridge.replay_state_changed.emit(state)

    def set_prefix(self, prefix: str) -> None:
        with self._state_lock:
            self._pending_prefix = prefix

    def get_intrinsics(self, topic: str, width: int, height: int) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
        k, _source = _eai_platform_bootstrap.resolve_intrinsics(
            topic, width, height, live=self._camera_intrinsics,
            live_sizes=self._camera_intrinsics_sizes,
        )
        return k

    def _camera_info_callback(self, msg: _eai_platform_bootstrap.CameraInfo, depth_topic: str) -> None:
        k = _eai_perception_image_geometry.intrinsics_from_camera_info(msg)
        if msg.width <= 0 or msg.height <= 0 or min(k[:2]) <= 0:
            return
        self._camera_intrinsics[depth_topic] = k
        self._camera_intrinsics_sizes[depth_topic] = (int(msg.width), int(msg.height))
        if msg.header.frame_id:
            self._camera_frame_ids[depth_topic] = msg.header.frame_id

    def _subscribe_camera_info(self, depth_topic: str, qos: _eai_platform_bootstrap.QoSProfile) -> None:
        if depth_topic in self.camera_info_subs:
            return
        all_topics = dict(self.get_topic_names_and_types())
        for candidate in _eai_perception_image_geometry.camera_info_candidates(depth_topic):
            if candidate not in all_topics:
                continue
            sub = self.create_subscription(
                _eai_platform_bootstrap.CameraInfo,
                candidate,
                lambda msg, t=depth_topic: self._camera_info_callback(msg, t),
                qos,
            )
            self.camera_info_subs[depth_topic] = sub
            self.get_logger().info(f"订阅 CameraInfo: {candidate} -> {depth_topic}")
            return

    def _on_timer(self) -> None:
        with self._state_lock:
            if self._pending_prefix is not None:
                self.prefix = self._pending_prefix
                self._pending_prefix = None
            pending_enabled = self._pending_enabled
            self._pending_enabled = None

        if pending_enabled is not None:
            self.enabled_topics = pending_enabled
            self._resubscribe()
            self.get_logger().debug(
                f"subscribed to {len(self.enabled_topics)} image topics"
            )

        self._refresh_topics()
        self._maybe_retry_subscriptions()
        self._maybe_retry_robot_subscriptions()

    def _refresh_topics(self) -> None:
        all_topics = self._cached_topic_graph()
        camera_topics = {
            name: types
            for name, types in sorted(all_topics.items())
            if name.startswith(self.prefix)
        }

        if camera_topics != self._last_camera_topics:
            self._last_camera_topics = camera_topics
            self.ros_bridge.topics_updated.emit(camera_topics)

        if not all_topics:
            message = (
                "未发现任何 topic — 请用 bash run.sh 启动（需 ROS_LOCALHOST_ONLY=1 + a2d DDS 配置）"
            )
        elif len(all_topics) <= 3:
            message = (
                f"仅发现 {len(all_topics)} 个 topic（robot 未连通）— 请确认 robot-service 已启动并用 bash run.sh 运行"
            )
        else:
            message = (
                f"共 {len(all_topics)} 个 topic，匹配 {len(camera_topics)} 个（前缀: {self.prefix}）"
            )
        if message != self._last_status_message:
            self._last_status_message = message
            self.ros_bridge.status_message.emit(message)

    def _qos_candidates_for_topic(self, topic: str) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.QoSProfile]:
        candidates: _eai_platform_bootstrap.List[_eai_platform_bootstrap.QoSProfile] = []

        derived = self._qos_from_publisher_info(topic)
        if derived is not None:
            candidates.append(derived)

        # depth 必须小：原始 Image 很大，depth=30 会在回调跟不上时把内存打爆并卡死 UI
        candidates.extend([
            _eai_platform_bootstrap.QoSProfile(
                reliability=_eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
            ),
            _eai_platform_bootstrap.QoSProfile(
                reliability=_eai_platform_bootstrap.ReliabilityPolicy.BEST_EFFORT,
                history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=_eai_platform_bootstrap.DurabilityPolicy.VOLATILE,
            ),
        ])
        return candidates

    def _qos_for_topic(self, topic: str, attempt: int = 0) -> _eai_platform_bootstrap.QoSProfile:
        candidates = self._qos_candidates_for_topic(topic)
        idx = min(attempt, len(candidates) - 1)
        qos = candidates[idx]
        # 强制夹紧 depth，即使从 publisher 推导出更大值
        try:
            if int(getattr(qos, "depth", 1) or 1) > 2:
                qos = _eai_platform_bootstrap.QoSProfile(
                    reliability=qos.reliability,
                    history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
                    depth=1,
                    durability=qos.durability,
                )
        except Exception:
            pass
        self.get_logger().debug(
            f"{topic}: QoS attempt={attempt} reliability={qos.reliability.name} depth={qos.depth}"
        )
        return qos

    @staticmethod
    def _image_type_fallbacks(types: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[str]] = None) -> _eai_platform_bootstrap.List[str]:
        """ROS graph 尚未发现时，按 Image / CompressedImage 尝试订阅。"""
        out: _eai_platform_bootstrap.List[str] = []
        for name in list(types or []) + [
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CompressedImage",
            "sensor_msgs/Image",
            "sensor_msgs/CompressedImage",
        ]:
            if name and name not in out:
                out.append(name)
        return out

    def _maybe_retry_subscriptions(self) -> None:
        # 不再自动反复重订阅：无图时刷 QoS 只会卡 UI / 刷屏。
        # 用户取消勾选再勾选，或点刷新，会走 set_enabled_topics → _resubscribe。
        return

    def _resubscribe(self) -> None:
        for sub in self.subscriptions_map.values():
            self.destroy_subscription(sub)
        for sub in self.camera_info_subs.values():
            self.destroy_subscription(sub)
        self.subscriptions_map.clear()
        self.camera_info_subs.clear()

        all_topics = self._cached_topic_graph()
        for topic in sorted(self.enabled_topics):
            types = self._image_type_fallbacks(all_topics.get(topic, []))
            attempt = self._subscribe_attempts.get(topic, 0)
            qos = self._qos_for_topic(topic, attempt)
            if self._subscribe_topic(topic, types, qos):
                self._last_subscribe_time[topic] = _eai_platform_bootstrap.time.time()
                self.get_logger().debug(f"订阅: {topic}")
                if _eai_perception_image_geometry.is_depth_topic(topic):
                    self._subscribe_camera_info(topic, qos)
            else:
                self.get_logger().warn(f"跳过非图像 topic: {topic}")

    def _subscribe_topic(self, topic: str, types: _eai_platform_bootstrap.List[str], qos: _eai_platform_bootstrap.QoSProfile) -> bool:
        for type_name in self._image_type_fallbacks(types):
            msg_type = _eai_configuration_perception.IMAGE_TYPES.get(type_name)
            if msg_type is _eai_platform_bootstrap.CompressedImage:
                sub = self.create_subscription(
                    _eai_platform_bootstrap.CompressedImage,
                    topic,
                    lambda msg, t=topic: self._compressed_callback(msg, t),
                    qos,
                )
                self.subscriptions_map[topic] = sub
                return True
            if msg_type is _eai_platform_bootstrap.Image:
                sub = self.create_subscription(
                    _eai_platform_bootstrap.Image,
                    topic,
                    lambda msg, t=topic: self._image_callback(msg, t),
                    qos,
                )
                self.subscriptions_map[topic] = sub
                return True
        return False

    def _convert_image(self, msg: _eai_platform_bootstrap.Image) -> _eai_platform_bootstrap.np.ndarray:
        encoding = (msg.encoding or "bgr8").lower()
        if encoding in ("rgb8", "bgr8"):
            desired = "bgr8" if encoding == "bgr8" else "rgb8"
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=desired)
            if encoding == "rgb8":
                cv_image = _eai_platform_bootstrap.cv2.cvtColor(cv_image, _eai_platform_bootstrap.cv2.COLOR_RGB2BGR)
            return cv_image
        if "16" in encoding or encoding in ("32fc1", "mono16"):
            return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if encoding in ("mono8", "8uc1"):
            gray = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return _eai_platform_bootstrap.cv2.cvtColor(gray, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _should_emit_frame_to_ui(self, topic: str, now: _eai_platform_bootstrap.Optional[float] = None) -> bool:
        now = _eai_platform_bootstrap.time.time() if now is None else now
        interval = _eai_configuration_policies.UI_DEPTH_MIN_INTERVAL_S if _eai_perception_image_geometry.is_depth_topic(topic) else _eai_configuration_policies.UI_IMAGE_MIN_INTERVAL_S
        last = self._last_ui_emit_time.get(topic, 0.0)
        if now - last < interval:
            return False
        self._last_ui_emit_time[topic] = now
        return True

    def take_pending_ui_frames(
        self,
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.np.ndarray], _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[int, float]]]:
        """UI 线程取走待显示帧（每 topic 最多一帧），清空合并队列。"""
        with self._pending_ui_lock:
            frames = self._pending_ui_frames
            stats = self._pending_ui_stats
            self._pending_ui_frames = {}
            self._pending_ui_stats = {}
            self._frame_tick_pending = False
            return frames, stats

