from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.robot.dependencies')

class SensorLayersMixin:
    def _set_depth_enabled(self, key: str, enabled: bool) -> None:
        self._depth_on[key] = bool(enabled)
        for cloud in self._clouds:
            if cloud.get("key") == key:
                cloud["enabled"] = bool(enabled)
        if self._gl is not None:
            self._gl.update()

    def _set_skeleton_enabled(self, enabled: bool) -> None:
        self._skel_on = bool(enabled)
        for skel in self._skels:
            skel["enabled"] = bool(enabled)
        if self._gl is not None:
            self._gl.update()

    def _set_wrench_enabled(self, enabled: bool) -> None:
        self._wrench_on = bool(enabled)
        for item in self._wrenches:
            item["enabled"] = bool(enabled) and item.get("force") is not None
        self._refresh_wrench_readout()
        if self._gl is not None:
            self._gl.update()

    def _refresh_wrench_readout(self) -> None:
        label = getattr(self, "_wrench_readout", None)
        if label is None:
            return
        label.setText(
            _deps.format_wrench_html(self._wrenches, enabled=self._wrench_on)
        )

    def set_lingbot_depth(
        self,
        points=None,
        colors=None,
        *,
        color_topic: str = "",
        note: str = "",
    ) -> None:
        """LingBot-Depth 点云（相机系）。挂在与实时深度同一套外参上。空则清掉。"""
        if points is None:
            self._lingbot_hold = None
            self._lingbot_colors = None
            self._lingbot_parent = ""
            self._lingbot_T = _deps.np.eye(4, dtype=_deps.np.float64)
            self._lingbot_note = ""
            self._sync_lingbot()
            return
        arr = _deps.np.asarray(points, dtype=_deps.np.float32).reshape(-1, 3)
        if arr.shape[0] == 0 or not _deps.np.isfinite(arr).all():
            self._lingbot_hold = None
            self._lingbot_colors = None
            self._lingbot_note = ""
            self._sync_lingbot()
            return
        keep = _deps.valid_depth_mask(arr[:, 2])
        if not _deps.np.any(keep):
            self.set_lingbot_depth(None)
            return
        if colors is not None:
            color_array = _deps.np.asarray(colors).reshape(-1, 3)
            colors = color_array[keep] if len(color_array) == len(arr) else None
        arr = arr[keep]
        if arr.shape[0] > 80000:
            step = int(_deps.np.ceil(arr.shape[0] / 80000.0))
            arr = arr[::step]
            if colors is not None:
                colors = _deps.np.asarray(colors, dtype=_deps.np.float32).reshape(-1, 3)[::step]
        rgb = None
        if colors is not None:
            rgb = _deps.np.asarray(colors, dtype=_deps.np.float32).reshape(-1, 3)
            if rgb.shape[0] != arr.shape[0] or not _deps.np.isfinite(rgb).all():
                rgb = None
            else:
                if float(_deps.np.nanmax(rgb)) > 1.5:
                    rgb = rgb / 255.0
                rgb = _deps.np.clip(rgb, 0.0, 1.0)
        topic = (color_topic or "").strip()
        if topic:
            parent, T, place_note = _deps._placement_for_color(topic)
        else:
            parent, T, place_note = "base_link", _deps.np.eye(4, dtype=_deps.np.float64), "无相机 topic，按 base_link"
        self._lingbot_hold = arr.copy()
        self._lingbot_colors = None if rgb is None else rgb.copy()
        self._lingbot_parent = parent
        self._lingbot_T = _deps.np.asarray(T, dtype=_deps.np.float64)
        self._lingbot_note = f"{note or f'LingBot {arr.shape[0]} 点'} · {place_note}"
        self._sync_lingbot()

    def set_lingbot_depth_from_ply(self, ply_path: str, *, color_topic: str = "") -> None:
        """从 LingBot-Depth 写出的 point_cloud.ply 加载。"""
        path = _deps.os.path.abspath(_deps.os.path.expanduser(ply_path or ""))
        if not path or not _deps.os.path.isfile(path):
            self.set_lingbot_depth(None)
            return
        xyz, rgb = _deps.load_ply_xyz_rgb(path)
        self.set_lingbot_depth(
            xyz,
            rgb,
            color_topic=color_topic,
            note=f"LingBot {xyz.shape[0]} 点",
        )

    def _make_skeletons(self) -> _deps.List[dict]:
        specs = (("left", "Link7_l"), ("right", "Link7_r"))
        skels = []
        for side, parent in specs:
            skels.append(
                {
                    "side": side,
                    "parent": parent,
                    "T": _deps.np.eye(4, dtype=_deps.np.float64),
                    "color": (1.0, 1.0, 1.0),
                    "enabled": bool(self._skel_on),
                    "pending": None,
                    "buf": None,
                    "count": 0,
                    "lock": _deps.threading.Lock(),
                }
            )
        return skels

    def _make_wrenches(self) -> _deps.List[dict]:
        items = []
        self._wrench_by_side = {}
        for side, topic, parent, label in _deps._WRENCH_TOPICS:
            item = {
                "side": side,
                "topic": topic,
                "label": label,
                "parent": parent,
                "T": _deps.np.eye(4, dtype=_deps.np.float64),
                "color": (1.0, 0.35, 0.12),
                "enabled": False,
                "pending": None,
                "buf": None,
                "count": 0,
                "lock": _deps.threading.Lock(),
                "force": None,
                "torque": None,
                "force_state": None,
                "stamp": 0.0,
                "error": "",
            }
            items.append(item)
            self._wrench_by_side[side] = item
        return items

    def _hand_motors(self):
        motors = {"left": {}, "right": {}}
        node = self._ros_node
        if node is not None and hasattr(node, "get_robot_state"):
            try:
                state = node.get_robot_state()
                motors["left"] = {name: value for name, value in state.left_hand}
                motors["right"] = {name: value for name, value in state.right_hand}
            except Exception:
                pass
        if self._simulation_q is not None:
            order = ('thumb_rotation', 'thumb_bend', 'index', 'middle', 'ring', 'pinky')
            motors = {side: dict(zip(order, self._simulation_hands.get(side, [0.] * 6)))
                      for side in ('left', 'right')}
        return motors

    def _update_skeleton(self) -> None:
        for skel in self._skels:
            skel['enabled'] = bool(self._skel_on)
        if not self._skel_on:
            return
        motors = self._hand_motors()
        for skel in self._skels:
            try:
                packed = _deps._skeleton_lines(skel["side"], motors.get(skel["side"]) or {})
            except Exception:
                continue
            with skel["lock"]:
                skel["pending"] = packed

    def _subscribe_depth(self) -> None:
        node = self._ros_node
        if node is None or not self._cloud_by_topic:
            return
        enqueue = getattr(node, "enqueue_ros_op", None)
        if callable(enqueue):
            enqueue(self._subscribe_depth_on_ros)
            return
        self._subscribe_depth_on_ros()

    def _subscribe_wrench(self) -> None:
        node = self._ros_node
        if node is None or not self._wrenches:
            return
        enqueue = getattr(node, "enqueue_ros_op", None)
        if callable(enqueue):
            enqueue(self._subscribe_wrench_on_ros)
            return
        self._subscribe_wrench_on_ros()

    def _subscribe_depth_on_ros(self) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = self._ros_node
        self._depth_subs = []
        for topic, cloud in self._cloud_by_topic.items():
            topic_qos = qos
            qos_for = getattr(node, "_qos_for_topic", None)
            if callable(qos_for):
                try:
                    topic_qos = qos_for(topic, 0)
                except Exception:
                    topic_qos = qos
            sub = node.create_subscription(
                Image,
                topic,
                lambda msg, bound=cloud: self._on_depth(bound, msg),
                topic_qos,
            )
            self._depth_subs.append(sub)

    def _subscribe_wrench_on_ros(self) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        try:
            ArmState = _deps.import_arm_state()
        except Exception as exc:
            for item in self._wrenches:
                item["error"] = f"缺少 genie_msgs：{exc}"
            return
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = self._ros_node
        self._wrench_subs = []
        for item in self._wrenches:
            topic = str(item.get("topic") or "")
            topic_qos = qos
            qos_for = getattr(node, "_qos_for_topic", None)
            if callable(qos_for):
                try:
                    topic_qos = qos_for(topic, 0)
                except Exception:
                    topic_qos = qos
            # ArmState 发布端是 RELIABLE，再挂一个可靠订阅兜底。
            try:
                from rclpy.qos import ReliabilityPolicy as _Rel

                reliable = QoSProfile(
                    reliability=_Rel.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    durability=DurabilityPolicy.VOLATILE,
                )
            except Exception:
                reliable = topic_qos
            try:
                sub = node.create_subscription(
                    ArmState,
                    topic,
                    lambda msg, bound=item: self._on_wrench(bound, msg),
                    reliable,
                )
            except Exception as exc:
                item["error"] = f"订阅失败：{exc}"
                continue
            self._wrench_subs.append(sub)

    def _on_depth(self, cloud: dict, msg) -> None:
        try:
            from eai_app.adapters.ros.depth_cloud import _depth_meters, _intrinsics

            depth = _depth_meters(msg)
            if self._ros_node is not None and hasattr(self._ros_node, "get_intrinsics"):
                fx, fy, cx, cy = self._ros_node.get_intrinsics(cloud["topic"], int(msg.width), int(msg.height))
            else:
                fx, fy, cx, cy = _intrinsics(cloud["cam"], int(msg.width), int(msg.height))
            with cloud["lock"]:
                color_bgr = cloud.get("color_bgr")
                if color_bgr is not None:
                    color_bgr = _deps.np.array(color_bgr, copy=True)
            packed = _deps._depth_points(
                depth,
                fx,
                fy,
                cx,
                cy,
                stride=4,
                color_bgr=color_bgr,
                fallback=tuple(cloud.get("color") or (0.7, 0.7, 0.7)),
            )
        except Exception as exc:
            cloud["error"] = str(exc)
            return
        with cloud["lock"]:
            cloud["pending"] = packed
            cloud["error"] = ""

    def _on_wrench(self, item: dict, msg) -> None:
        try:
            data = [float(v) for v in (msg.force_data or [])]
            if len(data) < 6 or not all(_deps.np.isfinite(data[:6])):
                raise ValueError(f"force_data 长度 {len(data)}")
            force = (data[0], data[1], data[2])
            torque = (data[3], data[4], data[5])
            packed = _deps._wrench_lines(force, torque)
            state = int(getattr(msg, "force_state", 0) or 0)
        except Exception as exc:
            with item["lock"]:
                item["error"] = str(exc)
            return
        with item["lock"]:
            item["pending"] = packed
            item["force"] = force
            item["torque"] = torque
            item["force_state"] = state
            item["stamp"] = _deps.time.time()
            item["error"] = "" if state == 0 else f"force_state={state}"
            item["enabled"] = bool(self._wrench_on)

    def _ingest_cached_depth(self) -> None:
        """用界面已经收到的深度帧。机器人页自己的订阅挂错过节点时，这里仍能画出来。"""
        frames = getattr(self._ros_node, "click_frames", None)
        if frames is None or not self._clouds:
            return
        grabbed = []
        with frames.lock:
            for cloud in self._clouds:
                queue = frames.frames.get(cloud.get("topic"))
                if not queue:
                    continue
                frame = queue[-1]
                color_frame = None
                color_queue = frames.frames.get(cloud.get("color_topic"))
                if color_queue:
                    color_frame = color_queue[-1]
                token = (id(frame), id(color_frame) if color_frame is not None else 0)
                if cloud.get("_token") == token:
                    continue
                color_img = None if color_frame is None else _deps.np.array(color_frame.image, copy=True)
                grabbed.append(
                    (cloud, _deps.np.array(frame.image, copy=True), str(frame.encoding), color_img, token)
                )
        for cloud, image, encoding, color_img, token in grabbed:
            intr = None
            if self._ros_node is not None and hasattr(self._ros_node, "get_intrinsics"):
                h, w = _deps.np.asarray(image).shape[:2]
                intr = self._ros_node.get_intrinsics(cloud["topic"], w, h)
            packed = _deps._pack_depth_image(
                cloud.get("cam") or {},
                image,
                encoding,
                color_bgr=color_img,
                intrinsics=intr,
                fallback=tuple(cloud.get("color") or (0.7, 0.7, 0.7)),
            )
            if packed is None:
                cloud["error"] = f"无法解析 {encoding}"
                continue
            with cloud["lock"]:
                if color_img is not None:
                    cloud["color_bgr"] = color_img
                cloud["pending"] = packed
                cloud["_token"] = token
                cloud["error"] = ""

    def _depth_status(self) -> str:
        parts = []
        poses = self._gl._poses if self._gl is not None else {}
        for cloud in self._clouds:
            if not cloud.get("enabled"):
                continue
            label = cloud.get("label") or cloud.get("key")
            err = cloud.get("error") or ""
            if err:
                parts.append(f"{label} {err}")
                continue
            if cloud.get("parent") not in poses:
                parts.append(f"{label} 缺少 {cloud.get('parent')}")
                continue
            count = int(cloud.get("count") or 0)
            parts.append(f"{label} {count} 点" if count else f"{label} 未收到")
        if not parts:
            return "深度未勾选。"
        return "深度 " + "，".join(parts) + "。"

    def _wrench_status(self) -> str:
        if not self._wrench_on:
            return ""
        parts = []
        now = _deps.time.time()
        for item in self._wrenches:
            label = item.get("label") or item.get("side")
            err = item.get("error") or ""
            force = item.get("force")
            torque = item.get("torque")
            stamp = float(item.get("stamp") or 0.0)
            if force is None or torque is None:
                parts.append(f"{label}{' ' + err if err else ' 未收到'}")
                continue
            age = now - stamp if stamp > 0 else 999.0
            f_norm = float(_deps.np.linalg.norm(force))
            m_norm = float(_deps.np.linalg.norm(torque))
            note = f"{label} F={f_norm:.1f}N M={m_norm:.2f}Nm"
            if age > 2.0:
                note += f" 停{age:.0f}s"
            if err:
                note += f" {err}"
            parts.append(note)
        if not parts:
            return ""
        return "六维力 " + "，".join(parts) + "。"

