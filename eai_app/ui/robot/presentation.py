from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.robot.dependencies')

class PresentationMixin:
    def _set_pose_enabled(self, key: str, enabled: bool) -> None:
        self._pose_on[key] = bool(enabled)
        item = self._pose_items.get(key)
        if item is not None:
            item["enabled"] = bool(enabled) and self._pose_hold.get(key) is not None
        if self._gl is not None:
            self._gl.update()

    def _show_wrist_accessories(self, enabled):
        if self._gl is not None:
            self._gl.show_wrist_accessories = bool(enabled)
            self._gl.update()

    def _set_target_poses_enabled(self, enabled):
        if self._target_pose_item is not None:
            self._target_pose_item['enabled'] = bool(enabled) and bool(self._target_pose_hold)
        if self._gl is not None:
            self._gl.update()

    def _set_sam3_enabled(self, enabled: bool) -> None:
        self._sam3_on = bool(enabled)
        if self._sam3 is not None:
            self._sam3["enabled"] = bool(enabled) and self._sam3_hold is not None
        if self._gl is not None:
            self._gl.update()

    def _set_lingbot_enabled(self, enabled: bool) -> None:
        self._lingbot_on = bool(enabled)
        if self._lingbot is not None:
            self._lingbot["enabled"] = bool(enabled) and self._lingbot_hold is not None
        if self._gl is not None:
            self._gl.update()

    def set_sam3_points(self, points) -> None:
        """SAM3 分割点，Nx3，已经在 base_link。空则清掉。"""
        if points is None:
            self._sam3_hold = None
            self._sam3_note = ""
        else:
            arr = _deps.np.asarray(points, dtype=_deps.np.float32).reshape(-1, 3)
            if arr.shape[0] == 0 or not _deps.np.isfinite(arr).all():
                self._sam3_hold = None
                self._sam3_note = ""
            else:
                self._sam3_hold = arr.copy()
                self._sam3_note = f"SAM3 {arr.shape[0]} 点"
        self._sync_sam3()

    def _sync_sam3(self) -> None:
        item = self._sam3
        if item is None:
            return
        item["enabled"] = bool(self._sam3_on) and self._sam3_hold is not None
        if self._sam3_hold is None:
            packed = _deps.np.zeros((0, 6), dtype=_deps.np.float32)
        else:
            rgb = _deps.np.array((0.95, 0.2, 0.85), dtype=_deps.np.float32)
            packed = _deps.np.zeros((self._sam3_hold.shape[0], 6), dtype=_deps.np.float32)
            packed[:, :3] = self._sam3_hold
            packed[:, 3:] = rgb
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def _sync_lingbot(self) -> None:
        item = self._lingbot
        if item is None:
            return
        item["enabled"] = bool(self._lingbot_on) and self._lingbot_hold is not None
        item["parent"] = self._lingbot_parent or ""
        item["T"] = _deps.np.asarray(self._lingbot_T, dtype=_deps.np.float64)
        if self._lingbot_hold is None:
            packed = _deps.np.zeros((0, 6), dtype=_deps.np.float32)
        else:
            packed = _deps.np.zeros((self._lingbot_hold.shape[0], 6), dtype=_deps.np.float32)
            packed[:, :3] = self._lingbot_hold
            if self._lingbot_colors is not None and self._lingbot_colors.shape[0] == packed.shape[0]:
                packed[:, 3:6] = self._lingbot_colors
            else:
                packed[:, 3:6] = (0.35, 0.95, 0.45)
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def set_method_pose(self, key: str, xyz, rotation, corners=None, extents=None) -> None:
        """把某一估计方法的 6D 位姿画在 base_link 上。xyz 为空则清掉该方法。"""
        label = next((text for name, text, _rgb in _deps._POSE_METHODS if name == key), key)
        if xyz is None or rotation is None:
            self._pose_hold.pop(key, None)
            self._pose_note.pop(key, None)
        else:
            point = _deps.np.asarray(xyz, dtype=_deps.np.float64).reshape(3).copy()
            matrix = _deps.np.asarray(rotation, dtype=_deps.np.float64).reshape(3, 3).copy()
            box = None if corners is None else _deps.np.asarray(corners, dtype=_deps.np.float64).copy()
            size = None if extents is None else tuple(float(v) for v in extents)
            self._pose_hold[key] = (point, matrix, box, size)
            self._pose_note[key] = f"{label} ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        self._sync_pose(key)

    def set_estimated_pose(self, xyz, rotation, corners=None, extents=None) -> None:
        self.set_method_pose("pca", xyz, rotation, corners, extents)

    def set_target_poses(self, poses) -> None:
        self._target_pose_hold = list(poses or [])
        item = self._target_pose_item
        if item is None:
            return
        colors = ((1.,.25,.25),(.25,1.,.35),(.25,.55,1.),(1.,.75,.15),(1.,.3,.9),(.3,1.,1.))
        rows = []
        for index, pose in enumerate(self._target_pose_hold):
            packed = _deps._pose6d_lines(pose['xyz'], pose['rotation'], pose.get('corners'),
                                   pose.get('extents'), colors[index % len(colors)])
            if len(packed):
                rows.append(packed)
        packed = _deps.np.concatenate(rows, axis=0) if rows else _deps.np.zeros((0,6), _deps.np.float32)
        with item['lock']:
            item['pending'] = packed
        item['enabled'] = self._target_poses_check.isChecked() and bool(rows)
        if self._gl is not None:
            self._gl.update()

    def set_collision_scene(self,scene,attachments=()):
        self._collision_scene_hold=(scene,list(attachments))
        if not self._collision_scene_items:return
        groups={'world':[], 'Link7_l':[], 'Link7_r':[]}
        for item in scene.get('world',[]):
            T=_deps.np.asarray(item['T_base_object'])
            groups['world'].append(_deps._pose6d_lines(T[:3,3],T[:3,:3],None,item['size_m'],(.9,.6,.15)))
        for item in attachments:
            T=_deps.np.asarray(item['T_link_object'])
            groups[item['link']].append(_deps._pose6d_lines(T[:3,3],T[:3,:3],None,item['size_m'],(.2,.9,.9)))
        for key,rows in groups.items():
            layer=self._collision_scene_items[key]
            with layer['lock']:layer['pending']=_deps.np.concatenate(rows) if rows else _deps.np.zeros((0,6),_deps.np.float32)
            layer['enabled']=bool(rows) and self._collision_scene_check.isChecked()
        self._collision_scene_check.setToolTip('物理仿真对象状态，不是真机标定' if scene.get('source')=='simulation_task' else
            ('已标定环境' if scene.get('calibrated') else '场景待标定，仅显示配置几何'))
        if self._gl is not None:self._gl.update()

    def _sync_pose(self, key: str) -> None:
        item = self._pose_items.get(key)
        if item is None:
            return
        hold = self._pose_hold.get(key)
        item["enabled"] = bool(self._pose_on.get(key, True)) and hold is not None
        if hold is None:
            packed = _deps.np.zeros((0, 6), dtype=_deps.np.float32)
        else:
            point, matrix, box, size = hold
            packed = _deps._pose6d_lines(point, matrix, box, size, item.get("box_rgb"))
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def _sync_pose6d(self) -> None:
        for key, _label, _rgb in _deps._POSE_METHODS:
            self._sync_pose(key)

    def set_running(self, running: bool) -> None:
        if running and self._gl is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._gl is not None:
            self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _load(self) -> None:
        try:
            urdf = _deps._resolve_urdf()
            visuals = _deps.load_visuals(urdf)
            self._hand_model = _deps.RuiyanHandModel()
            visuals.extend(self._hand_model.visuals)
            if not visuals:
                raise RuntimeError("URDF 里没有可用的视觉网格")
            self._kin = _deps._Kinematics(urdf)
            self._gl = _deps._RobotGL(visuals, self)
            self._gl.show_wrist_accessories = self._wrist_accessories_check.isChecked()
            self._clouds = self._make_clouds()
            self._skels = self._make_skeletons()
            self._wrenches = self._make_wrenches()
            self._pose_items = {
                key: self._make_pose6d(rgb) for key, _label, rgb in _deps._POSE_METHODS
            }
            self._sam3 = self._make_sam3()
            self._lingbot = self._make_lingbot()
            self._target_pose_item = self._make_pose6d((1., .3, .3))
            self._collision_scene_items={key:self._make_pose6d((1.,.6,.1)) for key in ('world','Link7_l','Link7_r')}
            for key,item in self._collision_scene_items.items():
                item['parent']='' if key=='world' else key;item['T']=_deps.np.eye(4)
            self._reach_items = {side: self._make_sam3() for side in ('l', 'r')}
            for item in self._reach_items.values():
                item['point_size'] = 2.0
            self._gl.set_clouds(self._clouds)
            self._gl.set_skeletons(self._skels)
            self._gl.set_wrenches(self._wrenches)
            self._gl.set_pose_items(
                list(self._pose_items.values()) + [self._sam3, self._lingbot, self._target_pose_item]
                + list(self._reach_items.values())
                + list(self._collision_scene_items.values())
            )
            self._sync_pose6d()
            self._sync_sam3()
            self._sync_lingbot()
            self.set_target_poses(self._target_pose_hold)
            self.set_collision_scene(*self._collision_scene_hold)
            self.layout().addWidget(self._gl, 1)
            try:
                self._subscribe_depth()
            except Exception as exc:
                self._status.setText(f"深度订阅失败：{exc}")
            try:
                self._subscribe_wrench()
            except Exception as exc:
                self._status.setText(f"六维力订阅失败：{exc}")
            self._status.setText("拖动旋转，右键平移，滚轮缩放。勾选要看的深度。")
            self._tick()
            if self.isVisible():
                self._timer.start()
        except Exception as exc:
            self._error = str(exc)
            self._status.setText(f"机器人视图加载失败：{exc}")

    def _make_clouds(self) -> _deps.List[dict]:
        from eai_app.adapters.ros.depth_cloud import _CAMERAS, _load_extrinsics, _placement

        cams = _load_extrinsics()
        key_of = {topic: key for topic, key, _label, _tip in _deps._DEPTH_KEYS}
        color_of = {
            "/camera/head_depth": (80 / 255.0, 220 / 255.0, 220 / 255.0),
            "/camera/hand_left_depth": (70 / 255.0, 140 / 255.0, 230 / 255.0),
            "/camera/hand_right_depth": (230 / 255.0, 140 / 255.0, 60 / 255.0),
        }
        clouds: _deps.List[dict] = []
        self._cloud_by_topic = {}
        for spec in _CAMERAS:
            cam = dict(cams.get(spec["color"]) or {}, color_topic=spec["color"])
            parent, xyz, quat, note = _placement(cam, spec)
            key = key_of[spec["depth"]]
            label = next(text for topic, name, text, _tip in _deps._DEPTH_KEYS if name == key)
            cloud = {
                "key": key,
                "label": label,
                "topic": spec["depth"],
                "color_topic": spec["color"],
                "parent": parent,
                "T": _deps._T_from_xyz_quat(xyz, quat),
                "color": color_of[spec["depth"]],
                "enabled": bool(self._depth_on.get(key, True)),
                "cam": cam,
                "pending": None,
                "buf": None,
                "count": 0,
                "lock": _deps.threading.Lock(),
            }
            clouds.append(cloud)
            self._cloud_by_topic[spec["depth"]] = cloud
            box = self._depth_checks.get(key)
            if box is not None:
                box.setToolTip(note)
        return clouds

    def _make_pose6d(self, box_rgb=(1.0, 0.86, 0.15)) -> dict:
        return {
            "parent": "",
            "T": _deps.np.eye(4, dtype=_deps.np.float64),
            "color": (1.0, 1.0, 1.0),
            "box_rgb": tuple(box_rgb),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": _deps.threading.Lock(),
        }

    def _make_sam3(self) -> dict:
        return {
            "parent": "",
            "T": _deps.np.eye(4, dtype=_deps.np.float64),
            "color": (0.95, 0.2, 0.85),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": _deps.threading.Lock(),
            "draw_mode": _deps._GL_POINTS,
            "point_size": 5.0,
        }

    def _make_lingbot(self) -> dict:
        return {
            "parent": self._lingbot_parent or "",
            "T": _deps.np.asarray(self._lingbot_T, dtype=_deps.np.float64),
            "color": (0.35, 0.95, 0.45),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": _deps.threading.Lock(),
            "draw_mode": _deps._GL_POINTS,
            "point_size": 3.0,
        }

    def _tick(self) -> None:
        for boxes in self._sim_hand_controls.values():
            for box in boxes:box.setEnabled(self._simulation_q is not None and self._simulation_manual_allowed())
        if self._gl is None or self._kin is None:
            return
        q16, age = _deps.read_q16()
        if self._simulation_q is not None:
            q16, age = self._simulation_q, 0.
        self._update_reachable_space(q16, age)
        self._ingest_cached_depth()
        self._update_skeleton()
        for item in self._wrenches:
            item["enabled"] = bool(self._wrench_on) and item.get("force") is not None
        self._refresh_wrench_readout()
        if q16 is None:
            q16 = [0.0] * 16
            prefix = "没有关节数据。请先启动 host_fk。"
        elif age > 2.0:
            prefix = f"关节数据已停 {age:.0f} 秒。"
        else:
            prefix = ""
        try:
            poses = self._kin.link_poses(q16)
            if self._hand_model is not None:
                poses.update(self._hand_model.poses(poses, self._hand_motors()))
            self._gl.set_poses(poses)
        except Exception as exc:
            self._status.setText(f"正运动学失败：{exc}")
            return
        pose_note = ""
        shown = [
            self._pose_note[key]
            for key, _label, _rgb in _deps._POSE_METHODS
            if self._pose_on.get(key, True) and self._pose_note.get(key)
        ]
        if shown:
            pose_note = " ".join(shown) + "。"
        sam3_note = ""
        if self._sam3_on and self._sam3_note:
            sam3_note = self._sam3_note + "。"
        lingbot_note = ""
        if self._lingbot_on and self._lingbot_note:
            lingbot_note = self._lingbot_note + "。"
        elif self._lingbot_on and self._lingbot is not None and self._lingbot_hold is None:
            lingbot_note = "LingBot 未收到。"
        self._status.setText(
            prefix
            + self._depth_status()
            + pose_note
            + sam3_note
            + lingbot_note
            + self._wrench_status()
            + " 拖动旋转，滚轮缩放。"
        )

