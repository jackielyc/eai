"""ui.window.shell force check essential cameras — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellForceCheckEssentialCamerasMixin:
    def _force_check_essential_cameras(self, reason: str = "") -> None:
        topic_checks = getattr(self, "topic_checks", None) or {}
        if not topic_checks:
            self.status_bar.showMessage(
                "未发现相机 topic：请确认 forwarder 已启动"
                "（scripts/start_cameras_minimal.sh start）",
                6000,
            )
            return
        enabled = {t for t, cb in topic_checks.items() if cb.isChecked()}
        added: _eai_platform_bootstrap.List[str] = []
        for topic in _eai_configuration_policies.ESSENTIAL_A2D_CAMERA_COLOR_TOPICS:
            cb = topic_checks.get(topic)
            if cb is None or topic in self._topic_user_unchecked:
                continue
            if not cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                added.append(topic)
            enabled.add(topic)
        if not added:
            return
        self._apply_selection(enabled)
        self._refresh_skeleton_camera_list()
        msg = "已订阅头/腕彩色" + (f"（{reason}）" if reason else "")
        msg += f"：{', '.join(added)}"
        self.status_bar.showMessage(msg, 5000)
        btn = getattr(self, "camera_connect_btn", None)
        if btn is not None:
            btn.setText("刷新 Topic")

    def _on_refresh_clicked(self) -> None:
        prefix = self.prefix_edit.text().strip() or "/camera"
        if prefix == "/":
            self.status_bar.showMessage("前缀不能为 /，已重置为 /camera")
            prefix = "/camera"
            self.prefix_edit.setText(prefix)
        self._ensure_essential_robot_cameras(reason="刷新 Topic")

    def _read_sim_robot_state(self) -> _eai_platform_bootstrap.Optional[dict]:
        if _eai_configuration_policies.read_sim_robot_state is None:
            return None
        try:
            return _eai_configuration_policies.read_sim_robot_state(self._sim_bridge_dir())
        except Exception:
            return None

    def _any_sim_process_running(self) -> bool:
        if getattr(self, "_sim_eval_launcher", None) and self._sim_eval_launcher.is_running():
            return True
        if getattr(self, "_mujoco_launcher", None) and self._mujoco_launcher.is_running():
            return True
        return False

    def _maybe_record_arm_home(
        self,
        arm_side: str,
        pose: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]],
    ) -> None:
        """记录该侧首次有效 EE，作为归位目标。"""
        side = "right" if arm_side == "right" else "left"
        if side in self._arm_home_pose:
            return
        xyz, quat = pose
        self._arm_home_pose[side] = (
            (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )
        if getattr(self, "_arm_home_q16", None) is None:
            q16 = self.node._load_host_fk_q16()
            if q16 is not None and len(q16) >= 16:
                self._arm_home_q16 = [float(v) for v in q16[:16]]

    def _write_sim_gui_cmd(self, **kwargs) -> bool:
        if _eai_configuration_policies.write_gui_robot_cmd is None:
            return False
        try:
            return bool(
                _eai_configuration_policies.write_gui_robot_cmd(root=self._sim_bridge_dir(), merge=True, **kwargs)
            )
        except Exception:
            return False

    def _apply_sim_gripper(self, side: str, position: float) -> bool:
        pos = max(0.0, min(1.0, float(position)))
        if side == "right":
            return self._write_sim_gui_cmd(right_gripper=pos, enabled=True)
        return self._write_sim_gui_cmd(left_gripper=pos, enabled=True)

    def _is_sim_arm_moving(self) -> bool:
        return self._sim_arm_move is not None and self._sim_arm_move_timer.isActive()

    def _cancel_sim_arm_move(self, message: str = "已取消仿真手臂移动") -> str:
        self._sim_arm_move = None
        self._sim_arm_move_timer.stop()
        return message

    def _start_sim_arm_move(self, arm_side: str, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> str:
        arm_side = self._sim_resolve_arm_side(arm_side)
        start = self._sim_tcp_pose(arm_side)
        goal_xyz = goal.position_xyz
        goal_quat = goal.quaternion_xyzw
        if start is None:
            # 尚无 TCP：仍直接写绝对目标，供 NoopModelClient / 策略侧消费（不插值）
            pose7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(goal_xyz, goal_quat)
            kwargs: dict = {"enabled": True}
            if arm_side == "right":
                kwargs["right_ee_pose"] = pose7
            else:
                kwargs["left_ee_pose"] = pose7
            if not self._write_sim_gui_cmd(**kwargs):
                return (
                    "无法移动: 尚无仿真 TCP，且写入 gui_robot_cmd 失败"
                    "（请确认策略=无 / Noop，且 ISAAC_CAM_BRIDGE_DIR 可写）"
                )
            side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
            return (
                f"已写入仿真{side_name}绝对目标（当前无 TCP 回读，跳过插值）→ {goal.label}"
            )
        start_xyz, start_quat = start
        # 明显下压时先水平对准（同 XY、保持当前高度），再落到终值，IK 成功率更高
        final_xyz = (
            float(goal_xyz[0]),
            float(goal_xyz[1]),
            float(goal_xyz[2]),
        )
        dz = float(start_xyz[2]) - float(final_xyz[2])
        approach_xyz = None
        if dz >= float(_eai_configuration_policies.SIM_ARM_APPROACH_MIN_DZ_M):
            approach_xyz = (
                float(final_xyz[0]),
                float(final_xyz[1]),
                float(start_xyz[2]),
            )
            active_xyz = approach_xyz
        else:
            active_xyz = final_xyz
        dist = _eai_platform_bootstrap.math.sqrt(
            sum((float(active_xyz[i]) - float(start_xyz[i])) ** 2 for i in range(3))
        )
        duration = self._sim_move_duration_s(dist)
        hold_side = "right" if arm_side == "left" else "left"
        hold = self._sim_tcp_pose(hold_side)
        # 目标夹爪中点（按指令姿态从终值 EE 反推）；settle 闭环按此伺服，不依赖姿态是否跟得上
        target_mid = None
        if self._sim_uses_finger_mid_match():
            target_mid = self._finger_mid_xyz_from_ee(final_xyz, goal_quat)
        self._sim_arm_move = {
            "arm_side": arm_side,
            "start_xyz": start_xyz,
            "start_quat": start_quat,
            "goal_xyz": active_xyz,
            "goal_quat": goal_quat,
            "final_xyz": final_xyz,
            "final_quat": goal_quat,
            "target_mid_xyz": target_mid,
            "approach_pending": approach_xyz is not None,
            "hold_side": hold_side,
            "hold_pose": hold,
            "t0": _eai_platform_bootstrap.time.time(),
            "duration": duration,
            "label": goal.label,
        }
        # 立即写一帧目标，避免等待首个 tick
        self._on_sim_arm_move_tick()
        if not self._sim_arm_move_timer.isActive():
            self._sim_arm_move_timer.start()
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        phase = "先水平对准" if approach_xyz is not None else "直达"
        return f"仿真{side_name}移动中… ({phase} {duration:.1f}s) → {goal.label}"

    def _on_left_hand_sliders_changed(self, _value: int = 0) -> None:
        self.left_hand_label_a.setText(
            _eai_adapters_robot_geometry.format_hand_gap_label(self.left_hand_gap_slider_a.value())
        )
        self.left_hand_label_b.setText(
            _eai_adapters_robot_geometry.format_hand_gap_label(self.left_hand_gap_slider_b.value())
        )
        self._update_left_hand_toggle_ui(self.node.is_left_hand_at_a())

    def _update_left_hand_toggle_ui(self, at_a: bool) -> None:
        gap_a = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(self.left_hand_gap_slider_a.value())
        gap_b = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(self.left_hand_gap_slider_b.value())
        next_preset = "B" if at_a else "A"
        next_gap = gap_b if at_a else gap_a
        self.left_hand_toggle_btn.setText(
            f"左手: 切到{next_preset} ({next_gap:.1f}cm)"
        )
        active_style = f"font-weight: bold; color: {_eai_configuration_policies.UI_ACCENT_BLUE};"
        idle_style = f"color: {_eai_configuration_policies.UI_TEXT_MUTED};"
        self.left_hand_label_a.setStyleSheet(active_style if at_a else idle_style)
        self.left_hand_label_b.setStyleSheet(active_style if not at_a else idle_style)

    def _on_left_hand_toggle(self) -> None:
        if self._sim_teleop_active():
            self.node._left_hand_at_a = not self.node._left_hand_at_a
            at_a = self.node._left_hand_at_a
            slider = (
                self.left_hand_gap_slider_a.value()
                if at_a
                else self.left_hand_gap_slider_b.value()
            )
            gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
            pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
            ok = self._apply_sim_gripper("left", pos)
            self._update_left_hand_toggle_ui(at_a)
            self.node.ros_bridge.left_hand_preset_changed.emit(at_a)
            preset = "A" if at_a else "B"
            self.status_bar.showMessage(
                f"仿真左手 → 状态{preset} {gap_cm:.1f}cm"
                + ("" if ok else "（写入失败）")
            )
            return
        if not self.node.is_hand_enabled():
            self.status_bar.showMessage("手未使能：请先点「启用手」", 4000)
            return
        slider_a = self.left_hand_gap_slider_a.value()
        slider_b = self.left_hand_gap_slider_b.value()
        at_a, sent_pos = self.node.toggle_left_hand_between(slider_a, slider_b)
        self._update_left_hand_toggle_ui(at_a)
        preset = "A" if at_a else "B"
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider_a if at_a else slider_b)
        self.status_bar.showMessage(
            f"左手已切到状态{preset} {gap_cm:.1f}cm (pos={sent_pos:.3f}) "
            f"-> {_eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC}"
        )

    def _on_right_hand_sliders_changed(self, _value: int = 0) -> None:
        self.right_hand_label_a.setText(
            _eai_adapters_robot_geometry.format_hand_gap_label(self.right_hand_gap_slider_a.value())
        )
        self.right_hand_label_b.setText(
            _eai_adapters_robot_geometry.format_hand_gap_label(self.right_hand_gap_slider_b.value())
        )
        self._update_right_hand_toggle_ui(self.node.is_right_hand_at_a())

    def _update_right_hand_toggle_ui(self, at_a: bool) -> None:
        gap_a = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(self.right_hand_gap_slider_a.value())
        gap_b = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(self.right_hand_gap_slider_b.value())
        next_preset = "B" if at_a else "A"
        next_gap = gap_b if at_a else gap_a
        self.right_hand_toggle_btn.setText(
            f"右手: 切到{next_preset} ({next_gap:.1f}cm)"
        )
        active_style = f"font-weight: bold; color: {_eai_configuration_policies.UI_ACCENT_BLUE};"
        idle_style = f"color: {_eai_configuration_policies.UI_TEXT_MUTED};"
        self.right_hand_label_a.setStyleSheet(active_style if at_a else idle_style)
        self.right_hand_label_b.setStyleSheet(active_style if not at_a else idle_style)

    def _on_right_hand_toggle(self) -> None:
        if self._sim_teleop_active():
            self.node._right_hand_at_a = not self.node._right_hand_at_a
            at_a = self.node._right_hand_at_a
            slider = (
                self.right_hand_gap_slider_a.value()
                if at_a
                else self.right_hand_gap_slider_b.value()
            )
            gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
            pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
            ok = self._apply_sim_gripper("right", pos)
            self._update_right_hand_toggle_ui(at_a)
            self.node.ros_bridge.right_hand_preset_changed.emit(at_a)
            preset = "A" if at_a else "B"
            self.status_bar.showMessage(
                f"仿真右手 → 状态{preset} {gap_cm:.1f}cm"
                + ("" if ok else "（写入失败）")
            )
            return
        if not self.node.is_hand_enabled():
            self.status_bar.showMessage("手未使能：请先点「启用手」", 4000)
            return
        slider_a = self.right_hand_gap_slider_a.value()
        slider_b = self.right_hand_gap_slider_b.value()
        at_a, sent_pos = self.node.toggle_right_hand_between(slider_a, slider_b)
        self._update_right_hand_toggle_ui(at_a)
        preset = "A" if at_a else "B"
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider_a if at_a else slider_b)
        self.status_bar.showMessage(
            f"右手已切到状态{preset} {gap_cm:.1f}cm (pos={sent_pos:.3f}) "
            f"-> {_eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC}"
        )

    def _update_move_offset_ui_visibility(self) -> None:
        """「分割位姿」时隐藏六向输入；仅「相对当前」显示。"""
        show_offset = self.move_target_relative_radio.isChecked()
        for side in ("left", "right"):
            title = getattr(self, "_arm_dir_row_titles", {}).get(side)
            if title is not None:
                title.setVisible(show_offset)
            for direction in _eai_perception_pose_estimation.ARM_DIR_ORDER:
                for bag in (
                    getattr(self, "_arm_dir_labels", {}),
                    getattr(self, "_arm_dir_spins", {}),
                    getattr(self, "_arm_dir_btns", {}),
                ):
                    w = bag.get(side, {}).get(direction)
                    if w is not None:
                        w.setVisible(show_offset)
                        if isinstance(w, _eai_platform_bootstrap.QDoubleSpinBox):
                            w.setEnabled(show_offset)
        for w in getattr(self, "_arm_dir_extra_widgets", []):
            w.setVisible(show_offset)
        if show_offset:
            self._update_arm_dir_btns_ui()

    def _sync_offset_spins_from_dir(self, arm_side: str) -> None:
        """把指定臂六向净偏移写回隐藏的 offset_*_spin（兼容旧存档/文本指令路径）。"""
        if not hasattr(self, "offset_x_spin"):
            return
        dx, dy, dz = self._arm_combined_offset_m(arm_side)
        for spin, meters in (
            (self.offset_x_spin, dx),
            (self.offset_y_spin, dy),
            (self.offset_z_spin, dz),
        ):
            spin.blockSignals(True)
            _eai_perception_pose_estimation.set_manual_offset_spin_m(spin, meters)
            spin.blockSignals(False)

    def _set_arm_dir_from_xyz_m(
        self, arm_side: str, dx: float, dy: float, dz: float
    ) -> None:
        """按有符号米偏移填入六向输入（对侧清零）。"""
        spins = getattr(self, "_arm_dir_spins", {}).get(arm_side) or {}
        if not spins:
            if hasattr(self, "offset_x_spin"):
                _eai_perception_pose_estimation.set_manual_offset_spin_m(self.offset_x_spin, dx)
                _eai_perception_pose_estimation.set_manual_offset_spin_m(self.offset_y_spin, dy)
                _eai_perception_pose_estimation.set_manual_offset_spin_m(self.offset_z_spin, dz)
            return

        def _set(name: str, cm: float) -> None:
            sp = spins.get(name)
            if sp is None:
                return
            sp.blockSignals(True)
            sp.setValue(max(0.0, min(_eai_configuration_chat.MANUAL_OFFSET_MAX_CM, float(cm))))
            sp.blockSignals(False)

        _set("forward", dx * 100.0 if dx > 0 else 0.0)
        _set("back", -dx * 100.0 if dx < 0 else 0.0)
        _set("left", dy * 100.0 if dy > 0 else 0.0)
        _set("right", -dy * 100.0 if dy < 0 else 0.0)
        _set("up", dz * 100.0 if dz > 0 else 0.0)
        _set("down", -dz * 100.0 if dz < 0 else 0.0)
        self._sync_offset_spins_from_dir(arm_side)

    def _resolve_direction_move_goal(
        self, arm_side: str, direction: str
    ) -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.ResolvedArmMoveGoal]:
        """仅沿单一 base_link 轴生成相对目标（步长取该方向输入框）。"""
        cm = self._arm_dir_spin_cm(arm_side, direction)
        if cm < 1e-6:
            return None
        step = cm * 0.01
        mapping = {
            "forward": (step, 0.0, 0.0),
            "back": (-step, 0.0, 0.0),
            "left": (0.0, step, 0.0),
            "right": (0.0, -step, 0.0),
            "up": (0.0, 0.0, step),
            "down": (0.0, 0.0, -step),
        }
        delta = mapping.get(direction)
        if delta is None:
            return None
        tcp = self._tcp_for_relative_goal(arm_side)
        if tcp is None:
            return None
        xyz, quat = tcp
        return _eai_perception_pose_estimation.compute_relative_move_goal(
            xyz, quat, delta[0], delta[1], delta[2], arm_side=arm_side
        )

    def _update_arm_dir_btns_ui(self) -> None:
        """六向「移动」按钮使能：相对模式 + TCP 就绪（真机需使能）。"""
        btns_map = getattr(self, "_arm_dir_btns", None)
        if not btns_map:
            return
        show_offset = True
        if hasattr(self, "move_target_relative_radio"):
            show_offset = bool(self.move_target_relative_radio.isChecked())
        busy = self.node.is_slow_motion_busy() or self._is_sim_arm_moving()
        for side, btn_by_dir in btns_map.items():
            if not show_offset or busy:
                can = False
            elif self._sim_teleop_active():
                can = self._sim_tcp_pose(side) is not None
            else:
                can = bool(
                    self.node.is_arm_enabled()
                    and self._tcp_for_relative_goal(side) is not None
                )
            for btn in btn_by_dir.values():
                btn.setEnabled(bool(can))
                btn.setVisible(show_offset)

    def _ensure_arm_home_recorded(self) -> None:
        """尽量补记左右臂初始位姿（仿真优先）。"""
        for side in ("left", "right"):
            if side in self._arm_home_pose:
                continue
            pose = self._tcp_for_relative_goal(side)
            if pose is not None:
                self._maybe_record_arm_home(side, pose)

    def _start_sim_dual_arm_home_move(self) -> str:
        """双臂同时插值回到已记录的初始 EE。"""
        left_home = self._arm_home_pose.get("left")
        right_home = self._arm_home_pose.get("right")
        if left_home is None and right_home is None:
            return "无法归位: 尚未记录初始臂位姿（请先等仿真/TCP 回读）"
        left_cur = self._sim_tcp_pose("left") if left_home else None
        right_cur = self._sim_tcp_pose("right") if right_home else None
        # 无当前位姿时直接跳到目标
        if left_home is not None and left_cur is None:
            left_cur = left_home
        if right_home is not None and right_cur is None:
            right_cur = right_home

        def _dist(a, b) -> float:
            return _eai_platform_bootstrap.math.sqrt(
                sum((float(a[0][i]) - float(b[0][i])) ** 2 for i in range(3))
            )

        dist = 0.0
        if left_home is not None and left_cur is not None:
            dist = max(dist, _dist(left_cur, left_home))
        if right_home is not None and right_cur is not None:
            dist = max(dist, _dist(right_cur, right_home))
        duration = self._sim_move_duration_s(dist)
        move: dict = {
            "mode": "dual",
            "arm_side": "both",
            "t0": _eai_platform_bootstrap.time.time(),
            "duration": duration,
            "label": "双臂归位",
        }
        if left_home is not None and left_cur is not None:
            move["left_start_xyz"] = left_cur[0]
            move["left_start_quat"] = left_cur[1]
            move["left_goal_xyz"] = left_home[0]
            move["left_goal_quat"] = left_home[1]
        if right_home is not None and right_cur is not None:
            move["right_start_xyz"] = right_cur[0]
            move["right_start_quat"] = right_cur[1]
            move["right_goal_xyz"] = right_home[0]
            move["right_goal_quat"] = right_home[1]
        self._sim_arm_move = move
        self._on_sim_arm_move_tick()
        if not self._sim_arm_move_timer.isActive():
            self._sim_arm_move_timer.start()
        sides = []
        if "left_goal_xyz" in move:
            sides.append("左")
        if "right_goal_xyz" in move:
            sides.append("右")
        return f"仿真{'+'.join(sides)}臂归位中… ({duration:.1f}s)"

    @staticmethod
    def _finger_mid_offset_in_ee(
        quat_xyzw: _eai_platform_bootstrap.Sequence[float],
        offset_ee: _eai_platform_bootstrap.Sequence[float] = _eai_configuration_policies.SIM_FINGER_MID_OFFSET_IN_EE_M,
    ) -> _eai_platform_bootstrap.Tuple[float, float, float]:
        R = _eai_adapters_robot_geometry.quat_to_rotation_matrix(
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        )
        o = _eai_platform_bootstrap.np.asarray(offset_ee, dtype=_eai_platform_bootstrap.np.float64)
        v = R @ o
        return (float(v[0]), float(v[1]), float(v[2]))

    def _finger_mid_xyz_from_ee(
        self,
        ee_xyz: _eai_platform_bootstrap.Sequence[float],
        ee_quat_xyzw: _eai_platform_bootstrap.Sequence[float],
    ) -> _eai_platform_bootstrap.Tuple[float, float, float]:
        dx, dy, dz = self._finger_mid_offset_in_ee(ee_quat_xyzw)
        return (
            float(ee_xyz[0]) + dx,
            float(ee_xyz[1]) + dy,
            float(ee_xyz[2]) + dz,
        )

    def _ee_xyz_from_finger_mid_target(
        self,
        finger_xyz: _eai_platform_bootstrap.Sequence[float],
        ee_quat_xyzw: _eai_platform_bootstrap.Sequence[float],
    ) -> _eai_platform_bootstrap.Tuple[float, float, float]:
        """手指结合中点目标 → EE/link6 指令位置。"""
        dx, dy, dz = self._finger_mid_offset_in_ee(ee_quat_xyzw)
        return (
            float(finger_xyz[0]) - dx,
            float(finger_xyz[1]) - dy,
            float(finger_xyz[2]) - dz,
        )

    def _resolve_move_goal(self, arm_side: str = "left") -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.ResolvedArmMoveGoal]:
        if self.move_target_segment_radio.isChecked() and self._last_segment_target is not None:
            return self.node.resolve_segment_move_goal(
                self._last_segment_target, arm_side=arm_side
            )
        tcp = self._tcp_for_relative_goal(arm_side)
        if tcp is None:
            return None
        xyz, quat = tcp
        dx, dy, dz = self._arm_combined_offset_m(arm_side)
        return _eai_perception_pose_estimation.compute_relative_move_goal(
            xyz,
            quat,
            dx,
            dy,
            dz,
            arm_side=arm_side,
        )

    def _ensure_host_ruiyan(self) -> None:
        """确保宿主机 Ruiyan hand_node 在订 /ry_hand/*/set_angles（无 Docker）。"""
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_ruiyan.sh",
        )
        if not _eai_platform_bootstrap.os.path.isfile(script):
            return
        try:
            _eai_platform_bootstrap.subprocess.Popen(
                ["bash", script, "start"],
                stdout=_eai_platform_bootstrap.subprocess.DEVNULL,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    def _ensure_host_fk(self) -> None:
        """确保宿主机 FK 发布 /mink_fk/*_tcp_pose（无 Docker；替代缺失的 fk_node）。"""
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_fk.sh",
        )
        if not _eai_platform_bootstrap.os.path.isfile(script):
            return
        try:
            _eai_platform_bootstrap.subprocess.Popen(
                ["bash", script, "start"],
                stdout=_eai_platform_bootstrap.subprocess.DEVNULL,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    def _ensure_host_arm_hand_bridge(self) -> None:
        """确保宿主机使能桥 + Ruiyan 在跑（无 Docker）。"""
        self._ensure_host_ruiyan()
        self._ensure_host_fk()
        if _eai_adapters_robot_services._is_stack_services_running():
            return
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_arm_hand.sh",
        )
        if not _eai_platform_bootstrap.os.path.isfile(script):
            return
        try:
            _eai_platform_bootstrap.subprocess.Popen(
                ["bash", script, "start"],
                stdout=_eai_platform_bootstrap.subprocess.DEVNULL,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    def _ensure_enable_state_subscriptions(self) -> None:
        try:
            self.node.ensure_enable_state_subscriptions()
        except Exception:
            pass

    def _on_model_mode_clicked(self) -> None:
        self._ensure_host_arm_hand_bridge()
        msg = self.node.request_model_control_mode()
        self.status_bar.showMessage(msg)
        self._update_enable_status_ui()
        self._update_prereq_gated_buttons()

    @staticmethod
    def _style_enable_label(label: _eai_platform_bootstrap.QLabel, received: bool, enabled: bool) -> None:
        if not received:
            label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        elif enabled:
            label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")

    def _stack_ready_for_enable_services(self) -> bool:
        """手/臂使能与 mode 切换所需：手/臂服务栈已运行。"""
        if self._sim_teleop_active():
            return False
        try:
            _, base = self._stack_launcher.get_cached_status()
        except Exception:
            base = ""
        return base == "运行中" or _eai_adapters_robot_services._is_stack_services_running()

    def _has_segment_seed(self) -> bool:
        topic = getattr(self, "_last_sam3_topic", "") or ""
        panel = self.panels.get(topic) if topic else None
        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._pending_seed_uv is not None:
            return True
        if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D) and panel._pending_seed_full_uv is not None:
            return True
        for p in self.panels.values():
            if isinstance(p, _eai_ui_cameras_panels.CameraPanel) and p._pending_seed_uv is not None:
                return True
            if isinstance(p, _eai_ui_cameras_panels.DepthPanel3D) and p._pending_seed_full_uv is not None:
                return True
        return False

    def _set_btn_prereq_enabled(
        self,
        btn: _eai_platform_bootstrap.QPushButton,
        enabled: bool,
        *,
        ok_tip: str,
        blockers: _eai_platform_bootstrap.Sequence[str],
    ) -> None:
        btn.setEnabled(bool(enabled))
        if enabled:
            btn.setToolTip(ok_tip)
        else:
            reasons = list(blockers) or ["先决条件未满足"]
            btn.setToolTip("不可用:\n" + "\n".join(f"• {r}" for r in reasons))

    def _update_enable_control_btns_ui(self) -> None:
        """按服务栈就绪状态控制「启用手臂/手」「切 mode=0」「手部应用」。"""
        sim = self._sim_teleop_active()
        stack_ok = self._stack_ready_for_enable_services()
        host_bridge = _eai_platform_bootstrap.os.path.isfile(
            _eai_platform_bootstrap.os.path.join(
                str(PROJECT_ROOT),
                "scripts",
                "start_host_arm_hand.sh",
            )
        )
        arm_on = self.node.is_arm_enabled()
        hand_on = self.node.is_hand_enabled()
        # 有宿主机桥脚本即可点：点击时自动拉起（无 Docker）
        can_arm = sim or stack_ok or arm_on or host_bridge
        can_hand = sim or stack_ok or hand_on or host_bridge
        can_mode = (not sim) and (stack_ok or host_bridge)
        can_apply = sim or hand_on
        stack_hint = "将自动启动宿主机使能桥（无 Docker）"
        arm_tip = (
            "仿真遥控中无需真机使能"
            if sim
            else "等同踏板 F2 开启/关闭手臂控制"
        )
        hand_tip = (
            "仿真遥控中使用夹爪滑块即可"
            if sim
            else "等同踏板 F1 开启/关闭手部控制"
        )
        mode_tip = (
            f"发布 {_eai_configuration_policies.CONTROL_MODE_TOPIC}={_eai_configuration_chat.MODEL_CONTROL_MODE}"
            "（模型控制，手臂移动/回放需要）"
        )
        apply_tip = "将主游标（当前激活状态 A 或 B）的角度立即发送"
        for btn in (
            getattr(self, "arm_enable_btn", None),
            getattr(self, "robot_arm_enable_btn", None),
        ):
            if btn is not None:
                self._set_btn_prereq_enabled(
                    btn,
                    can_arm,
                    ok_tip=arm_tip,
                    blockers=["仿真遥控中" if sim else stack_hint],
                )
        for btn in (
            getattr(self, "hand_enable_btn", None),
            getattr(self, "robot_hand_enable_btn", None),
        ):
            if btn is not None:
                self._set_btn_prereq_enabled(
                    btn,
                    can_hand,
                    ok_tip=hand_tip,
                    blockers=["仿真遥控中" if sim else stack_hint],
                )
        for btn in (
            getattr(self, "mode0_btn", None),
            getattr(self, "robot_mode0_btn", None),
        ):
            if btn is not None:
                self._set_btn_prereq_enabled(
                    btn,
                    can_mode,
                    ok_tip=mode_tip,
                    blockers=[
                        "仿真遥控中无需切 mode" if sim else stack_hint
                    ],
                )
        for btn in (
            getattr(self, "left_hand_apply_btn", None),
            getattr(self, "right_hand_apply_btn", None),
        ):
            if btn is not None:
                self._set_btn_prereq_enabled(
                    btn,
                    can_apply,
                    ok_tip=apply_tip,
                    blockers=["请先「启用手」或进入仿真遥控"],
                )

    def _update_segment_invoke_btns_ui(self) -> None:
        """「调用分割 / SAM3」按选点与彩色图就绪控制可点性。"""
        if hasattr(self, "stereo_invoke_btn"):
            has_seed = self._has_segment_seed()
            self._set_btn_prereq_enabled(
                self.stereo_invoke_btn,
                has_seed,
                ok_tip=(
                    "对当前提示点执行立体分割 + 6D 位姿（跟随上方分割/位姿后端设置）。\n"
                    "点击图像仅选点，不会自动分割。"
                ),
                blockers=["请先点击彩色/深度图像选择提示点"],
            )
        if hasattr(self, "sam3_invoke_btn"):
            if getattr(self, "_sam3_call_busy", False):
                self.sam3_invoke_btn.setEnabled(False)
                self.sam3_invoke_btn.setToolTip("SAM3 调用中…")
            else:
                has_color = self._pick_sam3_color_source() is not None
                backend = (
                    self.segment_backend_combo.currentData()
                    if hasattr(self, "segment_backend_combo")
                    else None
                )
                need_seed = backend != _eai_configuration_perception.SAM3_BACKEND_TEXT
                has_seed = self._has_segment_seed()
                can = has_color and (has_seed or not need_seed)
                blockers: _eai_platform_bootstrap.List[str] = []
                if not has_color:
                    blockers.append("无可用彩色图像（请勾选 color topic 并等待图像）")
                if need_seed and not has_seed:
                    blockers.append("请先点击图像选择提示点")
                self._set_btn_prereq_enabled(
                    self.sam3_invoke_btn,
                    can,
                    ok_tip=(
                        "对当前彩色图像调用 SAM3 分割，并在下方显示结果；"
                        "成功时会在图像上叠加 mask"
                    ),
                    blockers=blockers,
                )

    def _update_prereq_gated_buttons(self, *_args) -> None:
        """统一刷新依赖先决条件的按钮可点状态。"""
        self._update_enable_control_btns_ui()
        self._update_segment_invoke_btns_ui()
        if hasattr(self, "replay_start_btn"):
            self._update_replay_ui()
        if hasattr(self, "real_eval_start_btn"):
            self._refresh_real_eval_status()

    def _update_robot_enable_status_ui(self) -> None:
        arm_received = self.node._arm_enable_received_at > 0
        hand_received = self.node._hand_enable_received_at > 0
        arm_enabled = self.node.is_arm_enabled()
        hand_enabled = self.node.is_hand_enabled()

        if not arm_received:
            arm_text = "手臂: 等待"
        else:
            arm_text = f"手臂: {'已使能' if arm_enabled else '未使能'}"
        self.robot_arm_enable_label.setText(arm_text)
        self._style_enable_label(self.robot_arm_enable_label, arm_received, arm_enabled)

        if not hand_received:
            hand_text = "手: 等待"
        else:
            hand_text = f"手: {'已使能' if hand_enabled else '未使能'}"
        self.robot_hand_enable_label.setText(hand_text)
        self._style_enable_label(self.robot_hand_enable_label, hand_received, hand_enabled)

        mode_label = self.node.get_control_mode_label()
        self.robot_control_mode_label.setText(mode_label)
        if self.node._control_mode_received_at <= 0:
            self.robot_control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        elif self.node._control_mode == _eai_configuration_chat.MODEL_CONTROL_MODE:
            self.robot_control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.robot_control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")

    def _update_hand_enable_ui(self) -> None:
        self.hand_enable_label.setText(self.node.get_hand_enable_label())
        received = self.node._hand_enable_received_at > 0
        enabled = self.node.is_hand_enabled()
        self._style_enable_label(self.hand_enable_label, received, enabled)
        btn_text = "关闭手" if enabled else "启用手"
        self.hand_enable_btn.setText(btn_text)
        if hasattr(self, "robot_hand_enable_btn"):
            self.robot_hand_enable_btn.setText(btn_text)

    def _update_enable_status_ui(self) -> None:
        if self._sim_teleop_active():
            self.arm_enable_label.setText("手臂: 仿真遥控")
            self._style_enable_label(self.arm_enable_label, True, True)
            self.hand_enable_label.setText("手: 仿真夹爪")
            self._style_enable_label(self.hand_enable_label, True, True)
            self.control_mode_label.setText("mode: sim-gui")
            self.control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            if hasattr(self, "robot_arm_enable_label"):
                self.robot_arm_enable_label.setText("手臂: 仿真遥控")
                self._style_enable_label(self.robot_arm_enable_label, True, True)
            if hasattr(self, "robot_hand_enable_label"):
                self.robot_hand_enable_label.setText("手: 仿真夹爪")
                self._style_enable_label(self.robot_hand_enable_label, True, True)
            if hasattr(self, "robot_control_mode_label"):
                self.robot_control_mode_label.setText("mode: sim-gui")
                self.robot_control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self._update_enable_control_btns_ui()
            return
        self._update_arm_enable_ui()
        self._update_hand_enable_ui()
        self._update_robot_enable_status_ui()
        mode_text = self.node.get_control_mode_label()
        self.control_mode_label.setText(mode_text)
        if self.node._control_mode_received_at <= 0:
            self.control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        elif self.node._control_mode == _eai_configuration_chat.MODEL_CONTROL_MODE:
            self.control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.control_mode_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        self._update_enable_control_btns_ui()

    def _try_finish_pending_arm_move(self) -> None:
        if self._pending_arm_move_goal is None or not self.node.is_arm_enabled():
            return
        goal = self._pending_arm_move_goal
        self._pending_arm_move_goal = None
        self._arm_enable_wait_timer.stop()
        msg = self.node.request_slow_move_to_goal(goal)
        if str(msg).startswith("无法") or str(msg).startswith("已有移动"):
            if self._text_cmd_queue:
                self._text_cmd_queue.clear()
                msg += "（已取消后续文本指令）"
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

    def _update_arm_enable_ui(self) -> None:
        self.arm_enable_label.setText(self.node.get_arm_enable_label())
        received = self.node._arm_enable_received_at > 0
        enabled = self.node.is_arm_enabled()
        self._style_enable_label(self.arm_enable_label, received, enabled)
        btn_text = "关闭手臂" if enabled else "启用手臂"
        self.arm_enable_btn.setText(btn_text)
        if hasattr(self, "robot_arm_enable_btn"):
            self.robot_arm_enable_btn.setText(btn_text)

    def _can_start_arm_move(self, arm_side: str = "left") -> bool:
        if self._is_sim_arm_moving() or self.node.is_slow_motion_busy():
            return True
        # 有 TCP 且能解析出目标即可点（相对模式需非零偏移；默认 ΔX=5cm）
        if self._sim_teleop_active():
            if self._sim_tcp_pose(arm_side) is None:
                return False
            return self._resolve_move_goal(arm_side) is not None
        # 真机：必须先使能，关闭后按钮灰掉且不可自动再开
        if not self.node.is_arm_enabled():
            return False
        state = self.node.get_robot_state()
        tcp = state.left_tcp if arm_side == "left" else state.right_tcp
        if tcp is None or not tcp.valid:
            return False
        return self._resolve_move_goal(arm_side) is not None

