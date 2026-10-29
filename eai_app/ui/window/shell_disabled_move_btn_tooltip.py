"""ui.window.shell disabled move btn tooltip — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellDisabledMoveBtnTooltipMixin:
    def _disabled_move_btn_tooltip(self, arm_side: str = "left") -> str:
        hints: _eai_platform_bootstrap.List[str] = []
        if self._sim_teleop_active():
            left_ok = self._sim_tcp_pose("left") is not None
            right_ok = self._sim_tcp_pose("right") is not None
            if not left_ok or not right_ok:
                hints.append(
                    "等待仿真 TCP（请「只开界面」或策略=无，等待 sim_robot_state.json）"
                )
            side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
            if self.move_target_segment_radio.isChecked():
                if self._last_segment_target is None:
                    hints.append("请先点击图像选点，再点「调用分割」")
                elif self._resolve_move_goal(arm_side) is None:
                    frame = self._last_segment_target.camera_frame or "未知"
                    hints.append(f"分割 TF 不可用 ({frame} -> {_eai_configuration_policies.IK_TARGET_FRAME})")
            elif self._resolve_move_goal(arm_side) is None:
                hints.append(f"相对当前模式：请为{side_name}设置非零偏移（如 ΔX=5 cm）")
            if not hints:
                hints.append("仿真遥控就绪：设置偏移后点「移动」")
            return "\n".join(hints)
        blockers = self.node.get_arm_move_blockers(tf_timeout_s=_eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S)
        if blockers:
            hints.append("点击后可能无法移动:")
            hints.extend(blockers)
        state = self.node.get_robot_state()
        tcp = state.left_tcp if arm_side == "left" else state.right_tcp
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        if tcp is None or not tcp.valid:
            hints.append(
                f"等待{side_name} TCP（需 host_fk / fk_node 发布 /mink_fk/*；点「连接机器人状态」或启用手臂会自动拉起）"
            )
        if self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                hints.append("请先点击图像选点，再点「调用分割」")
            elif self._resolve_move_goal(arm_side) is None:
                frame = self._last_segment_target.camera_frame or "未知"
                hints.append(f"分割 TF 不可用 ({frame} -> {_eai_configuration_policies.IK_TARGET_FRAME})")
        elif self._resolve_move_goal(arm_side) is None:
            hints.append(f"相对当前模式：请为{side_name}设置非零偏移（如 ΔX=5 cm）")
        if not self.node.is_arm_enabled():
            hints.insert(0, "请先点「启用手臂」")
        if not hints:
            hints.append("暂不可移动")
        return "\n".join(hints)

    def _update_arm_pose_display(self) -> None:
        if self._sim_teleop_active():
            parts: _eai_platform_bootstrap.List[str] = []
            for side in ("left", "right"):
                tcp = self._sim_tcp_pose(side)
                name = _eai_perception_pose_estimation.arm_side_label(side)
                if tcp is None:
                    parts.append(f"{name} TCP: (等待仿真)")
                else:
                    xyz, quat = tcp
                    parts.append(_eai_perception_pose_estimation.format_xyz_rpy_line(name, xyz, quat) + "  [sim]")
            self.arm_pose_current_label.setText("当前  " + "  |  ".join(parts))
        else:
            self.arm_pose_current_label.setText(self.node.get_arm_move_current_label_both())
        left_goal = self._resolve_move_goal("left")
        right_goal = self._resolve_move_goal("right")
        if left_goal is not None or right_goal is not None:
            parts = []
            if left_goal is not None:
                parts.append(
                    _eai_perception_pose_estimation.format_xyz_rpy_line(
                        "左", left_goal.position_xyz, left_goal.quaternion_xyzw
                    )
                )
            if right_goal is not None:
                parts.append(
                    _eai_perception_pose_estimation.format_xyz_rpy_line(
                        "右", right_goal.position_xyz, right_goal.quaternion_xyzw
                    )
                )
            label_hint = ""
            if left_goal is not None:
                label_hint = left_goal.label
            elif right_goal is not None:
                label_hint = right_goal.label
            self.arm_pose_target_label.setText(
                f"目标  [{label_hint}]  " + "  |  ".join(parts)
            )
        elif self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                self.arm_pose_target_label.setText("目标  TCP: (请先选点并调用分割/FP)")
            else:
                frame = self._last_segment_target.camera_frame or "未知"
                self.arm_pose_target_label.setText(
                    f"目标  TCP: TF 不可用 ({frame} -> {_eai_configuration_policies.IK_TARGET_FRAME})"
                )
        else:
            self.arm_pose_target_label.setText(
                "目标  TCP: (设置前/后/左/右/上/下偏移，base_link 系)"
            )

    def _schedule_robot_ui_refresh(self, *_args) -> None:
        now = _eai_platform_bootstrap.time.time()
        if now - self._robot_ui_last_update < _eai_configuration_policies.UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._robot_ui_last_update = now
        if self._sim_teleop_active():
            self._update_enable_status_ui()
        self._update_arm_pose_display()
        if not self.node.is_slow_motion_busy() and not self._is_sim_arm_moving():
            for side, btn in (
                ("left", self.left_arm_move_btn),
                ("right", self.right_arm_move_btn),
            ):
                can_move = self._can_start_arm_move(side)
                if btn.isEnabled() != can_move:
                    btn.setEnabled(can_move)
                    if not can_move:
                        btn.setToolTip(self._disabled_move_btn_tooltip(side))

    def _update_one_arm_move_btn_ui(self, arm_side: str, force: bool = False) -> None:
        btn = self.left_arm_move_btn if arm_side == "left" else self.right_arm_move_btn
        idle_style = (
            self._left_arm_move_btn_idle_style
            if arm_side == "left"
            else self._right_arm_move_btn_idle_style
        )
        cancel_style = (
            self._left_arm_move_btn_cancel_style
            if arm_side == "left"
            else self._right_arm_move_btn_cancel_style
        )
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        if self._is_sim_arm_moving():
            moving = str((self._sim_arm_move or {}).get("arm_side", ""))
            if moving == arm_side:
                btn.setText("取消仿真移动")
                btn.setEnabled(True)
                btn.setStyleSheet(cancel_style)
                btn.setToolTip("停止当前仿真手臂插值移动")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在仿真移动")
            return
        pending = self._pending_arm_move_goal
        if pending is not None and not self.node.is_arm_enabled():
            if pending.arm_side == arm_side:
                btn.setText("使能中…")
                btn.setEnabled(True)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("正在自动启用手臂，完成后将开始移动")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在等待使能")
            return
        if self.node.is_slow_motion_preparing():
            if self.node._slow_motion_moving_side == arm_side:
                btn.setText("取消准备")
                btn.setEnabled(True)
                btn.setStyleSheet(cancel_style)
                btn.setToolTip("取消正在进行的 IK 同步与轨迹规划")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在准备移动")
            return
        if self.node.is_slow_motion_active():
            if self.node._slow_motion_moving_side == arm_side:
                btn.setText("取消移动")
                btn.setEnabled(True)
                btn.setStyleSheet(cancel_style)
                btn.setToolTip("停止当前移动")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在移动")
            return
        btn.setText(f"{side_name}: 移动")
        btn.setStyleSheet(idle_style)
        can_move = self._can_start_arm_move(arm_side)
        btn.setEnabled(can_move)
        if can_move:
            if self._sim_teleop_active():
                extra = ""
                if (
                    not self.move_target_segment_radio.isChecked()
                    and self._resolve_move_goal(arm_side) is None
                ):
                    extra = "\n请设置非零偏移后再点击"
                btn.setToolTip(
                    f"仿真遥控：将{side_name} TCP 移到目标（策略=无）。\n"
                    f"目标写入共享目录 gui_robot_cmd.json，由当前仿真后端执行。{extra}"
                )
            else:
                blockers = self.node.get_arm_move_blockers(tf_timeout_s=_eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S)
                extra = ""
                if blockers:
                    extra = "\n注意: " + "；".join(blockers)
                elif (
                    not self.move_target_segment_radio.isChecked()
                    and self._resolve_move_goal(arm_side) is None
                ):
                    extra = "\n请设置非零偏移后再点击"
                btn.setToolTip(
                    f"将{side_name} TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
                    f"目标可为分割位姿，或相对当前位置的手动偏移。\n"
                    f"须先「启用手臂」。{extra}"
                )
        else:
            btn.setToolTip(self._disabled_move_btn_tooltip(arm_side))

    def _update_arm_move_btns_ui(self, force: bool = False) -> None:
        # Guard: may be invoked from sim eval signals during build_heavy_ui.
        if not hasattr(self, "left_arm_move_btn") or not hasattr(
            self, "_left_arm_move_btn_idle_style"
        ):
            return
        if not hasattr(self, "arm_move_speed_slider"):
            return
        speed_busy = self.node.is_slow_motion_busy() or self._is_sim_arm_moving()
        self.arm_move_speed_slider.setEnabled(not speed_busy)
        self._update_one_arm_move_btn_ui("left", force=force)
        self._update_one_arm_move_btn_ui("right", force=force)
        self._update_arm_repeat_btn()
        self._update_arm_dir_btns_ui()
        self._update_arm_pose_display()

    def _update_left_arm_move_btn_ui(self, force: bool = False) -> None:
        """兼容旧调用点。"""
        self._update_arm_move_btns_ui(force=force)

    def _store_segment_target(self, target: _eai_perception_pose_estimation.SegmentPoseTarget) -> None:
        self._last_segment_target = target
        self.move_target_segment_radio.setEnabled(True)
        self.move_target_segment_radio.setChecked(True)
        # 选中绝对位姿目标时，偏移量不参与目标计算；清零并隐藏六向
        for side in ("left", "right"):
            self._set_arm_dir_from_xyz_m(side, 0.0, 0.0, 0.0)
        self._update_move_offset_ui_visibility()
        self._update_arm_move_btns_ui()
        self._update_arm_pose_display()
        self.status_bar.showMessage(
            f"已记录绝对位姿目标: {target.label or target.source_topic}"
        )

    def _try_auto_move_to_segment_target(self) -> None:
        """Compatibility entry point: never execute motion from a pose callback."""
        self.status_bar.showMessage("位姿已记录；请显式选择手臂并确认移动，不再自动移动左臂")

    def _start_next_mv6d_target(self) -> None:
        if not self._mv6d_estimate_queue:
            self._mv6d_batch_active = False
            self._mv6d_batch_capture = None
            self._mv6d_batch_depth_cache = None
            self._mv6d_active_target = None
            self.mv6d_once_btn.setEnabled(True)
            self.mv6d_once_btn.setText('估计一次')
            successes = sum(1 for result in self._mv6d_target_results.values() if result.get('ok'))
            self.status_bar.showMessage(
                f'多视角 6D：批量估计完成，成功 {successes}/{len(self._mv6d_targets)}')
            self._refresh_mv6d_target_combo()
            return
        target_id = self._mv6d_estimate_queue.pop(0)
        target = next((item for item in self._mv6d_targets if item['id'] == target_id), None)
        if target is None:
            self._start_next_mv6d_target()
            return
        self._set_mv6d_active_target(target, show_result=False)
        self.mv6d_once_btn.setText(
            f"估计中 {len(self._mv6d_targets)-len(self._mv6d_estimate_queue)}/{len(self._mv6d_targets)}…")
        self._update_mv6d_batch_summary()
        self._start_mv6d_estimate(silent=False)

    def _finish_mv6d_target(self, result: dict) -> None:
        target = getattr(self, '_mv6d_active_target', None)
        if target is not None:
            stored = dict(result)
            stored['target'] = dict(target)
            self._mv6d_target_results[target['id']] = stored
        self._sync_mv6d_targets_to_grasp()
        self._refresh_mv6d_target_combo(target['id'] if target else '')
        if self._mv6d_batch_active:
            _eai_platform_bootstrap.QTimer.singleShot(0, self._start_next_mv6d_target)

    def _show_mv6d_target_result(self, result: dict) -> None:
        if not result.get('ok'):
            self.mv6d_result_edit.setPlainText('目标估计失败：' + str(result.get('error') or '未知错误'))
            return
        snap = result.get('snap')
        if not isinstance(snap, _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
            return
        self._mv6d_last_snap = snap
        poses = result.get('method_poses') or {}
        self._show_mv6d_on_robot_view(snap, silent=False, method_poses=poses)
        observations = result.get('observations') or {}
        if observations:
            target_id = str(result.get('target', {}).get('id') or '')
            if not self.grasp_execution_panel.select_target_observation(target_id):
                self.grasp_execution_panel.set_method_observations(_eai_platform_bootstrap.copy.deepcopy(observations))
        self._refresh_mv6d_result_text(snap)
        self._apply_mv6d_overlays(snap)

    def _sync_mv6d_targets_to_grasp(self) -> None:
        panel = getattr(self, 'grasp_execution_panel', None)
        if panel is None:
            return
        entries = {}
        for index, target in enumerate(self._mv6d_targets, 1):
            result = self._mv6d_target_results.get(target['id'])
            observations = (result or {}).get('observations') or {}
            if observations:
                entries[target['id']] = {
                    'label': f"目标 {index} ({target['u']}, {target['v']})",
                    'methods': observations,
                }
        panel.set_target_observations(entries)
        visuals = []
        backend = str(self.mv6d_method_combo.currentData() or _eai_configuration_perception.POSE_BACKEND_PCA)
        for target in self._mv6d_targets:
            result = self._mv6d_target_results.get(target['id']) or {}
            observations = result.get('observations') or {}
            observation = observations.get(backend) or next(iter(observations.values()), None)
            if not observation:
                continue
            raw = observation.get('T_base_object')
            if raw is None:
                try:
                    raw = (_eai_platform_bootstrap.np.asarray(observation['T_base_camera']) @
                           _eai_platform_bootstrap.np.asarray(observation['object_pose']['T_camera_object'])).tolist()
                except Exception:
                    continue
            transform = _eai_platform_bootstrap.np.asarray(raw, dtype=float)
            extents = observation.get('object_pose', {}).get('extents_m')
            visuals.append({'target_id':target['id'], 'xyz':transform[:3,3].tolist(),
                            'rotation':transform[:3,:3].tolist(), 'extents':extents})
        self._robot_target_poses = visuals
        view = getattr(self, '_robot_view', None)
        if view is not None and hasattr(view, 'set_target_poses'):
            view.set_target_poses(visuals)

    def _open_grasp_profile_calibration(self) -> None:
        tabs = getattr(self, "arm_workspace_tabs", None)
        panel = getattr(self, "grasp_profile_calib_panel", None)
        if tabs is None or panel is None:
            return
        for index in range(tabs.count()):
            candidate = tabs.widget(index)
            if isinstance(candidate, _eai_platform_bootstrap.QScrollArea) and candidate.widget() is panel:
                tabs.setCurrentIndex(index)
                return

    def _set_mv6d_active_target(self, target: dict, *, show_result: bool = True) -> None:
        self._mv6d_active_target = target
        self.mv6d_head_u_spin.blockSignals(True)
        self.mv6d_head_v_spin.blockSignals(True)
        self.mv6d_head_u_spin.setValue(int(target['u']))
        self.mv6d_head_v_spin.setValue(int(target['v']))
        self.mv6d_head_u_spin.blockSignals(False)
        self.mv6d_head_v_spin.blockSignals(False)
        self._mv6d_object_picked = True
        self._mv6d_update_seed_status_label()
        self._refresh_mv6d_live_previews_for_pick()
        if show_result and target['id'] in self._mv6d_target_results:
            self._show_mv6d_target_result(self._mv6d_target_results[target['id']])

    def _refresh_mv6d_target_combo(self, selected_id: str = '') -> None:
        combo = self.mv6d_target_combo
        selected_id = selected_id or str(combo.currentData() or '')
        combo.blockSignals(True)
        combo.clear()
        for index, target in enumerate(self._mv6d_targets, 1):
            result = self._mv6d_target_results.get(target['id'])
            state = '✓' if result and result.get('ok') else ('×' if result else '○')
            combo.addItem(f"{state} 目标 {index} ({target['u']}, {target['v']})", target['id'])
        wanted = combo.findData(selected_id)
        combo.setCurrentIndex(wanted if wanted >= 0 else combo.count()-1)
        combo.blockSignals(False)
        self.mv6d_remove_target_btn.setEnabled(bool(self._mv6d_targets) and not self._mv6d_busy)
        self.mv6d_clear_targets_btn.setEnabled(bool(self._mv6d_targets) and not self._mv6d_busy)
        self._update_mv6d_batch_summary()

    def _update_mv6d_batch_summary(self) -> None:
        label = getattr(self, 'mv6d_batch_summary_label', None)
        if label is None:
            return
        parts = []
        active_id = (getattr(self, '_mv6d_active_target', None) or {}).get('id')
        for index, target in enumerate(self._mv6d_targets, 1):
            result = self._mv6d_target_results.get(target['id'])
            if target['id'] == active_id and self._mv6d_busy:
                state = '估计中'
            elif result and result.get('ok'):
                snap = result.get('snap')
                xyz = getattr(snap, 'position_xyz', None)
                state = ('成功' if xyz is None else
                         f"成功 XYZ=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})")
            elif result:
                state = '失败：' + str(result.get('error') or '未知错误')[:80]
            else:
                state = '等待'
            parts.append(f"目标{index} {state}")
        label.setText('目标估计：' + ('；'.join(parts) if parts else '尚未选择目标'))

    def _remove_mv6d_target(self) -> None:
        if self._mv6d_busy:
            return
        target_id = str(self.mv6d_target_combo.currentData() or '')
        self._mv6d_targets = [item for item in self._mv6d_targets if item['id'] != target_id]
        self._mv6d_target_results.pop(target_id, None)
        self._sync_mv6d_targets_to_grasp()
        self._refresh_mv6d_target_combo()
        if self._mv6d_targets:
            self._on_mv6d_target_changed()
        else:
            self._mv6d_active_target = None
            self._mv6d_object_picked = False
            self._mv6d_update_seed_status_label()
            self._refresh_mv6d_live_previews_for_pick()

    def _clear_mv6d_targets(self) -> None:
        if self._mv6d_busy:
            return
        self._mv6d_targets.clear()
        self._mv6d_target_results.clear()
        self._sync_mv6d_targets_to_grasp()
        self._mv6d_active_target = None
        self._mv6d_object_picked = False
        self._refresh_mv6d_target_combo()
        self._mv6d_update_seed_status_label()
        self._refresh_mv6d_live_previews_for_pick()

    def _draw_mv6d_seed_marker(self, bgr: _eai_platform_bootstrap.np.ndarray, u: int, v: int) -> _eai_platform_bootstrap.np.ndarray:
        out = _eai_platform_bootstrap.np.ascontiguousarray(bgr.copy())
        h, w = out.shape[:2]
        uu = max(0, min(w - 1, int(u)))
        vv = max(0, min(h - 1, int(v)))
        _eai_platform_bootstrap.cv2.drawMarker(
            out,
            (uu, vv),
            (0, 220, 255),
            markerType=_eai_platform_bootstrap.cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )
        _eai_platform_bootstrap.cv2.circle(out, (uu, vv), 10, (0, 180, 255), 2, lineType=_eai_platform_bootstrap.cv2.LINE_AA)
        return out

    def _draw_mv6d_seed_markers(self, bgr: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
        out = _eai_platform_bootstrap.np.ascontiguousarray(bgr.copy())
        targets = getattr(self, '_mv6d_targets', [])
        active_id = (getattr(self, '_mv6d_active_target', None) or {}).get('id')
        for index, target in enumerate(targets, 1):
            out = self._draw_mv6d_seed_marker(out, target['u'], target['v'])
            color = (0, 255, 255) if target['id'] == active_id else (255, 180, 0)
            _eai_platform_bootstrap.cv2.putText(out, str(index), (int(target['u'])+10, int(target['v'])-10),
                        _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, _eai_platform_bootstrap.cv2.LINE_AA)
        return out

    def _draw_mv6d_target_results(self, bgr: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
        check = getattr(self, 'mv6d_overlay_check', None)
        if check is None or not check.isChecked():
            return bgr
        out = _eai_platform_bootstrap.np.ascontiguousarray(bgr.copy())
        h, w = out.shape[:2]
        colors = ((40,40,255),(40,230,80),(255,120,30),(30,210,255),(220,50,220),(220,220,40))
        for index, target in enumerate(getattr(self, '_mv6d_targets', []), 1):
            result = self._mv6d_target_results.get(target['id']) or {}
            snap = result.get('snap')
            view = getattr(snap, 'head_result', None)
            if not result.get('ok') or view is None:
                continue
            saved_k = getattr(view, 'intrinsics', None)
            if saved_k is None:
                continue
            mh, mw = view.mask.shape[:2]
            intr = (saved_k[0]*w/mw, saved_k[1]*h/mh, saved_k[2]*w/mw, saved_k[3]*h/mh)
            corners = _eai_platform_bootstrap.np.asarray(view.obb_corners, dtype=float)
            position = _eai_platform_bootstrap.np.asarray(view.position_xyz, dtype=float)
            rotation = _eai_platform_bootstrap.np.asarray(view.rotation_matrix, dtype=float)
            color = colors[(index-1) % len(colors)]
            _eai_adapters_robot_geometry.draw_obb_on_image(out, corners, *intr, color=color, thickness=2)
            axis_len = max(.025, float(max(view.obb_extents))*.45)
            _eai_adapters_robot_geometry.draw_pose_axes_on_image(out, position, rotation, *intr, axis_len)
            cu, cv = _eai_perception_image_geometry.scale_uv_to_shape(int(view.centroid_uv[0]), int(view.centroid_uv[1]),
                                       (mh,mw), (h,w))
            _eai_platform_bootstrap.cv2.putText(out, f'T{index}', (int(cu)+8,int(cv)-8),
                        _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX,.7,color,2,_eai_platform_bootstrap.cv2.LINE_AA)
        return out

    def _refresh_mv6d_live_previews_for_pick(self) -> None:
        """刷头部/双腕实时画面，便于点选物体（未估计时也显示；无帧保留占位）。"""
        if getattr(self, "mv6d_head_preview", None) is None:
            return
        if self._arm_panel_selector_key() != "mv6d":
            return
        if not self._control_tab_title_is_current("手臂/手"):
            return
        self.mv6d_head_preview.setVisible(True)
        pending = {}
        head_src = self._pick_mv6d_color_frame("/camera/head_color")
        head_bgr = None
        if head_src is not None:
            head_bgr = _eai_platform_bootstrap.np.asarray(head_src[1])
            self._mv6d_live_head_bgr = _eai_platform_bootstrap.np.ascontiguousarray(head_bgr)
            show = head_bgr
            if self._mv6d_object_picked:
                show = self._draw_mv6d_seed_markers(show)
            show = self._draw_mv6d_target_results(show)
            pending["/camera/head_color"] = show
        else:
            self._mv6d_live_head_bgr = None
        wrist_pending = []
        for side, lab, cache, empty in (
            (
                "left",
                self.mv6d_left_wrist_preview,
                "_mv6d_left_wrist_pix",
                "左腕 · 等待画面\n勾选 /camera/hand_left_color 或 left_wrist_color",
            ),
            (
                "right",
                self.mv6d_right_wrist_preview,
                "_mv6d_right_wrist_pix",
                "右腕 · 等待画面\n勾选 /camera/hand_right_color 或 right_wrist_color",
            ),
        ):
            lab.setVisible(True)
            src = self._pick_mv6d_color_frame(side)
            if src is None:
                self._set_mv6d_preview_label(lab, None, empty, cache_attr=cache)
                continue
            pending[src[0]] = src[1]
            wrist_pending.append((lab, cache, empty, src[0]))
        drawn = self._mv6d_overlay_hand_skeleton(pending)
        if head_bgr is not None:
            self._set_mv6d_preview_label(
                self.mv6d_head_preview,
                drawn.get("/camera/head_color", pending.get("/camera/head_color")),
                "头部 · 点击选择物体（无帧）",
                cache_attr="_mv6d_head_pix",
                source_wh=(int(head_bgr.shape[1]), int(head_bgr.shape[0])),
            )
        else:
            self._set_mv6d_preview_label(
                self.mv6d_head_preview,
                None,
                "头部 · 等待画面\n勾选 /camera/head_color，或仿真「只开界面」后重进本页",
                cache_attr="_mv6d_head_pix",
            )
        for lab, cache, empty, topic in wrist_pending:
            self._set_mv6d_preview_label(
                lab,
                drawn.get(topic, pending.get(topic)),
                empty.replace("等待画面", "点击放大"),
                cache_attr=cache,
            )

    def _pick_mv6d_color_frame(
        self, preferred: str
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray]]:
        candidates = self._mv6d_color_topic_candidates(preferred)
        for topic in candidates:
            frame = self._frame_cache.get(topic)
            if frame is None:
                panel = self.panels.get(topic)
                if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
                    frame = panel._latest_image
            if frame is None:
                # 尝试从仿真共享目录读 color npy / png（未跑仿真时放宽过期，便于手臂页仍能看到画面）
                age = 120.0 if self._any_sim_process_running() else 900.0
                frame = self._load_sim_bridge_color_bgr(topic, max_age_s=age)
            if frame is not None and _eai_platform_bootstrap.np.asarray(frame).ndim >= 2:
                return topic, _eai_platform_bootstrap.np.asarray(frame)
        return None

    def _load_sim_bridge_color_bgr(
        self,
        color_topic: str,
        *,
        max_age_s: float = 45.0,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        """仿真 headless：按 topic 约定读共享彩色帧（过期帧不读，避免 Isaac 残留污染 LIBERO）。"""
        try:
            root = self._sim_bridge_dir()
        except Exception:
            return None
        if not root or not _eai_platform_bootstrap.os.path.isdir(root):
            return None
        stem = color_topic.rstrip("/").split("/")[-1]
        # /camera/head_color → cam_head 或 head_color
        aliases = [stem, stem.replace("_color", ""), stem]
        cam_map = {
            "head_color": "cam_head",
            "left_wrist_color": "cam_left_wrist",
            "right_wrist_color": "cam_right_wrist",
        }
        if stem in cam_map:
            aliases.insert(0, cam_map[stem])
        now = _eai_platform_bootstrap.time.time()
        for name in aliases:
            for ext in (".npy", ".png", ".jpg"):
                path = _eai_platform_bootstrap.os.path.join(root, f"{name}{ext}")
                if not _eai_platform_bootstrap.os.path.isfile(path):
                    path = _eai_platform_bootstrap.os.path.join(root, "cameras", f"{name}{ext}")
                if not _eai_platform_bootstrap.os.path.isfile(path):
                    continue
                stamp = path + ".stamp" if ext == ".npy" else path
                try:
                    mtime = _eai_platform_bootstrap.os.path.getmtime(
                        stamp if _eai_platform_bootstrap.os.path.isfile(stamp) else path
                    )
                except OSError:
                    continue
                if max_age_s > 0 and (now - mtime) > float(max_age_s):
                    continue
                try:
                    if ext == ".npy":
                        arr = _eai_platform_bootstrap.np.load(path)
                        if arr.ndim == 2:
                            return _eai_platform_bootstrap.cv2.cvtColor(arr.astype(_eai_platform_bootstrap.np.uint8), _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
                        if arr.shape[-1] == 3:
                            return _eai_platform_bootstrap.np.asarray(arr)
                        return arr
                    img = _eai_platform_bootstrap.cv2.imread(path, _eai_platform_bootstrap.cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                except Exception:
                    continue
        return None

    def _refine_depth_with_lingbot_sync(
        self,
        color_bgr: _eai_platform_bootstrap.np.ndarray,
        *,
        topic: str,
        seed_u: int,
        seed_v: int,
        sparse_z_m: float,
        get_intrinsics,
        tag: str = "view",
        depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        depth_topic: str = "",
        model: str = _eai_configuration_tabs.LINGBOT_DEPTH_MODEL_DEFAULT,
        device: str = "auto",
        result_callback=None,
        cancel_event=None,
    ) -> _eai_platform_bootstrap.np.ndarray:
        """后台精修 RGB-D；无 depth 时使用稀疏先验，失败抛错且不回退。"""
        script = _eai_configuration_tabs.LINGBOT_DEPTH_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            raise RuntimeError(f"未找到 {script}")
        h, w = color_bgr.shape[:2]
        rgb_for_depth = color_bgr
        if depth is not None:
            raw_shape = _eai_platform_bootstrap.np.asarray(depth).shape
            if len(raw_shape) != 2:
                raise RuntimeError(f"原始深度形状异常: {raw_shape}")
            h, w = raw_shape
            if color_bgr.shape[:2] != (h, w):
                # Keep the sensor's pixel grid and intrinsics for every pose backend.
                rgb_for_depth = _eai_platform_bootstrap.cv2.resize(color_bgr, (w, h), interpolation=_eai_platform_bootstrap.cv2.INTER_LINEAR)
        fx, fy, cx, cy = get_intrinsics(depth_topic or topic, w, h)
        import tempfile
        work_root = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_DEPTH_CACHE_DIR, "mv6d")
        _eai_platform_bootstrap.os.makedirs(work_root, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix=f"{tag}_", dir=work_root)
        from eai_app.perception.depth_io import write_camera_inputs, validate_intrinsics, run_cancelable
        K = _eai_platform_bootstrap.np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=_eai_platform_bootstrap.np.float64)
        depth_path = ""
        try:
            if depth is not None:
                rgb_path, depth_path, K_path = write_camera_inputs(out_dir, rgb_for_depth, depth, K)
            else:
                validate_intrinsics(K)
                rgb_path = _eai_platform_bootstrap.os.path.join(out_dir, "rgb.png")
                K_path = _eai_platform_bootstrap.os.path.join(out_dir, "intrinsics.txt")
                if not _eai_platform_bootstrap.cv2.imwrite(rgb_path, rgb_for_depth):
                    raise OSError(f"无法写出 LingBot RGB: {rgb_path}")
                _eai_platform_bootstrap.np.savetxt(K_path, K, fmt="%.8f")
        except (ValueError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc

        cache_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "huggingface")
        _eai_platform_bootstrap.os.makedirs(cache_dir, exist_ok=True)

        env = _eai_platform_bootstrap.os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["LINGBOT_DEPTH_ROOT"] = _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_DEPTH_PYTHON_DEFAULT):
            env["LINGBOT_DEPTH_PYTHON"] = _eai_configuration_tabs.LINGBOT_DEPTH_PYTHON_DEFAULT
        env["HF_HOME"] = cache_dir
        env["HUGGINGFACE_HUB_CACHE"] = cache_dir

        cmd = [
            "bash",
            script,
            "--out",
            out_dir,
            "--model",
            model,
            "--device",
            device,
            "--cache-dir",
            cache_dir,
            "--rgb",
            rgb_path,
            "--intrinsics",
            K_path,
        ]
        if depth_path:
            cmd.extend(["--depth", depth_path])
        else:
            cmd.extend([
                "--allow-empty-depth", "--sparse-z", f"{float(sparse_z_m):.4f}",
                "--seed-u", str(int(seed_u)), "--seed-v", str(int(seed_v)),
            ])
        proc = run_cancelable(
            cmd,
            cancel_event=cancel_event,
            cwd=_eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT,
            env=env,
            timeout=600,
        )
        result: _eai_platform_bootstrap.Dict[str, object] = {}
        for stream in (proc.stdout or "", proc.stderr or ""):
            for line in stream.splitlines():
                if line.startswith("[RESULT] "):
                    try:
                        result = _eai_platform_bootstrap.json.loads(line[len("[RESULT] ") :].strip())
                    except _eai_platform_bootstrap.json.JSONDecodeError:
                        pass
        if proc.returncode != 0 or not result.get("ok"):
            err = str(result.get("error") or proc.stderr or f"exit={proc.returncode}")
            raise RuntimeError(f"LingBot-Depth 失败 ({tag}): {err}")

        npy_path = _eai_platform_bootstrap.os.path.join(out_dir, "depth_refined.npy")
        if not _eai_platform_bootstrap.os.path.isfile(npy_path):
            raise RuntimeError(f"LingBot-Depth 未写出 depth_refined.npy: {out_dir}")
        depth_m = _eai_platform_bootstrap.np.load(npy_path)
        if depth_m.shape != (h, w):
            raise RuntimeError(f"depth_refined 形状异常: {depth_m.shape}，应为 {(h, w)}")
        valid = _eai_platform_bootstrap.valid_depth_mask(depth_m)
        if not _eai_platform_bootstrap.np.any(valid):
            raise RuntimeError(f"LingBot-Depth 没有有效深度 ({tag})")
        if result_callback is not None:
            result_callback(dict(result, color_topic=topic, tag=tag))
        return _eai_platform_bootstrap.np.ascontiguousarray(_eai_platform_bootstrap.np.where(valid, depth_m, 0.0), dtype=_eai_platform_bootstrap.np.float32)

