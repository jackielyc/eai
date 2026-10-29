"""ui.window.shell collect method poses — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_application_main_window = module_ref('eai_app.application.main_window')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ShellCollectMethodPosesMixin:
    def _collect_method_poses(
        self,
        snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
        *,
        selected: str,
        head_bgr,
        head_depth,
        head_topic: str,
        head_depth_topic: str,
        seed_u: int,
        seed_v: int,
        get_intrinsics,
        enabled_backends: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[str]] = None,
    ):
        """同一张 SAM3 掩码上计算已开启的方法，连同当前结果交给机器人页。"""
        enabled = tuple(_eai_configuration_perception.POSE_BACKENDS if enabled_backends is None else enabled_backends)
        labels = {
            _eai_configuration_perception.POSE_BACKEND_PCA: "PCA",
            _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE: "FoundPose",
            _eai_configuration_perception.POSE_BACKEND_ANY6D: "Any6D",
        }
        poses = {}
        notes: _eai_platform_bootstrap.List[str] = []
        selected_pose = self._mv6d_pose_in_base(snap)
        if selected_pose is not None and selected in enabled:
            poses[str(selected or _eai_configuration_perception.POSE_BACKEND_PCA)] = selected_pose
        result = snap.head_result
        if result is None or result.mask is None or head_bgr is None or head_depth is None:
            if selected_pose is None:
                notes.append("三种方法都缺少可放到 base_link 的位姿")
            return poses, notes
        depth = _eai_platform_bootstrap.np.asarray(head_depth)
        dh, dw = depth.shape[:2]
        raw_mask = getattr(result, "raw_mask", None)
        mask = _eai_platform_bootstrap.np.asarray(raw_mask if raw_mask is not None else result.mask)
        if mask.shape[:2] != (dh, dw):
            mask = _eai_perception_image_geometry.resize_mask_to_shape(mask, (dh, dw))
        ch, cw = _eai_platform_bootstrap.np.asarray(head_bgr).shape[:2]
        u_d, v_d = _eai_perception_image_geometry.scale_uv_to_shape(int(seed_u), int(seed_v), (ch, cw), (dh, dw))
        fx, fy, cx, cy = get_intrinsics(head_depth_topic or head_topic, dw, dh)
        for backend in enabled:
            if backend in poses:
                continue
            label = labels.get(backend, backend)
            try:
                view = self._pose_on_mask(
                    head_bgr,
                    depth,
                    mask,
                    u_d,
                    v_d,
                    fx,
                    fy,
                    cx,
                    cy,
                    backend,
                    method_tag="head",
                    seg_method="sam3",
                )
            except Exception as exc:
                notes.append(f"{label} 失败：{exc}")
                continue
            if view is None:
                notes.append(f"{label} 无结果")
                continue
            converted = self._camera_pose_to_base(snap, view)
            if converted is None:
                notes.append(f"{label} 缺少可用外参")
                continue
            poses[backend] = converted
        return poses, notes

    def _show_sam3_on_robot_view(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot, silent: bool = False) -> None:
        points, reason = self._mv6d_sam3_points_in_base(snap)
        if points is None:
            if reason and not silent:
                self.status_bar.showMessage(
                    "SAM3 分割点仍在相机系，缺少可用外参，机器人页未画出。", 8000
                )
            return
        self._robot_sam3_points = points
        view = getattr(self, "_robot_view", None)
        if view is not None and hasattr(view, "set_sam3_points"):
            view.set_sam3_points(points)

    def _show_mv6d_on_robot_view(
        self,
        snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
        silent: bool = False,
        method_poses=None,
    ) -> None:
        try:
            self._show_sam3_on_robot_view(snap, silent=silent)
        except Exception as exc:
            if not silent:
                self.status_bar.showMessage(f"机器人页未画出 SAM3：{exc}", 8000)
        poses = dict(method_poses or {})
        if not poses:
            converted = self._mv6d_pose_in_base(snap)
            if converted is not None:
                text = str(getattr(snap, "method", "") or "").lower()
                if "foundation" in text:
                    key = _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE
                elif "any6d" in text:
                    key = _eai_configuration_perception.POSE_BACKEND_ANY6D
                else:
                    key = _eai_configuration_perception.POSE_BACKEND_PCA
                poses[key] = converted
        enabled = _eai_application_main_window.CameraTopicWindow._mv6d_enabled_methods(self)
        poses = {key: pose for key, pose in poses.items() if key in enabled}
        if not poses:
            if not silent:
                self.status_bar.showMessage(
                    "位姿仍在相机系，缺少可用外参，机器人页未画出。", 8000
                )
            return
        stored = dict(getattr(self, "_robot_method_poses", {}) or {})
        stored.update(poses)
        self._robot_method_poses = stored
        view = getattr(self, "_robot_view", None)
        if view is not None and hasattr(view, "set_method_pose"):
            for key, pose in poses.items():
                view.set_method_pose(key, *pose)

    def _fill_arm_text_from_mv6d(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot) -> bool:
        """回填绝对目标：优先最近仿真物体中心（夹爪中点应对准物体中点），否则用 6D 估计。"""
        edit = getattr(self, "arm_text_cmd_edit", None)
        if edit is None:
            return False
        xyz_est, note = self._mv6d_xyz_in_base_link(snap)
        if xyz_est is None:
            self.status_bar.showMessage(note or "回填跳过：无有效 6D 估计", 8000)
            try:
                self.mv6d_result_edit.append(f"\n[回填跳过] {note}")
            except Exception:
                pass
            return False
        unreliable, why = self._mv6d_estimate_looks_unreliable(xyz_est, snap)
        xyz_gt, _quat_gt, gt_note = self._mv6d_refine_with_sim_gt(
            xyz_est,
            max_dist_m=float(_eai_configuration_policies.MV6D_FILL_GT_MAX_DIST_M),
            snap=snap,
        )
        if gt_note and xyz_gt is not None:
            xyz_base = (
                float(xyz_gt[0]),
                float(xyz_gt[1]),
                float(xyz_gt[2]),
            )
            note = f"{note}; 回填=仿真物体中心({gt_note})"
            try:
                self.mv6d_result_edit.append(
                    f"\n[回填] 6D→仿真物体中心 {gt_note} → "
                    f"({xyz_base[0]:.4f},{xyz_base[1]:.4f},{xyz_base[2]:.4f})"
                    f"（估计曾为 {xyz_est[0]:.4f},{xyz_est[1]:.4f},{xyz_est[2]:.4f}）"
                )
            except Exception:
                pass
        else:
            xyz_base = xyz_est
            note = f"{note}; 回填=6D估计"
            if unreliable:
                note = f"{note}; 警告:{why}"
            try:
                self.mv6d_result_edit.append(
                    f"\n[回填] 6D 估计 → "
                    f"({xyz_est[0]:.4f},{xyz_est[1]:.4f},{xyz_est[2]:.4f})"
                )
            except Exception:
                pass
        # 选离目标更近的臂回填（右腕 combo + 物体在左侧时，右臂误差会显得大很多）
        side = self._mv6d_cmd_arm_side_for_target(xyz_base)
        prefer = self._mv6d_cmd_arm_side()
        if side != prefer:
            note = f"{note}; 改用{_eai_perception_pose_estimation.arm_side_label(side)}(距目标更近，原偏好{_eai_perception_pose_estimation.arm_side_label(prefer)})"
        tcp = self._tcp_for_relative_goal(side)
        use_finger_mid = self._sim_uses_finger_mid_match()
        if use_finger_mid:
            note = f"{note}; 匹配点=手指结合中点"
        if tcp is not None:
            if use_finger_mid:
                cx, cy, cz = self._finger_mid_xyz_from_ee(tcp[0], tcp[1])
                cur_label = "手指中点"
            else:
                cx, cy, cz = tcp[0]
                cur_label = "TCP"
            dist = _eai_platform_bootstrap.math.sqrt(
                (xyz_base[0] - cx) ** 2
                + (xyz_base[1] - cy) ** 2
                + (xyz_base[2] - cz) ** 2
            )
            if dist > 0.85:
                msg = (
                    f"回填拒绝：目标距当前{_eai_perception_pose_estimation.arm_side_label(side)}{cur_label} "
                    f"约 {dist:.2f} m（目标 {xyz_base[0]:.3f},{xyz_base[1]:.3f},"
                    f"{xyz_base[2]:.3f}；{cur_label} {cx:.3f},{cy:.3f},{cz:.3f}）。"
                    f"多为仿真仍在用 A2D 真机 TF；请确认 Isaac 外参已写出并重新估计。"
                    f"（{note}）"
                )
                self.status_bar.showMessage(msg, 10000)
                try:
                    self.mv6d_result_edit.append(f"\n[回填拒绝] {msg}")
                except Exception:
                    pass
                return False
        text = _eai_platform_bootstrap.format_arm_abs_pose_text(side, xyz_base)
        # 若输入框里已有非本类指令，追加到末尾；若已是同侧绝对移到则替换
        prev = edit.text().strip()
        if prev:
            actions, _ = _eai_platform_bootstrap.parse_arm_hand_text(prev)
            only_same_abs = (
                len(actions) == 1
                and actions[0].kind == "arm"
                and actions[0].abs_xyz is not None
                and actions[0].side == side
            )
            if only_same_abs:
                edit.setText(text)
            else:
                # 去掉末尾重复的同侧绝对指令后再追加
                clauses = [c.strip() for c in _eai_platform_bootstrap.re.split(r"[\n；;]+", prev) if c.strip()]
                kept: _eai_platform_bootstrap.List[str] = []
                for c in clauses:
                    a, _ = _eai_platform_bootstrap.parse_arm_hand_text(c)
                    if (
                        len(a) == 1
                        and a[0].kind == "arm"
                        and a[0].abs_xyz is not None
                        and a[0].side == side
                    ):
                        continue
                    kept.append(c)
                kept.append(text)
                edit.setText("；".join(kept))
        else:
            edit.setText(text)
        edit.setCursorPosition(len(edit.text()))
        self.status_bar.showMessage(
            f"已回填{_eai_perception_pose_estimation.arm_side_label(side)}绝对目标（{note}）→ {text}", 5000
        )
        return True

    def _adopt_mv6d_segment_target(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot) -> bool:
        """把已变到 base_link 的 6D 位置记成「分割位姿」，单选框才可以选。"""
        if not hasattr(self, "move_target_segment_radio"):
            return False
        xyz, note = self._mv6d_xyz_in_base_link(snap)
        if xyz is None:
            return False
        quat = snap.quaternion_xyzw
        self._store_segment_target(
            _eai_perception_pose_estimation.SegmentPoseTarget(
                camera_frame=_eai_configuration_policies.BASE_LINK_FRAME,
                position_xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                quaternion_xyzw=(
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ),
                source_topic=snap.head_topic or "/camera/head_color",
                label=f"多视角6D ({note})",
                keep_tcp_orientation=True,
            )
        )
        return True

    def _save_grasp_task_data(self, task_key, panel_state):
        if not self._mv6d_targets or not self._mv6d_target_results:
            raise ValueError('没有可保存的多目标6D结果')
        from eai_app.infrastructure.observation_store import save_cache
        from eai_app.ui.grasp.workflow import TASKS
        saved_at = _eai_platform_bootstrap.time.time()
        payload = {
            'task_key': task_key,
            'saved_at': saved_at,
            'panel_state': _eai_platform_bootstrap.copy.deepcopy(panel_state),
            'task_definition': _eai_platform_bootstrap.copy.deepcopy(TASKS[task_key]),
            'targets': _eai_platform_bootstrap.copy.deepcopy(self._mv6d_targets),
            'target_results': _eai_platform_bootstrap.copy.deepcopy(list(self._mv6d_target_results.values())),
            'last_snap': self._mv6d_last_snap,
            'robot_method_poses': _eai_platform_bootstrap.copy.deepcopy(getattr(self, '_robot_method_poses', {})),
            'sam3_points': getattr(self, '_robot_sam3_points', None),
            'robot_lingbot_depth': _eai_platform_bootstrap.copy.deepcopy(getattr(self, '_robot_lingbot_depth', None)),
        }
        path = self._grasp_task_path(task_key)
        save_cache(path, payload)
        return '已保存图像、6D位姿和流程状态 → ' + path

    def _load_grasp_task_data(self, task_key):
        path = self._grasp_task_path(task_key)
        if not _eai_platform_bootstrap.os.path.isfile(path):
            self._mv6d_targets = []
            self._mv6d_target_results = {}
            self._mv6d_active_target = None
            self._mv6d_object_picked = False
            self._refresh_mv6d_target_combo()
            self._sync_mv6d_targets_to_grasp()
            view = getattr(self, '_robot_view', None)
            if view is not None and hasattr(view, 'set_target_poses'):
                view.set_target_poses([])
            return None
        from eai_app.infrastructure.observation_store import load_cache
        saved = load_cache(path, {
            'MultiView6DPoseSnapshot': _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
            'Object6DPoseResult': _eai_perception_pose_estimation.Object6DPoseResult,
        })
        if saved.get('task_key') != task_key:
            raise ValueError('任务缓存标识不匹配')
        self._mv6d_targets = [dict(target) for target in saved.get('targets') or []]
        results = saved.get('target_results') or []
        from grasp_perception import mark_cached
        for result in results:
            mark_cached(result.get('observations') or {})
        self._mv6d_target_results = {
            str(result.get('target', {}).get('id')): result
            for result in results if result.get('target', {}).get('id')
        }
        self._mv6d_target_serial = max(
            [int(str(target['id']).rsplit('-',1)[-1]) for target in self._mv6d_targets
             if str(target.get('id','')).rsplit('-',1)[-1].isdigit()] or [len(self._mv6d_targets)])
        self._mv6d_last_snap = saved.get('last_snap')
        self._robot_method_poses = saved.get('robot_method_poses') or {}
        self._robot_sam3_points = saved.get('sam3_points')
        self._robot_lingbot_depth = saved.get('robot_lingbot_depth')
        self._refresh_mv6d_target_combo()
        self._sync_mv6d_targets_to_grasp()
        if self._mv6d_targets:
            selected = str((saved.get('panel_state') or {}).get('selected_target') or '')
            index = self.mv6d_target_combo.findData(selected)
            self.mv6d_target_combo.setCurrentIndex(index if index >= 0 else 0)
            target_id = str(self.mv6d_target_combo.currentData() or '')
            target = next((item for item in self._mv6d_targets if item['id'] == target_id), None)
            if target is not None:
                self._set_mv6d_active_target(
                    target, show_result=hasattr(self, 'grasp_execution_panel'))
        saved_at = float(saved.get('saved_at') or 0.)
        return {
            'panel_state': saved.get('panel_state') or {},
            'task_definition': saved.get('task_definition'),
            'target_observations': {
                target_id: {'label': f"目标 {index} ({target['u']}, {target['v']})",
                            'methods': self._mv6d_target_results[target_id].get('observations') or {}}
                for index, target in enumerate(self._mv6d_targets, 1)
                for target_id in (target['id'],)
                if target_id in self._mv6d_target_results
            },
            'saved_at_text': (_eai_platform_bootstrap.time.strftime('%Y-%m-%d %H:%M:%S', _eai_platform_bootstrap.time.localtime(saved_at))
                              if saved_at else ''),
        }

    def _restore_mv6d_cache(self):
        path = self._mv6d_cache_path()
        if not _eai_platform_bootstrap.os.path.isfile(path) or self._mv6d_busy:
            return
        try:
            from eai_app.infrastructure.observation_store import load_cache
            cached = load_cache(path, {'MultiView6DPoseSnapshot': _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
                                       'Object6DPoseResult': _eai_perception_pose_estimation.Object6DPoseResult})
            snap = cached['snap']
            if not isinstance(snap, _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
                raise ValueError('缓存快照类型错误')
            self._mv6d_last_snap = snap
            self._robot_method_poses = cached['method_poses']
            self._robot_sam3_points = cached.get('sam3_points')
            self._robot_lingbot_depth = cached.get('robot_lingbot_depth')
            targets = cached.get('targets') or []
            target_results = cached.get('target_results') or []
            if targets:
                self._mv6d_targets = [dict(target) for target in targets]
                self._mv6d_target_results = {
                    str(result.get('target', {}).get('id')): result
                    for result in target_results
                    if result.get('target', {}).get('id')
                }
                self._mv6d_target_serial = max(
                    [int(str(target['id']).rsplit('-', 1)[-1]) for target in targets
                     if str(target.get('id', '')).rsplit('-', 1)[-1].isdigit()] or [len(targets)])
                self._refresh_mv6d_target_combo()
                self._sync_mv6d_targets_to_grasp()
                self._on_mv6d_target_changed()
            view = getattr(self, '_robot_view', None)
            if view is not None:
                for key, pose in self._robot_method_poses.items():
                    view.set_method_pose(key, *pose)
                view.set_sam3_points(self._robot_sam3_points)
                cloud = self._robot_lingbot_depth or {}
                if cloud.get('ply') and _eai_platform_bootstrap.os.path.isfile(cloud['ply']):
                    view.set_lingbot_depth_from_ply(cloud['ply'], color_topic=cloud.get('color_topic', ''))
            observations = cached['observations']
            from grasp_perception import mark_cached
            mark_cached(observations)
            for obs in observations.values():
                obs['source_label'] = obs.get('source_label', '') + ' / 历史缓存'
            self.grasp_execution_panel.set_method_observations(observations)
            self._refresh_mv6d_result_text(snap)
            # Put the cached source on screen immediately. Full overlay
            # rendering can depend on UI/TF helpers that are still starting.
            cached_head = getattr(getattr(snap, 'head_result', None), 'source_bgr', None)
            if cached_head is not None:
                source = _eai_platform_bootstrap.np.asarray(cached_head)
                self._set_mv6d_preview_label(
                    self.mv6d_head_preview, source,
                    '头部 · 历史缓存无图像', cache_attr='_mv6d_head_pix',
                    source_wh=(int(source.shape[1]), int(source.shape[0])),
                )
            try:
                self._apply_mv6d_overlays(snap)
            except Exception as exc:
                # Keep the cached source usable; live frames can refresh it too.
                self.status_bar.showMessage('已恢复6D缓存原图；叠加稍后重试：' + str(exc)[:120])
            stamp = _eai_platform_bootstrap.time.strftime('%Y-%m-%d %H:%M:%S', _eai_platform_bootstrap.time.localtime(cached['captured_at']))
            self.mv6d_result_edit.setPlainText('已加载历史缓存，估计时间：' + stamp + '\n' + self.mv6d_result_edit.toPlainText())
            self.status_bar.showMessage('已恢复上次6D结果；点击“估计一次”成功后更新。')
        except Exception as exc:
            self.status_bar.showMessage('6D缓存未加载：' + str(exc))

    def _pump_text_cmd_queue(self) -> str:
        if self._text_cmd_pumping:
            return ""
        self._text_cmd_pumping = True
        try:
            return self._pump_text_cmd_queue_body()
        finally:
            self._text_cmd_pumping = False

    def _pump_text_cmd_queue_body(self) -> str:
        notes: _eai_platform_bootstrap.List[str] = []
        while self._text_cmd_queue:
            action = self._text_cmd_queue[0]
            if action.kind == "stop":
                self._text_cmd_queue.clear()
                notes.append(self._cancel_arm_motion_for_text())
                break
            if action.kind == "hand":
                self._text_cmd_queue.pop(0)
                notes.append(self._execute_text_hand(action))
                continue
            if action.kind != "arm":
                self._text_cmd_queue.pop(0)
                continue
            if self._arm_text_motion_busy():
                notes.append(f"后续 {len(self._text_cmd_queue)} 段已排队")
                break
            self._text_cmd_queue.pop(0)
            status, msg = self._start_text_arm_action(action)
            notes.append(msg)
            if status == "busy":
                self._text_cmd_queue.insert(0, action)
                break
            if status == "fail":
                self._text_cmd_queue.clear()
                break
            if status == "skip":
                continue
            if self._text_cmd_queue:
                notes.append(f"完成后继续 {len(self._text_cmd_queue)} 段")
            break
        return "；".join(note for note in notes if note)

    def _cancel_arm_motion_for_text(self) -> str:
        self._pending_arm_move_goal = None
        self._arm_enable_wait_timer.stop()
        if self._is_sim_arm_moving():
            self._cancel_sim_arm_move()
            self._update_arm_move_btns_ui()
            return "已停止仿真手臂移动"
        if self.node.is_slow_motion_busy():
            msg = self.node.cancel_slow_motion()
            self._update_arm_move_btns_ui()
            return msg
        return "已停止"

    def _current_hand_vector(self, side: str) -> _eai_platform_bootstrap.List[float]:
        state = self.node.get_robot_state()
        joints = state.right_hand if side == "right" else state.left_hand
        out = [0.0] * len(_eai_platform_bootstrap.HAND_JOINT_ORDER)
        index = {name: i for i, name in enumerate(_eai_platform_bootstrap.HAND_JOINT_ORDER)}
        for name, val in joints:
            key = _eai_platform_bootstrap.canonical_hand_joint(name)
            if key is None or key not in index:
                continue
            out[index[key]] = max(0.0, min(1.0, float(val)))
        return out

    def _execute_text_hand(self, action: _eai_platform_bootstrap.ArmHandTextAction) -> str:
        sides = ["left", "right"] if action.side == "both" else [action.side]
        enable_note = ""
        if not self._sim_teleop_active() and not self.node.is_hand_enabled():
            enable_note = self.node.request_hand_enable(True)
        parts: _eai_platform_bootstrap.List[str] = []
        for side in sides:
            side_name = "右手" if side == "right" else "左手"
            if self._sim_teleop_active():
                if action.joints:
                    scalar = sum(float(v) for v in action.joints.values()) / len(action.joints)
                else:
                    scalar = 0.0 if action.position is None else float(action.position)
                scalar = max(0.0, min(1.0, scalar))
                ok = self._apply_sim_gripper(side, scalar)
                parts.append(
                    f"仿真{side_name} gripper={scalar:.2f}" + ("" if ok else "（写入失败）")
                )
                continue
            vals = self.node.apply_hand_joint_positions(
                side, self._hand_vector_for_action(side, action)
            )
            shown = ", ".join(f"{v:.2f}" for v in vals)
            parts.append(f"{side_name} [{shown}] -> {_eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC if side == 'right' else _eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC}")
        msg = "；".join(parts)
        if enable_note and "已使能" not in enable_note:
            msg = f"{enable_note}；{msg}"
        return msg

    def _start_text_arm_action(self, action: _eai_platform_bootstrap.ArmHandTextAction) -> _eai_platform_bootstrap.Tuple[str, str]:
        side = action.side if action.side in ("left", "right") else "left"
        remap_note = ""
        sim_cmd = self._sim_abs_cmd_ready()
        if sim_cmd:
            resolved = self._sim_resolve_arm_side(side)
            if resolved != side:
                remap_note = f"（{_eai_perception_pose_estimation.arm_side_label(side)}无 TCP → 改{_eai_perception_pose_estimation.arm_side_label(resolved)}）"
                side = resolved
        tcp = self._tcp_for_relative_goal(side)
        abs_xyz = getattr(action, "abs_xyz", None)
        if abs_xyz is not None:
            # 绝对目标：姿态优先用指令里的 quat，否则保持当前 TCP；
            # 仿真桥可用且尚无 TCP 时用单位四元数，直接写 gui_robot_cmd。
            # 仿真下 abs_xyz 视为「两指结合中点」匹配点，再换算成 EE/link6。
            abs_quat = getattr(action, "abs_quat_xyzw", None)
            if abs_quat is None:
                if tcp is not None:
                    abs_quat = tcp[1]
                elif sim_cmd:
                    abs_quat = (0.0, 0.0, 0.0, 1.0)
                else:
                    return (
                        "fail",
                        f"无法移动: 无{_eai_perception_pose_estimation.arm_side_label(side)} TCP，且仿真桥未就绪"
                        "（请确认 RoboDojo/评测在跑且写出 sim_robot_state）",
                    )
            match_xyz = (
                float(abs_xyz[0]),
                float(abs_xyz[1]),
                float(abs_xyz[2]),
            )
            use_finger_mid = self._sim_uses_finger_mid_match()
            if tcp is not None:
                if use_finger_mid:
                    cx, cy, cz = self._finger_mid_xyz_from_ee(tcp[0], tcp[1])
                    cur_label = "手指中点"
                else:
                    cx, cy, cz = tcp[0]
                    cur_label = "TCP"
                dist = _eai_platform_bootstrap.math.sqrt(
                    (match_xyz[0] - cx) ** 2
                    + (match_xyz[1] - cy) ** 2
                    + (match_xyz[2] - cz) ** 2
                )
                if dist > 0.85:
                    return (
                        "fail",
                        f"绝对目标距当前{cur_label} {dist:.2f} m 过大，已拒绝执行"
                        "（多为相机系坐标被当成 base_link；请重新估计并确认 TF）",
                    )
            if use_finger_mid:
                goal_xyz = self._ee_xyz_from_finger_mid_target(match_xyz, abs_quat)
                mid_note = (
                    f"手指中点({match_xyz[0]:.3f},{match_xyz[1]:.3f},{match_xyz[2]:.3f})"
                    f"→EE({goal_xyz[0]:.3f},{goal_xyz[1]:.3f},{goal_xyz[2]:.3f})"
                )
            else:
                goal_xyz = match_xyz
                mid_note = (
                    f"({match_xyz[0]:.3f}, {match_xyz[1]:.3f}, {match_xyz[2]:.3f}) m"
                )
            label = action.label or f"{_eai_perception_pose_estimation.arm_side_label(side)} → {mid_note}"
            if remap_note:
                label = f"{label}{remap_note}"
            goal = _eai_perception_pose_estimation.ResolvedArmMoveGoal(
                position_xyz=(
                    float(goal_xyz[0]),
                    float(goal_xyz[1]),
                    float(goal_xyz[2]),
                ),
                quaternion_xyzw=(
                    float(abs_quat[0]),
                    float(abs_quat[1]),
                    float(abs_quat[2]),
                    float(abs_quat[3]),
                ),
                label=label,
                arm_side=side,
            )
            # 绝对目标不走相对偏移 UI
            if hasattr(self, "move_target_segment_radio"):
                self.move_target_segment_radio.setEnabled(True)
                self.move_target_segment_radio.setChecked(True)
            for clear_side in ("left", "right"):
                self._set_arm_dir_from_xyz_m(clear_side, 0.0, 0.0, 0.0)
            self._update_move_offset_ui_visibility()
            self._update_arm_pose_display()
        else:
            if tcp is None:
                return "fail", f"无法移动: 无{_eai_perception_pose_estimation.arm_side_label(side)} TCP"
            xyz, quat = tcp
            goal = _eai_perception_pose_estimation.compute_relative_move_goal(
                xyz, quat, action.dx, action.dy, action.dz, arm_side=side
            )
            if goal is None:
                return "skip", "偏移为零，已跳过"
            goal = _eai_perception_pose_estimation.ResolvedArmMoveGoal(
                position_xyz=goal.position_xyz,
                quaternion_xyzw=goal.quaternion_xyzw,
                label=action.label or goal.label,
                arm_side=side,
            )
            self.move_target_relative_radio.setChecked(True)
            self._set_arm_dir_from_xyz_m(
                side, float(action.dx), float(action.dy), float(action.dz)
            )
            self._update_move_offset_ui_visibility()
            self._update_arm_pose_display()
        if sim_cmd or self._sim_teleop_active():
            msg = self._start_sim_arm_move(side, goal)
            self._update_arm_move_btns_ui()
            if str(msg).startswith("无法"):
                return "fail", msg
            return "ok", msg
        if self.node.is_slow_motion_busy() or self._pending_arm_move_goal is not None:
            return "busy", "已有移动进行中"
        if not self.node.is_arm_enabled():
            self._update_arm_move_btns_ui()
            return "fail", "手臂未使能（请先点「启用手臂」）"
        msg = self.node.request_slow_move_to_goal(goal)
        self._update_arm_move_btns_ui()
        if str(msg).startswith("已有移动"):
            return "busy", msg
        if str(msg).startswith("无法"):
            return "fail", msg
        return "ok", msg

    def _update_arm_repeat_btn(self) -> None:
        btn = getattr(self, "arm_repeat_btn", None)
        if btn is None:
            return
        if getattr(self, "_repeat_active", False):
            cycle = int(getattr(self, "_repeat_cycle", 0) or 0)
            btn.setText("取消重复")
            btn.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            btn.setEnabled(True)
            btn.setToolTip(f"停止重复执行（当前第 {cycle} 轮）")
            return
        btn.setText("重复执行")
        btn.setStyleSheet("")
        btn.setEnabled(True)
        btn.setToolTip(
            "左右臂按当前目标各移动一次，再回到界面启动时记下的初始关节位置，然后按间隔重复。\n"
            "回初始按上次下发的关节角插值，不依赖尚未更新的关节反馈。\n"
            "执行中再点一次取消。"
        )

    def _stop_arm_repeat(self, message: str, *, cancel_motion: bool) -> None:
        was_active = bool(getattr(self, "_repeat_active", False))
        self._repeat_active = False
        self._repeat_queue = []
        self._repeat_busy_retries = 0
        self._repeat_skip_retries = 0
        self._repeat_last_goal = None
        self._repeat_advance_pending = False
        wait_timer = getattr(self, "_repeat_wait_timer", None)
        if wait_timer is not None:
            wait_timer.stop()
        self._update_arm_repeat_btn()
        if cancel_motion and was_active:
            if self._is_sim_arm_moving():
                self._cancel_sim_arm_move("")
            elif self.node.is_slow_motion_busy():
                self.node.cancel_slow_motion()
        if message:
            self.status_bar.showMessage(message, 8000)
        self._update_arm_move_btns_ui()

    def _begin_repeat_cycle(self) -> None:
        if not self._repeat_active:
            return
        self._repeat_cycle += 1
        self._repeat_queue = list(self._repeat_out_goals) + list(self._repeat_home_goals)
        self._repeat_busy_retries = 0
        self._repeat_skip_retries = 0
        names = "、".join(_eai_perception_pose_estimation.arm_side_label(goal.arm_side) for goal in self._repeat_out_goals)
        self.status_bar.showMessage(f"重复执行第 {self._repeat_cycle} 轮：{names}", 5000)
        self._advance_repeat_queue()

    def _advance_repeat_queue(self) -> None:
        if not self._repeat_active:
            return
        if self._arm_text_motion_busy():
            self._repeat_busy_retries += 1
            if self._repeat_busy_retries > 25:
                self._stop_arm_repeat("重复执行中断：上一段移动未结束", cancel_motion=False)
                return
            self._schedule_repeat_advance()
            return
        self._repeat_busy_retries = 0
        if not self._repeat_queue:
            interval = 0.0
            spin = getattr(self, "arm_repeat_interval_spin", None)
            if spin is not None:
                interval = max(0.0, float(spin.value()))
            self.status_bar.showMessage(
                f"第 {self._repeat_cycle} 轮已回到初始位置，{interval:.1f} 秒后继续",
                5000,
            )
            self._repeat_wait_timer.start(int(round(interval * 1000.0)))
            return
        goal = self._repeat_queue.pop(0)
        self._kick_repeat_goal(goal)

    def _schedule_repeat_advance(self, delay_ms: int = 200) -> None:
        if self._repeat_advance_pending or not self._repeat_active:
            return
        self._repeat_advance_pending = True
        _eai_platform_bootstrap.QTimer.singleShot(max(0, int(delay_ms)), self._advance_repeat_queue_from_timer)

    def _repeat_goal_is_home(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> bool:
        return "回初始" in str(goal.label or "")

    def _repeat_arm_indexes(self, side: str):
        return range(9, 16) if side == "right" else range(2, 9)

    def _repeat_merge_commanded(self, q16, side: str) -> None:
        if not q16 or len(q16) < 16:
            return
        fresh = [float(v) for v in list(q16)[:16]]
        base = self._repeat_commanded_q16
        if not base or len(base) < 16:
            self._repeat_commanded_q16 = fresh
            return
        merged = [float(v) for v in base[:16]]
        for index in list(self._repeat_arm_indexes(side)) + [0, 1]:
            merged[index] = fresh[index]
        self._repeat_commanded_q16 = merged

    def _repeat_home_joints_far(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal, tol_rad: float = 0.02) -> bool:
        home = self._repeat_home_q16
        commanded = self._repeat_commanded_q16
        if not home or not commanded or len(home) < 16 or len(commanded) < 16:
            return False
        return any(
            abs(float(commanded[index]) - float(home[index])) > tol_rad
            for index in self._repeat_arm_indexes(goal.arm_side)
        )

    def _repeat_home_joint_target(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[float]]:
        home = self._repeat_home_q16
        commanded = self._repeat_commanded_q16
        if not home or not commanded or len(home) < 16 or len(commanded) < 16:
            return None
        target = [float(v) for v in commanded[:16]]
        for index in self._repeat_arm_indexes(goal.arm_side):
            target[index] = float(home[index])
        if not any(self._repeat_goal_is_home(item) for item in self._repeat_queue):
            target[0] = float(home[0])
            target[1] = float(home[1])
        return target

    def _repeat_tcp_near_goal(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal, tol_m: float = 0.015) -> bool:
        tcp = self._tcp_for_relative_goal(goal.arm_side)
        if tcp is None:
            return False
        dx = float(tcp[0][0]) - float(goal.position_xyz[0])
        dy = float(tcp[0][1]) - float(goal.position_xyz[1])
        dz = float(tcp[0][2]) - float(goal.position_xyz[2])
        return dx * dx + dy * dy + dz * dz <= tol_m * tol_m

    def _advance_repeat_queue_from_timer(self) -> None:
        self._repeat_advance_pending = False
        self._advance_repeat_queue()

    def _kick_repeat_goal(self, goal: _eai_perception_pose_estimation.ResolvedArmMoveGoal) -> None:
        sim = bool(self._sim_teleop_active() or self._sim_abs_cmd_ready())
        if sim:
            msg = self._start_sim_arm_move(goal.arm_side, goal)
        else:
            if not self.node.is_arm_enabled():
                self._stop_arm_repeat("手臂未使能，已停止重复", cancel_motion=False)
                return
            if self._repeat_goal_is_home(goal):
                target = self._repeat_home_joint_target(goal)
                start = self._repeat_commanded_q16
                if target is not None and start is not None:
                    msg = self.node.request_joint_space_move(start, target, goal)
                else:
                    msg = self.node.request_slow_move_to_goal(goal)
            else:
                if self._repeat_commanded_q16 is not None:
                    self.node._motion_seed_override = list(self._repeat_commanded_q16)
                msg = self.node.request_slow_move_to_goal(goal)
        self._repeat_last_goal = goal
        self._update_arm_move_btns_ui()
        text = str(msg)
        if text.startswith("无法"):
            self._repeat_queue.insert(0, goal)
            self._stop_arm_repeat(f"{text}（已停止重复）", cancel_motion=False)
            return
        if text.startswith("已有移动"):
            self._repeat_queue.insert(0, goal)
            self._schedule_repeat_advance()
            return
        if "无需移动" in text:
            self._on_repeat_step_finished(False, text)
            return
        self.status_bar.showMessage(
            f"重复第 {self._repeat_cycle} 轮：{goal.label}", 4000
        )
        if not self._arm_text_motion_busy():
            _eai_platform_bootstrap.QTimer.singleShot(0, lambda done=text: self._on_repeat_step_finished(True, done))

    def _on_repeat_wait_timeout(self) -> None:
        if not self._repeat_active:
            return
        self._begin_repeat_cycle()

    def _on_repeat_step_finished(self, ok: bool, text: str) -> None:
        if not self._repeat_active:
            return
        if not ok:
            goal = self._repeat_last_goal
            # 关节反馈还停在出发姿态时，笛卡尔回初始会被报成「无需移动」。
            # 这时不能当成已经回到初始位置。
            if "无需移动" in text and goal is not None:
                if self._repeat_goal_is_home(goal) and self._repeat_home_joints_far(goal):
                    self._repeat_skip_retries += 1
                    if self._repeat_skip_retries <= 8:
                        self._repeat_queue.insert(0, goal)
                        self._schedule_repeat_advance(400)
                        return
                    self._stop_arm_repeat(
                        "回初始失败：关节仍停在出发姿态（已停止重复）",
                        cancel_motion=False,
                    )
                    return
                if self._repeat_tcp_near_goal(goal) or (
                    self._repeat_goal_is_home(goal) and not self._repeat_home_joints_far(goal)
                ):
                    self._repeat_skip_retries = 0
                    self.status_bar.showMessage(f"{goal.label} 已在目标，继续重复", 3000)
                    self._schedule_repeat_advance(400)
                    return
                self._repeat_skip_retries += 1
                if self._repeat_skip_retries <= 8:
                    self._repeat_queue.insert(0, goal)
                    self._schedule_repeat_advance(400)
                    return
            self._stop_arm_repeat(f"{text}（已停止重复）", cancel_motion=False)
            return
        goal = self._repeat_last_goal
        commanded = getattr(self.node, "_commanded_q16", None)
        if goal is not None and isinstance(commanded, list):
            self._repeat_merge_commanded(commanded, goal.arm_side)
        self._repeat_skip_retries = 0
        # 等关节反馈跟上，再规划回初始，避免下一段被判成零位移。
        self._schedule_repeat_advance(500)

    def _after_arm_motion_settled(self, ok: bool, text: str) -> bool:
        if not getattr(self, "_repeat_active", False):
            return False
        self._on_repeat_step_finished(ok, text)
        return True

    def _on_left_arm_move_clicked(self) -> None:
        self._on_arm_move_clicked("left")

    def _on_slow_motion_progress(self, progress: float, text: str) -> None:
        self.status_bar.showMessage(text)
        self._update_arm_move_btns_ui()
        self._update_arm_pose_display()

