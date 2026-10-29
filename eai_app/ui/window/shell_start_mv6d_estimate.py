"""ui.window.shell start mv6d estimate — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')

class ShellStartMv6DEstimateMixin:
    def _start_mv6d_estimate(self, *, silent: bool = False) -> None:
        enabled_backends = self._mv6d_enabled_methods()
        if not enabled_backends:
            if not silent:
                self.status_bar.showMessage("请至少开启 PCA、FoundationPose、Any6D 中的一种方法")
            return
        try:
            self._ensure_mv6d_camera_feeds()
        except Exception:
            pass
        if self._mv6d_busy:
            if not silent:
                self.status_bar.showMessage("多视角 6D：正在估计中…")
            return
        head_src = self._pick_mv6d_color_frame("/camera/head_color")
        preferred_side = str(self.mv6d_wrist_combo.currentData() or "left").strip().lower()
        if preferred_side not in ("left", "right"):
            preferred_side = (
                "right"
                if "right" in preferred_side
                else "left"
            )
        preferred_wrist = self._resolve_wrist_color_topic(preferred_side)
        left_topic = self._resolve_wrist_color_topic("left")
        right_topic = self._resolve_wrist_color_topic("right")
        left_src = self._pick_mv6d_color_frame("left")
        right_src = self._pick_mv6d_color_frame("right")
        if head_src is None and left_src is None and right_src is None:
            if not silent and getattr(self, '_mv6d_last_snap', None) is None:
                self.mv6d_result_edit.setPlainText(
                    "错误: 未拿到头部或腕部彩色图。\n"
                    "请勾选 /camera/head_color 与 "
                    "/camera/hand_left_color|/camera/hand_right_color"
                    "（或 left_wrist/right_wrist），或启动带相机输出的仿真。"
                )
                self.status_bar.showMessage("多视角 6D：缺少彩色图")
            return

        head_topic, head_bgr = head_src if head_src else ("", None)
        left_wrist_topic, left_wrist_bgr = left_src if left_src else (left_topic, None)
        right_wrist_topic, right_wrist_bgr = (
            right_src if right_src else (right_topic, None)
        )
        head_depth = (
            self._get_paired_depth_frame(head_topic) if head_topic else None
        )
        left_wrist_depth = (
            self._get_paired_depth_frame(left_wrist_topic)
            if left_wrist_bgr is not None
            else None
        )
        right_wrist_depth = (
            self._get_paired_depth_frame(right_wrist_topic)
            if right_wrist_bgr is not None
            else None
        )
        selected_depth_frames = {}
        capture_sync_issues = []
        if not (getattr(self, '_mv6d_batch_active', False) and getattr(self, '_mv6d_batch_capture', None)):
            aligned = {}
            for label, topic, image, depth_image in (
                ('head', head_topic, head_bgr, head_depth),
                ('left', left_wrist_topic, left_wrist_bgr, left_wrist_depth),
                ('right', right_wrist_topic, right_wrist_bgr, right_wrist_depth)):
                if image is None or depth_image is None:
                    aligned[label] = depth_image
                    continue
                try:
                    _, depth_frame, _ = self.node.click_frames.pair(
                        topic, self._get_paired_depth_topic_name(topic) or '', image)
                    aligned[label] = depth_frame.image
                    selected_depth_frames[label] = depth_frame
                except Exception as exc:
                    aligned[label] = depth_image  # Diagnostic preview remains available.
                    capture_sync_issues.append(label+': '+str(exc))
            head_depth, left_wrist_depth, right_wrist_depth = aligned['head'], aligned['left'], aligned['right']
        batch_active = bool(getattr(self, '_mv6d_batch_active', False))
        batch_capture = getattr(self, '_mv6d_batch_capture', None) if batch_active else None
        if batch_capture is None and batch_active:
            def clone(value):
                return None if value is None else _eai_platform_bootstrap.np.ascontiguousarray(value).copy()
            batch_capture = {
                'head_topic': head_topic, 'head_bgr': clone(head_bgr), 'head_depth': clone(head_depth),
                'left_wrist_topic': left_wrist_topic, 'left_wrist_bgr': clone(left_wrist_bgr),
                'left_wrist_depth': clone(left_wrist_depth),
                'right_wrist_topic': right_wrist_topic, 'right_wrist_bgr': clone(right_wrist_bgr),
                'right_wrist_depth': clone(right_wrist_depth), 'preferred_wrist': preferred_wrist,
            }
            self._mv6d_batch_capture = batch_capture
        elif batch_capture is not None:
            head_topic, head_bgr, head_depth = (batch_capture['head_topic'], batch_capture['head_bgr'], batch_capture['head_depth'])
            left_wrist_topic, left_wrist_bgr, left_wrist_depth = (
                batch_capture['left_wrist_topic'], batch_capture['left_wrist_bgr'], batch_capture['left_wrist_depth'])
            right_wrist_topic, right_wrist_bgr, right_wrist_depth = (
                batch_capture['right_wrist_topic'], batch_capture['right_wrist_bgr'], batch_capture['right_wrist_depth'])
            preferred_wrist = batch_capture['preferred_wrist']
        use_lingbot = bool(
            getattr(self, "mv6d_lingbot_depth_check", None)
            and self.mv6d_lingbot_depth_check.isChecked()
        )
        launcher = getattr(self, "_lingbot_depth_launcher", None)
        if use_lingbot and launcher is not None and launcher.is_running():
            if not silent:
                self.status_bar.showMessage("独立深度精修正在运行，请等待或停止后再估计 6D")
            return
        cached_depth = getattr(self, '_mv6d_batch_depth_cache', None) if batch_active else None
        need_refine = use_lingbot and cached_depth is None
        lingbot_model = self._mv6d_lingbot_model_id() if use_lingbot else ""
        lingbot_device = (
            str(self.lingbot_depth_device_combo.currentData() or "auto")
            if use_lingbot else "auto"
        )
        any_wrist_depth = left_wrist_depth is not None or right_wrist_depth is not None
        rgb_only = head_depth is None and not any_wrist_depth and not need_refine
        if need_refine:
            if not silent:
                self.status_bar.showMessage(
                    "多视角 6D：LingBot-Depth 精修 → SAM3 → 已开启的位姿方法…"
                )
        elif rgb_only:
            if not silent:
                self.status_bar.showMessage(
                    "多视角 6D：无深度，改用 RGB-only（平面深度近似；LIBERO 默认无 depth）"
                )
        else:
            if head_bgr is not None and head_depth is None and not silent:
                self.status_bar.showMessage("多视角 6D：头部无深度 → 该视角 RGB-only")
            if not silent and (
                (left_wrist_bgr is not None and left_wrist_depth is None)
                or (right_wrist_bgr is not None and right_wrist_depth is None)
            ):
                self.status_bar.showMessage(
                    "多视角 6D：腕部无深度 → 看不见目标的腕不参与融合"
                )

        seed_u = int(self.mv6d_head_u_spin.value())
        seed_v = int(self.mv6d_head_v_spin.value())
        pose_backend = str(
            self.mv6d_method_combo.currentData() or _eai_configuration_perception.POSE_BACKEND_PCA
        )
        if pose_backend not in enabled_backends:
            pose_backend = enabled_backends[0]
        # 拷贝到线程，避免 UI 帧被覆盖
        head_bgr_c = None if head_bgr is None else _eai_platform_bootstrap.np.ascontiguousarray(head_bgr).copy()
        left_bgr_c = (
            None if left_wrist_bgr is None else _eai_platform_bootstrap.np.ascontiguousarray(left_wrist_bgr).copy()
        )
        right_bgr_c = (
            None if right_wrist_bgr is None else _eai_platform_bootstrap.np.ascontiguousarray(right_wrist_bgr).copy()
        )
        head_depth_c = None if head_depth is None else _eai_platform_bootstrap.np.ascontiguousarray(head_depth).copy()
        left_depth_c = (
            None if left_wrist_depth is None else _eai_platform_bootstrap.np.ascontiguousarray(left_wrist_depth).copy()
        )
        right_depth_c = (
            None
            if right_wrist_depth is None
            else _eai_platform_bootstrap.np.ascontiguousarray(right_wrist_depth).copy()
        )
        # RGB-only 假定深度必须是「相机光心到物体」的距离，不能用 base_link 下 TCP 的 Z
        # （后者曾导致回填绝对目标飞出工作区，越执行越远）
        assumed_z = 0.45

        head_depth_topic = (
            self._get_paired_depth_topic_name(head_topic) if head_topic else None
        ) or ""
        left_depth_topic = (
            self._get_paired_depth_topic_name(left_wrist_topic)
            if left_wrist_topic
            else None
        ) or ""
        right_depth_topic = (
            self._get_paired_depth_topic_name(right_wrist_topic)
            if right_wrist_topic
            else None
        ) or ""

        node = self.node
        tf_buffer = None
        try:
            tf_buffer = node.get_tf_buffer()
        except Exception:
            tf_buffer = None

        from grasp_perception import capture_provenance, CaptureTimeTF
        from eai_app.domain.calibration.cameras import STORE_PATH
        provenance = (batch_capture or {}).get('provenance')
        if provenance is None:
            try:
                provenance = capture_provenance(node.click_frames, {
                    'head': (head_topic, head_depth_topic, head_bgr),
                    'left': (left_wrist_topic, left_depth_topic, left_wrist_bgr if (use_lingbot or left_wrist_depth is not None) else None),
                    'right': (right_wrist_topic, right_depth_topic, right_wrist_bgr if (use_lingbot or right_wrist_depth is not None) else None),
                }, STORE_PATH, 'lingbot' if use_lingbot else 'sensor', tf_buffer, selected_depth_frames)
                provenance['issues'].extend(capture_sync_issues)
            except Exception as exc:
                provenance = {'version': 1, 'issues': [str(exc)], 'frames': {}, 'captured_at': _eai_platform_bootstrap.time.time()}
            if batch_capture is not None:
                batch_capture['provenance'] = _eai_platform_bootstrap.copy.deepcopy(provenance)
        if 'robot_q16' not in provenance:
            try:
                from eai_app.domain.grasp.execution import read_feedback
                provenance['robot_q16'] = read_feedback(_eai_platform_bootstrap.os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))['q16']
            except Exception as exc:
                provenance.setdefault('issues', []).append('采集时机器人反馈不可用：'+str(exc))
        capture_stamp = (provenance.get('frames', {}).get('head') or {}).get('stamp')
        if tf_buffer is not None and capture_stamp:
            tf_buffer = CaptureTimeTF(tf_buffer, capture_stamp)

        self._mv6d_busy = True
        if hasattr(self, "_on_mv6d_methods_changed"):
            self._on_mv6d_methods_changed()
        self._mv6d_lingbot_active = use_lingbot
        cancel_event = _eai_platform_bootstrap.threading.Event() if use_lingbot else None
        self._mv6d_lingbot_cancel = cancel_event
        if hasattr(self, "_update_lingbot_depth_ui"):
            self._update_lingbot_depth_ui()
        if not silent:
            self.mv6d_once_btn.setEnabled(False)
            self.mv6d_once_btn.setText("估计中…")
            method_name = {
                _eai_configuration_perception.POSE_BACKEND_PCA: "PCA",
                _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE: "FoundPose",
                _eai_configuration_perception.POSE_BACKEND_ANY6D: "Any6D",
            }.get(pose_backend, pose_backend)
            names = {_eai_configuration_perception.POSE_BACKEND_PCA: "PCA", _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE: "FoundationPose", _eai_configuration_perception.POSE_BACKEND_ANY6D: "Any6D"}
            active_names = "、".join(names[key] for key in enabled_backends)
            msg = f"正在用 SAM3 分割，再计算 {active_names}（结果文本用 {method_name}）…"
            if need_refine:
                msg = f"正在用 LingBot-Depth 补深度，随后 SAM3，并计算 {active_names}…"
            if getattr(self, '_mv6d_last_snap', None) is None:
                self.mv6d_result_edit.setPlainText(msg)
            else:
                self.status_bar.showMessage(msg + '（保留上次结果直到成功）')

        requested_target_id = (getattr(self, '_mv6d_active_target', None) or {}).get('id')
        def _work() -> None:
            t0 = _eai_platform_bootstrap.time.time()
            notes_extra: _eai_platform_bootstrap.List[str] = list(cached_depth[3]) if cached_depth is not None else []
            lingbot_results: _eai_platform_bootstrap.List[dict] = _eai_platform_bootstrap.copy.deepcopy(cached_depth[4]) if cached_depth is not None else []
            try:
                if cached_depth is not None:
                    hd, ld, rd = (None if value is None else _eai_platform_bootstrap.np.ascontiguousarray(value).copy()
                                  for value in cached_depth[:3])
                else:
                    hd, ld, rd = head_depth_c, left_depth_c, right_depth_c
                if need_refine:
                    refined = []
                    views = (
                        ("head", "头部", head_topic, head_depth_topic, head_bgr_c, hd, assumed_z),
                        ("left_wrist", "左腕", left_wrist_topic, left_depth_topic, left_bgr_c, ld, max(0.08, assumed_z * 0.7)),
                        ("right_wrist", "右腕", right_wrist_topic, right_depth_topic, right_bgr_c, rd, max(0.08, assumed_z * 0.7)),
                    )
                    for tag, label, topic, depth_topic, rgb, raw_depth, prior in views:
                        if cancel_event.is_set():
                            raise RuntimeError("LingBot-Depth 已停止，本轮 6D 已取消")
                        if rgb is None:
                            refined.append(None)
                            continue
                        ch, cw = rgb.shape[:2]
                        su = max(0, min(cw - 1, seed_u)) if tag == "head" else cw // 2
                        sv = max(0, min(ch - 1, seed_v)) if tag == "head" else ch // 2
                        try:
                            depth_m = self._refine_depth_with_lingbot_sync(
                                rgb, topic=topic, depth_topic=depth_topic, depth=raw_depth,
                                seed_u=su, seed_v=sv, sparse_z_m=prior,
                                get_intrinsics=node.get_intrinsics, tag=tag,
                                model=lingbot_model, device=lingbot_device,
                                result_callback=lingbot_results.append,
                                cancel_event=cancel_event,
                            )
                            if depth_m is None:
                                raise RuntimeError("未返回精修深度")
                        except Exception as exc:
                            raise RuntimeError(f"{label} LingBot-Depth 精修失败，本轮已停止：{exc}") from exc
                        refined.append(depth_m)
                        source_note = "原始 RGB-D 精修" if raw_depth is not None else f"稀疏先验 {prior:.3f}m 补全"
                        notes_extra.append(f"{label}深度 ← LingBot-Depth（{source_note}）")
                        if lingbot_results and lingbot_results[-1].get("tag") == tag:
                            info = lingbot_results[-1]
                            pixels = int(info.get("width", 0)) * int(info.get("height", 0))
                            if pixels and "input_valid_pixels" in info:
                                coverage = float(info["input_valid_pixels"]) / pixels
                                notes_extra.append(f"{label}原始有效深度覆盖率 {coverage:.1%}；其余为模型补全")
                    hd, ld, rd = refined
                    if batch_active:
                        self._mv6d_batch_depth_cache = (
                            None if hd is None else _eai_platform_bootstrap.np.ascontiguousarray(hd).copy(),
                            None if ld is None else _eai_platform_bootstrap.np.ascontiguousarray(ld).copy(),
                            None if rd is None else _eai_platform_bootstrap.np.ascontiguousarray(rd).copy(),
                            list(notes_extra), _eai_platform_bootstrap.copy.deepcopy(lingbot_results),
                        )
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("LingBot-Depth 已停止，本轮 6D 已取消")
                snap = self._compute_mv6d_snapshot(
                    head_topic=head_topic,
                    head_bgr=head_bgr_c,
                    head_depth=hd,
                    head_depth_topic=head_depth_topic,
                    left_wrist_topic=left_wrist_topic,
                    left_wrist_bgr=left_bgr_c,
                    left_wrist_depth=ld,
                    left_wrist_depth_topic=left_depth_topic,
                    right_wrist_topic=right_wrist_topic,
                    right_wrist_bgr=right_bgr_c,
                    right_wrist_depth=rd,
                    right_wrist_depth_topic=right_depth_topic,
                    preferred_wrist_topic=preferred_wrist,
                    seed_u=seed_u,
                    seed_v=seed_v,
                    pose_backend=pose_backend,
                    tf_buffer=tf_buffer,
                    get_intrinsics=node.get_intrinsics,
                    get_camera_frame=node.resolve_segment_camera_frame,
                    assumed_depth_m=assumed_z,
                )
                if notes_extra:
                    extra = "; ".join(notes_extra)
                    snap.notes = (
                        f"{snap.notes}; {extra}" if snap.notes else extra
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("已停止，本轮 6D 已取消")
                method_poses = {}
                try:
                    method_poses, pose_notes = self._collect_method_poses(
                        snap,
                        selected=pose_backend,
                        enabled_backends=enabled_backends,
                        head_bgr=head_bgr_c,
                        head_depth=hd,
                        head_topic=head_topic,
                        head_depth_topic=head_depth_topic,
                        seed_u=seed_u,
                        seed_v=seed_v,
                        get_intrinsics=node.get_intrinsics,
                    )
                    if pose_notes:
                        extra = "; ".join(pose_notes)
                        snap.notes = (
                            f"{snap.notes}; {extra}" if snap.notes else extra
                        )
                except Exception as exc:
                    snap.notes = (
                        f"{snap.notes}; 其余方法未画出：{exc}"
                        if snap.notes
                        else f"其余方法未画出：{exc}"
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("已停止，本轮 6D 已取消")
                if provenance.get('robot_q16') is not None:
                    try:
                        from eai_app.domain.grasp.execution import read_feedback
                        from eai_app.domain.motion.trajectory import exceeds
                        end_feedback = read_feedback(_eai_platform_bootstrap.os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))
                        if exceeds(end_feedback['q16'], provenance['robot_q16'], 'hold'):
                            provenance.setdefault('issues', []).append('估计期间机器人移动，需重新采集同步场景')
                    except Exception as exc:
                        provenance.setdefault('issues', []).append('估计结束时反馈不可用：'+str(exc))
                payload = {
                    "ok": True,
                    "snap": snap,
                    "method_poses": method_poses,
                    "captured_at": provenance['captured_at'],
                    "provenance": _eai_platform_bootstrap.copy.deepcopy(provenance),
                    "requested_target_id": requested_target_id,
                    "elapsed_s": _eai_platform_bootstrap.time.time() - t0,
                    "silent": silent,
                    "lingbot_notes": notes_extra,
                    "lingbot_results": lingbot_results,
                    "depth_source": "lingbot" if use_lingbot else "sensor",
                }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": str(exc),
                    "lingbot_results": lingbot_results,
                    "depth_source": "lingbot" if use_lingbot else "sensor",
                    "elapsed_s": _eai_platform_bootstrap.time.time() - t0,
                    "silent": silent,
                }
            self._mv6d_bridge.finished.emit(payload)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _compute_mv6d_snapshot(self, **kwargs):
        from eai_app.perception.multiview import compute_snapshot
        from types import SimpleNamespace
        services = SimpleNamespace(
            _estimate_view_6d=self._estimate_view_6d,
            _get_paired_depth_topic_name=self._get_paired_depth_topic_name,
            _mv6d_manual_extrinsic_matrix=self._mv6d_manual_extrinsic_matrix,
            _mv6d_refine_with_sim_gt=self._mv6d_refine_with_sim_gt,
            _mv6d_sim_extrinsic_matrix=self._mv6d_sim_extrinsic_matrix,
            _mv6d_try_wrist_view_if_visible=self._mv6d_try_wrist_view_if_visible,
        )
        return compute_snapshot(services, dependencies={
            "BASE_LINK_FRAME": _eai_configuration_policies.BASE_LINK_FRAME,
            "MV6D_EST_GT_SNAP_MAX_DIST_M": _eai_configuration_policies.MV6D_EST_GT_SNAP_MAX_DIST_M,
            "MV6D_FUSE_WEIGHT_HEAD": _eai_configuration_policies.MV6D_FUSE_WEIGHT_HEAD,
            "MV6D_FUSE_WEIGHT_WRIST": _eai_configuration_policies.MV6D_FUSE_WEIGHT_WRIST,
            "MultiView6DPoseSnapshot": _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
            "Object6DPoseResult": _eai_perception_pose_estimation.Object6DPoseResult,
            "POSE_BACKEND_ANY6D": _eai_configuration_perception.POSE_BACKEND_ANY6D,
            "POSE_BACKEND_FOUNDATIONPOSE": _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE,
            "POSE_BACKEND_PCA": _eai_configuration_perception.POSE_BACKEND_PCA,
            "estimate_6d_pose_pca": _eai_perception_pose_estimation.estimate_6d_pose_pca,
            "lookup_transform_matrix": _eai_adapters_robot_geometry.lookup_transform_matrix,
            "obb_corners": _eai_perception_pose_estimation.obb_corners,
            "quat_to_rotation_matrix": _eai_adapters_robot_geometry.quat_to_rotation_matrix,
            "refine_fused_object_points": _eai_perception_pose_estimation.refine_fused_object_points,
            "rotation_matrix_to_euler_xyz": _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz,
            "rotation_matrix_to_quaternion": _eai_perception_pose_estimation.rotation_matrix_to_quaternion,
            "transform_points_by_matrix": _eai_perception_pose_estimation.transform_points_by_matrix,
        }, **kwargs)

    def _refresh_mv6d_result_text(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot) -> None:
        gt_xyz, gt_R = self._mv6d_gt_pose()
        text = _eai_adapters_robot_geometry.format_multiview_6d_text(
            snap,
            gt_xyz=gt_xyz,
            gt_R=gt_R,
            gt_enabled=self.mv6d_gt_enable_check.isChecked(),
        )
        edit = self.mv6d_result_edit
        bar = edit.verticalScrollBar()
        selecting = edit.hasFocus() and edit.textCursor().hasSelection()
        if not selecting and not bar.isSliderDown() and edit.toPlainText() != text:
            scroll_value = bar.value()
            edit.setPlainText(text)
            bar.setValue(min(scroll_value, bar.maximum()))
            bar.raise_()
        if self.mv6d_gt_enable_check.isChecked() and gt_xyz is not None and gt_R is not None and _eai_adapters_robot_geometry.normalize_frame_id(snap.frame_id) == _eai_configuration_policies.BASE_LINK_FRAME:
            t_mm, r_deg, d_mm = _eai_perception_pose_estimation.pose_error_metrics(
                snap.position_xyz, snap.rotation_matrix, gt_xyz, gt_R
            )
            self.mv6d_error_label.setText(
                f"位姿参考差值（非 SAM3 分割误差） |Δt|={t_mm:.1f} mm  "
                f"(Δx={d_mm[0]:+.1f}, Δy={d_mm[1]:+.1f}, Δz={d_mm[2]:+.1f})  "
                f"|ΔR|={r_deg:.2f}°  |  frame={snap.frame_id}"
            )
            self.mv6d_error_label.setStyleSheet(
                f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE if t_mm > 20 or r_deg > 10 else _eai_configuration_policies.UI_ACCENT_BLUE};"
            )
        else:
            x, y, z = snap.position_xyz
            r, p, yw = snap.euler_rpy_rad
            self.mv6d_error_label.setText(
                f"估计 XYZ=({x:.4f}, {y:.4f}, {z:.4f}) m  "
                f"RPY=({_eai_platform_bootstrap.math.degrees(r):.1f}°, {_eai_platform_bootstrap.math.degrees(p):.1f}°, {_eai_platform_bootstrap.math.degrees(yw):.1f}°)  "
                f"| 参考位姿对比未启用或坐标系不一致"
            )
            self.mv6d_error_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
        self._sync_mv6d_lightbox()

    def _set_mv6d_preview_label(
        self,
        label: _eai_platform_bootstrap.QLabel,
        bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        placeholder: str,
        *,
        cache_attr: str = "",
        source_wh: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None,
    ) -> None:
        pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
        orig_wh = source_wh
        if bgr is not None and isinstance(bgr, _eai_platform_bootstrap.np.ndarray) and bgr.size > 0:
            h0, w0 = int(bgr.shape[0]), int(bgr.shape[1])
            if orig_wh is None:
                orig_wh = (w0, h0)
            small = self._mv6d_downscale_preview_bgr(bgr, 320)
            cand = _eai_adapters_robot_geometry.cv2_to_qpixmap(small, max_edge=320)
            if not cand.isNull():
                pix = cand
        if cache_attr:
            setattr(self, cache_attr, pix)
        if isinstance(label, _eai_ui_widgets_images.ScaledPixmapLabel):
            if pix is None:
                label.set_source_pixmap(None)
                label.setText(placeholder)
            else:
                # source_wh=原图像素，点选坐标映射回全分辨率
                label.set_source_pixmap(pix, source_wh=orig_wh)
            return
        if pix is None:
            label.setPixmap(_eai_platform_bootstrap.QPixmap())
            label.setText(placeholder)
            return
        label.setText("")
        label.setPixmap(
            pix.scaled(
                label.size() if label.width() > 8 else _eai_platform_bootstrap.QSize(320, 180),
                _eai_platform_bootstrap.Qt.KeepAspectRatio,
                _eai_platform_bootstrap.Qt.FastTransformation,
            )
        )

    def _open_mv6d_lightbox(self) -> None:
        win = self._mv6d_lightbox
        if win is not None:
            try:
                if win.isVisible() and not getattr(win, "_user_closed", False):
                    self._sync_mv6d_lightbox()
                    win.raise_()
                    win.activateWindow()
                    return
            except RuntimeError:
                self._mv6d_lightbox = None
                win = None
        win = _eai_ui_widgets_images.Mv6dLightboxWindow(self)
        win.closed_by_user.connect(self._on_mv6d_lightbox_closed)
        self._mv6d_lightbox = win
        self._sync_mv6d_lightbox()
        win.show()
        win.raise_()
        win.activateWindow()

    def _sync_mv6d_lightbox(self) -> None:
        win = getattr(self, "_mv6d_lightbox", None)
        if win is None:
            return
        try:
            if getattr(win, "_user_closed", False):
                return
            win.set_error_text(self.mv6d_error_label.text())
            win.set_views(
                getattr(self, "_mv6d_head_pix", None),
                getattr(self, "_mv6d_left_wrist_pix", None),
                getattr(self, "_mv6d_right_wrist_pix", None),
            )
        except RuntimeError:
            self._mv6d_lightbox = None

    def _resolve_mv6d_cam_pose_for_view(
        self,
        snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
        color_topic: str,
        depth_topic: str,
        view_result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray], str]:
        """优先把融合位姿变到该相机系；失败则退回该视角自己的估计。"""
        axis_note = snap.method
        if view_result is not None and (
            not snap.tf_ok or _eai_adapters_robot_geometry.normalize_frame_id(snap.frame_id) in ("", "camera")
        ):
            return (
                _eai_platform_bootstrap.np.asarray(view_result.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                _eai_platform_bootstrap.np.asarray(view_result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                _eai_platform_bootstrap.np.asarray(view_result.obb_corners, dtype=_eai_platform_bootstrap.np.float32),
                f"{axis_note}/local",
            )
        try:
            tf_buffer = self.node.get_tf_buffer()
        except Exception:
            tf_buffer = None
        cam_frame = ""
        try:
            cam_frame = self.node.resolve_segment_camera_frame(color_topic, depth_topic)
        except Exception:
            cam_frame = ""
        if cam_frame and snap.frame_id and tf_buffer is not None:
            transformed = _eai_adapters_robot_geometry.transform_pose_xyz_R(
                snap.position_xyz,
                snap.rotation_matrix,
                source_frame=snap.frame_id,
                target_frame=cam_frame,
                tf_buffer=tf_buffer,
            )
            if transformed is not None:
                xyz_c, R_c = transformed
                # OBB corners: transform fused corners if available
                corners = None
                if snap.obb_corners is not None and len(snap.obb_corners):
                    T = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, cam_frame, snap.frame_id)
                    if T is not None:
                        corners = _eai_perception_pose_estimation.transform_points_by_matrix(
                            _eai_platform_bootstrap.np.asarray(snap.obb_corners, dtype=_eai_platform_bootstrap.np.float32), T
                        )
                return xyz_c, R_c, corners, f"{axis_note}->cam"
        if view_result is not None:
            return (
                _eai_platform_bootstrap.np.asarray(view_result.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                _eai_platform_bootstrap.np.asarray(view_result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                _eai_platform_bootstrap.np.asarray(view_result.obb_corners, dtype=_eai_platform_bootstrap.np.float32),
                f"{axis_note}/fallback",
            )
        return (
            _eai_platform_bootstrap.np.asarray(snap.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
            _eai_platform_bootstrap.np.asarray(snap.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
            _eai_platform_bootstrap.np.asarray(snap.obb_corners, dtype=_eai_platform_bootstrap.np.float32)
            if snap.obb_corners is not None
            else None,
            axis_note,
        )

    def _apply_mv6d_overlays(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot) -> None:
        """把 6D 坐标轴/OBB/数字渲染到手臂页预览（不写主相机面板）。"""
        err_t, err_r = self._mv6d_error_pair(snap)
        self._mv6d_last_snap = snap

        def _render_one(
            color_topic: str,
            view_result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
            title: str,
        ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
            if not color_topic:
                return None
            src = getattr(view_result, "source_bgr", None)
            if src is None:
                src = self._frame_cache.get(color_topic)
            panel = self.panels.get(color_topic)
            if src is None and isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                src = panel._latest_image
            if src is None:
                src = self._load_sim_bridge_color_bgr(color_topic)
            if src is None:
                return None
            src = _eai_platform_bootstrap.np.asarray(src)
            depth_topic = self._get_paired_depth_topic_name(color_topic) or color_topic
            ch, cw = src.shape[:2]
            depth = self._get_paired_depth_frame(color_topic)
            if depth is not None:
                dh, dw = _eai_platform_bootstrap.np.asarray(depth).shape[:2]
                fx, fy, cx0, cy0 = self.node.get_intrinsics(depth_topic, dw, dh)
                sx = float(cw) / float(max(1, dw))
                sy = float(ch) / float(max(1, dh))
                intr = (fx * sx, fy * sy, cx0 * sx, cy0 * sy)
            else:
                fx, fy, cx0, cy0 = self.node.get_intrinsics(color_topic, cw, ch)
                intr = (fx, fy, cx0, cy0)

            saved_k = getattr(view_result, "intrinsics", None)
            if saved_k is not None:
                mh, mw = view_result.mask.shape[:2]
                intr = (saved_k[0] * cw / mw, saved_k[1] * ch / mh,
                        saved_k[2] * cw / mw, saved_k[3] * ch / mh)

            xyz_c, R_c, corners, note = self._resolve_mv6d_cam_pose_for_view(
                snap, color_topic, depth_topic, view_result
            )
            mask = None
            centroid = None
            if view_result is not None:
                raw_mask = getattr(view_result, "raw_mask", None)
                mask = _eai_perception_image_geometry.resize_mask_to_shape(raw_mask if raw_mask is not None else view_result.mask, (ch, cw))
                centroid = _eai_perception_image_geometry.scale_uv_to_shape(
                    int(round(view_result.centroid_uv[0])),
                    int(round(view_result.centroid_uv[1])),
                    view_result.mask.shape[:2],
                    (ch, cw),
                )
                centroid = (float(centroid[0]), float(centroid[1]))
            extents = (
                view_result.obb_extents
                if view_result is not None
                else snap.obb_extents
            )
            axis_len = float(max(extents) * 0.55) if extents else 0.08
            hud = {
                "title": title,
                "method": note,
                "err_t_mm": err_t,
                "err_r_deg": err_r,
            }
            rendered = _eai_adapters_robot_geometry.render_mv6d_view_bgr(
                src,
                mask=mask,
                pose_xyz=xyz_c,
                pose_R=R_c,
                obb_corners_cam=corners,
                intrinsics=intr,
                axis_len=axis_len,
                title=title,
                method=note,
                err_t_mm=err_t,
                err_r_deg=err_r,
                centroid_uv=centroid,
            )

            # 不写回主相机面板的 segment_overlay：一旦挂上会每帧 compose，主线程必卡。
            # 6D 结果只画在手臂页三个预览框（及 lightbox）。
            return rendered

        left_topic = self._resolve_wrist_color_topic("left")
        right_topic = self._resolve_wrist_color_topic("right")
        wrist_topic = str(snap.wrist_topic or "")
        # 仅「可见且参与融合」的腕画 mask/轴；不可见腕不参考其分割结果
        left_result = getattr(snap, "left_wrist_result", None)
        right_result = getattr(snap, "right_wrist_result", None)
        if left_result is None and right_result is None and snap.wrist_result is not None:
            # 兼容旧快照
            if "left" in wrist_topic:
                left_result = snap.wrist_result
            elif "right" in wrist_topic:
                right_result = snap.wrist_result
        if wrist_topic and "left" in wrist_topic:
            left_topic = wrist_topic
        if wrist_topic and "right" in wrist_topic:
            right_topic = wrist_topic

        def _clear_wrist_panel_overlay(color_topic: str) -> None:
            panel = self.panels.get(color_topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                try:
                    panel.image_label.clear_segment_overlay()
                except Exception:
                    pass
            depth_topic = self._get_paired_depth_topic_name(color_topic) or ""
            depth_panel = self.panels.get(depth_topic) if depth_topic else None
            if isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D):
                try:
                    depth_panel.depth_preview.clear_segment_overlay()
                except Exception:
                    pass

        rendered_head = _render_one(snap.head_topic, snap.head_result, "HEAD 6D")
        _skel_images = {}
        if rendered_head is not None:
            _skel_images[str(snap.head_topic)] = rendered_head
        # 不可见腕：只显示画面，不叠加伪位姿（避免“参考了不可见腕”的错觉）
        if left_result is not None:
            rendered_left = _render_one(left_topic, left_result, "L-WRIST 6D")
        else:
            _clear_wrist_panel_overlay(left_topic)
            src_l = self._pick_mv6d_color_frame(left_topic)
            rendered_left = None if src_l is None else _eai_platform_bootstrap.np.asarray(src_l[1]).copy()
            if rendered_left is not None:
                try:
                    _eai_platform_bootstrap.cv2.putText(
                        rendered_left,
                        "L-WRIST: skipped (not visible)",
                        (12, 28),
                        _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (80, 80, 220),
                        2,
                        _eai_platform_bootstrap.cv2.LINE_AA,
                    )
                except Exception:
                    pass
        if right_result is not None:
            rendered_right = _render_one(right_topic, right_result, "R-WRIST 6D")
        else:
            _clear_wrist_panel_overlay(right_topic)
            src_r = self._pick_mv6d_color_frame(right_topic)
            rendered_right = None if src_r is None else _eai_platform_bootstrap.np.asarray(src_r[1]).copy()
            if rendered_right is not None:
                try:
                    _eai_platform_bootstrap.cv2.putText(
                        rendered_right,
                        "R-WRIST: skipped (not visible)",
                        (12, 28),
                        _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (80, 80, 220),
                        2,
                        _eai_platform_bootstrap.cv2.LINE_AA,
                    )
                except Exception:
                    pass
        if rendered_left is not None:
            _skel_images[str(left_topic)] = rendered_left
        if rendered_right is not None:
            _skel_images[str(right_topic)] = rendered_right
        _skel_drawn = self._mv6d_overlay_hand_skeleton(_skel_images)
        rendered_head = _skel_drawn.get(str(snap.head_topic), rendered_head)
        rendered_left = _skel_drawn.get(str(left_topic), rendered_left)
        rendered_right = _skel_drawn.get(str(right_topic), rendered_right)
        self._set_mv6d_preview_label(
            self.mv6d_head_preview,
            rendered_head,
            "头部 · 等待画面\n勾选 /camera/head_color 或开仿真预览",
            cache_attr="_mv6d_head_pix",
        )
        self._set_mv6d_preview_label(
            self.mv6d_left_wrist_preview,
            rendered_left,
            "左腕 · 等待画面",
            cache_attr="_mv6d_left_wrist_pix",
        )
        self._set_mv6d_preview_label(
            self.mv6d_right_wrist_preview,
            rendered_right,
            "右腕 · 等待画面",
            cache_attr="_mv6d_right_wrist_pix",
        )
        # 始终保留三格预览，避免无帧时整行消失
        self.mv6d_head_preview.setVisible(True)
        self.mv6d_left_wrist_preview.setVisible(True)
        self.mv6d_right_wrist_preview.setVisible(True)
        self._mv6d_wrist_pix = self._mv6d_left_wrist_pix
        self._sync_mv6d_lightbox()

        # 写入仿真共享目录，便于外部查看 / 录屏对齐
        try:
            bridge = self._sim_bridge_dir()
            _eai_platform_bootstrap.os.makedirs(bridge, exist_ok=True)
            payload = {
                "frame_id": snap.frame_id,
                "xyz": list(snap.position_xyz),
                "quat_xyzw": list(snap.quaternion_xyzw),
                "rpy_rad": list(snap.euler_rpy_rad),
                "rpy_deg": [_eai_platform_bootstrap.math.degrees(v) for v in snap.euler_rpy_rad],
                "method": snap.method,
                "tf_ok": bool(snap.tf_ok),
                "head_topic": snap.head_topic,
                "wrist_topic": snap.wrist_topic,
                "left_wrist_topic": left_topic,
                "right_wrist_topic": right_topic,
                "fused_points": int(snap.fused_points),
                "timestamp": _eai_platform_bootstrap.time.time(),
            }
            if err_t is not None and err_r is not None:
                payload["error_t_mm"] = float(err_t)
                payload["error_r_deg"] = float(err_r)
                payload["error_kind"] = "pose_reference_difference"
            with open(
                _eai_platform_bootstrap.os.path.join(bridge, "object_6d_pose.json"), "w", encoding="utf-8"
            ) as f:
                _eai_platform_bootstrap.json.dump(payload, f, ensure_ascii=False, indent=2)
            if rendered_head is not None:
                _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(bridge, "mv6d_overlay_head.png"), rendered_head)
            if rendered_left is not None:
                _eai_platform_bootstrap.cv2.imwrite(
                    _eai_platform_bootstrap.os.path.join(bridge, "mv6d_overlay_left_wrist.png"), rendered_left
                )
                _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(bridge, "mv6d_overlay_wrist.png"), rendered_left)
            if rendered_right is not None:
                _eai_platform_bootstrap.cv2.imwrite(
                    _eai_platform_bootstrap.os.path.join(bridge, "mv6d_overlay_right_wrist.png"), rendered_right
                )
        except Exception:
            pass

