"""ui.window.shell set segment preview — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellSetSegmentPreviewMixin:
    def _set_segment_preview(
        self,
        image_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        seed_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
        centroid_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
        obb_corners: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None,
        pose_position: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        pose_rotation: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        pose_axis_len: _eai_platform_bootstrap.Optional[float] = None,
    ) -> None:
        """在分割 Tab 右侧显示 mask / 位姿叠加缩略图。"""
        if image_bgr is None:
            self.sam3_preview_label.clear()
            self.sam3_preview_label.setText("分割预览")
            return
        has_mask = mask is not None and _eai_platform_bootstrap.np.asarray(mask).any()
        has_pose = obb_corners is not None or pose_position is not None
        if not has_mask and not has_pose:
            self.sam3_preview_label.clear()
            self.sam3_preview_label.setText("分割预览")
            return
        overlay = _eai_adapters_robot_geometry.apply_segment_overlay(
            image_bgr,
            mask,
            centroid_uv=centroid_uv,
            seed_uv=seed_uv,
            obb_corners=obb_corners,
            intrinsics=intrinsics,
            pose_position=pose_position,
            pose_rotation=pose_rotation,
            pose_axis_len=pose_axis_len,
        )
        max_w, max_h = 240, 160
        h, w = overlay.shape[:2]
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 0.999:
            overlay = _eai_platform_bootstrap.cv2.resize(
                overlay,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
            )
        self.sam3_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(overlay))
        self.sam3_preview_label.setText("")

    def _run_click_pose(self, topic: str, u: int, v: int, displayed=None) -> None:
        if self._click_pose_busy:
            self.status_bar.showMessage("点击估计进行中，请等待完成后重新选点")
            return
        panel = self.panels.get(topic)
        if not isinstance(panel, _eai_ui_cameras_panels.CameraPanel) or panel._latest_image is None:
            return
        try:
            displayed = panel._latest_image if displayed is None else displayed
            # Resolve calibration using full sensor resolution, not preview size.
            with self.node.click_frames.lock:
                frames = list(self.node.click_frames.frames.get(topic, ()))
            selected = next((f for f in reversed(frames) if f.displayed is displayed), None)
            if selected is None:
                raise ValueError("没有点击图像对应的实时 ROS 原始帧")
            entry, K = _eai_platform_bootstrap.click_pose.load_calibration(self.click_pose_calibration_edit.text().strip(), topic, selected.image.shape)
            if entry["depth_topic"] not in self.node.enabled_topics:
                self.node.set_enabled_topics(set(self.node.enabled_topics) | {entry["depth_topic"]})
                raise ValueError("已请求订阅标定指定的深度话题，请等待新帧后再次点击")
            color, depth_frame, skew = self.node.click_frames.pair(topic, entry["depth_topic"], displayed)
            if color.frame_id != entry["optical_frame"] or depth_frame.frame_id != entry["optical_frame"]:
                raise ValueError("RGB-D frame_id 与标定 optical_frame 不一致，无法确认已注册到同一相机")
            depth = _eai_platform_bootstrap.click_pose.metric_depth(depth_frame, color.image.shape)
            image = color.image.copy()
            ph, pw = displayed.shape[:2]
            h, w = image.shape[:2]
            su = min(w - 1, max(0, int((u + 0.5) * w / pw)))
            sv = min(h - 1, max(0, int((v + 0.5) * h / ph)))
            segment_cfg, pose_cfg = _eai_perception_segmentation.get_segment_settings(), _eai_perception_segmentation.get_pose_settings()
        except Exception as exc:
            self.click_pose_result_edit.setPlainText(_eai_platform_bootstrap.click_pose.format_report({"ok": False, "error": str(exc)}))
            self.status_bar.showMessage("点击估计未开始：请查看分割页数据检查结果")
            return
        self._click_pose_busy = True
        self.click_pose_result_edit.setPlainText("正在分割并估计物体位姿…（仅展示）")

        def work():
            try:
                notes = []
                if segment_cfg.backend == _eai_configuration_perception.SAM3_BACKEND_GEOMETRY:
                    mask, method = _eai_perception_image_geometry.segment_object_at_click(su, sv, depth, color_bgr=image)
                    notes.append("几何分割预览：精细目标边界建议选择 SAM3 点提示")
                else:
                    # Clicking selects one instance; no broad text-concept fallback.
                    mask, method = _eai_perception_segmentation.run_sam3_segmentation(image, su, sv, settings=segment_cfg, tag="calibrated_click")
                points, valid_mask, quality = _eai_platform_bootstrap.click_pose.object_points(mask, depth, K, (su, sv))
                intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                if pose_cfg.backend == _eai_configuration_perception.POSE_BACKEND_PCA:
                    pose = _eai_platform_bootstrap.click_pose.geometric_preview(points)
                    notes.append("可见表面 PCA：不是完整物体重建，轴方向存在符号/对称性歧义")
                else:
                    if pose_cfg.backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE:
                        if not pose_cfg.fp_mesh.strip():
                            raise ValueError("FoundationPose 需要目标自身的米制模型；不能复用其他物体 mesh")
                        body = _eai_perception_segmentation.run_foundationpose_estimation(image, depth, valid_mask, *intr, settings=pose_cfg, tag="calibrated_click")
                    else:
                        body = _eai_perception_pose_estimation.run_any6d_estimation(image, depth, valid_mask, *intr, settings=pose_cfg, tag="calibrated_click")
                    T = _eai_platform_bootstrap.click_pose.validate_transform(body["pose_matrix"])
                    pose = {"T_camera_object": T.tolist(), "method": str(body.get("method", pose_cfg.backend)),
                            "pose_kind": "model_estimate", "orientation_ambiguous": None}
                    if pose_cfg.backend == _eai_configuration_perception.POSE_BACKEND_ANY6D and not pose_cfg.fp_mesh.strip():
                        pose["pose_kind"] = "partial_hull_estimate"
                        notes.append("Any6D 当前无 CAD 适配为可见点云凸包，未实现官方完整参考图重建流程")
                result = {"ok": True, "topic": topic, "camera_frame": entry["optical_frame"],
                          "stamp": color.stamp, "seed_uv": [su, sv], "skew_ms": skew * 1000,
                          "captured_at": _eai_platform_bootstrap.time.time() - (_eai_platform_bootstrap.time.monotonic() - color.received),
                          "observation_id": f"{topic}:{color.stamp:.9f}:{su}:{sv}",
                          "quality": quality, "object_pose": pose, "notes": notes,
                          "grasp_candidates": [], "execution_ready": False}
                if entry.get("T_base_camera") is not None:
                    result["T_base_camera"] = entry["T_base_camera"]
                    result["T_base_object"] = (_eai_platform_bootstrap.click_pose.validate_transform(entry["T_base_camera"]) @
                                               _eai_platform_bootstrap.click_pose.validate_transform(pose["T_camera_object"])).tolist()
                    result["base_frame"] = entry["base_frame"]
                if entry.get("graspgen"):
                    try:
                        result["grasps"] = _eai_platform_bootstrap.click_pose.run_graspgen(points, entry["graspgen"])
                    except Exception as exc:
                        notes.append(f"抓取模型失败（保留物体估计）: {exc}")
                # Persist actual full-resolution inputs for reproducible model evaluation.
                directory = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "click_pose", str(_eai_platform_bootstrap.time.time_ns()))
                _eai_platform_bootstrap.os.makedirs(directory, exist_ok=True)
                _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(directory, "rgb.png"), image)
                _eai_platform_bootstrap.np.savez_compressed(_eai_platform_bootstrap.os.path.join(directory, "inputs.npz"), depth_m=depth,
                                    mask=mask, K=K, points_camera=points)
                with open(_eai_platform_bootstrap.os.path.join(directory, "result.json"), "w", encoding="utf-8") as stream:
                    _eai_platform_bootstrap.json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                result["notes"].append("输入与结果已保存: " + directory)
                result["mask"] = mask
                result["image"] = image
                result["K"] = K
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            self._click_pose_bridge.finished.emit(result)
        _eai_platform_bootstrap.threading.Thread(target=work, daemon=True).start()

    def _calib_last_click(self) -> dict:
        topic = getattr(self, "_last_sam3_topic", "") or ""
        panel = self.panels.get(topic) if hasattr(self, "panels") else None
        shape = None
        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and getattr(panel, "_latest_image", None) is not None:
            shape = panel._latest_image.shape[:2]
        u = int(self.sam3_u_spin.value()) if hasattr(self, "sam3_u_spin") else -1
        v = int(self.sam3_v_spin.value()) if hasattr(self, "sam3_v_spin") else -1
        return {"topic": topic, "u": u, "v": v, "preview_hw": shape}

    def _calib_tcp(self, side: str):
        path = _eai_platform_bootstrap.os.environ.get("EAI_HOST_FK_STATE", "/tmp/eai_host_fk_state.json")
        try:
            if _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.time.time() - _eai_platform_bootstrap.os.path.getmtime(path) < 2.0:
                with open(path, encoding="utf-8") as stream:
                    state = _eai_platform_bootstrap.json.load(stream)
                arm = state.get(side) if isinstance(state, dict) else None
                if isinstance(arm, dict) and arm.get("xyz") and arm.get("xyzw"):
                    waist_at = float(state.get("waist_received_at") or 0)
                    return {
                        "xyz": tuple(float(v) for v in arm["xyz"]),
                        "xyzw": tuple(float(v) for v in arm["xyzw"]),
                        "waist_age_s": (_eai_platform_bootstrap.time.time() - waist_at) if waist_at else None,
                    }
        except Exception:
            pass
        pose = self._tcp_for_relative_goal(side)
        if pose is None:
            return None
        return {"xyz": pose[0], "xyzw": pose[1], "waist_age_s": None}

    def _fill_grasp_observation_extrinsic(self, observation: dict) -> dict:
        """把相机→base_link 写入观测。优先用 TF，没有时用「手眼标定」页保存的结果。"""
        if not isinstance(observation, dict) or observation.get("T_base_camera") is not None:
            return observation
        frame = str(observation.get("camera_frame") or observation.get("input_frame") or "").lstrip("/")
        if not frame or frame in ("camera", "unknown"):
            observation["extrinsic_error"] = "观测没有相机 frame_id，无法查询手眼 TF"
            return observation
        if _eai_adapters_robot_geometry.normalize_frame_id(frame) == _eai_adapters_robot_geometry.normalize_frame_id(_eai_configuration_policies.BASE_LINK_FRAME):
            observation["T_base_camera"] = _eai_platform_bootstrap.np.eye(4).tolist()
            observation["base_frame"] = _eai_configuration_policies.BASE_LINK_FRAME
            observation["extrinsic_source"] = "already_base_link"
            observation.pop("extrinsic_error", None)
            return observation
        T = _eai_adapters_robot_geometry.lookup_transform_matrix(
            self.node.get_tf_buffer(), _eai_configuration_policies.BASE_LINK_FRAME, frame, timeout_s=0.4
        )
        if T is None:
            from eai_app.domain.calibration.cameras import apply_saved_extrinsic

            if apply_saved_extrinsic(observation, self._calib_tcp):
                return observation
            if not observation.get("extrinsic_error"):
                observation["extrinsic_error"] = (
                    f"TF 中没有 {frame} → {_eai_configuration_policies.BASE_LINK_FRAME}。可在「手眼标定」页手工标定后重试"
                )
            return observation
        try:
            observation["T_base_camera"] = _eai_platform_bootstrap.click_pose.validate_transform(T).tolist()
        except Exception as exc:
            observation["extrinsic_error"] = str(exc)
            return observation
        observation["base_frame"] = _eai_configuration_policies.BASE_LINK_FRAME
        observation["extrinsic_source"] = f"tf:{frame}"
        observation.pop("extrinsic_error", None)
        return observation

    def _open_visual_6d_for_grasp(self):
        for index in range(self.arm_workspace_tabs.count()):
            if self.arm_workspace_tabs.tabText(index) == "手动控制":
                self.arm_workspace_tabs.setCurrentIndex(index)
                break
        self.arm_panel_mv6d_radio.setChecked(True)
        self._apply_arm_panel_stack()

    def _control_grasp_simulation_hand(self, side, values):
        panel = self.grasp_execution_panel
        if panel.is_simulation and not panel._program_active:
            panel.simulation_backend().hand(side, values)

    def _phone_task_live_frame(self):
        frames=self.node.click_frames
        with frames.lock:
            bucket=list(frames.frames.get('/camera/head_color',()))
        if not bucket or _eai_platform_bootstrap.time.monotonic()-bucket[-1].received>.8:return None
        return bucket[-1].image,bucket[-1].received

    def _request_grasp_reestimate(self, target_id, completion):
        if self._mv6d_busy or getattr(self,'_mv6d_batch_active',False):
            raise ValueError('已有视觉估计正在运行，请等待完成')
        target=next((item for item in self._mv6d_targets if item['id']==target_id),None)
        if target is None:raise ValueError('当前任务目标不在视觉目标列表中')
        self._grasp_reestimate_pending=(target_id,completion)
        self._set_mv6d_active_target(target,show_result=False)
        self._start_mv6d_estimate(silent=False)
        if not self._mv6d_busy:
            self._finish_grasp_reestimate(None,'视觉估计未启动')

    def _finish_grasp_reestimate(self, observations, error=None, target_id=None):
        pending=getattr(self,'_grasp_reestimate_pending',None)
        if pending is None:return
        self._grasp_reestimate_pending=None
        if observations is not None and target_id!=pending[0]:
            observations=None;error='重估结果目标标识不匹配'
        _eai_platform_bootstrap.QTimer.singleShot(0,lambda:pending[1](observations,error))

    def _show_runtime_scene(self,scene,attachments):
        self._robot_collision_scene=(_eai_platform_bootstrap.copy.deepcopy(scene),_eai_platform_bootstrap.copy.deepcopy(attachments))
        view=getattr(self,'_robot_view',None)
        if view is not None:view.set_collision_scene(scene,attachments)

    def _verify_grasp_observation(self, observation):
        from grasp_perception import verify_live
        from eai_app.domain.calibration.cameras import STORE_PATH
        return verify_live(observation, self.node.click_frames, STORE_PATH)

    def _ensure_grasp_wrench_monitor(self):
        if self._grasp_wrench_monitor_started:
            return
        self._grasp_wrench_monitor_started = True
        self.node.enqueue_ros_op(self._setup_grasp_wrench_monitor_on_ros)

    def _setup_grasp_wrench_monitor_on_ros(self):
        try:
            from eai_app.ui.robot.viewer import import_arm_state
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            ArmState = import_arm_state()
            qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.VOLATILE)
            for side, topic in (('left', '/hal/left_arm_data'), ('right', '/hal/right_arm_data')):
                sub = self.node.create_subscription(
                    ArmState, topic,
                    lambda msg, bound=side: self._on_grasp_wrench(bound, msg), qos)
                self._grasp_wrench_subs.append(sub)
        except Exception as exc:
            with self._grasp_wrench_lock:
                for side in ('left', 'right'):
                    self._grasp_wrench_cache[side] = {
                        'force': None, 'torque': None, 'force_state': -3,
                        'stamp': _eai_platform_bootstrap.time.time(), 'error': str(exc),
                    }

    def _display_grasp_simulation(self, joints, hands):
        self._robot_grasp_simulation = (joints, hands)
        task_console=getattr(getattr(self,'grasp_execution_panel',None),'_program_active',False)
        if joints is not None and getattr(self, '_robot_view', None) is None and not task_console:
            self._on_mv6d_show_robot()
        view = getattr(self, '_robot_view', None)
        if view is not None:
            view.set_simulation_state(joints, hands)
        if joints is not None and not getattr(self, '_grasp_sim_display_active', False) and not task_console:
            self._on_mv6d_show_robot()
        self._grasp_sim_display_active = joints is not None

    def _show_grasp_preview_on_robot(self, preview):
        self._robot_grasp_preview = preview
        if preview is not None:
            self._on_mv6d_show_robot()
        view = getattr(self, '_robot_view', None)
        if view is not None:
            view.set_grasp_preview(preview)

    def _render_grasp_preview(self, preview):
        result = getattr(self, "_last_click_pose_snapshot", None)
        if preview["observation_id"].startswith("mv6d:"):
            # No invented camera projection for an already-fused base-frame pose.
            return _eai_platform_bootstrap.QPixmap()
        if result is None or result.get("observation_id") != preview["observation_id"]:
            raise ValueError("预览图像与候选观测不一致，请重新点击")
        canvas = result["image"].copy()
        K = _eai_platform_bootstrap.np.asarray(result["K"])
        T_camera_base = _eai_platform_bootstrap.np.linalg.inv(_eai_platform_bootstrap.np.asarray(result["T_base_camera"]))
        previous = None
        for name, color in (("pregrasp", (220, 80, 220)), ("approach", (60, 220, 60)), ("lift", (0, 220, 255))):
            T = T_camera_base @ _eai_platform_bootstrap.np.asarray(preview["goals"][name])
            points = _eai_platform_bootstrap.np.vstack([T[:3, 3], T[:3, 3]+.04*T[:3, :3].T])
            if _eai_platform_bootstrap.np.any(points[:, 2] <= .01):
                continue
            projected = points @ K.T
            uv = _eai_platform_bootstrap.np.rint(projected[:, :2]/projected[:, 2:]).astype(int)
            if previous is not None:
                _eai_platform_bootstrap.cv2.line(canvas, previous, tuple(uv[0]), color, 2)
            previous = tuple(uv[0])
            _eai_platform_bootstrap.cv2.circle(canvas, previous, 5, color, -1)
            _eai_platform_bootstrap.cv2.putText(canvas, name, previous, _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            for i, axis_color in enumerate(((0,0,255), (0,255,0), (255,0,0))):
                _eai_platform_bootstrap.cv2.line(canvas, previous, tuple(uv[i+1]), axis_color, 2)
        _eai_platform_bootstrap.cv2.putText(canvas, "TCP SNAPSHOT / NO COLLISION CHECK", (10, 25), _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX, .6, (0,200,255), 2)
        h, w = canvas.shape[:2]
        scale = min(480/w, 190/h)
        return _eai_adapters_robot_geometry.cv2_to_qpixmap(_eai_platform_bootstrap.cv2.resize(canvas, (max(1,int(w*scale)), max(1,int(h*scale)))))

    def _pick_chat_camera_frame(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray]]:
        """供 AI 对话附带视觉输入：优先最近点击的彩色 topic / 头部彩色。"""
        source = self._pick_sam3_color_source()
        if source is None:
            return None
        topic, image, _panel = source
        return topic, image

    def _pick_sam3_color_source(
        self,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        if self._last_sam3_topic:
            panel = self.panels.get(self._last_sam3_topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
                return (
                    self._last_sam3_topic,
                    panel._latest_image.copy(),
                    panel,
                )
        for topic, panel in sorted(self.panels.items()):
            if _eai_perception_image_geometry.is_depth_topic(topic) or not isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                continue
            if panel._latest_image is not None:
                return topic, panel._latest_image.copy(), panel
        for topic in sorted(self._frame_cache.keys()):
            if _eai_perception_image_geometry.is_depth_topic(topic):
                continue
            image = self._frame_cache.get(topic)
            if image is None or _eai_platform_bootstrap.np.asarray(image).ndim < 2:
                continue
            panel = self.panels.get(topic)
            return (
                topic,
                _eai_platform_bootstrap.np.asarray(image).copy(),
                panel if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) else None,
            )
        return None

    def _set_fp_result_text(self, text: str) -> None:
        self.fp_result_edit.setPlainText(text)

    def _pick_fp_stereo_source(
        self,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray, str, _eai_platform_bootstrap.np.ndarray, int, int]]:
        color_source = self._pick_sam3_color_source()
        if color_source is None:
            return None
        color_topic, color_bgr, _panel = color_source
        if not self._is_paired_depth_enabled(color_topic):
            return None
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if not depth_topic:
            return None
        depth = self._get_paired_depth_frame(color_topic)
        if depth is None or _eai_platform_bootstrap.np.asarray(depth).ndim < 2:
            return None
        depth = _eai_platform_bootstrap.np.asarray(depth)
        ch, cw = color_bgr.shape[:2]
        u = max(0, min(cw - 1, self.sam3_u_spin.value()))
        v = max(0, min(ch - 1, self.sam3_v_spin.value()))
        return color_topic, color_bgr, depth_topic, depth, u, v

    def _refresh_sam3_status(self) -> None:
        cfg = _eai_perception_segmentation.get_segment_settings()
        if cfg.backend == _eai_configuration_perception.SAM3_BACKEND_GEOMETRY:
            self.sam3_status_label.setText("SAM3: 关")
            self.sam3_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            return
        if cfg.sam3_use_http:
            ok = _eai_perception_segmentation.check_sam3_server_health(cfg.sam3_server_url)
            if ok:
                self.sam3_status_label.setText("SAM3: 在线")
                self.sam3_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            else:
                self.sam3_status_label.setText("SAM3: 离线")
                self.sam3_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
        else:
            self.sam3_status_label.setText("SAM3: 子进程")
            self.sam3_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")

    def _on_pose_settings_changed(self, *_args) -> None:
        backend = self.pose_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = _eai_configuration_perception.POSE_BACKEND_PCA
        is_learned = backend in (
            _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE,
            _eai_configuration_perception.POSE_BACKEND_ANY6D,
        )
        self.fp_mesh_edit.setEnabled(is_learned or backend == _eai_configuration_perception.POSE_BACKEND_PCA)
        self.fp_mesh_browse_btn.setEnabled(is_learned)
        self.fp_http_check.setEnabled(is_learned)
        self.fp_invoke_btn.setEnabled(not self._fp_call_busy)
        mesh_path = self.fp_mesh_edit.text().strip()
        if backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE and not mesh_path:
            mesh_path = _eai_configuration_perception.resolve_fp_mesh_path()
        use_http = self.fp_http_check.isChecked()
        fp_url = (
            _eai_perception_segmentation.resolve_fp_viewer_server_url() if use_http else _eai_configuration_perception.FP_SERVER_URL_DEFAULT
        )
        any6d_url = (
            _eai_perception_pose_estimation.resolve_any6d_viewer_server_url()
            if use_http
            else _eai_configuration_perception.ANY6D_SERVER_URL_DEFAULT
        )
        _eai_perception_segmentation.set_pose_settings(
            backend=backend,
            fp_use_http=use_http and backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE,
            fp_server_url=fp_url,
            fp_mesh=mesh_path,
            any6d_use_http=use_http and backend == _eai_configuration_perception.POSE_BACKEND_ANY6D,
            any6d_server_url=any6d_url,
        )
        self._refresh_fp_status()

    def _refresh_fp_status(self) -> None:
        cfg = _eai_perception_segmentation.get_pose_settings()
        use_http = self.fp_http_check.isChecked()
        if cfg.backend == _eai_configuration_perception.POSE_BACKEND_ANY6D:
            status = _eai_perception_pose_estimation.evaluate_any6d_availability(
                cfg.fp_mesh,
                use_http=use_http,
                server_url=cfg.any6d_server_url
                if use_http
                else _eai_configuration_perception.ANY6D_SERVER_URL_DEFAULT,
            )
            prefix = "Any6D"
            pose_hint = "选点后点「调用位姿」/「调用分割」走 Any6D（mesh 可空）"
        elif cfg.backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE:
            status = _eai_perception_segmentation.evaluate_fp_availability(
                cfg.fp_mesh,
                use_http=use_http,
                server_url=(
                    _eai_perception_segmentation.resolve_fp_viewer_server_url() if use_http else cfg.fp_server_url
                ),
            )
            prefix = "FP"
            pose_hint = "选点后点「调用位姿」走 FoundationPose（需 mesh）"
        else:
            status = _eai_perception_segmentation.FpAvailabilityStatus(
                label="PCA",
                color="#bd93f9",
                detail="点云几何",
                tooltip="位姿后端=PCA，不调用 FP/Any6D",
                can_invoke=True,
            )
            prefix = "位姿"
            pose_hint = "当前 PCA；可改选 FoundationPose / Any6D"
        self.fp_status_label.setText(f"{prefix}: {status.label}")
        self.fp_status_label.setStyleSheet(f"color: {status.color};")
        self.fp_status_label.setToolTip(status.tooltip)
        self.fp_avail_detail_label.setText(status.detail)
        self.fp_avail_detail_label.setToolTip(f"{status.tooltip}\n{pose_hint}")
        if not self._fp_call_busy:
            # PCA 也可点；learned 后端看 can_invoke
            self.fp_invoke_btn.setEnabled(
                True
                if cfg.backend == _eai_configuration_perception.POSE_BACKEND_PCA
                else bool(status.can_invoke)
            )

    def _on_stack_status_changed(self, robot: str, base: str) -> None:
        self._update_robot_stack_ui(robot, base)
        self._update_robot_enable_status_ui()
        self._update_prereq_gated_buttons()
        if base == "运行中":
            _eai_platform_bootstrap.QTimer.singleShot(
                1000,
                lambda: self._ensure_essential_robot_cameras(reason="机器人栈就绪"),
            )

    def _update_robot_stack_ui(
        self,
        robot: _eai_platform_bootstrap.Optional[str] = None,
        base: _eai_platform_bootstrap.Optional[str] = None,
    ) -> None:
        if robot is None or base is None:
            robot, base = self._stack_launcher.get_cached_status()
        self.robot_stack_status.setText(f"robot: {robot} | stack: {base}")
        if base == "运行中":
            self.robot_stack_status.setStyleSheet("color: #88ff88;")
        elif base == "启动中" or robot == "启动中":
            self.robot_stack_status.setStyleSheet("color: #ffcc66;")
        else:
            self.robot_stack_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        busy = base in ("运行中", "启动中") or robot == "启动中"
        self.robot_stack_btn.setEnabled(not busy)
        if base == "运行中":
            self.robot_stack_btn.setText("栈已运行")
        elif robot == "启动中":
            self.robot_stack_btn.setText("robot 启动中…")
        elif base == "启动中":
            self.robot_stack_btn.setText("服务栈启动中…")
        else:
            self.robot_stack_btn.setText("启动机器人栈")
        # 真机评测页：A2D 同步栈按钮；R1 / Franka 独立脚本路径时保持可点
        if hasattr(self, "real_eval_stack_btn"):
            backend = self._real_robot_backend_id()
            label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
            if backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D:
                self.real_eval_stack_btn.setEnabled(not busy)
                if base == "运行中":
                    self.real_eval_stack_btn.setText(f"{label}栈已运行")
                elif robot == "启动中":
                    self.real_eval_stack_btn.setText(f"{label} robot 启动中…")
                elif base == "启动中":
                    self.real_eval_stack_btn.setText(f"{label}服务栈启动中…")
                else:
                    self.real_eval_stack_btn.setText(f"启动{label}栈")
            elif not self._real_eval_active:
                self.real_eval_stack_btn.setText(f"启动{label}栈")

    def _refresh_host_min_node_status(self) -> None:
        """刷新最小栈各 node 运行状态（回放页详表 + 手臂页单行摘要）。"""
        try:
            rows = _eai_adapters_robot_services.probe_host_min_node_status()
            detail = _eai_adapters_robot_services.format_host_min_node_status_text(rows, compact=False)
            compact = _eai_adapters_robot_services.format_host_min_node_status_text(rows, compact=True)
        except Exception as exc:
            detail = f"节点状态查询失败: {exc}"
            compact = detail
            rows = []
        n_ok = sum(1 for _n, ok, _note in rows if ok)
        n_all = len(rows) if rows else 0
        header = f"节点 ({n_ok}/{n_all} 运行):\n" if n_all else "节点:\n"
        detail_full = header + detail
        color = (
            _eai_configuration_policies.UI_ACCENT_GREEN
            if n_ok == n_all and n_all > 0
            else (_eai_configuration_policies.UI_ACCENT_ORANGE if n_ok > 0 else _eai_configuration_policies.UI_TEXT_MUTED)
        )
        robot_lab = getattr(self, "robot_nodes_status_label", None)
        if robot_lab is not None:
            robot_lab.setText(compact)
            robot_lab.setStyleSheet(f"color: {color};")
            robot_lab.setToolTip(detail_full)
        arm_lab = getattr(self, "arm_nodes_status_label", None)
        if arm_lab is not None:
            arm_lab.setText(compact)
            arm_lab.setStyleSheet(f"color: {color};")
            arm_lab.setToolTip(detail_full)

    def _host_script_path(self, name: str) -> str:
        return _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT), "scripts", name
        )

    def _run_host_script_async(self, script_name: str, action: str) -> None:
        script = self._host_script_path(script_name)
        if not _eai_platform_bootstrap.os.path.isfile(script):
            return
        try:
            _eai_platform_bootstrap.subprocess.Popen(
                ["bash", script, action],
                stdout=_eai_platform_bootstrap.subprocess.DEVNULL,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    def _start_host_robot_min_stack(self) -> None:
        """连接机器人：启动看图 + 控手 + 控臂所需宿主机节点。"""
        self._run_host_script_async("start_host_robot_min.sh", "start")

    def _stop_host_robot_min_stack(self) -> None:
        """取消连接：停止最小宿主机节点（不关机器人本体）。"""
        self._run_host_script_async("start_host_robot_min.sh", "stop")

    def _on_connect_robot_state_clicked(self) -> None:
        if getattr(self, "_robot_connect_busy", False):
            return
        # 已连接 → 取消连接：停节点 + 清订阅
        if self.node.robot_subscriptions_ready():
            self._robot_connect_busy = True
            try:
                try:
                    if self.node.is_slow_motion_busy():
                        self.node.cancel_slow_motion()
                except Exception:
                    pass
                self._pending_arm_move_goal = None
                self._stop_robot_tcp_bridge()
                self._stop_host_robot_min_stack()
                self.node.clear_robot_subscriptions()
                self._apply_robot_connect_ui("未连接", ok=False)
                self.status_bar.showMessage(
                    "已取消连接：已停止相机/手/臂最小节点", 5000
                )
                self._update_arm_move_btns_ui()
                _eai_platform_bootstrap.QTimer.singleShot(400, self._refresh_host_min_node_status)
                _eai_platform_bootstrap.QTimer.singleShot(2000, self._refresh_host_min_node_status)
            finally:
                self._robot_connect_busy = False
            return
        self._robot_connect_busy = True
        try:
            # 标连接 → 后台起最小栈 → TCP bridge → 稍后勾选头/腕图
            self.node.begin_robot_subscription_steps()
            self._apply_robot_connect_ui("已连接", ok=True)
            self.status_bar.showMessage(
                "正在启动：相机 + 手 + 手臂（最小栈）…", 5000
            )
            self._start_host_robot_min_stack()
            _eai_platform_bootstrap.QTimer.singleShot(80, self._ensure_robot_tcp_bridge)
            _eai_platform_bootstrap.QTimer.singleShot(200, self._ensure_enable_state_subscriptions)
            # 等 forwarder 起来后再勾选头/腕彩色（避免点连接瞬间卡死）
            _eai_platform_bootstrap.QTimer.singleShot(
                1500,
                lambda: self._ensure_essential_robot_cameras(reason="连接机器人"),
            )
            _eai_platform_bootstrap.QTimer.singleShot(400, self._update_arm_move_btns_ui)
            _eai_platform_bootstrap.QTimer.singleShot(2500, self._update_arm_move_btns_ui)
            _eai_platform_bootstrap.QTimer.singleShot(500, self._refresh_host_min_node_status)
            _eai_platform_bootstrap.QTimer.singleShot(2000, self._refresh_host_min_node_status)
            _eai_platform_bootstrap.QTimer.singleShot(5000, self._refresh_host_min_node_status)
        finally:
            self._robot_connect_busy = False

    def _ensure_robot_tcp_bridge(self, *, force_restart: bool = False) -> None:
        stale = self._tcp_bridge_json_age_s() > 3.0
        proc = getattr(self, "_robot_tcp_bridge_proc", None)
        alive = proc is not None and proc.poll() is None
        if alive and not force_restart and not stale:
            self._ensure_robot_tcp_poll_timer()
            return
        # 残留/过期 bridge：杀掉后用与 host_fk 相同的 shm+udp DDS 重启
        if force_restart or stale or alive:
            self._stop_robot_tcp_bridge()
        else:
            try:
                if (
                    _eai_platform_bootstrap.subprocess.run(
                        ["pgrep", "-f", r"/usr/bin/python3\.10 .*robot_tcp_bridge\.py"],
                        capture_output=True,
                        check=False,
                    ).returncode
                    == 0
                ):
                    if not stale:
                        self._ensure_robot_tcp_poll_timer()
                        return
                    self._stop_robot_tcp_bridge()
            except Exception:
                pass
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT), "compat/robot_tcp_bridge.py"
        )
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_bar.showMessage("缺少 compat/robot_tcp_bridge.py")
            return
        env = _eai_platform_bootstrap.os.environ.copy()
        env["EAI_ROBOT_TCP_STATE"] = self._robot_tcp_state_path()
        # 必须与 host_fk 同 profile，否则订不到 /mink_fk/*（udp_only 易断发现）
        dds_xml = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "dds",
            "fastdds_profiles_shm_udp.xml",
        )
        if not _eai_platform_bootstrap.os.path.isfile(dds_xml):
            dds_xml = _eai_platform_bootstrap.os.path.join(
                str(PROJECT_ROOT),
                "dds",
                "fastdds_profiles_udp_only.xml",
            )
        if _eai_platform_bootstrap.os.path.isfile(dds_xml):
            env["FASTRTPS_DEFAULT_PROFILES_FILE"] = dds_xml
        env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        env.setdefault("ROS_DOMAIN_ID", "0")
        env.setdefault("ROS_LOCALHOST_ONLY", "1")
        # 保证能找到 librcl_*（未完整 source 时 import rclpy 会挂）
        ros_lib = "/opt/ros/humble/lib"
        if _eai_platform_bootstrap.os.path.isdir(ros_lib):
            env["LD_LIBRARY_PATH"] = ros_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        py_exec = "/usr/bin/python3.10" if _eai_platform_bootstrap.os.path.isfile("/usr/bin/python3.10") else _eai_platform_bootstrap.sys.executable
        try:
            self._robot_tcp_bridge_proc = _eai_platform_bootstrap.subprocess.Popen(
                [py_exec, script],
                env=env,
                stdout=_eai_platform_bootstrap.subprocess.DEVNULL,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            self.status_bar.showMessage(f"启动 TCP bridge 失败: {exc}")
            return
        self._ensure_robot_tcp_poll_timer()

    def _ensure_robot_tcp_poll_timer(self) -> None:
        timer = getattr(self, "_robot_tcp_poll_timer", None)
        if timer is not None:
            return
        timer = _eai_platform_bootstrap.QTimer(self)
        timer.setInterval(200)
        timer.timeout.connect(self._poll_robot_tcp_bridge)
        timer.start()
        self._robot_tcp_poll_timer = timer

    def _poll_robot_tcp_bridge(self) -> None:
        path = self._robot_tcp_state_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
        except Exception:
            # JSON 长时间缺失：尝试拉起 bridge
            last = float(getattr(self, "_tcp_bridge_restart_at", 0.0) or 0.0)
            if _eai_platform_bootstrap.time.time() - last > 5.0:
                self._tcp_bridge_restart_at = _eai_platform_bootstrap.time.time()
                self._ensure_host_fk()
                self._ensure_robot_tcp_bridge(force_restart=True)
            return
        if not isinstance(data, dict):
            return
        updated = float(data.get("updated_at") or 0.0)
        if updated > 0 and _eai_platform_bootstrap.time.time() - updated > 5.0:
            last = float(getattr(self, "_tcp_bridge_restart_at", 0.0) or 0.0)
            if _eai_platform_bootstrap.time.time() - last > 5.0:
                self._tcp_bridge_restart_at = _eai_platform_bootstrap.time.time()
                self._ensure_host_fk()
                self._ensure_robot_tcp_bridge(force_restart=True)
            return
        had_tcp = False
        try:
            st = self.node.get_robot_state()
            had_tcp = bool(
                (st.left_tcp and st.left_tcp.valid)
                or (st.right_tcp and st.right_tcp.valid)
            )
        except Exception:
            pass
        try:
            self.node.apply_bridged_tcp_state(data)
        except Exception:
            pass
        # TCP 从无到有时刷新一次按钮（避免每 200ms 刷 UI）
        if not had_tcp:
            try:
                self._update_arm_move_btns_ui()
            except Exception:
                pass

    def _stop_robot_tcp_bridge(self) -> None:
        timer = getattr(self, "_robot_tcp_poll_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._robot_tcp_poll_timer = None
        proc = getattr(self, "_robot_tcp_bridge_proc", None)
        self._robot_tcp_bridge_proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
        # 清掉残留 bridge（多次连接可能留下孤儿进程）
        try:
            _eai_platform_bootstrap.subprocess.run(
                ["pkill", "-f", r"/usr/bin/python3\.10 .*robot_tcp_bridge\.py"],
                capture_output=True,
                check=False,
                timeout=2.0,
            )
        except Exception:
            pass

    def _lazy_attach_head_camera_tf(self) -> None:
        try:
            self.node.ensure_head_camera_tf_lazy()
        except Exception:
            pass

    def _apply_robot_connect_ui(self, msg: str, *, ok: bool) -> None:
        for lab in (
            getattr(self, "robot_state_connect_label", None),
            getattr(self, "arm_robot_state_connect_label", None),
        ):
            if lab is None:
                continue
            if ok:
                lab.setText("状态: 已连接")
                lab.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            elif msg in ("未连接", ""):
                lab.setText("状态: 未连接")
                lab.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            else:
                lab.setText(f"状态: {msg}")
                lab.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
        for btn in (
            getattr(self, "robot_state_connect_btn", None),
            getattr(self, "arm_robot_state_connect_btn", None),
        ):
            if btn is None:
                continue
            if ok:
                btn.setText("取消连接")
                btn.setEnabled(True)
            else:
                btn.setText("连接机器人")
                btn.setEnabled(True)
        if msg:
            self.status_bar.showMessage(msg, 5000)

    def _ensure_host_camera_forwarder(self) -> None:
        """宿主机 a2d_sdk forwarder：无 Docker 时桥接机器人头/腕相机。"""
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_cameras_minimal.sh",
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

    def _ensure_essential_robot_cameras(self, *, reason: str = "") -> None:
        """扫描并勾选尚未被用户取消的头/腕彩色。"""
        self._ensure_host_camera_forwarder()
        edit = getattr(self, "prefix_edit", None)
        prefix = edit.text().strip() if edit is not None else ""
        prefix = prefix or "/camera"
        try:
            self.node.set_prefix(prefix)
            self.node.ensure_topic_scan()
        except Exception as exc:
            self.status_bar.showMessage(f"相机扫描失败: {exc}", 5000)
            return
        # 等 graph 刷新后再勾选
        _eai_platform_bootstrap.QTimer.singleShot(800, lambda: self._force_check_essential_cameras(reason))
        _eai_platform_bootstrap.QTimer.singleShot(2500, lambda: self._force_check_essential_cameras(reason))

