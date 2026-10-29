"""ui.window.multiview mv6d try wrist view if visible — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class MultiviewMv6DTryWristViewIfVisibleMixin:
    def _mv6d_try_wrist_view_if_visible(
        self,
        *,
        label: str,
        head_result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
        head_topic: str,
        head_depth_topic: str,
        wrist_topic: str,
        wrist_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        wrist_depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        wrist_depth_topic: str,
        tf_buffer,
        get_intrinsics,
        get_camera_frame,
        assumed_depth_m: float,
        head_center_base: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        max_center_dist_m: float = 0.12,
        pose_backend: str = _eai_configuration_perception.POSE_BACKEND_PCA,
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult], _eai_platform_bootstrap.List[str]]:
        """仅当头部目标在该腕部视野内时才估计；否则跳过，不参考该腕图像。"""
        notes: _eai_platform_bootstrap.List[str] = []
        if wrist_bgr is None:
            notes.append(f"{label}：无 RGB，跳过")
            return None, notes
        if wrist_depth is None:
            notes.append(f"{label}：无深度，跳过（不参考该腕图像）")
            return None, notes
        if head_result is None:
            notes.append(f"{label}：无头部目标，跳过")
            return None, notes

        wh, ww = wrist_bgr.shape[:2]
        dh, dw = wrist_depth.shape[:2]
        fx, fy, cx, cy = get_intrinsics(wrist_depth_topic or wrist_topic, dw, dh)
        wu, wv = self._mv6d_reproject_head_seed_to_wrist(
            head_result=head_result,
            head_topic=head_topic,
            head_depth_topic=head_depth_topic,
            wrist_topic=wrist_topic,
            wrist_depth_topic=wrist_depth_topic,
            wrist_fx=fx,
            wrist_fy=fy,
            wrist_cx=cx,
            wrist_cy=cy,
            wrist_w=ww,
            wrist_h=wh,
            tf_buffer=tf_buffer,
            get_camera_frame=get_camera_frame,
        )
        if wu is None or wv is None:
            notes.append(f"{label}：目标不在视野内（反投影失败），不参考该腕图像")
            return None, notes

        # 种子处深度无效 → 视为看不见（遮挡/飞出工作区）
        try:
            depth_m = _eai_perception_image_geometry.depth_to_meters_array(wrist_depth)
            z_seed = float(
                depth_m[
                    max(0, min(dh - 1, int(wv))),
                    max(0, min(dw - 1, int(wu))),
                ]
            )
        except Exception:
            depth_m = None
            z_seed = 0.0
        if not _eai_platform_bootstrap.np.isfinite(z_seed) or z_seed < 0.05 or z_seed > 3.5:
            notes.append(
                f"{label}：种子({wu},{wv})深度无效 z={z_seed:.3f}，不参考该腕图像"
            )
            return None, notes

        # 种子邻域 + 全图抽样：无效深度过高则跳过（避免右腕空洞把融合中心带偏）
        if depth_m is not None:
            rad = 40
            u0 = max(0, int(wu) - rad)
            u1 = min(dw, int(wu) + rad + 1)
            v0 = max(0, int(wv) - rad)
            v1 = min(dh, int(wv) + rad + 1)
            patch = depth_m[v0:v1, u0:u1]
            if patch.size > 0:
                valid_p = (patch > 0.05) & (patch < 3.5) & _eai_platform_bootstrap.np.isfinite(patch)
                inv_p = 1.0 - float(_eai_platform_bootstrap.np.count_nonzero(valid_p)) / float(patch.size)
                if inv_p > float(_eai_configuration_policies.MV6D_WRIST_DEPTH_INVALID_MAX_RATIO):
                    notes.append(
                        f"{label}：种子邻域无效深度 {inv_p * 100:.0f}% "
                        f"> {_eai_configuration_policies.MV6D_WRIST_DEPTH_INVALID_MAX_RATIO * 100:.0f}%，不参考"
                    )
                    return None, notes
            # 全图稀疏抽样
            step = max(1, min(dh, dw) // 64)
            sample = depth_m[::step, ::step]
            if sample.size > 0:
                valid_s = (sample > 0.05) & (sample < 3.5) & _eai_platform_bootstrap.np.isfinite(sample)
                inv_s = 1.0 - float(_eai_platform_bootstrap.np.count_nonzero(valid_s)) / float(sample.size)
                if inv_s > float(_eai_configuration_policies.MV6D_WRIST_DEPTH_INVALID_MAX_RATIO) * 1.5:
                    notes.append(
                        f"{label}：全图无效深度≈{inv_s * 100:.0f}%，不参考该腕图像"
                    )
                    return None, notes

        notes.append(f"{label}：可见，种子({wu},{wv})")
        try:
            wrist_result = self._estimate_view_6d(
                wrist_bgr,
                wrist_depth,
                int(wu),
                int(wv),
                fx,
                fy,
                cx,
                cy,
                method_tag=label,
                assumed_depth_m=max(0.08, float(assumed_depth_m) * 0.7),
                pose_backend=pose_backend,
            )
        except Exception as exc:
            notes.append(f"{label}：SAM3/位姿失败，不参考该腕图像（{exc}）")
            return None, notes
        if wrist_result is None:
            notes.append(f"{label}：估计失败，跳过")
            return None, notes

        # 与头部目标在 base 下一致性：差过大说明切到了别的东西
        if head_center_base is not None:
            frame = ""
            try:
                frame = (
                    get_camera_frame(wrist_topic, wrist_depth_topic)
                    or get_camera_frame(wrist_depth_topic, wrist_topic)
                    or ""
                )
            except Exception:
                pass
            T = self._mv6d_sim_extrinsic_matrix(
                wrist_topic, wrist_depth_topic, frame
            )
            if T is None and tf_buffer is not None and frame:
                T = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, _eai_configuration_policies.BASE_LINK_FRAME, frame)
            if T is not None:
                p_w = _eai_perception_pose_estimation.transform_points_by_matrix(
                    _eai_platform_bootstrap.np.asarray([wrist_result.position_xyz], dtype=_eai_platform_bootstrap.np.float32),
                    T,
                )[0]
                dist = float(_eai_platform_bootstrap.np.linalg.norm(p_w - head_center_base))
                if dist > float(max_center_dist_m):
                    notes.append(
                        f"{label}：与头部目标差 {dist * 1000:.0f} mm > "
                        f"{max_center_dist_m * 1000:.0f} mm，判定非同一物体，不参考"
                    )
                    return None, notes

        return wrist_result, notes

    def _on_mv6d_copy_result(self) -> None:
        text = self.mv6d_result_edit.toPlainText().strip()
        if not text:
            text = self.mv6d_error_label.text().strip()
        clip = _eai_platform_bootstrap.QApplication.clipboard()
        if clip is None or not text:
            return
        clip.setText(text)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage("已复制多视角 6D 结果", 3000)

    def _mv6d_error_pair(
        self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[float], _eai_platform_bootstrap.Optional[float]]:
        gt_xyz, gt_R = self._mv6d_gt_pose()
        if (
            _eai_adapters_robot_geometry.normalize_frame_id(snap.frame_id) != _eai_configuration_policies.BASE_LINK_FRAME
            or not self.mv6d_gt_enable_check.isChecked()
            or gt_xyz is None
            or gt_R is None
        ):
            return None, None
        t_mm, r_deg, _ = _eai_perception_pose_estimation.pose_error_metrics(
            snap.position_xyz, snap.rotation_matrix, gt_xyz, gt_R
        )
        return t_mm, r_deg

    def _on_mv6d_lightbox_closed(self) -> None:
        self._mv6d_lightbox = None

    def _mv6d_cmd_arm_side(self) -> str:
        """文本指挥回填用哪只臂：跟随腕相机，但仿真无该侧 TCP 时自动改有状态的一侧。"""
        wrist = str(
            getattr(self, "mv6d_wrist_combo", None)
            and self.mv6d_wrist_combo.currentData()
            or ""
        )
        prefer = "right" if "right" in wrist else "left"
        if self._sim_teleop_active():
            return self._sim_resolve_arm_side(prefer)
        return prefer

    def _mv6d_cmd_arm_side_for_target(
        self, xyz: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]]
    ) -> str:
        """回填选臂：优先离目标更近的那只（避免目标在左侧却驱右臂，误差显得更大）。"""
        prefer = self._mv6d_cmd_arm_side()
        if xyz is None:
            return prefer
        use_mid = self._sim_uses_finger_mid_match()
        best_side = prefer
        best_d = 1e9
        prefer_d = 1e9
        hx, hy, hz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        for side in ("left", "right"):
            tcp = self._tcp_for_relative_goal(side)
            if tcp is None:
                continue
            if use_mid:
                px, py, pz = self._finger_mid_xyz_from_ee(tcp[0], tcp[1])
            else:
                px, py, pz = tcp[0]
            d = _eai_platform_bootstrap.math.sqrt((px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2)
            if side == prefer:
                prefer_d = d
            if d < best_d:
                best_d = d
                best_side = side
        # 偏好侧已足够近，或两侧差不多 → 跟 combo；否则改用更近的一侧
        if best_side != prefer and prefer_d < best_d + 0.12:
            return prefer
        if best_side != prefer and best_d + 0.08 < prefer_d:
            return best_side
        return prefer

    def _mv6d_sim_extrinsic_matrix(
        self,
        color_topic: str = "",
        depth_topic: str = "",
        frame: str = "",
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        """读 Isaac bridge 写出的 cam→base 外参（新鲜才用）。"""
        if _eai_configuration_policies.read_cam_extrinsic is None:
            return None
        keys: _eai_platform_bootstrap.List[str] = []
        for topic in (color_topic, depth_topic):
            k = _eai_configuration_policies._SIM_TOPIC_TO_CAM_KEY.get(str(topic or "").strip())
            if k and k not in keys:
                keys.append(k)
        fk = _eai_configuration_policies._SIM_FRAME_TO_CAM_KEY.get(_eai_adapters_robot_geometry.normalize_frame_id(frame))
        if fk and fk not in keys:
            keys.append(fk)
        if not keys:
            keys = ["cam_head", "cam_high"]
        bridge = _eai_platform_bootstrap.os.environ.get("ISAAC_CAM_BRIDGE_DIR", "").strip() or (
            _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
        )
        for key in keys:
            T = _eai_configuration_policies.read_cam_extrinsic(key, root=bridge, max_age_s=3.0)
            if T is not None:
                return _eai_platform_bootstrap.np.asarray(T, dtype=_eai_platform_bootstrap.np.float64)
        return None

    def _mv6d_manual_extrinsic_matrix(
        self, color_topic: str = "", frame: str = ""
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        """手眼标定页保存的外参。残差大于 2cm，或腕部相对 TCP 远于 35cm 的结果不用。"""
        try:
            from eai_app.domain.calibration.cameras import compose_base_camera, load_store
        except Exception:
            return None
        cameras = load_store().get("cameras") or {}
        entry = cameras.get(str(color_topic or ""))
        if entry is None and _eai_adapters_robot_geometry.normalize_frame_id(frame) == "camera_frame":
            entry = cameras.get("/camera/head_color")
        if not isinstance(entry, dict):
            return None
        calibrated, _reason = _eai_platform_bootstrap.validated_camera_transform(entry)
        if calibrated is None:
            return None
        if entry.get("model") == "eye_in_hand":
            raw = calibrated
            tcp = self._calib_tcp(str(entry.get("arm_side") or ""))
            if not tcp or raw is None:
                return None
            t_tcp = _eai_platform_bootstrap.np.asarray(raw, dtype=_eai_platform_bootstrap.np.float64)
            if t_tcp.shape != (4, 4) or float(_eai_platform_bootstrap.np.linalg.norm(t_tcp[:3, 3])) > 0.35:
                return None
            matrix = compose_base_camera(tcp["xyz"], tcp["xyzw"], t_tcp)
        else:
            raw = calibrated
            if raw is None:
                return None
            matrix = _eai_platform_bootstrap.np.asarray(raw, dtype=_eai_platform_bootstrap.np.float64)
        if getattr(matrix, "shape", None) != (4, 4) or not _eai_platform_bootstrap.np.isfinite(matrix).all():
            return None
        return matrix

    def _mv6d_reproject_head_seed_to_wrist(
        self,
        *,
        head_result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
        head_topic: str,
        head_depth_topic: str,
        wrist_topic: str,
        wrist_depth_topic: str,
        wrist_fx: float,
        wrist_fy: float,
        wrist_cx: float,
        wrist_cy: float,
        wrist_w: int,
        wrist_h: int,
        tf_buffer,
        get_camera_frame,
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[int], _eai_platform_bootstrap.Optional[int]]:
        """把头部估计中心变到腕部像素，作腕部分割种子。"""
        if head_result is None:
            return None, None
        head_frame = ""
        wrist_frame = ""
        try:
            head_frame = (
                get_camera_frame(head_topic, head_depth_topic)
                or get_camera_frame(head_depth_topic, head_topic)
                or ""
            )
            wrist_frame = (
                get_camera_frame(wrist_topic, wrist_depth_topic)
                or get_camera_frame(wrist_depth_topic, wrist_topic)
                or ""
            )
        except Exception:
            pass
        T_bh = self._mv6d_sim_extrinsic_matrix(
            head_topic, head_depth_topic, head_frame
        )
        T_bw = self._mv6d_sim_extrinsic_matrix(
            wrist_topic, wrist_depth_topic, wrist_frame
        )
        if T_bh is None and tf_buffer is not None and head_frame:
            T_bh = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, _eai_configuration_policies.BASE_LINK_FRAME, head_frame)
        if T_bw is None and tf_buffer is not None and wrist_frame:
            T_bw = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, _eai_configuration_policies.BASE_LINK_FRAME, wrist_frame)
        if T_bh is None or T_bw is None:
            return None, None
        try:
            T_wb = _eai_platform_bootstrap.np.linalg.inv(_eai_platform_bootstrap.np.asarray(T_bw, dtype=_eai_platform_bootstrap.np.float64))
        except _eai_platform_bootstrap.np.linalg.LinAlgError:
            return None, None
        p_h = _eai_platform_bootstrap.np.array(
            [
                float(head_result.position_xyz[0]),
                float(head_result.position_xyz[1]),
                float(head_result.position_xyz[2]),
                1.0,
            ],
            dtype=_eai_platform_bootstrap.np.float64,
        )
        p_b = _eai_platform_bootstrap.np.asarray(T_bh, dtype=_eai_platform_bootstrap.np.float64) @ p_h
        p_w = T_wb @ p_b
        z = float(p_w[2])
        if not _eai_platform_bootstrap.np.isfinite(z) or z < 0.05:
            return None, None
        u = wrist_fx * float(p_w[0]) / z + wrist_cx
        v = wrist_fy * float(p_w[1]) / z + wrist_cy
        if not (_eai_platform_bootstrap.np.isfinite(u) and _eai_platform_bootstrap.np.isfinite(v)):
            return None, None
        ui = int(round(u))
        vi = int(round(v))
        if ui < 0 or vi < 0 or ui >= int(wrist_w) or vi >= int(wrist_h):
            return None, None
        return ui, vi

    def _mv6d_pose_in_base(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
        """把估计位姿变到 base_link。变不了就返回 None，避免画到相机系坐标上。"""
        frame = _eai_adapters_robot_geometry.normalize_frame_id(snap.frame_id)
        base = _eai_adapters_robot_geometry.normalize_frame_id(_eai_configuration_policies.BASE_LINK_FRAME)
        xyz = _eai_platform_bootstrap.np.asarray(snap.position_xyz, dtype=_eai_platform_bootstrap.np.float64).reshape(3)
        rotation = _eai_platform_bootstrap.np.asarray(snap.rotation_matrix, dtype=_eai_platform_bootstrap.np.float64).reshape(3, 3)
        corners = None
        if snap.obb_corners is not None:
            arr = _eai_platform_bootstrap.np.asarray(snap.obb_corners, dtype=_eai_platform_bootstrap.np.float64)
            if arr.shape == (8, 3):
                corners = arr
        extents = snap.obb_extents
        if not _eai_platform_bootstrap.np.isfinite(xyz).all() or not _eai_platform_bootstrap.np.isfinite(rotation).all():
            return None
        if (snap.tf_ok and (not frame or frame == base)) or frame == base:
            return xyz, rotation, corners, extents
        if not frame or frame in ("camera", "cam", "optical_frame"):
            try:
                head_t = snap.head_topic or "/camera/head_color"
                frame = self.node.resolve_segment_camera_frame(head_t, "") or frame or "camera"
            except Exception:
                frame = frame or "camera"
        matrix = self._mv6d_sim_extrinsic_matrix(snap.head_topic or "", "", frame or "")
        if matrix is None:
            matrix = self._mv6d_manual_extrinsic_matrix(snap.head_topic or "", frame or "")
        if matrix is None:
            try:
                tf_buffer = self.node.get_tf_buffer()
            except Exception:
                tf_buffer = None
            if tf_buffer is not None and frame:
                matrix = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, _eai_configuration_policies.BASE_LINK_FRAME, frame)
        if matrix is None:
            return None
        matrix = _eai_platform_bootstrap.np.asarray(matrix, dtype=_eai_platform_bootstrap.np.float64)
        if matrix.shape != (4, 4) or not _eai_platform_bootstrap.np.isfinite(matrix).all():
            return None
        xyz_b = matrix[:3, :3] @ xyz + matrix[:3, 3]
        rotation_b = matrix[:3, :3] @ rotation
        corners_b = None
        if corners is not None:
            corners_b = _eai_perception_pose_estimation.transform_points_by_matrix(corners.astype(_eai_platform_bootstrap.np.float32), matrix)
        return xyz_b, rotation_b, corners_b, extents

    def _mv6d_T_base_from_head(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
        """头部相机到 base_link。腕部标定误差大，分割点只用头部外参。"""
        topic = snap.head_topic or "/camera/head_color"
        frame = ""
        try:
            frame = self.node.resolve_segment_camera_frame(topic, "") or ""
        except Exception:
            frame = ""
        matrix = self._mv6d_sim_extrinsic_matrix(topic, "", frame)
        if matrix is None:
            matrix = self._mv6d_manual_extrinsic_matrix(topic, frame)
        if matrix is None:
            try:
                tf_buffer = self.node.get_tf_buffer()
            except Exception:
                tf_buffer = None
            if tf_buffer is not None and frame:
                matrix = _eai_adapters_robot_geometry.lookup_transform_matrix(tf_buffer, _eai_configuration_policies.BASE_LINK_FRAME, frame)
        if matrix is None:
            return None
        matrix = _eai_platform_bootstrap.np.asarray(matrix, dtype=_eai_platform_bootstrap.np.float64)
        if matrix.shape != (4, 4) or not _eai_platform_bootstrap.np.isfinite(matrix).all():
            return None
        return matrix

    def _mv6d_sam3_points_in_base(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
        result = snap.head_result
        if result is None or result.points_3d is None:
            return None, ""
        points = _eai_platform_bootstrap.np.asarray(result.points_3d, dtype=_eai_platform_bootstrap.np.float32).reshape(-1, 3)
        if points.shape[0] == 0 or not _eai_platform_bootstrap.np.isfinite(points).all():
            return None, ""
        matrix = self._mv6d_T_base_from_head(snap)
        if matrix is None:
            return None, "外参"
        if points.shape[0] > 6000:
            step = int(_eai_platform_bootstrap.np.ceil(points.shape[0] / 6000.0))
            points = points[::step]
        return _eai_perception_pose_estimation.transform_points_by_matrix(points, matrix), ""

    def _mv6d_xyz_in_base_link(
        self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]], str]:
        """把 6D 估计位置变到 base_link，供文本指挥绝对目标使用。

        返回 (xyz_base, note)。失败时 xyz 为 None，note 为原因。
        """
        frame = _eai_adapters_robot_geometry.normalize_frame_id(snap.frame_id)
        base = _eai_adapters_robot_geometry.normalize_frame_id(_eai_configuration_policies.BASE_LINK_FRAME)
        xyz = (
            float(snap.position_xyz[0]),
            float(snap.position_xyz[1]),
            float(snap.position_xyz[2]),
        )
        quat = (
            float(snap.quaternion_xyzw[0]),
            float(snap.quaternion_xyzw[1]),
            float(snap.quaternion_xyzw[2]),
            float(snap.quaternion_xyzw[3]),
        )
        if snap.tf_ok and (not frame or frame == base):
            return xyz, "base_link(融合)"
        if frame == base:
            return xyz, "base_link"
        # 相机系：优先仿真外参，再 TF
        if not frame or frame in ("camera", "cam", "optical_frame"):
            try:
                head_t = snap.head_topic or "/camera/head_color"
                frame = (
                    self.node.resolve_segment_camera_frame(head_t, "")
                    or frame
                    or "camera"
                )
            except Exception:
                pass
        extrinsic_note = "sim外参"
        T_sim = self._mv6d_sim_extrinsic_matrix(
            snap.head_topic or "", "", frame or ""
        )
        if T_sim is None:
            T_sim = self._mv6d_manual_extrinsic_matrix(snap.head_topic or "", frame or "")
            extrinsic_note = "手工外参"
        if T_sim is not None:
            p = _eai_perception_pose_estimation.transform_points_by_matrix(
                _eai_platform_bootstrap.np.asarray([xyz], dtype=_eai_platform_bootstrap.np.float32), T_sim
            )[0]
            return (
                (float(p[0]), float(p[1]), float(p[2])),
                f"{frame or 'camera'}→base_link({extrinsic_note})",
            )
        tf_buffer = None
        try:
            tf_buffer = self.node.get_tf_buffer()
        except Exception:
            tf_buffer = None
        converted = _eai_adapters_robot_geometry.transform_pose_to_base(
            xyz, quat, frame, tf_buffer, base_frame=_eai_configuration_policies.BASE_LINK_FRAME
        )
        if converted is None:
            return (
                None,
                f"位姿在「{frame or 'camera'}」系且无法变换到 base_link（缺 TF）。"
                "请勿把相机系坐标当绝对目标执行，否则手臂会越走越远。",
            )
        bx, by, bz = converted[0]
        return (float(bx), float(by), float(bz)), f"{frame}→base_link"

    def _mv6d_sim_objects_list(
        self,
    ) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.Tuple[float, float, float]]]:
        """从仿真 sim_robot_state.objects 读可抓物体位姿（跳过 init_region）。"""
        if not self._mv6d_sim_reference_is_current():
            return []
        state = self._read_sim_robot_state()
        if not isinstance(state, dict):
            return []
        raw = state.get("objects")
        if not isinstance(raw, dict):
            return []
        out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.Tuple[float, float, float]]] = []
        for name, meta in raw.items():
            key = str(name)
            if "init_region" in key.lower():
                continue
            if not isinstance(meta, dict):
                continue
            pos = meta.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                continue
            try:
                xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
            except (TypeError, ValueError):
                continue
            if not all(_eai_platform_bootstrap.math.isfinite(v) for v in xyz):
                continue
            out.append((key, xyz))
        return out

    def _mv6d_nearest_sim_object(
        self,
        hint_xyz: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]],
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.Tuple[float, float, float]]]:
        """按水平距离选离 hint 最近的仿真物体（LIBERO GT）。"""
        objs = self._mv6d_sim_objects_list()
        if not objs:
            return None
        if hint_xyz is None:
            side = self._mv6d_cmd_arm_side()
            tcp = self._tcp_for_relative_goal(side)
            hint_xyz = tcp[0] if tcp is not None else (0.0, 0.0, 0.9)
        best: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.Tuple[float, float, float]]] = None
        best_d = 1e9
        hx, hy, hz = hint_xyz
        for name, (x, y, z) in objs:
            # 水平距离为主；高度差加权，避免飞到天花板估计附近的假匹配
            d = _eai_platform_bootstrap.math.sqrt((x - hx) ** 2 + (y - hy) ** 2 + 0.15 * (z - hz) ** 2)
            if d < best_d:
                best_d = d
                best = (name, (x, y, z))
        return best

    def _mv6d_estimate_looks_unreliable(
        self,
        xyz_est: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]],
        snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """LIBERO RGB-only / 错误 TF 时常把 Z 估到 1.6m+；用仿真物体高度校验。"""
        method = (snap.method or "") + " " + (snap.notes or "")
        if "rgb-only" in method.lower() or "假定平面" in method:
            return True, "RGB-only/平面深度，尺度不可靠"
        fused = int(getattr(snap, "fused_points", 0) or 0)
        if fused > 0 and fused < _eai_configuration_policies.MV6D_MIN_FUSED_POINTS:
            return True, f"融合点过少({fused}<{_eai_configuration_policies.MV6D_MIN_FUSED_POINTS})，中心/姿态不可靠"
        objs = self._mv6d_sim_objects_list()
        if xyz_est is None:
            return True, "无有效 base 估计"
        if objs:
            zs = [p[2] for _, p in objs]
            med_z = float(sorted(zs)[len(zs) // 2])
            if abs(float(xyz_est[2]) - med_z) > 0.25:
                return (
                    True,
                    f"估计 Z={xyz_est[2]:.3f} 与仿真物体高度≈{med_z:.3f} 差过大",
                )
        # 桌面操作工作区粗检
        if float(xyz_est[2]) > 1.25 or float(xyz_est[2]) < 0.2:
            return True, f"估计 Z={xyz_est[2]:.3f} 超出合理桌面范围"
        return False, ""

    def _mv6d_refine_with_sim_gt(
        self,
        xyz_est: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]],
        *,
        max_dist_m: float = 0.12,
        snap: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.MultiView6DPoseSnapshot] = None,
    ) -> _eai_platform_bootstrap.Tuple[
        _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]],
        _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]],
        str,
    ]:
        """若估计靠近仿真物体，用 GT 位置（及姿态）精修 → 仿真下手臂可达毫米级。"""
        if xyz_est is None:
            return None, None, ""
        objs = self._mv6d_sim_objects_list()
        if not objs:
            return xyz_est, None, ""
        fused = int(getattr(snap, "fused_points", 0) or 0) if snap is not None else 0
        # 点云过稀时放宽匹配半径，仍优先 GT 而非漂数 cm 的 PCA 中心
        if fused > 0 and fused < _eai_configuration_policies.MV6D_MIN_FUSED_POINTS:
            max_dist_m = max(float(max_dist_m), 0.22)
        # 扩展：从 state 再读 quat
        state = self._read_sim_robot_state()
        raw = state.get("objects") if isinstance(state, dict) else None
        best = None
        best_d = 1e9
        best_q = None
        hx, hy, hz = xyz_est
        for name, (x, y, z) in objs:
            d = _eai_platform_bootstrap.math.sqrt((x - hx) ** 2 + (y - hy) ** 2 + (z - hz) ** 2)
            if d < best_d:
                best_d = d
                best = (name, (x, y, z))
                best_q = None
                if isinstance(raw, dict) and name in raw:
                    q = raw[name].get("quat") if isinstance(raw[name], dict) else None
                    if isinstance(q, (list, tuple)) and len(q) >= 4:
                        # sim objects quat 为 wxyz
                        best_q = (
                            float(q[1]),
                            float(q[2]),
                            float(q[3]),
                            float(q[0]),
                        )
        if best is None or best_d > max_dist_m:
            return xyz_est, None, ""
        name, xyz_gt = best
        return (
            xyz_gt,
            best_q,
            f"sim-GT精修:{name} Δ={best_d * 1000:.1f}mm",
        )

    def _mv6d_tabletop_grasp_quat_xyzw(
        self,
        *,
        yaw_rad: float = 0.0,
    ) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
        """桌面抓取姿态：link6 +X（指尖方向）朝下，开合轴水平，绕 Z 偏航 yaw。"""
        cy, sy = _eai_platform_bootstrap.math.cos(float(yaw_rad)), _eai_platform_bootstrap.math.sin(float(yaw_rad))
        # Ry(+90°)：ee +X → world -Z；再 Rz(yaw)
        ry = _eai_platform_bootstrap.np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=_eai_platform_bootstrap.np.float64,
        )
        rz = _eai_platform_bootstrap.np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=_eai_platform_bootstrap.np.float64,
        )
        return _eai_perception_pose_estimation.rotation_matrix_to_quaternion(rz @ ry)

    def _mv6d_grasp_xyz_quat_for_fill(
        self,
        xyz_center: _eai_platform_bootstrap.Tuple[float, float, float],
        snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
        *,
        side: str,
        prefer_topdown: bool = True,
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]], str]:
        """物体中心 → 抓取匹配点（手指中点）+ 可选桌面朝下姿态。"""
        notes: _eai_platform_bootstrap.List[str] = []
        x, y, z = [float(v) for v in xyz_center]
        # Z：略抬到几何中心之上，避免插桌；幅度取 OBB 最小边的一部分
        extents = getattr(snap, "obb_extents", None)
        half_z = 0.02
        if isinstance(extents, (list, tuple)) and len(extents) >= 3:
            try:
                half_z = 0.5 * float(min(abs(float(v)) for v in extents[:3]))
            except (TypeError, ValueError):
                pass
        half_z = max(0.008, min(0.04, half_z))
        z_grasp = z + float(_eai_configuration_policies.MV6D_GRASP_Z_LIFT_FRAC) * half_z
        notes.append(f"抓取Z+{ (z_grasp - z) * 1000:.0f}mm")
        xyz_grasp = (x, y, z_grasp)

        quat: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None
        if prefer_topdown and self._sim_uses_finger_mid_match():
            tcp = self._tcp_for_relative_goal(side)
            yaw = 0.0
            use_topdown = True
            if tcp is not None:
                R = _eai_adapters_robot_geometry.quat_to_rotation_matrix(
                    float(tcp[1][0]),
                    float(tcp[1][1]),
                    float(tcp[1][2]),
                    float(tcp[1][3]),
                )
                approach_z = float(R[2, 0])  # ee +X 的世界 Z 分量
                # 已明显朝下则保持当前姿态，避免大幅拧腕
                if approach_z < -0.55:
                    use_topdown = False
                    quat = (
                        float(tcp[1][0]),
                        float(tcp[1][1]),
                        float(tcp[1][2]),
                        float(tcp[1][3]),
                    )
                    notes.append("姿态保持(已朝下)")
                else:
                    # 偏航：开合轴大致垂直于手臂→物体水平方向
                    dx = x - float(tcp[0][0])
                    dy = y - float(tcp[0][1])
                    if abs(dx) + abs(dy) > 1e-4:
                        yaw = _eai_platform_bootstrap.math.atan2(dy, dx) + 0.5 * _eai_platform_bootstrap.math.pi
            if use_topdown:
                quat = self._mv6d_tabletop_grasp_quat_xyzw(yaw_rad=yaw)
                notes.append("姿态=桌面朝下")
        return xyz_grasp, quat, "; ".join(notes)

    def _mv6d_auto_gt_from_sim(
        self, xyz_est: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]]
    ) -> None:
        """估计完成后自动把最近仿真物体写入 GT 旋钮，误差才有意义。"""
        if not self._mv6d_sim_reference_is_current():
            return
        if xyz_est is None:
            return
        if getattr(self, "mv6d_gt_enable_check", None) is None:
            return
        picked = self._mv6d_nearest_sim_object(xyz_est)
        if picked is None:
            return
        name, (x, y, z) = picked
        try:
            self.mv6d_gt_x_spin.setValue(float(x))
            self.mv6d_gt_y_spin.setValue(float(y))
            self.mv6d_gt_z_spin.setValue(float(z))
            state = self._read_sim_robot_state()
            raw = state.get("objects") if isinstance(state, dict) else None
            if isinstance(raw, dict) and name in raw:
                q = raw[name].get("quat") if isinstance(raw[name], dict) else None
                if isinstance(q, (list, tuple)) and len(q) >= 4:
                    R = _eai_adapters_robot_geometry.quat_to_rotation_matrix(
                        float(q[0]), float(q[1]), float(q[2]), float(q[3])
                    )
                    r, p, yw = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(R)
                    self.mv6d_gt_roll_spin.setValue(_eai_platform_bootstrap.math.degrees(r))
                    self.mv6d_gt_pitch_spin.setValue(_eai_platform_bootstrap.math.degrees(p))
                    self.mv6d_gt_yaw_spin.setValue(_eai_platform_bootstrap.math.degrees(yw))
            self.mv6d_gt_enable_check.setChecked(True)
            try:
                self.mv6d_result_edit.append(
                    f"\n[GT] 已自动对齐最近仿真物体 {name} "
                    f"@ ({x:.3f},{y:.3f},{z:.3f})"
                )
            except Exception:
                pass
        except Exception:
            pass

