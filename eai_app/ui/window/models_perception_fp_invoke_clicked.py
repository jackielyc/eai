"""ui.window.models perception fp invoke clicked — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ModelsPerceptionFpInvokeClickedMixin:
    def _on_fp_invoke_clicked(self) -> None:
        if self._fp_call_busy:
            return
        self._on_pose_settings_changed()
        pcfg = _eai_perception_segmentation.get_pose_settings()
        backend = pcfg.backend
        backend_label = {
            _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE: "FoundationPose",
            _eai_configuration_perception.POSE_BACKEND_ANY6D: "Any6D",
            _eai_configuration_perception.POSE_BACKEND_PCA: "PCA",
        }.get(backend, backend)

        if backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE:
            mesh_path, mesh_source = _eai_configuration_perception.resolve_fp_mesh_for_call(
                pcfg.fp_mesh,
                pcfg.fp_use_http,
                pcfg.fp_server_url,
            )
            mesh_display = mesh_path if mesh_source != "local" else pcfg.fp_mesh
            transport = "HTTP" if pcfg.fp_use_http else "子进程"
        elif backend == _eai_configuration_perception.POSE_BACKEND_ANY6D:
            mesh_display = (pcfg.fp_mesh or "").strip() or "(model-free 深度凸包)"
            mesh_source = "file" if (pcfg.fp_mesh or "").strip() else "depth-hull"
            transport = "HTTP" if pcfg.any6d_use_http else "子进程"
        else:
            mesh_display = "-"
            mesh_source = "pca"
            transport = "本地"

        source = self._pick_fp_stereo_source()
        if source is None:
            self._set_fp_result_text(
                "错误: 需要彩色图 + 已勾选的配对 depth topic\n"
                "请勾选 color 与对应 depth，并在图像上设置提示点 (u,v)"
            )
            self.status_bar.showMessage(f"{backend_label}: 缺少彩色/深度数据")
            return

        color_topic, color_bgr, depth_topic, depth, u, v = source
        dh, dw = depth.shape[:2]
        u_d, v_d = _eai_perception_image_geometry.scale_uv_to_shape(u, v, color_bgr.shape[:2], (dh, dw))
        fx, fy, cx, cy = self.node.get_intrinsics(depth_topic, dw, dh)

        self._fp_call_busy = True
        self.fp_invoke_btn.setEnabled(False)
        self.fp_invoke_btn.setText(f"{backend_label}…")
        self._set_fp_result_text(
            f"正在调用 {backend_label}…\n"
            f"彩色: {color_topic}\n"
            f"深度: {depth_topic}\n"
            f"点: color({u},{v}) depth({u_d},{v_d})\n"
            f"mesh: {mesh_display} ({mesh_source})\n"
            f"传输: {transport}"
        )
        scfg = _eai_perception_segmentation.get_segment_settings()

        def _work() -> None:
            t0 = _eai_platform_bootstrap.time.time()
            try:
                if backend == _eai_configuration_perception.POSE_BACKEND_PCA:
                    pose_result, _info = _eai_perception_pose_estimation.run_stereo_pose_pipeline(
                        depth,
                        u_d,
                        v_d,
                        fx,
                        fy,
                        cx,
                        cy,
                        color_bgr=color_bgr,
                        pose_settings=pcfg,
                    )
                    if pose_result is None:
                        raise RuntimeError(_info or "PCA 立体位姿失败")
                    result = _eai_perception_segmentation.FpCallResult(
                        ok=True,
                        color_topic=color_topic,
                        depth_topic=depth_topic,
                        u=u,
                        v=v,
                        mesh=mesh_display,
                        segment_method=str(pose_result.method or "pca"),
                        pose_method=str(pose_result.method or "pca"),
                        transport=transport,
                        elapsed_s=_eai_platform_bootstrap.time.time() - t0,
                        mask=pose_result.mask,
                        pose_result=pose_result,
                    )
                else:
                    mask, seg_method = _eai_perception_pose_estimation.segment_object_at_click_with_backend(
                        u_d,
                        v_d,
                        depth,
                        color_bgr=color_bgr,
                        settings=scfg,
                    )
                    if not mask.any():
                        raise RuntimeError("分割失败（无有效 mask）")
                    if backend == _eai_configuration_perception.POSE_BACKEND_ANY6D:
                        fp_body = _eai_perception_pose_estimation.run_any6d_estimation(
                            color_bgr,
                            depth,
                            mask,
                            fx,
                            fy,
                            cx,
                            cy,
                            settings=pcfg,
                            tag="invoke_btn",
                        )
                    else:
                        fp_body = _eai_perception_segmentation.run_foundationpose_estimation(
                            color_bgr,
                            depth,
                            mask,
                            fx,
                            fy,
                            cx,
                            cy,
                            settings=pcfg,
                            tag="invoke_btn",
                        )
                    pose_result = _eai_perception_segmentation.object6d_from_foundationpose(
                        mask,
                        depth,
                        u_d,
                        v_d,
                        fx,
                        fy,
                        cx,
                        cy,
                        fp_body,
                        segment_method=seg_method,
                    )
                    if pose_result is None:
                        raise RuntimeError(f"{backend_label} 响应无效")
                    result = _eai_perception_segmentation.FpCallResult(
                        ok=True,
                        color_topic=color_topic,
                        depth_topic=depth_topic,
                        u=u,
                        v=v,
                        mesh=mesh_display,
                        segment_method=seg_method,
                        pose_method=str(fp_body.get("method") or backend),
                        transport=transport,
                        elapsed_s=_eai_platform_bootstrap.time.time() - t0,
                        mask=mask,
                        pose_result=pose_result,
                    )
            except Exception as exc:
                result = _eai_perception_segmentation.FpCallResult(
                    ok=False,
                    color_topic=color_topic,
                    depth_topic=depth_topic,
                    u=u,
                    v=v,
                    mesh=mesh_display,
                    transport=transport,
                    elapsed_s=_eai_platform_bootstrap.time.time() - t0,
                    error=str(exc),
                )
            self._fp_call_bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_fp_call_finished(self, result: object) -> None:
        self._fp_call_busy = False
        self.fp_invoke_btn.setText("调用位姿")
        if not isinstance(result, _eai_perception_segmentation.FpCallResult):
            self._refresh_fp_status()
            return

        self._set_fp_result_text(_eai_perception_segmentation.format_fp_call_result_text(result))
        backend = _eai_perception_segmentation.get_pose_settings().backend
        backend_label = {
            _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE: "FoundationPose",
            _eai_configuration_perception.POSE_BACKEND_ANY6D: "Any6D",
            _eai_configuration_perception.POSE_BACKEND_PCA: "PCA",
        }.get(backend, "位姿")
        if not result.ok or result.pose_result is None:
            if result.ok:
                self.status_bar.showMessage(f"{backend_label} 成功")
            else:
                self.status_bar.showMessage(
                    f"{backend_label} 失败: {(result.error or '')[:120]}"
                )
            self._refresh_fp_status()
            return

        pose = result.pose_result
        color_panel = self.panels.get(result.color_topic)
        if isinstance(color_panel, _eai_ui_cameras_panels.CameraPanel) and color_panel._latest_image is not None:
            src = color_panel._latest_image
            display_mask = _eai_perception_image_geometry.resize_mask_to_shape(pose.mask, src.shape[:2])
            dh, dw = pose.mask.shape[:2]
            ch, cw = src.shape[:2]
            cu, cv_pt = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(pose.centroid_uv[0])),
                int(round(pose.centroid_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            contact_u, contact_v = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(pose.contact_uv[0])),
                int(round(pose.contact_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            # 位姿在 depth 相机坐标系；用 depth 内参并缩放到彩色图尺寸
            fx, fy, cx, cy = self.node.get_intrinsics(result.depth_topic, dw, dh)
            sx = float(cw) / float(max(1, dw))
            sy = float(ch) / float(max(1, dh))
            intr = (fx * sx, fy * sy, cx * sx, cy * sy)
            axis_len = float(max(pose.obb_extents) * 0.55)
            color_panel.image_label.set_segment_overlay(
                display_mask,
                (float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=intr,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(result.u), float(result.v)),
                pose_position=_eai_platform_bootstrap.np.asarray(pose.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                pose_rotation=_eai_platform_bootstrap.np.asarray(pose.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                pose_axis_len=axis_len,
            )
            self._set_segment_preview(
                src,
                display_mask,
                seed_uv=(float(result.u), float(result.v)),
                centroid_uv=(float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=intr,
                pose_position=_eai_platform_bootstrap.np.asarray(pose.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                pose_rotation=_eai_platform_bootstrap.np.asarray(pose.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                pose_axis_len=axis_len,
            )

        depth_panel = self.panels.get(result.depth_topic)
        if isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D):
            dh, dw = depth_panel._depth_full_shape
            mask_p = _eai_perception_image_geometry.resize_mask_to_shape(pose.mask, depth_panel._preview_shape)
            cu, cv_pt = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(pose.centroid_uv[0])),
                int(round(pose.centroid_uv[1])),
                (dh, dw),
                depth_panel._preview_shape,
            )
            contact_u, contact_v = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(pose.contact_uv[0])),
                int(round(pose.contact_uv[1])),
                (dh, dw),
                depth_panel._preview_shape,
            )
            depth_panel.depth_preview.set_segment_overlay(
                mask_p,
                (float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=depth_panel._preview_intrinsics,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(result.u), float(result.v)),
                pose_position=_eai_platform_bootstrap.np.asarray(pose.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                pose_rotation=_eai_platform_bootstrap.np.asarray(pose.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                pose_axis_len=float(max(pose.obb_extents) * 0.55),
            )
            depth_panel._show_pose_6d(pose)

        camera_frame = self.node.resolve_segment_camera_frame(
            result.color_topic, result.depth_topic
        )
        if camera_frame:
            # 切换到「分割位姿」绝对目标（FP 物体 6D），不要走「相对当前」
            self._store_segment_target(
                _eai_perception_pose_estimation.SegmentPoseTarget.from_pose_result(
                    pose, camera_frame, result.color_topic
                )
            )
            self.status_bar.showMessage(
                f"FoundationPose 成功 ({result.pose_method})，已设为左臂绝对目标"
            )
        else:
            self.status_bar.showMessage(
                f"FoundationPose 成功 ({result.pose_method})，"
                "但缺少相机 frame_id，无法设为左臂绝对目标"
            )
        self._refresh_fp_status()

    def _on_fp_mesh_browse_clicked(self) -> None:
        initial = self.fp_mesh_edit.text().strip()
        if not initial or not _eai_platform_bootstrap.os.path.isfile(initial):
            initial = _eai_platform_bootstrap.os.path.dirname(_eai_configuration_perception.resolve_fp_mesh_path())
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择物体 mesh",
            initial if _eai_platform_bootstrap.os.path.isdir(initial) else _eai_platform_bootstrap.os.path.expanduser("~"),
            "Mesh (*.obj *.OBJ);;All Files (*)",
        )
        if path:
            self.fp_mesh_edit.setText(path)
            self._on_pose_settings_changed()

