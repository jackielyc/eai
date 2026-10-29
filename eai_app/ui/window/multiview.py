"""ui.window.multiview — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_application_main_window = module_ref('eai_app.application.main_window')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

class MultiviewMixin:
    def _on_click_pose_finished(self, result: dict) -> None:
        self._click_pose_busy = False
        if result.get("ok"):
            self._last_click_pose_snapshot = result
        if result.get("ok") and hasattr(self, "grasp_execution_panel"):
            self.grasp_execution_panel.set_observation(result)
        self.click_pose_result_edit.setPlainText(_eai_platform_bootstrap.click_pose.format_report(result))
        if result.get("ok"):
            panel = self.panels.get(result["topic"])
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                # The report is a timestamped snapshot; no persistent overlay on moving video.
                self.sam3_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(
                    self._click_pose_preview(result)))
            self.status_bar.showMessage("点击估计完成：仅展示，结果未转换为机械臂运动目标")
        else:
            self.status_bar.showMessage("点击估计失败：" + str(result.get("error", ""))[:160])

    def _click_pose_preview(self, result: dict):
        mask = _eai_platform_bootstrap.np.asarray(result["mask"], dtype=bool)
        canvas = result["image"].copy()
        canvas[mask] = (canvas[mask] * 0.6 + _eai_platform_bootstrap.np.array([30, 180, 30]) * 0.4).astype(_eai_platform_bootstrap.np.uint8)
        T = _eai_platform_bootstrap.np.asarray(result["object_pose"]["T_camera_object"])
        xyz = _eai_platform_bootstrap.np.vstack([T[:3, 3], T[:3, 3] + 0.05 * T[:3, :3].T])
        if _eai_platform_bootstrap.np.all(xyz[:, 2] > 0.01):
            projected = xyz @ _eai_platform_bootstrap.np.asarray(result["K"]).T
            uv = _eai_platform_bootstrap.np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
            for i, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                _eai_platform_bootstrap.cv2.line(canvas, tuple(uv[0]), tuple(uv[i + 1]), color, 3)
        _eai_platform_bootstrap.cv2.putText(canvas, "SNAPSHOT / object axes / NOT TCP", (10, 24),
                    _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        h, w = canvas.shape[:2]
        return _eai_platform_bootstrap.cv2.resize(canvas, (320, max(1, int(h * 320 / w))))

    def _on_sam3_click_uv(self, topic: str, u: int, v: int) -> None:
        clicked_color = not _eai_perception_image_geometry.is_depth_topic(topic)
        if _eai_perception_image_geometry.is_depth_topic(topic):
            color_topic = _eai_perception_image_geometry.find_paired_color_topic(topic, self._cached_topic_names())
            if color_topic:
                topic = color_topic
        self._last_sam3_topic = topic
        self.sam3_u_spin.setValue(max(0, int(u)))
        self.sam3_v_spin.setValue(max(0, int(v)))
        if hasattr(self, "camera_calib_panel"):
            self.camera_calib_panel.note_click(topic, int(u), int(v))
        panel = self.panels.get(topic)
        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
            h, w = panel._latest_image.shape[:2]
            self.sam3_u_spin.setMaximum(max(u, w - 1))
            self.sam3_v_spin.setMaximum(max(v, h - 1))
        self._update_segment_invoke_btns_ui()
        if hasattr(self, "lingbot_depth_topic_combo"):
            source = str(self.lingbot_depth_source_combo.currentData() or "camera")
            if source == "camera":
                self._select_lingbot_depth_topic(topic, from_click=True)
        if hasattr(self, "lingbot_video_topic_combo"):
            self._select_preview_topic_combo(
                self.lingbot_video_topic_combo,
                topic,
                status_cb=self._on_lingbot_video_status,
            )
        if hasattr(self, "lingbot_world_topic_combo"):
            self._select_preview_topic_combo(
                self.lingbot_world_topic_combo,
                topic,
                status_cb=self._on_lingbot_world_status,
            )
        if (
            clicked_color
            and getattr(self, "click_pose_check", None) is not None
            and self.click_pose_check.isChecked()
        ):
            clicked_panel = self.panels.get(topic)
            clicked_image = getattr(clicked_panel, "_latest_image", None)
            _eai_platform_bootstrap.QTimer.singleShot(
                0,
                lambda t=topic, x=u, y=v, img=clicked_image: self._run_click_pose(t, x, y, img),
            )

    def _on_stereo_invoke_clicked(self) -> None:
        """按钮确认：按提示点执行立体分割 + 6D 位姿。"""
        u = self.sam3_u_spin.value()
        v = self.sam3_v_spin.value()
        topic = self._last_sam3_topic or ""
        panel = self.panels.get(topic) if topic else None

        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
            msg = panel.invoke_segment_at_seed(u, v)
            self.status_bar.showMessage(msg)
            return

        # 深度面板：把彩色提示点映射到 depth 坐标
        if topic and not _eai_perception_image_geometry.is_depth_topic(topic):
            depth_topic = self._get_paired_depth_topic_name(topic)
            if depth_topic:
                depth_panel = self.panels.get(depth_topic)
                if isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D):
                    color = None
                    cam = self.panels.get(topic)
                    if isinstance(cam, _eai_ui_cameras_panels.CameraPanel) and cam._latest_image is not None:
                        color = cam._latest_image
                    if color is not None and depth_panel._latest_depth is not None:
                        dh, dw = depth_panel._depth_full_shape
                        u_d, v_d = _eai_perception_image_geometry.scale_uv_to_shape(
                            u, v, color.shape[:2], (dh, dw)
                        )
                        msg = depth_panel.invoke_segment_at_seed(u_d, v_d)
                        self.status_bar.showMessage(msg)
                        return

        if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
            msg = panel.invoke_segment_at_seed()
            self.status_bar.showMessage(msg)
            return

        # 回退：任一面板上有待处理提示点
        for p in self.panels.values():
            if isinstance(p, _eai_ui_cameras_panels.CameraPanel) and p._pending_seed_uv is not None:
                msg = p.invoke_segment_at_seed()
                self.status_bar.showMessage(msg)
                return
            if isinstance(p, _eai_ui_cameras_panels.DepthPanel3D) and p._pending_seed_full_uv is not None:
                msg = p.invoke_segment_at_seed()
                self.status_bar.showMessage(msg)
                return

        self.status_bar.showMessage("请先点击彩色/深度图像选择提示点，再点「调用分割」")

    def _on_sam3_invoke_clicked(self) -> None:
        if self._sam3_call_busy:
            return
        source = self._pick_sam3_color_source()
        if source is None:
            self._set_sam3_result_text(
                "错误: 无可用彩色图像\n请勾选 color topic 并等待图像到达"
            )
            self.status_bar.showMessage("SAM3: 无可用彩色图像")
            return

        topic, image, _panel = source
        h, w = image.shape[:2]
        u = max(0, min(w - 1, self.sam3_u_spin.value()))
        v = max(0, min(h - 1, self.sam3_v_spin.value()))
        self.sam3_u_spin.setMaximum(max(u, w - 1))
        self.sam3_v_spin.setMaximum(max(v, h - 1))
        self.sam3_u_spin.setValue(u)
        self.sam3_v_spin.setValue(v)

        backend = self.segment_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = _eai_configuration_perception.SAM3_BACKEND_POINT
        if backend == _eai_configuration_perception.SAM3_BACKEND_GEOMETRY:
            backend = _eai_configuration_perception.SAM3_BACKEND_POINT

        self._on_segment_settings_changed()
        cfg = _eai_perception_segmentation.get_segment_settings()
        text = self.sam3_text_edit.text().strip() if backend == _eai_configuration_perception.SAM3_BACKEND_TEXT else ""
        transport = "HTTP" if cfg.sam3_use_http else "子进程"

        self._sam3_call_busy = True
        self.sam3_invoke_btn.setEnabled(False)
        self.sam3_invoke_btn.setText("SAM3 调用中…")
        self._set_sam3_result_text(
            f"正在调用 SAM3…\n图像: {topic}\n"
            f"模型: {cfg.sam3_model}\n"
            + (f"文本: {text}\n" if text else f"点: ({u}, {v})\n")
            + f"传输: {transport}"
        )

        def _work() -> None:
            t0 = _eai_platform_bootstrap.time.time()
            try:
                mask, method = _eai_perception_segmentation.run_sam3_segmentation(
                    image, u, v, text=text or None, settings=cfg, tag="invoke_btn"
                )
                if not mask.any():
                    raise RuntimeError("SAM3 返回空 mask")
                if int(mask.sum()) < _eai_configuration_perception.SEGMENT_MIN_AREA:
                    raise RuntimeError(
                        f"SAM3 mask 过小 ({int(mask.sum())} px < {_eai_configuration_perception.SEGMENT_MIN_AREA})"
                    )
                ys, xs = _eai_platform_bootstrap.np.where(mask)
                centroid = (
                    (float(_eai_platform_bootstrap.np.mean(xs)), float(_eai_platform_bootstrap.np.mean(ys)))
                    if len(xs) > 0
                    else (float(u), float(v))
                )
                result = _eai_perception_segmentation.Sam3CallResult(
                    ok=True,
                    topic=topic,
                    u=u,
                    v=v,
                    text=text,
                    method=method,
                    pixel_count=int(mask.sum()),
                    mask_shape=(mask.shape[0], mask.shape[1]),
                    centroid_uv=centroid,
                    transport=transport,
                    elapsed_s=_eai_platform_bootstrap.time.time() - t0,
                    mask=mask,
                )
            except Exception as exc:
                result = _eai_perception_segmentation.Sam3CallResult(
                    ok=False,
                    topic=topic,
                    u=u,
                    v=v,
                    text=text,
                    transport=transport,
                    elapsed_s=_eai_platform_bootstrap.time.time() - t0,
                    error=str(exc),
                )
            self._sam3_call_bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_sam3_call_finished(self, result: object) -> None:
        self._sam3_call_busy = False
        self.sam3_invoke_btn.setText("调用 SAM3")
        self._update_segment_invoke_btns_ui()
        if not isinstance(result, _eai_perception_segmentation.Sam3CallResult):
            return

        self._set_sam3_result_text(_eai_perception_segmentation.format_sam3_call_result_text(result))
        if result.ok and result.mask is not None:
            panel = self.panels.get(result.topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                panel.show_sam3_mask(result.mask, result.u, result.v)
                src = panel._latest_image
            else:
                src = None
            self._set_segment_preview(
                src,
                result.mask,
                seed_uv=(float(result.u), float(result.v)),
                centroid_uv=result.centroid_uv,
            )
            self.status_bar.showMessage(
                f"SAM3 成功: {result.pixel_count} px ({result.method})"
            )
        elif result.ok:
            self.status_bar.showMessage("SAM3 成功")
        else:
            self._set_segment_preview(None, None)
            self.status_bar.showMessage(f"SAM3 失败: {result.error[:120]}")

    def _on_segment_settings_changed(self, *_args) -> None:
        backend = self.segment_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = _eai_configuration_perception.SAM3_BACKEND_GEOMETRY
        use_sam3 = backend != _eai_configuration_perception.SAM3_BACKEND_GEOMETRY
        self.sam3_text_edit.setEnabled(backend == _eai_configuration_perception.SAM3_BACKEND_TEXT)
        self.sam3_http_check.setEnabled(use_sam3)
        _eai_perception_segmentation.set_segment_settings(
            backend=backend,
            sam3_text=self.sam3_text_edit.text().strip(),
            sam3_use_http=self.sam3_http_check.isChecked() and use_sam3,
            sam3_server_url=_eai_perception_segmentation.resolve_sam3_viewer_server_url()
            if (self.sam3_http_check.isChecked() and use_sam3)
            else _eai_configuration_perception.SAM3_SERVER_URL_DEFAULT,
            sam3_model=_eai_configuration_perception.resolve_sam3_model_path(),
        )
        self._refresh_sam3_status()
        self._update_segment_invoke_btns_ui()

    def _on_mv6d_enable_toggled(self, checked: bool) -> None:
        self._mv6d_timer.stop()
        self.mv6d_enable_check.blockSignals(True)
        self.mv6d_enable_check.setChecked(False)
        self.mv6d_enable_check.blockSignals(False)
        if checked:
            self.status_bar.showMessage('开发缓存模式：请点击“估计一次”更新结果')

    def _on_mv6d_interval_changed(self, value: float) -> None:
        ms = max(200, int(float(value) * 1000.0))
        self._mv6d_timer.setInterval(ms)

    def _on_mv6d_timer_tick(self) -> None:
        # Cached developer workflow: only the explicit once button updates results.
        return

    def _on_mv6d_estimate_once(self) -> None:
        if not self._mv6d_ensure_object_selected(silent=False):
            return
        targets = list(getattr(self, '_mv6d_targets', []))
        if not targets:
            return
        self._mv6d_batch_active = True
        self._mv6d_batch_capture = None
        self._mv6d_batch_depth_cache = None
        self._mv6d_estimate_queue = [target['id'] for target in targets]
        for target in targets:
            self._mv6d_target_results.pop(target['id'], None)
        self._refresh_mv6d_target_combo()
        self._start_next_mv6d_target()

    def _on_mv6d_show_robot(self) -> None:
        holder = getattr(self, "_robot_view_holder", None)
        tabs = getattr(self, "arm_workspace_tabs", None)
        if holder is None or tabs is None:
            return
        for index in range(tabs.count()):
            if _eai_ui_widgets_layouts._tab_page_matches(tabs.widget(index), holder):
                tabs.setCurrentIndex(index)
                break

    def _on_mv6d_open_rviz(self) -> None:
        """启动 RViz，并带上 robot_description / joint_states TF（宿主机脚本）。"""
        script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_rviz.sh",
        )
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_bar.showMessage(f"缺少脚本: {script}", 8000)
            return
        domain = _eai_platform_bootstrap.os.environ.get("ROS_DOMAIN_ID", "0")
        localhost = _eai_platform_bootstrap.os.environ.get("ROS_LOCALHOST_ONLY", "0")
        dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
        discovery = _eai_platform_bootstrap.os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        exports = [
            f"export ROS_DOMAIN_ID={_eai_platform_bootstrap.shlex.quote(str(domain))}",
            f"export ROS_LOCALHOST_ONLY={_eai_platform_bootstrap.shlex.quote(str(localhost))}",
            f"export ROS_AUTOMATIC_DISCOVERY_RANGE={_eai_platform_bootstrap.shlex.quote(str(discovery))}",
            "export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}",
        ]
        if dds and _eai_platform_bootstrap.os.path.isfile(dds):
            exports.append(
                f"export FASTRTPS_DEFAULT_PROFILES_FILE={_eai_platform_bootstrap.shlex.quote(dds)}"
            )
        cmd = " && ".join(exports + [f"exec bash {_eai_platform_bootstrap.shlex.quote(script)} start"])
        ok = _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", cmd])
        if ok:
            self.status_bar.showMessage(
                "正在启动 RViz（robot_description + TF）…", 5000
            )
        else:
            self.status_bar.showMessage("启动 RViz 失败", 8000)

    def _mv6d_update_seed_status_label(self) -> None:
        lab = getattr(self, "mv6d_seed_status_label", None)
        if lab is None:
            return
        if self._mv6d_object_picked:
            u = int(self.mv6d_head_u_spin.value())
            v = int(self.mv6d_head_v_spin.value())
            count = len(getattr(self, '_mv6d_targets', []))
            lab.setText(f"当前目标 ({u}, {v}) · 共 {count} 个 · 可批量估计")
            lab.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
        else:
            lab.setText("未选择物体 · 请点击头部预览")
            lab.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")

    def _mv6d_mark_object_picked(self, u: int, v: int, *, source: str = "预览") -> None:
        self._mv6d_target_serial += 1
        target = {'id': f'target-{self._mv6d_target_serial}', 'u': int(u), 'v': int(v)}
        self._mv6d_targets.append(target)
        self._refresh_mv6d_target_combo(target['id'])
        self.mv6d_gt_enable_check.setChecked(False)
        self._set_mv6d_active_target(target, show_result=False)
        self.status_bar.showMessage(
            f"多视角 6D：新增目标 {len(self._mv6d_targets)} ({int(u)}, {int(v)})（{source}）"
        )

    def _on_mv6d_target_changed(self, *_args) -> None:
        target_id = str(self.mv6d_target_combo.currentData() or '')
        target = next((item for item in self._mv6d_targets if item['id'] == target_id), None)
        if target is not None:
            self._set_mv6d_active_target(target)

    def _on_mv6d_seed_spin_changed(self, *_args) -> None:
        self.mv6d_gt_enable_check.setChecked(False)
        # 手动改 spin 也视为已选，避免只能点图
        if not getattr(self, "_mv6d_object_picked", False):
            # 未点选过时改数字不自动解锁，避免默认 320,240 误触发
            self._mv6d_update_seed_status_label()
            return
        target = getattr(self, '_mv6d_active_target', None)
        if target is not None:
            target['u'] = int(self.mv6d_head_u_spin.value())
            target['v'] = int(self.mv6d_head_v_spin.value())
            self._mv6d_target_results.pop(target['id'], None)
            self._refresh_mv6d_target_combo(target['id'])
        self._mv6d_update_seed_status_label()
        self._refresh_mv6d_live_previews_for_pick()

    def _on_mv6d_head_preview_picked(self, u: int, v: int) -> None:
        self._mv6d_mark_object_picked(u, v, source="头部预览点击")

    def _mv6d_downscale_preview_bgr(self, bgr: _eai_platform_bootstrap.np.ndarray, max_edge: int = 320) -> _eai_platform_bootstrap.np.ndarray:
        img = _eai_platform_bootstrap.np.asarray(bgr)
        if img.size == 0:
            return img
        h, w = img.shape[:2]
        edge = max(h, w)
        if edge <= max_edge or max_edge <= 0:
            return img
        scale = float(max_edge) / float(edge)
        return _eai_platform_bootstrap.cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
        )

    def _mv6d_overlay_hand_skeleton(self, images: dict) -> dict:
        label = getattr(self, "mv6d_hand_skel_label", None)
        check = getattr(self, "mv6d_hand_skel_check", None)
        if not images or check is None or not check.isChecked():
            if label is not None and check is not None and not check.isChecked():
                label.setText("手骨架：已关闭")
            return dict(images)
        try:
            from eai_app.domain.robot.hand_geometry import overlay_views

            state = self.node.get_robot_state()
            tcp = {}
            for side in ("left", "right"):
                pose = self._calib_tcp(side)
                if pose is not None:
                    tcp[side] = pose
            motors = {
                "left": {name: value for name, value in state.left_hand},
                "right": {name: value for name, value in state.right_hand},
            }
            drawn, summary = overlay_views(images, tcp, motors, self._calib_tcp)
            if label is not None:
                label.setText(summary)
            merged = dict(images)
            merged.update(drawn)
            return merged
        except Exception as exc:
            if label is not None:
                label.setText("手骨架失败：" + str(exc)[:120])
            return dict(images)

    def _mv6d_ensure_object_selected(self, *, silent: bool = False) -> bool:
        if getattr(self, "_mv6d_object_picked", False):
            return True
        self._refresh_mv6d_live_previews_for_pick()
        if not silent:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "多视角 6D",
                "请先在「头部」预览图上点击要估计的物体，再点「估计一次」。\n\n"
                "也可点「同步分割点」从分割页导入提示点。",
            )
            self.status_bar.showMessage("多视角 6D：请先在头部预览图上点击选择物体")
            self._mv6d_update_seed_status_label()
        return False

    def _on_mv6d_sync_seed_from_segment(self) -> None:
        if hasattr(self, "sam3_u_spin") and hasattr(self, "sam3_v_spin"):
            u = int(self.sam3_u_spin.value())
            v = int(self.sam3_v_spin.value())
            self._mv6d_mark_object_picked(u, v, source="同步分割点")
        else:
            self.status_bar.showMessage("分割页提示点控件不可用")

    def _on_mv6d_gt_params_changed(self, *_args) -> None:
        snap = self._mv6d_last_snap
        if snap is not None:
            self._refresh_mv6d_result_text(snap)
            if self.mv6d_overlay_check.isChecked():
                try:
                    self._apply_mv6d_overlays(snap)
                except Exception:
                    pass

    def _on_mv6d_overlay_toggled(self, checked: bool) -> None:
        snap = self._mv6d_last_snap
        if checked:
            if snap is not None:
                try:
                    self._apply_mv6d_overlays(snap)
                except Exception as exc:
                    self.status_bar.showMessage(f"多视角 6D 叠加失败: {exc}")
            self._refresh_mv6d_live_previews_for_pick()
            return
        # 关闭叠加时清掉头/双腕彩色与配对深度上的 6D 层
        topics = [
            "/camera/head_color",
            "/camera/left_wrist_color",
            "/camera/right_wrist_color",
        ]
        if snap is not None:
            for t in (snap.head_topic, snap.wrist_topic):
                if t and t not in topics:
                    topics.append(t)
        for color_topic in topics:
            panel = self.panels.get(color_topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                panel.image_label.clear_segment_overlay()
            depth_topic = self._get_paired_depth_topic_name(color_topic)
            if depth_topic:
                depth_panel = self.panels.get(depth_topic)
                if isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D):
                    depth_panel.depth_preview.clear_segment_overlay()
        self._set_mv6d_preview_label(
            self.mv6d_head_preview,
            None,
            "头部 · 点击选择物体",
            cache_attr="_mv6d_head_pix",
        )
        self._set_mv6d_preview_label(
            self.mv6d_left_wrist_preview,
            None,
            "左腕 · 点击放大",
            cache_attr="_mv6d_left_wrist_pix",
        )
        self._set_mv6d_preview_label(
            self.mv6d_right_wrist_preview,
            None,
            "右腕 · 点击放大",
            cache_attr="_mv6d_right_wrist_pix",
        )
        self._mv6d_wrist_pix = self._mv6d_left_wrist_pix
        self._sync_mv6d_lightbox()
        self._refresh_mv6d_live_previews_for_pick()

    def _mv6d_sim_reference_is_current(self) -> bool:
        state = self._read_sim_robot_state()
        return bool(_eai_configuration_policies.sim_state_is_fresh is not None and _eai_configuration_policies.sim_state_is_fresh(state, 5.0))

    def _mv6d_gt_pose(
        self,
    ) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]]:
        if not self.mv6d_gt_enable_check.isChecked():
            return None, None
        xyz = (
            float(self.mv6d_gt_x_spin.value()),
            float(self.mv6d_gt_y_spin.value()),
            float(self.mv6d_gt_z_spin.value()),
        )
        R = _eai_perception_pose_estimation.euler_xyz_to_rotation_matrix(
            _eai_platform_bootstrap.math.radians(float(self.mv6d_gt_roll_spin.value())),
            _eai_platform_bootstrap.math.radians(float(self.mv6d_gt_pitch_spin.value())),
            _eai_platform_bootstrap.math.radians(float(self.mv6d_gt_yaw_spin.value())),
        )
        return xyz, R

    def _on_mv6d_load_sim_gt(self) -> None:
        if not self._mv6d_sim_reference_is_current():
            self.status_bar.showMessage("没有当前运行的仿真状态；请手动填写并确认当前目标参考位姿")
            return
        root = ""
        try:
            root = self._sim_bridge_dir()
        except Exception:
            root = ""
        candidates = []
        if root:
            candidates.append(_eai_platform_bootstrap.os.path.join(root, "object_gt_pose.json"))
        candidates.append(_eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "object_gt_pose.json"))
        path = next((p for p in candidates if p and _eai_platform_bootstrap.os.path.isfile(p)), "")
        if not path:
            self.status_bar.showMessage(
                "未找到 object_gt_pose.json（仿真共享目录或 eai/.cache）"
            )
            return
        try:
            if _eai_platform_bootstrap.time.time() - _eai_platform_bootstrap.os.path.getmtime(path) > 5.0:
                raise ValueError("仿真参考文件已过期")
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            if _eai_adapters_robot_geometry.normalize_frame_id(str(data.get("frame_id") or _eai_configuration_policies.BASE_LINK_FRAME)) != _eai_configuration_policies.BASE_LINK_FRAME:
                raise ValueError("仿真参考必须使用 base_link 坐标系")
            xyz = data.get("xyz") or data.get("position") or data.get("t")
            if not xyz or len(xyz) < 3:
                raise ValueError("缺少 xyz")
            self.mv6d_gt_x_spin.setValue(float(xyz[0]))
            self.mv6d_gt_y_spin.setValue(float(xyz[1]))
            self.mv6d_gt_z_spin.setValue(float(xyz[2]))
            if data.get("rpy_deg") is not None:
                rpy = data["rpy_deg"]
                self.mv6d_gt_roll_spin.setValue(float(rpy[0]))
                self.mv6d_gt_pitch_spin.setValue(float(rpy[1]))
                self.mv6d_gt_yaw_spin.setValue(float(rpy[2]))
            elif data.get("rpy") is not None:
                rpy = data["rpy"]
                self.mv6d_gt_roll_spin.setValue(_eai_platform_bootstrap.math.degrees(float(rpy[0])))
                self.mv6d_gt_pitch_spin.setValue(_eai_platform_bootstrap.math.degrees(float(rpy[1])))
                self.mv6d_gt_yaw_spin.setValue(_eai_platform_bootstrap.math.degrees(float(rpy[2])))
            elif data.get("quat_xyzw") is not None:
                q = data["quat_xyzw"]
                R = _eai_adapters_robot_geometry.quat_to_rotation_matrix(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                r, p, y = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(R)
                self.mv6d_gt_roll_spin.setValue(_eai_platform_bootstrap.math.degrees(r))
                self.mv6d_gt_pitch_spin.setValue(_eai_platform_bootstrap.math.degrees(p))
                self.mv6d_gt_yaw_spin.setValue(_eai_platform_bootstrap.math.degrees(y))
            self.mv6d_gt_enable_check.setChecked(True)
            self._on_mv6d_gt_params_changed()
            self.status_bar.showMessage(f"已加载仿真 GT: {path}")
        except Exception as exc:
            self.status_bar.showMessage(f"加载 GT 失败: {exc}")

    def _mv6d_color_topic_candidates(self, preferred: str) -> _eai_platform_bootstrap.List[str]:
        """展开腕部 topic 别名（left_wrist ↔ hand_left 等）。"""
        preferred = str(preferred or "").strip()
        candidates: _eai_platform_bootstrap.List[str] = []
        if preferred in ("left", "right"):
            candidates.extend(self._wrist_color_topics(preferred))
        else:
            candidates.append(preferred)
            for side, aliases in _eai_configuration_policies.WRIST_COLOR_TOPIC_ALIASES.items():
                if preferred in aliases:
                    for t in aliases:
                        if t not in candidates:
                            candidates.append(t)
                    break
            # 逻辑侧名夹在路径里时也展开
            low = preferred.lower()
            if "hand_left" in low or "left_wrist" in low:
                for t in self._wrist_color_topics("left"):
                    if t not in candidates:
                        candidates.append(t)
            if "hand_right" in low or "right_wrist" in low:
                for t in self._wrist_color_topics("right"):
                    if t not in candidates:
                        candidates.append(t)
        known = self._cached_topic_names()
        for t in list(candidates):
            stem = t.rstrip("/").split("/")[-1]
            for k in known:
                if k not in candidates and stem and stem in k:
                    candidates.append(k)
        return [c for c in candidates if c]

    def _mv6d_lingbot_model_id(self) -> str:
        """跟随深度精修配置，优先使用用户指定的本地权重。"""
        edit = getattr(self, "lingbot_depth_model_edit", None)
        if edit is not None:
            custom = str(edit.text() or "").strip()
            if custom:
                return custom
        combo = getattr(self, "lingbot_depth_model_combo", None)
        if combo is not None:
            mid = str(combo.currentData() or "").strip()
            if mid:
                return mid
        return _eai_configuration_tabs.LINGBOT_DEPTH_MODEL_DEFAULT

    def _mv6d_enabled_methods(self) -> _eai_platform_bootstrap.Tuple[str, ...]:
        return tuple(
            backend for backend, attr in (
                (_eai_configuration_perception.POSE_BACKEND_PCA, "mv6d_pca_enabled_check"),
                (_eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE, "mv6d_foundationpose_enabled_check"),
                (_eai_configuration_perception.POSE_BACKEND_ANY6D, "mv6d_any6d_enabled_check"),
            )
            if getattr(self, attr, None) is None or getattr(self, attr).isChecked()
        )

    def _on_mv6d_methods_changed(self, *_args) -> None:
        enabled = self._mv6d_enabled_methods()
        busy = bool(getattr(self, "_mv6d_busy", False))
        for attr in ("mv6d_pca_enabled_check", "mv6d_foundationpose_enabled_check", "mv6d_any6d_enabled_check"):
            check = getattr(self, attr, None)
            if check is not None:
                check.setEnabled(not busy)
        combo = getattr(self, "mv6d_method_combo", None)
        if combo is not None:
            for index in range(combo.count()):
                combo.model().item(index).setEnabled(combo.itemData(index) in enabled)
            if enabled and combo.currentData() not in enabled:
                combo.setCurrentIndex(combo.findData(enabled[0]))
            combo.setEnabled(bool(enabled) and not busy)
        button = getattr(self, "mv6d_once_btn", None)
        if button is not None:
            button.setEnabled(bool(enabled) and not busy)
        realtime = getattr(self, "mv6d_enable_check", None)
        if realtime is not None:
            if not enabled:
                realtime.setChecked(False)
            realtime.setEnabled(bool(enabled))
        stored = getattr(self, "_robot_method_poses", {})
        view = getattr(self, "_robot_view", None)
        for backend in _eai_configuration_perception.POSE_BACKENDS:
            if backend not in enabled:
                stored.pop(backend, None)
                if view is not None and hasattr(view, "set_method_pose"):
                    view.set_method_pose(backend, None, None)

    @staticmethod
    def _estimate_view_6d(
        color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        seed_u: int,
        seed_v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        method_tag: str,
        assumed_depth_m: float = 0.45,
        pose_backend: str = _eai_configuration_perception.POSE_BACKEND_PCA,
    ) -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult]:
        """先用点击点做 SAM3，再按 PCA / FoundationPose / Any6D 出 6D。"""
        if color_bgr is None:
            return None
        backend = str(pose_backend or _eai_configuration_perception.POSE_BACKEND_PCA)
        ch, cw = color_bgr.shape[:2]
        su = max(0, min(cw - 1, int(seed_u)))
        sv = max(0, min(ch - 1, int(seed_v)))
        mask, seg_method = _eai_perception_segmentation.run_sam3_segmentation(
            color_bgr, su, sv, text=None, tag=f"mv6d-{method_tag}"
        )
        if mask is None or not _eai_platform_bootstrap.np.any(mask):
            raise RuntimeError("SAM3 没有分出区域")
        raw_mask = _eai_platform_bootstrap.np.asarray(mask, dtype=bool).copy()
        debug_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "sam3", f"mv6d-{method_tag}")
        _eai_platform_bootstrap.os.makedirs(debug_dir, exist_ok=True)
        _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(debug_dir, "rgb.png"), color_bgr)
        _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(debug_dir, "mask_raw.png"), raw_mask.astype(_eai_platform_bootstrap.np.uint8) * 255)
        overlay = color_bgr.copy()
        overlay[raw_mask] = (0.55 * overlay[raw_mask] + 0.45 * _eai_platform_bootstrap.np.array([0, 255, 0])).astype(_eai_platform_bootstrap.np.uint8)
        _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(debug_dir, "overlay_raw.png"), overlay)
        # Reset the postprocessed mask so a failed new run cannot show an older mask.
        _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(debug_dir, "mask_depth_filtered.png"), _eai_platform_bootstrap.np.zeros((ch, cw), dtype=_eai_platform_bootstrap.np.uint8))
        if depth is None:
            if backend != _eai_configuration_perception.POSE_BACKEND_PCA:
                raise RuntimeError(f"{backend} 需要深度图")
            depth_use = _eai_platform_bootstrap.np.full((ch, cw), float(assumed_depth_m), dtype=_eai_platform_bootstrap.np.float32)
            u_d, v_d = su, sv
            seg_method = f"{seg_method}+plane"
        else:
            dh, dw = depth.shape[:2]
            if mask.shape[:2] != (dh, dw):
                mask = _eai_perception_image_geometry.resize_mask_to_shape(mask, (dh, dw))
            u_d, v_d = _eai_perception_image_geometry.scale_uv_to_shape(su, sv, (ch, cw), (dh, dw))
            depth_use = depth
        result = _eai_application_main_window.CameraTopicWindow._pose_on_mask(
            color_bgr,
            depth_use,
            mask,
            u_d,
            v_d,
            fx,
            fy,
            cx,
            cy,
            backend,
            method_tag=method_tag,
            seg_method=seg_method,
        )

        if result is not None:
            result.raw_mask = raw_mask
            result.source_bgr = color_bgr.copy()
            result.intrinsics = (float(fx), float(fy), float(cx), float(cy))
            result.diagnostics_dir = debug_dir
            _eai_platform_bootstrap.cv2.imwrite(_eai_platform_bootstrap.os.path.join(debug_dir, "mask_depth_filtered.png"), result.mask.astype(_eai_platform_bootstrap.np.uint8) * 255)
        return result

    @staticmethod
    def _pose_on_mask(
        color_bgr,
        depth,
        mask,
        u_d: int,
        v_d: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        backend: str,
        *,
        method_tag: str,
        seg_method: str,
    ) -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult]:
        depth = _eai_platform_bootstrap.sanitize_depth(depth)
        mask = _eai_platform_bootstrap.np.asarray(mask, dtype=bool) & _eai_platform_bootstrap.valid_depth_mask(depth)
        if not _eai_platform_bootstrap.np.any(mask):
            raise RuntimeError("分割区域没有有效深度（0.08–2.5m）")
        if backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE:
            body = _eai_perception_segmentation.run_foundationpose_estimation(
                color_bgr,
                depth,
                mask,
                fx,
                fy,
                cx,
                cy,
                settings=_eai_perception_segmentation.get_pose_settings(),
                tag=f"mv6d-{method_tag}",
            )
            return _eai_perception_segmentation.object6d_from_foundationpose(
                mask,
                depth,
                u_d,
                v_d,
                fx,
                fy,
                cx,
                cy,
                body,
                segment_method=f"sam3/{seg_method}",
            )
        if backend == _eai_configuration_perception.POSE_BACKEND_ANY6D:
            body = _eai_perception_pose_estimation.run_any6d_estimation(
                color_bgr,
                depth,
                mask,
                fx,
                fy,
                cx,
                cy,
                settings=_eai_perception_segmentation.get_pose_settings(),
                tag=f"mv6d-{method_tag}",
            )
            return _eai_perception_segmentation.object6d_from_foundationpose(
                mask,
                depth,
                u_d,
                v_d,
                fx,
                fy,
                cx,
                cy,
                body,
                segment_method=f"sam3/{seg_method}",
            )
        return _eai_perception_pose_estimation.compute_stereo_segment_6d_pose(
            mask,
            depth,
            u_d,
            v_d,
            fx,
            fy,
            cx,
            cy,
            method=f"{method_tag}/sam3+pca/{seg_method}",
        )

