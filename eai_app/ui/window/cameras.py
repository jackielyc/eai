"""ui.window.cameras — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class CamerasMixin:
    def _on_local_cam_tab_help(self) -> None:
        _eai_platform_bootstrap.QMessageBox.information(self, "本地摄像头", _eai_configuration_chat.LOCAL_WEBCAM_TAB_HELP)

    def _on_local_cam_copy_mac_helper(self) -> None:
        cmd = f"cd {_eai_platform_bootstrap.shlex.quote(_eai_configuration_models.EAI_DIR)} && python3 examples/local_webcam_forward_ui.py"
        _eai_platform_bootstrap.QApplication.clipboard().setText(cmd)
        self._append_local_cam_log(f"[助手] 已复制: {cmd}")
        self.status_bar.showMessage("已复制 Mac 助手启动命令到剪贴板", 4000)

    def _on_local_cam_port_changed(self, value: int) -> None:
        url = f"http://127.0.0.1:{int(value)}/cam.mjpg"
        if hasattr(self, "local_cam_url_edit") and not getattr(
            self, "_local_cam_preview_active", False
        ):
            self.local_cam_url_edit.setText(url)
        if hasattr(self, "real_eval_cam_url_edit") and not getattr(
            self, "_real_eval_local_cam_active", False
        ):
            self.real_eval_cam_url_edit.setText(url)

    def _on_local_cam_refresh_devices(self) -> None:
        self._refresh_local_cam_device_list()
        self.status_bar.showMessage("已刷新本机摄像头设备列表", 3000)

    def _on_local_cam_device_combo_changed(self, _index: int) -> None:
        if getattr(self, "_local_cam_device_combo_syncing", False):
            return
        data = self.local_cam_device_combo.currentData()
        if data is None or not hasattr(self, "local_cam_device_edit"):
            return
        self.local_cam_device_edit.setText(str(data))

    def _on_local_cam_device_edit_finished(self) -> None:
        if not hasattr(self, "local_cam_device_combo"):
            return
        want = self.local_cam_device_edit.text().strip()
        if want.startswith("/dev/video") and want[len("/dev/video") :].isdigit():
            want = want[len("/dev/video") :]
            self.local_cam_device_edit.setText(want)
        for i in range(self.local_cam_device_combo.count()):
            if str(self.local_cam_device_combo.itemData(i)) == want:
                self._local_cam_device_combo_syncing = True
                self.local_cam_device_combo.setCurrentIndex(i)
                self._local_cam_device_combo_syncing = False
                return

    def _on_local_cam_mjpeg_toggled(self) -> None:
        launcher = getattr(self, "_local_cam_mjpeg_launcher", None)
        if launcher is None:
            return
        if launcher.is_running():
            launcher.stop()
            return
        if not self._confirm_local_cam_on_remote_host("启动推流"):
            return
        device = self.local_cam_device_edit.text().strip() or "0"
        port = int(self.local_cam_port_spin.value())
        launcher.start(device=device, port=port, bind="127.0.0.1")

    def _on_local_cam_mjpeg_status(self, msg: str) -> None:
        self.status_bar.showMessage(msg, 5000)
        self._append_local_cam_log(f"[推流] {msg}")

    def _on_local_cam_mjpeg_running(self, running: bool) -> None:
        if not hasattr(self, "local_cam_mjpeg_btn"):
            return
        self.local_cam_mjpeg_btn.setText("停止推流" if running else "启动推流")
        port = int(self.local_cam_port_spin.value())
        if running:
            self.local_cam_mjpeg_status.setText(f"推流: 运行中 :{port}")
            self.local_cam_mjpeg_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self._set_local_cam_device_controls_enabled(False)
            self.local_cam_port_spin.setEnabled(False)
        else:
            self.local_cam_mjpeg_status.setText("推流: 未启动")
            self.local_cam_mjpeg_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            if getattr(self, "_local_cam_preview_mode", "") != "device":
                self._set_local_cam_device_controls_enabled(True)
            self.local_cam_port_spin.setEnabled(True)

    def _on_local_cam_tunnel_toggled(self) -> None:
        launcher = getattr(self, "_local_cam_tunnel_launcher", None)
        if launcher is None:
            return
        if launcher.is_running():
            launcher.stop()
            return
        if not self._confirm_local_cam_on_remote_host("启动隧道"):
            return
        target = self.local_cam_ssh_target_edit.text().strip()
        if "remote-host" in target:
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "反向隧道",
                "请把「远端」改成真实地址，例如 yourname@10.0.0.5",
            )
            return
        launcher.start(
            target,
            local_port=int(self.local_cam_port_spin.value()),
            remote_port=int(self.local_cam_remote_port_spin.value()),
            extra_args=self.local_cam_ssh_extra_edit.text(),
        )

    def _on_local_cam_tunnel_status(self, msg: str) -> None:
        self.status_bar.showMessage(msg, 5000)
        self._append_local_cam_log(f"[隧道] {msg}")

    def _on_local_cam_tunnel_running(self, running: bool) -> None:
        if not hasattr(self, "local_cam_tunnel_btn"):
            return
        self.local_cam_tunnel_btn.setText("停止隧道" if running else "启动隧道")
        if running:
            target = self.local_cam_ssh_target_edit.text().strip()
            rport = int(self.local_cam_remote_port_spin.value())
            self.local_cam_tunnel_status.setText(f"隧道: 运行中 → {target}:{rport}")
            self.local_cam_tunnel_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self.local_cam_ssh_target_edit.setEnabled(False)
            self.local_cam_remote_port_spin.setEnabled(False)
            self.local_cam_ssh_extra_edit.setEnabled(False)
        else:
            self.local_cam_tunnel_status.setText("隧道: 未启动")
            self.local_cam_tunnel_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            self.local_cam_ssh_target_edit.setEnabled(True)
            self.local_cam_remote_port_spin.setEnabled(True)
            self.local_cam_ssh_extra_edit.setEnabled(True)

    def _on_local_cam_monitor_toggled(self) -> None:
        if self._local_cam_preview_active and getattr(
            self, "_local_cam_preview_mode", ""
        ) == "device":
            self._stop_local_cam_preview()
            self._append_local_cam_log("[监控] 已停止")
            self.status_bar.showMessage("本机摄像头监控已停止", 3000)
            return
        self._ensure_local_cam_monitor_open(silent=False)

    def _on_local_cam_preview_toggled(self) -> None:
        if self._local_cam_preview_active and getattr(
            self, "_local_cam_preview_mode", ""
        ) == "url":
            self._stop_local_cam_preview()
            self._append_local_cam_log("[预览] 已停止")
            return
        if self._local_cam_preview_active:
            self._stop_local_cam_preview()
        url = self.local_cam_url_edit.text().strip() or _eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT
        self.local_cam_url_edit.setText(url)
        self.local_cam_preview_status.setText(f"预览: 连接中… {url}")
        _eai_platform_bootstrap.QApplication.processEvents()
        ok, msg = self._open_local_cam_preview_source(url)
        if not ok:
            # 不再自动回退打开本机摄像头；仅手动点「打开本机监控」
            _eai_platform_bootstrap.QMessageBox.warning(self, "本地摄像头预览", msg)
            self.local_cam_preview_status.setText("预览: 连接失败")
            self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            return
        self._start_local_cam_preview_ui(msg, "url")
        self._append_local_cam_log(f"[预览] 已开启: {msg}")

    def _on_local_cam_rotate_clicked(self) -> None:
        cur = int(getattr(self, "_local_cam_rotate_deg", 0) or 0) % 360
        self._local_cam_rotate_deg = (cur + 90) % 360
        self._update_local_cam_rotate_btn()
        deg = self._local_cam_rotate_deg
        self._append_local_cam_log(f"[预览] 旋转 → {deg}°")
        self.status_bar.showMessage(f"本地摄像头画面旋转 {deg}°", 2500)

    def _on_local_cam_preview_tick(self) -> None:
        if not self._local_cam_preview_active or self._local_cam_preview_busy:
            return
        cap = self._local_cam_preview_cap
        if cap is None:
            return
        self._local_cam_preview_busy = True
        try:
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                self.local_cam_preview_status.setText("预览: 等待画面…")
                return
            frame = self._apply_local_cam_rotation(frame)
            max_w = max(320, self.local_cam_preview_label.width() - 8)
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                frame = _eai_platform_bootstrap.cv2.resize(
                    frame,
                    (max_w, max(1, int(h * scale))),
                    interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
                )
            self.local_cam_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(frame))
        except Exception as exc:
            self.local_cam_preview_status.setText(f"预览: 读帧错误 {exc}")
            self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
        finally:
            self._local_cam_preview_busy = False

    def _camera_pose_to_base(self, snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot, result: _eai_perception_pose_estimation.Object6DPoseResult):
        matrix = self._mv6d_T_base_from_head(snap)
        if matrix is None or result is None:
            return None
        xyz = _eai_platform_bootstrap.np.asarray(result.position_xyz, dtype=_eai_platform_bootstrap.np.float64).reshape(3)
        rotation = _eai_platform_bootstrap.np.asarray(result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float64).reshape(3, 3)
        if not _eai_platform_bootstrap.np.isfinite(xyz).all() or not _eai_platform_bootstrap.np.isfinite(rotation).all():
            return None
        xyz_b = matrix[:3, :3] @ xyz + matrix[:3, 3]
        rotation_b = matrix[:3, :3] @ rotation
        corners_b = None
        if result.obb_corners is not None:
            corners = _eai_platform_bootstrap.np.asarray(result.obb_corners, dtype=_eai_platform_bootstrap.np.float32)
            if corners.shape == (8, 3):
                corners_b = _eai_perception_pose_estimation.transform_points_by_matrix(corners, matrix)
        return xyz_b, rotation_b, corners_b, result.obb_extents

    def _on_topics_updated(self, topics: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]]) -> None:
        topics = self._merge_sim_preview_topics(topics)
        # 发现结果未变时不要整表重建 checkbox/panel（会卡死点击）
        sig = tuple(sorted((k, tuple(v)) for k, v in topics.items()))
        if sig == getattr(self, "_topics_updated_sig", None) and self.topic_checks:
            self._topic_types = topics
            return
        self._topics_updated_sig = sig
        self._topic_types = topics
        prev_checked = {
            topic: checkbox.isChecked()
            for topic, checkbox in self.topic_checks.items()
        }

        while self.topic_list_layout.count():
            item = self.topic_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.topic_checks.clear()

        if not topics:
            empty = _eai_platform_bootstrap.QLabel("未发现匹配的 topic")
            empty.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; padding: 8px;")
            self.topic_list_layout.addWidget(empty)
            # 仿真预览面板勿因 ROS 空发现被整表清掉；尊重用户取消勾选
            if self._sim_preview_active() or self.panels:
                keep = {
                    topic
                    for topic in self.panels.keys()
                    if topic not in self._topic_user_unchecked
                }
                self._rebuild_panels(keep)
                self._refresh_skeleton_camera_list()
                return
            self._rebuild_panels(set())
            self._refresh_skeleton_camera_list()
            return

        pending_topic_checks = getattr(self, "_pending_topic_checks", None)
        if isinstance(pending_topic_checks, dict) and pending_topic_checks:
            # 首次 topic 刷新时套用上次关闭前的勾选；之后交给 prev_checked
            for topic, checked in pending_topic_checks.items():
                if topic not in prev_checked:
                    prev_checked[topic] = bool(checked)
            self._pending_topic_checks = None

        default_enabled: set[str] = set()
        for topic, types in topics.items():
            type_str = ", ".join(t.split("/")[-1] for t in types)
            # 默认全部不勾选；用户手动勾选需要的 topic
            is_default = False
            if topic in prev_checked:
                checked = prev_checked[topic]
            elif topic in self._topic_user_unchecked:
                checked = False
            elif (
                topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS
                and self._sim_preview_active()
            ):
                # 新出现的仿真 topic：默认勾选（用户取消过的除外）
                checked = True
            elif topic in (_eai_configuration_policies.LIBERO_PREVIEW_TOPIC, _eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC):
                # File-backed LIBERO / Live HUD preview — keep visible by default.
                checked = True
            elif topic in _eai_configuration_policies.ESSENTIAL_A2D_CAMERA_COLOR_TOPICS:
                # 头/腕彩色默认勾选（用户明确取消过的除外，见 _topic_user_unchecked）
                checked = True
            else:
                checked = is_default
            label = f"{topic}  [{type_str}]"
            checkbox = _eai_platform_bootstrap.QCheckBox(label)
            checkbox.setChecked(checked)
            checkbox.stateChanged.connect(self._on_selection_changed)
            self.topic_checks[topic] = checkbox
            self.topic_list_layout.addWidget(checkbox)
            if checked:
                default_enabled.add(topic)

        self.topic_list_layout.addStretch()
        self._apply_selection(default_enabled)
        self._refresh_skeleton_camera_list()
        self._retry_pending_combo_ui_state()

    def _depth_compatible_with_color(
        self, color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray], depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]
    ) -> bool:
        if color_bgr is None or depth is None:
            return depth is not None
        try:
            ch, cw = _eai_platform_bootstrap.np.asarray(color_bgr).shape[:2]
            dh, dw = _eai_platform_bootstrap.np.asarray(depth).shape[:2]
        except Exception:
            return False
        if min(ch, cw, dh, dw) <= 0:
            return False
        scale = max(
            float(ch) / float(dh),
            float(cw) / float(dw),
            float(dh) / float(ch),
            float(dw) / float(cw),
        )
        return scale <= 1.8

    def _on_frame_tick(self) -> None:
        """合并投递的帧：一次事件处理所有 topic 最新帧，避免队列堆积。"""
        if _eai_platform_bootstrap.time.time() < float(getattr(self, "_ui_frame_paused_until", 0.0) or 0.0):
            # 切 tab 窗口：丢掉积压帧，优先响应点击
            try:
                self.node.take_pending_ui_frames()
            except Exception:
                pass
            return
        try:
            frames, stats = self.node.take_pending_ui_frames()
        except Exception:
            return
        for topic, cv_image in frames.items():
            self._on_frame_updated(topic, cv_image)
            st = stats.get(topic)
            if st is not None:
                self._on_frame_stats(topic, int(st[0]), float(st[1]))

    def _on_frame_updated(self, topic: str, cv_image: object) -> None:
        if _eai_platform_bootstrap.time.time() < float(getattr(self, "_ui_frame_paused_until", 0.0) or 0.0):
            return
        # 仿真评测预览期间：同名 /camera/* 可能来自真机，优先用共享 .npy
        if (
            topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS
            and getattr(self, "_sim_npy_preview_timer", None) is not None
            and self._sim_npy_preview_timer.isActive()
            and self._sim_eval_launcher.is_running()
        ):
            return
        image = _eai_platform_bootstrap.np.asarray(cv_image)
        self._frame_cache[topic] = image
        # 手臂页多视角预览：直接跟 ROS2 帧刷新（不依赖工作区 panel 是否创建）
        try:
            self._maybe_refresh_mv6d_previews_for_topics([topic])
        except Exception:
            pass
        panel = self.panels.get(topic)
        if panel is None:
            return
        if not panel.isVisible():
            return
        if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
            if image.ndim == 3:
                image = _eai_platform_bootstrap.cv2.cvtColor(image, _eai_platform_bootstrap.cv2.COLOR_BGR2GRAY)
            panel.update_depth(image)
        elif isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
            panel.update_frame(image)

    def _on_frame_stats(self, topic: str, count: int, _timestamp: float) -> None:
        self._received_topics[topic] = count

