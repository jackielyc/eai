"""ui.window.shell slow motion finished — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellSlowMotionFinishedMixin:
    def _on_slow_motion_finished(self, ok: bool, text: str) -> None:
        if self._after_arm_motion_settled(ok, text):
            self._update_arm_move_btns_ui()
            return
        if self._text_cmd_pumping:
            self.status_bar.showMessage(text)
            self._update_arm_move_btns_ui()
            return
        if not ok:
            if self._text_cmd_queue:
                self._text_cmd_queue.clear()
                text += "（已取消后续文本指令）"
        else:
            extra = self._pump_text_cmd_queue()
            if extra:
                text = f"{text}；{extra}"
        self.status_bar.showMessage(text)
        self._update_arm_move_btns_ui()

    def _on_left_hand_apply_active(self) -> None:
        at_a = self.node.is_left_hand_at_a()
        slider = (
            self.left_hand_gap_slider_a.value()
            if at_a
            else self.left_hand_gap_slider_b.value()
        )
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
        if self._sim_teleop_active():
            pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
            ok = self._apply_sim_gripper("left", pos)
            preset = "A" if at_a else "B"
            self.status_bar.showMessage(
                f"仿真左手状态{preset} {gap_cm:.1f}cm"
                + ("" if ok else "（写入失败）")
            )
            return
        if not self.node.is_hand_enabled():
            self.status_bar.showMessage("手未使能：请先点「启用手」", 4000)
            return
        sent_pos = self.node.apply_left_hand_angle(slider)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"左手状态{preset} 已应用 {gap_cm:.1f}cm (pos={sent_pos:.3f}) "
            f"-> {_eai_configuration_policies.ROBOT_LEFT_HAND_CMD_TOPIC}"
        )

    def _on_right_hand_apply_active(self) -> None:
        at_a = self.node.is_right_hand_at_a()
        slider = (
            self.right_hand_gap_slider_a.value()
            if at_a
            else self.right_hand_gap_slider_b.value()
        )
        gap_cm = _eai_adapters_robot_geometry.slider_to_hand_gap_cm(slider)
        if self._sim_teleop_active():
            pos = _eai_adapters_robot_geometry.slider_to_hand_position(slider)
            ok = self._apply_sim_gripper("right", pos)
            preset = "A" if at_a else "B"
            self.status_bar.showMessage(
                f"仿真右手状态{preset} {gap_cm:.1f}cm"
                + ("" if ok else "（写入失败）")
            )
            return
        if not self.node.is_hand_enabled():
            self.status_bar.showMessage("手未使能：请先点「启用手」", 4000)
            return
        sent_pos = self.node.apply_right_hand_angle(slider)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"右手状态{preset} 已应用 {gap_cm:.1f}cm (pos={sent_pos:.3f}) "
            f"-> {_eai_configuration_policies.ROBOT_RIGHT_HAND_CMD_TOPIC}"
        )

    def _refresh_skeleton_camera_list(self) -> None:
        current = self.skeleton_cam_combo.currentData()
        self.skeleton_cam_combo.blockSignals(True)
        self.skeleton_cam_combo.clear()

        # 本机摄像头
        local_indices = self._discover_local_webcam_indices()
        for idx in local_indices:
            dev = f"/dev/video{idx}"
            label = f"本地摄像头 {idx}"
            if _eai_platform_bootstrap.os.path.exists(dev):
                label = f"{label} ({dev})"
            self.skeleton_cam_combo.addItem(label, f"{_eai_configuration_chat.SKELETON_LOCAL_CAM_PREFIX}{idx}")

        # VNC / 跨机：网络流（URL 见下方输入框）
        self.skeleton_cam_combo.addItem("网络流 (下方 URL)", _eai_configuration_chat.SKELETON_NETWORK_CAM_SOURCE)

        topics: _eai_platform_bootstrap.List[str] = []
        for topic, types in sorted(self._topic_types.items()):
            if not self._is_image_topic(types):
                continue
            if _eai_perception_image_geometry.is_depth_topic(topic):
                continue
            topics.append(topic)
        if not topics:
            for topic in sorted(self._frame_cache.keys()):
                if not _eai_perception_image_geometry.is_depth_topic(topic):
                    topics.append(topic)
        for topic in topics:
            self.skeleton_cam_combo.addItem(topic, topic)

        prefer_network = False
        # 远端常见：没有真实 /dev/video，默认选网络流
        if not any(_eai_platform_bootstrap.os.path.exists(f"/dev/video{i}") for i in local_indices):
            prefer_network = True

        if current:
            idx = self.skeleton_cam_combo.findData(current)
            if idx >= 0:
                self.skeleton_cam_combo.setCurrentIndex(idx)
            elif prefer_network:
                idx = self.skeleton_cam_combo.findData(_eai_configuration_chat.SKELETON_NETWORK_CAM_SOURCE)
                self.skeleton_cam_combo.setCurrentIndex(max(0, idx))
        elif prefer_network:
            idx = self.skeleton_cam_combo.findData(_eai_configuration_chat.SKELETON_NETWORK_CAM_SOURCE)
            self.skeleton_cam_combo.setCurrentIndex(max(0, idx))
        elif self.skeleton_cam_combo.count() > 0:
            self.skeleton_cam_combo.setCurrentIndex(0)
        self.skeleton_cam_combo.blockSignals(False)
        self._on_skeleton_cam_source_changed()
        if self.skeleton_cam_combo.count() == 0:
            self.skeleton_status_label.setText("手骨架: 无可用相机")

    def _refresh_human_demo_task_match(self, path: str = "") -> None:
        p = path or self._human_demo_path or ""
        label = getattr(self, "human_demo_match_label", None)
        if not p:
            self._human_demo_task_match = None
            if label is not None:
                label.setText("匹配任务: —")
            return
        try:
            from eai_app.domain.demonstrations.matching import match_video_to_libero_task

            m = match_video_to_libero_task(p)
            self._human_demo_task_match = m
            if label is not None:
                label.setText(
                    f"匹配任务: {m.suite}[{m.task_id}] {m.language} ({m.reason})"
                )
        except Exception as exc:
            self._human_demo_task_match = None
            if label is not None:
                label.setText(f"匹配任务: 失败 ({exc})")

    def _apply_human_demo_match_to_libero_ui(self) -> None:
        m = self._human_demo_task_match
        if m is None:
            return
        if hasattr(self, "sim_backend_libero_radio"):
            self.sim_backend_libero_radio.setChecked(True)
        if hasattr(self, "libero_suite_combo"):
            idx = self.libero_suite_combo.findData(m.suite)
            if idx >= 0:
                self.libero_suite_combo.setCurrentIndex(idx)
        if hasattr(self, "_on_libero_suite_changed"):
            self._on_libero_suite_changed()
        if hasattr(self, "libero_task_id_spin"):
            self.libero_task_id_spin.setValue(int(m.task_id))

    def _set_human_demo_video(self, path: str) -> None:
        resolved = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
        self._stop_human_demo_playback()
        self._human_demo_path = resolved
        self._human_demo_traj = None
        self.human_demo_path_edit.setText(resolved)
        self.human_demo_run_btn.setEnabled(False)
        if hasattr(self, "human_demo_align_btn"):
            self.human_demo_align_btn.setEnabled(False)
        self.human_demo_preview_label.setPixmap(_eai_platform_bootstrap.QPixmap())
        self.human_demo_preview_label.setText("已选视频，点「分析」提取手部轨迹")
        self._refresh_human_demo_task_match(resolved)
        # 若有缓存则自动加载（cache_version < 2 则忽略，强制重分析）
        try:
            from eai_app.domain.demonstrations.imitation import (
                TRAJ_CACHE_VERSION,
                cache_path_for_video,
                load_trajectory,
                trajectory_cache_version,
            )
            import json as _json

            cache = cache_path_for_video(resolved, _eai_configuration_policies.HUMAN_DEMO_CACHE_DIR)
            if _eai_platform_bootstrap.os.path.isfile(cache):
                with open(cache, "r", encoding="utf-8") as f:
                    raw = _json.load(f)
                if trajectory_cache_version(raw) < TRAJ_CACHE_VERSION:
                    self.human_demo_status_label.setText(
                        f"演示: 缓存过旧(v{trajectory_cache_version(raw)})，请重新分析"
                    )
                    self.status_bar.showMessage("演示缓存需重分析以包含 world 深度", 5000)
                    return
                traj = load_trajectory(cache)
                if _eai_platform_bootstrap.os.path.abspath(traj.video_path) == resolved or not traj.video_path:
                    traj.video_path = resolved
                    self._human_demo_traj = traj
                    self.human_demo_run_btn.setEnabled(bool(traj.frames))
                    if hasattr(self, "human_demo_align_btn"):
                        self.human_demo_align_btn.setEnabled(bool(traj.frames))
                    self.human_demo_status_label.setText(
                        f"演示: 已加载缓存 · {traj.summary()}"
                    )
                    self._fill_arm_text_from_human_demo(traj)
                    self._show_human_demo_preview_frame(0)
                    self.status_bar.showMessage(f"已加载演示缓存: {cache}", 4000)
                    return
        except Exception as exc:
            self.human_demo_status_label.setText(f"演示: 缓存无效 ({exc})")
        self.human_demo_status_label.setText(
            f"演示: 已选 {_eai_platform_bootstrap.os.path.basename(resolved)}（待分析）"
        )
        self.status_bar.showMessage(f"已选择演示视频: {resolved}", 4000)

    def _fill_arm_text_from_human_demo(self, traj: object) -> bool:
        """把演示轨迹写成文本指挥，填入上方文本指挥输入框。成功返回 True。"""
        edit = getattr(self, "arm_text_cmd_edit", None)
        if edit is None or traj is None:
            return False
        try:
            from eai_app.domain.demonstrations.imitation import trajectory_to_arm_text

            mirror_w = getattr(self, "human_demo_mirror_check", None)
            left_w = getattr(self, "human_demo_ctrl_left_check", None)
            right_w = getattr(self, "human_demo_ctrl_right_check", None)
            arm_w = getattr(self, "human_demo_move_arm_check", None)
            scale_w = getattr(self, "human_demo_scale_spin", None)
            text = trajectory_to_arm_text(
                traj,  # type: ignore[arg-type]
                mirror=bool(mirror_w.isChecked()) if mirror_w is not None else True,
                ctrl_left=bool(left_w.isChecked()) if left_w is not None else True,
                ctrl_right=bool(right_w.isChecked()) if right_w is not None else True,
                move_arm=bool(arm_w.isChecked()) if arm_w is not None else True,
                scale_m=float(scale_w.value()) if scale_w is not None else 0.30,
            )
        except Exception as exc:
            self.status_bar.showMessage(f"轨迹转文本指挥失败: {exc}", 5000)
            return False
        text = (text or "").strip()
        if not text:
            return False
        edit.setText(text)
        edit.setCursorPosition(0)
        return True

    def _show_human_demo_preview_frame(self, sample_idx: int) -> None:
        traj = self._human_demo_traj
        if traj is None or not traj.frames:
            return
        idx = max(0, min(int(sample_idx), len(traj.frames) - 1))
        fr = traj.frames[idx]
        try:
            _eai_platform_bootstrap._ensure_cv2_numpy()
            from eai_app.domain.demonstrations.imitation import draw_demo_frame, read_video_frame

            bgr = read_video_frame(traj.video_path or self._human_demo_path, fr.index)
            if bgr is None:
                return
            title = f"#{fr.index} t={fr.t_sec:.2f}s hands={len(fr.hands)}"
            vis = draw_demo_frame(bgr, fr, title=title)
            vis = self._human_demo_preview_fit_bgr(vis)
            self.human_demo_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(vis))
            self.human_demo_preview_label.setText("")
        except Exception as exc:
            self.human_demo_preview_label.setText(f"预览失败: {exc}")

    def _stop_human_demo_playback(self) -> None:
        self._human_demo_playing = False
        if getattr(self, "_human_demo_timer", None) is not None:
            self._human_demo_timer.stop()
        cap = getattr(self, "_human_demo_video_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._human_demo_video_cap = None
        self._human_demo_play_idx = 0
        self._human_demo_origins = {}
        self._human_demo_world_origins = {}
        self._human_demo_tcp_base = {}
        self._human_demo_last_pose = {}
        self._human_demo_play_mode = "wrist"
        self._human_demo_aligned_waypoints = []
        if getattr(self, "human_demo_stop_btn", None) is not None:
            self.human_demo_stop_btn.setEnabled(False)
        if getattr(self, "human_demo_run_btn", None) is not None:
            self.human_demo_run_btn.setEnabled(self._human_demo_traj is not None)
        if getattr(self, "human_demo_align_btn", None) is not None:
            self.human_demo_align_btn.setEnabled(self._human_demo_traj is not None)
        if getattr(self, "human_demo_analyze_btn", None) is not None:
            self.human_demo_analyze_btn.setEnabled(not self._human_demo_analyzing)

    def _prepare_human_demo_playback_common(self, traj: object) -> bool:
        """共享：读 TCP、开视频、启动 timer。失败返回 False。"""
        self._human_demo_tcp_base = {}
        for side in ("left", "right"):
            if self._sim_teleop_active():
                tcp = self._sim_tcp_pose(side)
            else:
                tcp = self.node._tcp_pose_in_ik_frame(side, timeout_s=0.3)
            if tcp is not None:
                self._human_demo_tcp_base[side] = tcp
        need_arm = self.human_demo_move_arm_check.isChecked() or (
            self._human_demo_play_mode == "aligned"
        )
        if need_arm and not self._human_demo_tcp_base:
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "人类演示",
                "无法读取 TCP（真机需 /mink_fk/*_tcp_pose；仿真需评测写出状态）。\n"
                "可取消勾选「驱动手臂」仅复现手部开合。",
            )
            return False
        self._human_demo_last_pose = {
            side: (tuple(xyz), tuple(quat))  # type: ignore[arg-type]
            for side, (xyz, quat) in self._human_demo_tcp_base.items()
        }
        _eai_platform_bootstrap._ensure_cv2_numpy()
        cap = _eai_platform_bootstrap.cv2.VideoCapture(
            getattr(traj, "video_path", None) or self._human_demo_path
        )
        self._human_demo_video_cap = cap if cap.isOpened() else None
        self._human_demo_play_idx = 0
        self._human_demo_playing = True
        self.human_demo_run_btn.setEnabled(False)
        if hasattr(self, "human_demo_align_btn"):
            self.human_demo_align_btn.setEnabled(False)
        self.human_demo_stop_btn.setEnabled(True)
        self.human_demo_analyze_btn.setEnabled(False)
        if self._human_demo_play_mode == "aligned" and self._human_demo_aligned_waypoints:
            n = len(self._human_demo_aligned_waypoints)
            t0 = float(self._human_demo_aligned_waypoints[0].t_sec)
            t1 = float(self._human_demo_aligned_waypoints[-1].t_sec)
            interval_ms = max(40, int(round(1000.0 * max(0.05, (t1 - t0) / max(1, n - 1)))))
            total = n
            mode_tag = "对齐"
        else:
            stride = max(1, int(getattr(traj, "sample_stride", 1) or 1))
            fps = max(1.0, float(getattr(traj, "fps", 30.0) or 30.0))
            interval_ms = max(40, int(round(1000.0 * stride / fps)))
            total = len(getattr(traj, "frames", []) or [])
            mode_tag = "手腕"
        self._human_demo_timer.setInterval(interval_ms)
        arms = ",".join(sorted(self._human_demo_tcp_base)) or "无"
        self.human_demo_status_label.setText(
            f"演示: {mode_tag}执行中 0/{total} · {interval_ms}ms/步 · 臂[{arms}]"
        )
        self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        self._human_demo_timer.start()
        return True

    def _discover_local_webcam_indices(self, max_index: int = 5) -> _eai_platform_bootstrap.List[int]:
        """列出本机 /dev/video* 索引，不做 VideoCapture 探测（会卡 UI / 甚至 SIGSEGV）。"""
        if self._skeleton_tracking and self._skeleton_local_cap is not None:
            cur = self.skeleton_cam_combo.currentData()
            if self._skeleton_source_is_local(cur):
                return [self._skeleton_local_cam_index(cur)]
            return [0]

        candidates: _eai_platform_bootstrap.List[int] = []
        for i in range(max_index + 1):
            if _eai_platform_bootstrap.os.path.exists(f"/dev/video{i}"):
                candidates.append(i)
        return candidates

    def _close_skeleton_local_cam(self) -> None:
        cap = self._skeleton_local_cap
        self._skeleton_local_cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _open_skeleton_local_cam(self, index: int) -> _eai_platform_bootstrap.Tuple[bool, str]:
        self._close_skeleton_local_cam()
        index = int(index)
        dev = f"/dev/video{index}"
        if not _eai_platform_bootstrap.os.path.exists(dev):
            return False, f"本机无摄像头设备 {dev}"
        cap = None
        try:
            cap = _eai_platform_bootstrap.cv2.VideoCapture(index, _eai_platform_bootstrap.cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = _eai_platform_bootstrap.cv2.VideoCapture(index)
            if not cap.isOpened():
                return False, f"无法打开本地摄像头 {index}（{dev}）"
            try:
                cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                cap.release()
                return False, f"本地摄像头 {index} 已打开但读不到画面"
            self._skeleton_local_cap = cap
            return True, f"本地摄像头 {index}"
        except Exception as exc:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return False, f"打开本地摄像头失败: {exc}"

    def _open_skeleton_network_stream(self, url: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
        url = (url or "").strip()
        if not url:
            return False, "请填写网络流 URL（例如 http://127.0.0.1:8090/cam.mjpg）"
        self._close_skeleton_local_cam()
        cap = None
        try:
            cap = _eai_platform_bootstrap.cv2.VideoCapture(url, _eai_platform_bootstrap.cv2.CAP_FFMPEG)
            if not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = _eai_platform_bootstrap.cv2.VideoCapture(url)
            if not cap.isOpened():
                return (
                    False,
                    f"无法打开网络流:\n{url}\n\n请确认本机 MJPEG 服务与 SSH -R 隧道已建立（点「用法」）。",
                )
            try:
                cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # 网络流首帧可能稍慢
            deadline = _eai_platform_bootstrap.time.time() + 8.0
            frame = None
            ok = False
            while _eai_platform_bootstrap.time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    break
                _eai_platform_bootstrap.time.sleep(0.05)
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                cap.release()
                return False, f"网络流已连接但读不到画面:\n{url}"
            self._skeleton_local_cap = cap
            return True, url
        except Exception as exc:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return False, f"打开网络流失败: {exc}"

    def _read_skeleton_frame(self, source: object) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray], str]:
        """返回 (BGR 帧, 来源标签)。"""
        if self._skeleton_source_is_local(source):
            idx = self._skeleton_local_cam_index(source)
            cap = self._skeleton_local_cap
            if cap is None:
                return None, f"本地摄像头 {idx}（未打开）"
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                return None, f"本地摄像头 {idx}"
            return frame, f"本地摄像头 {idx}"
        if self._skeleton_source_is_network(source):
            url = self.skeleton_stream_url_edit.text().strip() or _eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT
            cap = self._skeleton_local_cap
            if cap is None:
                return None, f"网络流（未打开） {url}"
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                return None, url
            short = url if len(url) <= 48 else url[:45] + "..."
            return frame, short
        topic = str(source or "")
        if not topic:
            return None, "(未选相机)"
        frame = self._frame_cache.get(topic)
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, topic
        return frame, topic

    def _stop_skeleton_tracking(self) -> None:
        self._skeleton_tracking = False
        self._skeleton_timer.stop()
        self.skeleton_teleop_check.setChecked(False)
        self._close_skeleton_local_cam()
        if self._hand_skeleton_detector is not None:
            try:
                self._hand_skeleton_detector.close()
            except Exception:
                pass
            self._hand_skeleton_detector = None
        self.skeleton_track_btn.setText("开始识别")
        self.skeleton_status_label.setText("手骨架: 已停止")

    def _normalize_prefix(self, prefix: str) -> str:
        prefix = prefix.strip() or "/camera"
        if prefix == "/":
            return "/camera"
        return prefix

    def _is_image_topic(self, types: _eai_platform_bootstrap.List[str]) -> bool:
        return any(t in _eai_configuration_perception.IMAGE_TYPES for t in types)

    def _maybe_refresh_mv6d_previews_for_topics(self, topics: _eai_platform_bootstrap.Sequence[str]) -> None:
        """ROS2 / 仿真帧更新时，刷新手臂页「多视角6D」头/腕预览。"""
        if self._arm_panel_selector_key() != "mv6d":
            return
        # 不在「手臂/手」页时别刷预览（切到其它 tab 仍刷会卡点击）
        if not self._control_tab_title_is_current("手臂/手"):
            return
        topic_list = [str(t) for t in topics if t]
        if not topic_list:
            return
        color_keys = (
            "head_color",
            "left_wrist_color",
            "right_wrist_color",
            "hand_left_color",
            "hand_right_color",
        )
        if not any(
            any(k in t.lower() for k in color_keys) for t in topic_list
        ):
            return
        # 节流：实况预览最多 ~2Hz；绝不在这里跑 _apply_mv6d_overlays（会卡死主线程）
        now = _eai_platform_bootstrap.time.time()
        last = float(getattr(self, "_mv6d_preview_refresh_at", 0.0) or 0.0)
        if now - last < 0.5:
            return
        self._mv6d_preview_refresh_at = now
        # A restored result belongs to the previous capture. It must not block
        # live head/wrist frames used to select the next object.
        try:
            self._refresh_mv6d_live_previews_for_pick()
        except Exception:
            pass

    def _merge_sim_preview_topics(
        self, topics: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]]
    ) -> _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]]:
        """ROS 未发现 /camera 时，仍把仿真共享 .npy 对应的伪 topic 放进列表。"""
        out = dict(topics)
        img_types = ["sensor_msgs/msg/Image"]
        expose_all = self._sim_preview_active()
        bridge_dir = self._sim_bridge_dir_path()
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
            if topic in out:
                continue
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}.npy")
            if (
                expose_all
                or topic in self._frame_cache
                or topic in self.panels
                or _eai_platform_bootstrap.os.path.isfile(npy_path)
            ):
                out[topic] = list(img_types)
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_DEPTH_CAM_TOPICS:
            if topic in out:
                continue
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}_depth.npy")
            if (
                expose_all
                or topic in self._frame_cache
                or topic in self.panels
                or _eai_platform_bootstrap.os.path.isfile(npy_path)
            ):
                out[topic] = list(img_types)
        # LIBERO / Live HUD file-backed preview topics (not real ROS pubs).
        libero_running = bool(
            getattr(self, "_mujoco_launcher", None)
            and self._mujoco_launcher.is_running()
            and self._sim_backend_id() == "libero"
        )
        timer = getattr(self, "_rynnvalue_workspace_hud_timer", None)
        preview_on = bool(timer is not None and timer.isActive()) or libero_running
        for topic in (_eai_configuration_policies.LIBERO_PREVIEW_TOPIC, _eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC):
            if topic in out:
                continue
            if (
                preview_on
                or topic in self._frame_cache
                or topic in self.panels
                or _eai_platform_bootstrap.os.path.isfile(_eai_configuration_policies.LIBERO_PREVIEW_JPG)
            ):
                out[topic] = list(img_types)
        return out

    def _on_selection_changed(self) -> None:
        for topic, checkbox in self.topic_checks.items():
            if checkbox.isChecked():
                self._topic_user_unchecked.discard(topic)
            else:
                self._topic_user_unchecked.add(topic)
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        self._apply_selection(enabled)
        self._refresh_skeleton_camera_list()

    def _apply_selection(self, enabled: set[str]) -> None:
        # Pseudo file-backed topics are painted by the poller, not ROS.
        ros_enabled = {
            t
            for t in enabled
            if t not in (_eai_configuration_policies.LIBERO_PREVIEW_TOPIC, _eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC)
        }
        if ros_enabled:
            try:
                self.node.ensure_topic_scan()
            except Exception:
                pass
        # Acquisition dependencies do not require extra visible depth panels.
        for color_topic in list(ros_enabled):
            if not _eai_perception_image_geometry.is_depth_topic(color_topic):
                depth_topic = self._get_paired_depth_topic_name(color_topic)
                if depth_topic:
                    ros_enabled.add(depth_topic)
        self.node.set_enabled_topics(ros_enabled)
        self._rebuild_panels(enabled)

    def _select_all_images(self) -> None:
        for topic, checkbox in self.topic_checks.items():
            types = self._topic_types.get(topic, [])
            want = self._is_image_topic(types)
            if want:
                self._topic_user_unchecked.discard(topic)
            else:
                self._topic_user_unchecked.add(topic)
            checkbox.setChecked(want)

    def _clear_selection(self) -> None:
        for topic, checkbox in self.topic_checks.items():
            self._topic_user_unchecked.add(topic)
            checkbox.setChecked(False)

    def _grid_dimensions(self, topic_count: int, has_depth: bool) -> _eai_platform_bootstrap.Tuple[int, int]:
        """返回 (rows, cols)，使面板在可视区域内并排显示。"""
        del has_depth  # 深度与彩色面板统一按数量排布
        if topic_count <= 0:
            return 1, 1
        if topic_count == 1:
            return 1, 1
        if topic_count <= 3:
            return 1, topic_count
        if topic_count == 4:
            return 2, 2
        if topic_count <= 6:
            cols = 2
            return (topic_count + cols - 1) // cols, cols
        cols = 3
        return (topic_count + cols - 1) // cols, cols

    def _apply_grid_stretches(self, rows: int, cols: int) -> None:
        for r in range(16):
            self.grid_layout.setRowStretch(r, 1 if r < rows else 0)
        for c in range(16):
            self.grid_layout.setColumnStretch(c, 1 if c < cols else 0)

    def _rebuild_panels(self, enabled: set[str]) -> None:
        for topic in list(self.panels.keys()):
            if topic not in enabled:
                panel = self.panels.pop(topic)
                if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
                    try:
                        panel.shutdown()
                    except Exception:
                        pass
                panel.deleteLater()

        for topic in enabled:
            if topic not in self.panels:
                self.panels[topic] = _eai_ui_cameras_panels.create_camera_panel(
                    topic,
                    get_intrinsics=self.node.get_intrinsics,
                    get_paired_depth=self._get_paired_depth_frame,
                    get_paired_color=self._get_paired_color_frame,
                    get_depth_topic=self._get_paired_depth_topic_name,
                    is_paired_depth_enabled=self._is_paired_depth_enabled,
                    get_robot_state=self.node.get_robot_state,
                    get_camera_frame_id=self.node.get_camera_frame_id,
                    resolve_segment_camera_frame=self.node.resolve_segment_camera_frame,
                    get_tf_buffer=self.node.get_tf_buffer,
                    on_segment_pose=self._store_segment_target,
                    on_click_uv=self._on_sam3_click_uv,
                    status_callback=self.status_bar.showMessage,
                )

        while self.grid_layout.count():
            self.grid_layout.takeAt(0)

        order_index = {t: i for i, t in enumerate(_eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS)}
        # LIBERO / Live HUD preview sit first when present.
        order_index[_eai_configuration_policies.LIBERO_PREVIEW_TOPIC] = -2
        order_index[_eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC] = -1
        topics = sorted(
            self.panels.keys(),
            key=lambda t: (order_index.get(t, 1000), t),
        )
        has_depth = any(_eai_perception_image_geometry.is_depth_topic(t) for t in topics)
        rows, cols = self._grid_dimensions(len(topics), has_depth)
        for idx, topic in enumerate(topics):
            row, col = divmod(idx, cols)
            panel = self.panels[topic]
            panel.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
            self.grid_layout.addWidget(panel, row, col)
        self._apply_grid_stretches(rows, cols)

        self.status_bar.showMessage(f"显示 {len(topics)} 路图像")
        if hasattr(self, "lingbot_depth_topic_combo"):
            self._refresh_lingbot_depth_topic_combo()
        if hasattr(self, "lingbot_video_topic_combo"):
            self._refresh_lingbot_video_topic_combo()
        if hasattr(self, "lingbot_world_topic_combo"):
            self._refresh_lingbot_world_topic_combo()
        if hasattr(self, "reward_topic_combo"):
            self._refresh_reward_topic_combo()

    def _cached_topic_names(self) -> _eai_platform_bootstrap.List[str]:
        """已知 topic：缓存帧 + 列表勾选 + 发现类型 + 当前面板。"""
        names: _eai_platform_bootstrap.List[str] = []
        seen: set[str] = set()
        for src in (
            self._frame_cache.keys(),
            self.topic_checks.keys(),
            getattr(self, "_topic_types", {}).keys(),
            self.panels.keys(),
        ):
            for topic in src:
                if topic in seen:
                    continue
                seen.add(topic)
                names.append(topic)
        return names

    def _ensure_topic_enabled_for_capture(self, topic: str) -> bool:
        """勾选并订阅 topic；若 topic 列表尚无该项则返回 False。"""
        if not topic:
            return False
        checkbox = self.topic_checks.get(topic)
        if checkbox is None:
            return False
        if checkbox.isChecked() and topic in self.panels:
            return True
        checkbox.blockSignals(True)
        checkbox.setChecked(True)
        checkbox.blockSignals(False)
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        enabled.add(topic)
        self._apply_selection(enabled)
        return topic in self.panels or checkbox.isChecked()

    def _get_paired_depth_topic_name(self, color_topic: str) -> _eai_platform_bootstrap.Optional[str]:
        return _eai_perception_image_geometry.find_paired_depth_topic(color_topic, self._cached_topic_names())

    def _is_topic_enabled(self, topic: str) -> bool:
        checkbox = self.topic_checks.get(topic)
        return checkbox is not None and checkbox.isChecked()

    def _is_paired_depth_enabled(self, color_topic: str) -> bool:
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if depth_topic is None:
            return False
        return self._is_topic_enabled(depth_topic)

    def _get_paired_depth_frame(self, color_topic: str) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if depth_topic is None:
            return None
        color = self._frame_cache.get(color_topic)
        if color is None:
            panel_c = self.panels.get(color_topic)
            if isinstance(panel_c, _eai_ui_cameras_panels.CameraPanel) and panel_c._latest_image is not None:
                color = panel_c._latest_image
        depth = self._frame_cache.get(depth_topic)
        if depth is not None and self._depth_compatible_with_color(color, depth):
            return depth
        if depth is not None and not self._depth_compatible_with_color(color, depth):
            # 缓存里是残留深度（尺寸对不上当前彩色）→ 丢掉
            try:
                del self._frame_cache[depth_topic]
            except Exception:
                pass
            depth = None
        panel = self.panels.get(depth_topic)
        if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D) and panel._latest_depth is not None:
            cand = _eai_platform_bootstrap.np.asarray(panel._latest_depth)
            if self._depth_compatible_with_color(color, cand):
                return cand
        # 仿真 headless：无 ROS depth topic 时直读共享 .npy
        bridge_depth = self._load_sim_bridge_depth_u16(color_topic, color_bgr=color)
        if bridge_depth is not None:
            self._frame_cache[depth_topic] = bridge_depth
            return bridge_depth
        return None

    def _get_paired_color_frame(self, depth_topic: str) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        color_topic = _eai_perception_image_geometry.find_paired_color_topic(depth_topic, self._cached_topic_names())
        if color_topic is None:
            return None
        frame = self._frame_cache.get(color_topic)
        if frame is None or frame.ndim < 2:
            return None
        return frame

    def _update_waiting_hint(self) -> None:
        if not self.panels:
            return
        receiving = sum(1 for t in self.panels if self._received_topics.get(t, 0) > 0)
        waiting = len(self.panels) - receiving
        if waiting > 0 and receiving == 0:
            self.status_bar.showMessage(
                f"已订阅 {len(self.panels)} 路，但未收到图像 — 请确认 camera 节点正在发布数据"
            )
        elif waiting > 0:
            self.status_bar.showMessage(
                f"显示 {receiving}/{len(self.panels)} 路图像，{waiting} 路仍等待中"
            )

