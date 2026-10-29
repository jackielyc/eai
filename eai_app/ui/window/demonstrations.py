"""ui.window.demonstrations — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')

class DemonstrationsMixin:
    def _ctx_default_media_dir(self) -> str:
        if self._ctx_video_path and _eai_platform_bootstrap.os.path.isfile(self._ctx_video_path):
            return _eai_platform_bootstrap.os.path.dirname(self._ctx_video_path)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_policies.CTX_DEFAULT_VIDEO_PATH):
            return _eai_platform_bootstrap.os.path.dirname(_eai_configuration_policies.CTX_DEFAULT_VIDEO_PATH)
        repo_datasets = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT), "datasets"
        )
        if _eai_platform_bootstrap.os.path.isdir(repo_datasets):
            return repo_datasets
        d = _eai_adapters_robot_launch_utils.default_rrd_dataset_dir()
        return d if _eai_platform_bootstrap.os.path.isdir(d) else _eai_platform_bootstrap.os.path.expanduser("~")

    def _ctx_find_sibling(self, path: str, exts: tuple[str, ...]) -> str:
        stem, _ = _eai_platform_bootstrap.os.path.splitext(path)
        for ext in exts:
            candidate = stem + ext
            if _eai_platform_bootstrap.os.path.isfile(candidate):
                return _eai_platform_bootstrap.os.path.abspath(candidate)
        parent = _eai_platform_bootstrap.os.path.dirname(path)
        base = _eai_platform_bootstrap.os.path.basename(stem)
        try:
            for name in _eai_platform_bootstrap.os.listdir(parent):
                lower = name.lower()
                if not any(lower.endswith(e.lower()) for e in exts):
                    continue
                if _eai_platform_bootstrap.os.path.splitext(name)[0] == base or base in _eai_platform_bootstrap.os.path.splitext(name)[0]:
                    full = _eai_platform_bootstrap.os.path.join(parent, name)
                    if _eai_platform_bootstrap.os.path.isfile(full):
                        return _eai_platform_bootstrap.os.path.abspath(full)
        except OSError:
            pass
        return ""

    def _on_ctx_select_video_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return
        initial = self._ctx_default_media_dir()
        if self._ctx_video_path and _eai_platform_bootstrap.os.path.isfile(self._ctx_video_path):
            initial = _eai_platform_bootstrap.os.path.dirname(self._ctx_video_path)
        video_path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择演示视频",
            initial,
            "Video Files (*.mp4 *.avi *.mkv *.webm);;All Files (*)",
        )
        if not video_path:
            return
        resolved = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(video_path))
        self._ctx_video_path = resolved
        self.ctx_video_path_edit.setText(resolved)
        self.ctx_video_path_edit.setToolTip(resolved)
        if not self.ctx_video_player.load(resolved):
            _eai_platform_bootstrap.QMessageBox.warning(self, "上下文学习", f"无法打开视频：\n{resolved}")
        if not (self._ctx_rrd_path and _eai_platform_bootstrap.os.path.isfile(self._ctx_rrd_path)):
            sibling = self._ctx_find_sibling(resolved, (".rrd",))
            if sibling:
                self._ctx_rrd_path = sibling
                self.ctx_rrd_path_edit.setText(sibling)
                self.ctx_rrd_path_edit.setToolTip(sibling)
                self.status_bar.showMessage(f"已自动配对 RRD: {_eai_platform_bootstrap.os.path.basename(sibling)}", 5000)
        self._update_ctx_ui()

    def _on_ctx_select_rrd_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return
        initial = self._ctx_default_media_dir()
        if self._ctx_rrd_path and _eai_platform_bootstrap.os.path.isfile(self._ctx_rrd_path):
            initial = _eai_platform_bootstrap.os.path.dirname(self._ctx_rrd_path)
        rrd_path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择配对 RRD 轨迹",
            initial,
            "RRD Files (*.rrd);;All Files (*)",
        )
        if not rrd_path:
            return
        resolved = _eai_ui_datasets_notes.resolve_rrd_path(rrd_path)
        self._ctx_rrd_path = resolved
        self.ctx_rrd_path_edit.setText(resolved)
        self.ctx_rrd_path_edit.setToolTip(resolved)
        if not (self._ctx_video_path and _eai_platform_bootstrap.os.path.isfile(self._ctx_video_path)):
            sibling = self._ctx_find_sibling(
                resolved, (".mp4", ".avi", ".mkv", ".webm")
            )
            if sibling:
                self._ctx_video_path = sibling
                self.ctx_video_path_edit.setText(sibling)
                self.ctx_video_path_edit.setToolTip(sibling)
                self.ctx_video_player.load(sibling)
                self.status_bar.showMessage(
                    f"已自动配对视频: {_eai_platform_bootstrap.os.path.basename(sibling)}", 5000
                )
        self._update_ctx_ui()

    def _on_ctx_rerun_clicked(self) -> None:
        path = (self._ctx_rrd_path or "").strip()
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "Rerun 播放",
                "请先选择有效的 .rrd 文件。",
            )
            return
        if not self._open_rrd_in_native_rerun(path):
            return
        argv = _eai_ui_datasets_notes.resolve_rerun_argv() or ["rerun"]
        cmd_preview = " ".join(_eai_platform_bootstrap.shlex.quote(x) for x in list(argv) + [path])
        self.ctx_rerun_title.setText(f"Rerun · {_eai_platform_bootstrap.os.path.basename(path)}")
        self.ctx_rerun_placeholder.setText(
            "已用 rerun-sdk 原生 Viewer 打开。\n\n"
            f"文件:\n{path}\n\n"
            f"命令:\n{cmd_preview}"
        )

    def _on_ctx_sync_start_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return
        video = (self._ctx_video_path or "").strip()
        rrd = (self._ctx_rrd_path or "").strip()
        if not video or not _eai_platform_bootstrap.os.path.isfile(video):
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "上下文学习",
                "请先选择演示视频。",
            )
            return
        if not rrd or not _eai_platform_bootstrap.os.path.isfile(rrd):
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "上下文学习",
                "请先选择配对的 .rrd 轨迹文件。\n"
                "机器人跟做依赖关节回放；纯 RGB 视频无法驱臂。",
            )
            return
        if not self.ctx_video_player.is_open():
            if not self.ctx_video_player.load(video):
                _eai_platform_bootstrap.QMessageBox.warning(self, "上下文学习", f"无法打开视频：\n{video}")
                return
        if not self._start_rrd_replay(rrd, self.ctx_loop_spin.value()):
            return
        self.ctx_video_player.restart()
        self.status_bar.showMessage(
            f"上下文学习：视频 + RRD 同步开始（{_eai_platform_bootstrap.os.path.basename(rrd)}）",
            6000,
        )
        self._update_ctx_ui()

    def _on_ctx_stop_clicked(self) -> None:
        if getattr(self, "ctx_video_player", None) is not None:
            self.ctx_video_player.pause()
        if self._replay_launcher.is_running():
            self._replay_launcher.stop()
        self.ctx_status_label.setText("状态: 已停止")
        self._update_ctx_ui()

    def _on_human_demo_select_clicked(self) -> None:
        start = ""
        if self._human_demo_path and _eai_platform_bootstrap.os.path.isfile(self._human_demo_path):
            start = _eai_platform_bootstrap.os.path.dirname(self._human_demo_path)
        elif _eai_platform_bootstrap.os.path.isdir(_eai_configuration_models.EAI_DIR):
            start = _eai_configuration_models.EAI_DIR
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self, "选择人类演示视频", start, _eai_configuration_policies.HUMAN_DEMO_VIDEO_FILTER
        )
        if not path:
            return
        self._set_human_demo_video(path)

    def _on_human_demo_analyze_clicked(self) -> None:
        if self._human_demo_analyzing:
            self.status_bar.showMessage("正在分析中…")
            return
        path = self._human_demo_path or self.human_demo_path_edit.text().strip()
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            _eai_platform_bootstrap.QMessageBox.information(self, "人类演示", "请先选择演示视频。")
            return
        try:
            from eai_app.domain.demonstrations.imitation import mediapipe_available
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "人类演示", f"无法加载 human_demo_imitate:\n{exc}")
            return
        if not mediapipe_available():
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "人类演示",
                "未安装 mediapipe。请在当前环境执行:\n"
                "  python -m pip install 'mediapipe==0.10.14'",
            )
            return
        self._stop_human_demo_playback()
        self._human_demo_analyzing = True
        self.human_demo_analyze_btn.setEnabled(False)
        self.human_demo_run_btn.setEnabled(False)
        self.human_demo_status_label.setText("演示: 分析中…")
        self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")

        def _worker() -> None:
            try:
                from eai_app.domain.demonstrations.imitation import (
                    analyze_demo_video,
                    cache_path_for_video,
                    save_trajectory,
                )

                def _prog(pct: float, msg: str) -> None:
                    self._human_demo_bridge.progress.emit(float(pct), str(msg))

                traj = analyze_demo_video(
                    path,
                    target_fps=12.0,
                    flip_horizontal=False,
                    progress=_prog,
                )
                _eai_platform_bootstrap.os.makedirs(_eai_configuration_policies.HUMAN_DEMO_CACHE_DIR, exist_ok=True)
                cache = cache_path_for_video(path, _eai_configuration_policies.HUMAN_DEMO_CACHE_DIR)
                save_trajectory(traj, cache)
                self._human_demo_bridge.finished.emit(True, traj, cache)
            except Exception as exc:
                self._human_demo_bridge.finished.emit(False, None, str(exc))

        _eai_platform_bootstrap.threading.Thread(target=_worker, daemon=True).start()

    def _on_human_demo_analyze_progress(self, pct: float, msg: str) -> None:
        self.human_demo_status_label.setText(f"演示: {msg} ({pct * 100:.0f}%)")

    def _on_human_demo_analyze_finished(
        self, ok: bool, traj: object, detail: str
    ) -> None:
        self._human_demo_analyzing = False
        self.human_demo_analyze_btn.setEnabled(True)
        if not ok or traj is None:
            self.human_demo_status_label.setText(f"演示: 分析失败 — {detail}")
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            _eai_platform_bootstrap.QMessageBox.warning(self, "人类演示", f"分析失败:\n{detail}")
            return
        self._human_demo_traj = traj
        ready = bool(getattr(traj, "frames", None))
        self.human_demo_run_btn.setEnabled(ready)
        if hasattr(self, "human_demo_align_btn"):
            self.human_demo_align_btn.setEnabled(ready)
        summary = traj.summary() if hasattr(traj, "summary") else ""
        self.human_demo_status_label.setText(f"演示: 就绪 · {summary}")
        self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        filled = self._fill_arm_text_from_human_demo(traj)
        self._show_human_demo_preview_frame(0)
        if filled:
            self.status_bar.showMessage(
                f"演示分析完成，已填入文本指挥；缓存: {detail}", 5000
            )
        else:
            self.status_bar.showMessage(f"演示分析完成，缓存: {detail}", 5000)

    def _human_demo_preview_fit_bgr(self, bgr: "np.ndarray") -> "np.ndarray":
        """将预览帧缩放到预览区宽高内（保持比例，尽量铺满）。"""
        label = self.human_demo_preview_label
        max_w = max(520, int(label.width()) - 12)
        max_h = max(360, int(label.height()) - 12)
        h, w = bgr.shape[:2]
        if w < 1 or h < 1:
            return bgr
        scale = min(max_w / float(w), max_h / float(h))
        if abs(scale - 1.0) < 1e-3:
            return bgr
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        interp = _eai_platform_bootstrap.cv2.INTER_AREA if scale < 1.0 else _eai_platform_bootstrap.cv2.INTER_LINEAR
        return _eai_platform_bootstrap.cv2.resize(bgr, (nw, nh), interpolation=interp)

    def _human_demo_resolve_robot_side(self, robot_side: str) -> _eai_platform_bootstrap.Optional[str]:
        """人物映射侧 → 实际可控臂。LIBERO 等单臂仿真只有 left_ee 时，右臂映射回退到左臂。"""
        if robot_side in self._human_demo_tcp_base:
            return robot_side
        if len(self._human_demo_tcp_base) == 1:
            return next(iter(self._human_demo_tcp_base))
        return None

    def _on_human_demo_stop_clicked(self) -> None:
        was = self._human_demo_playing
        self._stop_human_demo_playback()
        if was:
            self.human_demo_status_label.setText("演示: 已停止")
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
            self.status_bar.showMessage("已停止人类演示复现")

    def _on_human_demo_run_clicked(self) -> None:
        traj = self._human_demo_traj
        if traj is None or not traj.frames:
            _eai_platform_bootstrap.QMessageBox.information(self, "人类演示", "请先分析视频。")
            return
        if self._human_demo_playing:
            return
        if self._skeleton_tracking and self.skeleton_teleop_check.isChecked():
            self.skeleton_teleop_check.setChecked(False)
        # 准备使能
        if not self._sim_teleop_active():
            self.node.request_model_control_mode()
            if not self.node.is_hand_enabled():
                self.node.request_hand_enable(True)
            if self.human_demo_move_arm_check.isChecked() and not self.node.is_arm_enabled():
                self.node.request_arm_enable(True)
        try:
            from eai_app.domain.demonstrations.imitation import (
                first_wrist_origins,
                first_wrist_world_origins,
            )
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "人类演示", f"加载失败: {exc}")
            return
        self._human_demo_origins = first_wrist_origins(traj)
        self._human_demo_world_origins = first_wrist_world_origins(traj)
        if not self._human_demo_origins:
            _eai_platform_bootstrap.QMessageBox.warning(
                self, "人类演示", "轨迹中未检测到手，请换一段手部更清晰的视频。"
            )
            return
        self._human_demo_play_mode = "wrist"
        self._human_demo_aligned_waypoints = []
        if not self._prepare_human_demo_playback_common(traj):
            return
        arms = ",".join(sorted(self._human_demo_tcp_base)) or "无"
        world_n = len(self._human_demo_world_origins)
        self.status_bar.showMessage(
            f"开始按人类演示复现…（可控臂: {arms}"
            + ("；单臂已自动折叠左右映射" if len(self._human_demo_tcp_base) == 1 else "")
            + (f"；world原点×{world_n}" if world_n else "；无world回退图像")
            + "）"
        )

    def _on_human_demo_scene_align_clicked(self) -> None:
        traj = self._human_demo_traj
        if traj is None or not traj.frames:
            _eai_platform_bootstrap.QMessageBox.information(self, "人类演示", "请先分析视频。")
            return
        if self._human_demo_playing:
            return
        if not self._sim_teleop_active():
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "场景对齐复现",
                "请先在仿真评测打开匹配任务的 LIBERO「只开界面」，\n"
                "确保 sim_robot_state 含 objects 后再点此按钮。",
            )
            return
        state = self._read_sim_robot_state() or {}
        if not isinstance(state.get("objects"), dict) or not state.get("objects"):
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "场景对齐复现",
                "当前仿真状态无 objects。\n"
                "请重启带任务序号的 LIBERO「只开界面」（需更新后的 run_libero_eval）。",
            )
            return
        try:
            from eai_app.domain.demonstrations.alignment import build_scene_aligned_waypoints

            wps = build_scene_aligned_waypoints(traj, state)
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "场景对齐复现", f"生成航点失败:\n{exc}")
            return
        if not wps:
            _eai_platform_bootstrap.QMessageBox.warning(self, "场景对齐复现", "未生成有效航点。")
            return
        if self._skeleton_tracking and self.skeleton_teleop_check.isChecked():
            self.skeleton_teleop_check.setChecked(False)
        self._human_demo_play_mode = "aligned"
        self._human_demo_aligned_waypoints = list(wps)
        if not self._prepare_human_demo_playback_common(traj):
            return
        note = wps[0].note if wps else ""
        self.status_bar.showMessage(
            f"场景对齐复现开始 · {len(wps)} 航点 · {note}", 6000
        )

    def _on_human_demo_policy_clicked(self) -> None:
        path = self._human_demo_path or self.human_demo_path_edit.text().strip()
        if not path:
            _eai_platform_bootstrap.QMessageBox.information(self, "匹配策略", "请先选择演示视频。")
            return
        self._refresh_human_demo_task_match(path)
        m = self._human_demo_task_match
        if m is None:
            _eai_platform_bootstrap.QMessageBox.warning(self, "匹配策略", "无法匹配 LIBERO 任务。")
            return
        self._apply_human_demo_match_to_libero_ui()
        reply = _eai_platform_bootstrap.QMessageBox.question(
            self,
            "用匹配任务跑策略",
            f"将启动 pi05_libero 单任务评测：\n\n"
            f"{m.suite}[{m.task_id}]\n{m.language}\n\n"
            f"（会部署/连接 Pi，并停止当前 viewer 若在运行）\n继续？",
            _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
            _eai_platform_bootstrap.QMessageBox.Yes,
        )
        if reply != _eai_platform_bootstrap.QMessageBox.Yes:
            return
        if self._mujoco_launcher.is_running():
            try:
                self._mujoco_launcher.stop()
            except Exception:
                pass
        if hasattr(self, "libero_max_tasks_spin"):
            self.libero_max_tasks_spin.setValue(1)
        self.status_bar.showMessage(f"启动匹配策略: {m.summary()}", 5000)
        self._on_libero_run_clicked(viewer_only=False)

    def _on_human_demo_play_tick(self) -> None:
        if not self._human_demo_playing:
            self._human_demo_timer.stop()
            return
        if self._human_demo_play_mode == "aligned":
            self._on_human_demo_aligned_tick()
            return
        traj = self._human_demo_traj
        if traj is None or not traj.frames:
            self._stop_human_demo_playback()
            return
        idx = self._human_demo_play_idx
        if idx >= len(traj.frames):
            self._stop_human_demo_playback()
            self.human_demo_status_label.setText(
                f"演示: 完成 · {traj.summary()}"
            )
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self.status_bar.showMessage("人类演示复现完成")
            self._show_human_demo_preview_frame(max(0, len(traj.frames) - 1))
            return
        fr = traj.frames[idx]
        self._human_demo_play_idx = idx + 1
        try:
            from eai_app.domain.demonstrations.imitation import (
                draw_demo_frame,
                map_person_hand_to_robot_side,
                wrist_delta_to_base_world,
            )

            mirror = self.human_demo_mirror_check.isChecked()
            ctrl_left = self.human_demo_ctrl_left_check.isChecked()
            ctrl_right = self.human_demo_ctrl_right_check.isChecked()
            move_arm = self.human_demo_move_arm_check.isChecked()
            scale = float(self.human_demo_scale_spin.value())
            sim = self._sim_teleop_active()

            for hand in fr.hands:
                person_key = (
                    "Left" if hand.handedness.lower().startswith("l") else "Right"
                )
                mapped = map_person_hand_to_robot_side(hand.handedness, mirror)
                # 仿真单臂：右臂映射折叠到唯一可控臂；真机保持左右独立
                robot_side = (
                    self._human_demo_resolve_robot_side(mapped) if sim else mapped
                )
                if robot_side is None:
                    continue
                if robot_side == "left" and not ctrl_left:
                    continue
                if robot_side == "right" and not ctrl_right:
                    continue
                if sim:
                    scalar = sum(float(v) for v in hand.joints) / max(
                        1, len(hand.joints)
                    )
                    self._apply_sim_gripper(robot_side, scalar)
                else:
                    self.node.apply_hand_joint_positions(robot_side, hand.joints)
                if move_arm and robot_side in self._human_demo_tcp_base:
                    origin = self._human_demo_origins.get(person_key)
                    if origin is None:
                        continue
                    ww0 = self._human_demo_world_origins.get(person_key)
                    ww_now = hand.wrist_world
                    dx, dy, dz = wrist_delta_to_base_world(
                        hand.wrist_xyz,
                        origin,
                        wrist_world_now=ww_now,
                        wrist_world0=ww0,
                        scale_m=scale,
                        mirror=mirror,
                    )
                    base_xyz, base_quat = self._human_demo_tcp_base[robot_side]
                    new_xyz = (
                        float(base_xyz[0]) + dx,
                        float(base_xyz[1]) + dy,
                        float(base_xyz[2]) + dz,
                    )
                    self._human_demo_last_pose[robot_side] = (new_xyz, base_quat)

            if move_arm:
                if sim:
                    # 持续发送最近目标；无手帧保持，避免把目标打回起点
                    kwargs: dict = {"enabled": True}
                    for side, pose in self._human_demo_last_pose.items():
                        key = "left_ee_pose" if side == "left" else "right_ee_pose"
                        kwargs[key] = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(pose[0], pose[1])
                    if len(kwargs) > 1:
                        self._write_sim_gui_cmd(**kwargs)
                else:
                    left_pose = self._human_demo_last_pose.get("left")
                    right_pose = self._human_demo_last_pose.get("right")
                    if left_pose is None:
                        left_pose = self._human_demo_tcp_base.get("left")
                        if left_pose is None:
                            left_pose = self.node._tcp_pose_in_ik_frame(
                                "left", timeout_s=0.05
                            )
                    if right_pose is None:
                        right_pose = self._human_demo_tcp_base.get("right")
                        if right_pose is None:
                            right_pose = self.node._tcp_pose_in_ik_frame(
                                "right", timeout_s=0.05
                            )
                    if left_pose is not None and right_pose is not None:
                        self.node._publish_dual_arm_targets(
                            left_pose[0],
                            left_pose[1],
                            right_pose[0],
                            right_pose[1],
                        )

            # 预览
            bgr = None
            cap = self._human_demo_video_cap
            if cap is not None and cap.isOpened():
                cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_POS_FRAMES, fr.index)
                ok, bgr = cap.read()
                if not ok:
                    bgr = None
            if bgr is not None:
                title = (
                    f"复现 {idx + 1}/{len(traj.frames)}  "
                    f"t={fr.t_sec:.2f}s  hands={len(fr.hands)}"
                )
                vis = draw_demo_frame(bgr, fr, title=title)
                vis = self._human_demo_preview_fit_bgr(vis)
                self.human_demo_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(vis))
                self.human_demo_preview_label.setText("")
            self.human_demo_status_label.setText(
                f"演示: 执行中 {idx + 1}/{len(traj.frames)} · t={fr.t_sec:.2f}s"
            )
        except Exception as exc:
            self.human_demo_status_label.setText(f"演示: 执行错误 — {exc}")
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            self._stop_human_demo_playback()

    def _on_human_demo_aligned_tick(self) -> None:
        wps = self._human_demo_aligned_waypoints or []
        idx = self._human_demo_play_idx
        if idx >= len(wps):
            self._stop_human_demo_playback()
            self.human_demo_status_label.setText(
                f"演示: 对齐完成 · {len(wps)} 航点"
            )
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self.status_bar.showMessage("场景对齐复现完成")
            return
        wp = wps[idx]
        self._human_demo_play_idx = idx + 1
        try:
            pose7 = list(wp.ee_pose7_wxyz)
            grip = float(wp.gripper01)
            if self._sim_teleop_active():
                self._write_sim_gui_cmd(
                    enabled=True,
                    left_ee_pose=pose7,
                    left_gripper=grip,
                )
            else:
                xyz, quat = _eai_perception_pose_estimation.xyz_xyzw_from_pose7_wxyz(pose7)
                right = self._human_demo_tcp_base.get("right") or self.node._tcp_pose_in_ik_frame(
                    "right", timeout_s=0.05
                )
                if right is not None:
                    self.node._publish_dual_arm_targets(
                        xyz, quat, right[0], right[1]
                    )
            self.human_demo_status_label.setText(
                f"演示: 对齐 {idx + 1}/{len(wps)} · t={wp.t_sec:.2f}s · "
                f"g={grip:.2f} · {wp.note}"
            )
            # 预览：按比例映射到视频帧
            traj = self._human_demo_traj
            if traj is not None and traj.frames:
                fi = int(round((idx / max(1, len(wps) - 1)) * (len(traj.frames) - 1)))
                self._show_human_demo_preview_frame(fi)
        except Exception as exc:
            self.human_demo_status_label.setText(f"演示: 对齐错误 — {exc}")
            self.human_demo_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            self._stop_human_demo_playback()

    def _on_skeleton_cam_source_changed(self, *_args) -> None:
        is_net = self._skeleton_source_is_network(self.skeleton_cam_combo.currentData())
        if hasattr(self, "skeleton_stream_url_edit"):
            self.skeleton_stream_url_edit.setEnabled(True)  # 始终可编辑，便于预先填好
            self.skeleton_stream_url_edit.setStyleSheet(
                f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY};" if is_net else f"color: {_eai_configuration_policies.UI_TEXT_MUTED};"
            )

    def _on_skeleton_network_help(self) -> None:
        _eai_platform_bootstrap.QMessageBox.information(self, "本机摄像头 → 远端（VNC）", _eai_configuration_chat.SKELETON_NETWORK_HELP)

    @staticmethod
    def _skeleton_source_is_local(source: object) -> bool:
        return isinstance(source, str) and source.startswith(_eai_configuration_chat.SKELETON_LOCAL_CAM_PREFIX)

    @staticmethod
    def _skeleton_source_is_network(source: object) -> bool:
        return source == _eai_configuration_chat.SKELETON_NETWORK_CAM_SOURCE

    @staticmethod
    def _skeleton_source_uses_capture(source: object) -> bool:
        return MainWindow._skeleton_source_is_local(source) or MainWindow._skeleton_source_is_network(
            source
        )

    @staticmethod
    def _skeleton_local_cam_index(source: object) -> int:
        if not isinstance(source, str) or not source.startswith(_eai_configuration_chat.SKELETON_LOCAL_CAM_PREFIX):
            return 0
        try:
            return max(0, int(source.split(":", 1)[1]))
        except ValueError:
            return 0

    def _on_skeleton_track_toggled(self) -> None:
        if self._skeleton_tracking:
            self._stop_skeleton_tracking()
            return
        try:
            from eai_app.domain.demonstrations.teleop import HandSkeletonDetector, mediapipe_available
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "手骨架遥控",
                f"无法加载 hand_skeleton_teleop:\n{exc}",
            )
            return
        if not mediapipe_available():
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "手骨架遥控",
                "未安装 mediapipe。请在当前 ROS conda 环境执行:\n"
                "  python -m pip install 'mediapipe==0.10.14'\n"
                "（本机 run_local.sh 使用 Python 3.11；勿强行降 numpy，"
                "conda 已带的 numpy 可与 cv_bridge 共存。）",
            )
            return
        # 保留当前选择（含网络流），刷新时不要冲掉
        prev = self.skeleton_cam_combo.currentData()
        self._refresh_skeleton_camera_list()
        if prev is not None:
            idx = self.skeleton_cam_combo.findData(prev)
            if idx >= 0:
                self.skeleton_cam_combo.setCurrentIndex(idx)
        if self.skeleton_cam_combo.count() == 0:
            self.status_bar.showMessage("无可用相机：请用网络流、本机摄像头或 ROS topic")
            return
        source = self.skeleton_cam_combo.currentData()
        if self._skeleton_source_is_local(source):
            ok, msg = self._open_skeleton_local_cam(self._skeleton_local_cam_index(source))
            if not ok:
                _eai_platform_bootstrap.QMessageBox.warning(self, "手骨架遥控", msg)
                return
        elif self._skeleton_source_is_network(source):
            url = self.skeleton_stream_url_edit.text().strip() or _eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT
            self.skeleton_stream_url_edit.setText(url)
            self.status_bar.showMessage(f"正在连接网络流: {url}")
            _eai_platform_bootstrap.QApplication.processEvents()
            ok, msg = self._open_skeleton_network_stream(url)
            if not ok:
                _eai_platform_bootstrap.QMessageBox.warning(self, "手骨架遥控", msg)
                return
        try:
            self._hand_skeleton_detector = HandSkeletonDetector(max_num_hands=2)
        except Exception as exc:
            self._close_skeleton_local_cam()
            _eai_platform_bootstrap.QMessageBox.warning(self, "手骨架遥控", f"初始化 MediaPipe Hands 失败:\n{exc}")
            self._hand_skeleton_detector = None
            return
        self._skeleton_tracking = True
        self.skeleton_track_btn.setText("停止识别")
        if self._skeleton_source_is_local(source):
            src_label = f"本地摄像头 {self._skeleton_local_cam_index(source)}"
        elif self._skeleton_source_is_network(source):
            src_label = self.skeleton_stream_url_edit.text().strip()
        else:
            src_label = str(source)
        self.skeleton_status_label.setText(f"手骨架: 识别中… @ {src_label}")
        self._skeleton_timer.start()

    def _on_skeleton_teleop_toggled(self, checked: bool) -> None:
        if checked and not self._skeleton_tracking:
            self.skeleton_teleop_check.blockSignals(True)
            self.skeleton_teleop_check.setChecked(False)
            self.skeleton_teleop_check.blockSignals(False)
            self.status_bar.showMessage("请先点击「开始识别」")
            return
        if checked and not self.node.is_hand_enabled():
            msg = self.node.request_hand_enable(True)
            self.status_bar.showMessage(f"{msg}（遥控需要手部使能）")
        if checked:
            self.skeleton_status_label.setText("手骨架: 识别中 + 遥控中")
            self.skeleton_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        elif self._skeleton_tracking:
            self.skeleton_status_label.setText("手骨架: 识别中（仅预览）")
            self.skeleton_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")

    def _on_skeleton_tick(self) -> None:
        if not self._skeleton_tracking or self._skeleton_busy:
            return
        if self._hand_skeleton_detector is None:
            return
        source = self.skeleton_cam_combo.currentData()
        if not source:
            return
        frame, src_label = self._read_skeleton_frame(source)
        if frame is None:
            self.skeleton_status_label.setText(f"手骨架: 等待图像 {src_label}")
            return
        self._skeleton_busy = True
        try:
            from eai_app.domain.demonstrations.teleop import (
                draw_hand_skeleton,
                map_person_hand_to_robot_side,
            )

            flip = self.skeleton_flip_check.isChecked()
            alpha = self.skeleton_smooth_slider.value() / 100.0
            hands = self._hand_skeleton_detector.detect(
                frame, flip_horizontal=flip, smooth_alpha=alpha
            )
            vis = draw_hand_skeleton(frame, hands, flip_horizontal=flip)
            # 预览缩放到合适宽度
            max_w = max(320, self.skeleton_preview_label.width() - 8)
            h, w = vis.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                vis = _eai_platform_bootstrap.cv2.resize(
                    vis,
                    (max_w, max(1, int(h * scale))),
                    interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
                )
            self.skeleton_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(vis))

            mirror = self.skeleton_mirror_map_check.isChecked()
            lines: _eai_platform_bootstrap.List[str] = []
            teleop = self.skeleton_teleop_check.isChecked()
            ctrl_left = self.skeleton_ctrl_left_check.isChecked()
            ctrl_right = self.skeleton_ctrl_right_check.isChecked()
            for hand in hands:
                robot_side = map_person_hand_to_robot_side(hand.handedness, mirror)
                j = hand.joints
                lines.append(
                    f"人{hand.handedness}→机{robot_side}: "
                    f"rot={j[0]:.2f} bend={j[1]:.2f} "
                    f"I={j[2]:.2f} M={j[3]:.2f} R={j[4]:.2f} P={j[5]:.2f}"
                )
                if teleop:
                    if robot_side == "left" and ctrl_left:
                        self.node.apply_hand_joint_positions("left", j)
                    elif robot_side == "right" and ctrl_right:
                        self.node.apply_hand_joint_positions("right", j)
            if lines:
                self.skeleton_joints_label.setText(" | ".join(lines))
                mode = "遥控" if teleop else "预览"
                self.skeleton_status_label.setText(
                    f"手骨架: {mode} 检测到 {len(hands)} 只手 @ {src_label}"
                )
            else:
                self.skeleton_joints_label.setText("关节: (未检测到手)")
                self.skeleton_status_label.setText(f"手骨架: 未检测到手 @ {src_label}")
        except Exception as exc:
            self.skeleton_status_label.setText(f"手骨架错误: {exc}")
        finally:
            self._skeleton_busy = False

