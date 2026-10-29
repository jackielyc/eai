"""ui.window.simulation mujoco start clicked — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationMujocoStartClickedMixin:
    def _on_mujoco_start_clicked(self, *, viewer_only: bool = False) -> None:
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("MuJoCo 正在运行")
            return
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        if self._sim_backend_id() != "mujoco":
            self._on_mujoco_status("请先将仿真后端切换为 MuJoCo")
            return
        policy = ""
        if hasattr(self, "mujoco_policy_combo"):
            policy = str(self.mujoco_policy_combo.currentData() or "")
        if (not viewer_only) and _eai_configuration_models.is_lingbot_va_policy(policy):
            self.mujoco_log_edit.clear()
            self._append_mujoco_log(
                "[mujoco/LingBot-VA] 启动 wan_va（serve_only）；"
                "MJCF viewer 无 VA closed-loop，完整评测请用 LIBERO/RoboTwin/Isaac"
            )
            self._launch_lingbot_va_backend("mujoco", label="mujoco-va")
            return
        if (not viewer_only) and _eai_configuration_models.is_vpp2_policy(policy):
            self.mujoco_log_edit.clear()
            self._append_mujoco_log(
                "[mujoco/VPP2] 启动 policy server（serve_only）；"
                "MJCF viewer 无 VPP2 closed-loop，完整评测请用 LIBERO/Isaac"
            )
            self._launch_vpp2_backend("mujoco", label="mujoco-vpp2")
            return
        mjcf = self._resolve_mujoco_mjcf_path()
        if not mjcf or not _eai_platform_bootstrap.os.path.isfile(mjcf):
            self._append_mujoco_log(f"[ERROR] 模型无效: {mjcf or '(空)'}")
            self._on_mujoco_status("请选择有效 MJCF")
            return
        if not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_mujoco_log(
                "[warn] DISPLAY 为空，若窗口未弹出请检查本机图形环境"
            )
        if (not viewer_only) and not self._ensure_live_hud_reward_or_abort("mujoco"):
            return
        self.mujoco_log_edit.clear()
        if viewer_only:
            self._append_mujoco_log("[mujoco] 只开界面：官方 MJCF viewer（无策略）")
        hud_env = (
            {}
            if viewer_only
            else self._rynnvalue_live_hud_child_env("mujoco")
        )
        if hud_env:
            # Force EGL path when Live HUD needs RGB frames from renderer.
            mode = str(self.mujoco_mode_combo.currentData() or "python")
            if mode in ("auto", "python"):
                mode = "egl"
                self._append_mujoco_log(
                    "[mujoco] Live HUD 开启：改用 EGL viewer 以便推送 RGB 帧"
                )
            else:
                mode = str(self.mujoco_mode_combo.currentData() or "python")
            self._expand_workspace_panel()
            self._append_mujoco_log(
                f"[mujoco] RynnValue Live HUD "
                f"url={hud_env.get('RYNNVALUE_SERVER_URL')}"
            )
        else:
            mode = str(self.mujoco_mode_combo.currentData() or "python")
            if viewer_only and mode in ("auto", "python"):
                mode = "egl"
        env_extra = dict(hud_env or {})
        env_extra.update(self._sim_bridge_child_env())
        tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
        prev = env_extra.get("PYTHONPATH") or ""
        env_extra["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
            [tools] + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p != tools]
        )
        self._mujoco_launcher.start(
            mjcf_path=mjcf,
            mujoco_root=self.mujoco_root_edit.text().strip(),
            python_bin=self.mujoco_python_edit.text().strip(),
            mode=mode,
            install=False,
            env_extra=env_extra,
        )
        self._update_mujoco_ui()

    def _on_mujoco_stop_clicked(self) -> None:
        self._append_mujoco_log("--- 用户停止 MuJoCo ---")
        self._set_sim_paused(False, log=False)
        self._mujoco_launcher.stop()
        self._update_mujoco_ui()

    def _on_rynnvalue_status(self, msg: str) -> None:
        if hasattr(self, "rynnvalue_status_label"):
            self.rynnvalue_status_label.setText(msg or "空闲")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _on_rynnvalue_log_dir(self, path: str) -> None:
        if hasattr(self, "rynnvalue_log_dir_edit"):
            self.rynnvalue_log_dir_edit.setText(path)
        if hasattr(self, "rynnvalue_open_log_btn"):
            self.rynnvalue_open_log_btn.setEnabled(
                bool(path) and _eai_platform_bootstrap.os.path.isdir(path)
            )

    def _on_rynnvalue_output_video(self, path: str) -> None:
        path = (path or "").strip()
        if not path:
            return
        self._rynnvalue_last_video = path
        if hasattr(self, "rynnvalue_open_video_btn"):
            self.rynnvalue_open_video_btn.setEnabled(_eai_platform_bootstrap.os.path.isfile(path))
        render_gui = bool(
            getattr(self, "rynnvalue_render_gui_check", None)
            and self.rynnvalue_render_gui_check.isChecked()
        )
        if render_gui and _eai_platform_bootstrap.os.path.isfile(path):
            self._play_rynnvalue_video(path)

    def _on_rynnvalue_root_browse(self) -> None:
        cur = self.rynnvalue_root_edit.text().strip() or _eai_configuration_tabs.RYNNVALUE_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 RynnValue 仓库根目录", cur
        )
        if selected:
            self.rynnvalue_root_edit.setText(selected)
            self.rynnvalue_python_edit.setText(_eai_adapters_models_discovery.resolve_rynnvalue_python(selected))
            example = _eai_platform_bootstrap.os.path.join(
                selected, "example", "Put_the_box_in_the_drawer_and_close_it.mp4"
            )
            if _eai_platform_bootstrap.os.path.isfile(example) and not self.rynnvalue_video_edit.text().strip():
                self.rynnvalue_video_edit.setText(example)

    def _on_rynnvalue_model_browse(self) -> None:
        cur = self.rynnvalue_model_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 RynnValue HuggingFace 模型目录", start
        )
        if selected:
            self.rynnvalue_model_edit.setText(selected)

    def _on_rynnvalue_ckpt_browse(self) -> None:
        cur = self.rynnvalue_ckpt_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 checkpoint 目录（含 model.pt）", start
        )
        if selected:
            self.rynnvalue_ckpt_edit.setText(selected)

    def _on_rynnvalue_video_browse(self) -> None:
        cur = self.rynnvalue_video_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) or _eai_platform_bootstrap.os.getcwd()
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择输入视频",
            start,
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)",
        )
        if selected:
            self.rynnvalue_video_edit.setText(selected)

    def _on_rynnvalue_job_changed(self, *_args) -> None:
        job = str(self.rynnvalue_job_combo.currentData() or "infer").strip()
        if hasattr(self, "rynnvalue_start_btn"):
            if job in ("reward_server", "server", "serve"):
                self.rynnvalue_start_btn.setText("启动服务")
            elif job in ("policy_ranking", "confusion_matrix"):
                self.rynnvalue_start_btn.setText("启动评测")
            else:
                self.rynnvalue_start_btn.setText("启动推理")
        self._update_rynnvalue_ui()

    def _on_rynnvalue_start_clicked(self) -> None:
        if self._rynnvalue_launcher.is_running():
            self._on_rynnvalue_status("RynnValue 正在运行")
            return
        job = str(self.rynnvalue_job_combo.currentData() or "infer").strip()
        port = int(self.rynnvalue_port_spin.value())
        # reward_server：先探测已部署实例，命中则跳过启动（无需再填 model）
        if job in ("reward_server", "server", "serve"):
            url = f"http://127.0.0.1:{port}"
            ok_srv, msg_srv, info_srv = _eai_adapters_models_discovery.probe_rynnvalue_reward_server(
                url, timeout_s=2.0
            )
            if ok_srv:
                self._append_rynnvalue_log(f"[adopt] 已部署，跳过启动: {msg_srv}")
                self._on_rynnvalue_status(f"已部署，跳过启动: {msg_srv}")
                self._update_rynnvalue_ui()
                return
            if info_srv.get("listening"):
                self._append_rynnvalue_log(f"[skip] {msg_srv}")
                self._on_rynnvalue_status(msg_srv)
                return
        model = self.rynnvalue_model_edit.text().strip()
        if not model and job not in ("reward_server",):
            self._append_rynnvalue_log("[ERROR] 请填写模型路径")
            self._on_rynnvalue_status("请填写模型路径")
            return
        if job in ("reward_server",) and not model and not self.rynnvalue_ckpt_edit.text().strip():
            self._append_rynnvalue_log("[ERROR] 请填写 model 或 checkpoint")
            self._on_rynnvalue_status("请填写 model 或 checkpoint")
            return
        render_gui = bool(self.rynnvalue_render_gui_check.isChecked())
        if render_gui and not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_rynnvalue_log(
                "[rynnvalue] 警告: DISPLAY 为空，图形窗口可能无法弹出"
            )
        extras_raw = self.rynnvalue_extra_edit.text().strip()
        extras = _eai_platform_bootstrap.shlex.split(extras_raw) if extras_raw else []
        self.rynnvalue_log_edit.clear()
        self.rynnvalue_log_dir_edit.clear()
        self.rynnvalue_open_log_btn.setEnabled(False)
        self._rynnvalue_launcher.start(
            job=job,
            repo_root=self.rynnvalue_root_edit.text().strip(),
            python_bin=self.rynnvalue_python_edit.text().strip(),
            model_path=model,
            checkpoint_path=self.rynnvalue_ckpt_edit.text().strip(),
            video_path=self.rynnvalue_video_edit.text().strip(),
            instruction=self.rynnvalue_instruction_edit.text().strip(),
            robot_description=self.rynnvalue_robot_edit.text().strip(),
            camera_description=self.rynnvalue_camera_edit.text().strip(),
            num_frames=int(self.rynnvalue_frames_spin.value()),
            num_steps=int(self.rynnvalue_steps_spin.value()),
            batch_size=int(self.rynnvalue_batch_spin.value()),
            max_image_side=640,
            max_new_tokens=int(self.rynnvalue_tokens_spin.value()),
            fps=int(self.rynnvalue_fps_spin.value()),
            port=int(self.rynnvalue_port_spin.value()),
            cuda_devices=self.rynnvalue_cuda_edit.text().strip(),
            render_gui=render_gui,
            extra_args=extras,
        )
        self._update_rynnvalue_ui()

    def _on_rynnvalue_stop_clicked(self) -> None:
        self._append_rynnvalue_log("--- 用户停止 RynnValue ---")
        msg = self._stop_rynnvalue_model(log=True)
        self._append_rynnvalue_log(f"[rynnvalue] {msg}")
        self._update_rynnvalue_ui()

    def _on_rynnvalue_open_log_clicked(self) -> None:
        path = self.rynnvalue_log_dir_edit.text().strip()
        if path and _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_rynnvalue_open_video_clicked(self) -> None:
        path = getattr(self, "_rynnvalue_last_video", "") or ""
        if path and _eai_platform_bootstrap.os.path.isfile(path):
            self._play_rynnvalue_video(path)
            return
        newest = self._find_newest_rynnvalue_trend_video()
        if newest:
            self._rynnvalue_last_video = newest
            if hasattr(self, "rynnvalue_log_dir_edit") and not self.rynnvalue_log_dir_edit.text().strip():
                self.rynnvalue_log_dir_edit.setText(_eai_platform_bootstrap.os.path.dirname(newest))
            self._play_rynnvalue_video(newest)
            return
        self._on_rynnvalue_status("暂无趋势视频可播放（请先跑一次 infer）")

    def _sim_bridge_dir(self) -> str:
        edit = getattr(self, "sim_bridge_dir_edit", None)
        raw = edit.text().strip() if edit is not None else ""
        return _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(raw or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT)
        )

    def _sim_bridge_child_env(self) -> _eai_platform_bootstrap.Dict[str, str]:
        """Env vars so any sim child can read/write gui_robot_cmd / sim_robot_state."""
        return {
            "ISAAC_CAM_BRIDGE_DIR": self._sim_bridge_dir(),
            "EAI_DIR": _eai_configuration_models.EAI_DIR,
        }

    def _sim_teleop_active(self) -> bool:
        """任一仿真后端运行且走共享指令桥时，「手臂/手」文本/滑块遥控仿真。"""
        if _eai_configuration_policies.write_gui_robot_cmd is None:
            return False
        isaac = getattr(self, "_sim_eval_launcher", None)
        mj = getattr(self, "_mujoco_launcher", None)
        isaac_on = bool(isaac is not None and isaac.is_running())
        mj_on = bool(mj is not None and mj.is_running())
        if not isaac_on and not mj_on:
            state = self._read_sim_robot_state()
            if _eai_configuration_policies.sim_state_is_fresh is None:
                return False
            # RoboDojo 可能只回写 gripper；时间戳新鲜即可走仿真遥控
            if bool(_eai_configuration_policies.sim_state_is_fresh(state, max(_eai_configuration_policies.SIM_TELEOP_STATE_MAX_AGE_S, 5.0))):
                return True
            age = self._sim_bridge_newest_age_s()
            return age is not None and age <= 30.0
        # MuJoCo / MolmoSpaces / Arena / LIBERO / RoboTwin（统一走 _mujoco_launcher）
        if mj_on and not isaac_on:
            return True
        # Isaac / RoboDojo：本窗口拥有评测进程时始终开桥（含策略=无与有策略）
        # 有策略时也可能写 gui_robot_cmd；TCP 缺失时走「直接写绝对目标」分支
        if isaac_on:
            return True
        state = self._read_sim_robot_state()
        if _eai_configuration_policies.sim_state_is_fresh is None:
            return False
        return bool(_eai_configuration_policies.sim_state_is_fresh(state, max(_eai_configuration_policies.SIM_TELEOP_STATE_MAX_AGE_S, 5.0)))

    def _sim_abs_cmd_ready(self) -> bool:
        """绝对位姿指令是否应写入仿真桥（即使瞬时判定 teleop 为假）。"""
        if _eai_configuration_policies.write_gui_robot_cmd is None:
            return False
        if self._sim_teleop_active():
            return True
        if self._any_sim_process_running():
            return True
        age = self._sim_bridge_newest_age_s()
        if age is not None and age <= 60.0:
            return True
        state = self._read_sim_robot_state()
        if state and _eai_configuration_policies.sim_state_is_fresh is not None:
            return bool(_eai_configuration_policies.sim_state_is_fresh(state, 60.0))
        return False

    def _sim_tcp_pose(
        self, arm_side: str
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
        state = self._read_sim_robot_state()
        if not state:
            return None
        if _eai_configuration_policies.sim_state_is_fresh is not None and not _eai_configuration_policies.sim_state_is_fresh(
            state, _eai_configuration_policies.SIM_TELEOP_STATE_MAX_AGE_S
        ):
            return None
        keys: _eai_platform_bootstrap.List[str]
        if arm_side == "right":
            keys = ["right_ee_pose"]
        else:
            # 单臂 / RoboDojo 常把唯一臂写在 left_ee_pose 或 ee_pose
            keys = ["left_ee_pose", "ee_pose"]
        for key in keys:
            pose = state.get(key)
            if not isinstance(pose, (list, tuple)) or len(pose) != 7:
                continue
            try:
                result = _eai_perception_pose_estimation.xyz_xyzw_from_pose7_wxyz(pose)
                self._maybe_record_arm_home(arm_side, result)
                return result
            except Exception:
                continue
        return None

    def _sim_resolve_arm_side(self, prefer: str) -> str:
        """选实际有仿真 TCP 的手臂；右腕无状态时回落到左臂（RoboDojo 常见）。"""
        prefer = prefer if prefer in ("left", "right") else "left"
        if self._sim_tcp_pose(prefer) is not None:
            return prefer
        other = "right" if prefer == "left" else "left"
        if self._sim_tcp_pose(other) is not None:
            return other
        # 尚无 TCP 时仍偏向左臂：NoopModelClient 单臂映射 left_ee_pose / ee_pose
        return "left"

    def _sim_move_duration_s(self, dist_m: float) -> float:
        slider = getattr(self, "arm_move_speed_slider", None)
        pct = 0.5 if slider is None else max(0.1, min(1.0, slider.value() / 100.0))
        # ~3–25 cm/s，与真机「慢速移动」手感接近
        speed_m_s = 0.03 + pct * 0.22
        return max(0.25, min(8.0, float(dist_m) / speed_m_s))

    def _on_sim_arm_move_tick(self) -> None:
        move = self._sim_arm_move
        if move is None:
            self._sim_arm_move_timer.stop()
            return
        duration = max(1e-3, float(move["duration"]))
        t = (_eai_platform_bootstrap.time.time() - float(move["t0"])) / duration
        done = t >= 1.0
        t = min(1.0, max(0.0, t))
        kwargs: dict = {"enabled": True}
        if move.get("mode") == "dual":
            for side_key, start_key, goal_key, quat_s_key, quat_g_key, out_key in (
                (
                    "left",
                    "left_start_xyz",
                    "left_goal_xyz",
                    "left_start_quat",
                    "left_goal_quat",
                    "left_ee_pose",
                ),
                (
                    "right",
                    "right_start_xyz",
                    "right_goal_xyz",
                    "right_start_quat",
                    "right_goal_quat",
                    "right_ee_pose",
                ),
            ):
                if start_key not in move or goal_key not in move:
                    continue
                sx, sy, sz = move[start_key]
                gx, gy, gz = move[goal_key]
                xyz = (
                    float(sx) + (float(gx) - float(sx)) * t,
                    float(sy) + (float(gy) - float(sy)) * t,
                    float(sz) + (float(gz) - float(sz)) * t,
                )
                quat = _eai_perception_pose_estimation.slerp_xyzw(move[quat_s_key], move[quat_g_key], t)
                kwargs[out_key] = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(xyz, quat)
            self._write_sim_gui_cmd(**kwargs)
            if done:
                settled, settle_note = self._sim_arm_move_try_settle(move)
                if not settled:
                    self.status_bar.showMessage(
                        f"仿真双臂指令已写完，等待 IK 到位… {settle_note}", 2000
                    )
                    self._update_arm_move_btns_ui()
                    return
                self._sim_arm_move = None
                self._sim_arm_move_timer.stop()
                label = str(move.get("label") or "双臂归位")
                done_msg = f"仿真双臂移动完成 → {label}{settle_note}"
                if not self._after_arm_motion_settled(True, done_msg):
                    if not self._text_cmd_pumping:
                        extra = self._pump_text_cmd_queue()
                        if extra:
                            done_msg = f"{done_msg}；{extra}"
                self.status_bar.showMessage(done_msg)
                self._update_arm_move_btns_ui()
                return
            self._update_arm_move_btns_ui()
            return

        sx, sy, sz = move["start_xyz"]
        gx, gy, gz = move["goal_xyz"]
        xyz = (
            float(sx) + (float(gx) - float(sx)) * t,
            float(sy) + (float(gy) - float(sy)) * t,
            float(sz) + (float(gz) - float(sz)) * t,
        )
        quat = _eai_perception_pose_estimation.slerp_xyzw(move["start_quat"], move["goal_quat"], t)
        pose7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(xyz, quat)
        if move["arm_side"] == "right":
            kwargs["right_ee_pose"] = pose7
        else:
            kwargs["left_ee_pose"] = pose7
        hold = move.get("hold_pose")
        if hold is not None:
            h_xyz, h_quat = hold
            hold7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(h_xyz, h_quat)
            if move["hold_side"] == "right":
                kwargs["right_ee_pose"] = hold7
            else:
                kwargs["left_ee_pose"] = hold7
        self._write_sim_gui_cmd(**kwargs)
        if done:
            settled, settle_note = self._sim_arm_move_try_settle(move)
            if not settled:
                self.status_bar.showMessage(
                    f"仿真指令已写完，等待 IK 到位… {settle_note}", 2000
                )
                self._update_arm_move_btns_ui()
                return
            # 水平对准完成后，再启动下压段
            if move.get("approach_pending") and move.get("final_xyz") is not None:
                cur = self._sim_tcp_pose(str(move["arm_side"]))
                start_xyz = cur[0] if cur is not None else move["goal_xyz"]
                start_quat = cur[1] if cur is not None else move["goal_quat"]
                final_xyz = move["final_xyz"]
                final_quat = move.get("final_quat") or move["goal_quat"]
                dist2 = _eai_platform_bootstrap.math.sqrt(
                    sum(
                        (float(final_xyz[i]) - float(start_xyz[i])) ** 2
                        for i in range(3)
                    )
                )
                move["start_xyz"] = start_xyz
                move["start_quat"] = start_quat
                move["goal_xyz"] = final_xyz
                move["goal_quat"] = final_quat
                move["approach_pending"] = False
                move.pop("settle_t0", None)
                move["t0"] = _eai_platform_bootstrap.time.time()
                move["duration"] = self._sim_move_duration_s(dist2)
                self.status_bar.showMessage(
                    f"仿真已水平对准，开始下压… ({move['duration']:.1f}s) {settle_note}",
                    3000,
                )
                self._update_arm_move_btns_ui()
                return
            side_name = _eai_perception_pose_estimation.arm_side_label(str(move["arm_side"]))
            label = str(move.get("label") or "")
            self._sim_arm_move = None
            self._sim_arm_move_timer.stop()
            done_msg = (
                f"仿真{side_name}移动完成"
                + (f" → {label}" if label else "")
                + settle_note
            )
            if not self._after_arm_motion_settled(True, done_msg):
                if not self._text_cmd_pumping:
                    extra = self._pump_text_cmd_queue()
                    if extra:
                        done_msg = f"{done_msg}；{extra}"
            self.status_bar.showMessage(done_msg)
            self._update_arm_move_btns_ui()
            return
        self._update_arm_move_btns_ui()

    def _sim_arm_move_try_settle(self, move: dict) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """插值结束后持有/修正目标，直到 EE（或夹爪中点）到位或超时。

        仿真下姿态常跟不住：对「手指中点匹配」用世界系闭环
        ``EE ← EE + (target_mid - cur_mid)``，保持当前实际姿态，把中点拉到目标。
        """
        now = _eai_platform_bootstrap.time.time()
        if "settle_t0" not in move:
            move["settle_t0"] = now
            # 把插值时钟钉在终点，后续 tick 继续走 done 分支
            move["t0"] = now - float(move.get("duration") or 1.0)

        use_mid = (
            move.get("mode") != "dual"
            and not move.get("approach_pending")
            and self._sim_uses_finger_mid_match()
            and move.get("target_mid_xyz") is not None
        )

        # 闭环：按实际中点残差改写 EE 目标（保留当前可达成姿态）
        if use_mid:
            side = str(move.get("arm_side") or "left")
            cur = self._sim_tcp_pose(side)
            if cur is not None:
                target_mid = move["target_mid_xyz"]
                cur_mid = self._finger_mid_xyz_from_ee(cur[0], cur[1])
                merr = _eai_platform_bootstrap.math.sqrt(
                    sum(
                        (float(cur_mid[i]) - float(target_mid[i])) ** 2
                        for i in range(3)
                    )
                )
                if merr > float(_eai_configuration_policies.SIM_ARM_SETTLE_TOL_M):
                    # 限幅单步修正，避免一次跳太远触发碰撞
                    max_step = 0.06
                    corr = [
                        float(target_mid[i]) - float(cur_mid[i]) for i in range(3)
                    ]
                    corr_n = _eai_platform_bootstrap.math.sqrt(sum(c * c for c in corr)) or 1.0
                    scale = min(1.0, max_step / corr_n)
                    new_ee = (
                        float(cur[0][0]) + corr[0] * scale,
                        float(cur[0][1]) + corr[1] * scale,
                        float(cur[0][2]) + corr[2] * scale,
                    )
                    move["goal_xyz"] = new_ee
                    move["goal_quat"] = (
                        float(cur[1][0]),
                        float(cur[1][1]),
                        float(cur[1][2]),
                        float(cur[1][3]),
                    )

        # 续写终值，防止 Noop 丢目标
        kwargs: dict = {"enabled": True}
        goals: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]] = []
        if move.get("mode") == "dual":
            for side, gxyz, gquat in (
                ("left", "left_goal_xyz", "left_goal_quat"),
                ("right", "right_goal_xyz", "right_goal_quat"),
            ):
                if gxyz not in move or gquat not in move:
                    continue
                pose7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(move[gxyz], move[gquat])
                kwargs[f"{side}_ee_pose"] = pose7
                goals.append((side, move[gxyz], move[gquat]))
        else:
            side = str(move.get("arm_side") or "left")
            pose7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(move["goal_xyz"], move["goal_quat"])
            kwargs["left_ee_pose" if side == "left" else "right_ee_pose"] = pose7
            goals.append((side, move["goal_xyz"], move["goal_quat"]))
            hold = move.get("hold_pose")
            if hold is not None:
                h_side = str(move.get("hold_side") or ("right" if side == "left" else "left"))
                hold7 = _eai_perception_pose_estimation.pose7_wxyz_from_xyz_xyzw(hold[0], hold[1])
                kwargs["left_ee_pose" if h_side == "left" else "right_ee_pose"] = hold7
        self._write_sim_gui_cmd(**kwargs)

        max_err = 0.0
        mid_err = 0.0
        parts: _eai_platform_bootstrap.List[str] = []
        for side, gxyz, gquat in goals:
            cur = self._sim_tcp_pose(side)
            if cur is None:
                parts.append(f"{_eai_perception_pose_estimation.arm_side_label(side)}无回读")
                max_err = max(max_err, 1.0)
                continue
            err = _eai_platform_bootstrap.math.sqrt(
                sum((float(cur[0][i]) - float(gxyz[i])) ** 2 for i in range(3))
            )
            max_err = max(max_err, err)
            cur_mid = self._finger_mid_xyz_from_ee(cur[0], cur[1])
            if move.get("target_mid_xyz") is not None and not move.get(
                "approach_pending"
            ):
                goal_mid = move["target_mid_xyz"]
            else:
                goal_mid = self._finger_mid_xyz_from_ee(gxyz, gquat)
            merr = _eai_platform_bootstrap.math.sqrt(
                sum((float(cur_mid[i]) - float(goal_mid[i])) ** 2 for i in range(3))
            )
            mid_err = max(mid_err, merr)
            parts.append(
                f"{_eai_perception_pose_estimation.arm_side_label(side)} EEΔ={err * 1000:.0f}mm 中点Δ={merr * 1000:.0f}mm"
            )
        note = "；".join(parts)
        # 手指中点模式以中点残差为准；否则看 EE
        done_err = mid_err if use_mid else max_err
        if done_err <= float(_eai_configuration_policies.SIM_ARM_SETTLE_TOL_M):
            return True, f"（到位 {note}）"
        if now - float(move["settle_t0"]) >= float(_eai_configuration_policies.SIM_ARM_SETTLE_MAX_S):
            return True, f"（超时未完全到位 {note}；请检查 IK/碰撞）"
        return False, note

    def _sim_uses_finger_mid_match(self) -> bool:
        """仿真遥控时绝对目标按两指结合中点匹配（X5 ee=link6≠指尖中点）。"""
        return bool(self._sim_teleop_active() or self._sim_abs_cmd_ready())

    def _sim_bridge_dir_path(self) -> str:
        edit = getattr(self, "sim_bridge_dir_edit", None)
        raw = edit.text().strip() if edit is not None else ""
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT))

    def _sim_preview_active(self) -> bool:
        eval_on = (
            getattr(self, "_sim_eval_launcher", None) is not None
            and self._sim_eval_launcher.is_running()
        )
        bridge_on = (
            getattr(self, "_sim_bridge_launcher", None) is not None
            and self._sim_bridge_launcher.is_running()
        )
        mj_on = (
            getattr(self, "_mujoco_launcher", None) is not None
            and self._mujoco_launcher.is_running()
        )
        return bool(eval_on or bridge_on or mj_on)

