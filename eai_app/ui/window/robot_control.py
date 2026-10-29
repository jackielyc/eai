"""ui.window.robot control — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

class RobotControlMixin:
    def _on_robot_stack_clicked(self) -> None:
        self._ensure_host_camera_forwarder()
        self._stack_launcher.start_stack()
        # 栈起来后延迟订阅头/腕（forwarder + graph 需要一点时间）
        _eai_platform_bootstrap.QTimer.singleShot(
            3000, lambda: self._ensure_essential_robot_cameras(reason="启动机器人栈")
        )
        _eai_platform_bootstrap.QTimer.singleShot(
            15000, lambda: self._ensure_essential_robot_cameras(reason="启动机器人栈")
        )

    def _arm_panel_selector_keys(self) -> _eai_platform_bootstrap.List[str]:
        bar = getattr(self, "arm_panel_tab_bar", None)
        if bar is None:
            return []
        return [str(bar.tabData(i) or "") for i in range(bar.count())]

    def _arm_panel_selector_key(self) -> str:
        for key, _label, attr in _eai_configuration_tabs.ARM_PANEL_SELECTOR_SPECS:
            radio = getattr(self, attr, None)
            if radio is not None and radio.isChecked():
                return key
        return _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS[0] if _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS else "human_demo"

    def _on_arm_panel_changed(self, checked: bool = False) -> None:
        if not checked:
            return
        self._sync_arm_panel_tab_from_radios()
        self._apply_arm_panel_stack()

    def _on_arm_panel_tab_changed(self, index: int) -> None:
        if getattr(self, "_arm_panel_tabs_guard", False):
            return
        bar = getattr(self, "arm_panel_tab_bar", None)
        if bar is None or index < 0:
            return
        key = str(bar.tabData(index) or "").strip()
        radio = (getattr(self, "_arm_panel_radio_by_key", None) or {}).get(key)
        if radio is None:
            return
        if not radio.isChecked():
            radio.setChecked(True)
        else:
            self._apply_arm_panel_stack()

    def _on_arm_panel_tab_moved(self, _from: int, _to: int) -> None:
        if getattr(self, "_arm_panel_tabs_order_guard", False):
            return
        keys = self._arm_panel_selector_keys()
        if not keys:
            return
        try:
            path = _eai_infrastructure_ui_storage.save_arm_panels_order(keys)
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"已保存手臂功能面板顺序 → {path}", 3000)
        except Exception as exc:
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"保存手臂功能面板顺序失败: {exc}", 4000)

    def _robot_grasp_action(self, action):
        panel = self.grasp_execution_panel
        if action == 'restore':
            view = getattr(self, '_robot_view', None)
            if not panel.is_simulation and view is not None:
                panel.scene_confirm.setChecked(view._restore_scene_check.isChecked())
                view._restore_scene_check.setChecked(False)
            panel.restore_startup()
        else:
            panel.stop()
        return panel.message

    def _grasp_wrench_feedback(self, side):
        with self._grasp_wrench_lock:
            sample = self._grasp_wrench_cache.get(str(side))
            return None if sample is None else dict(sample)

    def _on_grasp_wrench(self, side, msg):
        try:
            values = [float(v) for v in (msg.force_data or [])]
            valid = len(values) >= 6 and all(_eai_platform_bootstrap.math.isfinite(v) for v in values[:6])
            sample = {
                'force': values[:3] if valid else None,
                'torque': values[3:6] if valid else None,
                'force_state': int(getattr(msg, 'force_state', 0) or 0),
                'force_coordinate': int(getattr(msg, 'force_coordinate', 0) or 0),
                'stamp': _eai_platform_bootstrap.time.time(),
            }
        except Exception:
            sample = {'force': None, 'torque': None, 'force_state': -3, 'stamp': _eai_platform_bootstrap.time.time()}
        with self._grasp_wrench_lock:
            self._grasp_wrench_cache[str(side)] = sample

    def _robot_tcp_state_path(self) -> str:
        return _eai_platform_bootstrap.os.environ.get(
            "EAI_ROBOT_TCP_STATE",
            _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.tempfile.gettempdir(), "eai_robot_tcp_state.json"),
        )

    def _tcp_bridge_json_age_s(self) -> float:
        path = self._robot_tcp_state_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            updated = float(data.get("updated_at") or 0.0)
            if updated <= 0:
                return 1e9
            return max(0.0, _eai_platform_bootstrap.time.time() - updated)
        except Exception:
            return 1e9

    def _on_arm_dir_spin_changed(self, *_args) -> None:
        """六向输入变化 → 同步隐藏 ΔXYZ，并刷新目标显示。"""
        # 以当前编辑侧为准困难；左右都同步各自净偏移到展示用 spin（用左臂写旧三旋钮供兼容）
        self._sync_offset_spins_from_dir("left")
        self._on_move_target_params_changed()

    def _arm_dir_spin_cm(self, arm_side: str, direction: str) -> float:
        spins = getattr(self, "_arm_dir_spins", {}).get(arm_side) or {}
        spin = spins.get(direction)
        if spin is None:
            return 0.0
        return max(0.0, float(spin.value()))

    def _arm_combined_offset_m(self, arm_side: str) -> _eai_platform_bootstrap.Tuple[float, float, float]:
        """六向净合成：前-后 → X，左-右 → Y，上-下 → Z（米）。"""
        f = self._arm_dir_spin_cm(arm_side, "forward")
        b = self._arm_dir_spin_cm(arm_side, "back")
        l = self._arm_dir_spin_cm(arm_side, "left")
        r = self._arm_dir_spin_cm(arm_side, "right")
        u = self._arm_dir_spin_cm(arm_side, "up")
        d = self._arm_dir_spin_cm(arm_side, "down")
        return ((f - b) * 0.01, (l - r) * 0.01, (u - d) * 0.01)

    def _on_arm_direction_move(self, arm_side: str, direction: str) -> None:
        """前/后/左/右/上/下：只做对应方向单轴移动。"""
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        dir_cn = _eai_perception_pose_estimation.ARM_DIR_LABEL_CN.get(direction, direction)
        if self.node.is_slow_motion_busy() or self._is_sim_arm_moving():
            self.status_bar.showMessage("已有移动进行中，请等待或先取消")
            return
        if not self._sim_teleop_active() and not self.node.is_arm_enabled():
            self.status_bar.showMessage("手臂未使能：请先点「启用手臂」")
            self._update_arm_dir_btns_ui()
            return
        if hasattr(self, "move_target_relative_radio"):
            self.move_target_relative_radio.setChecked(True)
        if self._arm_dir_spin_cm(arm_side, direction) < 1e-6:
            self.status_bar.showMessage(
                f"请先填写{side_name}「{dir_cn}」方向的步长（cm）"
            )
            return
        goal = self._resolve_direction_move_goal(arm_side, direction)
        if goal is None:
            self.status_bar.showMessage(
                f"无法{side_name}向{dir_cn}移动：无 TCP 或步长无效"
            )
            return
        if self._sim_teleop_active():
            msg = self._start_sim_arm_move(arm_side, goal)
            self.status_bar.showMessage(msg)
            self._update_arm_move_btns_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

    def _on_move_target_params_changed(self, *_args) -> None:
        if self._last_segment_target is None and self.move_target_segment_radio.isChecked():
            self.move_target_relative_radio.setChecked(True)
        self._update_move_offset_ui_visibility()
        self._update_arm_pose_display()
        self._update_arm_move_btns_ui()

    def _tcp_for_relative_goal(
        self, arm_side: str
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
        if self._sim_teleop_active():
            sim_tcp = self._sim_tcp_pose(arm_side)
            if sim_tcp is not None:
                return sim_tcp
        tcp = self.node._tcp_pose_in_ik_frame(arm_side, timeout_s=_eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S)
        if tcp is not None:
            self._maybe_record_arm_home(arm_side, tcp)
            return tcp
        state = self.node.get_robot_state()
        raw = state.left_tcp if arm_side == "left" else state.right_tcp
        if raw is not None and raw.valid:
            frame = _eai_adapters_robot_geometry.normalize_frame_id(raw.frame_id)
            if frame == _eai_adapters_robot_geometry.normalize_frame_id(_eai_configuration_policies.IK_TARGET_FRAME) or frame in _eai_configuration_policies.MINK_FK_FRAME_ALIASES:
                pose = (raw.xyz, raw.quat_xyzw)
                self._maybe_record_arm_home(arm_side, pose)
                return pose
        return None

    def _on_arm_home_clicked(self) -> None:
        """文本指挥旁「归位」：清空队列，双臂回初始 EE。"""
        self._text_cmd_queue.clear()
        self._ensure_arm_home_recorded()
        # 先尽量刷一次当前位姿，便于补记 home
        for side in ("left", "right"):
            self._tcp_for_relative_goal(side)
        self._ensure_arm_home_recorded()
        if not self._arm_home_pose:
            self.status_bar.showMessage(
                "归位失败: 尚无臂位姿记录（请先启动仿真或等 TCP 回读）", 8000
            )
            return
        if self._sim_teleop_active() or self._sim_abs_cmd_ready():
            self._cancel_sim_arm_move("")
            msg = self._start_sim_dual_arm_home_move()
            self._update_arm_move_btns_ui()
            self.status_bar.showMessage(msg, 8000)
            return
        # 真机：排队两侧绝对目标（保持 home 姿态）
        actions: _eai_platform_bootstrap.List[_eai_platform_bootstrap.ArmHandTextAction] = []
        for side in ("left", "right"):
            home = self._arm_home_pose.get(side)
            if home is None:
                continue
            xyz, quat = home
            actions.append(
                _eai_platform_bootstrap.ArmHandTextAction(
                    kind="arm",
                    side=side,
                    abs_xyz=xyz,
                    abs_quat_xyzw=quat,
                    label=f"{_eai_perception_pose_estimation.arm_side_label(side)}归位",
                )
            )
        if not actions:
            self.status_bar.showMessage("归位失败: 无已记录的初始位姿", 8000)
            return
        stop_msg = self._cancel_arm_motion_for_text()
        self._text_cmd_queue.extend(actions)
        pump = self._pump_text_cmd_queue()
        parts = [p for p in (stop_msg, pump or "已开始归位") if p]
        self.status_bar.showMessage("；".join(parts), 8000)

    def _on_arm_move_speed_changed(self, value: int) -> None:
        speed = _eai_adapters_robot_geometry.slider_to_arm_move_speed(value)
        self.node.set_arm_move_joint_speed(speed)
        self.arm_move_speed_label.setText(_eai_adapters_robot_geometry.format_arm_move_speed_label(value))
        if not self.node.is_slow_motion_busy():
            self._update_arm_move_btns_ui()

    def _on_arm_enable_clicked(self) -> None:
        """开关手臂：打开/关闭对机器人手臂的控制（不扫相机，避免卡 UI）。"""
        self._ensure_host_arm_hand_bridge()
        self._ensure_enable_state_subscriptions()
        enable = not self.node.is_arm_enabled()
        if not enable:
            # 关闭时立刻停掉本进程轨迹，避免关使能后仍在发关节目标
            if getattr(self, "_repeat_active", False):
                self._repeat_active = False
                self._repeat_queue = []
                wait_timer = getattr(self, "_repeat_wait_timer", None)
                if wait_timer is not None:
                    wait_timer.stop()
                self._update_arm_repeat_btn()
            try:
                if self.node.is_slow_motion_busy():
                    self.status_bar.showMessage(self.node.cancel_slow_motion())
            except Exception:
                pass
            try:
                self._pending_arm_move_goal = None
                self._arm_enable_wait_timer.stop()
            except Exception:
                pass
            try:
                self._cancel_sim_arm_move("")
            except Exception:
                pass

        def _do_enable(e: bool = enable) -> None:
            msg = self.node.request_arm_enable(enable=e)
            self.status_bar.showMessage(
                ("已打开手臂控制" if e else "已关闭手臂控制") + f"：{msg}",
                5000,
            )
            self._update_enable_status_ui()
            self._update_arm_move_btns_ui()

        # 关使能：本地立即门控（request 内乐观更新），再异步确认桥
        # 开使能：桥已在跑则立即；否则稍等 UDP 监听就绪
        if (not enable) or _eai_adapters_robot_services._pgrep_pattern("host_arm_hand_bridge.py") or _eai_adapters_robot_services._is_stack_services_running():
            _do_enable()
        else:
            self.status_bar.showMessage("正在拉起宿主机使能桥…")
            _eai_platform_bootstrap.QTimer.singleShot(600, _do_enable)

    def _on_hand_enable_clicked(self) -> None:
        self._ensure_host_arm_hand_bridge()
        self._ensure_enable_state_subscriptions()
        enable = not self.node.is_hand_enabled()

        def _do_hand(e: bool = enable) -> None:
            msg = self.node.request_hand_enable(enable=e)
            self.status_bar.showMessage(
                ("已打开手控制" if e else "已关闭手控制") + f"：{msg}",
                5000,
            )
            self._update_enable_status_ui()

        if _eai_adapters_robot_services._pgrep_pattern("host_arm_hand_bridge.py") or _eai_adapters_robot_services._is_stack_services_running():
            _do_hand()
        else:
            self.status_bar.showMessage("正在拉起宿主机使能桥…")
            _eai_platform_bootstrap.QTimer.singleShot(600, _do_hand)

    def _on_arm_enable_ui_changed(self, enabled: bool) -> None:
        self._update_enable_status_ui()
        self._update_prereq_gated_buttons()
        if enabled:
            # 使能后确保 FK/TCP 在更新，否则移动按钮一直灰
            self._ensure_host_fk()
            self._ensure_robot_tcp_bridge()
            _eai_platform_bootstrap.QTimer.singleShot(300, self._update_arm_move_btns_ui)
            _eai_platform_bootstrap.QTimer.singleShot(1200, self._update_arm_move_btns_ui)
            self._try_finish_pending_arm_move()
        else:
            try:
                if self.node.is_slow_motion_busy():
                    self.node.cancel_slow_motion()
            except Exception:
                pass
            self._pending_arm_move_goal = None
        self._update_arm_move_btns_ui()

    def _on_hand_enable_ui_changed(self, _enabled: bool) -> None:
        self._update_enable_status_ui()
        self._update_prereq_gated_buttons()

    def _on_control_mode_ui_changed(self, _mode: int) -> None:
        self._update_enable_status_ui()
        self._update_prereq_gated_buttons()

    def _on_arm_enable_wait_tick(self) -> None:
        if self._pending_arm_move_goal is None:
            self._arm_enable_wait_timer.stop()
            return
        if self.node.is_arm_enabled():
            self._try_finish_pending_arm_move()
            return
        if _eai_platform_bootstrap.time.time() > self._arm_enable_wait_deadline:
            self._pending_arm_move_goal = None
            self._text_cmd_queue.clear()
            self._arm_enable_wait_timer.stop()
            self.status_bar.showMessage(
                "自动启用手臂超时，请确认 topic_router 已运行，或手动按 F2 / 点「启用手臂」"
            )
            self._update_arm_move_btns_ui()
        self._update_arm_move_btns_ui()

    def _on_arm_workspace_tab_changed(self, index: int) -> None:
        holder = getattr(self, "_robot_view_holder", None)
        tabs = getattr(self, "arm_workspace_tabs", None)
        view = getattr(self, "_robot_view", None)
        if holder is None or tabs is None:
            return
        active = _eai_ui_widgets_layouts._tab_page_matches(tabs.widget(index), holder)
        if active and view is None:
            try:
                from eai_app.ui.robot.viewer import RobotView3D

                view = RobotView3D(holder, ros_node=getattr(self, "node", None))
                holder.layout().addWidget(view, 1)
                self._robot_view = view
                view.simulation_hand_callback = self._control_grasp_simulation_hand
                view.simulation_controls_allowed = lambda: not self.grasp_execution_panel._program_active
                view.grasp_action_callback = self._robot_grasp_action
                sim_q, sim_hands = getattr(self, '_robot_grasp_simulation', (None, {}))
                view.set_simulation_state(sim_q, sim_hands)
                view.set_grasp_preview(getattr(self, '_robot_grasp_preview', None))
                view.set_collision_scene(*getattr(self, '_robot_collision_scene', ({}, [])))
                cached = getattr(self, "_robot_method_poses", None) or {}
                if not cached:
                    old = getattr(self, "_robot_pose6d", None)
                    if old is not None:
                        view.set_method_pose("pca", *old)
                for key, pose in cached.items():
                    view.set_method_pose(key, *pose)
                view.set_target_poses(getattr(self, '_robot_target_poses', []))
                sam3_pts = getattr(self, "_robot_sam3_points", None)
                if sam3_pts is not None:
                    view.set_sam3_points(sam3_pts)
                lingbot = getattr(self, "_robot_lingbot_depth", None) or {}
                ply = str(lingbot.get("ply") or "")
                if ply and _eai_platform_bootstrap.os.path.isfile(ply) and hasattr(view, "set_lingbot_depth_from_ply"):
                    try:
                        view.set_lingbot_depth_from_ply(
                            ply,
                            color_topic=str(lingbot.get("color_topic") or ""),
                        )
                    except Exception:
                        pass
            except Exception as exc:
                self.status_bar.showMessage(f"机器人视图加载失败: {exc}", 8000)
                return
        if view is not None:
            view.set_running(active)

    def _on_grasp_profile_calibrated(self, path: str) -> None:
        """Refresh only the local execution profile after explicit calibration."""
        panel = getattr(self, "grasp_execution_panel", None)
        if panel is None:
            return
        panel.profile_path.setText(str(path))
        panel.check_readiness()
        self.status_bar.showMessage("抓取配置已更新；请重新检查准备状态并重新预览目标")

    def _grasp_task_path(self, task_key):
        from eai_app.ui.grasp.workflow import TASKS
        if task_key not in TASKS:
            raise ValueError('未知任务：' + str(task_key))
        return _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, '.cache', 'grasp_tasks', task_key + '.npz')

    def _on_arm_text_command(self) -> None:
        text = self.arm_text_cmd_edit.text().strip()
        if not text:
            self.status_bar.showMessage("请输入指令，例如：左臂向前5cm；右手张开")
            return
        actions, errors = _eai_platform_bootstrap.parse_arm_hand_text(text)
        if not actions:
            hint = "；".join(errors) if errors else "无法识别"
            self.status_bar.showMessage(f"无法识别：{hint}")
            return
        if actions[0].kind == "stop":
            self._text_cmd_queue.clear()
            stop_msg = self._cancel_arm_motion_for_text()
            actions = actions[1:]
        else:
            stop_msg = ""
        self._text_cmd_queue.extend(actions)
        msg = self._pump_text_cmd_queue()
        parts = [p for p in (stop_msg, msg) if p]
        if errors:
            parts.append("未识别：" + "；".join(errors))
        self.status_bar.showMessage("；".join(parts) if parts else "没有可执行的指令")
        try:
            self.node.get_logger().info(f"文本指挥 [{text}] -> {self.status_bar.currentMessage()}")
        except Exception:
            pass

    def _arm_text_motion_busy(self) -> bool:
        return (
            self._is_sim_arm_moving()
            or self.node.is_slow_motion_busy()
            or self._pending_arm_move_goal is not None
        )

    def _hand_vector_for_action(self, side: str, action: _eai_platform_bootstrap.ArmHandTextAction) -> _eai_platform_bootstrap.List[float]:
        if action.joints:
            out = self._current_hand_vector(side)
            index = {name: i for i, name in enumerate(_eai_platform_bootstrap.HAND_JOINT_ORDER)}
            for name, val in action.joints.items():
                if name in index:
                    out[index[name]] = max(0.0, min(1.0, float(val)))
            return out
        pos = 0.0 if action.position is None else float(action.position)
        pos = max(0.0, min(1.0, pos))
        return [pos] * len(_eai_platform_bootstrap.HAND_JOINT_ORDER)

    def _on_arm_repeat_clicked(self) -> None:
        if self._repeat_active:
            self._stop_arm_repeat("已取消重复执行", cancel_motion=True)
            return
        if self._arm_text_motion_busy():
            self.status_bar.showMessage("已有移动进行中，请先停掉再重复执行", 6000)
            return
        sim = bool(self._sim_teleop_active() or self._sim_abs_cmd_ready())
        if not sim and not self.node.is_arm_enabled():
            self.status_bar.showMessage("手臂未使能：请先点「启用手臂」", 6000)
            return
        homes: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]] = {}
        outs: _eai_platform_bootstrap.List[_eai_perception_pose_estimation.ResolvedArmMoveGoal] = []
        for side in ("left", "right"):
            tcp = self._tcp_for_relative_goal(side)
            if tcp is None:
                continue
            homes[side] = (
                (float(tcp[0][0]), float(tcp[0][1]), float(tcp[0][2])),
                (float(tcp[1][0]), float(tcp[1][1]), float(tcp[1][2]), float(tcp[1][3])),
            )
            goal = self._resolve_move_goal(side)
            if goal is None:
                continue
            outs.append(goal)
        if not homes:
            self.status_bar.showMessage("无法重复：没有左右臂当前位置", 8000)
            return
        if not outs:
            self.status_bar.showMessage(
                "无法重复：请先给左臂或右臂设置偏移，或选择分割位姿", 8000
            )
            return
        home_goals = []
        for side in ("left", "right"):
            if side not in homes or not any(goal.arm_side == side for goal in outs):
                continue
            recorded = self._arm_home_pose.get(side)
            pose = recorded if recorded is not None else homes[side]
            home_goals.append(
                _eai_perception_pose_estimation.ResolvedArmMoveGoal(
                    position_xyz=pose[0],
                    quaternion_xyzw=pose[1],
                    label=f"{_eai_perception_pose_estimation.arm_side_label(side)}回初始",
                    arm_side=side,
                )
            )
        home_q = getattr(self, "_arm_home_q16", None)
        if not isinstance(home_q, list) or len(home_q) < 16:
            home_q = self.node._load_host_fk_q16()
        self._repeat_home_q16 = (
            [float(v) for v in home_q[:16]] if home_q and len(home_q) >= 16 else None
        )
        commanded = getattr(self.node, "_commanded_q16", None)
        if not isinstance(commanded, list) or len(commanded) < 16:
            commanded = self.node._load_host_fk_q16()
        self._repeat_commanded_q16 = (
            [float(v) for v in commanded[:16]] if commanded and len(commanded) >= 16 else None
        )
        self._text_cmd_queue.clear()
        self._repeat_out_goals = outs
        self._repeat_home_goals = home_goals
        self._repeat_cycle = 0
        self._repeat_busy_retries = 0
        self._repeat_skip_retries = 0
        self._repeat_last_goal = None
        self._repeat_active = True
        self._update_arm_repeat_btn()
        self._begin_repeat_cycle()

    def _on_arm_move_clicked(self, arm_side: str) -> None:
        if getattr(self, "_repeat_active", False) and not self._arm_text_motion_busy():
            self._stop_arm_repeat("已停止重复执行", cancel_motion=False)
        side_name = _eai_perception_pose_estimation.arm_side_label(arm_side)
        if self._is_sim_arm_moving():
            moving = str((self._sim_arm_move or {}).get("arm_side", ""))
            if moving != arm_side:
                return
            self._text_cmd_queue.clear()
            repeating = bool(self._repeat_active)
            if repeating:
                self._repeat_active = False
                self._repeat_queue = []
                self._repeat_wait_timer.stop()
            msg = self._cancel_sim_arm_move()
            self._update_arm_repeat_btn()
            self.status_bar.showMessage("已取消重复执行" if repeating else msg)
            self._update_arm_move_btns_ui()
            return
        if self.node.is_slow_motion_busy():
            if self.node._slow_motion_moving_side != arm_side:
                return
            self._text_cmd_queue.clear()
            self._pending_arm_move_goal = None
            self._arm_enable_wait_timer.stop()
            msg = self.node.cancel_slow_motion()
            self.status_bar.showMessage(msg)
            self._update_arm_move_btns_ui()
            return
        goal = self._resolve_move_goal(arm_side)
        if goal is None:
            if self.move_target_segment_radio.isChecked():
                if self._last_segment_target is None:
                    self.status_bar.showMessage("请先选点并点「调用分割 / FP」完成位姿估计")
                else:
                    frame = self._last_segment_target.camera_frame or "未知"
                    self.status_bar.showMessage(
                        f"无法将分割位姿变换到 {_eai_configuration_policies.IK_TARGET_FRAME}，"
                        f"请检查 {frame} 的 TF"
                    )
            else:
                self.status_bar.showMessage(
                    f"请设置非零偏移量，或等待{side_name} TCP 数据"
                )
            return
        self._text_cmd_queue.clear()
        if self._sim_teleop_active():
            msg = self._start_sim_arm_move(arm_side, goal)
            self.status_bar.showMessage(msg)
            self._update_arm_move_btns_ui()
            return
        if not self.node.is_arm_enabled():
            self.status_bar.showMessage("手臂未使能：请先点「启用手臂」")
            self._update_arm_move_btns_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

