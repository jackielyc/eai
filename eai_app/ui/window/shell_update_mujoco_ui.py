"""ui.window.shell update mujoco ui — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

class ShellUpdateMujocoUiMixin:
    def _update_mujoco_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_mujoco_launcher", None)
            and self._mujoco_launcher.is_running()
        )
        backend = self._sim_backend_id()
        # Keep workspace LIBERO/Live-HUD preview + /camera/* npy poll while eval runs.
        if hasattr(self, "_set_rynnvalue_workspace_hud_active"):
            timer = getattr(self, "_rynnvalue_workspace_hud_timer", None)
            if running and backend == "libero":
                if timer is not None and not timer.isActive():
                    self._set_rynnvalue_workspace_hud_active(True)
                self._activate_sim_camera_preview(force=False)
            elif running:
                # MuJoCo / Molmo / Arena / RoboTwin：若写出 cam_*.npy 则同样轮询
                if getattr(self, "_sim_npy_preview_timer", None) is not None:
                    if not self._sim_npy_preview_timer.isActive():
                        self._sim_npy_preview_timer.start()
            elif (
                not running
                and not self._sim_rynnvalue_hud_enabled()
                and timer is not None
                and timer.isActive()
            ):
                self._set_rynnvalue_workspace_hud_active(False)
        startable = (not running) and (
            backend
            in (
                "mujoco",
                "molmospaces",
                "psi_molmospaces",
                "arena",
                "libero",
                "robotwin",
            )
            or self._is_extended_sim_backend(backend)
        )
        if hasattr(self, "mujoco_start_btn"):
            self.mujoco_start_btn.setEnabled(startable)
        if hasattr(self, "mujoco_viewer_only_btn"):
            self.mujoco_viewer_only_btn.setEnabled(startable)
        if hasattr(self, "mujoco_stop_btn"):
            self.mujoco_stop_btn.setEnabled(running)
        self._update_sim_pause_ui()
        if hasattr(self, "mujoco_install_btn"):
            self.mujoco_install_btn.setEnabled(
                (not running) and backend == "mujoco"
            )
        for wname in (
            "mujoco_root_edit",
            "mujoco_python_edit",
            "mujoco_model_combo",
            "mujoco_mode_combo",
            "mujoco_refresh_btn",
            "mujoco_mjcf_browse_btn",
            "mujoco_root_browse_btn",
            "molmospaces_root_edit",
            "molmospaces_root_browse_btn",
            "molmospaces_recipe_combo",
            "molmospaces_robot_combo",
            "molmospaces_task_combo",
            "molmospaces_task_refresh_btn",
            "molmospaces_policy_combo",
            "molmospaces_episodes_spin",
            "molmospaces_pi_ckpt_edit",
            "molmospaces_pi_ckpt_browse_btn",
            "molmospaces_pi_start_btn",
            "molmospaces_pi_stop_btn",
            "molmospaces_pi_probe_btn",
            "arena_root_edit",
            "arena_root_browse_btn",
            "arena_task_combo",
            "arena_task_refresh_btn",
            "arena_policy_combo",
            "arena_policy_file_edit",
            "arena_policy_file_browse_btn",
            "arena_steps_spin",
            "arena_headless_check",
            "arena_gpu_check",
            "arena_gpu_spin",
            "arena_extra_edit",
            "libero_rlinf_edit",
            "libero_rlinf_browse_btn",
            "libero_suite_combo",
            "libero_task_id_spin",
            "libero_config_combo",
            "libero_mode_combo",
            "libero_ckpt_edit",
            "libero_ckpt_browse_btn",
            "libero_pi_start_btn",
            "libero_pi_probe_btn",
            "libero_max_tasks_spin",
            "libero_trials_spin",
            "libero_chunk_spin",
            "libero_steps_spin",
            "libero_render_gui_check",
            "robotwin_rlinf_edit",
            "robotwin_rlinf_browse_btn",
            "robotwin_root_edit",
            "robotwin_root_browse_btn",
            "robotwin_assets_edit",
            "robotwin_assets_browse_btn",
            "robotwin_task_combo",
            "robotwin_config_combo",
            "robotwin_gpu_spin",
            "robotwin_ckpt_edit",
            "robotwin_ckpt_browse_btn",
            "robotwin_num_envs_spin",
            "robotwin_max_steps_spin",
            "robotwin_render_gui_check",
            "mujoco_open_model_btn",
            "ext_bench_root_edit",
            "ext_bench_root_browse_btn",
            "ext_bench_python_edit",
            "ext_bench_mode_combo",
            "ext_bench_task_edit",
            "ext_bench_episodes_spin",
            "ext_bench_viewer_check",
            "ext_bench_probe_btn",
            "ext_bench_hint_btn",
            "ext_bench_docs_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)
        self._update_sim_backend_mutex_ui()

    def _resolve_mujoco_mjcf_path(self) -> str:
        raw = (
            str(self.mujoco_model_combo.currentData() or "").strip()
            or self.mujoco_model_combo.currentText().strip()
        )
        if not raw:
            return ""
        # Combo may show a short label; prefer stored userData path.
        if _eai_platform_bootstrap.os.path.isabs(raw) or _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.expanduser(raw)):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))
        # Relative to eai repo, or an absolute path (e.g. PSI_R1_MODEL_XML)
        eai_cand = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, raw)
        if _eai_platform_bootstrap.os.path.isfile(eai_cand):
            return _eai_platform_bootstrap.os.path.abspath(eai_cand)
        root = _eai_adapters_models_discovery.resolve_mujoco_root(self.mujoco_root_edit.text())
        cand = _eai_platform_bootstrap.os.path.join(root, "model", raw)
        if _eai_platform_bootstrap.os.path.isfile(cand):
            return cand
        cand2 = _eai_platform_bootstrap.os.path.join(root, raw)
        if _eai_platform_bootstrap.os.path.isfile(cand2):
            return cand2
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))

    def _refresh_mujoco_models(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "mujoco_model_combo"):
            return
        root = self.mujoco_root_edit.text().strip()
        names = _eai_adapters_models_discovery.list_mujoco_model_xmls(root)
        prefer_abs = (
            _eai_platform_bootstrap.os.path.abspath(prefer)
            if prefer and (_eai_platform_bootstrap.os.path.isabs(prefer) or _eai_platform_bootstrap.os.path.isfile(prefer))
            else prefer
        )
        current = prefer_abs or self.mujoco_model_combo.currentData() or ""
        if not current:
            current = self.mujoco_model_combo.currentText().strip()
        if not current:
            for p in _eai_configuration_tabs.MUJOCO_MODEL_PRESETS:
                p_abs = _eai_platform_bootstrap.os.path.abspath(p) if _eai_platform_bootstrap.os.path.isabs(p) else p
                if p_abs in names or p in names:
                    current = p_abs if p_abs in names else p
                    break
            if not current and names:
                current = names[0]
        self.mujoco_model_combo.blockSignals(True)
        self.mujoco_model_combo.clear()
        for name in names:
            self.mujoco_model_combo.addItem(self._mujoco_model_label(name), name)
        if current:
            idx = self.mujoco_model_combo.findData(current)
            if idx < 0:
                idx = self.mujoco_model_combo.findData(_eai_platform_bootstrap.os.path.abspath(current))
            if idx < 0:
                idx = self.mujoco_model_combo.findText(current)
            if idx >= 0:
                self.mujoco_model_combo.setCurrentIndex(idx)
            else:
                self.mujoco_model_combo.setEditText(str(current))
        self.mujoco_model_combo.blockSignals(False)
        n_local = sum(1 for n in names if _eai_platform_bootstrap.os.path.isabs(n))
        self._append_mujoco_log(
            f"已扫描 {len(names)} 个模型（eai 本地 {n_local} + "
            f"{_eai_adapters_models_discovery.resolve_mujoco_root(root)}/model）"
        )

    def _resolve_arena_root(self) -> str:
        raw = ""
        if hasattr(self, "arena_root_edit"):
            raw = self.arena_root_edit.text().strip()
        raw = raw or _eai_platform_bootstrap.os.environ.get("ISAACLAB_ARENA_ROOT", "") or _eai_configuration_tabs.ISAACLAB_ARENA_ROOT_DEFAULT
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))

    def _refresh_arena_tasks(self) -> None:
        if not hasattr(self, "arena_task_combo"):
            return
        root = self._resolve_arena_root()
        current = str(self.arena_task_combo.currentData() or "")
        names = _eai_adapters_models_discovery.list_isaaclab_arena_environments(root)
        self.arena_task_combo.blockSignals(True)
        self.arena_task_combo.clear()
        for name in names:
            self.arena_task_combo.addItem(name, name)
        if current:
            idx = self.arena_task_combo.findData(current)
            if idx >= 0:
                self.arena_task_combo.setCurrentIndex(idx)
        self.arena_task_combo.blockSignals(False)
        _eai_configuration_models.stamp_eval_task_combo(self.arena_task_combo)
        if hasattr(self, "arena_task_thumb"):
            self.arena_task_thumb.show_combo_current(self.arena_task_combo)
        self._append_mujoco_log(f"[arena] 已列出 {len(names)} 个评测任务")

    def _ensure_live_hud_reward_or_abort(self, tag: str) -> bool:
        """If Live HUD checked, ensure reward_server; return False to abort start."""
        if not self._sim_rynnvalue_hud_enabled():
            return True
        url = self._sim_rynnvalue_server_url()
        line = f"[{tag}] 确认 RynnValue reward_server（{url}）…"
        if hasattr(self, "_append_mujoco_log"):
            self._append_mujoco_log(line)
        if tag == "isaac" and hasattr(self, "_append_sim_log"):
            self._append_sim_log(line)
        ok, msg = self._ensure_rynnvalue_reward_server_ready(
            start_if_needed=True, wait_s=900.0
        )
        if not ok:
            err = f"[{tag}] RynnValue 预检失败: {msg}"
            if hasattr(self, "_append_mujoco_log"):
                self._append_mujoco_log(err)
            if tag == "isaac" and hasattr(self, "_append_sim_log"):
                self._append_sim_log(err)
            if hasattr(self, "_on_mujoco_status"):
                self._on_mujoco_status(f"RynnValue 未就绪: {msg}")
            return False
        ok_line = f"[{tag}] RynnValue 预检通过: {msg}"
        if hasattr(self, "_append_mujoco_log"):
            self._append_mujoco_log(ok_line)
        if tag == "isaac" and hasattr(self, "_append_sim_log"):
            self._append_sim_log(ok_line)
        return True

    def _start_isaac_rynnvalue_npy_hud(self, bridge_dir: str) -> None:
        """Poll Isaac cam_head.npy into Live HUD (GUI-side sidecar).

        OpenCV HighGUI is headless in ros-humble, so the popup is a Qt window.
        """
        self._stop_isaac_rynnvalue_npy_hud()
        if not self._sim_rynnvalue_hud_enabled():
            return
        settings = self._sim_rynnvalue_hud_settings()
        try:
            tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
            if tools not in _eai_platform_bootstrap.sys.path:
                _eai_platform_bootstrap.sys.path.insert(0, tools)
            from rynnvalue_live_hud import RynnValueLiveHud
            from rynnvalue_sim_bridge import get_hud_profile
        except Exception as exc:
            self._append_sim_log(f"[rynnvalue] Isaac HUD import failed: {exc}")
            return
        prof = get_hud_profile("isaac")
        status = _eai_platform_bootstrap.os.path.join(
            _eai_configuration_models.EAI_DIR, ".cache", "rynnvalue_live", "isaac_rynnvalue_live.json"
        )
        try:
            _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(status), exist_ok=True)
        except Exception:
            pass
        # Scoring only in-process; Qt owns the visible window (not OpenCV).
        hud = RynnValueLiveHud(
            server_url=str(settings["server_url"]),
            refresh_sec=float(settings["refresh_sec"]),
            num_frames=int(settings["num_frames"]),
            robot_description=prof.robot_description,
            camera_description=prof.camera_description,
            show_window=False,
            status_path=status,
            flip_ud=False,
        )
        hud.start()
        self._isaac_rynnvalue_hud = hud
        self._isaac_rynnvalue_bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(bridge_dir or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT)
        )
        self._isaac_rynnvalue_last_mtime = 0.0
        qt_hud = _eai_ui_widgets_video.RynnValueQtHudWindow(self)
        qt_hud.closed_by_user.connect(self._on_isaac_rynnvalue_qt_hud_closed)
        qt_hud.show()
        qt_hud.raise_()
        qt_hud.activateWindow()
        self._isaac_rynnvalue_qt_hud = qt_hud
        if not hasattr(self, "_isaac_rynnvalue_timer"):
            self._isaac_rynnvalue_timer = _eai_platform_bootstrap.QTimer(self)
            self._isaac_rynnvalue_timer.setInterval(120)
            self._isaac_rynnvalue_timer.timeout.connect(self._poll_isaac_rynnvalue_npy)
        self._isaac_rynnvalue_timer.start()
        self._append_sim_log(
            f"[rynnvalue] Isaac Live HUD Qt 窗口已弹出；轮询 "
            f"{self._isaac_rynnvalue_bridge_dir}/cam_head.npy"
            "（ros-humble OpenCV 无 HighGUI，改用 Qt）"
        )

    def _stop_isaac_rynnvalue_npy_hud(self) -> None:
        timer = getattr(self, "_isaac_rynnvalue_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        qt_hud = getattr(self, "_isaac_rynnvalue_qt_hud", None)
        self._isaac_rynnvalue_qt_hud = None
        if qt_hud is not None:
            try:
                qt_hud.closed_by_user.disconnect(
                    self._on_isaac_rynnvalue_qt_hud_closed
                )
            except Exception:
                pass
            try:
                qt_hud.close()
            except Exception:
                pass
        hud = getattr(self, "_isaac_rynnvalue_hud", None)
        if hud is not None:
            try:
                hud.stop()
            except Exception:
                pass
        self._isaac_rynnvalue_hud = None

    def _poll_isaac_rynnvalue_npy(self) -> None:
        hud = getattr(self, "_isaac_rynnvalue_hud", None)
        if hud is None:
            return
        bridge = getattr(self, "_isaac_rynnvalue_bridge_dir", "") or ""
        npy = _eai_platform_bootstrap.os.path.join(bridge, "cam_head.npy")
        if _eai_platform_bootstrap.os.path.isfile(npy):
            try:
                mtime = _eai_platform_bootstrap.os.path.getmtime(npy)
                if mtime > float(
                    getattr(self, "_isaac_rynnvalue_last_mtime", 0.0) or 0.0
                ):
                    import numpy as _np

                    arr = _np.load(npy)
                    self._isaac_rynnvalue_last_mtime = mtime
                    hud.push_frame(arr)
            except Exception:
                pass
        qt_hud = getattr(self, "_isaac_rynnvalue_qt_hud", None)
        if qt_hud is None or getattr(qt_hud, "_user_closed", False):
            return
        try:
            rgb = hud.last_overlay_rgb()
            if rgb is not None:
                qt_hud.set_overlay_rgb(rgb)
            snap = hud.snapshot()
            qt_hud.set_status_title(
                snap.remaining_s if snap.remaining_s is not None else None,
                error=str(snap.error or ""),
            )
        except Exception:
            pass

    def _expand_workspace_panel(self) -> None:
        """Ensure下方工作区 is visible (Topic / Live HUD 控件 / 图像预览)."""
        if getattr(self, "_workspace_collapsed", False):
            self._toggle_workspace_panel()

    def _set_rynnvalue_workspace_hud_active(self, active: bool) -> None:
        """Ensure LIBERO + Live HUD panels exist in 工作区「图像预览」."""
        if not hasattr(self, "topic_list_layout") or not hasattr(self, "grid_layout"):
            return
        topics = self._workspace_preview_topics()
        timer = getattr(self, "_rynnvalue_workspace_hud_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()
        if not active:
            # Keep last frames / timer; a new JPEG will refresh the panels.
            return

        self._expand_workspace_panel()
        types: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.List[str]] = {}
        for k, v in (getattr(self, "_topic_types", {}) or {}).items():
            if isinstance(v, (list, tuple)):
                types[k] = list(v)
            elif isinstance(v, str):
                types[k] = [v]
            else:
                types[k] = ["sensor_msgs/msg/Image"]
        for topic in topics:
            types[topic] = ["sensor_msgs/msg/Image"]
        self._on_topics_updated(types)
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        for topic in topics:
            checkbox = self.topic_checks.get(topic)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
            self._topic_user_unchecked.discard(topic)
            enabled.add(topic)
        # File-backed previews: show panels but do not ROS-subscribe.
        if hasattr(self, "node") and hasattr(self.node, "set_enabled_topics"):
            ros_enabled = {
                t
                for t in enabled
                if t not in (_eai_configuration_policies.LIBERO_PREVIEW_TOPIC, _eai_configuration_policies.RYNNVALUE_LIVE_HUD_TOPIC)
            }
            self.node.set_enabled_topics(ros_enabled)
        self._rebuild_panels(enabled)
        self._poll_rynnvalue_workspace_hud()

    def _poll_rynnvalue_workspace_hud(self) -> None:
        """Pull newest LIBERO preview / Live HUD overlay into the preview panel."""
        if not hasattr(self, "topic_list_layout") or not hasattr(self, "grid_layout"):
            return
        # Prefer Live HUD overlay when it is the newest file; else raw preview.
        best_path = ""
        best_mtime = 0.0
        for path in self._rynnvalue_workspace_overlay_candidates():
            try:
                if not _eai_platform_bootstrap.os.path.isfile(path):
                    continue
                mtime = float(_eai_platform_bootstrap.os.path.getmtime(path))
            except OSError:
                continue
            if mtime >= best_mtime:
                best_mtime = mtime
                best_path = path
        if not best_path:
            return
        # Auto-expose pseudo-topics as soon as a preview JPEG exists.
        checks = getattr(self, "topic_checks", {}) or {}
        panels = getattr(self, "panels", {}) or {}
        need_topics = [
            t
            for t in self._workspace_preview_topics()
            if t not in checks or t not in panels
        ]
        if need_topics and not getattr(self, "_rynnvalue_workspace_hud_activating", False):
            self._rynnvalue_workspace_hud_activating = True
            try:
                self._set_rynnvalue_workspace_hud_active(True)
                self._activate_sim_camera_preview(force=False)
            finally:
                self._rynnvalue_workspace_hud_activating = False
            # Activation itself calls poll; avoid double-read in this tick.
            return
        prev = float(getattr(self, "_rynnvalue_workspace_hud_mtime", 0.0) or 0.0)
        if best_mtime <= prev:
            return
        try:
            import cv2 as _cv2

            img = _cv2.imread(best_path, _cv2.IMREAD_COLOR)
        except Exception:
            return
        if img is None or getattr(img, "size", 0) == 0:
            return
        self._rynnvalue_workspace_hud_mtime = best_mtime
        # Route into both pseudo-topics so either panel name works.
        updated: _eai_platform_bootstrap.List[str] = []
        for topic in self._workspace_preview_topics():
            self._frame_cache[topic] = img
            panel = self.panels.get(topic) if hasattr(self, "panels") else None
            if panel is None:
                enabled = {
                    t
                    for t, cb in getattr(self, "topic_checks", {}).items()
                    if cb.isChecked()
                }
                enabled.add(topic)
                self._apply_selection(enabled)
                panel = self.panels.get(topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                panel.update_frame(img)
            updated.append(topic)
        # 无 cam_head.npy 时，把 HUD/LIBERO JPEG 镜像到头部彩色，供多视角 6D 预览
        head_topic = "/camera/head_color"
        head_mtime = float(self._sim_npy_last_mtime.get(head_topic, 0.0) or 0.0)
        if best_mtime >= head_mtime:
            self._frame_cache[head_topic] = img
            self._sim_npy_last_mtime[head_topic] = best_mtime
            self._received_topics[head_topic] = (
                self._received_topics.get(head_topic, 0) + 1
            )
            if head_topic not in getattr(self, "topic_checks", {}):
                self._on_topics_updated(dict(getattr(self, "_topic_types", {}) or {}))
            head_panel = self.panels.get(head_topic) if hasattr(self, "panels") else None
            if head_panel is None:
                enabled = {
                    t
                    for t, cb in getattr(self, "topic_checks", {}).items()
                    if cb.isChecked()
                }
                enabled.add(head_topic)
                self._apply_selection(enabled)
                head_panel = self.panels.get(head_topic)
            if isinstance(head_panel, _eai_ui_cameras_panels.CameraPanel):
                head_panel.update_frame(img)
            updated.append(head_topic)
            if not self._sim_npy_preview_timer.isActive():
                self._sim_npy_preview_timer.start()
        self._maybe_refresh_mv6d_previews_for_topics(updated)
        try:
            if best_path.endswith("_overlay.jpg"):
                status_guess = best_path.replace("_overlay.jpg", ".json")
                if _eai_platform_bootstrap.os.path.isfile(status_guess):
                    with open(status_guess, "r", encoding="utf-8") as fh:
                        payload = _eai_platform_bootstrap.json.load(fh)
                    rem = payload.get("remaining_s")
                    if rem is not None and hasattr(self, "_set_sim_rynnvalue_status"):
                        self._set_sim_rynnvalue_status(
                            f"Live HUD rem={float(rem):.2f}s "
                            f"step={payload.get('step', '?')}",
                            level="ok",
                        )
        except Exception:
            pass

    def _update_sim_rynnvalue_hud_controls(self) -> None:
        enabled = self._sim_rynnvalue_hud_enabled()
        launcher_running, listening, healthy = self._rynnvalue_reward_runtime()
        deploy_locked = bool(launcher_running or listening or healthy)
        for wname in (
            "sim_rynnvalue_refresh_spin",
            "sim_rynnvalue_url_edit",
            "sim_rynnvalue_frames_spin",
            "sim_rynnvalue_probe_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(enabled)
        deploy_btn = getattr(self, "sim_rynnvalue_deploy_btn", None)
        if deploy_btn is not None:
            # Deploy does not require Live HUD; only lock while loading/running.
            deploy_btn.setEnabled(not deploy_locked)
            if deploy_locked:
                deploy_btn.setToolTip(
                    "正在部署或模型已在运行，不可重复点击；请先「停止模型」"
                )
            else:
                deploy_btn.setToolTip(
                    "启动 RynnValue reward_server（模型路径沿用「RynnValue」页配置）。"
                    "与 Pi 同卡时请先「停止 Pi」。"
                )
        if hasattr(self, "sim_rynnvalue_stop_btn"):
            self.sim_rynnvalue_stop_btn.setEnabled(True)
        timer = getattr(self, "_sim_rynnvalue_status_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _update_libero_rynnvalue_hud_controls(self) -> None:
        self._update_sim_rynnvalue_hud_controls()

    def _set_sim_rynnvalue_status(self, text: str, *, level: str = "muted") -> None:
        if not hasattr(self, "sim_rynnvalue_status_label"):
            return
        color = {
            "ok": "#3dd68c",
            "warn": "#e6a23c",
            "err": "#f56c6c",
            "muted": _eai_configuration_policies.UI_TEXT_MUTED,
        }.get(level, _eai_configuration_policies.UI_TEXT_MUTED)
        self.sim_rynnvalue_status_label.setText(text)
        self.sim_rynnvalue_status_label.setStyleSheet(f"color: {color};")

    def _set_libero_rynnvalue_status(self, text: str, *, level: str = "muted") -> None:
        self._set_sim_rynnvalue_status(text, level=level)

    def _probe_and_show_rynnvalue_reward_status(
        self, *, log: bool = False, prefix: str = ""
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        # ``prefix`` kept for call-site compatibility; shared panel ignores it.
        _ = prefix
        url = self._sim_rynnvalue_server_url()
        launcher_running = bool(
            getattr(self, "_rynnvalue_launcher", None)
            and self._rynnvalue_launcher.is_running()
        )
        ok, msg, info = _eai_adapters_models_discovery.probe_rynnvalue_reward_server(url)
        self._sim_rynnvalue_last_ok = bool(ok)
        deploy_btn = getattr(self, "sim_rynnvalue_deploy_btn", None)
        if ok:
            self._set_sim_rynnvalue_status(f"● {msg}", level="ok")
            if deploy_btn is not None:
                deploy_btn.setText("已部署")
        elif info.get("listening") or launcher_running:
            detail = msg
            if launcher_running and not info.get("listening"):
                detail = "RynnValue 进程已启动，等待 /health…"
            self._set_sim_rynnvalue_status(f"◐ {detail}", level="warn")
            if deploy_btn is not None:
                deploy_btn.setText("部署中…")
        else:
            self._set_sim_rynnvalue_status(f"○ {msg}", level="err")
            if deploy_btn is not None:
                deploy_btn.setText("部署 RynnValue")
        self._update_sim_rynnvalue_hud_controls()
        if log:
            prev = getattr(self, "_sim_rynnvalue_last_logged_msg", "")
            if msg != prev:
                self._sim_rynnvalue_last_logged_msg = msg
                self._append_mujoco_log(f"[rynnvalue] {msg}")
        return ok, msg

    def _stop_rynnvalue_model(self, *, log: bool = True) -> str:
        """Stop launcher + reward_server port so the GPU model is unloaded."""
        if hasattr(self, "_rynnvalue_launcher") and self._rynnvalue_launcher.is_running():
            if log:
                self._append_mujoco_log("[rynnvalue] 停止模型进程…")
            self._rynnvalue_launcher.stop()
        url = self._sim_rynnvalue_server_url()
        _host, port = _eai_adapters_models_discovery.parse_rynnvalue_server_url(url)
        msg = _eai_adapters_models_discovery.stop_rynnvalue_server_on_port(
            port, log=self._append_mujoco_log if log else None
        )
        if log:
            self._append_mujoco_log(f"[rynnvalue] {msg}")
        self._update_rynnvalue_ui()
        self._update_sim_rynnvalue_hud_controls()
        self._probe_and_show_rynnvalue_reward_status(log=False)
        return msg

    def _cuda_mem_mib(self, index: int) -> _eai_platform_bootstrap.Tuple[int, int, int]:
        """Return (used, free, total) MiB for GPU ``index``; zeros on failure."""
        try:
            raw = _eai_platform_bootstrap.subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2.0,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
            )
        except Exception:
            return 0, 0, 0
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[0].isdigit() and int(parts[0]) == int(index):
                try:
                    return int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                except ValueError:
                    return 0, 0, 0
        return 0, 0, 0

    def _start_rynnvalue_reward_server(self) -> _eai_platform_bootstrap.Tuple[bool, str]:
        if not hasattr(self, "_rynnvalue_launcher"):
            return False, "RynnValue launcher 未初始化"
        url = self._sim_rynnvalue_server_url()
        # Prefer an already-healthy deploy (UI-owned or external) — do not reload GPU.
        ok_ext, msg_ext, _info_ext = _eai_adapters_models_discovery.probe_rynnvalue_reward_server(url, timeout_s=2.5)
        if ok_ext:
            live = ""
            model_info = _info_ext.get("model_info") if isinstance(_info_ext, dict) else None
            if isinstance(model_info, dict):
                live = str(model_info.get("reward_model") or "")
            health = _info_ext.get("health") if isinstance(_info_ext, dict) else None
            if not live and isinstance(health, dict):
                live = str(health.get("reward_model") or "")
            want = (self._rynnvalue_deploy_model_path() or self._rynnvalue_deploy_ckpt_path() or "").strip()
            same = _eai_adapters_models_qwen_local.model_identity_same("", live, "", want) if (live and want) else True
            if same is False:
                msg_diff = f"已在运行其它模型（{live}）。请先「停止模型」再部署。"
                self._append_mujoco_log(f"[rynnvalue] {msg_diff}")
                self._set_sim_rynnvalue_status(f"● {msg_ext}", level="ok")
                return False, msg_diff
            self._append_mujoco_log(f"[rynnvalue] 同一模型已部署，跳过重复部署: {msg_ext}")
            self._set_sim_rynnvalue_status(f"● {msg_ext}", level="ok")
            self._update_sim_rynnvalue_hud_controls()
            return True, msg_ext
        if self._rynnvalue_launcher.is_running():
            ok, msg, _info = _eai_adapters_models_discovery.probe_rynnvalue_reward_server(url)
            if ok:
                return True, msg
            return (
                False,
                "RynnValue 进程已在运行，但 /health 未就绪。"
                "若正在跑 infer/评测，请先在「RynnValue」页停止再部署 reward_server。",
            )
        _host, port = _eai_adapters_models_discovery.parse_rynnvalue_server_url(url)
        if _eai_configuration_models.port_listening(_host, int(port)):
            return (
                False,
                f"{_host}:{port} 已占用但 /health 未就绪（可能仍在加载，或非 RynnValue）。"
                "请先「停止模型」再部署，或等待就绪。",
            )
        model = self._rynnvalue_deploy_model_path()
        ckpt = self._rynnvalue_deploy_ckpt_path()
        if not model and not ckpt:
            return False, "未找到 RynnValue 模型，请先在「RynnValue」页填写模型路径"
        if model and not _eai_adapters_models_discovery._rynnvalue_hf_model_ready(model) and not ckpt:
            return False, f"模型目录不可用: {model}"

        num_frames = 8
        if hasattr(self, "sim_rynnvalue_frames_spin"):
            num_frames = int(self.sim_rynnvalue_frames_spin.value())
        cuda = "0"
        if hasattr(self, "rynnvalue_cuda_edit"):
            cuda = self.rynnvalue_cuda_edit.text().strip() or "0"
        repo = ""
        if hasattr(self, "rynnvalue_root_edit"):
            repo = self.rynnvalue_root_edit.text().strip()
        py = ""
        if hasattr(self, "rynnvalue_python_edit"):
            py = self.rynnvalue_python_edit.text().strip()
        py = py or _eai_adapters_models_discovery.resolve_rynnvalue_python(repo)
        ok_gpu, gpu_msg = self._rynnvalue_gpu_preflight(cuda=cuda, model=model)
        if not ok_gpu:
            return False, gpu_msg
        # Fail fast if server deps missing (otherwise process exits immediately → Connection refused).
        # Do NOT inherit GUI/ros-humble PYTHONPATH — rynnvalue is 3.10 and would
        # otherwise import ros-humble's 3.11 numpy and crash.
        try:
            import subprocess

            robometer_root = _eai_platform_bootstrap.os.path.join(_eai_adapters_models_discovery.resolve_rynnvalue_root(repo), "robometer")
            chk_env = {
                k: v
                for k, v in _eai_platform_bootstrap.os.environ.items()
                if k
                not in (
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "AMENT_PREFIX_PATH",
                    "ROS_DISTRO",
                    "ROS_VERSION",
                    "COLCON_PREFIX_PATH",
                )
            }
            chk_env["PYTHONPATH"] = robometer_root
            chk_env["PYTHONNOUSERSITE"] = "1"
            chk_env["PYTHONUNBUFFERED"] = "1"
            chk = subprocess.run(
                [
                    py,
                    "-c",
                    "import fastapi, uvicorn, omegaconf, hydra, loguru; "
                    "import robometer.evals.baseline_eval_server",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=robometer_root,
                env=chk_env,
            )
            if chk.returncode != 0:
                detail = (chk.stderr or chk.stdout or "").strip()[:400]
                return (
                    False,
                    f"reward_server 依赖未就绪 ({py}). "
                    f"缺少 fastapi/uvicorn/omegaconf 或 robometer 导入失败: {detail}",
                )
        except Exception as exc:  # noqa: BLE001
            return False, f"无法探测 RynnValue Python 依赖: {exc}"

        # Keep RynnValue tab port in sync with shared Live HUD URL.
        if hasattr(self, "rynnvalue_port_spin"):
            self.rynnvalue_port_spin.setValue(int(port))
        if hasattr(self, "rynnvalue_job_combo"):
            for i in range(self.rynnvalue_job_combo.count()):
                if str(self.rynnvalue_job_combo.itemData(i) or "") == "reward_server":
                    self.rynnvalue_job_combo.setCurrentIndex(i)
                    break

        self._append_mujoco_log(
            f"[rynnvalue] 部署 reward_server model={model or '(ckpt)'} "
            f"ckpt={ckpt or '-'} port={port} num_frames={num_frames} cuda={cuda}"
        )
        self._rynnvalue_launcher.start(
            job="reward_server",
            repo_root=repo or _eai_adapters_models_discovery.resolve_rynnvalue_root(),
            python_bin=py or _eai_adapters_models_discovery.resolve_rynnvalue_python(repo),
            model_path=model,
            checkpoint_path=ckpt,
            num_frames=num_frames,
            port=int(port),
            cuda_devices=cuda,
        )
        if not self._rynnvalue_launcher.is_running():
            return False, "启动 reward_server 失败（见 RynnValue 日志）"
        self._set_sim_rynnvalue_status("◐ 正在加载 RynnValue 模型…", level="warn")
        if hasattr(self, "sim_rynnvalue_deploy_btn"):
            self.sim_rynnvalue_deploy_btn.setText("部署中…")
        self._update_sim_rynnvalue_hud_controls()
        return True, "已启动 reward_server，等待健康检查"

    def _ensure_rynnvalue_reward_server_ready(
        self,
        *,
        start_if_needed: bool = True,
        wait_s: float = 600.0,
        prefix: str = "",
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        _ = prefix  # compatibility with older call sites
        ok, msg = self._probe_and_show_rynnvalue_reward_status(log=True)
        if ok:
            return True, msg

        if not start_if_needed:
            return False, msg

        started_ok, start_msg = self._start_rynnvalue_reward_server()
        if not started_ok:
            self._probe_and_show_rynnvalue_reward_status(log=True)
            return False, start_msg
        self._append_mujoco_log(f"[rynnvalue] {start_msg}")

        deadline = _eai_platform_bootstrap.time.time() + max(30.0, float(wait_s))
        last = start_msg
        t0 = _eai_platform_bootstrap.time.time()
        while _eai_platform_bootstrap.time.time() < deadline:
            _eai_platform_bootstrap.QApplication.processEvents()
            ok, last = self._probe_and_show_rynnvalue_reward_status(log=False)
            if ok:
                self._append_mujoco_log(f"[rynnvalue] {last}")
                return True, last
            launcher = getattr(self, "_rynnvalue_launcher", None)
            if launcher is not None and not launcher.is_running():
                ok, last = self._probe_and_show_rynnvalue_reward_status(log=True)
                if ok:
                    return True, last
                return False, f"reward_server 进程已退出: {last}"
            if _eai_platform_bootstrap.time.time() - t0 >= 45.0:
                cuda = "0"
                if hasattr(self, "rynnvalue_cuda_edit"):
                    cuda = self.rynnvalue_cuda_edit.text().strip() or "0"
                idx = (
                    int(cuda.split(",")[0].strip() or "0")
                    if str(cuda).split(",")[0].strip().isdigit()
                    else 0
                )
                _used, free, _total = self._cuda_mem_mib(idx)
                model = ""
                if hasattr(self, "_rynnvalue_deploy_model_path"):
                    try:
                        model = self._rynnvalue_deploy_model_path()
                    except Exception:
                        model = ""
                need = 28000 if "8B" in (model or "").upper() else 16000
                # Pi already on the card leaves too little headroom for load/cuBLAS.
                if free and free < need:
                    return (
                        False,
                        f"reward_server 45s 仍未就绪，GPU{idx} 剩余 {free}MiB"
                        f"（大约需要 ≥{need}MiB）。"
                        "请先在 LIBERO 点「停止 Pi」释放显存，再部署 RynnValue。",
                    )
            _eai_platform_bootstrap.time.sleep(2.0)
        return False, f"等待 RynnValue reward_server 超时: {last}"

