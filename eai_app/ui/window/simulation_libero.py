"""ui.window.simulation libero — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationLiberoMixin:
    def _libero_ckpt_text(self) -> str:
        if hasattr(self, "libero_ckpt_edit"):
            return self.libero_ckpt_edit.text().strip()
        return _eai_configuration_tabs.LIBERO_CKPT_DEFAULT

    def _libero_config_id(self) -> str:
        if hasattr(self, "libero_config_combo"):
            return str(self.libero_config_combo.currentData() or _eai_configuration_tabs.LIBERO_CONFIG_DEFAULT)
        return _eai_configuration_tabs.LIBERO_CONFIG_DEFAULT

    def _libero_mode_id(self) -> str:
        if hasattr(self, "libero_mode_combo"):
            return str(self.libero_mode_combo.currentData() or "local")
        return "local"

    def _on_libero_suite_changed(self, *_args) -> None:
        suite = "libero_spatial"
        if hasattr(self, "libero_suite_combo"):
            suite = str(self.libero_suite_combo.currentData() or suite)
        try:
            from eai_app.domain.demonstrations.matching import list_suite_tasks

            tasks = list_suite_tasks(suite)
        except Exception:
            tasks = []
        spin = getattr(self, "libero_task_id_spin", None)
        if spin is None:
            return
        n = max(1, len(tasks)) if tasks else 100
        cur = int(spin.value())
        spin.setRange(0, max(0, n - 1))
        spin.setValue(min(cur, max(0, n - 1)))
        tip = "suite 内任务下标"
        if tasks:
            lines = [f"[{i}] {lang}" for i, _name, lang in tasks[:12]]
            if len(tasks) > 12:
                lines.append(f"… 共 {len(tasks)} 个")
            tip = "\n".join(lines)
            idx = int(spin.value())
            if 0 <= idx < len(tasks):
                tip = f"当前: [{idx}] {tasks[idx][2]}\n\n" + tip
        spin.setToolTip(tip)
        combo = getattr(self, "libero_task_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.clear()
            if tasks:
                for i, _name, lang in tasks:
                    combo.addItem(f"[{i}] {lang}", i)
            else:
                cur = int(spin.value()) if spin is not None else 0
                combo.addItem(f"{suite} 任务 {cur}", cur)
            _eai_configuration_models.stamp_eval_task_combo(combo)
            if spin is not None:
                found = combo.findData(int(spin.value()))
                if found >= 0:
                    combo.setCurrentIndex(found)
            combo.blockSignals(False)
            if hasattr(self, "libero_task_thumb"):
                self.libero_task_thumb.show_combo_current(combo)

    def _on_libero_task_combo_changed(self, *_args) -> None:
        combo = getattr(self, "libero_task_combo", None)
        spin = getattr(self, "libero_task_id_spin", None)
        if combo is None:
            return
        data = combo.currentData()
        if spin is not None and data is not None:
            try:
                idx = int(data)
            except (TypeError, ValueError):
                idx = -1
            if idx >= 0 and spin.value() != idx:
                spin.blockSignals(True)
                spin.setValue(idx)
                spin.blockSignals(False)
                spin.setToolTip(combo.itemText(combo.currentIndex()))
        if hasattr(self, "libero_task_thumb"):
            self.libero_task_thumb.show_combo_current(combo)

    def _libero_task_id_value(self) -> int:
        if hasattr(self, "libero_task_id_spin"):
            return int(self.libero_task_id_spin.value())
        return 0

    def _on_libero_mode_changed(self, *_args) -> None:
        # Both modes may need Pi deploy (local = auto-deploy serve + remote client).
        for wname in ("libero_pi_start_btn", "libero_pi_probe_btn", "libero_pi_stop_btn"):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(True)
        if hasattr(self, "_update_molmospaces_pi_deploy_ui"):
            self._update_molmospaces_pi_deploy_ui()

    def _on_libero_rlinf_browse(self) -> None:
        start = ""
        if hasattr(self, "libero_rlinf_edit"):
            start = self.libero_rlinf_edit.text().strip()
        start = start or _eai_configuration_tabs.RLINF_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RLinf 仓库", start)
        if selected and hasattr(self, "libero_rlinf_edit"):
            self.libero_rlinf_edit.setText(selected)

    def _on_libero_ckpt_browse(self) -> None:
        start = self._libero_ckpt_text() or str(_eai_configuration_models.EAI_DIR)
        if start.startswith("gs://"):
            start = str(_eai_configuration_models.EAI_DIR / ".cache" / "openpi")
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 LIBERO checkpoint 目录", start)
        if selected and hasattr(self, "libero_ckpt_edit"):
            self.libero_ckpt_edit.setText(selected)

    def _on_libero_pi_start_clicked(self) -> None:
        # Sync LIBERO ckpt into MolmoSpaces Pi field so shared deploy path works.
        ckpt = self._libero_ckpt_text()
        cfg = self._libero_config_id()
        if hasattr(self, "molmospaces_pi_ckpt_edit") and ckpt:
            self.molmospaces_pi_ckpt_edit.setText(ckpt)
        try:
            self._append_mujoco_log(f"[libero] 部署 Pi config={cfg} ckpt={ckpt}")
            ok, msg = self._ensure_pi_policy_ready(
                start_if_needed=True,
                wait_s=900.0,
                prefer_ckpt=ckpt,
                prefer_config=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            self._append_mujoco_log(f"[libero] 部署异常: {msg}")
            self._on_mujoco_status(f"Pi 部署失败: {msg}")
            return
        if ok:
            self._on_mujoco_status(f"Pi 已部署: {msg}")
        else:
            self._on_mujoco_status(f"Pi 部署失败: {msg}")
        self._update_molmospaces_pi_deploy_ui()

    def _on_libero_pi_stop_clicked(self) -> None:
        self._on_molmospaces_pi_stop_clicked()

    def _libero_rynnvalue_server_url(self) -> str:
        return self._sim_rynnvalue_server_url()

    def _libero_rynnvalue_hud_enabled(self) -> bool:
        return self._sim_rynnvalue_hud_enabled()

    def _on_libero_rynnvalue_hud_toggled(self, checked: bool) -> None:
        self._on_sim_rynnvalue_hud_toggled(checked)

    def _on_libero_rynnvalue_url_changed(self) -> None:
        self._on_sim_rynnvalue_url_changed()

    def _on_libero_rynnvalue_status_tick(self) -> None:
        self._on_sim_rynnvalue_status_tick()

    def _on_libero_rynnvalue_probe_clicked(self) -> None:
        self._on_sim_rynnvalue_probe_clicked()

    def _on_libero_rynnvalue_deploy_clicked(self) -> None:
        self._on_sim_rynnvalue_deploy_clicked()

    def _on_libero_run_clicked(self, *, viewer_only: bool = False) -> None:
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        if self._sim_backend_id() != "libero":
            self._on_mujoco_status("请先将仿真后端切换为 LIBERO")
            return
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status(
                "MuJoCo / MolmoSpaces / Arena / LIBERO / RoboTwin 正在运行，请先停止"
            )
            return

        self.mujoco_log_edit.clear()
        mode = self._libero_mode_id()
        suite = "libero_spatial"
        if hasattr(self, "libero_suite_combo"):
            suite = str(self.libero_suite_combo.currentData() or suite)
        cfg = self._libero_config_id()
        ckpt = self._libero_ckpt_text()
        rlinf = ""
        if hasattr(self, "libero_rlinf_edit"):
            rlinf = self.libero_rlinf_edit.text().strip()
        py = _eai_configuration_models.resolve_libero_python()
        ok_deps, dep_msg = _eai_configuration_models.probe_libero_deps(py)
        if not ok_deps:
            self._append_mujoco_log(f"[libero] 依赖探测失败: {dep_msg}")
            self._append_mujoco_log(_eai_configuration_models.libero_install_hint(py))
            self._on_mujoco_status("LIBERO 依赖未就绪（见日志）")
            return

        max_tasks = 1
        trials = 1
        chunk = 5
        num_steps = 10
        task_id = self._libero_task_id_value()
        if hasattr(self, "libero_max_tasks_spin"):
            max_tasks = int(self.libero_max_tasks_spin.value())
        if hasattr(self, "libero_trials_spin"):
            trials = int(self.libero_trials_spin.value())
        if hasattr(self, "libero_chunk_spin"):
            chunk = int(self.libero_chunk_spin.value())
        if hasattr(self, "libero_steps_spin"):
            num_steps = int(self.libero_steps_spin.value())
        render_gui = True
        if hasattr(self, "libero_render_gui_check"):
            render_gui = bool(self.libero_render_gui_check.isChecked())
        if viewer_only:
            render_gui = True
            self._append_mujoco_log(
                "[libero] 只开界面：不连 Pi / 不推理；「手臂/手」文本指挥可用"
            )
            try:
                argv, cwd, env_extra = _eai_configuration_models.build_libero_eval_argv(
                    mode="remote",
                    task_suite=suite,
                    config_name=cfg,
                    checkpoint=ckpt,
                    max_tasks=1,
                    task_id=task_id,
                    num_trials_per_task=1,
                    action_chunk=chunk,
                    num_steps=num_steps,
                    python_bin=py,
                    rlinf_root=rlinf,
                    render_gui=True,
                    viewer_only=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._on_mujoco_status(f"无法组装 LIBERO viewer 命令: {exc}")
                return
            clean_env = {k: v for k, v in env_extra.items() if v is not None}
            clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
            clean_env.update(self._sim_bridge_child_env())
            tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
            prev = clean_env.get("PYTHONPATH") or ""
            clean_env["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
                [tools] + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p != tools]
            )
            self._expand_workspace_panel()
            self._set_rynnvalue_workspace_hud_active(True)
            self._append_mujoco_log(
                f"[libero] python={py} suite={suite} task_id={task_id} viewer_only"
            )
            self._mujoco_launcher.start_cwd_command(
                cwd=str(cwd),
                argv=argv,
                label=f"libero-viewer {suite}[{task_id}]",
                env_extra=clean_env,
            )
            self._update_mujoco_ui()
            return

        # LingBot-VA: official wan_va server + LIBERO client (not OpenPI).
        if _eai_configuration_models.is_lingbot_va_policy(cfg):
            ok_va, va_msg = _eai_configuration_models.probe_lingbot_va()
            self._append_mujoco_log(f"[libero/LingBot-VA] {va_msg}")
            if not ok_va:
                self._append_mujoco_log(_eai_configuration_models.lingbot_va_install_hint())
                self._on_mujoco_status("LingBot-VA 未就绪（见日志）")
                return
            task_end = int(task_id) + max(1, int(max_tasks))
            try:
                argv, cwd, env_extra = _eai_configuration_models.build_libero_va_eval_argv(
                    task_suite=suite,
                    checkpoint=ckpt,
                    test_num=trials,
                    task_start=int(task_id),
                    task_end=task_end,
                    python_bin=_eai_configuration_models.resolve_lingbot_va_python(),
                    start_server=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._on_mujoco_status(f"无法组装 LingBot-VA LIBERO 命令: {exc}")
                return
            clean_env = {k: v for k, v in env_extra.items() if v is not None}
            clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
            clean_env.update(self._sim_bridge_child_env())
            self._append_mujoco_log(
                f"[libero/LingBot-VA] suite={suite} task_id={task_id} "
                f"trials={trials} ckpt={ckpt}"
            )
            self._expand_workspace_panel()
            self._mujoco_launcher.start_cwd_command(
                cwd=str(cwd),
                argv=argv,
                label=f"libero-va {suite}",
                env_extra=clean_env,
            )
            self._update_mujoco_ui()
            return

        # VPP2: official scripts/libero/eval.sh
        if _eai_configuration_models.is_vpp2_policy(cfg):
            self._launch_vpp2_backend(
                "libero",
                task=suite,
                task_suite=suite,
                label=f"libero-vpp2 {suite}",
            )
            return

        # libero 客户端环境无 jax/openpi；「local」改为自动部署 serve + remote 评测。
        eval_mode = mode
        if mode == "local":
            self._append_mujoco_log(
                "[libero] local：libero 环境无 jax，自动部署 openpi serve_policy，"
                "再以 remote 客户端评测"
            )
            eval_mode = "remote"

        self._append_mujoco_log(f"[libero] 确认 Pi serve（config={cfg}）…")
        if hasattr(self, "molmospaces_pi_ckpt_edit") and ckpt:
            self.molmospaces_pi_ckpt_edit.setText(ckpt)
        ok, msg = self._ensure_pi_policy_ready(
            start_if_needed=True,
            wait_s=900.0,
            prefer_ckpt=ckpt,
            prefer_config=cfg,
        )
        if not ok:
            self._append_mujoco_log(f"[libero] Pi 预检失败: {msg}")
            self._on_mujoco_status(f"Pi 未就绪: {msg}")
            return
        self._append_mujoco_log(f"[libero] Pi 预检通过: {msg}")

        _rv = self._sim_rynnvalue_hud_settings()
        rynnvalue_live_hud = bool(_rv["enabled"])
        rynnvalue_refresh_sec = float(_rv["refresh_sec"])
        rynnvalue_server_url = str(_rv["server_url"])
        rynnvalue_num_frames = int(_rv["num_frames"])

        if rynnvalue_live_hud:
            ok_w, msg_w = self._warmup_pi_before_rynnvalue_hud()
            if not ok_w:
                self._append_mujoco_log(f"[libero] {msg_w}")
                self._on_mujoco_status("Pi 首次推理预热失败（见日志）")
                return
            self._append_mujoco_log(f"[libero] {msg_w}")
            self._append_mujoco_log(
                f"[libero] 确认 RynnValue reward_server（{rynnvalue_server_url}）…"
            )
            ok_rv, msg_rv = self._ensure_rynnvalue_reward_server_ready(
                start_if_needed=True,
                wait_s=900.0,
                prefix="libero",
            )
            if not ok_rv:
                self._append_mujoco_log(f"[libero] RynnValue 预检失败: {msg_rv}")
                self._on_mujoco_status(f"RynnValue 未就绪: {msg_rv}")
                return
            self._append_mujoco_log(f"[libero] RynnValue 预检通过: {msg_rv}")

        try:
            argv, cwd, env_extra = _eai_configuration_models.build_libero_eval_argv(
                mode=eval_mode,
                task_suite=suite,
                config_name=cfg,
                checkpoint=ckpt,
                host=_eai_configuration_models.LIBERO_HOST_DEFAULT,
                port=_eai_configuration_models.LIBERO_PORT_DEFAULT,
                num_trials_per_task=trials,
                max_tasks=max_tasks,
                task_id=task_id,
                action_chunk=chunk,
                num_steps=num_steps,
                python_bin=py,
                rlinf_root=rlinf,
                render_gui=render_gui,
                rynnvalue_live_hud=rynnvalue_live_hud,
                rynnvalue_server_url=rynnvalue_server_url,
                rynnvalue_refresh_sec=rynnvalue_refresh_sec,
                rynnvalue_num_frames=rynnvalue_num_frames,
                rynnvalue_show_window=bool(_rv["show_window"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._on_mujoco_status(f"无法组装 LIBERO 命令: {exc}")
            return

        self._append_mujoco_log(
            f"[libero] mode={mode}"
            + (f"→{eval_mode}" if eval_mode != mode else "")
            + f" suite={suite} task_id={task_id} config={cfg} render_gui={render_gui}"
            + f" rynnvalue_hud={rynnvalue_live_hud}"
        )
        self._append_mujoco_log(f"[libero] python={py}")
        self._append_mujoco_log(f"[libero] ckpt={ckpt}")
        self._append_mujoco_log(
            f"[libero] max_tasks={max_tasks} task_id={task_id} trials={trials} "
            f"chunk={chunk} num_steps={num_steps}"
        )
        if rynnvalue_live_hud:
            self._append_mujoco_log(
                f"[libero] RynnValue Live HUD "
                f"url={rynnvalue_server_url} refresh={rynnvalue_refresh_sec}s "
                f"num_frames={rynnvalue_num_frames}"
            )
        # Always feed agentview into 工作区图像预览 + /camera/*（多视角 6D）。
        # LIBERO 通常只有头（agentview）+ 右腕（eye-in-hand），清掉左腕等残留以免占位显示。
        purged = self._purge_sim_bridge_cam_keys(("cam_left_wrist",))
        if purged:
            self._append_mujoco_log(
                f"[libero] 已清除无摄像头残留帧 cam_left_wrist（{purged} 个文件）"
            )
        self._expand_workspace_panel()
        self._set_rynnvalue_workspace_hud_active(True)
        self._sim_preview_want_topics = None
        self._activate_sim_camera_preview(force=True)
        self._append_mujoco_log(
            f"[libero] 图像预览 ← 仅显示有画面的相机"
            f"（头/右腕；无左腕则不显示）← {_eai_configuration_policies.LIBERO_PREVIEW_JPG} + cam_*.npy"
        )
        self._append_mujoco_log(f"[libero] out={_eai_configuration_models.LIBERO_EVAL_OUTPUT_DIR}")
        if render_gui and not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_mujoco_log(
                "[libero] 警告: DISPLAY 为空，图形窗口可能无法弹出"
            )
        clean_env = {k: v for k, v in env_extra.items() if v is not None}
        if rynnvalue_live_hud:
            clean_env.update(self._rynnvalue_live_hud_child_env("libero"))
        # Force EAI_DIR so the child writes preview next to this UI cache.
        clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
        clean_env.update(self._sim_bridge_child_env())
        self._mujoco_launcher.start_cwd_command(
            cwd=str(cwd),
            argv=argv,
            label=f"libero {suite}",
            env_extra=clean_env,
        )
        self._update_mujoco_ui()

    def _on_libero_config_changed(self, *_args) -> None:
        cfg = self._libero_config_id()
        if _eai_configuration_models.is_vpp2_policy(cfg):
            if hasattr(self, "libero_ckpt_edit"):
                cur = self.libero_ckpt_edit.text().strip()
                if (not cur) or "pi05_libero" in cur or "pi0_libero" in cur:
                    self.libero_ckpt_edit.setText(
                        _eai_configuration_models.resolve_vpp2_bundle(
                            _eai_platform_bootstrap.os.environ.get("VPP2_BUNDLE", "joint2b_s100000")
                        )
                    )
            if hasattr(self, "libero_mode_combo"):
                idx = self.libero_mode_combo.findData("local")
                if idx >= 0:
                    self.libero_mode_combo.setCurrentIndex(idx)
            return
        if not _eai_configuration_models.is_lingbot_va_policy(cfg):
            return
        if hasattr(self, "libero_ckpt_edit"):
            cur = self.libero_ckpt_edit.text().strip()
            if (not cur) or "pi05_libero" in cur or "pi0_libero" in cur:
                self.libero_ckpt_edit.setText(
                    _eai_configuration_models.resolve_lingbot_va_checkpoint(
                        _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_CKPT", "")
                    )
                )
        if hasattr(self, "libero_mode_combo"):
            # VA path manages its own wan_va server.
            idx = self.libero_mode_combo.findData("local")
            if idx >= 0:
                self.libero_mode_combo.setCurrentIndex(idx)

