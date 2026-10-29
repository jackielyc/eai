"""ui.window.shell warmup pi before rynnvalue hud — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

class ShellWarmupPiBeforeRynnvalueHudMixin:
    def _warmup_pi_before_rynnvalue_hud(self) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Init Pi JAX/cuBLAS before RynnValue occupies the same GPU."""
        host = _eai_configuration_models.PI_SERVER_HOST_DEFAULT
        port = _eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT
        self._append_mujoco_log(
            "[libero] Pi 首次推理预热（cuBLAS；须在 RynnValue 占卡前完成）…"
        )
        ok, msg = _eai_configuration_models.warmup_pi_server_infer(host, port, timeout_s=180.0)
        if ok:
            return True, msg
        rv_running = bool(
            getattr(self, "_rynnvalue_launcher", None)
            and self._rynnvalue_launcher.is_running()
        )
        if not rv_running:
            return False, msg
        self._append_mujoco_log(
            "[libero] 预热失败且 RynnValue 已占卡，先停 RynnValue 再预热 Pi…"
        )
        self._append_mujoco_log(f"[libero] {msg}")
        try:
            self._rynnvalue_launcher.stop()
        except Exception as exc:  # noqa: BLE001
            return False, f"无法停止 RynnValue 以给 Pi 腾显存: {exc}"
        deadline = _eai_platform_bootstrap.time.time() + 30.0
        while _eai_platform_bootstrap.time.time() < deadline:
            _eai_platform_bootstrap.QApplication.processEvents()
            if not self._rynnvalue_launcher.is_running():
                break
            _eai_platform_bootstrap.time.sleep(0.4)
        _eai_platform_bootstrap.time.sleep(2.0)
        ok2, msg2 = _eai_configuration_models.warmup_pi_server_infer(host, port, timeout_s=180.0)
        if ok2:
            return True, f"已停 RynnValue 后预热成功。{msg2}"
        return False, msg2

    def _ensure_pi_policy_ready(
        self,
        *,
        start_if_needed: bool = True,
        wait_s: float = 600.0,
        prefer_ckpt: str = "",
        prefer_config: str = "",
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Validate Pi ckpt and confirm ws://host:port is callable."""
        prefer = (prefer_ckpt or "").strip() or self._molmospaces_pi_ckpt_text()
        # Force jointpos default config when path points at jointpos dirs / gs.
        default_cfg = (prefer_config or "").strip() or _eai_configuration_tabs.MOLMOSPACES_PI_CONFIG_DEFAULT
        ckpt = _eai_configuration_models.resolve_pi_checkpoint(prefer, config=default_cfg)
        path = ckpt.path
        config = ckpt.config or default_cfg
        if prefer_config:
            config = prefer_config.strip()
        elif "jointpos" in path.replace("\\", "/") and "polaris" not in _eai_platform_bootstrap.os.path.basename(
            path.rstrip("/")
        ):
            config = "pi05_droid_jointpos"
        if "libero" in path.replace("\\", "/").lower() and not prefer_config:
            # LIBERO checkpoints should not inherit droid jointpos default.
            if "pi05" in path:
                config = "pi05_libero"
            elif "pi0" in path:
                config = "pi0_libero"
        host = _eai_configuration_models.PI_SERVER_HOST_DEFAULT
        port = _eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT
        status = _eai_configuration_models.checkpoint_status(path)
        if prefer and prefer.rstrip("/") != path.rstrip("/"):
            self._append_mujoco_log(
                f"[pi] 已改用可用模型: {prefer} → {path} ({ckpt.source})"
            )
        self._append_mujoco_log(f"[pi] checkpoint={path}")
        self._append_mujoco_log(f"[pi] config={config} status={status}")
        if hasattr(self, "molmospaces_pi_ckpt_edit") and path:
            self.molmospaces_pi_ckpt_edit.setText(path)

        if path.startswith("gs://") or not _eai_configuration_models.is_local_checkpoint_usable(path):
            # gs or missing local → materialize (HF mirror for jointpos).
            materialize_src = path if path.startswith("gs://") else _eai_configuration_models.PI_CKPT_GS_DEFAULT
            if not path.startswith("gs://"):
                self._append_mujoco_log(
                    f"[pi] 本地不可用 ({status})，改拉官方 {materialize_src}"
                )
            ok_dl, local_or_err = _eai_configuration_models.ensure_checkpoint_materialized(
                materialize_src, log=self._append_mujoco_log
            )
            if not ok_dl:
                return False, local_or_err
            path = local_or_err
            config = _eai_configuration_models.infer_policy_config(path, config)
            if "jointpos" in path and "polaris" not in _eai_platform_bootstrap.os.path.basename(
                path.rstrip("/")
            ):
                config = "pi05_droid_jointpos"
            status = _eai_configuration_models.checkpoint_status(path)
            self._append_mujoco_log(f"[pi] 本地化后 checkpoint={path} ({status})")
            if hasattr(self, "molmospaces_pi_ckpt_edit"):
                self.molmospaces_pi_ckpt_edit.setText(path)

        ok, msg = _eai_configuration_models.probe_pi_server(host, port, timeout_s=3.0)
        if ok:
            match_ok, match_msg = _eai_configuration_models.pi_server_matches(
                port=port, config=config, checkpoint=path
            )
            if match_ok:
                self._append_mujoco_log(f"[pi] {msg} ({match_msg})")
                self._update_molmospaces_pi_deploy_ui()
                return True, msg
            self._append_mujoco_log(
                f"[pi] 已有服务与目标不匹配: {match_msg} → 停止并重部署 {config}"
            )
            if hasattr(self, "_pi_policy_launcher") and self._pi_policy_launcher.is_running():
                self._pi_policy_launcher.stop()
            stop_msg = _eai_configuration_models.stop_pi_server_on_port(port, log=self._append_mujoco_log)
            self._append_mujoco_log(f"[pi] {stop_msg}")
            # Fall through to start_if_needed redeploy below.
            msg = match_msg

        if _eai_configuration_models.port_listening(host, port):
            # Occupied but not a matching / probe-able openpi server.
            match_ok, match_msg = _eai_configuration_models.pi_server_matches(
                port=port, config=config, checkpoint=path
            )
            if not match_ok and start_if_needed:
                self._append_mujoco_log(
                    f"[pi] 端口占用且不匹配: {match_msg} → 停止后重部署"
                )
                if hasattr(self, "_pi_policy_launcher") and self._pi_policy_launcher.is_running():
                    self._pi_policy_launcher.stop()
                stop_msg = _eai_configuration_models.stop_pi_server_on_port(port, log=self._append_mujoco_log)
                self._append_mujoco_log(f"[pi] {stop_msg}")
            elif not match_ok:
                return (
                    False,
                    f"{host}:{port} 已占用但策略不匹配: {match_msg}。"
                    "请先点「部署停止」再启动。",
                )
            elif not ok:
                return (
                    False,
                    f"{host}:{port} 已占用但不是可用的 OpenPI 服务: {msg}。"
                    "请先点「部署停止」再启动。",
                )

        if not start_if_needed:
            return False, msg

        if not hasattr(self, "_pi_policy_launcher"):
            return False, "内部错误: Pi 服务 launcher 未初始化"

        if self._pi_policy_launcher.is_running():
            self._append_mujoco_log("[pi] serve_policy 已在启动中，继续等待…")
        else:
            try:
                argv, cwd, env_extra = _eai_configuration_models.build_serve_policy_argv(
                    checkpoint=path,
                    config=config,
                    port=port,
                )
            except Exception as exc:  # noqa: BLE001
                return False, f"无法组装 serve_policy 命令: {exc}"
            self._append_mujoco_log(
                f"[pi] 启动 serve_policy config={config} dir={path} port={port}"
            )
            self._append_mujoco_log(
                f"[pi] OPENPI_DATA_HOME={env_extra.get('OPENPI_DATA_HOME')}"
            )
            self._pi_policy_launcher.start_cwd_command(
                cwd=str(cwd),
                argv=argv,
                label=f"openpi serve_policy:{config}",
                env_extra=env_extra,
            )
            if hasattr(self, "_pi_policy_ready_timer"):
                self._pi_policy_ready_timer.start()
            self._update_molmospaces_pi_deploy_ui()

        deadline = _eai_platform_bootstrap.time.time() + max(30.0, float(wait_s))
        last = msg
        while _eai_platform_bootstrap.time.time() < deadline:
            _eai_platform_bootstrap.QApplication.processEvents()
            self._update_molmospaces_pi_deploy_ui()
            if not self._pi_policy_launcher.is_running() and not _eai_configuration_models.port_listening(
                host, port
            ):
                return False, "serve_policy 进程已退出，请查看上方日志"
            ok, last = _eai_configuration_models.probe_pi_server(host, port, timeout_s=4.0)
            if ok:
                match_ok, match_msg = _eai_configuration_models.pi_server_matches(
                    port=port, config=config, checkpoint=path
                )
                if match_ok:
                    self._append_mujoco_log(f"[pi] {last} ({match_msg})")
                    self._update_molmospaces_pi_deploy_ui()
                    return True, last
                last = match_msg
            _eai_platform_bootstrap.time.sleep(2.0)
        return False, f"等待策略服务超时: {last}"

    def _start_psi_molmospaces_viewer_only(self) -> None:
        """Open the currently selected Psi task scene (no model / no planner).

        Always uses ``run_molmospaces_eval.py`` + the task combo benchmark —
        never the bare Psi R1 MJCF quick_viewer path.
        """
        root = self._resolve_molmospaces_root()
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self._on_mujoco_status(f"Psi-MolmoSpaces 目录不存在: {root}")
            return
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("MuJoCo / MolmoSpaces 正在运行，请先停止")
            return
        task = None
        if hasattr(self, "molmospaces_task_combo"):
            task = self.molmospaces_task_combo.currentData()
        if not isinstance(task, dict) or not task.get("path"):
            self._on_mujoco_status("请先在任务下拉框选择一个 Psi 评测任务")
            return
        bench = str(task["path"])
        if not _eai_adapters_models_discovery.molmospaces_benchmark_ready(bench):
            self._append_mujoco_log(
                f"[psi_molmospaces] 任务数据未就绪: {bench}"
            )
            self._on_mujoco_status(f"任务未下载: {task.get('label')}")
            return
        ok_r1, r1_msg = _eai_adapters_models_discovery.ensure_psi_r1_robot_assets(root)
        if not ok_r1:
            self._append_mujoco_log(f"[psi_molmospaces] {r1_msg}")
            self._on_mujoco_status("缺少 Psi R1 机器人资源")
            return
        py = self._resolve_molmospaces_python()
        wrapper = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "tools",
            "run_molmospaces_eval.py",
        )
        if not _eai_platform_bootstrap.os.path.isfile(wrapper):
            self._on_mujoco_status(f"未找到: {wrapper}")
            return
        # PsiR1 robot+scene from planner config; VIEWER_ONLY swaps in DummyPolicy.
        policy = _eai_configuration_tabs.PSI_MOLMOSPACES_DEFAULT_POLICY
        out_dir = _eai_configuration_tabs.PSI_MOLMOSPACES_EVAL_OUTPUT_DIR
        _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
        max_eps = 1
        if hasattr(self, "molmospaces_episodes_spin"):
            max_eps = max(1, int(self.molmospaces_episodes_spin.value()))
        argv = [
            py,
            wrapper,
            policy,
            "--benchmark_dir",
            bench,
            "--no_wandb",
            "--num_workers",
            "1",
            "--output_dir",
            out_dir,
            "--max_episodes",
            str(max_eps),
        ]
        extra = task.get("extra") or ()
        argv.extend(str(x) for x in extra)
        vgl = (
            "/usr/bin/vglrun"
            if _eai_platform_bootstrap.os.path.isfile("/usr/bin/vglrun")
            else "/opt/VirtualGL/bin/vglrun"
            if _eai_platform_bootstrap.os.path.isfile("/opt/VirtualGL/bin/vglrun")
            else ""
        )
        if vgl:
            argv = [vgl, "-d", _eai_platform_bootstrap.os.environ.get("MUJOCO_VGL_DEVICE") or "egl", *argv]
        self.mujoco_log_edit.clear()
        self._append_mujoco_log(
            "[psi_molmospaces] 只开界面：按所选任务开场景（DummyPolicy，不推理）"
        )
        self._append_mujoco_log(f"[psi_molmospaces] task={task.get('label')}")
        self._append_mujoco_log(f"[psi_molmospaces] benchmark={bench}")
        self._append_mujoco_log(f"[psi_molmospaces] assets={r1_msg}")
        _vp_fp, _vp_cam, vp_label = self._molmospaces_viewer_cam_choice()
        self._append_mujoco_log(
            f"[psi_molmospaces] viewer: 官方 MuJoCo GLFW · {vp_label}"
        )
        self._append_mujoco_log(
            "[psi_molmospaces] 结束后保留窗口，点「停止」或关闭窗口再退出"
        )
        env_extra = {
            "MUJOCO_GL": "glfw",
            "MOLMOSPACES_ROOT": root,
            "RYNNVALUE_HUD_BACKEND": "psi_molmospaces",
            "MOLMOSPACES_EVAL_VIEWER": "1",
            "MOLMOSPACES_VIEWER_ONLY": "1",
            **self._molmospaces_viewer_cam_env(),
            "MOLMOSPACES_FORCE_EGL": "0",
            "MOLMOSPACES_VIEWER_KEEP_OPEN": "1",
            "EAI_DIR": _eai_configuration_models.EAI_DIR,
            "OPENPI_DATA_HOME": _eai_platform_bootstrap.os.environ.get("OPENPI_DATA_HOME")
            or _eai_configuration_models.OPENPI_DATA_HOME_DEFAULT,
            "PYTHONPATH": root,
            "PYTHONNOUSERSITE": "1",
            **_eai_adapters_models_discovery.molmospaces_cache_env(),
        }
        if self._sim_rynnvalue_hud_enabled():
            if not self._ensure_live_hud_reward_or_abort("psi_molmospaces"):
                return
            hud_env = self._rynnvalue_live_hud_child_env("psi_molmospaces")
            if hud_env:
                env_extra.update(hud_env)
                self._expand_workspace_panel()
                self._append_mujoco_log(
                    "[psi_molmospaces] RynnValue Live HUD "
                    f"url={hud_env.get('RYNNVALUE_SERVER_URL')}"
                )
        tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
        prev = env_extra.get("PYTHONPATH") or ""
        env_extra["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
            [root, tools]
            + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p not in (root, tools)]
        )
        self._mujoco_launcher.start_cwd_command(
            cwd=root,
            argv=argv,
            label=f"psi_molmospaces viewer {task.get('id')}",
            env_extra=env_extra,
        )
        self._update_mujoco_ui()

    def _append_dojo_rl_log(self, line: str) -> None:
        if not hasattr(self, "dojo_rl_log_edit"):
            return
        self.dojo_rl_log_edit.append(line)
        bar = self.dojo_rl_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_dojo_rl_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_dojo_rl_launcher", None)
            and self._dojo_rl_launcher.is_running()
        )
        if hasattr(self, "dojo_rl_start_btn"):
            self.dojo_rl_start_btn.setEnabled(not running)
        if hasattr(self, "dojo_rl_stop_btn"):
            self.dojo_rl_stop_btn.setEnabled(running)
        for wname in (
            "dojo_rl_mode_combo",
            "dojo_rl_config_combo",
            "dojo_rl_root_edit",
            "dojo_rl_python_edit",
            "dojo_rl_cuda_edit",
            "dojo_rl_overrides_edit",
            "dojo_rl_model_dir_edit",
            "dojo_rl_reward_ckpt_edit",
            "dojo_rl_resume_edit",
            "dojo_rl_refresh_btn",
            "dojo_rl_auto_gpu_check",
            "dojo_rl_root_browse_btn",
            "dojo_rl_model_browse_btn",
            "dojo_rl_reward_browse_btn",
            "dojo_rl_resume_browse_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)

    def _refresh_dojo_rl_configs(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "dojo_rl_config_combo"):
            return
        root = self.dojo_rl_root_edit.text().strip()
        names = _eai_adapters_models_discovery.list_robodojo_online_rl_configs(root)
        current = prefer or self.dojo_rl_config_combo.currentText().strip() or "rl_release"
        self.dojo_rl_config_combo.blockSignals(True)
        self.dojo_rl_config_combo.clear()
        for name in names:
            self.dojo_rl_config_combo.addItem(name, name)
        if not names:
            self.dojo_rl_config_combo.addItem("rl_release", "rl_release")
        idx = self.dojo_rl_config_combo.findData(current)
        if idx < 0:
            idx = self.dojo_rl_config_combo.findText(current)
        if idx >= 0:
            self.dojo_rl_config_combo.setCurrentIndex(idx)
        else:
            self.dojo_rl_config_combo.setEditText(current)
        self.dojo_rl_config_combo.blockSignals(False)

    def _append_robometer_log(self, line: str) -> None:
        if not hasattr(self, "robometer_log_edit"):
            return
        self.robometer_log_edit.append(line)
        bar = self.robometer_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_robometer_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_robometer_launcher", None)
            and self._robometer_launcher.is_running()
        )
        if hasattr(self, "robometer_start_btn"):
            self.robometer_start_btn.setEnabled(not running)
        if hasattr(self, "robometer_stop_btn"):
            self.robometer_stop_btn.setEnabled(running)
        for wname in (
            "robometer_job_combo",
            "robometer_config_combo",
            "robometer_root_edit",
            "robometer_python_edit",
            "robometer_cuda_edit",
            "robometer_overrides_edit",
            "robometer_ckpt_edit",
            "robometer_refresh_btn",
            "robometer_render_gui_check",
            "robometer_wandb_offline_check",
            "robometer_root_browse_btn",
            "robometer_ckpt_browse_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)

    def _refresh_robometer_configs(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "robometer_config_combo"):
            return
        root = self.robometer_root_edit.text().strip()
        names = _eai_adapters_models_discovery.list_robometer_configs(root)
        current = (
            prefer
            or self.robometer_config_combo.currentText().strip()
            or "libero_online_rl"
        )
        self.robometer_config_combo.blockSignals(True)
        self.robometer_config_combo.clear()
        for name in names:
            self.robometer_config_combo.addItem(name, name)
        if not names:
            self.robometer_config_combo.addItem(current, current)
        idx = self.robometer_config_combo.findData(current)
        if idx < 0:
            idx = self.robometer_config_combo.findText(current)
        if idx >= 0:
            self.robometer_config_combo.setCurrentIndex(idx)
        else:
            self.robometer_config_combo.setEditText(current)
        self.robometer_config_combo.blockSignals(False)

    def _append_rynnvalue_log(self, line: str) -> None:
        if _eai_adapters_models_discovery.rynnvalue_status_log_is_routine(line):
            return
        if hasattr(self, "rynnvalue_log_edit"):
            self.rynnvalue_log_edit.append(line)
            bar = self.rynnvalue_log_edit.verticalScrollBar()
            bar.setValue(bar.maximum())
        # Mirror into sim log while deploying/using Live HUD so failures are visible.
        if self._sim_rynnvalue_hud_enabled() and hasattr(self, "_append_mujoco_log"):
            text = (line or "").rstrip()
            if text:
                self._append_mujoco_log(f"[rynnvalue] {text}")

    def _update_rynnvalue_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_rynnvalue_launcher", None)
            and self._rynnvalue_launcher.is_running()
        )
        if hasattr(self, "rynnvalue_start_btn"):
            self.rynnvalue_start_btn.setEnabled(not running)
        if hasattr(self, "rynnvalue_stop_btn"):
            self.rynnvalue_stop_btn.setEnabled(running)
        if hasattr(self, "_update_sim_rynnvalue_hud_controls"):
            self._update_sim_rynnvalue_hud_controls()
        job = ""
        if hasattr(self, "rynnvalue_job_combo"):
            job = str(self.rynnvalue_job_combo.currentData() or "").strip()
        is_infer = job in ("infer", "inference", "demo", "")
        is_server = job in ("reward_server", "server", "serve")
        for wname in (
            "rynnvalue_job_combo",
            "rynnvalue_root_edit",
            "rynnvalue_python_edit",
            "rynnvalue_cuda_edit",
            "rynnvalue_model_edit",
            "rynnvalue_ckpt_edit",
            "rynnvalue_extra_edit",
            "rynnvalue_frames_spin",
            "rynnvalue_steps_spin",
            "rynnvalue_batch_spin",
            "rynnvalue_tokens_spin",
            "rynnvalue_fps_spin",
            "rynnvalue_port_spin",
            "rynnvalue_render_gui_check",
            "rynnvalue_root_browse_btn",
            "rynnvalue_model_browse_btn",
            "rynnvalue_ckpt_browse_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)
        for wname in (
            "rynnvalue_video_edit",
            "rynnvalue_video_browse_btn",
            "rynnvalue_instruction_edit",
            "rynnvalue_robot_edit",
            "rynnvalue_camera_edit",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running and is_infer)
        if hasattr(self, "rynnvalue_port_spin"):
            self.rynnvalue_port_spin.setEnabled(not running and is_server)
        if hasattr(self, "rynnvalue_open_video_btn"):
            # Always clickable: click handler resolves last session path or scans disk.
            self.rynnvalue_open_video_btn.setEnabled(True)

    def _play_rynnvalue_video(self, path: str) -> None:
        try:
            dlg = _eai_ui_widgets_video.SimEvalVideoDialog(path, self)
            dlg.setAttribute(_eai_platform_bootstrap.Qt.WA_DeleteOnClose, True)
            dlg.show()
            self._on_rynnvalue_status(f"播放: {_eai_platform_bootstrap.os.path.basename(path)}")
            return
        except Exception as exc:
            self._append_rynnvalue_log(f"[rynnvalue] 内置播放失败: {exc}")
        candidates: _eai_platform_bootstrap.List[_eai_platform_bootstrap.List[str]] = []
        ffplay = _eai_platform_bootstrap.shutil.which("ffplay")
        if ffplay:
            candidates.append(
                [ffplay, "-autoexit", "-window_title", _eai_platform_bootstrap.os.path.basename(path), path]
            )
        if _eai_platform_bootstrap.shutil.which("xdg-open"):
            candidates.append(["xdg-open", path])
        for cmd in candidates:
            try:
                _eai_platform_bootstrap.subprocess.Popen(cmd, start_new_session=True)
                self._on_rynnvalue_status(f"已用 {cmd[0]} 打开视频")
                return
            except Exception:
                continue
        self._append_rynnvalue_log(f"[rynnvalue] 无法打开视频: {path}")

    def _find_newest_rynnvalue_trend_video(self) -> str:
        """Locate newest output_with_trend.mp4 under known output roots."""
        roots: _eai_platform_bootstrap.List[str] = []
        log_dir = ""
        if hasattr(self, "rynnvalue_log_dir_edit"):
            log_dir = self.rynnvalue_log_dir_edit.text().strip()
        if log_dir:
            roots.append(log_dir)
        roots.append(_eai_configuration_tabs.RYNNVALUE_CACHE_DIR)
        repo = ""
        if hasattr(self, "rynnvalue_root_edit"):
            repo = self.rynnvalue_root_edit.text().strip()
        if not repo:
            repo = _eai_adapters_models_discovery.resolve_rynnvalue_root()
        if repo:
            roots.append(_eai_platform_bootstrap.os.path.join(repo, "rynn_infer", "outputs"))
            roots.append(_eai_platform_bootstrap.os.path.join(repo, "outputs"))
        newest = ""
        newest_mtime = -1.0
        seen = set()
        for root in roots:
            root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(root or ""))
            if not root or root in seen or not _eai_platform_bootstrap.os.path.isdir(root):
                continue
            seen.add(root)
            for dirpath, _dirs, files in _eai_platform_bootstrap.os.walk(root):
                for name in files:
                    if name != "output_with_trend.mp4" and not name.endswith(
                        "_with_trend.mp4"
                    ):
                        continue
                    cand = _eai_platform_bootstrap.os.path.join(dirpath, name)
                    try:
                        mtime = _eai_platform_bootstrap.os.path.getmtime(cand)
                    except OSError:
                        continue
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = cand
        return newest

    def _append_roboicl_log(self, line: str) -> None:
        if not hasattr(self, "roboicl_log_edit"):
            return
        self.roboicl_log_edit.append(line)
        bar = self.roboicl_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_roboicl_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_roboicl_launcher", None)
            and self._roboicl_launcher.is_running()
        )
        if hasattr(self, "roboicl_start_btn"):
            self.roboicl_start_btn.setEnabled(not running)
        if hasattr(self, "roboicl_stop_btn"):
            self.roboicl_stop_btn.setEnabled(running)
        for wname in (
            "roboicl_mode_combo",
            "roboicl_task_combo",
            "roboicl_protocol_combo",
            "roboicl_root_edit",
            "roboicl_policy_python_edit",
            "roboicl_sim_python_edit",
            "roboicl_shots_spin",
            "roboicl_seed_spin",
            "roboicl_layout_spin",
            "roboicl_gpu_spin",
            "roboicl_data_edit",
            "roboicl_results_edit",
            "roboicl_reference_edit",
            "roboicl_model_edit",
            "roboicl_api_key_edit",
            "roboicl_extra_edit",
            "roboicl_render_gui_check",
            "roboicl_refresh_btn",
            "roboicl_root_browse_btn",
            "roboicl_data_browse_btn",
            "roboicl_results_browse_btn",
            "roboicl_reference_browse_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)

    def _refresh_roboicl_tasks(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "roboicl_task_combo"):
            return
        root = self.roboicl_root_edit.text().strip()
        names = _eai_adapters_models_discovery.list_roboicl_tasks(root)
        current = (
            prefer
            or self.roboicl_task_combo.currentText().strip()
            or "deposit_coin"
        )
        self.roboicl_task_combo.blockSignals(True)
        self.roboicl_task_combo.clear()
        for name in names:
            self.roboicl_task_combo.addItem(name, name)
        if not names:
            self.roboicl_task_combo.addItem(current, current)
        idx = self.roboicl_task_combo.findData(current)
        if idx < 0:
            idx = self.roboicl_task_combo.findText(current)
        if idx >= 0:
            self.roboicl_task_combo.setCurrentIndex(idx)
        else:
            self.roboicl_task_combo.setEditText(current)
        self.roboicl_task_combo.blockSignals(False)

    def _append_lingbot_map_log(self, line: str) -> None:
        if not hasattr(self, "lingbot_map_log_edit"):
            return
        self.lingbot_map_log_edit.append(line)
        bar = self.lingbot_map_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_lingbot_map_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_lingbot_map_launcher", None)
            and self._lingbot_map_launcher.is_running()
        )
        scene = str(self.lingbot_map_scene_combo.currentData() or "courthouse")
        path_editable = scene in ("folder", "video")
        camera_mode = scene == "camera"
        self.lingbot_map_run_btn.setEnabled(not running)
        self.lingbot_map_stop_btn.setEnabled(running)
        self.lingbot_map_open_btn.setEnabled(
            bool(getattr(self, "_lingbot_map_viser_url", ""))
        )
        self.lingbot_map_scene_combo.setEnabled(not running)
        self.lingbot_map_ckpt_combo.setEnabled(not running)
        self.lingbot_map_port_spin.setEnabled(not running)
        self.lingbot_map_path_edit.setEnabled(not running and path_editable)
        self.lingbot_map_browse_btn.setEnabled(not running and path_editable)
        self.lingbot_map_model_edit.setEnabled(not running)
        self.lingbot_map_model_browse_btn.setEnabled(not running)
        self.lingbot_map_first_k_spin.setEnabled(not running)
        self.lingbot_map_stride_spin.setEnabled(not running)
        self.lingbot_map_fps_spin.setEnabled(not running)
        self.lingbot_map_mask_sky_check.setEnabled(not running)
        self.lingbot_map_sdpa_check.setEnabled(not running)
        self.lingbot_map_capture_spin.setEnabled(not running and camera_mode)
        self.lingbot_map_capture_btn.setEnabled(not running and camera_mode)
        if running:
            self.lingbot_map_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.lingbot_map_status_label.setStyleSheet("")

    def _resolve_lingbot_map_input(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, str]]:
        """返回 (image_folder, video_path)。"""
        scene = str(self.lingbot_map_scene_combo.currentData() or "courthouse")
        if scene in ("courthouse", "university", "loop"):
            folder = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT, "example", scene)
            if not _eai_platform_bootstrap.os.path.isdir(folder):
                self._append_lingbot_map_log(f"[ERROR] 示例目录不存在: {folder}")
                return None
            return folder, ""
        if scene == "camera":
            folder = self._lingbot_map_camera_dir()
            n = len(
                [
                    n
                    for n in _eai_platform_bootstrap.os.listdir(folder)
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if n < 8:
                self._append_lingbot_map_log(
                    f"[ERROR] 相机采集帧数过少（{n}），请先点「拍一帧」至少 8 张"
                )
                self._on_lingbot_map_status("采集帧不足")
                return None
            return folder, ""
        path = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(self.lingbot_map_path_edit.text().strip())
        )
        if scene == "video":
            if not _eai_platform_bootstrap.os.path.isfile(path):
                self._append_lingbot_map_log(f"[ERROR] 视频不存在: {path}")
                return None
            return "", path
        if not _eai_platform_bootstrap.os.path.isdir(path):
            self._append_lingbot_map_log(f"[ERROR] 图像目录不存在: {path}")
            return None
        return path, ""

    def _append_lingbot_video_log(self, line: str) -> None:
        if not hasattr(self, "lingbot_video_log_edit"):
            return
        self.lingbot_video_log_edit.append(line)
        bar = self.lingbot_video_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_lingbot_video_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_lingbot_video_launcher", None)
            and self._lingbot_video_launcher.is_running()
        )
        mode = str(self.lingbot_video_mode_combo.currentData() or "t2v")
        ti2v = mode == "ti2v"
        custom_preset = str(self.lingbot_video_preset_combo.currentData() or "") == "custom"
        self.lingbot_video_run_btn.setEnabled(not running)
        self.lingbot_video_stop_btn.setEnabled(running)
        self.lingbot_video_open_btn.setEnabled(
            bool(getattr(self, "_lingbot_video_last_output", ""))
            and _eai_platform_bootstrap.os.path.isfile(self._lingbot_video_last_output)
        )
        for w in (
            self.lingbot_video_mode_combo,
            self.lingbot_video_case_combo,
            self.lingbot_video_preset_combo,
            self.lingbot_video_prompt_edit,
            self.lingbot_video_prompt_browse_btn,
            self.lingbot_video_model_edit,
            self.lingbot_video_model_browse_btn,
            self.lingbot_video_seed_spin,
            self.lingbot_video_refiner_check,
        ):
            w.setEnabled(not running)
        for w in (
            self.lingbot_video_h_spin,
            self.lingbot_video_w_spin,
            self.lingbot_video_frames_spin,
            self.lingbot_video_steps_spin,
        ):
            w.setEnabled(not running and custom_preset)
        self.lingbot_video_image_edit.setEnabled(not running and ti2v)
        self.lingbot_video_image_browse_btn.setEnabled(not running and ti2v)
        self.lingbot_video_cam_btn.setEnabled(not running and ti2v)
        if hasattr(self, "lingbot_video_topic_combo"):
            self.lingbot_video_topic_combo.setEnabled(not running and ti2v)
            self.lingbot_video_topic_refresh_btn.setEnabled(not running and ti2v)
        if running:
            self.lingbot_video_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.lingbot_video_status_label.setStyleSheet("")

    def _populate_lingbot_video_cases(self) -> None:
        self.lingbot_video_case_combo.blockSignals(True)
        self.lingbot_video_case_combo.clear()
        mode = str(self.lingbot_video_mode_combo.currentData() or "t2v")
        manifest_path = _eai_platform_bootstrap.os.path.join(
            _eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT, "assets", "cases", "manifest.json"
        )
        cases: _eai_platform_bootstrap.List[dict] = []
        if _eai_platform_bootstrap.os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = _eai_platform_bootstrap.json.load(f)
                cases = [
                    c
                    for c in (data.get("examples") or [])
                    if str(c.get("mode") or "") == mode
                ]
            except Exception as exc:
                self._append_lingbot_video_log(f"[WARN] 读取 manifest 失败: {exc}")
        if not cases:
            cases_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT, "assets", "cases", mode)
            if _eai_platform_bootstrap.os.path.isdir(cases_dir):
                for name in sorted(_eai_platform_bootstrap.os.listdir(cases_dir)):
                    prompt = _eai_platform_bootstrap.os.path.join(cases_dir, name, "prompt.json")
                    if _eai_platform_bootstrap.os.path.isfile(prompt):
                        item = {"mode": mode, "name": name, "prompt_json": prompt}
                        img = _eai_platform_bootstrap.os.path.join(cases_dir, name, "first_frame.png")
                        if _eai_platform_bootstrap.os.path.isfile(img):
                            item["image"] = img
                        cases.append(item)
        for case in cases:
            name = str(case.get("name") or "case")
            prompt = str(case.get("prompt_json") or "")
            if not _eai_platform_bootstrap.os.path.isabs(prompt):
                prompt = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT, prompt)
            image = str(case.get("image") or "")
            if image and not _eai_platform_bootstrap.os.path.isabs(image):
                image = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT, image)
            self.lingbot_video_case_combo.addItem(
                name, {"prompt_json": prompt, "image": image}
            )
        self.lingbot_video_case_combo.addItem("自定义路径…", {"prompt_json": "", "image": ""})
        self.lingbot_video_case_combo.blockSignals(False)
        self._on_lingbot_video_case_changed()

    def _append_lingbot_world_log(self, line: str) -> None:
        if not hasattr(self, "lingbot_world_log_edit"):
            return
        self.lingbot_world_log_edit.append(line)
        bar = self.lingbot_world_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_lingbot_world_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_lingbot_world_launcher", None)
            and self._lingbot_world_launcher.is_running()
        )
        custom = str(self.lingbot_world_example_combo.currentData() or "") == "custom"
        self.lingbot_world_run_btn.setEnabled(not running)
        self.lingbot_world_stop_btn.setEnabled(running)
        self.lingbot_world_play_btn.setEnabled(
            bool(getattr(self, "_lingbot_world_last_output", ""))
            and _eai_platform_bootstrap.os.path.isfile(self._lingbot_world_last_output)
        )
        for w in (
            self.lingbot_world_example_combo,
            self.lingbot_world_size_combo,
            self.lingbot_world_frames_spin,
            self.lingbot_world_seed_spin,
            self.lingbot_world_t5_cpu_check,
            self.lingbot_world_offload_check,
            self.lingbot_world_ckpt_edit,
            self.lingbot_world_ckpt_browse_btn,
            self.lingbot_world_prompt_edit,
            self.lingbot_world_cam_btn,
        ):
            w.setEnabled(not running)
        if hasattr(self, "lingbot_world_topic_combo"):
            self.lingbot_world_topic_combo.setEnabled(not running)
            self.lingbot_world_topic_refresh_btn.setEnabled(not running)
        self.lingbot_world_image_edit.setEnabled(not running and custom)
        self.lingbot_world_image_browse_btn.setEnabled(not running and custom)
        self.lingbot_world_action_edit.setEnabled(not running and custom)
        self.lingbot_world_action_browse_btn.setEnabled(not running and custom)
        if running:
            self.lingbot_world_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.lingbot_world_status_label.setStyleSheet("")

    def _set_sam3_result_text(self, text: str) -> None:
        self.sam3_result_edit.setPlainText(text)

