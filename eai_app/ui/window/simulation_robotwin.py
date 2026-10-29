"""ui.window.simulation robotwin — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationRobotwinMixin:
    def _on_robotwin_rynnvalue_hud_toggled(self, checked: bool) -> None:
        self._on_sim_rynnvalue_hud_toggled(checked)

    def _on_robotwin_rynnvalue_url_changed(self) -> None:
        self._on_sim_rynnvalue_url_changed()

    def _on_robotwin_rynnvalue_status_tick(self) -> None:
        self._on_sim_rynnvalue_status_tick()

    def _on_robotwin_rynnvalue_probe_clicked(self) -> None:
        self._on_sim_rynnvalue_probe_clicked()

    def _on_robotwin_rynnvalue_deploy_clicked(self) -> None:
        self._on_sim_rynnvalue_deploy_clicked()

    def _robotwin_ckpt_text(self) -> str:
        if hasattr(self, "robotwin_ckpt_edit"):
            return self.robotwin_ckpt_edit.text().strip()
        return _eai_configuration_tabs.ROBOTWIN_CKPT_DEFAULT

    def _robotwin_config_id(self) -> str:
        if hasattr(self, "robotwin_config_combo"):
            return str(
                self.robotwin_config_combo.currentData() or _eai_configuration_tabs.ROBOTWIN_CONFIG_DEFAULT
            )
        return _eai_configuration_tabs.ROBOTWIN_CONFIG_DEFAULT

    def _robotwin_task_id(self) -> str:
        if hasattr(self, "robotwin_task_combo"):
            return str(self.robotwin_task_combo.currentData() or _eai_configuration_tabs.ROBOTWIN_TASK_DEFAULT)
        return _eai_configuration_tabs.ROBOTWIN_TASK_DEFAULT

    def _on_robotwin_rlinf_browse(self) -> None:
        start = ""
        if hasattr(self, "robotwin_rlinf_edit"):
            start = self.robotwin_rlinf_edit.text().strip()
        start = start or _eai_configuration_tabs.RLINF_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RLinf 仓库", start)
        if selected and hasattr(self, "robotwin_rlinf_edit"):
            self.robotwin_rlinf_edit.setText(selected)

    def _on_robotwin_root_browse(self) -> None:
        start = ""
        if hasattr(self, "robotwin_root_edit"):
            start = self.robotwin_root_edit.text().strip()
        start = start or str(_eai_configuration_models.ROBOTWIN_ROOT_DEFAULT)
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RoboTwin 仓库", start)
        if selected and hasattr(self, "robotwin_root_edit"):
            self.robotwin_root_edit.setText(selected)

    def _on_robotwin_assets_browse(self) -> None:
        start = ""
        if hasattr(self, "robotwin_assets_edit"):
            start = self.robotwin_assets_edit.text().strip()
        start = start or _eai_configuration_models.resolve_robotwin_assets()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RoboTwin assets", start)
        if selected and hasattr(self, "robotwin_assets_edit"):
            self.robotwin_assets_edit.setText(selected)

    def _on_robotwin_ckpt_browse(self) -> None:
        start = self._robotwin_ckpt_text() or str(_eai_configuration_models.EAI_DIR)
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 RoboTwin checkpoint 目录", start
        )
        if selected and hasattr(self, "robotwin_ckpt_edit"):
            self.robotwin_ckpt_edit.setText(selected)

    def _on_robotwin_run_clicked(self) -> None:
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        if self._sim_backend_id() != "robotwin":
            self._on_mujoco_status("请先将仿真后端切换为 RoboTwin")
            return
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status(
                "MuJoCo / MolmoSpaces / Arena / LIBERO / RoboTwin 正在运行，请先停止"
            )
            return

        self.mujoco_log_edit.clear()
        task = self._robotwin_task_id()
        cfg = self._robotwin_config_id()
        ckpt = self._robotwin_ckpt_text()
        rlinf = ""
        if hasattr(self, "robotwin_rlinf_edit"):
            rlinf = self.robotwin_rlinf_edit.text().strip()
        rt = ""
        if hasattr(self, "robotwin_root_edit"):
            rt = self.robotwin_root_edit.text().strip()
        assets = ""
        if hasattr(self, "robotwin_assets_edit"):
            assets = self.robotwin_assets_edit.text().strip()
        gpu = 0
        num_envs = 1
        max_steps = 200
        if hasattr(self, "robotwin_gpu_spin"):
            gpu = int(self.robotwin_gpu_spin.value())
        if hasattr(self, "robotwin_num_envs_spin"):
            num_envs = int(self.robotwin_num_envs_spin.value())
        if hasattr(self, "robotwin_max_steps_spin"):
            max_steps = int(self.robotwin_max_steps_spin.value())
        render_gui = True
        if hasattr(self, "robotwin_render_gui_check"):
            render_gui = bool(self.robotwin_render_gui_check.isChecked())
        _rv = self._sim_rynnvalue_hud_settings()
        rynnvalue_live_hud = bool(_rv["enabled"])
        rynnvalue_refresh_sec = float(_rv["refresh_sec"])
        rynnvalue_server_url = str(_rv["server_url"])
        rynnvalue_num_frames = int(_rv["num_frames"])
        if render_gui and num_envs > 1:
            self._append_mujoco_log(
                f"[robotwin] 图形窗口开启：并行环境 {num_envs} -> 1"
            )
            num_envs = 1
            if hasattr(self, "robotwin_num_envs_spin"):
                self.robotwin_num_envs_spin.setValue(1)

        if _eai_configuration_models.is_vpp2_policy(cfg):
            self._append_mujoco_log(
                "[robotwin/VPP2] serve_only：启动 VPP2 policy server；"
                "RoboTwin 无官方 closed-loop adapter"
            )
            self._launch_vpp2_backend(
                "robotwin", task=task, label="robotwin-vpp2"
            )
            return

        if _eai_configuration_models.is_lingbot_va_policy(cfg):
            ok_va, va_msg = _eai_configuration_models.probe_lingbot_va()
            self._append_mujoco_log(f"[robotwin/LingBot-VA] {va_msg}")
            if not ok_va:
                self._append_mujoco_log(_eai_configuration_models.lingbot_va_install_hint())
                self._on_mujoco_status("LingBot-VA 未就绪（见日志）")
                return
            try:
                argv, cwd, env_extra = _eai_configuration_models.build_robotwin_va_eval_argv(
                    task=task,
                    checkpoint=ckpt,
                    config_name=_eai_configuration_models.va_config_for_backend("robotwin"),
                    python_bin=_eai_configuration_models.resolve_lingbot_va_python(),
                    gpu=str(gpu),
                    start_server=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._on_mujoco_status(f"无法组装 LingBot-VA RoboTwin 命令: {exc}")
                return
            clean_env = {k: v for k, v in env_extra.items() if v is not None}
            clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
            clean_env.update(self._sim_bridge_child_env())
            self._append_mujoco_log(
                f"[robotwin/LingBot-VA] task={task} gpu={gpu} ckpt={ckpt}"
            )
            self._expand_workspace_panel()
            self._mujoco_launcher.start_cwd_command(
                cwd=str(cwd),
                argv=argv,
                label=f"robotwin-va {task}",
                env_extra=clean_env,
            )
            self._update_mujoco_ui()
            return

        py = _eai_configuration_models.resolve_robotwin_python()
        ok_deps, dep_msg = _eai_configuration_models.probe_robotwin_deps(
            py,
            rlinf_root=None,
            robotwin_root=_eai_configuration_models.resolve_robotwin_root(rt),
        )
        # Soft-fail: still allow launch so logs show install hints; hard-block only
        # when python itself is missing.
        if not ok_deps:
            self._append_mujoco_log(f"[robotwin] 依赖探测失败: {dep_msg}")
            self._append_mujoco_log(_eai_configuration_models.robotwin_install_hint(py))
            # Continue if VectorEnv missing but user may have env almost ready;
            # only abort when sapien/torch clearly missing.
            if "MISSING" in str(dep_msg):
                self._on_mujoco_status("RoboTwin 依赖未就绪（见日志）")
                return
            self._append_mujoco_log(
                "[robotwin] 继续启动（VectorEnv 可能需切到 RLinf_support）…"
            )

        if not ckpt:
            self._on_mujoco_status("请填写 RoboTwin checkpoint 路径")
            return
        if not _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.expanduser(ckpt)):
            self._append_mujoco_log(
                f"[robotwin] checkpoint 目录不存在，尝试从 HF 拉取: "
                f"{_eai_configuration_models.ROBOTWIN_CKPT_HF_DEFAULT}"
            )
            ok_dl, ckpt_or_err = _eai_configuration_models.ensure_robotwin_checkpoint(
                ckpt,
                cfg,
                hf_repo=_eai_configuration_models.ROBOTWIN_CKPT_HF_DEFAULT,
                log=self._append_mujoco_log,
            )
            if not ok_dl:
                self._append_mujoco_log(f"[robotwin] {ckpt_or_err}")
                self._on_mujoco_status(f"checkpoint 不可用: {ckpt}")
                return
            ckpt = ckpt_or_err
            if hasattr(self, "robotwin_ckpt_edit"):
                self.robotwin_ckpt_edit.setText(ckpt)

        assets = _eai_configuration_models.resolve_robotwin_assets(assets, robotwin_root=_eai_configuration_models.resolve_robotwin_root(rt))
        if hasattr(self, "robotwin_assets_edit"):
            self.robotwin_assets_edit.setText(assets)
        assets_objects = _eai_platform_bootstrap.os.path.join(assets, "assets", "objects")
        if not _eai_platform_bootstrap.os.path.isdir(assets_objects):
            self._append_mujoco_log(
                f"[robotwin] 资产未就绪: {assets_objects}\n"
                f"  请在 RoboTwin 根目录执行: bash script/_download_assets.sh"
            )
            self._on_mujoco_status("RoboTwin assets 未下载（见日志）")
            return

        if rynnvalue_live_hud:
            self._append_mujoco_log(
                f"[robotwin] 确认 RynnValue reward_server（{rynnvalue_server_url}）…"
            )
            ok_rv, msg_rv = self._ensure_rynnvalue_reward_server_ready(
                start_if_needed=True,
                wait_s=900.0,
                prefix="robotwin",
            )
            if not ok_rv:
                self._append_mujoco_log(f"[robotwin] RynnValue 预检失败: {msg_rv}")
                self._on_mujoco_status(f"RynnValue 未就绪: {msg_rv}")
                return
            self._append_mujoco_log(f"[robotwin] RynnValue 预检通过: {msg_rv}")

        try:
            argv, cwd, env_extra = _eai_configuration_models.build_robotwin_eval_argv(
                task=task,
                policy_config=cfg,
                checkpoint=ckpt,
                assets_path=assets,
                num_envs=num_envs,
                max_episode_steps=max_steps,
                python_bin=py,
                rlinf_root=rlinf,
                robotwin_root=rt,
                gpu=str(gpu),
                render_gui=render_gui,
                rynnvalue_live_hud=rynnvalue_live_hud,
                rynnvalue_server_url=rynnvalue_server_url,
                rynnvalue_refresh_sec=rynnvalue_refresh_sec,
                rynnvalue_num_frames=rynnvalue_num_frames,
                rynnvalue_show_window=bool(_rv["show_window"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._on_mujoco_status(f"无法组装 RoboTwin 命令: {exc}")
            return

        self._append_mujoco_log(
            f"[robotwin] task={task} config={cfg} gpu={gpu} envs={num_envs}"
            f" render_gui={render_gui} rynnvalue_hud={rynnvalue_live_hud}"
        )
        self._append_mujoco_log(f"[robotwin] python={py}")
        self._append_mujoco_log(f"[robotwin] ckpt={ckpt}")
        self._append_mujoco_log(f"[robotwin] assets={assets or _eai_configuration_models.resolve_robotwin_assets()}")
        self._append_mujoco_log(f"[robotwin] out={_eai_configuration_models.ROBOTWIN_EVAL_OUTPUT_DIR}")
        if rynnvalue_live_hud:
            self._append_mujoco_log(
                f"[robotwin] RynnValue Live HUD "
                f"url={rynnvalue_server_url} refresh={rynnvalue_refresh_sec}s "
                f"num_frames={rynnvalue_num_frames}"
            )
            self._expand_workspace_panel()
            self._set_rynnvalue_workspace_hud_active(True)
        if render_gui and not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_mujoco_log(
                "[robotwin] 警告: DISPLAY 为空，SAPIEN 图形窗口可能无法弹出"
            )
        clean_env = {k: v for k, v in env_extra.items() if v is not None}
        if rynnvalue_live_hud:
            clean_env.update(self._rynnvalue_live_hud_child_env("robotwin"))
        clean_env.update(self._sim_bridge_child_env())
        tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
        prev = clean_env.get("PYTHONPATH") or ""
        clean_env["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
            [tools] + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p != tools]
        )
        self._mujoco_launcher.start_cwd_command(
            cwd=str(cwd),
            argv=argv,
            label=f"robotwin {task}",
            env_extra=clean_env,
        )
        self._update_mujoco_ui()

    def _on_robotwin_viewer_only_clicked(self) -> None:
        """SAPIEN viewer without OpenPI — GUI teleop via gui_robot_cmd.json."""
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        if self._sim_backend_id() != "robotwin":
            self._on_mujoco_status("请先将仿真后端切换为 RoboTwin")
            return
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("其它仿真后端正在运行，请先停止")
            return
        self.mujoco_log_edit.clear()
        task = self._robotwin_task_id()
        rt = ""
        if hasattr(self, "robotwin_root_edit"):
            rt = self.robotwin_root_edit.text().strip()
        assets = ""
        if hasattr(self, "robotwin_assets_edit"):
            assets = self.robotwin_assets_edit.text().strip()
        py = _eai_configuration_models.resolve_robotwin_python()
        try:
            argv, cwd, env_extra = _eai_configuration_models.build_robotwin_viewer_argv(
                task=task,
                python_bin=py,
                robotwin_root=rt,
                assets_path=assets,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_mujoco_status(f"无法组装 RoboTwin viewer: {exc}")
            return
        clean_env = {k: v for k, v in env_extra.items() if v is not None}
        clean_env.update(self._sim_bridge_child_env())
        tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
        prev = clean_env.get("PYTHONPATH") or ""
        clean_env["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
            [tools] + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p != tools]
        )
        self._append_mujoco_log(
            f"[robotwin] 只开界面 task={task}；「手臂/手」文本指挥可用"
        )
        self._mujoco_launcher.start_cwd_command(
            cwd=str(cwd),
            argv=argv,
            label=f"robotwin-viewer {task}",
            env_extra=clean_env,
        )
        self._update_mujoco_ui()

