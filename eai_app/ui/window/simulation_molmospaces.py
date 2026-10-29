"""ui.window.simulation molmospaces — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationMolmospacesMixin:
    def _molmospaces_selected_robot(self) -> str:
        """Return --robot id from the MolmoSpaces 本体 combo (default droid)."""
        if hasattr(self, "molmospaces_robot_combo"):
            rid = self.molmospaces_robot_combo.currentData()
            if rid:
                return str(rid)
        return "droid"

    def _molmospaces_viewer_cam_choice(self) -> _eai_platform_bootstrap.Tuple[str, str, str]:
        """Return ``(first_person, cam_name, human_label)`` from the 视角 combo.

        Default is front third-person (robot + manipulation target overview).
        ``head`` uses ``head_camera_rgb`` first-person (arms + target).
        """
        cam_id = "front"
        if hasattr(self, "molmospaces_viewer_cam_combo"):
            cam_id = str(self.molmospaces_viewer_cam_combo.currentData() or "front")
        if cam_id == "head":
            return (
                "1",
                "head_camera_rgb",
                "头部第一视角 (head_camera_rgb)",
            )
        return ("0", "front", "正面第三视角 (front)")

    def _molmospaces_viewer_cam_env(self) -> _eai_platform_bootstrap.Dict[str, str]:
        """Env vars for ``run_molmospaces_eval.py`` viewer camera framing."""
        viewer_fp, viewer_cam, _label = self._molmospaces_viewer_cam_choice()
        env: _eai_platform_bootstrap.Dict[str, str] = {
            "MOLMOSPACES_VIEWER_FIRST_PERSON": viewer_fp,
            "MOLMOSPACES_VIEWER_CAM": viewer_cam,
        }
        if viewer_fp == "0":
            env["MOLMOSPACES_VIEWER_DISTANCE"] = _eai_configuration_tabs.MOLMOSPACES_VIEWER_FRONT_DISTANCE
            env["MOLMOSPACES_VIEWER_ELEVATION"] = _eai_configuration_tabs.MOLMOSPACES_VIEWER_FRONT_ELEVATION
        return env

    def _on_molmospaces_root_browse(self) -> None:
        cur = self._resolve_molmospaces_root()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 MolmoSpaces 仓库", cur
        )
        if selected and hasattr(self, "molmospaces_root_edit"):
            self.molmospaces_root_edit.setText(selected)
            self._refresh_molmospaces_tasks()

    def _on_molmospaces_policy_changed(self, *_args) -> None:
        policy = ""
        if hasattr(self, "molmospaces_policy_combo"):
            policy = str(self.molmospaces_policy_combo.currentData() or "")
        backend = ""
        if hasattr(self, "_sim_backend_id"):
            try:
                backend = str(self._sim_backend_id() or "")
            except Exception:
                backend = ""
        show_pi = (
            _eai_configuration_models.is_pi_policy_id(policy)
            and (not _eai_configuration_models.is_lingbot_va_policy(policy))
            and (not _eai_configuration_models.is_vpp2_policy(policy))
        )
        # On Psi-MolmoSpaces always keep the deploy row/status visible so switching
        # to Pick Planner / Dummy does not make the controls disappear.
        show_psi = _eai_adapters_models_discovery.needs_psi_remote_deploy(policy) or backend == "psi_molmospaces"
        if hasattr(self, "molmospaces_pi_row"):
            self.molmospaces_pi_row.setVisible(show_pi)
        if hasattr(self, "molmospaces_pi_status_label"):
            self.molmospaces_pi_status_label.setVisible(show_pi)
        if hasattr(self, "molmospaces_psi_row"):
            self.molmospaces_psi_row.setVisible(show_psi)
        if hasattr(self, "molmospaces_psi_status_label"):
            self.molmospaces_psi_status_label.setVisible(show_psi)
        if show_psi and hasattr(self, "molmospaces_psi_ckpt_label"):
            mode = _eai_adapters_models_discovery.psi_remote_mode_for_policy(policy)
            labels = {
                "wam82": "WAM模型",
                "native6d": "Native6D模型",
                "psi_policy": "Psi模型",
            }
            self.molmospaces_psi_ckpt_label.setText(labels.get(mode, "Psi/WAM模型"))
            # Auto-fill the Songzhe WAM-finetune default when switching to WAM,
            # or when the current path is empty / clearly a wrong LLM artifact.
            if mode == "wam82" and hasattr(self, "molmospaces_psi_ckpt_edit"):
                cur = self.molmospaces_psi_ckpt_edit.text().strip()
                cur_l = cur.replace("\\", "/").lower()
                need_default = (not cur) or cur.endswith("scheduler.pt") or (
                    "qwen" in cur_l and "wam" not in cur_l
                )
                if need_default and _eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT:
                    self.molmospaces_psi_ckpt_edit.setText(_eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT)
        self._update_molmospaces_pi_deploy_ui()
        self._update_molmospaces_psi_deploy_ui()

    def _molmospaces_pi_ckpt_text(self) -> str:
        if hasattr(self, "molmospaces_pi_ckpt_edit"):
            raw = self.molmospaces_pi_ckpt_edit.text().strip()
            if raw:
                return raw
        return _eai_configuration_tabs.MOLMOSPACES_PI_CKPT_DEFAULT

    def _on_molmospaces_pi_ckpt_browse(self) -> None:
        cur = self._molmospaces_pi_ckpt_text()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) if cur else ""
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 OpenPI / Pi checkpoint 目录", start
        )
        if selected and hasattr(self, "molmospaces_pi_ckpt_edit"):
            self.molmospaces_pi_ckpt_edit.setText(selected)

    def _on_molmospaces_pi_probe_clicked(self) -> None:
        ok, msg = _eai_configuration_models.probe_pi_server(
            _eai_configuration_models.PI_SERVER_HOST_DEFAULT, _eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT, timeout_s=5.0
        )
        self._append_mujoco_log(f"[pi] 验证: {msg}")
        if ok:
            self._on_mujoco_status(f"Pi 服务可用: {msg}")
        else:
            self._on_mujoco_status(f"Pi 服务不可用: {msg}")
        self._update_molmospaces_pi_deploy_ui()

    def _on_molmospaces_pi_start_clicked(self) -> None:
        try:
            self._append_mujoco_log("[pi] 正在部署启动…")
            ok, msg = self._ensure_pi_policy_ready(start_if_needed=True, wait_s=900.0)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            self._append_mujoco_log(f"[pi] 部署异常: {msg}")
            self._on_mujoco_status(f"Pi 部署失败: {msg}")
            self._update_molmospaces_pi_deploy_ui()
            return
        self._update_molmospaces_pi_deploy_ui()
        if ok:
            self._on_mujoco_status(f"Pi 已部署: {msg}")
        else:
            self._on_mujoco_status(f"Pi 部署失败: {msg}")

    def _on_molmospaces_pi_stop_clicked(self) -> None:
        if hasattr(self, "_pi_policy_launcher") and self._pi_policy_launcher.is_running():
            self._append_mujoco_log("[pi] 正在停止 UI 启动的 serve_policy…")
            self._pi_policy_launcher.stop()
        msg = _eai_configuration_models.stop_pi_server_on_port(
            _eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT, log=self._append_mujoco_log
        )
        self._append_mujoco_log(f"[pi] {msg}")
        self._update_molmospaces_pi_deploy_ui()
        self._on_mujoco_status(msg)

    def _molmospaces_selected_policy_id(self) -> str:
        if hasattr(self, "molmospaces_policy_combo"):
            return str(self.molmospaces_policy_combo.currentData() or "")
        return ""

    def _molmospaces_psi_mode(self) -> str:
        return _eai_adapters_models_discovery.psi_remote_mode_for_policy(self._molmospaces_selected_policy_id())

    def _molmospaces_psi_port(self) -> int:
        return int(_eai_adapters_models_discovery.psi_remote_default_port(self._molmospaces_psi_mode() or "psi_policy"))

    def _molmospaces_psi_ckpt_text(self) -> str:
        if hasattr(self, "molmospaces_psi_ckpt_edit"):
            return self.molmospaces_psi_ckpt_edit.text().strip()
        return _eai_configuration_tabs.PSI_REMOTE_CKPT_DEFAULT

    def _on_molmospaces_psi_ckpt_browse(self) -> None:
        cur = self._molmospaces_psi_ckpt_text()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) if cur else ""
        selected, _filter = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 Psi / WAM checkpoint (.pt)",
            start,
            "PyTorch checkpoint (*.pt);;All files (*)",
        )
        if selected and hasattr(self, "molmospaces_psi_ckpt_edit"):
            self.molmospaces_psi_ckpt_edit.setText(selected)

    def _on_molmospaces_psi_probe_clicked(self) -> None:
        mode = self._molmospaces_psi_mode()
        if not mode:
            self._on_mujoco_status("当前策略不需要 Psi/WAM 部署")
            return
        host, port = "localhost", self._molmospaces_psi_port()
        # Fast path: avoid long subprocess probe that freezes the GUI.
        if not _eai_configuration_models.port_listening(host, port, timeout=0.3):
            self._append_mujoco_log(
                f"[psi-remote/{mode}] 验证: ws://{host}:{port} 未监听"
            )
            self._on_mujoco_status(f"Psi/WAM 未就绪: :{port} 未监听")
            self._update_molmospaces_psi_deploy_ui()
            return
        py = _eai_adapters_models_discovery.resolve_psi_remote_python(_eai_adapters_models_discovery.resolve_psi_remote_policy_root())
        ok, msg = _eai_adapters_models_discovery.probe_psi_remote_server(mode, host, port, python_bin=py, timeout_s=3.0)
        self._append_mujoco_log(f"[psi-remote/{mode}] 验证: {msg}")
        self._on_mujoco_status(msg if ok else f"Psi/WAM 不可用: {msg}")
        self._update_molmospaces_psi_deploy_ui()

    def _on_molmospaces_psi_start_clicked(self) -> None:
        try:
            self._append_mujoco_log("[psi-remote] 正在部署启动…")
            self._psi_remote_last_error = ""
            self._psi_remote_user_stopped = False
            port = self._molmospaces_psi_port()
            mode = self._molmospaces_psi_mode()
            listening = _eai_configuration_models.port_listening("localhost", port, timeout=0.2)
            launcher_busy = bool(
                getattr(self, "_psi_remote_launcher", None)
                and self._psi_remote_launcher.is_running()
            )
            # Only force-clean a *stale listening* WAM. Do not kill a healthy
            # in-flight load just because :port is not up yet.
            force = bool(
                mode == "wam82"
                and listening
                and _eai_adapters_models_discovery.wam82_server_needs_restart(port)
            )
            if mode == "wam82" and (not listening) and launcher_busy:
                self._append_mujoco_log(
                    f"[psi-remote/{mode}] 服务仍在加载（:{port} 未监听），不重复拉起"
                )
                self._update_molmospaces_psi_deploy_ui()
                self._on_mujoco_status(f"WAM 加载中（:{port}），请等待「已就绪」")
                return
            ok, msg = self._start_psi_remote_server_async(force_restart=force)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            self._psi_remote_last_error = msg
            self._append_mujoco_log(f"[psi-remote] 部署异常: {msg}")
            self._on_mujoco_status(f"Psi/WAM 部署失败: {msg}")
            self._update_molmospaces_psi_deploy_ui()
            return
        if ok:
            self._psi_remote_last_error = ""
        else:
            self._psi_remote_last_error = msg
            self._append_mujoco_log(f"[psi-remote] {msg}")
        self._update_molmospaces_psi_deploy_ui()
        self._on_mujoco_status(msg if ok else f"Psi/WAM 部署失败: {msg}")

    def _on_molmospaces_psi_stop_clicked(self) -> None:
        mode = self._molmospaces_psi_mode() or "psi_policy"
        port = self._molmospaces_psi_port()
        self._psi_remote_last_error = ""
        self._psi_remote_user_stopped = True
        if hasattr(self, "_psi_remote_launcher") and self._psi_remote_launcher.is_running():
            self._append_mujoco_log(f"[psi-remote/{mode}] 正在停止 UI 启动的服务…")
            self._psi_remote_launcher.stop()
        msg = _eai_configuration_models.stop_pi_server_on_port(port, log=self._append_mujoco_log)
        self._append_mujoco_log(f"[psi-remote/{mode}] {msg}")
        self._update_molmospaces_psi_deploy_ui()
        self._on_mujoco_status(msg)

    def _on_molmospaces_run_clicked(self, *, viewer_only: bool = False) -> None:
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        backend = self._sim_backend_id()
        if not self._is_spaces_sim_backend(backend):
            self._on_mujoco_status(
                "请先将仿真后端切换为 MolmoSpaces / Psi-MolmoSpaces"
            )
            return
        # Psi「只开界面」：独立路径，强制用任务 combo 的 benchmark 开场景。
        if viewer_only and backend == "psi_molmospaces":
            self._start_psi_molmospaces_viewer_only()
            return
        spaces_tag = (
            "psi_molmospaces" if backend == "psi_molmospaces" else "molmospaces"
        )
        spaces_label = (
            "Psi-MolmoSpaces" if backend == "psi_molmospaces" else "MolmoSpaces"
        )
        root = self._resolve_molmospaces_root()
        # 只开界面：其它 MolmoSpaces 仍走快速演示 --viewer。
        if viewer_only:
            recipe = "quick_viewer"
            self._append_mujoco_log(
                f"[{spaces_tag}] 只开界面：快速演示 --viewer，不调用模型"
            )
        else:
            recipe = "quick_viewer"
            if hasattr(self, "molmospaces_recipe_combo"):
                recipe = str(
                    self.molmospaces_recipe_combo.currentData() or "quick_viewer"
                )
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self._on_mujoco_status(f"{spaces_label} 目录不存在: {root}")
            return
        if recipe == "open_dir":
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(root))
            self._append_mujoco_log(f"已打开目录: {root}")
            return
        if recipe == "open_terminal":
            for term, args in (
                ("xfce4-terminal", ["--working-directory", root]),
                ("x-terminal-emulator", []),
                ("gnome-terminal", [f"--working-directory={root}"]),
                ("xterm", ["-e", f"cd {_eai_platform_bootstrap.shlex.quote(root)}; exec $SHELL"]),
            ):
                if _eai_platform_bootstrap.shutil.which(term):
                    if term == "x-terminal-emulator":
                        ok = _eai_platform_bootstrap.QProcess.startDetached(
                            term,
                            ["-e", f"bash -lc 'cd {_eai_platform_bootstrap.shlex.quote(root)}; exec bash'"],
                        )
                    else:
                        ok = _eai_platform_bootstrap.QProcess.startDetached(term, args)
                    if ok:
                        self._append_mujoco_log(f"已打开终端 ({term}): {root}")
                        return
            self._on_mujoco_status("未找到可用终端模拟器")
            return

        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("MuJoCo / MolmoSpaces 正在运行，请先停止")
            return

        py = self._resolve_molmospaces_python()
        if recipe == "install_deps":
            self.mujoco_log_edit.clear()
            self._append_mujoco_log(
                f"[molmospaces] 安装依赖: molmospaces-resources=={_eai_configuration_tabs.MOLMOSPACES_RESOURCES_MIN}"
            )
            self._mujoco_launcher.start_cwd_command(
                cwd=root,
                argv=[
                    py,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    f"molmospaces-resources=={_eai_configuration_tabs.MOLMOSPACES_RESOURCES_MIN}",
                ],
                label="molmospaces install_deps",
                env_extra={"PYTHONUNBUFFERED": "1"},
            )
            self._update_mujoco_ui()
            return

        pipeline = _eai_platform_bootstrap.os.path.join(root, "scripts", "datagen", "run_pipeline.py")
        repaired: _eai_platform_bootstrap.List[str] = []
        if recipe == "eval_task":
            robot_id = self._molmospaces_selected_robot()
            # Psi-MolmoSpaces fork has first-class Psi R1; local MJCF is only a
            # MolmoSpaces quick_viewer fallback and should not block eval there.
            if (
                robot_id in _eai_configuration_tabs.MOLMOSPACES_LOCAL_MJCF_ROBOTS
                and backend != "psi_molmospaces"
            ):
                self._on_mujoco_status(
                    f"本体「{robot_id}」仅支持快速演示（本地 MJCF），请改选内置机型再评测"
                )
                return
            task = None
            if hasattr(self, "molmospaces_task_combo"):
                task = self.molmospaces_task_combo.currentData()
            if not isinstance(task, dict) or not task.get("path"):
                self._on_mujoco_status(f"请选择一个 {spaces_label} 评测任务")
                return
            bench = str(task["path"])
            if not _eai_adapters_models_discovery.molmospaces_benchmark_ready(bench):
                self._append_mujoco_log(
                    f"[{spaces_tag}] 任务数据未就绪: {bench}\n"
                    "  在仓库执行一次资源初始化后再刷新:\n"
                    f"  {py} -m molmo_spaces.molmo_spaces_constants"
                )
                self._on_mujoco_status(f"任务未下载: {task.get('label')}")
                return
            if backend == "psi_molmospaces":
                ok_r1, r1_msg = _eai_adapters_models_discovery.ensure_psi_r1_robot_assets(root)
                if not ok_r1:
                    self._append_mujoco_log(f"[psi_molmospaces] {r1_msg}")
                    self._on_mujoco_status("缺少 Psi R1 机器人资源")
                    return
                self._append_mujoco_log(
                    f"[psi_molmospaces] psi_r1_ruiyan assets={r1_msg}"
                )
            policy = "molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig"
            if hasattr(self, "molmospaces_policy_combo"):
                policy = str(
                    self.molmospaces_policy_combo.currentData() or policy
                )
            # 只开界面：不启远程模型 / VA / VPP2；Psi 任务必须用 Psi R1 配置加载场景。
            if viewer_only and backend == "psi_molmospaces":
                needs_remote = (
                    _eai_configuration_models.is_lingbot_va_policy(policy)
                    or _eai_configuration_models.is_vpp2_policy(policy)
                    or _eai_adapters_models_discovery.needs_psi_remote_deploy(policy)
                    or _eai_configuration_models.is_pi_policy_id(policy)
                )
                franka_only = (
                    "DummyBenchmarkEvalConfig" in policy
                    or "TeleopPolicyEvalConfig" in policy
                )
                if needs_remote or franka_only:
                    self._append_mujoco_log(
                        f"[{spaces_tag}] 只开界面：策略 "
                        f"{policy.split(':')[-1]} 不适合无模型开窗，"
                        f"改用 {_eai_configuration_tabs.PSI_MOLMOSPACES_DEFAULT_POLICY.split(':')[-1]}"
                    )
                    policy = _eai_configuration_tabs.PSI_MOLMOSPACES_DEFAULT_POLICY
            if (not viewer_only) and _eai_configuration_models.is_lingbot_va_policy(policy):
                self.mujoco_log_edit.clear()
                self._append_mujoco_log(
                    f"[{spaces_tag}/LingBot-VA] 启动 wan_va（serve_only）；"
                    f"{spaces_label} 尚无官方 VA closed-loop adapter"
                )
                self._launch_lingbot_va_backend(
                    spaces_tag,
                    task=str(task.get("id") or ""),
                    label=f"{spaces_tag}-va",
                )
                return
            if (not viewer_only) and _eai_configuration_models.is_vpp2_policy(policy):
                self.mujoco_log_edit.clear()
                self._append_mujoco_log(
                    f"[{spaces_tag}/VPP2] 启动 policy server（serve_only）；"
                    f"{spaces_label} 尚无官方 VPP2 closed-loop adapter"
                )
                self._launch_vpp2_backend(
                    spaces_tag,
                    task=str(task.get("id") or ""),
                    label=f"{spaces_tag}-vpp2",
                )
                return
            pi_ckpt = ""
            psi_remote_ckpt = ""
            psi_remote_mode = ""
            if (not viewer_only) and _eai_adapters_models_discovery.needs_psi_remote_deploy(policy):
                # Psi Policy / Native6D / WAM: deploy via GUI row (not openpi Pi).
                self.mujoco_log_edit.clear()
                psi_remote_mode = _eai_adapters_models_discovery.psi_remote_mode_for_policy(policy)
                self._append_mujoco_log(
                    f"[psi-remote/{psi_remote_mode}] 选用远程策略：确认服务可调用…"
                )
                ok, msg = self._ensure_psi_remote_policy_ready(
                    start_if_needed=True, wait_s=900.0
                )
                if not ok:
                    self._append_mujoco_log(f"[psi-remote/{psi_remote_mode}] 预检失败: {msg}")
                    self._on_mujoco_status(f"Psi/WAM 未就绪: {msg}")
                    return
                psi_remote_ckpt = _eai_adapters_models_discovery.resolve_psi_remote_checkpoint(
                    self._molmospaces_psi_ckpt_text()
                )
                self._append_mujoco_log(
                    f"[psi-remote/{psi_remote_mode}] 预检通过: {msg}"
                )
            elif (not viewer_only) and _eai_configuration_models.is_pi_policy_id(policy):
                self.mujoco_log_edit.clear()
                self._append_mujoco_log("[pi] 选用 Pi：先确认模型/服务可调用…")
                ok, msg = self._ensure_pi_policy_ready(
                    start_if_needed=True, wait_s=900.0
                )
                if not ok:
                    self._append_mujoco_log(f"[pi] 预检失败: {msg}")
                    self._on_mujoco_status(f"Pi 未就绪: {msg}")
                    return
                pi_ckpt = _eai_configuration_models.resolve_pi_checkpoint(self._molmospaces_pi_ckpt_text()).path
                self._append_mujoco_log(f"[pi] 预检通过: {msg}")
            wrapper = _eai_platform_bootstrap.os.path.join(
                str(PROJECT_ROOT),
                "tools",
                "run_molmospaces_eval.py",
            )
            if not _eai_platform_bootstrap.os.path.isfile(wrapper):
                self._on_mujoco_status(f"未找到: {wrapper}")
                return
            out_dir = (
                _eai_configuration_tabs.PSI_MOLMOSPACES_EVAL_OUTPUT_DIR
                if backend == "psi_molmospaces"
                else _eai_configuration_tabs.MOLMOSPACES_EVAL_OUTPUT_DIR
            )
            _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
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
            ]
            if pi_ckpt:
                argv.extend(["--checkpoint_path", pi_ckpt])
            elif psi_remote_ckpt:
                argv.extend(["--checkpoint_path", psi_remote_ckpt])
            max_eps = 1
            if hasattr(self, "molmospaces_episodes_spin"):
                max_eps = int(self.molmospaces_episodes_spin.value())
            if max_eps > 0:
                argv.extend(["--max_episodes", str(max_eps)])
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
            label = (
                f"{spaces_tag} viewer {task.get('id')}"
                if viewer_only
                else f"{spaces_tag} eval {task.get('id')}"
            )
            if not _eai_configuration_models.is_pi_policy_id(policy) and not _eai_adapters_models_discovery.needs_psi_remote_deploy(policy):
                self.mujoco_log_edit.clear()
            self._append_mujoco_log(f"[{spaces_tag}] task={task.get('label')}")
            self._append_mujoco_log(f"[{spaces_tag}] benchmark={bench}")
            self._append_mujoco_log(f"[{spaces_tag}] policy={policy}")
            if pi_ckpt:
                self._append_mujoco_log(f"[{spaces_tag}] checkpoint={pi_ckpt}")
            if psi_remote_ckpt:
                self._append_mujoco_log(
                    f"[{spaces_tag}] psi/wam checkpoint={psi_remote_ckpt}"
                )
            self._append_mujoco_log(
                f"[{spaces_tag}] max_episodes={'全部' if max_eps <= 0 else max_eps}"
            )
            self._append_mujoco_log(
                f"[{spaces_tag}] 结束后保留窗口，点「停止」或关闭窗口再退出"
            )
            if backend == "psi_molmospaces":
                _vp_fp, _vp_cam, vp_label = self._molmospaces_viewer_cam_choice()
                self._append_mujoco_log(
                    f"[{spaces_tag}] viewer: 官方 MuJoCo GLFW · {vp_label}"
                )
            else:
                self._append_mujoco_log(
                    f"[{spaces_tag}] viewer cam: "
                    f"{self._molmospaces_viewer_cam_choice()[2]}"
                )
            # GLFW path also needs RYNNVALUE_LIVE_HUD (viewer feed is hooked in
            # molmospaces_viewer_cam); previously viewer_only skipped HUD env.
            hud_env = (
                {}
                if not self._sim_rynnvalue_hud_enabled()
                else self._rynnvalue_live_hud_child_env(spaces_tag)
            )
            if hud_env and not self._ensure_live_hud_reward_or_abort(spaces_tag):
                return
            # Psi：官方 MuJoCo GLFW；视角由上方「视角」下拉决定（默认正面第三视角）。
            # 其它 Spaces 默认仍可 FORCE_EGL。
            force_egl = "0" if backend == "psi_molmospaces" else "1"
            env_extra = {
                "MUJOCO_GL": "glfw",
                "MOLMOSPACES_ROOT": root,
                "RYNNVALUE_HUD_BACKEND": spaces_tag,
                "MOLMOSPACES_EVAL_VIEWER": "1",
                **self._molmospaces_viewer_cam_env(),
                "MOLMOSPACES_FORCE_EGL": force_egl,
                # Keep last frame until GUI「停止」or window close.
                "MOLMOSPACES_VIEWER_KEEP_OPEN": "1",
                "EAI_DIR": _eai_configuration_models.EAI_DIR,
                "OPENPI_DATA_HOME": _eai_platform_bootstrap.os.environ.get("OPENPI_DATA_HOME")
                or _eai_configuration_models.OPENPI_DATA_HOME_DEFAULT,
                # Prefer checkout root over conda editable install (often upstream molmospaces).
                "PYTHONPATH": root,
                "PYTHONNOUSERSITE": "1",
                **_eai_adapters_models_discovery.molmospaces_cache_env(),
            }
            if psi_remote_mode:
                port = self._molmospaces_psi_port()
                env_extra["PSI_POLICY_SERVER_HOST"] = "127.0.0.1"
                env_extra["PSI_POLICY_SERVER_PORT"] = str(port)
                if psi_remote_mode == "wam82":
                    env_extra["WAM_POLICY_SERVER_HOST"] = "127.0.0.1"
                    env_extra["WAM_POLICY_SERVER_PORT"] = str(port)
                    env_extra["WAM_PSI_R1_ACTION_CHUNK_SIZE"] = "16"
                    env_extra["MLSPACES_JSON_EVAL_SCENE_VARIANT"] = _eai_platform_bootstrap.os.environ.get(
                        "MLSPACES_JSON_EVAL_SCENE_VARIANT", "base"
                    )
            if viewer_only:
                env_extra["MOLMOSPACES_VIEWER_ONLY"] = "1"
            tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
            prev = env_extra.get("PYTHONPATH") or ""
            env_extra["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
                [root, tools]
                + [
                    p
                    for p in prev.split(_eai_platform_bootstrap.os.pathsep)
                    if p and p not in (root, tools)
                ]
            )
            if hud_env:
                env_extra.update(hud_env)
                self._expand_workspace_panel()
                self._append_mujoco_log(
                    f"[{spaces_tag}] RynnValue Live HUD "
                    f"url={hud_env.get('RYNNVALUE_SERVER_URL')}"
                )
            self._mujoco_launcher.start_cwd_command(
                cwd=root,
                argv=argv,
                label=label,
                env_extra=env_extra,
            )
            self._update_mujoco_ui()
            return
        if recipe == "quick_viewer":
            robot_id = self._molmospaces_selected_robot()
            local_mjcf = _eai_configuration_tabs.MOLMOSPACES_LOCAL_MJCF_ROBOTS.get(robot_id, "")
            # Psi backend must never open bare MJCF — that drops the task scene.
            if local_mjcf and backend == "psi_molmospaces":
                self._append_mujoco_log(
                    "[psi_molmospaces] 已拦截裸 MJCF quick_viewer，改用所选任务场景"
                )
                self._start_psi_molmospaces_viewer_only()
                return
            if local_mjcf:
                mjcf = local_mjcf
                if not _eai_platform_bootstrap.os.path.isfile(mjcf):
                    # Re-resolve in case candidates were checked before assets existed.
                    if robot_id == "psi_r1":
                        for cand in _eai_configuration_models.PSI_R1_MODEL_XML_CANDIDATES:
                            if _eai_platform_bootstrap.os.path.isfile(cand):
                                mjcf = cand
                                break
                if not _eai_platform_bootstrap.os.path.isfile(mjcf):
                    self._append_mujoco_log(f"[ERROR] Psi R1 MJCF 不存在: {mjcf}")
                    self._on_mujoco_status("未找到 Psi R1 model.xml")
                    return
                self.mujoco_log_edit.clear()
                self._append_mujoco_log(
                    f"[molmospaces] 本体={robot_id} → 本地 MuJoCo viewer"
                )
                self._append_mujoco_log(f"[molmospaces] mjcf={mjcf}")
                self._mujoco_launcher.start(
                    mjcf_path=mjcf,
                    mujoco_root=self.mujoco_root_edit.text().strip()
                    if hasattr(self, "mujoco_root_edit")
                    else "",
                    python_bin=self.mujoco_python_edit.text().strip()
                    if hasattr(self, "mujoco_python_edit")
                    else "",
                    mode=str(
                        self.mujoco_mode_combo.currentData()
                        if hasattr(self, "mujoco_mode_combo")
                        else "auto"
                    )
                    or "auto",
                    install=False,
                )
                self._update_mujoco_ui()
                return
            if not _eai_platform_bootstrap.os.path.isfile(pipeline):
                self._on_mujoco_status(f"未找到: {pipeline}")
                return
            self._append_mujoco_log(
                f"[{spaces_tag}] 本体={robot_id} → MolmoSpaces pipeline --viewer\n"
                "  首次会建/校验资源与 LMDB 索引，可能要 1–3 分钟才弹出窗口；"
                "请看下方日志进度，勿重复点击。"
            )
            repaired = self._repair_molmospaces_cache_manifest()
            ok, msg = self._check_molmospaces_resources(py)
            if not ok:
                self._append_mujoco_log(f"[molmospaces] 依赖检查失败: {msg}")
                self._append_mujoco_log(
                    f"[molmospaces] 请在动作里选「安装/更新依赖」，或手动:\n"
                    f"  {py} -m pip install -U "
                    f"molmospaces-resources=={_eai_configuration_tabs.MOLMOSPACES_RESOURCES_MIN}"
                )
                self._on_mujoco_status(f"依赖不足: {msg}")
                return
            # Prefer VirtualGL→NVIDIA GLFW; wrapper only falls back to EGL+Tk if no GLX.
            wrapper = _eai_platform_bootstrap.os.path.join(
                str(PROJECT_ROOT),
                "tools",
                "run_molmospaces_pipeline.py",
            )
            if not _eai_platform_bootstrap.os.path.isfile(wrapper):
                self._on_mujoco_status(f"未找到: {wrapper}")
                return
            argv = [
                py,
                wrapper,
                "--viewer",
                "--seed",
                "1",
                "--robot",
                robot_id,
            ]
            task_type = _eai_configuration_tabs.MOLMOSPACES_QUICK_VIEWER_TASK.get(robot_id, "pick")
            if task_type != "pick":
                argv.extend(["--task_type", task_type])
            # Stash for the shared log block below.
            self._molmospaces_last_task_type = task_type
            vgl = (
                "/usr/bin/vglrun"
                if _eai_platform_bootstrap.os.path.isfile("/usr/bin/vglrun")
                else "/opt/VirtualGL/bin/vglrun"
                if _eai_platform_bootstrap.os.path.isfile("/opt/VirtualGL/bin/vglrun")
                else ""
            )
            if vgl:
                argv = [vgl, "-d", _eai_platform_bootstrap.os.environ.get("MUJOCO_VGL_DEVICE") or "egl", *argv]
            label = "molmospaces quick_viewer"
            resources_msg = msg
        else:
            self._on_mujoco_status(f"未知 MolmoSpaces 动作: {recipe}")
            return

        self.mujoco_log_edit.clear()
        if repaired:
            self._append_mujoco_log(
                "[molmospaces] 已修复 cache manifest: " + ", ".join(repaired)
            )
        self._append_mujoco_log(f"[molmospaces] root={root}")
        self._append_mujoco_log(f"[molmospaces] python={py}")
        self._append_mujoco_log(f"[molmospaces] robot={self._molmospaces_selected_robot()}")
        task_type = getattr(self, "_molmospaces_last_task_type", None) or (
            _eai_configuration_tabs.MOLMOSPACES_QUICK_VIEWER_TASK.get(self._molmospaces_selected_robot(), "pick")
        )
        self._append_mujoco_log(f"[molmospaces] task_type={task_type}")
        if self._molmospaces_selected_robot() == "rby1" and task_type == "nav_to_obj":
            self._append_mujoco_log(
                "[molmospaces] 提示: RBY1 无 parallel IK，PickPlanner 不可用，已改导航演示"
            )
        self._append_mujoco_log(f"[molmospaces] resources={resources_msg}")
        self._append_mujoco_log(
            "[molmospaces] viewer: VirtualGL→NVIDIA when available, else EGL+Tk"
        )
        _vp_fp, _vp_cam, vp_label = self._molmospaces_viewer_cam_choice()
        self._append_mujoco_log(f"[molmospaces] viewer cam: {vp_label}")
        self._append_mujoco_log(
            "[molmospaces] 结束后保留窗口，点「停止」或关闭窗口再退出"
        )
        if not self._ensure_live_hud_reward_or_abort(spaces_tag):
            return
        hud_env = self._rynnvalue_live_hud_child_env(spaces_tag)
        env_extra = {
            "MUJOCO_GL": "glfw",
            "MOLMOSPACES_ROOT": root,
            "RYNNVALUE_HUD_BACKEND": spaces_tag,
            **self._molmospaces_viewer_cam_env(),
            "MOLMOSPACES_FORCE_EGL": "1",
            "MOLMOSPACES_VIEWER_KEEP_OPEN": "1",
            "EAI_DIR": _eai_configuration_models.EAI_DIR,
            # Prefer checkout root over conda editable install (often upstream molmospaces).
            "PYTHONPATH": root,
            "PYTHONNOUSERSITE": "1",
            **_eai_adapters_models_discovery.molmospaces_cache_env(),
            **self._sim_bridge_child_env(),
        }
        tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
        prev = env_extra.get("PYTHONPATH") or ""
        env_extra["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(
            [root, tools]
            + [
                p
                for p in prev.split(_eai_platform_bootstrap.os.pathsep)
                if p and p not in (root, tools)
            ]
        )
        if hud_env:
            env_extra.update(hud_env)
            self._expand_workspace_panel()
            self._append_mujoco_log(
                f"[{spaces_tag}] RynnValue Live HUD "
                f"url={hud_env.get('RYNNVALUE_SERVER_URL')}"
            )
        self._mujoco_launcher.start_cwd_command(
            cwd=root,
            argv=argv,
            label=label,
            env_extra=env_extra,
        )
        self._update_mujoco_ui()

