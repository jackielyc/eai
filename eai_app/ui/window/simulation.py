"""ui.window.simulation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_simulation_robodojo = module_ref('eai_app.adapters.simulation.robodojo')
_eai_adapters_simulation_robodojo_discovery = module_ref('eai_app.adapters.simulation.robodojo_discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

class SimulationMixin:
    def _on_sim_bridge_browse_clicked(self) -> None:
        current = self.sim_bridge_dir_edit.text().strip()
        initial = current if current and _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_models.EAI_DIR
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 Isaac 共享帧目录", initial)
        if selected:
            self.sim_bridge_dir_edit.setText(selected)
            self._refresh_sim_frame_status()

    def _on_sim_bridge_start_clicked(self) -> None:
        if getattr(self, "_mujoco_launcher", None) and self._mujoco_launcher.is_running():
            self._append_sim_log("MuJoCo / MolmoSpaces 正在运行，请先停止后再启动相机桥")
            return
        if self._sim_backend_id() != "isaac":
            self._append_sim_log("相机桥仅用于 Isaac 后端")
            return
        self._sim_bridge_launcher.start(
            self.sim_bridge_dir_edit.text(),
            hz=float(self.sim_bridge_hz_spin.value()),
        )
        self._update_sim_bridge_ui()

    def _on_sim_bridge_stop_clicked(self) -> None:
        self._sim_bridge_launcher.stop()
        self._update_sim_bridge_ui()

    def _on_sim_clear_log_clicked(self) -> None:
        self.sim_log_edit.clear()

    def _on_sim_eval_policy_changed(self, _index: int = 0) -> None:
        policy = str(self.sim_eval_policy_combo.currentData() or "")
        no_policy = not policy
        running = self._sim_eval_launcher.is_running()
        self.sim_eval_ckpt_edit.setEnabled(not running and not no_policy)
        self.sim_eval_action_combo.setEnabled(not running and not no_policy)
        if hasattr(self, "sim_eval_bundled_check"):
            self.sim_eval_bundled_check.setEnabled(not running and not no_policy)
        if no_policy:
            return
        if "starVLA" in policy:
            cur = self.sim_eval_ckpt_edit.text().strip()
            if (
                (not cur)
                or cur == "demo"
                or (not _eai_configuration_policies._looks_like_starvla_ckpt(cur))
                or (not _eai_platform_bootstrap.os.path.isabs(cur))
            ):
                self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_starvla_ckpt_text())
            else:
                self._set_sim_eval_ckpt_path(cur)
            idx = self.sim_eval_action_combo.findData("joint")
            if idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(idx)
        elif _eai_configuration_models.is_lingbot_va_policy(policy):
            cur = self.sim_eval_ckpt_edit.text().strip()
            if (
                (not cur)
                or cur in ("demo", "hf_qwenpi_v3", "joint2b_s100000")
                or (not _eai_platform_bootstrap.os.path.isabs(cur))
            ):
                self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_lingbot_va_ckpt_text())
            else:
                self._set_sim_eval_ckpt_path(cur)
            idx = self.sim_eval_action_combo.findData("joint")
            if idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(idx)
        elif _eai_configuration_models.is_vpp2_policy(policy):
            cur = self.sim_eval_ckpt_edit.text().strip()
            if (not cur) or cur in ("demo", "hf_qwenpi_v3") or "RoboDojo-cotrain" in cur:
                self.sim_eval_ckpt_edit.setText(
                    _eai_platform_bootstrap.os.environ.get("VPP2_CKPT", "joint2b_s100000")
                )
            idx = self.sim_eval_action_combo.findData("ee")
            if idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(idx)
        elif _eai_configuration_policies._is_pi05_robodojo_policy(policy):
            cur = self.sim_eval_ckpt_edit.text().strip()
            if (
                (not cur)
                or cur == "demo"
                or (not _eai_platform_bootstrap.os.path.isabs(cur))
                or "hf_" in cur
                or "RoboDojo-cotrain" in cur
                or _eai_configuration_policies._ckpt_owned_by_lingbot_va_not_pi05(cur)
                or _eai_configuration_policies._looks_like_starvla_ckpt(cur)
            ):
                self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_pi05_ckpt_text())
            else:
                self._set_sim_eval_ckpt_path(cur)
            idx = self.sim_eval_action_combo.findData("joint")
            if idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(idx)
        else:
            cur = self.sim_eval_ckpt_edit.text().strip()
            if (
                (not cur)
                or "hf_" in cur
                or "RoboDojo-cotrain" in cur
                or _eai_configuration_policies._ckpt_owned_by_lingbot_va_not_pi05(cur)
            ):
                self.sim_eval_ckpt_edit.setText("demo")
            idx = self.sim_eval_action_combo.findData("ee")
            if idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(idx)

    def _sim_bridge_dir_resolved(self) -> str:
        edit = getattr(self, "sim_bridge_dir_edit", None)
        raw = edit.text().strip() if edit is not None else ""
        return _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(raw or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT)
        )

    def _sim_bridge_fresh_cam_keys(self, *, max_age_s: float = 120.0) -> set:
        """共享目录里有新鲜彩色帧的相机 key（如 cam_head）。无文件/过期不计入。"""
        bridge_dir = self._sim_bridge_dir_resolved()
        if not _eai_platform_bootstrap.os.path.isdir(bridge_dir):
            return set()
        now = _eai_platform_bootstrap.time.time()
        present: set = set()
        for cam_key, _topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}.npy")
            stamp_path = npy_path + ".stamp"
            try:
                mtime = _eai_platform_bootstrap.os.path.getmtime(
                    stamp_path if _eai_platform_bootstrap.os.path.isfile(stamp_path) else npy_path
                )
            except OSError:
                continue
            if now - mtime > float(max_age_s):
                continue
            if not _eai_platform_bootstrap.os.path.isfile(npy_path):
                continue
            present.add(cam_key)
        return present

    def _sim_bridge_fresh_depth_keys(self, *, max_age_s: float = 120.0) -> set:
        """有新鲜 depth npy 且对应彩色也存在的相机 key。"""
        color_keys = self._sim_bridge_fresh_cam_keys(max_age_s=max_age_s)
        if not color_keys:
            return set()
        bridge_dir = self._sim_bridge_dir_resolved()
        now = _eai_platform_bootstrap.time.time()
        present: set = set()
        for cam_key, _topic in _eai_configuration_policies.SIM_PREVIEW_DEPTH_CAM_TOPICS:
            if cam_key not in color_keys:
                continue
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}_depth.npy")
            stamp_path = npy_path + ".stamp"
            try:
                mtime = _eai_platform_bootstrap.os.path.getmtime(
                    stamp_path if _eai_platform_bootstrap.os.path.isfile(stamp_path) else npy_path
                )
            except OSError:
                continue
            if now - mtime > float(max_age_s):
                continue
            if not _eai_platform_bootstrap.os.path.isfile(npy_path):
                continue
            present.add(cam_key)
        return present

    def _sim_bridge_newest_age_s(self) -> _eai_platform_bootstrap.Optional[float]:
        """共享目录里最新一帧的年龄（秒）；无文件则 None。"""
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                self.sim_bridge_dir_edit.text().strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        if not _eai_platform_bootstrap.os.path.isdir(bridge_dir):
            return None
        now = _eai_platform_bootstrap.time.time()
        newest: _eai_platform_bootstrap.Optional[float] = None
        for key in _eai_configuration_policies.ISAAC_CAM_BRIDGE_KEYS:
            for base in (key, f"{key}_depth"):
                npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{base}.npy")
                stamp_path = npy_path + ".stamp"
                try:
                    mtime = _eai_platform_bootstrap.os.path.getmtime(
                        stamp_path if _eai_platform_bootstrap.os.path.isfile(stamp_path) else npy_path
                    )
                except OSError:
                    continue
                age = max(0.0, now - mtime)
                newest = age if newest is None else min(newest, age)
        return newest

    def _on_sim_eval_start_clicked(self) -> None:
        """Backward-compatible alias for「启动评测」."""
        self._start_sim_eval(viewer_only=False)

    def _on_sim_policy_deploy_running_changed(self, running: bool = False) -> None:
        if running:
            if hasattr(self, "_sim_policy_ready_timer"):
                self._sim_policy_ready_timer.start()
        else:
            if hasattr(self, "_sim_policy_ready_timer"):
                self._sim_policy_ready_timer.stop()
        self._update_sim_policy_deploy_ui()

    def _on_sim_policy_deploy_start_clicked(self) -> None:
        policy_data = self.sim_eval_policy_combo.currentData()
        policy_dir = "" if policy_data is None else str(policy_data)
        if not policy_dir.strip():
            self._append_sim_log("[策略] 请先选择策略（不能为「无」）再部署")
            return
        task = str(
            self.sim_eval_task_combo.currentData()
            or self.sim_eval_task_combo.currentText()
            or "build_tower"
        ).strip() or "build_tower"
        ckpt = self.sim_eval_ckpt_edit.text().strip() or "demo"
        action_type = str(self.sim_eval_action_combo.currentData() or "ee")
        policy_env = _eai_configuration_policies.ROBODOJO_ENV_DEFAULT
        if "starVLA" in policy_dir and _eai_platform_bootstrap.os.path.isdir(_eai_configuration_policies.ROBODOJO_STARVLA_ENV_DEFAULT):
            policy_env = _eai_configuration_policies.ROBODOJO_STARVLA_ENV_DEFAULT
        elif _eai_configuration_models.is_lingbot_va_policy(policy_dir):
            va_env = _eai_configuration_models.resolve_lingbot_va_conda_env(_eai_configuration_policies.ROBODOJO_LINGBOT_VA_ENV_DEFAULT)
            if va_env and _eai_platform_bootstrap.os.path.isdir(va_env):
                policy_env = va_env
        elif _eai_configuration_policies._is_pi05_robodojo_policy(policy_dir):
            if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_policies.ROBODOJO_OPENPI_ENV_DEFAULT):
                policy_env = _eai_configuration_policies.ROBODOJO_OPENPI_ENV_DEFAULT
        elif _eai_configuration_models.is_vpp2_policy(policy_dir):
            vpp2_env = _eai_configuration_models.resolve_vpp2_conda_env(_eai_configuration_policies.ROBODOJO_VPP2_ENV_DEFAULT)
            if vpp2_env and _eai_platform_bootstrap.os.path.isdir(vpp2_env):
                policy_env = vpp2_env
        root = _eai_configuration_policies.ROBODOJO_ROOT_DEFAULT
        robodojo_sh = _eai_platform_bootstrap.os.path.join(root, "scripts", "robodojo.sh")
        if not _eai_platform_bootstrap.os.path.isfile(robodojo_sh):
            self._append_sim_log(f"[策略] 未找到 {robodojo_sh}")
            return
        host = _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT
        port = (
            _eai_configuration_models.DEFAULT_VPP2_PORT
            if _eai_configuration_models.is_vpp2_policy(policy_dir)
            else _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
        )

        if (
            hasattr(self, "_robodojo_policy_launcher")
            and self._robodojo_policy_launcher.is_running()
        ):
            info_owned = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(host, port)
            if _eai_adapters_simulation_robodojo_discovery.robodojo_policy_matches_listener(
                info_owned, policy_dir=policy_dir, ckpt=ckpt, action_type=action_type
            ) or not info_owned.get("listening"):
                self._append_sim_log(
                    f"[策略] 同一模型已在运行（{host}:{port}），跳过重复部署"
                )
            else:
                pol = info_owned.get("policy_name") or "?"
                ck = info_owned.get("ckpt_name") or "?"
                self._append_sim_log(
                    f"[策略] 已有其它策略在跑（{pol}/{ck}），"
                    "与当前选择不一致。请先「部署停止」再部署。"
                )
            self._update_sim_policy_deploy_ui()
            return

        info = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(host, port)
        if info.get("listening") and _eai_adapters_simulation_robodojo_discovery.robodojo_policy_matches_listener(
            info, policy_dir=policy_dir, ckpt=ckpt, action_type=action_type
        ):
            pol = info.get("policy_name") or _eai_platform_bootstrap.os.path.basename(policy_dir.rstrip("/"))
            ck = info.get("ckpt_name") or ckpt
            act = info.get("action_type") or action_type
            pid = info.get("pid")
            self._append_sim_log(
                f"[策略] 已正常运行，跳过部署："
                f"{host}:{port} policy={pol} ckpt={ck} action={act}"
                + (f" pid={pid}" if pid else "")
            )
            self._update_sim_policy_deploy_ui()
            return
        if info.get("listening") and info.get("is_policy_server"):
            pol = info.get("policy_name") or "?"
            ck = info.get("ckpt_name") or "?"
            act = info.get("action_type") or "?"
            self._append_sim_log(
                f"[策略] {host}:{port} 已有其它策略在跑（{pol}/{ck} action={act}），"
                f"与当前选择 "
                f"{_eai_platform_bootstrap.os.path.basename(policy_dir.rstrip('/'))}/{ckpt} action={action_type} "
                "不一致。请先「部署停止」再部署，或改选与线上一致的策略/ckpt/action。"
            )
            self._update_sim_policy_deploy_ui()
            return
        if info.get("listening"):
            self._append_sim_log(
                f"[策略] {host}:{port} 已被其它进程占用（非 XPolicyLab 策略服务），"
                "请先释放端口或改 ROBODOJO_POLICY_PORT 后再部署"
            )
            self._update_sim_policy_deploy_ui()
            return

        exports = [
            "unset PYTHONPATH PYTHONHOME || true",
            "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1",
            "export CONDA_ROOT=/home/psibot/miniconda3",
            "export CONDA_EXE=/home/psibot/miniconda3/bin/conda",
            "export PATH=/home/psibot/miniconda3/bin:\"${PATH}\"",
            "export TMPDIR=/dev/shm/robodojo_tmp TEMP=/dev/shm/robodojo_tmp TMP=/dev/shm/robodojo_tmp",
            "mkdir -p /dev/shm/robodojo_tmp",
            f"export ROBODOJO_POLICY_ENV={_eai_platform_bootstrap.shlex.quote(policy_env)}",
        ]
        if "starVLA" in policy_dir:
            if not _eai_configuration_policies._looks_like_starvla_ckpt(ckpt):
                ckpt = _eai_configuration_policies._default_starvla_ckpt_text()
                self.sim_eval_ckpt_edit.setText(ckpt)
                self._append_sim_log(
                    f"[策略] starVLA ckpt 已纠正为 {ckpt}（对应 "
                    f"{_eai_configuration_policies.STARVLA_PI_V3_CKPT_FILE_DEFAULT}）"
                )
            variant = _eai_configuration_policies.resolve_starvla_hf_variant(ckpt)
            ckpt_path = _eai_configuration_policies.resolve_starvla_ckpt_file(variant, ckpt=ckpt)
            if not ckpt_path:
                self._append_sim_log(
                    f"[策略] 未找到 starVLA 权重 variant={variant}，"
                    f"期望 {_eai_configuration_policies.STARVLA_PI_V3_CKPT_FILE_DEFAULT}"
                )
                return
            exports.extend(
                [
                    f"export STARVLA_ENV={_eai_platform_bootstrap.shlex.quote(policy_env)}",
                    f"export STARVLA_HF_ROOT={_eai_platform_bootstrap.shlex.quote(_eai_configuration_policies.ROBODOJO_STARVLA_HF_ROOT_DEFAULT)}",
                    f"export STARVLA_BASE_VLM={_eai_platform_bootstrap.shlex.quote(_eai_configuration_policies.ROBODOJO_STARVLA_BASE_VLM_DEFAULT)}",
                    f"export STARVLA_CKPT_PATH={_eai_platform_bootstrap.shlex.quote(ckpt_path)}",
                    "export STARVLA_HF_VERIFY_ONLY=1",
                    "export STARVLA_HF_SKIP_WEIGHT_HASH=1",
                    "export STARVLA_INCLUDE_STATE=True",
                    "export STARVLA_UNNORM_KEY=arx_x5",
                ]
            )
            self._append_sim_log(
                f"[策略] STARVLA_CKPT_PATH={ckpt_path} STARVLA_ENV={policy_env}"
            )
        if _eai_configuration_models.is_lingbot_va_policy(policy_dir):
            base_model = _eai_configuration_models.resolve_lingbot_va_base_model(
                _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_BASE_MODEL_PATH", "")
            )
            if not base_model or not _eai_platform_bootstrap.os.path.isdir(base_model):
                self._append_sim_log(
                    "[策略] LingBot-VA 需要 LINGBOT_VA_BASE_MODEL_PATH；"
                    f"{_eai_configuration_models.lingbot_va_install_hint()}"
                )
                return
            va_ckpt = _eai_configuration_models.resolve_lingbot_va_checkpoint(ckpt)
            exports.extend(
                [
                    f"export LINGBOT_VA_BASE_MODEL_PATH={_eai_platform_bootstrap.shlex.quote(base_model)}",
                    f"export BASE_MODEL_PATH={_eai_platform_bootstrap.shlex.quote(base_model)}",
                    f"export LINGBOT_VA_CKPT={_eai_platform_bootstrap.shlex.quote(va_ckpt)}",
                    f"export CHECKPOINT_PATH={_eai_platform_bootstrap.shlex.quote(va_ckpt)}",
                    f"export CONFIG_NAME={_eai_platform_bootstrap.shlex.quote(_eai_configuration_models.va_config_for_backend('isaac'))}",
                    f"export LINGBOT_VA_VA_PORT={_eai_platform_bootstrap.shlex.quote(str(_eai_configuration_models.LINGBOT_VA_PORT_DEFAULT))}",
                ]
            )
            self._append_sim_log(
                f"[策略] LingBot-VA base={base_model} ckpt={va_ckpt}"
            )
        if _eai_configuration_models.is_vpp2_policy(policy_dir):
            try:
                ok_ad, ad_msg = _eai_configuration_models.ensure_vpp2_adapter(robodojo_root=root)
                self._append_sim_log(
                    f"[策略] VPP2 adapter {'OK' if ok_ad else 'NEED'} · {ad_msg}"
                )
            except Exception as exc:  # noqa: BLE001
                self._append_sim_log(f"[策略] VPP2 adapter: {exc}")
            bundle = _eai_configuration_models.resolve_vpp2_bundle(ckpt)
            wan = _eai_configuration_models.resolve_vpp2_wan_root()
            weights = str(_eai_configuration_models.resolve_vpp2_weights())
            ok_probe, probe_msg = _eai_configuration_models.probe_vpp2()
            if not ok_probe:
                self._append_sim_log(
                    f"[策略] VPP2 未就绪: {probe_msg}\n{_eai_configuration_models.vpp2_install_hint()}"
                )
                return
            exports.extend(
                [
                    f"export VPP2_WEIGHTS={_eai_platform_bootstrap.shlex.quote(weights)}",
                    f"export VPP2_BUNDLE={_eai_platform_bootstrap.shlex.quote(bundle)}",
                    f"export VPP2_WAN_ROOT={_eai_platform_bootstrap.shlex.quote(wan)}",
                    "export VPP2_TORCH_LOAD_MMAP=1",
                ]
            )
            # Prefer official vpp2.cli server (sets deploy.yml overrides).
            try:
                argv_v, cwd_v, env_v, note_v = _eai_configuration_models.build_backend_vpp2_argv(
                    "isaac",
                    python_bin=_eai_configuration_models.resolve_vpp2_python(conda_env=policy_env),
                    port=port,
                    gpu="0",
                )
                self._append_sim_log(f"[策略] {note_v}")
                clean_env = {k: v for k, v in env_v.items() if v is not None}
                self._robodojo_policy_launcher.start_cwd_command(
                    cwd=str(cwd_v),
                    argv=argv_v,
                    label="vpp2-server",
                    env_extra=clean_env,
                )
                self._update_sim_policy_deploy_ui()
                return
            except Exception as exc:  # noqa: BLE001
                self._append_sim_log(
                    f"[策略] VPP2 official server 失败，回退 robodojo.sh: {exc}"
                )
            self._append_sim_log(f"[策略] VPP2 bundle={bundle} wan={wan}")
        run_cmd = (
            f"exec bash {_eai_platform_bootstrap.shlex.quote(robodojo_sh)} server "
            f"--policy-dir {_eai_platform_bootstrap.shlex.quote(policy_dir)} "
            f"--task {_eai_platform_bootstrap.shlex.quote(task)} "
            f"--ckpt {_eai_platform_bootstrap.shlex.quote(ckpt)} "
            f"--policy-env {_eai_platform_bootstrap.shlex.quote(policy_env)} "
            f"--action-type {_eai_platform_bootstrap.shlex.quote(action_type)} "
            f"--policy-port {int(port)} "
            f"--bind-host {_eai_platform_bootstrap.shlex.quote(host)} "
            f"--policy-gpu 0 "
            f"--seed 0"
        )
        cmd = " && ".join(exports) + f" && cd {_eai_platform_bootstrap.shlex.quote(root)} && {run_cmd}"
        self._append_sim_log(
            f"[策略] 部署启动 {policy_dir} ckpt={ckpt} → {host}:{port}"
        )
        self._robodojo_policy_launcher.start_cwd_command(
            cwd=root,
            argv=["bash", "-lc", cmd],
            label=f"robodojo policy:{_eai_platform_bootstrap.os.path.basename(policy_dir.rstrip('/'))}",
            env_extra={
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": None,
                "PYTHONHOME": None,
            },
        )
        if hasattr(self, "_sim_policy_ready_timer"):
            self._sim_policy_ready_timer.start()
        self._update_sim_policy_deploy_ui()

    def _on_sim_policy_deploy_stop_clicked(self) -> None:
        self._append_sim_log("[策略] 正在停止部署…")
        if (
            hasattr(self, "_robodojo_policy_launcher")
            and self._robodojo_policy_launcher.is_running()
        ):
            self._robodojo_policy_launcher.stop()
        else:
            # 外部已就绪的策略服务：尽量结束监听端口上的进程树
            info = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(
                _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT, _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
            )
            pid = info.get("pid")
            if pid and info.get("is_policy_server"):
                try:
                    _eai_platform_bootstrap.os.kill(int(pid), _eai_platform_bootstrap.signal.SIGTERM)
                    self._append_sim_log(f"[策略] 已向 pid={pid} 发送 SIGTERM")
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    self._append_sim_log(f"[策略] 无法结束 pid={pid}: {exc}")
            else:
                self._append_sim_log("[策略] 没有本窗口管理的部署进程")
        self._update_sim_policy_deploy_ui()

    def _on_sim_eval_stop_run_clicked(self) -> None:
        self._append_sim_log("--- 用户点击停止评测（停止全部仿真）---")
        self._set_sim_paused(False, log=False)
        self._stop_isaac_rynnvalue_npy_hud()
        self._sim_eval_launcher.stop()
        if self._sim_eval_auto_bridge and self._sim_bridge_launcher.is_running():
            self._append_sim_log("--- 一并停止自动启动的相机桥 ---")
            self._sim_bridge_launcher.stop()
            self._sim_eval_auto_bridge = False
            self._update_sim_bridge_ui()
        self._update_sim_eval_run_ui()

    def _sim_paused_state(self) -> bool:
        if _eai_configuration_policies.is_sim_paused is None:
            return bool(getattr(self, "_sim_paused_ui", False))
        try:
            return bool(_eai_configuration_policies.is_sim_paused(self._sim_bridge_dir()))
        except Exception:
            return bool(getattr(self, "_sim_paused_ui", False))

    def _on_sim_pause_clicked(self) -> None:
        try:
            ext_isaac = bool(
                _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.isaac_sim_status().get("running")
            )
        except Exception:
            ext_isaac = False
        if not self._any_sim_process_running() and not ext_isaac:
            self._set_sim_paused(False, log=False)
            return
        # Prefer the clicked button's checked state; fall back to toggle.
        sender = self.sender()
        if sender is not None and hasattr(sender, "isChecked"):
            want_paused = bool(sender.isChecked())
        else:
            want_paused = not self._sim_paused_state()
        self._set_sim_paused(want_paused, log=True)

    def _sim_eval_root_path(self) -> str:
        return _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                self.sim_eval_root_edit.text().strip() or _eai_configuration_policies.ROBODOJO_EVAL_RESULT_ROOT_DEFAULT
            )
        )

    @staticmethod
    def _sim_eval_read_rate(result_json: str) -> str:
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            rate = data.get("success_rate", data.get("score"))
            if rate is None:
                return "?"
            return f"{float(rate):.2f}"
        except Exception:
            return "?"

    def _on_sim_eval_root_browse_clicked(self) -> None:
        current = self._sim_eval_root_path()
        initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_models.EAI_DIR
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RoboDojo 评测结果目录", initial)
        if selected:
            self.sim_eval_root_edit.setText(selected)
            self._refresh_sim_eval_tree()

    def _on_sim_eval_filter_changed(self, _text: str = "") -> None:
        self._refresh_sim_eval_tree()
        if self._sim_eval_outcome_keyword():
            self._refresh_sim_eval_mp4_by_outcome()

    def _sim_eval_outcome_keyword(self) -> str:
        data = self.sim_eval_outcome_combo.currentData()
        return str(data or "").strip().lower()

    def _on_sim_eval_outcome_changed(self, _index: int = 0) -> None:
        keyword = self._sim_eval_outcome_keyword()
        if not keyword:
            self._on_sim_eval_selection_changed(self.sim_eval_tree.currentItem(), None)
            return
        self._refresh_sim_eval_mp4_by_outcome()

    def _on_sim_eval_item_expanded(self, item: _eai_platform_bootstrap.QTreeWidgetItem) -> None:
        if item.childCount() == 1 and item.child(0).text(0) == "…":
            item.takeChild(0)
            self._fill_sim_eval_children(item)

    def _on_sim_eval_selection_changed(
        self, current: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QTreeWidgetItem], _previous: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QTreeWidgetItem]
    ) -> None:
        self.sim_eval_detail_edit.clear()
        outcome_kw = self._sim_eval_outcome_keyword()
        if not outcome_kw:
            self.sim_eval_files_list.clear()
        if current is None:
            return
        path = str(current.data(0, _eai_platform_bootstrap.Qt.UserRole) or "")
        if not path:
            return
        kind = str(current.data(0, _eai_platform_bootstrap.Qt.UserRole + 1) or "")
        result_json = _eai_platform_bootstrap.os.path.join(path, "_result.json")
        if kind != "run" and not _eai_platform_bootstrap.os.path.isfile(result_json):
            if not outcome_kw:
                self.sim_eval_detail_edit.setPlainText(path)
            return
        lines: _eai_platform_bootstrap.List[str] = [path, ""]
        if _eai_platform_bootstrap.os.path.isfile(result_json):
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    data = _eai_platform_bootstrap.json.load(f)
                lines.append(_eai_platform_bootstrap.json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as exc:
                lines.append(f"读取 _result.json 失败: {exc}")
        else:
            lines.append("(无 _result.json)")
        self.sim_eval_detail_edit.setPlainText("\n".join(lines))
        # 已按 success/failed 全局列出时，保留扫描结果，不按当前目录覆盖
        if outcome_kw:
            return
        try:
            files = sorted(_eai_platform_bootstrap.os.listdir(path))
        except OSError:
            files = []
        for name in files:
            if not name.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                continue
            item = _eai_platform_bootstrap.QListWidgetItem(name)
            item.setData(_eai_platform_bootstrap.Qt.UserRole, _eai_platform_bootstrap.os.path.join(path, name))
            self.sim_eval_files_list.addItem(item)
        if self.sim_eval_files_list.count() > 0:
            self.sim_eval_files_list.setCurrentRow(0)

    def _on_sim_eval_open_dir_clicked(self) -> None:
        item = self.sim_eval_tree.currentItem()
        path = ""
        if item is not None:
            path = str(item.data(0, _eai_platform_bootstrap.Qt.UserRole) or "")
        if not path or not _eai_platform_bootstrap.os.path.isdir(path):
            path = self._sim_eval_root_path()
        if not _eai_platform_bootstrap.os.path.isdir(path):
            self.status_bar.showMessage(f"目录不存在: {path}", 4000)
            return
        try:
            _eai_platform_bootstrap.subprocess.Popen(["xdg-open", path], start_new_session=True)
        except Exception as exc:
            self.status_bar.showMessage(f"打开目录失败: {exc}", 5000)

    def _on_sim_eval_open_video_clicked(self) -> None:
        item = self.sim_eval_files_list.currentItem()
        if item is None and self.sim_eval_files_list.count() > 0:
            self.sim_eval_files_list.setCurrentRow(0)
            item = self.sim_eval_files_list.currentItem()
        if item is None:
            self.status_bar.showMessage("请先在右侧列表选择一个视频", 3000)
            return
        path = str(item.data(_eai_platform_bootstrap.Qt.UserRole) or "")
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            self.status_bar.showMessage("视频文件不存在", 3000)
            return
        # 内置播放器（不依赖系统默认关联）
        try:
            dlg = _eai_ui_widgets_video.SimEvalVideoDialog(path, self)
            dlg.setAttribute(_eai_platform_bootstrap.Qt.WA_DeleteOnClose, True)
            dlg.show()
            self.status_bar.showMessage(f"播放: {_eai_platform_bootstrap.os.path.basename(path)}", 3000)
            return
        except Exception as exc:
            self.status_bar.showMessage(f"内置播放失败: {exc}", 4000)
        # 回退 ffplay / xdg-open
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
                self.status_bar.showMessage(f"已用 {cmd[0]} 打开视频", 3000)
                return
            except Exception:
                continue
        self.status_bar.showMessage("无法播放视频（内置与外部播放器均失败）", 5000)

    def _sim_backend_selector_keys(self) -> _eai_platform_bootstrap.List[str]:
        bar = getattr(self, "sim_backend_tab_bar", None)
        if bar is None:
            return []
        out: _eai_platform_bootstrap.List[str] = []
        for i in range(bar.count()):
            key = str(bar.tabData(i) or "").strip()
            if key:
                out.append(key)
        return out

    def _sim_backend_selector_key(self) -> str:
        """当前选中的后端选择条 key（扩展套件统一为 extended）。"""
        by_key = getattr(self, "_sim_backend_radio_by_key", None) or {}
        for key, radio in by_key.items():
            if radio is not None and radio.isChecked():
                return str(key)
        return "isaac"

    def _on_sim_backend_tab_changed(self, index: int) -> None:
        if getattr(self, "_sim_backend_tabs_guard", False):
            return
        bar = getattr(self, "sim_backend_tab_bar", None)
        if bar is None or index < 0:
            return
        key = str(bar.tabData(index) or "").strip()
        radio = (getattr(self, "_sim_backend_radio_by_key", None) or {}).get(key)
        if radio is None:
            return
        if radio.isChecked():
            # 顺序拖动后 currentChanged 可能仍指向同一 radio，无需重复触发
            return
        radio.setChecked(True)

    def _on_sim_backend_tab_moved(self, _from: int, _to: int) -> None:
        if getattr(self, "_sim_backend_tabs_order_guard", False):
            return
        keys = self._sim_backend_selector_keys()
        if not keys:
            return
        try:
            path = _eai_infrastructure_ui_storage.save_sim_backends_order(keys)
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"已保存仿真后端顺序 → {path}", 3000)
        except Exception as exc:
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"保存仿真后端顺序失败: {exc}", 4000)

    def _sim_bridge_cam_key_for_color(self, color_topic: str) -> _eai_platform_bootstrap.Optional[str]:
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
            if topic == color_topic:
                return cam_key
        # 命名约定：/camera/head_color → cam_head
        name = color_topic.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith("_color"):
            stem = name[: -len("_color")]
            if stem.startswith("cam_"):
                return stem
            return f"cam_{stem}"
        return None

    def _sim_bridge_npy_mtime(self, cam_key: str, *, depth: bool = False) -> _eai_platform_bootstrap.Optional[float]:
        root = self._sim_bridge_dir()
        base = f"{cam_key}_depth.npy" if depth else f"{cam_key}.npy"
        path = _eai_platform_bootstrap.os.path.join(root, base)
        stamp = path + ".stamp"
        try:
            return float(_eai_platform_bootstrap.os.path.getmtime(stamp if _eai_platform_bootstrap.os.path.isfile(stamp) else path))
        except OSError:
            return None

    def _on_mujoco_status(self, msg: str) -> None:
        if hasattr(self, "mujoco_status_label"):
            text = msg or "空闲"
            if text in ("空闲", ""):
                backend = self._sim_backend_id()
                if backend == "mujoco":
                    text = "空闲 · MuJoCo"
                elif backend == "molmospaces":
                    text = "空闲 · MolmoSpaces"
                elif backend == "psi_molmospaces":
                    text = "空闲 · Psi-MolmoSpaces"
                elif backend == "arena":
                    text = "空闲 · IsaacLab-Arena"
                elif backend == "libero":
                    text = "空闲 · LIBERO"
                elif backend == "robotwin":
                    text = "空闲 · RoboTwin"
                elif backend not in (
                    "isaac",
                    "mujoco",
                    "molmospaces",
                    "psi_molmospaces",
                ):
                    text = f"空闲 · {backend}"
            self.mujoco_status_label.setText(text)
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _sim_backend_id(self) -> str:
        """Return native backend id, or extended suite key when「扩展评测」selected."""
        if (
            getattr(self, "sim_backend_extended_radio", None) is not None
            and self.sim_backend_extended_radio.isChecked()
        ):
            if hasattr(self, "sim_extended_combo"):
                key = str(self.sim_extended_combo.currentData() or "").strip()
                if key:
                    return _eai_configuration_models.normalize_sim_backend_key(key)
            return "simpler"
        if getattr(self, "sim_backend_robotwin_radio", None) is not None and self.sim_backend_robotwin_radio.isChecked():
            return "robotwin"
        if getattr(self, "sim_backend_libero_radio", None) is not None and self.sim_backend_libero_radio.isChecked():
            return "libero"
        if getattr(self, "sim_backend_arena_radio", None) is not None and self.sim_backend_arena_radio.isChecked():
            return "arena"
        if (
            getattr(self, "sim_backend_psi_spaces_radio", None) is not None
            and self.sim_backend_psi_spaces_radio.isChecked()
        ):
            return "psi_molmospaces"
        if getattr(self, "sim_backend_spaces_radio", None) is not None and self.sim_backend_spaces_radio.isChecked():
            return "molmospaces"
        if getattr(self, "sim_backend_mujoco_radio", None) is not None and self.sim_backend_mujoco_radio.isChecked():
            return "mujoco"
        return "isaac"

    def _mujoco_backend_is_mujoco(self) -> bool:
        return self._sim_backend_id() == "mujoco"

    def _on_sim_leaderboard_toggled(self, checked: bool) -> None:
        show = bool(checked)
        for w in (
            getattr(self, "sim_leaderboard_meta_label", None),
            getattr(self, "sim_leaderboard_source_btn", None),
            getattr(self, "sim_leaderboard_refresh_btn", None),
            getattr(self, "sim_leaderboard_reload_btn", None),
            getattr(self, "sim_leaderboard_merge_btn", None),
            getattr(self, "sim_leaderboard_remove_local_btn", None),
            getattr(self, "sim_leaderboard_table", None),
            getattr(self, "sim_leaderboard_note_label", None),
        ):
            if w is not None:
                w.setVisible(show)

    def _on_sim_leaderboard_reload_clicked(self) -> None:
        try:
            _eai_configuration_models.clear_sim_eval_leaderboard_cache()
        except Exception:
            pass
        self._refresh_sim_leaderboard()
        if hasattr(self, "sim_leaderboard_meta_label"):
            cur = self.sim_leaderboard_meta_label.text()
            if "本地重载" not in cur:
                self.sim_leaderboard_meta_label.setText(
                    (cur + " · 本地重载").strip(" ·")
                )

    def _on_sim_leaderboard_refresh_clicked(self) -> None:
        """Fetch current backend leaderboard from public sources (background)."""
        if getattr(self, "_sim_leaderboard_refreshing", False):
            return
        backend = self._sim_backend_id()
        self._sim_leaderboard_refreshing = True
        if hasattr(self, "sim_leaderboard_refresh_btn"):
            self.sim_leaderboard_refresh_btn.setEnabled(False)
            self.sim_leaderboard_refresh_btn.setText("刷新中…")
        if hasattr(self, "sim_leaderboard_meta_label"):
            self.sim_leaderboard_meta_label.setText(
                f"正在从公开源拉取「{backend}」实时榜单…"
            )

        result: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any] = {"done": False, "ok": False, "backend": backend}

        def _work() -> None:
            try:
                board = _eai_configuration_models.refresh_sim_eval_backend(backend, persist=True)
                result["ok"] = True
                result["board"] = board
            except Exception as exc:
                result["ok"] = False
                result["error"] = str(exc)
            finally:
                result["done"] = True

        def _done() -> None:
            self._sim_leaderboard_refreshing = False
            if hasattr(self, "sim_leaderboard_refresh_btn"):
                self.sim_leaderboard_refresh_btn.setEnabled(True)
                self.sim_leaderboard_refresh_btn.setText("刷新")
            if result.get("ok") and isinstance(result.get("board"), dict):
                try:
                    _eai_configuration_models.clear_sim_eval_leaderboard_cache()
                except Exception:
                    pass
                self._apply_sim_leaderboard_board(result["board"])
                fetched = str(result["board"].get("_fetched_at") or "")
                n_local = int(result["board"].get("_local_count") or 0)
                if hasattr(self, "status_bar") and self.status_bar is not None:
                    extra = f" · 含 {n_local} 条本地对比" if n_local else ""
                    self.status_bar.showMessage(
                        f"实时榜单已刷新 · {backend}"
                        + (f" · {fetched}" if fetched else "")
                        + extra,
                        5000,
                    )
            else:
                err = str(result.get("error") or "未知错误")
                try:
                    self._refresh_sim_leaderboard(backend)
                except Exception:
                    pass
                if hasattr(self, "sim_leaderboard_meta_label"):
                    self.sim_leaderboard_meta_label.setText(
                        f"刷新失败（仍显示本地）: {err}"
                    )

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

        def _poll() -> None:
            if not result.get("done"):
                _eai_platform_bootstrap.QTimer.singleShot(200, _poll)
                return
            _done()

        _eai_platform_bootstrap.QTimer.singleShot(200, _poll)

    def _on_sim_leaderboard_source_clicked(self) -> None:
        url = (getattr(self, "_sim_leaderboard_source_url", None) or "").strip()
        if not url:
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(url))

