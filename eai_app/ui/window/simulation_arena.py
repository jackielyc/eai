"""ui.window.simulation arena — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationArenaMixin:
    def _on_arena_root_browse(self) -> None:
        cur = self._resolve_arena_root()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 IsaacLab-Arena 仓库", cur)
        if selected and hasattr(self, "arena_root_edit"):
            self.arena_root_edit.setText(selected)
            self._refresh_arena_tasks()

    def _on_arena_policy_changed(self, *_args) -> None:
        policy = "zero_action"
        if hasattr(self, "arena_policy_combo"):
            policy = str(self.arena_policy_combo.currentData() or policy)
        show = policy in ("replay", "rsl_rl")
        if hasattr(self, "arena_policy_file_row"):
            self.arena_policy_file_row.setVisible(show)
        if not show or not hasattr(self, "arena_policy_file_label"):
            return
        if policy == "replay":
            self.arena_policy_file_label.setText("回放文件")
            self.arena_policy_file_edit.setPlaceholderText("HDF5，传给 --replay_file_path")
        else:
            self.arena_policy_file_label.setText("checkpoint")
            self.arena_policy_file_edit.setPlaceholderText(".pt，传给 --checkpoint_path")

    def _on_arena_policy_file_browse(self) -> None:
        policy = str(self.arena_policy_combo.currentData() or "")
        if policy == "rsl_rl":
            caption = "选择 RSL-RL checkpoint"
            filt = "Checkpoint (*.pt *.pth);;All (*)"
        else:
            caption = "选择回放 HDF5"
            filt = "HDF5 (*.hdf5 *.h5);;All (*)"
        cur = self.arena_policy_file_edit.text().strip()
        start = cur if _eai_platform_bootstrap.os.path.isfile(cur) else _eai_platform_bootstrap.os.path.dirname(cur) if cur else ""
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(self, caption, start, filt)
        if selected:
            self.arena_policy_file_edit.setText(selected)

    def _on_arena_run_clicked(self, *, viewer_only: bool = False) -> None:
        if (
            self._sim_eval_launcher.is_running()
            or self._sim_bridge_launcher.is_running()
        ):
            self._on_mujoco_status("Isaac 评测/相机桥正在运行，请先停止")
            return
        if self._sim_backend_id() != "arena":
            self._on_mujoco_status("请先将仿真后端切换为 IsaacLab-Arena")
            return
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("MuJoCo / MolmoSpaces / Arena 正在运行，请先停止")
            return
        root = self._resolve_arena_root()
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self._on_mujoco_status(f"IsaacLab-Arena 目录不存在: {root}")
            return
        # policy_runner is invoked via tools/run_arena_eval.py (Live HUD hook).
        env_name = ""
        if hasattr(self, "arena_task_combo"):
            env_name = str(self.arena_task_combo.currentData() or "").strip()
        if not env_name:
            self._on_mujoco_status("请选择一个 IsaacLab-Arena 任务")
            return
        # 只开界面：强制 zero_action + Kit，不加载策略权重 / Live HUD。
        if viewer_only:
            policy = "zero_action"
            show_gui = True
        else:
            policy = "zero_action"
            if hasattr(self, "arena_policy_combo"):
                policy = str(self.arena_policy_combo.currentData() or policy)
            show_gui = not (
                hasattr(self, "arena_headless_check")
                and self.arena_headless_check.isChecked()
            )
        if (not viewer_only) and _eai_configuration_models.is_lingbot_va_policy(policy):
            self.mujoco_log_edit.clear()
            self._append_mujoco_log(
                "[arena/LingBot-VA] 启动 wan_va（serve_only）；"
                "Arena Kit 暂无原生 VA policy_type，仅起推理服务"
            )
            self._launch_lingbot_va_backend(
                "arena",
                task=env_name,
                label="arena-va",
            )
            return
        steps = 200
        if hasattr(self, "arena_steps_spin"):
            steps = int(self.arena_steps_spin.value())
        if viewer_only:
            steps = max(steps, 2000)
        py = _eai_adapters_models_discovery.resolve_isaaclab_arena_python()
        runner = _eai_platform_bootstrap.os.path.join(root, "isaaclab_arena", "evaluation", "policy_runner.py")
        wrapper = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools", "run_arena_eval.py")
        if not _eai_platform_bootstrap.os.path.isfile(runner):
            self._on_mujoco_status(f"未找到: {runner}")
            return
        if (not viewer_only) and not self._ensure_live_hud_reward_or_abort("arena"):
            return
        hud_env = (
            {}
            if viewer_only
            else self._rynnvalue_live_hud_child_env("arena")
        )
        if hud_env:
            self._expand_workspace_panel()
        argv = [
            py,
            wrapper,
            f"--arena-root={root}",
            "--",
            "--policy_type",
            policy,
            "--num_steps",
            str(steps),
        ]
        use_gpu = True
        gpu_id = 0
        if hasattr(self, "arena_gpu_check"):
            use_gpu = self.arena_gpu_check.isChecked()
        if hasattr(self, "arena_gpu_spin"):
            gpu_id = int(self.arena_gpu_spin.value())
        if use_gpu:
            argv.extend(["--device", "cuda:0"])
        else:
            argv.extend(["--device", "cpu"])
        if (not viewer_only) and policy in ("replay", "rsl_rl"):
            policy_file = ""
            if hasattr(self, "arena_policy_file_edit"):
                policy_file = self.arena_policy_file_edit.text().strip()
            if not policy_file or not _eai_platform_bootstrap.os.path.isfile(policy_file):
                need = "HDF5 回放文件" if policy == "replay" else "checkpoint（.pt）"
                self._on_mujoco_status(f"{policy} 需要{need}，未启动 Kit")
                return
            flag = "--replay_file_path" if policy == "replay" else "--checkpoint_path"
            argv.extend([flag, _eai_platform_bootstrap.os.path.abspath(policy_file)])
        if show_gui:
            argv.extend(["--viz", "kit"])
        else:
            argv.extend(["--viz", "none"])
        argv.append(env_name)
        extra = ""
        if hasattr(self, "arena_extra_edit"):
            extra = self.arena_extra_edit.text().strip()
        if extra:
            try:
                argv.extend(_eai_platform_bootstrap.shlex.split(extra))
            except ValueError as exc:
                self._on_mujoco_status(f"额外参数无法解析: {exc}")
                return
        self.mujoco_log_edit.clear()
        if viewer_only:
            self._append_mujoco_log(
                "[arena] 只开界面：zero_action + Kit，不加载策略/不推理"
            )
        self._append_mujoco_log(f"[arena] root={root}")
        self._append_mujoco_log(f"[arena] python={py}")
        device_label = f"cuda:{gpu_id}" if use_gpu else "cpu"
        viz_label = "kit" if show_gui else "none"
        self._append_mujoco_log(
            f"[arena] task={env_name} policy={policy} steps={steps} "
            f"device={device_label} viz={viz_label}"
        )
        env_extra = _eai_adapters_models_discovery.isaaclab_arena_child_env(
            root, py, cuda_visible=str(gpu_id) if use_gpu else ""
        )
        if hud_env:
            env_extra = {**(env_extra or {}), **hud_env}
            # Ensure EAI tools importable inside Arena python
            tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
            prev = (env_extra.get("PYTHONPATH") or _eai_platform_bootstrap.os.environ.get("PYTHONPATH") or "")
            parts = [tools] + [p for p in prev.split(_eai_platform_bootstrap.os.pathsep) if p and p != tools]
            env_extra["PYTHONPATH"] = _eai_platform_bootstrap.os.pathsep.join(parts)
            self._append_mujoco_log(
                f"[arena] RynnValue Live HUD "
                f"url={hud_env.get('RYNNVALUE_SERVER_URL')}"
            )
        self._mujoco_launcher.start_cwd_command(
            cwd=root,
            argv=argv,
            label=f"arena {env_name}",
            env_extra=env_extra,
        )
        self._update_mujoco_ui()

