"""adapters.simulation.robodojo parts.actions restart keepalive — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ActionsRestartKeepaliveMixin:
    def _restart_keepalive(self) -> None:
        self._restart_pending = False
        if not self._keep_alive or self._stopping:
            self.running_changed.emit(False)
            return
        kwargs = dict(self._last_start_kwargs or {})
        if not kwargs:
            self.running_changed.emit(False)
            return
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            return
        # 日志 tail 抖动时 persist/Isaac 可能仍在：先接管，禁止再起一套。
        if self.isaac_sim_status().get("running"):
            adopted = self.adopt_running_eval(
                str(kwargs.get("task") or ""),
                prefer_task=str(kwargs.get("task") or ""),
                **{
                    k: kwargs[k]
                    for k in (
                        "bridge_dir",
                        "policy_dir",
                        "ckpt",
                        "action_type",
                        "policy_env",
                        "eval_env",
                        "robodojo_root",
                        "display",
                        "use_gui",
                    )
                    if k in kwargs
                },
            )
            if adopted is None:
                adopted = self.adopt_persisted_eval(
                    str(kwargs.get("task") or ""),
                    **{
                        k: kwargs[k]
                        for k in (
                            "bridge_dir",
                            "policy_dir",
                            "ckpt",
                            "action_type",
                            "policy_env",
                            "eval_env",
                            "robodojo_root",
                            "display",
                            "use_gui",
                        )
                        if k in kwargs
                    },
                )
            if adopted is not None:
                self.log_line.emit("--- 常驻：日志退出但 Isaac 仍在，已重新接管 ---")
                return
            self.log_line.emit(
                "--- 常驻重启跳过：检测到 Isaac 仍在运行但无法接管，避免重复启动 ---"
            )
            self.running_changed.emit(False)
            return
        self.log_line.emit("--- 常驻模式：重新启动评测 ---")
        self.start(reuse_existing=False, **kwargs)
        if not self.is_running():
            self._keep_alive = False
            self.log_line.emit("--- 常驻重启失败，已停止自动重启 ---")
            self.status_message.emit("评测常驻重启失败")
            self.running_changed.emit(False)

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        was_stopping = self._stopping
        self._process = None
        self._pgid = None
        self._stopping = False
        if was_stopping:
            self._keep_alive = False
            self._restart_pending = False
            self.running_changed.emit(False)
            self.log_line.emit("--- RoboDojo 评测已停止 ---")
            self.status_message.emit("RoboDojo 评测已停止")
            return
        if self._keep_alive:
            self.log_line.emit(
                f"--- RoboDojo 评测退出 (code={exit_code})，准备常驻重启 ---"
            )
            self._schedule_keepalive_restart()
            return
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- RoboDojo 评测正常结束 ---")
            self.status_message.emit("RoboDojo 评测已结束")
        else:
            self.log_line.emit(f"--- RoboDojo 评测退出 (code={exit_code}) ---")
            self.status_message.emit(f"RoboDojo 评测异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"RoboDojo 评测进程错误: {error}")

    def _force_kill_process(self) -> None:
        # 兼容旧调用路径（二次点击 / 定时器）
        self._force_kill_all_sims()

    @staticmethod
    def _pkill_eval_leftovers(sig: str = "-TERM") -> int:
        """对所有仿真相关进程发信号。返回成功匹配的 pkill 次数。"""
        patterns = (
            "src/eval_client/main.py",
            "scripts/run_gui_eval.sh",
            "run_gui_eval.sh",
            "run_gui_starvla_pi_v3.sh",
            "eval_hf_robodojo.sh",
            "scripts/robodojo.sh eval",
            "scripts/robodojo.sh",
            "scripts/eval_policy.sh",
            "isaacsim.replicator",
            "isaacsim.sensors",
            "/kit/kit",
            "kit/kit ",
            "omni.isaac",
            "IsaacSim",
        )
        hits = 0
        for pat in patterns:
            try:
                result = _eai_platform_bootstrap.subprocess.run(
                    ["pkill", sig, "-f", pat],
                    capture_output=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    hits += 1
            except Exception:
                pass
        return hits

