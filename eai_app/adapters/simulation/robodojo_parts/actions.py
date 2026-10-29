"""adapters.simulation.robodojo parts.actions — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ActionsMixin:
    @staticmethod
    def list_running_eval_clients() -> _eai_platform_bootstrap.List[dict]:
        """发现系统中已在跑的 src/eval_client/main.py（含任务名）。"""
        found: _eai_platform_bootstrap.List[dict] = []
        try:
            result = _eai_platform_bootstrap.subprocess.run(
                ["pgrep", "-af", "src/eval_client/main.py"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return found
        if result.returncode not in (0, 1):
            return found
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or "pgrep" in line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            cmd = parts[1]
            if "src/eval_client/main.py" not in cmd:
                continue
            task = ""
            tokens = cmd.split()
            for i, tok in enumerate(tokens):
                if tok == "--task_name" and i + 1 < len(tokens):
                    task = tokens[i + 1].strip()
                    break
                if tok.startswith("--task_name="):
                    task = tok.split("=", 1)[1].strip()
                    break
            found.append({"pid": pid, "task": task, "cmd": cmd})
        found.sort(key=lambda x: x["pid"])
        return found

    @staticmethod
    def list_running_isaac_sims() -> _eai_platform_bootstrap.List[dict]:
        """发现系统中已在跑的 Isaac Sim / Kit 进程。"""
        patterns = (
            r"/kit/kit",
            r"kit/kit ",
            r"isaac-sim",
            r"isaacsim\.",
            r"IsaacSim",
            r"omni\.isaac",
            r"isaacsim\.replicator",
        )
        found: _eai_platform_bootstrap.List[dict] = []
        seen = set()  # type: ignore[var-annotated]
        for pat in patterns:
            try:
                result = _eai_platform_bootstrap.subprocess.run(
                    ["pgrep", "-af", pat],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            except Exception:
                continue
            if result.returncode not in (0, 1):
                continue
            for line in (result.stdout or "").splitlines():
                line = line.strip()
                if not line or "pgrep" in line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                if pid in seen or pid <= 1:
                    continue
                cmd = parts[1]
                # 排除纯文本编辑 / 本探测自身
                low = cmd.lower()
                if "show_camera_topics" in low or "pgrep" in low:
                    continue
                if not any(
                    k in low
                    for k in (
                        "kit/kit",
                        "isaac-sim",
                        "isaacsim",
                        "omni.isaac",
                        "isaac sim",
                    )
                ):
                    continue
                seen.add(pid)
                found.append({"pid": pid, "cmd": cmd})
        found.sort(key=lambda x: x["pid"])
        return found

    @classmethod
    def isaac_sim_status(cls) -> dict:
        """汇总 Isaac / eval_client / 持久化 session 运行状态。"""
        clients = cls.list_running_eval_clients()
        kits = cls.list_running_isaac_sims()
        _, pid_path = _eai_adapters_models_qwen_local.persistent_slot_paths(cls.PERSIST_SLOT)
        persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(pid_path)
        return {
            "clients": clients,
            "kits": kits,
            "persist_pid": persist_pid,
            "running": bool(clients or kits or persist_pid > 1),
        }

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 1:
            return False
        try:
            _eai_platform_bootstrap.os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def is_running(self) -> bool:
        if self._stopping or self._restart_pending:
            return True
        if self._process is not None:
            if self._process.state() in (_eai_platform_bootstrap.QProcess.Starting, _eai_platform_bootstrap.QProcess.Running):
                return True
        persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path)
        if persist_pid > 1 and persist_pid not in self._external_pids:
            # 持久化 session 仍在，但本窗口尚未接管 → 不算「本窗口在跑」
            pass
        if self._external_pids:
            alive = [p for p in self._external_pids if self._pid_alive(p)]
            self._external_pids = alive
            if alive:
                return True
            if self._adopted:
                self._adopted = False
                self._pgid = None
        return False

    def adopt_running_eval(
        self,
        task: str = "",
        *,
        bridge_dir: str = "",
        prefer_task: str = "",
        **kwargs,
    ) -> _eai_platform_bootstrap.Optional[dict]:
        """接管已存在的评测进程。成功返回 {pid, task, ...}，否则 None。"""
        if self.is_running() and not self._adopted:
            return None
        clients = self.list_running_eval_clients()
        if not clients:
            return None
        prefer = (prefer_task or task or "").strip()
        chosen = None
        if prefer:
            for c in reversed(clients):
                if c.get("task") == prefer:
                    chosen = c
                    break
        if chosen is None:
            chosen = clients[-1]  # 最新 PID
        pid = int(chosen["pid"])
        cur_task = str(chosen.get("task") or "").strip()
        self._process = None
        self._adopted = True
        self._keep_alive = False  # 接管的进程退出后不自动再拉起
        self._stopping = False
        self._restart_pending = False
        self._external_pids = [pid]
        try:
            self._pgid = _eai_platform_bootstrap.os.getpgid(pid)
        except OSError:
            self._pgid = pid
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                (bridge_dir or "").strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        self._last_start_kwargs = {
            "task": cur_task or prefer or task,
            "bridge_dir": bridge_dir,
            "policy_dir": kwargs.get("policy_dir", ""),
            "ckpt": kwargs.get("ckpt", "demo"),
            "action_type": kwargs.get("action_type", "ee"),
            "policy_env": kwargs.get("policy_env", ""),
            "eval_env": kwargs.get("eval_env", ""),
            "robodojo_root": kwargs.get("robodojo_root", ""),
            "display": kwargs.get("display", ""),
            "use_gui": bool(kwargs.get("use_gui", True)),
        }
        extras = [c for c in clients if int(c["pid"]) != pid]
        self.log_line.emit(
            f"复用已运行仿真: pid={pid} task={cur_task or '?'} "
            f"(共发现 {len(clients)} 个 eval_client)"
        )
        if extras:
            extra_s = ", ".join(
                f"{c['pid']}:{c.get('task') or '?'}" for c in extras
            )
            self.log_line.emit(
                f"[WARN] 另有评测进程未接管（请手动停掉以免抢 GPU/帧）: {extra_s}"
            )
        self.status_message.emit(
            f"已复用运行中仿真 pid={pid}（{cur_task or 'unknown'}）"
        )
        if not self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.start()
        self.running_changed.emit(True)
        return chosen

    def adopt_persisted_eval(
        self,
        task: str = "",
        *,
        bridge_dir: str = "",
        **kwargs,
    ) -> _eai_platform_bootstrap.Optional[dict]:
        """接管 EAI 退出后仍存活的持久化评测 session。"""
        if self.is_running() and not self._adopted:
            return None
        persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path)
        if persist_pid <= 1:
            return None
        # 优先用 eval_client 拿到任务名；否则只挂持久化 session pid。
        clients = self.list_running_eval_clients()
        chosen = None
        prefer = (task or "").strip()
        if prefer:
            for c in reversed(clients):
                if c.get("task") == prefer:
                    chosen = c
                    break
        if chosen is None and clients:
            chosen = clients[-1]
        if chosen is not None:
            return self.adopt_running_eval(
                task,
                bridge_dir=bridge_dir,
                prefer_task=str(chosen.get("task") or prefer),
                **kwargs,
            )
        # 仅有持久化 shell / Isaac，无 eval_client 元数据时仍接管 session。
        self._process = None
        self._adopted = True
        self._keep_alive = False
        self._stopping = False
        self._restart_pending = False
        self._external_pids = [persist_pid]
        try:
            self._pgid = _eai_platform_bootstrap.os.getpgid(persist_pid)
        except OSError:
            self._pgid = persist_pid
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                (bridge_dir or "").strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        self._last_start_kwargs = {
            "task": prefer or task,
            "bridge_dir": bridge_dir,
            "policy_dir": kwargs.get("policy_dir", ""),
            "ckpt": kwargs.get("ckpt", "demo"),
            "action_type": kwargs.get("action_type", "ee"),
            "policy_env": kwargs.get("policy_env", ""),
            "eval_env": kwargs.get("eval_env", ""),
            "robodojo_root": kwargs.get("robodojo_root", ""),
            "display": kwargs.get("display", ""),
            "use_gui": bool(kwargs.get("use_gui", True)),
        }
        kits = self.list_running_isaac_sims()
        self.log_line.emit(
            f"复用持久化仿真 session: pid={persist_pid}"
            + (f" IsaacKit×{len(kits)}" if kits else "")
        )
        self.status_message.emit(
            f"已复用持久化 Isaac/评测 session pid={persist_pid}"
        )
        if not self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.start()
        self.running_changed.emit(True)
        return {"pid": persist_pid, "task": prefer or task, "persisted": True}

    def _on_adopt_watch_tick(self) -> None:
        if not self._adopted:
            self._adopt_watch_timer.stop()
            return
        if self.is_running():
            return
        # 持久化 session 或 eval_client 可能刚退出；再扫一遍 Isaac。
        if _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path) > 1:
            return
        if self.list_running_eval_clients() or self.list_running_isaac_sims():
            return
        self._adopt_watch_timer.stop()
        self._adopted = False
        self._external_pids = []
        self._pgid = None
        self.log_line.emit("--- 已接管的评测进程已退出 ---")
        self.status_message.emit("复用的评测已结束")
        self.running_changed.emit(False)

    def start(
        self,
        task: str,
        *,
        bridge_dir: str = "",
        policy_dir: str = "XPolicyLab/policy/demo_policy",
        ckpt: str = "demo",
        action_type: str = "ee",
        policy_env: str = "",
        eval_env: str = "",
        robodojo_root: str = "",
        display: str = "",
        use_gui: bool = True,
        reuse_existing: bool = True,
        attach_external: bool = False,
        policy_host: str = "",
        policy_port: int = 0,
    ) -> str:
        """启动或复用评测。返回 started / reused / switched / busy / error。"""
        task = (task or "").strip()
        if not task:
            self.status_message.emit("请选择评测任务")
            return "error"

        if self.is_running():
            cur = self.current_task()
            if cur and cur == task:
                self.status_message.emit(f"评测已在运行（任务 {task}）")
                return "reused"
            if self.request_switch_task(task, bridge_dir=bridge_dir):
                return "switched"
            return "busy"

        adopt_kwargs = dict(
            bridge_dir=bridge_dir,
            policy_dir=policy_dir,
            ckpt=ckpt,
            action_type=action_type,
            policy_env=policy_env,
            eval_env=eval_env,
            robodojo_root=robodojo_root,
            display=display,
            use_gui=use_gui,
        )

        if reuse_existing:
            adopted = self.adopt_running_eval(
                task,
                prefer_task=task,
                **adopt_kwargs,
            )
            if adopted is not None:
                cur = str(adopted.get("task") or "").strip()
                if cur and cur != task:
                    if self.request_switch_task(task, bridge_dir=bridge_dir):
                        return "switched"
                    return "reused"
                self.status_message.emit(f"已复用仿真（任务 {cur or task}）")
                return "reused"

            persisted = self.adopt_persisted_eval(task, **adopt_kwargs)
            if persisted is not None:
                cur = str(persisted.get("task") or "").strip()
                if cur and cur != task:
                    if self.request_switch_task(task, bridge_dir=bridge_dir):
                        return "switched"
                    return "reused"
                self.status_message.emit(
                    f"已复用持久化仿真（任务 {cur or task or '?'}）"
                )
                return "reused"

            # Isaac Kit 已在跑但尚无/未识别到 eval_client：禁止再起一套。
            status = self.isaac_sim_status()
            if status.get("running"):
                kits = status.get("kits") or []
                clients = status.get("clients") or []
                persist_pid = int(status.get("persist_pid") or 0)
                detail_parts = []
                if clients:
                    detail_parts.append(
                        "eval_client="
                        + ",".join(
                            f"{c['pid']}:{c.get('task') or '?'}" for c in clients
                        )
                    )
                if kits:
                    detail_parts.append(
                        "IsaacKit=" + ",".join(str(k["pid"]) for k in kits[:6])
                    )
                if persist_pid > 1:
                    detail_parts.append(f"persist={persist_pid}")
                detail = "; ".join(detail_parts) or "unknown"
                self.log_line.emit(
                    f"[WARN] 检测到 Isaac/评测已在运行，拒绝重复启动: {detail}"
                )
                # 最后再试一次接管（竞态：扫描时进程刚起来）
                adopted = self.adopt_running_eval(
                    task, prefer_task=task, **adopt_kwargs
                )
                if adopted is None:
                    adopted = self.adopt_persisted_eval(task, **adopt_kwargs)
                if adopted is not None:
                    return "reused"
                self.status_message.emit(
                    "Isaac Sim 已在运行，请先「停止评测」或等待接管，勿重复启动"
                )
                return "busy"

        return self._start_new_process(
            task,
            bridge_dir=bridge_dir,
            policy_dir=policy_dir,
            ckpt=ckpt,
            action_type=action_type,
            policy_env=policy_env,
            eval_env=eval_env,
            robodojo_root=robodojo_root,
            display=display,
            use_gui=use_gui,
            attach_external=attach_external,
            policy_host=policy_host,
            policy_port=policy_port,
        )

