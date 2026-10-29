"""adapters.models.policy launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class RynnValueLauncher(_eai_platform_bootstrap.QObject):
    """启动 RynnValue 视频推理 / Reward 服务 / Robometer 评测。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    log_dir_ready = _eai_platform_bootstrap.pyqtSignal(str)
    output_video_ready = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._log_dir: str = ""
        self._render_gui: bool = False
        self._pending_video: str = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start(
        self,
        *,
        job: str,
        repo_root: str,
        python_bin: str = "",
        model_path: str = "",
        checkpoint_path: str = "",
        video_path: str = "",
        instruction: str = "",
        robot_description: str = "",
        camera_description: str = "",
        output_path: str = "",
        num_frames: int = 16,
        num_steps: int = 32,
        batch_size: int = 1,
        max_image_side: int = 640,
        max_new_tokens: int = 128,
        fps: int = 30,
        port: int = 8001,
        cuda_devices: str = "",
        datasets_path: str = "",
        render_gui: bool = False,
        extra_args: _eai_platform_bootstrap.Sequence[str] = (),
    ) -> None:
        if self.is_running():
            self.status_message.emit("RynnValue 正在运行")
            return
        script = _eai_configuration_tabs.RYNNVALUE_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_rynnvalue_root(repo_root)
        if not _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.join(root, "rynn_infer")):
            self.status_message.emit(f"无效仓库: {root}")
            return
        job_norm = (job or "infer").strip().lower().replace("-", "_")
        # Persistent reward_server: reuse healthy deploy (avoid second GPU load).
        if job_norm in ("reward_server", "server", "serve"):
            url = f"http://127.0.0.1:{max(1, int(port))}"
            ok_srv, msg_srv, info_srv = _eai_adapters_models_discovery.probe_rynnvalue_reward_server(
                url, timeout_s=2.0
            )
            if ok_srv:
                live = ""
                model_info = info_srv.get("model_info") if isinstance(info_srv, dict) else None
                if isinstance(model_info, dict):
                    live = str(model_info.get("reward_model") or "")
                health = info_srv.get("health") if isinstance(info_srv, dict) else None
                if not live and isinstance(health, dict):
                    live = str(health.get("reward_model") or "")
                want = (model_path or checkpoint_path or "").strip()
                same = (
                    _eai_adapters_models_qwen_local.model_identity_same("", live, "", want) if (live and want) else True
                )
                if same is False:
                    self.status_message.emit(
                        f"RynnValue 已在运行其它模型（{live}）。请先停止再部署。"
                    )
                    return
                self.log_line.emit(f"[adopt] {msg_srv}")
                self.status_message.emit(f"同一模型已部署，跳过重复部署: {msg_srv}")
                return
            if info_srv.get("listening"):
                self.log_line.emit(f"[skip] {msg_srv}")
                self.status_message.emit(msg_srv)
                return
        model = (model_path or "").strip()
        if job_norm in ("infer", "inference", "demo", "policy_ranking", "ranking",
                        "confusion_matrix", "confusion") and not model:
            self.status_message.emit("请填写 model path")
            return
        if job_norm in ("reward_server", "server", "serve") and not model and not (
            checkpoint_path or ""
        ).strip():
            self.status_message.emit("请填写 model path 或 checkpoint")
            return
        if job_norm in ("infer", "inference", "demo"):
            vid = (video_path or "").strip() or _eai_platform_bootstrap.os.path.join(
                root, "example", "Put_the_box_in_the_drawer_and_close_it.mp4"
            )
            if not _eai_platform_bootstrap.os.path.isfile(vid):
                self.status_message.emit(f"视频不存在: {vid}")
                return

        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_rynnvalue_python(root)
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.RYNNVALUE_CACHE_DIR, exist_ok=True)
        self._render_gui = bool(render_gui)
        self._pending_video = ""
        self._log_dir = _eai_platform_bootstrap.os.path.join(root, "rynn_infer", "outputs")

        args = [
            "--job",
            job_norm,
            "--root",
            root,
            "--python",
            py,
            "--num-frames",
            str(max(1, int(num_frames))),
            "--num-steps",
            str(max(0, int(num_steps))),
            "--batch-size",
            str(max(1, int(batch_size))),
            "--max-image-side",
            str(max(0, int(max_image_side))),
            "--max-new-tokens",
            str(max(1, int(max_new_tokens))),
            "--fps",
            str(max(1, int(fps))),
            "--port",
            str(max(1, int(port))),
        ]
        if model:
            args.extend(["--model-path", model])
        ckpt = (checkpoint_path or "").strip()
        if ckpt:
            args.extend(["--checkpoint-path", ckpt])
        if job_norm in ("infer", "inference", "demo"):
            args.extend(["--video-path", vid])
            instr = (instruction or "").strip() or _eai_configuration_tabs.RYNNVALUE_DEFAULT_INSTRUCTION
            args.extend(["--instruction", instr])
            robot = (robot_description or "").strip() or _eai_configuration_tabs.RYNNVALUE_DEFAULT_ROBOT
            camera = (camera_description or "").strip() or _eai_configuration_tabs.RYNNVALUE_DEFAULT_CAMERA
            args.extend(["--robot-description", robot])
            args.extend(["--camera-description", camera])
            out = (output_path or "").strip()
            if out:
                args.extend(["--output-path", out])
                self._log_dir = out
        ds = (datasets_path or "").strip()
        if ds:
            args.extend(["--datasets-path", ds])
        if self._render_gui:
            args.append("--gui")
        extras = [str(x).strip() for x in extra_args if str(x).strip()]
        if extras:
            args.append("--")
            args.extend(extras)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        # Strip GUI/ros-humble paths so rynnvalue (py3.10) never loads
        # ros-humble (py3.11) numpy/site-packages.
        for _k in (
            "PYTHONPATH",
            "PYTHONHOME",
            "AMENT_PREFIX_PATH",
            "ROS_DISTRO",
            "ROS_VERSION",
            "COLCON_PREFIX_PATH",
        ):
            qenv.remove(_k)
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("RYNNVALUE_ROOT", root)
        qenv.insert("RYNNVALUE_PYTHON", py)
        qenv.insert("MPLBACKEND", "Agg")
        qenv.insert("HF_ENDPOINT", qenv.value("HF_ENDPOINT") or "https://hf-mirror.com")
        cuda = (cuda_devices or "").strip()
        if cuda:
            qenv.insert("CUDA_VISIBLE_DEVICES", cuda)

        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root)
        proc.setProcessEnvironment(qenv)
        persist_server = job_norm in ("reward_server", "server", "serve")
        if persist_server:
            inner = "exec bash " + " ".join(
                _eai_platform_bootstrap.shlex.quote(a) for a in [script, *args]
            )
            shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(inner, "rynnvalue_reward")
            self._persist_pid_path = pid_path
            proc.start("setsid", ["bash", "-lc", shell])
        else:
            self._persist_pid_path = ""
            proc.start("setsid", ["bash", script, *args])
        self._process = proc
        self.running_changed.emit(True)
        if self._log_dir:
            self.log_dir_ready.emit(self._log_dir)
        gui_flag = " --gui" if self._render_gui else ""
        self.log_line.emit(
            f"$ bash run_rynnvalue.sh --job {job_norm}{gui_flag}"
        )
        self.status_message.emit(f"正在启动 RynnValue（{job_norm}）…")

    def stop(self) -> None:
        if getattr(self, "_persist_pid_path", ""):
            _eai_adapters_models_qwen_local.stop_persistent_model(self._persist_pid_path)
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 RynnValue")
            return
        self.status_message.emit("正在停止 RynnValue…")
        if self._process is not None:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.terminate()
            else:
                self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(4000, self._force_kill)

    def shutdown(self) -> None:
        # reward_server 在独立 session：界面退出只结束日志 tail。
        if getattr(self, "_persist_pid_path", ""):
            proc = self._process
            self._process = None
            if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
                proc.terminate()
                proc.waitForFinished(400)
            return
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.terminate()
            else:
                self._process.terminate()
            self._process.waitForFinished(2000)
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.kill()
            else:
                self._process.kill()
            self._process.waitForFinished(800)
        self._process = None
        self.running_changed.emit(False)

    def _force_kill(self) -> None:
        if self._process is None or self._process.state() == _eai_platform_bootstrap.QProcess.NotRunning:
            return
        pid = int(self._process.processId())
        if pid > 0:
            try:
                _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._process.kill()

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            text = line.rstrip()
            if not text:
                continue
            if text.startswith("[rynnvalue] log_dir="):
                self._log_dir = text.split("=", 1)[-1].strip()
                if self._log_dir:
                    self.log_dir_ready.emit(self._log_dir)
            if text.startswith("[rynnvalue] output_video="):
                self._pending_video = text.split("=", 1)[-1].strip()
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- RynnValue 正常退出 ---")
            self.status_message.emit("RynnValue 已完成")
            if self._pending_video and _eai_platform_bootstrap.os.path.isfile(self._pending_video):
                self.output_video_ready.emit(self._pending_video)
        else:
            self.log_line.emit(f"--- RynnValue 退出 (code={exit_code}) ---")
            self.status_message.emit(f"RynnValue 异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 RynnValue 进程")
            self.status_message.emit("无法启动 RynnValue")
            self.running_changed.emit(False)


class RoboICLLauncher(_eai_platform_bootstrap.QObject):
    """启动 RoboICL dry-run / capture-only / 完整 rollout。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    log_dir_ready = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._log_dir: str = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start(
        self,
        *,
        mode: str,
        task: str,
        shots: int,
        seed: int = 0,
        layout: int = 0,
        gpu: int = 0,
        repo_root: str = "",
        policy_python: str = "",
        sim_python: str = "",
        profile: str = "",
        data_root: str = "",
        results_root: str = "",
        reference: str = "",
        model: str = "",
        api_key: str = "",
        render_gui: bool = False,
        extra_args: _eai_platform_bootstrap.Sequence[str] = (),
    ) -> None:
        if self.is_running():
            self.status_message.emit("RoboICL 正在运行")
            return
        script = _eai_configuration_tabs.ROBOICL_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_roboicl_root(repo_root)
        if not _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(root, "roboicl", "run.py")):
            self.status_message.emit(f"无效仓库: {root}")
            return
        mode_norm = (mode or "run").strip().lower().replace("-", "_")
        if mode_norm not in ("dry_run", "capture_only", "run", "prepare_assets"):
            self.status_message.emit(f"未知 mode: {mode}")
            return
        task_name = (task or "").strip()
        if mode_norm == "prepare_assets":
            task_name = task_name or "deposit_coin"
        elif not task_name:
            self.status_message.emit("请选择任务")
            return
        policy_py = (policy_python or "").strip() or _eai_adapters_models_discovery.resolve_roboicl_policy_python(root)
        sim_py = (sim_python or "").strip() or _eai_adapters_models_discovery.resolve_roboicl_sim_python(root)
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.ROBOICL_CACHE_DIR, exist_ok=True)

        results = (results_root or "").strip() or _eai_platform_bootstrap.os.path.join(root, "results")
        data = (data_root or "").strip() or _eai_platform_bootstrap.os.path.join(root, "data")
        self._log_dir = results if mode_norm != "prepare_assets" else data

        args = [
            "--mode",
            mode_norm,
            "--task",
            task_name,
            "--shots",
            str(int(shots)),
            "--seed",
            str(max(0, int(seed))),
            "--layout",
            str(max(0, int(layout))),
            "--gpu",
            str(max(0, int(gpu))),
            "--root",
            root,
            "--policy-python",
            policy_py,
            "--sim-python",
            sim_py,
            "--data-root",
            data,
            "--results-root",
            results,
        ]
        key = (api_key or "").strip()
        if mode_norm != "prepare_assets":
            prof = (profile or "").strip()
            if prof:
                args.extend(["--profile", prof])
            ref = (reference or "").strip()
            if ref:
                args.extend(["--reference", ref])
            mdl = (model or "").strip()
            if mdl:
                args.extend(["--model", mdl])
            if key:
                args.extend(["--api-key", key])
            if render_gui:
                args.append("--gui")
        extras = [str(x).strip() for x in extra_args if str(x).strip()]
        if extras:
            args.append("--")
            args.extend(extras)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("ROBOICL_ROOT", root)
        qenv.insert("ROBOICL_CODE_ROOT", root)
        qenv.insert("ROBOICL_POLICY_PYTHON", policy_py)
        qenv.insert("ROBOICL_SIM_PYTHON", sim_py)
        qenv.insert("ROBOICL_DATA_ROOT", data)
        qenv.insert("ROBOICL_RESULTS_ROOT", results)
        qenv.insert("CUDA_VISIBLE_DEVICES", str(max(0, int(gpu))))
        # Avoid shared /tmp/isaaclab/logs PermissionError (root-owned / contested).
        tmpdir = _eai_platform_bootstrap.os.path.join(results, ".tmpdir")
        try:
            _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.join(tmpdir, "isaaclab", "logs"), exist_ok=True)
        except OSError:
            tmpdir = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.ROBOICL_CACHE_DIR, "tmpdir")
            _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.join(tmpdir, "isaaclab", "logs"), exist_ok=True)
        qenv.insert("ROBOICL_TMPDIR", tmpdir)
        qenv.insert("TMPDIR", tmpdir)
        qenv.insert("TEMP", tmpdir)
        qenv.insert("TMP", tmpdir)
        if key:
            qenv.insert("ASTRA_API_KEY", key)
        # Clash Verge Sidecar mixed-port（本机无 systemd，TUN/系统代理对子进程不可靠）
        if not (
            qenv.value("HTTPS_PROXY")
            or qenv.value("https_proxy")
            or qenv.value("ALL_PROXY")
            or qenv.value("all_proxy")
        ):
            clash_host = _eai_platform_bootstrap.os.environ.get("CLASH_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
            clash_port = _eai_platform_bootstrap.os.environ.get("CLASH_MIXED_PORT", "7897").strip() or "7897"
            listening = False
            try:
                import socket

                with socket.create_connection((clash_host, int(clash_port)), timeout=0.35):
                    listening = True
            except OSError:
                listening = False
            if listening:
                proxy_url = f"http://{clash_host}:{clash_port}"
                for env_name in (
                    "http_proxy",
                    "https_proxy",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "all_proxy",
                    "ALL_PROXY",
                ):
                    qenv.insert(env_name, proxy_url)
                no_proxy = (
                    "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
                )
                qenv.insert("no_proxy", no_proxy)
                qenv.insert("NO_PROXY", no_proxy)
                self.log_line.emit(f"[roboicl] proxy={proxy_url}")

        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root)
        proc.setProcessEnvironment(qenv)
        proc.start("setsid", ["bash", script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_dir_ready.emit(self._log_dir)
        gui_flag = " --gui" if render_gui else ""
        self.log_line.emit(
            f"$ bash run_roboicl.sh --mode {mode_norm} --task {task_name}"
            f" --shots {int(shots)}{gui_flag}"
        )
        self.status_message.emit(
            f"正在启动 RoboICL（{mode_norm} / {task_name}）…"
        )

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 RoboICL")
            return
        self.status_message.emit("正在停止 RoboICL…")
        if self._process is not None:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.terminate()
            else:
                self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(4000, self._force_kill)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.terminate()
            else:
                self._process.terminate()
            self._process.waitForFinished(2000)
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.kill()
            else:
                self._process.kill()
            self._process.waitForFinished(800)
        self._process = None
        self.running_changed.emit(False)

    def _force_kill(self) -> None:
        if self._process is None or self._process.state() == _eai_platform_bootstrap.QProcess.NotRunning:
            return
        pid = int(self._process.processId())
        if pid > 0:
            try:
                _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._process.kill()

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            text = line.rstrip()
            if not text:
                continue
            if text.startswith("[roboicl] log_dir="):
                self._log_dir = text.split("=", 1)[-1].strip()
                if self._log_dir:
                    self.log_dir_ready.emit(self._log_dir)
            if text.startswith("Run: "):
                run_dir = text[len("Run: ") :].strip()
                if run_dir and _eai_platform_bootstrap.os.path.isdir(run_dir):
                    self._log_dir = run_dir
                    self.log_dir_ready.emit(self._log_dir)
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- RoboICL 正常退出 ---")
            self.status_message.emit("RoboICL 已完成")
        else:
            self.log_line.emit(f"--- RoboICL 退出 (code={exit_code}) ---")
            self.status_message.emit(f"RoboICL 异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 RoboICL 进程")
            self.status_message.emit("无法启动 RoboICL")
            self.running_changed.emit(False)

