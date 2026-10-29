"""adapters.simulation.mujoco launcher — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class MuJoCoViewerLauncher(_eai_platform_bootstrap.QObject):
    """启动 MuJoCo 交互式图形界面（viewer / simulate）。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._label = ""
        self._last_exit_code: _eai_platform_bootstrap.Optional[int] = None
        # Bumped on each start/force_clear so delayed _force_kill cannot
        # SIGKILL a newly launched process after relaunch.
        self._run_generation: int = 0
        # Policy / Pi / Psi servers stay up after the UI exits.
        self._keep_on_exit = False
        self._persist_slot = ""
        self._persist_pid_path = ""

    def enable_persist_on_exit(self, slot: str) -> None:
        self._keep_on_exit = True
        self._persist_slot = (slot or "model").strip() or "model"

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def _disconnect_process(self, proc: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess]) -> None:
        if proc is None:
            return
        for signal_name, slot in (
            ("readyReadStandardOutput", self._on_process_output),
            ("finished", self._on_process_finished),
            ("errorOccurred", self._on_process_error),
        ):
            try:
                getattr(proc, signal_name).disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _is_active_process(self, proc: object) -> bool:
        return proc is not None and proc is self._process

    @property
    def last_exit_code(self) -> _eai_platform_bootstrap.Optional[int]:
        return self._last_exit_code

    @property
    def label(self) -> str:
        return self._label or ""

    def start(
        self,
        *,
        mjcf_path: str,
        mujoco_root: str = "",
        python_bin: str = "",
        mode: str = "python",
        install: bool = False,
        env_extra: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, str]] = None,
    ) -> None:
        if self.is_running():
            self.status_message.emit("MuJoCo 正在运行")
            return
        script = _eai_configuration_tabs.MUJOCO_VIEWER_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_mujoco_root(mujoco_root)
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_mujoco_python(root)
        mode_norm = (mode or "auto").strip().lower() or "auto"
        if mode_norm not in ("auto", "python", "simulate", "egl"):
            self.status_message.emit(f"未知 mode: {mode}")
            return

        args: _eai_platform_bootstrap.List[str] = [
            "--mujoco-root",
            root,
            "--python",
            py,
            "--mode",
            mode_norm,
        ]
        if install:
            args.append("--install")
            self._label = "install mujoco"
        else:
            mjcf = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((mjcf_path or "").strip()))
            if not mjcf:
                self.status_message.emit("请选择 MJCF 模型")
                return
            if not _eai_platform_bootstrap.os.path.isfile(mjcf):
                self.status_message.emit(f"模型不存在: {mjcf}")
                return
            args.extend(["--mjcf", mjcf])
            self._label = _eai_platform_bootstrap.os.path.basename(mjcf)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("MUJOCO_ROOT", root)
        qenv.insert("MUJOCO_PYTHON", py)
        if env_extra:
            for k, v in env_extra.items():
                if v is None:
                    continue
                qenv.insert(str(k), str(v))
        # Prefer VirtualGL → NVIDIA for interactive viewer (TurboVNC has no GPU GLX).
        if mode_norm != "egl":
            qenv.insert("MUJOCO_GL", "glfw")
            qenv.insert("MUJOCO_USE_VGL", qenv.value("MUJOCO_USE_VGL") or "1")
            qenv.insert("MUJOCO_VGL_DEVICE", qenv.value("MUJOCO_VGL_DEVICE") or "egl")
        else:
            qenv.insert("MUJOCO_GL", qenv.value("MUJOCO_GL") or "egl")
        # GUI 需要显示；保留宿主 DISPLAY / XAUTHORITY
        if not (qenv.value("DISPLAY") or "").strip():
            self.log_line.emit(
                "[warn] DISPLAY 为空，图形窗口可能无法弹出（本机桌面/X11 转发）"
            )

        self._run_generation += 1
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root if _eai_platform_bootstrap.os.path.isdir(root) else _eai_configuration_models.EAI_DIR)
        proc.setProcessEnvironment(qenv)
        proc.start("setsid", ["bash", script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ bash scripts/launch/run_mujoco_viewer.sh {' '.join(args)}")
        if install:
            self.status_message.emit("正在安装 mujoco…")
        else:
            self.status_message.emit(f"正在启动 MuJoCo: {self._label}")

    def start_cwd_command(
        self,
        *,
        cwd: str,
        argv: _eai_platform_bootstrap.List[str],
        label: str = "",
        env_extra: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, str]] = None,
    ) -> None:
        """在指定目录启动通用命令（如 MolmoSpaces）。"""
        if self.is_running():
            self.status_message.emit("MuJoCo / MolmoSpaces 正在运行")
            return
        work = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((cwd or "").strip()))
        if not _eai_platform_bootstrap.os.path.isdir(work):
            self.status_message.emit(f"目录不存在: {work}")
            return
        if not argv:
            self.status_message.emit("命令为空")
            return
        self._label = (label or argv[0] or "cmd").strip() or "cmd"
        launch_argv = list(argv)
        if self._keep_on_exit:
            inner = " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in launch_argv)
            shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(
                inner, self._persist_slot or "model"
            )
            self._persist_pid_path = pid_path
            launch_argv = ["bash", "-lc", shell]
        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.insert("PYTHONUNBUFFERED", "1")
        if env_extra:
            for k, v in env_extra.items():
                key = str(k)
                if v is None:
                    qenv.remove(key)
                else:
                    qenv.insert(key, str(v))
        if not (qenv.value("DISPLAY") or "").strip():
            self.log_line.emit(
                "[warn] DISPLAY 为空，图形窗口可能无法弹出（本机桌面/X11 转发）"
            )
        self._run_generation += 1
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(work)
        proc.setProcessEnvironment(qenv)
        # setsid so stop() can kill the whole process group.
        # Persisted models put the real server in an inner session; this
        # process is only the log tail.
        proc.start("setsid", launch_argv)
        self._process = proc
        self._last_exit_code = None
        self.running_changed.emit(True)
        shown = " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in argv)
        self.log_line.emit(f"$ (cd {_eai_platform_bootstrap.shlex.quote(work)} && {shown})")
        self.status_message.emit(f"正在启动: {self._label}")

    def stop(self) -> None:
        if self._keep_on_exit and self._persist_pid_path:
            _eai_adapters_models_qwen_local.stop_persistent_model(self._persist_pid_path)
        if not self.is_running():
            if self._keep_on_exit:
                self.status_message.emit("已停止持久化模型进程")
                self.running_changed.emit(False)
            else:
                self.status_message.emit("当前没有运行中的 MuJoCo")
            return
        self.status_message.emit("正在停止 MuJoCo…")
        if self._process is not None:
            gen = self._run_generation
            pid = int(self._process.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._process.terminate()
            else:
                self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, lambda g=gen: self._force_kill(expected_gen=g))

    def shutdown(self) -> None:
        proc = self._process
        self._run_generation += 1
        self._process = None
        self._disconnect_process(proc)
        if self._keep_on_exit:
            # 日志 tail 可以结束；模型 session 留给下次界面复用。
            if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
                proc.terminate()
                proc.waitForFinished(400)
            if proc is not None:
                proc.deleteLater()
            return
        if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(proc.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.terminate()
            else:
                proc.terminate()
            proc.waitForFinished(1500)
        if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(proc.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
            else:
                proc.kill()
            proc.waitForFinished(800)
        if proc is not None:
            proc.deleteLater()
        self.running_changed.emit(False)

    def force_clear(self) -> None:
        """Hard-kill process group and drop launcher state so relaunch can proceed.

        Used by Psi/WAM redeploy when a previous QProcess is stuck or still
        marked Running after SIGTERM. Unlike stop(), this does not wait on a
        timer — it SIGKILLs immediately and clears ``_process``.

        Disconnects the old QProcess before clearing so a late finished(code=9)
        cannot wipe a newly started process reference.
        """
        if self._keep_on_exit and self._persist_pid_path:
            _eai_adapters_models_qwen_local.stop_persistent_model(self._persist_pid_path)
        proc = self._process
        self._run_generation += 1
        self._process = None
        self._disconnect_process(proc)
        if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            pid = int(proc.processId())
            if pid > 0:
                try:
                    _eai_platform_bootstrap.os.killpg(pid, _eai_platform_bootstrap.signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.kill()
                    except Exception:
                        pass
            else:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.waitForFinished(500)
            except Exception:
                pass
        if proc is not None:
            proc.deleteLater()
        self.running_changed.emit(False)

    def _force_kill(self, expected_gen: _eai_platform_bootstrap.Optional[int] = None) -> None:
        if expected_gen is not None and expected_gen != self._run_generation:
            return
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
        proc = self.sender()
        if not self._is_active_process(proc):
            return
        assert self._process is not None
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            text = line.rstrip()
            if text:
                self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        proc = self.sender()
        # Stale QProcess (killed during relaunch) must not clear the new one.
        if not self._is_active_process(proc):
            return
        self._last_exit_code = int(exit_code)
        self.running_changed.emit(False)
        label = self._label or "MuJoCo"
        if exit_code == 0:
            self.log_line.emit(f"--- MuJoCo 正常退出 ({label}) ---")
            self.status_message.emit(f"MuJoCo 已退出: {label}")
        else:
            self.log_line.emit(f"--- MuJoCo 退出 (code={exit_code}, {label}) ---")
            self.status_message.emit(f"MuJoCo 异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if not self._is_active_process(self.sender()):
            return
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self._last_exit_code = -1
            self.log_line.emit("[ERROR] 无法启动 MuJoCo 进程")
            self.status_message.emit("无法启动 MuJoCo")
            self.running_changed.emit(False)
            self._process = None

