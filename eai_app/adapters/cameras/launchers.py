"""adapters.cameras.launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class IsaacCamBridgeLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 Isaac 共享帧 → ROS2 相机桥（scripts/launch/run_isaac_cam_bridge.sh）。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def start(self, bridge_dir: str, *, hz: float = 30.0) -> None:
        if self.is_running():
            self.status_message.emit("相机桥已在运行")
            return
        if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_policies.ISAAC_CAM_BRIDGE_SCRIPT):
            self.status_message.emit(f"未找到脚本: {_eai_configuration_policies.ISAAC_CAM_BRIDGE_SCRIPT}")
            return
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((bridge_dir or "").strip()))
        if not bridge_dir:
            self.status_message.emit("请填写共享帧目录")
            return
        _eai_platform_bootstrap.os.makedirs(bridge_dir, exist_ok=True)
        hz_val = max(1.0, float(hz))
        env_exports = (
            f"export ISAAC_CAM_BRIDGE_DIR={_eai_platform_bootstrap.shlex.quote(bridge_dir)} "
            f"PYTHONUNBUFFERED=1 && "
            f"exec bash {_eai_platform_bootstrap.shlex.quote(_eai_configuration_policies.ISAAC_CAM_BRIDGE_SCRIPT)} --hz {hz_val}"
        )
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_models.EAI_DIR)
        proc.start("setsid", ["bash", "-lc", env_exports])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ ISAAC_CAM_BRIDGE_DIR={bridge_dir} bash scripts/launch/run_isaac_cam_bridge.sh --hz {hz_val}")
        self.status_message.emit("正在启动 Isaac 相机桥…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的相机桥")
            return
        self.status_message.emit("正在停止相机桥…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill_process)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None
        self.running_changed.emit(False)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                self.log_line.emit(line.rstrip())

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- 相机桥正常退出 ---")
            self.status_message.emit("Isaac 相机桥已停止")
        else:
            self.log_line.emit(f"--- 相机桥退出 (code={exit_code}) ---")
            self.status_message.emit(f"Isaac 相机桥异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"相机桥进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- 相机桥已被强制终止 ---")
            self.status_message.emit("相机桥已强制停止")


class LocalWebcamMjpegLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止本机 MJPEG 推流（examples/local_webcam_mjpeg_server.py）。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def start(self, device: str = "0", port: int = _eai_configuration_chat.LOCAL_WEBCAM_DEFAULT_PORT, bind: str = "127.0.0.1") -> None:
        if self.is_running():
            self.status_message.emit("本机推流已在运行")
            return
        if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_chat.LOCAL_WEBCAM_MJPEG_SCRIPT):
            self.status_message.emit(f"未找到推流脚本: {_eai_configuration_chat.LOCAL_WEBCAM_MJPEG_SCRIPT}")
            return
        device = (device or "0").strip() or "0"
        port = int(port)
        bind = (bind or "127.0.0.1").strip() or "127.0.0.1"
        args = [
            _eai_platform_bootstrap.sys.executable,
            _eai_configuration_chat.LOCAL_WEBCAM_MJPEG_SCRIPT,
            "--device",
            device,
            "--port",
            str(port),
            "--bind",
            bind,
        ]
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_models.EAI_DIR)
        env = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.start(args[0], args[1:])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit("$ " + " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in args))
        self.status_message.emit(f"正在启动本机推流 :{port} …")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的推流")
            return
        self.status_message.emit("正在停止本机推流…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill_process)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
            if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
                self._process.kill()
                self._process.waitForFinished(1000)
        self._process = None
        self.running_changed.emit(False)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                self.log_line.emit(line.rstrip())

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- 本机推流正常退出 ---")
            self.status_message.emit("本机推流已停止")
        else:
            self.log_line.emit(f"--- 本机推流退出 (code={exit_code}) ---")
            self.status_message.emit(f"本机推流异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.status_message.emit("本机推流启动失败（请检查 Python/OpenCV）")
            self.log_line.emit("推流进程 FailedToStart")
        elif error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"本机推流进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- 本机推流已被强制终止 ---")
            self.status_message.emit("本机推流已强制停止")


class SshReverseTunnelLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 SSH 反向隧道：ssh -N -R remote:127.0.0.1:local target。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def start(
        self,
        target: str,
        *,
        local_port: int = _eai_configuration_chat.LOCAL_WEBCAM_DEFAULT_PORT,
        remote_port: int = _eai_configuration_chat.LOCAL_WEBCAM_DEFAULT_PORT,
        extra_args: str = "",
    ) -> None:
        if self.is_running():
            self.status_message.emit("反向隧道已在运行")
            return
        target = (target or "").strip()
        if not target or "@" not in target:
            self.status_message.emit("请填写 SSH 远端，例如 user@10.0.0.5")
            return
        local_port = int(local_port)
        remote_port = int(remote_port)
        args = ["ssh", "-N", "-R", f"{remote_port}:127.0.0.1:{local_port}"]
        try:
            extras = _eai_platform_bootstrap.shlex.split(extra_args or "")
        except ValueError as exc:
            self.status_message.emit(f"额外 ssh 参数解析失败: {exc}")
            return
        args.extend(extras)
        args.append(target)
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.start(args[0], args[1:])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit("$ " + " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in args))
        self.status_message.emit(f"正在建立反向隧道 → {target}:{remote_port} …")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的隧道")
            return
        self.status_message.emit("正在停止反向隧道…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill_process)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
            if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
                self._process.kill()
                self._process.waitForFinished(1000)
        self._process = None
        self.running_changed.emit(False)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                self.log_line.emit(line.rstrip())

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- 反向隧道正常退出 ---")
            self.status_message.emit("反向隧道已停止")
        else:
            self.log_line.emit(f"--- 反向隧道退出 (code={exit_code}) ---")
            self.status_message.emit(f"反向隧道异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.status_message.emit("ssh 启动失败（请确认已安装 OpenSSH 客户端）")
            self.log_line.emit("隧道进程 FailedToStart")
        elif error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"隧道进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- 反向隧道已被强制终止 ---")
            self.status_message.emit("反向隧道已强制停止")

