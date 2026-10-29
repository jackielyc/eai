"""adapters.robot.launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')

class LocalAiLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止本地 AI：SAM3 分割 HTTP 服务 + Ollama 对话服务。"""

    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None

    def is_sam3_running(self) -> bool:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            return True
        if _eai_perception_segmentation._sam3_server_process_running():
            return True
        try:
            return bool(
                _eai_perception_segmentation.check_sam3_server_health(
                    _eai_perception_segmentation.resolve_sam3_viewer_server_url(), timeout_s=0.6
                )
            )
        except Exception:
            return False

    def start_deploy(self) -> None:
        self._ensure_ollama()
        self._start_sam3()

    def stop(self) -> None:
        _eai_adapters_models_qwen_local.stop_persistent_model(
            getattr(self, "_persist_pid_path", "") or _eai_adapters_models_qwen_local.persistent_slot_paths("sam3")[1]
        )
        if self._process is not None:
            if self._process.state() == _eai_platform_bootstrap.QProcess.Running:
                self._process.terminate()
                _eai_platform_bootstrap.QTimer.singleShot(2000, self._force_kill_process)
            else:
                self._process = None
        _eai_perception_segmentation._stop_sam3_server_processes()
        self.running_changed.emit(False)
        self.status_message.emit("已停止本地 SAM3 服务")

    def shutdown(self) -> None:
        # 只结束日志 tail。SAM3 在独立 session，界面退出后继续服务。
        proc = self._process
        self._process = None
        if proc is not None and proc.state() == _eai_platform_bootstrap.QProcess.Running:
            proc.terminate()
            proc.waitForFinished(400)

    def _ensure_ollama(self) -> None:
        if _eai_perception_segmentation.check_ollama_server_health():
            self.status_message.emit("Ollama 已部署，跳过启动")
            return
        ollama_bin = _eai_platform_bootstrap.shutil.which("ollama")
        if not ollama_bin:
            self.status_message.emit(
                "未检测到 Ollama，请安装: https://ollama.com 后执行 ollama pull qwen3.5:4b"
            )
            return
        if _eai_platform_bootstrap.QProcess.startDetached(ollama_bin, ["serve"]):
            self.status_message.emit("正在启动 Ollama 服务 (127.0.0.1:11434)…")
        else:
            self.status_message.emit("Ollama 启动失败")

    def _start_sam3(self) -> None:
        url = _eai_perception_segmentation.resolve_sam3_viewer_server_url()
        if _eai_perception_segmentation.check_sam3_server_health(url, timeout_s=1.0):
            self.status_message.emit(f"SAM3 已部署，跳过启动: {url}")
            self.running_changed.emit(True)
            return
        if self.is_sam3_running():
            self.status_message.emit("SAM3 服务已在运行")
            self.running_changed.emit(True)
            return

        script = _eai_perception_segmentation.resolve_sam3_run_script()
        if script is None:
            hint = (
                "cd eai && bash run_sam3.sh --host 0.0.0.0"
                if _eai_perception_segmentation.is_running_in_docker()
                else f"未找到 {_eai_configuration_perception.SAM3_RUN_SCRIPT_NAME}，请确认与 show_camera_topics.py 同目录"
            )
            self.status_message.emit(f"无法启动 SAM3: {hint}")
            return

        bind_host = _eai_perception_segmentation.resolve_sam3_bind_host()
        script_dir = _eai_platform_bootstrap.os.path.dirname(script)
        cmd = f"exec bash {_eai_platform_bootstrap.shlex.quote(_eai_platform_bootstrap.os.path.basename(script))} --host {_eai_platform_bootstrap.shlex.quote(bind_host)}"
        shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(cmd, "sam3")
        self._persist_pid_path = pid_path
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(script_dir)
        proc.start("setsid", ["bash", "-lc", shell])
        self._process = proc
        self.running_changed.emit(True)
        viewer_url = _eai_perception_segmentation.resolve_sam3_viewer_server_url()
        self.status_message.emit(
            f"正在启动 SAM3 ({bind_host}:{_eai_configuration_perception.SAM3_DEFAULT_PORT})，viewer URL: {viewer_url}"
        )

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            text = line.strip()
            if text:
                self.status_message.emit(f"[SAM3] {text}")

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code != 0:
            self.status_message.emit(f"SAM3 进程退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"SAM3 进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None


class RobotStackLauncher(_eai_platform_bootstrap.QObject):
    """后台启动 robot-service 与手/臂精简服务栈（不阻塞 UI）。"""

    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    stack_status_changed = _eai_platform_bootstrap.pyqtSignal(str, str)
    start_robot_on_main = _eai_platform_bootstrap.pyqtSignal()
    start_base_on_main = _eai_platform_bootstrap.pyqtSignal()

    def __init__(self, node: "CameraTopicNode", parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._robot_status = "未运行"
        self._base_status = "未运行"
        self._robot_launch_pending = False
        self._base_launch_pending = False
        self._poll_stop = _eai_platform_bootstrap.threading.Event()
        self._status_lock = _eai_platform_bootstrap.threading.Lock()
        self.start_robot_on_main.connect(self._launch_robot_service, _eai_platform_bootstrap.Qt.QueuedConnection)
        self.start_base_on_main.connect(self._launch_base_services, _eai_platform_bootstrap.Qt.QueuedConnection)
        self._poll_thread = _eai_platform_bootstrap.threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def shutdown(self) -> None:
        self._poll_stop.set()
        self._poll_thread.join(timeout=1.0)

    def get_cached_status(self) -> _eai_platform_bootstrap.Tuple[str, str]:
        with self._status_lock:
            return self._robot_status, self._base_status

    def start_stack(self) -> None:
        robot, base = self.get_cached_status()
        if base == "运行中":
            self.status_message.emit("手/臂服务栈已在运行")
            return
        if base == "启动中" or self._base_launch_pending:
            self.status_message.emit("手/臂服务栈正在启动…")
            return
        if self._node.is_hal_arm_ready() or robot == "就绪":
            self._launch_stack_services()
            return
        if robot == "启动中" or self._robot_launch_pending:
            self.status_message.emit("robot-service 启动中，请等待 (~3 分钟)…")
            return
        self._launch_robot_service()

    def _launch_robot_service(self) -> None:
        if self._robot_launch_pending:
            self.status_message.emit("robot-service 已在启动中")
            return
        script = self._resolve_script(_eai_configuration_policies.ROBOT_SERVICE_SCRIPT)
        if script is None:
            self.status_message.emit("未找到 start_ros_service.sh，请设置 A2D_SCRIPTS_DIR")
            return
        script_dir = _eai_platform_bootstrap.os.path.dirname(script)
        cmd = (
            f"{_eai_adapters_robot_services.build_ros_shell_prefix()} && cd {script_dir} && "
            f"exec bash {_eai_platform_bootstrap.os.path.basename(script)}"
        )
        if not _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self.status_message.emit("robot-service 启动失败（startDetached 返回 false）")
            return
        self._robot_launch_pending = True
        self.status_message.emit("正在启动 robot-service（约 3 分钟）…")

    def _launch_stack_services(self) -> None:
        robot, base = self.get_cached_status()
        if base == "运行中":
            self.status_message.emit("手/臂服务栈已在运行")
            return
        if self._base_launch_pending or base == "启动中":
            self.status_message.emit("手/臂服务栈正在启动…")
            return
        # 永远不用 Docker：宿主机使能桥 + Ruiyan hand_node
        host_script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_arm_hand.sh",
        )
        ruiyan_script = _eai_platform_bootstrap.os.path.join(
            str(PROJECT_ROOT),
            "scripts",
            "start_host_ruiyan.sh",
        )
        if _eai_platform_bootstrap.os.path.isfile(host_script):
            if not _eai_platform_bootstrap.QProcess.startDetached("bash", [host_script, "start"]):
                self.status_message.emit("宿主机手/臂桥启动失败")
                return
            if _eai_platform_bootstrap.os.path.isfile(ruiyan_script):
                _eai_platform_bootstrap.QProcess.startDetached("bash", [ruiyan_script, "start"])
            self._base_launch_pending = True
            self.status_message.emit(
                "正在启动宿主机手/臂使能桥 + Ruiyan（无 Docker）…"
            )
            return
        script = self._resolve_script(_eai_configuration_policies.STACK_SERVICES_SCRIPT)
        if script is None:
            tried = _eai_adapters_robot_launch_utils.resolve_scripts_pack_dir() or "(未找到 scripts_pack 目录)"
            self.status_message.emit(
                f"未找到宿主机桥与 {_eai_configuration_policies.STACK_SERVICES_SCRIPT}（已查: {tried}）"
            )
            return
        script_dir = _eai_platform_bootstrap.os.path.dirname(script)
        cmd = (
            f"{_eai_adapters_robot_services.build_ros_shell_prefix()} && cd {script_dir} && "
            f"exec bash {_eai_platform_bootstrap.os.path.basename(script)}"
        )
        if not _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self.status_message.emit("手/臂服务栈启动失败（startDetached 返回 false）")
            return
        self._base_launch_pending = True
        self.status_message.emit(
            "正在启动手/臂服务（本机 scripts_pack，无 Docker）…"
        )

    def _launch_base_services(self) -> None:
        """兼容旧信号连接，实际启动精简栈。"""
        self._launch_stack_services()

    def _resolve_script(self, relative_name: str) -> _eai_platform_bootstrap.Optional[str]:
        scripts_dir = _eai_adapters_robot_launch_utils.resolve_scripts_pack_dir()
        if scripts_dir is None:
            return None
        path = _eai_platform_bootstrap.os.path.join(scripts_dir, relative_name)
        return path if _eai_platform_bootstrap.os.path.isfile(path) else None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            self._refresh_status_cache()
            robot, base = self.get_cached_status()
            self.stack_status_changed.emit(robot, base)
            interval = 15.0 if base == "运行中" else 5.0
            self._poll_stop.wait(interval)

    def _refresh_status_cache(self) -> None:
        base_running = _eai_adapters_robot_services._is_stack_services_running()
        robot_running = _eai_adapters_robot_services._pgrep_pattern("robot-service")
        hal_ready = self._node.is_hal_arm_ready()

        if base_running:
            self._base_launch_pending = False
            base_status = "运行中"
        elif self._base_launch_pending:
            base_status = "启动中"
        else:
            base_status = "未运行"

        if robot_running:
            robot_status = "启动中"
        elif hal_ready:
            robot_status = "就绪"
        else:
            robot_status = "未运行"

        with self._status_lock:
            self._robot_status = robot_status
            self._base_status = base_status

        self._maybe_continue_stack_startup(base_running, robot_running, hal_ready)

    def _maybe_continue_stack_startup(
        self,
        base_running: bool,
        robot_running: bool,
        hal_ready: bool,
    ) -> None:
        if base_running or self._base_launch_pending:
            return
        if not self._robot_launch_pending:
            return
        if robot_running:
            return
        if not hal_ready:
            return
        self._robot_launch_pending = False
        self.status_message.emit("robot-service 完成，正在启动手/臂服务栈…")
        self.start_base_on_main.emit()


class RrdReplayLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 rrd_replay 节点，并在就绪后自动调用 start_replay 服务。"""

    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    node_active_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, node: "CameraTopicNode", parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._rrd_path = ""
        self._target_loops = _eai_configuration_policies.REPLAY_LOOP_COUNT_DEFAULT
        self._completed_loops = 0
        self._loop_restart_pending = False
        self._finishing_after_complete = False
        self._auto_start_attempts = 0
        self._max_auto_start_attempts = 240
        self._auto_start_min_attempts = 6  # launch 后至少等待 ~3s
        self._auto_start_timer = _eai_platform_bootstrap.QTimer()
        self._auto_start_timer.setInterval(500)
        self._auto_start_timer.timeout.connect(self._try_auto_start_replay)
        node.ros_bridge.replay_state_changed.connect(self._on_replay_state_changed)

    def get_loop_progress(self) -> _eai_platform_bootstrap.Tuple[int, int]:
        return self._completed_loops, self._target_loops

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def current_rrd_path(self) -> str:
        return self._rrd_path

    def start(self, rrd_path: str, loop_count: int = _eai_configuration_policies.REPLAY_LOOP_COUNT_DEFAULT) -> None:
        if self.is_running():
            self.status_message.emit("RRD 回放已在运行")
            return

        self._target_loops = max(1, min(int(loop_count), _eai_configuration_policies.REPLAY_LOOP_COUNT_MAX))
        self._completed_loops = 0
        self._loop_restart_pending = False

        resolved_path = _eai_ui_datasets_notes.resolve_rrd_path(rrd_path)
        if not _eai_platform_bootstrap.os.path.isfile(resolved_path):
            self.status_message.emit(f"RRD 文件不存在: {resolved_path}")
            return

        if (
            _eai_adapters_robot_services._pgrep_pattern("rrd_replay.launch.py")
            or _eai_adapters_robot_services._pgrep_pattern("rrd_replay_node")
        ):
            self.status_message.emit("正在停止已有 rrd_replay 进程…")
            _eai_adapters_robot_services._stop_existing_rrd_replay()

        self._rrd_path = resolved_path
        quoted = resolved_path.replace("'", "'\\''")
        workspace = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
        # 与 a2d-tele GUI 保持一致的 launch 参数
        cmd = (
            f"source /opt/ros/humble/setup.bash && "
            f"cd {workspace} && source install/setup.bash && "
            f"ros2 launch rrd_replay rrd_replay.launch.py "
            f"rrd_path:='{quoted}' force_generate_csv:=false enable_keyboard_stop:=false"
        )
        dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
        if dds and _eai_platform_bootstrap.os.path.isfile(dds):
            cmd = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && {cmd}"
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(workspace)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.node_active_changed.emit(True)
        self._node.prepare_for_rrd_replay()
        self.status_message.emit(
            f"正在启动 RRD 回放: {_eai_platform_bootstrap.os.path.basename(resolved_path)} "
            f"（共 {self._target_loops} 次）"
        )
        self._auto_start_attempts = 0
        self._auto_start_timer.start()

    def stop(self, *, auto_complete: bool = False) -> None:
        completed = self._completed_loops
        total = self._target_loops
        self._auto_start_timer.stop()
        self._node._stop_replay_prep_mode_timer()
        self._loop_restart_pending = False
        self._completed_loops = 0
        self._target_loops = _eai_configuration_policies.REPLAY_LOOP_COUNT_DEFAULT
        if self.is_running() or _eai_adapters_robot_services._ros_service_exists(_eai_configuration_policies.REPLAY_STOP_SERVICE):
            _eai_adapters_robot_services._ros2_trigger_service(_eai_configuration_policies.REPLAY_STOP_SERVICE, timeout_sec=5.0)
            self._node.request_replay_stop()
        if self._process is not None:
            if self._process.state() == _eai_platform_bootstrap.QProcess.Running:
                self._process.terminate()
                _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill_process)
            else:
                self._process = None
                self.node_active_changed.emit(False)
        else:
            self.node_active_changed.emit(False)
        if auto_complete:
            self.status_message.emit(f"全部回放完成 ({completed}/{total})")
        else:
            self.status_message.emit("正在停止 RRD 回放…")

    def shutdown(self) -> None:
        self._auto_start_timer.stop()
        self._node._stop_replay_prep_mode_timer()
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None
        self.node_active_changed.emit(False)

    def _try_auto_start_replay(self) -> None:
        self._auto_start_attempts += 1
        if self._auto_start_attempts < self._auto_start_min_attempts:
            return
        if not _eai_adapters_robot_services._ros_service_exists(_eai_configuration_policies.REPLAY_START_SERVICE):
            if self._auto_start_attempts >= self._max_auto_start_attempts:
                self._auto_start_timer.stop()
                self.status_message.emit("等待 rrd_replay 服务超时（120s）")
            elif self._auto_start_attempts % 4 == 0:
                self.status_message.emit("等待 rrd_replay 服务就绪…")
            return

        ok, detail = _eai_adapters_robot_services._ros2_trigger_service(_eai_configuration_policies.REPLAY_START_SERVICE, timeout_sec=15.0)
        if ok:
            self._auto_start_timer.stop()
            self._node.get_logger().info(f"auto start_replay OK: {detail}")
            self.status_message.emit(
                f"回放已启动 第 {self._completed_loops + 1}/{self._target_loops} 次"
            )
            return

        self._node.get_logger().warn(f"auto start_replay 失败: {detail}")
        if self._auto_start_attempts >= self._max_auto_start_attempts:
            self._auto_start_timer.stop()
            self.status_message.emit(f"自动启动回放失败: {detail[:120]}")
        elif self._auto_start_attempts % 4 == 0:
            self.status_message.emit(f"重试 start_replay… ({detail[:80]})")

    def _on_replay_state_changed(self, state: int) -> None:
        if state != _eai_configuration_policies.REPLAY_STATE_FINISHED:
            return
        if self._loop_restart_pending:
            return
        if not self.is_running() and not _eai_adapters_robot_services._ros_service_exists(_eai_configuration_policies.REPLAY_START_SERVICE):
            return

        self._completed_loops += 1
        self.status_message.emit(
            f"回放完成 {self._completed_loops}/{self._target_loops}"
        )
        self._node.get_logger().info(
            f"rrd replay finished loop {self._completed_loops}/{self._target_loops}"
        )
        if self._completed_loops >= self._target_loops:
            _eai_platform_bootstrap.QTimer.singleShot(300, self._finish_all_replay_loops)
            return

        self._loop_restart_pending = True
        _eai_platform_bootstrap.QTimer.singleShot(400, self._restart_next_loop)

    def _finish_all_replay_loops(self) -> None:
        if not self.is_running() and not _eai_adapters_robot_services._ros_service_exists(_eai_configuration_policies.REPLAY_STOP_SERVICE):
            return
        self._finishing_after_complete = True
        self.stop(auto_complete=True)

    def _restart_next_loop(self) -> None:
        if self._completed_loops >= self._target_loops:
            self._loop_restart_pending = False
            return
        ok_stop, stop_detail = _eai_adapters_robot_services._ros2_trigger_service(_eai_configuration_policies.REPLAY_STOP_SERVICE, timeout_sec=8.0)
        if not ok_stop:
            self._node.get_logger().warn(f"loop stop_replay failed: {stop_detail}")
        _eai_platform_bootstrap.QTimer.singleShot(300, self._start_next_loop)

    def _start_next_loop(self) -> None:
        self._loop_restart_pending = False
        if self._completed_loops >= self._target_loops:
            return
        ok, detail = _eai_adapters_robot_services._ros2_trigger_service(_eai_configuration_policies.REPLAY_START_SERVICE, timeout_sec=15.0)
        if ok:
            self.status_message.emit(
                f"开始第 {self._completed_loops + 1}/{self._target_loops} 次回放"
            )
            self._node.get_logger().info(
                f"rrd replay loop {self._completed_loops + 1}/{self._target_loops} started"
            )
        else:
            self.status_message.emit(f"下一轮回放启动失败: {detail[:120]}")
            self._node.get_logger().warn(f"loop start_replay failed: {detail}")

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            text = line.strip()
            if text:
                self._node.get_logger().info(f"[rrd_replay] {text}")

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._auto_start_timer.stop()
        self._process = None
        self.node_active_changed.emit(False)
        if self._finishing_after_complete:
            self._finishing_after_complete = False
            return
        self.status_message.emit(f"RRD 回放节点已退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"RRD 回放进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()

