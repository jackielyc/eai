"""adapters.models.perception launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class LingbotVisionLauncher(_eai_platform_bootstrap.QObject):
    """在独立进程跑 LingBot-Vision PCA，避免污染 viewer 的 ROS PYTHONPATH。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    result_ready = _eai_platform_bootstrap.pyqtSignal(dict)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._result: _eai_platform_bootstrap.Optional[dict] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start(
        self,
        input_path: str,
        *,
        variant: str = "small",
        ckpt: str = "",
        size: int = 512,
        mode: str = "square",
        dtype: str = "auto",
        device: str = "",
        out_dir: str = "",
    ) -> None:
        if self.is_running():
            self.status_message.emit("LingBot-Vision 正在运行")
            return
        script = _eai_configuration_tabs.LINGBOT_VISION_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        input_path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(input_path))
        if not _eai_platform_bootstrap.os.path.isfile(input_path):
            self.status_message.emit(f"输入图像不存在: {input_path}")
            return
        out_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(out_dir or _eai_configuration_tabs.LINGBOT_VISION_CACHE_DIR)
        )
        _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
        cache_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "huggingface")
        _eai_platform_bootstrap.os.makedirs(cache_dir, exist_ok=True)

        args = [
            "--input",
            input_path,
            "--out",
            out_dir,
            "--variant",
            variant,
            "--size",
            str(int(size)),
            "--mode",
            mode,
            "--dtype",
            dtype or "auto",
            "--cache-dir",
            cache_dir,
        ]
        if ckpt.strip():
            args.extend(["--ckpt", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(ckpt.strip()))])
        if device.strip():
            args.extend(["--device", device.strip()])

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("LINGBOT_VISION_ROOT", _eai_configuration_models.LINGBOT_VISION_ROOT_DEFAULT)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_VISION_PYTHON_DEFAULT):
            qenv.insert("LINGBOT_VISION_PYTHON", _eai_configuration_tabs.LINGBOT_VISION_PYTHON_DEFAULT)
        qenv.insert("HF_HOME", cache_dir)
        qenv.insert("HUGGINGFACE_HUB_CACHE", cache_dir)

        self._result = None
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_models.LINGBOT_VISION_ROOT_DEFAULT)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", [script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ bash scripts/launch/run_lingbot_vision.sh --variant {variant} --input {input_path}")
        self.status_message.emit(f"正在运行 LingBot-Vision PCA（{variant}）…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 LingBot-Vision")
            return
        self.status_message.emit("正在停止 LingBot-Vision…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(2500, self._force_kill)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(1500)
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(800)
        self._process = None
        self.running_changed.emit(False)

    def _force_kill(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
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
            if text.startswith("[RESULT] "):
                raw = text[len("[RESULT] ") :].strip()
                try:
                    self._result = _eai_platform_bootstrap.json.loads(raw)
                except _eai_platform_bootstrap.json.JSONDecodeError:
                    self.log_line.emit(text)
                    continue
                continue
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        result = self._result or {}
        if result.get("ok"):
            self.result_ready.emit(result)
            self.status_message.emit("LingBot-Vision PCA 完成")
        else:
            err = str(result.get("error") or f"退出码 {exit_code}")
            self.status_message.emit(f"LingBot-Vision 失败: {err}")
            self.log_line.emit(f"[ERROR] {err}")
            self.result_ready.emit({"ok": False, "error": err})
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 scripts/launch/run_lingbot_vision.sh")
            self.status_message.emit("无法启动 LingBot-Vision")
            self.running_changed.emit(False)


class LingbotDepthLauncher(_eai_platform_bootstrap.QObject):
    """在独立进程跑 LingBot-Depth，避免污染 viewer 的 ROS PYTHONPATH。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    result_ready = _eai_platform_bootstrap.pyqtSignal(dict)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._result: _eai_platform_bootstrap.Optional[dict] = None
        from eai_app.perception.depth_io import ResultStream
        self._output = ResultStream()
        self._cancelled = False
        self._result_context = {}

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start(
        self,
        *,
        example: str = "",
        rgb: str = "",
        depth: str = "",
        intrinsics: str = "",
        model: str = "",
        device: str = "auto",
        no_mask: bool = False,
        out_dir: str = "",
        allow_empty_depth: bool = False,
        sparse_z: float = 0.45,
        seed_u: _eai_platform_bootstrap.Optional[int] = None,
        seed_v: _eai_platform_bootstrap.Optional[int] = None,
        color_topic: str = "",
    ) -> None:
        if self.is_running():
            self.status_message.emit("LingBot-Depth 正在运行")
            return
        script = _eai_configuration_tabs.LINGBOT_DEPTH_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        if not out_dir:
            import tempfile
            _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_DEPTH_CACHE_DIR, exist_ok=True)
            out_dir = tempfile.mkdtemp(prefix="run_", dir=_eai_configuration_tabs.LINGBOT_DEPTH_CACHE_DIR)
        out_dir = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(out_dir))
        _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
        cache_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "huggingface")
        _eai_platform_bootstrap.os.makedirs(cache_dir, exist_ok=True)

        model = (model or _eai_configuration_tabs.LINGBOT_DEPTH_MODEL_DEFAULT).strip()
        args = [
            "--out",
            out_dir,
            "--model",
            model,
            "--device",
            device or "auto",
            "--cache-dir",
            cache_dir,
        ]
        if example.strip():
            args.extend(["--example", example.strip()])
        else:
            args.extend(
                [
                    "--rgb",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(rgb)),
                    "--intrinsics",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(intrinsics)),
                ]
            )
            depth_s = (depth or "").strip()
            if depth_s and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(depth_s))):
                args.extend(
                    ["--depth", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(depth_s))]
                )
            elif allow_empty_depth:
                args.append("--allow-empty-depth")
                args.extend(["--sparse-z", f"{float(sparse_z):.4f}"])
                if seed_u is not None:
                    args.extend(["--seed-u", str(int(seed_u))])
                if seed_v is not None:
                    args.extend(["--seed-v", str(int(seed_v))])
            else:
                args.extend(
                    [
                        "--depth",
                        _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(depth_s or "")),
                    ]
                )
        if no_mask:
            args.append("--no-mask")

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("LINGBOT_DEPTH_ROOT", _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_DEPTH_PYTHON_DEFAULT):
            qenv.insert("LINGBOT_DEPTH_PYTHON", _eai_configuration_tabs.LINGBOT_DEPTH_PYTHON_DEFAULT)
        qenv.insert("HF_HOME", cache_dir)
        qenv.insert("HUGGINGFACE_HUB_CACHE", cache_dir)

        from eai_app.perception.depth_io import ResultStream
        self._result = None
        self._output = ResultStream()
        self._cancelled = False
        self._result_context = {"color_topic": color_topic, "request_started_at": _eai_platform_bootstrap.time.time()}
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT)
        proc.setProcessEnvironment(qenv)
        self._process = proc
        proc.start("bash", [script, *args])
        self.running_changed.emit(True)
        src = f"example={example}" if example.strip() else f"rgb={rgb}"
        self.log_line.emit(f"$ bash scripts/launch/run_lingbot_depth.sh --model {model} {src}")
        self.status_message.emit("正在运行 LingBot-Depth…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 LingBot-Depth")
            return
        self._cancelled = True
        self.status_message.emit("正在停止 LingBot-Depth…")
        if self._process is not None:
            proc = self._process
            proc.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(2500, lambda: self._force_kill(proc))

    def shutdown(self) -> None:
        self._cancelled = True
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(1500)
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(800)
        self._process = None
        self.running_changed.emit(False)

    def _force_kill(self, proc=None) -> None:
        proc = proc if proc is not None else self._process
        try:
            if proc is not None and proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
                proc.kill()
        except RuntimeError:
            pass  # The captured previous QProcess has already been deleted.

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        for line in self._output.feed(bytes(self._process.readAllStandardOutput())):
            self.log_line.emit(line)
        self._result = self._output.result

    def _on_process_finished(self, exit_code: int, status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._on_process_output()
        for line in self._output.feed(final=True):
            self.log_line.emit(line)
        result = dict(self._output.result or {})
        result.update(self._result_context)
        proc = self._process
        self._process = None
        self.running_changed.emit(False)
        if result.get("ok") and exit_code == 0 and status == _eai_platform_bootstrap.QProcess.NormalExit and not self._cancelled:
            self.result_ready.emit(result)
            self.status_message.emit("LingBot-Depth 完成")
        else:
            err = "已停止" if self._cancelled else str(result.get("error") or f"退出码 {exit_code}，未返回有效结果")
            self.status_message.emit(f"LingBot-Depth 失败: {err}")
            self.log_line.emit(f"[ERROR] {err}")
            self.result_ready.emit(dict(self._result_context, ok=False, error=err))
        if proc is not None:
            proc.deleteLater()

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 scripts/launch/run_lingbot_depth.sh")
            self.status_message.emit("无法启动 LingBot-Depth")
            self.running_changed.emit(False)
            self._process = None
            self.result_ready.emit(dict(self._result_context, ok=False, error="无法启动 LingBot-Depth"))

