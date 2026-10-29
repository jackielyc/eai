"""adapters.models.generation launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class LingbotMapLauncher(_eai_platform_bootstrap.QObject):
    """在独立进程跑 LingBot-Map demo.py（viser 交互式 3D 重建）。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    viser_ready = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._viser_url: str = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start(
        self,
        *,
        image_folder: str = "",
        video_path: str = "",
        model: str = "",
        port: int = _eai_configuration_tabs.LINGBOT_MAP_PORT_DEFAULT,
        first_k: int = 0,
        stride: int = 1,
        fps: int = 10,
        mode: str = "streaming",
        mask_sky: bool = False,
        use_sdpa: bool = True,
    ) -> None:
        if self.is_running():
            self.status_message.emit("LingBot-Map 正在运行")
            return
        script = _eai_configuration_tabs.LINGBOT_MAP_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        if not image_folder and not video_path:
            self.status_message.emit("需要图像目录或视频")
            return

        cache_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "huggingface")
        _eai_platform_bootstrap.os.makedirs(cache_dir, exist_ok=True)
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_MAP_MODEL_CACHE, exist_ok=True)

        args = [
            "--port",
            str(int(port)),
            "--mode",
            mode or "streaming",
            "--stride",
            str(max(1, int(stride))),
            "--fps",
            str(max(1, int(fps))),
            "--cache-dir",
            _eai_configuration_tabs.LINGBOT_MAP_MODEL_CACHE,
            "--model",
            (model or _eai_configuration_tabs.LINGBOT_MAP_CKPT_DEFAULT).strip(),
        ]
        if image_folder:
            args.extend(
                ["--image_folder", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image_folder))]
            )
        if video_path:
            args.extend(
                ["--video_path", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(video_path))]
            )
        if int(first_k) > 0:
            args.extend(["--first_k", str(int(first_k))])
        if mask_sky:
            args.append("--mask_sky")
        if use_sdpa:
            args.append("--use_sdpa")
        else:
            args.append("--no_use_sdpa")

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("LINGBOT_MAP_ROOT", _eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_MAP_PYTHON_DEFAULT):
            qenv.insert("LINGBOT_MAP_PYTHON", _eai_configuration_tabs.LINGBOT_MAP_PYTHON_DEFAULT)
        qenv.insert("HF_HOME", cache_dir)
        qenv.insert("HUGGINGFACE_HUB_CACHE", cache_dir)

        self._viser_url = f"http://127.0.0.1:{int(port)}"
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", [script, *args])
        self._process = proc
        self.running_changed.emit(True)
        src = image_folder or video_path
        self.log_line.emit(f"$ bash run_lingbot_map.sh … {src}")
        self.status_message.emit(f"正在运行 LingBot-Map（viser {self._viser_url}）…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 LingBot-Map")
            return
        self.status_message.emit("正在停止 LingBot-Map…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(2000)
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
            if text.startswith("[VISER] "):
                url = text[len("[VISER] ") :].strip()
                if url:
                    self._viser_url = url
                    self.viser_ready.emit(url)
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.status_message.emit("LingBot-Map 已结束")
        else:
            self.status_message.emit(f"LingBot-Map 退出码 {exit_code}")
            self.log_line.emit(f"[ERROR] 退出码 {exit_code}")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 run_lingbot_map.sh")
            self.status_message.emit("无法启动 LingBot-Map")
            self.running_changed.emit(False)


class LingbotVideoLauncher(_eai_platform_bootstrap.QObject):
    """在独立进程跑 LingBot-Video DiT 推理。"""

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
        *,
        mode: str,
        output: str,
        prompt_json: str = "",
        prompt: str = "",
        image: str = "",
        model_dir: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        steps: int = 40,
        seed: int = 42,
        fps: int = 24,
        guidance_scale: float = 3.0,
        run_refiner: bool = False,
    ) -> None:
        if self.is_running():
            self.status_message.emit("LingBot-Video 正在运行")
            return
        script = _eai_configuration_tabs.LINGBOT_VIDEO_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        if not prompt_json and not prompt:
            self.status_message.emit("需要 prompt_json 或文本 prompt")
            return

        out_path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(output))
        _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(out_path) or ".", exist_ok=True)
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_VIDEO_MODEL_CACHE, exist_ok=True)
        cache_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "huggingface")
        _eai_platform_bootstrap.os.makedirs(cache_dir, exist_ok=True)

        args = [
            "--mode",
            mode,
            "--output",
            out_path,
            "--cache-dir",
            _eai_configuration_tabs.LINGBOT_VIDEO_MODEL_CACHE,
            "--height",
            str(int(height)),
            "--width",
            str(int(width)),
            "--num_frames",
            str(int(num_frames)),
            "--steps",
            str(int(steps)),
            "--seed",
            str(int(seed)),
            "--fps",
            str(int(fps)),
            "--guidance_scale",
            str(float(guidance_scale)),
            "--backend",
            "diffusers",
        ]
        if model_dir.strip():
            args.extend(
                ["--model_dir", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(model_dir.strip()))]
            )
        if prompt_json.strip():
            args.extend(
                [
                    "--prompt_json",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(prompt_json.strip())),
                ]
            )
        if prompt.strip():
            args.extend(["--prompt", prompt.strip()])
        if image.strip():
            args.extend(
                ["--image", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image.strip()))]
            )
        if run_refiner:
            args.append("--run_refiner")

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("LINGBOT_VIDEO_ROOT", _eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_VIDEO_PYTHON_DEFAULT):
            qenv.insert("LINGBOT_VIDEO_PYTHON", _eai_configuration_tabs.LINGBOT_VIDEO_PYTHON_DEFAULT)
        qenv.insert("HF_HOME", cache_dir)
        qenv.insert("HUGGINGFACE_HUB_CACHE", cache_dir)
        qenv.insert("DIFFUSERS_ATTN_BACKEND", "_native_flash")
        qenv.insert("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa")

        self._result = None
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", [script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ bash run_lingbot_video.sh --mode {mode} --output {out_path}")
        self.status_message.emit(f"正在运行 LingBot-Video（{mode}）…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 LingBot-Video")
            return
        self.status_message.emit("正在停止 LingBot-Video…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(4000, self._force_kill)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(2000)
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
            self.status_message.emit("LingBot-Video 完成")
        else:
            err = str(result.get("error") or f"退出码 {exit_code}")
            self.status_message.emit(f"LingBot-Video 失败: {err}")
            self.log_line.emit(f"[ERROR] {err}")
            self.result_ready.emit({"ok": False, "error": err})
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 run_lingbot_video.sh")
            self.status_message.emit("无法启动 LingBot-Video")
            self.running_changed.emit(False)


class LingbotWorldLauncher(_eai_platform_bootstrap.QObject):
    """在独立进程跑 LingBot-World-V2 generate.py。"""

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
        *,
        image: str,
        action_path: str,
        prompt: str,
        output: str,
        ckpt_dir: str = "",
        size: str = "480*832",
        frame_num: int = 17,
        seed: int = 42,
        local_attn_size: int = 18,
        sink_size: int = 6,
        t5_cpu: bool = True,
        offload_model: bool = True,
    ) -> None:
        if self.is_running():
            self.status_message.emit("LingBot-World 正在运行")
            return
        script = _eai_configuration_tabs.LINGBOT_WORLD_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        out_path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(output))
        _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(out_path) or ".", exist_ok=True)

        args = [
            "--image",
            _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image)),
            "--action_path",
            _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(action_path)),
            "--prompt",
            prompt or _eai_configuration_tabs.LINGBOT_WORLD_PROMPT_DEFAULT,
            "--output",
            out_path,
            "--ckpt_dir",
            (ckpt_dir or _eai_configuration_tabs.LINGBOT_WORLD_CKPT_DEFAULT).strip(),
            "--size",
            size or "480*832",
            "--frame_num",
            str(int(frame_num)),
            "--seed",
            str(int(seed)),
            "--local_attn_size",
            str(int(local_attn_size)),
            "--sink_size",
            str(int(sink_size)),
        ]
        if t5_cpu:
            args.append("--t5_cpu")
        else:
            args.append("--no_t5_cpu")
        if offload_model:
            args.append("--offload_model")
        else:
            args.append("--no_offload_model")

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("LINGBOT_WORLD_ROOT", _eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT)
        if _eai_platform_bootstrap.os.path.isfile(_eai_configuration_tabs.LINGBOT_WORLD_PYTHON_DEFAULT):
            qenv.insert("LINGBOT_WORLD_PYTHON", _eai_configuration_tabs.LINGBOT_WORLD_PYTHON_DEFAULT)

        self._result = None
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", [script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ bash run_lingbot_world.sh --image {image}")
        self.status_message.emit("正在运行 LingBot-World…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 LingBot-World")
            return
        self.status_message.emit("正在停止 LingBot-World…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(5000, self._force_kill)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(2500)
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
            self.status_message.emit("LingBot-World 完成")
        else:
            err = str(result.get("error") or f"退出码 {exit_code}")
            self.status_message.emit(f"LingBot-World 失败: {err}")
            self.log_line.emit(f"[ERROR] {err}")
            self.result_ready.emit({"ok": False, "error": err})
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 run_lingbot_world.sh")
            self.status_message.emit("无法启动 LingBot-World")
            self.running_changed.emit(False)

