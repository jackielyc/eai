"""adapters.models.reward launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class RewardModelLauncher(_eai_platform_bootstrap.QObject):
    """跑 RLinf reward 打分（single/pair/batch）或真机 teleop。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    result_ready = _eai_platform_bootstrap.pyqtSignal(dict)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._result: _eai_platform_bootstrap.Optional[dict] = None
        self._job: str = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start_score(
        self,
        *,
        mode: str,
        rlinf_root: str,
        python_bin: str = "",
        ckpt: str,
        model_type: str = "resnet",
        arch: str = "resnet18",
        image_size: str = "3,224,224",
        device: str = "auto",
        threshold: float = 0.5,
        task: str = "",
        image: str = "",
        image_a: str = "",
        image_b: str = "",
        dataset: str = "",
        max_samples: int = 0,
        batch_size: int = 16,
    ) -> None:
        if self.is_running():
            self.status_message.emit("Reward 评测正在运行")
            return
        script = _eai_configuration_tabs.RLINF_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_rlinf_root(rlinf_root)
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_rlinf_python(root)
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.RLINF_REWARD_CACHE_DIR, exist_ok=True)

        args = [
            "--mode",
            mode,
            "--ckpt",
            _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(ckpt)),
            "--model-type",
            model_type,
            "--arch",
            arch,
            "--image-size",
            image_size,
            "--device",
            device or "auto",
            "--threshold",
            str(threshold),
            "--batch-size",
            str(int(batch_size)),
            "--precision",
            "bf16" if model_type in ("vlm", "history_vlm") else "fp32",
        ]
        if task.strip():
            args.extend(["--task", task.strip()])
        if mode == "single":
            args.extend(["--image", _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image))])
        elif mode == "pair":
            args.extend(
                [
                    "--image-a",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image_a)),
                    "--image-b",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(image_b)),
                ]
            )
        elif mode == "batch":
            args.extend(
                [
                    "--dataset",
                    _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(dataset)),
                    "--max-samples",
                    str(int(max_samples)),
                ]
            )
        else:
            self.status_message.emit(f"未知 mode: {mode}")
            return

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("RLINF_ROOT", root)
        qenv.insert("RLINF_PYTHON", py)

        self._result = None
        self._job = mode
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", [script, *args])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ bash scripts/launch/run_reward_model.sh --mode {mode} --ckpt {ckpt}")
        self.status_message.emit(f"正在运行 Reward 评测（{mode}）…")

    def start_teleop(
        self,
        *,
        rlinf_root: str,
        python_bin: str = "",
        config_name: str = "realworld_teleop",
        ckpt: str = "",
        threshold: float = 0.2,
        model_type: str = "resnet",
        arch: str = "resnet18",
        image_size: str = "3,128,128",
    ) -> None:
        if self.is_running():
            self.status_message.emit("Reward teleop 正在运行")
            return
        root = _eai_adapters_models_discovery.resolve_rlinf_root(rlinf_root)
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_rlinf_python(root)
        src = _eai_platform_bootstrap.os.path.join(root, "examples", "reward", "eval_realworld_teleop.py")
        cfg_dir = _eai_platform_bootstrap.os.path.join(root, "examples", "reward", "config")
        if not _eai_platform_bootstrap.os.path.isfile(src):
            self.status_message.emit(f"未找到: {src}")
            return
        log_dir = _eai_platform_bootstrap.os.path.join(
            root, "logs", _eai_platform_bootstrap.time.strftime("%Y%m%d-%H:%M:%S") + f"-{config_name}"
        )
        _eai_platform_bootstrap.os.makedirs(log_dir, exist_ok=True)
        overrides = [
            f"runner.logger.log_path={log_dir}",
            f"reward.model.model_type={model_type}",
            f"reward.model.arch={arch}",
            f"reward.model.image_size=[{image_size.replace('x', ',').replace(' ', '')}]",
            f"reward.reward_threshold={threshold}",
            "reward.use_reward_model=True",
            "reward.standalone_realworld=True",
        ]
        if ckpt.strip():
            ckpt_abs = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(ckpt.strip()))
            overrides.append(f"reward.model.model_path={ckpt_abs}")
            overrides.append("reward.model.pretrained=False")

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("RLINF_ROOT", root)
        qenv.insert("REWARD_PATH", _eai_platform_bootstrap.os.path.join(root, "examples", "reward"))
        qenv.insert("REPO_PATH", root)
        qenv.insert("PYTHONPATH", root)

        self._result = None
        self._job = "teleop"
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root)
        proc.setProcessEnvironment(qenv)
        args = [
            src,
            f"--config-path={cfg_dir}",
            f"--config-name={config_name}",
            *overrides,
        ]
        proc.start(py, args)
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(
            f"$ {py} examples/reward/eval_realworld_teleop.py "
            f"--config-name {config_name} …"
        )
        self.status_message.emit("正在启动 Reward 真机 teleop…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 Reward 任务")
            return
        self.status_message.emit("正在停止 Reward 任务…")
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
        job = self._job or "reward"
        if job == "teleop":
            if exit_code == 0:
                self.status_message.emit("Reward teleop 已退出")
                self.result_ready.emit({"ok": True, "mode": "teleop"})
            else:
                err = f"退出码 {exit_code}"
                self.status_message.emit(f"Reward teleop 失败: {err}")
                self.log_line.emit(f"[ERROR] {err}")
                self.result_ready.emit({"ok": False, "error": err, "mode": "teleop"})
        elif result.get("ok"):
            self.result_ready.emit(result)
            self.status_message.emit(f"Reward 评测完成（{result.get('mode', job)}）")
        else:
            err = str(result.get("error") or f"退出码 {exit_code}")
            self.status_message.emit(f"Reward 评测失败: {err}")
            self.log_line.emit(f"[ERROR] {err}")
            self.result_ready.emit({"ok": False, "error": err, "mode": job})
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 Reward 进程")
            self.status_message.emit("无法启动 Reward 评测")
            self.running_changed.emit(False)


class RewardWorkflowLauncher(_eai_platform_bootstrap.QObject):
    """启动 RLinf 奖励模型工作流（采集 / 预处理 / 训练 / 预处理→训练）。"""

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
        rlinf_root: str,
        python_bin: str = "",
        config_name: str = "",
        dataset_type: str = "resnet",
        raw_data_path: str = "",
        output_dir: str = "",
        num_samples: int = 0,
        val_split: float = 0.2,
        fail_success_ratio: float = 2.0,
        seed: int = 42,
        keep_last_frame: bool = True,
        cuda_devices: str = "",
        hydra_overrides: _eai_platform_bootstrap.Sequence[str] = (),
    ) -> None:
        if self.is_running():
            self.status_message.emit("Reward 训练工作流正在运行")
            return
        script = _eai_configuration_tabs.RLINF_REWARD_WORKFLOW_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_rlinf_root(rlinf_root)
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_rlinf_python(root)
        mode_norm = (mode or "preprocess").strip().lower()
        if mode_norm not in ("preprocess", "train", "collect", "pipeline"):
            self.status_message.emit(f"未知 mode: {mode}")
            return
        ds = (dataset_type or "resnet").strip().lower() or "resnet"
        if ds not in ("resnet", "qwentrend"):
            self.status_message.emit(f"未知 dataset_type: {dataset_type}")
            return
        if mode_norm in ("preprocess", "pipeline"):
            raw = (raw_data_path or "").strip()
            if not raw:
                self.status_message.emit("请填写 raw-data-path（采集数据目录）")
                return
            if not _eai_platform_bootstrap.os.path.isdir(raw):
                self.status_message.emit(f"raw-data-path 不是目录: {raw}")
                return
        cfg = (config_name or "").strip()
        if mode_norm == "train" and not cfg:
            cfg = "reward_training"
        if mode_norm == "collect" and not cfg:
            cfg = "realworld_collect_dataset"
        if mode_norm == "pipeline" and not cfg:
            cfg = "reward_training"
        if mode_norm in ("train", "pipeline") and cfg:
            cfg_path = _eai_platform_bootstrap.os.path.join(
                root, "examples", "reward", "config", f"{cfg}.yaml"
            )
            if not _eai_platform_bootstrap.os.path.isfile(cfg_path):
                self.status_message.emit(f"配置不存在: {cfg_path}")
                return
        if mode_norm == "collect" and cfg:
            cfg_path = _eai_platform_bootstrap.os.path.join(
                root, "examples", "reward", "config", f"{cfg}.yaml"
            )
            if not _eai_platform_bootstrap.os.path.isfile(cfg_path):
                self.status_message.emit(f"配置不存在: {cfg_path}")
                return
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.RLINF_REWARD_WORKFLOW_CACHE_DIR, exist_ok=True)

        args = [
            "--mode",
            mode_norm,
            "--dataset-type",
            ds,
            "--rlinf-root",
            root,
            "--python",
            py,
            "--num-samples",
            str(int(num_samples)),
            "--val-split",
            str(val_split),
            "--fail-success-ratio",
            str(fail_success_ratio),
            "--seed",
            str(int(seed)),
        ]
        if cfg:
            args.extend(["--config", cfg])
        raw = (raw_data_path or "").strip()
        if raw:
            args.extend(["--raw-data-path", raw])
        out = (output_dir or "").strip()
        if out:
            args.extend(["--output-dir", out])
        if not keep_last_frame:
            args.append("--no-keep-last-frame")
        overrides = [str(x).strip() for x in hydra_overrides if str(x).strip()]
        if overrides:
            args.append("--")
            args.extend(overrides)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("RLINF_ROOT", root)
        qenv.insert("RLINF_PYTHON", py)
        cuda = (cuda_devices or "").strip()
        if cuda:
            qenv.insert("CUDA_VISIBLE_DEVICES", cuda)

        self._log_dir = ""
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
        self.log_line.emit(
            f"$ bash scripts/launch/run_reward_workflow.sh --mode {mode_norm} "
            f"--dataset-type {ds}"
            + (f" --config {cfg}" if cfg else "")
        )
        self.status_message.emit(
            f"正在启动 Reward 工作流（{mode_norm}）…"
        )

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 Reward 工作流")
            return
        self.status_message.emit("正在停止 Reward 工作流…")
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
            if text.startswith("[reward-workflow] log_dir="):
                self._log_dir = text.split("=", 1)[-1].strip()
                if self._log_dir:
                    self.log_dir_ready.emit(self._log_dir)
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- Reward 工作流正常退出 ---")
            self.status_message.emit("Reward 工作流已完成")
        else:
            self.log_line.emit(f"--- Reward 工作流退出 (code={exit_code}) ---")
            self.status_message.emit(f"Reward 工作流异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 Reward 工作流进程")
            self.status_message.emit("无法启动 Reward 工作流")
            self.running_changed.emit(False)


class RoboDojoOnlineRLLauncher(_eai_platform_bootstrap.QObject):
    """在 RoboDojo/RISE 中启动基于 RLinf 的在线强化学习（policy_online）。"""

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
        config_name: str,
        rise_root: str,
        python_bin: str = "",
        cuda_devices: str = "",
        hydra_overrides: _eai_platform_bootstrap.Sequence[str] = (),
        auto_single_gpu: bool = True,
    ) -> None:
        if self.is_running():
            self.status_message.emit("仿真强化学习训练 正在运行")
            return
        script = _eai_configuration_tabs.ROBODOJO_ONLINE_RL_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_rise_root(rise_root)
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_rise_python(root)
        cfg = (config_name or "").strip()
        if not cfg:
            self.status_message.emit("请选择 Hydra config")
            return
        cfg_path = _eai_platform_bootstrap.os.path.join(
            root,
            "policy_and_value",
            "policy_online",
            "examples",
            "embodiment",
            "config",
            f"{cfg}.yaml",
        )
        if not _eai_platform_bootstrap.os.path.isfile(cfg_path):
            self.status_message.emit(f"配置不存在: {cfg_path}")
            return
        mode_norm = (mode or "single").strip().lower()
        if mode_norm in ("sync", "embodiment"):
            mode_norm = "single"
        if mode_norm in ("ray", "multinode"):
            mode_norm = "multi"
        if mode_norm not in ("single", "multi"):
            self.status_message.emit(f"未知 mode: {mode}")
            return
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.ROBODOJO_ONLINE_RL_CACHE_DIR, exist_ok=True)

        args = [
            "--mode",
            mode_norm,
            "--config",
            cfg,
            "--rise-root",
            root,
            "--python",
            py,
        ]
        if not auto_single_gpu:
            args.append("--no-auto-single-gpu")
        overrides = [str(x).strip() for x in hydra_overrides if str(x).strip()]
        if overrides:
            args.append("--")
            args.extend(overrides)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("RISE_ROOT", root)
        qenv.insert("RISE_PYTHON", py)
        qenv.insert("MUJOCO_GL", qenv.value("MUJOCO_GL") or "egl")
        qenv.insert("PYOPENGL_PLATFORM", qenv.value("PYOPENGL_PLATFORM") or "egl")
        cuda = (cuda_devices or "").strip()
        if cuda:
            qenv.insert("CUDA_VISIBLE_DEVICES", cuda)

        self._log_dir = ""
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
        self.log_line.emit(
            f"$ bash scripts/launch/run_robodojo_online_rl.sh --mode {mode_norm} --config {cfg}"
        )
        self.status_message.emit(
            f"正在启动 仿真强化学习训练（{mode_norm} / {cfg}）…"
        )

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 仿真强化学习训练")
            return
        self.status_message.emit("正在停止 仿真强化学习训练…")
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
            if text.startswith("[robodojo-online-rl] log_dir="):
                self._log_dir = text.split("=", 1)[-1].strip()
                if self._log_dir:
                    self.log_dir_ready.emit(self._log_dir)
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- 仿真强化学习训练 正常退出 ---")
            self.status_message.emit("仿真强化学习训练 已完成")
        else:
            self.log_line.emit(f"--- 仿真强化学习训练 退出 (code={exit_code}) ---")
            self.status_message.emit(f"仿真强化学习训练 异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 仿真强化学习训练 进程")
            self.status_message.emit("无法启动 仿真强化学习训练")
            self.running_changed.emit(False)


class RoboMeterPolicyLauncher(_eai_platform_bootstrap.QObject):
    """启动 robometer-policy-learning 训练 / 评测。"""

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
        job: str,
        config_name: str,
        repo_root: str,
        python_bin: str = "",
        cuda_devices: str = "",
        render_gui: bool = False,
        hydra_overrides: _eai_platform_bootstrap.Sequence[str] = (),
    ) -> None:
        if self.is_running():
            self.status_message.emit("RoboMeter 正在运行")
            return
        script = _eai_configuration_tabs.ROBOMETER_RUN_SCRIPT
        if not _eai_platform_bootstrap.os.path.isfile(script):
            self.status_message.emit(f"未找到脚本: {script}")
            return
        root = _eai_adapters_models_discovery.resolve_robometer_root(repo_root)
        if not _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.join(root, "robometer_policy_learning")):
            self.status_message.emit(f"无效仓库: {root}")
            return
        py = (python_bin or "").strip() or _eai_adapters_models_discovery.resolve_robometer_python(root)
        cfg = (config_name or "").strip()
        if not cfg:
            self.status_message.emit("请选择 Hydra config")
            return
        cfg_path = _eai_platform_bootstrap.os.path.join(
            root, "robometer_policy_learning", "configs", f"{cfg}.yaml"
        )
        if not _eai_platform_bootstrap.os.path.isfile(cfg_path):
            self.status_message.emit(f"配置不存在: {cfg_path}")
            return
        job_norm = (job or "train").strip().lower().replace("-", "_")
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.ROBOMETER_CACHE_DIR, exist_ok=True)

        args = [
            "--job",
            job_norm,
            "--config",
            cfg,
            "--root",
            root,
            "--python",
            py,
        ]
        if render_gui:
            args.append("--gui")
        overrides = [str(x).strip() for x in hydra_overrides if str(x).strip()]
        if overrides:
            args.append("--")
            args.extend(overrides)

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.remove("QT_XCB_GL_INTEGRATION")
        qenv.remove("QT_OPENGL")
        qenv.remove("QT_QUICK_BACKEND")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("ROBOMETER_ROOT", root)
        qenv.insert("ROBOMETER_PYTHON", py)
        qenv.insert("PYTHONPATH", root)
        qenv.insert("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        # train / train_async use OffScreenRenderEnv; glfw under Qt parent often
        # triggers cudnn "Invalid handle. Cannot load symbol cudnnGetVersion".
        use_glfw = bool(render_gui) and job_norm not in ("train", "train_async", "online_rl")
        if use_glfw:
            qenv.insert("MUJOCO_GL", "glfw")
            qenv.remove("PYOPENGL_PLATFORM")
        else:
            qenv.insert("MUJOCO_GL", "egl")
            qenv.insert("PYOPENGL_PLATFORM", "egl")
        cuda = (cuda_devices or "").strip()
        if cuda:
            qenv.insert("CUDA_VISIBLE_DEVICES", cuda)

        self._log_dir = ""
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
        gui_flag = " --gui" if render_gui else ""
        self.log_line.emit(
            f"$ bash scripts/launch/run_robometer_policy.sh --job {job_norm} --config {cfg}{gui_flag}"
        )
        self.status_message.emit(f"正在启动 RoboMeter（{job_norm} / {cfg}）…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 RoboMeter")
            return
        self.status_message.emit("正在停止 RoboMeter…")
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
            if text.startswith("[robometer] log_dir="):
                self._log_dir = text.split("=", 1)[-1].strip()
                if self._log_dir:
                    self.log_dir_ready.emit(self._log_dir)
            self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- RoboMeter 正常退出 ---")
            self.status_message.emit("RoboMeter 已完成")
        else:
            self.log_line.emit(f"--- RoboMeter 退出 (code={exit_code}) ---")
            self.status_message.emit(f"RoboMeter 异常退出 (code={exit_code})")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.log_line.emit("[ERROR] 无法启动 RoboMeter 进程")
            self.status_message.emit("无法启动 RoboMeter")
            self.running_changed.emit(False)

