"""adapters.models.training — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class CadMeshLauncher(_eai_platform_bootstrap.QObject):
    """启动照片/mesh → FoundationPose .obj 重建，日志转发到 UI。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    mesh_ready = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._last_mesh = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def last_mesh(self) -> str:
        return self._last_mesh

    def start_from_images(
        self,
        images_dir: str,
        name: str,
        *,
        target_extent_m: float = _eai_configuration_policies.CAD_TARGET_EXTENT_DEFAULT_M,
        poisson_depth: int = _eai_configuration_policies.CAD_POISSON_DEPTH_DEFAULT,
        min_images: int = _eai_configuration_policies.CAD_MIN_IMAGES_DEFAULT,
        out_dir: _eai_platform_bootstrap.Optional[str] = None,
    ) -> None:
        images_dir = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(images_dir.strip()))
        if not _eai_platform_bootstrap.os.path.isdir(images_dir):
            self.status_message.emit(f"照片目录不存在: {images_dir}")
            return
        self._start(
            [
                "--images",
                images_dir,
                "--name",
                name,
                "--target-extent-m",
                str(float(target_extent_m)),
                "--poisson-depth",
                str(int(poisson_depth)),
                "--min-images",
                str(int(min_images)),
            ],
            out_dir=out_dir,
            expected_name=name,
        )

    def start_from_mesh(
        self,
        mesh_path: str,
        name: str,
        *,
        target_extent_m: float = _eai_configuration_policies.CAD_TARGET_EXTENT_DEFAULT_M,
        out_dir: _eai_platform_bootstrap.Optional[str] = None,
    ) -> None:
        mesh_path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(mesh_path.strip()))
        if not _eai_platform_bootstrap.os.path.isfile(mesh_path):
            self.status_message.emit(f"mesh 不存在: {mesh_path}")
            return
        self._start(
            [
                "--import-mesh",
                mesh_path,
                "--name",
                name,
                "--target-extent-m",
                str(float(target_extent_m)),
            ],
            out_dir=out_dir,
            expected_name=name,
        )

    def _start(
        self,
        extra_args: _eai_platform_bootstrap.List[str],
        *,
        out_dir: _eai_platform_bootstrap.Optional[str],
        expected_name: str,
    ) -> None:
        if self.is_running():
            self.status_message.emit("CAD 重建已在运行")
            return

        name = (expected_name or "").strip().replace(" ", "_")
        if not name or "/" in name or "\\" in name:
            self.status_message.emit("请填写有效的物体名称")
            return

        out = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(out_dir or _eai_configuration_policies.CAD_MESHES_DIR))
        _eai_platform_bootstrap.os.makedirs(out, exist_ok=True)
        self._expected_obj = _eai_platform_bootstrap.os.path.join(out, name, "reconstructed.obj")
        self._last_mesh = ""

        if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_policies.CAD_PHOTOGRAMMETRY_PY):
            self.status_message.emit(f"未找到重建脚本: {_eai_configuration_policies.CAD_PHOTOGRAMMETRY_PY}")
            return

        mesh_python = _eai_configuration_policies.resolve_cad_mesh_python()
        launch_args = [
            _eai_configuration_policies.CAD_PHOTOGRAMMETRY_PY,
            "--name",
            name,
            "--out-dir",
            out,
        ] + extra_args
        quoted = " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in launch_args)
        cmd = (
            f"export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "
            f"MESH_PYTHON={_eai_platform_bootstrap.shlex.quote(mesh_python)} "
            f"QT_QPA_PLATFORM=offscreen LIBGL_ALWAYS_SOFTWARE=1 && "
            f"cd {_eai_platform_bootstrap.shlex.quote(_eai_configuration_models.EAI_DIR)} && "
            f"exec {_eai_platform_bootstrap.shlex.quote(mesh_python)} {quoted}"
        )

        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(_eai_configuration_models.EAI_DIR)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ {mesh_python} {' '.join(launch_args)}")
        self.status_message.emit(f"正在生成 CAD: {name}…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 CAD 重建")
            return
        self.status_message.emit("正在停止 CAD 重建…")
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
            text = line.rstrip()
            if not text:
                continue
            self.log_line.emit(text)
            if text.startswith("{") and '"mesh"' in text:
                try:
                    payload = _eai_platform_bootstrap.json.loads(text)
                    mesh = str(payload.get("mesh") or "").strip()
                    if mesh and _eai_platform_bootstrap.os.path.isfile(mesh):
                        self._last_mesh = _eai_platform_bootstrap.os.path.abspath(mesh)
                except Exception:
                    pass

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        mesh = self._last_mesh
        if not mesh and getattr(self, "_expected_obj", "") and _eai_platform_bootstrap.os.path.isfile(self._expected_obj):
            mesh = _eai_platform_bootstrap.os.path.abspath(self._expected_obj)
            self._last_mesh = mesh
        if exit_code == 0 and mesh:
            self.log_line.emit(f"--- CAD 完成: {mesh} ---")
            self.status_message.emit(f"CAD 已生成: {_eai_platform_bootstrap.os.path.basename(_eai_platform_bootstrap.os.path.dirname(mesh))}")
            self.mesh_ready.emit(mesh)
        elif exit_code == 0:
            self.log_line.emit("--- CAD 进程正常退出（未找到输出 mesh）---")
            self.status_message.emit("CAD 完成但未找到 reconstructed.obj")
        else:
            self.log_line.emit(f"--- CAD 进程退出 (code={exit_code}) ---")
            self.status_message.emit(f"CAD 重建失败 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"CAD 重建进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- CAD 进程已被强制终止 ---")
            self.status_message.emit("CAD 重建已强制停止")


class PsiPolicyTrainLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 psi-policy 训练，并将 stdout/stderr 实时转发到 UI。"""

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
        config_name: str,
        datahouse_id: str,
        view_id: str,
        *,
        repo_dir: _eai_platform_bootstrap.Optional[str] = None,
        num_epochs: int = 0,
        batch_size: int = 0,
        logging_mode: str = "offline",
        num_processes: int = 1,
    ) -> None:
        if self.is_running():
            self.status_message.emit("psi-policy 训练已在运行")
            return

        repo = repo_dir or _eai_perception_segmentation.resolve_psi_policy_dir()
        if repo is None:
            self.status_message.emit(
                "未找到 psi-policy 目录，请点击「选择路径」指定仓库"
            )
            return

        config = (config_name or _eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT).strip()
        if not config:
            self.status_message.emit("请选择训练配置 (config-name)")
            return

        overrides: _eai_platform_bootstrap.List[str] = [f"logging.mode={logging_mode}"]
        dh_override = _eai_perception_segmentation.format_hydra_override(
            "task.dataset.sources.0.datahouse_id", datahouse_id
        )
        if dh_override:
            overrides.append(dh_override)
        view_override = _eai_perception_segmentation.format_hydra_override(
            "task.dataset.sources.0.view_id", view_id
        )
        if view_override:
            overrides.append(view_override)
        if num_epochs > 0:
            overrides.append(f"training.num_epochs={int(num_epochs)}")
        if batch_size > 0:
            overrides.append(f"train_dataloader.batch_size={int(batch_size)}")

        python_bin = _eai_perception_segmentation.resolve_psi_policy_python(repo)
        train_args = ["psi_policy/train.py", "--config-name", config] + overrides
        if num_processes > 1:
            launch_args = [
                "-m",
                "accelerate.commands.launch",
                f"--num_processes={int(num_processes)}",
            ] + train_args
        else:
            launch_args = train_args

        quoted_args = " ".join(_eai_platform_bootstrap.shlex.quote(arg) for arg in launch_args)
        conda_activate = _eai_platform_bootstrap.os.path.join(
            _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.dirname(python_bin)), "bin", "activate"
        )
        prefix = ""
        if _eai_platform_bootstrap.os.path.isfile(conda_activate):
            prefix = f"source {_eai_platform_bootstrap.shlex.quote(conda_activate)} && "
        cmd = (
            f"{prefix}cd {_eai_platform_bootstrap.shlex.quote(repo)} && "
            f"export PYTHONUNBUFFERED=1 && "
            f"exec {_eai_platform_bootstrap.shlex.quote(python_bin)} {quoted_args}"
        )

        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(repo)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ cd {repo}")
        self.log_line.emit(f"$ {python_bin} {' '.join(launch_args)}")
        self.status_message.emit(f"正在启动 psi-policy 训练 ({config})…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的训练")
            return
        self.status_message.emit("正在停止训练…")
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
            self.log_line.emit("--- 训练进程正常退出 ---")
            self.status_message.emit("psi-policy 训练已完成")
        else:
            self.log_line.emit(f"--- 训练进程退出 (code={exit_code}) ---")
            self.status_message.emit(f"psi-policy 训练异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"psi-policy 训练进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- 训练进程已被强制终止 ---")
            self.status_message.emit("训练已强制停止")

