"""adapters.models.bagel launchers — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_bagel_runtime = module_ref('eai_app.adapters.models.bagel_runtime')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class HumanEgoLauncher(_eai_platform_bootstrap.QObject):
    """在 HumanEgo 仓库目录中跑 setup / download / preprocess / train / 自定义命令。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._label = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() in (
            _eai_platform_bootstrap.QProcess.Starting,
            _eai_platform_bootstrap.QProcess.Running,
        )

    def start_command(
        self,
        *,
        repo_dir: str,
        python_bin: str,
        command: str,
        label: str = "",
    ) -> None:
        if self.is_running():
            self.status_message.emit("HumanEgo 任务仍在运行，请先停止")
            return
        repo = _eai_configuration_tabs.resolve_humanego_root(repo_dir)
        if not _eai_platform_bootstrap.os.path.isdir(repo):
            self.status_message.emit(f"仓库不存在: {repo}")
            return
        cmd = (command or "").strip()
        if not cmd:
            self.status_message.emit("命令为空")
            return
        py = (python_bin or "").strip() or _eai_configuration_tabs.resolve_humanego_python(repo)
        label = (label or "").strip() or cmd[:80]
        self._label = label

        # 用 bash -lc，保证 conda activate / PATH 与终端一致；PYTHON 注入供命令里用 $PYTHON
        env_exports = [
            f"export HUMANOGO_ROOT={_eai_platform_bootstrap.shlex.quote(repo)}",
            f"export PYTHON={_eai_platform_bootstrap.shlex.quote(py)}",
            "export PYTHONUNBUFFERED=1",
        ]
        # 若 python 在 conda env 下，尝试 source conda.sh + activate
        conda_prefix = ""
        py_dir = _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.abspath(py))
        if py_dir.endswith("/bin"):
            maybe_prefix = _eai_platform_bootstrap.os.path.dirname(py_dir)
            if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(maybe_prefix, "conda-meta", "history")) or _eai_platform_bootstrap.os.path.isdir(
                _eai_platform_bootstrap.os.path.join(maybe_prefix, "conda-meta")
            ):
                conda_prefix = maybe_prefix
        activate_bits: _eai_platform_bootstrap.List[str] = []
        if conda_prefix:
            conda_sh_candidates = [
                _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.dirname(conda_prefix)), "etc", "profile.d", "conda.sh"),
                "/home/psibot/miniconda3/etc/profile.d/conda.sh",
                "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/etc/profile.d/conda.sh",
            ]
            for sh_path in conda_sh_candidates:
                if _eai_platform_bootstrap.os.path.isfile(sh_path):
                    env_name = _eai_platform_bootstrap.os.path.basename(conda_prefix)
                    activate_bits = [
                        f"source {_eai_platform_bootstrap.shlex.quote(sh_path)}",
                        f"conda activate {_eai_platform_bootstrap.shlex.quote(env_name)} || true",
                    ]
                    break
            env_exports.append(f"export PATH={_eai_platform_bootstrap.shlex.quote(py_dir)}:$PATH")
        shell = " && ".join(
            [
                f"cd {_eai_platform_bootstrap.shlex.quote(repo)}",
                *activate_bits,
                *env_exports,
                cmd,
            ]
        )

        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.insert("PYTHONUNBUFFERED", "1")
        qenv.insert("HUMANOGO_ROOT", repo)

        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(repo)
        proc.setProcessEnvironment(qenv)
        proc.start("bash", ["-lc", shell])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ {cmd}")
        self.status_message.emit(f"HumanEgo 运行中: {label}")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 HumanEgo 任务")
            return
        self.status_message.emit("正在停止 HumanEgo…")
        if self._process is not None:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill)

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
            if text:
                self.log_line.emit(text)

    def _on_process_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self.running_changed.emit(False)
        label = self._label or "任务"
        if exit_code == 0:
            self.status_message.emit(f"HumanEgo 完成: {label}")
            self.log_line.emit(f"[done] exit=0  {label}")
        else:
            self.status_message.emit(f"HumanEgo 失败 (exit={exit_code}): {label}")
            self.log_line.emit(f"[ERROR] exit={exit_code}  {label}")
        self._process = None

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.status_message.emit("HumanEgo 启动失败（找不到 bash？）")
            self.log_line.emit("[ERROR] FailedToStart")
            self.running_changed.emit(False)
            self._process = None


class BagelAppLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 Bagel Gradio app.py，日志转发到 UI。"""

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    ready_url = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._ready_emitted = False
        self._url = ""
        self._mode = 1
        self._adopted = False
        self._adopt_host = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        self._adopt_port = _eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT
        self._adopt_watch_timer = _eai_platform_bootstrap.QTimer(self)
        self._adopt_watch_timer.setInterval(2500)
        self._adopt_watch_timer.timeout.connect(self._on_adopt_watch_tick)

    def is_running(self) -> bool:
        if self._adopted:
            return True
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def current_url(self) -> str:
        return self._url

    def current_mode(self) -> int:
        return int(self._mode or 1)

    def try_adopt(
        self,
        *,
        server_name: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
        server_port: int = _eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT,
        mode: int = 1,
    ) -> bool:
        """复用已在监听的 Bagel Gradio，避免重复加载权重。"""
        host = (server_name or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(server_port)
        if not _eai_adapters_models_bagel_runtime.bagel_probe_gradio_ready(host, port, timeout_s=2.0):
            return False
        self._adopted = True
        self._adopt_host = host
        self._adopt_port = port
        self._mode = int(mode or 1)
        found = _eai_platform_bootstrap.re.search(r"--mode(?:\s+|=)(\d+)", _eai_adapters_models_qwen_local.listener_cmdline(port))
        if found:
            try:
                self._mode = int(found.group(1))
            except (TypeError, ValueError):
                pass
        self._url = _eai_adapters_models_bagel_runtime.bagel_app_url(host, port)
        self._ready_emitted = True
        self.log_line.emit(f"[adopt] 复用已部署 Gradio {self._url.rstrip('/')}")
        self.status_message.emit(
            f"Bagel Gradio 已部署，跳过启动: {self._url}  ·  "
            f"mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._mode)}"
        )
        self.running_changed.emit(True)
        self.ready_url.emit(self._url)
        if not self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.start()
        return True

    def _clear_adopted(self, *, emit_stopped: bool = True) -> None:
        was = self._adopted
        self._adopted = False
        self._ready_emitted = False
        if self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.stop()
        if was and emit_stopped:
            self.running_changed.emit(False)

    def _on_adopt_watch_tick(self) -> None:
        if not self._adopted:
            self._adopt_watch_timer.stop()
            return
        if _eai_adapters_models_bagel_runtime.bagel_probe_gradio_ready(self._adopt_host, self._adopt_port, timeout_s=1.5):
            return
        self.log_line.emit("--- 复用的 Bagel Gradio 已不可用 ---")
        self.status_message.emit("复用的 Bagel Gradio 已退出")
        self._clear_adopted(emit_stopped=True)

    def start(
        self,
        *,
        repo_dir: str = "",
        model_path: str = "",
        python_bin: str = "",
        server_name: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
        server_port: int = _eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT,
        mode: int = 1,
        zh: bool = False,
        share: bool = False,
        force_new: bool = False,
    ) -> None:
        host = (server_name or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(server_port)
        repo_for_match = _eai_adapters_models_bagel_runtime.resolve_bagel_root(repo_dir)
        model_for_match = (model_path or "").strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
        model_abs_for_match = _eai_adapters_models_bagel_runtime.resolve_bagel_model_path(repo_for_match, model_for_match)
        if not force_new and (
            self.is_running() or _eai_adapters_models_bagel_runtime.bagel_probe_gradio_ready(host, port)
        ):
            same = _eai_adapters_models_bagel_runtime.bagel_listener_matches(
                port, mode=int(mode or 1), model_path=model_abs_for_match
            )
            if same is False:
                self.status_message.emit(
                    "Bagel Gradio 已在运行其它模型或模式。请先停止再部署。"
                )
                return
            if self.try_adopt(server_name=host, server_port=port, mode=int(mode or 1)):
                self.status_message.emit(
                    f"同一模型已部署，跳过重复部署: {self._url}  ·  "
                    f"mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self.current_mode())}"
                )
                return
            if self.is_running():
                self.status_message.emit(
                    f"同一模型正在启动，跳过重复部署 "
                    f"(mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self.current_mode())})"
                )
                return
        self._clear_adopted(emit_stopped=False)
        repo = _eai_adapters_models_bagel_runtime.resolve_bagel_root(repo_dir)
        app_py = _eai_platform_bootstrap.os.path.join(repo, "app.py")
        if not _eai_platform_bootstrap.os.path.isfile(app_py):
            self.status_message.emit(f"未找到 app.py: {app_py}")
            return
        py = (python_bin or "").strip() or _eai_adapters_models_bagel_runtime.resolve_bagel_python(repo)
        if _eai_adapters_models_bagel_runtime._bagel_python_looks_wrong(py) and _eai_platform_bootstrap.os.path.isfile(_eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT):
            self.log_line.emit(
                f"[warn] 跳过 {py}（无兼容 gradio），改用 {_eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT}"
            )
            py = _eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT
        if not _eai_platform_bootstrap.os.path.isfile(py):
            self.status_message.emit(f"Python 不存在: {py}")
            return
        model = (model_path or "").strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
        model_abs = _eai_adapters_models_bagel_runtime.resolve_bagel_model_path(repo, model)
        if not _eai_platform_bootstrap.os.path.isdir(model_abs):
            self.log_line.emit(
                f"[warn] 模型目录不存在: {model_abs}\n"
                "请点「下载模型」，或按 README 下载 ByteDance-Seed/BAGEL-7B-MoT。\n"
                f"也可用模型路径: {_eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT}"
            )
        # Preflight: avoid cryptic ModuleNotFoundError in subprocess
        pre = (
            f"{_eai_platform_bootstrap.shlex.quote(py)} -c "
            f"{_eai_platform_bootstrap.shlex.quote('import gradio,torch,transformers,flash_attn')}"
        )
        # Use absolute model path so cwd 无关
        model_arg = model_abs if _eai_platform_bootstrap.os.path.isdir(model_abs) else model
        host = (server_name or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(server_port)
        self._mode = int(mode or 1)
        self._url = _eai_adapters_models_bagel_runtime.bagel_app_url(host, port)
        self._ready_emitted = False

        args = [
            "app.py",
            "--server_name",
            host,
            "--server_port",
            str(port),
            "--model_path",
            model_arg,
            "--mode",
            str(self._mode),
        ]
        if zh:
            args.append("--zh")
        if share:
            args.append("--share")

        quoted = " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in args)
        conda_activate = _eai_platform_bootstrap.os.path.join(
            _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.dirname(py)), "bin", "activate"
        )
        prefix = (
            f"source {_eai_platform_bootstrap.shlex.quote(conda_activate)} && "
            if _eai_platform_bootstrap.os.path.isfile(conda_activate)
            else ""
        )
        cmd = (
            "unset PYTHONPATH PYTHONHOME || true && "
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true && "
            "export NO_PROXY=127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY} "
            "no_proxy=127.0.0.1,localhost,::1${no_proxy:+,$no_proxy} "
            "PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False && "
            f"{prefix}cd {_eai_platform_bootstrap.shlex.quote(repo)} && "
            f"({pre}) || {{ echo '[error] Python 环境缺依赖 (gradio/torch/transformers/flash_attn)。"
            f"请将 Python 设为 miniconda3/envs/bagel 或先 pip install -r requirements.txt'; exit 2; }} && "
            f"exec {_eai_platform_bootstrap.shlex.quote(py)} {quoted}"
        )

        shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(cmd, "bagel_gradio")
        self._persist_pid_path = pid_path
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(repo)
        proc.setProcessEnvironment(_eai_adapters_models_bagel_runtime.bagel_isolated_process_env(py))
        proc.start("setsid", ["bash", "-lc", shell])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ cd {repo}")
        self.log_line.emit(f"$ {py} {' '.join(args)}")
        self.status_message.emit(
            f"正在启动 Bagel Gradio ({self._url}) · mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._mode)}…"
        )

    def stop(self) -> None:
        self.status_message.emit("正在停止 Bagel Gradio，并清理相关进程…")
        _eai_adapters_models_qwen_local.stop_persistent_model(
            getattr(self, "_persist_pid_path", "")
            or _eai_adapters_models_qwen_local.persistent_slot_paths("bagel_gradio")[1]
        )
        self._clear_adopted(emit_stopped=False)
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(4000, self._force_kill_process)
        hits = _eai_adapters_models_bagel_runtime.bagel_kill_related_processes(sig="-TERM")
        self.log_line.emit(f"[stop] pkill TERM 匹配组≈{hits}")
        _eai_platform_bootstrap.QTimer.singleShot(2000, lambda: _eai_adapters_models_bagel_runtime.bagel_kill_related_processes(sig="-KILL"))
        if self._process is None or self._process.state() != _eai_platform_bootstrap.QProcess.Running:
            self._process = None
            self._ready_emitted = False
            self.running_changed.emit(False)
            self.status_message.emit("已请求停止 Bagel 相关进程")

    def shutdown(self) -> None:
        # 只结束日志 tail，不 pkill。Gradio 在独立 session，界面退出后继续服务。
        self._clear_adopted(emit_stopped=False)
        proc = self._process
        self._process = None
        if proc is not None and proc.state() == _eai_platform_bootstrap.QProcess.Running:
            proc.terminate()
            proc.waitForFinished(400)

    def _maybe_emit_ready(self, line: str) -> None:
        if self._ready_emitted:
            return
        low = line.lower()
        if "running on" in low or "http://" in low or "local url" in low:
            self._ready_emitted = True
            self.ready_url.emit(self._url)
            self.status_message.emit(
                f"Bagel 已就绪: {self._url}  ·  mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self.current_mode())}"
            )

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                text = line.rstrip()
                if _eai_adapters_models_bagel_runtime._is_noisy_bagel_log_line(text):
                    continue
                self.log_line.emit(text)
                self._maybe_emit_ready(text)

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- Bagel 进程正常退出 ---")
            self.status_message.emit("Bagel 已退出")
        else:
            self.log_line.emit(f"--- Bagel 进程退出 (code={exit_code}) ---")
            self.status_message.emit(f"Bagel 异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"Bagel 进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- Bagel 进程已被强制终止 ---")
            self.status_message.emit("Bagel 已强制停止")


class BagelApiLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止 Bagel serve_api.py（无 Gradio，仅模型 HTTP）。

    若目标端口上已有就绪的 serve_api，部署时直接复用（adopt），不重复加载权重。
    停止时会清理本进程并 pkill 所有相关 Bagel 进程（API + Gradio）。
    """

    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    ready_url = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._ready_emitted = False
        self._url = ""
        self._api_base = ""
        self._mode = 1
        self._adopted = False
        self._adopt_host = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        self._adopt_port = _eai_configuration_models.BAGEL_API_PORT_DEFAULT
        self._adopt_watch_timer = _eai_platform_bootstrap.QTimer(self)
        self._adopt_watch_timer.setInterval(2500)
        self._adopt_watch_timer.timeout.connect(self._on_adopt_watch_tick)

    def is_running(self) -> bool:
        if self._adopted:
            return True
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def is_adopted(self) -> bool:
        return bool(self._adopted)

    def current_url(self) -> str:
        return self._url

    def api_base(self) -> str:
        return self._api_base

    def current_mode(self) -> int:
        return int(self._mode or 1)

    def try_adopt(
        self,
        *,
        server_name: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
        server_port: int = _eai_configuration_models.BAGEL_API_PORT_DEFAULT,
    ) -> bool:
        """复用已在运行且 /health 就绪的推理 API。"""
        host = (server_name or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(server_port)
        body = _eai_adapters_models_bagel_runtime.bagel_probe_api_health(host, port, timeout_s=2.0)
        if body is None:
            return False
        self._adopted = True
        self._adopt_host = host
        self._adopt_port = port
        self._mode = int(body.get("mode") or 1)
        self._url = _eai_adapters_models_bagel_runtime.bagel_app_url(host, port)
        self._api_base = self._url.rstrip("/") + "/v1"
        self._ready_emitted = True
        model = str(body.get("model") or body.get("model_path") or "")
        self.log_line.emit(
            f"[adopt] 复用已部署推理 API {self._url.rstrip('/')}  "
            f"mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._mode)}"
            + (f"  model={model}" if model else "")
        )
        self.status_message.emit(
            f"已复用推理 API: {self._url}  ·  mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._mode)}"
        )
        self.running_changed.emit(True)
        self.ready_url.emit(self._url)
        if not self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.start()
        return True

    def _clear_adopted(self, *, emit_stopped: bool = True) -> None:
        was = self._adopted
        self._adopted = False
        self._ready_emitted = False
        if self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.stop()
        if was and emit_stopped:
            self.running_changed.emit(False)

    def _on_adopt_watch_tick(self) -> None:
        if not self._adopted:
            self._adopt_watch_timer.stop()
            return
        body = _eai_adapters_models_bagel_runtime.bagel_probe_api_health(
            self._adopt_host, self._adopt_port, timeout_s=1.5
        )
        if body is None:
            self.log_line.emit("--- 复用的推理 API 已不可用 ---")
            self.status_message.emit("复用的推理 API 已退出")
            self._clear_adopted(emit_stopped=True)

    def start(
        self,
        *,
        repo_dir: str = "",
        model_path: str = "",
        python_bin: str = "",
        server_name: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
        server_port: int = _eai_configuration_models.BAGEL_API_PORT_DEFAULT,
        mode: int = 1,
        force_new: bool = False,
    ) -> None:
        host = (server_name or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(server_port)
        repo_for_match = _eai_adapters_models_bagel_runtime.resolve_bagel_root(repo_dir)
        model_for_match = (model_path or "").strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
        model_abs_for_match = _eai_adapters_models_bagel_runtime.resolve_bagel_model_path(repo_for_match, model_for_match)
        if not force_new:
            body = _eai_adapters_models_bagel_runtime.bagel_probe_api_health(host, port, timeout_s=2.0)
            if body is not None:
                same = _eai_adapters_models_qwen_local.model_identity_same(
                    str(body.get("mode") or ""),
                    str(body.get("model_path") or ""),
                    str(int(mode or 1)),
                    model_abs_for_match,
                )
                if same is False:
                    live = (
                        body.get("model_path")
                        or body.get("model")
                        or body.get("mode")
                        or "未知模型"
                    )
                    self.status_message.emit(
                        f"推理 API 已在运行其它模型（{live}）。请先停止再部署。"
                    )
                    return
                if self.try_adopt(server_name=host, server_port=port):
                    self.status_message.emit(
                        f"同一模型已部署，跳过重复部署: {self._url}  ·  "
                        f"mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self.current_mode())}"
                    )
                    return
            elif self.is_running() or _eai_adapters_models_qwen_local.persistent_slot_blob("bagel_api"):
                same_loading = _eai_adapters_models_bagel_runtime.bagel_listener_matches(
                    port, mode=int(mode or 1), model_path=model_abs_for_match
                )
                if same_loading is False:
                    self.status_message.emit(
                        "推理 API 正在加载其它模型。请先停止再部署。"
                    )
                    return
                self.status_message.emit(
                    f"同一模型正在启动，跳过重复部署 "
                    f"(mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(int(mode or 1))})"
                )
                self.running_changed.emit(True)
                return

        repo = _eai_adapters_models_bagel_runtime.resolve_bagel_root(repo_dir)
        serve_py = _eai_platform_bootstrap.os.path.join(repo, "serve_api.py")
        if not _eai_platform_bootstrap.os.path.isfile(serve_py):
            self.status_message.emit(f"未找到 serve_api.py: {serve_py}")
            return
        py = (python_bin or "").strip() or _eai_adapters_models_bagel_runtime.resolve_bagel_python(repo)
        if _eai_adapters_models_bagel_runtime._bagel_python_looks_wrong(py) and _eai_platform_bootstrap.os.path.isfile(_eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT):
            self.log_line.emit(
                f"[warn] 跳过 {py}（无兼容环境），改用 {_eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT}"
            )
            py = _eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT
        if not _eai_platform_bootstrap.os.path.isfile(py):
            self.status_message.emit(f"Python 不存在: {py}")
            return
        model = (model_path or "").strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
        model_abs = _eai_adapters_models_bagel_runtime.resolve_bagel_model_path(repo, model)
        if not _eai_platform_bootstrap.os.path.isdir(model_abs):
            self.status_message.emit(f"模型目录不存在: {model_abs}")
            self.log_line.emit(
                f"[error] 模型目录不存在: {model_abs}\n"
                "请先点「下载模型」。"
            )
            return
        self._clear_adopted(emit_stopped=False)
        self._mode = int(mode or 1)
        self._adopt_host = host
        self._adopt_port = port
        self._url = _eai_adapters_models_bagel_runtime.bagel_app_url(host, port)
        self._api_base = self._url.rstrip("/") + "/v1"
        self._ready_emitted = False

        args = [
            "serve_api.py",
            "--host",
            host,
            "--port",
            str(port),
            "--model_path",
            model_abs,
            "--mode",
            str(self._mode),
        ]
        quoted = " ".join(_eai_platform_bootstrap.shlex.quote(a) for a in args)
        conda_activate = _eai_platform_bootstrap.os.path.join(
            _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.dirname(py)), "bin", "activate"
        )
        prefix = (
            f"source {_eai_platform_bootstrap.shlex.quote(conda_activate)} && "
            if _eai_platform_bootstrap.os.path.isfile(conda_activate)
            else ""
        )
        pre = (
            f"{_eai_platform_bootstrap.shlex.quote(py)} -c "
            f"{_eai_platform_bootstrap.shlex.quote('import fastapi,uvicorn,torch,transformers,flash_attn')}"
        )
        cmd = (
            "unset PYTHONPATH PYTHONHOME || true && "
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true && "
            "export NO_PROXY=127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY} "
            "no_proxy=127.0.0.1,localhost,::1${no_proxy:+,$no_proxy} "
            "PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 && "
            f"{prefix}cd {_eai_platform_bootstrap.shlex.quote(repo)} && "
            f"({pre}) || {{ echo '[error] Python 环境缺依赖 (fastapi/uvicorn/torch)。"
            f"请使用 miniconda3/envs/bagel'; exit 2; }} && "
            f"exec {_eai_platform_bootstrap.shlex.quote(py)} {quoted}"
        )

        shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(cmd, "bagel_api")
        self._persist_pid_path = pid_path
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(repo)
        proc.setProcessEnvironment(_eai_adapters_models_bagel_runtime.bagel_isolated_process_env(py))
        proc.start("setsid", ["bash", "-lc", shell])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ cd {repo}")
        self.log_line.emit(f"$ {py} {' '.join(args)}")
        self.status_message.emit(
            f"正在部署 Bagel 推理 API ({self._url}) · mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._mode)}…"
        )

    def stop(self) -> None:
        """停止本窗口管理的进程，并清理所有相关 Bagel 模型进程。"""
        self.status_message.emit("正在停止全部 Bagel 相关进程…")
        _eai_adapters_models_qwen_local.stop_persistent_model(
            getattr(self, "_persist_pid_path", "")
            or _eai_adapters_models_qwen_local.persistent_slot_paths("bagel_api")[1]
        )
        if self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.stop()
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.terminate()
            _eai_platform_bootstrap.QTimer.singleShot(3500, self._force_kill_process)
        hits = _eai_adapters_models_bagel_runtime.bagel_kill_related_processes(sig="-TERM")
        self.log_line.emit(f"[stop] pkill TERM 匹配组≈{hits}")
        _eai_platform_bootstrap.QTimer.singleShot(2000, self._force_kill_all_related)
        self._clear_adopted(emit_stopped=False)
        if self._process is None or self._process.state() != _eai_platform_bootstrap.QProcess.Running:
            self._process = None
            self._ready_emitted = False
            self.running_changed.emit(False)
            self.status_message.emit("已请求停止全部 Bagel 相关进程")

    def shutdown(self) -> None:
        # 只结束日志 tail，不 pkill。推理 API 在独立 session，界面退出后继续服务。
        if self._adopt_watch_timer.isActive():
            self._adopt_watch_timer.stop()
        proc = self._process
        self._process = None
        if proc is not None and proc.state() == _eai_platform_bootstrap.QProcess.Running:
            proc.terminate()
            proc.waitForFinished(400)

    def _force_kill_all_related(self) -> None:
        hits = _eai_adapters_models_bagel_runtime.bagel_kill_related_processes(sig="-KILL")
        if hits:
            self.log_line.emit(f"[stop] pkill KILL 匹配组≈{hits}")
        self._process = None
        self._adopted = False
        self._ready_emitted = False
        self.running_changed.emit(False)
        self.log_line.emit("--- 全部 Bagel 相关进程已清理 ---")
        self.status_message.emit("Bagel 相关进程已全部停止")

    def _maybe_emit_ready(self, line: str) -> None:
        if self._ready_emitted:
            return
        low = line.lower()
        if "[bagel-api] ready" in low or "uvicorn running on" in low:
            self._ready_emitted = True
            self.ready_url.emit(self._url)
            self.status_message.emit(
                f"Bagel 推理 API 已就绪: {self._url}  ·  mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self.current_mode())}"
            )

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                text = line.rstrip()
                if _eai_adapters_models_bagel_runtime._is_noisy_bagel_log_line(text):
                    continue
                self.log_line.emit(text)
                self._maybe_emit_ready(text)

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self._ready_emitted = False
        if self._adopted:
            # 本窗口进程退出但可能仍在复用外部；由 adopt watch 决定
            return
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- Bagel 推理 API 正常退出 ---")
            self.status_message.emit("Bagel 推理 API 已退出")
        else:
            self.log_line.emit(f"--- Bagel 推理 API 退出 (code={exit_code}) ---")
            self.status_message.emit(f"Bagel 推理 API 异常退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"Bagel 推理 API 进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()
            self._process = None
            if not self._adopted:
                self.running_changed.emit(False)
                self.log_line.emit("--- Bagel 推理 API 已被强制终止 ---")
                self.status_message.emit("Bagel 推理 API 已强制停止")

