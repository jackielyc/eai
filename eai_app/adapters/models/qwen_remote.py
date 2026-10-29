"""adapters.models.qwen remote — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def load_remote_qwen_ctl():
    """动态加载 compat/remote_qwen_ctl.py（避免 ROS/Qt 环境下的包导入问题）。"""
    import importlib.util

    path = _eai_configuration_policies.REMOTE_QWEN_CTL_SCRIPT
    if not _eai_platform_bootstrap.os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {path}")
    spec = importlib.util.spec_from_file_location("remote_qwen_ctl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_remote_qwen_hostctl_url() -> str:
    return _eai_configuration_policies.REMOTE_QWEN_HOSTCTL_URL.rstrip("/")


def check_remote_qwen_hostctl_health(timeout_s: float = 1.0) -> bool:
    url = resolve_remote_qwen_hostctl_url() + "/health"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def remote_qwen_hostctl_request(
    path: str,
    method: str = "GET",
    timeout_s: float = 180.0,
    payload: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None,
) -> _eai_platform_bootstrap.Tuple[bool, _eai_platform_bootstrap.Dict[str, object]]:
    url = resolve_remote_qwen_hostctl_url().rstrip("/") + path
    data = None
    if method == "POST":
        data = _eai_platform_bootstrap.json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = _eai_platform_bootstrap.urllib.request.Request(url, data=data, method=method)
    if method == "POST":
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok", True)), body if isinstance(body, dict) else {}
    except Exception as exc:
        return False, {"ok": False, "message": str(exc)}


def remote_qwen_host_label(host_id: str) -> str:
    for hid, label in _eai_configuration_policies.REMOTE_QWEN_HOSTS:
        if hid == host_id:
            return label
    return host_id


def remote_qwen_api_base_for_host(host_id: str) -> str:
    try:
        ctl = load_remote_qwen_ctl()
        prof = ctl.apply_host(host_id)
        return str(prof.get("api_base") or _eai_configuration_policies.REMOTE_QWEN_API_BASE_DEFAULT)
    except Exception:
        port = _eai_configuration_policies.REMOTE_QWEN_HOST_LOCAL_PORTS.get(host_id, _eai_configuration_policies.REMOTE_QWEN_LOCAL_PORT)
        return f"http://127.0.0.1:{port}/v1"


def remote_qwen_profile(host_id: str) -> _eai_platform_bootstrap.Dict[str, object]:
    try:
        ctl = load_remote_qwen_ctl()
        return dict(ctl.apply_host(host_id))
    except Exception:
        return {
            "id": host_id,
            "ssh_host": host_id,
            "api_base": remote_qwen_api_base_for_host(host_id),
            "local_port": _eai_configuration_policies.REMOTE_QWEN_HOST_LOCAL_PORTS.get(
                host_id, _eai_configuration_policies.REMOTE_QWEN_LOCAL_PORT
            ),
            "model_root": "",
            "python": "",
            "remote_work": "",
        }


class RemoteQwenDeployBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(object)


class DeployModelListBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(object)


class HumanDemoAnalyzeBridge(_eai_platform_bootstrap.QObject):
    """后台分析人类演示视频：progress(pct, msg) / finished(ok, traj_or_none, error)。"""

    progress = _eai_platform_bootstrap.pyqtSignal(float, str)
    finished = _eai_platform_bootstrap.pyqtSignal(bool, object, str)


class RemoteQwenServiceLauncher(_eai_platform_bootstrap.QObject):
    """经 SSH 在远程机部署 Qwen，并通过本地端口转发连接。"""

    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    log_line = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._starting = False
        self._fail_notified = False
        self._host_id = _eai_configuration_policies.REMOTE_QWEN_SSH_HOST
        self._last_model_id = "qwen3.5-35b-a3b"
        self._last_model_label = "Qwen3.5-35B-A3B"
        self._last_api_base = remote_qwen_api_base_for_host(self._host_id)
        self._bridge = RemoteQwenDeployBridge()
        self._bridge.finished.connect(self._on_action_finished)
        self._pending_action = ""

    def set_host(self, host_id: str) -> None:
        hid = (host_id or _eai_configuration_policies.REMOTE_QWEN_SSH_HOST).strip()
        self._host_id = hid
        self._last_api_base = remote_qwen_api_base_for_host(hid)

    def host_id(self) -> str:
        return self._host_id

    def api_base(self) -> str:
        return self._last_api_base or remote_qwen_api_base_for_host(self._host_id)

    def last_model_id(self) -> str:
        return self._last_model_id

    def last_model_label(self) -> str:
        return self._last_model_label

    def is_healthy(self) -> bool:
        return bool(_eai_adapters_models_qwen_local.check_local_qwen_server_health(self.api_base()))

    def status(self) -> _eai_platform_bootstrap.Dict[str, object]:
        try:
            ctl = load_remote_qwen_ctl()
            ctl.apply_host(self._host_id)
            return dict(ctl.status_payload())
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def start(
        self,
        model_key: _eai_platform_bootstrap.Optional[str] = None,
        host_id: _eai_platform_bootstrap.Optional[str] = None,
        model_dir: _eai_platform_bootstrap.Optional[str] = None,
        model_id: _eai_platform_bootstrap.Optional[str] = None,
        model_label: _eai_platform_bootstrap.Optional[str] = None,
    ) -> None:
        if host_id:
            self.set_host(host_id)
        deploy_key = model_key or "qwen3.5-35b-a3b"
        if model_dir:
            self._last_model_id = str(model_id or _eai_platform_bootstrap.os.path.basename(model_dir.rstrip("/")))
            self._last_model_label = str(model_label or self._last_model_id)
        # 先探测隧道：避免上次「等待中」把 _starting 卡死，挡住已就绪的服务
        if self.is_healthy():
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(self.api_base()) or {}
            mid = str(info.get("model") or "")
            mdir = str(info.get("model_dir") or "")
            want_id = str(model_id or self._last_model_id or "")
            want_dir = str(model_dir or "")
            same = _eai_adapters_models_qwen_local.model_identity_same(mid, mdir, want_id, want_dir)
            self._starting = False
            self._fail_notified = False
            self.running_changed.emit(True)
            if same is False:
                shown = mid or mdir or "未知模型"
                self.status_message.emit(
                    f"远程已在运行其它模型（{shown}）。"
                    f"若要换成 {model_label or want_id or deploy_key}，请先停止再部署。"
                )
                self.log_line.emit(
                    f"远程模型不一致: host={self._host_id} live={shown} "
                    f"want={want_id or want_dir or deploy_key}"
                )
                return
            if mid:
                self._last_model_id = mid
            self.status_message.emit(
                f"同一模型已在运行，跳过重复部署: {self.api_base()} "
                f"({mid or self._last_model_id})"
            )
            self.log_line.emit(
                f"远程已就绪: host={self._host_id} model={mid or self._last_model_id} "
                f"api={self.api_base()}"
            )
            return
        if self._starting:
            self.status_message.emit("远程部署进行中…")
            return

        self._starting = True
        self._fail_notified = False
        self._pending_action = "deploy"
        self.running_changed.emit(True)
        label = remote_qwen_host_label(self._host_id)
        self.status_message.emit(f"正在远程部署（{label}）…")
        self.log_line.emit(
            f"远程部署开始: host={self._host_id} model_key={deploy_key} "
            f"dir={model_dir or '-'} api={self.api_base()}"
        )
        host = self._host_id
        deploy_payload: _eai_platform_bootstrap.Dict[str, object] = {"host_id": host, "model_key": deploy_key}
        if model_dir:
            deploy_payload["model_dir"] = model_dir
        if model_id:
            deploy_payload["model_id"] = model_id
        if model_label:
            deploy_payload["model_label"] = model_label

        def _work() -> None:
            try:
                if check_remote_qwen_hostctl_health():
                    self.log_line.emit(
                        f"经宿主机 remote hostctl 部署: {resolve_remote_qwen_hostctl_url()}"
                    )
                    _ok, result = remote_qwen_hostctl_request(
                        "/deploy",
                        method="POST",
                        timeout_s=300.0,
                        payload=deploy_payload,
                    )
                    if not isinstance(result, dict):
                        result = {"ok": False, "message": str(result)}
                    if not _ok and "ok" not in result:
                        result = {"ok": False, "message": str(result.get("message") or result)}
                else:
                    if _eai_perception_segmentation.is_running_in_docker():
                        result = {
                            "ok": False,
                            "message": (
                                "未检测到宿主机 remote hostctl (:18103)。"
                                "请重新 bash scripts/launch/run_in_docker.sh，或在宿主机执行: "
                                f"python3 {_eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, 'compat/remote_qwen_hostctl.py')}"
                            ),
                        }
                    else:
                        ctl = load_remote_qwen_ctl()
                        ctl.apply_host(host)
                        result = ctl.deploy(
                            model_key=deploy_key,
                            model_dir=model_dir,
                            model_id=model_id,
                            model_label=model_label,
                        )
            except Exception as exc:
                result = {"ok": False, "message": str(exc)}
            self._bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def stop(self, host_id: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if host_id:
            self.set_host(host_id)
        self._starting = True
        self._pending_action = "stop"
        self.status_message.emit(f"正在停止远程服务与隧道（{self._host_id}）…")
        host = self._host_id

        def _work() -> None:
            try:
                if check_remote_qwen_hostctl_health():
                    self.log_line.emit(
                        f"经宿主机 remote hostctl 停止: {resolve_remote_qwen_hostctl_url()}"
                    )
                    _ok, result = remote_qwen_hostctl_request(
                        "/stop",
                        method="POST",
                        timeout_s=120.0,
                        payload={"host_id": host},
                    )
                    if not isinstance(result, dict):
                        result = {"ok": False, "message": str(result)}
                else:
                    ctl = load_remote_qwen_ctl()
                    ctl.apply_host(host)
                    result = ctl.stop_all()
            except Exception as exc:
                result = {"ok": False, "message": str(exc)}
            self._bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_action_finished(self, result: object) -> None:
        data = result if isinstance(result, dict) else {"ok": False, "message": str(result)}
        ok = bool(data.get("ok"))
        msg = str(data.get("message") or "")
        action = self._pending_action
        self._pending_action = ""
        self.log_line.emit(msg)

        if data.get("api_base"):
            self._last_api_base = str(data.get("api_base"))
        if data.get("host_id"):
            self._host_id = str(data.get("host_id"))

        if action == "deploy":
            if data.get("model_id"):
                self._last_model_id = str(data.get("model_id"))
            if data.get("model_label"):
                self._last_model_label = str(data.get("model_label"))
            if ok:
                # 模型加载中：隧道可能已通但 /health 尚未 ok
                self._starting = True
                self.status_message.emit(
                    f"远程已启动，等待模型就绪 → {self.api_base()}"
                )
                self.running_changed.emit(True)
            else:
                self._starting = False
                self._fail_notified = True
                self.running_changed.emit(False)
                self.status_message.emit(f"远程部署失败: {msg}")
            return

        # stop
        self._starting = False
        self.running_changed.emit(False)
        self.status_message.emit(msg if ok else f"远程停止异常: {msg}")

    def shutdown(self) -> None:
        # 界面退出不停止远程模型，也不拆隧道，下次打开仍能看到在线状态。
        return


IDLE_CONTROL_MODE = 99

