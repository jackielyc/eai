"""adapters.models.qwen local — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_adapters_simulation_robodojo_discovery = module_ref('eai_app.adapters.simulation.robodojo_discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

TEST_SCENARIO_LABELS = (
    "手机装配",
    "商超零售拆盒子",
    "真人数采",
    "THT电容插件装配",
    "叠盒子",
    "麻将机器人",
    "微波炉加热食品",
    "飞机盒装物",
)


TEST_SCENARIO_IMAGE_FILES: _eai_platform_bootstrap.Dict[str, str] = {}


def qwen_model_path_from_spec(dirname_or_path: str, *, root: str = "") -> str:
    """将部署 spec 解析为绝对路径（相对名则拼到 root / 本地 models）。"""
    if dirname_or_path.startswith("/"):
        return dirname_or_path
    base = root or _eai_configuration_policies.LOCAL_QWEN_MODELS_ROOT
    return _eai_platform_bootstrap.os.path.join(base, dirname_or_path)


def qwen_model_dir_valid(path: str) -> bool:
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(path, "config.json")) or _eai_platform_bootstrap.os.path.isfile(
        _eai_platform_bootstrap.os.path.join(path, "adapter_config.json")
    )


def qwen_model_is_lora(path: str) -> bool:
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(path, "adapter_config.json"))


def qwen_deploy_path_spec_for_key(model_key: str) -> str:
    for key, _label, _mid, spec in _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return spec
    return model_key


def remote_qwen_python_for_key(model_key: str, host_id: str = "") -> _eai_platform_bootstrap.Optional[str]:
    if model_key.startswith("lake-"):
        return _eai_configuration_policies.REMOTE_LAKE_QWEN_PYTHON
    return None


def local_qwen_model_dir_for_key(model_key: str) -> _eai_platform_bootstrap.Optional[str]:
    """按部署 key 解析权重目录（全量需 config.json，LoRA 需 adapter_config.json）。"""
    spec = qwen_deploy_path_spec_for_key(model_key)
    if spec.startswith("/"):
        return spec if qwen_model_dir_valid(spec) else None
    for root in (
        _eai_configuration_policies.LOCAL_QWEN_MODELS_ROOT,
        _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "models", "Qwen"),
    ):
        path = _eai_platform_bootstrap.os.path.join(root, spec)
        if qwen_model_dir_valid(path):
            return path
    return None


def local_qwen_model_id_for_key(model_key: str) -> str:
    for key, _label, mid, _dirname in _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return mid
    return _eai_configuration_policies.LOCAL_QWEN_MODEL_ID


def local_qwen_label_for_key(model_key: str) -> str:
    for key, label, _mid, _dirname in _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return label
    return model_key


def local_deploy_scan_roots() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    roots: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = [
        (_eai_configuration_policies.LOCAL_QWEN_MODELS_ROOT, "本地 Qwen 基座"),
    ]
    if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_policies.LAKE_QWEN35_OUTPUT_ROOT):
        roots.append((_eai_configuration_policies.LAKE_QWEN35_OUTPUT_ROOT, "Lake 训练 output"))
    return roots


def remote_deploy_scan_roots(host_id: str) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    try:
        ctl = _eai_adapters_models_qwen_remote.load_remote_qwen_ctl()
        return [(r, lbl) for r, lbl in ctl.deploy_scan_roots(host_id)]
    except Exception:
        prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id)
        return [
            (str(prof.get("model_root") or _eai_configuration_policies.LOCAL_QWEN_MODELS_ROOT), "Qwen 基座"),
            (_eai_configuration_policies.LAKE_QWEN35_OUTPUT_ROOT, "Lake 训练 output"),
        ]


def deploy_model_kind_label(kind: str) -> str:
    return "LoRA" if kind == "lora" else "全量"


def qwen_deploy_mode_suffix(
    info: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None,
    *,
    model_dir: str = "",
    kind: str = "",
) -> str:
    """部署成功后状态栏用的 mode 片段，如 ' · mode=LoRA'。"""
    k = (kind or "").strip().lower()
    if not k and isinstance(info, dict):
        k = str(info.get("kind") or "").strip().lower()
    path = (model_dir or "").strip()
    if not path and isinstance(info, dict):
        path = str(info.get("model_dir") or "").strip()
    if not k and path:
        k = "lora" if qwen_model_is_lora(path) else "full"
    if not k:
        return ""
    return f" · mode={deploy_model_kind_label(k)}"


def fetch_deploy_model_dirs(
    *,
    target: str,
    host_id: str = "",
    root: str = "",
) -> _eai_platform_bootstrap.Tuple[bool, _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]], str]:
    scan_roots = [root.strip()] if root.strip() else None
    try:
        ctl = _eai_adapters_models_qwen_remote.load_remote_qwen_ctl()
        if target == "local":
            scan = scan_roots or [r for r, _ in local_deploy_scan_roots()]
            models = ctl.list_deploy_model_dirs_local(scan)
            return True, models, ""

        host = (host_id or _eai_configuration_policies.REMOTE_QWEN_SSH_HOST).strip()

        def _via_ctl() -> _eai_platform_bootstrap.Tuple[bool, _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]], str]:
            if _eai_perception_segmentation.is_running_in_docker():
                return (
                    False,
                    [],
                    "Docker 内列举远程目录需宿主机 remote hostctl (:18103)。"
                    "请重新执行 bash eai/run_in_docker.sh 以重启 hostctl。",
                )
            result = ctl.list_deploy_model_dirs(roots=scan_roots, host_id=host)
            if result.get("ok"):
                raw = result.get("models") or []
                return True, [dict(x) for x in raw if isinstance(x, dict)], ""
            return False, [], str(result.get("message") or "列举失败")

        qs = f"?host={_eai_platform_bootstrap.urllib.parse.quote(host)}"
        if root.strip():
            qs += f"&root={_eai_platform_bootstrap.urllib.parse.quote(root.strip())}"
        if _eai_adapters_models_qwen_remote.check_remote_qwen_hostctl_health():
            ok, body = _eai_adapters_models_qwen_remote.remote_qwen_hostctl_request(
                f"/list_models{qs}",
                method="GET",
                timeout_s=120.0,
            )
            if ok:
                raw = body.get("models") or []
                return True, [dict(x) for x in raw if isinstance(x, dict)], ""
            msg = str(body.get("message") or "list_models 失败")
            if "404" in msg or "unknown /list_models" in msg:
                return _via_ctl()
            return False, [], msg
        return _via_ctl()
    except Exception as exc:
        return False, [], str(exc)


def resolve_deploy_python_for_spec(
    spec: _eai_platform_bootstrap.Dict[str, str], *, target: str, host_id: str = ""
) -> str:
    path = str(spec.get("path") or "")
    kind = str(spec.get("kind") or "")
    if target == "local":
        if kind == "lora" or _eai_configuration_policies.LAKE_QWEN35_OUTPUT_ROOT in path:
            return _eai_configuration_policies.REMOTE_LAKE_QWEN_PYTHON
        return _eai_configuration_policies.LOCAL_QWEN_PYTHON_DEFAULT
    if kind == "lora" or _eai_configuration_policies.LAKE_QWEN35_OUTPUT_ROOT in path or "lake_qwen35" in path:
        return _eai_configuration_policies.REMOTE_LAKE_QWEN_PYTHON
    prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id or _eai_configuration_policies.REMOTE_QWEN_SSH_HOST)
    return str(prof.get("python") or _eai_configuration_policies.LOCAL_QWEN_PYTHON_DEFAULT)


def resolve_local_qwen_model_dir(model_key: _eai_platform_bootstrap.Optional[str] = None) -> _eai_platform_bootstrap.Optional[str]:
    """解析本地 Qwen HuggingFace 权重目录。"""
    if model_key:
        found = local_qwen_model_dir_for_key(model_key)
        if found:
            return found
    env = _eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_MODEL_DIR", "").strip()
    candidates = []
    if env:
        candidates.append(_eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env)))
    candidates.append(_eai_configuration_policies.LOCAL_QWEN_MODEL_DIR_DEFAULT)
    for _key, _label, _mid, dirname in _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS:
        candidates.append(_eai_platform_bootstrap.os.path.join(_eai_configuration_policies.LOCAL_QWEN_MODELS_ROOT, dirname))
        candidates.append(_eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "models", "Qwen", dirname))
    for path in candidates:
        if qwen_model_dir_valid(path):
            return path
    return None


def resolve_local_qwen_run_script() -> _eai_platform_bootstrap.Optional[str]:
    path = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, _eai_configuration_policies.LOCAL_QWEN_RUN_SCRIPT_NAME)
    return path if _eai_platform_bootstrap.os.path.isfile(path) else None


def resolve_local_qwen_bind_host() -> str:
    return "0.0.0.0" if _eai_perception_segmentation.is_running_in_docker() else "127.0.0.1"


def resolve_local_qwen_viewer_api_base() -> str:
    """viewer 访问推理服务的 API。容器为 host 网络时统一用 127.0.0.1。"""
    env = _eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_API_BASE", "").strip()
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{_eai_configuration_policies.LOCAL_QWEN_PORT_DEFAULT}/v1"


def resolve_local_qwen_ctl_url() -> str:
    env = _eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_CTL_URL", "").strip()
    if env:
        return env.rstrip("/")
    return _eai_configuration_policies.LOCAL_QWEN_CTL_URL_DEFAULT.rstrip("/")


def check_local_qwen_server_health(
    api_base: _eai_platform_bootstrap.Optional[str] = None, timeout_s: float = 2.0
) -> bool:
    info = fetch_local_qwen_server_info(api_base=api_base, timeout_s=timeout_s)
    return bool(info and info.get("ok"))


def fetch_local_qwen_server_info(
    api_base: _eai_platform_bootstrap.Optional[str] = None, timeout_s: float = 2.0
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]]:
    base = (api_base or resolve_local_qwen_viewer_api_base()).rstrip("/")
    root = base.removesuffix("/v1").rstrip("/")
    url = f"{root}/health"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict):
                return body
    except Exception:
        return None
    return None


def check_local_qwen_hostctl_health(timeout_s: float = 1.0) -> bool:
    url = resolve_local_qwen_ctl_url().rstrip("/") + "/health"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def local_qwen_hostctl_request(
    path: str,
    method: str = "GET",
    timeout_s: float = 10.0,
    payload: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None,
) -> _eai_platform_bootstrap.Tuple[bool, _eai_platform_bootstrap.Dict[str, object]]:
    url = resolve_local_qwen_ctl_url().rstrip("/") + path
    data = None
    if method == "POST":
        data = _eai_platform_bootstrap.json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = _eai_platform_bootstrap.urllib.request.Request(url, data=data, method=method)
    if method == "POST":
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok", True)), body
    except Exception as exc:
        return False, {"ok": False, "message": str(exc)}


def _norm_model_token(value: object) -> str:
    """Normalize a model id or path for equality checks."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""
    if text.startswith("gs://"):
        return text.rstrip("/")
    expanded = _eai_platform_bootstrap.os.path.expanduser(text)
    if _eai_platform_bootstrap.os.path.isabs(expanded):
        try:
            return _eai_platform_bootstrap.os.path.realpath(expanded).rstrip("/")
        except OSError:
            return expanded.rstrip("/")
    return text


def model_identity_same(
    live_id: object,
    live_dir: object,
    want_id: object,
    want_dir: object,
) -> _eai_platform_bootstrap.Optional[bool]:
    """True if the running model is the requested one.

    False when both sides are known and differ. None when there is not
    enough information to decide.
    """
    lid = str(live_id or "").strip()
    wid = str(want_id or "").strip()
    ld = _norm_model_token(live_dir)
    wd = _norm_model_token(want_dir)
    if ld and wd and (_eai_platform_bootstrap.os.path.isabs(ld) or _eai_platform_bootstrap.os.path.isabs(wd) or ld.startswith("gs:") or wd.startswith("gs:")):
        same_dir = ld == wd or (
            _eai_platform_bootstrap.os.path.basename(ld) == _eai_platform_bootstrap.os.path.basename(wd) and bool(_eai_platform_bootstrap.os.path.basename(ld))
        )
        if not same_dir:
            return False
        if lid and wid and lid != wid:
            return False
        return True
    if lid and wid:
        return lid == wid
    return None


PERSIST_LOG_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "log", "persist")


def persistent_model_shell(inner: str, slot: str) -> _eai_platform_bootstrap.Tuple[str, str]:
    """Run ``inner`` in its own session and tail its log for the UI.

    The QProcess child is ``tail``. Closing the UI kills only the tail;
    the model session keeps running. The pid file records the model session.
    """
    safe = _eai_platform_bootstrap.re.sub(r"[^A-Za-z0-9_.-]+", "_", slot).strip("._")[:80] or "model"
    _eai_platform_bootstrap.os.makedirs(PERSIST_LOG_DIR, exist_ok=True)
    log_path = _eai_platform_bootstrap.os.path.join(PERSIST_LOG_DIR, safe + ".log")
    pid_path = _eai_platform_bootstrap.os.path.join(PERSIST_LOG_DIR, safe + ".pid")
    quoted_inner = (
        "trap '' PIPE; "
        f"exec >> {_eai_platform_bootstrap.shlex.quote(log_path)} 2>&1; "
        + inner
    )
    cmd = (
        "trap '' PIPE; "
        f"mkdir -p {_eai_platform_bootstrap.shlex.quote(PERSIST_LOG_DIR)}; "
        f": > {_eai_platform_bootstrap.shlex.quote(log_path)}; "
        f"setsid bash -lc {_eai_platform_bootstrap.shlex.quote(quoted_inner)} </dev/null >/dev/null 2>&1 & "
        f"echo $! > {_eai_platform_bootstrap.shlex.quote(pid_path)}; "
        f"exec tail -n +1 -F --pid=$(cat {_eai_platform_bootstrap.shlex.quote(pid_path)}) {_eai_platform_bootstrap.shlex.quote(log_path)}"
    )
    return cmd, pid_path


def persistent_slot_paths(slot: str) -> _eai_platform_bootstrap.Tuple[str, str]:
    safe = _eai_platform_bootstrap.re.sub(r"[^A-Za-z0-9_.-]+", "_", slot).strip("._")[:80] or "model"
    return (
        _eai_platform_bootstrap.os.path.join(PERSIST_LOG_DIR, safe + ".log"),
        _eai_platform_bootstrap.os.path.join(PERSIST_LOG_DIR, safe + ".pid"),
    )


def read_persistent_pid(pid_path: str) -> int:
    try:
        pid = int(open(pid_path, "r", encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return 0
    if pid <= 1:
        return 0
    try:
        _eai_platform_bootstrap.os.kill(pid, 0)
    except OSError:
        return 0
    return pid


def persistent_slot_blob(slot: str) -> str:
    """Cmdline + environ of a still-running persisted model, else empty."""
    _log_path, pid_path = persistent_slot_paths(slot)
    pid = read_persistent_pid(pid_path)
    if pid <= 1:
        return ""
    parts: _eai_platform_bootstrap.List[str] = []
    for name in ("cmdline", "environ"):
        try:
            with open(f"/proc/{pid}/{name}", "rb") as fh:
                parts.append(fh.read().replace(b"\0", b" ").decode("utf-8", "replace"))
        except OSError:
            continue
    return " ".join(parts)


def stop_persistent_model(pid_path: str) -> None:
    """SIGTERM then SIGKILL the persisted model session (not the UI log tail)."""
    pid = read_persistent_pid(pid_path)
    if pid <= 1:
        return
    for sig in (_eai_platform_bootstrap.signal.SIGTERM, _eai_platform_bootstrap.signal.SIGKILL):
        try:
            _eai_platform_bootstrap.os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        except OSError:
            try:
                _eai_platform_bootstrap.os.kill(pid, sig)
            except OSError:
                return
        if sig == _eai_platform_bootstrap.signal.SIGTERM:
            _eai_platform_bootstrap.time.sleep(0.15)


def listener_cmdline(port: int) -> str:
    """Cmdline and environ of the process listening on ``port``."""
    try:
        pid = _eai_adapters_simulation_robodojo_discovery._pid_listening_on_tcp_port(int(port))
    except Exception:
        return ""
    if not pid or int(pid) <= 1:
        return ""
    parts: _eai_platform_bootstrap.List[str] = []
    for name in ("cmdline", "environ"):
        try:
            with open(f"/proc/{int(pid)}/{name}", "rb") as fh:
                parts.append(fh.read().replace(b"\0", b" ").decode("utf-8", "replace"))
        except OSError:
            continue
    return " ".join(parts)


def blob_mentions_model(blob: str, model_id: str = "", model_dir: str = "") -> _eai_platform_bootstrap.Optional[bool]:
    """Whether ``blob`` looks like ``model_id`` / ``model_dir``.

    None if the blob does not contain an obvious model reference.
    """
    text = blob or ""
    if not text.strip():
        return None
    mid = str(model_id or "").strip()
    md = _norm_model_token(model_dir)
    if md and (md in text or _eai_platform_bootstrap.os.path.basename(md) and _eai_platform_bootstrap.os.path.basename(md) in text):
        return True
    if mid and mid in text:
        return True
    if md or mid:
        return False
    return None


class LocalQwenServiceLauncher(_eai_platform_bootstrap.QObject):
    """启动/停止本地 Qwen OpenAI 兼容推理服务。

    Docker 内优先走宿主机 hostctl（:18101），以便使用宿主机 GPU / conda。
    """

    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    log_line = _eai_platform_bootstrap.pyqtSignal(str)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._api_base = resolve_local_qwen_viewer_api_base()
        self._via_hostctl = False
        self._last_model_id = _eai_configuration_policies.LOCAL_QWEN_MODEL_ID
        self._last_model_label = _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS[0][1]
        self._starting = False
        self._fail_notified = False

    def api_base(self) -> str:
        return self._api_base

    def last_model_id(self) -> str:
        return self._last_model_id

    def last_model_label(self) -> str:
        return self._last_model_label

    def is_running(self) -> bool:
        if check_local_qwen_server_health(self._api_base):
            return True
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            return True
        return False

    def start(
        self,
        model_key: _eai_platform_bootstrap.Optional[str] = None,
        model_dir: _eai_platform_bootstrap.Optional[str] = None,
        model_id: _eai_platform_bootstrap.Optional[str] = None,
        model_label: _eai_platform_bootstrap.Optional[str] = None,
    ) -> None:
        self._api_base = resolve_local_qwen_viewer_api_base()
        key = model_key or _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS[0][0]
        resolved_dir = model_dir or resolve_local_qwen_model_dir(key)
        resolved_id = model_id or local_qwen_model_id_for_key(key)
        resolved_label = model_label or local_qwen_label_for_key(key)
        self._last_model_id = resolved_id
        self._last_model_label = resolved_label

        if check_local_qwen_server_health(self._api_base):
            info = fetch_local_qwen_server_info(self._api_base) or {}
            running_id = str(info.get("model") or "")
            running_dir = str(info.get("model_dir") or "")
            self._starting = False
            self.running_changed.emit(True)
            same = model_identity_same(
                running_id, running_dir, resolved_id, resolved_dir or ""
            )
            if same is False:
                shown = running_id or running_dir or "未知模型"
                self.status_message.emit(
                    f"本地 Qwen 已在运行其它模型（{shown}）: {self._api_base}。"
                    f"若要换成 {resolved_label}，请先停止再部署。"
                )
            else:
                self.status_message.emit(
                    f"同一模型已在运行，跳过重复部署: {self._api_base} ({resolved_label})"
                )
            return

        loading_blob = persistent_slot_blob("local_qwen")
        if loading_blob:
            same_loading = blob_mentions_model(
                loading_blob, resolved_id, resolved_dir or ""
            )
            self._starting = True
            self.running_changed.emit(True)
            if same_loading is False:
                self.status_message.emit(
                    f"已有其它模型正在启动。若要换成 {resolved_label}，请先停止再部署。"
                )
            else:
                self.status_message.emit(
                    f"同一模型正在启动，跳过重复部署: {resolved_label}"
                )
            return

        if resolved_dir is None:
            self._starting = False
            self.running_changed.emit(False)
            self.status_message.emit(
                f"未找到 {resolved_label} 权重（models/Qwen/…）"
            )
            return

        start_payload: _eai_platform_bootstrap.Dict[str, object] = {
            "model_dir": resolved_dir,
            "model_id": resolved_id,
            "model_label": resolved_label,
        }
        self._starting = True
        self._fail_notified = False

        if check_local_qwen_hostctl_health():
            self.log_line.emit(
                f"经宿主机 hostctl 启动 {resolved_label}: "
                f"{resolve_local_qwen_ctl_url()}"
            )
            ok, body = local_qwen_hostctl_request(
                "/start",
                method="POST",
                timeout_s=30.0,
                payload=start_payload,
            )
            msg = str(body.get("message") or body)
            self.log_line.emit(msg)
            if ok:
                self._via_hostctl = True
                self.running_changed.emit(True)
                self.status_message.emit(
                    f"已请求宿主机启动 {resolved_label} → {self._api_base}"
                )
            else:
                self._starting = False
                self.running_changed.emit(False)
                self.status_message.emit(f"hostctl 启动失败: {msg}")
            return

        if _eai_perception_segmentation.is_running_in_docker():
            self._starting = False
            self.running_changed.emit(False)
            self.status_message.emit(
                "未检测到宿主机 hostctl。请在宿主机执行: "
                f"python3 {_eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, 'local_qwen_hostctl.py')} "
                "或重新 bash run_in_docker.sh"
            )
            self.log_line.emit(
                "容器内无 NVIDIA 驱动库，不能直接加载 GPU 模型；"
                "必须由宿主机 hostctl / run_local_qwen.sh 启动。"
            )
            return

        script = resolve_local_qwen_run_script()
        if script is None:
            self.status_message.emit(
                f"未找到 {_eai_configuration_policies.LOCAL_QWEN_RUN_SCRIPT_NAME}，请确认与 show_camera_topics.py 同目录"
            )
            return

        bind_host = resolve_local_qwen_bind_host()
        script_dir = _eai_platform_bootstrap.os.path.dirname(script)
        py = _eai_configuration_policies.LOCAL_QWEN_PYTHON_DEFAULT
        cmd = (
            f"exec bash {_eai_platform_bootstrap.shlex.quote(_eai_platform_bootstrap.os.path.basename(script))} "
            f"--host {_eai_platform_bootstrap.shlex.quote(bind_host)} "
            f"--port {_eai_configuration_policies.LOCAL_QWEN_PORT_DEFAULT} "
            f"--model {_eai_platform_bootstrap.shlex.quote(resolved_dir)} "
            f"--model-id {_eai_platform_bootstrap.shlex.quote(resolved_id)} "
            f"--python {_eai_platform_bootstrap.shlex.quote(py)}"
        )
        # 模型在独立 session 里跑；QProcess 只 tail 日志，界面退出不杀模型。
        shell, pid_path = persistent_model_shell(cmd, "local_qwen")
        self._persist_pid_path = pid_path
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(script_dir)
        proc.start("setsid", ["bash", "-lc", shell])
        self._process = proc
        self._via_hostctl = False
        self.running_changed.emit(True)
        self.status_message.emit(
            f"正在启动 {resolved_label} ({bind_host}:{_eai_configuration_policies.LOCAL_QWEN_PORT_DEFAULT})，"
            f"viewer API: {self._api_base}"
        )
        self.log_line.emit(f"启动: {cmd}")

    def stop(self) -> None:
        self._starting = False
        stop_persistent_model(getattr(self, "_persist_pid_path", "") or persistent_slot_paths("local_qwen")[1])
        if self._via_hostctl or check_local_qwen_hostctl_health():
            ok, body = local_qwen_hostctl_request("/stop", method="POST", timeout_s=15.0)
            msg = str(body.get("message") or body)
            self.log_line.emit(msg)
            self._via_hostctl = False
            self.running_changed.emit(False)
            self.status_message.emit(
                "已停止本地 Qwen 推理服务" if ok else f"停止失败: {msg}"
            )
            return

        if self._process is not None:
            if self._process.state() == _eai_platform_bootstrap.QProcess.Running:
                self._process.terminate()
                _eai_platform_bootstrap.QTimer.singleShot(3000, self._force_kill_process)
            else:
                self._process = None
        try:
            _eai_platform_bootstrap.subprocess.run(
                ["pkill", "-f", "local_qwen_worker.py"],
                capture_output=True,
                timeout=2,
                check=False,
            )
        except Exception:
            pass
        self.running_changed.emit(False)
        self.status_message.emit("已停止本地 Qwen 推理服务")

    def shutdown(self) -> None:
        # 只结束日志 tail。推理进程在独立 session，界面退出后继续服务。
        proc = self._process
        self._process = None
        if proc is not None and proc.state() == _eai_platform_bootstrap.QProcess.Running:
            proc.terminate()
            proc.waitForFinished(400)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            line = line.strip()
            if line:
                self.log_line.emit(line)

    def _on_process_finished(self, exit_code: int, _exit_status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        self._process = None
        self._starting = False
        self.running_changed.emit(False)
        self.status_message.emit(f"本地 Qwen 服务已退出 (code={exit_code})")

    def _on_process_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error != _eai_platform_bootstrap.QProcess.Crashed:
            self.status_message.emit(f"本地 Qwen 服务进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running:
            self._process.kill()

