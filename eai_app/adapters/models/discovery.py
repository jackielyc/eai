"""adapters.models.discovery — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_rlinf_root(path: str = "") -> str:
    raw = (path or "").strip() or _eai_platform_bootstrap.os.environ.get("RLINF_ROOT", "") or _eai_configuration_tabs.RLINF_ROOT_DEFAULT
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_rlinf_python(repo: str = "") -> str:
    env = (_eai_platform_bootstrap.os.environ.get("RLINF_PYTHON") or "").strip()
    if env and _eai_platform_bootstrap.os.path.isfile(env):
        return _eai_platform_bootstrap.os.path.abspath(env)
    for path in _eai_configuration_tabs.RLINF_PYTHON_CANDIDATES:
        if _eai_platform_bootstrap.os.path.isfile(path):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return "python3"


def list_reward_workflow_configs(
    rlinf_root: str = "",
    *,
    kind: str = "train",
) -> _eai_platform_bootstrap.List[str]:
    """扫描 examples/reward/config 下 yaml 名（train / collect）。"""
    root = resolve_rlinf_root(rlinf_root)
    cfg_dir = _eai_platform_bootstrap.os.path.join(root, "examples", "reward", "config")
    if not _eai_platform_bootstrap.os.path.isdir(cfg_dir):
        return []
    kind_key = (kind or "train").strip().lower()
    names: _eai_platform_bootstrap.List[str] = []
    try:
        for fname in _eai_platform_bootstrap.os.listdir(cfg_dir):
            if not fname.endswith(".yaml"):
                continue
            stem = fname[: -len(".yaml")]
            low = stem.lower()
            if kind_key == "collect":
                if "collect" not in low and "teleop" not in low:
                    continue
                if "teleop" in low:
                    continue
            else:
                # train：排除真机采集 / teleop
                if "collect" in low or "teleop" in low or low.startswith("realworld_"):
                    continue
            names.append(stem)
    except OSError:
        return []
    return sorted(names)


def resolve_rise_root(path: str = "") -> str:
    raw = (path or "").strip() or _eai_platform_bootstrap.os.environ.get("RISE_ROOT", "") or _eai_configuration_policies.RISE_ROOT_DEFAULT
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_rise_python(repo: str = "") -> str:
    for env_key in ("RISE_PYTHON", "RLINF_PYTHON"):
        env = (_eai_platform_bootstrap.os.environ.get(env_key) or "").strip()
        if env and _eai_platform_bootstrap.os.path.isfile(env):
            return _eai_platform_bootstrap.os.path.abspath(env)
    for path in _eai_configuration_policies.RISE_PYTHON_CANDIDATES:
        if _eai_platform_bootstrap.os.path.isfile(path):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return "python3"


def is_rise_openpi_ckpt_dir(path: str) -> bool:
    """OpenPI/RISE ckpt：目录内有 model.safetensors（可选 assets/）。"""
    raw = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((path or "").strip()))
    if not raw or not _eai_platform_bootstrap.os.path.isdir(raw):
        return False
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(raw, "model.safetensors"))


def resolve_rise_policy_ckpt(prefer: str = "") -> str:
    """解析可用策略 ckpt 目录；环境变量 RISE_POLICY_CKPT 优先。"""
    for cand in (
        (prefer or "").strip(),
        (_eai_platform_bootstrap.os.environ.get("RISE_POLICY_CKPT") or "").strip(),
        *_eai_configuration_policies.RISE_POLICY_CKPT_CANDIDATES,
    ):
        if is_rise_openpi_ckpt_dir(cand):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cand))
    return ""


def resolve_rise_reward_ckpt(prefer: str = "", *, policy_ckpt: str = "") -> str:
    """解析可用奖励/value ckpt；无独立产物时回退到策略 ckpt。"""
    for cand in (
        (prefer or "").strip(),
        (_eai_platform_bootstrap.os.environ.get("RISE_REWARD_CKPT") or "").strip(),
        *_eai_configuration_policies.RISE_REWARD_CKPT_CANDIDATES,
    ):
        if is_rise_openpi_ckpt_dir(cand):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cand))
    fallback = resolve_rise_policy_ckpt(policy_ckpt)
    return fallback


def list_robodojo_online_rl_configs(rise_root: str = "") -> _eai_platform_bootstrap.List[str]:
    """扫描 RISE policy_online/examples/embodiment/config 下顶层 yaml。"""
    root = resolve_rise_root(rise_root)
    cfg_dir = _eai_platform_bootstrap.os.path.join(
        root, "policy_and_value", "policy_online", "examples", "embodiment", "config"
    )
    if not _eai_platform_bootstrap.os.path.isdir(cfg_dir):
        return []
    names: _eai_platform_bootstrap.List[str] = []
    try:
        for fname in _eai_platform_bootstrap.os.listdir(cfg_dir):
            if not fname.endswith(".yaml"):
                continue
            names.append(fname[: -len(".yaml")])
    except OSError:
        return []
    return sorted(names)


def resolve_robometer_root(path: str = "") -> str:
    raw = (
        (path or "").strip()
        or _eai_platform_bootstrap.os.environ.get("ROBOMETER_ROOT", "")
        or _eai_configuration_tabs.ROBOMETER_ROOT_DEFAULT
    )
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_robometer_python(repo: str = "") -> str:
    for path in (
        (_eai_platform_bootstrap.os.environ.get("ROBOMETER_PYTHON") or "").strip(),
        *_eai_configuration_tabs.ROBOMETER_PYTHON_CANDIDATES,
    ):
        if path and _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
    which = _eai_platform_bootstrap.shutil.which("python3") or _eai_platform_bootstrap.shutil.which("python")
    return which or "python3"


def list_robometer_configs(repo: str = "") -> _eai_platform_bootstrap.List[str]:
    """扫描 robometer_policy_learning/configs 下顶层 yaml。"""
    root = resolve_robometer_root(repo)
    cfg_dir = _eai_platform_bootstrap.os.path.join(root, "robometer_policy_learning", "configs")
    if not _eai_platform_bootstrap.os.path.isdir(cfg_dir):
        return []
    names: _eai_platform_bootstrap.List[str] = []
    try:
        for fname in _eai_platform_bootstrap.os.listdir(cfg_dir):
            if not fname.endswith(".yaml"):
                continue
            names.append(fname[: -len(".yaml")])
    except OSError:
        return []
    return sorted(names)


def resolve_rynnvalue_root(path: str = "") -> str:
    raw = (
        (path or "").strip()
        or _eai_platform_bootstrap.os.environ.get("RYNNVALUE_ROOT", "")
        or _eai_configuration_tabs.RYNNVALUE_ROOT_DEFAULT
    )
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_rynnvalue_python(repo: str = "") -> str:
    for path in (
        (_eai_platform_bootstrap.os.environ.get("RYNNVALUE_PYTHON") or "").strip(),
        *_eai_configuration_tabs.RYNNVALUE_PYTHON_CANDIDATES,
    ):
        if path and _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
    which = _eai_platform_bootstrap.shutil.which("python3") or _eai_platform_bootstrap.shutil.which("python")
    return which or "python3"


def resolve_roboicl_root(path: str = "") -> str:
    raw = (
        (path or "").strip()
        or _eai_platform_bootstrap.os.environ.get("ROBOICL_ROOT", "")
        or _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
    )
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_roboicl_policy_python(repo: str = "") -> str:
    for path in (
        (_eai_platform_bootstrap.os.environ.get("ROBOICL_POLICY_PYTHON") or "").strip(),
        *_eai_configuration_tabs.ROBOICL_POLICY_PYTHON_CANDIDATES,
    ):
        if path and _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
    which = _eai_platform_bootstrap.shutil.which("python3") or _eai_platform_bootstrap.shutil.which("python")
    return which or "python3"


def resolve_roboicl_sim_python(repo: str = "") -> str:
    for path in (
        (_eai_platform_bootstrap.os.environ.get("ROBOICL_SIM_PYTHON") or "").strip(),
        *_eai_configuration_tabs.ROBOICL_SIM_PYTHON_CANDIDATES,
    ):
        if path and _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
    return resolve_roboicl_policy_python(repo)


def list_roboicl_tasks(repo: str = "") -> _eai_platform_bootstrap.List[str]:
    """扫描 RoboICL configs/tasks/*.json。"""
    root = resolve_roboicl_root(repo)
    tasks_dir = _eai_platform_bootstrap.os.path.join(root, "configs", "tasks")
    if not _eai_platform_bootstrap.os.path.isdir(tasks_dir):
        return []
    names: _eai_platform_bootstrap.List[str] = []
    try:
        for fname in _eai_platform_bootstrap.os.listdir(tasks_dir):
            if not fname.endswith(".json"):
                continue
            names.append(fname[: -len(".json")])
    except OSError:
        return []
    return sorted(names)


def _rynnvalue_hf_model_ready(path: str) -> bool:
    """True if path looks like a usable RynnValue HuggingFace export."""
    root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((path or "").strip()))
    if not root or not _eai_platform_bootstrap.os.path.isdir(root):
        return False
    if not _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(root, "config.json")):
        return False
    for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(root, name)):
            return True
    try:
        for fname in _eai_platform_bootstrap.os.listdir(root):
            if fname.endswith(".safetensors") or fname.endswith(".bin"):
                return True
    except OSError:
        return False
    return False


def _rynnvalue_ckpt_ready(path: str) -> bool:
    """True if path is a training ckpt dir containing model.pt."""
    root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((path or "").strip()))
    if not root or not _eai_platform_bootstrap.os.path.isdir(root):
        return False
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(root, "model.pt"))


def resolve_rynnvalue_model(path: str = "") -> str:
    """Prefer explicit / env / models/ HF export; empty if none ready."""
    models_root = "/share_data/projects/mahjong/share/personal/liyichao/models"
    candidates = [
        (path or "").strip(),
        (_eai_platform_bootstrap.os.environ.get("RYNNVALUE_MODEL_PATH") or "").strip(),
        _eai_configuration_tabs.RYNNVALUE_MODEL_DEFAULT,
        _eai_platform_bootstrap.os.path.join(models_root, "RynnValue-4B"),
        _eai_platform_bootstrap.os.path.join(models_root, "RynnValue-8B"),
        _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "weights", "RynnValue-4B"),
        _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "weights", "RynnValue-8B"),
    ]
    for cand in candidates:
        if cand and _rynnvalue_hf_model_ready(cand):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cand))
    # Fall back to default path even if still downloading (UI placeholder).
    for cand in candidates:
        if cand:
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cand))
    return _eai_configuration_tabs.RYNNVALUE_MODEL_DEFAULT


def resolve_rynnvalue_ckpt(path: str = "") -> str:
    """Prefer explicit / env / local training checkpoint; empty if none."""
    candidates = [
        (path or "").strip(),
        (_eai_platform_bootstrap.os.environ.get("RYNNVALUE_CKPT_PATH") or "").strip(),
        _eai_configuration_tabs.RYNNVALUE_CKPT_DEFAULT,
    ]
    for cand in candidates:
        if cand and _rynnvalue_ckpt_ready(cand):
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cand))
    # Scan cache for checkpoint_model_* / */model.pt
    cache = _eai_configuration_tabs.RYNNVALUE_CACHE_DIR
    if _eai_platform_bootstrap.os.path.isdir(cache):
        try:
            names = sorted(_eai_platform_bootstrap.os.listdir(cache), reverse=True)
        except OSError:
            names = []
        for name in names:
            cand = _eai_platform_bootstrap.os.path.join(cache, name)
            if _rynnvalue_ckpt_ready(cand):
                return cand
            nested = _eai_platform_bootstrap.os.path.join(cand, "model.pt")
            if _eai_platform_bootstrap.os.path.isfile(nested):
                return cand
    return ""


def parse_rynnvalue_server_url(url: str) -> _eai_platform_bootstrap.Tuple[str, int]:
    """Return (host, port) from a reward_server base URL."""
    raw = (url or "").strip() or "http://127.0.0.1:8001"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        host = (parsed.hostname or "127.0.0.1").strip() or "127.0.0.1"
        port = int(parsed.port or 8001)
        return host, port
    except Exception:
        return "127.0.0.1", 8001


def probe_rynnvalue_reward_server(
    url: str, timeout_s: float = 2.5
) -> _eai_platform_bootstrap.Tuple[bool, str, _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]]:
    """Probe RynnValue reward_server /health (+ optional /model_info)."""
    base = (url or "").strip().rstrip("/") or "http://127.0.0.1:8001"
    info: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any] = {"url": base}
    ok, body, err = _eai_adapters_models_chat_client._http_get_json(f"{base}/health", timeout_s=timeout_s)
    if not ok:
        host, port = parse_rynnvalue_server_url(base)
        listening = False
        try:
            listening = bool(_eai_configuration_models.port_listening(host, port))
        except Exception:
            listening = False
        if listening:
            return (
                False,
                f"端口 {host}:{port} 已占用，但 /health 未就绪（可能仍在加载模型）",
                {**info, "listening": True, "error": err},
            )
        return False, f"未部署 / 不可达: {err or 'connection failed'}", {
            **info,
            "listening": False,
            "error": err,
        }

    status = ""
    reward_model = ""
    if isinstance(body, dict):
        status = str(body.get("status") or "")
        reward_model = str(body.get("reward_model") or "")
        info["health"] = body
    healthy = (not status) or status.lower() in ("healthy", "ok", "ready")
    if not healthy:
        return False, f"/health 异常: {body}", info

    ok_info, model_body, _ = _eai_adapters_models_chat_client._http_get_json(f"{base}/model_info", timeout_s=timeout_s)
    if ok_info and isinstance(model_body, dict):
        info["model_info"] = model_body
        reward_model = str(model_body.get("reward_model") or reward_model)

    label = reward_model or "reward_server"
    return True, f"已就绪（{label} @ {base}）", info


def stop_rynnvalue_server_on_port(
    port: int = 8001,
    *,
    log: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], None]] = None,
) -> str:
    """Best-effort stop of listeners on the RynnValue reward_server port."""
    port = int(port)

    def _log(msg: str) -> None:
        if log:
            log(msg)

    pids: _eai_platform_bootstrap.List[int] = []
    try:
        out = _eai_platform_bootstrap.subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            stderr=_eai_platform_bootstrap.subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        for tok in out.replace("\n", " ").split():
            if tok.isdigit():
                pids.append(int(tok))
    except (_eai_platform_bootstrap.subprocess.CalledProcessError, FileNotFoundError, _eai_platform_bootstrap.subprocess.TimeoutExpired):
        pass

    listening = False
    try:
        listening = bool(_eai_configuration_models.port_listening("127.0.0.1", port))
    except Exception:
        listening = False
    if not pids and not listening:
        return f"端口 {port} 未占用"

    for pid in pids:
        try:
            _eai_platform_bootstrap.os.kill(pid, _eai_platform_bootstrap.signal.SIGTERM)
            _log(f"[rynnvalue] SIGTERM pid={pid} (port {port})")
        except (ProcessLookupError, PermissionError) as exc:
            _log(f"[rynnvalue] 无法结束 pid={pid}: {exc}")

    deadline = _eai_platform_bootstrap.time.time() + 8.0
    while _eai_platform_bootstrap.time.time() < deadline:
        still = False
        try:
            still = bool(_eai_configuration_models.port_listening("127.0.0.1", port))
        except Exception:
            still = False
        if not still:
            break
        _eai_platform_bootstrap.time.sleep(0.3)

    still_up = False
    try:
        still_up = bool(_eai_configuration_models.port_listening("127.0.0.1", port))
    except Exception:
        still_up = False
    if still_up:
        for pid in pids:
            try:
                _eai_platform_bootstrap.os.kill(pid, _eai_platform_bootstrap.signal.SIGKILL)
                _log(f"[rynnvalue] SIGKILL pid={pid}")
            except (ProcessLookupError, PermissionError):
                pass
        _eai_platform_bootstrap.time.sleep(0.5)

    try:
        still_up = bool(_eai_configuration_models.port_listening("127.0.0.1", port))
    except Exception:
        still_up = False
    if still_up:
        return f"端口 {port} 仍被占用，请手动结束进程"
    return f"已停止端口 {port} 上的 RynnValue 服务"


def rynnvalue_status_log_is_routine(line: str) -> bool:
    """High-frequency RynnValue status lines (scores, health, 2xx access).

    Remaining time already shows on the Live HUD window and the status label.
    These lines would otherwise fill the log once per refresh.
    """
    text = (line or "").strip()
    if not text:
        return False
    if "Live HUD score step=" in text:
        return True
    lower = text.lower()
    if "http/1." not in lower:
        return False
    if not any(
        path in lower
        for path in ("/health", "/evaluate", "/model_info", "/predict")
    ):
        return False
    # Keep failed requests visible.
    if any(
        code in text
        for code in (" 400 ", " 404 ", " 422 ", " 500 ", " 502 ", " 503 ", " 504 ")
    ):
        return False
    return True


def resolve_mujoco_root(path: str = "") -> str:
    raw = (path or "").strip() or _eai_platform_bootstrap.os.environ.get("MUJOCO_ROOT", "") or _eai_configuration_tabs.MUJOCO_ROOT_DEFAULT
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_mujoco_python(repo: str = "") -> str:
    for env_key in ("MUJOCO_PYTHON", "PYTHON"):
        env = (_eai_platform_bootstrap.os.environ.get(env_key) or "").strip()
        if env and _eai_platform_bootstrap.os.path.isfile(env):
            return _eai_platform_bootstrap.os.path.abspath(env)
    candidates = list(_eai_configuration_tabs.MUJOCO_PYTHON_CANDIDATES)
    # Prefer interpreters that already have mujoco installed.
    for path in candidates:
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            continue
        try:
            import subprocess

            r = subprocess.run(
                [path, "-c", "import mujoco"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            if r.returncode == 0:
                return _eai_platform_bootstrap.os.path.abspath(path)
        except Exception:
            continue
    for path in candidates:
        if path and _eai_platform_bootstrap.os.path.isfile(path):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return "python3"


def molmospaces_assets_dir() -> str:
    """Resolved ``MLSPACES_ASSETS_DIR`` used by MolmoSpaces / Psi-MolmoSpaces."""
    assets_parent = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "molmospaces", "assets")
    assets = assets_parent
    if _eai_platform_bootstrap.os.path.isdir(assets_parent):
        subs = []
        try:
            for name in sorted(_eai_platform_bootstrap.os.listdir(assets_parent)):
                path = _eai_platform_bootstrap.os.path.join(assets_parent, name)
                if _eai_platform_bootstrap.os.path.isdir(path):
                    subs.append(path)
        except OSError:
            subs = []
        if len(subs) == 1:
            assets = subs[0]
    return assets


def molmospaces_cache_env() -> dict:
    """Env so MolmoSpaces reads eai/.cache instead of ~/.cache."""
    cache = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "molmo-spaces-resources")
    return {
        "MLSPACES_CACHE_DIR": cache,
        "MLSPACES_ASSETS_DIR": molmospaces_assets_dir(),
    }


def ensure_psi_r1_robot_assets(repo_root: str = "") -> _eai_platform_bootstrap.Tuple[bool, str]:
    """Ensure ``ASSETS_DIR/robots/psi_r1_ruiyan`` exists for Psi eval.

    Official molmospaces-resources does not ship Psi R1; the fork keeps it under
    ``symlink/resources/robots/psi_r1_ruiyan``. Symlink that (or a local
    ``psi_r1_ruiyan`` checkout) into the EAI assets cache.
    """
    assets = molmospaces_assets_dir()
    dest = _eai_platform_bootstrap.os.path.join(assets, "robots", "psi_r1_ruiyan")
    model = _eai_platform_bootstrap.os.path.join(dest, "urdf", "model.xml")
    if _eai_platform_bootstrap.os.path.isfile(model):
        return True, dest

    root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(repo_root or ""))
    candidates: _eai_platform_bootstrap.List[str] = []
    if root:
        candidates.append(
            _eai_platform_bootstrap.os.path.join(root, "symlink", "resources", "robots", "psi_r1_ruiyan")
        )
    candidates.extend(
        [
            _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_models.EAI_DIR), "psi_r1_ruiyan"),
            _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "assets", "psi_r1_ruiyan"),
        ]
    )
    src = ""
    for cand in candidates:
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(cand, "urdf", "model.xml")):
            src = cand
            break
    if not src:
        return False, (
            "未找到 psi_r1_ruiyan（需含 urdf/model.xml）。"
            "请确认 psi-molmospaces/symlink/resources/robots/psi_r1_ruiyan 存在。"
        )

    robots_dir = _eai_platform_bootstrap.os.path.join(assets, "robots")
    _eai_platform_bootstrap.os.makedirs(robots_dir, exist_ok=True)
    try:
        if _eai_platform_bootstrap.os.path.islink(dest) or _eai_platform_bootstrap.os.path.exists(dest):
            if _eai_platform_bootstrap.os.path.islink(dest) or _eai_platform_bootstrap.os.path.isfile(dest):
                _eai_platform_bootstrap.os.remove(dest)
            else:
                return False, f"目标已存在且不是链接，请手动处理: {dest}"
        _eai_platform_bootstrap.os.symlink(src, dest)
    except OSError as exc:
        return False, f"无法链接 psi_r1_ruiyan → {dest}: {exc}"
    if not _eai_platform_bootstrap.os.path.isfile(model):
        return False, f"链接后仍缺少 {model}"
    return True, dest


def molmospaces_asset_roots(repo_root: str = "") -> _eai_platform_bootstrap.List[str]:
    """Directories that may contain benchmarks/ (assets cache, then repo assets/)."""
    roots: _eai_platform_bootstrap.List[str] = []
    env = (_eai_platform_bootstrap.os.environ.get("MLSPACES_ASSETS_DIR") or "").strip()
    if env:
        roots.append(_eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env)))
    cache = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "molmospaces", "assets")
    if _eai_platform_bootstrap.os.path.isdir(cache):
        try:
            for name in sorted(_eai_platform_bootstrap.os.listdir(cache)):
                path = _eai_platform_bootstrap.os.path.join(cache, name)
                if _eai_platform_bootstrap.os.path.isdir(path):
                    roots.append(path)
        except OSError:
            pass
    if repo_root:
        roots.append(_eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.abspath(repo_root), "assets"))
    seen = set()
    out: _eai_platform_bootstrap.List[str] = []
    for path in roots:
        if path in seen or not _eai_platform_bootstrap.os.path.isdir(path):
            continue
        seen.add(path)
        out.append(path)
    return out


def molmospaces_benchmark_ready(path: str) -> bool:
    if not path or not _eai_platform_bootstrap.os.path.isdir(path):
        return False
    if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(path, "benchmark.json")):
        return True
    try:
        for name in _eai_platform_bootstrap.os.listdir(path):
            if name.startswith("house_") and _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.join(path, name)):
                return True
    except OSError:
        return False
    return False


def resolve_molmospaces_benchmark(rel: str, repo_root: str = "") -> str:
    """Resolve a catalog-relative benchmark dir. Missing dirs still get a preferred path."""
    raw = (rel or "").strip()
    if _eai_platform_bootstrap.os.path.isabs(raw):
        return _eai_platform_bootstrap.os.path.abspath(raw)
    rel = raw.lstrip("/")
    rel_under_assets = rel[len("assets/") :] if rel.startswith("assets/") else rel
    for root in molmospaces_asset_roots(repo_root):
        for cand in (
            _eai_platform_bootstrap.os.path.join(root, rel_under_assets),
            _eai_platform_bootstrap.os.path.join(root, rel),
        ):
            if _eai_platform_bootstrap.os.path.isdir(cand):
                return _eai_platform_bootstrap.os.path.abspath(cand)
    roots = molmospaces_asset_roots(repo_root)
    base = roots[0] if roots else _eai_platform_bootstrap.os.path.join(repo_root or ".", "assets")
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.join(base, rel_under_assets))


def stage_psi_benchmark_json(json_path: str) -> str:
    """Return a directory containing ``benchmark.json`` for eval_main --benchmark_dir.

    ``psi_molmo_spaces`` often ships episode lists as ``benchmark_*.json`` files;
    MolmoSpaces eval expects a directory with a file named ``benchmark.json``.
    """
    src = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(json_path or ""))
    if not src or not _eai_platform_bootstrap.os.path.isfile(src):
        return src
    base = _eai_platform_bootstrap.os.path.basename(src)
    parent = _eai_platform_bootstrap.os.path.dirname(src)
    if base == "benchmark.json":
        return parent
    stem = _eai_platform_bootstrap.os.path.splitext(base)[0]
    stage = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "psi_molmospaces_bench", stem)
    _eai_platform_bootstrap.os.makedirs(stage, exist_ok=True)
    dest = _eai_platform_bootstrap.os.path.join(stage, "benchmark.json")
    try:
        if _eai_platform_bootstrap.os.path.islink(dest) or _eai_platform_bootstrap.os.path.isfile(dest):
            if _eai_platform_bootstrap.os.path.islink(dest) and _eai_platform_bootstrap.os.path.realpath(dest) == _eai_platform_bootstrap.os.path.realpath(src):
                return stage
            _eai_platform_bootstrap.os.remove(dest)
        _eai_platform_bootstrap.os.symlink(src, dest)
    except OSError:
        import shutil

        shutil.copy2(src, dest)
    return stage


def _psi_benchmark_label(stem: str, episodes: list) -> str:
    """Human label for a Psi BK file (prefer unique language task description)."""
    from collections import Counter

    descs: Counter = Counter()
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        lang = ep.get("language") or {}
        if not isinstance(lang, dict):
            continue
        desc = lang.get("task_description") or lang.get("instruction") or ""
        if isinstance(desc, str) and desc.strip():
            descs[desc.strip().rstrip(".")] += 1
    n = len(episodes)
    if len(descs) == 1:
        only = next(iter(descs))
        return f"[Psi] {only} ({n} ep)"
    if stem == "benchmark":
        return f"[Psi] psi_molmo_spaces ({n} ep)"
    return f"[Psi] {stem} ({n} ep)"


def list_psi_molmospaces_local_tasks(repo_root: str = "") -> _eai_platform_bootstrap.List[dict]:
    """Local Psi-MolmoSpaces benches under ``psi_molmo_spaces/`` (init_molmospaces).

    On ``init_molmospaces`` the canonical bench is ``psi_molmo_spaces/benchmark.json``.
    Extra JSON files (e.g. outputs of ``filter_benchmark_by_task.py``) are also listed.
    """
    tasks: _eai_platform_bootstrap.List[dict] = []
    root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(repo_root or ""))
    if not root:
        return tasks
    psi_dir = _eai_platform_bootstrap.os.path.join(root, "psi_molmo_spaces")
    if not _eai_platform_bootstrap.os.path.isdir(psi_dir):
        return tasks
    skip_names = {
        "robot_base_pose_cache.json",
        "ik_feasible_samples.txt",
    }
    try:
        names = sorted(_eai_platform_bootstrap.os.listdir(psi_dir))
    except OSError:
        return tasks
    # Prefer the canonical benchmark.json first.
    ordered = []
    if "benchmark.json" in names:
        ordered.append("benchmark.json")
    ordered.extend(n for n in names if n != "benchmark.json")
    for name in ordered:
        if name in skip_names or not name.endswith(".json"):
            continue
        if "cache" in name.lower():
            continue
        full = _eai_platform_bootstrap.os.path.join(psi_dir, name)
        if not _eai_platform_bootstrap.os.path.isfile(full):
            continue
        try:
            import json

            with open(full, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list) or not data:
                continue
            if not isinstance(data[0], dict):
                continue
        except Exception:
            continue
        path = stage_psi_benchmark_json(full)
        stem = _eai_platform_bootstrap.os.path.splitext(name)[0]
        tasks.append(
            {
                "id": f"psi:{stem}",
                "label": _psi_benchmark_label(stem, data),
                "path": path,
                "ready": True,
                "extra": (),
            }
        )
    return tasks


_MOLMOSPACES_TASK_CACHE: _eai_platform_bootstrap.Dict[_eai_platform_bootstrap.Tuple[str, bool], _eai_platform_bootstrap.Tuple[float, _eai_platform_bootstrap.List[dict]]] = {}


_MOLMOSPACES_TASK_TTL_S = 45.0


def list_molmospaces_eval_tasks(
    repo_root: str = "", *, psi_only: bool = False
) -> _eai_platform_bootstrap.List[dict]:
    """Official MS/MB benchmarks, plus any local benchmark.json not in the catalog.

    When ``psi_only=True`` (Psi-MolmoSpaces backend), return only
    ``psi_molmo_spaces`` benches — matching the init_molmospaces workflow.
    """
    root_key = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(repo_root or ""))
    cache_key = (root_key, bool(psi_only))
    now = _eai_platform_bootstrap.time.time()
    hit = _MOLMOSPACES_TASK_CACHE.get(cache_key)
    if hit is not None and (now - hit[0]) < _MOLMOSPACES_TASK_TTL_S:
        return hit[1]

    psi_tasks = list_psi_molmospaces_local_tasks(repo_root)
    if psi_only:
        _MOLMOSPACES_TASK_CACHE[cache_key] = (now, psi_tasks)
        return psi_tasks

    tasks: _eai_platform_bootstrap.List[dict] = []
    known_paths = set()
    for task_id, label, rel, extra in _eai_configuration_tabs.MOLMOSPACES_EVAL_TASKS:
        path = resolve_molmospaces_benchmark(rel, repo_root)
        ready = molmospaces_benchmark_ready(path)
        known_paths.add(_eai_platform_bootstrap.os.path.abspath(path))
        tasks.append(
            {
                "id": task_id,
                "label": label,
                "path": path,
                "ready": ready,
                "extra": tuple(extra),
            }
        )
    for root in molmospaces_asset_roots(repo_root):
        bench_root = _eai_platform_bootstrap.os.path.join(root, "benchmarks")
        if not _eai_platform_bootstrap.os.path.isdir(bench_root):
            continue
        for dirpath, dirnames, filenames in _eai_platform_bootstrap.os.walk(bench_root):
            # Skip episode trees (house_*) and VCS junk — they dominate walk time.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "__pycache__"} and not d.startswith("house_")
            ]
            if "benchmark.json" not in filenames:
                continue
            full = _eai_platform_bootstrap.os.path.abspath(dirpath)
            # Found a bench leaf; do not descend into its scene payloads.
            dirnames.clear()
            if full in known_paths:
                continue
            known_paths.add(full)
            name = _eai_platform_bootstrap.os.path.basename(full)
            tasks.append(
                {
                    "id": "local:" + name,
                    "label": name,
                    "path": full,
                    "ready": True,
                    "extra": (),
                }
            )
    if psi_tasks:
        tasks = psi_tasks + tasks
    _MOLMOSPACES_TASK_CACHE[cache_key] = (now, tasks)
    return tasks


try:
    from psi_remote_policy_runtime import (  # noqa: E402
        PSI_NATIVE6D_EVAL_ID,
        PSI_NATIVE6D_PORT_DEFAULT,
        PSI_POLICY_EVAL_ID,
        PSI_POLICY_PORT_DEFAULT as PSI_REMOTE_POLICY_PORT_DEFAULT,
        WAM_PSI_R1_EVAL_ID,
        WAM_PSI_R1_PORT_DEFAULT,
        build_psi_remote_server_argv,
        default_port_for_mode as psi_remote_default_port,
        is_psi_native6d_policy_id,
        is_psi_policy_id,
        is_wam_psi_r1_policy_id,
        needs_psi_remote_deploy,
        probe_psi_remote_server,
        psi_remote_mode_for_policy,
        resolve_checkpoint_path as resolve_psi_remote_checkpoint,
        resolve_default_wam_finetune_checkpoint,
        resolve_psi_policy_python as resolve_psi_remote_python,
        resolve_psi_policy_root as resolve_psi_remote_policy_root,
        resolve_wan22_dir,
        wam82_server_needs_restart,
    )
except ImportError:  # pragma: no cover
    PSI_NATIVE6D_EVAL_ID = (
        "molmo_spaces.evaluation.configs.psi_native6d_eval_config:"
        "PsiNative6DPolicyEvalConfig"
    )
    PSI_POLICY_EVAL_ID = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PsiPolicyEvalConfig"
    )
    WAM_PSI_R1_EVAL_ID = (
        "molmo_spaces.configs.wam_psi_r1_config:WamPsiR1RemotePickEvalConfig"
    )
    PSI_REMOTE_POLICY_PORT_DEFAULT = 8080
    PSI_NATIVE6D_PORT_DEFAULT = 8081
    WAM_PSI_R1_PORT_DEFAULT = 9317

    def is_psi_policy_id(policy_id: str) -> bool:  # type: ignore[misc]
        return "PsiPolicyEvalConfig" in (policy_id or "")

    def is_psi_native6d_policy_id(policy_id: str) -> bool:  # type: ignore[misc]
        return "PsiNative6DPolicyEvalConfig" in (policy_id or "")

    def is_wam_psi_r1_policy_id(policy_id: str) -> bool:  # type: ignore[misc]
        return "WamPsiR1RemotePickEvalConfig" in (policy_id or "")

    def needs_psi_remote_deploy(policy_id: str) -> bool:  # type: ignore[misc]
        return (
            is_psi_policy_id(policy_id)
            or is_psi_native6d_policy_id(policy_id)
            or is_wam_psi_r1_policy_id(policy_id)
        )

    def psi_remote_mode_for_policy(policy_id: str) -> str:  # type: ignore[misc]
        if is_wam_psi_r1_policy_id(policy_id):
            return "wam82"
        if is_psi_native6d_policy_id(policy_id):
            return "native6d"
        if is_psi_policy_id(policy_id):
            return "psi_policy"
        return ""

    def psi_remote_default_port(mode: str) -> int:  # type: ignore[misc]
        if mode == "wam82":
            return WAM_PSI_R1_PORT_DEFAULT
        if mode == "native6d":
            return PSI_NATIVE6D_PORT_DEFAULT
        return PSI_REMOTE_POLICY_PORT_DEFAULT

    def build_psi_remote_server_argv(**_kwargs):  # type: ignore[misc]
        raise RuntimeError("psi_remote_policy_runtime 未找到")

    def probe_psi_remote_server(*_a, **_k):  # type: ignore[misc]
        return False, "psi_remote_policy_runtime 未找到"

    def resolve_psi_remote_checkpoint(path: str) -> str:  # type: ignore[misc]
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path or ""))

    def resolve_default_wam_finetune_checkpoint(user: str = "") -> str:  # type: ignore[misc]
        return (
            user
            or _eai_platform_bootstrap.os.environ.get("PSI_REMOTE_CKPT", "").strip()
            or (
                "/share_data_prj/maobo/wam_train_eval/train_runs/"
                "pick_apple_tomato_856_native6d_quantile_songzhe_repro_wam_finetune_4gpu/"
                "checkpoints/policy/topk/step=00016000-train_loss=0.349923.pt"
            )
        )

    def resolve_psi_remote_python(repo_dir: str = "") -> str:  # type: ignore[misc]
        return "python3"

    def resolve_psi_remote_policy_root(user: str = "") -> str:  # type: ignore[misc]
        return ""

    def resolve_wan22_dir(user: str = "") -> str:  # type: ignore[misc]
        return ""

    def wam82_server_needs_restart(port: int) -> bool:  # type: ignore[misc]
        return False


def list_isaaclab_arena_environments(root: str) -> _eai_platform_bootstrap.List[str]:
    """CLI names of @register_environment classes under isaaclab_arena_environments/."""
    env_dir = _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.abspath(root or ""), "isaaclab_arena_environments")
    if not _eai_platform_bootstrap.os.path.isdir(env_dir):
        return []
    skip = {"cli.py", "example_environment_base.py", "__init__.py"}
    names: _eai_platform_bootstrap.List[str] = []
    try:
        files = sorted(_eai_platform_bootstrap.os.listdir(env_dir))
    except OSError:
        return []
    for fname in files:
        if not fname.endswith(".py") or fname in skip:
            continue
        path = _eai_platform_bootstrap.os.path.join(env_dir, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        if "@register_environment" not in text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("name"):
                continue
            if "=" not in stripped:
                continue
            quote = '"' if '"' in stripped else "'" if "'" in stripped else ""
            if not quote:
                continue
            parts = stripped.split(quote)
            if len(parts) >= 2 and parts[1]:
                names.append(parts[1])
                break
    return names


def resolve_isaaclab_arena_python() -> str:
    for cand in _eai_configuration_tabs.ISAACLAB_ARENA_PYTHON_CANDIDATES:
        if cand and _eai_platform_bootstrap.os.path.isfile(cand):
            return _eai_platform_bootstrap.os.path.abspath(cand)
    which = _eai_platform_bootstrap.shutil.which("python3") or "python3"
    return which


def _path_without_markers(value: str, markers: _eai_platform_bootstrap.Tuple[str, ...]) -> str:
    kept: _eai_platform_bootstrap.List[str] = []
    for part in (value or "").split(_eai_platform_bootstrap.os.pathsep):
        if not part:
            continue
        if any(marker in part for marker in markers):
            continue
        kept.append(part)
    return _eai_platform_bootstrap.os.pathsep.join(kept)


def isaaclab_arena_child_env(
    root: str, python_bin: str, cuda_visible: str = ""
) -> _eai_platform_bootstrap.Dict[str, str]:
    """Env for Arena's Python. Drop the GUI's ROS Humble 3.11 site-packages."""
    prefix = _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.abspath(python_bin)))
    markers = ("ros-humble", "/opt/ros/")
    path = _path_without_markers(_eai_platform_bootstrap.os.environ.get("PATH", ""), markers)
    bin_dir = _eai_platform_bootstrap.os.path.join(prefix, "bin")
    if path:
        path = bin_dir + _eai_platform_bootstrap.os.pathsep + path
    else:
        path = bin_dir
    ld = _path_without_markers(_eai_platform_bootstrap.os.environ.get("LD_LIBRARY_PATH", ""), markers)
    lib_dir = _eai_platform_bootstrap.os.path.join(prefix, "lib")
    if _eai_platform_bootstrap.os.path.isdir(lib_dir):
        ld = lib_dir + (_eai_platform_bootstrap.os.pathsep + ld if ld else "")
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _eai_platform_bootstrap.os.path.abspath(root),
        "PATH": path,
        "LD_LIBRARY_PATH": ld,
        "CONDA_PREFIX": prefix,
        "CONDA_DEFAULT_ENV": _eai_platform_bootstrap.os.path.basename(prefix),
        "AMENT_PREFIX_PATH": _path_without_markers(
            _eai_platform_bootstrap.os.environ.get("AMENT_PREFIX_PATH", ""), markers
        ),
        "ISAACLAB_ARENA_ROOT": _eai_platform_bootstrap.os.path.abspath(root),
    }
    if cuda_visible != "":
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    return env


def list_eai_mujoco_model_xmls() -> _eai_platform_bootstrap.List[str]:
    """eai 仓库本地 MJCF（绝对路径），优先 psi_r1 / assets。"""
    out: _eai_platform_bootstrap.List[str] = []
    candidates = (
        _eai_configuration_models.PSI_R1_MODEL_XML,
    )
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path):
            out.append(_eai_platform_bootstrap.os.path.abspath(path))
    # Other MuJoCo MJCF under assets (skip ROS package.xml etc.)
    assets_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "assets")
    if _eai_platform_bootstrap.os.path.isdir(assets_dir):
        try:
            for dirpath, dirnames, filenames in _eai_platform_bootstrap.os.walk(assets_dir):
                # skip bulky unrelated trees
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in {".git", "__pycache__", "meshes"}
                ]
                for fname in filenames:
                    if not fname.endswith(".xml"):
                        continue
                    if fname in {"package.xml", "CMakeLists.xml"}:
                        continue
                    full = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.join(dirpath, fname))
                    if full in out:
                        continue
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                            head = fh.read(256)
                        if "<mujoco" not in head.lower():
                            continue
                    except OSError:
                        continue
                    out.append(full)
        except OSError:
            pass
    return out


def list_mujoco_model_xmls(mujoco_root: str = "", *, limit: int = 400) -> _eai_platform_bootstrap.List[str]:
    """扫描 eai 本地 + mujoco/model 下 *.xml。

    eai 本地模型返回绝对路径；官方示例返回相对 model/ 的路径。
    """
    root = resolve_mujoco_root(mujoco_root)
    model_dir = _eai_platform_bootstrap.os.path.join(root, "model")
    out: _eai_platform_bootstrap.List[str] = []
    # Local project models first (selectable in UI).
    for path in list_eai_mujoco_model_xmls():
        out.append(path)
        if len(out) >= int(limit):
            return out
    if _eai_platform_bootstrap.os.path.isdir(model_dir):
        try:
            for dirpath, _dirnames, filenames in _eai_platform_bootstrap.os.walk(model_dir):
                for fname in filenames:
                    if not fname.endswith(".xml"):
                        continue
                    full = _eai_platform_bootstrap.os.path.join(dirpath, fname)
                    rel = _eai_platform_bootstrap.os.path.relpath(full, model_dir).replace(_eai_platform_bootstrap.os.sep, "/")
                    out.append(rel)
                    if len(out) >= int(limit):
                        break
                if len(out) >= int(limit):
                    break
        except OSError:
            pass
    preferred = (
        _eai_configuration_models.PSI_R1_MODEL_XML,
        "humanoid/humanoid.xml",
        "welcome/welcome.xml",
        "car/car.xml",
        "cube/cube_3x3x3.xml",
        "mug/mug.xml",
    )
    preferred_norm = [_eai_platform_bootstrap.os.path.abspath(p) if _eai_platform_bootstrap.os.path.isabs(p) else p for p in preferred]

    def _rank_key(p: str):
        abs_p = _eai_platform_bootstrap.os.path.abspath(p) if _eai_platform_bootstrap.os.path.isabs(p) else p
        try:
            pref_i = preferred_norm.index(abs_p)
        except ValueError:
            try:
                pref_i = preferred_norm.index(p)
            except ValueError:
                pref_i = 999
        return (0 if _eai_platform_bootstrap.os.path.isabs(p) else 1, pref_i, p.count("/"), p.lower())

    return sorted(out, key=_rank_key)


def count_visible_cuda_devices(cuda_devices: str = "") -> int:
    """根据 CUDA_VISIBLE_DEVICES / nvidia-smi 估计可见 GPU 数。"""
    raw = (cuda_devices or "").strip()
    if not raw:
        raw = (_eai_platform_bootstrap.os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if parts:
            return len(parts)
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return 0


def astra_url_prefers_external_browser(url: str) -> bool:
    """含视频的 Gallery 页：Qt5 WebEngine 常播不了 H.264，优先系统浏览器。"""
    u = (url or "").lower()
    return ("gallery.html" in u) or ("anonymous-report-421.github.io" in u)

