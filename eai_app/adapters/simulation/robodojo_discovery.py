"""adapters.simulation.robodojo discovery — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def list_robodojo_tasks() -> _eai_platform_bootstrap.List[str]:
    """从 RoboDojo task config 列出可评测任务名。"""
    cfg_dir = _eai_configuration_policies.ROBODOJO_TASK_CONFIG_DIR
    if not _eai_platform_bootstrap.os.path.isdir(cfg_dir):
        return []
    names: _eai_platform_bootstrap.List[str] = []
    try:
        for fn in _eai_platform_bootstrap.os.listdir(cfg_dir):
            if not fn.endswith(".yml") or fn.startswith("_"):
                continue
            names.append(fn[: -len(".yml")])
    except OSError:
        return []
    return sorted(names)


def list_robodojo_policies(
    robodojo_root: str = "",
) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    """列出 XPolicyLab/policy 下可部署策略：(显示名, 相对 RoboDojo 根的 policy_dir)。

    判定：目录内存在 deploy.py（与 robodojo.sh server/client 一致）。
    """
    root = _eai_platform_bootstrap.os.path.abspath(
        _eai_platform_bootstrap.os.path.expanduser((robodojo_root or "").strip() or _eai_configuration_policies.ROBODOJO_ROOT_DEFAULT)
    )
    policy_root = _eai_platform_bootstrap.os.path.join(root, "XPolicyLab", "policy")
    if not _eai_platform_bootstrap.os.path.isdir(policy_root):
        return []
    out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = []
    try:
        for name in sorted(_eai_platform_bootstrap.os.listdir(policy_root), key=str.lower):
            if name.startswith("_") or name.startswith("."):
                continue
            d = _eai_platform_bootstrap.os.path.join(policy_root, name)
            if not _eai_platform_bootstrap.os.path.isdir(d):
                continue
            if not _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(d, "deploy.py")):
                continue
            rel = f"XPolicyLab/policy/{name}"
            out.append((name, rel))
    except OSError:
        return []
    return out


def _pid_listening_on_tcp_port(port: int) -> _eai_platform_bootstrap.Optional[int]:
    """Best-effort PID of a process listening on TCP port (Linux ss/fuser)."""
    port = int(port)
    try:
        out = _eai_platform_bootstrap.subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        for ln in out.splitlines():
            if "LISTEN" not in ln:
                continue
            m = _eai_platform_bootstrap.re.search(r"pid=(\d+)", ln)
            if m:
                return int(m.group(1))
    except (_eai_platform_bootstrap.subprocess.CalledProcessError, FileNotFoundError, _eai_platform_bootstrap.subprocess.TimeoutExpired):
        pass
    try:
        out = _eai_platform_bootstrap.subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            stderr=_eai_platform_bootstrap.subprocess.STDOUT,
            text=True,
            timeout=2,
        )
        for tok in out.replace("\n", " ").split():
            if tok.isdigit():
                return int(tok)
    except (_eai_platform_bootstrap.subprocess.CalledProcessError, FileNotFoundError, _eai_platform_bootstrap.subprocess.TimeoutExpired):
        pass
    return None


def inspect_robodojo_policy_listener(
    host: str = "",
    port: int = 0,
) -> dict:
    """Inspect whatever is on the RoboDojo policy deploy port.

    Returns keys: listening, pid, cmdline, policy_name, ckpt_name, action_type,
    is_policy_server.
    """
    host = (host or _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT).strip() or "127.0.0.1"
    port = int(port or _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT)
    info: dict = {
        "host": host,
        "port": port,
        "listening": False,
        "pid": None,
        "cmdline": "",
        "policy_name": "",
        "ckpt_name": "",
        "action_type": "",
        "is_policy_server": False,
    }
    try:
        info["listening"] = bool(_eai_configuration_models.port_listening(host, port))
    except Exception:
        info["listening"] = False
    if not info["listening"]:
        return info
    pid = _pid_listening_on_tcp_port(port)
    info["pid"] = pid
    cmdline = ""
    if pid and pid > 1:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            cmdline = ""
    info["cmdline"] = cmdline
    # setup_policy_server.py … policy_name=starVLA ckpt_name=hf_qwenpi_v3
    # or robodojo.sh server --policy-dir …/starVLA --ckpt …
    m_pol = _eai_platform_bootstrap.re.search(r"policy_name=([^\s]+)", cmdline)
    if not m_pol:
        m_pol = _eai_platform_bootstrap.re.search(r"XPolicyLab/policy/([^\s/]+)", cmdline)
    if m_pol:
        info["policy_name"] = m_pol.group(1).strip().strip("'\"")
    m_ckpt = _eai_platform_bootstrap.re.search(r"ckpt_name=([^\s]+)", cmdline)
    if not m_ckpt:
        m_ckpt = _eai_platform_bootstrap.re.search(r"--ckpt[=\s]+([^\s]+)", cmdline)
    if m_ckpt:
        info["ckpt_name"] = m_ckpt.group(1).strip().strip("'\"")
    m_act = _eai_platform_bootstrap.re.search(r"action_type=([^\s]+)", cmdline)
    if not m_act:
        m_act = _eai_platform_bootstrap.re.search(r"--action-type[=\s]+([^\s]+)", cmdline)
    if m_act:
        info["action_type"] = m_act.group(1).strip().strip("'\"")
    markers = (
        "setup_policy_server.py",
        "setup_eval_policy_server.sh",
        "robodojo.sh server",
        "XPolicyLab/policy/",
        "model_server",
        "server_policy.py",
    )
    info["is_policy_server"] = any(m in cmdline for m in markers) or bool(
        info["policy_name"]
    )
    return info


def robodojo_policy_matches_listener(
    info: dict,
    *,
    policy_dir: str,
    ckpt: str = "",
    action_type: str = "",
) -> bool:
    """True if listener looks like the requested policy (and ckpt/action when known)."""
    if not info or not info.get("listening") or not info.get("is_policy_server"):
        return False
    want = _eai_platform_bootstrap.os.path.basename((policy_dir or "").rstrip("/"))
    got = str(info.get("policy_name") or "").strip()
    if want and got and want.lower() != got.lower():
        return False
    if want and not got:
        # cmdline may omit policy_name; require path fragment
        cmd = str(info.get("cmdline") or "")
        if want not in cmd and f"policy/{want}" not in cmd:
            return False
    want_ckpt = (ckpt or "").strip()
    got_ckpt = str(info.get("ckpt_name") or "").strip()
    if want_ckpt and got_ckpt and want_ckpt != got_ckpt:
        # allow soft match: hf_qwenpi_v3 vs path ending
        if want_ckpt not in got_ckpt and got_ckpt not in want_ckpt:
            return False
    want_act = (action_type or "").strip().lower()
    got_act = str(info.get("action_type") or "").strip().lower()
    if want_act and got_act and want_act != got_act:
        return False
    return True

