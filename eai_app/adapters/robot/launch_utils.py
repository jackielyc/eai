"""adapters.robot.launch utils — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_scripts_pack_dir() -> _eai_platform_bootstrap.Optional[str]:
    env_dir = _eai_platform_bootstrap.os.environ.get("A2D_SCRIPTS_DIR", "").strip()
    if env_dir and _eai_platform_bootstrap.os.path.isdir(env_dir):
        return _eai_platform_bootstrap.os.path.abspath(env_dir)
    here = str(PROJECT_ROOT)
    psibot_home = _eai_platform_bootstrap.os.environ.get("PSIBOT_HOME", "/home/psibot").strip() or "/home/psibot"
    candidates = [
        _eai_platform_bootstrap.os.path.join(psibot_home, "workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"),
        _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.join(here, "..", "install", "scripts_pack", "share", "scripts_pack", "scripts")),
        _eai_platform_bootstrap.os.path.join(psibot_home, "workspace_liyichao/a2d-tele/install/scripts_pack/share/scripts_pack/scripts"),
        _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"),
        _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/a2d-tele/install/scripts_pack/share/scripts_pack/scripts"),
        "/opt/psi/rt/a2d-tele/install/scripts_pack/share/scripts_pack/scripts",
    ]
    stack_script = _eai_configuration_policies.STACK_SERVICES_SCRIPT
    resolved_candidates: _eai_platform_bootstrap.List[str] = []
    for path in candidates:
        resolved = _eai_platform_bootstrap.os.path.abspath(path)
        if resolved in resolved_candidates:
            continue
        resolved_candidates.append(resolved)
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(resolved, stack_script)):
            return resolved
    for resolved in resolved_candidates:
        if _eai_platform_bootstrap.os.path.isdir(resolved):
            return resolved
    return None


def resolve_a2d_workspace_dir() -> str:
    """与 GUI 一致，优先使用含 install/setup.bash 的 a2d-tele 工作区。

    可用 A2D_WORKSPACE / PSIBOT_HOME 覆盖。若均未安装，仍返回常见默认路径
    （供 UI 展示）；真正启动前须再校验 install 是否存在。
    """
    env = _eai_platform_bootstrap.os.environ.get("A2D_WORKSPACE", "").strip()
    psibot_home = _eai_platform_bootstrap.os.environ.get("PSIBOT_HOME", "/home/psibot").strip() or "/home/psibot"
    candidates: _eai_platform_bootstrap.List[str] = []
    if env:
        candidates.append(_eai_platform_bootstrap.os.path.expanduser(env))
    candidates.extend(
        [
            "/opt/psi/rt/a2d-tele",
            _eai_platform_bootstrap.os.path.join(psibot_home, "workspace_liyichao", "a2d-tele"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/a2d-tele"),
            _eai_platform_bootstrap.os.path.join(psibot_home, "workspace_liyichao"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao"),
        ]
    )
    seen = set()  # type: ignore[var-annotated]
    ordered: _eai_platform_bootstrap.List[str] = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    for candidate in ordered:
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(candidate, "install", "setup.bash")):
            return candidate
    for candidate in ordered:
        if _eai_platform_bootstrap.os.path.isdir(candidate):
            return candidate
    return ordered[0] if ordered else "/opt/psi/rt/a2d-tele"


def normalize_real_robot_backend(backend: str) -> str:
    key = (backend or "").strip().lower().replace("-", "_")
    if key in ("r1", "psi_r1", "ruiyan_r1"):
        return _eai_configuration_policies.REAL_ROBOT_BACKEND_R1
    if key in ("franka", "franka_panda", "panda", "frankapanda", "libfranka"):
        return _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA
    return _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D


def real_robot_backend_label(backend: str) -> str:
    key = normalize_real_robot_backend(backend)
    return _eai_configuration_policies.REAL_ROBOT_BACKEND_LABELS.get(key, key.upper())


def _resolve_tele_scripts_pack_dir(
    *,
    env_scripts: str,
    workspace_names: _eai_platform_bootstrap.Sequence[str],
    opt_roots: _eai_platform_bootstrap.Sequence[str],
) -> _eai_platform_bootstrap.Optional[str]:
    """通用：按 ENV / 工作区名 /opt 路径解析 scripts_pack。"""
    env_dir = _eai_platform_bootstrap.os.environ.get(env_scripts, "").strip()
    if env_dir and _eai_platform_bootstrap.os.path.isdir(env_dir):
        return _eai_platform_bootstrap.os.path.abspath(env_dir)
    here = str(PROJECT_ROOT)
    psibot_home = _eai_platform_bootstrap.os.environ.get("PSIBOT_HOME", "/home/psibot").strip() or "/home/psibot"
    candidates: _eai_platform_bootstrap.List[str] = []
    for ws_name in workspace_names:
        candidates.append(
            _eai_platform_bootstrap.os.path.join(
                psibot_home,
                f"workspace_liyichao/{ws_name}/install/scripts_pack/share/scripts_pack/scripts",
            )
        )
        candidates.append(
            _eai_platform_bootstrap.os.path.expanduser(
                f"~/workspace_liyichao/{ws_name}/install/scripts_pack/share/scripts_pack/scripts"
            )
        )
    candidates.append(
        _eai_platform_bootstrap.os.path.join(
            psibot_home, "workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"
        )
    )
    candidates.append(
        _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.join(here, "..", "install", "scripts_pack", "share", "scripts_pack", "scripts")
        )
    )
    for root in opt_roots:
        candidates.append(
            _eai_platform_bootstrap.os.path.join(root, "install/scripts_pack/share/scripts_pack/scripts")
        )
    stack_script = _eai_configuration_policies.STACK_SERVICES_SCRIPT
    resolved_candidates: _eai_platform_bootstrap.List[str] = []
    for path in candidates:
        resolved = _eai_platform_bootstrap.os.path.abspath(path)
        if resolved in resolved_candidates:
            continue
        resolved_candidates.append(resolved)
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(resolved, stack_script)):
            return resolved
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(resolved, _eai_configuration_policies.ROBOT_SERVICE_SCRIPT)):
            return resolved
    for resolved in resolved_candidates:
        if _eai_platform_bootstrap.os.path.isdir(resolved):
            return resolved
    return None


def _resolve_tele_workspace_dir(
    *,
    env_workspace: str,
    workspace_names: _eai_platform_bootstrap.Sequence[str],
    opt_roots: _eai_platform_bootstrap.Sequence[str],
    default_root: str,
) -> str:
    env_ws = _eai_platform_bootstrap.os.environ.get(env_workspace, "").strip()
    if env_ws and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(env_ws, "install/setup.bash")):
        return _eai_platform_bootstrap.os.path.abspath(env_ws)
    psibot_home = _eai_platform_bootstrap.os.environ.get("PSIBOT_HOME", "/home/psibot").strip() or "/home/psibot"
    candidates: _eai_platform_bootstrap.List[str] = list(opt_roots)
    for ws_name in workspace_names:
        candidates.append(_eai_platform_bootstrap.os.path.expanduser(f"~/workspace_liyichao/{ws_name}"))
        candidates.append(_eai_platform_bootstrap.os.path.join(psibot_home, f"workspace_liyichao/{ws_name}"))
    for candidate in candidates:
        if _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(candidate, "install/setup.bash")):
            return candidate
    return default_root


def resolve_r1_scripts_pack_dir() -> _eai_platform_bootstrap.Optional[str]:
    """R1 真机 scripts_pack；可通过 R1_SCRIPTS_DIR 覆盖。"""
    return _resolve_tele_scripts_pack_dir(
        env_scripts="R1_SCRIPTS_DIR",
        workspace_names=("r1-tele",),
        opt_roots=("/opt/psi/rt/r1-tele",),
    )


def resolve_r1_workspace_dir() -> str:
    """R1 真机工作区；优先 r1-tele install。"""
    return _resolve_tele_workspace_dir(
        env_workspace="R1_WORKSPACE_DIR",
        workspace_names=("r1-tele",),
        opt_roots=("/opt/psi/rt/r1-tele",),
        default_root="/opt/psi/rt/r1-tele",
    )


def resolve_franka_scripts_pack_dir() -> _eai_platform_bootstrap.Optional[str]:
    """Franka 真机 scripts_pack；可通过 FRANKA_SCRIPTS_DIR 覆盖。"""
    return _resolve_tele_scripts_pack_dir(
        env_scripts="FRANKA_SCRIPTS_DIR",
        workspace_names=("franka-tele", "franka_tele", "panda-tele"),
        opt_roots=(
            "/opt/psi/rt/franka-tele",
            "/opt/psi/rt/franka_tele",
            "/opt/psi/rt/panda-tele",
        ),
    )


def resolve_franka_workspace_dir() -> str:
    """Franka 真机工作区；优先 franka-tele install。"""
    return _resolve_tele_workspace_dir(
        env_workspace="FRANKA_WORKSPACE_DIR",
        workspace_names=("franka-tele", "franka_tele", "panda-tele"),
        opt_roots=(
            "/opt/psi/rt/franka-tele",
            "/opt/psi/rt/franka_tele",
            "/opt/psi/rt/panda-tele",
        ),
        default_root="/opt/psi/rt/franka-tele",
    )


def resolve_real_robot_scripts_dir(backend: str = _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D) -> _eai_platform_bootstrap.Optional[str]:
    key = normalize_real_robot_backend(backend)
    if key == _eai_configuration_policies.REAL_ROBOT_BACKEND_R1:
        return resolve_r1_scripts_pack_dir()
    if key == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA:
        return resolve_franka_scripts_pack_dir()
    return resolve_scripts_pack_dir()


def resolve_real_robot_workspace_dir(backend: str = _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D) -> str:
    key = normalize_real_robot_backend(backend)
    if key == _eai_configuration_policies.REAL_ROBOT_BACKEND_R1:
        return resolve_r1_workspace_dir()
    if key == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA:
        return resolve_franka_workspace_dir()
    return resolve_a2d_workspace_dir()


def build_real_robot_ros_shell_prefix(backend: str = _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D) -> str:
    """按真机后端拼 source + DDS 环境（A2D / R1 / Franka 工作区不同）。"""
    key = normalize_real_robot_backend(backend)
    parts = ["set -e", "source /opt/ros/humble/setup.bash"]
    if key == _eai_configuration_policies.REAL_ROBOT_BACKEND_R1:
        setups = (
            "/opt/psi/rt/r1-tele/install/setup.bash",
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/r1-tele/install/setup.bash"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/install/setup.bash"),
        )
        scripts_env = "R1_SCRIPTS_DIR"
    elif key == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA:
        setups = (
            "/opt/psi/rt/franka-tele/install/setup.bash",
            "/opt/psi/rt/franka_tele/install/setup.bash",
            "/opt/psi/rt/panda-tele/install/setup.bash",
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/franka-tele/install/setup.bash"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/franka_tele/install/setup.bash"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/panda-tele/install/setup.bash"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/install/setup.bash"),
        )
        scripts_env = "FRANKA_SCRIPTS_DIR"
    else:
        setups = (
            "/opt/psi/rt/a2d-tele/install/setup.bash",
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/a2d-tele/install/setup.bash"),
            _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/install/setup.bash"),
        )
        scripts_env = "A2D_SCRIPTS_DIR"
    for setup in setups:
        if _eai_platform_bootstrap.os.path.isfile(setup):
            parts.append(f"source {setup}")
    scripts_dir = resolve_real_robot_scripts_dir(key)
    if scripts_dir:
        parts.append(f"export {scripts_env}={scripts_dir}")
    dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    if dds and _eai_platform_bootstrap.os.path.isfile(dds):
        parts.append(f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds}")
    parts.append("export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}")
    return " && ".join(parts)


def real_robot_stack_ready_hint(backend: str) -> str:
    """真机栈脚本是否可解析的提示文案。"""
    key = normalize_real_robot_backend(backend)
    label = real_robot_backend_label(key)
    scripts = resolve_real_robot_scripts_dir(key)
    if scripts and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(scripts, _eai_configuration_policies.ROBOT_SERVICE_SCRIPT)):
        return f"{label} 栈脚本: {scripts}"
    env_name = {
        _eai_configuration_policies.REAL_ROBOT_BACKEND_R1: "R1_SCRIPTS_DIR",
        _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA: "FRANKA_SCRIPTS_DIR",
    }.get(key, "A2D_SCRIPTS_DIR")
    if scripts:
        return f"{label} 已找到 scripts 目录但缺少 {_eai_configuration_policies.ROBOT_SERVICE_SCRIPT}（{scripts}）"
    return f"{label} 未找到栈脚本，请设置 {env_name}"


def default_rrd_dataset_dir() -> str:
    env_dir = _eai_platform_bootstrap.os.environ.get("RRD_DATASET_DIR", "").strip()
    if env_dir and _eai_platform_bootstrap.os.path.isdir(env_dir):
        return env_dir
    for candidate in ("/root/dataset", "/home/psibot/dataset"):
        if _eai_platform_bootstrap.os.path.isdir(candidate):
            return candidate
    return "/home/psibot/dataset"


def dataset_browser_presets() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    """数据集浏览预设：(显示名, 绝对路径)。"""
    items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = [
        ("eai/datasets (.rrd)", _eai_configuration_chat.EAI_DATASETS_DIR),
        ("Bagel 历史", _eai_configuration_chat.BAGEL_HISTORY_DIR),
        ("Bagel 输出缓存", _eai_configuration_models.BAGEL_API_OUTPUT_DIR),
        ("场景图 images/", _eai_configuration_chat.EAI_IMAGES_DIR),
        ("RRD 数据目录", default_rrd_dataset_dir()),
    ]
    out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = []
    seen = set()
    for label, path in items:
        abs_path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path))
        if abs_path in seen:
            continue
        seen.add(abs_path)
        out.append((label, abs_path))
    return out


def embodied_open_datasets_notes_path() -> str:
    """开源具身数据集分析文档路径。"""
    return _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "docs", "embodied-open-datasets-notes.md")


def _html_escape_text(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

