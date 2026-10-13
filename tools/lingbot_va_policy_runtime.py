#!/usr/bin/env python3
"""Resolve / launch LingBot-VA (World-Action Model) across EAI sim backends.

Official adapter lives under RoboDojo XPolicyLab::

  RoboDojo/XPolicyLab/policy/LingBot_VA/

Upstream eval clients:
  - LIBERO:   lingbot_va/evaluation/libero/{launch_server,client}.py
  - RoboTwin: lingbot_va/evaluation/robotwin/{launch_server,client}.sh
  - RoboDojo: deploy.py + launch_wan_va_server.sh (va_server :10001)

Policy id used in EAI UI combo boxes: ``lingbot_va``.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

EAI_DIR = Path(__file__).resolve().parent.parent
_LIY = Path("/share_data/projects/mahjong/share/personal/liyichao")

LINGBOT_VA_POLICY_ID = "lingbot_va"
LINGBOT_VA_POLICY_LABEL = "LingBot-VA (WAM)"

ROBODOJO_ROOT_DEFAULT = _LIY / "RoboDojo"
LINGBOT_VA_POLICY_REL = "XPolicyLab/policy/LingBot_VA"
LINGBOT_VA_ROOT_DEFAULT = ROBODOJO_ROOT_DEFAULT / LINGBOT_VA_POLICY_REL

LINGBOT_VA_ENV_CANDIDATES: Tuple[str, ...] = (
    str(_LIY / "RoboDojo_cache" / "envs" / "lingbot_va"),
    str(_LIY / "miniconda3" / "envs" / "lingbot_va"),
    "/home/psibot/miniconda3/envs/lingbot_va",
    str(_LIY / "RoboDojo_cache" / "envs" / "RoboDojo"),
)

LINGBOT_VA_BASE_MODEL_CANDIDATES: Tuple[str, ...] = (
    str(_LIY / "RoboDojo_cache" / "models" / "lingbot-va-base"),
    str(_LIY / "models" / "lingbot-va-base"),
    str(EAI_DIR / ".cache" / "lingbot_va" / "lingbot-va-base"),
)

LINGBOT_VA_CKPT_CANDIDATES: Tuple[str, ...] = (
    str(LINGBOT_VA_ROOT_DEFAULT / "checkpoints"),
    str(EAI_DIR / ".cache" / "lingbot_va" / "checkpoints"),
    str(_LIY / "models" / "lingbot_va"),
)

# (label, config_id) — config_id is wan_va --config-name or deploy.yml config_name
LINGBOT_VA_CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("LingBot-VA · RoboDojo (robotwin30_train)", "robotwin30_train"),
    ("LingBot-VA · RoboTwin (robotwin)", "robotwin"),
    ("LingBot-VA · LIBERO (libero)", "libero"),
)

# Backend → preferred wan_va config_name
BACKEND_VA_CONFIG: Dict[str, str] = {
    "isaac": "robotwin30_train",
    "libero": "libero",
    "libero_plus": "libero",
    "libero_pro": "libero",
    "libero_mem": "libero",
    "robotwin": "robotwin",
    "molmospaces": "robotwin30_train",
    "arena": "robotwin30_train",
    "mujoco": "robotwin30_train",
    "calvin": "robotwin30_train",
    "simpler": "robotwin30_train",
    "robocasa": "robotwin30_train",
    "robocasa365": "robotwin30_train",
    "maniskill3": "robotwin30_train",
    "maniskill2": "robotwin30_train",
    "metaworld": "robotwin30_train",
    "rlbench": "robotwin30_train",
    "behavior1k": "robotwin30_train",
    "furniturebench": "robotwin30_train",
    "vlabench": "robotwin30_train",
    "genesis": "robotwin30_train",
    "mikasa": "robotwin30_train",
    "robocerebra": "robotwin30_train",
    "robomme": "robotwin30_train",
    "duobench": "robotwin30_train",
    "kinetix": "robotwin30_train",
    "vla_harness": "robotwin30_train",
}

# native_eval: full client+server path in eai
# robodojo_adapter: XPolicyLab deploy.py
# serve_only: start wan_va; client adapter not wired for that bench yet
BACKEND_VA_SUPPORT: Dict[str, str] = {
    "isaac": "robodojo_adapter",
    "libero": "native_eval",
    "libero_plus": "native_eval",
    "libero_pro": "native_eval",
    "libero_mem": "native_eval",
    "robotwin": "native_eval",
    "molmospaces": "serve_only",
    "arena": "serve_only",
    "mujoco": "serve_only",
    "calvin": "serve_only",
    "simpler": "serve_only",
    "robocasa": "serve_only",
    "robocasa365": "serve_only",
    "maniskill3": "serve_only",
    "maniskill2": "serve_only",
    "metaworld": "serve_only",
    "rlbench": "serve_only",
    "behavior1k": "serve_only",
    "furniturebench": "serve_only",
    "vlabench": "serve_only",
    "genesis": "serve_only",
    "mikasa": "serve_only",
    "robocerebra": "serve_only",
    "robomme": "serve_only",
    "duobench": "serve_only",
    "kinetix": "serve_only",
    "vla_harness": "serve_only",
}

DEFAULT_VA_HOST = "127.0.0.1"
DEFAULT_VA_PORT = 10001
LIBERO_VA_PORT_DEFAULT = 29056
ROBOTWIN_VA_PORT_DEFAULT = 29056


@dataclass(frozen=True)
class LingbotVaPaths:
    policy_root: Path
    upstream_root: Path
    base_model: str
    checkpoint: str
    config_name: str
    python_bin: str
    conda_env: str
    source: str


def is_lingbot_va_policy(policy_id_or_dir: str) -> bool:
    raw = (policy_id_or_dir or "").strip()
    if not raw:
        return False
    low = raw.lower().replace("-", "_")
    if low in (LINGBOT_VA_POLICY_ID, "lingbot_va", "lingbotva"):
        return True
    if "lingbot_va" in low or "lingbot-va" in raw.lower():
        return True
    return False


def va_support_level(backend: str) -> str:
    return BACKEND_VA_SUPPORT.get((backend or "").strip().lower(), "serve_only")


def va_config_for_backend(backend: str, override: str = "") -> str:
    if (override or "").strip():
        return override.strip()
    return BACKEND_VA_CONFIG.get((backend or "").strip().lower(), "robotwin30_train")


def resolve_robodojo_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("ROBODOJO_ROOT", ""),
        str(ROBODOJO_ROOT_DEFAULT),
    ):
        p = Path(os.path.expanduser(cand or "")).resolve()
        if (p / LINGBOT_VA_POLICY_REL / "deploy.py").is_file():
            return p
        if (p / "XPolicyLab" / "policy").is_dir():
            return p
    return ROBODOJO_ROOT_DEFAULT


def resolve_lingbot_va_root(raw: str = "", robodojo_root: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("LINGBOT_VA_ROOT", ""),
        os.environ.get("LINGBOT_VA_POLICY_DIR", ""),
    ):
        p = Path(os.path.expanduser(cand or "")).resolve()
        if (p / "deploy.py").is_file() or (p / "lingbot_va" / "wan_va").is_dir():
            return p
    rd = resolve_robodojo_root(robodojo_root)
    p = (rd / LINGBOT_VA_POLICY_REL).resolve()
    if p.is_dir():
        return p
    return LINGBOT_VA_ROOT_DEFAULT


def resolve_lingbot_va_python(prefer: str = "", conda_env: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("LINGBOT_VA_PYTHON", ""),
    ):
        c = (cand or "").strip()
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    env_dir = (conda_env or "").strip()
    if env_dir and os.path.isdir(env_dir):
        py = os.path.join(env_dir, "bin", "python")
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    for env_path in LINGBOT_VA_ENV_CANDIDATES:
        py = os.path.join(env_path, "bin", "python")
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    return shutil.which("python3") or "python3"


def resolve_lingbot_va_conda_env(prefer: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("LINGBOT_VA_CONDA_ENV", ""),
        os.environ.get("ROBODOJO_POLICY_ENV", ""),
        *LINGBOT_VA_ENV_CANDIDATES,
    ):
        c = (cand or "").strip()
        if c and os.path.isdir(c):
            return c
    return LINGBOT_VA_ENV_CANDIDATES[0]


def resolve_base_model(prefer: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("LINGBOT_VA_BASE_MODEL_PATH", ""),
        os.environ.get("BASE_MODEL_PATH", ""),
        *LINGBOT_VA_BASE_MODEL_CANDIDATES,
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c))
        if p.is_dir():
            return str(p.resolve())
    # Return preferred default even if missing (caller validates).
    return LINGBOT_VA_BASE_MODEL_CANDIDATES[0]


def _ckpt_looks_present(path: str) -> bool:
    p = Path(os.path.expanduser((path or "").strip()))
    if not p.exists():
        return False
    if p.is_file():
        return True
    if not p.is_dir():
        return False
    markers = (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
        "transformer",
    )
    if any((p / m).exists() for m in markers):
        return True
    try:
        for child in p.iterdir():
            if child.is_dir() and any((child / m).exists() for m in markers):
                return True
    except OSError:
        pass
    return False


def resolve_checkpoint(prefer: str = "", policy_root: Optional[Path] = None) -> str:
    raw = (prefer or "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
        if p.exists():
            return str(p.resolve())
        # Relative ckpt name under policy checkpoints/
        root = policy_root or resolve_lingbot_va_root()
        under = root / "checkpoints" / raw
        if under.exists():
            return str(under.resolve())
        return raw  # let upstream resolve short names
    env = (os.environ.get("LINGBOT_VA_CKPT") or os.environ.get("CHECKPOINT_PATH") or "").strip()
    if env:
        return env
    root = policy_root or resolve_lingbot_va_root()
    ckpt_dir = root / "checkpoints"
    if ckpt_dir.is_dir():
        try:
            subs = sorted(
                [c for c in ckpt_dir.iterdir() if c.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for sub in subs:
                if _ckpt_looks_present(str(sub)):
                    return str(sub.resolve())
        except OSError:
            pass
    for cand in LINGBOT_VA_CKPT_CANDIDATES:
        if _ckpt_looks_present(cand):
            return cand
    return str(ckpt_dir)


def resolve_paths(
    *,
    prefer_root: str = "",
    prefer_ckpt: str = "",
    prefer_base: str = "",
    config_name: str = "",
    python_bin: str = "",
    conda_env: str = "",
    backend: str = "",
) -> LingbotVaPaths:
    root = resolve_lingbot_va_root(prefer_root)
    cfg = va_config_for_backend(backend, config_name)
    base = resolve_base_model(prefer_base)
    ckpt = resolve_checkpoint(prefer_ckpt, policy_root=root)
    env = resolve_lingbot_va_conda_env(conda_env)
    py = resolve_lingbot_va_python(python_bin, conda_env=env)
    upstream = root / "lingbot_va"
    source = "ok" if (root / "deploy.py").is_file() else "missing_deploy"
    return LingbotVaPaths(
        policy_root=root,
        upstream_root=upstream,
        base_model=base,
        checkpoint=ckpt,
        config_name=cfg,
        python_bin=py,
        conda_env=env,
        source=source,
    )


def port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def probe_lingbot_va(paths: Optional[LingbotVaPaths] = None) -> Tuple[bool, str]:
    p = paths or resolve_paths()
    missing: List[str] = []
    if not (p.policy_root / "deploy.py").is_file():
        missing.append(f"deploy.py@{p.policy_root}")
    if not (p.upstream_root / "wan_va" / "wan_va_server.py").is_file():
        missing.append(f"wan_va_server@{p.upstream_root}")
    if not os.path.isdir(p.base_model):
        missing.append(f"base_model={p.base_model}")
    if missing:
        return False, "MISSING " + ", ".join(missing)
    ckpt_ok = _ckpt_looks_present(p.checkpoint)
    note = "ckpt_ok" if ckpt_ok else f"ckpt_pending={p.checkpoint}"
    return True, f"OK root={p.policy_root} config={p.config_name} {note}"


def lingbot_va_child_env(
    paths: LingbotVaPaths,
    *,
    extra: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Optional[str]]:
    py = paths.python_bin
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    bin_dir = os.path.join(prefix, "bin")
    markers = ("ros-humble", "/opt/ros/", "PyQt5/Qt5/lib", "/cv2/")
    path_parts = [bin_dir]
    for part in (os.environ.get("PATH") or "").split(os.pathsep):
        if not part or any(m in part for m in markers):
            continue
        if part not in path_parts:
            path_parts.append(part)

    py_path = [
        str(EAI_DIR / "tools"),
        str(paths.policy_root),
        str(paths.upstream_root),
        str(resolve_robodojo_root()),
        str(resolve_robodojo_root() / "XPolicyLab"),
    ]
    nvidia_dirs: List[str] = []
    sp = Path(prefix) / "lib"
    for child in sp.glob("python*/site-packages/nvidia/*/lib"):
        if child.is_dir():
            nvidia_dirs.append(str(child))

    env: Dict[str, Optional[str]] = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHOME": None,
        "PYTHONPATH": os.pathsep.join(py_path),
        "PATH": os.pathsep.join(path_parts),
        "LD_LIBRARY_PATH": os.pathsep.join(nvidia_dirs) if nvidia_dirs else "",
        "CONDA_PREFIX": prefix if os.path.isdir(prefix) else None,
        "AMENT_PREFIX_PATH": None,
        "ROS_DISTRO": None,
        "ROS_VERSION": None,
        "LINGBOT_VA_ROOT": str(paths.policy_root),
        "LINGBOT_VA_BASE_MODEL_PATH": paths.base_model,
        "BASE_MODEL_PATH": paths.base_model,
        "CHECKPOINT_PATH": paths.checkpoint,
        "LINGBOT_VA_CKPT": paths.checkpoint,
        "CONFIG_NAME": paths.config_name,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    if extra:
        env.update(extra)
    return env


def build_wan_va_server_argv(
    *,
    backend: str = "isaac",
    checkpoint: str = "",
    base_model: str = "",
    config_name: str = "",
    port: Optional[int] = None,
    gpu: str = "0",
    python_bin: str = "",
    policy_root: str = "",
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    """Return (argv, cwd, env) to launch official launch_wan_va_server.sh."""
    paths = resolve_paths(
        prefer_root=policy_root,
        prefer_ckpt=checkpoint,
        prefer_base=base_model,
        config_name=config_name,
        python_bin=python_bin,
        backend=backend,
    )
    script = paths.policy_root / "launch_wan_va_server.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    va_port = int(port or DEFAULT_VA_PORT)
    # Prefer bash wrapper so prepare_merged_ckpt runs first.
    argv = ["bash", str(script), str(gpu), str(va_port)]
    env = lingbot_va_child_env(
        paths,
        extra={
            "CHECKPOINT_PATH": paths.checkpoint,
            "BASE_MODEL_PATH": paths.base_model,
            "CONFIG_NAME": paths.config_name,
            "CUDA_VISIBLE_DEVICES": str(gpu),
        },
    )
    return argv, paths.policy_root, env


def build_libero_va_eval_argv(
    *,
    task_suite: str = "libero_10",
    checkpoint: str = "",
    base_model: str = "",
    host: str = DEFAULT_VA_HOST,
    port: int = LIBERO_VA_PORT_DEFAULT,
    test_num: int = 1,
    task_start: int = 0,
    task_end: int = 1,
    out_dir: str = "",
    python_bin: str = "",
    policy_root: str = "",
    start_server: bool = True,
    gpu: str = "0",
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    """Client-side LIBERO eval against wan_va (optionally start server first)."""
    paths = resolve_paths(
        prefer_root=policy_root,
        prefer_ckpt=checkpoint,
        prefer_base=base_model,
        config_name="libero",
        python_bin=python_bin,
        backend="libero",
    )
    wrapper = Path(__file__).resolve().parent / "run_lingbot_va_eval.py"
    out = out_dir or str(EAI_DIR / ".cache" / "lingbot_va_eval" / "libero")
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = [
        paths.python_bin,
        str(wrapper),
        "--backend=libero",
        f"--config-name={paths.config_name}",
        f"--checkpoint={paths.checkpoint}",
        f"--base-model={paths.base_model}",
        f"--policy-root={paths.policy_root}",
        f"--host={host}",
        f"--port={int(port)}",
        f"--task-suite={task_suite}",
        f"--test-num={int(test_num)}",
        f"--task-start={int(task_start)}",
        f"--task-end={int(task_end)}",
        f"--out-dir={out}",
        f"--gpu={gpu}",
        "--start-server" if start_server else "--no-start-server",
    ]
    env = lingbot_va_child_env(paths)
    return argv, paths.upstream_root, env


def build_robotwin_va_eval_argv(
    *,
    task: str = "adjust_bottle",
    checkpoint: str = "",
    base_model: str = "",
    host: str = DEFAULT_VA_HOST,
    port: int = ROBOTWIN_VA_PORT_DEFAULT,
    python_bin: str = "",
    policy_root: str = "",
    start_server: bool = True,
    gpu: str = "0",
    config_name: str = "robotwin",
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    paths = resolve_paths(
        prefer_root=policy_root,
        prefer_ckpt=checkpoint,
        prefer_base=base_model,
        config_name=config_name or "robotwin",
        python_bin=python_bin,
        backend="robotwin",
    )
    wrapper = Path(__file__).resolve().parent / "run_lingbot_va_eval.py"
    argv = [
        paths.python_bin,
        str(wrapper),
        "--backend=robotwin",
        f"--config-name={paths.config_name}",
        f"--checkpoint={paths.checkpoint}",
        f"--base-model={paths.base_model}",
        f"--policy-root={paths.policy_root}",
        f"--host={host}",
        f"--port={int(port)}",
        f"--task={task}",
        f"--gpu={gpu}",
        "--start-server" if start_server else "--no-start-server",
    ]
    env = lingbot_va_child_env(paths)
    return argv, paths.upstream_root, env


def build_backend_va_argv(
    backend: str,
    *,
    checkpoint: str = "",
    base_model: str = "",
    config_name: str = "",
    task: str = "",
    task_suite: str = "",
    python_bin: str = "",
    policy_root: str = "",
    port: Optional[int] = None,
    gpu: str = "0",
    start_server: bool = True,
) -> Tuple[List[str], Path, Dict[str, Optional[str]], str]:
    """Dispatch LingBot-VA launch for any registered sim backend.

    Returns (argv, cwd, env, note).
    """
    key = (backend or "").strip().lower()
    level = va_support_level(key)
    cfg = va_config_for_backend(key, config_name)

    if key in ("libero", "libero_plus", "libero_pro", "libero_mem") or level == "native_eval" and key.startswith("libero"):
        suite = (task_suite or task or "libero_10").strip() or "libero_10"
        argv, cwd, env = build_libero_va_eval_argv(
            task_suite=suite,
            checkpoint=checkpoint,
            base_model=base_model,
            port=int(port or LIBERO_VA_PORT_DEFAULT),
            python_bin=python_bin,
            policy_root=policy_root,
            start_server=start_server,
            gpu=gpu,
        )
        return argv, cwd, env, f"LingBot-VA native LIBERO eval · suite={suite}"

    if key == "robotwin":
        argv, cwd, env = build_robotwin_va_eval_argv(
            task=(task or "adjust_bottle").strip() or "adjust_bottle",
            checkpoint=checkpoint,
            base_model=base_model,
            config_name=cfg,
            port=int(port or ROBOTWIN_VA_PORT_DEFAULT),
            python_bin=python_bin,
            policy_root=policy_root,
            start_server=start_server,
            gpu=gpu,
        )
        return argv, cwd, env, f"LingBot-VA native RoboTwin eval · task={task or 'adjust_bottle'}"

    if key == "isaac" or level == "robodojo_adapter":
        # Isaac path is handled in show_camera_topics (XPolicyLab deploy).
        # Here we expose wan_va server launch for attach_external / pre-deploy.
        argv, cwd, env = build_wan_va_server_argv(
            backend="isaac",
            checkpoint=checkpoint,
            base_model=base_model,
            config_name=cfg,
            port=int(port or DEFAULT_VA_PORT),
            gpu=gpu,
            python_bin=python_bin,
            policy_root=policy_root,
        )
        return (
            argv,
            cwd,
            env,
            "LingBot-VA RoboDojo: start wan_va (:10001); use XPolicyLab LingBot_VA deploy for Isaac client",
        )

    # serve_only / extended: start wan_va and print adapter note
    argv, cwd, env = build_wan_va_server_argv(
        backend=key,
        checkpoint=checkpoint,
        base_model=base_model,
        config_name=cfg,
        port=int(port or DEFAULT_VA_PORT),
        gpu=gpu,
        python_bin=python_bin,
        policy_root=policy_root,
    )
    note = (
        f"LingBot-VA serve_only for backend={key} (config={cfg}). "
        "wan_va websocket 已启动；该后端尚无官方 closed-loop adapter，"
        "可用 WebsocketClientPolicy 对接或改走 Isaac/LIBERO/RoboTwin。"
    )
    return argv, cwd, env, note


def robodojo_policy_rel() -> str:
    return LINGBOT_VA_POLICY_REL


def install_hint(python_bin: str = "") -> str:
    py = resolve_lingbot_va_python(python_bin)
    root = resolve_lingbot_va_root()
    base = resolve_base_model()
    return (
        "LingBot-VA 依赖未就绪。安装与配置示例：\n"
        f"  1) cd {root} && bash install.sh\n"
        f"     conda activate lingbot_va  # 或设置 LINGBOT_VA_PYTHON={py}\n"
        f"  2) export LINGBOT_VA_BASE_MODEL_PATH=<lingbot-va-base>  # 当前探测: {base}\n"
        f"  3) 将 post-train ckpt 放到 {root}/checkpoints/<run_name>\n"
        "  4) Isaac/RoboDojo: 策略选 LingBot_VA，action=joint，再「一体启动」或先部署\n"
        "  5) LIBERO/RoboTwin: 配置选 LingBot-VA，走 tools/run_lingbot_va_eval.py\n"
        "  官方仓库: RoboDojo/XPolicyLab/policy/LingBot_VA/"
    )


def append_policy_combo_entries(
    entries: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Ensure LingBot-VA appears in (label, id) policy lists without duplicating."""
    out = list(entries)
    ids = {e[1] for e in out}
    if LINGBOT_VA_POLICY_ID not in ids and "lingbot_va" not in {
        x.lower().replace("-", "_") for x in ids
    }:
        out.append((LINGBOT_VA_POLICY_LABEL, LINGBOT_VA_POLICY_ID))
    return out


__all__ = [
    "BACKEND_VA_CONFIG",
    "BACKEND_VA_SUPPORT",
    "DEFAULT_VA_HOST",
    "DEFAULT_VA_PORT",
    "LIBERO_VA_PORT_DEFAULT",
    "LINGBOT_VA_CONFIGS",
    "LINGBOT_VA_POLICY_ID",
    "LINGBOT_VA_POLICY_LABEL",
    "LINGBOT_VA_POLICY_REL",
    "LINGBOT_VA_ROOT_DEFAULT",
    "LingbotVaPaths",
    "append_policy_combo_entries",
    "build_backend_va_argv",
    "build_libero_va_eval_argv",
    "build_robotwin_va_eval_argv",
    "build_wan_va_server_argv",
    "install_hint",
    "is_lingbot_va_policy",
    "lingbot_va_child_env",
    "port_listening",
    "probe_lingbot_va",
    "resolve_base_model",
    "resolve_checkpoint",
    "resolve_lingbot_va_conda_env",
    "resolve_lingbot_va_python",
    "resolve_lingbot_va_root",
    "resolve_paths",
    "resolve_robodojo_root",
    "robodojo_policy_rel",
    "va_config_for_backend",
    "va_support_level",
]
