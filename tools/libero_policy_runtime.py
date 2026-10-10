#!/usr/bin/env python3
"""Resolve / launch LIBERO sim eval (RLinf + openpi patterns).

RLinf reference:
  toolkits/eval_scripts_openpi/libero_eval.py
OpenPI reference (websocket client):
  examples/libero/main.py

Default mode is ``remote``: connect to openpi ``serve_policy`` (Pi 部署),
matching MolmoSpaces. ``local`` loads the checkpoint in-process like RLinf.
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
RLINF_ROOT_DEFAULT = Path("/share_data/projects/mahjong/share/personal/liyichao/RLinf")
OPENPI_ROOT_DEFAULT = Path(
    "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/openpi"
)
LIBERO_SRC_DEFAULT = OPENPI_ROOT_DEFAULT / "third_party" / "libero"
LIBERO_CONFIG_DIR_DEFAULT = EAI_DIR / ".cache" / "libero"
LIBERO_EVAL_OUTPUT_DIR = EAI_DIR / ".cache" / "libero_eval"

LIBERO_PYTHON_CANDIDATES: Tuple[str, ...] = (
    str(EAI_DIR / ".cache" / "envs" / "libero" / "bin" / "python"),
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/libero/bin/python",
    "/home/psibot/miniconda3/envs/libero/bin/python",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/openpi/bin/python",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python",
    "/home/psibot/miniconda3/envs/openpi/bin/python",
)

LIBERO_TASK_SUITES: Tuple[Tuple[str, str], ...] = (
    ("libero_spatial", "libero_spatial"),
    ("libero_object", "libero_object"),
    ("libero_goal", "libero_goal"),
    ("libero_10 (long)", "libero_10"),
    ("libero_90", "libero_90"),
)

LIBERO_POLICY_CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("π0.5-LIBERO (pi05_libero)", "pi05_libero"),
    ("π0-LIBERO (pi0_libero)", "pi0_libero"),
)

LIBERO_CKPT_GS_DEFAULT = "gs://openpi-assets/checkpoints/pi05_libero"
# Official openpi JAX layout mirror (params/ + assets/); GCS often needs anon/gsutil.
LIBERO_CKPT_HF_DEFAULT = "bf-jeon/pi05_libero"
LIBERO_CKPT_HF_LOCAL_DEFAULT = str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_libero")
LIBERO_CKPT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    LIBERO_CKPT_HF_LOCAL_DEFAULT,
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_libero",
    str(EAI_DIR / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi0_libero",
    str(EAI_DIR / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi0_libero"),
)

LIBERO_HOST_DEFAULT = "localhost"
LIBERO_PORT_DEFAULT = 8080


@dataclass(frozen=True)
class LiberoCheckpoint:
    path: str
    config: str
    source: str


def resolve_rlinf_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("RLINF_ROOT", ""),
        str(RLINF_ROOT_DEFAULT),
    ):
        p = Path(os.path.expanduser(cand or "")).resolve()
        if (p / "toolkits" / "eval_scripts_openpi" / "libero_eval.py").is_file():
            return p
    return RLINF_ROOT_DEFAULT


def resolve_openpi_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("OPENPI_ROOT", ""),
        str(OPENPI_ROOT_DEFAULT),
    ):
        p = Path(os.path.expanduser(cand or "")).resolve()
        if (p / "scripts" / "serve_policy.py").is_file():
            return p
    return OPENPI_ROOT_DEFAULT


def resolve_libero_src(openpi_root: Optional[Path] = None) -> Path:
    env = (os.environ.get("LIBERO_PATH") or "").strip()
    if env:
        p = Path(os.path.expanduser(env)).resolve()
        if p.is_dir():
            return p
    root = openpi_root or resolve_openpi_root()
    for cand in (
        root / "third_party" / "libero",
        LIBERO_SRC_DEFAULT,
        resolve_rlinf_root() / ".venv" / "libero",
    ):
        if cand.is_dir():
            return cand.resolve()
    return LIBERO_SRC_DEFAULT


def resolve_libero_python(prefer: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("LIBERO_PYTHON", ""),
        *LIBERO_PYTHON_CANDIDATES,
        shutil.which("python3") or "",
    ):
        c = (cand or "").strip()
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "python3"


def ensure_libero_config(config_dir: Optional[Path] = None, libero_src: Optional[Path] = None) -> Path:
    """Write non-interactive LIBERO config.yaml (avoids stdin prompt on first import)."""
    cfg_dir = Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH") or LIBERO_CONFIG_DIR_DEFAULT)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.is_file():
        return cfg_dir

    src = libero_src or resolve_libero_src()
    # Package root that contains bddl_files / assets (…/libero/libero).
    pkg_root = src / "libero" / "libero"
    if not pkg_root.is_dir():
        pkg_root = src / "libero"
    if not pkg_root.is_dir():
        pkg_root = src

    paths = {
        "benchmark_root": str(pkg_root),
        "bddl_files": str(pkg_root / "bddl_files"),
        "init_states": str(pkg_root / "init_files"),
        "datasets": str(src / "datasets"),
        "assets": str(pkg_root / "assets"),
    }
    # Prefer PyYAML if present; else minimal YAML.
    try:
        import yaml  # type: ignore

        cfg_file.write_text(yaml.safe_dump(paths, sort_keys=False), encoding="utf-8")
    except Exception:
        lines = [f"{k}: {v}" for k, v in paths.items()]
        cfg_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg_dir


def _libero_ckpt_usable(path: str) -> bool:
    """Prefer pi_policy_runtime allocation check; fall back to layout-only."""
    try:
        from pi_policy_runtime import is_local_checkpoint_usable  # type: ignore

        return bool(is_local_checkpoint_usable(path))
    except Exception:
        p = Path(os.path.expanduser((path or "").strip()))
        if not p.is_dir():
            return False
        if (p / ".cache" / "huggingface" / "download").is_dir():
            for dirpath, _dns, fns in os.walk(p / ".cache" / "huggingface" / "download"):
                for name in fns:
                    if name.endswith(".incomplete"):
                        return False
        return (p / "model.safetensors").is_file() or (p / "params").is_dir()


def resolve_libero_checkpoint(prefer: str = "", config: str = "") -> LiberoCheckpoint:
    """Pick LIBERO Pi ckpt: user → usable local/HF cache → env → HF dest → gs://."""
    cfg = (config or "").strip() or "pi05_libero"
    raw = (prefer or "").strip()
    if raw:
        # If user points at gs:// but a usable local mirror exists, prefer local.
        if raw.startswith("gs://") and "pi05_libero" in raw:
            for cand in LIBERO_CKPT_LOCAL_CANDIDATES:
                if _libero_ckpt_usable(cand):
                    return LiberoCheckpoint(path=cand, config=cfg, source="local")
        return LiberoCheckpoint(path=raw, config=cfg, source="user")
    for cand in LIBERO_CKPT_LOCAL_CANDIDATES:
        if _libero_ckpt_usable(cand):
            return LiberoCheckpoint(path=cand, config=cfg, source="local")
    env = (os.environ.get("LIBERO_CKPT") or "").strip()
    if env:
        return LiberoCheckpoint(path=env, config=cfg, source="env")
    # Default deploy target: HF cache path (materialized on first 部署 via pi runtime).
    return LiberoCheckpoint(path=LIBERO_CKPT_HF_LOCAL_DEFAULT, config=cfg, source="hf")


def probe_libero_deps(python_bin: str, libero_src: Optional[Path] = None) -> Tuple[bool, str]:
    """Return (ok, message) after probing robosuite + libero imports."""
    src = libero_src or resolve_libero_src()
    cfg_dir = ensure_libero_config(libero_src=src)
    py_paths = [
        str(src),
    ]
    code = (
        "import sys\n"
        "missing=[]\n"
        "for m in ('robosuite','libero','imageio','numpy','matplotlib'):\n"
        "  try:\n"
        "    __import__(m)\n"
        "  except Exception as e:\n"
        "    missing.append(f'{m}:{type(e).__name__}')\n"
        "if missing:\n"
        "  print('MISSING', ','.join(missing)); raise SystemExit(2)\n"
        "from libero.libero import benchmark\n"
        "print('OK', ','.join(sorted(benchmark.get_benchmark_dict().keys())[:6]))\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTHONHOME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "AMENT_PREFIX_PATH",
            "ROS_DISTRO",
        )
    }
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(py_paths),
            "LIBERO_CONFIG_PATH": str(cfg_dir),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL") or "egl",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"探测异常: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0 and out.startswith("OK"):
        return True, out
    detail = out or err or f"exit={proc.returncode}"
    return False, detail[:500]


def libero_child_env(
    *,
    python_bin: str = "",
    rlinf_root: str = "",
    openpi_root: str = "",
    libero_src: str = "",
    render_gui: bool = False,
) -> Dict[str, Optional[str]]:
    """Clean env for LIBERO client / local eval subprocesses."""
    py = resolve_libero_python(python_bin)
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    bin_dir = os.path.join(prefix, "bin")
    rlinf = resolve_rlinf_root(rlinf_root)
    openpi = resolve_openpi_root(openpi_root)
    src = Path(libero_src).resolve() if libero_src else resolve_libero_src(openpi)
    cfg_dir = ensure_libero_config(libero_src=src)

    markers = ("ros-humble", "/opt/ros/", "PyQt5/Qt5/lib", "/cv2/")
    path_parts = [bin_dir]
    for part in (os.environ.get("PATH") or "").split(os.pathsep):
        if not part or any(m in part for m in markers):
            continue
        if part not in path_parts:
            path_parts.append(part)

    py_path = [
        str(EAI_DIR / "tools"),
        str(rlinf),
        str(openpi / "src"),
        # Official openpi layout: parent of package dir so `from libero.libero import …`
        str(src),
    ]
    # Prefer pip nvidia libs when present (same idea as openpi_child_env).
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
        "LIBERO_CONFIG_PATH": str(cfg_dir),
        "LIBERO_PATH": str(src),
        "OPENPI_ROOT": str(openpi),
        "RLINF_ROOT": str(rlinf),
        "OPENPI_DATA_HOME": os.environ.get("OPENPI_DATA_HOME")
        or str(EAI_DIR / ".cache" / "openpi"),
        "TOKENIZERS_PARALLELISM": "false",
        "OPENPI_DISABLE_TORCH_COMPILE": os.environ.get(
            "OPENPI_DISABLE_TORCH_COMPILE", "1"
        ),
    }
    if render_gui:
        # On-screen robosuite viewer (same VirtualGL hint as MuJoCo viewer).
        env["MUJOCO_GL"] = "glfw"
        env["MUJOCO_USE_VGL"] = os.environ.get("MUJOCO_USE_VGL") or "1"
        env["MUJOCO_VGL_DEVICE"] = os.environ.get("MUJOCO_VGL_DEVICE") or "egl"
        if (os.environ.get("DISPLAY") or "").strip():
            env["DISPLAY"] = os.environ.get("DISPLAY")
        if (os.environ.get("XAUTHORITY") or "").strip():
            env["XAUTHORITY"] = os.environ.get("XAUTHORITY")
    else:
        env["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
    return env


def port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def build_libero_eval_argv(
    *,
    mode: str = "remote",
    task_suite: str = "libero_spatial",
    config_name: str = "pi05_libero",
    checkpoint: str = "",
    host: str = LIBERO_HOST_DEFAULT,
    port: int = LIBERO_PORT_DEFAULT,
    num_trials_per_task: int = 1,
    max_tasks: int = 1,
    action_chunk: int = 5,
    num_steps: int = 10,
    seed: int = 7,
    num_save_videos: int = 1,
    exp_name: str = "",
    python_bin: str = "",
    rlinf_root: str = "",
    openpi_root: str = "",
    render_gui: bool = True,
    viewer_only: bool = False,
    rynnvalue_live_hud: bool = False,
    rynnvalue_server_url: str = "http://127.0.0.1:8001",
    rynnvalue_refresh_sec: float = 1.0,
    rynnvalue_num_frames: int = 8,
    rynnvalue_show_window: bool = True,
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    """Return (argv, cwd, env_extra) for LIBERO eval.

    ``mode``:
      - remote: websocket client (openpi examples / eai wrapper)
      - local: in-process policy (RLinf libero_eval style via eai wrapper)

    ``viewer_only=True`` opens the on-screen env and holds it (no policy / Pi).
    """
    py = resolve_libero_python(python_bin)
    rlinf = resolve_rlinf_root(rlinf_root)
    openpi = resolve_openpi_root(openpi_root)
    wrapper = Path(__file__).resolve().parent / "run_libero_eval.py"
    ckpt = resolve_libero_checkpoint(checkpoint, config_name)
    exp = (exp_name or "").strip() or f"{task_suite}_{config_name}"
    if viewer_only:
        exp = f"{exp}_viewer"
    out_dir = LIBERO_EVAL_OUTPUT_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Viewer-only always needs the GLFW on-screen window.
    want_gui = bool(render_gui or viewer_only)
    argv = [
        py,
        str(wrapper),
        f"--mode={mode}",
        f"--task_suite_name={task_suite}",
        f"--config_name={ckpt.config}",
        f"--pretrained_path={ckpt.path}",
        f"--host={host}",
        f"--port={int(port)}",
        f"--num_trials_per_task={int(num_trials_per_task)}",
        f"--max_tasks={int(max_tasks)}",
        f"--action_chunk={int(action_chunk)}",
        f"--num_steps={int(num_steps)}",
        f"--seed={int(seed)}",
        f"--num_save_videos={int(num_save_videos)}",
        f"--exp_name={exp}",
        f"--log_dir={out_dir}",
        f"--rlinf_root={rlinf}",
        f"--openpi_root={openpi}",
    ]
    if want_gui:
        argv.append("--render_gui")
    else:
        argv.append("--no_render_gui")
    if viewer_only:
        argv.append("--viewer_only")
    from rynnvalue_sim_bridge import append_rynnvalue_live_hud_argv

    append_rynnvalue_live_hud_argv(
        argv,
        enabled=bool(rynnvalue_live_hud) and not viewer_only,
        server_url=rynnvalue_server_url,
        refresh_sec=rynnvalue_refresh_sec,
        num_frames=rynnvalue_num_frames,
        show_window=rynnvalue_show_window,
    )
    env_extra = libero_child_env(
        python_bin=py,
        rlinf_root=str(rlinf),
        openpi_root=str(openpi),
        render_gui=bool(want_gui),
    )
    return argv, openpi, env_extra


def install_hint(python_bin: str = "") -> str:
    py = resolve_libero_python(python_bin)
    src = resolve_libero_src()
    return (
        "LIBERO 依赖未就绪（需要 robosuite + libero）。示例安装：\n"
        f"  1) RLinf: cd {resolve_rlinf_root()} && bash requirements/install.sh libero\n"
        f"  2) 或: {py} -m pip install 'robosuite==1.4.1' imageio tqdm pyyaml\n"
        f"     && {py} -m pip install -e {src}\n"
        f"  并设置 LIBERO_PYTHON={py}"
    )


__all__ = [
    "LIBERO_CKPT_GS_DEFAULT",
    "LIBERO_EVAL_OUTPUT_DIR",
    "LIBERO_HOST_DEFAULT",
    "LIBERO_POLICY_CONFIGS",
    "LIBERO_PORT_DEFAULT",
    "LIBERO_TASK_SUITES",
    "LiberoCheckpoint",
    "build_libero_eval_argv",
    "ensure_libero_config",
    "install_hint",
    "libero_child_env",
    "port_listening",
    "probe_libero_deps",
    "resolve_libero_checkpoint",
    "resolve_libero_python",
    "resolve_libero_src",
    "resolve_openpi_root",
    "resolve_rlinf_root",
]
