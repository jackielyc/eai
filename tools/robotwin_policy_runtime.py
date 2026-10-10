#!/usr/bin/env python3
"""Resolve / launch RoboTwin sim eval (RLinf OpenPI patterns).

RLinf reference:
  examples/embodiment/eval_embodiment.sh
  docs/.../examples/embodied/robotwin.rst
  rlinf/envs/robotwin/robotwin_env.py

Unlike LIBERO (standalone openpi websocket client), RoboTwin official eval
runs through RLinf ``EmbodiedEvalRunner`` + SAPIEN VectorEnv. This module
assembles that launch with GUI-friendly overrides (1 env, single GPU).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

EAI_DIR = Path(__file__).resolve().parent.parent
RLINF_ROOT_DEFAULT = Path("/share_data/projects/mahjong/share/personal/liyichao/RLinf")
ROBOTWIN_ROOT_DEFAULT = Path(
    "/share_data/projects/mahjong/share/personal/liyichao/RoboTwin"
)
ROBOTWIN_EVAL_OUTPUT_DIR = EAI_DIR / ".cache" / "robotwin_eval"

ROBOTWIN_PYTHON_CANDIDATES: Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python",
    str(RLINF_ROOT_DEFAULT / ".venv" / "bin" / "python"),
    "/home/psibot/miniconda3/envs/RLinf/bin/python",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/openpi/bin/python",
)

# (label, task_name) — tasks with ready Hydra env YAMLs under examples/embodiment/config/env/
ROBOTWIN_TASKS: Tuple[Tuple[str, str], ...] = (
    ("adjust_bottle", "adjust_bottle"),
    ("place_empty_cup", "place_empty_cup"),
    ("place_container_plate", "place_container_plate"),
    ("place_shoe", "place_shoe"),
    ("pick_dual_bottles", "pick_dual_bottles"),
    ("move_can_pot", "move_can_pot"),
    ("lift_pot", "lift_pot"),
    ("handover_block", "handover_block"),
    ("beat_block_hammer", "beat_block_hammer"),
    ("click_bell", "click_bell"),
)

ROBOTWIN_POLICY_CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("π0.5 Aloha (pi05_aloha_robotwin)", "pi05_aloha_robotwin"),
    ("π0 Aloha (pi0_aloha_robotwin)", "pi0_aloha_robotwin"),
)

# Preferred Hydra config basename (no .yaml) for OpenPI eval.
_TASK_HYDRA_EVAL: Dict[Tuple[str, str], str] = {
    ("adjust_bottle", "pi05_aloha_robotwin"): "robotwin_adjust_bottle_ppo_openpi_pi05_eval",
    ("adjust_bottle", "pi0_aloha_robotwin"): "robotwin_adjust_bottle_ppo_openpi_eval",
    ("place_empty_cup", "pi0_aloha_robotwin"): "robotwin_place_empty_cup_openpi_eval",
    ("place_empty_cup", "pi05_aloha_robotwin"): "robotwin_adjust_bottle_ppo_openpi_pi05_eval",
}

ROBOTWIN_CKPT_HF_DEFAULT = "RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
ROBOTWIN_CKPT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    str(EAI_DIR / ".cache" / "robotwin" / "hf" / "RLinf-Pi05-RoboTwin-SFT-adjust_bottle"),
    "/share_data/projects/mahjong/share/personal/liyichao/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_aloha_robotwin",
    str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_aloha_robotwin"),
)

# ASSETS_PATH must be a *root that contains* ``assets/objects`` (RoboTwin repo root
# after ``script/_download_assets.sh``, or a dedicated robotwin_assets root).
# Not the inner ``.../assets`` folder itself — RoboTwin joins ``ASSETS_PATH/assets/...``.
ROBOTWIN_ASSETS_CANDIDATES: Tuple[str, ...] = (
    str(ROBOTWIN_ROOT_DEFAULT),
    "/share_data/projects/mahjong/share/personal/liyichao/robotwin_assets",
    str(EAI_DIR / ".cache" / "robotwin" / "assets_root"),
)


@dataclass(frozen=True)
class RobotwinCheckpoint:
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
        if (p / "examples" / "embodiment" / "eval_embodiment.sh").is_file():
            return p
        if (p / "rlinf" / "envs" / "robotwin" / "robotwin_env.py").is_file():
            return p
    return RLINF_ROOT_DEFAULT


def resolve_robotwin_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("ROBOTWIN_PATH", ""),
        str(ROBOTWIN_ROOT_DEFAULT),
    ):
        p = Path(os.path.expanduser(cand or "")).resolve()
        if not p.is_dir():
            continue
        # RLinf_support layout: robotwin/envs/vector_env.py
        if (p / "robotwin" / "envs" / "vector_env.py").is_file():
            return p
        # main-branch layout still usable as checkout root (needs branch switch).
        if (p / "envs" / "_base_task.py").is_file() or (p / "README.md").is_file():
            return p
    return ROBOTWIN_ROOT_DEFAULT


def _assets_root_ok(path: Path) -> bool:
    """True if ``path/assets/objects`` (or embodiments) exists."""
    if not path.is_dir():
        return False
    return (path / "assets" / "objects").is_dir() or (
        path / "assets" / "embodiments"
    ).is_dir()


def _normalize_assets_root(path: Path) -> Optional[Path]:
    """Accept either assets root or the inner ``assets/`` directory."""
    p = path.expanduser().resolve()
    if _assets_root_ok(p):
        return p
    # User / old default pointed at ``.../RoboTwin/assets`` → walk up.
    if p.name == "assets" and _assets_root_ok(p.parent):
        return p.parent
    if p.is_dir():
        # Still usable as ASSETS_PATH once downloads finish; prefer parent of assets/.
        if (p / "assets").is_dir():
            return p
        if p.name == "assets":
            return p.parent
    return None


def resolve_robotwin_assets(raw: str = "", robotwin_root: Optional[Path] = None) -> str:
    root = robotwin_root or resolve_robotwin_root()
    for cand in (
        (raw or "").strip(),
        os.environ.get("ROBOTWIN_ASSETS", ""),
        os.environ.get("ASSETS_PATH", ""),
        str(root),
        *ROBOTWIN_ASSETS_CANDIDATES,
    ):
        c = (cand or "").strip()
        if not c:
            continue
        norm = _normalize_assets_root(Path(c))
        if norm is not None and _assets_root_ok(norm):
            return str(norm)
    # Fall back to RoboTwin checkout root (correct join target even before download).
    for cand in ((raw or "").strip(), str(root), *ROBOTWIN_ASSETS_CANDIDATES):
        c = (cand or "").strip()
        if not c:
            continue
        norm = _normalize_assets_root(Path(c))
        if norm is not None:
            return str(norm)
    return str(root.resolve())


def resolve_robotwin_python(prefer: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("ROBOTWIN_PYTHON", ""),
        os.environ.get("RLINF_PYTHON", ""),
        *ROBOTWIN_PYTHON_CANDIDATES,
        shutil.which("python3") or "",
    ):
        c = (cand or "").strip()
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "python3"


def _ckpt_looks_present(path: str) -> bool:
    p = Path(os.path.expanduser((path or "").strip()))
    if not p.is_dir():
        return False
    return (
        (p / "model.safetensors").is_file()
        or (p / "model.safetensors.index.json").is_file()
        or (p / "params").is_dir()
        or (p / "config.json").is_file()
    )


def resolve_robotwin_checkpoint(prefer: str = "", config: str = "") -> RobotwinCheckpoint:
    cfg = (config or "").strip() or "pi05_aloha_robotwin"
    raw = (prefer or "").strip()
    if raw:
        return RobotwinCheckpoint(path=raw, config=cfg, source="user")
    for cand in ROBOTWIN_CKPT_LOCAL_CANDIDATES:
        if _ckpt_looks_present(cand):
            return RobotwinCheckpoint(path=cand, config=cfg, source="local")
    env = (os.environ.get("ROBOTWIN_CKPT") or "").strip()
    if env:
        return RobotwinCheckpoint(path=env, config=cfg, source="env")
    # Default: HF repo id / local download dest (materialize separately if needed).
    dest = ROBOTWIN_CKPT_LOCAL_CANDIDATES[0]
    return RobotwinCheckpoint(path=dest, config=cfg, source="hf")


def ensure_robotwin_checkpoint(
    prefer: str = "",
    config: str = "",
    *,
    hf_repo: str = "",
    log=None,
) -> Tuple[bool, str]:
    """Ensure local RoboTwin OpenPI checkpoint exists; download from HF if missing.

    Returns ``(ok, local_path_or_error)``.
    """
    ckpt = resolve_robotwin_checkpoint(prefer, config)
    dest = Path(os.path.expanduser(ckpt.path)).resolve()
    if _ckpt_looks_present(str(dest)):
        return True, str(dest)
    repo = (hf_repo or "").strip() or ROBOTWIN_CKPT_HF_DEFAULT
    if log:
        log(f"[robotwin] 下载 HF checkpoint: {repo} → {dest}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        return False, f"缺少 huggingface_hub: {exc}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(dest))
    except Exception as exc:  # noqa: BLE001
        return False, f"HF 下载失败 ({repo}): {type(exc).__name__}: {exc}"
    if not _ckpt_looks_present(str(dest)):
        return False, f"下载后仍不可用: {dest}"
    if log:
        log(f"[robotwin] checkpoint 就绪: {dest}")
    return True, str(dest)


def hydra_config_for(task: str, policy_config: str) -> str:
    """Pick Hydra eval config basename; fall back to pi05 adjust_bottle eval."""
    task = (task or "").strip() or "adjust_bottle"
    cfg = (policy_config or "").strip() or "pi05_aloha_robotwin"
    if (task, cfg) in _TASK_HYDRA_EVAL:
        return _TASK_HYDRA_EVAL[(task, cfg)]
    if cfg.startswith("pi0_") and not cfg.startswith("pi05_"):
        return "robotwin_adjust_bottle_ppo_openpi_eval"
    return "robotwin_adjust_bottle_ppo_openpi_pi05_eval"


def probe_robotwin_deps(
    python_bin: str,
    *,
    rlinf_root: Optional[Path] = None,
    robotwin_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Return (ok, message) after probing sapien + robotwin VectorEnv import."""
    rlinf = rlinf_root or resolve_rlinf_root()
    rt = robotwin_root or resolve_robotwin_root()
    code = (
        "import sys\n"
        "missing=[]\n"
        "for m in ('sapien','gymnasium','omegaconf','torch','openpi'):\n"
        "  try:\n"
        "    __import__(m)\n"
        "  except Exception as e:\n"
        "    missing.append(f'{m}:{type(e).__name__}')\n"
        "if missing:\n"
        "  print('MISSING', ','.join(missing)); raise SystemExit(2)\n"
        "try:\n"
        "  from robotwin.envs.vector_env import VectorEnv  # noqa: F401\n"
        "  print('OK vector_env')\n"
        "except Exception as e:\n"
        "  print('NO_VECTOR_ENV', type(e).__name__, e)\n"
        "  raise SystemExit(3)\n"
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
    assets = resolve_robotwin_assets(robotwin_root=rt)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join([str(rlinf), str(rt)]),
            "ROBOTWIN_PATH": str(rt),
            "ASSETS_PATH": assets,
            "ROBOTWIN_ASSETS": assets,
            "ROBOT_PLATFORM": "ALOHA",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"探测异常: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0 and out.startswith("OK"):
        return True, out
    detail = out or err or f"exit={proc.returncode}"
    return False, detail[:800]


def robotwin_child_env(
    *,
    python_bin: str = "",
    rlinf_root: str = "",
    robotwin_root: str = "",
    assets_path: str = "",
) -> Dict[str, Optional[str]]:
    py = resolve_robotwin_python(python_bin)
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    bin_dir = os.path.join(prefix, "bin")
    rlinf = resolve_rlinf_root(rlinf_root)
    rt = resolve_robotwin_root(robotwin_root)
    assets = resolve_robotwin_assets(assets_path, robotwin_root=rt)

    markers = ("ros-humble", "/opt/ros/", "PyQt5/Qt5/lib", "/cv2/")
    path_parts = [bin_dir]
    for part in (os.environ.get("PATH") or "").split(os.pathsep):
        if not part or any(m in part for m in markers):
            continue
        if part not in path_parts:
            path_parts.append(part)

    # Ray workers need nvidia pip libs on LD_LIBRARY_PATH (cudnn_graph.so.9 etc.);
    # the GUI ROS env often omits them and OpenPI conv then aborts in cudnnCreate.
    nvidia_dirs: List[str] = []
    sp_root = Path(prefix) / "lib"
    for child in sp_root.glob("python*/site-packages/nvidia/*/lib"):
        if not child.is_dir():
            continue
        # Torch is cu12x; skip accidental cu13 wheels that shadow the right runtime.
        if "/nvidia/cu13/" in str(child).replace("\\", "/"):
            continue
        nvidia_dirs.append(str(child.resolve()))
    # Prefer cudnn / cublas / cuda_runtime near the front of the search path.
    def _nv_prio(p: str) -> int:
        low = p.replace("\\", "/").lower()
        for i, key in enumerate(
            (
                "/nvidia/cudnn/",
                "/nvidia/cublas/",
                "/nvidia/cuda_runtime/",
                "/nvidia/cufft/",
                "/nvidia/cusolver/",
                "/nvidia/cusparse/",
                "/nvidia/nvjitlink/",
                "/nvidia/nccl/",
            )
        ):
            if key in low:
                return i
        return 100

    nvidia_dirs.sort(key=_nv_prio)
    lib_dir = os.path.join(prefix, "lib")
    if os.path.isdir(lib_dir):
        nvidia_dirs.insert(0, lib_dir)
    # Keep non-ROS entries from the parent so CUDA driver stubs remain reachable.
    for part in (os.environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep):
        if not part or any(m in part for m in markers):
            continue
        if part not in nvidia_dirs:
            nvidia_dirs.append(part)

    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHOME": None,
        "PYTHONPATH": os.pathsep.join(
            [str(EAI_DIR / "tools"), str(rlinf), str(rt)]
        ),
        "PATH": os.pathsep.join(path_parts),
        "LD_LIBRARY_PATH": os.pathsep.join(nvidia_dirs) if nvidia_dirs else "",
        "CONDA_PREFIX": prefix if os.path.isdir(prefix) else None,
        "AMENT_PREFIX_PATH": None,
        "ROS_DISTRO": None,
        "ROS_VERSION": None,
        "RLINF_ROOT": str(rlinf),
        "REPO_PATH": str(rlinf),
        "EMBODIED_PATH": str(rlinf / "examples" / "embodiment"),
        "ROBOTWIN_PATH": str(rt),
        "ASSETS_PATH": assets,
        "ROBOTWIN_ASSETS": assets,
        "ROBOT_PLATFORM": "ALOHA",
        "MUJOCO_GL": os.environ.get("MUJOCO_GL") or "osmesa",
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM") or "osmesa",
        "HYDRA_FULL_ERROR": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OPENPI_DISABLE_TORCH_COMPILE": os.environ.get(
            "OPENPI_DISABLE_TORCH_COMPILE", "1"
        ),
        # Mute Ray docker CPU warning in this containerized host.
        "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
    }


def build_robotwin_eval_argv(
    *,
    task: str = "adjust_bottle",
    policy_config: str = "pi05_aloha_robotwin",
    checkpoint: str = "",
    assets_path: str = "",
    num_envs: int = 1,
    max_episode_steps: int = 200,
    seed: int = 0,
    exp_name: str = "",
    python_bin: str = "",
    rlinf_root: str = "",
    robotwin_root: str = "",
    hydra_config: str = "",
    gpu: str = "0",
    render_gui: bool = False,
    rynnvalue_live_hud: bool = False,
    rynnvalue_server_url: str = "http://127.0.0.1:8001",
    rynnvalue_refresh_sec: float = 1.0,
    rynnvalue_num_frames: int = 8,
    rynnvalue_show_window: bool = True,
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    """Return (argv, cwd, env_extra) for RoboTwin RLinf eval.

    Launches ``examples/embodiment/eval_embodied_agent.py`` with Hydra overrides
    suitable for GUI smoke (1 env, single GPU placement).

    ``render_gui=True`` sets ``task_config.render_freq>0`` (SAPIEN Viewer) and
    forces ``num_envs=1``.

    ``rynnvalue_live_hud=True`` uses ``robotwin_eval_agent`` + frame bus so
    EnvWorker publishes RGB while the driver shows Live HUD.
    """
    py = resolve_robotwin_python(python_bin)
    rlinf = resolve_rlinf_root(rlinf_root)
    rt = resolve_robotwin_root(robotwin_root)
    assets = resolve_robotwin_assets(assets_path, robotwin_root=rt)
    ckpt = resolve_robotwin_checkpoint(checkpoint, policy_config)
    task_id = (task or "").strip() or "adjust_bottle"
    cfg_name = (hydra_config or "").strip() or hydra_config_for(task_id, ckpt.config)
    env_yaml = f"robotwin_{task_id}"

    exp = (exp_name or "").strip() or f"{task_id}_{ckpt.config}"
    out_dir = ROBOTWIN_EVAL_OUTPUT_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, int(num_envs))
    if render_gui and n_envs > 1:
        n_envs = 1

    wrapper = Path(__file__).resolve().parent / "run_robotwin_eval.py"
    argv = [
        py,
        str(wrapper),
        f"--task={task_id}",
        f"--policy_config={ckpt.config}",
        f"--hydra_config={cfg_name}",
        f"--env_config={env_yaml}",
        f"--model_path={ckpt.path}",
        f"--assets_path={assets}",
        f"--num_envs={n_envs}",
        f"--max_episode_steps={int(max_episode_steps)}",
        f"--seed={int(seed)}",
        f"--gpu={gpu}",
        f"--exp_name={exp}",
        f"--log_dir={out_dir}",
        f"--rlinf_root={rlinf}",
        f"--robotwin_root={rt}",
        "--render_gui" if render_gui else "--no_render_gui",
    ]
    from rynnvalue_sim_bridge import append_rynnvalue_live_hud_argv

    append_rynnvalue_live_hud_argv(
        argv,
        enabled=bool(rynnvalue_live_hud),
        server_url=rynnvalue_server_url,
        refresh_sec=rynnvalue_refresh_sec,
        num_frames=rynnvalue_num_frames,
        show_window=rynnvalue_show_window,
    )
    env_extra = robotwin_child_env(
        python_bin=py,
        rlinf_root=str(rlinf),
        robotwin_root=str(rt),
        assets_path=assets,
    )
    if render_gui or rynnvalue_live_hud:
        display = (os.environ.get("DISPLAY") or "").strip()
        if display:
            env_extra["DISPLAY"] = display
        xauth = (os.environ.get("XAUTHORITY") or "").strip()
        if xauth:
            env_extra["XAUTHORITY"] = xauth
    return argv, rlinf / "examples" / "embodiment", env_extra


def build_robotwin_viewer_argv(
    *,
    task: str = "adjust_bottle",
    task_config: str = "demo_clean",
    python_bin: str = "",
    robotwin_root: str = "",
    assets_path: str = "",
    seed: int = 0,
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    """Return (argv, cwd, env_extra) for SAPIEN viewer without policy inference."""
    py = resolve_robotwin_python(python_bin)
    rt = resolve_robotwin_root(robotwin_root)
    assets = resolve_robotwin_assets(assets_path, robotwin_root=rt)
    wrapper = Path(__file__).resolve().parent / "run_robotwin_viewer.py"
    task_id = (task or "").strip() or "adjust_bottle"
    cfg = (task_config or "").strip() or "demo_clean"
    argv = [
        py,
        str(wrapper),
        f"--task={task_id}",
        f"--task_config={cfg}",
        f"--robotwin_root={rt}",
        f"--assets_path={assets}",
        f"--seed={int(seed)}",
    ]
    env_extra = robotwin_child_env(
        python_bin=py,
        robotwin_root=str(rt),
        assets_path=assets,
    )
    display = (os.environ.get("DISPLAY") or "").strip()
    if display:
        env_extra["DISPLAY"] = display
    xauth = (os.environ.get("XAUTHORITY") or "").strip()
    if xauth:
        env_extra["XAUTHORITY"] = xauth
    return argv, rt, env_extra


def install_hint(python_bin: str = "") -> str:
    py = resolve_robotwin_python(python_bin)
    rlinf = resolve_rlinf_root()
    rt = resolve_robotwin_root()
    return (
        "RoboTwin 依赖未就绪（需要 RLinf embodied + RoboTwin RLinf_support + assets + openpi）。\n"
        f"  1) cd {rlinf} && bash requirements/install.sh embodied --model openpi --env robotwin\n"
        f"     或在当前 python 安装：{py} -m pip install 'git+https://github.com/RLinf/openpi'\n"
        f"  2) RoboTwin 切到 RLinf_support：\n"
        f"       cd {rt} && git fetch origin RLinf_support && git checkout RLinf_support\n"
        f"  3) 下载资产：cd {rt} && bash script/_download_assets.sh  （或设置 ASSETS_PATH）\n"
        f"  4) export ROBOTWIN_PATH={rt} ROBOT_PLATFORM=ALOHA ROBOTWIN_PYTHON={py}\n"
        f"  HF 权重示例：{ROBOTWIN_CKPT_HF_DEFAULT}"
    )


__all__ = [
    "ROBOTWIN_ASSETS_CANDIDATES",
    "ROBOTWIN_CKPT_HF_DEFAULT",
    "ROBOTWIN_CKPT_LOCAL_CANDIDATES",
    "ROBOTWIN_EVAL_OUTPUT_DIR",
    "ROBOTWIN_POLICY_CONFIGS",
    "ROBOTWIN_ROOT_DEFAULT",
    "ROBOTWIN_TASKS",
    "RobotwinCheckpoint",
    "build_robotwin_eval_argv",
    "build_robotwin_viewer_argv",
    "ensure_robotwin_checkpoint",
    "hydra_config_for",
    "install_hint",
    "probe_robotwin_deps",
    "resolve_robotwin_assets",
    "resolve_robotwin_checkpoint",
    "resolve_robotwin_python",
    "resolve_robotwin_root",
    "resolve_rlinf_root",
    "robotwin_child_env",
]
