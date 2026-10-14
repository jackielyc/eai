#!/usr/bin/env python3
"""Resolve / launch VPP2 (Video Prediction Policy 2) across EAI sim backends.

Upstream:
  https://github.com/roboterax/video-prediction-policy-2
  https://huggingface.co/Haodong082399/VPP2

RoboDojo XPolicyLab adapter: ``XPolicyLab/policy/VPP2``
  (installed via ``scripts/robodojo/install_adapter.sh`` or copied from
  ``video-prediction-policy-2/policy/VPP2``).

Policy id used in EAI UI combo boxes: ``vpp2``.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

EAI_DIR = Path(__file__).resolve().parent.parent
_LIY = Path("/share_data/projects/mahjong/share/personal/liyichao")

VPP2_POLICY_ID = "vpp2"
VPP2_POLICY_LABEL = "VPP2 (WAM)"

VPP2_CODE_ROOT_DEFAULT = _LIY / "video-prediction-policy-2"
ROBODOJO_ROOT_DEFAULT = _LIY / "RoboDojo"
VPP2_POLICY_REL = "XPolicyLab/policy/VPP2"
VPP2_WEIGHTS_DEFAULT = _LIY / "models" / "VPP2"

# Exactly one miniconda3 env for this stack (see tools/model_conda_envs.py).
VPP2_ENV_DEFAULT = str(_LIY / "miniconda3" / "envs" / "vpp2")
try:
    import sys as _sys

    _tools_dir = str(Path(__file__).resolve().parent)
    if _tools_dir not in _sys.path:
        _sys.path.insert(0, _tools_dir)
    from model_conda_envs import env_dir as _model_env_dir

    VPP2_ENV_DEFAULT = _model_env_dir("vpp2")
except Exception:  # noqa: BLE001
    pass
VPP2_ENV_CANDIDATES: Tuple[str, ...] = (VPP2_ENV_DEFAULT,)

# Backend → support level
#   robodojo_adapter: XPolicyLab VPP2 deploy
#   native_eval: scripts/libero/eval.sh or robodojo server/eval
#   serve_only: start policy server / note for custom clients
BACKEND_VPP2_SUPPORT: Dict[str, str] = {
    "isaac": "robodojo_adapter",
    "libero": "native_eval",
    "libero_plus": "native_eval",
    "libero_pro": "native_eval",
    "libero_mem": "native_eval",
    "robotwin": "serve_only",
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

DEFAULT_VPP2_PORT = 19960


@dataclass(frozen=True)
class Vpp2Paths:
    code_root: Path
    policy_root: Path
    weights_root: Path
    bundle: str  # joint2b_s100000 dir
    wan_root: str
    libero_dir: str
    python_bin: str
    conda_env: str
    source: str


def is_vpp2_policy(policy_id_or_dir: str) -> bool:
    raw = (policy_id_or_dir or "").strip()
    if not raw:
        return False
    low = raw.lower().replace("-", "_")
    if low in (VPP2_POLICY_ID, "vpp2", "video_prediction_policy_2"):
        return True
    if "vpp2" in low or "/policy/vpp2" in low:
        return True
    return False


def vpp2_support_level(backend: str) -> str:
    return BACKEND_VPP2_SUPPORT.get((backend or "").strip().lower(), "serve_only")


def resolve_vpp2_code_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("VPP2_ROOT", ""),
        str(VPP2_CODE_ROOT_DEFAULT),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c)).resolve()
        if (p / "scripts" / "robodojo" / "server.sh").is_file():
            return p
        if (p / "src" / "vpp2").is_dir():
            return p
    return VPP2_CODE_ROOT_DEFAULT


def resolve_robodojo_root(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("ROBODOJO_ROOT", ""),
        str(ROBODOJO_ROOT_DEFAULT),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c)).resolve()
        if (p / VPP2_POLICY_REL / "deploy.py").is_file():
            return p
        if (p / "XPolicyLab" / "policy").is_dir():
            return p
    return ROBODOJO_ROOT_DEFAULT


def resolve_vpp2_policy_root(raw: str = "", robodojo_root: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("VPP2_POLICY_DIR", ""),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c)).resolve()
        if (p / "deploy.py").is_file():
            return p
    rd = resolve_robodojo_root(robodojo_root)
    p = (rd / VPP2_POLICY_REL).resolve()
    if (p / "deploy.py").is_file():
        return p
    # Fall back to upstream adapter template
    upstream = resolve_vpp2_code_root() / "policy" / "VPP2"
    return upstream if (upstream / "deploy.py").is_file() else p


def resolve_vpp2_weights(raw: str = "") -> Path:
    for cand in (
        (raw or "").strip(),
        os.environ.get("VPP2_WEIGHTS", ""),
        str(VPP2_WEIGHTS_DEFAULT),
        str(EAI_DIR / ".cache" / "vpp2"),
        str(resolve_vpp2_code_root() / "weights"),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c)).resolve()
        if p.is_dir():
            return p
    return VPP2_WEIGHTS_DEFAULT


def resolve_vpp2_bundle(prefer: str = "", weights: Optional[Path] = None) -> str:
    w = weights or resolve_vpp2_weights()
    default = (w / "checkpoints" / "joint2b_s100000").resolve()
    for cand in (
        (prefer or "").strip(),
        os.environ.get("VPP2_BUNDLE", ""),
        str(default),
        str(resolve_vpp2_code_root() / "weights" / "joint2b_s100000"),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c))
        # Prefer short names like joint2b_s100000 under weights/checkpoints.
        if not p.is_absolute() and "/" not in c and "\\" not in c:
            p = w / "checkpoints" / c
        if (p / "action.pt").is_file() and (p / "video.pt").is_file():
            return str(p.resolve())
        if p.is_dir() and any(p.iterdir()):
            return str(p.resolve())
    return str(default)


def resolve_vpp2_wan_root(prefer: str = "", weights: Optional[Path] = None) -> str:
    w = weights or resolve_vpp2_weights()
    default = (w / "checkpoints" / "Wan2.1-I2V-14B-480P").resolve()
    for cand in (
        (prefer or "").strip(),
        os.environ.get("VPP2_WAN_ROOT", ""),
        str(default),
        str(w / "Wan2.1-I2V-14B-480P"),
        str(_LIY / "models" / "Wan2.1-I2V-14B-480P"),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c))
        if p.is_dir():
            return str(p.resolve())
    return str(default)


def resolve_vpp2_libero_dir(prefer: str = "", weights: Optional[Path] = None) -> str:
    w = weights or resolve_vpp2_weights()
    default = (w / "checkpoints" / "libero").resolve()
    for cand in (
        (prefer or "").strip(),
        os.environ.get("VPP2_LIBERO_DIR", ""),
        str(default),
        str(w / "libero"),
    ):
        c = (cand or "").strip()
        if not c:
            continue
        p = Path(os.path.expanduser(c))
        if p.is_dir():
            return str(p.resolve())
    return str(default)


def resolve_vpp2_python(prefer: str = "", conda_env: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("VPP2_PYTHON", ""),
        os.environ.get("PYTHON_BIN", ""),
    ):
        c = (cand or "").strip()
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    env_dir = (conda_env or "").strip()
    if env_dir and os.path.isdir(env_dir):
        py = os.path.join(env_dir, "bin", "python")
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    for env_path in VPP2_ENV_CANDIDATES:
        py = os.path.join(env_path, "bin", "python")
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    return shutil.which("python3") or "python3"


def resolve_vpp2_conda_env(prefer: str = "") -> str:
    for cand in (
        (prefer or "").strip(),
        os.environ.get("VPP2_CONDA_ENV", ""),
        VPP2_ENV_DEFAULT,
    ):
        c = (cand or "").strip()
        if not c:
            continue
        path = c if (os.path.sep in c or c.startswith("~")) else str(
            _LIY / "miniconda3" / "envs" / c
        )
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(path):
            return path
    return VPP2_ENV_DEFAULT


def resolve_paths(
    *,
    prefer_code: str = "",
    prefer_weights: str = "",
    prefer_bundle: str = "",
    prefer_wan: str = "",
    prefer_libero: str = "",
    python_bin: str = "",
    conda_env: str = "",
) -> Vpp2Paths:
    code = resolve_vpp2_code_root(prefer_code)
    weights = resolve_vpp2_weights(prefer_weights)
    policy = resolve_vpp2_policy_root()
    env = resolve_vpp2_conda_env(conda_env)
    py = resolve_vpp2_python(python_bin, conda_env=env)
    bundle = resolve_vpp2_bundle(prefer_bundle, weights)
    wan = resolve_vpp2_wan_root(prefer_wan, weights)
    libero = resolve_vpp2_libero_dir(prefer_libero, weights)
    src = "ok" if (policy / "deploy.py").is_file() else "missing_adapter"
    return Vpp2Paths(
        code_root=code,
        policy_root=policy,
        weights_root=weights,
        bundle=bundle,
        wan_root=wan,
        libero_dir=libero,
        python_bin=py,
        conda_env=env,
        source=src,
    )


def probe_vpp2(paths: Optional[Vpp2Paths] = None) -> Tuple[bool, str]:
    p = paths or resolve_paths()
    missing: List[str] = []
    if not (p.code_root / "src" / "vpp2").is_dir():
        missing.append(f"code@{p.code_root}")
    if not (p.policy_root / "deploy.py").is_file():
        missing.append(f"adapter@{p.policy_root}")
    bundle = Path(p.bundle)
    if not ((bundle / "action.pt").is_file() and (bundle / "video.pt").is_file()):
        missing.append(f"bundle={p.bundle}")
    elif not (bundle / "dataset_stats.json").is_file():
        missing.append(f"stats@{bundle / 'dataset_stats.json'}")
    wan = Path(p.wan_root)
    for name in (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ):
        if not (wan / name).is_file():
            missing.append(f"wan/{name}")
    if not (wan / "google" / "umt5-xxl").is_dir():
        missing.append("wan/google/umt5-xxl")
    libero = Path(p.libero_dir)
    libero_ok = (libero / "action_step030000.pt").is_file() and (
        libero / "video_step010000.pt"
    ).is_file()
    if missing:
        return False, "MISSING " + ", ".join(missing)
    note = "libero_ok" if libero_ok else f"libero_pending={p.libero_dir}"
    return True, f"OK code={p.code_root} bundle={p.bundle} {note}"


def ensure_adapter_installed(robodojo_root: str = "") -> Tuple[bool, str]:
    """Copy upstream policy/VPP2 into RoboDojo if missing."""
    rd = resolve_robodojo_root(robodojo_root)
    dest = rd / VPP2_POLICY_REL
    if not (dest / "deploy.py").is_file():
        src = resolve_vpp2_code_root() / "policy" / "VPP2"
        if not (src / "deploy.py").is_file():
            return False, f"upstream adapter missing: {src}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("deploy.py", "deploy.yml", "model.py", "__init__.py"):
            s = src / name
            if s.is_file():
                shutil.copy2(s, dest / name)
    ok_mat, mat_msg = materialize_deploy_yml(robodojo_root=str(rd))
    if not ok_mat:
        return (dest / "deploy.py").is_file(), f"{dest} ({mat_msg})"
    return (dest / "deploy.py").is_file(), str(dest)


def materialize_deploy_yml(
    *,
    robodojo_root: str = "",
    prefer_bundle: str = "",
    prefer_wan: str = "",
) -> Tuple[bool, str]:
    """Write absolute checkpoint/wan paths into RoboDojo VPP2 deploy.yml.

    Official ``vpp2.cli server`` passes these via ``--overrides``; for EAI's
    ``robodojo.sh`` path we bake them into deploy.yml so Isaac deploy works.
    """
    rd = resolve_robodojo_root(robodojo_root)
    yml = rd / VPP2_POLICY_REL / "deploy.yml"
    if not yml.is_file():
        return False, f"missing {yml}"
    paths = resolve_paths(prefer_bundle=prefer_bundle, prefer_wan=prefer_wan)
    bundle = Path(paths.bundle)
    action = bundle / "action.pt"
    video = bundle / "video.pt"
    stats = bundle / "dataset_stats.json"
    wan = Path(paths.wan_root)
    # Keep yaml even if weights are still downloading — paths are absolute.
    text = yml.read_text(encoding="utf-8")
    replacements = {
        "checkpoint_path:": f"checkpoint_path: {action}",
        "video_checkpoint_path:": f"video_checkpoint_path: {video}",
        "dataset_stats_path:": f"dataset_stats_path: {stats}",
        "wan_model_dir:": f"wan_model_dir: {wan}",
        "ckpt_name:": f"ckpt_name: {bundle.name}",
        "port:": f"port: {DEFAULT_VPP2_PORT}",
    }
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        key = next((k for k in replacements if stripped.startswith(k)), None)
        if key is None:
            lines.append(line)
            continue
        indent = line[: len(line) - len(stripped)]
        lines.append(f"{indent}{replacements[key]}")
    yml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [str(p) for p in (action, video, stats, wan) if not p.exists()]
    if missing:
        return False, "deploy.yml updated; waiting weights: " + ", ".join(missing)
    return True, str(yml)


def vpp2_child_env(
    paths: Vpp2Paths,
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
        str(paths.code_root / "src"),
        str(paths.code_root),
        str(resolve_robodojo_root()),
        str(resolve_robodojo_root() / "XPolicyLab"),
    ]
    env: Dict[str, Optional[str]] = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHOME": None,
        "PYTHONPATH": os.pathsep.join(py_path),
        "PATH": os.pathsep.join(path_parts),
        "CONDA_PREFIX": prefix if os.path.isdir(prefix) else None,
        "AMENT_PREFIX_PATH": None,
        "ROS_DISTRO": None,
        "VPP2_ROOT": str(paths.code_root),
        "VPP2_WEIGHTS": str(paths.weights_root),
        "VPP2_BUNDLE": paths.bundle,
        "VPP2_WAN_ROOT": paths.wan_root,
        "VPP2_LIBERO_DIR": paths.libero_dir,
        "PYTHON_BIN": py,
        "TOKENIZERS_PARALLELISM": "false",
    }
    if extra:
        env.update(extra)
    return env


def build_robodojo_server_argv(
    *,
    port: int = DEFAULT_VPP2_PORT,
    gpu: str = "0",
    python_bin: str = "",
    robodojo_root: str = "",
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    paths = resolve_paths(python_bin=python_bin)
    ensure_adapter_installed(robodojo_root)
    script = paths.code_root / "scripts" / "robodojo" / "server.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    rd = resolve_robodojo_root(robodojo_root)
    argv = [
        "bash",
        str(script),
        f"--robodojo-root={rd}",
        f"--bundle={paths.bundle}",
        f"--wan={paths.wan_root}",
        f"--gpu={gpu}",
        f"--port={int(port)}",
    ]
    env = vpp2_child_env(
        paths,
        extra={
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VPP2_SIM_ROOT": str(rd),
        },
    )
    return argv, paths.code_root, env


def build_libero_eval_argv(
    *,
    config: str = "configs/libero/eval_standard.yaml",
    gpus: str = "0",
    python_bin: str = "",
) -> Tuple[List[str], Path, Dict[str, Optional[str]]]:
    paths = resolve_paths(python_bin=python_bin)
    script = paths.code_root / "scripts" / "libero" / "eval.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    cfg = (config or "").strip() or "configs/libero/eval_standard.yaml"
    # Prefer absolute config under code root
    cfg_path = Path(cfg)
    if not cfg_path.is_file():
        cfg_path = paths.code_root / cfg
    argv = [
        "bash",
        str(script),
        f"--config={cfg_path}",
        f"--gpus={gpus}",
    ]
    env = vpp2_child_env(paths)
    # Official eval expects weights under code_root/weights/libero via symlinks;
    # expose env so users can override in their configs.
    env["VPP2_LIBERO_VIDEO"] = str(Path(paths.libero_dir) / "video_step010000.pt")
    env["VPP2_LIBERO_ACTION"] = str(Path(paths.libero_dir) / "action_step030000.pt")
    return argv, paths.code_root, env


def build_backend_vpp2_argv(
    backend: str,
    *,
    python_bin: str = "",
    port: Optional[int] = None,
    gpu: str = "0",
    task: str = "",
    task_suite: str = "",
) -> Tuple[List[str], Path, Dict[str, Optional[str]], str]:
    key = (backend or "").strip().lower()
    level = vpp2_support_level(key)
    paths = resolve_paths(python_bin=python_bin)
    ensure_adapter_installed()

    if key.startswith("libero") or (level == "native_eval" and key.startswith("libero")):
        cfg = "configs/libero/eval_standard.yaml"
        if "pro" in key:
            cfg = "configs/libero/eval_pro.yaml"
        elif "plus" in key or "ood" in (task_suite or task or "").lower():
            cfg = "configs/libero/eval_ood.yaml"
        argv, cwd, env = build_libero_eval_argv(
            config=cfg, gpus=gpu, python_bin=python_bin
        )
        return argv, cwd, env, f"VPP2 native LIBERO eval · {cfg}"

    if key == "isaac" or level == "robodojo_adapter":
        # Prefer starting the official VPP2 policy server; Isaac client uses
        # XPolicyLab VPP2 via robodojo.sh / run_gui_eval.sh.
        argv, cwd, env = build_robodojo_server_argv(
            port=int(port or DEFAULT_VPP2_PORT),
            gpu=gpu,
            python_bin=python_bin,
        )
        return (
            argv,
            cwd,
            env,
            "VPP2 RoboDojo: start policy server; use XPolicyLab policy=VPP2 + attach_external / 一体启动",
        )

    # serve_only / extended
    argv, cwd, env = build_robodojo_server_argv(
        port=int(port or DEFAULT_VPP2_PORT),
        gpu=gpu,
        python_bin=python_bin,
    )
    note = (
        f"VPP2 serve_only for backend={key}. "
        f"policy server :{port or DEFAULT_VPP2_PORT}; "
        "该后端无官方 closed-loop adapter，优先用 Isaac/LIBERO。"
    )
    return argv, cwd, env, note


def robodojo_policy_rel() -> str:
    return VPP2_POLICY_REL


def install_hint(python_bin: str = "") -> str:
    py = resolve_vpp2_python(python_bin)
    code = resolve_vpp2_code_root()
    weights = resolve_vpp2_weights()
    return (
        "VPP2 依赖未就绪。安装与配置示例：\n"
        f"  1) 代码: {code} （已 git clone video-prediction-policy-2）\n"
        f"  2) conda create -n vpp2 python=3.10 && conda activate vpp2\n"
        f"     cd {code} && pip install -r requirements.txt\n"
        f"  3) 权重: {weights}\n"
        "     需含 checkpoints/joint2b_s100000/{{video,action}}.pt\n"
        "          checkpoints/Wan2.1-I2V-14B-480P/\n"
        "          checkpoints/libero/{{video_step010000,action_step030000}}.pt\n"
        f"  4) Isaac: 策略选 VPP2，action=ee，ckpt=joint2b_s100000\n"
        f"  5) LIBERO: 配置选 VPP2，走 scripts/libero/eval.sh\n"
        f"  PYTHON: {py}\n"
        "  HF: https://huggingface.co/Haodong082399/VPP2"
    )


def append_policy_combo_entries(
    entries: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    out = list(entries)
    ids = {e[1] for e in out}
    if VPP2_POLICY_ID not in ids and "vpp2" not in {
        x.lower().replace("-", "_") for x in ids
    }:
        out.append((VPP2_POLICY_LABEL, VPP2_POLICY_ID))
    return out


__all__ = [
    "BACKEND_VPP2_SUPPORT",
    "DEFAULT_VPP2_PORT",
    "VPP2_POLICY_ID",
    "VPP2_POLICY_LABEL",
    "VPP2_POLICY_REL",
    "VPP2_WEIGHTS_DEFAULT",
    "Vpp2Paths",
    "append_policy_combo_entries",
    "build_backend_vpp2_argv",
    "build_libero_eval_argv",
    "build_robodojo_server_argv",
    "ensure_adapter_installed",
    "install_hint",
    "is_vpp2_policy",
    "materialize_deploy_yml",
    "probe_vpp2",
    "resolve_paths",
    "resolve_vpp2_bundle",
    "resolve_vpp2_code_root",
    "resolve_vpp2_conda_env",
    "resolve_vpp2_libero_dir",
    "resolve_vpp2_policy_root",
    "resolve_vpp2_python",
    "resolve_vpp2_wan_root",
    "resolve_vpp2_weights",
    "robodojo_policy_rel",
    "vpp2_child_env",
    "vpp2_support_level",
]
