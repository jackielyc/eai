"""Launch / probe Psi-MolmoSpaces remote policy servers from the EAI GUI.

Supports:
- Psi Policy (legacy fingertip / openpi-compatible websocket)
- Psi Policy Native6D (ruiyan motor joints)
- WAM Psi-R1 Remote (psi_wam_inference protocol via serve_wam82_policy)
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from typing import Dict, List, Optional, Tuple


PSI_POLICY_EVAL_ID = (
    "molmo_spaces.evaluation.configs.evaluation_configs:PsiPolicyEvalConfig"
)
PSI_NATIVE6D_EVAL_ID = (
    "molmo_spaces.evaluation.configs.psi_native6d_eval_config:PsiNative6DPolicyEvalConfig"
)
WAM_PSI_R1_EVAL_ID = (
    "molmo_spaces.configs.wam_psi_r1_config:WamPsiR1RemotePickEvalConfig"
)

PSI_REMOTE_MODES = ("psi_policy", "native6d", "wam82")

PSI_POLICY_PORT_DEFAULT = int(os.environ.get("PSI_POLICY_SERVER_PORT") or 8080)
PSI_NATIVE6D_PORT_DEFAULT = int(os.environ.get("PSI_NATIVE6D_SERVER_PORT") or 8081)
WAM_PSI_R1_PORT_DEFAULT = int(os.environ.get("WAM_POLICY_SERVER_PORT") or 9317)

_LIY = "/share_data/projects/mahjong/share/personal/liyichao"
WAN22_DEFAULT_CANDIDATES = (
    os.environ.get("WAN22_TI2V_5B_CHECKPOINT_DIR", "").strip(),
    f"{_LIY}/models/Wan2.2-TI2V-5B",
    "/share_data_prj/maobo/wam_train_eval/Wan2.2-TI2V-5B",
)
# Local maobo WAM checkpoint mirrored under models/; prefer this when present.
WAM_FINETUNE_CKPT_LOCAL = (
    f"{_LIY}/models/psi-molospaces-maobo-step=00022000-train_loss=0.018139.pt"
)
# Songzhe WAM-finetune pick_apple_tomato checkpoint used in reproduction evals.
WAM_FINETUNE_CKPT_NAME = "step=00016000-train_loss=0.349923.pt"
WAM_FINETUNE_CKPT_REL = (
    "train_runs/pick_apple_tomato_856_native6d_quantile_songzhe_repro_wam_finetune_4gpu/"
    "checkpoints/policy/topk/step=00016000-train_loss=0.349923.pt"
)
WAM_TRAIN_EVAL_ROOT_CANDIDATES = (
    os.environ.get("WAM_TRAIN_EVAL_ROOT", "").strip(),
    os.environ.get("PSI_REMOTE_CKPT_ROOT", "").strip(),
    "/share_data_prj/maobo/wam_train_eval",
    f"{_LIY}/wam_train_eval",
    f"{_LIY}/psi-policy/wam_train_eval",
    "/share_data/projects/mahjong/share/personal/maobo/wam_train_eval",
)
WAM_FINETUNE_CKPT_ABSOLUTE_CANDIDATES = (
    os.environ.get("PSI_REMOTE_CKPT", "").strip(),
    os.environ.get("WAM_FINETUNE_CKPT", "").strip(),
    WAM_FINETUNE_CKPT_LOCAL,
    "/share_data_prj/maobo/wam_train_eval/"
    "train_runs/pick_apple_tomato_856_native6d_quantile_songzhe_repro_wam_finetune_4gpu/"
    "checkpoints/policy/topk/step=00016000-train_loss=0.349923.pt",
)
# Exactly one miniconda3 env for Psi Policy (see tools/model_conda_envs.py).
PSI_POLICY_PYTHON_CANDIDATES = (
    os.environ.get("PSI_POLICY_PYTHON", "").strip(),
    f"{_LIY}/miniconda3/envs/psi-policy/bin/python",
)
PSI_POLICY_ROOT_CANDIDATES = (
    os.environ.get("PSI_POLICY_ROOT", "").strip(),
    os.environ.get("PSI_POLICY_DIR", "").strip(),
    f"{_LIY}/psi-policy",
    "/home/maobo/psi-policy",
)


def is_psi_policy_id(policy_id: str) -> bool:
    return "PsiPolicyEvalConfig" in (policy_id or "")


def is_psi_native6d_policy_id(policy_id: str) -> bool:
    return "PsiNative6DPolicyEvalConfig" in (policy_id or "")


def is_wam_psi_r1_policy_id(policy_id: str) -> bool:
    return "WamPsiR1RemotePickEvalConfig" in (policy_id or "")


def needs_psi_remote_deploy(policy_id: str) -> bool:
    return (
        is_psi_policy_id(policy_id)
        or is_psi_native6d_policy_id(policy_id)
        or is_wam_psi_r1_policy_id(policy_id)
    )


def psi_remote_mode_for_policy(policy_id: str) -> str:
    if is_wam_psi_r1_policy_id(policy_id):
        return "wam82"
    if is_psi_native6d_policy_id(policy_id):
        return "native6d"
    if is_psi_policy_id(policy_id):
        return "psi_policy"
    return ""


def default_port_for_mode(mode: str) -> int:
    if mode == "wam82":
        return WAM_PSI_R1_PORT_DEFAULT
    if mode == "native6d":
        return PSI_NATIVE6D_PORT_DEFAULT
    return PSI_POLICY_PORT_DEFAULT


def resolve_psi_policy_root(user: str = "") -> str:
    candidates = [user, *PSI_POLICY_ROOT_CANDIDATES]
    for raw in candidates:
        path = os.path.abspath(os.path.expanduser((raw or "").strip()))
        if path and os.path.isfile(os.path.join(path, "psi_policy", "train.py")):
            return path
        if path and os.path.isfile(
            os.path.join(path, "psi_policy", "molmospaces_adapter", "serve_wam82_policy.py")
        ):
            return path
    return ""


def resolve_psi_policy_python(repo_dir: str = "") -> str:
    for raw in PSI_POLICY_PYTHON_CANDIDATES:
        path = os.path.abspath(os.path.expanduser((raw or "").strip()))
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    if repo_dir:
        for rel in (".venv/bin/python", "venv/bin/python"):
            cand = os.path.join(repo_dir, rel)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return "python3"


def _is_wan22_checkpoint_dir(path: str) -> bool:
    """True only for a real Wan2.2-TI2V-5B tree (must contain Wan2.2_VAE.pth)."""
    root = (path or "").strip()
    if not root or not os.path.isdir(root):
        return False
    return os.path.isfile(os.path.join(root, "Wan2.2_VAE.pth"))


def resolve_wan22_dir(user: str = "") -> str:
    # Skip empty strings: abspath("") == cwd and must never be treated as Wan root.
    for raw in (user, *WAN22_DEFAULT_CANDIDATES):
        text = (raw or "").strip()
        if not text:
            continue
        path = os.path.abspath(os.path.expanduser(text))
        if _is_wan22_checkpoint_dir(path):
            return path
    return ""


def resolve_default_wam_finetune_checkpoint(user: str = "") -> str:
    """Resolve the Songzhe WAM-finetune step=00016000 checkpoint.

    Prefers an existing file among env overrides / known train_eval roots.
    Falls back to the canonical maobo path even if currently missing, so the
    GUI can show the expected default.
    """
    candidates: list[str] = []
    if user:
        candidates.append(user)
    candidates.extend(WAM_FINETUNE_CKPT_ABSOLUTE_CANDIDATES)
    for root in WAM_TRAIN_EVAL_ROOT_CANDIDATES:
        if not root:
            continue
        candidates.append(os.path.join(root, WAM_FINETUNE_CKPT_REL))
        candidates.append(
            os.path.join(
                root,
                "checkpoints",
                "policy",
                "topk",
                WAM_FINETUNE_CKPT_NAME,
            )
        )
    # Also accept a bare .pt dropped into local models/.
    candidates.append(WAM_FINETUNE_CKPT_LOCAL)
    candidates.append(os.path.join(_LIY, "models", WAM_FINETUNE_CKPT_NAME))
    candidates.append(os.path.join(_LIY, "eai", ".cache", "wam", WAM_FINETUNE_CKPT_NAME))

    existing = ""
    canonical = ""
    for raw in candidates:
        text = (raw or "").strip()
        if not text:
            continue
        path = os.path.abspath(os.path.expanduser(text))
        base = os.path.basename(path)
        if not canonical and (
            base == os.path.basename(WAM_FINETUNE_CKPT_LOCAL)
            or WAM_FINETUNE_CKPT_NAME in base
        ):
            canonical = path
        if os.path.isfile(path):
            existing = path
            break
    if existing:
        return existing
    if canonical:
        return canonical
    return os.path.abspath(os.path.expanduser(WAM_FINETUNE_CKPT_LOCAL))


def resolve_checkpoint_path(path: str) -> str:
    raw = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not raw:
        return ""
    if os.path.isfile(raw):
        return raw
    if not os.path.isdir(raw):
        return raw
    preferred = (
        os.path.join(raw, "checkpoints", "policy", "latest.pt"),
        os.path.join(raw, "policy", "latest.pt"),
        os.path.join(raw, "latest.pt"),
    )
    for cand in preferred:
        if os.path.isfile(cand):
            return cand
    newest = ""
    newest_mtime = -1.0
    for root, _dirs, files in os.walk(raw):
        for name in files:
            if not name.endswith(".pt"):
                continue
            full = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = full
    return newest or raw


def port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _pids_listening_on_port(port: int) -> list[int]:
    """Best-effort: return PIDs that have a TCP listen socket on ``port``."""
    port = int(port)
    pids: list[int] = []
    try:
        proc = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    for match in re.finditer(r"pid=(\d+)", text):
        try:
            pid = int(match.group(1))
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def listener_cmdline_on_port(port: int) -> str:
    """Return the first listener process cmdline on ``port``, or empty string."""
    for pid in _pids_listening_on_port(port):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw = fh.read()
            if raw:
                return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            continue
    return ""


def _listener_environ_value(port: int, key: str) -> str:
    for pid in _pids_listening_on_port(port):
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        for item in raw.split(b"\0"):
            if not item or b"=" not in item:
                continue
            k, _, v = item.partition(b"=")
            if k.decode("utf-8", errors="replace") == key:
                return v.decode("utf-8", errors="replace")
    return ""


def wam82_server_needs_restart(port: int) -> bool:
    """True if :port hosts a stale/polluted serve_wam82 that must be relaunched."""
    cmd = listener_cmdline_on_port(port)
    if not cmd:
        return False
    if "serve_wam82_policy" not in cmd:
        return False
    # Pre-fix: missing local prompt embedding dir remap.
    if "--prompt-embedding-dir" not in cmd:
        return True
    # After ``env -u LD_PRELOAD … exec python``, cmdline no longer contains
    # ``env -u`` — judge pollution from the live process environ instead.
    preload = _listener_environ_value(port, "LD_PRELOAD")
    if "vglfaker" in preload or "dlfaker" in preload:
        return True
    return False


def probe_openpi_compatible_server(
    host: str,
    port: int,
    *,
    python_bin: str = "",
    timeout_s: float = 6.0,
) -> Tuple[bool, str]:
    """Probe openpi-style metadata handshake (Psi Policy / Native6D)."""
    host = (host or "localhost").strip() or "localhost"
    port = int(port)
    py = (python_bin or "").strip() or "python3"
    code = (
        "import sys\n"
        "host, port = sys.argv[1], int(sys.argv[2])\n"
        "uri = f'ws://{host}:{port}'\n"
        "try:\n"
        "    from openpi_client.websocket_client_policy import WebsocketClientPolicy\n"
        "    meta = WebsocketClientPolicy(host=host, port=port).get_server_metadata()\n"
        "    name = ''\n"
        "    if isinstance(meta, dict):\n"
        "        name = str(meta.get('policy_name') or meta.get('policy_checkpoint') or '')\n"
        "    print('OK', name or 'metadata')\n"
        "except Exception as exc:\n"
        "    import websockets.sync.client as ws_client\n"
        "    with ws_client.connect(uri, compression=None, max_size=None, open_timeout=5) as conn:\n"
        "        raw = conn.recv(timeout=5)\n"
        "        print('OK', type(raw).__name__, str(exc)[:80])\n"
    )
    try:
        proc = subprocess.run(
            [py, "-c", code, host, str(port)],
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_s)),
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
    except subprocess.TimeoutExpired:
        return False, f"探测超时 ws://{host}:{port}"
    except OSError as exc:
        return False, f"无法启动探测进程: {exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    for line in reversed(out.splitlines()):
        if line.startswith("OK"):
            detail = line[2:].strip()
            return True, f"ws://{host}:{port} 可用" + (f" ({detail})" if detail else "")
    return False, out or f"探测失败 code={proc.returncode}"


def probe_wam_server(
    host: str,
    port: int,
    *,
    python_bin: str = "",
    timeout_s: float = 8.0,
) -> Tuple[bool, str]:
    """Probe psi_wam_inference handshake used by serve_wam82_policy."""
    host = (host or "localhost").strip() or "localhost"
    port = int(port)
    py = (python_bin or "").strip() or "python3"
    code = (
        "import sys\n"
        "host, port = sys.argv[1], int(sys.argv[2])\n"
        "uri = f'ws://{host}:{port}'\n"
        "import msgpack_numpy\n"
        "import websockets.sync.client as ws_client\n"
        "with ws_client.connect(uri, compression=None, max_size=None, open_timeout=5) as conn:\n"
        "    raw = conn.recv(timeout=5)\n"
        "    meta = msgpack_numpy.unpackb(raw)\n"
        "    proto = meta.get('protocol') if isinstance(meta, dict) else None\n"
        "    if proto != 'psi_wam_inference':\n"
        "        raise RuntimeError(f'unexpected protocol={proto!r}')\n"
        "    print('OK', meta.get('server_address') or proto)\n"
    )
    try:
        proc = subprocess.run(
            [py, "-c", code, host, str(port)],
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_s)),
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
    except subprocess.TimeoutExpired:
        return False, f"WAM 探测超时 ws://{host}:{port}"
    except OSError as exc:
        return False, f"无法启动 WAM 探测进程: {exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    for line in reversed(out.splitlines()):
        if line.startswith("OK"):
            detail = line[2:].strip()
            return True, f"ws://{host}:{port} WAM 可用" + (f" ({detail})" if detail else "")
    return False, out or f"WAM 探测失败 code={proc.returncode}"


def probe_psi_remote_server(
    mode: str,
    host: str,
    port: int,
    *,
    python_bin: str = "",
    timeout_s: float = 8.0,
) -> Tuple[bool, str]:
    if mode == "wam82":
        return probe_wam_server(host, port, python_bin=python_bin, timeout_s=timeout_s)
    return probe_openpi_compatible_server(
        host, port, python_bin=python_bin, timeout_s=timeout_s
    )


def build_psi_remote_server_argv(
    *,
    mode: str,
    checkpoint: str,
    molmospaces_root: str,
    psi_policy_root: str = "",
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    device: str = "cuda:0",
    execute_step: Optional[int] = None,
    prompt_text: str = "Pick up the apple",
    gpu_id: str = "0",
) -> Tuple[List[str], str, Dict[str, Optional[str]], str]:
    """Return (argv, cwd, env_extra, label).

    ``env_extra`` values of ``None`` mean ``QProcessEnvironment.remove`` so the
    child does not inherit GUI/ROS/VirtualGL CUDA pollution (common cause of
    ``Invalid handle. Cannot load symbol cudnnGetVersion`` when loading Wan VAE).
    """
    mode = (mode or "").strip()
    if mode not in PSI_REMOTE_MODES:
        raise ValueError(f"unsupported psi remote mode: {mode!r}")
    ckpt = resolve_checkpoint_path(checkpoint)
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    port = int(port if port is not None else default_port_for_mode(mode))
    molmo_root = os.path.abspath(os.path.expanduser(molmospaces_root))
    if not os.path.isdir(molmo_root):
        raise FileNotFoundError(f"psi-molmospaces 不存在: {molmo_root}")
    wan_dir = resolve_wan22_dir()
    policy_root = resolve_psi_policy_root(psi_policy_root)
    python_bin = resolve_psi_policy_python(policy_root)
    # Strip VirtualGL / ROS library injection inherited from scripts/launch/run_local.sh GUI.
    env: Dict[str, Optional[str]] = {
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu_id).strip() or "0",
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "MUJOCO_USE_VGL": None,
        "MUJOCO_VGL_DEVICE": None,
    }
    if wan_dir:
        env["WAN22_TI2V_5B_CHECKPOINT_DIR"] = wan_dir
    if policy_root:
        env["PSI_POLICY_ROOT"] = policy_root

    if mode == "wam82":
        if not wan_dir:
            raise FileNotFoundError(
                "未找到 Wan2.2-TI2V-5B（目录内需有 Wan2.2_VAE.pth）。"
                "请设置 WAN22_TI2V_5B_CHECKPOINT_DIR，"
                f"或放置到 {_LIY}/models/Wan2.2-TI2V-5B"
            )
        if not policy_root:
            raise FileNotFoundError(
                "未找到 psi-policy（需含 molmospaces_adapter/serve_wam82_policy.py）"
            )
        step = int(execute_step if execute_step is not None else 16)
        prompt_emb_dir = os.environ.get("WAM_PROMPT_EMBEDDING_DIR", "").strip() or (
            f"{_LIY}/eai/.cache/wam/prompt_embeddings"
        )
        try:
            os.makedirs(prompt_emb_dir, exist_ok=True)
        except OSError:
            pass
        # Use /usr/bin/env -u so VirtualGL LD_PRELOAD cannot leak in even if the
        # GUI QProcessEnvironment scrubbing is incomplete (causes
        # "Cannot load symbol cudnnGetVersion" on Wan VAE load).
        argv = [
            "/usr/bin/env",
            "-u",
            "LD_PRELOAD",
            "-u",
            "LD_LIBRARY_PATH",
            "-u",
            "MUJOCO_USE_VGL",
            python_bin,
            "-u",
            "-m",
            "psi_policy.molmospaces_adapter.serve_wam82_policy",
            "--checkpoint",
            ckpt,
            "--device",
            device,
            "--host",
            host,
            "--port",
            str(port),
            "--execute-step",
            str(step),
            "--num-inference-steps",
            "16",
            "--action-samples",
            "1",
            "--seed",
            "42",
            "--prompt-source",
            "task_description",
            "--prompt-embedding-dir",
            prompt_emb_dir,
        ]
        env["WAM_PROMPT_EMBEDDING_DIR"] = prompt_emb_dir
        env["PYTHONPATH"] = policy_root
        label = f"wam82 serve:{os.path.basename(ckpt)}"
        return argv, policy_root, env, label

    server_dir = os.path.join(molmo_root, "psi-policy-server")
    if not os.path.isdir(server_dir):
        raise FileNotFoundError(f"缺少 psi-policy-server: {server_dir}")
    if mode == "native6d":
        script = os.path.join(server_dir, "scripts", "serve_policy_psi_policy_native6d.py")
        step = int(execute_step if execute_step is not None else 16)
        action_dim = 82
        label = f"psi native6d:{os.path.basename(ckpt)}"
    else:
        script = os.path.join(server_dir, "scripts", "serve_policy_psi_policy.py")
        step = int(execute_step if execute_step is not None else 8)
        action_dim = 82
        label = f"psi policy:{os.path.basename(ckpt)}"
    if not os.path.isfile(script):
        raise FileNotFoundError(f"缺少服务脚本: {script}")
    if not policy_root:
        raise FileNotFoundError("未找到 PSI_POLICY_ROOT（微调 psi-policy 仓库）")
    argv = [
        python_bin,
        "-u",
        script,
        f"--host={host}",
        f"--port={port}",
        f"--device={device}",
        f"--action-dim={action_dim}",
        f"--execute-step={step}",
        f"--prompt-text={prompt_text}",
        "policy:checkpoint",
        "--policy.config=psi_r1",
        f"--policy.dir={ckpt}",
    ]
    return argv, server_dir, env, label


__all__ = [
    "PSI_NATIVE6D_EVAL_ID",
    "PSI_NATIVE6D_PORT_DEFAULT",
    "PSI_POLICY_EVAL_ID",
    "PSI_POLICY_PORT_DEFAULT",
    "PSI_REMOTE_MODES",
    "WAM_FINETUNE_CKPT_NAME",
    "WAM_PSI_R1_EVAL_ID",
    "WAM_PSI_R1_PORT_DEFAULT",
    "build_psi_remote_server_argv",
    "default_port_for_mode",
    "is_psi_native6d_policy_id",
    "is_psi_policy_id",
    "is_wam_psi_r1_policy_id",
    "needs_psi_remote_deploy",
    "listener_cmdline_on_port",
    "port_listening",
    "probe_psi_remote_server",
    "psi_remote_mode_for_policy",
    "resolve_checkpoint_path",
    "resolve_default_wam_finetune_checkpoint",
    "resolve_psi_policy_python",
    "resolve_psi_policy_root",
    "resolve_wan22_dir",
    "wam82_server_needs_restart",
]
