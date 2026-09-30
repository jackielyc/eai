#!/usr/bin/env python3
"""Resolve / probe / serve OpenPI (Pi) checkpoints for MolmoSpaces eval.

MolmoSpaces PiPolicy connects to ws://localhost:8080 by default. This module:
  - picks a usable checkpoint (local non-sparse dir, openpi cache, or gs:// URL)
  - builds ``serve_policy.py`` argv
  - probes the websocket server (metadata handshake)
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

EAI_DIR = Path(__file__).resolve().parent.parent
OPENPI_ROOT_DEFAULT = Path(
    "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/openpi"
)
OPENPI_PYTHON_CANDIDATES: Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/openpi/bin/python",
    "/home/psibot/miniconda3/envs/openpi/bin/python",
)
OPENPI_DATA_HOME_DEFAULT = str(EAI_DIR / ".cache" / "openpi")
# MolmoSpaces eval default: joint-position π0.5-DROID.
PI_CKPT_CONFIG_DEFAULT = "pi05_droid_jointpos"
PI_CKPT_GS_DEFAULT = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
PI_CKPT_HF_DEFAULT = "Kovavavvavava/pi05_droid_jointpos"
PI_CKPT_HF_LOCAL_DEFAULT = str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_droid_jointpos")
# openpi JAX pi05_libero mirror (GCS often fails without gsutil/anon).
PI05_LIBERO_HF_DEFAULT = "bf-jeon/pi05_libero"
PI05_LIBERO_HF_LOCAL_DEFAULT = str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_libero")
PI_SERVER_HOST_DEFAULT = "localhost"
PI_SERVER_PORT_DEFAULT = 8080
# Local dirs that look like Pi ckpts (may be sparse / incomplete on this host).
# Prefer models/ then eai HF/GCS cache; UI defaults to first present local.
PI_CKPT_LOCAL_CANDIDATES: Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_droid_jointpos",
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_droid_jointpos_polaris",
    str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_droid_jointpos"),
    str(EAI_DIR / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid_jointpos"),
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_droid",
    str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_droid"),
    str(EAI_DIR / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid"),
)
# Require this much allocated data before treating a local dir as usable.
_MIN_ALLOCATED_BYTES = 512 * (1 << 20)  # 512 MiB


@dataclass(frozen=True)
class PiCheckpoint:
    path: str
    config: str
    source: str  # env | cache | local | local_present | gs | fallback


def _allocated_and_size(root: Path) -> Tuple[int, int]:
    alloc = 0
    size = 0
    if not root.exists():
        return 0, 0
    if root.is_file():
        st = root.stat()
        return st.st_blocks * 512, st.st_size
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            # Ignore HF download scratch; it inflates "usable" checks.
            if name.endswith(".incomplete") or name.endswith(".lock"):
                continue
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            alloc += st.st_blocks * 512
            size += st.st_size
    return alloc, size


def _has_incomplete_hf_download(root: Path) -> bool:
    cache = root / ".cache" / "huggingface" / "download"
    if not cache.is_dir():
        return False
    for dirpath, _dirnames, filenames in os.walk(cache):
        for name in filenames:
            if name.endswith(".incomplete"):
                return True
    return False


def is_local_checkpoint_present(path: str) -> bool:
    """True if path looks like an OpenPI ckpt dir (weights may still be sparse)."""
    raw = (path or "").strip()
    if not raw or raw.startswith("gs://") or "://" in raw:
        return False
    root = Path(os.path.expanduser(raw)).resolve()
    if not root.is_dir():
        return False
    has_jax = (root / "params" / "_CHECKPOINT_METADATA").is_file() or (
        root / "params"
    ).is_dir()
    has_pt = (root / "model.safetensors").is_file()
    return bool(has_jax or has_pt)


def is_local_checkpoint_usable(path: str) -> bool:
    """True if path is a local OpenPI ckpt with real (non-sparse) weights."""
    if not is_local_checkpoint_present(path):
        return False
    root = Path(os.path.expanduser((path or "").strip())).resolve()
    if _has_incomplete_hf_download(root):
        return False
    alloc, size = _allocated_and_size(root)
    if alloc < _MIN_ALLOCATED_BYTES:
        return False
    # Reject mostly-sparse stubs (apparent size huge, almost nothing allocated).
    if size > 0 and alloc < max(_MIN_ALLOCATED_BYTES, int(size * 0.02)):
        return False
    return True


def infer_policy_config(path: str, override: str = "") -> str:
    ov = (override or "").strip()
    if ov:
        return ov
    raw = (path or "").strip().rstrip("/")
    if not raw:
        return PI_CKPT_CONFIG_DEFAULT
    name = (
        urllib.parse.urlparse(raw).path.rstrip("/").split("/")[-1]
        if "://" in raw
        else Path(raw).name
    )
    # Symlink / renamed dirs: map known aliases to openpi TrainConfig names.
    aliases = {
        "pi05_droid_jointpos_polaris": "pi05_droid_jointpos_polaris",
        "pi05_droid_jointpos": "pi05_droid_jointpos",
        "pi05_droid": "pi05_droid",
    }
    return aliases.get(name, name or PI_CKPT_CONFIG_DEFAULT)


def resolve_openpi_python(prefer: str = "") -> str:
    for cand in ((prefer or "").strip(), *OPENPI_PYTHON_CANDIDATES):
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return "python3"


def resolve_openpi_root(prefer: str = "") -> Path:
    for cand in (
        (prefer or "").strip(),
        (os.environ.get("OPENPI_ROOT") or "").strip(),
        str(OPENPI_ROOT_DEFAULT),
    ):
        if not cand:
            continue
        root = Path(os.path.expanduser(cand)).resolve()
        script = root / "scripts" / "serve_policy.py"
        if script.is_file():
            return root
    return OPENPI_ROOT_DEFAULT


def resolve_pi_checkpoint(
    prefer: str = "",
    *,
    config: str = "",
    allow_gs: bool = True,
    prefer_present_local: bool = True,
) -> PiCheckpoint:
    """Pick the best available Pi checkpoint.

    Priority: prefer arg → env → usable local/cache → present local (UI) → gs://.
    """
    env_path = (
        os.environ.get("MOLMOSPACES_PI_CKPT") or os.environ.get("OPENPI_PI_CKPT") or ""
    ).strip()
    candidates: List[Tuple[str, str]] = []
    if (prefer or "").strip():
        candidates.append(((prefer or "").strip(), "prefer"))
    if env_path:
        candidates.append((env_path, "env"))
    for p in PI_CKPT_LOCAL_CANDIDATES:
        candidates.append((p, "local"))

    seen = set()
    present: Optional[PiCheckpoint] = None
    for path, source in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.startswith("gs://") or "://" in path:
            continue
        abs_path = os.path.abspath(os.path.expanduser(path))
        if is_local_checkpoint_usable(abs_path):
            return PiCheckpoint(
                path=abs_path,
                config=infer_policy_config(abs_path, config),
                source=source,
            )
        if prefer_present_local and present is None and is_local_checkpoint_present(abs_path):
            present = PiCheckpoint(
                path=abs_path,
                config=infer_policy_config(abs_path, config),
                source="local_present",
            )

    if present is not None:
        return present

    if allow_gs:
        return PiCheckpoint(
            path=PI_CKPT_GS_DEFAULT,
            config=infer_policy_config(PI_CKPT_GS_DEFAULT, config),
            source="gs",
        )

    fallback = (
        prefer or env_path or PI_CKPT_LOCAL_CANDIDATES[0] or PI_CKPT_GS_DEFAULT
    ).strip()
    return PiCheckpoint(
        path=fallback,
        config=infer_policy_config(fallback, config),
        source="fallback",
    )


def checkpoint_status(path: str) -> str:
    """Human-readable status for UI / logs."""
    raw = (path or "").strip()
    if not raw:
        return "空路径"
    if raw.startswith("gs://"):
        return "远程 GCS（首次 serve 时会下载到 OPENPI_DATA_HOME）"
    if is_local_checkpoint_usable(raw):
        alloc, size = _allocated_and_size(Path(os.path.expanduser(raw)))
        return f"本地可用（已分配 {alloc / 1e9:.1f}G / 标称 {size / 1e9:.1f}G）"
    root = Path(os.path.expanduser(raw))
    if not root.exists():
        return "路径不存在"
    if _has_incomplete_hf_download(root):
        return "HF 下载未完成（存在 .incomplete），请重新部署以下载完整权重"
    alloc, size = _allocated_and_size(root)
    if size > _MIN_ALLOCATED_BYTES and alloc < _MIN_ALLOCATED_BYTES:
        return (
            f"本地占位/稀疏文件（已分配 {alloc / 1e6:.0f}MB，标称 {size / 1e9:.1f}G）"
            " — 不可用，请换完整 ckpt 或 gs:// 路径"
        )
    return f"未识别为 OpenPI ckpt（alloc={alloc / 1e6:.0f}MB）"


def _download_hf_repo(
    repo_id: str,
    dest: str,
    *,
    log: Optional[Callable[[str], None]] = None,
    openpi_python: str = "",
) -> Tuple[bool, str]:
    if is_local_checkpoint_usable(dest):
        return True, dest

    def _log(msg: str) -> None:
        if log:
            log(msg)

    py = resolve_openpi_python(openpi_python)
    code = (
        "from huggingface_hub import snapshot_download\n"
        f"p=snapshot_download(repo_id={repo_id!r}, local_dir={dest!r})\n"
        "print(p)\n"
    )
    _log(f"[pi] 从 HuggingFace 下载 {repo_id} → {dest}")
    try:
        clean = {k: v for k, v in openpi_child_env(openpi_python=py).items() if v is not None}
        # Drop GUI/ros python+cuda path pollution before applying openpi env.
        base = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH")
        }
        proc = subprocess.run(
            [py, "-c", code],
            capture_output=True,
            text=True,
            timeout=None,
            env={**base, **clean, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"HF 下载异常: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"HF 下载失败 (code={proc.returncode}): {err[-800:]}"
    local_s = (proc.stdout or "").strip().splitlines()[-1].strip() or dest
    if is_local_checkpoint_usable(local_s):
        _log(f"[pi] HF 下载完成: {local_s}")
        return True, local_s
    return False, f"HF 下载后仍不可用: {checkpoint_status(local_s)}"


def ensure_norm_assets_cached(
    checkpoint: str,
    *,
    config: str = "",
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Copy ckpt assets/ into OPENPI_DATA_HOME GCS-layout cache (avoids GCS download).

    openpi TrainConfig for pi05_droid_jointpos points assets_dir at
    ``gs://openpi-assets/checkpoints/<name>/assets``; maybe_download maps that to
    ``$OPENPI_DATA_HOME/openpi-assets/checkpoints/<name>/assets``.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    ckpt = Path(os.path.abspath(os.path.expanduser((checkpoint or "").strip())))
    if not ckpt.is_dir():
        return False, f"checkpoint 不存在: {ckpt}"
    src_assets = ckpt / "assets"
    if not src_assets.is_dir():
        return False, f"checkpoint 缺少 assets/: {src_assets}"

    cfg = infer_policy_config(str(ckpt), config)
    cache_root = Path(
        os.environ.get("OPENPI_DATA_HOME") or OPENPI_DATA_HOME_DEFAULT
    ).expanduser().resolve()
    dst_assets = (
        cache_root / "openpi-assets" / "checkpoints" / cfg / "assets"
    )
    try:
        dst_assets.mkdir(parents=True, exist_ok=True)
        for child in src_assets.iterdir():
            if not child.is_dir():
                continue
            # Skip HF lock placeholder dirs.
            if child.name in (".cache",):
                continue
            dst = dst_assets / child.name
            marker = dst / "norm_stats.json"
            if marker.is_file():
                continue
            # Only materialize real asset packs (norm_stats), skip HF lock stubs.
            if not (child / "norm_stats.json").is_file():
                continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
            _log(f"[pi] 已缓存 assets/{child.name} → {dst}")
        # Remove stale download locks that block maybe_download.
        for lock in dst_assets.glob("*.lock"):
            try:
                lock.unlink()
            except OSError:
                pass
    except Exception as exc:  # noqa: BLE001
        return False, f"缓存 assets 失败: {type(exc).__name__}: {exc}"
    return True, str(dst_assets)


def ensure_checkpoint_materialized(
    path: str,
    *,
    log: Optional[Callable[[str], None]] = None,
    openpi_python: str = "",
) -> Tuple[bool, str]:
    """Materialize a usable local Pi checkpoint.

    Prefer already-cached HF/GCS dirs. For official jointpos / pi05_droid URLs,
    try HuggingFace mirrors first (faster than anon GCS on this host).
    """
    raw = (path or "").strip()
    if not raw:
        return False, "空 checkpoint 路径"

    def _finish(local: str) -> Tuple[bool, str]:
        ok_a, msg_a = ensure_norm_assets_cached(local, log=log)
        if not ok_a and log:
            log(f"[pi] assets 缓存警告: {msg_a}")
        return True, local

    if not raw.startswith("gs://"):
        abs_local = os.path.abspath(os.path.expanduser(raw))
        if is_local_checkpoint_usable(abs_local):
            return _finish(abs_local)
        # Incomplete HF dir: resume download when it matches known HF dest.
        if (
            os.path.abspath(abs_local) == os.path.abspath(PI_CKPT_HF_LOCAL_DEFAULT)
            or Path(abs_local).name == "pi05_droid_jointpos"
        ):
            ok_hf, hf_path = _download_hf_repo(
                PI_CKPT_HF_DEFAULT,
                PI_CKPT_HF_LOCAL_DEFAULT,
                log=log,
                openpi_python=openpi_python,
            )
            if ok_hf:
                return _finish(hf_path)
        if (
            os.path.abspath(abs_local) == os.path.abspath(PI05_LIBERO_HF_LOCAL_DEFAULT)
            or Path(abs_local).name == "pi05_libero"
        ):
            ok_hf, hf_path = _download_hf_repo(
                PI05_LIBERO_HF_DEFAULT,
                PI05_LIBERO_HF_LOCAL_DEFAULT,
                log=log,
                openpi_python=openpi_python,
            )
            if ok_hf:
                return _finish(hf_path)
        return False, checkpoint_status(raw)

    name = raw.rstrip("/").split("/")[-1]
    if name == "pi05_droid_jointpos" or "pi05_droid_jointpos" in raw:
        ok_hf, hf_path = _download_hf_repo(
            PI_CKPT_HF_DEFAULT,
            PI_CKPT_HF_LOCAL_DEFAULT,
            log=log,
            openpi_python=openpi_python,
        )
        if ok_hf:
            return _finish(hf_path)
    elif name == "pi05_droid" or raw.rstrip("/").endswith("/pi05_droid"):
        ok_hf, hf_path = _download_hf_repo(
            "ankile/openpi-pi05-droid-pretrained",
            str(EAI_DIR / ".cache" / "openpi" / "hf" / "pi05_droid"),
            log=log,
            openpi_python=openpi_python,
        )
        if ok_hf:
            return _finish(hf_path)
    elif name == "pi05_libero" or "pi05_libero" in raw:
        ok_hf, hf_path = _download_hf_repo(
            PI05_LIBERO_HF_DEFAULT,
            PI05_LIBERO_HF_LOCAL_DEFAULT,
            log=log,
            openpi_python=openpi_python,
        )
        if ok_hf:
            return _finish(hf_path)

    def _log(msg: str) -> None:
        if log:
            log(msg)

    parsed = urllib.parse.urlparse(raw)
    cache_root = Path(
        os.environ.get("OPENPI_DATA_HOME") or OPENPI_DATA_HOME_DEFAULT
    ).expanduser().resolve()
    cached = cache_root / parsed.netloc / parsed.path.strip("/")
    if is_local_checkpoint_usable(str(cached)):
        _log(f"[pi] 使用本地缓存: {cached}")
        return _finish(str(cached))

    py = resolve_openpi_python(openpi_python)
    code = (
        "import os\n"
        f"os.environ['OPENPI_DATA_HOME']={OPENPI_DATA_HOME_DEFAULT!r}\n"
        "from openpi.shared import download\n"
        f"p=download.maybe_download({raw!r}, token='anon')\n"
        "print(p)\n"
    )
    _log(f"[pi] 下载 checkpoint: {raw}")
    try:
        clean = {k: v for k, v in openpi_child_env(openpi_python=py).items() if v is not None}
        proc = subprocess.run(
            [py, "-c", code],
            capture_output=True,
            text=True,
            timeout=None,
            env={**{k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH")}, **clean},
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"下载进程异常: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"下载失败 (code={proc.returncode}): {err[-800:]}"
    local_s = (proc.stdout or "").strip().splitlines()[-1].strip()
    if is_local_checkpoint_usable(local_s):
        _log(f"[pi] 下载完成: {local_s}")
        return _finish(local_s)
    return False, f"下载后仍不可用: {checkpoint_status(local_s)} ({local_s})"


def _openpi_nvidia_lib_dirs(python_bin: str) -> List[str]:
    """Pip nvidia-* wheel lib dirs under the openpi env site-packages."""
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(python_bin)))
    candidates = [
        os.path.join(prefix, "lib", "python3.12", "site-packages", "nvidia"),
        os.path.join(prefix, "lib", "python3.11", "site-packages", "nvidia"),
        os.path.join(prefix, "lib", "python3.10", "site-packages", "nvidia"),
    ]
    out: List[str] = []
    seen = set()
    for nvidia_root in candidates:
        if not os.path.isdir(nvidia_root):
            continue
        try:
            for name in sorted(os.listdir(nvidia_root)):
                lib = os.path.join(nvidia_root, name, "lib")
                if os.path.isdir(lib) and lib not in seen:
                    seen.add(lib)
                    out.append(lib)
        except OSError:
            continue
        if out:
            break
    lib_dir = os.path.join(prefix, "lib")
    if os.path.isdir(lib_dir) and lib_dir not in seen:
        out.append(lib_dir)
    return out


def openpi_child_env(
    *,
    openpi_python: str = "",
    openpi_root: str = "",
) -> dict:
    """Clean env for openpi subprocesses (no GUI/ros-humble PYTHONPATH/LD)."""
    py = resolve_openpi_python(openpi_python)
    root = resolve_openpi_root(openpi_root)
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    bin_dir = os.path.join(prefix, "bin")
    markers = ("ros-humble", "/opt/ros/", "PyQt5/Qt5/lib", "/cv2/")
    path_parts = [bin_dir]
    for part in (os.environ.get("PATH") or "").split(os.pathsep):
        if not part:
            continue
        if any(m in part for m in markers):
            continue
        if part not in path_parts:
            path_parts.append(part)
    nvidia_libs = _openpi_nvidia_lib_dirs(py)
    ld = os.pathsep.join(nvidia_libs)
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "OPENPI_DATA_HOME": os.environ.get("OPENPI_DATA_HOME") or OPENPI_DATA_HOME_DEFAULT,
        # ONLY openpi src — do NOT inherit GUI/ros-humble PYTHONPATH (breaks numpy).
        "PYTHONPATH": str(root / "src"),
        # Drop parent conda/ros PYTHONHOME so openpi's own prefix is used.
        "PYTHONHOME": None,
        "PATH": os.pathsep.join(path_parts),
        # Prefer pip nvidia-* libs; drop Qt/ros LD that breaks cudnnCreate.
        "LD_LIBRARY_PATH": ld,
        "CONDA_PREFIX": prefix,
        "CONDA_DEFAULT_ENV": os.path.basename(prefix),
        "AMENT_PREFIX_PATH": None,
        "ROS_DISTRO": None,
        "ROS_VERSION": None,
    }


def build_serve_policy_argv(
    *,
    checkpoint: str,
    config: str = "",
    port: int = PI_SERVER_PORT_DEFAULT,
    openpi_python: str = "",
    openpi_root: str = "",
) -> Tuple[List[str], Path, dict]:
    """Return (argv, cwd, env_extra) for serve_policy.

    Uses ``tools/run_openpi_serve_policy.py`` so first infer does not run
    ``torch.compile(mode=max-autotune)`` (that blocks long enough for the
    MolmoSpaces websocket keepalive to drop and the arm never moves).
    """
    py = resolve_openpi_python(openpi_python)
    root = resolve_openpi_root(openpi_root)
    wrapper = Path(__file__).resolve().parent / "run_openpi_serve_policy.py"
    stock = root / "scripts" / "serve_policy.py"
    script = wrapper if wrapper.is_file() else stock
    cfg = infer_policy_config(checkpoint, config)
    ckpt = (checkpoint or "").strip()
    argv = [
        py,
        str(script),
        f"--port={int(port)}",
        "policy:checkpoint",
        f"--policy.config={cfg}",
        f"--policy.dir={ckpt}",
    ]
    env_extra = openpi_child_env(openpi_python=py, openpi_root=str(root))
    env_extra["OPENPI_ROOT"] = str(root)
    env_extra["OPENPI_SERVE_POLICY_SCRIPT"] = str(stock)
    # Default on; allow override from parent env if explicitly set.
    env_extra["OPENPI_DISABLE_TORCH_COMPILE"] = os.environ.get(
        "OPENPI_DISABLE_TORCH_COMPILE", "1"
    )
    return argv, root, env_extra


def port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if something is listening on host:port.

    Prefer ``ss``/``fuser`` so we do not open a bare TCP connection against an
    OpenPI websocket server (that spams handshake errors in serve_policy logs).
    """
    port = int(port)
    try:
        out = subprocess.check_output(
            ["ss", "-ltn", f"sport = :{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(0.5, float(timeout)),
        )
        if any("LISTEN" in ln for ln in out.splitlines()):
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(0.5, float(timeout)),
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Last resort: TCP connect (may log handshake errors on WS servers).
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def stop_pi_server_on_port(
    port: int = PI_SERVER_PORT_DEFAULT,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Best-effort stop of listeners on the Pi serve port (SIGTERM then SIGKILL)."""
    port = int(port)

    def _log(msg: str) -> None:
        if log:
            log(msg)

    pids: List[int] = []
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        for tok in out.replace("\n", " ").split():
            if tok.isdigit():
                pids.append(int(tok))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not pids and not port_listening("127.0.0.1", port):
        return f"端口 {port} 未占用"

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            _log(f"[pi] SIGTERM pid={pid} (port {port})")
        except (ProcessLookupError, PermissionError) as exc:
            _log(f"[pi] 无法结束 pid={pid}: {exc}")

    deadline = time.time() + 8.0
    while time.time() < deadline and port_listening("127.0.0.1", port):
        time.sleep(0.3)

    if port_listening("127.0.0.1", port):
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                _log(f"[pi] SIGKILL pid={pid}")
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.5)

    if port_listening("127.0.0.1", port):
        return f"端口 {port} 仍被占用，请手动结束进程"
    return f"已停止端口 {port} 上的服务"


def _pids_on_tcp_port(port: int) -> List[int]:
    pids: List[int] = []
    try:
        out = subprocess.check_output(
            ["fuser", f"{int(port)}/tcp"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        for tok in out.replace("\n", " ").split():
            if tok.isdigit():
                pids.append(int(tok))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return pids


def inspect_pi_server_on_port(port: int = PI_SERVER_PORT_DEFAULT) -> Dict[str, str]:
    """Inspect openpi serve_policy process on ``port`` via /proc cmdline.

    Returns keys: pid, config, dir, cmdline (empty strings when unknown).
    """
    info = {"pid": "", "config": "", "dir": "", "cmdline": ""}
    for pid in _pids_on_tcp_port(port):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not cmdline:
            continue
        # Prefer the serve_policy / run_openpi_serve_policy process.
        if "serve_policy" not in cmdline and "run_openpi_serve_policy" not in cmdline:
            # Keep scanning — fuser may list children / wrappers first.
            if not info["pid"]:
                info["pid"] = str(pid)
                info["cmdline"] = cmdline
            continue
        info["pid"] = str(pid)
        info["cmdline"] = cmdline
        # Tyro-style: --policy.config=NAME or --policy.config NAME
        m = re.search(r"--policy\.config(?:=|\s+)(\S+)", cmdline)
        if m:
            info["config"] = m.group(1).strip().strip("'\"")
        m = re.search(r"--policy\.dir(?:=|\s+)(\S+)", cmdline)
        if m:
            info["dir"] = m.group(1).strip().strip("'\"")
        break
    return info


def pi_server_matches(
    *,
    port: int,
    config: str = "",
    checkpoint: str = "",
) -> Tuple[bool, str]:
    """True if listener on ``port`` looks like the requested config/checkpoint."""
    want_cfg = (config or "").strip()
    raw_ckpt = (checkpoint or "").strip()
    want_dir = ""
    if raw_ckpt and not raw_ckpt.startswith("gs://"):
        want_dir = os.path.abspath(os.path.expanduser(raw_ckpt))
    elif raw_ckpt:
        want_dir = raw_ckpt

    info = inspect_pi_server_on_port(port)
    if not info.get("pid"):
        return False, "端口无 openpi serve 进程"
    have_cfg = (info.get("config") or "").strip()
    have_dir = (info.get("dir") or "").strip()
    if have_dir and not have_dir.startswith("gs://"):
        have_dir = os.path.abspath(os.path.expanduser(have_dir))

    parts = [f"pid={info['pid']}"]
    if have_cfg:
        parts.append(f"config={have_cfg}")
    if have_dir:
        parts.append(f"dir={have_dir}")
    detail = " ".join(parts)

    if want_cfg and have_cfg and want_cfg != have_cfg:
        return False, f"策略不匹配（需要 {want_cfg}，当前 {have_cfg}）; {detail}"
    if want_cfg and not have_cfg:
        return False, f"无法确认策略配置（需要 {want_cfg}）; {detail}"

    if want_dir and have_dir and not want_dir.startswith("gs://"):
        same = want_dir.rstrip("/") == have_dir.rstrip("/")
        if not same:
            try:
                same = os.path.realpath(want_dir) == os.path.realpath(have_dir)
            except OSError:
                same = False
        if not same:
            wb = os.path.basename(want_dir.rstrip("/"))
            hb = os.path.basename(have_dir.rstrip("/"))
            same = bool(wb and hb and wb == hb)
        if not same:
            return False, f"权重不匹配（需要 {want_dir}，当前 {have_dir}）; {detail}"
    return True, detail



def probe_pi_server(
    host: str = PI_SERVER_HOST_DEFAULT,
    port: int = PI_SERVER_PORT_DEFAULT,
    *,
    timeout_s: float = 8.0,
    openpi_python: str = "",
) -> Tuple[bool, str]:
    """Connect to OpenPI websocket and read server metadata.

    Always probes via the openpi env (subprocess) so the GUI/ros-humble
    interpreter — which lacks websockets/openpi_client — does not fall back to a
    bare TCP connect (that spams the server with handshake errors).
    """
    host = (host or PI_SERVER_HOST_DEFAULT).strip() or PI_SERVER_HOST_DEFAULT
    port = int(port)
    uri = f"ws://{host}:{port}"
    py = resolve_openpi_python(openpi_python)
    # Minimal handshake: recv the metadata frame openpi sends on connect.
    code = (
        "import sys\n"
        "host, port = sys.argv[1], int(sys.argv[2])\n"
        "uri = f'ws://{host}:{port}'\n"
        "try:\n"
        "    from openpi_client import websocket_client_policy as wcp\n"
        "    meta = wcp.WebsocketClientPolicy(host=host, port=port).get_server_metadata()\n"
        "    if isinstance(meta, dict):\n"
        "        keys = ','.join(sorted(str(k) for k in list(meta)[:8]))\n"
        "        print('OK', f'metadata_keys=[{keys}]')\n"
        "    else:\n"
        "        print('OK', type(meta).__name__)\n"
        "except Exception:\n"
        "    import websockets.sync.client as ws_client\n"
        "    with ws_client.connect(uri, compression=None, max_size=None, open_timeout=5) as conn:\n"
        "        try:\n"
        "            raw = conn.recv(timeout=5)\n"
        "        except TypeError:\n"
        "            raw = conn.recv()\n"
        "        print('OK', type(raw).__name__)\n"
    )
    clean = {k: v for k, v in openpi_child_env(openpi_python=py).items() if v is not None}
    base = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTHONHOME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "AMENT_PREFIX_PATH",
        )
    }
    try:
        proc = subprocess.run(
            [py, "-c", code, host, str(port)],
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_s)),
            env={**base, **clean},
        )
    except subprocess.TimeoutExpired:
        return False, f"握手超时 {uri}"
    except Exception as exc:  # noqa: BLE001
        return False, f"探测异常 {uri}: {type(exc).__name__}: {exc}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0 and out.startswith("OK"):
        detail = out[2:].strip() or "ws ok"
        return True, f"已连接 {uri} {detail}".rstrip()
    # Connection refused / not ready yet — keep message short for wait loop.
    blob = (err or out or f"exit={proc.returncode}").replace("\n", " ")
    if "ConnectError" in blob or "Connection refused" in blob or "Errno 111" in blob:
        return False, f"端口未就绪: {host}:{port}"
    return False, f"握手失败 {uri}: {blob[-240:]}"


def wait_for_pi_server(
    host: str = PI_SERVER_HOST_DEFAULT,
    port: int = PI_SERVER_PORT_DEFAULT,
    *,
    timeout_s: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Poll until Pi websocket answers or timeout (model load can take minutes)."""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    deadline = time.time() + max(5.0, float(timeout_s))
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ok, msg = probe_pi_server(host, port, timeout_s=5.0)
        if ok:
            return True, msg
        if attempt == 1 or attempt % 6 == 0:
            _log(f"[pi] 等待策略服务就绪… ({msg})")
        time.sleep(5.0)
    return False, f"超时 {timeout_s:.0f}s 仍无法调用 {host}:{port}"


def is_pi_policy_id(policy_id: str) -> bool:
    return "PiPolicyEvalConfig" in (policy_id or "")


__all__ = [
    "PI_CKPT_CONFIG_DEFAULT",
    "PI_CKPT_GS_DEFAULT",
    "PI_CKPT_HF_DEFAULT",
    "PI_CKPT_HF_LOCAL_DEFAULT",
    "PI05_LIBERO_HF_DEFAULT",
    "PI05_LIBERO_HF_LOCAL_DEFAULT",
    "PI_SERVER_HOST_DEFAULT",
    "PI_SERVER_PORT_DEFAULT",
    "OPENPI_DATA_HOME_DEFAULT",
    "PiCheckpoint",
    "ensure_checkpoint_materialized",
    "build_serve_policy_argv",
    "checkpoint_status",
    "infer_policy_config",
    "is_local_checkpoint_present",
    "is_local_checkpoint_usable",
    "is_pi_policy_id",
    "port_listening",
    "probe_pi_server",
    "inspect_pi_server_on_port",
    "pi_server_matches",
    "resolve_openpi_python",
    "resolve_openpi_root",
    "resolve_pi_checkpoint",
    "stop_pi_server_on_port",
    "wait_for_pi_server",
]
