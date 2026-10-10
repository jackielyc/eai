"""Resolve paths / launch argv for extended sim-eval backends."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow importing registry from eai root when run as script.
_EAI = Path(__file__).resolve().parent.parent
if str(_EAI) not in sys.path:
    sys.path.insert(0, str(_EAI))

from sim_backends_registry import (  # noqa: E402
    EXTENDED_BACKENDS,
    get_backend,
    normalize_backend_key,
    resolve_default_root,
)

HARNESS_GHCR = "ghcr.io/allenai/vla-evaluation-harness"


def resolve_python(prefer: str = "") -> str:
    raw = (prefer or "").strip()
    if raw and os.path.isfile(raw) and os.access(raw, os.X_OK):
        return raw
    return sys.executable


def resolve_root(backend: str, prefer: str = "") -> str:
    raw = (prefer or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    spec = get_backend(backend)
    if spec is None:
        return ""
    return resolve_default_root(spec) or ""


def probe_deps(backend: str, python_bin: str = "", root: str = "") -> Tuple[bool, str]:
    """Lightweight readiness check. Returns (ok, message)."""
    key = normalize_backend_key(backend)
    spec = get_backend(key)
    if spec is None:
        return False, f"unknown backend: {backend}"
    py = resolve_python(python_bin)
    root_p = resolve_root(key, root)

    if key == "vla_harness":
        docker = shutil.which("docker")
        if docker:
            return True, f"docker available ({docker}); use harness image pull"
        return False, "需要 docker 以运行 AllenAI VLA harness 镜像"

    if key == "genesis":
        # Prefer import check
        code = (
            "import importlib.util as u;"
            "print('ok' if u.find_spec('genesis') else 'missing')"
        )
        try:
            import subprocess

            out = subprocess.check_output(
                [py, "-c", code], text=True, stderr=subprocess.STDOUT, timeout=30
            )
            if "ok" in out:
                return True, f"genesis importable via {py}"
        except Exception as exc:
            pass
        if root_p and os.path.isdir(root_p):
            return True, f"repo present: {root_p}（import 可能未装）"
        return False, "未找到 genesis；" + (spec.install_hint or "")

    if root_p and os.path.isdir(root_p):
        return True, f"repo: {root_p}"
    if spec.harness_image:
        return (
            False,
            f"本地仓库未找到；可 clone 到 {spec.default_roots[0] if spec.default_roots else '…'} "
            f"或 docker pull {HARNESS_GHCR}/{spec.harness_image}",
        )
    return False, "本地仓库未找到；" + (spec.install_hint or "见官方文档")


def install_hint(backend: str, python_bin: str = "") -> str:
    spec = get_backend(backend)
    if spec is None:
        return f"unknown backend: {backend}"
    lines = [
        f"# {spec.label} ({spec.key})",
        f"# engine={spec.engine or 'n/a'}",
    ]
    if spec.install_hint:
        lines.append(spec.install_hint)
    if spec.harness_image:
        lines.append(
            f"docker pull {HARNESS_GHCR}/{spec.harness_image}"
        )
    if spec.source_url:
        lines.append(f"# docs: {spec.source_url}")
    if python_bin:
        lines.append(f"# python: {python_bin}")
    return "\n".join(lines)


def build_harness_docker_argv(
    backend: str,
    *,
    extra: Sequence[str] = (),
) -> List[str]:
    spec = get_backend(backend)
    if spec is None or not spec.harness_image:
        raise ValueError(f"no harness image for {backend}")
    image = f"{HARNESS_GHCR}/{spec.harness_image}"
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-it",
        image,
    ]
    argv.extend(list(extra))
    return argv


def build_eval_argv(
    backend: str,
    *,
    root: str = "",
    python_bin: str = "",
    mode: str = "probe",
    task: str = "",
    episodes: int = 1,
    viewer: bool = True,
    harness: bool = False,
    extra: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a launch plan: {cwd, argv, env, note}."""
    key = normalize_backend_key(backend)
    spec = get_backend(key)
    if spec is None:
        raise KeyError(f"unknown backend: {backend}")

    py = resolve_python(python_bin)
    root_p = resolve_root(key, root)
    env = {
        "EAI_EXTERNAL_BENCH": key,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if root_p:
        env["EXTERNAL_BENCH_ROOT"] = root_p

    runner = str(_EAI / "tools" / "run_external_bench.py")

    if harness and spec.harness_image:
        argv = build_harness_docker_argv(key)
        return {
            "cwd": root_p or str(_EAI),
            "argv": argv,
            "env": env,
            "note": f"AllenAI harness docker · {spec.harness_image}",
            "python": py,
        }

    argv = [
        py,
        runner,
        "--backend",
        key,
        "--mode",
        mode,
        "--episodes",
        str(max(1, int(episodes))),
    ]
    if root_p:
        argv.extend(["--root", root_p])
    if task:
        argv.extend(["--task", task])
    if viewer:
        argv.append("--viewer")
    else:
        argv.append("--no-viewer")
    argv.extend(list(extra))

    return {
        "cwd": root_p or str(_EAI),
        "argv": argv,
        "env": env,
        "note": f"{spec.label} via run_external_bench.py",
        "python": py,
    }


def list_extended_choices() -> List[Tuple[str, str]]:
    """(label, key) for GUI combo."""
    return [(b.label, b.key) for b in EXTENDED_BACKENDS]
