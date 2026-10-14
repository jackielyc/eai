#!/usr/bin/env python3
"""Unified launcher for extended EAI sim-eval backends.

Modes:
  probe   — dependency / repo check (default)
  hint    — print install + docker harness commands
  viewer  — try to open a minimal viewer / demo if available
  eval    — try local eval entrypoints; else print harness docker command
  harness — run AllenAI vla-evaluation-harness docker image (if docker present)

This does NOT vendor full benchmark stacks. It wires the GUI to local clones
and/or the industry-standard Docker harness so missing benches are actionable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_EAI = _HERE.parent
if str(_EAI) not in sys.path:
    sys.path.insert(0, str(_EAI))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from external_bench_runtime import (  # noqa: E402
    HARNESS_GHCR,
    build_harness_docker_argv,
    install_hint,
    probe_deps,
    resolve_python,
    resolve_root,
)
from sim_backends_registry import get_backend, normalize_backend_key  # noqa: E402


def _log(msg: str) -> None:
    print(f"[external-bench] {msg}", flush=True)


def _find_script(root: str, candidates: List[str]) -> Optional[str]:
    if not root:
        return None
    for rel in candidates:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def _run(argv: List[str], *, cwd: str = "", env: Optional[dict] = None) -> int:
    _log("exec: " + " ".join(argv))
    if cwd:
        _log(f"cwd: {cwd}")
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items() if v is not None})
    # Best-effort: honor EAI「暂停」via gym/gymnasium Env.step patch.
    full_env.setdefault("EAI_SIM_PAUSE", "1")
    tools = str(_HERE)
    prev_pp = full_env.get("PYTHONPATH") or ""
    parts = [tools] + [p for p in prev_pp.split(os.pathsep) if p and p != tools]
    full_env["PYTHONPATH"] = os.pathsep.join(parts)
    # Prepend bootstrap import for python script launches.
    py_name = os.path.basename(str(argv[0] if argv else ""))
    if len(argv) >= 2 and py_name.startswith("python"):
        script = argv[1]
        if script.endswith(".py") and os.path.isfile(script):
            boot = (
                "import sys; import eai_sim_pause_bootstrap; "
                f"sys.argv=[{script!r}]+sys.argv[1:]; "
                f"import runpy; runpy.run_path({script!r}, run_name='__main__')"
            )
            argv = [argv[0], "-c", boot, *argv[2:]]
            _log("pause bootstrap injected for gym/gymnasium Env.step")
    try:
        return int(
            subprocess.call(argv, cwd=cwd or None, env=full_env)
        )
    except FileNotFoundError as exc:
        _log(f"command not found: {exc}")
        return 127


def _mode_probe(args: argparse.Namespace) -> int:
    ok, msg = probe_deps(args.backend, args.python, args.root)
    _log(("OK · " if ok else "NEED · ") + msg)
    hint = install_hint(args.backend, args.python)
    print(hint, flush=True)
    return 0 if ok else 2


def _mode_hint(args: argparse.Namespace) -> int:
    print(install_hint(args.backend, args.python), flush=True)
    spec = get_backend(args.backend)
    if spec and spec.harness_image:
        _log(
            "example harness:\n  "
            + " ".join(build_harness_docker_argv(args.backend))
        )
    return 0


def _try_local_eval(args: argparse.Namespace, *, viewer: bool) -> Optional[int]:
    """Return exit code if a local entrypoint was found and launched; else None."""
    key = normalize_backend_key(args.backend)
    root = resolve_root(key, args.root)
    py = resolve_python(args.python)
    if not root or not os.path.isdir(root):
        return None

    # Backend-specific entrypoints (best-effort; evolve as local clones appear).
    scripts = {
        "simpler": [
            "simpler_env/eval_simpler.py",
            "scripts/eval.py",
            "tools/eval.py",
        ],
        "calvin": [
            "calvin_models/calvin_agent/evaluation/evaluate_policy.py",
            "evaluation/evaluate_policy.py",
        ],
        "robocasa": ["robocasa/scripts/eval.py", "scripts/eval.py"],
        "robocasa365": ["robocasa/scripts/eval.py", "scripts/eval.py"],
        "maniskill3": ["examples/demo.py", "mani_skill/examples/demo.py"],
        "maniskill2": ["examples/demo.py"],
        "metaworld": [],
        "rlbench": ["examples/single_task_rl.py"],
        "genesis": ["examples/rigid/franka_cube.py", "examples/tutorials/hello_genesis.py"],
        "furniturebench": ["furniture_bench/scripts/run.py"],
        "vlabench": ["scripts/eval.py", "eval.py"],
        "behavior1k": [],
        "libero_plus": [],
        "libero_pro": [],
        "libero_mem": [],
    }.get(key, ["eval.py", "scripts/eval.py", "examples/demo.py"])

    script = _find_script(root, scripts) if scripts else None

    if key == "metaworld":
        # Tiny smoke: import + list envs
        code = (
            "import metaworld, sys;\n"
            "print('metaworld', getattr(metaworld, '__version__', '?'));\n"
            "print('OK');\n"
        )
        return _run([py, "-c", code], cwd=root)

    if key == "genesis" and script is None:
        code = (
            "import genesis as gs\n"
            "gs.init()\n"
            "print('genesis ok')\n"
        )
        return _run([py, "-c", code], cwd=root)

    if script is None:
        return None

    argv = [py, script]
    if args.task:
        argv.extend(["--task", args.task])
    if viewer:
        # Many scripts use --gui / --render; pass both as harmless extras via env
        os.environ.setdefault("EXTERNAL_BENCH_VIEWER", "1")
    if args.episodes:
        argv.extend(["--episodes", str(args.episodes)])
    # Append raw extras after --
    if args.extra:
        argv.extend(args.extra)
    return _run(argv, cwd=root)


def _mode_harness(args: argparse.Namespace) -> int:
    spec = get_backend(args.backend)
    if spec is None:
        _log(f"unknown backend: {args.backend}")
        return 2
    if args.backend == "vla_harness" or not spec.harness_image:
        image = f"{HARNESS_GHCR}/base"
        argv = ["docker", "run", "--rm", "--gpus", "all", "-it", image]
    else:
        try:
            argv = build_harness_docker_argv(args.backend, extra=args.extra or ())
        except ValueError as exc:
            _log(str(exc))
            return 2
    if not shutil.which("docker"):
        _log("docker 未安装；无法启动 harness 镜像")
        _log("命令预览: " + " ".join(argv))
        return 3
    return _run(argv)


def _mode_lingbot_va(args: argparse.Namespace) -> int:
    """Delegate to tools/run_lingbot_va_eval.py for policy=lingbot_va."""
    va = _HERE / "run_lingbot_va_eval.py"
    if not va.is_file():
        _log(f"missing {va}")
        return 2
    py = resolve_python(args.python)
    mode = "probe" if args.mode == "probe" else "eval"
    if args.mode == "hint":
        try:
            from lingbot_va_policy_runtime import install_hint as va_hint

            print(va_hint(py), flush=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"lingbot_va hint failed: {exc}")
        return 0
    argv = [
        py,
        str(va),
        f"--backend={args.backend}",
        f"--mode={mode}",
    ]
    if args.task:
        if str(args.backend).startswith("libero"):
            argv.append(f"--task-suite={args.task}")
        else:
            argv.append(f"--task={args.task}")
    return _run(argv, cwd=str(_EAI))


def _mode_vpp2(args: argparse.Namespace) -> int:
    """Delegate to tools/run_vpp2_eval.py for policy=vpp2."""
    runner = _HERE / "run_vpp2_eval.py"
    if not runner.is_file():
        _log(f"missing {runner}")
        return 2
    py = resolve_python(args.python)
    mode = "probe" if args.mode == "probe" else "eval"
    if args.mode == "hint":
        try:
            from vpp2_policy_runtime import install_hint as vpp2_hint

            print(vpp2_hint(py), flush=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"vpp2 hint failed: {exc}")
        return 0
    argv = [
        py,
        str(runner),
        f"--backend={args.backend}",
        f"--mode={mode}",
    ]
    if args.task:
        if str(args.backend).startswith("libero"):
            argv.append(f"--task-suite={args.task}")
        else:
            argv.append(f"--task={args.task}")
    return _run(argv, cwd=str(_EAI))


def _mode_eval_or_viewer(args: argparse.Namespace, *, viewer: bool) -> int:
    key = normalize_backend_key(args.backend)
    if key == "vla_harness":
        return _mode_harness(args)

    rc = _try_local_eval(args, viewer=viewer)
    if rc is not None:
        return rc

    _log("未找到本地可执行入口，打印安装与 harness 指引：")
    print(install_hint(key, args.python), flush=True)
    spec = get_backend(key)
    if spec and spec.harness_image:
        _log("也可改用 --mode harness 启动 Docker 评测镜像")
        _log("  " + " ".join(build_harness_docker_argv(key)))
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description="EAI extended sim-eval launcher")
    p.add_argument("--backend", required=True)
    p.add_argument(
        "--mode",
        default="probe",
        choices=("probe", "hint", "viewer", "eval", "harness"),
    )
    p.add_argument("--root", default="")
    p.add_argument("--python", default="")
    p.add_argument("--task", default="")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument(
        "--policy",
        default="",
        help="optional policy id (e.g. lingbot_va / vpp2)",
    )
    p.add_argument("extra", nargs="*", help="extra args forwarded to local script")
    args = p.parse_args()
    if args.no_viewer:
        args.viewer = False
    args.backend = normalize_backend_key(args.backend)
    if get_backend(args.backend) is None:
        _log(f"unknown backend: {args.backend}")
        return 2

    pol = (args.policy or os.environ.get("EAI_EXTERNAL_POLICY") or "").strip()
    _log(f"backend={args.backend} mode={args.mode} policy={pol or '-'}")
    if pol and "lingbot_va" in pol.lower().replace("-", "_"):
        return _mode_lingbot_va(args)
    if pol and "vpp2" in pol.lower().replace("-", "_"):
        return _mode_vpp2(args)
    if args.mode == "probe":
        return _mode_probe(args)
    if args.mode == "hint":
        return _mode_hint(args)
    if args.mode == "harness":
        return _mode_harness(args)
    if args.mode == "viewer":
        return _mode_eval_or_viewer(args, viewer=True)
    return _mode_eval_or_viewer(args, viewer=args.viewer)


if __name__ == "__main__":
    raise SystemExit(main())
