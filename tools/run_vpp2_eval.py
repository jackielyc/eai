#!/usr/bin/env python3
"""Launch VPP2 policy server and/or official eval entrypoints.

Examples::

  python tools/run_vpp2_eval.py --backend isaac --mode serve
  python tools/run_vpp2_eval.py --backend libero --mode eval
  python tools/run_vpp2_eval.py --backend isaac --mode probe
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vpp2_policy_runtime import (  # noqa: E402
    build_backend_vpp2_argv,
    ensure_adapter_installed,
    install_hint,
    probe_vpp2,
    resolve_paths,
    vpp2_support_level,
)


def _log(msg: str) -> None:
    print(f"[vpp2] {msg}", flush=True)


def _env_clean(env: dict) -> dict:
    out = os.environ.copy()
    for k, v in env.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = str(v)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True)
    ap.add_argument(
        "--mode",
        default="eval",
        choices=("eval", "serve", "probe"),
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--task", default="")
    ap.add_argument("--task-suite", default="")
    args = ap.parse_args(argv)

    backend = args.backend.strip().lower()
    ok_ad, ad_msg = ensure_adapter_installed()
    _log(("adapter OK · " if ok_ad else "adapter NEED · ") + ad_msg)
    paths = resolve_paths()
    ok, msg = probe_vpp2(paths)
    _log(("OK · " if ok else "NEED · ") + msg)
    if args.mode == "probe":
        if not ok:
            print(install_hint(), flush=True)
        return 0 if ok else 2
    if not ok:
        _log("abort: VPP2 weights/adapter incomplete")
        print(install_hint(), flush=True)
        return 2

    level = vpp2_support_level(backend)
    mode = "serve" if args.mode == "serve" or level in ("serve_only", "robodojo_adapter") else "eval"
    # For isaac, default to server (client is GUI/robodojo); for libero use eval.
    if backend == "isaac" and args.mode == "eval":
        mode = "serve"
        _log("isaac: launching policy server (use GUI client with policy=VPP2)")

    argv_l, cwd, env, note = build_backend_vpp2_argv(
        backend if mode != "serve" or backend.startswith("libero") else "isaac",
        gpu=args.gpu,
        port=args.port or None,
        task=args.task,
        task_suite=args.task_suite,
    )
    # Force server path when mode=serve
    if mode == "serve" and not backend.startswith("libero"):
        from vpp2_policy_runtime import build_robodojo_server_argv

        argv_l, cwd, env = build_robodojo_server_argv(
            port=args.port or 19960, gpu=args.gpu
        )
        note = f"VPP2 serve · port={args.port or 19960}"

    _log(note)
    _log("exec: " + " ".join(argv_l))
    _log(f"cwd={cwd}")
    return int(subprocess.call(argv_l, cwd=str(cwd), env=_env_clean(env)))


if __name__ == "__main__":
    raise SystemExit(main())
