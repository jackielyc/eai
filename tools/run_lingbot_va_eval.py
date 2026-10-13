#!/usr/bin/env python3
"""Launch LingBot-VA wan_va server and/or official eval clients.

Examples::

  # LIBERO (starts wan_va then client)
  python tools/run_lingbot_va_eval.py --backend libero --task-suite libero_10

  # RoboTwin
  python tools/run_lingbot_va_eval.py --backend robotwin --task adjust_bottle

  # Serve only (MolmoSpaces / Arena / extended benches)
  python tools/run_lingbot_va_eval.py --backend molmospaces --mode serve
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lingbot_va_policy_runtime import (  # noqa: E402
    DEFAULT_VA_HOST,
    DEFAULT_VA_PORT,
    LIBERO_VA_PORT_DEFAULT,
    ROBOTWIN_VA_PORT_DEFAULT,
    build_wan_va_server_argv,
    lingbot_va_child_env,
    port_listening,
    probe_lingbot_va,
    resolve_paths,
    va_config_for_backend,
    va_support_level,
)


def _log(msg: str) -> None:
    print(f"[lingbot-va] {msg}", flush=True)


def _env_clean(env: dict) -> dict:
    out = os.environ.copy()
    for k, v in env.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = str(v)
    return out


def _start_server(
    *,
    backend: str,
    paths,
    port: int,
    gpu: str,
) -> Optional[subprocess.Popen]:
    argv, cwd, env = build_wan_va_server_argv(
        backend=backend,
        checkpoint=paths.checkpoint,
        base_model=paths.base_model,
        config_name=paths.config_name,
        port=port,
        gpu=gpu,
        python_bin=paths.python_bin,
        policy_root=str(paths.policy_root),
    )
    _log("server: " + " ".join(argv))
    _log(f"cwd={cwd} port={port} config={paths.config_name}")
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=_env_clean(env),
        start_new_session=True,
    )
    host = DEFAULT_VA_HOST
    deadline = time.time() + 600.0
    while time.time() < deadline:
        if proc.poll() is not None:
            _log(f"server exited early code={proc.returncode}")
            return None
        if port_listening(host, port):
            _log(f"server ready {host}:{port}")
            return proc
        time.sleep(1.0)
    _log("server wait timeout")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    return None


def _run_libero_client(args: argparse.Namespace, paths) -> int:
    client = paths.upstream_root / "evaluation" / "libero" / "client.py"
    if not client.is_file():
        _log(f"missing client: {client}")
        return 2
    out = Path(args.out_dir or (_HERE.parent / ".cache" / "lingbot_va_eval" / "libero"))
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        paths.python_bin,
        str(client),
        f"--libero-benchmark={args.task_suite}",
        f"--port={int(args.port)}",
        f"--test-num={int(args.test_num)}",
        "--task-range",
        str(int(args.task_start)),
        str(int(args.task_end)),
        f"--out-dir={out}",
    ]
    # Some clients take --host; ignore if unsupported via env.
    env = lingbot_va_child_env(
        paths,
        extra={"LINGBOT_VA_VA_HOST": args.host, "LINGBOT_VA_VA_PORT": str(args.port)},
    )
    _log("client: " + " ".join(argv))
    return int(subprocess.call(argv, cwd=str(paths.upstream_root), env=_env_clean(env)))


def _run_robotwin_client(args: argparse.Namespace, paths) -> int:
    """Best-effort: prefer launch_client.sh, else websocket smoke."""
    launch = paths.upstream_root / "evaluation" / "robotwin" / "launch_client.sh"
    env = lingbot_va_child_env(
        paths,
        extra={
            "START_PORT": str(args.port),
            "LINGBOT_VA_VA_HOST": args.host,
            "LINGBOT_VA_VA_PORT": str(args.port),
            "TASK_NAME": args.task or "adjust_bottle",
        },
    )
    if launch.is_file():
        argv = ["bash", str(launch)]
        _log("client: " + " ".join(argv))
        return int(
            subprocess.call(argv, cwd=str(launch.parent), env=_env_clean(env))
        )
    # Fallback: import WebsocketClientPolicy and ping
    code = (
        "from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy "
        "import WebsocketClientPolicy\n"
        f"c=WebsocketClientPolicy(host={args.host!r}, port={int(args.port)})\n"
        "print('OK', getattr(c, 'metadata', {}))\n"
    )
    _log("robotwin launch_client.sh missing; probing websocket")
    return int(
        subprocess.call(
            [paths.python_bin, "-c", code],
            cwd=str(paths.upstream_root),
            env=_env_clean(env),
        )
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, help="sim backend key")
    ap.add_argument(
        "--mode",
        default="eval",
        choices=("eval", "serve", "probe"),
        help="eval=server+client when available; serve=wan_va only; probe=deps",
    )
    ap.add_argument("--config-name", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--base-model", default="")
    ap.add_argument("--policy-root", default="")
    ap.add_argument("--host", default=DEFAULT_VA_HOST)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--task", default="")
    ap.add_argument("--task-suite", default="libero_10")
    ap.add_argument("--test-num", type=int, default=1)
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-end", type=int, default=1)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)

    backend = args.backend.strip().lower()
    cfg = va_config_for_backend(backend, args.config_name)
    paths = resolve_paths(
        prefer_root=args.policy_root,
        prefer_ckpt=args.checkpoint,
        prefer_base=args.base_model,
        config_name=cfg,
        backend=backend,
    )
    ok, msg = probe_lingbot_va(paths)
    _log(("OK · " if ok else "NEED · ") + msg)
    if args.mode == "probe":
        return 0 if ok else 2
    if not ok and args.mode != "serve":
        _log("abort: LingBot-VA not ready (set LINGBOT_VA_BASE_MODEL_PATH / ckpt)")
        return 2

    level = va_support_level(backend)
    if args.port <= 0:
        if backend.startswith("libero") or backend == "robotwin":
            args.port = (
                LIBERO_VA_PORT_DEFAULT
                if backend.startswith("libero")
                else ROBOTWIN_VA_PORT_DEFAULT
            )
        else:
            args.port = DEFAULT_VA_PORT

    server_proc: Optional[subprocess.Popen] = None
    try:
        need_server = bool(args.start_server) or args.mode == "serve"
        if need_server and not port_listening(args.host, args.port):
            server_proc = _start_server(
                backend=backend, paths=paths, port=args.port, gpu=args.gpu
            )
            if server_proc is None:
                return 3
        elif port_listening(args.host, args.port):
            _log(f"reuse existing server {args.host}:{args.port}")

        if args.mode == "serve":
            _log(
                f"serve_only backend={backend} level={level} "
                f"ws={args.host}:{args.port} config={paths.config_name}"
            )
            if server_proc is None:
                return 0
            # Keep foreground until killed.
            return int(server_proc.wait())

        if backend.startswith("libero"):
            return _run_libero_client(args, paths)
        if backend == "robotwin":
            return _run_robotwin_client(args, paths)

        _log(
            f"backend={backend} support={level}: no native closed-loop client in eai; "
            f"wan_va is up at {args.host}:{args.port}. "
            "Use Isaac LingBot_VA deploy, or wire WebsocketClientPolicy."
        )
        if server_proc is not None:
            return int(server_proc.wait())
        return 0
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    finally:
        if server_proc is not None and server_proc.poll() is None and args.mode != "serve":
            _log("stopping wan_va server")
            try:
                os.killpg(server_proc.pid, signal.SIGTERM)
            except OSError:
                try:
                    server_proc.terminate()
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
