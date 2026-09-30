#!/usr/bin/env python3
"""EAI wrapper for IsaacLab-Arena policy_runner with optional RynnValue Live HUD.

Usage (GUI)::

  python tools/run_arena_eval.py --arena-root /path/to/IsaacLab-Arena \\
      -- --policy_type zero_action --num_steps 200 ... <env_name>

When ``RYNNVALUE_LIVE_HUD=1``, patches ``rollout_policy`` to push camera frames.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="EAI IsaacLab-Arena eval + Live HUD")
    p.add_argument(
        "--arena-root",
        default=os.environ.get(
            "ISAACLAB_ARENA_ROOT",
            "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/IsaacLab-Arena",
        ),
    )
    p.add_argument(
        "--runner",
        default="",
        help="policy_runner.py path (default: <arena-root>/isaaclab_arena/evaluation/policy_runner.py)",
    )
    args, rest = p.parse_known_args()
    # Allow "--" separator
    if rest and rest[0] == "--":
        rest = rest[1:]

    here = Path(__file__).resolve().parent
    arena_root = Path(os.path.expanduser(args.arena_root)).resolve()
    runner = Path(
        args.runner
        or (arena_root / "isaaclab_arena" / "evaluation" / "policy_runner.py")
    ).resolve()
    if not runner.is_file():
        print(f"[arena] missing policy_runner: {runner}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(here))
    sys.path.insert(0, str(arena_root))
    os.environ.setdefault("RYNNVALUE_HUD_BACKEND", "arena")
    os.environ.setdefault("EAI_DIR", str(here.parent))

    try:
        from rynnvalue_hud_attach import install_arena_rollout_hook

        if install_arena_rollout_hook():
            print("[arena] RynnValue Live HUD hook enabled", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[arena] Live HUD hook skipped: {exc}", flush=True)

    # policy_runner uses sys.argv; drop our wrapper flags.
    sys.argv = [str(runner), *rest]
    os.chdir(str(arena_root))
    print(f"[arena] runner={runner}", flush=True)
    print(f"[arena] argv={sys.argv}", flush=True)
    runpy.run_path(str(runner), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
