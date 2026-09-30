#!/usr/bin/env python3
"""RoboTwin eval entry used by EAI 仿真评测.

Wraps RLinf ``examples/embodiment/eval_embodied_agent.py`` (same as
``eval_embodiment.sh``) with GUI-friendly Hydra overrides:

  - 1 env / single-GPU placement by default
  - override task env YAML + model_path + assets_path
  - ROBOT_PLATFORM=ALOHA, ROBOTWIN_PATH on PYTHONPATH
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EAI RoboTwin / RLinf OpenPI eval")
    p.add_argument("--task", default="adjust_bottle")
    p.add_argument("--policy_config", default="pi05_aloha_robotwin")
    p.add_argument("--hydra_config", default="")
    p.add_argument("--env_config", default="")
    p.add_argument("--model_path", default="")
    p.add_argument("--assets_path", default="")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0", help="CUDA device id(s) for placement, e.g. 0 or 0-0")
    p.add_argument("--exp_name", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument(
        "--rlinf_root",
        default=os.environ.get(
            "RLINF_ROOT",
            "/share_data/projects/mahjong/share/personal/liyichao/RLinf",
        ),
    )
    p.add_argument(
        "--robotwin_root",
        default=os.environ.get(
            "ROBOTWIN_PATH",
            "/share_data/projects/mahjong/share/personal/liyichao/RoboTwin",
        ),
    )
    p.add_argument(
        "--render_gui",
        action="store_true",
        help="Open SAPIEN Viewer (task_config.render_freq>0; forces num_envs=1)",
    )
    p.add_argument(
        "--no_render_gui",
        action="store_true",
        help="Headless eval (default): keep render_freq=0",
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra Hydra overrides appended as-is",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    from robotwin_policy_runtime import (  # noqa: E402
        hydra_config_for,
        resolve_robotwin_assets,
        resolve_robotwin_root,
        resolve_rlinf_root,
        robotwin_child_env,
    )

    rlinf = resolve_rlinf_root(args.rlinf_root)
    rt = resolve_robotwin_root(args.robotwin_root)
    assets = resolve_robotwin_assets(args.assets_path, robotwin_root=rt)
    embodied = rlinf / "examples" / "embodiment"
    render_gui = bool(args.render_gui) and not bool(args.no_render_gui)
    # GUI mode: EAI agent holds SAPIEN Viewer until「停止」; else stock RLinf entry.
    if render_gui:
        src = here / "robotwin_eval_agent.py"
    else:
        src = embodied / "eval_embodied_agent.py"
    if not src.is_file():
        print(f"[robotwin] missing {src}", file=sys.stderr)
        return 2

    task = (args.task or "adjust_bottle").strip()
    policy_cfg = (args.policy_config or "pi05_aloha_robotwin").strip()
    hydra_name = (args.hydra_config or "").strip() or hydra_config_for(task, policy_cfg)
    env_cfg = (args.env_config or "").strip() or f"robotwin_{task}"
    model_path = (args.model_path or "").strip()
    if not model_path:
        print("[robotwin] --model_path required", file=sys.stderr)
        return 2

    log_dir = pathlib.Path(
        args.log_dir
        or (here.parent / ".cache" / "robotwin_eval" / (args.exp_name or task))
    ).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu = (args.gpu or "0").strip()
    n_envs = max(1, int(args.num_envs))
    max_steps = max(1, int(args.max_episode_steps))
    render_freq = 10 if render_gui else 0
    if render_gui and n_envs > 1:
        print(
            f"[robotwin] render_gui=True：SAPIEN Viewer 仅支持单环境，"
            f"将 num_envs {n_envs} -> 1",
            flush=True,
        )
        n_envs = 1

    # Env YAML packages mainly differ by task_name; override fields in-place
    # (avoid fragile Hydra config-group swaps from the CLI).
    _ = env_cfg  # retained for logging / future package swap
    overrides = [
        f"runner.logger.log_path={log_dir}",
        "runner.only_eval=True",
        f"actor.model.model_path={model_path}",
        f"actor.model.openpi.config_name={policy_cfg}",
        f"env.train.assets_path={assets}",
        f"env.eval.assets_path={assets}",
        f"env.train.total_num_envs={n_envs}",
        f"env.eval.total_num_envs={n_envs}",
        f"env.train.max_episode_steps={max_steps}",
        f"env.eval.max_episode_steps={max_steps}",
        f"env.train.max_steps_per_rollout_epoch={max_steps}",
        f"env.eval.max_steps_per_rollout_epoch={max_steps}",
        f"env.train.task_config.task_name={task}",
        f"env.eval.task_config.task_name={task}",
        f"env.train.task_config.render_freq={render_freq}",
        f"env.eval.task_config.render_freq={render_freq}",
        "env.train.task_config.embodiment=[aloha-agilex]",
        "env.eval.task_config.embodiment=[aloha-agilex]",
        "env.train.task_config.camera.collect_wrist_camera=true",
        "env.eval.task_config.camera.collect_wrist_camera=true",
        "env.train.center_crop=False",
        "env.eval.center_crop=False",
        "env.eval.is_eval=True",
        "env.eval.auto_reset=True",
        "env.eval.use_fixed_reset_state_ids=True",
        # YAML key is literally "actor, env, rollout" (comma+spaces) which Hydra
        # CLI cannot override; delete the map and re-add per-component keys.
        "~cluster.component_placement",
        f"+cluster.component_placement.actor={gpu}",
        f"+cluster.component_placement.env={gpu}",
        f"+cluster.component_placement.rollout={gpu}",
    ]
    if args.extra:
        overrides.extend(str(x) for x in args.extra)

    env_extra = robotwin_child_env(
        python_bin=sys.executable,
        rlinf_root=str(rlinf),
        robotwin_root=str(rt),
        assets_path=assets,
    )
    # Merge into process env for the child.
    child_env = {k: v for k, v in os.environ.items()}
    for k, v in env_extra.items():
        if v is None:
            child_env.pop(k, None)
        else:
            child_env[k] = v
    child_env["ROBOT_PLATFORM"] = "ALOHA"
    child_env["ROBOTWIN_PATH"] = str(rt)
    child_env["ASSETS_PATH"] = assets
    child_env["EMBODIED_PATH"] = str(embodied)
    if render_gui:
        # SAPIEN Viewer needs a display; keep parent DISPLAY / XAUTHORITY.
        display = (os.environ.get("DISPLAY") or "").strip()
        if display:
            child_env["DISPLAY"] = display
        else:
            print(
                "[robotwin] 警告: DISPLAY 为空，SAPIEN 图形窗口可能无法弹出",
                flush=True,
            )
        xauth = (os.environ.get("XAUTHORITY") or "").strip()
        if xauth:
            child_env["XAUTHORITY"] = xauth
        # Keep viewer after eval until EAI「停止」(SIGTERM on process group).
        child_env["ROBOTWIN_HOLD_VIEWER"] = "1"
    # Pin visible GPUs for GUI smoke (Hydra placement still set above).
    if gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Use first id when user passes "0" / "0-0" / "0,1"
        first = gpu.replace("-", ",").split(",")[0].strip()
        if first.isdigit():
            child_env["CUDA_VISIBLE_DEVICES"] = first
            # After remapping, placement should be logical GPU 0.
            overrides = [
                o
                if not o.startswith("+cluster.component_placement.")
                else o.rsplit("=", 1)[0] + "=0"
                for o in overrides
            ]
    # Ensure RoboTwin + RLinf on PYTHONPATH (eval_embodiment.sh pattern).
    py_path = [str(rlinf), str(rt)]
    existing = child_env.get("PYTHONPATH") or ""
    for part in existing.split(os.pathsep):
        if part and part not in py_path:
            py_path.append(part)
    child_env["PYTHONPATH"] = os.pathsep.join(py_path)

    cmd = [
        sys.executable,
        str(src),
        f"--config-path={embodied / 'config'}",
        f"--config-name={hydra_name}",
        *overrides,
    ]
    print("[robotwin] cwd=", embodied, flush=True)
    print("[robotwin] ROBOTWIN_PATH=", rt, flush=True)
    print("[robotwin] assets=", assets, flush=True)
    print("[robotwin] model=", model_path, flush=True)
    print("[robotwin] hydra=", hydra_name, "task=", task, flush=True)
    print(
        f"[robotwin] render_gui={render_gui} render_freq={render_freq} n_envs={n_envs}",
        flush=True,
    )
    print("[robotwin] cmd=", " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd=str(embodied), env=child_env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
