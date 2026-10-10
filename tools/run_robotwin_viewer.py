#!/usr/bin/env python3
"""Open a RoboTwin SAPIEN Viewer for one task — no policy / no inference.

Used by EAI 仿真评测「只开界面」。Mirrors ``RoboTwin/script/collect_data.py``
setup, then holds the Viewer until SIGINT/SIGTERM (UI「停止」).
"""
from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import time
from pathlib import Path


def _load_task_args(robotwin_root: Path, task_name: str, task_config: str) -> tuple:
    import yaml

    os.chdir(str(robotwin_root))
    if str(robotwin_root) not in sys.path:
        sys.path.insert(0, str(robotwin_root))

    from envs import CONFIGS_PATH  # noqa: E402

    envs_module = importlib.import_module(f"envs.{task_name}")
    if not hasattr(envs_module, task_name):
        raise SystemExit(f"No task class envs.{task_name}.{task_name}")
    task = getattr(envs_module, task_name)()

    cfg_path = robotwin_root / "task_config" / f"{task_config}.yml"
    if not cfg_path.is_file():
        raise SystemExit(f"missing task config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        args = yaml.load(fh.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = max(1, int(args.get("render_freq") or 0) or 10)
    args["collect_data"] = False
    args["save_data"] = False
    args["need_plan"] = False
    args["eval_mode"] = True
    args["episode_num"] = 1

    embodiment_type = args.get("embodiment") or ["aloha-agilex"]
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as fh:
        embodiment_types = yaml.load(fh.read(), Loader=yaml.FullLoader)

    def _embodiment_file(etype: str) -> str:
        entry = embodiment_types.get(etype) or {}
        robot_file = entry.get("file_path")
        if not robot_file:
            raise SystemExit(f"missing embodiment file_path for {etype!r}")
        return robot_file

    def _embodiment_config(robot_file: str) -> dict:
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as fh:
            return yaml.load(fh.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise SystemExit("embodiment config must have 1 or 3 entries")

    args["left_embodiment_config"] = _embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _embodiment_config(args["right_robot_file"])
    args["save_path"] = str(
        Path(
            os.environ.get("EAI_DIR")
            or Path(__file__).resolve().parent.parent
        )
        / ".cache"
        / "robotwin_viewer"
        / task_name
    )
    Path(args["save_path"]).mkdir(parents=True, exist_ok=True)
    return task, args


def main() -> int:
    p = argparse.ArgumentParser(description="RoboTwin SAPIEN viewer (no policy)")
    p.add_argument("--task", default="adjust_bottle")
    p.add_argument("--task_config", default="demo_clean")
    p.add_argument("--robotwin_root", default="")
    p.add_argument("--assets_path", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    eai = here.parent
    os.environ.setdefault("EAI_DIR", str(eai))

    from robotwin_policy_runtime import (  # noqa: E402
        resolve_robotwin_assets,
        resolve_robotwin_root,
        robotwin_child_env,
    )

    rt = resolve_robotwin_root(args.robotwin_root)
    assets = resolve_robotwin_assets(args.assets_path, robotwin_root=rt)
    os.environ["ASSETS_PATH"] = assets
    os.environ["ROBOTWIN_PATH"] = str(rt)
    # Apply env tweaks used by eval (DISPLAY etc. already inherited).
    for k, v in robotwin_child_env(
        python_bin=sys.executable,
        robotwin_root=str(rt),
        assets_path=assets,
    ).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    if not (rt / "envs" / "_base_task.py").is_file():
        print(
            f"[robotwin-viewer] need main-branch RoboTwin checkout with envs/: {rt}",
            file=sys.stderr,
        )
        return 2

    display = (os.environ.get("DISPLAY") or "").strip()
    if not display:
        print("[robotwin-viewer] 警告: DISPLAY 为空，SAPIEN Viewer 可能无法弹出", flush=True)

    task_name = (args.task or "adjust_bottle").strip()
    task_config = (args.task_config or "demo_clean").strip()
    print(
        f"[robotwin-viewer] root={rt} task={task_name} config={task_config} "
        f"assets={assets}",
        flush=True,
    )

    task, task_args = _load_task_args(rt, task_name, task_config)
    task_args["seed"] = int(args.seed)
    task.setup_demo(now_ep_num=0, seed=int(args.seed), **task_args)
    # Allow many take_action calls for GUI teleop.
    try:
        task.step_lim = max(int(getattr(task, "step_lim", 0) or 0), 10**9)
    except Exception:
        pass

    from sim_gui_teleop import (  # noqa: E402
        GuiTeleopSession,
        robotwin_action_from_cmd,
        robotwin_pose_to_pose7,
    )

    teleop = GuiTeleopSession(label="robotwin")
    print(
        "[robotwin-viewer] 界面已打开（无策略推理）；「手臂/手」文本指挥可用；"
        "点击 UI「停止」或 Ctrl+C 关闭",
        flush=True,
    )

    stop = {"flag": False}

    def _on_signal(signum, _frame) -> None:
        stop["flag"] = True
        print(f"[robotwin-viewer] signal {signum} — closing", flush=True)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)
    try:
        while not stop["flag"]:
            try:
                left = task.get_arm_pose("left")
                right = task.get_arm_pose("right")
                lg = float(task.robot.get_left_gripper_val())
                rg = float(task.robot.get_right_gripper_val())
                teleop.write_state(
                    left_ee_pose=robotwin_pose_to_pose7(left),
                    right_ee_pose=robotwin_pose_to_pose7(right),
                    # GUI 0=张 1=合；RoboTwin gripper 越大越开
                    left_gripper=max(0.0, min(1.0, 1.0 - lg)),
                    right_gripper=max(0.0, min(1.0, 1.0 - rg)),
                )
                cmd = teleop.read_cmd()
                if cmd and teleop.cmd_is_new(cmd) and not teleop._busy:
                    action = robotwin_action_from_cmd(left, right, lg, rg, cmd)
                    teleop._busy = True
                    try:
                        task.take_action(action, action_type="ee")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[robotwin-viewer] take_action: {exc}", flush=True)
                    finally:
                        teleop._busy = False
            except Exception as exc:  # noqa: BLE001
                print(f"[robotwin-viewer] teleop tick: {exc}", flush=True)

            viewer = getattr(task, "viewer", None)
            if viewer is not None:
                try:
                    viewer.render()
                except Exception as exc:  # noqa: BLE001
                    print(f"[robotwin-viewer] viewer.render: {exc}", flush=True)
                    break
            else:
                # Still step the scene so the window (if any) stays alive.
                try:
                    task.scene.step()
                    task.scene.update_render()
                except Exception:
                    pass
            time.sleep(0.03)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        try:
            task.close_env()
        except Exception:
            pass
        try:
            if getattr(task, "viewer", None) is not None:
                task.viewer.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
