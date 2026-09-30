#!/usr/bin/env python3
"""LIBERO eval entry used by EAI 仿真评测.

Mirrors RLinf ``toolkits/eval_scripts_openpi/libero_eval.py`` and openpi
``examples/libero/main.py``:

  --mode remote  → WebsocketClientPolicy (Pi serve_policy)
  --mode local   → openpi create_trained_policy (in-process, RLinf-style)
"""
from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import pathlib
import sys
from typing import Any, Deque, Optional


def _bootstrap_paths(args: argparse.Namespace) -> None:
    here = pathlib.Path(__file__).resolve().parent
    eai = here.parent
    openpi_root = pathlib.Path(
        args.openpi_root
        or os.environ.get("OPENPI_ROOT")
        or "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/openpi"
    ).resolve()
    rlinf_root = pathlib.Path(
        args.rlinf_root
        or os.environ.get("RLINF_ROOT")
        or "/share_data/projects/mahjong/share/personal/liyichao/RLinf"
    ).resolve()
    libero_src = pathlib.Path(
        os.environ.get("LIBERO_PATH")
        or (openpi_root / "third_party" / "libero")
    ).resolve()

    # Official openpi README: PYTHONPATH=$PWD/third_party/libero
    # Do NOT add …/third_party/libero/libero — that makes `import libero` resolve
    # to the inner package and breaks `from libero.libero import …`.
    # Prepend so we win over a polluted parent PYTHONPATH.
    ordered = [
        str(here),
        str(libero_src),
        str(openpi_root / "src"),
        str(rlinf_root),
    ]
    bad_suffix = str(libero_src / "libero")
    sys.path[:] = [p for p in sys.path if p not in ordered and p != bad_suffix]
    for p in reversed(ordered):
        sys.path.insert(0, p)

    # Drop bad path from env for child imports.
    cur = os.environ.get("PYTHONPATH") or ""
    parts = [p for p in cur.split(os.pathsep) if p and p != bad_suffix]
    if str(libero_src) not in parts:
        parts.insert(0, str(libero_src))
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    # Non-interactive LIBERO config.
    from libero_policy_runtime import ensure_libero_config, resolve_libero_src

    cfg = ensure_libero_config(libero_src=resolve_libero_src(openpi_root))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(cfg))
    # On-screen viewer needs glfw; headless/offscreen uses egl.
    if getattr(args, "render_gui", False):
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("EAI_DIR", str(eai))
    os.environ.setdefault("LIBERO_PATH", str(libero_src))


def _quat2axisangle(quat):
    import numpy as np

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed, *, render_gui: bool = False):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    common = dict(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    if render_gui:
        # Keep offscreen cameras for policy obs; also open robosuite GLFW window.
        env = ControlEnv(
            **common,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="agentview",
        )
    else:
        env = OffScreenRenderEnv(**common)
    env.seed(seed)
    return env, task_description


def _render_gui_frame(env) -> None:
    """Best-effort on-screen robosuite/mujoco render."""
    inner = getattr(env, "env", env)
    render = getattr(inner, "render", None) or getattr(env, "render", None)
    if callable(render):
        try:
            render()
        except Exception as exc:  # noqa: BLE001
            logging.debug("render skipped: %s", exc)


def _max_steps_for_suite(name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[name]


class _RemotePolicy:
    def __init__(self, host: str, port: int):
        from openpi_client import websocket_client_policy as wcp

        self._client = wcp.WebsocketClientPolicy(host, port)

    def reset(self) -> None:
        return None

    def infer(self, obs: dict) -> dict:
        try:
            return self._client.infer(obs)
        except RuntimeError as exc:
            text = str(exc)
            # Empty server error string usually means wrong policy / bad obs keys.
            if text.strip().endswith(":\n") or text.strip().endswith(":"):
                raise RuntimeError(
                    f"{text.rstrip()}\n"
                    "(服务端未返回详情。常见原因：:8080 上部署的不是 pi05_libero，"
                    "而是其它策略如 pi05_droid_jointpos。请在 GUI 重新「部署 Pi」"
                    "或停止旧 serve_policy 后再评测。)"
                ) from exc
            raise



class _LocalPolicy:
    def __init__(self, config_name: str, checkpoint: str, num_steps: int):
        try:
            import jax  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "local 进程内加载需要 openpi 环境（含 jax）。"
                "GUI 请选「本地 ckpt（自动部署 serve）」或「远程 Pi」；"
                "CLI 请改用 --mode remote，并用 openpi 的 serve_policy 提供权重。"
            ) from exc
        # Prefer stock openpi configs (avoid RLinf→ray import chain).
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        # Optional: disable compile for first-infer WS-like hangs when local.
        os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
        try:
            import dataclasses

            train_config = _config.get_config(config_name)
            model = train_config.model
            if hasattr(model, "pytorch_compile_mode") and model.pytorch_compile_mode:
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(model, pytorch_compile_mode=None),
                )
        except Exception:
            train_config = _config.get_config(config_name)

        self._policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint,
            sample_kwargs={"num_steps": int(num_steps)},
        )

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, obs: dict) -> dict:
        return self._policy.infer(obs)


def _setup_policy(args: argparse.Namespace) -> Any:
    if args.mode == "remote":
        logging.info("policy=remote ws://%s:%s", args.host, args.port)
        return _RemotePolicy(args.host, int(args.port))
    if not args.pretrained_path:
        raise SystemExit("--pretrained_path required for --mode local")
    logging.info(
        "policy=local config=%s ckpt=%s num_steps=%s",
        args.config_name,
        args.pretrained_path,
        args.num_steps,
    )
    return _LocalPolicy(args.config_name, args.pretrained_path, args.num_steps)


def main(args: argparse.Namespace) -> int:
    _bootstrap_paths(args)

    import imageio
    import numpy as np
    from libero.libero import benchmark
    from tqdm import tqdm

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{args.exp_name}.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    live_hud = None
    if bool(getattr(args, "rynnvalue_live_hud", False)):
        from rynnvalue_live_hud import RynnValueLiveHud

        live_hud = RynnValueLiveHud(
            server_url=str(getattr(args, "rynnvalue_server_url", "") or "http://127.0.0.1:8001"),
            refresh_sec=float(getattr(args, "rynnvalue_refresh_sec", 1.0) or 1.0),
            num_frames=int(getattr(args, "rynnvalue_num_frames", 8) or 8),
            robot_description=str(
                getattr(args, "rynnvalue_robot_description", "")
                or "a Franka single-arm robot"
            ),
            camera_description=str(
                getattr(args, "rynnvalue_camera_description", "")
                or "the third-person agentview camera"
            ),
            show_window=bool(getattr(args, "rynnvalue_show_window", True)),
            status_path=str(log_dir / "rynnvalue_live.json"),
            timeout_s=float(getattr(args, "rynnvalue_timeout_s", 60.0) or 60.0),
        )
        live_hud.start()

    logging.info(
        "LIBERO eval mode=%s suite=%s trials/task=%s max_tasks=%s render_gui=%s live_hud=%s",
        args.mode,
        args.task_suite_name,
        args.num_trials_per_task,
        args.max_tasks,
        bool(getattr(args, "render_gui", False)),
        bool(live_hud is not None),
    )

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    if args.max_tasks > 0:
        num_tasks = min(num_tasks, int(args.max_tasks))
    max_steps = _max_steps_for_suite(args.task_suite_name)
    resolution = 256
    dummy = [0.0] * 6 + [-1.0]

    try:
        policy = _setup_policy(args)
    except Exception as exc:
        logging.error("policy setup failed: %s", exc)
        logging.error(
            "Hint: for remote mode, deploy Pi with config=%s then retry; "
            "for local mode, install openpi + checkpoint.",
            args.config_name,
        )
        if live_hud is not None:
            live_hud.stop()
        raise

    total_episodes = 0
    total_successes = 0
    results_per_task: dict[str, float] = {}

    try:
        for task_id in tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(
                task, resolution, args.seed, render_gui=bool(args.render_gui)
            )
            task_episodes = 0
            task_successes = 0

            n_trials = min(int(args.num_trials_per_task), len(initial_states))
            for episode_idx in range(n_trials):
                logging.info("Task: %s", task_description)
                logging.info("Starting episode %s...", task_episodes + 1)
                policy.reset()
                env.reset()
                action_plan: Deque[Any] = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
                replay_images = []
                hud_images = []
                done = False
                if live_hud is not None:
                    live_hud.reset_episode(str(task_description))

                for t in range(max_steps + args.num_steps_wait):
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(dummy)
                        if args.render_gui:
                            _render_gui_frame(env)
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    # Remote openpi client expects 224 padded uint8; local openpi
                    # transforms usually resize internally — still send uint8 HxWx3.
                    try:
                        from openpi_client import image_tools

                        img_in = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, 224, 224)
                        )
                        wrist_in = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, 224, 224)
                        )
                    except Exception:
                        img_in, wrist_in = img, wrist_img

                    replay_images.append(img_in)
                    if live_hud is not None:
                        live_hud.push_frame(img, step=t - args.num_steps_wait)
                        overlay_rgb = live_hud.last_overlay_rgb()
                        if overlay_rgb is not None:
                            hud_images.append(overlay_rgb)
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    if not action_plan:
                        observation = {
                            "observation/image": img_in,
                            "observation/wrist_image": wrist_in,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(observation)["actions"]
                        assert len(action_chunk) >= args.action_chunk, (
                            f"replan every {args.action_chunk} but got {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[: args.action_chunk])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if args.render_gui:
                        _render_gui_frame(env)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                task_episodes += 1
                total_episodes += 1
                if total_episodes <= args.num_save_videos and replay_images:
                    suffix = "success" if done else "failure"
                    task_segment = task_description.replace(" ", "_")
                    out_path = (
                        log_dir
                        / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
                    )
                    imageio.mimwrite(
                        out_path,
                        [np.asarray(x) for x in replay_images[:: args.video_temp_subsample]],
                        fps=max(1, 30 // max(1, args.video_temp_subsample)),
                    )
                    logging.info("saved video %s", out_path)
                    if hud_images:
                        hud_path = (
                            log_dir
                            / f"rollout_{task_segment}_{episode_idx}_{suffix}_rynnvalue_hud.mp4"
                        )
                        imageio.mimwrite(
                            hud_path,
                            [np.asarray(x) for x in hud_images[:: args.video_temp_subsample]],
                            fps=max(1, 30 // max(1, args.video_temp_subsample)),
                        )
                        logging.info("saved Live HUD video %s", hud_path)

                logging.info("Success: %s", done)
                logging.info(
                    "# episodes=%s successes=%s (%.1f%%)",
                    total_episodes,
                    total_successes,
                    100.0 * total_successes / max(1, total_episodes),
                )

            env.close()
            rate = task_successes / task_episodes if task_episodes else 0.0
            results_per_task[task_description] = rate
            logging.info(
                "Task done: %s %s/%s (%.1f%%)",
                task_description,
                task_successes,
                task_episodes,
                100.0 * rate,
            )

        logging.info("======== LIBERO SUMMARY ========")
        for name, rate in results_per_task.items():
            logging.info("  %s: %.1f%%", name, 100.0 * rate)
        overall = total_successes / total_episodes if total_episodes else 0.0
        logging.info(
            "Overall: %s/%s = %.1f%%",
            total_successes,
            total_episodes,
            100.0 * overall,
        )
        return 0
    finally:
        if live_hud is not None:
            live_hud.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EAI LIBERO eval (RLinf/openpi style)")
    p.add_argument("--mode", choices=("remote", "local"), default="remote")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--config_name", default="pi05_libero")
    p.add_argument("--pretrained_path", default="")
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=1, help="0 = all tasks in suite")
    p.add_argument("--action_chunk", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=10, help="flow denoise steps (local)")
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_save_videos", type=int, default=1)
    p.add_argument("--video_temp_subsample", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--render_gui",
        action="store_true",
        help="Open robosuite on-screen viewer (has_renderer=True, MUJOCO_GL=glfw)",
    )
    p.add_argument(
        "--no_render_gui",
        action="store_true",
        help="Force headless OffScreenRenderEnv (default when --render_gui omitted)",
    )
    p.add_argument(
        "--rynnvalue_live_hud",
        action="store_true",
        help="Enable async RynnValue Live HUD (needs reward_server on --rynnvalue_server_url)",
    )
    p.add_argument(
        "--rynnvalue_server_url",
        default="http://127.0.0.1:8001",
        help="RynnValue reward_server base URL",
    )
    p.add_argument(
        "--rynnvalue_refresh_sec",
        type=float,
        default=1.0,
        help="Live HUD wall-clock refresh interval in seconds",
    )
    p.add_argument(
        "--rynnvalue_num_frames",
        type=int,
        default=8,
        help="Frames subsampled per Live HUD score request",
    )
    p.add_argument(
        "--rynnvalue_timeout_s",
        type=float,
        default=60.0,
        help="HTTP timeout for each Live HUD score request",
    )
    p.add_argument(
        "--rynnvalue_robot_description",
        default="a Franka single-arm robot",
    )
    p.add_argument(
        "--rynnvalue_camera_description",
        default="the third-person agentview camera",
    )
    p.add_argument(
        "--rynnvalue_show_window",
        action="store_true",
        default=True,
        help="Show OpenCV Live HUD window (default on)",
    )
    p.add_argument(
        "--no_rynnvalue_show_window",
        action="store_true",
        help="Disable OpenCV Live HUD window (still scores + writes status/video)",
    )
    p.add_argument("--exp_name", default="libero_eval")
    p.add_argument("--log_dir", default=str(pathlib.Path.cwd() / "logs"))
    p.add_argument("--rlinf_root", default="")
    p.add_argument("--openpi_root", default="")
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.no_render_gui:
        _args.render_gui = False
    if _args.no_rynnvalue_show_window:
        _args.rynnvalue_show_window = False
    raise SystemExit(main(_args))
