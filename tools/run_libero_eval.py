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


def _get_libero_env(
    task,
    resolution,
    seed,
    *,
    render_gui: bool = False,
    ignore_done: bool = False,
    hard_reset: bool = True,
):
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
        ignore_done=ignore_done,
        hard_reset=hard_reset,
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


# robosuite OpenCVRenderer window (utils/opencv_renderer.py).
_LIBERO_GUI_WINDOW = "offscreen render"
_gui_center_state = {"left": 0, "moved": 0}
# User closed the on-screen viewer (title-bar X / Esc / q). Once set, skip imshow
# so robosuite render() cannot pop the window back open.
_gui_dismiss_state = {"dismissed": False, "seen_open": False, "missing_streak": 0}
_opencv_renderer_close_patched = False
# Shared JPEG for eai 工作区「图像预览」(written every few steps).
_PREVIEW_WRITE_STATE = {"last_at": 0.0, "path": "", "logged_ok": 0, "logged_err": 0}


def _libero_preview_path() -> pathlib.Path:
    eai = pathlib.Path(os.environ.get("EAI_DIR") or pathlib.Path(__file__).resolve().parent.parent)
    return eai / ".cache" / "rynnvalue_live" / "libero_preview.jpg"


def _write_libero_preview_rgb(rgb: Any, *, min_interval_s: float = 0.1) -> None:
    """Atomically write agentview RGB for the eai GUI preview poller."""
    import time as _time

    now = _time.time()
    if now - float(_PREVIEW_WRITE_STATE["last_at"] or 0.0) < float(min_interval_s):
        return
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        if not _PREVIEW_WRITE_STATE["logged_err"]:
            logging.warning("libero preview write: cv2/numpy unavailable: %s", exc)
            _PREVIEW_WRITE_STATE["logged_err"] = 1
        return
    try:
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
        path = _libero_preview_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".writing.jpg")
        if not cv2.imwrite(str(tmp), bgr):
            raise RuntimeError(f"cv2.imwrite failed for {tmp}")
        tmp.replace(path)
        _PREVIEW_WRITE_STATE["last_at"] = now
        _PREVIEW_WRITE_STATE["path"] = str(path)
        if not _PREVIEW_WRITE_STATE["logged_ok"]:
            logging.info("libero preview → %s (%sx%s)", path, arr.shape[1], arr.shape[0])
            _PREVIEW_WRITE_STATE["logged_ok"] = 1
    except Exception as exc:  # noqa: BLE001
        if not _PREVIEW_WRITE_STATE["logged_err"]:
            logging.warning("libero preview write failed: %s", exc)
            _PREVIEW_WRITE_STATE["logged_err"] = 1


def _patch_opencv_renderer_close() -> None:
    """Stop robosuite from ``destroyAllWindows()`` (that also kills Live HUD)."""
    global _opencv_renderer_close_patched
    if _opencv_renderer_close_patched:
        return
    try:
        from robosuite.utils.opencv_renderer import OpenCVRenderer
    except Exception as exc:  # noqa: BLE001
        logging.debug("OpenCVRenderer patch skipped: %s", exc)
        return

    def _close(self) -> None:  # noqa: ANN001
        self.sim = None
        try:
            import cv2

            cv2.destroyWindow(_LIBERO_GUI_WINDOW)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001
            pass

    OpenCVRenderer.close = _close  # type: ignore[method-assign]
    _opencv_renderer_close_patched = True
    logging.info(
        "Patched OpenCVRenderer.close to destroy only %r (keep Live HUD)",
        _LIBERO_GUI_WINDOW,
    )


def _arm_libero_gui_center() -> None:
    """Recenter after the next show. hard_reset destroys and recreates the window."""
    _gui_center_state["left"] = 30
    _gui_center_state["moved"] = 0
    # hard_reset briefly destroys the OpenCV window — do not treat that as user X.
    if not _gui_dismiss_state["dismissed"]:
        _gui_dismiss_state["seen_open"] = False
        _gui_dismiss_state["missing_streak"] = 0


def _libero_gui_dismissed() -> bool:
    return bool(_gui_dismiss_state["dismissed"])


def _dismiss_libero_gui(reason: str = "") -> None:
    """Close the LIBERO OpenCV viewer and stop further on-screen renders."""
    if _gui_dismiss_state["dismissed"]:
        return
    _gui_dismiss_state["dismissed"] = True
    try:
        import cv2

        cv2.destroyWindow(_LIBERO_GUI_WINDOW)
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001
        pass
    extra = f" ({reason})" if reason else ""
    logging.info("LIBERO 图形窗口已关闭%s", extra)
    print(f"[libero] 图形窗口已关闭{extra}", flush=True)


def _libero_gui_window_is_open() -> bool:
    if _gui_dismiss_state["dismissed"]:
        return False
    try:
        import cv2

        visible = cv2.getWindowProperty(
            _LIBERO_GUI_WINDOW, cv2.WND_PROP_VISIBLE
        )
        return float(visible) >= 1.0
    except Exception:  # noqa: BLE001
        return False


def _pump_libero_gui_close_events() -> bool:
    """Detect Esc/q / title-bar X. Returns True if the GUI was dismissed."""
    if _gui_dismiss_state["dismissed"]:
        return True
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return False
    try:
        key = int(cv2.waitKey(1) & 0xFF)
    except Exception:  # noqa: BLE001
        return False
    if key in (27, ord("q"), ord("Q")):
        _dismiss_libero_gui("Esc/q")
        return True
    if _libero_gui_window_is_open():
        _gui_dismiss_state["seen_open"] = True
        _gui_dismiss_state["missing_streak"] = 0
    else:
        # QT often reports -1 during create/move; require a stable open first,
        # then several consecutive misses before treating as user X.
        _gui_dismiss_state["missing_streak"] += 1
        if (
            _gui_dismiss_state["seen_open"]
            and _gui_dismiss_state["missing_streak"] >= 8
        ):
            _dismiss_libero_gui("title-bar X")
            return True
    return bool(_gui_dismiss_state["dismissed"])


def _primary_monitor_rect() -> tuple[int, int, int, int]:
    """Return (x, y, width, height) of the primary monitor in virtual-screen coords."""
    try:
        import glfw

        if glfw.init():
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon) if mon else None
            if mon is not None and mode is not None and mode.size.width > 0:
                ox, oy = glfw.get_monitor_pos(mon)
                return int(ox), int(oy), int(mode.size.width), int(mode.size.height)
    except Exception:  # noqa: BLE001
        pass
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
        root.destroy()
        if sw > 0 and sh > 0:
            return 0, 0, sw, sh
    except Exception:  # noqa: BLE001
        pass
    return 0, 0, 1920, 1080


def _maybe_center_libero_gui() -> None:
    """Place the robosuite on-screen viewer at the center of the primary monitor."""
    if _gui_center_state["left"] <= 0 or _gui_center_state["moved"] >= 2:
        return
    _gui_center_state["left"] -= 1
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return
    name = _LIBERO_GUI_WINDOW
    width, height = 1280, 800
    try:
        rect = cv2.getWindowImageRect(name)
        if rect is not None and int(rect[2]) > 0 and int(rect[3]) > 0:
            width, height = int(rect[2]), int(rect[3])
    except Exception:  # noqa: BLE001
        pass
    ox, oy, sw, sh = _primary_monitor_rect()
    x = int(ox + max(0, (sw - width) // 2))
    y = int(oy + max(0, (sh - height) // 2))
    try:
        cv2.moveWindow(name, x, y)
    except Exception as exc:  # noqa: BLE001
        logging.debug("center libero window skipped: %s", exc)
        return
    _gui_center_state["moved"] += 1
    if _gui_center_state["moved"] >= 2:
        logging.info("LIBERO 图形窗口已置于屏幕中央 (%s, %s)", x, y)


def _render_gui_frame(env) -> bool:
    """Best-effort on-screen robosuite/mujoco render.

    Returns False if the user dismissed the window (X / Esc / q); callers must
    stop rendering so OpenCV does not recreate it via imshow.
    """
    if _libero_gui_dismissed():
        return False
    inner = getattr(env, "env", env)
    render = getattr(inner, "render", None) or getattr(env, "render", None)
    if callable(render):
        try:
            render()
        except Exception as exc:  # noqa: BLE001
            logging.debug("render skipped: %s", exc)
            return not _pump_libero_gui_close_events()
    _maybe_center_libero_gui()
    return not _pump_libero_gui_close_events()


def _hold_gui_until_stop(env) -> None:
    """Keep the on-screen viewer open until window close / Esc / SIGTERM."""
    import signal
    import time

    if _libero_gui_dismissed():
        return

    stop = {"flag": False}

    def _on_signal(signum, _frame) -> None:
        stop["flag"] = True
        logging.info("received signal %s — closing GUI", signum)

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)
    logging.info(
        "评测完成，图形窗口保持打开；关闭窗口 / Esc / q，或 UI「停止」/ Ctrl+C 后退出"
    )
    try:
        while not stop["flag"] and not _libero_gui_dismissed():
            if not _render_gui_frame(env):
                break
            time.sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


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
            if "BLAS" in text or "cublas" in text.lower():
                raise RuntimeError(
                    f"{text.rstrip()}\n"
                    "(Pi JAX 无法初始化 cuBLAS。单卡上 Pi05≈31G 与 RynnValue≈14G "
                    "几乎占满 46G：必须先完成一次 Pi 推理再加载 RynnValue。"
                    "请停 Live HUD/reward_server 后重新部署 Pi，或再点一次评测"
                    "（界面会先停 RynnValue、预热 Pi、再拉起 HUD）。)"
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


def _main_viewer_only(args: argparse.Namespace) -> int:
    """Open first-task LIBERO env GUI and hold — no policy / no Pi.

    Accepts EAI「手臂/手」文本指挥 / 滑块 via ``gui_robot_cmd.json``.
    """
    import numpy as np
    from libero.libero import benchmark
    from rynnvalue_sim_bridge import LiberoHudBridge
    from sim_gui_teleop import (
        GuiTeleopSession,
        libero_action_from_cmd,
        libero_obs_to_state,
    )

    _patch_opencv_renderer_close()
    args.render_gui = True
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(0)
    initial_states = task_suite.get_task_init_states(0)
    # ignore_done: horizon (~1000 steps) must not terminate the episode —
    # otherwise viewer_only hard-resets and the OpenCV window repeatedly pops.
    # hard_reset=False: if we do reset, keep the same window (soft sim.reset).
    env, task_description = _get_libero_env(
        task,
        256,
        args.seed,
        render_gui=True,
        ignore_done=True,
        hard_reset=False,
    )
    hud_bridge = LiberoHudBridge()
    teleop = GuiTeleopSession(label="libero")
    hold = [0.0] * 6 + [-1.0]
    logging.info(
        "LIBERO viewer_only suite=%s task=%s (no policy; GUI teleop on; "
        "ignore_done, soft reset)",
        args.task_suite_name,
        task_description,
    )
    try:
        env.reset()
        _arm_libero_gui_center()
        obs = env.set_init_state(initial_states[0])
        try:
            _write_libero_preview_rgb(
                hud_bridge.extract_rgb(obs, apply_flip=False), min_interval_s=0.0
            )
        except Exception:  # noqa: BLE001
            pass
        logging.info(
            "界面已打开（无策略推理）；「手臂/手」文本指挥可用；"
            "关闭窗口 / Esc / q，或 UI「停止」/ Ctrl+C 退出"
        )
        import signal
        import time

        stop = {"flag": False}

        def _on_signal(signum, _frame) -> None:
            stop["flag"] = True
            logging.info("received signal %s — closing GUI", signum)

        prev_int = signal.signal(signal.SIGINT, _on_signal)
        prev_term = signal.signal(signal.SIGTERM, _on_signal)

        def _viewer_reset() -> Any:
            """Soft reset + restore init pose; do not destroy the GUI window."""
            # Belt-and-suspenders: force soft reset even if env was created hard.
            inner = getattr(env, "env", None)
            if inner is not None and hasattr(inner, "hard_reset"):
                inner.hard_reset = False
            env.reset()
            return env.set_init_state(initial_states[0])

        try:
            while not stop["flag"] and not _libero_gui_dismissed():
                pose7, grip = libero_obs_to_state(obs)
                teleop.write_state(left_ee_pose=pose7, left_gripper=grip)
                cmd = teleop.read_cmd()
                action = libero_action_from_cmd(obs, cmd) if cmd else hold
                try:
                    obs, _reward, done, _info = env.step(action)
                except ValueError as exc:
                    # robosuite: "executing action in terminated episode"
                    if "terminated episode" not in str(exc).lower():
                        raise
                    logging.info("episode ended — soft-resetting viewer env")
                    obs = _viewer_reset()
                    done = False
                if done:
                    logging.info("task done — soft-resetting viewer env (keep GUI open)")
                    obs = _viewer_reset()
                try:
                    _write_libero_preview_rgb(
                        hud_bridge.extract_rgb(obs, apply_flip=False)
                    )
                except Exception:  # noqa: BLE001
                    pass
                if not _render_gui_frame(env):
                    break
                time.sleep(0.02)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
        return 0
    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            logging.debug("env.close skipped: %s", exc)


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

    if bool(getattr(args, "viewer_only", False)):
        return _main_viewer_only(args)

    from rynnvalue_sim_bridge import LiberoHudBridge, maybe_create_live_hud

    # robosuite hard_reset used destroyAllWindows() which also killed Live HUD.
    _patch_opencv_renderer_close()

    hud_bridge = LiberoHudBridge()
    live_hud = maybe_create_live_hud(
        args,
        bridge=hud_bridge,
        status_path=log_dir / "rynnvalue_live.json",
    )

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
    env = None

    try:
        gui_closed = False
        for task_id in tqdm(range(num_tasks)):
            if env is not None:
                env.close()
                env = None
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
                if args.render_gui:
                    # hard_reset closes the OpenCV viewer; recenter the new window.
                    _arm_libero_gui_center()
                action_plan: Deque[Any] = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
                replay_images = []
                hud_images = []
                done = False
                if live_hud is not None:
                    live_hud.reset_episode(hud_bridge.instruction(task_description))

                for t in range(max_steps + args.num_steps_wait):
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(dummy)
                        if args.render_gui and not _render_gui_frame(env):
                            gui_closed = True
                            break
                        # Feed eai 图像预览 during settle steps too.
                        try:
                            _write_libero_preview_rgb(
                                hud_bridge.extract_rgb(obs, apply_flip=False)
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        continue

                    # openpi/LIBERO policy expects 180°-flipped agentview.
                    img = hud_bridge.extract_rgb(obs, apply_flip=True)
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
                    # Unflipped agentview → eai workspace preview (always).
                    hud_img = hud_bridge.extract_rgb(obs, apply_flip=False)
                    _write_libero_preview_rgb(hud_img)
                    if live_hud is not None:
                        # Unflipped: score + display match sim viewer.
                        live_hud.push_frame(
                            hud_img, step=t - args.num_steps_wait
                        )
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
                    if args.render_gui and not _render_gui_frame(env):
                        gui_closed = True
                        break
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                if gui_closed:
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

            if task_episodes:
                rate = task_successes / task_episodes
                results_per_task[task_description] = rate
                logging.info(
                    "Task %s: %s %s/%s (%.1f%%)",
                    "interrupted" if gui_closed else "done",
                    task_description,
                    task_successes,
                    task_episodes,
                    100.0 * rate,
                )
            if gui_closed:
                logging.info("LIBERO 图形窗口已关闭，跳过剩余任务")
                break

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
        # Keep last scene visible until window close / Esc / UI「停止」.
        if (
            bool(getattr(args, "render_gui", False))
            and env is not None
            and not _libero_gui_dismissed()
        ):
            _hold_gui_until_stop(env)
        return 0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                logging.debug("env.close skipped: %s", exc)
            env = None
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
        "--viewer_only",
        action="store_true",
        help="Open on-screen env and hold (no policy / no Pi inference)",
    )
    from rynnvalue_sim_bridge import add_rynnvalue_live_hud_args

    add_rynnvalue_live_hud_args(p, profile="libero")
    p.add_argument("--exp_name", default="libero_eval")
    p.add_argument("--log_dir", default=str(pathlib.Path.cwd() / "logs"))
    p.add_argument("--rlinf_root", default="")
    p.add_argument("--openpi_root", default="")
    return p


if __name__ == "__main__":
    from rynnvalue_sim_bridge import finalize_rynnvalue_live_hud_args

    _args = build_parser().parse_args()
    if _args.viewer_only:
        _args.render_gui = True
        _args.no_render_gui = False
    if _args.no_render_gui:
        _args.render_gui = False
    finalize_rynnvalue_live_hud_args(_args)
    raise SystemExit(main(_args))
