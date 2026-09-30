#!/usr/bin/env python3
"""Attach RynnValue Live HUD from process env (any sim child).

Env knobs (same as RoboTwin / LIBERO)::

  RYNNVALUE_LIVE_HUD=1
  RYNNVALUE_SERVER_URL=http://127.0.0.1:8001
  RYNNVALUE_REFRESH_SEC=1.0
  RYNNVALUE_NUM_FRAMES=8
  RYNNVALUE_ROBOT_DESCRIPTION=...
  RYNNVALUE_CAMERA_DESCRIPTION=...
  RYNNVALUE_STATUS_PATH=...
  RYNNVALUE_NO_SHOW_WINDOW=1
  RYNNVALUE_FRAME_BUS=<dir>   # optional: publish frames instead of local HUD
  RYNNVALUE_HUD_BACKEND=mujoco|molmospaces|arena|isaac|...
  RYNNVALUE_INSTRUCTION=...   # optional default language
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_hud = None
_bus_writer = None
_step = 0
_last_push_t = 0.0
_instruction = ""


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes", "YES")


def enabled() -> bool:
    return _truthy("RYNNVALUE_LIVE_HUD")


def _as_uint8_rgb(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"unexpected frame shape: {getattr(arr, 'shape', None)}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _profile_meta() -> tuple[str, str]:
    backend = (os.environ.get("RYNNVALUE_HUD_BACKEND") or "mujoco").strip().lower()
    try:
        from rynnvalue_sim_bridge import get_hud_profile

        prof = get_hud_profile(backend)
        robot = (
            os.environ.get("RYNNVALUE_ROBOT_DESCRIPTION") or ""
        ).strip() or prof.robot_description
        camera = (
            os.environ.get("RYNNVALUE_CAMERA_DESCRIPTION") or ""
        ).strip() or prof.camera_description
        return robot, camera
    except Exception:
        return (
            (os.environ.get("RYNNVALUE_ROBOT_DESCRIPTION") or "a robot").strip(),
            (os.environ.get("RYNNVALUE_CAMERA_DESCRIPTION") or "the third-person camera").strip(),
        )


def ensure_started() -> Any:
    """Start local OpenCV HUD (or prepare frame-bus writer)."""
    global _hud, _bus_writer, _instruction
    if not enabled():
        return None
    with _lock:
        bus = (os.environ.get("RYNNVALUE_FRAME_BUS") or "").strip()
        if bus:
            if _bus_writer is None:
                from rynnvalue_frame_bus import FrameBusWriter

                _bus_writer = FrameBusWriter(bus)
                logger.info("RynnValue frame-bus writer → %s", bus)
            return _bus_writer

        if _hud is not None:
            return _hud
        from rynnvalue_live_hud import RynnValueLiveHud

        robot, camera = _profile_meta()
        status = (os.environ.get("RYNNVALUE_STATUS_PATH") or "").strip() or None
        _instruction = (os.environ.get("RYNNVALUE_INSTRUCTION") or "").strip()
        _hud = RynnValueLiveHud(
            server_url=(
                os.environ.get("RYNNVALUE_SERVER_URL") or "http://127.0.0.1:8001"
            ).strip(),
            instruction=_instruction,
            refresh_sec=float(os.environ.get("RYNNVALUE_REFRESH_SEC") or 1.0),
            num_frames=int(os.environ.get("RYNNVALUE_NUM_FRAMES") or 8),
            robot_description=robot,
            camera_description=camera,
            show_window=not _truthy("RYNNVALUE_NO_SHOW_WINDOW"),
            status_path=status,
            timeout_s=float(os.environ.get("RYNNVALUE_TIMEOUT_S") or 60.0),
        )
        _hud.start()
        logger.info("RynnValue Live HUD attached url=%s", _hud.server_url)
        return _hud


def set_instruction(instruction: str) -> None:
    global _instruction
    _instruction = instruction or ""
    if _hud is not None:
        _hud.reset_episode(_instruction)


def push_rgb(
    frame: Any,
    *,
    instruction: str = "",
    step: Optional[int] = None,
    min_interval_s: float = 0.12,
) -> None:
    """Push one RGB frame (throttled). No-op when Live HUD disabled."""
    global _step, _last_push_t, _instruction
    if not enabled():
        return
    try:
        ensure_started()
        now = time.time()
        if now - _last_push_t < max(0.05, float(min_interval_s)):
            return
        rgb = _as_uint8_rgb(frame)
        instr = (instruction or _instruction or "").strip()
        if instr and instr != _instruction:
            set_instruction(instr)
        with _lock:
            if step is None:
                _step += 1
                step_i = _step
            else:
                step_i = int(step)
                _step = step_i
            _last_push_t = now
            if _bus_writer is not None:
                _bus_writer.push(rgb, step=step_i, instruction=_instruction)
            elif _hud is not None:
                _hud.push_frame(rgb, step=step_i)
    except Exception as exc:  # noqa: BLE001
        logger.debug("rynnvalue push skipped: %s", exc)


def stop() -> None:
    global _hud, _bus_writer
    with _lock:
        if _hud is not None:
            try:
                _hud.stop()
            except Exception:  # noqa: BLE001
                pass
            _hud = None
        _bus_writer = None


def extract_obs_rgb(obs: Any) -> Optional[np.ndarray]:
    """Best-effort RGB from common sim observation layouts."""
    if obs is None:
        return None
    if isinstance(obs, np.ndarray) or hasattr(obs, "detach"):
        try:
            return _as_uint8_rgb(obs)
        except Exception:
            return None
    if not isinstance(obs, dict):
        return None
    # Nested camera_obs (IsaacLab-Arena)
    cam_obs = obs.get("camera_obs")
    if isinstance(cam_obs, dict) and cam_obs:
        for key in (
            "front_cam",
            "rgb",
            "camera",
            "agentview",
            "exo_camera_1",
            "wrist_camera",
        ):
            if key in cam_obs and cam_obs[key] is not None:
                try:
                    return _as_uint8_rgb(cam_obs[key])
                except Exception:
                    continue
        for _k, v in cam_obs.items():
            if v is None:
                continue
            try:
                return _as_uint8_rgb(v)
            except Exception:
                continue
    for key in (
        "exo_camera_1",
        "wrist_camera",
        "agentview_image",
        "full_image",
        "main_images",
        "image",
        "rgb",
    ):
        if key in obs and obs[key] is not None:
            try:
                return _as_uint8_rgb(obs[key])
            except Exception:
                continue
    return None


def install_arena_rollout_hook() -> bool:
    """Monkeypatch IsaacLab-Arena ``rollout_policy`` to publish frames."""
    if not enabled():
        return False
    try:
        import isaaclab_arena.evaluation.policy_runner as pr
    except Exception as exc:  # noqa: BLE001
        logger.warning("arena policy_runner import failed: %s", exc)
        return False
    if getattr(pr, "_rynnvalue_hud_hooked", False):
        return True
    orig = pr.rollout_policy

    def _wrapped(env, policy, num_steps, num_episodes, language_instruction=None):
        ensure_started()
        instr = language_instruction
        try:
            if not instr:
                instr = getattr(getattr(env, "unwrapped", env).cfg, "task_description", "") or ""
        except Exception:
            instr = language_instruction or ""
        if instr:
            set_instruction(str(instr))

        # Wrap env.step to see observations after each action.
        raw_step = env.step

        def step_with_hud(actions):
            out = raw_step(actions)
            try:
                obs = out[0] if isinstance(out, tuple) else out
                rgb = extract_obs_rgb(obs)
                if rgb is not None:
                    push_rgb(rgb, instruction=str(instr or ""))
            except Exception as exc:  # noqa: BLE001
                logger.debug("arena hud step push: %s", exc)
            return out

        env.step = step_with_hud  # type: ignore[method-assign]
        try:
            return orig(env, policy, num_steps, num_episodes, language_instruction)
        finally:
            env.step = raw_step  # type: ignore[method-assign]
            stop()

    pr.rollout_policy = _wrapped  # type: ignore[assignment]
    pr._rynnvalue_hud_hooked = True  # type: ignore[attr-defined]
    logger.info("Arena rollout_policy HUD hook installed")
    return True


def child_env_from_settings(
    settings: dict,
    *,
    backend: str,
    status_path: str = "",
    frame_bus: str = "",
    instruction: str = "",
) -> dict:
    """Build env dict for QProcess / subprocess from GUI HUD settings."""
    if not settings.get("enabled"):
        return {}
    env = {
        "RYNNVALUE_LIVE_HUD": "1",
        "RYNNVALUE_SERVER_URL": str(settings.get("server_url") or "http://127.0.0.1:8001"),
        "RYNNVALUE_REFRESH_SEC": str(float(settings.get("refresh_sec") or 1.0)),
        "RYNNVALUE_NUM_FRAMES": str(int(settings.get("num_frames") or 8)),
        "RYNNVALUE_HUD_BACKEND": str(backend or "mujoco"),
    }
    if not settings.get("show_window", True):
        env["RYNNVALUE_NO_SHOW_WINDOW"] = "1"
    if status_path:
        env["RYNNVALUE_STATUS_PATH"] = status_path
    if frame_bus:
        env["RYNNVALUE_FRAME_BUS"] = frame_bus
    if instruction:
        env["RYNNVALUE_INSTRUCTION"] = instruction
    return env


__all__ = [
    "child_env_from_settings",
    "enabled",
    "ensure_started",
    "extract_obs_rgb",
    "install_arena_rollout_hook",
    "push_rgb",
    "set_instruction",
    "stop",
]
