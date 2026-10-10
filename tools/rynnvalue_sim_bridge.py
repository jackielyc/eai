#!/usr/bin/env python3
"""Env-agnostic adapters for RynnValue Live HUD.

``RynnValueLiveHud`` only needs RGB frames + text meta. Each sim backend
implements a thin ``SimHudBridge`` that:

  - extracts RGB (H,W,3) uint8 from that env's observation / render
  - supplies robot / camera descriptions for reward_server
  - maps episode context → language instruction

Shared CLI / argv helpers keep ``--rynnvalue_*`` identical across
``run_*_eval.py`` wrappers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from rynnvalue_live_hud import (
    DEFAULT_CAMERA_DESCRIPTION,
    DEFAULT_ROBOT_DESCRIPTION,
    DEFAULT_SERVER_URL,
    RynnValueLiveHud,
)


@dataclass(frozen=True)
class HudBackendProfile:
    """Default meta for a simulation backend."""

    name: str
    robot_description: str
    camera_description: str
    camera_key: str = ""


# Profiles are defaults only; CLI flags always override when non-empty.
HUD_PROFILES: Dict[str, HudBackendProfile] = {
    "libero": HudBackendProfile(
        name="libero",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    "robotwin": HudBackendProfile(
        name="robotwin",
        robot_description="a dual-arm ALOHA robot workstation",
        camera_description="the third-person head / full scene camera",
        camera_key="full_image",
    ),
    "molmospaces": HudBackendProfile(
        name="molmospaces",
        robot_description="a mobile manipulator in MolmoSpaces",
        camera_description="the exo / third-person scene camera",
        camera_key="exo_camera_1",
    ),
    "mujoco": HudBackendProfile(
        name="mujoco",
        robot_description="a robot in MuJoCo",
        camera_description="the free / third-person MuJoCo camera",
        camera_key="",
    ),
    "arena": HudBackendProfile(
        name="arena",
        robot_description="a robot in IsaacLab-Arena",
        camera_description="the embodiment-mounted camera",
        camera_key="camera_obs",
    ),
    "isaac": HudBackendProfile(
        name="isaac",
        robot_description="a dual-arm robot in Isaac Sim / RoboDojo",
        camera_description="the head / third-person camera",
        camera_key="cam_head",
    ),
    "simpler": HudBackendProfile(
        name="simpler",
        robot_description="a Google Robot or WidowX arm in SimplerEnv",
        camera_description="the third-person / wrist camera",
        camera_key="",
    ),
    "calvin": HudBackendProfile(
        name="calvin",
        robot_description="a Franka Panda in CALVIN",
        camera_description="the static / gripper camera",
        camera_key="",
    ),
    "robocasa": HudBackendProfile(
        name="robocasa",
        robot_description="a Franka kitchen robot in RoboCasa",
        camera_description="the third-person kitchen camera",
        camera_key="",
    ),
    "robocasa365": HudBackendProfile(
        name="robocasa365",
        robot_description="a kitchen robot in RoboCasa365",
        camera_description="the third-person kitchen camera",
        camera_key="",
    ),
    "maniskill3": HudBackendProfile(
        name="maniskill3",
        robot_description="a robot in ManiSkill3",
        camera_description="the third-person / wrist camera",
        camera_key="",
    ),
    "maniskill2": HudBackendProfile(
        name="maniskill2",
        robot_description="a robot in ManiSkill2",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "metaworld": HudBackendProfile(
        name="metaworld",
        robot_description="a Sawyer arm in Meta-World",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "rlbench": HudBackendProfile(
        name="rlbench",
        robot_description="a Franka Panda in RLBench",
        camera_description="the front / wrist camera",
        camera_key="",
    ),
    "libero_plus": HudBackendProfile(
        name="libero_plus",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    "libero_pro": HudBackendProfile(
        name="libero_pro",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    "libero_mem": HudBackendProfile(
        name="libero_mem",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    "behavior1k": HudBackendProfile(
        name="behavior1k",
        robot_description="a mobile manipulator in BEHAVIOR-1K",
        camera_description="the egocentric RGB-D camera",
        camera_key="",
    ),
    "furniturebench": HudBackendProfile(
        name="furniturebench",
        robot_description="a Franka assembling furniture",
        camera_description="the third-person / wrist camera",
        camera_key="",
    ),
    "vlabench": HudBackendProfile(
        name="vlabench",
        robot_description="a robot in VLABench",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "genesis": HudBackendProfile(
        name="genesis",
        robot_description="a robot in Genesis",
        camera_description="the free camera",
        camera_key="",
    ),
    "mikasa": HudBackendProfile(
        name="mikasa",
        robot_description="a robot in MIKASA-Robo",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "robocerebra": HudBackendProfile(
        name="robocerebra",
        robot_description="a robot in RoboCerebra",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "robomme": HudBackendProfile(
        name="robomme",
        robot_description="a robot in RoboMME",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "duobench": HudBackendProfile(
        name="duobench",
        robot_description="a dual-arm robot in DuoBench",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "kinetix": HudBackendProfile(
        name="kinetix",
        robot_description="a robot in Kinetix",
        camera_description="the third-person camera",
        camera_key="",
    ),
    "vla_harness": HudBackendProfile(
        name="vla_harness",
        robot_description="a VLA policy under unified harness evaluation",
        camera_description="benchmark-specific cameras",
        camera_key="",
    ),
}


def get_hud_profile(name: str) -> HudBackendProfile:
    key = (name or "").strip().lower() or "libero"
    if key in HUD_PROFILES:
        return HUD_PROFILES[key]
    return HudBackendProfile(
        name=key,
        robot_description=DEFAULT_ROBOT_DESCRIPTION,
        camera_description=DEFAULT_CAMERA_DESCRIPTION,
    )


@runtime_checkable
class SimHudBridge(Protocol):
    """Minimal contract for wiring any sim loop to Live HUD."""

    def robot_description(self) -> str: ...

    def camera_description(self) -> str: ...

    def instruction(self, episode_ctx: Any) -> str: ...

    def extract_rgb(self, obs_or_render: Any) -> np.ndarray: ...


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unexpected frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _instruction_from_ctx(episode_ctx: Any) -> str:
    if isinstance(episode_ctx, Mapping):
        for key in ("instruction", "language", "task", "prompt", "task_descriptions"):
            val = episode_ctx.get(key)
            if val is None:
                continue
            if isinstance(val, (list, tuple)) and val:
                return str(val[0])
            if val:
                return str(val)
    return str(episode_ctx or "")


@dataclass
class LiberoHudBridge:
    """LIBERO / robosuite agentview RGB + Franka meta.

    Default ``flip_hw=False``: Live HUD score/display match the on-screen
    robosuite viewer. Pass ``apply_flip=True`` only when you explicitly need
    the openpi/LIBERO policy orientation (180° ``[::-1, ::-1]``).
    """

    profile: HudBackendProfile = HUD_PROFILES["libero"]
    flip_hw: bool = False

    def robot_description(self) -> str:
        return self.profile.robot_description

    def camera_description(self) -> str:
        return self.profile.camera_description

    def instruction(self, episode_ctx: Any) -> str:
        return _instruction_from_ctx(episode_ctx)

    def extract_rgb(
        self, obs_or_render: Any, *, apply_flip: Optional[bool] = None
    ) -> np.ndarray:
        if isinstance(obs_or_render, np.ndarray):
            frame = obs_or_render
        elif isinstance(obs_or_render, Mapping):
            key = self.profile.camera_key or "agentview_image"
            if key not in obs_or_render:
                raise KeyError(
                    f"LIBERO obs missing camera key {key!r}; "
                    f"keys={list(obs_or_render.keys())[:12]}"
                )
            frame = obs_or_render[key]
        else:
            raise TypeError(f"unsupported LIBERO frame source: {type(obs_or_render)}")
        rgb = _as_uint8_rgb(frame)
        do_flip = self.flip_hw if apply_flip is None else bool(apply_flip)
        if do_flip:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        return rgb


@dataclass
class RoboTwinHudBridge:
    """RoboTwin / ALOHA: ``full_image`` or extracted ``main_images[0]`` RGB."""

    profile: HudBackendProfile = HUD_PROFILES["robotwin"]

    def robot_description(self) -> str:
        return self.profile.robot_description

    def camera_description(self) -> str:
        return self.profile.camera_description

    def instruction(self, episode_ctx: Any) -> str:
        return _instruction_from_ctx(episode_ctx)

    def extract_rgb(self, obs_or_render: Any) -> np.ndarray:
        if isinstance(obs_or_render, np.ndarray):
            return _as_uint8_rgb(obs_or_render)
        if isinstance(obs_or_render, Mapping):
            for key in (
                self.profile.camera_key,
                "full_image",
                "main_images",
                "head_camera",
                "image",
            ):
                if not key or key not in obs_or_render:
                    continue
                frame = obs_or_render[key]
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                arr = np.asarray(frame)
                if arr.ndim == 4:
                    arr = arr[0]
                return _as_uint8_rgb(arr)
            raise KeyError(
                "RoboTwin obs missing camera frame; "
                f"keys={list(obs_or_render.keys())[:12]}"
            )
        raise TypeError(f"unsupported RoboTwin frame source: {type(obs_or_render)}")


def make_hud_bridge(backend: str) -> SimHudBridge:
    """Factory for known backends (unknown → Libero-style defaults via profile)."""
    key = (backend or "").strip().lower()
    if key in ("robotwin", "robo_twin", "aloha"):
        return RoboTwinHudBridge()
    if key in ("libero", ""):
        return LiberoHudBridge()
    # Generic: reuse RoboTwin extractor (no flip) with that profile's meta.
    bridge = RoboTwinHudBridge(profile=get_hud_profile(key))
    return bridge

def add_rynnvalue_live_hud_args(
    parser: argparse.ArgumentParser,
    *,
    profile: str = "libero",
) -> argparse.ArgumentParser:
    """Attach the shared ``--rynnvalue_*`` flags (identical across sim evals)."""
    prof = get_hud_profile(profile)
    parser.add_argument(
        "--rynnvalue_live_hud",
        action="store_true",
        help="Enable async RynnValue Live HUD (needs reward_server on --rynnvalue_server_url)",
    )
    parser.add_argument(
        "--rynnvalue_server_url",
        default=DEFAULT_SERVER_URL,
        help="RynnValue reward_server base URL",
    )
    parser.add_argument(
        "--rynnvalue_refresh_sec",
        type=float,
        default=1.0,
        help="Live HUD wall-clock refresh interval in seconds",
    )
    parser.add_argument(
        "--rynnvalue_num_frames",
        type=int,
        default=8,
        help="Frames subsampled per Live HUD score request",
    )
    parser.add_argument(
        "--rynnvalue_timeout_s",
        type=float,
        default=60.0,
        help="HTTP timeout for each Live HUD score request",
    )
    parser.add_argument(
        "--rynnvalue_robot_description",
        default=prof.robot_description,
        help=f"Robot text meta for reward_server (default profile={prof.name})",
    )
    parser.add_argument(
        "--rynnvalue_camera_description",
        default=prof.camera_description,
        help=f"Camera text meta for reward_server (default profile={prof.name})",
    )
    parser.add_argument(
        "--rynnvalue_show_window",
        action="store_true",
        default=True,
        help="Show OpenCV Live HUD window (default on)",
    )
    parser.add_argument(
        "--no_rynnvalue_show_window",
        action="store_true",
        help="Disable OpenCV Live HUD window (still scores + writes status/video)",
    )
    return parser


def finalize_rynnvalue_live_hud_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply ``--no_rynnvalue_show_window`` after parse."""
    if bool(getattr(args, "no_rynnvalue_show_window", False)):
        args.rynnvalue_show_window = False
    return args


def append_rynnvalue_live_hud_argv(
    argv: List[str],
    *,
    enabled: bool = False,
    server_url: str = DEFAULT_SERVER_URL,
    refresh_sec: float = 1.0,
    num_frames: int = 8,
    show_window: bool = True,
    robot_description: str = "",
    camera_description: str = "",
    timeout_s: Optional[float] = None,
) -> List[str]:
    """Append shared Live HUD flags to a child-process argv (no-op if disabled)."""
    if not enabled:
        return argv
    argv.append("--rynnvalue_live_hud")
    argv.append(f"--rynnvalue_server_url={server_url or DEFAULT_SERVER_URL}")
    argv.append(f"--rynnvalue_refresh_sec={float(refresh_sec)}")
    argv.append(f"--rynnvalue_num_frames={int(num_frames)}")
    if timeout_s is not None:
        argv.append(f"--rynnvalue_timeout_s={float(timeout_s)}")
    if (robot_description or "").strip():
        argv.append(f"--rynnvalue_robot_description={robot_description.strip()}")
    if (camera_description or "").strip():
        argv.append(f"--rynnvalue_camera_description={camera_description.strip()}")
    if show_window:
        argv.append("--rynnvalue_show_window")
    else:
        argv.append("--no_rynnvalue_show_window")
    return argv


def maybe_create_live_hud(
    args: Any,
    *,
    bridge: SimHudBridge,
    status_path: Optional[str | Path] = None,
) -> Optional[RynnValueLiveHud]:
    """Build + start Live HUD from parsed args + bridge defaults; None if disabled."""
    if not bool(getattr(args, "rynnvalue_live_hud", False)):
        return None
    import os

    # eai GUI injects RYNNVALUE_STATUS_PATH so overlay lands in workspace cache.
    env_status = (os.environ.get("RYNNVALUE_STATUS_PATH") or "").strip()
    resolved_status = env_status or (str(status_path) if status_path else None)
    robot = str(getattr(args, "rynnvalue_robot_description", "") or "").strip()
    camera = str(getattr(args, "rynnvalue_camera_description", "") or "").strip()
    hud = RynnValueLiveHud(
        server_url=str(getattr(args, "rynnvalue_server_url", "") or DEFAULT_SERVER_URL),
        refresh_sec=float(getattr(args, "rynnvalue_refresh_sec", 1.0) or 1.0),
        num_frames=int(getattr(args, "rynnvalue_num_frames", 8) or 8),
        robot_description=robot or bridge.robot_description(),
        camera_description=camera or bridge.camera_description(),
        show_window=bool(getattr(args, "rynnvalue_show_window", True)),
        status_path=resolved_status,
        timeout_s=float(getattr(args, "rynnvalue_timeout_s", 60.0) or 60.0),
        flip_ud=bool(getattr(args, "rynnvalue_flip_ud", True)),
    )
    hud.start()
    return hud


__all__ = [
    "HUD_PROFILES",
    "HudBackendProfile",
    "LiberoHudBridge",
    "RoboTwinHudBridge",
    "SimHudBridge",
    "add_rynnvalue_live_hud_args",
    "append_rynnvalue_live_hud_argv",
    "finalize_rynnvalue_live_hud_args",
    "get_hud_profile",
    "make_hud_bridge",
    "maybe_create_live_hud",
]
