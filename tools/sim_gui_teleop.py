"""Shared GUI teleop bridge for all EAI simulation backends.

Reads ``gui_robot_cmd.json`` / writes ``sim_robot_state.json`` under
``ISAAC_CAM_BRIDGE_DIR`` (same protocol as RoboDojo ``NoopModelClient``).

Pose format: ``[x, y, z, qw, qx, qy, qz]``. Gripper: ``0=open, 1=closed``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EAI_DIR = os.path.abspath(
    os.environ.get("EAI_DIR")
    or os.path.join(os.path.dirname(__file__), "..")
)
if _EAI_DIR not in sys.path:
    sys.path.insert(0, _EAI_DIR)


def _load_bridge():
    try:
        import isaac_cam_bridge_sink as sink  # type: ignore

        return sink
    except Exception:
        sink_path = os.path.join(_EAI_DIR, "isaac_cam_bridge_sink.py")
        if not os.path.isfile(sink_path):
            return None
        import importlib.util

        spec = importlib.util.spec_from_file_location("isaac_cam_bridge_sink", sink_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def ensure_bridge_dir(root: str = "") -> str:
    d = (root or os.environ.get("ISAAC_CAM_BRIDGE_DIR") or "").strip()
    if not d:
        d = os.path.join(_EAI_DIR, ".cache", "isaac_cam_bridge")
    os.makedirs(d, exist_ok=True)
    os.environ["ISAAC_CAM_BRIDGE_DIR"] = d
    os.environ.setdefault("EAI_DIR", _EAI_DIR)
    return d


def pose7_ok(val: Any) -> Optional[List[float]]:
    if val is None:
        return None
    try:
        arr = [float(x) for x in list(val)]
    except (TypeError, ValueError):
        return None
    if len(arr) != 7 or not all(np.isfinite(arr)):
        return None
    return arr


def grip01(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple)) and val:
            v = float(val[0])
        else:
            v = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(np.clip(v, 0.0, 1.0))


class GuiTeleopSession:
    """Poll GUI cmd JSON and publish sim state for one backend loop."""

    def __init__(self, bridge_dir: str = "", *, label: str = "sim") -> None:
        self.bridge_dir = ensure_bridge_dir(bridge_dir)
        self.label = label or "sim"
        self._bridge = _load_bridge()
        self._logged = False
        self._busy = False
        self._last_cmd_ts = 0.0
        if self._bridge is None:
            print(f"[sim-gui-teleop:{self.label}] bridge unavailable", flush=True)
        else:
            print(
                f"[sim-gui-teleop:{self.label}] dir={self.bridge_dir}",
                flush=True,
            )

    def read_cmd(self) -> Dict[str, Any]:
        if self._bridge is None:
            return {}
        try:
            cmd = self._bridge.read_gui_robot_cmd(root=self.bridge_dir)
        except Exception:
            return {}
        if not cmd or not cmd.get("enabled", True):
            return {}
        if not self._logged:
            print(
                f"[sim-gui-teleop:{self.label}] GUI teleop active → "
                f"{self.bridge_dir}/gui_robot_cmd.json",
                flush=True,
            )
            self._logged = True
        return cmd

    def write_state(
        self,
        *,
        left_ee_pose: Any = None,
        right_ee_pose: Any = None,
        left_gripper: Any = None,
        right_gripper: Any = None,
        objects: Any = None,
    ) -> None:
        if self._bridge is None:
            return
        try:
            self._bridge.write_sim_robot_state(
                root=self.bridge_dir,
                left_ee_pose=left_ee_pose,
                right_ee_pose=right_ee_pose,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                objects=objects,
            )
        except Exception:
            pass

    def cmd_is_new(self, cmd: Dict[str, Any]) -> bool:
        try:
            ts = float(cmd.get("ts", 0.0))
        except (TypeError, ValueError):
            return False
        if ts <= 0 or ts <= self._last_cmd_ts + 1e-6:
            return False
        self._last_cmd_ts = ts
        return True


def libero_obs_to_state(obs: dict) -> Tuple[Optional[List[float]], Optional[float]]:
    """Return (left_ee_pose7_wxyz, gripper01) from LIBERO obs."""
    try:
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
        # robosuite eef_quat is usually (x,y,z,w)
        xyzw = quat
        pose7 = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(xyzw[3]),
            float(xyzw[0]),
            float(xyzw[1]),
            float(xyzw[2]),
        ]
    except Exception:
        pose7 = None
    grip = None
    try:
        q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
        # wider fingers → more open; map roughly to [0,1] closedness
        open_amt = float(np.mean(np.abs(q)))
        grip = float(np.clip(1.0 - open_amt / 0.04, 0.0, 1.0))
    except Exception:
        pass
    return pose7, grip


def libero_objects_to_state(env: Any) -> Dict[str, Dict[str, Any]]:
    """Extract movable object / drawer poses from a LIBERO ControlEnv wrapper."""
    out: Dict[str, Dict[str, Any]] = {}
    inner = getattr(env, "env", env)
    states = getattr(inner, "object_states_dict", None)
    if not isinstance(states, dict):
        return out
    for name, st in states.items():
        entry: Dict[str, Any] = {}
        try:
            geom = st.get_geom_state()
            if isinstance(geom, dict):
                pos = geom.get("pos")
                quat = geom.get("quat")
                if pos is not None:
                    p = np.asarray(pos, dtype=np.float64).reshape(-1)
                    if p.size >= 3:
                        entry["pos"] = [float(p[0]), float(p[1]), float(p[2])]
                if quat is not None:
                    q = np.asarray(quat, dtype=np.float64).reshape(-1)
                    if q.size >= 4:
                        entry["quat"] = [
                            float(q[0]),
                            float(q[1]),
                            float(q[2]),
                            float(q[3]),
                        ]
        except Exception:
            pass
        try:
            if hasattr(st, "get_joint_state"):
                js = st.get_joint_state()
                if js is not None:
                    arr = np.asarray(js, dtype=np.float64).reshape(-1)
                    if arr.size:
                        entry["joint"] = float(arr[0])
        except Exception:
            pass
        if entry:
            out[str(name)] = entry
    return out


def libero_action_from_cmd(
    obs: dict,
    cmd: Dict[str, Any],
    *,
    max_pos_step: float = 0.04,
    max_ori_step: float = 0.25,
) -> List[float]:
    """Build 7-D OSC_POSE action toward GUI absolute target (or hold)."""
    cur_pose, cur_grip = libero_obs_to_state(obs)
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    if cur_grip is not None:
        # LIBERO: -1 open, +1 close
        action[6] = float(np.clip(2.0 * cur_grip - 1.0, -1.0, 1.0))

    target = pose7_ok(cmd.get("left_ee_pose") or cmd.get("right_ee_pose"))
    if cur_pose is not None and target is not None:
        dx = float(np.clip(target[0] - cur_pose[0], -max_pos_step, max_pos_step))
        dy = float(np.clip(target[1] - cur_pose[1], -max_pos_step, max_pos_step))
        dz = float(np.clip(target[2] - cur_pose[2], -max_pos_step, max_pos_step))
        action[0], action[1], action[2] = dx, dy, dz
        # orientation: small axis-angle from quat difference (skip if tiny)
        try:
            q_cur = np.array(
                [cur_pose[4], cur_pose[5], cur_pose[6], cur_pose[3]], dtype=np.float64
            )
            q_tgt = np.array(
                [target[4], target[5], target[6], target[3]], dtype=np.float64
            )
            # relative rotation approx via vector part difference
            if np.dot(q_cur, q_tgt) < 0:
                q_tgt = -q_tgt
            dxyz = np.clip(q_tgt[:3] - q_cur[:3], -max_ori_step, max_ori_step)
            action[3], action[4], action[5] = (
                float(dxyz[0]),
                float(dxyz[1]),
                float(dxyz[2]),
            )
        except Exception:
            pass

    g = grip01(cmd.get("left_gripper"))
    if g is None:
        g = grip01(cmd.get("right_gripper"))
    if g is not None:
        action[6] = float(np.clip(2.0 * g - 1.0, -1.0, 1.0))
    return action


def robotwin_pose_to_pose7(pose: Sequence[float]) -> Optional[List[float]]:
    return pose7_ok(pose)


def robotwin_action_from_cmd(
    left_pose: Sequence[float],
    right_pose: Sequence[float],
    left_grip_open: float,
    right_grip_open: float,
    cmd: Dict[str, Any],
) -> List[float]:
    """16-D ee action: left_pose7 + left_grip_open + right_pose7 + right_grip_open.

    RoboTwin gripper: larger = more open. GUI gripper: 0=open, 1=closed.
    """
    lp = pose7_ok(cmd.get("left_ee_pose")) or pose7_ok(left_pose) or list(left_pose)
    rp = pose7_ok(cmd.get("right_ee_pose")) or pose7_ok(right_pose) or list(right_pose)
    lg = grip01(cmd.get("left_gripper"))
    rg = grip01(cmd.get("right_gripper"))
    left_open = (1.0 - lg) if lg is not None else float(left_grip_open)
    right_open = (1.0 - rg) if rg is not None else float(right_grip_open)
    return list(lp) + [left_open] + list(rp) + [right_open]


def mujoco_guess_ee_pose(model, data, side: str = "left") -> Optional[List[float]]:
    """Best-effort EE pose from site/body name heuristics."""
    import mujoco

    tokens = (
        (f"{side}_tcp", f"{side}_ee", f"{side}_hand", f"{side}_gripper", "tcp", "ee")
        if side in ("left", "right")
        else ("tcp", "ee", "end_effector", "hand")
    )
    for name in tokens:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            pos = data.site_xpos[sid].copy()
            rot = data.site_xmat[sid].reshape(3, 3)
            # mat → wxyz
            qw = float(np.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0)
            qx = float((rot[2, 1] - rot[1, 2]) / (4.0 * qw + 1e-9))
            qy = float((rot[0, 2] - rot[2, 0]) / (4.0 * qw + 1e-9))
            qz = float((rot[1, 0] - rot[0, 1]) / (4.0 * qw + 1e-9))
            return [float(pos[0]), float(pos[1]), float(pos[2]), qw, qx, qy, qz]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos = data.xpos[bid].copy()
            return [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                1.0,
                0.0,
                0.0,
                0.0,
            ]
    return None


__all__ = [
    "GuiTeleopSession",
    "ensure_bridge_dir",
    "grip01",
    "libero_action_from_cmd",
    "libero_objects_to_state",
    "libero_obs_to_state",
    "mujoco_guess_ee_pose",
    "pose7_ok",
    "robotwin_action_from_cmd",
    "robotwin_pose_to_pose7",
]
