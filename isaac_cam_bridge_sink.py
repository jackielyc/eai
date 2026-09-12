#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac / RoboDojo → 共享目录帧写出（无 ROS 依赖，可在 Python 3.11 仿真进程内调用）。

环境变量:
  ISAAC_CAM_BRIDGE_DIR   共享目录，例如 /tmp/isaac_cam_bridge
  ISAAC_CAM_BRIDGE_EVERY 每隔 N 帧写一次（默认 1）

用法（在 RoboDojo eval_env._stream_vision 中）::

    from isaac_cam_bridge_sink import write_cam_frame, write_cam_depth
    write_cam_frame(cam_key, color_rgb)
    write_cam_depth(cam_key, depth_m_or_u16)

帧文件:
  {DIR}/{cam_key}.npy         HxWx3 uint8 RGB
  {DIR}/{cam_key}_depth.npy   HxW uint16 毫米深度（与真机 /camera/*_depth 一致）

遥控（策略=无）:
  {DIR}/gui_robot_cmd.json    UI → Isaac 目标位姿 / 夹爪
  {DIR}/sim_robot_state.json  Isaac → UI 当前 TCP / 夹爪
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

_counters: Dict[str, int] = {}
_depth_counters: Dict[str, int] = {}
_last_log_s = 0.0


def bridge_dir() -> str:
    return os.environ.get("ISAAC_CAM_BRIDGE_DIR", "").strip()


def _bridge_every(every: Optional[int]) -> int:
    if every is not None:
        return max(1, int(every))
    try:
        return max(1, int(os.environ.get("ISAAC_CAM_BRIDGE_EVERY", "1")))
    except ValueError:
        return 1


def _atomic_npy_write(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".writing"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    with open(path + ".stamp", "w", encoding="utf-8") as f:
        f.write(f"{time.time():.6f}\n")


def _log_fail(key: str, exc: Exception) -> None:
    global _last_log_s
    now = time.time()
    if now - _last_log_s > 5.0:
        _last_log_s = now
        print(f"[isaac_cam_bridge_sink] write failed ({key}): {exc}", flush=True)


def write_cam_frame(cam_key: str, color: object, *, every: Optional[int] = None) -> bool:
    """把一帧 RGB 写入共享目录。成功返回 True。"""
    root = bridge_dir()
    if not root:
        return False
    key = str(cam_key or "").strip()
    if not key:
        return False
    try:
        arr = np.asarray(color)
    except Exception:
        return False
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False

    every_n = _bridge_every(every)
    n = _counters.get(key, 0) + 1
    _counters[key] = n
    if (n - 1) % every_n != 0:
        return False

    rgb = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    try:
        _atomic_npy_write(os.path.join(root, f"{key}.npy"), rgb)
    except Exception as exc:
        _log_fail(key, exc)
        return False
    return True


def _depth_to_u16_mm(depth: object) -> Optional[np.ndarray]:
    """归一化为 HxW uint16 毫米深度。"""
    try:
        arr = np.asarray(depth)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr)
    depth_f = arr.astype(np.float32)
    finite = depth_f[np.isfinite(depth_f)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16)
    # 常见：米制 float（场景深度通常 < 20m）；否则按毫米写入
    if float(np.nanmax(finite)) < 100.0:
        depth_u16 = np.clip(depth_f * 1000.0, 0, 65535).astype(np.uint16)
    else:
        depth_u16 = np.clip(depth_f, 0, 65535).astype(np.uint16)
    # 无效/过远置 0
    bad = ~np.isfinite(depth_f) | (depth_f <= 0)
    depth_u16[bad] = 0
    return np.ascontiguousarray(depth_u16)


def write_cam_depth(cam_key: str, depth: object, *, every: Optional[int] = None) -> bool:
    """把一帧深度写入共享目录（uint16 mm）。成功返回 True。"""
    root = bridge_dir()
    if not root:
        return False
    key = str(cam_key or "").strip()
    if not key:
        return False
    depth_u16 = _depth_to_u16_mm(depth)
    if depth_u16 is None:
        return False

    every_n = _bridge_every(every)
    n = _depth_counters.get(key, 0) + 1
    _depth_counters[key] = n
    if (n - 1) % every_n != 0:
        return False

    try:
        _atomic_npy_write(os.path.join(root, f"{key}_depth.npy"), depth_u16)
    except Exception as exc:
        _log_fail(f"{key}_depth", exc)
        return False
    return True


# ---------------------------------------------------------------------------
# GUI ↔ Isaac 手臂/手遥控（策略=无 / NoopModelClient 时生效）
# ---------------------------------------------------------------------------
# gui_robot_cmd.json   UI 写入的目标位姿/夹爪
# sim_robot_state.json 仿真回写的当前 TCP / 夹爪
#
# ee_pose: [x, y, z, qw, qx, qy, qz]（与 RoboDojo get_real_endpose 一致）
# gripper: 0=开, 1=合

GUI_ROBOT_CMD_NAME = "gui_robot_cmd.json"
SIM_ROBOT_STATE_NAME = "sim_robot_state.json"


def _atomic_json_write(path: str, data: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".writing"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _as_pose7(val: Any) -> Optional[list]:
    if val is None:
        return None
    try:
        arr = [float(x) for x in list(val)]
    except (TypeError, ValueError):
        return None
    if len(arr) != 7:
        return None
    if not all(np.isfinite(x) for x in arr):
        return None
    return arr


def _as_gripper(val: Any) -> Optional[float]:
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


def gui_robot_cmd_path(root: Optional[str] = None) -> str:
    d = (root or bridge_dir() or "").strip()
    return os.path.join(d, GUI_ROBOT_CMD_NAME) if d else ""


def sim_robot_state_path(root: Optional[str] = None) -> str:
    d = (root or bridge_dir() or "").strip()
    return os.path.join(d, SIM_ROBOT_STATE_NAME) if d else ""


def read_gui_robot_cmd(root: Optional[str] = None) -> Optional[dict]:
    path = gui_robot_cmd_path(root)
    if not path:
        return None
    data = _read_json(path)
    if not data:
        return None
    return data


def write_gui_robot_cmd(
    *,
    root: Optional[str] = None,
    left_ee_pose: Any = None,
    right_ee_pose: Any = None,
    left_gripper: Any = None,
    right_gripper: Any = None,
    enabled: bool = True,
    merge: bool = True,
) -> bool:
    """写入 GUI 遥控指令。merge=True 时保留未传入字段的旧值。"""
    d = (root or bridge_dir() or "").strip()
    if not d:
        return False
    path = os.path.join(d, GUI_ROBOT_CMD_NAME)
    prev = _read_json(path) if merge else None
    if not isinstance(prev, dict):
        prev = {}
    out: dict = {
        "ts": time.time(),
        "enabled": bool(enabled if enabled is not None else prev.get("enabled", True)),
    }
    for key, val, caster in (
        ("left_ee_pose", left_ee_pose, _as_pose7),
        ("right_ee_pose", right_ee_pose, _as_pose7),
        ("left_gripper", left_gripper, _as_gripper),
        ("right_gripper", right_gripper, _as_gripper),
    ):
        casted = caster(val) if val is not None else None
        if casted is not None:
            out[key] = casted
        elif merge and key in prev:
            out[key] = prev[key]
    try:
        _atomic_json_write(path, out)
    except Exception as exc:
        _log_fail("gui_robot_cmd", exc)
        return False
    return True


def clear_gui_robot_cmd(root: Optional[str] = None) -> None:
    path = gui_robot_cmd_path(root)
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def read_sim_robot_state(root: Optional[str] = None) -> Optional[dict]:
    path = sim_robot_state_path(root)
    if not path:
        return None
    return _read_json(path)


def write_sim_robot_state(
    *,
    root: Optional[str] = None,
    left_ee_pose: Any = None,
    right_ee_pose: Any = None,
    left_gripper: Any = None,
    right_gripper: Any = None,
) -> bool:
    d = (root or bridge_dir() or "").strip()
    if not d:
        return False
    out: dict = {"ts": time.time()}
    lp = _as_pose7(left_ee_pose)
    rp = _as_pose7(right_ee_pose)
    lg = _as_gripper(left_gripper)
    rg = _as_gripper(right_gripper)
    if lp is not None:
        out["left_ee_pose"] = lp
    if rp is not None:
        out["right_ee_pose"] = rp
    if lg is not None:
        out["left_gripper"] = lg
    if rg is not None:
        out["right_gripper"] = rg
    try:
        _atomic_json_write(os.path.join(d, SIM_ROBOT_STATE_NAME), out)
    except Exception as exc:
        _log_fail("sim_robot_state", exc)
        return False
    return True


def sim_state_is_fresh(state: Optional[Mapping[str, Any]], max_age_s: float = 2.0) -> bool:
    if not state:
        return False
    try:
        ts = float(state.get("ts", 0.0))
    except (TypeError, ValueError):
        return False
    return ts > 0 and (time.time() - ts) <= max_age_s
