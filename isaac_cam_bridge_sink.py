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
  {DIR}/{cam_key}_extrinsic.npy  4x4 float64，相机→base/env 相对系（与 TCP 同系）
      约定：列为相机轴；默认已是 OpenCV/ROS optical（X右 Y下 Z前）。
      若来自 USD 相机 xform，写出前需做 OpenGL→optical 转换。

遥控（策略=无）:
  {DIR}/gui_robot_cmd.json    UI → Isaac 目标位姿 / 夹爪
  {DIR}/sim_robot_state.json  Isaac → UI 当前 TCP / 夹爪

任务暂停（所有仿真后端）:
  {DIR}/sim_control.json      UI → paused: true|false；子进程在 step 前轮询
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

_counters: Dict[str, int] = {}
_depth_counters: Dict[str, int] = {}
_extrinsic_counters: Dict[str, int] = {}
_last_log_s = 0.0

# USD/OpenGL 相机（X右 Y上 -Z前）→ ROS/OpenCV optical（X右 Y下 +Z前）
_R_USD_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def usd_cam_to_optical_extrinsic(T_base_usd: np.ndarray) -> np.ndarray:
    """把 camera-to-base（USD 轴）转为 optical-to-base。"""
    T = np.asarray(T_base_usd, dtype=np.float64).reshape(4, 4).copy()
    T_usd_from_opt = np.eye(4, dtype=np.float64)
    T_usd_from_opt[:3, :3] = _R_USD_FROM_OPTICAL
    return T @ T_usd_from_opt


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


def write_cam_extrinsic(
    cam_key: str,
    T_cam2base: object,
    *,
    every: Optional[int] = None,
    already_optical: bool = True,
) -> bool:
    """写出 4x4 相机→base 外参（与仿真 TCP 同一相对系）。

    ``already_optical=False`` 时按 USD/OpenGL 相机轴做转换后再写。
    """
    root = bridge_dir()
    if not root:
        return False
    key = str(cam_key or "").strip()
    if not key:
        return False
    try:
        T = np.asarray(T_cam2base, dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    if not np.isfinite(T).all():
        return False
    if not already_optical:
        T = usd_cam_to_optical_extrinsic(T)

    every_n = _bridge_every(every)
    n = _extrinsic_counters.get(key, 0) + 1
    _extrinsic_counters[key] = n
    if (n - 1) % every_n != 0:
        return False

    try:
        _atomic_npy_write(
            os.path.join(root, f"{key}_extrinsic.npy"),
            np.ascontiguousarray(T, dtype=np.float64),
        )
    except Exception as exc:
        _log_fail(f"{key}_extrinsic", exc)
        return False
    return True


def read_cam_extrinsic(
    cam_key: str, *, root: Optional[str] = None, max_age_s: float = 3.0
) -> Optional[np.ndarray]:
    """读新鲜的相机→base 外参；过期或不存在返回 None。"""
    d = (root or bridge_dir() or "").strip()
    key = str(cam_key or "").strip()
    if not d or not key:
        return None
    path = os.path.join(d, f"{key}_extrinsic.npy")
    stamp_path = path + ".stamp"
    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            ts = float(f.read().strip())
        if max_age_s > 0 and (time.time() - ts) > float(max_age_s):
            return None
        with open(path, "rb") as f:
            T = np.load(f)
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    if not np.isfinite(T).all():
        return None
    return T


def sim_extrinsics_active(*, root: Optional[str] = None, max_age_s: float = 3.0) -> bool:
    """Isaac 外参是否在刷（用于暂停 A2D 真机 TF，避免污染仿真 6D）。"""
    d = (root or bridge_dir() or "").strip()
    if not d:
        d = os.environ.get("ISAAC_CAM_BRIDGE_DIR", "").strip()
    if not d:
        return False
    for key in ("cam_head", "cam_high", "head_camera"):
        if read_cam_extrinsic(key, root=d, max_age_s=max_age_s) is not None:
            return True
    return False


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
# sim_control.json     UI 写入的暂停/继续（所有仿真后端共用）
#
# ee_pose: [x, y, z, qw, qx, qy, qz]（与 RoboDojo get_real_endpose 一致）
# gripper: 0=开, 1=合

GUI_ROBOT_CMD_NAME = "gui_robot_cmd.json"
SIM_ROBOT_STATE_NAME = "sim_robot_state.json"
SIM_CONTROL_NAME = "sim_control.json"


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
    objects: Any = None,
    merge: bool = True,
) -> bool:
    d = (root or bridge_dir() or "").strip()
    if not d:
        return False
    prev = _read_json(os.path.join(d, SIM_ROBOT_STATE_NAME)) if merge else None
    if not isinstance(prev, dict):
        prev = {}
    out: dict = {"ts": time.time()}
    lp = _as_pose7(left_ee_pose)
    rp = _as_pose7(right_ee_pose)
    lg = _as_gripper(left_gripper)
    rg = _as_gripper(right_gripper)
    # merge：未传的字段保留上一帧，避免 TCP 写与 objects 写互相覆盖
    if lp is not None:
        out["left_ee_pose"] = lp
    elif "left_ee_pose" in prev:
        out["left_ee_pose"] = prev["left_ee_pose"]
    if rp is not None:
        out["right_ee_pose"] = rp
    elif "right_ee_pose" in prev:
        out["right_ee_pose"] = prev["right_ee_pose"]
    if lg is not None:
        out["left_gripper"] = lg
    elif "left_gripper" in prev:
        out["left_gripper"] = prev["left_gripper"]
    if rg is not None:
        out["right_gripper"] = rg
    elif "right_gripper" in prev:
        out["right_gripper"] = prev["right_gripper"]
    if objects is None and isinstance(prev.get("objects"), dict):
        objects = prev["objects"]
    if isinstance(objects, dict) and objects:
        # 只保留可 JSON 序列化的简单位姿字段
        clean: dict = {}
        for name, meta in objects.items():
            if not isinstance(meta, dict):
                continue
            entry: dict = {}
            pos = meta.get("pos")
            quat = meta.get("quat")
            joint = meta.get("joint")
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                try:
                    entry["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
                except (TypeError, ValueError):
                    pass
            if isinstance(quat, (list, tuple)) and len(quat) >= 4:
                try:
                    entry["quat"] = [
                        float(quat[0]),
                        float(quat[1]),
                        float(quat[2]),
                        float(quat[3]),
                    ]
                except (TypeError, ValueError):
                    pass
            if joint is not None:
                try:
                    entry["joint"] = float(joint)
                except (TypeError, ValueError):
                    pass
            if entry:
                clean[str(name)] = entry
        if clean:
            out["objects"] = clean
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


# ---------------------------------------------------------------------------
# GUI ↔ 仿真任务暂停 / 继续（所有后端在 step 前轮询）
# ---------------------------------------------------------------------------


def sim_control_path(root: Optional[str] = None) -> str:
    d = (root or bridge_dir() or "").strip()
    return os.path.join(d, SIM_CONTROL_NAME) if d else ""


def read_sim_control(root: Optional[str] = None) -> Optional[dict]:
    path = sim_control_path(root)
    if not path:
        return None
    return _read_json(path)


def write_sim_control(
    *,
    root: Optional[str] = None,
    paused: Optional[bool] = None,
    merge: bool = True,
) -> bool:
    """Write sim pause/control flags. merge=True keeps unspecified fields."""
    d = (root or bridge_dir() or "").strip()
    if not d:
        return False
    path = os.path.join(d, SIM_CONTROL_NAME)
    prev = _read_json(path) if merge else None
    if not isinstance(prev, dict):
        prev = {}
    out: dict = {"ts": time.time()}
    if paused is not None:
        out["paused"] = bool(paused)
    elif merge and "paused" in prev:
        out["paused"] = bool(prev.get("paused"))
    else:
        out["paused"] = False
    try:
        _atomic_json_write(path, out)
    except Exception as exc:
        _log_fail("sim_control", exc)
        return False
    return True


def clear_sim_control(root: Optional[str] = None) -> None:
    path = sim_control_path(root)
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def is_sim_paused(root: Optional[str] = None) -> bool:
    data = read_sim_control(root)
    if not data:
        return False
    return bool(data.get("paused"))


def wait_while_paused(
    root: Optional[str] = None,
    *,
    poll_s: float = 0.05,
    stop_flag: Optional[Any] = None,
) -> bool:
    """Block while ``sim_control.json`` says paused.

    Returns False if ``stop_flag`` became truthy while waiting (caller should exit);
    otherwise True (running / resumed).
    """
    while is_sim_paused(root):
        if stop_flag is not None:
            try:
                if callable(stop_flag) and stop_flag():
                    return False
                if not callable(stop_flag) and stop_flag:
                    return False
            except Exception:
                pass
        time.sleep(max(0.01, float(poll_s)))
    return True
