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
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

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
