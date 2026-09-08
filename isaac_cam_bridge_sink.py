#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac / RoboDojo → 共享目录帧写出（无 ROS 依赖，可在 Python 3.11 仿真进程内调用）。

环境变量:
  ISAAC_CAM_BRIDGE_DIR   共享目录，例如 /tmp/isaac_cam_bridge
  ISAAC_CAM_BRIDGE_EVERY 每隔 N 帧写一次（默认 1）

用法（在 RoboDojo eval_env._stream_vision 中）::

    from isaac_cam_bridge_sink import write_cam_frame
    write_cam_frame(cam_key, color_rgb)

帧文件: {DIR}/{cam_key}.npy  （HxWx3 uint8 RGB）
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

_counters: Dict[str, int] = {}
_last_log_s = 0.0


def bridge_dir() -> str:
    return os.environ.get("ISAAC_CAM_BRIDGE_DIR", "").strip()


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

    if every is None:
        try:
            every = max(1, int(os.environ.get("ISAAC_CAM_BRIDGE_EVERY", "1")))
        except ValueError:
            every = 1
    n = _counters.get(key, 0) + 1
    _counters[key] = n
    if (n - 1) % every != 0:
        return False

    rgb = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    try:
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"{key}.npy")
        tmp = path + ".writing"
        with open(tmp, "wb") as f:
            np.save(f, rgb)
        os.replace(tmp, path)
        # 旁路时间戳，供 ROS 端判断是否有新帧
        with open(path + ".stamp", "w", encoding="utf-8") as f:
            f.write(f"{time.time():.6f}\n")
    except Exception as exc:
        global _last_log_s
        now = time.time()
        if now - _last_log_s > 5.0:
            _last_log_s = now
            print(f"[isaac_cam_bridge_sink] write failed ({key}): {exc}", flush=True)
        return False
    return True
