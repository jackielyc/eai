#!/usr/bin/env python3
"""Docker 内相机 → 宿主机可见帧桥接（写共享目录 npy）。

容器内相机多为 FastDDS「仅 SHM」，宿主机 DDS 收不到大图。
本节点在容器内订阅 /camera/*，写入双方都可访问的目录（默认 /var/psi/data/eai_cam），
宿主机 run_local 通过已有 cam_*.npy 预览链路显示。

文件约定（与仿真 ISAAC_CAM_BRIDGE 一致）:
  cam_head.npy / cam_head.npy.stamp
  cam_left_wrist.npy / cam_right_wrist.npy
  cam_*_depth.npy （深度）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    CvBridge = None  # type: ignore

DEFAULT_OUT_DIR = "/var/psi/data/eai_cam"

# (ros topic, file stem) — stem 与 show_camera_topics.SIM_PREVIEW_* 一致
COLOR_TOPICS: Tuple[Tuple[str, str], ...] = (
    ("/camera/head_color", "cam_head"),
    ("/camera/left_wrist_color", "cam_left_wrist"),
    ("/camera/right_wrist_color", "cam_right_wrist"),
)
DEPTH_TOPICS: Tuple[Tuple[str, str], ...] = (
    ("/camera/head_depth", "cam_head_depth"),
    ("/camera/left_wrist_depth", "cam_left_wrist_depth"),
    ("/camera/right_wrist_depth", "cam_right_wrist_depth"),
)


def _qos_reliable() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
    )


def _qos_sensor() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
    )


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".writing"
    stamp = path + ".stamp"
    np.save(tmp, arr)
    os.replace(tmp, path)
    # 空 stamp 文件：mtime 供宿主机判断新鲜度
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(f"{time.time():.6f}\n")


class CameraHostBridge(Node):
    def __init__(self, out_dir: str, *, max_hz: float = 15.0) -> None:
        super().__init__("camera_host_bridge")
        if CvBridge is None:
            raise RuntimeError("需要 cv_bridge")
        self._bridge = CvBridge()
        self._out_dir = out_dir
        os.makedirs(self._out_dir, exist_ok=True)
        self._min_interval = 1.0 / max(1.0, float(max_hz))
        self._last_write: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._subs = []

        for topic, stem in COLOR_TOPICS + DEPTH_TOPICS:
            self._bind(topic, stem, is_depth=("depth" in stem))

        self.create_timer(5.0, self._log_stats)
        self.get_logger().info(
            f"file bridge → {self._out_dir} (cam_*.npy, max_hz={max_hz})"
        )

    def _bind(self, topic: str, stem: str, *, is_depth: bool) -> None:
        def _cb(msg: Image, _stem=stem, _depth=is_depth) -> None:
            now = time.time()
            last = self._last_write.get(_stem, 0.0)
            if now - last < self._min_interval:
                return
            try:
                if _depth:
                    arr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    arr = np.asarray(arr)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    if arr.dtype != np.uint16:
                        # 米制 float → mm uint16（与仿真桥一致）
                        depth_f = arr.astype(np.float32)
                        finite = depth_f[np.isfinite(depth_f)]
                        if finite.size and float(np.nanmax(finite)) < 100.0:
                            arr = np.clip(depth_f * 1000.0, 0, 65535).astype(np.uint16)
                        else:
                            arr = np.clip(depth_f, 0, 65535).astype(np.uint16)
                else:
                    enc = (msg.encoding or "bgr8").lower()
                    if enc == "rgb8":
                        arr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                        import cv2

                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        arr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    arr = np.ascontiguousarray(arr)
                path = os.path.join(self._out_dir, f"{_stem}.npy")
                _atomic_save_npy(path, arr)
                self._last_write[_stem] = now
                self._counts[_stem] = self._counts.get(_stem, 0) + 1
            except Exception as exc:
                self.get_logger().warn(f"{_stem}: write failed: {exc}", throttle_duration_sec=2.0)

        for qos in (_qos_reliable(), _qos_sensor()):
            try:
                self._subs.append(
                    self.create_subscription(Image, topic, _cb, qos)
                )
            except Exception:
                continue

    def _log_stats(self) -> None:
        if not self._counts:
            self.get_logger().warn(
                "尚无写入 — 确认 /camera/* 有发布（SHM）"
            )
            return
        parts = [f"{k}={v}" for k, v in sorted(self._counts.items())]
        self.get_logger().info("wrote " + " ".join(parts))
        self._counts.clear()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Docker camera → host shared npy")
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("EAI_CAMERA_BRIDGE_DIR", DEFAULT_OUT_DIR),
        help="共享输出目录（宿主机与容器都能访问）",
    )
    parser.add_argument("--max-hz", type=float, default=15.0)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = CameraHostBridge(args.out_dir, max_hz=args.max_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
