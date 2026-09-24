#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac 共享帧目录 → ROS2 Image 发布桥（Python 3.10 + Humble）。

读取 ISAAC_CAM_BRIDGE_DIR 下的 {cam_key}.npy / {cam_key}_depth.npy，
发布到 /camera/*_color 与 /camera/*_depth。

默认映射:
  cam_head / cam_high     → /camera/head_color + /camera/head_depth
  cam_left_wrist          → /camera/left_wrist_color + /camera/left_wrist_depth
  cam_right_wrist         → /camera/right_wrist_color + /camera/right_wrist_depth

深度编码: 16UC1（毫米），与真机 D405 约定一致。

启动示例::

    export ISAAC_CAM_BRIDGE_DIR=/tmp/isaac_cam_bridge
    bash run_isaac_cam_bridge.sh

    # 另开终端启动 Isaac 评测时同样 export ISAAC_CAM_BRIDGE_DIR
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".cache", "isaac_cam_bridge"
)
DEFAULT_HZ = 30.0

# cam_key → (color_topic, depth_topic, frame_id)
CAMERA_MAP: Dict[str, Tuple[str, str, str]] = {
    "cam_head": ("/camera/head_color", "/camera/head_depth", "camera_frame"),
    "cam_high": ("/camera/head_color", "/camera/head_depth", "camera_frame"),
    "head_camera": ("/camera/head_color", "/camera/head_depth", "camera_frame"),
    "cam_left_wrist": (
        "/camera/left_wrist_color",
        "/camera/left_wrist_depth",
        "left_wrist_camera_frame",
    ),
    "cam_right_wrist": (
        "/camera/right_wrist_color",
        "/camera/right_wrist_depth",
        "right_wrist_camera_frame",
    ),
}


def _load_rgb(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            arr = np.load(f)
        arr = np.asarray(arr)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return np.ascontiguousarray(arr[..., :3], dtype=np.uint8)


def _load_depth_u16(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            arr = np.load(f)
        arr = np.asarray(arr)
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
    if finite.size and float(np.nanmax(finite)) < 100.0:
        out = np.clip(depth_f * 1000.0, 0, 65535).astype(np.uint16)
    else:
        out = np.clip(depth_f, 0, 65535).astype(np.uint16)
    return np.ascontiguousarray(out)


def _read_stamp(path: str) -> str:
    stamp_path = path + ".stamp"
    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        try:
            return str(os.path.getmtime(path))
        except Exception:
            return ""


def _numpy_to_imgmsg(
    arr: np.ndarray,
    *,
    encoding: str,
    frame_id: str,
    stamp,
) -> "Image":
    """Build sensor_msgs/Image without cv_bridge / OpenCV (RoboStack ABI drift)."""
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    cont = np.ascontiguousarray(arr)
    if encoding == "bgr8":
        if cont.ndim != 3 or cont.shape[2] < 3:
            raise ValueError(f"bgr8 expects HxWx3, got {cont.shape}")
        # Input is RGB; sensor_msgs bgr8 wants BGR channel order.
        payload = cont[..., :3][..., ::-1]
        msg.step = msg.width * 3
        msg.data = payload.tobytes()
    elif encoding == "rgb8":
        if cont.ndim != 3 or cont.shape[2] < 3:
            raise ValueError(f"rgb8 expects HxWx3, got {cont.shape}")
        msg.step = msg.width * 3
        msg.data = cont[..., :3].tobytes()
    elif encoding == "16UC1":
        if cont.ndim != 2:
            raise ValueError(f"16UC1 expects HxW, got {cont.shape}")
        u16 = cont.astype(np.uint16, copy=False)
        msg.step = msg.width * 2
        msg.data = np.ascontiguousarray(u16).tobytes()
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    return msg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Isaac cam → ROS2 Image bridge")
    parser.add_argument(
        "--dir",
        default=os.environ.get("ISAAC_CAM_BRIDGE_DIR", DEFAULT_DIR),
        help="共享帧目录",
    )
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="轮询频率")
    parser.add_argument(
        "--qos-best-effort",
        action="store_true",
        help="用 BEST_EFFORT QoS（与多数相机驱动一致）",
    )
    args = parser.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    class IsaacCamBridge(Node):
        def __init__(self) -> None:
            super().__init__("isaac_cam_ros_bridge")
            self._dir = os.path.abspath(args.dir)
            self._last_color_stamp: Dict[str, str] = {}
            self._last_depth_stamp: Dict[str, str] = {}
            self._pubs: Dict[str, object] = {}
            self._qos = QoSProfile(
                reliability=(
                    ReliabilityPolicy.BEST_EFFORT
                    if args.qos_best_effort
                    else ReliabilityPolicy.RELIABLE
                ),
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            for _cam_key, (color_topic, depth_topic, _fid) in CAMERA_MAP.items():
                if color_topic not in self._pubs:
                    self._pubs[color_topic] = self.create_publisher(
                        Image, color_topic, self._qos
                    )
                if depth_topic not in self._pubs:
                    self._pubs[depth_topic] = self.create_publisher(
                        Image, depth_topic, self._qos
                    )
            os.makedirs(self._dir, exist_ok=True)
            period = 1.0 / max(1.0, float(args.hz))
            self.create_timer(period, self._on_timer)
            self.get_logger().info(
                f"watching {self._dir} → color+depth "
                f"{sorted(self._pubs.keys())} (numpy Image encode, no cv_bridge)"
            )

        def _ensure_pub(self, topic: str) -> object:
            pub = self._pubs.get(topic)
            if pub is not None:
                return pub
            pub = self.create_publisher(Image, topic, self._qos)
            self._pubs[topic] = pub
            return pub

        def _resolve_mapping(self, cam_key: str) -> Tuple[str, str, str]:
            mapping = CAMERA_MAP.get(cam_key)
            if mapping is not None:
                return mapping
            return (
                f"/camera/{cam_key}_color",
                f"/camera/{cam_key}_depth",
                cam_key,
            )

        def _cam_keys(self) -> list:
            order = [
                "cam_head",
                "cam_high",
                "head_camera",
                "cam_left_wrist",
                "cam_right_wrist",
            ]
            seen = set(order)
            if not os.path.isdir(self._dir):
                return order
            for name in os.listdir(self._dir):
                if name.endswith("_depth.npy"):
                    key = name[: -len("_depth.npy")]
                elif name.endswith(".npy"):
                    key = name[: -len(".npy")]
                else:
                    continue
                if key.endswith("_depth"):
                    continue
                if key not in seen:
                    order.append(key)
                    seen.add(key)
            return order

        def _publish_color(self, cam_key: str, color_topic: str, frame_id: str) -> None:
            path = os.path.join(self._dir, f"{cam_key}.npy")
            if not os.path.isfile(path):
                return
            stamp_val = _read_stamp(path)
            if stamp_val and self._last_color_stamp.get(cam_key) == stamp_val:
                return
            rgb = _load_rgb(path)
            if rgb is None:
                return
            try:
                msg = _numpy_to_imgmsg(
                    rgb,
                    encoding="bgr8",
                    frame_id=frame_id,
                    stamp=self.get_clock().now().to_msg(),
                )
            except Exception as exc:
                self.get_logger().warning(f"encode color {cam_key} failed: {exc}")
                return
            self._ensure_pub(color_topic).publish(msg)
            self._last_color_stamp[cam_key] = stamp_val

        def _publish_depth(self, cam_key: str, depth_topic: str, frame_id: str) -> None:
            path = os.path.join(self._dir, f"{cam_key}_depth.npy")
            if not os.path.isfile(path):
                return
            stamp_val = _read_stamp(path)
            if stamp_val and self._last_depth_stamp.get(cam_key) == stamp_val:
                return
            depth_u16 = _load_depth_u16(path)
            if depth_u16 is None:
                return
            try:
                msg = _numpy_to_imgmsg(
                    depth_u16,
                    encoding="16UC1",
                    frame_id=frame_id,
                    stamp=self.get_clock().now().to_msg(),
                )
            except Exception as exc:
                self.get_logger().warning(f"encode depth {cam_key} failed: {exc}")
                return
            self._ensure_pub(depth_topic).publish(msg)
            self._last_depth_stamp[cam_key] = stamp_val

        def _on_timer(self) -> None:
            if not os.path.isdir(self._dir):
                return
            published_color = set()
            published_depth = set()
            for cam_key in self._cam_keys():
                color_topic, depth_topic, frame_id = self._resolve_mapping(cam_key)
                if color_topic not in published_color:
                    self._publish_color(cam_key, color_topic, frame_id)
                    published_color.add(color_topic)
                if depth_topic not in published_depth:
                    self._publish_depth(cam_key, depth_topic, frame_id)
                    published_depth.add(depth_topic)

    rclpy.init(args=None)
    node = IsaacCamBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
