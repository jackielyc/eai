#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac 共享帧目录 → ROS2 Image 发布桥（Python 3.10 + Humble）。

读取 ISAAC_CAM_BRIDGE_DIR 下的 {cam_key}.npy，发布到 /camera/*_color。

默认映射:
  cam_head / cam_high     → /camera/head_color      (frame_id=camera_frame)
  cam_left_wrist          → /camera/left_wrist_color
  cam_right_wrist         → /camera/right_wrist_color

启动示例::

    export ISAAC_CAM_BRIDGE_DIR=/tmp/isaac_cam_bridge
    bash run_isaac_cam_bridge.sh

    # 另开终端启动 Isaac 评测时同样 export ISAAC_CAM_BRIDGE_DIR
    # 然后用 eai viewer 勾选 /camera/head_color
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

# cam_key → (topic, frame_id)
CAMERA_MAP: Dict[str, Tuple[str, str]] = {
    "cam_head": ("/camera/head_color", "camera_frame"),
    "cam_high": ("/camera/head_color", "camera_frame"),
    "head_camera": ("/camera/head_color", "camera_frame"),
    "cam_left_wrist": ("/camera/left_wrist_color", "left_wrist_camera_frame"),
    "cam_right_wrist": ("/camera/right_wrist_color", "right_wrist_camera_frame"),
}


def _load_rgb(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            arr = np.load(f)
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        return np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    except Exception:
        return None


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
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    class IsaacCamBridge(Node):
        def __init__(self) -> None:
            super().__init__("isaac_cam_ros_bridge")
            self._dir = os.path.abspath(args.dir)
            self._bridge = CvBridge()
            self._last_stamp: Dict[str, str] = {}
            self._pubs: Dict[str, object] = {}
            qos = QoSProfile(
                reliability=(
                    ReliabilityPolicy.BEST_EFFORT
                    if args.qos_best_effort
                    else ReliabilityPolicy.RELIABLE
                ),
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            # 按 topic 去重创建 publisher（cam_head/cam_high 共用）
            for cam_key, (topic, _fid) in CAMERA_MAP.items():
                if topic not in self._pubs:
                    self._pubs[topic] = self.create_publisher(Image, topic, qos)
            os.makedirs(self._dir, exist_ok=True)
            period = 1.0 / max(1.0, float(args.hz))
            self.create_timer(period, self._on_timer)
            self.get_logger().info(
                f"watching {self._dir} → {sorted(self._pubs.keys())}"
            )

        def _on_timer(self) -> None:
            if not os.path.isdir(self._dir):
                return
            # 每个 topic 只发一次（优先 cam_head）
            published_topics = set()
            order = [
                "cam_head",
                "cam_high",
                "head_camera",
                "cam_left_wrist",
                "cam_right_wrist",
            ]
            seen = set(order)
            for name in os.listdir(self._dir):
                if name.endswith(".npy"):
                    key = name[: -len(".npy")]
                    if key not in seen:
                        order.append(key)
                        seen.add(key)

            for cam_key in order:
                mapping = CAMERA_MAP.get(cam_key)
                if mapping is None:
                    # 未知相机：发到 /camera/<key>_color
                    topic = f"/camera/{cam_key}_color"
                    frame_id = cam_key
                else:
                    topic, frame_id = mapping
                if topic in published_topics:
                    continue

                path = os.path.join(self._dir, f"{cam_key}.npy")
                if not os.path.isfile(path):
                    continue
                stamp_path = path + ".stamp"
                stamp_val = ""
                try:
                    with open(stamp_path, "r", encoding="utf-8") as f:
                        stamp_val = f.read().strip()
                except Exception:
                    try:
                        stamp_val = str(os.path.getmtime(path))
                    except Exception:
                        stamp_val = ""
                if stamp_val and self._last_stamp.get(cam_key) == stamp_val:
                    continue

                rgb = _load_rgb(path)
                if rgb is None:
                    continue
                try:
                    # cv_bridge 要 BGR
                    import cv2

                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
                except Exception as exc:
                    self.get_logger().warning(f"encode {cam_key} failed: {exc}")
                    continue
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = frame_id
                pub = self._pubs.get(topic)
                if pub is None:
                    pub = self.create_publisher(
                        Image,
                        topic,
                        QoSProfile(
                            reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST,
                            depth=1,
                        ),
                    )
                    self._pubs[topic] = pub
                pub.publish(msg)
                self._last_stamp[cam_key] = stamp_val
                published_topics.add(topic)

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
