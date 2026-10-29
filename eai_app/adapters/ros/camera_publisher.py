#!/usr/bin/env python3
"""宿主机 USB 相机 → ROS /camera/*（不依赖 Docker / a2d_sdk）。

本机已插 RealSense 时，可用 OpenCV 打开 V4L 设备并发布到与真机一致的 topic，
供 run_local / show_camera_topics 勾选预览。

用法（先 source ROS）:
  python3 host_camera_publisher.py --head /dev/video4
  python3 host_camera_publisher.py --head /dev/video4 --left /dev/videoX --right /dev/videoY
  python3 host_camera_publisher.py --list
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _list_devices() -> None:
    print("V4L 探测（OpenCV）:")
    for i in range(10):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        shape = None if frame is None else getattr(frame, "shape", None)
        print(f"  {path}: opened={cap.isOpened()} read={ok} shape={shape}")
        cap.release()


def _open_capture(dev: str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    if dev.startswith("/dev/video"):
        idx = int(dev.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    else:
        try:
            idx = int(dev)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        except ValueError:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {dev}")
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Host USB → /camera/* Image topics")
    parser.add_argument("--list", action="store_true", help="列出可用 V4L 设备")
    parser.add_argument("--head", default="", help="头部彩色设备，如 /dev/video4")
    parser.add_argument("--left", default="", help="左腕彩色 → /camera/hand_left_color")
    parser.add_argument("--right", default="", help="右腕彩色 → /camera/hand_right_color")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--frame-id",
        default="camera_frame",
        help="Image header.frame_id",
    )
    args = parser.parse_args(argv)

    if args.list:
        _list_devices()
        return 0

    mapping: List[Tuple[str, str]] = []
    if args.head:
        mapping.append((args.head, "/camera/head_color"))
    if args.left:
        mapping.append((args.left, "/camera/hand_left_color"))
    if args.right:
        mapping.append((args.right, "/camera/hand_right_color"))
    if not mapping:
        # 默认：本机探测到的第一路彩色 → head
        _list_devices()
        print("未指定 --head/--left/--right，尝试自动选彩色设备…", file=sys.stderr)
        for i in range(10):
            path = f"/dev/video{i}"
            if not os.path.exists(path):
                continue
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if ok and frame is not None and getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
                mapping.append((path, "/camera/head_color"))
                print(f"自动选择 {path} → /camera/head_color")
                break
        if not mapping:
            print("错误: 没有可用彩色 V4L 设备", file=sys.stderr)
            return 1

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )

    class HostCamNode(Node):
        def __init__(self) -> None:
            super().__init__("host_camera_publisher")
            self._bridge = CvBridge()
            self._caps: Dict[str, cv2.VideoCapture] = {}
            self._pubs = {}
            self._period = 1.0 / max(1.0, float(args.fps))
            for dev, topic in mapping:
                self._caps[topic] = _open_capture(
                    dev, args.width, args.height, args.fps
                )
                self._pubs[topic] = self.create_publisher(Image, topic, qos)
                self.get_logger().info(f"{dev} → {topic}")
            self.create_timer(self._period, self._tick)

        def _tick(self) -> None:
            now = self.get_clock().now().to_msg()
            for topic, cap in self._caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                msg = self._bridge.cv2_to_imgmsg(np.ascontiguousarray(frame), encoding="bgr8")
                msg.header.stamp = now
                msg.header.frame_id = args.frame_id
                self._pubs[topic].publish(msg)

        def destroy_node(self) -> bool:
            for cap in self._caps.values():
                try:
                    cap.release()
                except Exception:
                    pass
            return super().destroy_node()

    rclpy.init(args=None)
    node = HostCamNode()
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
