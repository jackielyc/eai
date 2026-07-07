#!/usr/bin/env python3
"""
订阅 /camera 彩色与深度话题，统计实际帧率并采样中心像素深度。

用法（在 ROS Docker 容器内）:
  source /opt/ros/humble/setup.bash
  python3 depth_probe.py
  python3 depth_probe.py --duration 10 --pairs hand_left,hand_right,head
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

PAIRS = {
    "hand_left": ("/camera/hand_left_color", "/camera/hand_left_depth"),
    "hand_right": ("/camera/hand_right_color", "/camera/hand_right_depth"),
    "head": ("/camera/head_color", "/camera/head_depth"),
}


def depth_value_meters(depth: np.ndarray, u: int, v: int) -> float:
    value = float(depth[v, u])
    if depth.dtype == np.uint16:
        return value / 1000.0
    return value


class DepthProbe(Node):
    def __init__(self, pairs: List[Tuple[str, str, str]]) -> None:
        super().__init__("depth_probe")
        self._counts: Dict[str, int] = {}
        self._meta: Dict[str, Tuple[int, int, str]] = {}
        self._latest: Dict[str, np.ndarray] = {}
        self._intrinsics: Dict[str, Tuple[float, float, float, float]] = {}
        self._pairs = pairs

        for label, color_topic, depth_topic in pairs:
            self._counts[color_topic] = 0
            self._counts[depth_topic] = 0

            def make_cb(topic: str):
                def cb(msg: Image) -> None:
                    self._counts[topic] += 1
                    if topic not in self._meta:
                        self._meta[topic] = (
                            int(msg.width),
                            int(msg.height),
                            str(msg.encoding),
                        )
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    enc = (msg.encoding or "").lower()
                    if "16" in enc:
                        arr = arr.view(np.uint16).reshape(msg.height, msg.width)
                    elif enc in ("32fc1",):
                        arr = arr.view(np.float32).reshape(msg.height, msg.width)
                    else:
                        return
                    self._latest[topic] = arr.copy()

                return cb

            self.create_subscription(Image, color_topic, make_cb(color_topic), 10)
            self.create_subscription(Image, depth_topic, make_cb(depth_topic), 10)
            for candidate in (
                f"{depth_topic}/camera_info",
                color_topic.replace("_color", "_depth") + "/camera_info",
                color_topic.replace("_color", "_color") + "/camera_info",
            ):
                all_topics = dict(self.get_topic_names_and_types())
                if candidate not in all_topics:
                    continue

                def info_cb(msg: CameraInfo, key: str = depth_topic) -> None:
                    k = msg.k
                    self._intrinsics[key] = (
                        float(k[0]),
                        float(k[4]),
                        float(k[2]),
                        float(k[5]),
                    )

                self.create_subscription(CameraInfo, candidate, info_cb, 10)
                break

    def report(self, duration_s: float) -> None:
        print(f"\n=== 深度探测 ({duration_s:.1f}s) ===")
        for label, color_topic, depth_topic in self._pairs:
            color_n = self._counts.get(color_topic, 0)
            depth_n = self._counts.get(depth_topic, 0)
            color_hz = color_n / max(duration_s, 1e-6)
            depth_hz = depth_n / max(duration_s, 1e-6)
            color_meta = self._meta.get(color_topic)
            depth_meta = self._meta.get(depth_topic)

            print(f"\n[{label}]")
            print(f"  color: {color_topic}")
            print(f"    frames={color_n}  hz={color_hz:.1f}  meta={color_meta}")
            print(f"  depth: {depth_topic}")
            print(f"    frames={depth_n}  hz={depth_hz:.1f}  meta={depth_meta}")

            if depth_n == 0:
                print("    ⚠ 有 publisher 但无实际帧 → 检查机器人端 hand_*_depth_node / D405 驱动")
                continue

            depth = self._latest.get(depth_topic)
            if depth is None:
                continue
            h, w = depth.shape[:2]
            u, v = w // 2, h // 2
            d_m = depth_value_meters(depth, u, v)
            print(f"    center ({u},{v}) depth={d_m:.3f} m")
            intr = self._intrinsics.get(depth_topic)
            if intr:
                fx, fy, cx, cy = intr
                x = (u - cx) * d_m / fx
                y = (v - cy) * d_m / fy
                print(f"    center XYZ=({x:.3f}, {y:.3f}, {d_m:.3f}) m  (from camera_info)")
            else:
                print("    camera_info: 未收到，无法计算 XYZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探测 /camera 深度话题是否有实际数据")
    parser.add_argument(
        "--pairs",
        default="hand_left,hand_right,head",
        help="逗号分隔: hand_left,hand_right,head",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="采样时长（秒）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = []
    for name in args.pairs.split(","):
        name = name.strip()
        if name not in PAIRS:
            print(f"未知 pair: {name}，可选: {', '.join(PAIRS)}", file=sys.stderr)
            return 2
        color_topic, depth_topic = PAIRS[name]
        selected.append((name, color_topic, depth_topic))

    rclpy.init()
    node = DepthProbe(selected)
    t0 = time.time()
    while time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.report(args.duration)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
