#!/usr/bin/python3.10
"""独立进程订阅左右 TCP，写 JSON；避免在 Qt 进程里 create_subscription 占 GIL。"""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time

STATE_PATH = os.environ.get(
    "EAI_ROBOT_TCP_STATE",
    os.path.join(tempfile.gettempdir(), "eai_robot_tcp_state.json"),
)
LEFT_TOPIC = "/mink_fk/left_tcp_pose"
RIGHT_TOPIC = "/mink_fk/right_tcp_pose"


def _pose_to_dict(msg) -> dict:
    p = msg.pose.position
    o = msg.pose.orientation
    return {
        "frame": msg.header.frame_id or "map",
        "xyz": [float(p.x), float(p.y), float(p.z)],
        "xyzw": [float(o.x), float(o.y), float(o.z), float(o.w)],
        "t": time.time(),
    }


def main() -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    stop = {"v": False}

    def _sig(*_a):
        stop["v"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    rclpy.init()
    node = Node("eai_robot_tcp_bridge")
    state = {"left": None, "right": None, "ready": False, "updated_at": 0.0}
    dirty = {"v": True}

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )

    def _on_left(msg: PoseStamped) -> None:
        state["left"] = _pose_to_dict(msg)
        state["updated_at"] = time.time()
        state["ready"] = True
        dirty["v"] = True

    def _on_right(msg: PoseStamped) -> None:
        state["right"] = _pose_to_dict(msg)
        state["updated_at"] = time.time()
        state["ready"] = True
        dirty["v"] = True

    # 两个 create 之间 sleep，减轻本进程内连续订阅压力
    node.create_subscription(PoseStamped, LEFT_TOPIC, _on_left, qos)
    time.sleep(0.15)
    node.create_subscription(PoseStamped, RIGHT_TOPIC, _on_right, qos)
    node.get_logger().info(f"tcp bridge up → {STATE_PATH}")

    last_write = 0.0
    while rclpy.ok() and not stop["v"]:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.time()
        if dirty["v"] and now - last_write >= 0.1:
            dirty["v"] = False
            last_write = now
            tmp = STATE_PATH + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(state, f, separators=(",", ":"))
                os.replace(tmp, STATE_PATH)
            except Exception:
                pass

    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
