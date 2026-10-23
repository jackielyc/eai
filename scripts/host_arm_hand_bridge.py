#!/usr/bin/python3.10
"""宿主机手/臂使能桥（无 Docker）。

替代容器内 topic_router 的 enable API + hand_cmd_relay：
  - ROS: /arm/set_enable、/hand/set_enable → 发布 /arm|/hand/enable_state
  - UDP 127.0.0.1:19283：兼容 UI 的 arm_enable / hand_enable / control_mode / hand / wbc

用法:
  python3.10 scripts/host_arm_hand_bridge.py [--port 19283]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt32
from std_srvs.srv import SetBool

DEFAULT_PORT = 19283
LEFT_HAND = "/ry_hand/left/set_angles"
RIGHT_HAND = "/ry_hand/right/set_angles"
LEFT_EE = "/wbc/left_ee_command"
RIGHT_EE = "/wbc/right_ee_command"
WBC_TOPIC = "/wbc/target_joints"
# HAL arm_command_node 订此话题；无 arm_control/ik_node 时由本桥转发 14 臂关节
ARM_COMMAND_TOPIC = "/wbc/arm_command"
CONTROL_MODE = "/control_mode"
ARM_ENABLE_STATE = "/arm/enable_state"
HAND_ENABLE_STATE = "/hand/enable_state"
ARM_SET_ENABLE = "/arm/set_enable"
HAND_SET_ENABLE = "/hand/set_enable"


class HostArmHandBridge(Node):
    def __init__(self, port: int) -> None:
        super().__init__("eai_host_arm_hand_bridge")
        self._cg = ReentrantCallbackGroup()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._arm_enabled = False
        self._hand_enabled = False
        self._last_drop_wbc_log = 0.0
        self._arm_pub = self.create_publisher(UInt32, ARM_ENABLE_STATE, qos)
        self._hand_pub = self.create_publisher(UInt32, HAND_ENABLE_STATE, qos)
        self._mode_pub = self.create_publisher(UInt32, CONTROL_MODE, 10)
        self._left_pub = self.create_publisher(JointState, LEFT_HAND, 10)
        self._right_pub = self.create_publisher(JointState, RIGHT_HAND, 10)
        self._left_ee_pub = self.create_publisher(JointState, LEFT_EE, 10)
        self._right_ee_pub = self.create_publisher(JointState, RIGHT_EE, 10)
        self._wbc_pub = self.create_publisher(JointState, WBC_TOPIC, 10)
        self._arm_cmd_pub = self.create_publisher(JointState, ARM_COMMAND_TOPIC, 10)
        self.create_service(
            SetBool, ARM_SET_ENABLE, self._on_arm_srv, callback_group=self._cg
        )
        self.create_service(
            SetBool, HAND_SET_ENABLE, self._on_hand_srv, callback_group=self._cg
        )
        self.create_timer(0.05, self._publish_states)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.05)
        self.create_timer(0.01, self._poll_udp)
        self._publish_states()
        self.get_logger().info(
            f"host arm/hand bridge up: ROS set_enable + udp://127.0.0.1:{port}"
        )

    def _publish_states(self) -> None:
        a = UInt32()
        a.data = 1 if self._arm_enabled else 0
        h = UInt32()
        h.data = 1 if self._hand_enabled else 0
        self._arm_pub.publish(a)
        self._hand_pub.publish(h)

    def _on_arm_srv(
        self, req: SetBool.Request, res: SetBool.Response
    ) -> SetBool.Response:
        self._set_arm_enabled(bool(req.data), src="srv")
        res.success = True
        res.message = f"arm enable set to {1 if self._arm_enabled else 0}"
        return res

    def _on_hand_srv(
        self, req: SetBool.Request, res: SetBool.Response
    ) -> SetBool.Response:
        self._hand_enabled = bool(req.data)
        self._publish_states()
        res.success = True
        res.message = f"hand enable set to {1 if self._hand_enabled else 0}"
        self.get_logger().info(res.message)
        return res

    def _poll_udp(self) -> None:
        for _ in range(64):
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                break
            except OSError as exc:
                self.get_logger().warn(f"udp recv: {exc}")
                break
            self._handle_udp(data)

    def _handle_udp(self, data: bytes) -> None:
        try:
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                return
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                ptype = str(payload.get("type") or "hand").lower()
                if ptype in ("hand", ""):
                    self._udp_hand(payload)
                elif ptype in ("wbc_joints", "wbc", "arm_joints"):
                    self._udp_wbc(payload)
                elif ptype in ("control_mode", "mode"):
                    msg = UInt32()
                    msg.data = int(payload.get("mode", 0))
                    self._mode_pub.publish(msg)
                elif ptype in ("arm_enable", "enable_arm"):
                    self._set_arm_enabled(bool(payload.get("data", True)), src="udp")
                elif ptype in ("hand_enable", "enable_hand"):
                    self._hand_enabled = bool(payload.get("data", True))
                    self._publish_states()
                    self.get_logger().info(
                        f"udp hand_enable -> {1 if self._hand_enabled else 0}"
                    )
                elif ptype in ("arm_reset", "reset_arm", "reset_to_initial"):
                    if not self._arm_enabled:
                        self.get_logger().warn("arm_reset ignored: arm disabled")
                        return
                    self.get_logger().warn(
                        "arm_reset: 无 Docker 时本桥不调用 /arm/reset_to_initial"
                    )
        except Exception as exc:
            self.get_logger().warn(f"bad udp: {exc}")

    def _set_arm_enabled(self, enabled: bool, *, src: str) -> None:
        prev = self._arm_enabled
        self._arm_enabled = bool(enabled)
        self._publish_states()
        if prev != self._arm_enabled:
            self.get_logger().info(
                f"{src} arm_enable -> {1 if self._arm_enabled else 0} "
                f"({'接受臂控' if self._arm_enabled else '拒绝臂控指令'})"
            )

    def _udp_hand(self, payload: dict) -> None:
        # 手部使能关闭时：set_angles 与 ee 一律不转发
        if not self._hand_enabled:
            import time as _time

            now = _time.time()
            last = float(getattr(self, "_last_drop_hand_log", 0.0) or 0.0)
            if now - last > 2.0:
                self._last_drop_hand_log = now
                self.get_logger().warn(
                    "drop hand cmd: hand disabled（点「启用手」后再控手）"
                )
            return
        side = str(payload.get("side", "left")).lower()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(payload.get("frame_id") or f"{side}_hand")
        msg.name = [str(x) for x in (payload.get("name") or [])]
        msg.position = [float(x) for x in (payload.get("position") or [])]
        msg.velocity = [float(x) for x in (payload.get("velocity") or [])]
        msg.effort = [float(x) for x in (payload.get("effort") or [])]
        if side == "right":
            self._right_pub.publish(msg)
            self._right_ee_pub.publish(msg)
        else:
            self._left_pub.publish(msg)
            self._left_ee_pub.publish(msg)

    def _udp_wbc(self, payload: dict) -> None:
        if not self._arm_enabled:
            import time as _time

            now = _time.time()
            if now - self._last_drop_wbc_log > 2.0:
                self._last_drop_wbc_log = now
                self.get_logger().warn(
                    "drop wbc_joints: arm disabled（点「启用手臂」后再控臂）"
                )
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(payload.get("frame_id") or "base_link")
        msg.name = [str(x) for x in (payload.get("name") or [])]
        msg.position = [float(x) for x in (payload.get("position") or [])]
        msg.velocity = [float(x) for x in (payload.get("velocity") or [])]
        msg.effort = [float(x) for x in (payload.get("effort") or [])]
        self._wbc_pub.publish(msg)
        # 16 = 腰2+臂14；HAL 要 14 臂关节。无 name 时按 position 截取。
        arm_msg = JointState()
        arm_msg.header = msg.header
        if len(msg.position) >= 16:
            arm_msg.position = [float(x) for x in msg.position[2:16]]
        elif len(msg.position) >= 14:
            arm_msg.position = [float(x) for x in msg.position[:14]]
        else:
            return
        if msg.name and len(msg.name) >= 16:
            arm_msg.name = [str(x) for x in msg.name[2:16]]
        self._arm_cmd_pub.publish(arm_msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    rclpy.init()
    node = HostArmHandBridge(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
