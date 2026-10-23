#!/usr/bin/env python3
"""Docker 内命令中继：收宿主机 UDP，再转到容器内 ROS（手 / 臂 / 使能 / mode）。

根因：FastDDS 仅 SHM 时宿主机→容器发令到不了（能收相机）。
本中继走 127.0.0.1 UDP（容器 network=host）。

协议（每包一行 JSON）:
  手（兼容旧格式，可省略 type）:
    {"type":"hand","side":"left"|"right","name":[...],"position":[...],...}
  臂关节目标:
    {"type":"wbc_joints","position":[...],"name":[...optional...],"frame_id":"..."}
  控制模式:
    {"type":"control_mode","mode":0}
  使能:
    {"type":"arm_enable","data":true}
    {"type":"hand_enable","data":true}
  手臂回初始位:
    {"type":"arm_reset","data":true}

用法（容器内）:
  python3 /path/hand_cmd_relay.py --port 19283
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt32
from std_srvs.srv import SetBool


DEFAULT_PORT = 19283
LEFT_TOPIC = "/ry_hand/left/set_angles"
RIGHT_TOPIC = "/ry_hand/right/set_angles"
WBC_TOPIC = "/wbc/target_joints"
CONTROL_MODE_TOPIC = "/control_mode"
ARM_ENABLE_SRV = "/arm/set_enable"
HAND_ENABLE_SRV = "/hand/set_enable"
ARM_RESET_SRV = "/arm/reset_to_initial"


class RobotCmdRelay(Node):
    def __init__(self, port: int) -> None:
        super().__init__("eai_hand_cmd_relay")
        self._left_pub = self.create_publisher(JointState, LEFT_TOPIC, 10)
        self._right_pub = self.create_publisher(JointState, RIGHT_TOPIC, 10)
        self._wbc_pub = self.create_publisher(JointState, WBC_TOPIC, 10)
        self._mode_pub = self.create_publisher(UInt32, CONTROL_MODE_TOPIC, 10)
        self._arm_enable_cli = self.create_client(SetBool, ARM_ENABLE_SRV)
        self._hand_enable_cli = self.create_client(SetBool, HAND_ENABLE_SRV)
        self._arm_reset_cli = self.create_client(SetBool, ARM_RESET_SRV)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.05)
        self._port = port
        self._ok = 0
        self._err = 0
        self.create_timer(0.01, self._poll)
        self.get_logger().info(
            f"robot_cmd_relay udp://127.0.0.1:{port} -> "
            f"hand/wbc/control_mode/arm_enable/hand_enable/arm_reset"
        )

    def _poll(self) -> None:
        for _ in range(64):
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                break
            except OSError as exc:
                self._err += 1
                if self._err % 50 == 1:
                    self.get_logger().warn(f"recv error: {exc}")
                break
            self._handle(data)

    def _handle(self, data: bytes) -> None:
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
                    self._on_hand(payload)
                elif ptype in ("wbc_joints", "wbc", "arm_joints"):
                    self._on_wbc(payload)
                elif ptype in ("control_mode", "mode"):
                    self._on_mode(payload)
                elif ptype in ("arm_enable", "enable_arm"):
                    self._on_enable(self._arm_enable_cli, ARM_ENABLE_SRV, payload)
                elif ptype in ("hand_enable", "enable_hand"):
                    self._on_enable(self._hand_enable_cli, HAND_ENABLE_SRV, payload)
                elif ptype in ("arm_reset", "reset_arm", "reset_to_initial"):
                    self._on_enable(self._arm_reset_cli, ARM_RESET_SRV, payload)
                else:
                    self.get_logger().warn(f"unknown type={ptype}")
        except Exception as exc:
            self._err += 1
            if self._err <= 5 or self._err % 20 == 0:
                self.get_logger().warn(f"bad packet: {exc}")

    def _on_hand(self, payload: dict) -> None:
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
        else:
            self._left_pub.publish(msg)
        self._ok += 1
        if self._ok <= 3 or self._ok % 50 == 0:
            pos0 = msg.position[0] if msg.position else float("nan")
            self.get_logger().info(f"relay hand #{self._ok} {side} pos0={pos0:.3f}")

    def _on_wbc(self, payload: dict) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(payload.get("frame_id") or "base_link")
        msg.name = [str(x) for x in (payload.get("name") or [])]
        msg.position = [float(x) for x in (payload.get("position") or [])]
        msg.velocity = [float(x) for x in (payload.get("velocity") or [])]
        msg.effort = [float(x) for x in (payload.get("effort") or [])]
        self._wbc_pub.publish(msg)
        self._ok += 1
        if self._ok <= 5 or self._ok % 100 == 0:
            n = len(msg.position)
            self.get_logger().info(f"relay wbc #{self._ok} joints={n}")

    def _on_mode(self, payload: dict) -> None:
        msg = UInt32()
        msg.data = int(payload.get("mode", 0))
        self._mode_pub.publish(msg)
        self.get_logger().info(f"relay control_mode={msg.data}")

    def _on_enable(self, client, name: str, payload: dict) -> None:
        data = bool(payload.get("data", True))
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(f"service {name} not ready")
                return
        req = SetBool.Request()
        req.data = data
        fut = client.call_async(req)

        def _done(f) -> None:
            try:
                res = f.result()
                ok = bool(res and res.success)
                self.get_logger().info(
                    f"relay {name}({data}) -> {ok} {getattr(res, 'message', '')}"
                )
            except Exception as exc:
                self.get_logger().warn(f"relay {name} failed: {exc}")

        fut.add_done_callback(_done)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    rclpy.init()
    node = RobotCmdRelay(args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
