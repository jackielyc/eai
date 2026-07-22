#!/usr/bin/python3.10
"""Publish A2D head-camera TF so camera_frame can reach base_link.

Robot image topics use header.frame_id=\"camera_frame\", but the running stack
often only publishes map/fk TCP frames. This node computes
base_link -> head_camera_depth_link from URDF + waist/neck joints, then aliases
camera_frame to that link.

Usage (inside a2d-tele container / same ROS domain as the robot):
  python3.10 a2d_head_camera_tf.py
  # or imported by show_camera_topics.CameraTopicNode
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_URDF_CANDIDATES = (
    "/opt/psi/rt/a2d-tele/install/a2d_description/share/a2d_description/urdf/A2D_RuiYan_D405.urdf",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "third_party",
        "A2D_RuiYan_D405.urdf",
    ),
    os.path.expanduser(
        "~/a2d-tele/install/a2d_description/share/a2d_description/urdf/A2D_RuiYan_D405.urdf"
    ),
)

# HAL / FK names -> URDF joint names
JOINT_NAME_MAP = {
    "joint_lift_body": "joint_waist_link",
    "joint_body_pitch": "joint_body_link",
    "joint_head_yaw": "joint_head_link_1",
    "joint_head_pitch": "joint_head_link_2",
}

# Chain used for head depth camera (URDF joint names, root -> leaf)
HEAD_DEPTH_CHAIN = (
    "joint_waist_link",
    "joint_body_link",
    "joint_head_link_1",
    "joint_head_link_2",
    "joint_head_camera_depth_link",
)

CAMERA_FRAME_ID = "camera_frame"
CAMERA_LINK_ID = "head_camera_depth_link"
BASE_FRAME_ID = "base_link"


def _parse_floats(text: Optional[str], n: int = 3) -> np.ndarray:
    if not text:
        return np.zeros(n, dtype=np.float64)
    vals = [float(x) for x in text.replace(",", " ").split()]
    while len(vals) < n:
        vals.append(0.0)
    return np.asarray(vals[:n], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def make_transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray

    def transform(self, q: float = 0.0) -> np.ndarray:
        T_origin = make_transform(self.origin_xyz, rpy_to_matrix(self.origin_rpy))
        if self.joint_type == "fixed":
            return T_origin
        if self.joint_type == "prismatic":
            axis = self.axis / (np.linalg.norm(self.axis) + 1e-12)
            T_joint = make_transform(axis * float(q), np.eye(3))
            return T_origin @ T_joint
        # revolute / continuous
        T_joint = make_transform(np.zeros(3), axis_angle_to_matrix(self.axis, float(q)))
        return T_origin @ T_joint


def load_urdf_joints(urdf_path: str) -> Dict[str, UrdfJoint]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, UrdfJoint] = {}
    for j in root.findall("joint"):
        name = j.get("name") or ""
        parent_el = j.find("parent")
        child_el = j.find("child")
        if not name or parent_el is None or child_el is None:
            continue
        origin = j.find("origin")
        axis = j.find("axis")
        joints[name] = UrdfJoint(
            name=name,
            joint_type=(j.get("type") or "fixed").lower(),
            parent=parent_el.get("link") or "",
            child=child_el.get("link") or "",
            origin_xyz=_parse_floats(origin.get("xyz") if origin is not None else None),
            origin_rpy=_parse_floats(origin.get("rpy") if origin is not None else None),
            axis=_parse_floats(axis.get("xyz") if axis is not None else "0 0 1"),
        )
    return joints


def a2d_ruiyan_d405_fallback_joints() -> Dict[str, UrdfJoint]:
    """Hardcoded A2D_RuiYan_D405 head-depth chain when URDF file is missing."""
    return {
        "joint_waist_link": UrdfJoint(
            "joint_waist_link",
            "prismatic",
            "base_link",
            "waist_link",
            np.array([0.0, 0.0, 0.6485]),
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
        ),
        "joint_body_link": UrdfJoint(
            "joint_body_link",
            "revolute",
            "waist_link",
            "body_link",
            np.array([0.131, 0.0, 0.0]),
            np.array([-1.5707982, 1.5707928, 0.0000018]),
            np.array([0.0, 0.0, 1.0]),
        ),
        "joint_head_link_1": UrdfJoint(
            "joint_head_link_1",
            "revolute",
            "body_link",
            "head_link_1",
            np.array([-0.441, 0.0, 0.0]),
            np.array([-1.5707928, -0.0000002, 1.5707927]),
            np.array([0.0, 0.0, 1.0]),
        ),
        "joint_head_link_2": UrdfJoint(
            "joint_head_link_2",
            "revolute",
            "head_link_1",
            "head_link_2",
            np.array([-0.050238, 0.0, 0.060065]),
            np.array([1.5708, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        "joint_head_camera_depth_link": UrdfJoint(
            "joint_head_camera_depth_link",
            "fixed",
            "head_link_2",
            "head_camera_depth_link",
            np.array([-0.10207, 0.04137, 0.04856]),
            np.array([1.5707964, -0.0000001, 3.1415925]),
            np.array([0.0, 0.0, 1.0]),
        ),
    }


def find_default_urdf() -> Optional[str]:
    env = os.environ.get("A2D_URDF", "").strip()
    if env and os.path.isfile(env):
        return env
    for path in DEFAULT_URDF_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def matrix_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12
    return qx / n, qy / n, qz / n, qw / n


class HeadCameraFk:
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        chain: Tuple[str, ...] = HEAD_DEPTH_CHAIN,
    ) -> None:
        if urdf_path:
            self.joints = load_urdf_joints(urdf_path)
            self.source = urdf_path
        else:
            self.joints = a2d_ruiyan_d405_fallback_joints()
            self.source = "builtin:A2D_RuiYan_D405_fallback"
        missing = [n for n in chain if n not in self.joints]
        if missing:
            raise RuntimeError(f"URDF 缺少关节: {missing} ({self.source})")
        self.chain = chain
        self.q: Dict[str, float] = {
            n: 0.0 for n in chain if self.joints[n].joint_type != "fixed"
        }

    def set_positions(self, name_to_q: Dict[str, float]) -> None:
        for raw_name, value in name_to_q.items():
            urdf_name = JOINT_NAME_MAP.get(raw_name, raw_name)
            if urdf_name in self.q:
                self.q[urdf_name] = float(value)

    def base_to_camera_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for name in self.chain:
            joint = self.joints[name]
            q = 0.0 if joint.joint_type == "fixed" else self.q.get(name, 0.0)
            T = T @ joint.transform(q)
        return T


class HeadCameraTfPublisher:
    """Attach to an existing rclpy Node and publish camera TF."""

    def __init__(
        self,
        node,
        urdf_path: Optional[str] = None,
        publish_hz: float = 30.0,
        camera_frame: str = CAMERA_FRAME_ID,
        camera_link: str = CAMERA_LINK_ID,
        base_frame: str = BASE_FRAME_ID,
    ) -> None:
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import TransformBroadcaster

        self.node = node
        self.camera_frame = camera_frame
        self.camera_link = camera_link
        self.base_frame = base_frame
        self._TransformStamped = TransformStamped
        path = urdf_path or find_default_urdf()
        self.fk = HeadCameraFk(path)  # path may be None -> builtin fallback
        self._lock = threading.Lock()
        self._broadcaster = TransformBroadcaster(node)
        self._last_log = 0.0
        self._have_waist = False
        self._have_head = False

        # Waist: prefer /fk/joint_states (JointState)
        node.create_subscription(
            __import__("sensor_msgs.msg", fromlist=["JointState"]).JointState,
            "/fk/joint_states",
            self._on_fk_joint_states,
            10,
        )
        # Head/waist HAL states (genie_msgs), optional
        self._try_subscribe_genie_states()

        period = 1.0 / max(1.0, float(publish_hz))
        self._timer = node.create_timer(period, self._on_timer)
        node.get_logger().info(
            f"HeadCameraTfPublisher: model={self.fk.source}; "
            f"publishing {base_frame} -> {camera_link} / {camera_frame}"
        )

    def _try_subscribe_genie_states(self) -> None:
        try:
            from genie_msgs.msg import HeadState, WaistState  # type: ignore
        except Exception as exc:
            self.node.get_logger().warning(
                f"无法导入 genie_msgs（头部/腰部状态）: {exc}；"
                "将仅使用 /fk/joint_states 中的腰部关节，头部关节默认为 0"
            )
            return
        self.node.create_subscription(HeadState, "/hal/neck_state", self._on_neck_state, 10)
        self.node.create_subscription(WaistState, "/hal/waist_state", self._on_waist_state, 10)

    def _on_fk_joint_states(self, msg) -> None:
        mapping = {}
        for name, pos in zip(list(msg.name), list(msg.position)):
            if name in ("joint_lift_body", "joint_body_pitch"):
                mapping[str(name)] = float(pos)
        if not mapping:
            return
        with self._lock:
            self.fk.set_positions(mapping)
            self._have_waist = True

    def _on_waist_state(self, msg) -> None:
        mapping = {}
        names = list(getattr(msg, "name", []) or [])
        motors = list(getattr(msg, "motor_states", []) or [])
        for i, name in enumerate(names):
            if i < len(motors):
                mapping[str(name)] = float(motors[i].position)
        if not mapping:
            return
        with self._lock:
            self.fk.set_positions(mapping)
            self._have_waist = True

    def _on_neck_state(self, msg) -> None:
        mapping = {}
        names = list(getattr(msg, "name", []) or [])
        motors = list(getattr(msg, "motor_states", []) or [])
        for i, name in enumerate(names):
            if i < len(motors):
                mapping[str(name)] = float(motors[i].position)
        if not mapping:
            return
        with self._lock:
            self.fk.set_positions(mapping)
            self._have_head = True

    def _stamp_now(self):
        return self.node.get_clock().now().to_msg()

    def _make_tf(self, parent: str, child: str, T: np.ndarray):
        msg = self._TransformStamped()
        msg.header.stamp = self._stamp_now()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])
        qx, qy, qz, qw = matrix_to_quat_xyzw(T[:3, :3])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        return msg

    def _on_timer(self) -> None:
        with self._lock:
            T = self.fk.base_to_camera_matrix()
            have_waist = self._have_waist
            have_head = self._have_head
        # Always publish: even with zero joints this unblocks TF lookups.
        tfs = [
            self._make_tf(self.base_frame, self.camera_link, T),
            self._make_tf(self.base_frame, self.camera_frame, T),
        ]
        self._broadcaster.sendTransform(tfs)
        now = time.time()
        if now - self._last_log > 5.0:
            self._last_log = now
            xyz = T[:3, 3]
            self.node.get_logger().info(
                f"camera TF ok: {self.base_frame}-> {self.camera_frame} "
                f"xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) "
                f"waist={'yes' if have_waist else 'no'} head={'yes' if have_head else 'no'}"
            )


def attach_to_node(node, urdf_path: Optional[str] = None) -> Optional[HeadCameraTfPublisher]:
    """Best-effort attach; returns None if URDF missing."""
    try:
        return HeadCameraTfPublisher(node, urdf_path=urdf_path)
    except Exception as exc:
        try:
            node.get_logger().warning(f"HeadCameraTfPublisher 未启动: {exc}")
        except Exception:
            print(f"HeadCameraTfPublisher 未启动: {exc}", file=sys.stderr)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Publish A2D head camera TF (camera_frame)")
    parser.add_argument("--urdf", default="", help="Path to A2D_RuiYan_D405.urdf")
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args(argv)

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=None)
    node = Node("a2d_head_camera_tf")
    try:
        HeadCameraTfPublisher(
            node,
            urdf_path=args.urdf or None,
            publish_hz=args.hz,
        )
    except Exception as exc:
        node.get_logger().error(str(exc))
        node.destroy_node()
        rclpy.shutdown()
        return 1
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
