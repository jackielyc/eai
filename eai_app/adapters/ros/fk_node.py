#!/usr/bin/python3.10
"""宿主机 FK：由 /hal/arm_joint_state + /hal/waist_state 发布 /mink_fk/*_tcp_pose。

替代缺失的 wbc fk_node（无 Docker）。数值与官方 fkSolve 同为 world/base 系 Link7，
frame_id 标为 map（与 mink_fk 一致）。
"""
from __future__ import annotations

from eai_app.paths import PROJECT_ROOT

import json
import hashlib
import os
import signal
import sys
import tempfile
import time
from typing import Dict, List, Optional

# ROS overlays may contain an older Pinocchio ABI. Prefer the explicitly
# selected virtual environment's cmeel package when one is installed.
import sys as _ik_sys
from pathlib import Path as _IKPath
_ik_prefix = _IKPath(_ik_sys.prefix) / "lib" / f"python{_ik_sys.version_info.major}.{_ik_sys.version_info.minor}" / "site-packages" / "cmeel.prefix" / "lib" / f"python{_ik_sys.version_info.major}.{_ik_sys.version_info.minor}" / "site-packages"
if (_ik_prefix / "pinocchio").is_dir():
    _ik_sys.path.insert(0, str(_ik_prefix))

import numpy as np

FK_STATE_PATH = os.environ.get(
    "EAI_HOST_FK_STATE",
    os.path.join(tempfile.gettempdir(), "eai_host_fk_state.json"),
)

DEFAULT_URDF = os.path.join(
    str(PROJECT_ROOT),
    "..",
    "a2d-tele",
    "modules",
    "control",
    "wbc",
    "assets",
    "A2D_NoHand",
    "A2D_NoHand_Flattened.urdf",
)
# eai 与 a2d-tele 同级时的备用路径
_ALT_URDF = os.path.join(
    str(PROJECT_ROOT),
    "a2d-tele",
    "modules",
    "control",
    "wbc",
    "assets",
    "A2D_NoHand",
    "A2D_NoHand_Flattened.urdf",
)

JOINT_NAMES: List[str] = [
    "joint_lift_body",
    "joint_body_pitch",
    "joint1_l",
    "joint2_l",
    "joint3_l",
    "joint4_l",
    "joint5_l",
    "joint6_l",
    "joint7_l",
    "joint1_r",
    "joint2_r",
    "joint3_r",
    "joint4_r",
    "joint5_r",
    "joint6_r",
    "joint7_r",
]
LEFT_EE = "Link7_l"
RIGHT_EE = "Link7_r"
BASE_FRAME = "map"
ARM_TOPIC = "/hal/arm_joint_state"
WAIST_TOPIC = "/hal/waist_state"
LEFT_TCP_TOPIC = "/mink_fk/left_tcp_pose"
RIGHT_TCP_TOPIC = "/mink_fk/right_tcp_pose"
LEFT_Z_UP_TOPIC = "/mink_fk/left_tcp_pose_z_up"
RIGHT_Z_UP_TOPIC = "/mink_fk/right_tcp_pose_z_up"
JOINT_STATES_TOPIC = "/joint_states"

# [x,y,z, qx,qy,qz,qw] — 与 a2d_tele.wbc.yaml fk_solver 一致
LEFT_TCP_OFFSET = (0.0, 0.0, 0.0, 0.5, -0.5, 0.5, 0.5)
RIGHT_TCP_OFFSET = (0.0, 0.0, 0.0, -0.5, -0.5, -0.5, 0.5)


def _resolve_urdf() -> str:
    env = os.environ.get("EAI_HOST_FK_URDF", "").strip()
    if env and os.path.isfile(env):
        return env
    for path in (
        os.path.normpath(DEFAULT_URDF),
        os.path.normpath(_ALT_URDF),
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf",
    ):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("A2D_NoHand_Flattened.urdf not found; set EAI_HOST_FK_URDF")


def _quat_mul(q1, q2):
    """Hamilton product, both (x,y,z,w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _apply_tcp_offset(xyz, xyzw, offset7):
    ox, oy, oz, qx, qy, qz, qw = offset7
    # R_pose * R_offset, t = t_pose + R_pose * t_offset
    import pinocchio as pin

    R = pin.Quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2]).matrix()
    t_off = R @ np.array([ox, oy, oz], dtype=float)
    new_xyz = (np.asarray(xyz, dtype=float) + t_off).tolist()
    new_q = _quat_mul(xyzw, (qx, qy, qz, qw))
    n = (new_q[0] ** 2 + new_q[1] ** 2 + new_q[2] ** 2 + new_q[3] ** 2) ** 0.5
    new_q = tuple(c / n for c in new_q)
    return new_xyz, new_q


class HostFK:
    def __init__(self, urdf_path: str) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._q_idx: Dict[str, int] = {}
        for name in JOINT_NAMES:
            if not self.model.existJointName(name):
                raise RuntimeError(f"URDF missing joint {name}")
            jid = self.model.getJointId(name)
            self._q_idx[name] = self.model.joints[jid].idx_q
        self._left_fid = self.model.getFrameId(LEFT_EE)
        self._right_fid = self.model.getFrameId(RIGHT_EE)
        self.q = pin.neutral(self.model)
        self.has_waist = False
        self.has_arm = False

    def set_waist(self, name_to_pos: Dict[str, float]) -> None:
        for name in ("joint_lift_body", "joint_body_pitch"):
            if name not in name_to_pos:
                return
            self.q[self._q_idx[name]] = float(name_to_pos[name])
        self.has_waist = True

    def set_arm_positions(self, positions: List[float]) -> None:
        if len(positions) < 14:
            return
        arm_names = JOINT_NAMES[2:]
        for i, name in enumerate(arm_names):
            self.q[self._q_idx[name]] = float(positions[i])
        self.has_arm = True

    def solve(self):
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        def _pose(fid):
            oMf = self.data.oMf[fid]
            xyz = oMf.translation.tolist()
            quat = pin.Quaternion(oMf.rotation)
            # pinocchio Quaternion stores (x,y,z,w) via .coeffs()
            c = quat.coeffs()  # x,y,z,w
            return xyz, (float(c[0]), float(c[1]), float(c[2]), float(c[3]))

        return _pose(self._left_fid), _pose(self._right_fid)

    def q16(self) -> List[float]:
        return [float(self.q[self._q_idx[n]]) for n in JOINT_NAMES]


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
    from sensor_msgs.msg import JointState

    try:
        from genie_msgs.msg import WaistState
    except ImportError:
        print(
            "genie_msgs 未找到：请 source 含 genie_msgs 的 install（如 robot-control-library）",
            file=sys.stderr,
        )
        return 1

    urdf = _resolve_urdf()
    fk = HostFK(urdf)
    with open(urdf, "rb") as stream:
        urdf_sha256 = hashlib.sha256(stream.read()).hexdigest()
    received = {"arm": 0.0, "waist": 0.0}

    stop = {"v": False}

    def _sig(*_a):
        stop["v"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    rclpy.init()
    node = Node("eai_host_fk_node")

    qos_be = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    qos_rel = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    left_pub = node.create_publisher(PoseStamped, LEFT_TCP_TOPIC, 100)
    right_pub = node.create_publisher(PoseStamped, RIGHT_TCP_TOPIC, 100)
    left_z_pub = node.create_publisher(PoseStamped, LEFT_Z_UP_TOPIC, 100)
    right_z_pub = node.create_publisher(PoseStamped, RIGHT_Z_UP_TOPIC, 100)
    joint_pub = node.create_publisher(JointState, JOINT_STATES_TOPIC, 100)

    pub_count = {"n": 0}
    last_warn = {"t": 0.0}

    def _fill(msg: PoseStamped, xyz, xyzw) -> None:
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(xyzw[0])
        msg.pose.orientation.y = float(xyzw[1])
        msg.pose.orientation.z = float(xyzw[2])
        msg.pose.orientation.w = float(xyzw[3])

    def _publish() -> None:
        if not fk.has_arm:
            return
        # 腰未到时用当前 q（neutral/上次）仍发布，避免按钮永久灰掉
        if not fk.has_waist:
            now = time.time()
            if now - last_warn["t"] > 2.0:
                last_warn["t"] = now
                node.get_logger().warn("waiting /hal/waist_state; publishing with waist=0/last")
        stamp = node.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = BASE_FRAME
        js.name = list(JOINT_NAMES)
        js.position = fk.q16()
        joint_pub.publish(js)
        (l_xyz, l_q), (r_xyz, r_q) = fk.solve()
        left_msg = PoseStamped()
        right_msg = PoseStamped()
        _fill(left_msg, l_xyz, l_q)
        _fill(right_msg, r_xyz, r_q)
        left_pub.publish(left_msg)
        right_pub.publish(right_msg)

        lz_xyz, lz_q = _apply_tcp_offset(l_xyz, l_q, LEFT_TCP_OFFSET)
        rz_xyz, rz_q = _apply_tcp_offset(r_xyz, r_q, RIGHT_TCP_OFFSET)
        lz = PoseStamped()
        rz = PoseStamped()
        _fill(lz, lz_xyz, lz_q)
        _fill(rz, rz_xyz, rz_q)
        left_z_pub.publish(lz)
        right_z_pub.publish(rz)
        # 供 UI 宿主机 IK 读种子关节（无 ik_node 时）
        try:
            state = {
                "q16": fk.q16(),
                "arm_received_at": received["arm"],
                "waist_received_at": received["waist"],
                "urdf_sha256": urdf_sha256,
                "left": {"xyz": l_xyz, "xyzw": list(l_q)},
                "right": {"xyz": r_xyz, "xyzw": list(r_q)},
                "t": time.time(),
            }
            tmp = FK_STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, separators=(",", ":"))
            os.replace(tmp, FK_STATE_PATH)
        except Exception:
            pass
        pub_count["n"] += 1
        if pub_count["n"] == 1:
            node.get_logger().info(
                f"first TCP L=[{l_xyz[0]:.3f},{l_xyz[1]:.3f},{l_xyz[2]:.3f}] "
                f"R=[{r_xyz[0]:.3f},{r_xyz[1]:.3f},{r_xyz[2]:.3f}] urdf={urdf}"
            )

    def _on_waist(msg: WaistState) -> None:
        name_to_pos: Dict[str, float] = {}
        n = min(len(msg.name), len(msg.motor_states))
        for i in range(n):
            name_to_pos[str(msg.name[i])] = float(msg.motor_states[i].position)
        if all(name in name_to_pos and np.isfinite(name_to_pos[name])
               for name in ("joint_lift_body", "joint_body_pitch")):
            fk.set_waist(name_to_pos)
            received["waist"] = time.time()

    def _on_arm(msg: JointState) -> None:
        if len(msg.position) < 14 or not np.isfinite(msg.position[:14]).all():
            return
        fk.set_arm_positions(list(msg.position))
        received["arm"] = time.time()
        _publish()

    node.create_subscription(WaistState, WAIST_TOPIC, _on_waist, qos_rel)
    time.sleep(0.05)
    node.create_subscription(JointState, ARM_TOPIC, _on_arm, qos_be)
    node.get_logger().info(f"host FK up → {LEFT_TCP_TOPIC} / {RIGHT_TCP_TOPIC}")

    while rclpy.ok() and not stop["v"]:
        rclpy.spin_once(node, timeout_sec=0.05)

    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
