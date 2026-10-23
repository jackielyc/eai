#!/usr/bin/python3.10
"""宿主机手臂 IK（pinocchio CLIK），无 Docker / 无 ik_node 时供臂移动使用。"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 与 host_fk_node 共用 URDF / 关节名
_JOINT_NAMES = [
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
_LEFT_EE = "Link7_l"
_RIGHT_EE = "Link7_r"


def _resolve_urdf() -> str:
    env = os.environ.get("EAI_HOST_FK_URDF", "").strip()
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.normpath(
            os.path.join(
                here,
                "..",
                "a2d-tele",
                "modules",
                "control",
                "wbc",
                "assets",
                "A2D_NoHand",
                "A2D_NoHand_Flattened.urdf",
            )
        ),
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf",
    ):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("A2D URDF not found")


class HostArmIK:
    """16-DoF（腰2 + 双臂14）CLIK；移动一侧时冻结腰与另一侧。"""

    def __init__(self, urdf_path: Optional[str] = None) -> None:
        import pinocchio as pin

        self._pin = pin
        path = urdf_path or _resolve_urdf()
        self.model = pin.buildModelFromUrdf(path)
        self.data = self.model.createData()
        self._q_idx = {}
        for name in _JOINT_NAMES:
            jid = self.model.getJointId(name)
            self._q_idx[name] = self.model.joints[jid].idx_q
        self._left_fid = self.model.getFrameId(_LEFT_EE)
        self._right_fid = self.model.getFrameId(_RIGHT_EE)
        self._left_idxs = [self._q_idx[n] for n in _JOINT_NAMES[2:9]]
        self._right_idxs = [self._q_idx[n] for n in _JOINT_NAMES[9:16]]

    def q_from_list(self, q16: Sequence[float]):
        pin = self._pin
        q = pin.neutral(self.model)
        for i, name in enumerate(_JOINT_NAMES):
            if i < len(q16):
                q[self._q_idx[name]] = float(q16[i])
        return q

    def list_from_q(self, q) -> List[float]:
        return [float(q[self._q_idx[n]]) for n in _JOINT_NAMES]

    def solve(
        self,
        q16_seed: Sequence[float],
        *,
        moving_side: str,
        goal_xyz: Tuple[float, float, float],
        goal_xyzw: Tuple[float, float, float, float],
        max_iters: int = 80,
        pos_tol: float = 0.004,
        ori_tol: float = 0.04,
        damp: float = 1e-3,
        step: float = 0.4,
    ) -> Optional[List[float]]:
        """求解 moving_side 的 Link7 到目标位姿；失败返回 None。"""
        pin = self._pin
        q = self.q_from_list(q16_seed)
        fid = self._left_fid if moving_side == "left" else self._right_fid
        idxs = self._left_idxs if moving_side == "left" else self._right_idxs
        oMg = pin.SE3(
            pin.Quaternion(
                float(goal_xyzw[3]),
                float(goal_xyzw[0]),
                float(goal_xyzw[1]),
                float(goal_xyzw[2]),
            ).matrix(),
            np.array(goal_xyz, dtype=float),
        )
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[fid]
            iMd = oMf.actInv(oMg)
            err = pin.log6(iMd).vector  # 6
            if float(np.linalg.norm(err[:3])) < pos_tol and float(
                np.linalg.norm(err[3:])
            ) < ori_tol:
                return self.list_from_q(q)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL
            )
            # 只取移动臂 7 关节列
            J7 = J[:, idxs]
            JJt = J7 @ J7.T
            dq7 = J7.T @ np.linalg.solve(JJt + damp * np.eye(6), err)
            for k, iq in enumerate(idxs):
                q[iq] = float(q[iq] + step * dq7[k])
        # 最后一次检查
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        iMd = self.data.oMf[fid].actInv(oMg)
        err = pin.log6(iMd).vector
        if float(np.linalg.norm(err[:3])) < pos_tol * 2.5:
            return self.list_from_q(q)
        return None


_IK_SINGLETON: Optional[HostArmIK] = None


def get_host_arm_ik() -> HostArmIK:
    global _IK_SINGLETON
    if _IK_SINGLETON is None:
        _IK_SINGLETON = HostArmIK()
    return _IK_SINGLETON
