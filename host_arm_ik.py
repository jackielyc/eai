#!/usr/bin/python3.10
"""宿主机手臂 IK（pinocchio CLIK），无 Docker / 无 ik_node 时供臂移动使用。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

# ROS overlays may contain an older Pinocchio ABI. Prefer the explicitly
# selected virtual environment's cmeel package when one is installed.
import sys as _ik_sys
from pathlib import Path as _IKPath
_ik_prefix = _IKPath(_ik_sys.prefix) / "lib" / f"python{_ik_sys.version_info.major}.{_ik_sys.version_info.minor}" / "site-packages" / "cmeel.prefix" / "lib" / f"python{_ik_sys.version_info.major}.{_ik_sys.version_info.minor}" / "site-packages"
if (_ik_prefix / "pinocchio").is_dir():
    _ik_sys.path.insert(0, str(_ik_prefix))

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
        self._waist_idxs = [self._q_idx[n] for n in _JOINT_NAMES[0:2]]
        self._left_idxs = [self._q_idx[n] for n in _JOINT_NAMES[2:9]]
        self._right_idxs = [self._q_idx[n] for n in _JOINT_NAMES[9:16]]
        self.last_info: Dict[str, float] = {}

    def q_from_list(self, q16: Sequence[float]):
        pin = self._pin
        q = pin.neutral(self.model)
        for i, name in enumerate(_JOINT_NAMES):
            if i < len(q16):
                q[self._q_idx[name]] = float(q16[i])
        return q

    def list_from_q(self, q) -> List[float]:
        return [float(q[self._q_idx[n]]) for n in _JOINT_NAMES]

    def ee_pose(
        self, q16: Sequence[float], moving_side: str
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pin = self._pin
        q = self.q_from_list(q16)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self._left_fid if moving_side == "left" else self._right_fid
        oMf = self.data.oMf[fid]
        xyz = (float(oMf.translation[0]), float(oMf.translation[1]), float(oMf.translation[2]))
        quat = pin.Quaternion(oMf.rotation)
        return xyz, (
            float(quat.x),
            float(quat.y),
            float(quat.z),
            float(quat.w),
        )

    def _clik(
        self,
        q,
        *,
        fid: int,
        idxs: List[int],
        oMg,
        max_iters: int,
        pos_tol: float,
        ori_tol: float,
        damp: float,
        step: float,
        ori_weight: float,
    ):
        pin = self._pin
        err = None
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMf[fid].actInv(oMg)
            err = pin.log6(iMd).vector.copy()
            pos_n = float(np.linalg.norm(err[:3]))
            ori_n = float(np.linalg.norm(err[3:]))
            if pos_n < pos_tol and (ori_weight <= 0.0 or ori_n < ori_tol):
                return True, err
            weighted = err.copy()
            weighted[3:] *= float(ori_weight)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL
            )
            J7 = J[:, idxs]
            if ori_weight <= 0.0:
                J7 = J7[:3, :]
                weighted = weighted[:3]
                eye = np.eye(3)
            else:
                eye = np.eye(6)
            dq7 = J7.T @ np.linalg.solve(J7 @ J7.T + damp * eye, weighted)
            for k, iq in enumerate(idxs):
                q[iq] = float(q[iq] + step * dq7[k])
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        err = pin.log6(self.data.oMf[fid].actInv(oMg)).vector
        return float(np.linalg.norm(err[:3])) < pos_tol * 2.5, err

    def solve(
        self,
        q16_seed: Sequence[float],
        *,
        moving_side: str,
        goal_xyz: Tuple[float, float, float],
        goal_xyzw: Tuple[float, float, float, float],
        max_iters: int = 120,
        pos_tol: float = 0.006,
        ori_tol: float = 0.08,
        damp: float = 1e-3,
        step: float = 0.35,
        prefer_position: bool = True,
    ) -> Optional[List[float]]:
        """求解 moving_side 的 Link7 到目标位姿；失败返回 None。

        姿态与位置同时很难时，默认再试「保位置、弱姿态 / 只位置」。
        """
        pin = self._pin
        fid = self._left_fid if moving_side == "left" else self._right_fid
        arm_idxs = self._left_idxs if moving_side == "left" else self._right_idxs
        # 用当前 FK 姿态作为兜底姿态，避免 TF 姿态与 seed 不一致时把位置求解拖垮
        seed_xyz, seed_xyzw = self.ee_pose(q16_seed, moving_side)
        travel = float(
            np.linalg.norm(np.asarray(goal_xyz, dtype=float) - np.asarray(seed_xyz, dtype=float))
        )
        # 大位移时放开腰部两轴，否则单臂 7 关节经常够不到桌面目标
        idx_sets = [arm_idxs]
        if travel > 0.12:
            idx_sets.append(self._waist_idxs + arm_idxs)

        def _goal_se3(xyzw: Sequence[float]):
            return pin.SE3(
                pin.Quaternion(
                    float(xyzw[3]),
                    float(xyzw[0]),
                    float(xyzw[1]),
                    float(xyzw[2]),
                ).matrix(),
                np.array(goal_xyz, dtype=float),
            )

        attempts = [
            ("full", goal_xyzw, 1.0, max_iters, pos_tol, ori_tol),
            ("weak_ori", goal_xyzw, 0.08, max_iters, pos_tol, ori_tol * 4),
        ]
        if prefer_position:
            attempts.append(("seed_ori", seed_xyzw, 0.15, max_iters, pos_tol, ori_tol * 3))
            attempts.append(("pos_only", seed_xyzw, 0.0, max_iters, pos_tol, 1.0))

        best_q = None
        best_pos = 1e9
        best_ori = 1e9
        best_mode = ""
        for idxs in idx_sets:
            for mode, xyzw, ori_w, iters, p_tol, o_tol in attempts:
                q = self.q_from_list(q16_seed)
                ok, err = self._clik(
                    q,
                    fid=fid,
                    idxs=idxs,
                    oMg=_goal_se3(xyzw),
                    max_iters=iters,
                    pos_tol=p_tol,
                    ori_tol=o_tol,
                    damp=damp,
                    step=step,
                    ori_weight=ori_w,
                )
                pos_n = float(np.linalg.norm(err[:3]))
                ori_n = float(np.linalg.norm(err[3:]))
                tag = mode + ("+waist" if len(idxs) > 7 else "")
                if pos_n < best_pos:
                    best_pos = pos_n
                    best_ori = ori_n
                    best_q = self.list_from_q(q)
                    best_mode = tag
                if ok:
                    self.last_info = {
                        "travel_m": travel,
                        "pos_err_m": pos_n,
                        "ori_err_rad": ori_n,
                        "mode": float(len(idxs)),
                    }
                    return self.list_from_q(q)

        self.last_info = {
            "travel_m": travel,
            "pos_err_m": best_pos,
            "ori_err_rad": best_ori,
            "mode": float(len(best_mode)),
        }
        # 位置已经够近时仍返回（姿态可能未对齐）
        if best_q is not None and best_pos < max(0.015, pos_tol * 2.5):
            return best_q
        return None


_IK_SINGLETON: Optional[HostArmIK] = None


def get_host_arm_ik() -> HostArmIK:
    global _IK_SINGLETON
    if _IK_SINGLETON is None:
        _IK_SINGLETON = HostArmIK()
    return _IK_SINGLETON
