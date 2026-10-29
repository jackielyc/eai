#!/usr/bin/python3.10
"""宿主机手臂 IK（pinocchio CLIK），无 Docker / 无 ik_node 时供臂移动使用。"""
from __future__ import annotations

from eai_app.paths import PROJECT_ROOT

import os
import time
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
    here = str(PROJECT_ROOT)
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
    """Bounded full-pose IK; explicitly selected joints and optional dual TCP hold."""

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

    def solve_targets(self, q16_seed, targets, variable_names, *, pos_tol=.002,
                      ori_tol=.02, max_iters=250, attempts=1, max_change=None,
                      deadline=None):
        """Solve all (frame_id, SE3) targets jointly, within URDF bounds.

        No pose relaxation or waist fallback. All returned targets are checked
        against the original position and rotation, not merely solver status.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise RuntimeError('IK 环境缺少兼容的 SciPy，请安装 requirements/requirements-grasp-ik.txt') from exc
        pin = self._pin
        seed = np.asarray(q16_seed, dtype=float)
        if seed.shape != (16,) or not np.isfinite(seed).all():
            raise ValueError('IK 需要完整、有限的16关节起点')
        if self.model.nq != self.model.nv or self.model.nq != 16:
            raise ValueError('IK 只支持当前16自由度A2D模型')
        if not targets or pos_tol <= 0 or ori_tol <= 0:
            raise ValueError('IK 目标和容差无效')
        all_q = [self._q_idx[n] for n in _JOINT_NAMES]
        lo, hi = self.model.lowerPositionLimit[all_q], self.model.upperPositionLimit[all_q]
        if not np.isfinite([lo, hi]).all() or np.any(seed < lo-1e-3) or np.any(seed > hi+1e-3):
            raise ValueError('IK 起点超出URDF关节限位')
        seed = np.clip(seed, lo, hi)
        pin.forwardKinematics(self.model, self.data, self.q_from_list(seed))
        pin.updateFramePlacements(self.model, self.data)
        initial_errors = [(float(np.linalg.norm(self.data.oMf[f].translation-t.translation)),
                           float(np.linalg.norm(pin.log3(t.rotation.T@self.data.oMf[f].rotation))))
                          for f,t in targets]
        if all(p <= pos_tol and r <= ori_tol for p,r in initial_errors):
            self.last_info = {'pos_err_m': max(p for p,r in initial_errors),
                              'ori_err_rad': max(r for p,r in initial_errors),
                              'attempts': 0, 'solver': 'bounded_joint_ik'}
            return seed.tolist()
        columns = [_JOINT_NAMES.index(name) for name in variable_names]
        velocity_columns = [self.model.joints[self.model.getJointId(n)].idx_v for n in variable_names]
        lower, upper = lo[columns].copy(), hi[columns].copy()
        if max_change is not None:
            limit = np.broadcast_to(np.asarray(max_change, dtype=float), seed.shape)[columns]
            if np.any(limit <= 0) or not np.isfinite(limit).all():
                raise ValueError('IK 局部步长限制无效')
            lower = np.maximum(lower, seed[columns]-limit)
            upper = np.minimum(upper, seed[columns]+limit)
        scale = np.array([1/pos_tol]*3 + [1/ori_tol]*3)
        span = np.maximum(hi[columns]-lo[columns], .01)
        cached_x = None
        residual = jacobian = None

        def evaluate(x):
            nonlocal cached_x, residual, jacobian
            if deadline is not None and time.monotonic() > deadline:
                raise ValueError('IK 计算超时，未生成可执行轨迹')
            if cached_x is not None and np.array_equal(x, cached_x):
                return
            q16 = seed.copy(); q16[columns] = x
            q = self.q_from_list(q16)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            errors, derivatives = [], []
            for fid, desired in targets:
                delta = self.data.oMf[fid].actInv(desired)
                errors.append(pin.log6(delta).vector * scale)
                J = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL)
                derivatives.append((-pin.Jlog6(delta.inverse()) @ J)[:, velocity_columns] * scale[:, None])
            # Prefer nearby configurations without changing task tolerances.
            errors.append(.01*(x-seed[columns])/span)
            derivatives.append(np.diag(.01/span))
            residual, jacobian = np.concatenate(errors), np.vstack(derivatives)
            cached_x = x.copy()

        def fun(x):
            evaluate(x)
            return residual

        def jac(x):
            evaluate(x)
            return jacobian

        rng = np.random.default_rng(17)
        best = float('inf')
        for trial in range(max(1, attempts)):
            x0 = seed[columns].copy()
            if trial:
                perturb = np.array([.02 if name == 'joint_lift_body' else .25 for name in variable_names])
                x0 += rng.uniform(-1, 1, len(columns))*perturb
            x0 = np.clip(x0, lower+1e-10, upper-1e-10)
            fit = least_squares(fun, x0, jac=jac, bounds=(lower, upper), method='trf',
                                x_scale='jac', max_nfev=max_iters,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            solved = seed.copy(); solved[columns] = fit.x
            pin.forwardKinematics(self.model,self.data,self.q_from_list(solved))
            pin.updateFramePlacements(self.model,self.data)
            errors = [(float(np.linalg.norm(self.data.oMf[f].translation-t.translation)),
                       float(np.linalg.norm(pin.log3(t.rotation.T@self.data.oMf[f].rotation))))
                      for f,t in targets]
            score = max(max(p/pos_tol, r/ori_tol) for p,r in errors)
            if score < best:
                best = score
                self.last_info = {'pos_err_m': max(p for p,r in errors),
                                  'ori_err_rad': max(r for p,r in errors),
                                  'attempts': trial+1, 'solver': 'bounded_joint_ik'}
            if all(p <= pos_tol and r <= ori_tol for p,r in errors):
                return solved.tolist()
        return None

    def solve_coordinated(self, seed, *, moving_side, target, allow_waist=False,
                          held_target=None, **kwargs):
        if moving_side not in ('left','right'):
            raise ValueError('IK 执行侧必须是 left/right')
        active = self._left_fid if moving_side == 'left' else self._right_fid
        other = self._right_fid if moving_side == 'left' else self._left_fid
        targets = [(active, target)]
        if allow_waist:
            if held_target is None:
                self._pin.forwardKinematics(self.model,self.data,self.q_from_list(seed))
                self._pin.updateFramePlacements(self.model,self.data)
                held_target = self.data.oMf[other].copy()
            targets.append((other, held_target))
            names = _JOINT_NAMES
        else:
            names = _JOINT_NAMES[2:9] if moving_side == 'left' else _JOINT_NAMES[9:16]
        return self.solve_targets(seed, targets, names, **kwargs)

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
            J7 = (pin.Jlog6(iMd.inverse()) @ J)[:, idxs]
            if ori_weight <= 0.0:
                J7 = J7[:3, :]
                weighted = weighted[:3]
                eye = np.eye(3)
            else:
                J7[3:, :] *= float(ori_weight)
                eye = np.eye(6)
            dq7 = J7.T @ np.linalg.solve(J7 @ J7.T + damp * eye, weighted)
            for k, iq in enumerate(idxs):
                q[iq] = float(q[iq] + step * dq7[k])
                q[iq] = float(np.clip(q[iq], self.model.lowerPositionLimit[iq], self.model.upperPositionLimit[iq]))
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        err = pin.log6(self.data.oMf[fid].actInv(oMg)).vector
        return (float(np.linalg.norm(err[:3])) < pos_tol and
                (ori_weight <= 0. or float(np.linalg.norm(err[3:])) < ori_tol)), err

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
        prefer_position: bool = False,
        allow_waist: bool = True,
    ) -> Optional[List[float]]:
        """Bounded full-pose solve for manual control; never silently relax orientation.

        Legacy damp/step/prefer_position arguments remain accepted for callers.
        allow_waist controls the variables even for small displacements.
        """
        del damp, step, prefer_position
        if moving_side not in ('left', 'right'):
            raise ValueError('IK 执行侧必须是 left/right')
        quat = np.asarray(goal_xyzw, dtype=float)
        xyz = np.asarray(goal_xyz, dtype=float)
        if quat.shape != (4,) or xyz.shape != (3,) or not np.isfinite(quat).all() or not np.isfinite(xyz).all() or np.linalg.norm(quat) < 1e-9:
            raise ValueError('IK 目标位姿无效')
        quat /= np.linalg.norm(quat)
        target = self._pin.SE3(self._pin.Quaternion(quat[3], *quat[:3]).matrix(), xyz)
        fid = self._left_fid if moving_side == 'left' else self._right_fid
        names = list(_JOINT_NAMES[2:9] if moving_side == 'left' else _JOINT_NAMES[9:16])
        if allow_waist:
            names = _JOINT_NAMES[:2] + names
        start, _ = self.ee_pose(q16_seed, moving_side)
        result = self.solve_targets(q16_seed, [(fid, target)], names, pos_tol=pos_tol,
                                    ori_tol=ori_tol, max_iters=max_iters, attempts=3,
                                    deadline=time.monotonic()+10.)
        self.last_info['travel_m'] = float(np.linalg.norm(xyz-start))
        return result


_IK_SINGLETON: Optional[HostArmIK] = None


def get_host_arm_ik() -> HostArmIK:
    global _IK_SINGLETON
    if _IK_SINGLETON is None:
        _IK_SINGLETON = HostArmIK()
    return _IK_SINGLETON
