"""Read-only Cartesian IK preflight; never publishes or assumes collision safety."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from click_pose import validate_transform
from grasp_execution import timed_waypoints


def prepare(request):
    from host_arm_ik import HostArmIK, _JOINT_NAMES
    profile, preview, stage = request['profile'], request['preview'], request['stage']
    ik = HostArmIK(profile['urdf_path'])
    pin = ik._pin
    if ik.model.nq != 16:
        raise ValueError('当前宿主机控制只支持腰2+双臂14自由度模型')
    side = profile['arm_side']
    fid = ik._left_fid if side == 'left' else ik._right_fid
    seed = np.asarray(request['start_joints'], dtype=float)
    if seed.shape != (16,) or not np.isfinite(seed).all():
        raise ValueError('无效实测关节种子')
    indices = [ik._q_idx[name] for name in _JOINT_NAMES]
    lower, upper = ik.model.lowerPositionLimit[indices], ik.model.upperPositionLimit[indices]
    def check(q):
        if not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            raise ValueError('IK 超出 URDF 关节限位')
        pin.forwardKinematics(ik.model, ik.data, ik.q_from_list(q))
        pin.updateFramePlacements(ik.model, ik.data)
        return ik.data.oMf[fid].copy()
    start = check(seed)
    T = validate_transform(preview['goals'][stage])
    dist = np.linalg.norm(T[:3, 3]-start.translation)
    delta_rotation = pin.log3(start.rotation.T @ T[:3, :3])
    angle = np.linalg.norm(delta_rotation)
    if dist > .35 or angle > 1.6:
        raise ValueError('单段位移>0.35m或旋转>约90度，请先手动移动到附近')
    steps = max(2, int(np.ceil(max(dist/.005, angle/.03))))
    rows = [seed.tolist()]
    for alpha in np.linspace(0, 1, steps+1)[1:]:
        xyz = start.translation*(1-alpha)+T[:3, 3]*alpha
        R = start.rotation @ pin.exp3(delta_rotation*alpha)
        quat = pin.Quaternion(R).coeffs()
        solved = ik.solve(rows[-1], moving_side=side, goal_xyz=tuple(xyz),
                          goal_xyzw=tuple(quat), pos_tol=.002, ori_tol=.02, max_iters=150)
        if solved is None:
            raise ValueError(f'IK 未收敛（路径 {alpha:.0%}）')
        actual = check(np.asarray(solved))
        if np.linalg.norm(actual.translation-xyz) > .004 or np.linalg.norm(pin.log3(R.T @ actual.rotation)) > .04:
            raise ValueError('IK 位姿误差过大（包括旋转），不能只按位置通过')
        if np.max(np.abs(np.asarray(solved)-rows[-1])) > .12:
            raise ValueError('相邻 IK 解跳变超过 0.12 rad')
        fixed = [0, 1, *range(9,16)] if side == 'left' else list(range(9))
        if np.max(np.abs(np.asarray(solved)[fixed]-seed[fixed])) > 1e-7:
            raise ValueError('IK 改动了腰部或另一只手臂')
        rows.append(solved)
    # Check interpolated paths as well, including TCP workspace bounds.
    bounds = np.asarray(profile['workspace_m'])
    for a, b in zip(rows, rows[1:]):
        for q in np.linspace(a, b, 6):
            actual = check(q)
            if np.any(actual.translation < bounds[0]) or np.any(actual.translation > bounds[1]):
                raise ValueError('插值路径超出工作空间')
    times = timed_waypoints(rows, profile['joint_velocity_rad_s'], profile['joint_acceleration_rad_s2'])
    if times[-1] > 180:
        raise ValueError('单段轨迹超过 180 秒，请缩短移动距离')
    return {'ok': True, 'preview_id': preview['id'], 'stage': stage, 'arm_side': side,
            'backend': 'host_ik_manual', 'ik_checked': True, 'joint_limits_checked': True,
            'collision_checked': False, 'planned_at': time.time(), 'target': T.tolist(),
            'joint_waypoints': rows, 'times_s': times,
            'urdf_sha256': hashlib.sha256(Path(profile['urdf_path']).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = prepare(json.loads(Path(args.request).read_text()))
    except Exception as exc:
        result = {'ok': False, 'error': str(exc)}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
