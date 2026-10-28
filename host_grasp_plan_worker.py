"""Read-only IK preflight; never publishes or assumes collision safety.

Pregrasp uses a joint-space transit to a verified terminal IK solution. Approach
and lift retain Cartesian interpolation because their path geometry matters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from click_pose import validate_transform
from grasp_execution import (IK_ADJACENT_JUMP_LIMIT_RAD, TCP_POSITION_TOLERANCE_M,
                             TCP_ROTATION_TOLERANCE_RAD, timed_waypoints)


MAX_CARTESIAN_SEGMENT_M = .60
MAX_ROTATION_SEGMENT_RAD = 1.6


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
    def check(q, context=''):
        if not np.isfinite(q).all():
            raise ValueError('IK 解包含非有限关节值' + (f'（{context}）' if context else ''))
        over = np.maximum(lower-q, q-upper)
        bad = np.flatnonzero(over > 1e-3)
        if bad.size:
            i = int(bad[0])
            where = f'，{context}' if context else ''
            raise ValueError(
                f'IK 超出 URDF 关节限位：{_JOINT_NAMES[i]}={q[i]:.4f} rad，'
                f'允许 [{lower[i]:.4f}, {upper[i]:.4f}]{where}'
            )
        np.clip(q, lower, upper, out=q)
        pin.forwardKinematics(ik.model, ik.data, ik.q_from_list(q))
        pin.updateFramePlacements(ik.model, ik.data)
        return ik.data.oMf[fid].copy()
    start = check(seed, '起点')
    held_side = 'right' if side == 'left' else 'left'
    held_fid = ik._right_fid if side == 'left' else ik._left_fid
    held_start = ik.data.oMf[held_fid].copy()
    T = validate_transform(preview['goals'][stage])
    dist = np.linalg.norm(T[:3, 3]-start.translation)
    delta_rotation = pin.log3(start.rotation.T @ T[:3, :3])
    angle = np.linalg.norm(delta_rotation)
    simulation = request.get('execution_mode') == 'simulation'
    allow_waist = request.get('allow_waist') is True
    max_dist = MAX_CARTESIAN_SEGMENT_M
    max_angle = MAX_ROTATION_SEGMENT_RAD
    if dist > max_dist or angle > max_angle:
        mode = '仿真' if simulation else '真机'
        raise ValueError(
            f'{mode}单段距离 {dist:.3f}m（上限 {max_dist:.2f}m），'
            f'旋转 {np.degrees(angle):.1f}°（上限 {np.degrees(max_angle):.0f}°）；'
            '请调整目标或先移动到附近'
        )
    fixed = ((list(range(9,16)) if allow_waist else [0, 1, *range(9,16)])
             if side == 'left' else
             (list(range(2,9)) if allow_waist else list(range(9))))
    if stage == 'pregrasp':
        quat = pin.Quaternion(T[:3, :3]).coeffs()
        solved = ik.solve(seed.tolist(), moving_side=side, goal_xyz=tuple(T[:3, 3]),
                          goal_xyzw=tuple(quat), pos_tol=.002, ori_tol=.02,
                          max_iters=500, allow_waist=allow_waist)
        if solved is None:
            if not allow_waist:
                with_waist = ik.solve(
                    seed.tolist(), moving_side=side, goal_xyz=tuple(T[:3, 3]),
                    goal_xyzw=tuple(quat), pos_tol=.002, ori_tol=.02,
                    max_iters=500, allow_waist=True,
                )
                if with_waist is not None:
                    waist_delta = np.asarray(with_waist[:2], dtype=float) - seed[:2]
                    raise ValueError(
                        '预抓取终点在腰部固定时无解；允许腰部参与后可求解，'
                        f'需要升降变化 {waist_delta[0]:+.3f}m、俯仰变化 '
                        f'{np.degrees(waist_delta[1]):+.1f}°。可勾选“腰部参与 IK”后重试'
                    )
            raise ValueError('预抓取终点 IK 未收敛（已允许腰部参与）' if allow_waist else
                             '预抓取终点 IK 未收敛：固定腰部和允许腰部参与两种求解均失败')
        solved = np.asarray(solved, dtype=float)
        actual = check(solved, '预抓取终点')
        position_error = float(np.linalg.norm(actual.translation-T[:3, 3]))
        rotation_error = float(np.linalg.norm(pin.log3(T[:3, :3].T @ actual.rotation)))
        if position_error > TCP_POSITION_TOLERANCE_M or rotation_error > TCP_ROTATION_TOLERANCE_RAD:
            raise ValueError(
                f'预抓取终点 IK 误差过大：位置 {position_error*1000:.1f}mm，'
                f'旋转 {np.degrees(rotation_error):.2f}°'
            )
        if np.max(np.abs(solved[fixed]-seed[fixed])) > 1e-7:
            raise ValueError('IK 改动了腰部或另一只手臂')
        steps = max(2, int(np.ceil(np.max(np.abs(solved-seed))/.03)))
        rows = np.linspace(seed, solved, steps+1).tolist()
        path_kind = 'joint_space_pregrasp'
    else:
        steps = max(2, int(np.ceil(max(dist/.005, angle/.03))))
        rows = [seed.tolist()]
        for alpha in np.linspace(0, 1, steps+1)[1:]:
            xyz = start.translation*(1-alpha)+T[:3, 3]*alpha
            R = start.rotation @ pin.exp3(delta_rotation*alpha)
            quat = pin.Quaternion(R).coeffs()
            solved = ik.solve(rows[-1], moving_side=side, goal_xyz=tuple(xyz),
                              goal_xyzw=tuple(quat), pos_tol=.002, ori_tol=.02,
                              max_iters=150, allow_waist=allow_waist)
            if solved is None:
                raise ValueError(f'IK 未收敛（路径 {alpha:.0%}）')
            solved = np.asarray(solved, dtype=float)
            actual = check(solved, f'路径 {alpha:.0%}')
            position_error = float(np.linalg.norm(actual.translation-xyz))
            rotation_error = float(np.linalg.norm(pin.log3(R.T @ actual.rotation)))
            if position_error > TCP_POSITION_TOLERANCE_M or rotation_error > TCP_ROTATION_TOLERANCE_RAD:
                raise ValueError(
                    f'IK 位姿误差过大：路径 {alpha:.0%}，位置误差 {position_error*1000:.1f}mm'
                    f'（上限 {TCP_POSITION_TOLERANCE_M*1000:.1f}mm），'
                    f'旋转误差 {np.degrees(rotation_error):.2f}°'
                    f'（上限 {np.degrees(TCP_ROTATION_TOLERANCE_RAD):.2f}°）'
                )
            jump = np.abs(solved-np.asarray(rows[-1]))
            if np.max(jump) > IK_ADJACENT_JUMP_LIMIT_RAD:
                joint = int(np.argmax(jump))
                raise ValueError(
                    f'相邻 IK 解跳变：路径 {alpha:.0%}，{_JOINT_NAMES[joint]} '
                    f'变化 {jump[joint]:.3f}rad，超过 {IK_ADJACENT_JUMP_LIMIT_RAD:.2f}rad')
            if np.max(np.abs(solved[fixed]-seed[fixed])) > 1e-7:
                raise ValueError('IK 改动了腰部或另一只手臂')
            rows.append(solved.tolist())
        path_kind = 'cartesian'
    other_tcp_held = False
    if allow_waist:
        held_quat = pin.Quaternion(held_start.rotation).coeffs()
        active_indices = list(range(2,9)) if side == 'left' else list(range(9,16))
        compensated = [list(rows[0])]
        for row_index, row in enumerate(rows[1:], 1):
            before = np.asarray(row, dtype=float)
            solved = ik.solve(
                before.tolist(), moving_side=held_side,
                goal_xyz=tuple(held_start.translation), goal_xyzw=tuple(held_quat),
                pos_tol=.002, ori_tol=.02, max_iters=250,
                prefer_position=False, allow_waist=False,
            )
            if solved is None:
                raise ValueError(f'腰部补偿失败：{held_side} 手 TCP 在路径 {row_index/(len(rows)-1):.0%} 无解')
            solved = np.asarray(solved, dtype=float)
            check(solved, f'另一侧 TCP 补偿 {row_index/(len(rows)-1):.0%}')
            if np.max(np.abs(solved[[0,1,*active_indices]]-before[[0,1,*active_indices]])) > 1e-7:
                raise ValueError('另一侧 TCP 补偿改动了腰部或执行臂')
            held_actual = ik.data.oMf[held_fid]
            held_pos_error = float(np.linalg.norm(held_actual.translation-held_start.translation))
            held_rot_error = float(np.linalg.norm(pin.log3(held_start.rotation.T@held_actual.rotation)))
            if held_pos_error > TCP_POSITION_TOLERANCE_M or held_rot_error > TCP_ROTATION_TOLERANCE_RAD:
                raise ValueError(
                    f'另一侧 TCP 保持误差过大：路径 {row_index/(len(rows)-1):.0%}，'
                    f'位置 {held_pos_error*1000:.1f}mm，旋转 {np.degrees(held_rot_error):.1f}°')
            jump = np.max(np.abs(solved-np.asarray(compensated[-1])))
            if jump > IK_ADJACENT_JUMP_LIMIT_RAD:
                raise ValueError(f'另一侧 TCP 补偿关节跳变 {jump:.3f}rad，超过 {IK_ADJACENT_JUMP_LIMIT_RAD:.2f}rad')
            compensated.append(solved.tolist())
        rows = compensated
        other_tcp_held = True
    # Check interpolated paths as well, including TCP workspace bounds.
    bounds = np.asarray(profile['workspace_m'])
    for a, b in zip(rows, rows[1:]):
        for q in np.linspace(a, b, 6):
            actual = check(q)
            if np.any(actual.translation < bounds[0]) or np.any(actual.translation > bounds[1]):
                raise ValueError('插值路径超出工作空间')
            if other_tcp_held:
                held_actual = ik.data.oMf[held_fid]
                held_pos_error = float(np.linalg.norm(held_actual.translation-held_start.translation))
                held_rot_error = float(np.linalg.norm(pin.log3(held_start.rotation.T@held_actual.rotation)))
                if held_pos_error > TCP_POSITION_TOLERANCE_M or held_rot_error > TCP_ROTATION_TOLERANCE_RAD:
                    raise ValueError('插值路径无法保持另一只手的末端空间位置')
    times = timed_waypoints(rows, profile['joint_velocity_rad_s'], profile['joint_acceleration_rad_s2'])
    if times[-1] > 180:
        raise ValueError('单段轨迹超过 180 秒，请缩短移动距离')
    return {'ok': True, 'preview_id': preview['id'], 'stage': stage, 'arm_side': side,
            'backend': 'host_ik_manual', 'ik_checked': True, 'joint_limits_checked': True,
            'collision_checked': False, 'path_kind': path_kind,
            'waist_included': allow_waist,
            'other_tcp_held': other_tcp_held, 'held_side': held_side,
            'held_target': held_start.homogeneous.tolist(),
            'planned_at': time.time(), 'target': T.tolist(),
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
