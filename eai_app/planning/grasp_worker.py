"""Read-only bounded joint IK and adaptive path preflight. No hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from eai_app.domain.geometry.transforms import validate_transform
from eai_app.domain.grasp.execution import (IK_ADJACENT_JUMP_LIMIT_RAD, TCP_POSITION_TOLERANCE_M,
                             TCP_ROTATION_TOLERANCE_RAD, timed_waypoints)

MAX_CARTESIAN_SEGMENT_M = .60
MAX_ROTATION_SEGMENT_RAD = 1.6
MAX_PREFLIGHT_SECONDS = 60.


def prepare(request):
    from eai_app.planning.arm_ik import HostArmIK, _JOINT_NAMES
    profile, preview, stage = request['profile'], request['preview'], request['stage']
    precision=preview.get('precision') or {}
    position_tolerance=float(precision.get('position_tolerance_m',TCP_POSITION_TOLERANCE_M))
    rotation_tolerance=float(precision.get('rotation_tolerance_rad',TCP_ROTATION_TOLERANCE_RAD))
    if stage not in ('pregrasp', 'approach', 'lift'):
        raise ValueError('未知运动阶段')
    side = profile['arm_side']
    if side not in ('left', 'right'):
        raise ValueError('必须明确指定 left/right')
    ik = HostArmIK(profile['urdf_path'])
    pin = ik._pin
    if ik.model.nq != 16:
        raise ValueError('当前宿主机控制只支持腰2+双臂14自由度模型')
    allow_waist = request.get('allow_waist') is True
    fid = ik._left_fid if side == 'left' else ik._right_fid
    held_side = 'right' if side == 'left' else 'left'
    held_fid = ik._right_fid if side == 'left' else ik._left_fid
    seed = np.asarray(request['start_joints'], dtype=float)
    if seed.shape != (16,) or not np.isfinite(seed).all():
        raise ValueError('无效实测关节种子')
    indices = [ik._q_idx[name] for name in _JOINT_NAMES]
    lower, upper = ik.model.lowerPositionLimit[indices], ik.model.upperPositionLimit[indices]
    if np.any(seed < lower-1e-3) or np.any(seed > upper+1e-3):
        raise ValueError('起点超出 URDF 关节限位')
    seed = np.clip(seed, lower, upper)
    bounds = np.asarray(profile['workspace_m'], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('工作空间配置无效')

    def fk(q):
        if not np.isfinite(q).all() or np.any(q < lower-1e-9) or np.any(q > upper+1e-9):
            raise ValueError('轨迹超出 URDF 关节限位')
        pin.forwardKinematics(ik.model, ik.data, ik.q_from_list(q))
        pin.updateFramePlacements(ik.model, ik.data)
        return ik.data.oMf[fid].copy(), ik.data.oMf[held_fid].copy()

    start, held = fk(seed)
    raw_target = validate_transform(preview['goals'][stage])
    goal = pin.SE3(raw_target[:3, :3], raw_target[:3, 3])
    dist = float(np.linalg.norm(goal.translation-start.translation))
    angle = float(np.linalg.norm(pin.log3(start.rotation.T @ goal.rotation)))
    if dist > MAX_CARTESIAN_SEGMENT_M or angle > MAX_ROTATION_SEGMENT_RAD:
        mode = '仿真' if request.get('execution_mode') == 'simulation' else '真机'
        raise ValueError(f'{mode}单段距离 {dist:.3f}m（上限 {MAX_CARTESIAN_SEGMENT_M:.2f}m），'
                         f'旋转 {np.degrees(angle):.1f}°（上限 {np.degrees(MAX_ROTATION_SEGMENT_RAD):.0f}°）')
    deadline = time.monotonic()+MAX_PREFLIGHT_SECONDS
    fixed = ([0, 1, *range(9, 16)] if side == 'left' else list(range(9))) if not allow_waist else []
    evaluations = 0
    refinements = 0
    step_limits = np.full(16, IK_ADJACENT_JUMP_LIMIT_RAD)
    step_limits[0] = .02  # Lift coordinate is metres, not radians.

    def interpolate(a, b, fraction):
        return pin.SE3(a.rotation @ pin.exp3(pin.log3(a.rotation.T @ b.rotation)*fraction),
                       a.translation*(1-fraction)+b.translation*fraction)

    def within(actual, desired):
        return (np.linalg.norm(actual.translation-desired.translation) <= position_tolerance
                and np.linalg.norm(pin.log3(desired.rotation.T @ actual.rotation)) <= rotation_tolerance)

    def solve(q, target, local=True):
        nonlocal evaluations
        evaluations += 1
        return ik.solve_coordinated(q, moving_side=side, target=target,
            allow_waist=allow_waist, held_target=held, pos_tol=min(.002,position_tolerance*.5), ori_tol=min(.02,rotation_tolerance*.5),
            max_iters=250 if local else 500, attempts=1 if local else 4,
            max_change=step_limits if local else None, deadline=deadline)

    def check_segment(q0, q1, pose0, pose1, cartesian):
        if np.any(np.abs(q1-q0) > step_limits+1e-8):
            raise ValueError('相邻关节变化过大')
        if fixed and np.max(np.abs(q1[fixed]-seed[fixed])) > 1e-8:
            raise ValueError('轨迹改动了固定关节')
        for fraction in np.linspace(0, 1, 7):
            actual, other = fk(q0*(1-fraction)+q1*fraction)
            if np.any(actual.translation < bounds[0]) or np.any(actual.translation > bounds[1]):
                raise ValueError('插值路径超出工作空间')
            if allow_waist and not within(other, held):
                raise ValueError('插值路径无法保持另一只手末端位姿')
            if cartesian and not within(actual, interpolate(pose0, pose1, fraction)):
                raise ValueError('关节插值偏离笛卡尔路径')

    def advance(q0, pose0, pose1, depth=0):
        nonlocal refinements
        if time.monotonic() > deadline:
            raise ValueError('IK 预检超过60秒计算预算')
        solved = solve(q0, pose1)
        reason = '局部有界联合 IK 未收敛'
        if solved is not None:
            q1 = np.asarray(solved)
            try:
                check_segment(q0, q1, pose0, pose1, True)
                return [q1]
            except ValueError as exc:
                reason = str(exc)
        if depth >= 5:
            info = ik.last_info
            raise ValueError(f'{reason}（细分5次；位置残差 {info.get("pos_err_m", 0)*1000:.2f}mm，'
                             f'旋转残差 {np.degrees(info.get("ori_err_rad", 0)):.2f}°）；未证明目标不可达')
        refinements += 1
        middle = interpolate(pose0, pose1, .5)
        left = advance(q0, pose0, middle, depth+1)
        right = advance(left[-1], middle, pose1, depth+1)
        return left+right

    def cartesian_path():
        steps = max(2, int(np.ceil(max(dist/.005, angle/.03))))
        rows = [seed.copy()]
        previous_pose = start
        for n, fraction in enumerate(np.linspace(0, 1, steps+1)[1:], 1):
            target = interpolate(start, goal, fraction)
            try:
                rows.extend(advance(rows[-1], previous_pose, target))
            except ValueError as exc:
                raise ValueError(f'路径 {n/steps:.0%}：{exc}') from exc
            previous_pose = target
        return rows

    # Coupled waist motion needs jointly solved waypoints to preserve the
    # other TCP throughout; single-arm pregrasp can use joint interpolation.
    path_kind = 'cartesian_joint_ik'
    if stage == 'pregrasp' and not allow_waist:
        endpoint = solve(seed, goal, local=False)
        if endpoint is None:
            raise ValueError('预抓取有界 IK 未收敛；已尝试多个初值，未证明目标不可达')
        endpoint = np.asarray(endpoint)
        steps = max(2, int(np.ceil(np.max(np.abs(endpoint-seed)/.03))))
        rows = list(np.linspace(seed, endpoint, steps+1))
        for a, b in zip(rows, rows[1:]):
            check_segment(a, b, fk(a)[0], fk(b)[0], False)
        path_kind = 'joint_space_pregrasp'
    else:
        try:
            rows = cartesian_path()
        except ValueError:
            if stage != 'pregrasp' or time.monotonic() > deadline:
                raise
            # Try an alternative transit, using a bounded terminal posture as
            # a guide and re-solving both TCP tasks at every waypoint.
            endpoint = solve(seed, goal, local=False)
            if endpoint is None:
                raise ValueError('预抓取联合 IK 未找到合法终点；未证明目标不可达')
            endpoint = np.asarray(endpoint)
            steps = max(2, int(np.ceil(np.max(np.abs(endpoint-seed)/.03))))
            rows = [seed.copy()]
            previous_pose = start
            for fraction in np.linspace(0, 1, steps+1)[1:]:
                target = fk(seed*(1-fraction)+endpoint*fraction)[0]
                rows.extend(advance(rows[-1], previous_pose, target))
                previous_pose = target
            path_kind = 'joint_guided_pregrasp'
    final, other = fk(rows[-1])
    if not within(final, goal) or (allow_waist and not within(other, held)):
        raise ValueError('终点位姿未达到验收精度')
    rows = np.asarray(rows).tolist()
    from eai_app.domain.motion.trajectory import parameterize,limits_from_profile,sample
    motion_limits=limits_from_profile(profile)
    motion_limits['velocity']*=float(precision.get('velocity_scale',1.))
    motion_limits['velocity']=np.minimum(motion_limits['velocity'],ik.model.velocityLimit[indices])
    timing=parameterize(rows,motion_limits)
    timed_plan=dict(timing,joint_waypoints=rows)
    # Validate the actual polynomial, not just the original IK knots.
    def validate_timing():
        for i,(t0,t1) in enumerate(zip(timing['times_s'],timing['times_s'][1:])):
            pose0,pose1=fk(np.asarray(rows[i]))[0],fk(np.asarray(rows[i+1]))[0]
            for t in np.linspace(t0,t1,11):
                q=sample(timed_plan,float(t));actual,other=fk(q)
                if np.any(actual.translation<bounds[0]) or np.any(actual.translation>bounds[1]):
                    raise ValueError('平滑轨迹超出工作空间')
                if allow_waist and not within(other,held):raise ValueError('平滑轨迹无法保持另一只手')
                if path_kind=='cartesian_joint_ik' and not within(actual,interpolate(pose0,pose1,(t-t0)/(t1-t0))):
                    raise ValueError('平滑轨迹偏离笛卡尔路径')
    try:validate_timing()
    except ValueError:
        timing=parameterize(rows,motion_limits,blend=False);timed_plan=dict(timing,joint_waypoints=rows)
        validate_timing()
    collision_info={'collision_checked':False}
    if request.get('collision_required'):
        from eai_app.planning.scene import CollisionScene,scene_digest
        scene=request.get('scene')
        if not scene:raise ValueError('缺少碰撞场景，禁止空场景回退')
        if request.get('execution_mode')=='hardware' and (not scene.get('calibrated') or not scene.get('world')):
            raise ValueError('真机场景待标定或环境几何为空')
        checker=CollisionScene(profile['urdf_path'],scene,hands=request.get('hands'),attachments=request.get('attachments',[]),stage=stage)
        samples=[sample(timed_plan,float(t)).tolist() for t in np.linspace(0,timing['times_s'][-1],
            max(2,int(np.ceil(timing['times_s'][-1]/.05))+1))]
        try:count=checker.validate_path(samples)
        except ValueError:
            if stage!='pregrasp' or allow_waist:raise
            from eai_app.planning.rrt import plan_transit
            def valid(q):
                actual,_=fk(q)
                if np.any(actual.translation<bounds[0]) or np.any(actual.translation>bounds[1]):
                    raise ValueError('绕行超出工作空间')
                checker.check(q)
            rows=plan_transit(rows[0],rows[-1],lower,upper,
                range(2,9) if side=='left' else range(9,16),valid)
            timing=parameterize(rows,motion_limits,blend=False)
            count=checker.validate_path(rows)
            path_kind='collision_aware_joint_transit'
        collision_info={'collision_checked':True,'collision_method':'sampled_full_geometry',
            'collision_samples':count,'scene_sha256':scene_digest(scene),'geometry_sha256':checker.geometry_hash,
            'scene_file_sha256':request.get('scene_file_sha256'),'scene_path':request.get('scene_path'),
            'hands':request['hands'],'attachments':request.get('attachments',[])}
    cartesian_speed=request.get('cartesian_speed_m_s')
    if cartesian_speed is not None:
        cartesian_speed=float(cartesian_speed)
        if not np.isfinite(cartesian_speed) or not 0<cartesian_speed<=.3:raise ValueError('工具速度上限需在0~0.3m/s')
        offset=np.asarray(request.get('tool_offset_link_m',[0,0,0]),float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('工具偏移无效')
        columns=[ik.model.joints[ik.model.getJointId(name)].idx_v for name in _JOINT_NAMES]
        maximum=0.
        for i,c in enumerate(np.asarray(timing['trajectory_coefficients'])):
            duration=timing['times_s'][i+1]-timing['times_s'][i]
            for u in np.linspace(0,1,31):
                q=np.polynomial.polynomial.polyval(u,c)
                dq=np.polynomial.polynomial.polyval(u,np.arange(1,6)[:,None]*c[1:])/duration
                J=pin.computeFrameJacobian(ik.model,ik.data,ik.q_from_list(q),fid,pin.LOCAL)[:,columns]
                velocity=J[:3]@dq+np.cross(J[3:]@dq,offset)
                maximum=max(maximum,float(np.linalg.norm(velocity)))
        scale=max(1.,maximum/cartesian_speed*1.05)
        timing['times_s']=(np.asarray(timing['times_s'])*scale).tolist()
    times = timing['times_s']
    if times[-1] > 180:
        raise ValueError('单段轨迹超过180秒，请缩短移动距离')
    return dict(ok=True, preview_id=preview['id'], stage=stage, arm_side=side,
                backend='host_ik_manual', ik_checked=True, joint_limits_checked=True,
                **collision_info, path_kind=path_kind, solver='bounded_dual_tcp',
                waist_included=allow_waist, other_tcp_held=allow_waist, held_side=held_side,
                held_target=held.homogeneous.tolist(), planned_at=time.time(), target=raw_target.tolist(),
                joint_waypoints=rows, **timing, ik_calls=evaluations, path_refinements=refinements,
                assembly_file_sha256=request.get('assembly_file_sha256'),
                profile_sha256=request.get('profile_sha256'),
                operation=preview.get('operation','grasp'), position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
                cartesian_speed_limit_m_s=cartesian_speed,
                urdf_sha256=hashlib.sha256(Path(profile['urdf_path']).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        request=json.loads(Path(args.request).read_text())
        if request.get('mode')=='rank_candidates':
            from grasp_candidates import rank
            result=rank(request)
        else:result = prepare(request)
    except Exception as exc:
        result = {'ok': False, 'error': str(exc)}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
