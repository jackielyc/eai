"""Prepare a bounded joint-space return to the captured startup arm state."""
import hashlib
from pathlib import Path
import time
import numpy as np
from grasp_execution import timed_waypoints


def prepare_restore(profile, feedback, startup, preview_id):
    from robot_view3d import _Kinematics, JOINT_NAMES
    kin = _Kinematics(profile['urdf_path'])
    digest = hashlib.sha256(Path(profile['urdf_path']).read_bytes()).hexdigest()
    if digest != feedback.get('urdf_sha256') or digest != startup.get('urdf_sha256'):
        raise ValueError('启动状态与当前反馈/模型不一致，不能恢复')
    start = np.asarray(feedback['q16'], dtype=float)
    end = np.asarray(startup['q16'], dtype=float).copy()
    if start.shape != (16,) or end.shape != (16,) or not np.isfinite([start,end]).all():
        raise ValueError('恢复需要完整有效的16关节状态')
    end[:2] = start[:2]  # Restore arms only; do not move the waist.
    idx = [kin._q_idx[name] for name in JOINT_NAMES]
    lo, hi = kin.model.lowerPositionLimit[idx], kin.model.upperPositionLimit[idx]
    if np.any(np.minimum(start,end) < lo) or np.any(np.maximum(start,end) > hi):
        raise ValueError('恢复路径超出URDF关节限位')
    rows = [start.tolist(), end.tolist()]
    times = timed_waypoints(rows, profile['joint_velocity_rad_s'], profile['joint_acceleration_rad_s2'])
    if times[-1] > 180:
        raise ValueError('恢复轨迹超过180秒，请先将手臂移动至初始位置附近')
    poses = kin.link_poses(end)
    targets = {side: poses[link].tolist() for side,link in (('left','Link7_l'),('right','Link7_r'))}
    return dict(ok=True, preview_id=preview_id, stage='restore', arm_side='both',
                backend='host_joint_restore', joint_limits_checked=True, collision_checked=False,
                planned_at=time.time(), target=targets['right'], targets=targets,
                joint_waypoints=rows, times_s=times, urdf_sha256=digest)
