"""Adapt the visual/demo 6D snapshot without changing its coordinate frame."""
import numpy as np
from click_pose import validate_transform


def observation_from_snapshot(snapshot, captured_at, observation_id, base_pose=None):
    """Create a grasp observation from a visual snapshot.

    ``base_pose`` is supplied by the UI only after it has resolved a real
    hand-eye/TF transform.  It prevents an untransformed camera-space pose
    from being mislabeled as a robot target.
    """
    frame_id = snapshot.frame_id
    rotation = np.asarray(snapshot.rotation_matrix, dtype=float)
    position = np.asarray(snapshot.position_xyz, dtype=float)
    transformed = False
    if base_pose is not None:
        position, rotation = base_pose[:2]
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        frame_id = "base_link"
        transformed = True
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    validate_transform(T)
    methods = [str(snapshot.method), str(snapshot.notes)]
    for name in ('head_result', 'left_wrist_result', 'right_wrist_result', 'wrist_result'):
        part = getattr(snapshot, name, None)
        if part is not None:
            methods.append(str(part.method))
    description = ' '.join(methods).lower()
    blockers = []
    if frame_id.lstrip('/') != 'base_link' or (not snapshot.tf_ok and not transformed):
        blockers.append('视觉估计尚未可靠转换到 base_link，请检查相机外参/TF')
    if any(token in description for token in ('rgb-only', 'simgt', 'lingbot', '假定', '平面深度')):
        blockers.append('当前结果含假定/推断深度或仿真真值修正，仅用于预览，不用于真机抓取')
    result = dict(ok=True, source='visual_demo_6d', source_label='手动控制 / 多视角6D位姿',
                  observation_id=observation_id, captured_at=float(captured_at),
                  camera_frame=frame_id, input_frame=snapshot.frame_id,
                  object_pose={'T_camera_object': T.tolist(), 'method': snapshot.method,
                               'pose_kind': 'multiview_estimate', 'extents_m': list(snapshot.obb_extents)},
                  execution_blockers=blockers, execution_ready=False,
                  notes=[snapshot.notes], grasps={})
    if frame_id.lstrip('/') == 'base_link' and (snapshot.tf_ok or transformed):
        # Legacy schema calls the input frame 'camera'; here it is already base.
        result.update(base_frame='base_link', T_base_camera=np.eye(4).tolist(), T_base_object=T.tolist())
    return result
