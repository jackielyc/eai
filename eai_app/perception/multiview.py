"""Multi-view pose computation with explicit geometry and inference callbacks. No Qt dependency."""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def compute_snapshot(
    services,
    *,
    dependencies,
    head_topic: str,
    head_bgr: Optional[np.ndarray],
    head_depth: Optional[np.ndarray],
    head_depth_topic: str,
    wrist_topic: str = "",
    wrist_bgr: Optional[np.ndarray] = None,
    wrist_depth: Optional[np.ndarray] = None,
    wrist_depth_topic: str = "",
    left_wrist_topic: str = "/camera/left_wrist_color",
    left_wrist_bgr: Optional[np.ndarray] = None,
    left_wrist_depth: Optional[np.ndarray] = None,
    left_wrist_depth_topic: str = "",
    right_wrist_topic: str = "/camera/right_wrist_color",
    right_wrist_bgr: Optional[np.ndarray] = None,
    right_wrist_depth: Optional[np.ndarray] = None,
    right_wrist_depth_topic: str = "",
    preferred_wrist_topic: str = "",
    seed_u: int,
    seed_v: int,
    tf_buffer,
    get_intrinsics,
    get_camera_frame,
    pose_backend: str = "pca",
    assumed_depth_m: float = 0.45,
) -> MultiView6DPoseSnapshot:
    BASE_LINK_FRAME = dependencies["BASE_LINK_FRAME"]
    MV6D_EST_GT_SNAP_MAX_DIST_M = dependencies["MV6D_EST_GT_SNAP_MAX_DIST_M"]
    MV6D_FUSE_WEIGHT_HEAD = dependencies["MV6D_FUSE_WEIGHT_HEAD"]
    MV6D_FUSE_WEIGHT_WRIST = dependencies["MV6D_FUSE_WEIGHT_WRIST"]
    MultiView6DPoseSnapshot = dependencies["MultiView6DPoseSnapshot"]
    Object6DPoseResult = dependencies["Object6DPoseResult"]
    POSE_BACKEND_ANY6D = dependencies["POSE_BACKEND_ANY6D"]
    POSE_BACKEND_FOUNDATIONPOSE = dependencies["POSE_BACKEND_FOUNDATIONPOSE"]
    POSE_BACKEND_PCA = dependencies["POSE_BACKEND_PCA"]
    estimate_6d_pose_pca = dependencies["estimate_6d_pose_pca"]
    lookup_transform_matrix = dependencies["lookup_transform_matrix"]
    obb_corners = dependencies["obb_corners"]
    quat_to_rotation_matrix = dependencies["quat_to_rotation_matrix"]
    refine_fused_object_points = dependencies["refine_fused_object_points"]
    rotation_matrix_to_euler_xyz = dependencies["rotation_matrix_to_euler_xyz"]
    rotation_matrix_to_quaternion = dependencies["rotation_matrix_to_quaternion"]
    transform_points_by_matrix = dependencies["transform_points_by_matrix"]
    head_result = None
    notes: List[str] = []
    assumed_depth_m = float(assumed_depth_m or 0.45)

    # 兼容旧单腕参数：未传左右腕时退化为 wrist_*
    if left_wrist_bgr is None and right_wrist_bgr is None and wrist_bgr is not None:
        if "right" in str(wrist_topic):
            right_wrist_topic = wrist_topic
            right_wrist_bgr = wrist_bgr
            right_wrist_depth = wrist_depth
            right_wrist_depth_topic = wrist_depth_topic
        else:
            left_wrist_topic = wrist_topic or left_wrist_topic
            left_wrist_bgr = wrist_bgr
            left_wrist_depth = wrist_depth
            left_wrist_depth_topic = wrist_depth_topic

    if head_bgr is not None:
        if head_depth is not None:
            dh, dw = head_depth.shape[:2]
            fx, fy, cx, cy = get_intrinsics(head_depth_topic or head_topic, dw, dh)
        else:
            ch, cw = head_bgr.shape[:2]
            fx, fy, cx, cy = get_intrinsics(head_topic, cw, ch)
            notes.append("头部无深度 → RGB-only（假定平面深度）")
        ch, cw = head_bgr.shape[:2]
        su = max(0, min(cw - 1, seed_u))
        sv = max(0, min(ch - 1, seed_v))
        try:
            head_result = services._estimate_view_6d(
                head_bgr,
                head_depth,
                su,
                sv,
                fx,
                fy,
                cx,
                cy,
                method_tag="head",
                assumed_depth_m=assumed_depth_m,
                pose_backend=pose_backend,
            )
        except Exception as exc:
            raise RuntimeError(f"头部估计失败：{exc}") from exc
        if head_result is None:
            raise RuntimeError("头部估计失败：没有 SAM3 分割结果")
    else:
        notes.append("缺少头部 RGB")

    # 头部中心 → base，供腕部一致性检查
    head_center_base: Optional[np.ndarray] = None
    if head_result is not None:
        try:
            hf = (
                get_camera_frame(head_topic, head_depth_topic)
                or get_camera_frame(head_depth_topic, head_topic)
                or ""
            )
        except Exception:
            hf = ""
        T_h = services._mv6d_sim_extrinsic_matrix(
            head_topic, head_depth_topic, hf
        )
        if T_h is None:
            T_h = services._mv6d_manual_extrinsic_matrix(head_topic, hf)
        if T_h is None and tf_buffer is not None and hf:
            T_h = lookup_transform_matrix(tf_buffer, BASE_LINK_FRAME, hf)
        if T_h is not None:
            head_center_base = transform_points_by_matrix(
                np.asarray([head_result.position_xyz], dtype=np.float32),
                T_h,
            )[0]

    # FP / Any6D：用头部学习位姿变换到 base，不再跑腕部（昂贵且 PCA 融合会冲掉学习结果）
    if (
        pose_backend in (POSE_BACKEND_FOUNDATIONPOSE, POSE_BACKEND_ANY6D)
        and head_result is not None
    ):
        notes.append(
            f"方法={pose_backend}，分割=SAM3：仅头部，跳过腕部融合"
        )
        frame_id = BASE_LINK_FRAME
        R_cam = np.asarray(head_result.rotation_matrix, dtype=np.float32)
        t_cam = np.asarray(head_result.position_xyz, dtype=np.float32)
        if T_h is not None:
            R_b = (T_h[:3, :3] @ R_cam).astype(np.float32)
            t_b = (
                T_h[:3, :3] @ t_cam + T_h[:3, 3]
            ).astype(np.float32)
            tf_ok = True
        else:
            R_b, t_b = R_cam, t_cam
            frame_id = hf or "camera"
            tf_ok = False
            notes.append("无外参/TF，位姿仍在相机系")
        half = np.array(
            [
                float(head_result.obb_extents[0]) * 0.5,
                float(head_result.obb_extents[1]) * 0.5,
                float(head_result.obb_extents[2]) * 0.5,
            ],
            dtype=np.float32,
        )
        corners = obb_corners(t_b, R_b, half)
        rpy = rotation_matrix_to_euler_xyz(R_b)
        quat = rotation_matrix_to_quaternion(R_b)
        method = f"{head_result.method}+mv6d-head-only"
        # 仿真 GT 精修（与融合路径一致）
        center_xyz = (float(t_b[0]), float(t_b[1]), float(t_b[2]))
        if tf_ok:
            try:
                xyz_gt, quat_gt, gt_note = services._mv6d_refine_with_sim_gt(
                    center_xyz,
                    max_dist_m=float(MV6D_EST_GT_SNAP_MAX_DIST_M),
                    snap=None,
                )
                if gt_note and xyz_gt is not None:
                    center_xyz = (
                        float(xyz_gt[0]),
                        float(xyz_gt[1]),
                        float(xyz_gt[2]),
                    )
                    t_b = np.asarray(center_xyz, dtype=np.float32)
                    corners = obb_corners(t_b, R_b, half)
                    method = f"{method}+simGT"
                    notes.append(gt_note)
                    if quat_gt is not None:
                        qx, qy, qz, qw = [float(v) for v in quat_gt[:4]]
                        quat = (qx, qy, qz, qw)
                        R_b = quat_to_rotation_matrix(qx, qy, qz, qw).astype(
                            np.float32
                        )
                        rpy = rotation_matrix_to_euler_xyz(R_b)
                        corners = obb_corners(t_b, R_b, half)
            except Exception:
                pass
        return MultiView6DPoseSnapshot(
            frame_id=frame_id,
            position_xyz=center_xyz,
            euler_rpy_rad=rpy,
            quaternion_xyzw=quat,
            rotation_matrix=np.asarray(R_b, dtype=np.float32),
            obb_extents=tuple(float(x) for x in head_result.obb_extents),
            obb_corners=np.asarray(corners, dtype=np.float32),
            head_topic=head_topic,
            wrist_topic="",
            head_points=int(head_result.point_count),
            wrist_points=0,
            fused_points=int(head_result.point_count),
            method=method,
            head_result=head_result,
            wrist_result=None,
            left_wrist_result=None,
            right_wrist_result=None,
            tf_ok=tf_ok,
            notes="; ".join(notes),
        )

    left_wrist_result, n_l = services._mv6d_try_wrist_view_if_visible(
        label="左腕",
        head_result=head_result,
        head_topic=head_topic,
        head_depth_topic=head_depth_topic,
        wrist_topic=left_wrist_topic,
        wrist_bgr=left_wrist_bgr,
        wrist_depth=left_wrist_depth,
        wrist_depth_topic=left_wrist_depth_topic
        or services._get_paired_depth_topic_name(left_wrist_topic)
        or "",
        tf_buffer=tf_buffer,
        get_intrinsics=get_intrinsics,
        get_camera_frame=get_camera_frame,
        assumed_depth_m=assumed_depth_m,
        head_center_base=head_center_base,
        pose_backend=pose_backend,
    )
    notes.extend(n_l)
    right_wrist_result, n_r = services._mv6d_try_wrist_view_if_visible(
        label="右腕",
        head_result=head_result,
        head_topic=head_topic,
        head_depth_topic=head_depth_topic,
        wrist_topic=right_wrist_topic,
        wrist_bgr=right_wrist_bgr,
        wrist_depth=right_wrist_depth,
        wrist_depth_topic=right_wrist_depth_topic
        or services._get_paired_depth_topic_name(right_wrist_topic)
        or "",
        tf_buffer=tf_buffer,
        get_intrinsics=get_intrinsics,
        get_camera_frame=get_camera_frame,
        assumed_depth_m=assumed_depth_m,
        head_center_base=head_center_base,
        pose_backend=pose_backend,
    )
    notes.extend(n_r)

    # 主腕：优先 combo 偏好且可见；否则任一可见腕
    pref = str(preferred_wrist_topic or wrist_topic or "")
    wrist_result: Optional[Object6DPoseResult] = None
    wrist_topic_used = ""
    wrist_depth_topic_used = ""
    if "right" in pref and right_wrist_result is not None:
        wrist_result = right_wrist_result
        wrist_topic_used = right_wrist_topic
        wrist_depth_topic_used = right_wrist_depth_topic
    elif "left" in pref and left_wrist_result is not None:
        wrist_result = left_wrist_result
        wrist_topic_used = left_wrist_topic
        wrist_depth_topic_used = left_wrist_depth_topic
    elif left_wrist_result is not None:
        wrist_result = left_wrist_result
        wrist_topic_used = left_wrist_topic
        wrist_depth_topic_used = left_wrist_depth_topic
    elif right_wrist_result is not None:
        wrist_result = right_wrist_result
        wrist_topic_used = right_wrist_topic
        wrist_depth_topic_used = right_wrist_depth_topic

    if head_result is None and left_wrist_result is None and right_wrist_result is None:
        raise RuntimeError("头/腕均未能估计出 6D 位姿（检查彩色图与提示点）")

    # 尝试变换到 base_link 后融合点云（头部权重大；腕部仅弱补充）
    fused_chunks: List[Tuple[np.ndarray, float]] = []
    tf_ok = False
    target_frame = BASE_LINK_FRAME

    def _view_points_in_target(
        result: Object6DPoseResult,
        color_topic: str,
        depth_topic: str,
    ) -> Optional[np.ndarray]:
        nonlocal tf_ok
        frame = get_camera_frame(color_topic, depth_topic) or get_camera_frame(
            depth_topic, color_topic
        )
        pts = np.asarray(result.points_3d, dtype=np.float32)
        if len(pts) == 0:
            return None
        # 优先 Isaac 外参（与 TCP 同系）；避免 A2D 真机 TF 把桌面点抬到 Z≈1.8
        T = services._mv6d_sim_extrinsic_matrix(color_topic, depth_topic, frame)
        if T is None:
            T = services._mv6d_manual_extrinsic_matrix(color_topic, frame)
        if T is None:
            if not frame:
                return None
            T = lookup_transform_matrix(tf_buffer, target_frame, frame)
        if T is None:
            return None
        tf_ok = True
        return transform_points_by_matrix(pts, T)

    if head_result is not None:
        pts = _view_points_in_target(head_result, head_topic, head_depth_topic)
        if pts is not None and len(pts) > 0:
            fused_chunks.append((pts, float(MV6D_FUSE_WEIGHT_HEAD)))
    if left_wrist_result is not None:
        pts = _view_points_in_target(
            left_wrist_result,
            left_wrist_topic,
            left_wrist_depth_topic
            or services._get_paired_depth_topic_name(left_wrist_topic)
            or "",
        )
        if pts is not None and len(pts) > 0:
            fused_chunks.append((pts, float(MV6D_FUSE_WEIGHT_WRIST)))
    if right_wrist_result is not None:
        pts = _view_points_in_target(
            right_wrist_result,
            right_wrist_topic,
            right_wrist_depth_topic
            or services._get_paired_depth_topic_name(right_wrist_topic)
            or "",
        )
        if pts is not None and len(pts) > 0:
            fused_chunks.append((pts, float(MV6D_FUSE_WEIGHT_WRIST)))

    if fused_chunks:
        weighted: List[np.ndarray] = []
        for pts, w in fused_chunks:
            n_rep = max(1, int(round(float(w))))
            for _ in range(n_rep):
                weighted.append(pts)
        all_pts = np.concatenate(weighted, axis=0)
        # 子采样避免腕部重复后过大
        if len(all_pts) > 20000:
            rng = np.random.default_rng(0)
            all_pts = all_pts[rng.choice(len(all_pts), 20000, replace=False)]
        seed_for_pca = head_center_base
        all_pts, clean_note = refine_fused_object_points(
            all_pts, seed_xyz=seed_for_pca
        )
        notes.append(f"点云清洗:{clean_note}")
        pose = estimate_6d_pose_pca(
            all_pts, align_gravity=True, seed_xyz=seed_for_pca
        )
        if pose is None:
            raise RuntimeError("融合点云 PCA 失败")
        center, rotation, half_ext, rpy, quat = pose
        frame_id = target_frame
        method = "sam3+pca-multiview-base+gravity+clean"
        corners = obb_corners(center, rotation, half_ext)
        extents = (
            float(half_ext[0] * 2),
            float(half_ext[1] * 2),
            float(half_ext[2] * 2),
        )
        fused_n = int(sum(len(p) for p, _ in fused_chunks))
    else:
        # 无 TF：以头部结果为主，腕部仅作备注
        primary = head_result or wrist_result
        assert primary is not None
        frame_id = (
            get_camera_frame(head_topic, head_depth_topic)
            if head_result is not None
            else get_camera_frame(wrist_topic_used, wrist_depth_topic_used)
        ) or "camera"
        center = np.asarray(primary.position_xyz, dtype=np.float32)
        rotation = np.asarray(primary.rotation_matrix, dtype=np.float32)
        rpy = primary.euler_rpy_rad
        quat = primary.quaternion_xyzw
        corners = primary.obb_corners
        extents = primary.obb_extents
        method = "sam3+pca-single-view"
        fused_n = int(primary.point_count)
        notes.append("无 TF→base_link，使用单视角相机系结果（未几何融合）")

    wrist_pts = 0
    if left_wrist_result is not None:
        wrist_pts += int(left_wrist_result.point_count)
    if right_wrist_result is not None:
        wrist_pts += int(right_wrist_result.point_count)

    # 仿真可用且估计已靠近某物体：位置吸到物体中心（姿态仍用视觉），显著减小平移误差
    center_xyz = (float(center[0]), float(center[1]), float(center[2]))
    if tf_ok and str(frame_id) == str(BASE_LINK_FRAME):
        try:
            xyz_gt, quat_gt, gt_note = services._mv6d_refine_with_sim_gt(
                center_xyz,
                max_dist_m=float(MV6D_EST_GT_SNAP_MAX_DIST_M),
                snap=None,
            )
            if gt_note and xyz_gt is not None:
                center_xyz = (
                    float(xyz_gt[0]),
                    float(xyz_gt[1]),
                    float(xyz_gt[2]),
                )
                center = np.asarray(center_xyz, dtype=np.float32)
                half_ext_arr = np.asarray(
                    [
                        float(extents[0]) * 0.5,
                        float(extents[1]) * 0.5,
                        float(extents[2]) * 0.5,
                    ],
                    dtype=np.float32,
                )
                corners = obb_corners(center, rotation, half_ext_arr)
                method = f"{method}+simGT"
                notes.append(gt_note)
                if quat_gt is not None:
                    # _mv6d_refine_with_sim_gt 已返回 xyzw
                    qx, qy, qz, qw = [float(v) for v in quat_gt[:4]]
                    quat = (qx, qy, qz, qw)
                    rotation = quat_to_rotation_matrix(qx, qy, qz, qw).astype(
                        np.float32
                    )
                    rpy = rotation_matrix_to_euler_xyz(rotation)
                    corners = obb_corners(center, rotation, half_ext_arr)
        except Exception:
            pass

    return MultiView6DPoseSnapshot(
        frame_id=frame_id,
        position_xyz=center_xyz,
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=np.asarray(rotation, dtype=np.float32),
        obb_extents=extents,
        obb_corners=np.asarray(corners, dtype=np.float32),
        head_topic=head_topic,
        wrist_topic=wrist_topic_used,
        head_points=int(head_result.point_count) if head_result else 0,
        wrist_points=int(wrist_pts),
        fused_points=int(fused_n),
        method=method,
        head_result=head_result,
        wrist_result=wrist_result,
        left_wrist_result=left_wrist_result,
        right_wrist_result=right_wrist_result,
        tf_ok=tf_ok,
        notes="; ".join(notes),
    )

