"""adapters.robot.geometry — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def lookup_transform_matrix(
    tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer],
    target_frame: str,
    source_frame: str,
    timeout_s: float = 0.15,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
    tgt = normalize_frame_id(target_frame)
    src = normalize_frame_id(source_frame)
    if not tgt or not src:
        return None
    if tgt == src:
        return _eai_platform_bootstrap.np.eye(4, dtype=_eai_platform_bootstrap.np.float64)
    if tf_buffer is None:
        return None
    try:
        tf_msg = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            _eai_platform_bootstrap.Time(),
            timeout=_eai_platform_bootstrap.Duration(seconds=timeout_s),
        )
        return transform_matrix_from_tf(tf_msg).astype(_eai_platform_bootstrap.np.float64)
    except Exception:
        return None


def transform_pose_xyz_R(
    xyz: _eai_platform_bootstrap.Sequence[float],
    rotation: _eai_platform_bootstrap.np.ndarray,
    *,
    source_frame: str,
    target_frame: str,
    tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer],
    timeout_s: float = 0.15,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]]:
    """把 (xyz, R) 从 source_frame 变到 target_frame。"""
    T = lookup_transform_matrix(
        tf_buffer, target_frame, source_frame, timeout_s=timeout_s
    )
    if T is None:
        return None
    quat = _eai_perception_pose_estimation.rotation_matrix_to_quaternion(_eai_platform_bootstrap.np.asarray(rotation, dtype=_eai_platform_bootstrap.np.float64))
    T_src = pose_to_matrix(xyz, quat)
    T_tgt = T @ T_src
    out_xyz, out_quat = matrix_to_pose(T_tgt)
    out_R = quat_to_rotation_matrix(*out_quat)
    return _eai_platform_bootstrap.np.asarray(out_xyz, dtype=_eai_platform_bootstrap.np.float32), out_R.astype(_eai_platform_bootstrap.np.float32)


def draw_mv6d_hud_banner(
    image: _eai_platform_bootstrap.np.ndarray,
    *,
    title: str,
    xyz: _eai_platform_bootstrap.Sequence[float],
    rpy_rad: _eai_platform_bootstrap.Sequence[float],
    err_t_mm: _eai_platform_bootstrap.Optional[float] = None,
    err_r_deg: _eai_platform_bootstrap.Optional[float] = None,
    method: str = "",
) -> None:
    """在图像顶部画 6D 数字条（估计 + 可选误差）。"""
    h, w = image.shape[:2]
    panel_h = min(92, max(64, h // 7))
    overlay = image.copy()
    _eai_platform_bootstrap.cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 22), -1)
    _eai_platform_bootstrap.cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    x, y, z = [float(v) for v in xyz]
    r, p, yw = [_eai_platform_bootstrap.math.degrees(float(v)) for v in rpy_rad]
    lines = [
        f"{title}  {method}".strip(),
        f"XYZ ({x:.3f}, {y:.3f}, {z:.3f}) m",
        f"RPY ({r:.1f}, {p:.1f}, {yw:.1f}) deg",
    ]
    if err_t_mm is not None and err_r_deg is not None:
        lines.append(f"pose-ref delta |t|={err_t_mm:.1f}mm |R|={err_r_deg:.1f}deg")
    font = _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45 if w >= 640 else 0.38
    y0 = 18
    for i, line in enumerate(lines):
        _eai_platform_bootstrap.cv2.putText(
            image,
            line,
            (10, y0 + i * 18),
            font,
            scale,
            (0, 255, 200) if i < 3 else (0, 180, 255),
            1,
            _eai_platform_bootstrap.cv2.LINE_AA,
        )


def render_mv6d_view_bgr(
    color_bgr: _eai_platform_bootstrap.np.ndarray,
    *,
    mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    pose_xyz: _eai_platform_bootstrap.Sequence[float],
    pose_R: _eai_platform_bootstrap.np.ndarray,
    obb_corners_cam: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    intrinsics: _eai_platform_bootstrap.Tuple[float, float, float, float],
    axis_len: float,
    title: str,
    method: str = "",
    err_t_mm: _eai_platform_bootstrap.Optional[float] = None,
    err_r_deg: _eai_platform_bootstrap.Optional[float] = None,
    centroid_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
) -> _eai_platform_bootstrap.np.ndarray:
    """合成带坐标轴 / OBB / 数字条的可视化图（用于仿真预览与面板）。"""
    rpy = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(_eai_platform_bootstrap.np.asarray(pose_R, dtype=_eai_platform_bootstrap.np.float64))
    display = apply_segment_overlay(
        color_bgr,
        mask,
        centroid_uv,
        obb_corners=obb_corners_cam,
        intrinsics=intrinsics,
        contact_uv=centroid_uv,
        seed_uv=centroid_uv,
        pose_position=_eai_platform_bootstrap.np.asarray(pose_xyz, dtype=_eai_platform_bootstrap.np.float32),
        pose_rotation=_eai_platform_bootstrap.np.asarray(pose_R, dtype=_eai_platform_bootstrap.np.float32),
        pose_axis_len=float(axis_len),
    )
    draw_mv6d_hud_banner(
        display,
        title=title,
        xyz=pose_xyz,
        rpy_rad=rpy,
        err_t_mm=err_t_mm,
        err_r_deg=err_r_deg,
        method=method,
    )
    return display


def format_multiview_6d_text(
    snap: _eai_perception_pose_estimation.MultiView6DPoseSnapshot,
    gt_xyz: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[float]] = None,
    gt_R: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    gt_enabled: bool = False,
) -> str:
    x, y, z = snap.position_xyz
    r, p, yw = snap.euler_rpy_rad
    qx, qy, qz, qw = snap.quaternion_xyzw
    lines = [
        f"坐标系: {snap.frame_id or 'camera'}  |  方法: {snap.method}"
        f"  |  TF={'OK' if snap.tf_ok else '无/降级'}",
        f"估计 XYZ=({x:.4f}, {y:.4f}, {z:.4f}) m",
        f"估计 RPY=({_eai_platform_bootstrap.math.degrees(r):.2f}°, {_eai_platform_bootstrap.math.degrees(p):.2f}°, {_eai_platform_bootstrap.math.degrees(yw):.2f}°)",
        f"估计 quat xyzw=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})",
        f"点云: 头={snap.head_points} 腕={snap.wrist_points} 融合={snap.fused_points}"
        f"  OBB=({snap.obb_extents[0]:.3f}, {snap.obb_extents[1]:.3f}, {snap.obb_extents[2]:.3f}) m",
    ]
    if snap.head_topic or snap.wrist_topic:
        used = []
        if getattr(snap, "left_wrist_result", None) is not None:
            used.append("左腕")
        if getattr(snap, "right_wrist_result", None) is not None:
            used.append("右腕")
        wrist_used = "+".join(used) if used else "无（均不可见/跳过）"
        lines.append(
            f"头: {snap.head_topic or '-'}  |  参与融合腕: {wrist_used}"
            f"  |  主腕: {snap.wrist_topic or '-'}"
        )
    if gt_enabled and gt_xyz is not None and gt_R is not None and normalize_frame_id(snap.frame_id) == _eai_configuration_policies.BASE_LINK_FRAME:
        gx, gy, gz = [float(v) for v in gt_xyz]
        gr, gp, gyaw = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(_eai_platform_bootstrap.np.asarray(gt_R, dtype=_eai_platform_bootstrap.np.float64))
        lines.append(f"参考 XYZ=({gx:.4f}, {gy:.4f}, {gz:.4f}) m")
        lines.append(
            f"参考 RPY=({_eai_platform_bootstrap.math.degrees(gr):.2f}°, {_eai_platform_bootstrap.math.degrees(gp):.2f}°, "
            f"{_eai_platform_bootstrap.math.degrees(gyaw):.2f}°)"
        )
        t_mm, r_deg, d_mm = _eai_perception_pose_estimation.pose_error_metrics(snap.position_xyz, snap.rotation_matrix, gt_xyz, gt_R)
        lines.append(
            f"位姿参考差值（非 SAM3 分割误差） |Δt|={t_mm:.1f} mm  (Δx={d_mm[0]:+.1f}, Δy={d_mm[1]:+.1f}, Δz={d_mm[2]:+.1f})  "
            f"|ΔR|={r_deg:.2f}°"
        )
    elif gt_enabled:
        lines.append("参考比较不可用：请确认参考已设置且估计位于 base_link")
    else:
        lines.append("参考位姿对比: 未启用（选择当前目标后确认参考值）")
    if snap.head_result is not None:
        debug_dir = getattr(snap.head_result, "diagnostics_dir", "")
        if debug_dir:
            lines.append(f"SAM3 原始二维掩码 / 深度过滤掩码: {debug_dir}")
    if snap.notes:
        lines.append(snap.notes)
    return "\n".join(lines)


def project_points_to_uv(
    points_3d: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> _eai_platform_bootstrap.np.ndarray:
    z = _eai_platform_bootstrap.np.maximum(points_3d[:, 2], 1e-3)
    u = fx * points_3d[:, 0] / z + cx
    v = fy * points_3d[:, 1] / z + cy
    return _eai_platform_bootstrap.np.stack([u, v], axis=-1)


def draw_obb_on_image(
    image: _eai_platform_bootstrap.np.ndarray,
    corners_3d: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color: _eai_platform_bootstrap.Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 2,
) -> None:
    if len(corners_3d) == 0:
        return
    uv = project_points_to_uv(corners_3d, fx, fy, cx, cy).astype(_eai_platform_bootstrap.np.int32)
    h, w = image.shape[:2]
    for a, b in _eai_perception_pose_estimation.OBB_EDGE_PAIRS:
        if corners_3d[a, 2] <= 0.01 or corners_3d[b, 2] <= 0.01:
            continue
        pa = (int(_eai_platform_bootstrap.np.clip(uv[a, 0], 0, w - 1)), int(_eai_platform_bootstrap.np.clip(uv[a, 1], 0, h - 1)))
        pb = (int(_eai_platform_bootstrap.np.clip(uv[b, 0], 0, w - 1)), int(_eai_platform_bootstrap.np.clip(uv[b, 1], 0, h - 1)))
        _eai_platform_bootstrap.cv2.line(image, pa, pb, color, thickness, _eai_platform_bootstrap.cv2.LINE_AA)


def draw_pose_axes_on_image(
    image: _eai_platform_bootstrap.np.ndarray,
    position_xyz: _eai_platform_bootstrap.np.ndarray,
    rotation: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    axis_len: float,
) -> None:
    """在图像上画 RGB = XYZ 坐标轴。"""
    if float(position_xyz[2]) <= 0.01:
        return
    origin = _eai_platform_bootstrap.np.asarray(position_xyz, dtype=_eai_platform_bootstrap.np.float32).reshape(3)
    R = _eai_platform_bootstrap.np.asarray(rotation, dtype=_eai_platform_bootstrap.np.float32).reshape(3, 3)
    # BGR: X红 Y绿 Z蓝
    axis_bgr = ((0, 0, 255), (0, 220, 0), (255, 120, 0))
    labels = ("X", "Y", "Z")
    h, w = image.shape[:2]
    pts = [origin] + [origin + R[:, i] * axis_len for i in range(3)]
    uv = project_points_to_uv(_eai_platform_bootstrap.np.stack(pts, axis=0), fx, fy, cx, cy).astype(_eai_platform_bootstrap.np.int32)
    o = (int(_eai_platform_bootstrap.np.clip(uv[0, 0], 0, w - 1)), int(_eai_platform_bootstrap.np.clip(uv[0, 1], 0, h - 1)))
    _eai_platform_bootstrap.cv2.circle(image, o, 5, (255, 255, 255), -1, _eai_platform_bootstrap.cv2.LINE_AA)
    for i in range(3):
        if pts[i + 1][2] <= 0.01:
            continue
        p = (int(_eai_platform_bootstrap.np.clip(uv[i + 1, 0], 0, w - 1)), int(_eai_platform_bootstrap.np.clip(uv[i + 1, 1], 0, h - 1)))
        _eai_platform_bootstrap.cv2.arrowedLine(image, o, p, axis_bgr[i], 3, _eai_platform_bootstrap.cv2.LINE_AA, tipLength=0.2)
        _eai_platform_bootstrap.cv2.putText(
            image,
            labels[i],
            (p[0] + 4, p[1] - 4),
            _eai_platform_bootstrap.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            axis_bgr[i],
            2,
            _eai_platform_bootstrap.cv2.LINE_AA,
        )


def apply_segment_overlay(
    image: _eai_platform_bootstrap.np.ndarray,
    mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    centroid_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
    obb_corners: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None,
    contact_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
    seed_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
    pose_position: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    pose_rotation: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    pose_axis_len: _eai_platform_bootstrap.Optional[float] = None,
) -> _eai_platform_bootstrap.np.ndarray:
    if image.ndim == 2:
        display = _eai_platform_bootstrap.cv2.cvtColor(image, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    mask_bool: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    if mask is not None:
        if mask.shape[:2] != display.shape[:2]:
            mask = _eai_perception_image_geometry.resize_mask_to_shape(mask, display.shape[:2])
        mask_bool = _eai_platform_bootstrap.np.asarray(mask).astype(bool)

    if mask_bool is not None and mask_bool.any():
        ys, xs = _eai_platform_bootstrap.np.where(mask_bool)
        pad = 12
        h, w = display.shape[:2]
        v0 = max(0, int(ys.min()) - pad)
        v1 = min(h, int(ys.max()) + pad + 1)
        u0 = max(0, int(xs.min()) - pad)
        u1 = min(w, int(xs.max()) + pad + 1)

        roi = display[v0:v1, u0:u1]
        roi_mask = mask_bool[v0:v1, u0:u1]
        tint = _eai_platform_bootstrap.np.array([40, 220, 80], dtype=_eai_platform_bootstrap.np.float32)
        blended = roi.astype(_eai_platform_bootstrap.np.float32)
        blended[roi_mask] = blended[roi_mask] * 0.42 + tint * 0.58
        display[v0:v1, u0:u1] = _eai_platform_bootstrap.np.clip(blended, 0, 255).astype(_eai_platform_bootstrap.np.uint8)

        mask_u8 = (mask_bool.astype(_eai_platform_bootstrap.np.uint8) * 255)
        contours, _ = _eai_platform_bootstrap.cv2.findContours(mask_u8, _eai_platform_bootstrap.cv2.RETR_EXTERNAL, _eai_platform_bootstrap.cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            _eai_platform_bootstrap.cv2.drawContours(display, contours, -1, (0, 255, 255), 2, _eai_platform_bootstrap.cv2.LINE_AA)

    if obb_corners is not None and intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        draw_obb_on_image(display, obb_corners, fx, fy, cx, cy, color=(0, 220, 255), thickness=2)

    if (
        pose_position is not None
        and pose_rotation is not None
        and intrinsics is not None
    ):
        fx, fy, cx, cy = intrinsics
        axis_len = float(pose_axis_len) if pose_axis_len is not None else 0.05
        if axis_len <= 1e-4:
            axis_len = 0.05
        draw_pose_axes_on_image(
            display,
            _eai_platform_bootstrap.np.asarray(pose_position, dtype=_eai_platform_bootstrap.np.float32),
            _eai_platform_bootstrap.np.asarray(pose_rotation, dtype=_eai_platform_bootstrap.np.float32),
            fx,
            fy,
            cx,
            cy,
            axis_len,
        )

    if centroid_uv is not None:
        cu, cv_pt = int(round(centroid_uv[0])), int(round(centroid_uv[1]))
        _eai_platform_bootstrap.cv2.drawMarker(
            display, (cu, cv_pt), (0, 255, 255), _eai_platform_bootstrap.cv2.MARKER_CROSS, 16, 2, _eai_platform_bootstrap.cv2.LINE_AA
        )
        _eai_platform_bootstrap.cv2.circle(display, (cu, cv_pt), 6, (0, 255, 255), 1, _eai_platform_bootstrap.cv2.LINE_AA)

    if contact_uv is not None:
        tu, tv = int(round(contact_uv[0])), int(round(contact_uv[1]))
        _eai_platform_bootstrap.cv2.drawMarker(
            display, (tu, tv), (0, 128, 255), _eai_platform_bootstrap.cv2.MARKER_TILTED_CROSS, 18, 2, _eai_platform_bootstrap.cv2.LINE_AA
        )
        _eai_platform_bootstrap.cv2.circle(display, (tu, tv), 8, (0, 128, 255), 2, _eai_platform_bootstrap.cv2.LINE_AA)

    if seed_uv is not None:
        su, sv = int(round(seed_uv[0])), int(round(seed_uv[1]))
        _eai_platform_bootstrap.cv2.drawMarker(
            display, (su, sv), (0, 80, 255), _eai_platform_bootstrap.cv2.MARKER_STAR, 18, 2, _eai_platform_bootstrap.cv2.LINE_AA
        )
        _eai_platform_bootstrap.cv2.circle(display, (su, sv), 10, (0, 80, 255), 2, _eai_platform_bootstrap.cv2.LINE_AA)
    return display


def cv2_to_qpixmap(
    cv_image: _eai_platform_bootstrap.np.ndarray,
    *,
    max_edge: _eai_platform_bootstrap.Optional[int] = None,
) -> _eai_platform_bootstrap.QPixmap:
    display = _eai_platform_bootstrap.np.asarray(cv_image)
    if len(display.shape) == 2:
        display = _eai_platform_bootstrap.cv2.normalize(display, None, 0, 255, _eai_platform_bootstrap.cv2.NORM_MINMAX, dtype=_eai_platform_bootstrap.cv2.CV_8U)
        display = _eai_platform_bootstrap.cv2.applyColorMap(display, _eai_platform_bootstrap.cv2.COLORMAP_JET)

    if max_edge is None:
        max_edge = _eai_configuration_policies.UI_PREVIEW_MAX_EDGE
    if max_edge and max_edge > 0:
        h0, w0 = display.shape[:2]
        edge = max(h0, w0)
        if edge > max_edge:
            scale = float(max_edge) / float(edge)
            display = _eai_platform_bootstrap.cv2.resize(
                display,
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
            )

    if display.ndim == 2:
        rgb = _eai_platform_bootstrap.cv2.cvtColor(display, _eai_platform_bootstrap.cv2.COLOR_GRAY2RGB)
    else:
        rgb = _eai_platform_bootstrap.cv2.cvtColor(display, _eai_platform_bootstrap.cv2.COLOR_BGR2RGB)
    # 必须 contiguous，且 QImage 不拥有 buffer → copy
    rgb = _eai_platform_bootstrap.np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = _eai_platform_bootstrap.QImage(rgb.data, w, h, ch * w, _eai_platform_bootstrap.QImage.Format_RGB888)
    return _eai_platform_bootstrap.QPixmap.fromImage(qimg.copy())


@_eai_platform_bootstrap.dataclass
class TcpPoseSnapshot:
    frame_id: str
    xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float]
    valid: bool = True


@_eai_platform_bootstrap.dataclass
class RobotStateSnapshot:
    left_arm: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]
    right_arm: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]
    left_hand: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]
    right_hand: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]
    waist: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]
    left_tcp: _eai_platform_bootstrap.Optional[TcpPoseSnapshot]
    right_tcp: _eai_platform_bootstrap.Optional[TcpPoseSnapshot]
    updated_at: float = 0.0


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> _eai_platform_bootstrap.np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return _eai_platform_bootstrap.np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=_eai_platform_bootstrap.np.float64,
    )


def pose_to_matrix(
    xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
) -> _eai_platform_bootstrap.np.ndarray:
    qx, qy, qz, qw = quat_xyzw
    mat = _eai_platform_bootstrap.np.eye(4, dtype=_eai_platform_bootstrap.np.float64)
    mat[:3, :3] = quat_to_rotation_matrix(qx, qy, qz, qw)
    mat[:3, 3] = xyz
    return mat


def matrix_to_pose(mat: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]:
    xyz = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    R = mat[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = _eai_platform_bootstrap.math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = _eai_platform_bootstrap.math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-8
    return xyz, (qx / norm, qy / norm, qz / norm, qw / norm)


def transform_matrix_from_tf(transform) -> _eai_platform_bootstrap.np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    return pose_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def transform_tcp_to_camera(
    tcp: _eai_platform_bootstrap.Optional[TcpPoseSnapshot],
    camera_frame: str,
    tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer],
    timeout_s: float = _eai_configuration_policies.UI_TF_LOOKUP_TIMEOUT_S,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]]:
    if tcp is None or not tcp.valid or not camera_frame or tf_buffer is None:
        return None
    if normalize_frame_id(tcp.frame_id) == normalize_frame_id(camera_frame):
        R = quat_to_rotation_matrix(*tcp.quat_xyzw)
        return _eai_platform_bootstrap.np.array(tcp.xyz, dtype=_eai_platform_bootstrap.np.float32), R.astype(_eai_platform_bootstrap.np.float32)
    try:
        tf_msg = tf_buffer.lookup_transform(
            camera_frame,
            tcp.frame_id,
            _eai_platform_bootstrap.Time(),
            timeout=_eai_platform_bootstrap.Duration(seconds=timeout_s),
        )
        T_cam_src = transform_matrix_from_tf(tf_msg)
        T_src_tcp = pose_to_matrix(tcp.xyz, tcp.quat_xyzw)
        T_cam_tcp = T_cam_src @ T_src_tcp
        xyz, quat = matrix_to_pose(T_cam_tcp)
        R = quat_to_rotation_matrix(*quat)
        return _eai_platform_bootstrap.np.array(xyz, dtype=_eai_platform_bootstrap.np.float32), R.astype(_eai_platform_bootstrap.np.float32)
    except Exception:
        return None


def quaternion_slerp(
    q0: _eai_platform_bootstrap.Tuple[float, float, float, float],
    q1: _eai_platform_bootstrap.Tuple[float, float, float, float],
    t: float,
) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    qa = _eai_platform_bootstrap.np.array(q0, dtype=_eai_platform_bootstrap.np.float64)
    qb = _eai_platform_bootstrap.np.array(q1, dtype=_eai_platform_bootstrap.np.float64)
    qa /= _eai_platform_bootstrap.np.linalg.norm(qa) + 1e-12
    qb /= _eai_platform_bootstrap.np.linalg.norm(qb) + 1e-12
    dot = float(_eai_platform_bootstrap.np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        out = qa + t * (qb - qa)
        out /= _eai_platform_bootstrap.np.linalg.norm(out) + 1e-12
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    theta0 = _eai_platform_bootstrap.math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = _eai_platform_bootstrap.math.sin(theta0)
    theta = theta0 * t
    s0 = _eai_platform_bootstrap.math.sin(theta0 - theta) / sin_theta0
    s1 = _eai_platform_bootstrap.math.sin(theta) / sin_theta0
    out = s0 * qa + s1 * qb
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def normalize_frame_id(frame: str) -> str:
    return (frame or "").strip().strip("/")


def transform_pose_to_base(
    xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
    source_frame: str,
    tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer],
    base_frame: str = _eai_configuration_policies.BASE_LINK_FRAME,
    timeout_s: float = _eai_configuration_policies.MOVE_TF_LOOKUP_TIMEOUT_S,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
    src = normalize_frame_id(source_frame)
    base = normalize_frame_id(base_frame)
    if not src:
        return None
    if src == base:
        return xyz, quat_xyzw
    if tf_buffer is None:
        return None
    try:
        tf_msg = tf_buffer.lookup_transform(
            base_frame,
            source_frame,
            _eai_platform_bootstrap.Time(),
            timeout=_eai_platform_bootstrap.Duration(seconds=timeout_s),
        )
        T_base_src = transform_matrix_from_tf(tf_msg)
        T_src_obj = pose_to_matrix(xyz, quat_xyzw)
        T_base_obj = T_base_src @ T_src_obj
        return matrix_to_pose(T_base_obj)
    except Exception:
        return None


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_interpolated_poses(
    start_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    start_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    goal_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    goal_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    steps: int,
) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]]:
    if steps < 1:
        return [(goal_xyz, goal_quat)]
    trajectory: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]] = []
    for i in range(1, steps + 1):
        t = _smoothstep01(i / steps)
        xyz = tuple(start_xyz[j] + t * (goal_xyz[j] - start_xyz[j]) for j in range(3))
        quat = quaternion_slerp(start_quat, goal_quat, t)
        trajectory.append((xyz, quat))
    return trajectory


def build_interpolated_joints(
    start_joints: _eai_platform_bootstrap.List[float],
    goal_joints: _eai_platform_bootstrap.List[float],
    steps: int,
    *,
    linear: bool = True,
) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.List[float]]:
    if len(start_joints) < 16 or len(goal_joints) < 16:
        return [list(goal_joints[:16])]
    if steps < 1:
        return [list(goal_joints[:16])]
    trajectory: _eai_platform_bootstrap.List[_eai_platform_bootstrap.List[float]] = []
    for i in range(1, steps + 1):
        t = (i / steps) if linear else _smoothstep01(i / steps)
        trajectory.append(
            [
                start_joints[j] + t * (goal_joints[j] - start_joints[j])
                for j in range(16)
            ]
        )
    return trajectory


def slider_to_arm_move_speed(slider_val: int) -> float:
    pct = (max(10, min(100, int(slider_val))) - 10) / 90.0
    return _eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S + pct * (
        _eai_configuration_chat.ARM_MOVE_SPEED_MAX_RAD_S - _eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S
    )


def arm_move_speed_to_slider(speed: float) -> int:
    pct = (float(speed) - _eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S) / (_eai_configuration_chat.ARM_MOVE_SPEED_MAX_RAD_S - _eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S)
    return max(10, min(100, int(round(10 + 90*pct))))


def format_arm_move_speed_label(slider_val: int) -> str:
    speed = slider_to_arm_move_speed(slider_val)
    return f"{speed:.2f} rad/s"


def compute_arm_move_duration_s(
    start_joints: _eai_platform_bootstrap.List[float],
    goal_joints: _eai_platform_bootstrap.List[float],
    joint_speed_rad_s: float = _eai_configuration_chat.ARM_MOVE_SPEED_DEFAULT_RAD_S,
) -> float:
    if len(start_joints) < 16 or len(goal_joints) < 16:
        return _eai_configuration_chat.ARM_MOVE_MIN_DURATION_S
    max_delta = max(abs(goal_joints[i] - start_joints[i]) for i in range(16))
    if max_delta < 1e-4:
        return 0.0
    speed = max(_eai_configuration_chat.ARM_MOVE_SPEED_MIN_RAD_S, min(_eai_configuration_chat.ARM_MOVE_SPEED_MAX_RAD_S, float(joint_speed_rad_s)))
    # Linear interpolation: do not truncate long durations and exceed the limit.
    # The first coordinate is the lift (metres), capped at 0.5 m/s in WBC config.
    duration = max(max_delta / speed, abs(goal_joints[0]-start_joints[0]) / .5)
    return max(_eai_configuration_chat.ARM_MOVE_MIN_DURATION_S, duration)


def _joint_dict_from_msg(msg: _eai_platform_bootstrap.JointState) -> _eai_platform_bootstrap.Dict[str, float]:
    result: _eai_platform_bootstrap.Dict[str, float] = {}
    for i, name in enumerate(msg.name):
        if i < len(msg.position):
            result[name] = float(msg.position[i])
    return result


def _split_arm_joints(joints: _eai_platform_bootstrap.Dict[str, float]) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]], _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]], _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]]:
    waist: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]] = []
    left: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]] = []
    right: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]] = []
    for name in _eai_perception_image_geometry.WAIST_JOINT_NAMES:
        if name in joints:
            waist.append((name, joints[name]))
    for name in _eai_perception_image_geometry.LEFT_ARM_JOINT_NAMES:
        if name in joints:
            left.append((name, joints[name]))
    for name in _eai_perception_image_geometry.RIGHT_ARM_JOINT_NAMES:
        if name in joints:
            right.append((name, joints[name]))
    return left, right, waist


def _arm_from_raw_positions(names: _eai_platform_bootstrap.List[str], positions: _eai_platform_bootstrap.List[float]) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]], _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]]]:
    if names:
        left, right, _ = _split_arm_joints(dict(zip(names, positions)))
        return left, right
    if len(positions) >= 14:
        left = list(zip(_eai_perception_image_geometry.LEFT_ARM_JOINT_NAMES, positions[:7]))
        right = list(zip(_eai_perception_image_geometry.RIGHT_ARM_JOINT_NAMES, positions[7:14]))
        return left, right
    return [], []


def _pose_from_msg(msg: _eai_platform_bootstrap.PoseStamped) -> TcpPoseSnapshot:
    p = msg.pose.position
    q = msg.pose.orientation
    return TcpPoseSnapshot(
        frame_id=msg.header.frame_id or "base_link",
        xyz=(float(p.x), float(p.y), float(p.z)),
        quat_xyzw=(float(q.x), float(q.y), float(q.z), float(q.w)),
        valid=True,
    )


def _tcp_pose_priority(snapshot: _eai_platform_bootstrap.Optional[TcpPoseSnapshot]) -> int:
    if snapshot is None or not snapshot.valid:
        return -1
    frame = normalize_frame_id(snapshot.frame_id)
    if frame == normalize_frame_id(_eai_configuration_policies.IK_TARGET_FRAME):
        return 2
    if frame in _eai_configuration_policies.MINK_FK_FRAME_ALIASES:
        return 1
    return 0


def _should_accept_tcp_update(
    current: _eai_platform_bootstrap.Optional[TcpPoseSnapshot],
    incoming: TcpPoseSnapshot,
) -> bool:
    if current is None or not current.valid:
        return True
    return _tcp_pose_priority(incoming) >= _tcp_pose_priority(current)


def _format_joint_group(title: str, joints: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]], unit: str = "rad") -> str:
    if not joints:
        return f"{title}: (无数据)\n"
    lines = [f"{title}:"]
    for name, val in joints:
        if unit == "rad":
            lines.append(f"  {name}: {val:.3f} rad ({_eai_platform_bootstrap.math.degrees(val):.1f}°)")
        elif unit == "mixed":
            if "lift" in name:
                lines.append(f"  {name}: {val:.3f} m")
            else:
                lines.append(f"  {name}: {val:.3f} rad ({_eai_platform_bootstrap.math.degrees(val):.1f}°)")
        elif unit == "norm":
            lines.append(f"  {name}: {val:.3f}")
        else:
            lines.append(f"  {name}: {val:.3f} {unit}")
    return "\n".join(lines) + "\n"


def _html_span(text: str, color: str, bold: bool = False) -> str:
    weight = "font-weight:bold;" if bold else ""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return f'<span style="color:{color};{weight}">{safe}</span>'


def _format_joint_group_html(
    title: str,
    joints: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, float]],
    title_color: str,
    value_color: str,
    unit: str = "rad",
) -> str:
    if not joints:
        return _html_span(f"{title}: (无数据)", _eai_configuration_policies.UI_TEXT_MUTED) + "<br/>"
    lines = [_html_span(f"{title}:", title_color, bold=True)]
    for name, val in joints:
        if unit == "rad":
            detail = f"  {name}: {val:.3f} rad ({_eai_platform_bootstrap.math.degrees(val):.1f}°)"
        elif unit == "mixed":
            if "lift" in name:
                detail = f"  {name}: {val:.3f} m"
            else:
                detail = f"  {name}: {val:.3f} rad ({_eai_platform_bootstrap.math.degrees(val):.1f}°)"
        elif unit == "norm":
            detail = f"  {name}: {val:.3f}"
        else:
            detail = f"  {name}: {val:.3f} {unit}"
        lines.append(_html_span(detail, value_color))
    return "<br/>".join(lines) + "<br/>"


def format_robot_state_html(
    state: RobotStateSnapshot,
    camera_frame: str,
    left_cam: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    right_cam: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
) -> str:
    age = _eai_platform_bootstrap.time.time() - state.updated_at if state.updated_at > 0 else -1.0
    header = f"══ 机器人状态 ══  相机系: {camera_frame or '未知'}"
    if age >= 0:
        header += f"  ({age:.1f}s 前)"

    parts = [
        "<div style='font-family: Monospace, Consolas, monospace; font-size: 10pt; "
        "line-height: 1.45;'>",
        _html_span(header, "#f0f0f0", bold=True),
        "<br/><br/>",
    ]

    if left_cam is not None:
        tcp_text = (
            f"左臂 TCP (相机系): XYZ=({left_cam[0]:.3f}, {left_cam[1]:.3f}, {left_cam[2]:.3f}) m"
        )
        parts.append(_html_span(tcp_text, "#ff79c6", bold=True))
    elif state.left_tcp and state.left_tcp.valid:
        x, y, z = state.left_tcp.xyz
        tcp_text = (
            f"左臂 TCP ({state.left_tcp.frame_id}): "
            f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
        parts.append(_html_span(tcp_text, "#ff79c6", bold=True))
    else:
        parts.append(_html_span("左臂 TCP: (无数据)", _eai_configuration_policies.UI_TEXT_MUTED))

    parts.append("<br/>")

    if right_cam is not None:
        tcp_text = (
            f"右臂 TCP (相机系): XYZ=({right_cam[0]:.3f}, {right_cam[1]:.3f}, {right_cam[2]:.3f}) m"
        )
        parts.append(_html_span(tcp_text, "#79d8ff", bold=True))
    elif state.right_tcp and state.right_tcp.valid:
        x, y, z = state.right_tcp.xyz
        tcp_text = (
            f"右臂 TCP ({state.right_tcp.frame_id}): "
            f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
        parts.append(_html_span(tcp_text, "#79d8ff", bold=True))
    else:
        parts.append(_html_span("右臂 TCP: (无数据)", _eai_configuration_policies.UI_TEXT_MUTED))

    parts.append("<br/><br/>")
    parts.append(_format_joint_group_html("左臂关节", state.left_arm, "#ff79c6", "#ffd0ea"))
    parts.append(_format_joint_group_html("右臂关节", state.right_arm, "#79d8ff", "#ccefff"))
    if state.waist:
        parts.append(_format_joint_group_html("腰关节", state.waist, "#f1fa8c", "#fffacd", unit="mixed"))
    parts.append(_format_joint_group_html("左手关节", state.left_hand, "#ffb86c", "#ffe8cc", unit="norm"))
    parts.append(_format_joint_group_html("右手关节", state.right_hand, "#8be9fd", "#d4f7ff", unit="norm"))
    parts.append("</div>")
    return "".join(parts)


def format_robot_state_text(
    state: RobotStateSnapshot,
    camera_frame: str,
    left_cam: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    right_cam: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
) -> str:
    age = _eai_platform_bootstrap.time.time() - state.updated_at if state.updated_at > 0 else -1.0
    header = f"══ 机器人状态 ══  相机系: {camera_frame or '未知'}"
    if age >= 0:
        header += f"  ({age:.1f}s 前)"
    parts = [header, ""]

    if left_cam is not None:
        parts.append(
            f"左臂 TCP (相机系): XYZ=({left_cam[0]:.3f}, {left_cam[1]:.3f}, {left_cam[2]:.3f}) m"
        )
    elif state.left_tcp and state.left_tcp.valid:
        x, y, z = state.left_tcp.xyz
        parts.append(
            f"左臂 TCP ({state.left_tcp.frame_id}): XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
    else:
        parts.append("左臂 TCP: (无数据)")

    if right_cam is not None:
        parts.append(
            f"右臂 TCP (相机系): XYZ=({right_cam[0]:.3f}, {right_cam[1]:.3f}, {right_cam[2]:.3f}) m"
        )
    elif state.right_tcp and state.right_tcp.valid:
        x, y, z = state.right_tcp.xyz
        parts.append(
            f"右臂 TCP ({state.right_tcp.frame_id}): XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
    else:
        parts.append("右臂 TCP: (无数据)")

    parts.append("")
    parts.append(_format_joint_group("左臂关节", state.left_arm))
    parts.append(_format_joint_group("右臂关节", state.right_arm))
    if state.waist:
        parts.append(_format_joint_group("腰关节", state.waist, unit="mixed"))
    parts.append(_format_joint_group("左手关节", state.left_hand, unit="norm"))
    parts.append(_format_joint_group("右手关节", state.right_hand, unit="norm"))
    return "\n".join(parts).strip()


def make_tcp_axis_lines(center: _eai_platform_bootstrap.np.ndarray, rotation: _eai_platform_bootstrap.np.ndarray, scale: float = 0.08) -> _eai_platform_bootstrap.np.ndarray:
    origin = center.astype(_eai_platform_bootstrap.np.float32)
    lines = []
    for i in range(3):
        end = origin + rotation[:, i] * scale
        lines.extend([origin, end])
    return _eai_platform_bootstrap.np.stack(lines, axis=0).astype(_eai_platform_bootstrap.np.float32)


def hand_gap_cm_to_slider(cm: float) -> int:
    cm = max(0.0, min(_eai_configuration_chat.HAND_FINGER_GAP_MAX_CM, float(cm)))
    return int(round(cm / _eai_configuration_chat.HAND_FINGER_GAP_STEP_CM))


def slider_to_hand_gap_cm(slider: int) -> float:
    slider = max(0, min(_eai_configuration_chat.HAND_FINGER_GAP_SLIDER_MAX, int(slider)))
    return slider * _eai_configuration_chat.HAND_FINGER_GAP_STEP_CM


def hand_gap_cm_to_position(cm: float) -> float:
    """指间距离 cm → 关节 position：0cm=全合(1)，max=全张(0)。"""
    cm = max(0.0, min(_eai_configuration_chat.HAND_FINGER_GAP_MAX_CM, float(cm)))
    if _eai_configuration_chat.HAND_FINGER_GAP_MAX_CM <= 1e-9:
        return 1.0
    return 1.0 - (cm / _eai_configuration_chat.HAND_FINGER_GAP_MAX_CM)


def slider_to_hand_position(slider: int) -> float:
    """游标（0.1cm 步进）→ 关节 position。"""
    return hand_gap_cm_to_position(slider_to_hand_gap_cm(slider))


def format_hand_gap_label(slider: int) -> str:
    return f"{slider_to_hand_gap_cm(slider):.1f} cm"


format_hand_angle_label = format_hand_gap_label


def make_hand_command(
    side: str,
    position: float | _eai_platform_bootstrap.Sequence[float],
) -> _eai_platform_bootstrap.JointState:
    """构建睿研灵巧手控制命令。

    position: 标量则六指同开合；序列则按 HAND_JOINT_NAMES 顺序逐关节 (0=张, 1=合)。
    """
    side = "right" if side == "right" else "left"
    msg = _eai_platform_bootstrap.JointState()
    msg.header.frame_id = f"{side}_hand"
    names = _eai_configuration_chat.RIGHT_HAND_JOINT_NAMES if side == "right" else _eai_configuration_chat.LEFT_HAND_JOINT_NAMES
    msg.name = list(names)
    if isinstance(position, (int, float)):
        pos = max(0.0, min(1.0, float(position)))
        msg.position = [pos] * len(names)
    else:
        vals = [max(0.0, min(1.0, float(v))) for v in position]
        if len(vals) < len(names):
            vals = vals + [vals[-1] if vals else 0.0] * (len(names) - len(vals))
        msg.position = vals[: len(names)]
    msg.velocity = list(_eai_configuration_chat.HAND_CMD_VELOCITY)
    msg.effort = list(_eai_configuration_chat.HAND_CMD_EFFORT)
    return msg


def make_left_hand_command(position: float | _eai_platform_bootstrap.Sequence[float]) -> _eai_platform_bootstrap.JointState:
    return make_hand_command("left", position)


def make_right_hand_command(position: float | _eai_platform_bootstrap.Sequence[float]) -> _eai_platform_bootstrap.JointState:
    return make_hand_command("right", position)


_hand_cmd_relay_sock: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.socket.socket] = None


_hand_cmd_relay_warned = False


def _hand_cmd_relay_socket() -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.socket.socket]:
    global _hand_cmd_relay_sock, _hand_cmd_relay_warned
    if not _eai_configuration_policies.HAND_CMD_RELAY_ENABLED:
        return None
    if _hand_cmd_relay_sock is not None:
        return _hand_cmd_relay_sock
    try:
        sock = _eai_platform_bootstrap.socket.socket(_eai_platform_bootstrap.socket.AF_INET, _eai_platform_bootstrap.socket.SOCK_DGRAM)
        sock.setblocking(False)
        _hand_cmd_relay_sock = sock
        return sock
    except OSError as exc:
        if not _hand_cmd_relay_warned:
            _hand_cmd_relay_warned = True
            print(f"[hand_cmd_relay] UDP socket 创建失败: {exc}", file=_eai_platform_bootstrap.sys.stderr)
        return None


def _relay_udp_payload(payload: dict) -> None:
    """经 UDP 中继把 JSON 送进 Docker（与 ROS pub 并行；DDS 不通时仍可控制）。"""
    sock = _hand_cmd_relay_socket()
    if sock is None:
        return
    try:
        sock.sendto(
            (_eai_platform_bootstrap.json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"),
            (_eai_configuration_policies.HAND_CMD_RELAY_HOST, _eai_configuration_policies.HAND_CMD_RELAY_PORT),
        )
    except OSError:
        pass


def relay_hand_command(side: str, msg: _eai_platform_bootstrap.JointState) -> None:
    side = "right" if side == "right" else "left"
    _relay_udp_payload(
        {
            "type": "hand",
            "side": side,
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        }
    )


def relay_wbc_joints(msg: _eai_platform_bootstrap.JointState) -> None:
    _relay_udp_payload(
        {
            "type": "wbc_joints",
            "frame_id": str(msg.header.frame_id or ""),
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        }
    )


def relay_control_mode(mode: int) -> None:
    _relay_udp_payload({"type": "control_mode", "mode": int(mode)})


def relay_arm_enable(enable: bool = True) -> None:
    _relay_udp_payload({"type": "arm_enable", "data": bool(enable)})


def relay_hand_enable(enable: bool = True) -> None:
    _relay_udp_payload({"type": "hand_enable", "data": bool(enable)})


def relay_arm_reset_to_initial(data: bool = True) -> None:
    _relay_udp_payload({"type": "arm_reset", "data": bool(data)})

