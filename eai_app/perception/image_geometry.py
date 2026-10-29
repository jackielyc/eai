"""perception.image geometry — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def list_local_v4l_devices() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    """扫描 /dev/video*，返回 [(index, 简述), ...]（跳过无像素格式的元数据节点）。"""
    items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = []
    for i in range(16):
        path = f"/dev/video{i}"
        if not _eai_platform_bootstrap.os.path.exists(path):
            continue
        fourcc = ""
        kind = "采集"
        try:
            import subprocess as _sp

            out = _sp.check_output(
                ["v4l2-ctl", "-d", path, "--list-formats-ext"],
                text=True,
                stderr=_sp.STDOUT,
                timeout=2,
            )
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("[") and "]:" in s and "'" in s:
                    parts = s.split("'")
                    if len(parts) >= 2:
                        fourcc = parts[1].strip()
                        break
            if not fourcc:
                continue
            up = fourcc.upper()
            if up in ("YUYV", "YUY2", "MJPG", "JPEG", "RGB3", "BGR3", "NV12"):
                kind = "彩色"
            elif up.startswith("Z16"):
                kind = "深度"
            elif up in ("GREY", "Y8  ", "Y8", "Y16 ", "Y16", "BYR2"):
                kind = "红外/灰"
            label = f"{i}  {path}  {kind}  ({fourcc})"
        except Exception:
            label = f"{i}  {path}"
        items.append((str(i), label))
    return items


def suggest_local_webcam_device() -> str:
    """优先彩色节点（YUYV/MJPG），否则列表第一项，再否则 0。"""
    devices = list_local_v4l_devices()
    for idx, label in devices:
        if "彩色" in label:
            return idx
    if devices:
        return devices[0][0]
    return "0"


RIGHT_HAND_GAP_A_DEFAULT_CM = _eai_configuration_chat.HAND_FINGER_GAP_MAX_CM


RIGHT_HAND_GAP_B_DEFAULT_CM = 6.0


LEFT_ARM_JOINT_NAMES = [f"joint{i}_l" for i in range(1, 8)]


RIGHT_ARM_JOINT_NAMES = [f"joint{i}_r" for i in range(1, 8)]


WAIST_JOINT_NAMES = ["joint_lift_body", "joint_body_pitch"]


def is_depth_topic(topic: str) -> bool:
    name = topic.lower()
    return "depth" in name or name.endswith("_z")


def is_color_image_topic(topic: str, types: _eai_platform_bootstrap.List[str]) -> bool:
    """color 图像 topic（名称含 color）。"""
    if not any(t in _eai_configuration_perception.IMAGE_TYPES for t in types):
        return False
    return "color" in topic.lower()


def is_head_color_topic(topic: str) -> bool:
    """头部彩色相机（排除 hand_* / wrist 等）。"""
    name = topic.lower().rstrip("/")
    if is_depth_topic(name):
        return False
    if name in {t.lower() for t in _eai_configuration_policies.CAD_CAPTURE_TOPIC_CANDIDATES}:
        return True
    if "head" not in name:
        return False
    if "hand" in name or "wrist" in name or "fisheye" in name:
        return False
    return "color" in name or name.endswith("/rgb") or name.endswith("_rgb")


def resolve_head_color_topic(available: _eai_platform_bootstrap.List[str]) -> _eai_platform_bootstrap.Optional[str]:
    """在可用 topic 中优先解析头部彩色相机。"""
    lowered = {t.lower(): t for t in available}
    for pref in _eai_configuration_policies.CAD_CAPTURE_TOPIC_CANDIDATES:
        hit = lowered.get(pref.lower())
        if hit is not None:
            return hit
    heads = [t for t in available if is_head_color_topic(t)]
    if not heads:
        return None
    heads.sort(key=lambda t: (0 if "head_color" in t.lower() else 1, t))
    return heads[0]


def default_viewer_geometry() -> _eai_platform_bootstrap.Tuple[int, int, int, int]:
    """根据屏幕可用区域返回窗口 (x, y, width, height)。"""
    app = _eai_platform_bootstrap.QApplication.instance()
    if app is not None:
        screen = app.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = max(1480, int(avail.width() * 0.96))
            height = max(900, int(avail.height() * 0.92))
            x = avail.x() + max(0, (avail.width() - width) // 2)
            y = avail.y() + max(0, (avail.height() - height) // 2)
            return x, y, width, height
    return 80, 60, 1680, 960


def default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    fx = fy = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def intrinsics_from_camera_info(msg: _eai_platform_bootstrap.CameraInfo) -> tuple[float, float, float, float]:
    k = msg.k
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def depth_to_point_cloud(
    depth: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = _eai_configuration_perception.DEFAULT_MAX_DEPTH_M,
    stride: int = _eai_configuration_perception.DEPTH_STRIDE,
) -> tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]:
    depth_f = depth.astype(_eai_platform_bootstrap.np.float32)
    if depth.dtype == _eai_platform_bootstrap.np.uint16:
        depth_f /= 1000.0

    depth_sub = depth_f[::stride, ::stride]
    h, w = depth_sub.shape
    u_coords, v_coords = _eai_platform_bootstrap.np.meshgrid(
        _eai_platform_bootstrap.np.arange(0, w, dtype=_eai_platform_bootstrap.np.float32) * stride,
        _eai_platform_bootstrap.np.arange(0, h, dtype=_eai_platform_bootstrap.np.float32) * stride,
    )

    valid = _eai_platform_bootstrap.valid_depth_mask(depth_sub, _eai_platform_bootstrap.DEPTH_MIN_M, min(float(max_depth), _eai_platform_bootstrap.DEPTH_MAX_M))
    if not _eai_platform_bootstrap.np.any(valid):
        return _eai_platform_bootstrap.np.empty((0, 3), dtype=_eai_platform_bootstrap.np.float32), _eai_platform_bootstrap.np.empty((0, 4), dtype=_eai_platform_bootstrap.np.float32)

    z = depth_sub[valid]
    u = u_coords[valid]
    v = v_coords[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = _eai_platform_bootstrap.np.stack([x, y, z], axis=-1).astype(_eai_platform_bootstrap.np.float32)

    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    jet = _eai_platform_bootstrap.cv2.applyColorMap((z_norm * 255).astype(_eai_platform_bootstrap.np.uint8), _eai_platform_bootstrap.cv2.COLORMAP_JET)
    rgb = jet.reshape(-1, 3)[:, ::-1].astype(_eai_platform_bootstrap.np.float32) / 255.0
    alpha = _eai_platform_bootstrap.np.ones((len(points), 1), dtype=_eai_platform_bootstrap.np.float32)
    colors = _eai_platform_bootstrap.np.hstack([rgb, alpha])
    return points, colors


def subsample_points_with_colors(
    points: _eai_platform_bootstrap.np.ndarray,
    colors: _eai_platform_bootstrap.np.ndarray,
    max_points: int,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = _eai_platform_bootstrap.np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx], colors[idx]


def build_depth_preview_image(
    depth: _eai_platform_bootstrap.np.ndarray,
    max_preview_w: int = _eai_configuration_perception.DEPTH_PREVIEW_MAX_W,
) -> _eai_platform_bootstrap.np.ndarray:
    preview = depth
    if preview.ndim == 2:
        preview_vis = _eai_platform_bootstrap.cv2.normalize(preview, None, 0, 255, _eai_platform_bootstrap.cv2.NORM_MINMAX, dtype=_eai_platform_bootstrap.cv2.CV_8U)
        preview_vis = _eai_platform_bootstrap.cv2.applyColorMap(preview_vis, _eai_platform_bootstrap.cv2.COLORMAP_JET)
    else:
        preview_vis = preview.copy()
    ph, pw = preview_vis.shape[:2]
    if pw > max_preview_w:
        scale = max_preview_w / pw
        preview_vis = _eai_platform_bootstrap.cv2.resize(
            preview_vis,
            (int(pw * scale), int(ph * scale)),
            interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
        )
    return preview_vis


@_eai_platform_bootstrap.dataclass
class DepthVizResult:
    preview_vis: _eai_platform_bootstrap.np.ndarray
    preview_shape: _eai_platform_bootstrap.Tuple[int, int]
    preview_intrinsics: _eai_platform_bootstrap.Tuple[float, float, float, float]
    points: _eai_platform_bootstrap.np.ndarray
    colors: _eai_platform_bootstrap.np.ndarray
    point_count: int
    full_shape: _eai_platform_bootstrap.Tuple[int, int]


def build_depth_viz_data(
    depth: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> DepthVizResult:
    h, w = depth.shape[:2]
    preview_vis = build_depth_preview_image(depth)
    preview_shape = preview_vis.shape[:2]
    pfx = fx * (preview_shape[1] / w)
    pfy = fy * (preview_shape[0] / h)
    pcx = cx * (preview_shape[1] / w)
    pcy = cy * (preview_shape[0] / h)
    points, colors = depth_to_point_cloud(
        depth, fx, fy, cx, cy, stride=_eai_configuration_perception.DEPTH_DISPLAY_STRIDE
    )
    points, colors = subsample_points_with_colors(points, colors, _eai_configuration_perception.MAX_GL_DEPTH_POINTS)
    return DepthVizResult(
        preview_vis=preview_vis,
        preview_shape=preview_shape,
        preview_intrinsics=(pfx, pfy, pcx, pcy),
        points=points,
        colors=colors,
        point_count=len(points),
        full_shape=(h, w),
    )


def camera_info_candidates(depth_topic: str) -> _eai_platform_bootstrap.List[str]:
    base = depth_topic.replace("_depth", "")
    return [
        f"{depth_topic}/camera_info",
        f"{base}_color/camera_info",
        f"{base}/camera_info",
    ]


def depth_value_meters(depth: _eai_platform_bootstrap.np.ndarray, u: int, v: int) -> float:
    value = float(depth[v, u])
    if depth.dtype == _eai_platform_bootstrap.np.uint16:
        return value / 1000.0
    return value


def pixel_to_camera_xyz(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> _eai_platform_bootstrap.Tuple[float, float, float]:
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return x, y, depth_m


def format_color_pixel_info(u: int, v: int, image: _eai_platform_bootstrap.np.ndarray) -> str:
    h, w = image.shape[:2]
    if u < 0 or v < 0 or u >= w or v >= h:
        return ""
    if image.ndim == 2:
        val = float(image[v, u])
        return f"像素 ({u}, {v})  |  gray={val:.1f}"
    b, g, r = (int(c) for c in image[v, u][:3])
    return f"像素 ({u}, {v})  |  BGR=({b}, {g}, {r})"


def format_depth_pixel_info(
    u: int,
    v: int,
    depth: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> str:
    h, w = depth.shape[:2]
    if u < 0 or v < 0 or u >= w or v >= h:
        return ""
    depth_m = depth_value_meters(depth, u, v)
    if not _eai_platform_bootstrap.np.isfinite(depth_m) or depth_m <= 0.01:
        return f"像素 ({u}, {v})  |  无效深度"
    x, y, z = pixel_to_camera_xyz(u, v, depth_m, fx, fy, cx, cy)
    return (
        f"像素 ({u}, {v})  |  depth={depth_m:.3f} m  |  "
        f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m"
    )


def depth_to_meters_array(depth: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
    depth_m = depth.astype(_eai_platform_bootstrap.np.float32)
    if depth.dtype == _eai_platform_bootstrap.np.uint16:
        depth_m /= 1000.0
    return depth_m


def scale_uv_to_shape(
    u: int, v: int, src_shape: _eai_platform_bootstrap.Tuple[int, int], dst_shape: _eai_platform_bootstrap.Tuple[int, int]
) -> _eai_platform_bootstrap.Tuple[int, int]:
    sh, sw = src_shape
    dh, dw = dst_shape
    if (sh, sw) == (dh, dw):
        return u, v
    u_d = int(round(u * dw / max(sw, 1)))
    v_d = int(round(v * dh / max(sh, 1)))
    return max(0, min(dw - 1, u_d)), max(0, min(dh - 1, v_d))


def resize_mask_to_shape(mask: _eai_platform_bootstrap.np.ndarray, shape: _eai_platform_bootstrap.Tuple[int, int]) -> _eai_platform_bootstrap.np.ndarray:
    h, w = shape
    if mask.shape[:2] == (h, w):
        return mask
    resized = _eai_platform_bootstrap.cv2.resize(
        mask.astype(_eai_platform_bootstrap.np.uint8), (w, h), interpolation=_eai_platform_bootstrap.cv2.INTER_NEAREST
    )
    return resized > 0


def find_paired_depth_topic(color_topic: str, available: _eai_platform_bootstrap.List[str]) -> _eai_platform_bootstrap.Optional[str]:
    depth_topics = [t for t in available if is_depth_topic(t)]
    candidates = [
        color_topic.replace("_color", "_depth"),
        color_topic.replace("_color", "_depth_z"),
        color_topic.replace("color", "depth"),
    ]
    for c in candidates:
        if c in depth_topics:
            return c
    stem = color_topic.replace("_color", "").replace("color", "").rstrip("_")
    for t in depth_topics:
        if stem and stem in t:
            return t
    # 尚无 depth 帧入缓存时，仍返回命名约定（便于自动勾选 / 订阅）
    for c in candidates:
        if c != color_topic and is_depth_topic(c):
            return c
    if is_head_color_topic(color_topic):
        return _eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT
    return None


def find_paired_color_topic(depth_topic: str, available: _eai_platform_bootstrap.List[str]) -> _eai_platform_bootstrap.Optional[str]:
    color_topics = [t for t in available if not is_depth_topic(t)]
    if not color_topics:
        return None
    candidates = [
        depth_topic.replace("_depth", "_color"),
        depth_topic.replace("_depth_z", "_color"),
        depth_topic.replace("depth", "color"),
    ]
    for c in candidates:
        if c in color_topics:
            return c
    stem = depth_topic.replace("_depth", "").replace("_depth_z", "").replace("depth", "").rstrip("_")
    for t in color_topics:
        if stem and stem in t and "color" in t.lower():
            return t
    return None


def segment_by_color_floodfill(
    seed_u: int,
    seed_v: int,
    color_bgr: _eai_platform_bootstrap.np.ndarray,
    lo_diff: int = 22,
    hi_diff: int = 22,
) -> _eai_platform_bootstrap.np.ndarray:
    h, w = color_bgr.shape[:2]
    work = color_bgr.copy()
    flood_mask = _eai_platform_bootstrap.np.zeros((h + 2, w + 2), dtype=_eai_platform_bootstrap.np.uint8)
    flags = 4 | _eai_platform_bootstrap.cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    _eai_platform_bootstrap.cv2.floodFill(
        work,
        flood_mask,
        (seed_u, seed_v),
        (0, 0, 0),
        (lo_diff, lo_diff, lo_diff),
        (hi_diff, hi_diff, hi_diff),
        flags,
    )
    return flood_mask[1:-1, 1:-1] > 0


def _connected_component_at_seed(valid: _eai_platform_bootstrap.np.ndarray, seed_u: int, seed_v: int) -> _eai_platform_bootstrap.np.ndarray:
    mask_u8 = valid.astype(_eai_platform_bootstrap.np.uint8)
    num_labels, labels = _eai_platform_bootstrap.cv2.connectedComponents(mask_u8)
    seed_label = labels[seed_v, seed_u]
    if seed_label == 0:
        return _eai_platform_bootstrap.np.zeros(valid.shape, dtype=bool)
    return labels == seed_label


def crop_roi(
    u: int, v: int, height: int, width: int, radius: int = _eai_configuration_perception.SEGMENT_ROI_RADIUS
) -> _eai_platform_bootstrap.Tuple[int, int, int, int]:
    u0 = max(0, u - radius)
    u1 = min(width, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(height, v + radius + 1)
    return u0, v0, u1, v1


def subsample_mask(mask: _eai_platform_bootstrap.np.ndarray, max_points: int = _eai_configuration_perception.SEGMENT_MAX_3D_POINTS) -> _eai_platform_bootstrap.np.ndarray:
    count = int(mask.sum())
    if count <= max_points:
        return mask
    ys, xs = _eai_platform_bootstrap.np.where(mask)
    rng = _eai_platform_bootstrap.np.random.default_rng(0)
    pick = rng.choice(count, max_points, replace=False)
    slim = _eai_platform_bootstrap.np.zeros_like(mask)
    slim[ys[pick], xs[pick]] = True
    return slim


def subsample_points(points: _eai_platform_bootstrap.np.ndarray, max_points: int = _eai_configuration_policies.MAX_GL_SEGMENT_POINTS) -> _eai_platform_bootstrap.np.ndarray:
    if len(points) <= max_points:
        return points
    rng = _eai_platform_bootstrap.np.random.default_rng(1)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def downsample_image_for_compute(
    image: _eai_platform_bootstrap.np.ndarray,
    max_dim: int = _eai_configuration_perception.SEGMENT_COMPUTE_MAX_DIM,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 0.999:
        return image, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = _eai_platform_bootstrap.cv2.INTER_NEAREST if image.ndim == 2 else _eai_platform_bootstrap.cv2.INTER_AREA
    return _eai_platform_bootstrap.cv2.resize(image, (new_w, new_h), interpolation=interp), scale


def scale_intrinsics_by(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    if scale >= 0.999:
        return fx, fy, cx, cy
    return fx * scale, fy * scale, cx * scale, cy * scale


def compute_contact_point_3d(
    seed_u: int,
    seed_v: int,
    depth: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    points_3d: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    patch_radius: int = 2,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float]]:
    """点击像素处的 3D 接触点（相机系），作为左臂 TCP 目标位置。"""
    depth_m = depth_to_meters_array(depth)
    h, w = depth_m.shape
    su = max(0, min(w - 1, int(seed_u)))
    sv = max(0, min(h - 1, int(seed_v)))

    u0 = max(0, su - patch_radius)
    u1 = min(w, su + patch_radius + 1)
    v0 = max(0, sv - patch_radius)
    v1 = min(h, sv + patch_radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[(patch > 0.01) & _eai_platform_bootstrap.np.isfinite(patch)]
    if len(valid) >= 1:
        z = float(_eai_platform_bootstrap.np.median(valid))
        x = (su - cx) * z / fx
        y = (sv - cy) * z / fy
        return (float(x), float(y), float(z))

    if points_3d is not None and len(points_3d) > 0:
        zs = _eai_platform_bootstrap.np.maximum(points_3d[:, 2], 1e-3)
        us = fx * points_3d[:, 0] / zs + cx
        vs = fy * points_3d[:, 1] / zs + cy
        dist = (us - su) ** 2 + (vs - sv) ** 2
        nearest = points_3d[int(_eai_platform_bootstrap.np.argmin(dist))]
        return (float(nearest[0]), float(nearest[1]), float(nearest[2]))
    return None


def upscale_segment_result(result: _eai_perception_pose_estimation.Object6DPoseResult, scale: float) -> _eai_perception_pose_estimation.Object6DPoseResult:
    if scale >= 0.999:
        return result
    inv = 1.0 / scale
    full_h = max(1, int(round(result.mask.shape[0] * inv)))
    full_w = max(1, int(round(result.mask.shape[1] * inv)))
    result.mask = resize_mask_to_shape(result.mask, (full_h, full_w))
    cu, cv = result.centroid_uv
    result.centroid_uv = (cu * inv, cv * inv)
    cu2, cv2 = result.contact_uv
    result.contact_uv = (cu2 * inv, cv2 * inv)
    return result


def build_pose_overlay_image(
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    result: _eai_perception_pose_estimation.Object6DPoseResult,
    intrinsics: _eai_platform_bootstrap.Tuple[float, float, float, float],
    max_dim: int = _eai_configuration_perception.SEGMENT_OVERLAY_MAX_DIM,
) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
    if color_bgr is None:
        return None
    display, scale = downsample_image_for_compute(color_bgr, max_dim)
    fx, fy, cx, cy = scale_intrinsics_by(*intrinsics, scale)
    mask = resize_mask_to_shape(result.mask, display.shape[:2])
    cu = result.centroid_uv[0] * scale
    cv = result.centroid_uv[1] * scale
    contact_u = result.contact_uv[0] * scale
    contact_v = result.contact_uv[1] * scale
    return _eai_adapters_robot_geometry.apply_segment_overlay(
        display,
        mask,
        (cu, cv),
        obb_corners=result.obb_corners,
        intrinsics=(fx, fy, cx, cy),
        contact_uv=(contact_u, contact_v),
        pose_position=_eai_platform_bootstrap.np.asarray(result.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
        pose_rotation=_eai_platform_bootstrap.np.asarray(result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
        pose_axis_len=float(max(result.obb_extents) * 0.55),
    )


def _segment_object_in_crop(
    seed_u: int,
    seed_v: int,
    depth_m: _eai_platform_bootstrap.np.ndarray,
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
    depth_tol: float,
    color_tol: float,
    min_area: int,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    h, w = depth_m.shape[:2]
    seed_u = max(0, min(w - 1, seed_u))
    seed_v = max(0, min(h - 1, seed_v))
    seed_d = float(depth_m[seed_v, seed_u])

    if not _eai_platform_bootstrap.np.isfinite(seed_d) or seed_d <= 0.01:
        if color_bgr is not None:
            mask = segment_by_color_floodfill(seed_u, seed_v, color_bgr)
            return mask, "color"
        return _eai_platform_bootstrap.np.zeros((h, w), dtype=bool), "none"

    depth_diff = _eai_platform_bootstrap.np.abs(depth_m - seed_d)
    valid = (depth_m > 0.01) & _eai_platform_bootstrap.np.isfinite(depth_m) & (depth_diff <= depth_tol)
    method = "depth"

    if color_bgr is not None:
        lab = _eai_platform_bootstrap.cv2.cvtColor(color_bgr, _eai_platform_bootstrap.cv2.COLOR_BGR2LAB).astype(_eai_platform_bootstrap.np.float32)
        seed_lab = lab[seed_v, seed_u]
        color_diff = _eai_platform_bootstrap.np.linalg.norm(lab - seed_lab, axis=2)
        valid = valid & (color_diff <= color_tol)
        method = "depth+color"

    mask = _connected_component_at_seed(valid, seed_u, seed_v)
    if not mask.any():
        if color_bgr is not None:
            mask = segment_by_color_floodfill(seed_u, seed_v, color_bgr)
            return mask, "color"
        return mask, method

    kernel = _eai_platform_bootstrap.cv2.getStructuringElement(_eai_platform_bootstrap.cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = (mask.astype(_eai_platform_bootstrap.np.uint8) * 255)
    mask_u8 = _eai_platform_bootstrap.cv2.morphologyEx(mask_u8, _eai_platform_bootstrap.cv2.MORPH_OPEN, kernel)
    mask_u8 = _eai_platform_bootstrap.cv2.morphologyEx(mask_u8, _eai_platform_bootstrap.cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0

    if int(mask.sum()) < min_area:
        valid_tight = (depth_m > 0.01) & _eai_platform_bootstrap.np.isfinite(depth_m) & (depth_diff <= depth_tol * 0.6)
        if color_bgr is not None:
            valid_tight = valid_tight & (color_diff <= color_tol * 0.75)
        mask = _connected_component_at_seed(valid_tight, seed_u, seed_v)

    return mask, method


def segment_object_at_click(
    seed_u: int,
    seed_v: int,
    depth: _eai_platform_bootstrap.np.ndarray,
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    depth_tol: float = _eai_configuration_perception.SEGMENT_DEPTH_TOL_M,
    color_tol: float = _eai_configuration_perception.SEGMENT_COLOR_TOL,
    min_area: int = _eai_configuration_perception.SEGMENT_MIN_AREA,
    roi_radius: int = _eai_configuration_perception.SEGMENT_ROI_RADIUS,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    """基于深度连续性 + 可选颜色约束分割点击处物体，返回 (mask, method)。"""
    h, w = depth.shape[:2]
    seed_u = max(0, min(w - 1, seed_u))
    seed_v = max(0, min(h - 1, seed_v))
    u0, v0, u1, v1 = crop_roi(seed_u, seed_v, h, w, roi_radius)

    depth_crop = depth[v0:v1, u0:u1]
    depth_m = depth_to_meters_array(depth_crop)
    local_u = seed_u - u0
    local_v = seed_v - v0

    color_crop: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    if color_bgr is not None:
        ch, cw = color_bgr.shape[:2]
        depth_crop_h, depth_crop_w = depth_crop.shape[:2]
        if (ch, cw) != (h, w):
            # 先把整幅彩色缩放到深度尺寸，再裁同一 ROI，避免 crop 形状不一致
            # （例如 LIBERO 256² RGB + 残留 Isaac 480×640 depth）
            color_resized = _eai_platform_bootstrap.cv2.resize(
                color_bgr, (w, h), interpolation=_eai_platform_bootstrap.cv2.INTER_LINEAR
            )
            color_crop = color_resized[v0:v1, u0:u1]
        else:
            color_crop = color_bgr[v0:v1, u0:u1]
        if color_crop.shape[:2] != (depth_crop_h, depth_crop_w):
            color_crop = _eai_platform_bootstrap.cv2.resize(
                color_crop,
                (depth_crop_w, depth_crop_h),
                interpolation=_eai_platform_bootstrap.cv2.INTER_LINEAR,
            )

    mask_crop, method = _segment_object_in_crop(
        local_u, local_v, depth_m, color_crop, depth_tol, color_tol, min_area
    )

    mask_full = _eai_platform_bootstrap.np.zeros((h, w), dtype=bool)
    mask_full[v0:v1, u0:u1] = mask_crop
    return mask_full, method


@_eai_platform_bootstrap.dataclass
class SegmentSettings:
    backend: str = _eai_configuration_perception.SAM3_BACKEND_GEOMETRY
    sam3_text: str = ""
    sam3_use_http: bool = _eai_configuration_perception.SAM3_USE_HTTP_DEFAULT
    sam3_server_url: str = _eai_configuration_perception.SAM3_SERVER_URL_DEFAULT
    sam3_python: str = _eai_configuration_perception.SAM3_PYTHON_DEFAULT
    sam3_model: str = _eai_configuration_perception.SAM3_MODEL_DEFAULT


@_eai_platform_bootstrap.dataclass
class PoseSettings:
    backend: str = _eai_configuration_perception.POSE_BACKEND_PCA
    fp_use_http: bool = _eai_configuration_perception.FP_USE_HTTP_DEFAULT
    fp_server_url: str = _eai_configuration_perception.FP_SERVER_URL_DEFAULT
    fp_python: str = _eai_configuration_perception.FP_PYTHON_DEFAULT
    fp_mesh: str = _eai_configuration_models.FP_MESH_DEFAULT
    fp_mode: str = "register"
    any6d_use_http: bool = _eai_configuration_perception.ANY6D_USE_HTTP_DEFAULT
    any6d_server_url: str = _eai_configuration_perception.ANY6D_SERVER_URL_DEFAULT
    any6d_python: str = _eai_configuration_perception.ANY6D_PYTHON_DEFAULT


_segment_settings = SegmentSettings()


_segment_settings_lock = _eai_platform_bootstrap.threading.Lock()


_pose_settings = PoseSettings()


_pose_settings_lock = _eai_platform_bootstrap.threading.Lock()

