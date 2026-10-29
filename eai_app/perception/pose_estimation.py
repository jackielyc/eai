"""perception.pose estimation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_any6d_viewer_server_url(port: int = _eai_configuration_perception.ANY6D_DEFAULT_PORT) -> str:
    env = _eai_platform_bootstrap.os.environ.get("ANY6D_SERVER_URL", "").strip()
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{int(port)}"


def fetch_any6d_health(url: str, timeout_s: float = 1.5) -> _eai_platform_bootstrap.Dict[str, object]:
    try:
        req = _eai_platform_bootstrap.urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
        return {
            "online": True,
            "any6d_ready": bool(body.get("any6d_ready", body.get("ok"))),
            "any6d_root": str(body.get("any6d_root") or ""),
            "missing_deps": list(body.get("missing_deps") or []),
            "mesh": str(body.get("mesh") or ""),
            "model_free": bool(body.get("model_free", True)),
            "raw": body,
        }
    except Exception as exc:
        return {"online": False, "error": str(exc)}


def evaluate_any6d_availability(
    mesh_path: str,
    *,
    use_http: bool,
    server_url: str,
    timeout_s: float = 1.2,
) -> _eai_perception_segmentation.FpAvailabilityStatus:
    """复用 FpAvailabilityStatus；Any6D 允许无 mesh（深度凸包）。"""
    worker_ok = _eai_platform_bootstrap.os.path.isfile(_eai_configuration_perception.ANY6D_WORKER_SCRIPT)
    if use_http:
        health = fetch_any6d_health(server_url, timeout_s)
        if not health.get("online"):
            return _eai_perception_segmentation.FpAvailabilityStatus(
                label="离线",
                color="#ff6b6b",
                detail="Any6D HTTP 未启动",
                tooltip=(
                    f"服务不可达: {server_url}\n"
                    f"请运行: bash {_eai_configuration_perception.ANY6D_RUN_SCRIPT_NAME}"
                ),
                can_invoke=False,
                http_online=False,
                worker_ok=worker_ok,
            )
        ready = bool(health.get("any6d_ready"))
        missing = health.get("missing_deps") or []
        if not ready:
            return _eai_perception_segmentation.FpAvailabilityStatus(
                label="缺依赖",
                color="#ffb86c",
                detail="服务在线但依赖未齐",
                tooltip=(
                    f"root={health.get('any6d_root')}\n"
                    f"缺少: {', '.join(map(str, missing)) or '?'}\n"
                    f"bash {_eai_configuration_perception.ANY6D_RUN_SCRIPT_NAME} --check"
                ),
                can_invoke=False,
                http_online=True,
                fp_ready=False,
                worker_ok=worker_ok,
            )
        mesh_note = str(health.get("mesh") or "model-free")
        return _eai_perception_segmentation.FpAvailabilityStatus(
            label="就绪",
            color="#50fa7b",
            detail=f"Any6D HTTP · {mesh_note}",
            tooltip=f"{server_url}\nroot={health.get('any6d_root')}\nmesh 可空=深度凸包",
            can_invoke=True,
            http_online=True,
            fp_ready=True,
            mesh_ok=True,
            worker_ok=worker_ok,
        )
    if not worker_ok:
        return _eai_perception_segmentation.FpAvailabilityStatus(
            label="无worker",
            color="#ff6b6b",
            detail="缺少 any6d_worker.py",
            tooltip=f"未找到 {_eai_configuration_perception.ANY6D_WORKER_SCRIPT}",
            can_invoke=False,
            worker_ok=False,
        )
    mesh = (mesh_path or "").strip()
    mesh_ok = (not mesh) or _eai_configuration_perception.fp_mesh_path_exists(mesh)
    return _eai_perception_segmentation.FpAvailabilityStatus(
        label="子进程",
        color="#8be9fd" if mesh_ok else "#ffb86c",
        detail=("model-free" if not mesh else "有 mesh"),
        tooltip=(
            f"将用 {_eai_configuration_perception.ANY6D_PYTHON_DEFAULT} 调 any6d_worker.py --once\n"
            "mesh 留空 → 深度凸包任意物体；有 mesh 更准"
        ),
        can_invoke=True,
        mesh_ok=mesh_ok,
        worker_ok=True,
    )


def _any6d_pose_via_http(
    payload: _eai_platform_bootstrap.Dict[str, object], server_url: str, timeout_s: float
) -> _eai_platform_bootstrap.Dict[str, object]:
    data = _eai_platform_bootstrap.json.dumps(payload).encode("utf-8")
    req = _eai_platform_bootstrap.urllib.request.Request(
        f"{server_url.rstrip('/')}/pose",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
    except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Any6D HTTP {exc.code}: {err_msg[:400]}") from exc
    except _eai_platform_bootstrap.urllib.error.URLError as exc:
        raise RuntimeError(
            f"Any6D 服务不可达 ({server_url})，请先: bash {_eai_configuration_perception.ANY6D_RUN_SCRIPT_NAME}"
        ) from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def _any6d_pose_via_subprocess(
    payload: _eai_platform_bootstrap.Dict[str, object], python_exe: str, timeout_s: float
) -> _eai_platform_bootstrap.Dict[str, object]:
    if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_perception.ANY6D_WORKER_SCRIPT):
        raise RuntimeError(f"未找到 worker: {_eai_configuration_perception.ANY6D_WORKER_SCRIPT}")
    proc = _eai_platform_bootstrap.subprocess.run(
        [python_exe, _eai_configuration_perception.ANY6D_WORKER_SCRIPT, "--once"],
        input=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
        stdout=_eai_platform_bootstrap.subprocess.PIPE,
        stderr=_eai_platform_bootstrap.subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(err or stdout or f"Any6D worker 退出码 {proc.returncode}")
    try:
        body = _eai_platform_bootstrap.json.loads(stdout.splitlines()[-1] if stdout else "{}")
    except _eai_platform_bootstrap.json.JSONDecodeError as exc:
        raise RuntimeError(f"Any6D worker 输出非 JSON: {stdout[:200]}") from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def run_any6d_estimation(
    color_bgr: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    mask: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    settings: _eai_platform_bootstrap.Optional[_eai_perception_image_geometry.PoseSettings] = None,
    *,
    tag: str = "viewer",
) -> _eai_platform_bootstrap.Dict[str, object]:
    """调用 Any6D；mesh 可空（深度凸包 model-free）。响应格式同 FoundationPose。"""
    cfg = settings or _eai_perception_segmentation.get_pose_settings()
    mesh_path = (cfg.fp_mesh or "").strip()
    # 复用 FP payload 编码；mesh 允许空字符串
    payload = _eai_perception_segmentation._fp_build_request_payload(
        color_bgr,
        depth,
        mask,
        fx,
        fy,
        cx,
        cy,
        mesh_path,
        mode="register",
    )
    payload["model_free"] = not bool(mesh_path and _eai_configuration_perception.fp_mesh_path_exists(mesh_path))
    request_summary = _eai_perception_segmentation._summarize_fp_payload(payload)
    request_summary["backend"] = "any6d"
    request_summary["transport"] = "HTTP" if cfg.any6d_use_http else "subprocess"
    t0 = _eai_platform_bootstrap.time.time()
    try:
        if cfg.any6d_use_http:
            body = _any6d_pose_via_http(
                payload, cfg.any6d_server_url, _eai_configuration_perception.ANY6D_TIMEOUT_S
            )
        else:
            body = _any6d_pose_via_subprocess(
                payload, cfg.any6d_python, _eai_configuration_perception.ANY6D_TIMEOUT_S
            )
        _eai_perception_segmentation._log_fp_request_result(
            f"any6d:{tag}",
            request_summary,
            {"ok": True, "method": body.get("method"), "position_xyz": body.get("position_xyz")},
            _eai_platform_bootstrap.time.time() - t0,
        )
        return body
    except Exception as exc:
        _eai_perception_segmentation._log_fp_request_result(
            f"any6d:{tag}",
            request_summary,
            {"ok": False, "error": str(exc)},
            _eai_platform_bootstrap.time.time() - t0,
        )
        raise


def segment_object_at_click_with_backend(
    seed_u: int,
    seed_v: int,
    depth: _eai_platform_bootstrap.np.ndarray,
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    settings: _eai_platform_bootstrap.Optional[_eai_perception_image_geometry.SegmentSettings] = None,
    depth_tol: float = _eai_configuration_perception.SEGMENT_DEPTH_TOL_M,
    color_tol: float = _eai_configuration_perception.SEGMENT_COLOR_TOL,
    min_area: int = _eai_configuration_perception.SEGMENT_MIN_AREA,
    roi_radius: int = _eai_configuration_perception.SEGMENT_ROI_RADIUS,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    cfg = settings or _eai_perception_segmentation.get_segment_settings()
    if cfg.backend == _eai_configuration_perception.SAM3_BACKEND_GEOMETRY:
        return _eai_perception_image_geometry.segment_object_at_click(
            seed_u,
            seed_v,
            depth,
            color_bgr=color_bgr,
            depth_tol=depth_tol,
            color_tol=color_tol,
            min_area=min_area,
            roi_radius=roi_radius,
        )
    if color_bgr is None:
        raise RuntimeError("SAM3 分割需要彩色图（请同时订阅 color topic）")
    text = cfg.sam3_text.strip() if cfg.backend == _eai_configuration_perception.SAM3_BACKEND_TEXT else None
    mask, method = _eai_perception_segmentation.run_sam3_segmentation(
        color_bgr, seed_u, seed_v, text=text, settings=cfg, tag="stereo_click"
    )
    if mask.shape[:2] != depth.shape[:2]:
        mask = _eai_perception_image_geometry.resize_mask_to_shape(mask, depth.shape[:2])
    if int(mask.sum()) < min_area:
        raise RuntimeError(f"SAM3 mask 过小 ({int(mask.sum())} px)")
    return mask, method


def mask_to_points_3d(
    mask: _eai_platform_bootstrap.np.ndarray,
    depth_m: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]:
    ys, xs = _eai_platform_bootstrap.np.where(mask)
    if len(xs) == 0:
        return _eai_platform_bootstrap.np.empty((0, 3), dtype=_eai_platform_bootstrap.np.float32), _eai_platform_bootstrap.np.empty((0, 2), dtype=_eai_platform_bootstrap.np.float32)
    zs = depth_m[ys, xs]
    valid = _eai_platform_bootstrap.valid_depth_mask(zs)
    xs = xs[valid].astype(_eai_platform_bootstrap.np.float32)
    ys = ys[valid].astype(_eai_platform_bootstrap.np.float32)
    zs = zs[valid]
    if len(xs) == 0:
        return _eai_platform_bootstrap.np.empty((0, 3), dtype=_eai_platform_bootstrap.np.float32), _eai_platform_bootstrap.np.empty((0, 2), dtype=_eai_platform_bootstrap.np.float32)
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points = _eai_platform_bootstrap.np.stack([x, y, zs], axis=-1).astype(_eai_platform_bootstrap.np.float32)
    uv = _eai_platform_bootstrap.np.stack([xs, ys], axis=-1).astype(_eai_platform_bootstrap.np.float32)
    return points, uv


def uv_to_mask(uv: _eai_platform_bootstrap.np.ndarray, shape: _eai_platform_bootstrap.Tuple[int, int]) -> _eai_platform_bootstrap.np.ndarray:
    h, w = shape
    mask = _eai_platform_bootstrap.np.zeros((h, w), dtype=bool)
    if len(uv) == 0:
        return mask
    us = _eai_platform_bootstrap.np.clip(uv[:, 0].astype(_eai_platform_bootstrap.np.int32), 0, w - 1)
    vs = _eai_platform_bootstrap.np.clip(uv[:, 1].astype(_eai_platform_bootstrap.np.int32), 0, h - 1)
    mask[vs, us] = True
    return mask


def region_grow_3d(
    points: _eai_platform_bootstrap.np.ndarray,
    uv: _eai_platform_bootstrap.np.ndarray,
    seed_u: float,
    seed_v: float,
    eps: float = _eai_configuration_perception.SEGMENT_3D_EPS_M,
    max_cluster: int = _eai_configuration_perception.SEGMENT_MAX_3D_POINTS,
) -> _eai_platform_bootstrap.np.ndarray:
    """3D 区域生长，在体素网格上做连通聚类。"""
    n = len(points)
    if n == 0:
        return _eai_platform_bootstrap.np.array([], dtype=_eai_platform_bootstrap.np.int64)
    if n > max_cluster:
        rng = _eai_platform_bootstrap.np.random.default_rng(2)
        keep = rng.choice(n, max_cluster, replace=False)
        seed_dist = (uv[:, 0] - seed_u) ** 2 + (uv[:, 1] - seed_v) ** 2
        seed_i_full = int(_eai_platform_bootstrap.np.argmin(seed_dist))
        if seed_i_full not in keep:
            keep[0] = seed_i_full
        points = points[keep]
        uv = uv[keep]
        n = len(points)

    seed_i = int(_eai_platform_bootstrap.np.argmin((uv[:, 0] - seed_u) ** 2 + (uv[:, 1] - seed_v) ** 2))
    inv_cell = 1.0 / max(eps, 1e-4)
    keys = _eai_platform_bootstrap.np.round(points * inv_cell).astype(_eai_platform_bootstrap.np.int32)
    grid: _eai_platform_bootstrap.Dict[_eai_platform_bootstrap.Tuple[int, int, int], _eai_platform_bootstrap.List[int]] = {}
    for i in range(n):
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        grid.setdefault(key, []).append(i)

    visited = _eai_platform_bootstrap.np.zeros(n, dtype=bool)
    queue: _eai_platform_bootstrap.deque[int] = _eai_platform_bootstrap.deque([seed_i])
    visited[seed_i] = True
    cluster: _eai_platform_bootstrap.List[int] = []
    eps_sq = eps * eps

    while queue:
        i = queue.popleft()
        cluster.append(i)
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        if visited[j]:
                            continue
                        diff = points[j] - points[i]
                        if float(_eai_platform_bootstrap.np.dot(diff, diff)) <= eps_sq:
                            visited[j] = True
                            queue.append(j)
    return _eai_platform_bootstrap.np.array(cluster, dtype=_eai_platform_bootstrap.np.int64)


def remove_statistical_outliers(points: _eai_platform_bootstrap.np.ndarray, k_mad: float = 2.5) -> _eai_platform_bootstrap.np.ndarray:
    if len(points) < 8:
        return _eai_platform_bootstrap.np.ones(len(points), dtype=bool)
    center = _eai_platform_bootstrap.np.median(points, axis=0)
    dist = _eai_platform_bootstrap.np.linalg.norm(points - center, axis=1)
    med = float(_eai_platform_bootstrap.np.median(dist))
    mad = float(_eai_platform_bootstrap.np.median(_eai_platform_bootstrap.np.abs(dist - med))) + 1e-6
    return dist <= med + k_mad * mad * 1.4826


def rotation_matrix_to_euler_xyz(R: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[float, float, float]:
    """相机坐标系下 roll(X), pitch(Y), yaw(Z)，单位弧度。"""
    sy = _eai_platform_bootstrap.math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy >= 1e-6:
        roll = _eai_platform_bootstrap.math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = _eai_platform_bootstrap.math.atan2(float(-R[2, 0]), sy)
        yaw = _eai_platform_bootstrap.math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = _eai_platform_bootstrap.math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = _eai_platform_bootstrap.math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def quat_xyzw_to_rpy_deg(
    quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
) -> _eai_platform_bootstrap.Tuple[float, float, float]:
    R = _eai_adapters_robot_geometry.quat_to_rotation_matrix(*quat_xyzw)
    roll, pitch, yaw = rotation_matrix_to_euler_xyz(R)
    return _eai_platform_bootstrap.math.degrees(roll), _eai_platform_bootstrap.math.degrees(pitch), _eai_platform_bootstrap.math.degrees(yaw)


def format_xyz_rpy_line(
    prefix: str,
    xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    quat_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float],
) -> str:
    roll, pitch, yaw = quat_xyzw_to_rpy_deg(quat_xyzw)
    return (
        f"{prefix} XYZ=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m  "
        f"RPY=({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
    )


def format_left_hand_state_line(state: _eai_adapters_robot_geometry.RobotStateSnapshot) -> str:
    if not state.left_hand:
        return "左手: (无数据)"
    vals = [v for _, v in state.left_hand]
    avg = sum(vals) / len(vals)
    return f"左手: avg={avg:.3f} ({len(vals)} 关节)"


def rotation_matrix_to_quaternion(R: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = _eai_platform_bootstrap.math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + float(R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + float(R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = _eai_platform_bootstrap.math.sqrt(1.0 + float(R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = _eai_platform_bootstrap.math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-8
    return qx / norm, qy / norm, qz / norm, qw / norm


def obb_corners(center: _eai_platform_bootstrap.np.ndarray, rotation: _eai_platform_bootstrap.np.ndarray, half_extents: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
    """8 corners in a fixed order matching draw_obb_on_image / FoundationPose worker."""
    signs = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    corners = []
    for sx, sy, sz in signs:
        local = _eai_platform_bootstrap.np.array([sx, sy, sz], dtype=_eai_platform_bootstrap.np.float32) * half_extents
        corners.append(center + rotation @ local)
    return _eai_platform_bootstrap.np.stack(corners, axis=0).astype(_eai_platform_bootstrap.np.float32)


OBB_EDGE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # z-min face
    (4, 5), (5, 6), (6, 7), (7, 4),  # z-max face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def obb_wireframe_edges(corners: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
    lines = []
    for a, b in OBB_EDGE_PAIRS:
        lines.append(corners[a])
        lines.append(corners[b])
    return _eai_platform_bootstrap.np.stack(lines, axis=0).astype(_eai_platform_bootstrap.np.float32)


def pose_axes_lines(center: _eai_platform_bootstrap.np.ndarray, rotation: _eai_platform_bootstrap.np.ndarray, scale: float) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]:
    origin = center.astype(_eai_platform_bootstrap.np.float32)
    lines = []
    colors = []
    axis_colors = [
        (1.0, 0.2, 0.2, 1.0),
        (0.2, 1.0, 0.2, 1.0),
        (0.2, 0.4, 1.0, 1.0),
    ]
    for i in range(3):
        end = origin + rotation[:, i] * scale
        lines.extend([origin, end])
        colors.extend([axis_colors[i], axis_colors[i]])
    return _eai_platform_bootstrap.np.stack(lines, axis=0).astype(_eai_platform_bootstrap.np.float32), _eai_platform_bootstrap.np.stack(colors, axis=0).astype(_eai_platform_bootstrap.np.float32)


def refine_fused_object_points(
    points: _eai_platform_bootstrap.np.ndarray,
    *,
    seed_xyz: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    xy_radius_m: float = _eai_configuration_policies.MV6D_FUSE_XY_RADIUS_M,
    table_clear_m: float = _eai_configuration_policies.MV6D_FUSE_TABLE_CLEAR_M,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    """融合点云清洗：种子邻域 + 去桌面 + 最密体素中心邻域。

    返回 (cleaned_points, note)。点过少时原样返回。
    """
    _eai_platform_bootstrap._ensure_cv2_numpy()
    pts = _eai_platform_bootstrap.np.asarray(points, dtype=_eai_platform_bootstrap.np.float32).reshape(-1, 3)
    if len(pts) < _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        return pts, "点过少跳过清洗"
    notes: _eai_platform_bootstrap.List[str] = []
    seed = (
        _eai_platform_bootstrap.np.asarray(seed_xyz, dtype=_eai_platform_bootstrap.np.float32).reshape(3)
        if seed_xyz is not None
        else _eai_platform_bootstrap.np.median(pts, axis=0).astype(_eai_platform_bootstrap.np.float32)
    )

    # 1) XY 半径：挡住邻物 / 腕部误分割
    d_xy = _eai_platform_bootstrap.np.linalg.norm(pts[:, :2] - seed[:2], axis=1)
    rad = float(max(0.04, xy_radius_m))
    sel = d_xy <= rad
    if int(sel.sum()) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        pts = pts[sel]
        notes.append(f"XY≤{rad * 1000:.0f}mm")
    else:
        # 放宽到 1.6×
        sel = d_xy <= rad * 1.6
        if int(sel.sum()) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
            pts = pts[sel]
            notes.append(f"XY≤{rad * 1.6 * 1000:.0f}mm(放宽)")

    # 2) 去桌面：用 Z 直方图低端峰作桌面高度
    z = pts[:, 2]
    z_min = float(_eai_platform_bootstrap.np.min(z))
    z_max = float(_eai_platform_bootstrap.np.max(z))
    if z_max - z_min > 0.025:
        bins = max(24, min(64, int((z_max - z_min) / 0.004)))
        hist, edges = _eai_platform_bootstrap.np.histogram(z, bins=bins)
        # 只在下半高度找峰，避免把物体顶面当桌面
        mid = max(1, bins // 2)
        peak_i = int(_eai_platform_bootstrap.np.argmax(hist[:mid]))
        table_z = 0.5 * float(edges[peak_i] + edges[peak_i + 1])
        clear = float(max(0.008, table_clear_m))
        above = z >= (table_z + clear)
        if int(above.sum()) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
            pts = pts[above]
            notes.append(f"去桌面Z>{table_z + clear:.3f}")

    # 3) 统计离群
    keep = remove_statistical_outliers(pts, k_mad=2.2)
    if int(keep.sum()) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        pts = pts[keep]

    # 4) 最密 2cm 体素 → 再收紧到该体素中心附近（抑双峰/邻物）
    if len(pts) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        voxel = 0.02
        keys = _eai_platform_bootstrap.np.floor(pts / voxel).astype(_eai_platform_bootstrap.np.int32)
        # 哈希：打包成一维
        flat = (
            keys[:, 0].astype(_eai_platform_bootstrap.np.int64) * 73856093
            ^ keys[:, 1].astype(_eai_platform_bootstrap.np.int64) * 19349663
            ^ keys[:, 2].astype(_eai_platform_bootstrap.np.int64) * 83492791
        )
        uniq, inv, counts = _eai_platform_bootstrap.np.unique(flat, return_inverse=True, return_counts=True)
        best = int(_eai_platform_bootstrap.np.argmax(counts))
        dens_center = _eai_platform_bootstrap.np.mean(pts[inv == best], axis=0).astype(_eai_platform_bootstrap.np.float32)
        # 若种子更可信，向种子靠拢
        dens_center = (0.65 * dens_center + 0.35 * seed).astype(_eai_platform_bootstrap.np.float32)
        d = _eai_platform_bootstrap.np.linalg.norm(pts - dens_center, axis=1)
        thr = float(_eai_platform_bootstrap.np.percentile(d, 75))
        thr = max(0.025, min(float(xy_radius_m), thr))
        tight = d <= thr
        if int(tight.sum()) >= _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
            pts = pts[tight]
            notes.append("最密体素邻域")

    return pts, ("+".join(notes) if notes else "无额外清洗")


def estimate_6d_pose_pca(
    points: _eai_platform_bootstrap.np.ndarray,
    *,
    align_gravity: bool = True,
    gravity_up: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    seed_xyz: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
) -> _eai_platform_bootstrap.Optional[
    _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]
]:
    """PCA-OBB 6D（模型无关）。

    改进相对朴素 PCA：
    - 统计离群点剔除 + 中心邻域迭代修剪（抑深度飞点）
    - 可选种子吸引，避免中心漂到邻物/桌面
    - 桌面场景默认把最近竖直轴对齐重力（base_link +Z），姿态更稳
    """
    _eai_platform_bootstrap._ensure_cv2_numpy()
    pts = _eai_platform_bootstrap.np.asarray(points, dtype=_eai_platform_bootstrap.np.float32).reshape(-1, 3)
    if len(pts) < _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        return None
    keep = remove_statistical_outliers(pts, k_mad=2.5)
    pts = pts[keep]
    if len(pts) < _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
        return None
    # 迭代修剪到中位数附近，抑制 mask 边缘/桌面混入
    for _ in range(3):
        c = _eai_platform_bootstrap.np.median(pts, axis=0)
        if seed_xyz is not None:
            s = _eai_platform_bootstrap.np.asarray(seed_xyz, dtype=_eai_platform_bootstrap.np.float32).reshape(3)
            c = (0.7 * c + 0.3 * s).astype(_eai_platform_bootstrap.np.float32)
        d = _eai_platform_bootstrap.np.linalg.norm(pts - c, axis=1)
        med = float(_eai_platform_bootstrap.np.median(d))
        mad = float(_eai_platform_bootstrap.np.median(_eai_platform_bootstrap.np.abs(d - med))) + 1e-6
        thr = med + 2.5 * mad * 1.4826
        thr = max(thr, 0.018)
        sel = d <= thr
        if int(sel.sum()) < _eai_configuration_perception.SEGMENT_3D_MIN_POINTS:
            break
        pts = pts[sel]

    centroid = _eai_platform_bootstrap.np.median(pts, axis=0).astype(_eai_platform_bootstrap.np.float32)
    if seed_xyz is not None:
        s = _eai_platform_bootstrap.np.asarray(seed_xyz, dtype=_eai_platform_bootstrap.np.float32).reshape(3)
        # XY 更信种子（点击/头部中心），Z 仍信点云
        centroid = _eai_platform_bootstrap.np.array(
            [
                0.55 * float(centroid[0]) + 0.45 * float(s[0]),
                0.55 * float(centroid[1]) + 0.45 * float(s[1]),
                0.85 * float(centroid[2]) + 0.15 * float(s[2]),
            ],
            dtype=_eai_platform_bootstrap.np.float32,
        )
    centered = pts - centroid
    cov = _eai_platform_bootstrap.np.cov(centered.astype(_eai_platform_bootstrap.np.float64), rowvar=False)
    evals, evecs = _eai_platform_bootstrap.np.linalg.eigh(cov)
    order = _eai_platform_bootstrap.np.argsort(evals)[::-1]
    rotation = evecs[:, order].astype(_eai_platform_bootstrap.np.float32)
    if _eai_platform_bootstrap.np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0

    if align_gravity:
        up = _eai_platform_bootstrap.np.asarray(
            gravity_up if gravity_up is not None else (0.0, 0.0, 1.0),
            dtype=_eai_platform_bootstrap.np.float32,
        ).reshape(3)
        nrm = float(_eai_platform_bootstrap.np.linalg.norm(up)) + 1e-8
        up = up / nrm
        # 选与重力最对齐的主轴，放到第 2 列（物体「竖直」轴）
        dots = _eai_platform_bootstrap.np.abs(rotation.T @ up)
        zi = int(_eai_platform_bootstrap.np.argmax(dots))
        if zi != 2:
            rotation[:, [zi, 2]] = rotation[:, [2, zi]]
        if float(_eai_platform_bootstrap.np.dot(rotation[:, 2], up)) < 0.0:
            rotation[:, 2] *= -1.0
        # 右手系重正交
        rotation[:, 0] = _eai_platform_bootstrap.np.cross(rotation[:, 1], rotation[:, 2])
        n0 = float(_eai_platform_bootstrap.np.linalg.norm(rotation[:, 0])) + 1e-8
        rotation[:, 0] /= n0
        rotation[:, 1] = _eai_platform_bootstrap.np.cross(rotation[:, 2], rotation[:, 0])
        if _eai_platform_bootstrap.np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0

    local = centered @ rotation
    # 用更紧的百分位，减少桌面/飞点拉大 OBB
    ext_min = _eai_platform_bootstrap.np.percentile(local, 8, axis=0)
    ext_max = _eai_platform_bootstrap.np.percentile(local, 92, axis=0)
    half_extents = ((ext_max - ext_min) * 0.5).astype(_eai_platform_bootstrap.np.float32)
    half_extents = _eai_platform_bootstrap.np.maximum(half_extents, 0.005)
    obb_center = centroid + rotation @ ((ext_max + ext_min) * 0.5).astype(_eai_platform_bootstrap.np.float32)

    roll, pitch, yaw = rotation_matrix_to_euler_xyz(rotation)
    quat = rotation_matrix_to_quaternion(rotation)
    return obb_center, rotation, half_extents, (roll, pitch, yaw), quat


def refine_stereo_segment(
    mask: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]:
    """2D 掩码 → 3D 点云聚类 refine，返回 (refined_mask, points_3d)。"""
    mask = _eai_perception_image_geometry.subsample_mask(mask)
    if not mask.any():
        return mask, _eai_platform_bootstrap.np.empty((0, 3), dtype=_eai_platform_bootstrap.np.float32)

    ys, xs = _eai_platform_bootstrap.np.where(mask)
    pad = 10
    h, w = mask.shape
    v0 = max(0, int(ys.min()) - pad)
    v1 = min(h, int(ys.max()) + pad + 1)
    u0 = max(0, int(xs.min()) - pad)
    u1 = min(w, int(xs.max()) + pad + 1)

    mask_crop = mask[v0:v1, u0:u1]
    depth_crop = depth[v0:v1, u0:u1]
    depth_m = _eai_perception_image_geometry.depth_to_meters_array(depth_crop)
    local_seed_u = seed_u - u0
    local_seed_v = seed_v - v0
    local_cx = cx - u0
    local_cy = cy - v0

    points, uv = mask_to_points_3d(mask_crop, depth_m, fx, fy, local_cx, local_cy)
    if len(points) == 0:
        return mask, points

    cluster_idx = region_grow_3d(points, uv, float(local_seed_u), float(local_seed_v))
    if len(cluster_idx) < _eai_configuration_perception.SEGMENT_3D_MIN_POINTS // 2:
        cluster_idx = _eai_platform_bootstrap.np.arange(len(points), dtype=_eai_platform_bootstrap.np.int64)

    inlier = remove_statistical_outliers(points[cluster_idx])
    cluster_idx = cluster_idx[inlier]
    if len(cluster_idx) < 5:
        return mask, points

    refined_points = points[cluster_idx]

    refined_uv = uv[cluster_idx]
    refined_mask_crop = uv_to_mask(refined_uv, mask_crop.shape)
    refined_mask = _eai_platform_bootstrap.np.zeros((h, w), dtype=bool)
    refined_mask[v0:v1, u0:u1] = refined_mask_crop
    return refined_mask, refined_points


def run_stereo_pose_pipeline(
    depth: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    pose_settings: _eai_platform_bootstrap.Optional[_eai_perception_image_geometry.PoseSettings] = None,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[Object6DPoseResult], str]:
    pcfg = pose_settings or _eai_perception_segmentation.get_pose_settings()
    try:
        mask, method = segment_object_at_click_with_backend(
            seed_u, seed_v, depth, color_bgr=color_bgr
        )
    except Exception as exc:
        return None, f"点击 ({seed_u}, {seed_v})  |  分割失败: {exc}"
    if not mask.any():
        return None, f"点击 ({seed_u}, {seed_v})  |  立体分割失败（无有效区域）"

    if pcfg.backend in (_eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE, _eai_configuration_perception.POSE_BACKEND_ANY6D):
        backend_name = (
            "FoundationPose"
            if pcfg.backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE
            else "Any6D"
        )
        if color_bgr is None:
            return None, (
                f"点击 ({seed_u}, {seed_v})  |  {backend_name} 需要彩色图"
                "（请同时订阅 color topic）"
            )
        try:
            if pcfg.backend == _eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE:
                fp_body = _eai_perception_segmentation.run_foundationpose_estimation(
                    color_bgr,
                    depth,
                    mask,
                    fx,
                    fy,
                    cx,
                    cy,
                    settings=pcfg,
                    tag="stereo_click",
                )
            else:
                fp_body = run_any6d_estimation(
                    color_bgr,
                    depth,
                    mask,
                    fx,
                    fy,
                    cx,
                    cy,
                    settings=pcfg,
                    tag="stereo_click",
                )
            result = _eai_perception_segmentation.object6d_from_foundationpose(
                mask,
                depth,
                seed_u,
                seed_v,
                fx,
                fy,
                cx,
                cy,
                fp_body,
                segment_method=method,
            )
        except Exception as exc:
            return None, f"点击 ({seed_u}, {seed_v})  |  {backend_name} 失败: {exc}"
        if result is None:
            return None, f"点击 ({seed_u}, {seed_v})  |  {backend_name} 响应无效"
        return result, format_pose_6d_info(result, seed_u, seed_v)

    result = compute_stereo_segment_6d_pose(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy, method=method
    )
    if result is None:
        return None, f"点击 ({seed_u}, {seed_v})  |  立体分割失败（深度无效或点云过少）"
    return result, format_pose_6d_info(result, seed_u, seed_v)


def run_stereo_pose_pipeline_resampled(
    depth: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[Object6DPoseResult], str]:
    """降采样后在后台线程运行分割，减轻卡顿。"""
    depth_s, scale = _eai_perception_image_geometry.downsample_image_for_compute(depth)
    color_s = None
    if color_bgr is not None:
        color_s, _ = _eai_perception_image_geometry.downsample_image_for_compute(color_bgr)
    su = max(0, min(depth_s.shape[1] - 1, int(round(seed_u * scale))))
    sv = max(0, min(depth_s.shape[0] - 1, int(round(seed_v * scale))))
    fx_s, fy_s, cx_s, cy_s = _eai_perception_image_geometry.scale_intrinsics_by(fx, fy, cx, cy, scale)
    result, info = run_stereo_pose_pipeline(
        depth_s, su, sv, fx_s, fy_s, cx_s, cy_s, color_bgr=color_s
    )
    if result is None:
        return None, info
    if scale < 0.999:
        result = _eai_perception_image_geometry.upscale_segment_result(result, scale)
        info = format_pose_6d_info(result, seed_u, seed_v)
    return result, info


@_eai_platform_bootstrap.dataclass
class Object6DPoseResult:
    mask: _eai_platform_bootstrap.np.ndarray
    points_3d: _eai_platform_bootstrap.np.ndarray
    centroid_uv: _eai_platform_bootstrap.Tuple[float, float]
    contact_uv: _eai_platform_bootstrap.Tuple[float, float]
    contact_xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    position_xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    euler_rpy_rad: _eai_platform_bootstrap.Tuple[float, float, float]
    quaternion_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float]
    rotation_matrix: _eai_platform_bootstrap.np.ndarray
    obb_center: _eai_platform_bootstrap.Tuple[float, float, float]
    obb_extents: _eai_platform_bootstrap.Tuple[float, float, float]
    obb_corners: _eai_platform_bootstrap.np.ndarray
    depth_m: float
    pixel_count: int
    point_count: int
    method: str
    raw_mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    source_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None
    diagnostics_dir: str = ""


@_eai_platform_bootstrap.dataclass
class SegmentPoseTarget:
    """相机系下分割得到的 6D 位姿目标（接触 TCP → 左臂 IK）。"""

    camera_frame: str
    position_xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    quaternion_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float]
    source_topic: str
    label: str = ""
    contact_uv: _eai_platform_bootstrap.Tuple[float, float] = (0.0, 0.0)
    obb_center_xyz: _eai_platform_bootstrap.Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # FoundationPose 物体姿态通常不可直接作为 TCP 姿态；位置绝对、姿态保持当前 TCP
    keep_tcp_orientation: bool = False

    @classmethod
    def from_pose_result(
        cls,
        result: Object6DPoseResult,
        camera_frame: str,
        source_topic: str,
    ) -> SegmentPoseTarget:
        method = (result.method or "").lower()
        # FoundationPose：物体中心绝对位置；几何分割：点击处接触点
        use_object_pose = "foundationpose" in method
        if use_object_pose:
            position = result.obb_center
            label = (
                f"{source_topic.split('/')[-1]} FoundationPose "
                f"({result.point_count} pts)"
            )
        else:
            position = result.contact_xyz
            label = (
                f"{source_topic.split('/')[-1]} 接触TCP "
                f"({result.point_count} pts)"
            )
        return cls(
            camera_frame=camera_frame,
            position_xyz=position,
            quaternion_xyzw=result.quaternion_xyzw,
            source_topic=source_topic,
            contact_uv=result.contact_uv,
            obb_center_xyz=result.obb_center,
            label=label,
            keep_tcp_orientation=use_object_pose,
        )


@_eai_platform_bootstrap.dataclass
class ResolvedArmMoveGoal:
    position_xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    quaternion_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float]
    label: str
    arm_side: str = "left"  # "left" | "right"


def arm_side_label(side: str) -> str:
    return "左臂" if side == "left" else "右臂"


def xyzw_to_wxyz(
    quat_xyzw: _eai_platform_bootstrap.Sequence[float],
) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return (w, x, y, z)


def wxyz_to_xyzw(
    quat_wxyz: _eai_platform_bootstrap.Sequence[float],
) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return (x, y, z, w)


def pose7_wxyz_from_xyz_xyzw(
    xyz: _eai_platform_bootstrap.Sequence[float],
    quat_xyzw: _eai_platform_bootstrap.Sequence[float],
) -> _eai_platform_bootstrap.List[float]:
    w, x, y, z = xyzw_to_wxyz(quat_xyzw)
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), w, x, y, z]


def xyz_xyzw_from_pose7_wxyz(
    pose7: _eai_platform_bootstrap.Sequence[float],
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[float, float, float], _eai_platform_bootstrap.Tuple[float, float, float, float]]:
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose7]
    return (x, y, z), (qx, qy, qz, qw)


def slerp_xyzw(
    q0: _eai_platform_bootstrap.Sequence[float],
    q1: _eai_platform_bootstrap.Sequence[float],
    t: float,
) -> _eai_platform_bootstrap.Tuple[float, float, float, float]:
    """Spherical linear interpolation for xyzw quaternions."""
    a = _eai_platform_bootstrap.np.asarray(q0, dtype=_eai_platform_bootstrap.np.float64)
    b = _eai_platform_bootstrap.np.asarray(q1, dtype=_eai_platform_bootstrap.np.float64)
    a = a / (_eai_platform_bootstrap.np.linalg.norm(a) + 1e-12)
    b = b / (_eai_platform_bootstrap.np.linalg.norm(b) + 1e-12)
    dot = float(_eai_platform_bootstrap.np.clip(_eai_platform_bootstrap.np.dot(a, b), -1.0, 1.0))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        out = out / (_eai_platform_bootstrap.np.linalg.norm(out) + 1e-12)
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    theta_0 = _eai_platform_bootstrap.math.acos(dot)
    sin_0 = _eai_platform_bootstrap.math.sin(theta_0)
    theta = theta_0 * t
    s0 = _eai_platform_bootstrap.math.sin(theta_0 - theta) / sin_0
    s1 = _eai_platform_bootstrap.math.sin(theta) / sin_0
    out = s0 * a + s1 * b
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def make_manual_offset_spinbox() -> _eai_platform_bootstrap.QDoubleSpinBox:
    spin = _eai_platform_bootstrap.QDoubleSpinBox()
    spin.setRange(-_eai_configuration_chat.MANUAL_OFFSET_MAX_CM, _eai_configuration_chat.MANUAL_OFFSET_MAX_CM)
    spin.setSingleStep(_eai_configuration_chat.MANUAL_OFFSET_STEP_CM)
    spin.setDecimals(1)
    spin.setSuffix(" cm")
    spin.setFixedWidth(84)
    return spin


def make_positive_offset_spinbox(default_cm: float = 0.0) -> _eai_platform_bootstrap.QDoubleSpinBox:
    """单方向步长（cm，≥0），用于前/后/左/右/上/下各自输入框。"""
    spin = _eai_platform_bootstrap.QDoubleSpinBox()
    spin.setRange(0.0, _eai_configuration_chat.MANUAL_OFFSET_MAX_CM)
    spin.setSingleStep(_eai_configuration_chat.MANUAL_OFFSET_STEP_CM)
    spin.setDecimals(1)
    spin.setSuffix(" cm")
    spin.setFixedWidth(84)
    spin.setValue(float(default_cm))
    return spin


ARM_DIR_ORDER: _eai_platform_bootstrap.Tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "up",
    "down",
)


ARM_DIR_LABEL_CN: _eai_platform_bootstrap.Dict[str, str] = {
    "forward": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "up": "上",
    "down": "下",
}


def manual_offset_spin_to_m(spin: _eai_platform_bootstrap.QDoubleSpinBox) -> float:
    """UI 偏移旋钮（cm）→ 米。"""
    return float(spin.value()) * 0.01


def set_manual_offset_spin_m(spin: _eai_platform_bootstrap.QDoubleSpinBox, meters: float) -> None:
    """按米写入偏移旋钮（内部换算为 cm 并夹紧）。"""
    cm = float(meters) * 100.0
    cm = max(-_eai_configuration_chat.MANUAL_OFFSET_MAX_CM, min(_eai_configuration_chat.MANUAL_OFFSET_MAX_CM, cm))
    spin.setValue(cm)


def compute_relative_move_goal(
    current_xyz: _eai_platform_bootstrap.Tuple[float, float, float],
    current_quat: _eai_platform_bootstrap.Tuple[float, float, float, float],
    dx: float,
    dy: float,
    dz: float,
    arm_side: str = "left",
) -> _eai_platform_bootstrap.Optional[ResolvedArmMoveGoal]:
    if abs(dx) + abs(dy) + abs(dz) < 1e-6:
        return None
    goal_xyz = (current_xyz[0] + dx, current_xyz[1] + dy, current_xyz[2] + dz)
    side_name = arm_side_label(arm_side)
    return ResolvedArmMoveGoal(
        position_xyz=goal_xyz,
        quaternion_xyzw=current_quat,
        label=(
            f"{side_name}相对偏移 "
            f"Δ({dx * 100:+.1f}, {dy * 100:+.1f}, {dz * 100:+.1f}) cm"
        ),
        arm_side=arm_side,
    )


def compute_stereo_segment_6d_pose(
    mask: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    method: str = "",
) -> _eai_platform_bootstrap.Optional[Object6DPoseResult]:
    refined_mask, points = refine_stereo_segment(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy
    )
    pose = estimate_6d_pose_pca(points)
    if pose is None:
        return None

    obb_center, rotation, half_extents, rpy, quat = pose
    corners = obb_corners(obb_center, rotation, half_extents)
    depth_m = float(obb_center[2])

    us = (points[:, 0] * fx / points[:, 2] + cx) if len(points) else _eai_platform_bootstrap.np.array([seed_u])
    vs = (points[:, 1] * fy / points[:, 2] + cy) if len(points) else _eai_platform_bootstrap.np.array([seed_v])
    centroid_uv = (float(_eai_platform_bootstrap.np.median(us)), float(_eai_platform_bootstrap.np.median(vs)))
    contact_xyz = _eai_perception_image_geometry.compute_contact_point_3d(
        seed_u, seed_v, depth, fx, fy, cx, cy, points_3d=points
    )
    if contact_xyz is None:
        contact_xyz = (float(obb_center[0]), float(obb_center[1]), float(obb_center[2]))

    return Object6DPoseResult(
        mask=refined_mask,
        points_3d=points,
        centroid_uv=centroid_uv,
        contact_uv=(float(seed_u), float(seed_v)),
        contact_xyz=contact_xyz,
        position_xyz=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=rotation,
        obb_center=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        obb_extents=(float(half_extents[0] * 2), float(half_extents[1] * 2), float(half_extents[2] * 2)),
        obb_corners=corners,
        depth_m=depth_m,
        pixel_count=int(refined_mask.sum()),
        point_count=len(points),
        method=method,
    )


def compute_rgb_only_6d_pose(
    color_bgr: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    assumed_depth_m: float = 0.45,
    method: str = "rgb-only",
) -> _eai_platform_bootstrap.Optional[Object6DPoseResult]:
    """无深度时的单目近似 6D：颜色分割 + 假定平面深度上的点云 PCA。

    适用于 LIBERO 等默认未开 ``camera_depths`` 的后端；尺度依赖 assumed_depth_m，
    姿态主要反映图像平面内主方向，不可当作真值级 metric 6D。
    """
    if color_bgr is None or _eai_platform_bootstrap.np.asarray(color_bgr).size == 0:
        return None
    img = _eai_platform_bootstrap.np.asarray(color_bgr)
    h, w = img.shape[:2]
    su = max(0, min(w - 1, int(seed_u)))
    sv = max(0, min(h - 1, int(seed_v)))
    mask = _eai_perception_image_geometry.segment_by_color_floodfill(su, sv, img)
    seg_method = "color"
    if not mask.any() or int(mask.sum()) < 30:
        rad = 36
        mask = _eai_platform_bootstrap.np.zeros((h, w), dtype=bool)
        mask[
            max(0, sv - rad) : min(h, sv + rad + 1),
            max(0, su - rad) : min(w, su + rad + 1),
        ] = True
        seg_method = "seed-box"
    ys, xs = _eai_platform_bootstrap.np.where(mask)
    if len(xs) < 8:
        return None
    # 子采样加速
    if len(xs) > 4000:
        step = max(1, len(xs) // 4000)
        xs = xs[::step]
        ys = ys[::step]
    z = float(max(0.05, assumed_depth_m))
    x = (xs.astype(_eai_platform_bootstrap.np.float32) - float(cx)) * z / float(max(1e-6, fx))
    y = (ys.astype(_eai_platform_bootstrap.np.float32) - float(cy)) * z / float(max(1e-6, fy))
    points = _eai_platform_bootstrap.np.stack([x, y, _eai_platform_bootstrap.np.full_like(x, z)], axis=1)
    pose = estimate_6d_pose_pca(points)
    if pose is None:
        return None
    obb_center, rotation, half_extents, rpy, quat = pose
    corners = obb_corners(obb_center, rotation, half_extents)
    cu = float(_eai_platform_bootstrap.np.median(xs))
    cv = float(_eai_platform_bootstrap.np.median(ys))
    refined = _eai_platform_bootstrap.np.zeros((h, w), dtype=bool)
    refined[ys, xs] = True
    return Object6DPoseResult(
        mask=refined,
        points_3d=points.astype(_eai_platform_bootstrap.np.float32),
        centroid_uv=(cu, cv),
        contact_uv=(float(su), float(sv)),
        contact_xyz=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        position_xyz=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=rotation,
        obb_center=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        obb_extents=(
            float(half_extents[0] * 2),
            float(half_extents[1] * 2),
            float(half_extents[2] * 2),
        ),
        obb_corners=corners,
        depth_m=float(obb_center[2]),
        pixel_count=int(refined.sum()),
        point_count=len(points),
        method=f"{method}/{seg_method}",
    )


def compute_object_3d_from_mask(
    mask: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    seed_u: int = 0,
    seed_v: int = 0,
    method: str = "",
) -> _eai_platform_bootstrap.Optional[Object6DPoseResult]:
    return compute_stereo_segment_6d_pose(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy, method=method
    )


def format_pose_6d_info(result: Object6DPoseResult, seed_u: int, seed_v: int) -> str:
    cx, cy, cz = result.contact_xyz
    ox, oy, oz = result.position_xyz
    roll, pitch, yaw = result.euler_rpy_rad
    qx, qy, qz, qw = result.quaternion_xyzw
    dx, dy, dz = result.obb_extents
    r_deg = _eai_platform_bootstrap.math.degrees(roll)
    p_deg = _eai_platform_bootstrap.math.degrees(pitch)
    y_deg = _eai_platform_bootstrap.math.degrees(yaw)
    method = result.method or "stereo3d"
    return (
        f"点击 ({seed_u}, {seed_v})  |  立体分割 {result.point_count} pts / {result.pixel_count} px ({method})\n"
        f"接触 TCP XYZ=({cx:.3f}, {cy:.3f}, {cz:.3f}) m  |  "
        f"OBB中心=({ox:.3f}, {oy:.3f}, {oz:.3f}) m\n"
        f"RPY=({r_deg:.1f}°, {p_deg:.1f}°, {y_deg:.1f}°)  |  "
        f"OBB=({dx:.3f}, {dy:.3f}, {dz:.3f}) m  |  "
        f"四元数 xyzw=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
    )


def format_segment_info(result: Object6DPoseResult, seed_u: int, seed_v: int) -> str:
    return format_pose_6d_info(result, seed_u, seed_v)


@_eai_platform_bootstrap.dataclass
class MultiView6DPoseSnapshot:
    """头+腕 RGB-D 融合后的 6D 位姿快照（默认 base_link，否则主相机系）。"""

    frame_id: str
    position_xyz: _eai_platform_bootstrap.Tuple[float, float, float]
    euler_rpy_rad: _eai_platform_bootstrap.Tuple[float, float, float]
    quaternion_xyzw: _eai_platform_bootstrap.Tuple[float, float, float, float]
    rotation_matrix: _eai_platform_bootstrap.np.ndarray
    obb_extents: _eai_platform_bootstrap.Tuple[float, float, float]
    obb_corners: _eai_platform_bootstrap.np.ndarray
    head_topic: str = ""
    wrist_topic: str = ""
    head_points: int = 0
    wrist_points: int = 0
    fused_points: int = 0
    method: str = "pca-multiview"
    head_result: _eai_platform_bootstrap.Optional[Object6DPoseResult] = None
    wrist_result: _eai_platform_bootstrap.Optional[Object6DPoseResult] = None
    left_wrist_result: _eai_platform_bootstrap.Optional[Object6DPoseResult] = None
    right_wrist_result: _eai_platform_bootstrap.Optional[Object6DPoseResult] = None
    tf_ok: bool = False
    notes: str = ""


def rotation_angle_error_deg(R_est: _eai_platform_bootstrap.np.ndarray, R_gt: _eai_platform_bootstrap.np.ndarray) -> float:
    """测地线旋转误差（度）。"""
    R = _eai_platform_bootstrap.np.asarray(R_est, dtype=_eai_platform_bootstrap.np.float64) @ _eai_platform_bootstrap.np.asarray(R_gt, dtype=_eai_platform_bootstrap.np.float64).T
    cos_theta = (_eai_platform_bootstrap.np.trace(R) - 1.0) * 0.5
    cos_theta = float(_eai_platform_bootstrap.np.clip(cos_theta, -1.0, 1.0))
    return _eai_platform_bootstrap.math.degrees(_eai_platform_bootstrap.math.acos(cos_theta))


def pose_error_metrics(
    est_xyz: _eai_platform_bootstrap.Sequence[float],
    est_R: _eai_platform_bootstrap.np.ndarray,
    gt_xyz: _eai_platform_bootstrap.Sequence[float],
    gt_R: _eai_platform_bootstrap.np.ndarray,
) -> _eai_platform_bootstrap.Tuple[float, float, _eai_platform_bootstrap.Tuple[float, float, float]]:
    """返回 (平移误差 mm, 旋转误差 deg, 平移误差分量 mm)。"""
    d = _eai_platform_bootstrap.np.asarray(est_xyz, dtype=_eai_platform_bootstrap.np.float64) - _eai_platform_bootstrap.np.asarray(gt_xyz, dtype=_eai_platform_bootstrap.np.float64)
    trans_mm = float(_eai_platform_bootstrap.np.linalg.norm(d) * 1000.0)
    rot_deg = rotation_angle_error_deg(est_R, gt_R)
    return trans_mm, rot_deg, (float(d[0] * 1000.0), float(d[1] * 1000.0), float(d[2] * 1000.0))


def euler_xyz_to_rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> _eai_platform_bootstrap.np.ndarray:
    cr, sr = _eai_platform_bootstrap.math.cos(roll), _eai_platform_bootstrap.math.sin(roll)
    cp, sp = _eai_platform_bootstrap.math.cos(pitch), _eai_platform_bootstrap.math.sin(pitch)
    cy, sy = _eai_platform_bootstrap.math.cos(yaw), _eai_platform_bootstrap.math.sin(yaw)
    return _eai_platform_bootstrap.np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=_eai_platform_bootstrap.np.float64,
    )


def transform_points_by_matrix(points: _eai_platform_bootstrap.np.ndarray, T: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
    if points is None or len(points) == 0:
        return _eai_platform_bootstrap.np.empty((0, 3), dtype=_eai_platform_bootstrap.np.float32)
    pts = _eai_platform_bootstrap.np.asarray(points, dtype=_eai_platform_bootstrap.np.float64)
    ones = _eai_platform_bootstrap.np.ones((pts.shape[0], 1), dtype=_eai_platform_bootstrap.np.float64)
    homo = _eai_platform_bootstrap.np.concatenate([pts, ones], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out.astype(_eai_platform_bootstrap.np.float32)

