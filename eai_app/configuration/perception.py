"""configuration.perception — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

try:
    from eai_app.adapters.ros.head_camera_tf import attach_to_node as attach_head_camera_tf
except Exception:  # pragma: no cover - optional helper
    attach_head_camera_tf = None  # type: ignore


ENABLE_STATE_QOS = _eai_platform_bootstrap.QoSProfile(
    history=_eai_platform_bootstrap.HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=_eai_platform_bootstrap.ReliabilityPolicy.RELIABLE,
    durability=_eai_platform_bootstrap.DurabilityPolicy.TRANSIENT_LOCAL,
)


IMAGE_TYPES = {
    "sensor_msgs/msg/Image": _eai_platform_bootstrap.Image,
    "sensor_msgs/Image": _eai_platform_bootstrap.Image,
    "sensor_msgs/msg/CompressedImage": _eai_platform_bootstrap.CompressedImage,
    "sensor_msgs/CompressedImage": _eai_platform_bootstrap.CompressedImage,
}


DEFAULT_MAX_DEPTH_M = 3.0


DEPTH_STRIDE = 4


DEPTH_DISPLAY_STRIDE = 8


DEPTH_PREVIEW_MAX_W = 480


MAX_GL_DEPTH_POINTS = 8000


SEGMENT_DEPTH_TOL_M = 0.025


SEGMENT_COLOR_TOL = 32.0


SEGMENT_MIN_AREA = 40


SEGMENT_3D_EPS_M = 0.018


SEGMENT_3D_MIN_POINTS = 30


SEGMENT_ROI_RADIUS = 120


SEGMENT_MAX_3D_POINTS = 4000


SEGMENT_COMPUTE_MAX_DIM = 640


SEGMENT_OVERLAY_MAX_DIM = 960


SAM3_BACKEND_GEOMETRY = "geometry"


SAM3_BACKEND_POINT = "sam3_point"


SAM3_BACKEND_TEXT = "sam3_text"


SAM3_SEGMENT_BACKENDS = (
    SAM3_BACKEND_GEOMETRY,
    SAM3_BACKEND_POINT,
    SAM3_BACKEND_TEXT,
)


SAM3_WORKER_SCRIPT = _eai_platform_bootstrap.os.path.join(
    str(PROJECT_ROOT), "compat/sam3_segment_worker.py"
)


def resolve_sam3_python() -> str:
    """找 sam3 这个 conda 环境的解释器。环境变量优先，其次本机 conda。"""
    env = _eai_platform_bootstrap.os.environ.get("SAM3_PYTHON", "").strip()
    if env:
        expanded = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env))
        if _eai_platform_bootstrap.os.path.isfile(expanded):
            return expanded
    candidates = [
        "/opt/miniconda3/envs/sam3/bin/python",
        _eai_platform_bootstrap.os.path.expanduser("~/miniconda3/envs/sam3/bin/python"),
        "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/sam3/bin/python",
    ]
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path):
            return path
    if env:
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env))
    return candidates[0]


SAM3_PYTHON_DEFAULT = resolve_sam3_python()


def resolve_sam3_model_path() -> str:
    env = _eai_platform_bootstrap.os.environ.get("SAM3_MODEL", "").strip()
    if env and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.expanduser(env)):
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env))
    here = str(PROJECT_ROOT)
    candidates = [
        _eai_platform_bootstrap.os.path.join(here, "models", "sam3.pt"),
        _eai_platform_bootstrap.os.path.join(here, "sam3.pt"),
        _eai_platform_bootstrap.os.path.join(
            here,
            ".cache",
            "modelscope",
            "models",
            "facebook--sam3",
            "snapshots",
            "master",
            "sam3.pt",
        ),
    ]
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env or _eai_platform_bootstrap.os.path.join(here, "models", "sam3.pt")))


SAM3_MODEL_DEFAULT = resolve_sam3_model_path()


SAM3_SERVER_URL_DEFAULT = _eai_platform_bootstrap.os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:8765")


SAM3_USE_HTTP_DEFAULT = _eai_platform_bootstrap.os.environ.get("SAM3_USE_HTTP", "0").strip() in ("1", "true", "yes")


SAM3_TIMEOUT_S = float(_eai_platform_bootstrap.os.environ.get("SAM3_TIMEOUT_S", "120"))


SAM3_RUN_SCRIPT_NAME = "scripts/launch/run_sam3.sh"


SAM3_DEFAULT_PORT = int(_eai_platform_bootstrap.os.environ.get("SAM3_PORT", "8765"))


OLLAMA_API_BASE_DEFAULT = "http://127.0.0.1:11434/v1"


POSE_BACKEND_PCA = "pca"


POSE_BACKEND_FOUNDATIONPOSE = "foundationpose"


POSE_BACKEND_ANY6D = "any6d"


POSE_BACKENDS = (POSE_BACKEND_PCA, POSE_BACKEND_FOUNDATIONPOSE, POSE_BACKEND_ANY6D)


FP_WORKER_SCRIPT = _eai_platform_bootstrap.os.path.join(
    str(PROJECT_ROOT), "compat/foundationpose_worker.py"
)


FP_RUN_SCRIPT_NAME = "scripts/launch/run_foundationpose.sh"


def resolve_fp_python() -> str:
    """找 foundationpose 环境的解释器。环境变量优先，其次本机 conda/venv。"""
    env = _eai_platform_bootstrap.os.environ.get("FP_PYTHON", "").strip()
    if env and env not in ("python3", "python"):
        expanded = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env))
        if _eai_platform_bootstrap.os.path.isfile(expanded) and _eai_platform_bootstrap.os.access(expanded, _eai_platform_bootstrap.os.X_OK):
            return expanded
    candidates = [
        "/opt/miniconda3/envs/foundationpose/bin/python",
        _eai_platform_bootstrap.os.path.expanduser("~/miniconda3/envs/foundationpose/bin/python"),
        "/home/psibot/miniconda3/envs/foundationpose/bin/python",
        _eai_platform_bootstrap.os.path.expanduser("~/anaconda3/envs/foundationpose/bin/python"),
    ]
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return path
    if env:
        return env
    return _eai_platform_bootstrap.shutil.which("python3") or "python3"


FP_PYTHON_DEFAULT = resolve_fp_python()


FP_SERVER_URL_DEFAULT = _eai_platform_bootstrap.os.environ.get("FP_SERVER_URL", "http://127.0.0.1:8766")


FP_USE_HTTP_DEFAULT = _eai_platform_bootstrap.os.environ.get("FP_USE_HTTP", "0").strip() in ("1", "true", "yes")


FP_TIMEOUT_S = float(_eai_platform_bootstrap.os.environ.get("FP_TIMEOUT_S", "180"))


FP_DEFAULT_PORT = int(_eai_platform_bootstrap.os.environ.get("FP_PORT", "8766"))


ANY6D_WORKER_SCRIPT = _eai_platform_bootstrap.os.path.join(
    str(PROJECT_ROOT), "compat/any6d_worker.py"
)


ANY6D_RUN_SCRIPT_NAME = "scripts/launch/run_any6d.sh"


def resolve_any6d_python() -> str:
    """找 any6d 环境的解释器。环境变量优先，其次本机 conda/venv。"""
    env = _eai_platform_bootstrap.os.environ.get("ANY6D_PYTHON", "").strip()
    if env and env not in ("python3", "python"):
        expanded = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env))
        if _eai_platform_bootstrap.os.path.isfile(expanded) and _eai_platform_bootstrap.os.access(expanded, _eai_platform_bootstrap.os.X_OK):
            return expanded
    candidates = [
        "/opt/miniconda3/envs/any6d/bin/python",
        "/opt/miniconda3/envs/Any6D/bin/python",
        _eai_platform_bootstrap.os.path.expanduser("~/miniconda3/envs/any6d/bin/python"),
        _eai_platform_bootstrap.os.path.expanduser("~/miniconda3/envs/Any6D/bin/python"),
        "/home/psibot/miniconda3/envs/any6d/bin/python",
        "/home/psibot/miniconda3/envs/Any6D/bin/python",
    ]
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return path
    if env:
        return env
    return _eai_platform_bootstrap.shutil.which("python3") or "python3"


ANY6D_PYTHON_DEFAULT = resolve_any6d_python()


ANY6D_SERVER_URL_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "ANY6D_SERVER_URL", "http://127.0.0.1:8767"
)


ANY6D_USE_HTTP_DEFAULT = _eai_platform_bootstrap.os.environ.get("ANY6D_USE_HTTP", "0").strip() in (
    "1",
    "true",
    "yes",
)


ANY6D_TIMEOUT_S = float(_eai_platform_bootstrap.os.environ.get("ANY6D_TIMEOUT_S", "240"))


ANY6D_DEFAULT_PORT = int(_eai_platform_bootstrap.os.environ.get("ANY6D_PORT", "8767"))


def fp_mesh_path_exists(mesh_path: str) -> bool:
    return resolve_fp_mesh_absolute(mesh_path) is not None


def resolve_fp_mesh_absolute(mesh_path: str) -> _eai_platform_bootstrap.Optional[str]:
    path = (mesh_path or "").strip()
    if not path:
        path = resolve_fp_mesh_path()
    here = str(PROJECT_ROOT)
    candidates = [_eai_platform_bootstrap.os.path.expanduser(path)]
    if not _eai_platform_bootstrap.os.path.isabs(path):
        candidates.append(_eai_platform_bootstrap.os.path.join(here, path))
    for candidate in candidates:
        if _eai_platform_bootstrap.os.path.isfile(candidate):
            return _eai_platform_bootstrap.os.path.abspath(candidate)
    return None


def resolve_fp_mesh_for_call(
    mesh_path: str,
    use_http: bool,
    server_url: str,
) -> _eai_platform_bootstrap.Tuple[str, str]:
    """解析调用时传给 worker 的 mesh 路径；Docker viewer 可用 worker 侧默认 mesh。"""
    local = resolve_fp_mesh_absolute(mesh_path)
    if local:
        return local, "local"
    if use_http:
        health = _eai_perception_segmentation.fetch_foundationpose_health(server_url, timeout_s=0.8)
        worker_mesh = str(health.get("mesh_resolved") or "").strip()
        if worker_mesh:
            return worker_mesh, "worker"
        worker_default = str(health.get("mesh") or "").strip()
        if worker_default and worker_default != "not_set":
            return worker_default, "worker_default"
    stripped = (mesh_path or "").strip()
    if stripped:
        return stripped, "path"
    return resolve_fp_mesh_path(), "default"


def fp_mesh_available(mesh_path: str, use_http: bool, server_url: str) -> _eai_platform_bootstrap.Tuple[bool, bool]:
    """返回 (本地 mesh 可用, 服务/worker mesh 可用)。"""
    local_ok = resolve_fp_mesh_absolute(mesh_path) is not None
    worker_ok = False
    if use_http:
        health = _eai_perception_segmentation.fetch_foundationpose_health(server_url, timeout_s=0.8)
        worker_ok = bool(health.get("mesh_ok")) or bool(
            str(health.get("mesh_resolved") or "").strip()
        )
    return local_ok, worker_ok


def resolve_fp_mesh_path() -> str:
    env = _eai_platform_bootstrap.os.environ.get("FP_MESH", "").strip()
    here = str(PROJECT_ROOT)
    if env:
        for candidate in (
            _eai_platform_bootstrap.os.path.expanduser(env),
            _eai_platform_bootstrap.os.path.join(here, env),
        ):
            if _eai_platform_bootstrap.os.path.isfile(candidate):
                return _eai_platform_bootstrap.os.path.abspath(candidate)
    # 方案 A：优先使用 meshes/<name>/reconstructed.obj（按修改时间最新）
    meshes_root = _eai_platform_bootstrap.os.path.join(here, "meshes")
    if _eai_platform_bootstrap.os.path.isdir(meshes_root):
        recon: list[tuple[float, str]] = []
        for name in _eai_platform_bootstrap.os.listdir(meshes_root):
            path = _eai_platform_bootstrap.os.path.join(meshes_root, name, "reconstructed.obj")
            if _eai_platform_bootstrap.os.path.isfile(path):
                recon.append((_eai_platform_bootstrap.os.path.getmtime(path), path))
        if recon:
            recon.sort(key=lambda x: x[0], reverse=True)
            return _eai_platform_bootstrap.os.path.abspath(recon[0][1])
    candidates = [
        _eai_platform_bootstrap.os.path.join(
            here,
            "FoundationPose",
            "demo_data",
            "mustard0",
            "mesh",
            "textured_simple.obj",
        ),
        _eai_platform_bootstrap.os.path.join(here, "demo_data", "mustard0", "mesh", "textured_simple.obj"),
        _eai_platform_bootstrap.os.path.expanduser("~/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"),
        _eai_platform_bootstrap.os.path.expanduser(
            "~/workspace_liyichao/eai/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"
        ),
    ]
    fp_root = _eai_platform_bootstrap.os.environ.get("FOUNDATIONPOSE_ROOT", "").strip()
    if fp_root:
        candidates.insert(
            0,
            _eai_platform_bootstrap.os.path.join(
                fp_root,
                "demo_data",
                "mustard0",
                "mesh",
                "textured_simple.obj",
            ),
        )
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(env or candidates[-1]))

