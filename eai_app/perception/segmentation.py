"""perception.segmentation — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def get_segment_settings() -> _eai_perception_image_geometry.SegmentSettings:
    with _eai_perception_image_geometry._segment_settings_lock:
        return _eai_perception_image_geometry.SegmentSettings(
            backend=_eai_perception_image_geometry._segment_settings.backend,
            sam3_text=_eai_perception_image_geometry._segment_settings.sam3_text,
            sam3_use_http=_eai_perception_image_geometry._segment_settings.sam3_use_http,
            sam3_server_url=_eai_perception_image_geometry._segment_settings.sam3_server_url,
            sam3_python=_eai_perception_image_geometry._segment_settings.sam3_python,
            sam3_model=_eai_perception_image_geometry._segment_settings.sam3_model,
        )


def get_pose_settings() -> _eai_perception_image_geometry.PoseSettings:
    with _eai_perception_image_geometry._pose_settings_lock:
        return _eai_perception_image_geometry.PoseSettings(
            backend=_eai_perception_image_geometry._pose_settings.backend,
            fp_use_http=_eai_perception_image_geometry._pose_settings.fp_use_http,
            fp_server_url=_eai_perception_image_geometry._pose_settings.fp_server_url,
            fp_python=_eai_perception_image_geometry._pose_settings.fp_python,
            fp_mesh=_eai_perception_image_geometry._pose_settings.fp_mesh,
            fp_mode=_eai_perception_image_geometry._pose_settings.fp_mode,
            any6d_use_http=_eai_perception_image_geometry._pose_settings.any6d_use_http,
            any6d_server_url=_eai_perception_image_geometry._pose_settings.any6d_server_url,
            any6d_python=_eai_perception_image_geometry._pose_settings.any6d_python,
        )


def set_segment_settings(**kwargs) -> None:
    with _eai_perception_image_geometry._segment_settings_lock:
        for key, value in kwargs.items():
            if hasattr(_eai_perception_image_geometry._segment_settings, key):
                setattr(_eai_perception_image_geometry._segment_settings, key, value)


def set_pose_settings(**kwargs) -> None:
    with _eai_perception_image_geometry._pose_settings_lock:
        for key, value in kwargs.items():
            if hasattr(_eai_perception_image_geometry._pose_settings, key):
                setattr(_eai_perception_image_geometry._pose_settings, key, value)


def _encode_image_bgr_b64(image_bgr: _eai_platform_bootstrap.np.ndarray) -> str:
    ok, buf = _eai_platform_bootstrap.cv2.imencode(".jpg", image_bgr, [int(_eai_platform_bootstrap.cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("图像 JPEG 编码失败")
    return _eai_platform_bootstrap.base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_sam3_mask_payload(payload: _eai_platform_bootstrap.Dict[str, object]) -> _eai_platform_bootstrap.np.ndarray:
    h = int(payload["h"])  # type: ignore[arg-type]
    w = int(payload["w"])  # type: ignore[arg-type]
    packed = _eai_platform_bootstrap.np.frombuffer(
        _eai_platform_bootstrap.base64.b64decode(str(payload["packed_b64"])), dtype=_eai_platform_bootstrap.np.uint8
    )
    flat = _eai_platform_bootstrap.np.unpackbits(packed)[: h * w]
    return flat.reshape(h, w).astype(bool)


def _summarize_sam3_payload(payload: _eai_platform_bootstrap.Dict[str, object]) -> _eai_platform_bootstrap.Dict[str, object]:
    summary: _eai_platform_bootstrap.Dict[str, object] = {}
    for key, value in payload.items():
        if key == "image_b64":
            summary[key] = f"<base64 {len(str(value))} chars>"
        else:
            summary[key] = value
    return summary


def _summarize_sam3_mask_result(mask: _eai_platform_bootstrap.np.ndarray, method: str) -> _eai_platform_bootstrap.Dict[str, object]:
    return {
        "ok": True,
        "method": method,
        "mask": {
            "h": int(mask.shape[0]),
            "w": int(mask.shape[1]),
            "pixels": int(mask.sum()),
        },
    }


def _log_sam3_request_result(
    tag: str,
    request_summary: _eai_platform_bootstrap.Dict[str, object],
    result_summary: _eai_platform_bootstrap.Dict[str, object],
    elapsed_s: float,
) -> None:
    print(
        f"[show_camera SAM3 {tag}] request: "
        f"{_eai_platform_bootstrap.json.dumps(request_summary, ensure_ascii=False)}",
        file=_eai_platform_bootstrap.sys.stderr,
        flush=True,
    )
    print(
        f"[show_camera SAM3 {tag}] result ({elapsed_s:.3f}s): "
        f"{_eai_platform_bootstrap.json.dumps(result_summary, ensure_ascii=False)}",
        file=_eai_platform_bootstrap.sys.stderr,
        flush=True,
    )


def _sam3_build_request_payload(
    image_bgr: _eai_platform_bootstrap.np.ndarray,
    u: int,
    v: int,
    text: _eai_platform_bootstrap.Optional[str],
    model_path: str,
) -> _eai_platform_bootstrap.Dict[str, object]:
    payload: _eai_platform_bootstrap.Dict[str, object] = {
        "image_b64": _encode_image_bgr_b64(image_bgr),
        "model": model_path,
    }
    if text and text.strip():
        payload["text"] = text.strip()
    else:
        payload["u"] = int(u)
        payload["v"] = int(v)
    return payload


def _sam3_segment_via_http(
    payload: _eai_platform_bootstrap.Dict[str, object],
    server_url: str,
    timeout_s: float,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    url = server_url.rstrip("/") + "/segment"
    req = _eai_platform_bootstrap.urllib.request.Request(
        url,
        data=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
    except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SAM3 HTTP {exc.code}: {detail[:400]}") from exc
    except _eai_platform_bootstrap.urllib.error.URLError as exc:
        raise RuntimeError(
            f"SAM3 服务不可达 ({server_url})，请先运行: "
            f"python3 sam3_segment_worker.py --serve --model sam3.pt"
        ) from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    mask_payload = body.get("mask")
    if not isinstance(mask_payload, dict):
        raise RuntimeError(f"SAM3 响应缺少 mask: {body}")
    return _decode_sam3_mask_payload(mask_payload), str(body.get("method") or "sam3")


def _json_object_from_worker_output(stdout: str) -> dict:
    """Worker 有时会在 JSON 前打印 Ultralytics 警告。取最后一行完整 JSON。"""
    text = (stdout or "").strip()
    if not text:
        raise _eai_platform_bootstrap.json.JSONDecodeError("empty", "", 0)
    try:
        body = _eai_platform_bootstrap.json.loads(text)
    except _eai_platform_bootstrap.json.JSONDecodeError:
        body = None
    if isinstance(body, dict):
        return body
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            body = _eai_platform_bootstrap.json.loads(line)
        except _eai_platform_bootstrap.json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            return body
    start = text.rfind("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        body = _eai_platform_bootstrap.json.loads(text[start : end + 1])
        if isinstance(body, dict):
            return body
    raise _eai_platform_bootstrap.json.JSONDecodeError("no json object", text, 0)


def _sam3_segment_via_subprocess(
    payload: _eai_platform_bootstrap.Dict[str, object],
    python_exe: str,
    timeout_s: float,
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_perception.SAM3_WORKER_SCRIPT):
        raise RuntimeError(f"未找到 worker: {_eai_configuration_perception.SAM3_WORKER_SCRIPT}")
    if not python_exe or not _eai_platform_bootstrap.os.path.isfile(python_exe):
        raise RuntimeError(
            f"找不到 SAM3 的 Python：{python_exe or '（空）'}。"
            "请先创建 /opt/miniconda3/envs/sam3，并把 sam3.pt 放到 eai/models/ 或设置 SAM3_MODEL。"
        )
    worker_env = _eai_platform_bootstrap.os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env.pop("PYTHONHOME", None)
    proc = _eai_platform_bootstrap.subprocess.run(
        [python_exe, _eai_configuration_perception.SAM3_WORKER_SCRIPT, "--once"],
        input=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
        env=worker_env,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0 and not stdout:
        raise RuntimeError(stderr or f"SAM3 worker 退出码 {proc.returncode}")
    try:
        body = _json_object_from_worker_output(stdout)
    except _eai_platform_bootstrap.json.JSONDecodeError as exc:
        detail = stderr or stdout
        raise RuntimeError(f"SAM3 worker 输出非 JSON: {detail[:400]}") from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    mask_payload = body.get("mask")
    if not isinstance(mask_payload, dict):
        raise RuntimeError(f"SAM3 响应缺少 mask: {body}")
    return _decode_sam3_mask_payload(mask_payload), str(body.get("method") or "sam3")


def run_sam3_segmentation(
    image_bgr: _eai_platform_bootstrap.np.ndarray,
    u: int,
    v: int,
    text: _eai_platform_bootstrap.Optional[str] = None,
    settings: _eai_platform_bootstrap.Optional[_eai_perception_image_geometry.SegmentSettings] = None,
    *,
    tag: str = "viewer",
) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, str]:
    cfg = settings or get_segment_settings()
    payload = _sam3_build_request_payload(image_bgr, u, v, text, cfg.sam3_model)
    request_summary = _summarize_sam3_payload(payload)
    request_summary["image_shape"] = [
        int(image_bgr.shape[0]),
        int(image_bgr.shape[1]),
        int(image_bgr.shape[2]),
    ]
    request_summary["transport"] = "HTTP" if cfg.sam3_use_http else "subprocess"
    if cfg.sam3_use_http:
        request_summary["server_url"] = cfg.sam3_server_url
    else:
        request_summary["python"] = cfg.sam3_python
    t0 = _eai_platform_bootstrap.time.time()
    try:
        if cfg.sam3_use_http:
            mask, method = _sam3_segment_via_http(
                payload, cfg.sam3_server_url, _eai_configuration_perception.SAM3_TIMEOUT_S
            )
        else:
            mask, method = _sam3_segment_via_subprocess(
                payload, cfg.sam3_python, _eai_configuration_perception.SAM3_TIMEOUT_S
            )
        result_summary = _summarize_sam3_mask_result(mask, method)
        _log_sam3_request_result(tag, request_summary, result_summary, _eai_platform_bootstrap.time.time() - t0)
        return mask, method
    except Exception as exc:
        _log_sam3_request_result(
            tag,
            request_summary,
            {"ok": False, "error": str(exc)},
            _eai_platform_bootstrap.time.time() - t0,
        )
        raise


@_eai_platform_bootstrap.dataclass
class Sam3CallResult:
    ok: bool
    topic: str = ""
    u: int = 0
    v: int = 0
    text: str = ""
    method: str = ""
    pixel_count: int = 0
    mask_shape: _eai_platform_bootstrap.Tuple[int, int] = (0, 0)
    centroid_uv: _eai_platform_bootstrap.Tuple[float, float] = (0.0, 0.0)
    transport: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None


def format_sam3_call_result_text(result: Sam3CallResult) -> str:
    lines = [
        f"时间: {_eai_platform_bootstrap.time.strftime('%H:%M:%S')}",
        f"图像: {result.topic or '--'}",
    ]
    if result.text:
        lines.append(f"文本提示: {result.text}")
    else:
        lines.append(f"点提示: ({result.u}, {result.v})")
    lines.append(f"传输: {result.transport or '--'}")
    if result.ok:
        lines.extend(
            [
                "状态: 成功",
                f"方法: {result.method}",
                f"mask: {result.mask_shape[1]}x{result.mask_shape[0]}",
                f"像素数: {result.pixel_count}",
                f"质心: ({result.centroid_uv[0]:.1f}, {result.centroid_uv[1]:.1f})",
                f"耗时: {result.elapsed_s:.2f}s",
            ]
        )
    else:
        lines.extend(["状态: 失败", f"错误: {result.error}"])
    return "\n".join(lines)


class Sam3CallBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(object)


@_eai_platform_bootstrap.dataclass
class FpCallResult:
    ok: bool
    color_topic: str = ""
    depth_topic: str = ""
    u: int = 0
    v: int = 0
    mesh: str = ""
    segment_method: str = ""
    pose_method: str = ""
    transport: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
    pose_result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult] = None


def format_fp_call_result_text(result: FpCallResult) -> str:
    lines = [
        f"时间: {_eai_platform_bootstrap.time.strftime('%H:%M:%S')}",
        f"彩色: {result.color_topic or '--'}",
        f"深度: {result.depth_topic or '--'}",
        f"提示点: ({result.u}, {result.v})",
        f"mesh: {result.mesh or '--'}",
        f"传输: {result.transport or '--'}",
    ]
    if result.ok and result.pose_result is not None:
        pose = result.pose_result
        lines.extend(
            [
                "状态: 成功",
                f"分割: {result.segment_method}",
                f"位姿: {result.pose_method}",
                f"mask 像素: {pose.pixel_count}",
                f"点云: {pose.point_count} pts",
                f"耗时: {result.elapsed_s:.2f}s",
                _eai_perception_pose_estimation.format_pose_6d_info(pose, result.u, result.v),
            ]
        )
    elif result.ok:
        lines.extend(["状态: 成功", f"耗时: {result.elapsed_s:.2f}s"])
    else:
        err = format_foundationpose_error(result.error)
        lines.extend(["状态: 失败", f"错误: {err}"])
    return "\n".join(lines)


class FpCallBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(object)


class Mv6dCallBridge(_eai_platform_bootstrap.QObject):
    """头+腕多视角 6D 估计完成信号。"""

    finished = _eai_platform_bootstrap.pyqtSignal(object)


def check_sam3_server_health(server_url: str, timeout_s: float = 2.0) -> bool:
    url = server_url.rstrip("/") + "/health"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def check_ollama_server_health(
    api_base: str = _eai_configuration_perception.OLLAMA_API_BASE_DEFAULT, timeout_s: float = 2.0
) -> bool:
    root = api_base.rstrip("/").removesuffix("/v1").rstrip("/")
    url = f"{root}/api/tags"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_running_in_docker() -> bool:
    return _eai_platform_bootstrap.os.path.isfile("/.dockerenv")


def resolve_sam3_run_script() -> _eai_platform_bootstrap.Optional[str]:
    here = str(PROJECT_ROOT)
    path = _eai_platform_bootstrap.os.path.join(here, _eai_configuration_perception.SAM3_RUN_SCRIPT_NAME)
    return path if _eai_platform_bootstrap.os.path.isfile(path) else None


def resolve_psi_policy_dir(user_dir: _eai_platform_bootstrap.Optional[str] = None) -> _eai_platform_bootstrap.Optional[str]:
    if user_dir:
        path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(user_dir.strip()))
        train_py = _eai_platform_bootstrap.os.path.join(path, "psi_policy", "train.py")
        if _eai_platform_bootstrap.os.path.isfile(train_py):
            return path
    env = _eai_platform_bootstrap.os.environ.get(_eai_configuration_policies.PSI_POLICY_DIR_ENV, "").strip()
    if env and _eai_platform_bootstrap.os.path.isdir(env):
        train_py = _eai_platform_bootstrap.os.path.join(env, "psi_policy", "train.py")
        if _eai_platform_bootstrap.os.path.isfile(train_py):
            return _eai_platform_bootstrap.os.path.abspath(env)
    here = str(PROJECT_ROOT)
    candidates = [
        _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.dirname(here), "psi-policy"),
        _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/psi-policy"),
    ]
    for path in candidates:
        train_py = _eai_platform_bootstrap.os.path.join(path, "psi_policy", "train.py")
        if _eai_platform_bootstrap.os.path.isfile(train_py):
            return _eai_platform_bootstrap.os.path.abspath(path)
    return None


def is_valid_psi_policy_dir(path: str) -> bool:
    if not path or not _eai_platform_bootstrap.os.path.isdir(path):
        return False
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(path, "psi_policy", "train.py"))


def resolve_psi_policy_python(repo_dir: str) -> str:
    env = _eai_platform_bootstrap.os.environ.get("PSI_POLICY_PYTHON", "").strip()
    if env:
        return env
    for cand in (
        "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/psi-policy/bin/python",
        "/home/psibot/miniconda3/envs/psi-policy/bin/python",
    ):
        if _eai_platform_bootstrap.os.path.isfile(cand):
            return cand
    return "python3"


def list_psi_policy_workspace_configs(repo_dir: str) -> _eai_platform_bootstrap.List[str]:
    config_dir = _eai_platform_bootstrap.os.path.join(repo_dir, "psi_policy", "config")
    if not _eai_platform_bootstrap.os.path.isdir(config_dir):
        return []
    names: _eai_platform_bootstrap.List[str] = []
    for fname in sorted(_eai_platform_bootstrap.os.listdir(config_dir)):
        if fname.startswith("example_workspace_") and fname.endswith(".yaml"):
            names.append(fname[:-5])
    return names


def format_hydra_override(key: str, value: str) -> str:
    val = value.strip()
    if not val:
        return ""
    if any(ch in val for ch in ' \t"\'=,'):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key}="{escaped}"'
    return f"{key}={val}"


def resolve_sam3_bind_host() -> str:
    env = _eai_platform_bootstrap.os.environ.get("SAM3_HOST", "").strip()
    if env:
        return env
    return "0.0.0.0" if is_running_in_docker() else "127.0.0.1"


def resolve_sam3_viewer_server_url(port: int = _eai_configuration_perception.SAM3_DEFAULT_PORT) -> str:
    env = _eai_platform_bootstrap.os.environ.get("SAM3_SERVER_URL", "").strip()
    if env:
        return env
    if is_running_in_docker():
        candidates = (
            "host.docker.internal",
            "172.17.0.1",
            "127.0.0.1",
        )
        for host in candidates:
            url = f"http://{host}:{port}"
            if check_sam3_server_health(url, timeout_s=0.6):
                return url
        return f"http://172.17.0.1:{port}"
    return f"http://127.0.0.1:{port}"


def _stop_sam3_server_processes() -> None:
    for pattern in ("sam3_segment_worker.py --serve", "sam3_segment_worker --serve"):
        try:
            _eai_platform_bootstrap.subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=1.0,
                check=False,
            )
        except Exception:
            pass


def _sam3_server_process_running() -> bool:
    return _eai_adapters_robot_services._pgrep_pattern("sam3_segment_worker.py --serve") or _eai_adapters_robot_services._pgrep_pattern(
        "sam3_segment_worker --serve"
    )


def resolve_fp_run_script() -> _eai_platform_bootstrap.Optional[str]:
    here = str(PROJECT_ROOT)
    path = _eai_platform_bootstrap.os.path.join(here, _eai_configuration_perception.FP_RUN_SCRIPT_NAME)
    return path if _eai_platform_bootstrap.os.path.isfile(path) else None


def resolve_fp_viewer_server_url(port: int = _eai_configuration_perception.FP_DEFAULT_PORT) -> str:
    env = _eai_platform_bootstrap.os.environ.get("FP_SERVER_URL", "").strip()
    if env:
        return env
    if is_running_in_docker():
        candidates = ("host.docker.internal", "172.17.0.1", "127.0.0.1")
        for host in candidates:
            url = f"http://{host}:{port}"
            if check_foundationpose_server_health(url, timeout_s=0.6):
                return url
        return f"http://172.17.0.1:{port}"
    return f"http://127.0.0.1:{port}"


def check_foundationpose_server_health(url: str, timeout_s: float = 1.5) -> bool:
    return fetch_foundationpose_health(url, timeout_s).get("online", False)


def format_foundationpose_error(message: str) -> str:
    msg = (message or "").strip()
    if "nvdiffrast" in msg:
        return (
            f"{msg}\n\n"
            "缺少 nvdiffrast（需在宿主机 foundationpose conda 环境安装）。\n"
            "请在本机终端执行：\n"
            "  cd ~/workspace_liyichao/eai\n"
            "  bash run_foundationpose.sh --install-gpu\n"
            "  bash run_foundationpose.sh --mesh "
            "FoundationPose/demo_data/mustard0/mesh/textured_simple.obj\n"
            "然后 viewer 中重试「调用 FP」。"
        )
    if "pytorch3d" in msg:
        return (
            f"{msg}\n\n"
            "请运行: bash run_foundationpose.sh --install-gpu"
        )
    return msg


def fetch_foundationpose_health(url: str, timeout_s: float = 1.5) -> _eai_platform_bootstrap.Dict[str, object]:
    health_url = url.rstrip("/") + "/health"
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(health_url, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
        online = resp.status == 200 and bool(body.get("ok", True))
        fp_ready = bool(body.get("fp_ready", body.get("ok", False)))
        missing_deps = body.get("missing_deps") or []
        install_hints = body.get("install_hints") or []
        return {
            "online": online,
            "fp_ready": fp_ready,
            "foundationpose_root": str(body.get("foundationpose_root") or ""),
            "mesh": str(body.get("mesh") or ""),
            "mesh_resolved": str(body.get("mesh_resolved") or ""),
            "missing_deps": list(missing_deps),
            "install_hints": list(install_hints),
        }
    except Exception as exc:
        return {"online": False, "fp_ready": False, "error": str(exc)}


@_eai_platform_bootstrap.dataclass
class FpAvailabilityStatus:
    label: str
    detail: str
    tooltip: str
    color: str
    can_invoke: bool
    http_online: bool = False
    fp_ready: bool = False
    mesh_ok: bool = False
    worker_ok: bool = False


def evaluate_fp_availability(
    mesh_path: str,
    use_http: bool,
    server_url: str,
    timeout_s: float = 0.8,
) -> FpAvailabilityStatus:
    worker_ok = _eai_platform_bootstrap.os.path.isfile(_eai_configuration_perception.FP_WORKER_SCRIPT)
    local_mesh_ok, server_mesh_ok = _eai_configuration_perception.fp_mesh_available(mesh_path, use_http, server_url)
    mesh_ok = local_mesh_ok or server_mesh_ok
    if use_http:
        health = fetch_foundationpose_health(server_url, timeout_s)
        http_online = bool(health.get("online"))
        fp_ready = bool(health.get("fp_ready"))
        fp_root = str(health.get("foundationpose_root") or "")
        mesh_local_tag = "✓" if local_mesh_ok else "✗"
        mesh_srv_tag = "✓" if server_mesh_ok else "✗"
        checks = [
            f"HTTP: {'✓' if http_online else '✗'}",
            f"FP源码: {'✓' if fp_ready else '✗'}",
            f"mesh本地:{mesh_local_tag}",
            f"mesh服务:{mesh_srv_tag}",
        ]
        detail = "  ".join(checks)
        service_ready = http_online and fp_ready
        can_invoke = service_ready
        if service_ready and mesh_ok:
            return FpAvailabilityStatus(
                label="可用",
                detail=detail,
                tooltip=f"FoundationPose 可调用\n{detail}",
                color=_eai_configuration_policies.UI_ACCENT_GREEN,
                can_invoke=True,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        if service_ready:
            worker_mesh = str(health.get("mesh_resolved") or health.get("mesh") or "")
            return FpAvailabilityStatus(
                label="可用",
                detail=detail,
                tooltip=(
                    "服务已就绪，可点击「调用 FP」尝试。\n"
                    "viewer 本地未找到 mesh（Docker 内常见）；"
                    "将使用 worker 启动时指定的 mesh。\n"
                    + (f"worker mesh: {worker_mesh}\n" if worker_mesh else "")
                    + "若未配置 worker mesh，调用时会报错。"
                ),
                color=_eai_configuration_policies.UI_ACCENT_GREEN,
                can_invoke=True,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        if http_online:
            missing = health.get("missing_deps") or []
            hints = health.get("install_hints") or []
            hint_text = "\n".join(hints) if hints else (
                "bash run_foundationpose.sh --install-gpu"
            )
            missing_text = ", ".join(missing) if missing else "未知"
            return FpAvailabilityStatus(
                label="未就绪",
                detail=detail,
                tooltip=(
                    "HTTP 服务已启动，但 FoundationPose 依赖未就绪。\n"
                    f"缺少: {missing_text}\n"
                    f"检测: {fp_root or '未找到 eai/FoundationPose'}\n\n"
                    f"请执行:\n{hint_text}"
                ),
                color=_eai_configuration_policies.UI_ACCENT_ORANGE,
                can_invoke=False,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        err = str(health.get("error") or "")
        return FpAvailabilityStatus(
            label="离线",
            detail=detail,
            tooltip=(
                f"无法连接 {server_url}\n"
                "宿主机请运行: bash run_foundationpose.sh --host 0.0.0.0\n"
                + (f"错误: {err}" if err else "")
            ),
            color=_eai_configuration_policies.UI_ACCENT_RED,
            can_invoke=False,
            http_online=http_online,
            fp_ready=fp_ready,
            mesh_ok=mesh_ok,
            worker_ok=worker_ok,
        )

    checks = [
        f"worker: {'✓' if worker_ok else '✗'}",
        f"mesh本地:{'✓' if local_mesh_ok else '✗'}",
    ]
    detail = "  ".join(checks)
    can_invoke = worker_ok
    if can_invoke and local_mesh_ok:
        return FpAvailabilityStatus(
            label="可用(本地)",
            detail=detail,
            tooltip=f"子进程模式可调用\n{detail}",
            color=_eai_configuration_policies.UI_TEXT_SECONDARY,
            can_invoke=True,
            mesh_ok=True,
            worker_ok=worker_ok,
        )
    if can_invoke:
        return FpAvailabilityStatus(
            label="可用(本地)",
            detail=detail,
            tooltip=(
                "子进程 worker 可用，可点击尝试。\n"
                "本地未找到 mesh 时将把路径传给 worker 解析。"
            ),
            color=_eai_configuration_policies.UI_TEXT_SECONDARY,
            can_invoke=True,
            mesh_ok=local_mesh_ok,
            worker_ok=worker_ok,
        )
    if not worker_ok:
        return FpAvailabilityStatus(
            label="无 worker",
            detail=detail,
            tooltip=f"未找到 {_eai_configuration_perception.FP_WORKER_SCRIPT}",
            color=_eai_configuration_policies.UI_ACCENT_ORANGE,
            can_invoke=False,
            mesh_ok=local_mesh_ok,
            worker_ok=worker_ok,
        )
    return FpAvailabilityStatus(
        label="离线",
        detail=detail,
        tooltip="子进程模式不可用",
        color=_eai_configuration_policies.UI_ACCENT_RED,
        can_invoke=False,
        mesh_ok=local_mesh_ok,
        worker_ok=worker_ok,
    )


def _fp_server_process_running() -> bool:
    return _eai_adapters_robot_services._pgrep_pattern("foundationpose_worker.py --serve") or _eai_adapters_robot_services._pgrep_pattern(
        "foundationpose_worker --serve"
    )


def _encode_depth_b64(depth: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[str, int, int, str]:
    depth_m = _eai_perception_image_geometry.depth_to_meters_array(depth)
    h, w = depth_m.shape[:2]
    raw = depth_m.astype(_eai_platform_bootstrap.np.float32).tobytes()
    return _eai_platform_bootstrap.base64.b64encode(raw).decode("ascii"), h, w, "float32"


def _encode_mask_payload(mask: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Dict[str, object]:
    flat = mask.reshape(-1).astype(bool)
    packed = _eai_platform_bootstrap.np.packbits(flat)
    return {
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "packed_b64": _eai_platform_bootstrap.base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _summarize_fp_payload(payload: _eai_platform_bootstrap.Dict[str, object]) -> _eai_platform_bootstrap.Dict[str, object]:
    summary: _eai_platform_bootstrap.Dict[str, object] = {}
    for key, value in payload.items():
        if key in ("rgb_b64", "depth_b64"):
            summary[key] = f"<base64 {len(str(value))} chars>"
        elif key == "mask" and isinstance(value, dict):
            summary[key] = {
                "h": value.get("h"),
                "w": value.get("w"),
                "packed_b64": f"<base64 {len(str(value.get('packed_b64', '')))} chars>",
            }
        else:
            summary[key] = value
    return summary


def _log_fp_request_result(
    tag: str,
    request_summary: _eai_platform_bootstrap.Dict[str, object],
    result_summary: _eai_platform_bootstrap.Dict[str, object],
    elapsed_s: float,
) -> None:
    print(
        f"[show_camera FP {tag}] request: "
        f"{_eai_platform_bootstrap.json.dumps(request_summary, ensure_ascii=False)}",
        file=_eai_platform_bootstrap.sys.stderr,
        flush=True,
    )
    print(
        f"[show_camera FP {tag}] result ({elapsed_s:.3f}s): "
        f"{_eai_platform_bootstrap.json.dumps(result_summary, ensure_ascii=False)}",
        file=_eai_platform_bootstrap.sys.stderr,
        flush=True,
    )


def _fp_build_request_payload(
    color_bgr: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    mask: _eai_platform_bootstrap.np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mesh_path: str,
    mode: str = "register",
) -> _eai_platform_bootstrap.Dict[str, object]:
    depth_b64, dh, dw, depth_dtype = _encode_depth_b64(depth)
    return {
        "rgb_b64": _encode_image_bgr_b64(color_bgr),
        "depth_b64": depth_b64,
        "depth_h": dh,
        "depth_w": dw,
        "depth_dtype": depth_dtype,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "mask": _encode_mask_payload(mask),
        "mesh": mesh_path,
        "mode": mode,
    }


def _fp_pose_via_http(
    payload: _eai_platform_bootstrap.Dict[str, object],
    server_url: str,
    timeout_s: float,
) -> _eai_platform_bootstrap.Dict[str, object]:
    url = server_url.rstrip("/") + "/pose"
    req = _eai_platform_bootstrap.urllib.request.Request(
        url,
        data=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
    except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            err_body = _eai_platform_bootstrap.json.loads(detail)
            err_msg = str(err_body.get("error") or detail)
        except _eai_platform_bootstrap.json.JSONDecodeError:
            err_msg = detail
        raise RuntimeError(
            format_foundationpose_error(f"FoundationPose HTTP {exc.code}: {err_msg[:400]}")
        ) from exc
    except _eai_platform_bootstrap.urllib.error.URLError as exc:
        raise RuntimeError(
            f"FoundationPose 服务不可达 ({server_url})，请先运行: "
            f"bash run_foundationpose.sh --host 0.0.0.0 --mesh <object.obj>"
        ) from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def _fp_pose_via_subprocess(
    payload: _eai_platform_bootstrap.Dict[str, object],
    python_exe: str,
    timeout_s: float,
) -> _eai_platform_bootstrap.Dict[str, object]:
    if not _eai_platform_bootstrap.os.path.isfile(_eai_configuration_perception.FP_WORKER_SCRIPT):
        raise RuntimeError(f"未找到 worker: {_eai_configuration_perception.FP_WORKER_SCRIPT}")
    proc = _eai_platform_bootstrap.subprocess.run(
        [python_exe, _eai_configuration_perception.FP_WORKER_SCRIPT, "--once"],
        input=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or stdout or f"FoundationPose worker 退出码 {proc.returncode}")
    try:
        body = _eai_platform_bootstrap.json.loads(stdout)
    except _eai_platform_bootstrap.json.JSONDecodeError as exc:
        raise RuntimeError(f"FoundationPose worker 输出非 JSON: {stdout[:200]}") from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def run_foundationpose_estimation(
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
    cfg = settings or get_pose_settings()
    mesh_path, mesh_source = _eai_configuration_perception.resolve_fp_mesh_for_call(
        cfg.fp_mesh.strip() or _eai_configuration_models.FP_MESH_DEFAULT,
        cfg.fp_use_http,
        cfg.fp_server_url,
    )
    payload = _fp_build_request_payload(
        color_bgr, depth, mask, fx, fy, cx, cy, mesh_path, mode=cfg.fp_mode
    )
    request_summary = _summarize_fp_payload(payload)
    request_summary["mesh_source"] = mesh_source
    request_summary["transport"] = "HTTP" if cfg.fp_use_http else "subprocess"
    if cfg.fp_use_http:
        request_summary["server_url"] = cfg.fp_server_url
    else:
        request_summary["python"] = cfg.fp_python
    t0 = _eai_platform_bootstrap.time.time()
    try:
        if cfg.fp_use_http:
            body = _fp_pose_via_http(payload, cfg.fp_server_url, _eai_configuration_perception.FP_TIMEOUT_S)
        else:
            body = _fp_pose_via_subprocess(payload, cfg.fp_python, _eai_configuration_perception.FP_TIMEOUT_S)
        result_summary = {
            "ok": True,
            "method": body.get("method"),
            "position_xyz": body.get("position_xyz"),
        }
        _log_fp_request_result(tag, request_summary, result_summary, _eai_platform_bootstrap.time.time() - t0)
        return body
    except Exception as exc:
        _log_fp_request_result(
            tag,
            request_summary,
            {"ok": False, "error": str(exc)},
            _eai_platform_bootstrap.time.time() - t0,
        )
        raise


def object6d_from_foundationpose(
    mask: _eai_platform_bootstrap.np.ndarray,
    depth: _eai_platform_bootstrap.np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fp_body: _eai_platform_bootstrap.Dict[str, object],
    segment_method: str = "",
) -> _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult]:
    pose_flat = fp_body.get("pose_matrix")
    if not isinstance(pose_flat, list) or len(pose_flat) != 16:
        return None
    pose = _eai_platform_bootstrap.np.asarray(pose_flat, dtype=_eai_platform_bootstrap.np.float64).reshape(4, 4)
    rotation = pose[:3, :3].astype(_eai_platform_bootstrap.np.float32)
    mesh_origin = pose[:3, 3]
    rpy = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(rotation)
    quat = _eai_perception_pose_estimation.rotation_matrix_to_quaternion(rotation)

    corners_flat = fp_body.get("obb_corners")
    if isinstance(corners_flat, list) and len(corners_flat) == 24:
        corners = _eai_platform_bootstrap.np.asarray(corners_flat, dtype=_eai_platform_bootstrap.np.float32).reshape(8, 3)
        # OBB 几何中心（mesh 原点经 to_origin 后可能偏离包围盒中心）
        position = corners.mean(axis=0).astype(_eai_platform_bootstrap.np.float64)
    else:
        position = mesh_origin
        half = _eai_platform_bootstrap.np.array(fp_body.get("obb_extents") or [0.05, 0.05, 0.05], dtype=_eai_platform_bootstrap.np.float32) * 0.5
        corners = _eai_perception_pose_estimation.obb_corners(position, rotation, half)

    depth_m = _eai_perception_image_geometry.depth_to_meters_array(depth)
    refined_mask, points = _eai_perception_pose_estimation.refine_stereo_segment(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy
    )
    us = (points[:, 0] * fx / points[:, 2] + cx) if len(points) else _eai_platform_bootstrap.np.array([seed_u])
    vs = (points[:, 1] * fy / points[:, 2] + cy) if len(points) else _eai_platform_bootstrap.np.array([seed_v])
    centroid_uv = (float(_eai_platform_bootstrap.np.median(us)), float(_eai_platform_bootstrap.np.median(vs)))
    contact_xyz = _eai_perception_image_geometry.compute_contact_point_3d(
        seed_u, seed_v, depth, fx, fy, cx, cy, points_3d=points
    )
    if contact_xyz is None:
        contact_xyz = (float(position[0]), float(position[1]), float(position[2]))

    obb_extents = fp_body.get("obb_extents")
    if isinstance(obb_extents, list) and len(obb_extents) == 3:
        extents = tuple(float(x) for x in obb_extents)
    else:
        extents = (
            float(_eai_platform_bootstrap.np.linalg.norm(corners[0] - corners[6])),
            float(_eai_platform_bootstrap.np.linalg.norm(corners[1] - corners[3])),
            float(_eai_platform_bootstrap.np.linalg.norm(corners[0] - corners[4])),
        )

    method = str(fp_body.get("method") or "foundationpose")
    if segment_method:
        method = f"{method}+{segment_method}"

    return _eai_perception_pose_estimation.Object6DPoseResult(
        mask=refined_mask,
        points_3d=points,
        centroid_uv=centroid_uv,
        contact_uv=(float(seed_u), float(seed_v)),
        contact_xyz=contact_xyz,
        position_xyz=(float(position[0]), float(position[1]), float(position[2])),
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=rotation,
        obb_center=(float(position[0]), float(position[1]), float(position[2])),
        obb_extents=extents,
        obb_corners=corners,
        depth_m=float(mesh_origin[2]),
        pixel_count=int(refined_mask.sum()),
        point_count=len(points),
        method=method,
    )

