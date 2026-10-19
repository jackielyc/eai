#!/usr/bin/env python3
"""
Any6D 6D 位姿估计 worker（HTTP / 单次 stdin 模式）。

任意新型物体：可无 CAD。若请求未给 mesh，则用当前 mask+深度做凸包粗 mesh，
再走 Any6D.register_any6d（尺度+位姿联合精修）。

依赖: 已克隆 taeyeopl/Any6D，设置 ANY6D_ROOT；FoundationPose 权重默认放
models/FoundationPose/weights（由 run_any6d.sh 软链到 Any6D/foundationpose/weights）。

stdin/POST JSON 与 FoundationPose worker 兼容，额外字段:
  {
    "rgb_b64": "...", "depth_b64": "...", "depth_h": H, "depth_w": W,
    "fx","fy","cx","cy", "mask": {...},
    "mesh": "/optional/path.obj",   # 空则深度凸包
    "iteration": 5,
    "model_free": true
  }

响应同 FP: ok / pose_matrix / position_xyz / obb_corners / obb_extents / method
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_ESTIMATORS: Dict[str, object] = {}
_MESH_META: Dict[str, Dict[str, Any]] = {}
_SERVER_DEFAULT_MESH = ""
_ANY6D_ROOT = ""


def _any6d_root_candidates() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    raw = (
        os.environ.get("ANY6D_ROOT", "").strip(),
        str(script_dir / "Any6D"),
        os.path.expanduser("~/Any6D"),
        os.path.expanduser("~/workspace_liyichao/Any6D"),
        os.path.expanduser("~/workspace_liyichao/eai/Any6D"),
    )
    seen: set[str] = set()
    out: list[Path] = []
    for item in raw:
        if not item:
            continue
        path = Path(item).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _resolve_any6d_root() -> Path:
    global _ANY6D_ROOT
    if _ANY6D_ROOT:
        return Path(_ANY6D_ROOT)
    for path in _any6d_root_candidates():
        if (path / "estimater.py").is_file():
            _ANY6D_ROOT = str(path.resolve())
            return path
    raise RuntimeError(
        "未找到 Any6D 源码。请: git clone https://github.com/taeyeopl/Any6D.git eai/Any6D "
        "并设置 ANY6D_ROOT"
    )


def _probe_any6d_dependencies() -> Tuple[bool, str, list[str], list[str]]:
    try:
        root = str(_resolve_any6d_root())
    except Exception as exc:
        return False, str(exc), [], [
            "git clone https://github.com/taeyeopl/Any6D.git eai/Any6D",
            "bash run_any6d.sh --check",
        ]

    missing: list[str] = []
    for import_path, label in (
        ("trimesh", "trimesh"),
        ("torch", "torch"),
        ("open3d", "open3d"),
        ("nvdiffrast.torch", "nvdiffrast"),
    ):
        try:
            __import__(import_path)
        except ImportError:
            missing.append(label)

    hints: list[str] = []
    if missing:
        hints.append("按 Any6D README 创建 conda 环境 Any6D 并安装依赖")
        hints.append("bash run_any6d.sh --check")
    weights = Path(root) / "foundationpose" / "weights"
    models_w = Path(
        os.environ.get(
            "FOUNDATIONPOSE_WEIGHTS",
            "/share_data/projects/mahjong/share/personal/liyichao/models/FoundationPose/weights",
        )
    )
    fp_w = Path(__file__).resolve().parent / "FoundationPose" / "weights"
    # 自动软链: models → Any6D/foundationpose/weights
    if (not weights.exists() or (weights.is_dir() and not any(weights.iterdir()))) and models_w.is_dir():
        try:
            if weights.is_symlink() or weights.exists():
                weights.unlink()
            weights.symlink_to(models_w)
        except OSError as exc:
            hints.append(f"软链权重失败: {exc}")
    need_weights = True
    if weights.is_dir():
        # 需含 scorer + refiner 两个 run 目录
        scorers = list(weights.glob("2024-01-11-20-02-45/model_best.pth"))
        refiners = list(weights.glob("2023-10-28-18-33-37/model_best.pth"))
        need_weights = not (scorers and refiners)
    if need_weights:
        hints.append(
            f"下载 FoundationPose 权重到 {models_w} "
            f"（HF: gpue/foundationpose-weights），或软链到 {weights}"
        )
    return len(missing) == 0, root, missing, hints


def _ensure_any6d_imports() -> Path:
    root = _resolve_any6d_root()
    # Any6D 仓库根需在 path；其内部 import foundationpose.*
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_mesh_path(requested: Any) -> str:
    global _SERVER_DEFAULT_MESH
    req = str(requested or "").strip()
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if req:
        req_path = Path(req).expanduser()
        candidates.append(req_path)
        if not req_path.is_absolute():
            candidates.append(script_dir / req_path)
    if _SERVER_DEFAULT_MESH:
        default_path = Path(_SERVER_DEFAULT_MESH).expanduser()
        candidates.append(default_path)
        if not default_path.is_absolute():
            candidates.append(script_dir / default_path)
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return req


def _decode_image_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码 rgb_b64")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_depth_b64(payload: Dict[str, Any]) -> np.ndarray:
    depth_b64 = payload.get("depth_b64")
    if not depth_b64:
        raise ValueError("缺少 depth_b64")
    h = int(payload["depth_h"])
    w = int(payload["depth_w"])
    dtype = str(payload.get("depth_dtype") or "float32")
    raw = base64.b64decode(str(depth_b64))
    if dtype == "uint16":
        depth = np.frombuffer(raw, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    else:
        depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    return depth


def _decode_mask_payload(payload: Dict[str, Any]) -> np.ndarray:
    mask_obj = payload.get("mask")
    if not isinstance(mask_obj, dict):
        raise ValueError("缺少 mask")
    h, w = int(mask_obj["h"]), int(mask_obj["w"])
    packed = np.frombuffer(base64.b64decode(str(mask_obj["packed_b64"])), dtype=np.uint8)
    flat = np.unpackbits(packed)[: h * w]
    return flat.reshape(h, w).astype(bool)


def _intrinsics_matrix(payload: Dict[str, Any]) -> np.ndarray:
    if "K" in payload and isinstance(payload["K"], list) and len(payload["K"]) == 9:
        return np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
    fx = float(payload["fx"])
    fy = float(payload["fy"])
    cx = float(payload["cx"])
    cy = float(payload["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _bbox_corners_from_extents(extents: np.ndarray) -> np.ndarray:
    mn = -extents / 2.0
    mx = extents / 2.0
    return np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )


def _transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    out = (pose @ hom.T).T
    return out[:, :3].astype(np.float32)


def _coarse_mesh_from_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    max_points: int = 8000,
) -> Tuple[Any, str]:
    """无 CAD 时：mask 内深度点 → 凸包 mesh（model-free 粗形状）。"""
    import trimesh

    ys, xs = np.where(mask)
    if len(xs) < 30:
        raise RuntimeError("mask 内有效像素过少，无法建粗 mesh")
    z = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05) & (z < 3.5)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(z) < 30:
        raise RuntimeError("mask 内有效深度过少，无法建粗 mesh")
    if len(z) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z), max_points, replace=False)
        xs, ys, z = xs[idx], ys[idx], z[idx]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    try:
        mesh = trimesh.convex.convex_hull(pts)
    except Exception:
        # 凸包失败 → 轴对齐盒
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = 0.5 * (mn + mx)
        extents = np.maximum(mx - mn, 0.01)
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
    if mesh is None or len(getattr(mesh, "vertices", [])) < 4:
        raise RuntimeError("深度凸包 mesh 无效")
    # 导出到临时文件供缓存 key / 调试
    tmp = tempfile.NamedTemporaryFile(prefix="any6d_coarse_", suffix=".obj", delete=False)
    tmp_path = tmp.name
    tmp.close()
    mesh.export(tmp_path)
    return mesh, tmp_path


def _mesh_meta_from_trimesh(mesh: Any) -> Dict[str, Any]:
    import trimesh

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    extents = np.asarray(extents, dtype=np.float32)
    return {
        "to_origin": np.asarray(to_origin, dtype=np.float64),
        "extents": extents,
        "local_corners": _bbox_corners_from_extents(extents),
    }


def _get_estimator(mesh_path: str, mesh_obj: Any = None):
    key = mesh_path or id(mesh_obj)
    key_s = str(key)
    if key_s in _ESTIMATORS:
        return _ESTIMATORS[key_s], _MESH_META[key_s]

    _ensure_any6d_imports()
    import trimesh  # noqa: WPS433
    import nvdiffrast.torch as dr  # noqa: WPS433
    from estimater import Any6D  # noqa: WPS433

    if mesh_obj is None:
        mesh = trimesh.load(mesh_path, process=False)
    else:
        mesh = mesh_obj
    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"无法加载 mesh: {mesh_path}")

    meta = _mesh_meta_from_trimesh(mesh)
    debug_dir = os.environ.get("ANY6D_DEBUG_DIR", "/tmp/any6d_debug")
    os.makedirs(debug_dir, exist_ok=True)
    glctx = dr.RasterizeCudaContext()
    est = Any6D(
        symmetry_tfs=None,
        mesh=mesh,
        debug_dir=debug_dir,
        debug=0,
        glctx=glctx,
    )
    _ESTIMATORS[key_s] = est
    _MESH_META[key_s] = meta
    return est, meta


def _pose_to_response(pose: np.ndarray, meta: Dict[str, Any], method: str) -> Dict[str, Any]:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    # Any6D 返回相机系下物体位姿；中心用 to_origin 对齐 OBB
    try:
        center_pose = pose @ np.linalg.inv(meta["to_origin"])
    except Exception:
        center_pose = pose
    obb_corners = _transform_points(center_pose, meta["local_corners"])
    rotation = pose[:3, :3]
    # 位置取 OBB 中心（更接近物体几何中心）
    position = obb_corners.mean(axis=0)
    return {
        "ok": True,
        "method": method,
        "pose_matrix": pose.reshape(-1).tolist(),
        "position_xyz": [float(position[0]), float(position[1]), float(position[2])],
        "rotation_matrix": rotation.reshape(-1).tolist(),
        "obb_corners": obb_corners.reshape(-1).tolist(),
        "obb_extents": [float(x) for x in meta["extents"]],
    }


def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, object] = {}
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


def _log_request_result(tag: str, req: Dict[str, Any], res: Dict[str, Any], elapsed_s: float) -> None:
    print(
        f"[any6d {tag}] request: {json.dumps(req, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[any6d {tag}] result ({elapsed_s:.3f}s): {json.dumps(res, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )


def pose_request(payload: Dict[str, Any], *, tag: str = "worker") -> Dict[str, Any]:
    t0 = time.time()
    request_summary = _summarize_payload(payload)
    try:
        rgb = _decode_image_b64(str(payload["rgb_b64"]))
        depth = _decode_depth_b64(payload)
        mask = _decode_mask_payload(payload)
        K = _intrinsics_matrix(payload)

        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        mesh_path = _resolve_mesh_path(payload.get("mesh"))
        mesh_obj = None
        method_suffix = "mesh"
        if not mesh_path or not os.path.isfile(mesh_path):
            # model-free：深度凸包粗 mesh
            mesh_obj, mesh_path = _coarse_mesh_from_depth(depth, mask, K)
            method_suffix = "depth-hull"
            request_summary["mesh_source"] = "depth_convex_hull"
        else:
            request_summary["mesh_source"] = "file"
        request_summary["mesh_resolved"] = mesh_path

        iteration = int(payload.get("iteration") or payload.get("est_refine_iter") or 5)
        est, meta = _get_estimator(mesh_path, mesh_obj=mesh_obj)
        # 每次新物体建议重建；若 mesh 变了 key 已换
        pose = est.register_any6d(
            K=K,
            rgb=rgb,
            depth=depth.astype(np.float32),
            ob_mask=mask.astype(bool),
            iteration=iteration,
            name=str(payload.get("name") or "query"),
        )
        result = _pose_to_response(pose, meta, f"any6d-{method_suffix}")
        # 更新 extents（register 可能缩放 mesh）
        try:
            if hasattr(est, "mesh") and est.mesh is not None:
                meta2 = _mesh_meta_from_trimesh(est.mesh)
                result = _pose_to_response(pose, meta2, f"any6d-{method_suffix}")
        except Exception:
            pass
        _log_request_result(tag, request_summary, result, time.time() - t0)
        return result
    except Exception as exc:
        error_result = {"ok": False, "error": str(exc)}
        _log_request_result(tag, request_summary, error_result, time.time() - t0)
        raise


def run_once() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "stdin 为空"}), flush=True)
        return 1
    try:
        payload = json.loads(raw)
        result = pose_request(payload, tag="once")
        print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return 1


class _Handler(BaseHTTPRequestHandler):
    server_version = "Any6DWorker/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            ready, root, missing, hints = _probe_any6d_dependencies()
            mesh = _SERVER_DEFAULT_MESH or ""
            mesh_ok = bool(mesh) and os.path.isfile(mesh)
            body = json.dumps(
                {
                    "ok": True,
                    "any6d_ready": ready,
                    "missing_deps": missing,
                    "install_hints": hints,
                    "mesh": mesh or "model-free(depth-hull)",
                    "mesh_ok": mesh_ok or True,  # 无 mesh 也可 model-free
                    "any6d_root": root,
                    "model_free": True,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/pose":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            result = pose_request(payload, tag=f"http:{self.address_string()}")
            body = json.dumps(result).encode("utf-8")
            code = 200
        except Exception as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            code = 400
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_serve(host: str, port: int, mesh_path: str) -> None:
    global _SERVER_DEFAULT_MESH
    resolved = _resolve_mesh_path(mesh_path) if mesh_path else ""
    _SERVER_DEFAULT_MESH = resolved if resolved and os.path.isfile(resolved) else ""
    ready, root, missing, hints = _probe_any6d_dependencies()
    if not ready:
        print(
            f"WARNING: Any6D 依赖未就绪 ({', '.join(missing) or root})",
            file=sys.stderr,
            flush=True,
        )
        for line in hints:
            print(f"  → {line}", file=sys.stderr, flush=True)
    server = HTTPServer((host, port), _Handler)
    print(
        f"Any6D worker listening on http://{host}:{port}  "
        f"mesh={_SERVER_DEFAULT_MESH or '(model-free depth-hull)'}  root={root}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Any6D HTTP worker")
    parser.add_argument("--once", action="store_true", help="单次 stdin/stdout 模式")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--mesh",
        default="",
        help="可选默认 mesh；为空则每次用深度凸包（任意物体）",
    )
    parser.add_argument("--check", action="store_true", help="仅检查依赖")
    args = parser.parse_args()
    if args.check:
        ready, root, missing, hints = _probe_any6d_dependencies()
        print(json.dumps({
            "ok": ready,
            "any6d_root": root,
            "missing": missing,
            "hints": hints,
        }, ensure_ascii=False, indent=2))
        return 0 if ready else 1
    if args.once:
        return run_once()
    if args.serve:
        run_serve(args.host, args.port, args.mesh)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
