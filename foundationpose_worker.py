#!/usr/bin/env python3
"""
FoundationPose 6D 位姿估计 worker（HTTP / 单次 stdin 模式）。

依赖: 已克隆并安装 NVlabs/FoundationPose，设置 FOUNDATIONPOSE_ROOT。

stdin/POST JSON 示例:
  {
    "rgb_b64": "<jpeg>",
    "depth_b64": "<float32 raw bytes b64>",
    "depth_h": 480, "depth_w": 640,
    "fx": 600, "fy": 600, "cx": 320, "cy": 240,
    "mask": {"h": 480, "w": 640, "packed_b64": "..."},
    "mesh": "/path/to/object.obj",
    "mode": "register"
  }

响应:
  {"ok": true, "method": "foundationpose-register", "pose": {...}, "obb_corners": [[x,y,z], ...]}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_ESTIMATORS: Dict[str, object] = {}
_MESH_META: Dict[str, Dict[str, Any]] = {}
_SERVER_DEFAULT_MESH = ""
_FP_ROOT = ""


def _fp_root_candidates() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    raw_candidates = (
        os.environ.get("FOUNDATIONPOSE_ROOT", "").strip(),
        str(script_dir / "FoundationPose"),
        os.path.expanduser("~/FoundationPose"),
        os.path.expanduser("~/workspace_liyichao/FoundationPose"),
        os.path.expanduser("~/workspace_liyichao/eai/FoundationPose"),
    )
    seen: set[str] = set()
    candidates: list[Path] = []
    for item in raw_candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    return candidates


def _resolve_fp_root() -> Path:
    global _FP_ROOT
    if _FP_ROOT:
        return Path(_FP_ROOT)
    for path in _fp_root_candidates():
        if (path / "estimater.py").is_file():
            _FP_ROOT = str(path.resolve())
            return path
    raise RuntimeError(
        "未找到 FoundationPose 源码目录。请 clone NVlabs/FoundationPose 并设置 "
        "FOUNDATIONPOSE_ROOT=/path/to/FoundationPose"
    )


def _probe_fp_root() -> Tuple[bool, str]:
    try:
        root = str(_resolve_fp_root())
    except Exception as exc:
        return False, str(exc)
    missing: list[str] = []
    for mod in ("trimesh", "torch", "nvdiffrast"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"{root}  |  缺少: {', '.join(missing)}"
    return True, root


def _ensure_fp_imports():
    root = _resolve_fp_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


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


def _encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    flat = mask.reshape(-1).astype(bool)
    packed = np.packbits(flat)
    return {
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "packed_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


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


def _get_estimator(mesh_path: str):
    if mesh_path in _ESTIMATORS:
        return _ESTIMATORS[mesh_path], _MESH_META[mesh_path]

    _ensure_fp_imports()
    import trimesh  # noqa: WPS433
    import nvdiffrast.torch as dr  # noqa: WPS433
    from estimater import FoundationPose  # noqa: WPS433
    from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: WPS433
    from learning.training.predict_score import ScorePredictor  # noqa: WPS433

    mesh = trimesh.load(mesh_path, process=False)
    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"无法加载 mesh: {mesh_path}")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=0,
        debug_dir="/tmp/foundationpose_debug",
        glctx=glctx,
    )
    meta = {
        "to_origin": np.asarray(to_origin, dtype=np.float64),
        "extents": np.asarray(extents, dtype=np.float32),
        "local_corners": _bbox_corners_from_extents(np.asarray(extents, dtype=np.float32)),
    }
    _ESTIMATORS[mesh_path] = est
    _MESH_META[mesh_path] = meta
    return est, meta


def _pose_to_response(
    pose: np.ndarray,
    meta: Dict[str, Any],
    method: str,
) -> Dict[str, Any]:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    center_pose = pose @ np.linalg.inv(meta["to_origin"])
    obb_corners = _transform_points(center_pose, meta["local_corners"])
    rotation = pose[:3, :3]
    position = pose[:3, 3]
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
        f"[foundationpose {tag}] request: {json.dumps(req, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[foundationpose {tag}] result ({elapsed_s:.3f}s): "
        f"{json.dumps(res, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )


def pose_request(payload: Dict[str, Any], *, tag: str = "worker") -> Dict[str, Any]:
    t0 = time.time()
    request_summary = _summarize_payload(payload)
    try:
        mesh_path = _resolve_mesh_path(payload.get("mesh"))
        if not mesh_path or not os.path.isfile(mesh_path):
            raise ValueError(f"mesh 文件不存在: {mesh_path or '(未指定)'}")

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

        mode = str(payload.get("mode") or "register").strip().lower()
        est_refine_iter = int(payload.get("est_refine_iter") or 5)
        track_refine_iter = int(payload.get("track_refine_iter") or 2)

        est, meta = _get_estimator(mesh_path)
        request_summary["mesh_resolved"] = mesh_path
        request_summary["depth_shape"] = [int(depth.shape[0]), int(depth.shape[1])]

        if mode == "track":
            pose = est.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=track_refine_iter,
            )
            method = "foundationpose-track"
        else:
            if bool(payload.get("reset")):
                est.pose_last = None
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask.astype(np.uint8),
                iteration=est_refine_iter,
            )
            method = "foundationpose-register"

        result = _pose_to_response(pose, meta, method)
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
    server_version = "FoundationPoseWorker/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            mesh = _SERVER_DEFAULT_MESH or "not_set"
            fp_ready, root = _probe_fp_root()
            mesh_resolved = ""
            mesh_ok = False
            if mesh and mesh != "not_set":
                try:
                    mesh_resolved = _resolve_mesh_path(mesh)
                    mesh_ok = os.path.isfile(mesh_resolved)
                except Exception:
                    mesh_resolved = mesh
            body = json.dumps(
                {
                    "ok": True,
                    "fp_ready": fp_ready,
                    "mesh": mesh,
                    "mesh_resolved": mesh_resolved,
                    "mesh_ok": mesh_ok,
                    "foundationpose_root": root,
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
            client = self.address_string()
            result = pose_request(payload, tag=f"http:{client}")
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
    resolved = _resolve_mesh_path(mesh_path)
    _SERVER_DEFAULT_MESH = resolved
    server = HTTPServer((host, port), _Handler)
    print(
        f"FoundationPose worker listening on http://{host}:{port}  mesh={resolved}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="FoundationPose HTTP worker")
    parser.add_argument("--once", action="store_true", help="单次 stdin/stdout 模式")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--mesh",
        default="",
        help="默认物体 mesh (.obj)，可在请求里覆盖",
    )
    args = parser.parse_args()
    if args.once:
        return run_once()
    if args.serve:
        run_serve(args.host, args.port, args.mesh)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
