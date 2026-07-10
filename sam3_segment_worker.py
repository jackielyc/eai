#!/usr/bin/env python3
"""
SAM3 本地分割 worker（Ultralytics SAM 点提示 / 可选文本概念分割）。

用法:
  # 单次（stdin JSON -> stdout JSON）
  python3 sam3_segment_worker.py --once

  # 常驻 HTTP 服务（模型只加载一次，推荐）
  python3 sam3_segment_worker.py --serve --port 8765 --model sam3.pt

stdin/POST JSON 示例:
  {"image_b64": "<jpeg b64>", "u": 320, "v": 240, "model": "sam3.pt"}
  {"image_b64": "...", "text": "red cup", "model": "sam3.pt"}

stdout/response:
  {"ok": true, "method": "sam3-point", "mask": {"h": 480, "w": 640, "packed_b64": "..."}}
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_MODEL = None
_MODEL_PATH = ""
_SERVER_DEFAULT_MODEL = ""


def _resolve_model_path(requested: Any) -> str:
    """优先使用已加载模型或存在的权重文件，避免请求里写死 sam3.pt。"""
    global _MODEL_PATH, _SERVER_DEFAULT_MODEL
    req = str(requested or "").strip()
    candidates: list[Path] = []
    if req:
        candidates.append(Path(req).expanduser())
    if _MODEL_PATH:
        candidates.append(Path(_MODEL_PATH).expanduser())
    if _SERVER_DEFAULT_MODEL:
        candidates.append(Path(_SERVER_DEFAULT_MODEL).expanduser())
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    if _MODEL is not None and _MODEL_PATH:
        return _MODEL_PATH
    return req or "sam3.pt"


def _load_sam(model_path: str):
    global _MODEL, _MODEL_PATH
    if _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL
    from ultralytics import SAM

    _MODEL = SAM(model_path)
    _MODEL_PATH = model_path
    return _MODEL


def _mask_from_results(results) -> Optional[np.ndarray]:
    if not results:
        return None
    r0 = results[0]
    if r0.masks is None or r0.masks.data is None or len(r0.masks.data) == 0:
        return None
    mask = r0.masks.data[0].detach().cpu().numpy()
    return (mask > 0.5).astype(bool)


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def _encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    flat = mask.reshape(-1).astype(bool)
    packed = np.packbits(flat)
    return {
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "packed_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def decode_mask_payload(payload: Dict[str, Any]) -> np.ndarray:
    h, w = int(payload["h"]), int(payload["w"])
    packed = np.frombuffer(base64.b64decode(payload["packed_b64"]), dtype=np.uint8)
    flat = np.unpackbits(packed)[: h * w]
    return flat.reshape(h, w).astype(bool)


def _decode_image_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码 image_b64")
    return image


def segment_with_point(
    image_bgr: np.ndarray,
    u: int,
    v: int,
    model_path: str,
) -> Tuple[np.ndarray, str]:
    model = _load_sam(model_path)
    h, w = image_bgr.shape[:2]
    u = max(0, min(w - 1, int(u)))
    v = max(0, min(h - 1, int(v)))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
        cv2.imwrite(path, image_bgr)
    try:
        results = model.predict(
            source=path,
            points=[u, v],
            labels=[1],
            verbose=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    mask = _mask_from_results(results)
    if mask is None:
        raise RuntimeError("SAM3 点提示未返回有效 mask")
    return _resize_mask(mask, h, w), "sam3-point"


def segment_with_text(
    image_bgr: np.ndarray,
    text: str,
    model_path: str,
) -> Tuple[np.ndarray, str]:
    h, w = image_bgr.shape[:2]
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 文本分割需要 ultralytics>=8.3.237 且 Python 3.12+"
        ) from exc

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(image_bgr)
    results = predictor(text=[text.strip()])
    if not results:
        raise RuntimeError("SAM3 文本分割无结果")
    mask = _mask_from_results(results)
    if mask is None:
        raise RuntimeError("SAM3 文本分割未返回 mask")
    return _resize_mask(mask, h, w), f"sam3-text:{text.strip()}"


def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "image_b64":
            text = str(value)
            summary["image_b64"] = f"<base64 {len(text)} chars>"
        else:
            summary[key] = value
    return summary


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}
    summary: Dict[str, Any] = {
        "ok": True,
        "method": result.get("method"),
    }
    mask_payload = result.get("mask")
    if isinstance(mask_payload, dict):
        h = int(mask_payload.get("h", 0))
        w = int(mask_payload.get("w", 0))
        try:
            mask = decode_mask_payload(mask_payload)
            summary["mask"] = {
                "h": h,
                "w": w,
                "pixels": int(mask.sum()),
            }
        except Exception as exc:
            summary["mask"] = {"h": h, "w": w, "decode_error": str(exc)}
    return summary


def _log_request_result(
    tag: str,
    request_summary: Dict[str, Any],
    result_summary: Dict[str, Any],
    elapsed_s: float,
) -> None:
    print(
        f"[SAM3 {tag}] request: {json.dumps(request_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[SAM3 {tag}] result ({elapsed_s:.3f}s): "
        f"{json.dumps(result_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )


def segment_request(payload: Dict[str, Any], *, tag: str = "segment") -> Dict[str, Any]:
    t0 = time.time()
    request_summary = _summarize_payload(payload)
    try:
        model_path = _resolve_model_path(payload.get("model"))
        request_summary["model_resolved"] = model_path
        image_b64 = payload.get("image_b64")
        if not image_b64:
            raise ValueError("缺少 image_b64")
        image = _decode_image_b64(str(image_b64))
        h, w = image.shape[:2]
        request_summary["image_shape"] = [int(h), int(w), int(image.shape[2])]
        text = str(payload.get("text") or "").strip()
        if text:
            mask, method = segment_with_text(image, text, model_path)
        else:
            if "u" not in payload or "v" not in payload:
                raise ValueError("点提示需要 u, v；或提供 text")
            mask, method = segment_with_point(
                image, int(payload["u"]), int(payload["v"]), model_path
            )
        if not mask.any():
            raise RuntimeError("SAM3 返回空 mask")
        result = {"ok": True, "method": method, "mask": _encode_mask(mask)}
        _log_request_result(tag, request_summary, _summarize_result(result), time.time() - t0)
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
        result = segment_request(payload, tag="once")
        print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return 1


class _Handler(BaseHTTPRequestHandler):
    server_version = "SAM3SegmentWorker/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            body = json.dumps({"ok": True, "model": _MODEL_PATH or "not_loaded"}).encode(
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/segment":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            client = self.address_string()
            result = segment_request(payload, tag=f"http:{client}")
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


def run_serve(host: str, port: int, model_path: str) -> None:
    global _SERVER_DEFAULT_MODEL
    resolved = _resolve_model_path(model_path)
    _SERVER_DEFAULT_MODEL = resolved
    _load_sam(resolved)
    server = HTTPServer((host, port), _Handler)
    print(
        f"SAM3 worker listening on http://{host}:{port}  model={resolved}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3 local segmentation worker")
    parser.add_argument("--once", action="store_true", help="单次 stdin/stdout 模式")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model",
        default="sam3.pt",
        help="sam3.pt 路径（需先在 HuggingFace 申请并下载）",
    )
    args = parser.parse_args()
    if args.once:
        return run_once()
    if args.serve:
        run_serve(args.host, args.port, args.model)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
