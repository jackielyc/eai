#!/usr/bin/env python3
"""
Hy-Embodied-RxBrain-1.0 OpenAI-compatible HTTP worker (VQA path).

Exposes:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions

Requires a local checkout of:
  https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0
and weights:
  https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0

Usage:
  export HY_RXBRAIN_REPO=/path/to/Hy-Embodied-RxBrain-1.0
  export HY_RXBRAIN_CKPT=/path/to/Hy-Embodied-RxBrain-1.0   # local weights dir
  python hy_rxbrain_worker.py --host 0.0.0.0 --port 8090

Or: bash run_hy_rxbrain.sh
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


MODEL_ID = os.environ.get("HY_RXBRAIN_MODEL", "hy-rxbrain")
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False,
    "error": "",
    "model": None,
    "processor": None,
    "device": None,
    "dtype": None,
    "repo": "",
    "ckpt": "",
}


def _log(msg: str) -> None:
    print(f"[hy_rxbrain] {msg}", flush=True)


def resolve_repo(user: Optional[str] = None) -> Optional[str]:
    candidates = []
    if user:
        candidates.append(os.path.abspath(os.path.expanduser(user)))
    env = os.environ.get("HY_RXBRAIN_REPO", "").strip()
    if env:
        candidates.append(os.path.abspath(os.path.expanduser(env)))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(here, "third_party", "Hy-Embodied-RxBrain-1.0"),
            os.path.expanduser("~/Hy-Embodied-RxBrain-1.0"),
            os.path.expanduser("~/workspace_liyichao/Hy-Embodied-RxBrain-1.0"),
        ]
    )
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "vqa_inference.py")):
            return path
    return None


def resolve_ckpt(user: Optional[str] = None) -> Optional[str]:
    candidates = []
    if user:
        candidates.append(os.path.abspath(os.path.expanduser(user)))
    env = os.environ.get("HY_RXBRAIN_CKPT", "").strip()
    if env:
        candidates.append(os.path.abspath(os.path.expanduser(env)))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(here, "weights", "Hy-Embodied-RxBrain-1.0"),
            os.path.expanduser("~/Hy-Embodied-RxBrain-1.0"),
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--tencent--Hy-Embodied-RxBrain-1.0"
            ),
        ]
    )
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        # Hub snapshot or local dir with config
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        snaps = os.path.join(path, "snapshots")
        if os.path.isdir(snaps):
            for name in sorted(os.listdir(snaps), reverse=True):
                snap = os.path.join(snaps, name)
                if os.path.isfile(os.path.join(snap, "config.json")):
                    return snap
    return None


def load_model(repo: str, ckpt: str) -> None:
    import torch
    from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor

    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    _log(f"loading processor from {ckpt}")
    processor = HunYuanVLMoTProcessor.from_pretrained(ckpt, trust_remote_code=True)
    _log(f"loading model from {ckpt} ({dtype}) on {device}")
    model = UnifiedMoTForConditionalGeneration.from_pretrained(ckpt, dtype=dtype)
    maybe_init_generation_path(model, model_load_path=ckpt)
    model.to(device).eval()
    with _LOCK:
        _STATE.update(
            {
                "ready": True,
                "error": "",
                "model": model,
                "processor": processor,
                "device": device,
                "dtype": dtype,
                "repo": repo,
                "ckpt": ckpt,
            }
        )
    _log("model ready")


def _decode_data_url(url: str) -> bytes:
    if not url.startswith("data:"):
        raise ValueError("only data:image/...;base64, is supported")
    _, b64 = url.split(",", 1)
    return base64.b64decode(b64)


def _extract_user_content(messages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Return (question, list of temp image file paths)."""
    question_parts: List[str] = []
    image_paths: List[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            question_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                question_parts.append(str(part.get("text") or ""))
            elif ptype == "image_url":
                image_url = (part.get("image_url") or {}).get("url") or ""
                raw = _decode_data_url(str(image_url))
                fd, path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                with open(path, "wb") as f:
                    f.write(raw)
                image_paths.append(path)
    question = "\n".join(p for p in question_parts if p.strip()).strip()
    return question, image_paths


def run_vqa(question: str, image_paths: List[str], max_new_tokens: int) -> str:
    with _LOCK:
        if not _STATE["ready"]:
            raise RuntimeError(_STATE.get("error") or "model not loaded")
        model = _STATE["model"]
        processor = _STATE["processor"]
        device = _STATE["device"]
        dtype = _STATE["dtype"]
        repo = _STATE["repo"]

    if repo not in sys.path:
        sys.path.insert(0, repo)
    from vqa_inference import answer

    if not image_paths:
        raise RuntimeError("RxBrain VQA 需要至少一张图像（勾选「附带相机图」）")
    if not question:
        question = "Describe the image."
    text = answer(
        model,
        processor,
        image_paths=image_paths,
        question=question,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    return str(text).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "HyRxBrainWorker/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        _log("%s - %s" % (self.address_string(), fmt % args))

    def _send_json(self, code: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            self._send_json(
                200,
                {
                    "ok": bool(_STATE["ready"]),
                    "model": MODEL_ID,
                    "ckpt": _STATE.get("ckpt") or "",
                    "error": _STATE.get("error") or "",
                },
            )
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "owned_by": "tencent",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if path != "/v1/chat/completions":
            self._send_json(404, {"error": f"unknown path {path}"})
            return

        messages = payload.get("messages") or []
        max_tokens = int(payload.get("max_tokens") or 256)
        tmp_paths: List[str] = []
        try:
            question, tmp_paths = _extract_user_content(messages)
            answer_text = run_vqa(question, tmp_paths, max_new_tokens=max_tokens)
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-rxbrain-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload.get("model") or MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer_text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        except Exception as exc:
            detail = f"{exc}\n{traceback.format_exc()}"
            _log(detail)
            self._send_json(500, {"error": str(exc)})
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Hy-Embodied-RxBrain OpenAI worker")
    parser.add_argument("--host", default=os.environ.get("HY_RXBRAIN_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("HY_RXBRAIN_PORT", "8090"))
    )
    parser.add_argument("--repo", default=os.environ.get("HY_RXBRAIN_REPO", ""))
    parser.add_argument("--ckpt", default=os.environ.get("HY_RXBRAIN_CKPT", ""))
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="start HTTP first; load model on first request (default: load now)",
    )
    args = parser.parse_args()

    repo = resolve_repo(args.repo or None)
    ckpt = resolve_ckpt(args.ckpt or None)
    if not repo:
        _log(
            "ERROR: 未找到 Hy-Embodied-RxBrain-1.0 仓库。"
            "请 git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0 "
            "并设置 HY_RXBRAIN_REPO"
        )
        return 1
    if not ckpt:
        _log(
            "ERROR: 未找到本地权重目录。"
            "请 hf download tencent/Hy-Embodied-RxBrain-1.0 --local-dir ./weights/... "
            "并设置 HY_RXBRAIN_CKPT"
        )
        return 1

    _STATE["repo"] = repo
    _STATE["ckpt"] = ckpt
    if not args.lazy:
        try:
            load_model(repo, ckpt)
        except Exception as exc:
            _STATE["error"] = str(exc)
            _log(f"load failed: {exc}")
            traceback.print_exc()
            return 1

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"serving OpenAI-compatible API on http://{args.host}:{args.port}/v1")
    _log(f"repo={repo}")
    _log(f"ckpt={ckpt}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
