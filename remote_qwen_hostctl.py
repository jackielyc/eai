#!/usr/bin/env python3
"""宿主机侧远程 Qwen 部署控制（供 Docker 内 viewer 调用）。

Docker 容器与宿主机 PID 命名空间隔离：容器内无法可靠杀掉宿主机 ssh 隧道，
且偶发 SSH Host 解析失败。因此 deploy/tunnel/stop 统一在宿主机执行。

监听: http://127.0.0.1:18103
  GET  /health
  GET  /status?host=...
  GET  /list_models?host=...&root=...  （root 可重复）
  POST /deploy  JSON: host_id, model_key | model_dir, model_id, model_label
  POST /tunnel  JSON: host_id
  POST /stop    JSON: host_id
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


HERE = os.path.dirname(os.path.abspath(__file__))
CTL_HOST = os.environ.get("REMOTE_QWEN_CTL_HOST", "127.0.0.1")
CTL_PORT = int(os.environ.get("REMOTE_QWEN_CTL_PORT", "18103"))
LOG_PATH = os.environ.get(
    "REMOTE_QWEN_CTL_LOG",
    os.path.join(HERE, "log", "remote_qwen_hostctl.log"),
)
REMOTE_CTL_PATH = os.path.join(HERE, "remote_qwen_ctl.py")

_LOCK = threading.Lock()
_BUSY = False
_LAST: Dict[str, Any] = {}


def _log(msg: str) -> None:
    line = f"[remote_qwen_hostctl] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_remote_ctl():
    spec = importlib.util.spec_from_file_location("remote_qwen_ctl_host", REMOTE_CTL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {REMOTE_CTL_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_action(
    action: str,
    host_id: str,
    model_key: str = "qwen3.5-35b-a3b",
    *,
    model_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    model_label: Optional[str] = None,
    scan_roots: Optional[list] = None,
    local_scan: bool = False,
) -> Dict[str, Any]:
    global _BUSY, _LAST
    with _LOCK:
        if _BUSY and action in ("deploy", "tunnel", "stop"):
            return {
                "ok": False,
                "message": "宿主机 remote hostctl 正忙，请稍后重试",
                "busy": True,
            }
        _BUSY = True
    try:
        ctl = load_remote_ctl()
        ctl.apply_host(host_id)
        if action == "deploy":
            result = ctl.deploy(
                model_key=model_key,
                model_dir=model_dir,
                model_id=model_id,
                model_label=model_label,
            )
        elif action == "tunnel":
            result = ctl.start_tunnel()
        elif action == "stop":
            result = ctl.stop_all()
        elif action == "stop_tunnel":
            result = ctl.stop_tunnel()
        elif action == "status":
            result = ctl.status_payload()
        elif action == "list_models":
            result = ctl.list_deploy_model_dirs(
                roots=scan_roots,
                host_id=host_id,
                local=local_scan,
            )
        else:
            result = {"ok": False, "message": f"unknown action: {action}"}
        if not isinstance(result, dict):
            result = {"ok": False, "message": str(result)}
        result.setdefault("host_id", host_id)
        result.setdefault("via", "remote_qwen_hostctl")
        _LAST = dict(result)
        _log(f"{action} host={host_id}: {result.get('message') or result.get('ok')}")
        return result
    except Exception as exc:
        err = {"ok": False, "message": str(exc), "host_id": host_id, "via": "remote_qwen_hostctl"}
        _LAST = dict(err)
        _log(f"{action} host={host_id} ERROR: {exc}")
        return err
    finally:
        with _LOCK:
            _BUSY = False


class Handler(BaseHTTPRequestHandler):
    server_version = "RemoteQwenHostCtl/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        _log("%s - %s" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        host_id = (qs.get("host") or qs.get("host_id") or ["psi_motus_2_for_liyichao"])[0]
        if path in ("/health", "/"):
            self._send(
                200,
                {
                    "ok": True,
                    "service": "remote_qwen_hostctl",
                    "busy": _BUSY,
                    "port": CTL_PORT,
                    "features": [
                        "status",
                        "list_models",
                        "deploy",
                        "tunnel",
                        "stop",
                    ],
                },
            )
            return
        if path == "/status":
            self._send(200, run_action("status", host_id))
            return
        if path == "/list_models":
            roots = qs.get("root") or []
            local_scan = (qs.get("local") or ["0"])[0] in ("1", "true", "yes")
            self._send(
                200,
                run_action(
                    "list_models",
                    host_id,
                    scan_roots=roots or None,
                    local_scan=local_scan,
                ),
            )
            return
        if path == "/last":
            self._send(200, {"ok": True, "last": _LAST, "busy": _BUSY})
            return
        self._send(404, {"ok": False, "message": f"unknown {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload = self._read_json_body()
        host_id = str(payload.get("host_id") or payload.get("host") or "psi_motus_2_for_liyichao")
        model_key = str(payload.get("model_key") or "qwen3.5-35b-a3b")
        model_dir = str(payload.get("model_dir") or "").strip() or None
        model_id = str(payload.get("model_id") or "").strip() or None
        model_label = str(payload.get("model_label") or "").strip() or None
        if path == "/deploy":
            body = run_action(
                "deploy",
                host_id,
                model_key,
                model_dir=model_dir,
                model_id=model_id,
                model_label=model_label,
            )
            self._send(200 if body.get("ok") else 500, body)
            return
        if path == "/tunnel":
            body = run_action("tunnel", host_id)
            self._send(200 if body.get("ok") else 500, body)
            return
        if path == "/stop":
            body = run_action("stop", host_id)
            self._send(200 if body.get("ok") else 500, body)
            return
        if path == "/stop_tunnel":
            body = run_action("stop_tunnel", host_id)
            self._send(200 if body.get("ok") else 500, body)
            return
        self._send(404, {"ok": False, "message": f"unknown {path}"})


def already_running() -> bool:
    try:
        with urlopen(f"http://{CTL_HOST}:{CTL_PORT}/health", timeout=0.8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def main() -> int:
    if already_running():
        _log(f"already listening on http://{CTL_HOST}:{CTL_PORT}")
        return 0
    if not os.path.isfile(REMOTE_CTL_PATH):
        _log(f"missing {REMOTE_CTL_PATH}")
        return 1
    server = ThreadingHTTPServer((CTL_HOST, CTL_PORT), Handler)
    _log(f"listening on http://{CTL_HOST}:{CTL_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
