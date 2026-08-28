#!/usr/bin/env python3
"""
宿主机侧本地 Qwen 启停控制（供 Docker 内 viewer 调用）。

监听: http://127.0.0.1:18101
  GET  /health
  GET  /status
  POST /start   JSON 可选: model_dir / model_id / model_label
  POST /stop

由 run_in_docker.sh 在宿主机后台拉起；viewer 在 host 网络下直接访问 127.0.0.1。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import urlopen


HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(HERE)
RUN_SCRIPT = os.path.join(HERE, "run_local_qwen.sh")
CTL_HOST = os.environ.get("LOCAL_QWEN_CTL_HOST", "127.0.0.1")
CTL_PORT = int(os.environ.get("LOCAL_QWEN_CTL_PORT", "18101"))
QWEN_PORT = int(os.environ.get("LOCAL_QWEN_PORT", "8100"))
LOG_PATH = os.environ.get(
    "LOCAL_QWEN_CTL_LOG",
    os.path.join(HERE, "log", "local_qwen_service.log"),
)
DEFAULT_MODEL_DIR = os.path.join(WORKSPACE_DIR, "models", "Qwen", "Qwen3.5-4B")
DEFAULT_MODEL_ID = "qwen3.5-4b"

_LOCK = threading.Lock()
_PROC: Optional[subprocess.Popen] = None
_LOG_FH = None
_LAST_OPTS: Dict[str, str] = {
    "model_dir": DEFAULT_MODEL_DIR,
    "model_id": DEFAULT_MODEL_ID,
    "model_label": "Qwen3.5-4B",
}
_LAST_EXIT_CODE: Optional[int] = None
_LAST_ERROR: str = ""


def _log(msg: str) -> None:
    line = f"[local_qwen_hostctl] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _tail_log_errors(max_lines: int = 40) -> str:
    """从服务日志尾部提取最近一次 load failed / Error 摘要。"""
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-max_lines:]
    except OSError:
        return ""
    interesting: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if (
            "load failed" in low
            or "error" in low
            or "traceback" in low
            or "outofmemory" in low
            or "cuda out of memory" in low
            or "valueerror" in low
        ):
            interesting.append(s)
    if interesting:
        return " | ".join(interesting[-3:])
    for line in reversed(lines):
        s = line.strip()
        if s and not s.startswith("[local_qwen_hostctl]"):
            return s
    return ""


def qwen_health_ok(timeout_s: float = 1.5) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{QWEN_PORT}/health", timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def _refresh_proc_state() -> None:
    """若 worker 已退出，清理 _PROC 并记录退出码。"""
    global _PROC, _LAST_EXIT_CODE, _LAST_ERROR, _LOG_FH
    with _LOCK:
        if _PROC is None:
            return
        code = _PROC.poll()
        if code is None:
            return
        _LAST_EXIT_CODE = int(code)
        if code != 0:
            _LAST_ERROR = _tail_log_errors() or f"进程退出 code={code}"
            _log(f"worker exited code={code}: {_LAST_ERROR[:200]}")
        else:
            _LAST_ERROR = ""
        _PROC = None
        if _LOG_FH is not None:
            try:
                _LOG_FH.flush()
            except Exception:
                pass


def status_payload() -> Dict[str, Any]:
    _refresh_proc_state()
    with _LOCK:
        running_proc = _PROC is not None and _PROC.poll() is None
        pid = _PROC.pid if running_proc and _PROC is not None else None
        opts = dict(_LAST_OPTS)
        last_exit = _LAST_EXIT_CODE
        last_error = _LAST_ERROR
    healthy = qwen_health_ok()
    return {
        "ok": True,
        "ctl": "local_qwen_hostctl",
        "process_running": running_proc,
        "pid": pid,
        "service_healthy": healthy,
        "starting": bool(running_proc and not healthy),
        "api_base": f"http://127.0.0.1:{QWEN_PORT}/v1",
        "model_dir": opts.get("model_dir") or "",
        "model_id": opts.get("model_id") or "",
        "model_label": opts.get("model_label") or "",
        "last_exit_code": last_exit,
        "last_error": last_error,
        "log": LOG_PATH,
    }


def _normalize_start_opts(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    raw = raw or {}
    model_dir = str(raw.get("model_dir") or "").strip()
    model_id = str(raw.get("model_id") or "").strip()
    model_label = str(raw.get("model_label") or "").strip()
    if not model_dir:
        model_dir = os.environ.get("LOCAL_QWEN_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR
    if not model_id:
        model_id = os.environ.get("LOCAL_QWEN_MODEL_ID", "").strip() or DEFAULT_MODEL_ID
    if not model_label:
        model_label = os.path.basename(model_dir.rstrip("/")) or model_id
    return {
        "model_dir": os.path.abspath(os.path.expanduser(model_dir)),
        "model_id": model_id,
        "model_label": model_label,
    }


def start_service(opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _PROC, _LOG_FH, _LAST_OPTS, _LAST_EXIT_CODE, _LAST_ERROR
    normalized = _normalize_start_opts(opts)
    _refresh_proc_state()
    if qwen_health_ok():
        return {"ok": True, "message": "服务已在运行", **status_payload()}
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            return {"ok": True, "message": "正在启动中", **status_payload()}
        if not os.path.isfile(RUN_SCRIPT):
            return {"ok": False, "message": f"未找到 {RUN_SCRIPT}"}
        model_dir = normalized["model_dir"]
        if not (
            os.path.isfile(os.path.join(model_dir, "config.json"))
            or os.path.isfile(os.path.join(model_dir, "adapter_config.json"))
        ):
            return {
                "ok": False,
                "message": (
                    f"未找到模型权重目录（需含 config.json 或 adapter_config.json）: "
                    f"{model_dir}"
                ),
            }
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if _LOG_FH is not None:
            try:
                _LOG_FH.close()
            except Exception:
                pass
        _LOG_FH = open(LOG_PATH, "a", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("LOCAL_QWEN_HOST", "0.0.0.0")
        env.setdefault("LOCAL_QWEN_PORT", str(QWEN_PORT))
        # 宿主机绝对路径，避免 Docker 内 HOME=/root 找不到 conda
        env.setdefault(
            "LOCAL_QWEN_PYTHON",
            os.path.expanduser("~/miniconda3/envs/psi-policy/bin/python"),
        )
        env["LOCAL_QWEN_MODEL_DIR"] = model_dir
        env["LOCAL_QWEN_MODEL_ID"] = normalized["model_id"]
        cmd = [
            "bash",
            RUN_SCRIPT,
            "--host",
            "0.0.0.0",
            "--port",
            str(QWEN_PORT),
            "--model",
            model_dir,
            "--model-id",
            normalized["model_id"],
        ]
        _LAST_OPTS = normalized
        _LAST_EXIT_CODE = None
        _LAST_ERROR = ""
        _log(
            f"start {normalized['model_label']}: {' '.join(cmd)}"
        )
        _PROC = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=_LOG_FH,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "ok": True,
        "message": (
            f"已启动 {normalized['model_label']} "
            f"pid={_PROC.pid} model_id={normalized['model_id']}"
        ),
        **status_payload(),
    }


def stop_service() -> Dict[str, Any]:
    global _PROC, _LOG_FH
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            _log(f"stop pid={_PROC.pid}")
            try:
                os.killpg(os.getpgid(_PROC.pid), signal.SIGTERM)
            except Exception:
                try:
                    _PROC.terminate()
                except Exception:
                    pass
            try:
                _PROC.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(_PROC.pid), signal.SIGKILL)
                except Exception:
                    pass
            _PROC = None
        if _LOG_FH is not None:
            try:
                _LOG_FH.close()
            except Exception:
                pass
            _LOG_FH = None
    # 清理残留 worker
    try:
        subprocess.run(
            ["pkill", "-f", "local_qwen_worker.py"],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except Exception:
        pass
    return {"ok": True, "message": "已停止", **status_payload()}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQwenHostCtl/1.0"

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
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            self._send(200, {"ok": True, "service": "local_qwen_hostctl"})
            return
        if path == "/status":
            self._send(200, status_payload())
            return
        self._send(404, {"ok": False, "message": f"unknown {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload = self._read_json_body()
        if path == "/start":
            body = start_service(payload)
            self._send(200 if body.get("ok") else 500, body)
            return
        if path == "/stop":
            body = stop_service()
            self._send(200, body)
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
    server = ThreadingHTTPServer((CTL_HOST, CTL_PORT), Handler)
    _log(f"listening on http://{CTL_HOST}:{CTL_PORT}")
    _log(f"will manage Qwen service on :{QWEN_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
        stop_service()
    return 0


if __name__ == "__main__":
    sys.exit(main())
