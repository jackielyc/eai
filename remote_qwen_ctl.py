#!/usr/bin/env python3
"""远程 Qwen 推理部署控制（SSH + 本地端口转发）。

支持多台远程机（Host Profile）。默认:
  psi_motus_2_for_liyichao  — 既有 8×A800
  tione-develop             — 新机（SSH Port 10666）

用法:
  python3 remote_qwen_ctl.py --host tione-develop sync
  python3 remote_qwen_ctl.py --host tione-develop deploy --model-key qwen3.5-35b-a3b
  python3 remote_qwen_ctl.py --host psi_motus_2_for_liyichao status
  python3 remote_qwen_ctl.py --host tione-develop stop
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import urlopen


HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "log")

# key, label, model_id, dirname or absolute path
LAKE_QWEN35_OUTPUT_ROOT = (
    "/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/output"
)
REMOTE_LAKE_QWEN_PYTHON = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python"
)
REMOTE_MODELS: Tuple[Tuple[str, str, str, str], ...] = (
    ("qwen3.5-4b", "Qwen3.5-4B", "qwen3.5-4b", "Qwen3.5-4B"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
    (
        "lake-qwen35-4b-lora",
        "Lake Qwen3.5-4B LoRA",
        "lake-qwen35-4b-lora",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake",
    ),
    (
        "lake-qwen35-4b-lora-smoke",
        "Lake Qwen3.5-4B LoRA (smoke)",
        "lake-qwen35-4b-lora-smoke",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake-smoke",
    ),
)

SYNC_FILES = (
    "local_qwen_worker.py",
    "run_local_qwen.sh",
)

# 每台远程机独立配置；可用环境变量覆盖（见 apply_host）
HOST_PROFILES: Dict[str, Dict[str, Any]] = {
    "psi_motus_2_for_liyichao": {
        "id": "psi_motus_2_for_liyichao",
        "label": "psi_motus（8×A800）",
        "ssh_host": "psi_motus_2_for_liyichao",
        "remote_work": "/share_data/liyichao/eai_qwen_runtime",
        "model_root": (
            "/share_data/projects/mahjong/share/personal/liyichao/models/Qwen"
        ),
        "python": "/root/anaconda3/envs/psi-policy/bin/python",
        "remote_port": 8100,
        "local_port": 18100,
    },
    "tione-develop": {
        "id": "tione-develop",
        "label": "tione-develop（7×A800）",
        "ssh_host": "tione-develop",
        "remote_work": "/root/eai_qwen_runtime",
        "model_root": (
            "/share_data/projects/mahjong/share/personal/liyichao/models/Qwen"
        ),
        "python": "/opt/conda/envs/eai-qwen/bin/python",
        "remote_port": 8100,
        "local_port": 18102,
    },
}

DEFAULT_HOST_ID = os.environ.get(
    "REMOTE_QWEN_HOST_ID",
    os.environ.get("REMOTE_QWEN_SSH_HOST", "psi_motus_2_for_liyichao"),
)

# 运行时生效的主机参数（由 apply_host 填充）
SSH_HOST = ""
REMOTE_WORK = ""
REMOTE_MODEL_ROOT = ""
REMOTE_PYTHON = ""
REMOTE_PORT = 8100
LOCAL_PORT = 18100
TUNNEL_PID_FILE = ""
REMOTE_LOG = ""
ACTIVE_HOST_ID = ""


def list_host_ids() -> List[str]:
    return list(HOST_PROFILES.keys())


def host_profile(host_id: str) -> Dict[str, Any]:
    if host_id in HOST_PROFILES:
        return dict(HOST_PROFILES[host_id])
    # 允许直接传 ssh Host 别名
    for pid, prof in HOST_PROFILES.items():
        if prof.get("ssh_host") == host_id:
            return dict(prof)
    raise KeyError(f"未知远程主机: {host_id}（可选: {', '.join(list_host_ids())}）")


def apply_host(host_id: Optional[str] = None) -> Dict[str, Any]:
    """切换当前操作的远程主机；返回生效 profile。"""
    global SSH_HOST, REMOTE_WORK, REMOTE_MODEL_ROOT, REMOTE_PYTHON
    global REMOTE_PORT, LOCAL_PORT, TUNNEL_PID_FILE, REMOTE_LOG, ACTIVE_HOST_ID

    hid = (host_id or DEFAULT_HOST_ID or "psi_motus_2_for_liyichao").strip()
    prof = host_profile(hid)
    ACTIVE_HOST_ID = str(prof["id"])
    # 以 profile 为准；仅当显式 FORCE_ENV 或选中默认主机时才吃全局环境变量
    use_env = (
        os.environ.get("REMOTE_QWEN_FORCE_ENV", "").strip() == "1"
        or hid == DEFAULT_HOST_ID
    )

    def _pick(env_key: str, default: object) -> str:
        if use_env:
            val = os.environ.get(env_key, "").strip()
            if val:
                return val
        return str(default).strip()

    SSH_HOST = _pick("REMOTE_QWEN_SSH_HOST", prof["ssh_host"])
    REMOTE_WORK = _pick("REMOTE_QWEN_REMOTE_WORK", prof["remote_work"])
    REMOTE_MODEL_ROOT = _pick("REMOTE_QWEN_MODEL_ROOT", prof["model_root"])
    REMOTE_PYTHON = _pick("REMOTE_QWEN_PYTHON", prof["python"])
    REMOTE_PORT = int(_pick("REMOTE_QWEN_PORT", prof["remote_port"]))
    LOCAL_PORT = int(_pick("REMOTE_QWEN_LOCAL_PORT", prof["local_port"]))
    safe = ACTIVE_HOST_ID.replace("/", "_")
    TUNNEL_PID_FILE = os.path.join(LOG_DIR, f"remote_qwen_tunnel_{safe}.pid")
    REMOTE_LOG = os.path.join(REMOTE_WORK, "local_qwen_service.log")
    return {
        "id": ACTIVE_HOST_ID,
        "label": prof.get("label") or ACTIVE_HOST_ID,
        "ssh_host": SSH_HOST,
        "remote_work": REMOTE_WORK,
        "model_root": REMOTE_MODEL_ROOT,
        "python": REMOTE_PYTHON,
        "remote_port": REMOTE_PORT,
        "local_port": LOCAL_PORT,
        "api_base": api_base(),
    }


def api_base() -> str:
    return f"http://127.0.0.1:{LOCAL_PORT}/v1"


def deploy_scan_roots(host_id: Optional[str] = None) -> List[Tuple[str, str]]:
    """可扫描的权重根目录：(绝对路径, 显示名)。"""
    apply_host(host_id)
    roots: List[Tuple[str, str]] = [
        (REMOTE_MODEL_ROOT, "Qwen 基座"),
        (LAKE_QWEN35_OUTPUT_ROOT, "Lake 训练 output"),
    ]
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for path, label in roots:
        p = path.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append((p, label))
    return out


def _is_deployable_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
        os.path.join(path, "adapter_config.json")
    )


def _model_entry_from_path(path: str, root: str = "") -> Dict[str, str]:
    name = os.path.basename(path.rstrip("/"))
    kind = (
        "lora"
        if os.path.isfile(os.path.join(path, "adapter_config.json"))
        else "full"
    )
    return {
        "path": path,
        "name": name,
        "model_id": name,
        "label": name,
        "kind": kind,
        "root": root or os.path.dirname(path),
    }


def list_deploy_model_dirs_local(roots: Sequence[str]) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    seen: set = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            path = os.path.join(root, name)
            if not os.path.isdir(path) or not _is_deployable_model_dir(path):
                continue
            if path in seen:
                continue
            seen.add(path)
            found.append(_model_entry_from_path(path, root))
    found.sort(key=lambda x: (x.get("root") or "", x.get("name") or ""))
    return found


def list_deploy_model_dirs_remote(
    roots: Optional[Sequence[str]] = None,
    *,
    host_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    apply_host(host_id)
    scan = [r for r, _ in (deploy_scan_roots(host_id) if roots is None else [])]
    if roots is not None:
        scan = [str(r).strip() for r in roots if str(r).strip()]
    if not scan:
        scan = [r for r, _ in deploy_scan_roots(host_id)]
    py_script = f"""
import json, os
roots = json.loads({json.dumps(json.dumps(list(scan)))})
seen = set()
rows = []
for root in roots:
    if not os.path.isdir(root):
        continue
    try:
        names = sorted(os.listdir(root))
    except OSError:
        continue
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if not (os.path.isfile(os.path.join(path, 'config.json')) or os.path.isfile(os.path.join(path, 'adapter_config.json'))):
            continue
        if path in seen:
            continue
        seen.add(path)
        kind = 'lora' if os.path.isfile(os.path.join(path, 'adapter_config.json')) else 'full'
        rows.append({{'root': root, 'path': path, 'name': name, 'model_id': name, 'label': name, 'kind': kind}})
rows.sort(key=lambda x: (x.get('root') or '', x.get('name') or ''))
print(json.dumps(rows, ensure_ascii=False))
"""
    proc = ssh_bash(
        f"python3 - <<'PY'\n{py_script}\nPY",
        timeout_s=90.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"远程列举模型目录失败 ({SSH_HOST}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    raw = (proc.stdout or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"远程列举返回非 JSON: {raw[:200]}") from exc
    if not isinstance(data, list):
        return []
    return [dict(x) for x in data if isinstance(x, dict) and x.get("path")]


def list_deploy_model_dirs(
    roots: Optional[Sequence[str]] = None,
    *,
    host_id: Optional[str] = None,
    local: bool = False,
) -> Dict[str, object]:
    """列举可部署权重目录。local=True 时只扫本机；否则 SSH 到当前 host。"""
    apply_host(host_id)
    scan = [str(r).strip() for r in roots] if roots else [r for r, _ in deploy_scan_roots(host_id)]
    scan = [r for r in scan if r]
    try:
        if local:
            models = list_deploy_model_dirs_local(scan)
        else:
            models = list_deploy_model_dirs_remote(scan, host_id=host_id)
        return {
            "ok": True,
            "host_id": ACTIVE_HOST_ID,
            "ssh_host": SSH_HOST,
            "roots": scan,
            "models": models,
            "count": len(models),
        }
    except Exception as exc:
        return {
            "ok": False,
            "host_id": ACTIVE_HOST_ID,
            "ssh_host": SSH_HOST,
            "roots": scan,
            "models": [],
            "message": str(exc),
        }


def infer_python_for_model_dir(model_dir: str, model_key: str = "") -> str:
    if model_key.startswith("lake-"):
        return REMOTE_LAKE_QWEN_PYTHON
    if LAKE_QWEN35_OUTPUT_ROOT in model_dir:
        return REMOTE_LAKE_QWEN_PYTHON
    if model_dir.rstrip("/").endswith(("adapter_config.json",)):
        return REMOTE_LAKE_QWEN_PYTHON
    if os.path.isfile(os.path.join(model_dir, "adapter_config.json")):
        return REMOTE_LAKE_QWEN_PYTHON
    return REMOTE_PYTHON


def model_spec(model_key: str) -> Tuple[str, str, str, str]:
    for item in REMOTE_MODELS:
        if item[0] == model_key:
            return item
    return REMOTE_MODELS[-1]


def model_dir_for_key(model_key: str) -> str:
    _key, _label, _model_id, path_spec = model_spec(model_key)
    if path_spec.startswith("/"):
        return path_spec
    return f"{REMOTE_MODEL_ROOT}/{path_spec}"


def python_for_model_key(model_key: str) -> str:
    if model_key.startswith("lake-"):
        return REMOTE_LAKE_QWEN_PYTHON
    return REMOTE_PYTHON


def ssh_common_opts(
    *, exit_on_forward_failure: str = "no", clear_all_forwardings: bool = True
) -> List[str]:
    opts: List[str] = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if clear_all_forwardings:
        # 仅用于 scp/远程 bash：与 -L 同用时会把命令行端口转发一并清掉
        opts.extend(["-o", "ClearAllForwardings=yes"])
    opts.extend(["-o", f"ExitOnForwardFailure={exit_on_forward_failure}"])
    return opts


def ssh_base_cmd() -> List[str]:
    return ["ssh", *ssh_common_opts(exit_on_forward_failure="no"), SSH_HOST]


def ssh_tunnel_cmd() -> List[str]:
    """本地端口转发：-L/-N 必须在 hostname 之前，且不能 ClearAllForwardings。"""
    return [
        "ssh",
        *ssh_common_opts(exit_on_forward_failure="yes", clear_all_forwardings=False),
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        f"{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}",
        SSH_HOST,
    ]


def run_cmd(
    cmd: Sequence[str],
    *,
    timeout_s: Optional[float] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=check,
    )


def ssh_bash(script: str, timeout_s: float = 120.0) -> subprocess.CompletedProcess:
    """在远程执行 bash 脚本（经 stdin，避免引号/换行被吃掉）。"""
    cmd = ssh_base_cmd() + ["bash", "-s"]
    return subprocess.run(
        cmd,
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def sync_runtime() -> Dict[str, object]:
    """把 worker / 启动脚本同步到远程工作目录。"""
    ensure_log_dir()
    missing = [f for f in SYNC_FILES if not os.path.isfile(os.path.join(HERE, f))]
    if missing:
        return {"ok": False, "message": f"本地缺少文件: {missing}"}

    prep = ssh_bash(f"mkdir -p {json.dumps(REMOTE_WORK)}")
    if prep.returncode != 0:
        return {
            "ok": False,
            "message": (
                f"SSH 创建远程目录失败 ({SSH_HOST}): "
                f"{(prep.stderr or prep.stdout).strip()}"
            ),
        }

    for name in SYNC_FILES:
        local = os.path.join(HERE, name)
        remote = f"{SSH_HOST}:{REMOTE_WORK}/{name}"
        proc = run_cmd(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ClearAllForwardings=yes",
                "-o",
                "ExitOnForwardFailure=no",
                local,
                remote,
            ],
            timeout_s=120.0,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "message": f"scp {name} 失败: {(proc.stderr or proc.stdout).strip()}",
            }

    chmod = ssh_bash(f"chmod +x {REMOTE_WORK}/run_local_qwen.sh")
    if chmod.returncode != 0:
        return {
            "ok": False,
            "message": f"chmod 失败: {(chmod.stderr or chmod.stdout).strip()}",
        }
    return {"ok": True, "message": f"已同步到 {SSH_HOST}:{REMOTE_WORK}"}


def remote_health_ok(timeout_s: float = 2.0) -> bool:
    script = (
        f"curl -fsS --max-time {max(1, int(timeout_s))} "
        f"http://127.0.0.1:{REMOTE_PORT}/health"
    )
    proc = ssh_bash(script, timeout_s=timeout_s + 10)
    return proc.returncode == 0


def local_tunnel_health(timeout_s: float = 2.0) -> Tuple[bool, Dict[str, object]]:
    url = f"http://127.0.0.1:{LOCAL_PORT}/health"
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok")), body if isinstance(body, dict) else {}
    except Exception:
        return False, {}


def local_tunnel_port_listening(timeout_s: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", LOCAL_PORT), timeout=timeout_s):
            return True
    except OSError:
        return False


def read_tunnel_pid() -> Optional[int]:
    try:
        with open(TUNNEL_PID_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def tunnel_running() -> bool:
    pid = read_tunnel_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return local_tunnel_port_listening()


def stop_tunnel() -> Dict[str, object]:
    pid = read_tunnel_pid()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        if os.path.isfile(TUNNEL_PID_FILE):
            os.remove(TUNNEL_PID_FILE)
    except OSError:
        pass
    port_fwd = f"-L {LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}"
    for pattern in (
        f"ssh .* {port_fwd} .*{SSH_HOST}",
        f"ssh .*{SSH_HOST} .* {port_fwd}",
        f"ssh .* {port_fwd}",
        f"ssh .* -L {LOCAL_PORT}:",
    ):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=2,
                check=False,
            )
        except Exception:
            pass
    # 强制释放本地端口（Docker/宿主机 PID 命名空间不一致时 pkill 可能杀不到）
    for cmd in (
        ["fuser", "-k", f"{LOCAL_PORT}/tcp"],
        ["ss", "-K", "sport", "=", str(LOCAL_PORT)],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=2, check=False)
        except Exception:
            pass
    time.sleep(0.2)
    return {"ok": True, "message": f"已停止本地 SSH 隧道 (:{LOCAL_PORT} / {SSH_HOST})"}


def start_tunnel() -> Dict[str, object]:
    ensure_log_dir()
    if tunnel_running():
        ok, _ = local_tunnel_health()
        if ok:
            return {
                "ok": True,
                "message": f"隧道已在运行 :{LOCAL_PORT}",
                "api_base": api_base(),
            }
        stop_tunnel()
    else:
        # 清理旧版错误命令留下的 zombie ssh（进程在但端口未监听）
        stop_tunnel()

    safe = ACTIVE_HOST_ID.replace("/", "_") or "default"
    log_path = os.path.join(LOG_DIR, f"remote_qwen_tunnel_{safe}.log")
    cmd = ssh_tunnel_cmd()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n>>> tunnel start {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with open(TUNNEL_PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    for _ in range(20):
        time.sleep(0.25)
        if proc.poll() is not None:
            return {
                "ok": False,
                "message": f"SSH 隧道启动失败，见 {log_path}",
            }
        ok, _ = local_tunnel_health(timeout_s=0.8)
        if ok:
            return {
                "ok": True,
                "message": (
                    f"隧道已建立 localhost:{LOCAL_PORT} -> {SSH_HOST}:{REMOTE_PORT}"
                ),
                "pid": proc.pid,
                "api_base": api_base(),
            }
    if proc.poll() is None:
        if local_tunnel_port_listening():
            return {
                "ok": True,
                "message": (
                    f"隧道已监听 localhost:{LOCAL_PORT}，等待远端模型就绪"
                ),
                "pid": proc.pid,
                "api_base": api_base(),
                "waiting_service": True,
            }
        try:
            proc.kill()
        except OSError:
            pass
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8") as logf:
                tail = logf.read()[-800:]
        except OSError:
            pass
        hint = tail.strip().splitlines()[-1] if tail.strip() else ""
        msg = f"SSH 隧道未绑定 localhost:{LOCAL_PORT}，见 {log_path}"
        if hint:
            msg += f" ({hint})"
        return {"ok": False, "message": msg}
    return {"ok": False, "message": f"SSH 隧道异常退出，见 {log_path}"}


def stop_remote_service() -> Dict[str, object]:
    script = f"""
pkill -f 'local_qwen_worker.py' >/dev/null 2>&1 || true
pkill -f 'run_local_qwen.sh' >/dev/null 2>&1 || true
sleep 0.5
if curl -fsS --max-time 1 http://127.0.0.1:{REMOTE_PORT}/health >/dev/null 2>&1; then
  echo STILL_UP
  exit 1
fi
echo STOPPED
"""
    proc = ssh_bash(script, timeout_s=30.0)
    msg = (proc.stdout or proc.stderr or "").strip()
    return {
        "ok": proc.returncode == 0,
        "message": msg
        or ("已停止远程服务" if proc.returncode == 0 else "停止远程服务失败"),
    }


def start_remote_service(
    model_key: str = "qwen3.5-35b-a3b",
    *,
    model_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    model_label: Optional[str] = None,
    python: Optional[str] = None,
) -> Dict[str, object]:
    if model_dir and str(model_dir).strip():
        path = str(model_dir).strip()
        mid = (model_id or os.path.basename(path.rstrip("/"))).strip()
        label = (model_label or mid).strip()
        key = model_key or mid
        py_default = python or infer_python_for_model_dir(path, key)
        model_dir_resolved = path
    else:
        key, label, mid_default, _path_spec = model_spec(model_key)
        model_dir_resolved = model_dir_for_key(model_key)
        mid = (model_id or mid_default).strip()
        label = (model_label or label).strip()
        py_default = python or python_for_model_key(model_key)
    model_dir = model_dir_resolved
    sync = sync_runtime()
    if not sync.get("ok"):
        return sync

    if remote_health_ok():
        return {
            "ok": True,
            "message": f"远程服务已在运行 ({label} @ {SSH_HOST})",
            "model_key": key,
            "model_id": model_id,
            "model_label": label,
            "already_running": True,
        }

    script = f"""
set -e
MODEL_DIR={json.dumps(model_dir)}
if [[ ! -f "$MODEL_DIR/config.json" && ! -f "$MODEL_DIR/adapter_config.json" ]]; then
  echo "MISSING_MODEL:$MODEL_DIR" >&2
  exit 2
fi
PY={json.dumps(py_default)}
if [[ ! -x "$PY" ]]; then
  # 回退常见路径
  for cand in \\
    {json.dumps(REMOTE_LAKE_QWEN_PYTHON)} \\
    /opt/conda/envs/eai-qwen/bin/python \\
    /root/anaconda3/envs/psi-policy/bin/python \\
    /opt/conda/bin/python \\
    /usr/bin/python3
  do
    if [[ -x "$cand" ]]; then PY="$cand"; break; fi
  done
fi
if [[ ! -x "$PY" ]]; then
  echo "MISSING_PYTHON:{json.dumps(py_default)}" >&2
  exit 3
fi
cd {json.dumps(REMOTE_WORK)}
export LOCAL_QWEN_PYTHON="$PY"
export LOCAL_QWEN_MODEL_DIR="$MODEL_DIR"
export LOCAL_QWEN_MODEL_ID={json.dumps(model_id)}
export LOCAL_QWEN_HOST=127.0.0.1
export LOCAL_QWEN_PORT={REMOTE_PORT}
export LOCAL_QWEN_CUDNN=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nohup bash ./run_local_qwen.sh \\
  --host 127.0.0.1 \\
  --port {REMOTE_PORT} \\
  --model "$MODEL_DIR" \\
  --model-id {json.dumps(model_id)} \\
  --python "$PY" \\
  >> {json.dumps(REMOTE_LOG)} 2>&1 &
echo $!
"""
    proc = ssh_bash(script, timeout_s=60.0)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {
            "ok": False,
            "message": f"远程启动失败 ({SSH_HOST}): {err or out}",
            "model_label": label,
        }
    return {
        "ok": True,
        "message": f"已在 {SSH_HOST} 启动 {label} (pid={out})，日志: {REMOTE_LOG}",
        "remote_pid": out,
        "model_key": key,
        "model_id": model_id,
        "model_label": label,
        "model_dir": model_dir,
        "ssh_host": SSH_HOST,
    }


def status_payload() -> Dict[str, object]:
    remote_ok = remote_health_ok(timeout_s=2.0)
    tunnel_ok = tunnel_running()
    local_ok, health = local_tunnel_health(timeout_s=1.5)
    return {
        "ok": True,
        "host_id": ACTIVE_HOST_ID,
        "ssh_host": SSH_HOST,
        "remote_work": REMOTE_WORK,
        "model_root": REMOTE_MODEL_ROOT,
        "python": REMOTE_PYTHON,
        "remote_port": REMOTE_PORT,
        "local_port": LOCAL_PORT,
        "api_base": api_base(),
        "remote_healthy": remote_ok,
        "tunnel_running": tunnel_ok,
        "local_healthy": local_ok,
        "health": health,
        "model": str(health.get("model") or ""),
        "model_dir": str(health.get("model_dir") or ""),
    }


def deploy(
    model_key: str = "qwen3.5-35b-a3b",
    *,
    model_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    model_label: Optional[str] = None,
    python: Optional[str] = None,
) -> Dict[str, object]:
    """同步 + 远程启动 + 建立隧道。"""
    started = start_remote_service(
        model_key=model_key,
        model_dir=model_dir,
        model_id=model_id,
        model_label=model_label,
        python=python,
    )
    if not started.get("ok"):
        return started
    tun = start_tunnel()
    if not tun.get("ok"):
        return {
            "ok": False,
            "message": f"远程已启动，但隧道失败: {tun.get('message')}",
            "remote": started,
            "tunnel": tun,
        }
    return {
        "ok": True,
        "message": f"{started.get('message')}; {tun.get('message')}",
        "api_base": api_base(),
        "model_id": started.get("model_id"),
        "model_label": started.get("model_label"),
        "ssh_host": SSH_HOST,
        "host_id": ACTIVE_HOST_ID,
        "remote": started,
        "tunnel": tun,
    }


def stop_all() -> Dict[str, object]:
    t = stop_tunnel()
    r = stop_remote_service()
    return {
        "ok": bool(t.get("ok") and r.get("ok")),
        "message": f"{t.get('message')}; {r.get('message')}",
        "tunnel": t,
        "remote": r,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Remote Qwen deploy controller")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST_ID,
        help=f"远程主机 id（可选: {', '.join(list_host_ids())}）",
    )
    parser.add_argument(
        "action",
        choices=(
            "sync",
            "start",
            "tunnel",
            "deploy",
            "stop",
            "status",
            "hosts",
            "list-models",
        ),
    )
    parser.add_argument(
        "--model-key",
        default="qwen3.5-35b-a3b",
        help="预设模型 key（与 --model-dir 二选一）",
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help="直接指定权重目录（优先于 --model-key）",
    )
    parser.add_argument("--model-id", default="", help="API model id（默认同目录名）")
    parser.add_argument("--model-label", default="", help="显示名（默认同 model-id）")
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="list-models 时只扫描该根目录（可重复）",
    )
    parser.add_argument(
        "--local-scan",
        action="store_true",
        help="list-models 在本机扫描（不 SSH）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.action == "hosts":
        print(json.dumps(list(HOST_PROFILES.values()), ensure_ascii=False, indent=2))
        return 0

    apply_host(args.host)

    deploy_kwargs = {
        "model_key": args.model_key,
        "model_dir": args.model_dir.strip() or None,
        "model_id": args.model_id.strip() or None,
        "model_label": args.model_label.strip() or None,
    }

    if args.action == "sync":
        result = sync_runtime()
    elif args.action == "list-models":
        roots = args.scan_root if args.scan_root else None
        result = list_deploy_model_dirs(
            roots=roots,
            host_id=args.host,
            local=bool(args.local_scan),
        )
    elif args.action == "start":
        result = start_remote_service(**deploy_kwargs)
    elif args.action == "tunnel":
        result = start_tunnel()
    elif args.action == "deploy":
        result = deploy(**deploy_kwargs)
    elif args.action == "stop":
        result = stop_all()
    else:
        result = status_payload()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


# 模块导入时应用默认主机，兼容旧调用方
apply_host(DEFAULT_HOST_ID)


if __name__ == "__main__":
    sys.exit(main())
