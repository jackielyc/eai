"""本地 Qwen 模型目录扫描、状态查询与部署/停止（供 EmbodiedEvo Tab 使用）。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EAI_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = EAI_DIR.parent
MODELS_ROOT = Path(
    os.environ.get(
        "LOCAL_QWEN_MODELS_ROOT",
        str(WORKSPACE_DIR / "models" / "Qwen"),
    )
)
LAKE_OUTPUT_ROOT = EAI_DIR / "train" / "lake_qwen35" / "output"
RUN_SCRIPT = EAI_DIR / "scripts/launch/run_local_qwen.sh"
VLLM_SCRIPT = EAI_DIR / "scripts/launch/run_vllm_coder.sh"
WORKER_NAME = "compat/local_qwen_worker.py"
VLLM_PROC_MATCH = "vllm.entrypoints.openai.api_server"
WORKER_PORT = int(os.environ.get("LOCAL_QWEN_PORT", "8100"))
VLLM_PORT = int(os.environ.get("LOCAL_VLLM_PORT", "8000"))
# Prefer vLLM (option A) when unset; LOCAL_QWEN_API_BASE still wins.
API_PORT = int(os.environ.get("LOCAL_API_PORT", str(VLLM_PORT)))
CTL_URL = os.environ.get("LOCAL_QWEN_CTL_URL", "http://127.0.0.1:18101").rstrip("/")
API_BASE = os.environ.get(
    "LOCAL_QWEN_API_BASE", f"http://127.0.0.1:{API_PORT}/v1"
).rstrip("/")
DEFAULT_PYTHON = os.environ.get(
    "LOCAL_QWEN_PYTHON",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python",
)
LOG_PATH = EAI_DIR / "log" / "embodied_evo_local_qwen.log"

# (key, label, model_id, path_or_dirname)
_PRESETS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "qwen3-coder-30b-a3b",
        "Qwen3-Coder-30B-A3B (vLLM)",
        "qwen3-coder-30b-a3b",
        "Qwen3-Coder-30B-A3B",
    ),
    ("qwen3.5-4b", "Qwen3.5-4B", "qwen3.5-4b", "Qwen3.5-4B"),
    ("qwen3.5-vl-4b", "Qwen3.5-VL-4B", "qwen3.5-vl-4b", "Qwen3.5-VL-4B"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
    ("qwen3.5-vl", "Qwen3.5-VL (→35B)", "qwen3.5-vl", "Qwen3.5-VL"),
    (
        "lake-qwen35-4b-lora",
        "Lake Qwen3.5-4B LoRA",
        "lake-qwen35-4b-lora",
        str(LAKE_OUTPUT_ROOT / "qwen35-4b-lora-lake"),
    ),
    (
        "lake-qwen35-4b-lora-smoke",
        "Lake Qwen3.5-4B LoRA (smoke)",
        "lake-qwen35-4b-lora-smoke",
        str(LAKE_OUTPUT_ROOT / "qwen35-4b-lora-lake-smoke"),
    ),
)


@dataclass(frozen=True)
class LocalModel:
    key: str
    label: str
    model_id: str
    model_dir: str
    available: bool
    kind: str  # full | lora


@dataclass
class ModelStatus:
    healthy: bool
    loading: bool
    model_id: str
    model_dir: str
    device: str
    kind: str
    error: str
    process_hint: str
    ctl_ok: bool
    api_base: str = API_BASE

    @property
    def state_text(self) -> str:
        if self.healthy:
            mid = self.model_id or "?"
            dev = f" @ {self.device}" if self.device else ""
            return f"运行中 · {mid}{dev}"
        if self.loading:
            return f"加载中 · {self.model_id or '…'}"
        if self.error:
            return f"异常 · {self.error[:80]}"
        if self.process_hint:
            return f"已停止（残留: {self.process_hint}）"
        return "已停止"


def _model_dir_valid(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "config.json").is_file() or (path / "adapter_config.json").is_file()


def _kind_for(path: Path) -> str:
    return "lora" if (path / "adapter_config.json").is_file() else "full"


def _resolve_dir(spec: str) -> Optional[Path]:
    p = Path(spec).expanduser()
    if p.is_absolute():
        return p if _model_dir_valid(p) else None
    cand = MODELS_ROOT / spec
    return cand if _model_dir_valid(cand) else None


def list_local_models(*, only_available: bool = True) -> List[LocalModel]:
    """列出预设 + models/Qwen 扫描到的可用权重。"""
    seen_dirs: set[str] = set()
    out: List[LocalModel] = []

    def _add(key: str, label: str, model_id: str, path: Optional[Path]) -> None:
        if path is None:
            if only_available:
                return
            out.append(
                LocalModel(
                    key=key,
                    label=label,
                    model_id=model_id,
                    model_dir="",
                    available=False,
                    kind="full",
                )
            )
            return
        resolved = str(path.resolve())
        if resolved in seen_dirs:
            return
        seen_dirs.add(resolved)
        out.append(
            LocalModel(
                key=key,
                label=label,
                model_id=model_id,
                model_dir=resolved,
                available=True,
                kind=_kind_for(path),
            )
        )

    for key, label, mid, spec in _PRESETS:
        _add(key, label, mid, _resolve_dir(spec))

    # 自动扫描仅纳入 Qwen3.5*（避免把 Omni 等不适配 worker 的权重塞进下拉）
    if MODELS_ROOT.is_dir():
        for child in sorted(MODELS_ROOT.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not child.name.startswith("Qwen3.5"):
                continue
            if not _model_dir_valid(child):
                continue
            key = child.name.lower().replace("_", "-")
            mid = key
            _add(key, child.name, mid, child)

    # Lake LoRA：只列名称含 qwen35 的近期产物，避免下拉过长
    if LAKE_OUTPUT_ROOT.is_dir():
        lake_items: List[LocalModel] = []
        for child in sorted(LAKE_OUTPUT_ROOT.iterdir(), reverse=True):
            if not child.is_dir() or not _model_dir_valid(child):
                continue
            low = child.name.lower()
            if "qwen35" not in low and "qwen3.5" not in low:
                continue
            key = f"lake-{low.replace('_', '-')}"
            before = len(seen_dirs)
            _add(key, f"Lake {child.name}", key, child)
            if len(seen_dirs) > before:
                lake_items.append(out[-1])
            if len(lake_items) >= 8:
                break

    return out


def _http_json(url: str, *, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout_s: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return True, body if isinstance(body, dict) else {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return False, {}


def uses_vllm(model: Optional[LocalModel] = None) -> bool:
    """Coder / explicit backend → vLLM; else HF worker."""
    forced = (os.environ.get("LOCAL_QWEN_BACKEND") or "").strip().lower()
    if forced in ("vllm", "worker"):
        return forced == "vllm"
    if model is None:
        return True  # option A default
    blob = f"{model.key} {model.model_id} {model.model_dir} {model.label}".lower()
    return "coder" in blob


def api_base_for(model: Optional[LocalModel] = None) -> str:
    """按后端选 API base。

    scripts/launch/run_local.sh 会导出 LOCAL_QWEN_API_BASE=:8100（HF worker 遗留）。
    Coder / vLLM 必须走 :8000，不能被该变量盖掉。
    """
    if uses_vllm(model):
        return f"http://127.0.0.1:{VLLM_PORT}/v1"
    override = (os.environ.get("LOCAL_QWEN_API_BASE") or "").strip().rstrip("/")
    # worker 模型：可用 LOCAL_QWEN_API_BASE，但忽略误指向 vLLM 端口的值
    if override and f":{VLLM_PORT}" not in override:
        return override
    return f"http://127.0.0.1:{WORKER_PORT}/v1"


def fetch_health(timeout_s: float = 1.5) -> Dict[str, Any]:
    ok, body = _http_json(f"http://127.0.0.1:{WORKER_PORT}/health", timeout_s=timeout_s)
    return body if ok else {}


def hostctl_ok(timeout_s: float = 0.8) -> bool:
    ok, body = _http_json(f"{CTL_URL}/health", timeout_s=timeout_s)
    return ok and bool(body.get("ok"))


def _pgrep_f(pattern: str) -> List[int]:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return []
    pids: List[int] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _worker_pids() -> List[int]:
    return _pgrep_f(WORKER_NAME)


def _vllm_pids() -> List[int]:
    return _pgrep_f(VLLM_PROC_MATCH)


def _status_from_vllm(timeout_s: float = 1.5) -> Optional[ModelStatus]:
    ok, body = _http_json(
        f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout_s=timeout_s
    )
    if not ok:
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        # Port open but still loading / empty
        if _vllm_pids():
            return ModelStatus(
                healthy=False,
                loading=True,
                model_id="",
                model_dir="",
                device="cuda",
                kind="vllm",
                error="",
                process_hint="pid=" + ",".join(str(p) for p in _vllm_pids()[:3]),
                ctl_ok=hostctl_ok(),
                api_base=f"http://127.0.0.1:{VLLM_PORT}/v1",
            )
        return None
    first = data[0] if isinstance(data[0], dict) else {}
    mid = str(first.get("id") or "")
    mdir = str(first.get("root") or first.get("model_dir") or "")
    return ModelStatus(
        healthy=True,
        loading=False,
        model_id=mid,
        model_dir=mdir,
        device="cuda",
        kind="vllm",
        error="",
        process_hint="",
        ctl_ok=hostctl_ok(),
        api_base=f"http://127.0.0.1:{VLLM_PORT}/v1",
    )


def fetch_status(timeout_s: float = 1.5) -> ModelStatus:
    ctl = hostctl_ok()
    vllm = _status_from_vllm(timeout_s=timeout_s)
    if vllm is not None:
        return vllm

    body = fetch_health(timeout_s=timeout_s)
    pids = _worker_pids() + _vllm_pids()
    if body.get("ok"):
        return ModelStatus(
            healthy=True,
            loading=bool(body.get("loading")),
            model_id=str(body.get("model") or ""),
            model_dir=str(body.get("model_dir") or ""),
            device=str(body.get("device") or ""),
            kind=str(body.get("kind") or ""),
            error=str(body.get("error") or ""),
            process_hint="",
            ctl_ok=ctl,
            api_base=f"http://127.0.0.1:{WORKER_PORT}/v1",
        )
    if body.get("loading"):
        return ModelStatus(
            healthy=False,
            loading=True,
            model_id=str(body.get("model") or ""),
            model_dir=str(body.get("model_dir") or ""),
            device=str(body.get("device") or ""),
            kind=str(body.get("kind") or ""),
            error=str(body.get("error") or ""),
            process_hint="",
            ctl_ok=ctl,
            api_base=f"http://127.0.0.1:{WORKER_PORT}/v1",
        )
    hint = ""
    if pids:
        hint = "pid=" + ",".join(str(p) for p in pids[:3])
    err = str(body.get("error") or "")
    return ModelStatus(
        healthy=False,
        loading=False,
        model_id=str(body.get("model") or ""),
        model_dir=str(body.get("model_dir") or ""),
        device="",
        kind="",
        error=err,
        process_hint=hint,
        ctl_ok=ctl,
        api_base=API_BASE,
    )


def find_model(key_or_id: str) -> Optional[LocalModel]:
    want = (key_or_id or "").strip()
    if not want:
        return None
    for m in list_local_models(only_available=False):
        if m.key == want or m.model_id == want:
            return m if m.available else None
    return None


def sync_codex_local_model(model: LocalModel, *, reasoning: str = "high") -> Tuple[bool, str]:
    """把 Codex HOME 指向本地 Responses 服务并写入所选 model id。"""
    from embodied_evo.catalog import ensure_codex_home, resolve_codex_home

    base = api_base_for(model)
    via = "vLLM" if uses_vllm(model) else "local_qwen_worker"
    home = resolve_codex_home()
    ok, msg, home = ensure_codex_home(
        provider="compatible",
        base_url=base,
        api_key=os.environ.get("OPENAI_API_KEY", "local-qwen"),
    )
    config = home / "config.toml"
    # Codex 默认以为上下文 ~258k；本地 vLLM 实际只有 MAX_MODEL_LEN（默认 16k）。
    # 不写 model_context_window 会导致不 compact，直到 vLLM 直接 400。
    ctx = int(os.environ.get("LOCAL_QWEN_CONTEXT_WINDOW", "16384"))
    compact = int(os.environ.get("LOCAL_QWEN_AUTO_COMPACT", str(max(4096, ctx - 4096))))
    text = (
        f"# Local {model.label} via {via} (Responses)\n"
        f'model = "{model.model_id}"\n'
        f'model_reasoning_effort = "{reasoning}"\n'
        f"model_context_window = {ctx}\n"
        f"model_auto_compact_token_limit = {compact}\n"
        'model_provider = "compatible"\n'
        'cli_auth_credentials_store = "file"\n'
        "\n"
        "[model_providers.compatible]\n"
        f'name = "Local {model.label}"\n'
        f'base_url = "{base}"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = false\n"
    )
    config.write_text(text, encoding="utf-8")
    try:
        config.chmod(0o600)
    except OSError:
        pass
    os.environ["EMBODIEDEVO_CODEX_HOME"] = str(home)
    return True, f"已同步 Codex → {model.model_id} @ {base} ({msg})"


def _deploy_vllm(model: LocalModel) -> Tuple[bool, str]:
    if not VLLM_SCRIPT.is_file():
        return False, f"未找到 {VLLM_SCRIPT}"
    env = os.environ.copy()
    env["MODEL_DIR"] = model.model_dir
    env["SERVED_MODEL_NAME"] = model.model_id
    env["PORT"] = str(VLLM_PORT)
    env["HOST"] = "127.0.0.1"
    try:
        proc = subprocess.run(
            ["bash", str(VLLM_SCRIPT)],
            cwd=str(EAI_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return False, f"vLLM 启动失败: {exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, out or f"vLLM 启动退出码 {proc.returncode}"
    base = api_base_for(model)
    return True, out or f"已启动 {model.label} (vLLM) → {base}"


def deploy_model(model: LocalModel) -> Tuple[bool, str]:
    """部署所选本地模型。Coder → vLLM :8000；其它 → local_qwen_worker :8100。"""
    if not model.available or not model.model_dir:
        return False, f"模型不可用: {model.label}"

    status = fetch_status()
    if status.healthy:
        live = status.model_id or status.model_dir
        same_id = status.model_id == model.model_id
        same_dir = (
            status.model_dir.rstrip("/") == model.model_dir.rstrip("/")
            or Path(status.model_dir).name == Path(model.model_dir).name
        )
        if same_id or same_dir:
            return True, f"同一模型已在运行，跳过部署 ({model.label})"
        return (
            False,
            f"已在运行其它模型（{live}）。请先停止再部署 {model.label}。",
        )

    if uses_vllm(model):
        return _deploy_vllm(model)

    payload = {
        "model_dir": model.model_dir,
        "model_id": model.model_id,
        "model_label": model.label,
    }
    if hostctl_ok():
        ok, body = _http_json(
            f"{CTL_URL}/start",
            method="POST",
            payload=payload,
            timeout_s=30.0,
        )
        msg = str(body.get("message") or body)
        return (ok and bool(body.get("ok", ok))), msg

    if not RUN_SCRIPT.is_file():
        return False, f"未找到 {RUN_SCRIPT}"

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "a", encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["LOCAL_QWEN_MODEL_DIR"] = model.model_dir
    env["LOCAL_QWEN_MODEL_ID"] = model.model_id
    env["LOCAL_QWEN_PORT"] = str(WORKER_PORT)
    py = str(env.get("LOCAL_QWEN_PYTHON") or DEFAULT_PYTHON).strip()
    cmd = [
        "bash",
        str(RUN_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(WORKER_PORT),
        "--model",
        model.model_dir,
        "--model-id",
        model.model_id,
        "--python",
        py,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(EAI_DIR),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        log_fh.close()
        return False, f"启动失败: {exc}"
    log_fh.write(f"[embodied_evo] start pid={proc.pid} {' '.join(cmd)}\n")
    log_fh.flush()
    base = api_base_for(model)
    return True, f"已启动 {model.label} pid={proc.pid} → {base}"


def stop_model() -> Tuple[bool, str]:
    """停止本地推理服务（vLLM :8000 与/或 worker :8100）。"""
    if hostctl_ok():
        ok, body = _http_json(f"{CTL_URL}/stop", method="POST", timeout_s=15.0)
        msg = str(body.get("message") or body)
        # still try local ports below
        if ok and body.get("ok", ok):
            pass

    if VLLM_SCRIPT.is_file():
        try:
            subprocess.run(
                ["bash", str(VLLM_SCRIPT), "stop"],
                cwd=str(EAI_DIR),
                capture_output=True,
                timeout=15,
                check=False,
            )
        except Exception:
            pass

    for port in (VLLM_PORT, WORKER_PORT):
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                timeout=3,
                check=False,
            )
        except Exception:
            pass
    for pid in _worker_pids() + _vllm_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    return True, "已停止本地推理服务（vLLM/worker）"
