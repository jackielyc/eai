"""adapters.models.bagel runtime — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_bagel_root(path: str = "") -> str:
    raw = (path or "").strip() or _eai_platform_bootstrap.os.environ.get("BAGEL_DIR", "") or _eai_configuration_models.BAGEL_ROOT_DEFAULT
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def _bagel_python_looks_wrong(path: str) -> bool:
    """ros-humble / Qwen2.5-VL 没有兼容的 gradio 栈，不能跑 Bagel。"""
    p = _eai_platform_bootstrap.os.path.normpath(path or "")
    return ("ros-humble" in p) or ("Qwen2.5-VL" in p)


def resolve_bagel_python(repo: str) -> str:
    """优先 BAGEL_PYTHON / miniconda bagel，最后当前解释器。"""
    env = (_eai_platform_bootstrap.os.environ.get("BAGEL_PYTHON") or "").strip()
    if env and _eai_platform_bootstrap.os.path.isfile(env) and not _bagel_python_looks_wrong(env):
        return env
    candidates = [
        _eai_configuration_models.BAGEL_VENV_PYTHON_DEFAULT,
        "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/bagel/bin/python",
        "/home/psibot/miniconda3/envs/bagel/bin/python",
        "/home/psibot/miniconda3/envs/Bagel/bin/python",
        _eai_platform_bootstrap.sys.executable,
    ]
    for path in candidates:
        if path and _eai_platform_bootstrap.os.path.isfile(path) and not _bagel_python_looks_wrong(path):
            return path
    return _eai_platform_bootstrap.sys.executable or "python3"


def bagel_mode_label(mode: int) -> str:
    """Bagel --mode 的可读标签。"""
    labels = {
        1: "1 · 全精度 (bf16)",
        2: "2 · NF4 量化",
        3: "3 · INT8 量化",
    }
    try:
        m = int(mode)
    except (TypeError, ValueError):
        m = 1
    return labels.get(m, f"{m}")


def resolve_bagel_model_path(repo: str, model_path: str = "") -> str:
    """解析模型目录：相对仓库路径 / 绝对路径 / models 默认。"""
    raw = (model_path or "").strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
    if _eai_platform_bootstrap.os.path.isabs(raw):
        cand = raw
    else:
        cand = _eai_platform_bootstrap.os.path.join(repo, raw)
    if _eai_platform_bootstrap.os.path.isdir(cand):
        return cand
    if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT):
        return _eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT
    return cand


def bagel_isolated_process_env(python_bin: str = "") -> _eai_platform_bootstrap.QProcessEnvironment:
    """去掉 RoboStack PYTHONPATH，避免 bagel env (3.10) 误用 ros-humble (3.11) 的 numpy。"""
    qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
    qenv.remove("PYTHONPATH")
    qenv.remove("PYTHONHOME")
    # Proxy often breaks Gradio's localhost reachability check.
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        qenv.remove(key)
    no_proxy = qenv.value("NO_PROXY", "") or qenv.value("no_proxy", "")
    local_no = "127.0.0.1,localhost,::1"
    merged = ",".join(
        p for p in (local_no + ("," + no_proxy if no_proxy else "")).split(",") if p
    )
    qenv.insert("NO_PROXY", merged)
    qenv.insert("no_proxy", merged)
    bagel_tmp = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "bagel_tmp")
    try:
        _eai_platform_bootstrap.os.makedirs(bagel_tmp, exist_ok=True)
    except OSError:
        bagel_tmp = "/tmp"
    qenv.insert("TMPDIR", bagel_tmp)
    qenv.insert("TEMP", bagel_tmp)
    qenv.insert("TMP", bagel_tmp)
    qenv.insert("GRADIO_TEMP_DIR", bagel_tmp)
    qenv.insert("PYTHONNOUSERSITE", "1")
    qenv.insert("PYTHONUNBUFFERED", "1")
    qenv.insert("GRADIO_ANALYTICS_ENABLED", "False")
    # bitsandbytes INT8/NF4 会刷屏 MatMul8bitLt cast 警告；抑制后日志可读。
    qenv.insert("BITSANDBYTES_NOWELCOME", "1")
    existing_pw = (qenv.value("PYTHONWARNINGS", "") or "").strip()
    matmul_filter = "ignore:MatMul8bitLt:UserWarning"
    if matmul_filter not in existing_pw:
        qenv.insert(
            "PYTHONWARNINGS",
            f"{existing_pw},{matmul_filter}" if existing_pw else matmul_filter,
        )
    py = (python_bin or "").strip()
    if py and _eai_platform_bootstrap.os.path.isfile(py):
        venv_bin = _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.abspath(py))
        venv_root = _eai_platform_bootstrap.os.path.dirname(venv_bin)
        qenv.insert("VIRTUAL_ENV", venv_root)
        path_now = qenv.value("PATH", "")
        parts = [p for p in path_now.split(":") if p] if path_now else []
        if venv_bin not in parts:
            qenv.insert("PATH", f"{venv_bin}:{path_now}" if path_now else venv_bin)
    return qenv


def _is_noisy_bagel_log_line(line: str) -> bool:
    """bitsandbytes 量化推理时的无用刷屏行。"""
    text = (line or "").strip()
    if not text:
        return True
    if "MatMul8bitLt" in text:
        return True
    if "inputs will be cast from" in text and "during quantization" in text:
        return True
    return False


def bagel_app_url(host: str, port: int) -> str:
    h = (host or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT).strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
    if h in ("0.0.0.0", "::"):
        h = "127.0.0.1"
    return f"http://{h}:{int(port)}/"


BAGEL_PROCESS_PATTERNS: _eai_platform_bootstrap.Tuple[str, ...] = (
    "serve_api.py",
    "app.py --server_name",
    "app.py --server_port",
)


def bagel_probe_api_health(
    host: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
    port: int = _eai_configuration_models.BAGEL_API_PORT_DEFAULT,
    *,
    timeout_s: float = 2.0,
) -> _eai_platform_bootstrap.Optional[dict]:
    """探测已部署的 Bagel serve_api /health；确认是本服务才返回 body，否则 None。

    仅 ``{"ok": true}`` 不够：其它服务的 /health 也会误触发「复用已部署」。
    需同时具备 Bagel 字段，并校验 ``/v1/models`` 归属 bagel。
    """
    base = bagel_app_url(host, port).rstrip("/")
    ok, body, _err = _eai_adapters_models_chat_client._http_get_json(f"{base}/health", timeout_s=timeout_s)
    if not ok or not isinstance(body, dict):
        return None
    if not body.get("ok"):
        return None
    # Bagel serve_api 健康体特征
    if "loaded_at" not in body:
        return None
    try:
        mode = int(body.get("mode"))
    except (TypeError, ValueError):
        return None
    if mode not in (1, 2, 3):
        return None
    model = str(body.get("model") or "").strip()
    model_path = str(body.get("model_path") or "").strip()
    if not model and not model_path:
        return None
    # 二次确认：/v1/models 必须是 bagel
    ok2, models_body, _err2 = _eai_adapters_models_chat_client._http_get_json(f"{base}/v1/models", timeout_s=timeout_s)
    if not ok2 or not isinstance(models_body, dict):
        return None
    data = models_body.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0] if isinstance(data[0], dict) else {}
    owned = str(first.get("owned_by") or "").strip().lower()
    mid = str(first.get("id") or "").strip().lower()
    if owned != "bagel" and "bagel" not in mid:
        return None
    return body


def bagel_listener_matches(port: int, *, mode: int, model_path: str) -> _eai_platform_bootstrap.Optional[bool]:
    """Compare a listening Bagel process with the requested mode and weights."""
    text = _eai_adapters_models_qwen_local.listener_cmdline(int(port))
    if not text.strip():
        return None
    live_mode = ""
    found = _eai_platform_bootstrap.re.search(r"--mode(?:\s+|=)(\d+)", text)
    if found:
        live_mode = found.group(1)
    live_model = ""
    found = _eai_platform_bootstrap.re.search(r"--model_path(?:\s+|=)(\S+)", text)
    if found:
        live_model = found.group(1).strip("'\"")
    if live_mode and int(live_mode) != int(mode or 1):
        return False
    if live_model and model_path:
        return _eai_adapters_models_qwen_local.model_identity_same("", live_model, "", model_path)
    if live_mode and int(live_mode) == int(mode or 1):
        return True
    if model_path and _eai_platform_bootstrap.os.path.basename(model_path.rstrip("/")) in text:
        return True
    return None


def bagel_probe_gradio_ready(
    host: str = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT,
    port: int = _eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT,
    *,
    timeout_s: float = 2.0,
) -> bool:
    """探测已部署的 Bagel Gradio（非 serve_api）。命中则应复用，勿再启一份。"""
    # 同端口若是推理 API，不算 Gradio
    if bagel_probe_api_health(host, port, timeout_s=min(1.5, timeout_s)) is not None:
        return False
    url = bagel_app_url(host, port).rstrip("/") + "/"
    try:
        req = _eai_platform_bootstrap.urllib.request.Request(url, method="GET")
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(8192).decode("utf-8", errors="replace").lower()
            if "gradio" in raw or "svelte" in raw or "bagel" in raw:
                return True
            # Gradio 偶发极简 HTML；端口在听且非 API 时也视为已部署
            return int(getattr(resp, "status", 200) or 200) < 500
    except Exception:
        pass
    probe_host = (host or "").strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
    if probe_host in ("0.0.0.0", "::"):
        probe_host = "127.0.0.1"
    try:
        return bool(_eai_configuration_models.port_listening(probe_host, int(port)))
    except Exception:
        return False


def bagel_kill_related_processes(sig: str = "-TERM") -> int:
    """对所有 Bagel 相关进程发信号。返回成功匹配的 pkill 次数。"""
    hits = 0
    for pat in BAGEL_PROCESS_PATTERNS:
        try:
            result = _eai_platform_bootstrap.subprocess.run(
                ["pkill", sig, "-f", pat],
                check=False,
                capture_output=True,
                text=True,
            )
            # pkill: 0=匹配到, 1=无匹配
            if result.returncode == 0:
                hits += 1
        except Exception:
            continue
    return hits


_BAGEL_INJECT_IMAGE_JS = r"""
(function(b64, mime, name) {
  function b64ToUint8(b64Data) {
    var bin = atob(b64Data);
    var len = bin.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  var blob = new Blob([b64ToUint8(b64)], {type: mime || "image/png"});
  var file = new File([blob], name || "upload.png", {type: mime || "image/png"});
  var dt = new DataTransfer();
  dt.items.add(file);
  var inputs = Array.prototype.slice.call(document.querySelectorAll('input[type="file"]'));
  var candidates = inputs.filter(function(el) {
    if (el.disabled) return false;
    var acc = (el.accept || "").toLowerCase();
    return !acc || acc.indexOf("image") >= 0 || acc.indexOf("*") >= 0 || acc === "";
  });
  if (!candidates.length) return "no-file-input";
  var target = null;
  for (var i = 0; i < candidates.length; i++) {
    var el = candidates[i];
    var block = el.closest(".block, .form, [data-testid]") || el.parentElement;
    if (block && block.offsetParent !== null) target = el;
  }
  if (!target) target = candidates[candidates.length - 1];
  try {
    target.files = dt.files;
  } catch (e) {
    return "set-files-failed:" + String(e);
  }
  target.dispatchEvent(new Event("input", {bubbles: true}));
  target.dispatchEvent(new Event("change", {bubbles: true}));
  return "ok:" + (target.accept || "") + ":" + file.name + ":" + file.size;
})
"""


if _eai_platform_bootstrap.QWebEnginePage is not None:

    class BagelWebEnginePage(_eai_platform_bootstrap.QWebEnginePage):
        """用系统文件对话框选图，避免 WebEngine 默认上传得到空文件。"""

        def chooseFiles(self, mode, old_files, accepted_mime_types):  # type: ignore[override]
            mime = " ".join(accepted_mime_types or [])
            if "image" in mime.lower() or not mime.strip():
                filt = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All (*)"
            else:
                filt = "All (*)"
            parent = self.view().window() if self.view() is not None else None
            multi = False
            try:
                multi = mode == _eai_platform_bootstrap.QWebEnginePage.FileSelectOpenMultiple
            except Exception:
                multi = int(mode) == 1
            if multi:
                paths, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileNames(
                    parent, "选择图片", "", filt
                )
                return paths
            path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(parent, "选择图片", "", filt)
            return [path] if path else []

else:
    BagelWebEnginePage = None  # type: ignore

