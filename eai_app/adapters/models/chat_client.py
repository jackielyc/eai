"""adapters.models.chat client — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

@_eai_platform_bootstrap.dataclass
class LlmChatConfig:
    api_base: str = _eai_configuration_chat.LLM_API_BASE_DEFAULT
    model: str = _eai_configuration_chat.LLM_MODEL_DEFAULT
    api_key: str = ""
    enable_thinking: bool = False
    max_tokens: int = 1536
    system_prompt: str = _eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> LlmChatConfig:
        settings = _eai_infrastructure_ui_storage.load_chat_user_settings()
        system_prompt = str(
            settings.get("system_prompt") or _eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        ).strip() or _eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        return cls(
            api_base=_eai_platform_bootstrap.os.environ.get("LLM_API_BASE", _eai_configuration_chat.LLM_API_BASE_DEFAULT),
            model=_eai_platform_bootstrap.os.environ.get("LLM_MODEL", _eai_configuration_chat.LLM_MODEL_DEFAULT),
            api_key=_eai_platform_bootstrap.os.environ.get(_eai_configuration_chat.LLM_API_KEY_ENV, "").strip(),
            system_prompt=system_prompt,
        )


def encode_bgr_image_jpeg_b64(image_bgr: _eai_platform_bootstrap.np.ndarray, max_side: int = 1280) -> str:
    """BGR 图像 → JPEG base64（供 OpenAI vision image_url）。

    max_side<=0 时不缩放，保持原始宽高。
    """
    img = _eai_platform_bootstrap.np.asarray(image_bgr)
    if img.ndim == 2:
        img = _eai_platform_bootstrap.cv2.cvtColor(img, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if max_side and max_side > 0:
        scale = min(1.0, float(max_side) / float(max(h, w, 1)))
        if scale < 0.999:
            img = _eai_platform_bootstrap.cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
            )
    ok, buf = _eai_platform_bootstrap.cv2.imencode(".jpg", img, [int(_eai_platform_bootstrap.cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return _eai_platform_bootstrap.base64.b64encode(buf.tobytes()).decode("ascii")


def _http_get_json(
    url: str,
    headers: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, str]] = None,
    timeout_s: float = 5.0,
) -> _eai_platform_bootstrap.Tuple[bool, object, str]:
    req = _eai_platform_bootstrap.urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, _eai_platform_bootstrap.json.loads(raw), ""
            except Exception:
                return True, raw, ""
    except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, None, f"HTTP {exc.code}: {detail[:200]}"
    except Exception as exc:
        return False, None, str(exc)


def _http_post_json(
    url: str,
    payload: _eai_platform_bootstrap.Dict[str, object],
    headers: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, str]] = None,
    timeout_s: float = 300.0,
) -> _eai_platform_bootstrap.Tuple[bool, object, str]:
    data = _eai_platform_bootstrap.json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        hdrs.update(headers)
    req = _eai_platform_bootstrap.urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, _eai_platform_bootstrap.json.loads(raw), ""
            except Exception:
                return True, raw, ""
    except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        # FastAPI JSON {"detail": "..."} or plain text
        parsed = ""
        if detail.startswith("{"):
            try:
                obj = _eai_platform_bootstrap.json.loads(detail)
                if isinstance(obj, dict):
                    parsed = str(obj.get("detail") or obj.get("error") or detail)
            except Exception:
                parsed = detail
        else:
            parsed = detail
        msg = parsed or exc.reason or "Internal Server Error"
        # 裸 "Internal Server Error" 多半是服务端异常处理器丢了 detail / 进程崩溃
        if msg.strip() in ("Internal Server Error", "Internal Server Error."):
            msg = (
                "Internal Server Error（服务端未返回 detail；"
                "常见原因：Bagel 仍在加载、CUDA OOM、或异常处理器异常。"
                "请查看 eai/.cache/bagel_serve_api_errors.log 并重启 Bagel API）"
            )
        return False, None, f"HTTP {exc.code}: {msg[:800]}"
    except Exception as exc:
        return False, None, str(exc)


def bagel_size_from_bgr(image_bgr: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[int, int]:
    """从输入图取输出 (W, H)，对齐 16 且不超过 1024。"""
    h, w = image_bgr.shape[:2]
    w = max(16, (int(w) // 16) * 16)
    h = max(16, (int(h) // 16) * 16)
    return (min(w, 1024), min(h, 1024))


def call_bagel_inference(
    api_base: str,
    prompt: str,
    *,
    image_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    task: str = "",
    num_timesteps: int = _eai_configuration_models.BAGEL_NUM_TIMESTEPS_DEFAULT,
    image_ratio: str = _eai_configuration_models.BAGEL_IMAGE_SIZE_DEFAULT,
    image_width: int = 0,
    image_height: int = 0,
    cfg_interval: float = _eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT,
    enable_taylorseer: bool = False,
    timeout_s: float = 300.0,
) -> _eai_platform_bootstrap.Tuple[str, float, _eai_platform_bootstrap.Dict[str, object]]:
    """调用 Bagel serve_api.py：有图则理解，无图则文生图。

    返回 (文本或 BAGEL_IMAGE::路径, 耗时秒, 元信息)。
    元信息可能含 model_input_width/height、image_width/height 等。
    """
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/v1"):
        root = base
    else:
        root = base + "/v1"
    prompt = (prompt or "").strip()
    if not prompt:
        raise RuntimeError("prompt 为空")
    kind = (task or "").strip() or ("understand" if image_bgr is not None else "text2image")
    payload: _eai_platform_bootstrap.Dict[str, object] = {"prompt": prompt}
    steps = max(1, int(num_timesteps or _eai_configuration_models.BAGEL_NUM_TIMESTEPS_DEFAULT))
    ratio = (image_ratio or _eai_configuration_models.BAGEL_IMAGE_SIZE_DEFAULT).strip() or _eai_configuration_models.BAGEL_IMAGE_SIZE_DEFAULT
    width = int(image_width or 0)
    height = int(image_height or 0)
    cfg_iv = max(0.0, min(1.0, float(cfg_interval)))
    # 默认「跟随输入」：有输入图则用其尺寸
    if ratio in ("input", "auto", "follow") and width <= 0 and height <= 0:
        if image_bgr is not None:
            width, height = bagel_size_from_bgr(image_bgr)
            ratio = "custom"
        else:
            ratio = _eai_configuration_models.BAGEL_IMAGE_SIZE_FALLBACK
    # 纯数字边长（如 224/320）：显式传宽高，兼容旧 serve_api 未登记的映射
    if ratio.isdigit() and width <= 0 and height <= 0:
        side = max(16, int(ratio))
        width = height = side
        ratio = "custom"
    if kind in ("understand", "edit"):
        if image_bgr is None:
            raise RuntimeError("图像理解/编辑需要输入图")
        payload["image_base64"] = encode_bgr_image_jpeg_b64(image_bgr, max_side=0)
        url = f"{root}/{kind}"
        if kind == "edit":
            payload["num_timesteps"] = steps
            payload["cfg_interval"] = cfg_iv
            payload["enable_taylorseer"] = bool(enable_taylorseer)
            if width > 0 and height > 0:
                payload["image_width"] = width
                payload["image_height"] = height
            elif ratio not in ("input", "auto", "follow", "custom", ""):
                payload["image_ratio"] = ratio
    else:
        url = f"{root}/text2image"
        payload["image_ratio"] = ratio if ratio != "custom" else _eai_configuration_models.BAGEL_IMAGE_SIZE_FALLBACK
        payload["num_timesteps"] = steps
        payload["cfg_interval"] = cfg_iv
        payload["enable_taylorseer"] = bool(enable_taylorseer)
        if width > 0 and height > 0:
            payload["image_width"] = width
            payload["image_height"] = height
    t0 = _eai_platform_bootstrap.time.perf_counter()
    ok, body, err = _http_post_json(url, payload, timeout_s=timeout_s)
    wall_s = _eai_platform_bootstrap.time.perf_counter() - t0
    if not ok:
        raise RuntimeError(err or "Bagel 调用失败")
    if not isinstance(body, dict):
        raise RuntimeError(f"Bagel 返回异常: {body}")
    if body.get("ok") is False:
        raise RuntimeError(str(body.get("detail") or body.get("error") or body))
    try:
        latency_s = float(body.get("latency_s"))
    except (TypeError, ValueError):
        latency_s = wall_s
    if latency_s <= 0:
        latency_s = wall_s
    meta: _eai_platform_bootstrap.Dict[str, object] = {"latency_s": latency_s}
    for key in (
        "model_input_width",
        "model_input_height",
        "gen_width",
        "gen_height",
        "image_width",
        "image_height",
    ):
        if key in body and body.get(key) is not None:
            try:
                meta[key] = int(body.get(key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
    text = str(body.get("text") or "").strip()
    b64 = str(body.get("image_base64") or "").strip()
    if b64:
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_models.BAGEL_API_OUTPUT_DIR, exist_ok=True)
        out_path = _eai_platform_bootstrap.os.path.join(
            _eai_configuration_models.BAGEL_API_OUTPUT_DIR, f"bagel_{int(_eai_platform_bootstrap.time.time())}_{_eai_platform_bootstrap.uuid.uuid4().hex[:8]}.png"
        )
        raw = b64
        if "," in raw and raw.lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        with open(out_path, "wb") as f:
            f.write(_eai_platform_bootstrap.base64.b64decode(raw))
        prefix = f"BAGEL_IMAGE::{out_path}"
        return prefix + (f"\n{text}" if text else ""), latency_s, meta
    return (text or "(Bagel 无文本输出)"), latency_s, meta


def classify_llm_service_kind(
    api_base: str,
    model_ids: _eai_platform_bootstrap.Sequence[str],
    *,
    ollama_ok: bool = False,
    health: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None,
    configured_model: str = "",
) -> str:
    base = (api_base or "").lower()
    ids = [str(x).strip() for x in model_ids if str(x).strip()]
    ids_l = " ".join(ids).lower()
    cfg = (configured_model or "").lower()
    health = health or {}

    if "dashscope.aliyuncs.com" in base or "compatible-mode" in base:
        return "阿里云百炼 / DashScope"
    if "api.openai.com" in base:
        return "OpenAI 官方"
    if "api.anthropic.com" in base:
        return "Anthropic"
    if ":8090" in base or "hy-rxbrain" in ids_l or "hy-rxbrain" in cfg:
        return "Hy-Embodied-RxBrain"
    if ":8080" in base or "hy_a3b" in ids_l or "hy_a3b" in cfg or "hy-embodied" in ids_l:
        return "Hy-Embodied-VLM"
    if ollama_ok or ":11434" in base:
        if "qwen3-vl" in ids_l or "qwen3-vl" in cfg:
            return "Ollama · Qwen3-VL"
        if "qwen3.5" in ids_l or "qwen3.5" in cfg:
            return "Ollama · Qwen3.5"
        return "Ollama"
    model_dir = str(health.get("model_dir") or "").lower()
    live = str(health.get("model") or "").lower()
    blob = f"{ids_l} {cfg} {model_dir} {live}"
    if (
        live
        or model_dir
        or "qwen3.5-4b" in blob
        or "qwen3.5-35b" in blob
        or "/models/qwen/" in model_dir
    ):
        if "35b" in blob or "a3b" in blob:
            return "本地 Qwen3.5-35B-A3B 服务"
        if "4b" in blob:
            return "本地 Qwen3.5-4B 服务"
        return "本地 Qwen 推理服务"
    if ids:
        return "OpenAI 兼容服务"
    return "未知服务"


def probe_llm_endpoint(
    api_base: str,
    api_key: str = "",
    configured_model: str = "",
    timeout_s: float = 6.0,
) -> _eai_platform_bootstrap.Dict[str, object]:
    """探测 API 地址上的服务类型与可用模型列表。"""
    base = (api_base or "").strip().rstrip("/") or _eai_configuration_chat.LLM_API_BASE_DEFAULT
    root = base.removesuffix("/v1").rstrip("/")
    key = (api_key or "").strip() or "EMPTY"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    details: _eai_platform_bootstrap.List[str] = []
    model_ids: _eai_platform_bootstrap.List[str] = []
    health: _eai_platform_bootstrap.Dict[str, object] = {}
    ollama_ok = False
    reachable = False

    ok, body, err = _http_get_json(f"{root}/health", headers=headers, timeout_s=timeout_s)
    if ok:
        reachable = True
        details.append("健康检查: /health 可达")
        if isinstance(body, dict):
            health = body
            if body.get("model"):
                mid = str(body.get("model"))
                if mid not in model_ids:
                    model_ids.append(mid)
            if body.get("model_dir"):
                details.append(f"model_dir={body.get('model_dir')}")
            if body.get("device"):
                details.append(f"device={body.get('device')}")
            if body.get("ok") is False and body.get("error"):
                details.append(f"服务未就绪: {body.get('error')}")
    else:
        details.append(f"健康检查: /health 不可用 ({err})")

    ok, body, err = _http_get_json(
        f"{base}/models", headers=headers, timeout_s=timeout_s
    )
    if ok:
        reachable = True
        details.append("模型列表: /models 可达")
        if isinstance(body, dict):
            for item in body.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"])
                    if mid not in model_ids:
                        model_ids.append(mid)
        elif isinstance(body, list):
            for item in body:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"])
                    if mid not in model_ids:
                        model_ids.append(mid)
    else:
        details.append(f"模型列表: /models 不可用 ({err})")

    ok, body, err = _http_get_json(f"{root}/api/tags", timeout_s=timeout_s)
    if ok:
        reachable = True
        ollama_ok = True
        details.append("Ollama: /api/tags 可达")
        if isinstance(body, dict):
            for item in body.get("models") or []:
                if isinstance(item, dict):
                    mid = str(item.get("name") or item.get("model") or "").strip()
                    if mid and mid not in model_ids:
                        model_ids.append(mid)
    else:
        details.append(f"Ollama: /api/tags 不可用 ({err})")

    kind = classify_llm_service_kind(
        base,
        model_ids,
        ollama_ok=ollama_ok,
        health=health,
        configured_model=configured_model,
    )
    matched = False
    if configured_model:
        cfg_l = configured_model.lower()
        matched = any(cfg_l == m.lower() or cfg_l in m.lower() or m.lower() in cfg_l for m in model_ids)

    return {
        "ok": reachable,
        "kind": kind,
        "api_base": base,
        "root": root,
        "configured_model": configured_model,
        "configured_model_matched": matched,
        "models": model_ids,
        "health": health,
        "details": details,
        "error": "" if reachable else "无法连接当前 API",
    }


def discover_running_inference_services(
    timeout_s: float = 1.0,
    extra_candidates: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[_eai_platform_bootstrap.Tuple[str, str, str, str]]] = None,
) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]]:
    """并行探测本机/隧道上已运行的推理服务，供 AI 对话选择。

    仅返回健康检查或 /models / Ollama tags 可达的本地类服务（不含云端百炼/OpenAI）。
    """
    candidates: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str, str, str]] = [
        (
            "本地 Qwen",
            _eai_adapters_models_qwen_local.resolve_local_qwen_viewer_api_base(),
            "EMPTY",
            "",
        ),
        (
            "远程 Qwen",
            _eai_configuration_policies.REMOTE_QWEN_API_BASE_DEFAULT,
            "EMPTY",
            "",
        ),
        (
            "Ollama",
            _eai_configuration_perception.OLLAMA_API_BASE_DEFAULT,
            "ollama",
            "",
        ),
        (
            "Hy-Embodied-VLM",
            _eai_configuration_chat.HY_EMBODIED_VLM_API_BASE,
            "EMPTY",
            _eai_configuration_chat.HY_EMBODIED_VLM_MODEL,
        ),
        (
            "Hy-Embodied-RxBrain",
            _eai_configuration_chat.HY_RXBRAIN_API_BASE,
            "EMPTY",
            _eai_configuration_chat.HY_RXBRAIN_MODEL,
        ),
        (
            "Bagel",
            f"http://127.0.0.1:{_eai_configuration_models.BAGEL_API_PORT_DEFAULT}/v1",
            "EMPTY",
            "BAGEL-7B-MoT",
        ),
    ]
    if extra_candidates:
        candidates.extend(extra_candidates)

    def _probe_one(
        family: str, api_base: str, api_key: str, hint: str
    ) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]]:
        result = probe_llm_endpoint(
            api_base,
            api_key=api_key,
            configured_model=hint,
            timeout_s=timeout_s,
        )
        if not result.get("ok"):
            return []
        health = result.get("health") or {}
        if isinstance(health, dict) and health.get("ok") is False:
            return []
        models = [
            str(m).strip()
            for m in (result.get("models") or [])
            if str(m).strip()
        ]
        if not models:
            live = ""
            if isinstance(health, dict):
                live = str(health.get("model") or "").strip()
            models = [m for m in (live, hint) if m]
        if not models:
            return []
        base = str(result.get("api_base") or api_base).rstrip("/")
        out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]] = []
        for mid in models:
            out.append(
                {
                    "label": f"{family} · {mid}",
                    "family": family,
                    "api_base": base,
                    "model": mid,
                    "api_key": api_key,
                }
            )
        return out

    found: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, str]] = []
    seen: set = set()
    with _eai_platform_bootstrap.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futs = [
            pool.submit(_probe_one, fam, base, key, hint)
            for fam, base, key, hint in candidates
        ]
        for fut in _eai_platform_bootstrap.as_completed(futs):
            try:
                items = fut.result()
            except Exception:
                continue
            for item in items:
                dedupe = f"{item['api_base']}|{item['model']}"
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                found.append(item)

    def _sort_key(item: _eai_platform_bootstrap.Dict[str, str]) -> _eai_platform_bootstrap.Tuple[int, str, str]:
        family = item.get("family") or ""
        order = {
            "本地 Qwen": 0,
            "远程 Qwen": 1,
            "Ollama": 2,
            "Hy-Embodied-VLM": 3,
            "Hy-Embodied-RxBrain": 4,
            "Bagel": 5,
        }.get(family, 9)
        return (order, family, item.get("model") or "")

    found.sort(key=_sort_key)
    return found


def format_chat_prompt_dump(
    messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]],
    *,
    api_base: str = "",
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 0,
    enable_thinking: bool = False,
    timeout_s: float = 0.0,
) -> str:
    """把即将发给 API 的提示词格式化为可读文本（图片 base64 截断）。"""
    lines: _eai_platform_bootstrap.List[str] = [
        "======== 完整请求提示词 ========",
        f"api_base: {api_base}",
        f"model: {model}",
        f"temperature: {temperature}",
        f"max_tokens: {max_tokens}",
        f"chat_template_kwargs.enable_thinking: {bool(enable_thinking)}",
    ]
    if timeout_s > 0:
        lines.append(f"timeout_s: {timeout_s}")
    lines.append(f"messages({len(messages)}):")
    for i, msg in enumerate(messages):
        role = str(msg.get("role") or "")
        content = msg.get("content")
        lines.append(f"----- [{i}] role={role} -----")
        if isinstance(content, str):
            lines.append(content if content.strip() else "(空文本)")
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    lines.append(f"  part[{j}]: {part!r}")
                    continue
                ptype = str(part.get("type") or "")
                if ptype == "text":
                    lines.append(f"  part[{j}] text: {part.get('text') or ''}")
                elif ptype in ("image_url", "image"):
                    url = ""
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url") or "")
                    elif isinstance(image_url, str):
                        url = image_url
                    elif part.get("url"):
                        url = str(part.get("url") or "")
                    if url.startswith("data:"):
                        header, _, rest = url.partition(",")
                        lines.append(
                            f"  part[{j}] image: {header},<base64 len={len(rest)}>"
                        )
                    elif url:
                        lines.append(f"  part[{j}] image_url: {url[:200]}")
                    else:
                        lines.append(f"  part[{j}] image: (无 url)")
                else:
                    lines.append(f"  part[{j}] type={ptype}: {part!r}"[:500])
        elif content is None:
            lines.append("(无 content)")
        else:
            lines.append(repr(content)[:1000])
    lines.append("======== 提示词结束 ========")
    return "\n".join(lines)


class LlmChatClient:
    """OpenAI 兼容 Chat Completions API 客户端（支持 OpenAI / Ollama / Hy-Embodied 等）。"""

    def __init__(self, config: LlmChatConfig) -> None:
        self.config = config

    def _auth_headers(self) -> _eai_platform_bootstrap.Dict[str, str]:
        api_key = self.config.api_key.strip() or "EMPTY"
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def probe(self, timeout_s: float = 6.0) -> _eai_platform_bootstrap.Dict[str, object]:
        """探测当前 API 可达性与模型种类。"""
        return probe_llm_endpoint(
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            configured_model=self.config.model,
            timeout_s=timeout_s,
        )

    def chat(
        self,
        messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]],
        timeout_s: _eai_platform_bootstrap.Optional[float] = None,
    ) -> str:
        api_key = self.config.api_key.strip()
        if not api_key:
            raise RuntimeError(
                f"未配置 API Key，请设置环境变量 {_eai_configuration_chat.LLM_API_KEY_ENV}，"
                "或在对话面板中填写（Ollama / Hy-Embodied 本地服务可填 EMPTY）"
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload: _eai_platform_bootstrap.Dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": int(self.config.max_tokens),
            # Qwen3.5 默认会进 <think>；未勾选 thinking 时显式关闭，避免长思考挤掉答案
            "chat_template_kwargs": {
                "enable_thinking": bool(self.config.enable_thinking)
            },
        }
        # 带图推理更慢（视觉编码 + 长输出），单独放宽超时
        if timeout_s is None:
            has_image = False
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and str(part.get("type") or "") in (
                            "image_url",
                            "image",
                        ):
                            has_image = True
                            break
                if has_image:
                    break
            timeout_s = (
                _eai_configuration_chat.LLM_CHAT_VISION_TIMEOUT_S if has_image else _eai_configuration_chat.LLM_CHAT_TIMEOUT_S
            )
        req = _eai_platform_bootstrap.urllib.request.Request(
            url,
            data=_eai_platform_bootstrap.json.dumps(payload).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )
        try:
            with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = _eai_platform_bootstrap.json.loads(resp.read().decode("utf-8"))
        except _eai_platform_bootstrap.urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except _eai_platform_bootstrap.urllib.error.URLError as exc:
            raise RuntimeError(f"网络错误: {exc.reason}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 无有效回复: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        parts: _eai_platform_bootstrap.List[str] = []
        if reasoning:
            parts.append(f"[thinking]\n{str(reasoning).strip()}")
        if content:
            parts.append(str(content).strip() if not reasoning else f"[answer]\n{str(content).strip()}")
        if not parts:
            raise RuntimeError(f"API 返回空内容: {body}")
        return "\n\n".join(parts).strip()


class LlmChatBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(str, bool)


class LlmProbeBridge(_eai_platform_bootstrap.QObject):
    finished = _eai_platform_bootstrap.pyqtSignal(object)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _utc_now_iso() -> str:
    return _eai_platform_bootstrap.datetime.now(_eai_platform_bootstrap.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

