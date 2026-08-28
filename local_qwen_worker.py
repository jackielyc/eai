#!/usr/bin/env python3
"""
本地 Qwen3.5 OpenAI 兼容推理服务。

读取 HuggingFace 权重目录（默认 workspace/models/Qwen/Qwen3.5-4B，
也可指向 Qwen3.5-35B-A3B 等），提供:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions

用法:
  bash run_local_qwen.sh
  bash run_local_qwen.sh --model /path/to/Qwen3.5-35B-A3B --model-id qwen3.5-35b-a3b
  LOCAL_QWEN_PYTHON=~/miniconda3/envs/psi-policy/bin/python \\
    python local_qwen_worker.py --model /path/to/Qwen3.5-4B --host 0.0.0.0 --port 8100
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


MODEL_ID = os.environ.get("LOCAL_QWEN_MODEL_ID", "qwen3.5-4b")
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False,
    "error": "",
    "model": None,
    "processor": None,
    "model_dir": "",
    "device": "",
}


def _log(msg: str) -> None:
    print(f"[local_qwen] {msg}", flush=True)


def _configure_torch_backends() -> None:
    """在首次 CUDA/视觉算子前配置后端。

    部分机房/驱动组合下 cuDNN 句柄初始化失败，会导致带图推理
    CUDNN_STATUS_NOT_INITIALIZED，纯文本却正常。默认关闭 cuDNN，
    走原生卷积（稍慢但稳）。可用 LOCAL_QWEN_CUDNN=1 强制开启。
    """
    try:
        import torch
    except Exception:
        return
    want = os.environ.get("LOCAL_QWEN_CUDNN", "").strip().lower()
    enable = want in ("1", "true", "yes", "on")
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
        torch.backends.cudnn.enabled = enable
        torch.backends.cudnn.benchmark = False
        _log(f"torch.backends.cudnn.enabled={enable}")
    except Exception as exc:
        _log(f"configure torch backends failed: {exc}")


def _ensure_cudnn_ready() -> None:
    """加载后确认后端；若仍开启 cuDNN 则做一次预热。"""
    _configure_torch_backends()
    import torch

    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        x = torch.zeros(1, 3, 2, 16, 16, device="cuda", dtype=torch.float16)
        w = torch.zeros(8, 3, 2, 3, 3, device="cuda", dtype=torch.float16)
        y = torch.nn.functional.conv3d(x, w, padding=1)
        del x, w, y
        torch.cuda.synchronize()
        _log("cuDNN warmup ok")
    except Exception as exc:
        _log(f"cuDNN warmup failed, disabling cudnn: {exc}")
        try:
            torch.backends.cudnn.enabled = False
        except Exception:
            pass


def resolve_model_dir(user: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if user:
        candidates.append(os.path.abspath(os.path.expanduser(user)))
    env = os.environ.get("LOCAL_QWEN_MODEL_DIR", "").strip()
    if env:
        candidates.append(os.path.abspath(os.path.expanduser(env)))
    here = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(here)
    for name in ("Qwen3.5-4B", "Qwen3.5-35B-A3B"):
        candidates.append(os.path.join(workspace, "models", "Qwen", name))
        candidates.append(os.path.join(here, "models", "Qwen", name))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        if os.path.isfile(os.path.join(path, "adapter_config.json")):
            return path
    return None


def _decode_data_url_to_path(url: str, tmp_paths: List[str]) -> Optional[str]:
    if not url.startswith("data:"):
        return None
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return None
    ext = ".jpg"
    if "png" in header:
        ext = ".png"
    elif "webp" in header:
        ext = ".webp"
    raw = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=ext, prefix="local_qwen_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(raw)
    tmp_paths.append(path)
    return path


def _extract_user_content(messages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """从 OpenAI messages 提取最后一条 user 文本与图片路径。"""
    tmp_paths: List[str] = []
    text_parts: List[str] = []
    image_paths: List[str] = []
    for msg in messages:
        if str(msg.get("role") or "") != "user":
            continue
        content = msg.get("content")
        text_parts.clear()
        image_paths.clear()
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "")
            if ptype == "text":
                text_parts.append(str(part.get("text") or ""))
            elif ptype == "image_url":
                image_url = part.get("image_url") or {}
                url = ""
                if isinstance(image_url, dict):
                    url = str(image_url.get("url") or "")
                elif isinstance(image_url, str):
                    url = image_url
                path = _decode_data_url_to_path(url, tmp_paths)
                if path:
                    image_paths.append(path)
    question = "\n".join(t.strip() for t in text_parts if t and t.strip()).strip()
    return question or "请描述图像。", image_paths


def load_model(model_dir: str) -> None:
    if os.path.isfile(os.path.join(model_dir, "adapter_config.json")):
        load_lora_model(model_dir)
        return
    load_full_model(model_dir)


def load_full_model(model_dir: str) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _log(f"loading model from {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    # 新版 transformers 推荐 dtype=；旧版仍认 torch_dtype
    load_kwargs["torch_dtype"] = dtype

    if torch.cuda.is_available():
        # 单卡放不下（如 35B MoE）时 accelerate 会把溢出权重落到 disk，
        # MoE 必须显式提供 offload_folder，否则直接报错退出。
        offload_root = os.environ.get("LOCAL_QWEN_OFFLOAD_DIR", "").strip()
        if not offload_root:
            offload_root = os.path.join(
                tempfile.gettempdir(), "local_qwen_offload", os.path.basename(model_dir)
            )
        os.makedirs(offload_root, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        total_gib = max(1, int(props.total_memory / (1024**3)))
        # 预留约 2GiB 给 KV / 激活，避免占满后 generate OOM
        gpu_budget = max(4, total_gib - 2)
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: f"{gpu_budget}GiB", "cpu": "64GiB"}
        load_kwargs["offload_folder"] = offload_root
        _log(
            f"device_map=auto max_memory[0]={gpu_budget}GiB "
            f"offload_folder={offload_root}"
        )
    else:
        load_kwargs["device_map"] = None

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **load_kwargs)
    except TypeError:
        # 极旧版本可能不接受部分 kwargs
        load_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **load_kwargs)

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"
    with _LOCK:
        _STATE["processor"] = processor
        _STATE["model"] = model
        _STATE["model_dir"] = model_dir
        _STATE["device"] = device
        _STATE["ready"] = True
        _STATE["error"] = ""
    _ensure_cudnn_ready()
    _log(f"model ready on {device}")


def load_lora_model(adapter_dir: str) -> None:
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_dir = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
    if not base_dir or not os.path.isfile(os.path.join(base_dir, "config.json")):
        raise FileNotFoundError(f"LoRA base model not found: {base_dir}")

    _log(f"loading LoRA adapter from {adapter_dir}, base={base_dir}")
    proc_dir = (
        adapter_dir
        if os.path.isfile(os.path.join(adapter_dir, "tokenizer.json"))
        else base_dir
    )
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if torch.cuda.is_available():
        offload_root = os.environ.get("LOCAL_QWEN_OFFLOAD_DIR", "").strip()
        if not offload_root:
            offload_root = os.path.join(
                tempfile.gettempdir(),
                "local_qwen_offload",
                os.path.basename(base_dir),
            )
        os.makedirs(offload_root, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        total_gib = max(1, int(props.total_memory / (1024**3)))
        gpu_budget = max(4, total_gib - 2)
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: f"{gpu_budget}GiB", "cpu": "64GiB"}
        load_kwargs["offload_folder"] = offload_root
        _log(
            f"LoRA device_map=auto max_memory[0]={gpu_budget}GiB "
            f"offload_folder={offload_root}"
        )
    else:
        load_kwargs["device_map"] = None

    try:
        base_model = AutoModelForImageTextToText.from_pretrained(base_dir, **load_kwargs)
    except TypeError:
        load_kwargs.pop("low_cpu_mem_usage", None)
        base_model = AutoModelForImageTextToText.from_pretrained(base_dir, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"
    with _LOCK:
        _STATE["processor"] = processor
        _STATE["model"] = model
        _STATE["model_dir"] = adapter_dir
        _STATE["device"] = device
        _STATE["ready"] = True
        _STATE["error"] = ""
    _ensure_cudnn_ready()
    _log(f"LoRA model ready on {device}")


def run_chat(
    messages: List[Dict[str, Any]],
    image_paths: List[str],
    max_new_tokens: int,
    *,
    enable_thinking: bool = False,
    temperature: float = 0.0,
) -> str:
    with _LOCK:
        if not _STATE["ready"]:
            raise RuntimeError(_STATE.get("error") or "model not ready")
        model = _STATE["model"]
        processor = _STATE["processor"]

    # 构造 chat 模板输入；有图时附加 image
    chat_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content.strip():
                chat_messages.append({"role": "system", "content": content.strip()})
            continue
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            chat_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # 简化：仅保留文本；图片走 processor images
            texts = [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            chat_messages.append(
                {"role": role, "content": "\n".join(t for t in texts if t).strip()}
            )

    if not chat_messages:
        chat_messages = [{"role": "user", "content": "你好"}]

    # 若有图，把图挂到最后一条 user
    images = None
    if image_paths:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        # Qwen3.5 processor 通常期望 messages 中带 image 占位；这里用 images= 参数
        last_user_idx = None
        for i in range(len(chat_messages) - 1, -1, -1):
            if chat_messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            text = chat_messages[last_user_idx].get("content") or ""
            content_parts: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
            content_parts.append({"type": "text", "text": text})
            chat_messages[last_user_idx] = {
                "role": "user",
                "content": content_parts,
            }

    # Qwen3.5 默认会打开 <think>；关闭后模板写入空 think 块，避免长篇思考占满 max_tokens
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": bool(enable_thinking),
    }
    try:
        prompt = processor.apply_chat_template(chat_messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = processor.apply_chat_template(chat_messages, **template_kwargs)
    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    import torch

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max(16, int(max_new_tokens)),
    }
    if temperature and float(temperature) > 1e-6:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    def _generate_once() -> Any:
        with torch.inference_mode():
            return model.generate(**inputs, **gen_kwargs)

    try:
        generated = _generate_once()
    except RuntimeError as exc:
        msg = str(exc)
        if "CUDNN_STATUS_NOT_INITIALIZED" in msg or "cuDNN" in msg:
            _log(f"generate hit cuDNN error, retry with cudnn disabled: {msg}")
            _ensure_cudnn_ready()
            try:
                generated = _generate_once()
            except RuntimeError:
                torch.backends.cudnn.enabled = False
                _log("retry generate with torch.backends.cudnn.enabled=False")
                generated = _generate_once()
        else:
            raise
    # 只解码新生成部分
    in_len = int(inputs["input_ids"].shape[-1])
    out_ids = generated[:, in_len:]
    text = processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    text = str(text).strip()
    # 兜底：若仍带上 think 块，只保留正文
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    elif text.startswith("<think>"):
        # 未闭合则整段当思考丢弃
        text = ""
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQwenWorker/1.0"

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
                    "model_dir": _STATE.get("model_dir") or "",
                    "device": _STATE.get("device") or "",
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
                            "owned_by": "local",
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
        max_tokens = int(payload.get("max_tokens") or 512)
        temperature = float(payload.get("temperature") or 0.0)
        enable_thinking = False
        ctk = payload.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            enable_thinking = bool(ctk.get("enable_thinking"))
        elif "enable_thinking" in payload:
            enable_thinking = bool(payload.get("enable_thinking"))
        tmp_paths: List[str] = []
        try:
            _, image_paths = _extract_user_content(messages)
            tmp_paths.extend(image_paths)
            answer = run_chat(
                messages,
                image_paths,
                max_new_tokens=max_tokens,
                enable_thinking=enable_thinking,
                temperature=temperature,
            )
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-local-qwen-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload.get("model") or MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
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
    global MODEL_ID
    parser = argparse.ArgumentParser(description="Local Qwen3.5 OpenAI worker")
    parser.add_argument("--host", default=os.environ.get("LOCAL_QWEN_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("LOCAL_QWEN_PORT", "8100"))
    )
    parser.add_argument("--model", default=os.environ.get("LOCAL_QWEN_MODEL_DIR", ""))
    parser.add_argument(
        "--model-id",
        default=os.environ.get("LOCAL_QWEN_MODEL_ID", "qwen3.5-4b"),
        help="OpenAI /v1/models 中暴露的 model id",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="先起 HTTP，首请求再加载模型（默认启动时加载）",
    )
    args = parser.parse_args()
    MODEL_ID = str(args.model_id or "").strip() or "qwen3.5-4b"
    os.environ["LOCAL_QWEN_MODEL_ID"] = MODEL_ID
    _configure_torch_backends()

    model_dir = resolve_model_dir(args.model or None)
    if not model_dir:
        _log(
            "ERROR: 未找到本地 Qwen 权重目录（需含 config.json 或 adapter_config.json）。"
            "请设置 LOCAL_QWEN_MODEL_DIR 或 --model"
        )
        return 1

    _STATE["model_dir"] = model_dir
    if not args.lazy:
        try:
            load_model(model_dir)
        except Exception as exc:
            _STATE["error"] = str(exc)
            _log(f"load failed: {exc}")
            traceback.print_exc()
            return 1
    else:
        def _bg_load() -> None:
            try:
                load_model(model_dir)
            except Exception as exc:
                _STATE["error"] = str(exc)
                _log(f"lazy load failed: {exc}")
                traceback.print_exc()

        threading.Thread(target=_bg_load, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"serving OpenAI-compatible API on http://{args.host}:{args.port}/v1")
    _log(f"model_id={MODEL_ID}")
    _log(f"model_dir={model_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
