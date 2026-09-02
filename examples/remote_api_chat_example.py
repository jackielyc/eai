#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI 兼容远程 API 调用示例（与 show_camera_topics.LlmChatClient 对齐）。

仅依赖 Python 标准库，无需额外 pip 包。

快速开始::

    # 本地 Ollama
    export LLM_API_BASE=http://127.0.0.1:11434/v1
    export LLM_MODEL=qwen2.5
    export LLM_API_KEY=ollama
    python3 remote_api_chat_example.py --probe
    python3 remote_api_chat_example.py "你好，介绍一下自己"

    # 带图多模态
    python3 remote_api_chat_example.py --image /path/to.jpg "图里有什么？"

详见同目录 README.md
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 默认值（与 show_camera_topics.py 保持一致）
# ---------------------------------------------------------------------------

LLM_API_BASE_DEFAULT = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_CHAT_TIMEOUT_S = float(os.environ.get("LLM_CHAT_TIMEOUT_S", "120"))
LLM_CHAT_VISION_TIMEOUT_S = float(os.environ.get("LLM_CHAT_VISION_TIMEOUT_S", "300"))

# 常用预设：(api_base, model, api_key_placeholder)
PROVIDER_PRESETS: Dict[str, Tuple[str, str, str]] = {
    "local-qwen": (
        os.environ.get("LOCAL_QWEN_API_BASE", "http://127.0.0.1:8100/v1"),
        "qwen3.5-4b",
        "EMPTY",
    ),
    "remote-qwen": (
        os.environ.get("REMOTE_QWEN_API_BASE", "http://127.0.0.1:18100/v1"),
        "qwen3.5-35b-a3b",
        "EMPTY",
    ),
    "ollama": (
        "http://127.0.0.1:11434/v1",
        "qwen2.5",
        "ollama",
    ),
    "hy-vlm": (
        os.environ.get("HY_EMBODIED_VLM_API_BASE", "http://127.0.0.1:8080/v1"),
        "hy_a3b",
        "EMPTY",
    ),
    "hy-rxbrain": (
        os.environ.get("HY_RXBRAIN_API_BASE", "http://127.0.0.1:8090/v1"),
        "hy-rxbrain",
        "EMPTY",
    ),
    "dashscope": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        "",
    ),
    "openai": (
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        "",
    ),
}


# ---------------------------------------------------------------------------
# HTTP 工具
# ---------------------------------------------------------------------------

def _http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 5.0,
) -> Tuple[bool, Any, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(raw), ""
            except Exception:
                return True, raw, ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, None, f"HTTP {exc.code}: {detail[:200]}"
    except Exception as exc:
        return False, None, str(exc)


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"网络错误: {exc.reason}") from exc


def encode_image_file_b64(path: str, max_side: int = 1280) -> Tuple[str, str]:
    """本地图片 → (mime, base64)。有 OpenCV 时会缩放，否则原样编码。"""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    try:
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图片: {path}")
        h, w = img.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w, 1)))
        if scale < 0.999:
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return "image/jpeg", base64.b64encode(buf.tobytes()).decode("ascii")
    except ImportError:
        with open(path, "rb") as f:
            return mime, base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

@dataclass
class LlmChatConfig:
    api_base: str = LLM_API_BASE_DEFAULT
    model: str = LLM_MODEL_DEFAULT
    api_key: str = ""
    enable_thinking: bool = False
    max_tokens: int = 1536
    temperature: float = 0.2
    system_prompt: str = "You are a helpful assistant."
    # 稳定调用：失败自动重试
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    @classmethod
    def from_env(cls, preset: Optional[str] = None) -> "LlmChatConfig":
        cfg = cls(
            api_base=os.environ.get("LLM_API_BASE", LLM_API_BASE_DEFAULT),
            model=os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
            api_key=os.environ.get(LLM_API_KEY_ENV, "").strip(),
            enable_thinking=os.environ.get("LLM_ENABLE_THINKING", "").lower()
            in ("1", "true", "yes"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1536")),
            system_prompt=os.environ.get(
                "LLM_SYSTEM_PROMPT", "You are a helpful assistant."
            ),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        )
        if preset:
            if preset not in PROVIDER_PRESETS:
                raise SystemExit(
                    f"未知预设 {preset!r}，可选: {', '.join(PROVIDER_PRESETS)}"
                )
            base, model, key = PROVIDER_PRESETS[preset]
            cfg.api_base = base
            cfg.model = model
            if not cfg.api_key and key:
                cfg.api_key = key
        return cfg


@dataclass
class LlmChatClient:
    """OpenAI 兼容 Chat Completions 客户端。"""

    config: LlmChatConfig = field(default_factory=LlmChatConfig.from_env)

    def _auth_headers(self) -> Dict[str, str]:
        api_key = self.config.api_key.strip() or "EMPTY"
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def probe(self, timeout_s: float = 6.0) -> Dict[str, Any]:
        """探测 API 可达性与可用模型。"""
        base = (self.config.api_base or "").strip().rstrip("/") or LLM_API_BASE_DEFAULT
        root = base.removesuffix("/v1").rstrip("/")
        headers = self._auth_headers()
        details: List[str] = []
        model_ids: List[str] = []
        reachable = False

        ok, body, err = _http_get_json(
            f"{root}/health", headers=headers, timeout_s=timeout_s
        )
        if ok:
            reachable = True
            details.append("健康检查: /health 可达")
            if isinstance(body, dict) and body.get("model"):
                model_ids.append(str(body["model"]))
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
        else:
            details.append(f"模型列表: /models 不可用 ({err})")

        ok, body, err = _http_get_json(f"{root}/api/tags", timeout_s=timeout_s)
        if ok:
            reachable = True
            details.append("Ollama: /api/tags 可达")
            if isinstance(body, dict):
                for item in body.get("models") or []:
                    if isinstance(item, dict):
                        mid = str(item.get("name") or item.get("model") or "").strip()
                        if mid and mid not in model_ids:
                            model_ids.append(mid)
        else:
            details.append(f"Ollama: /api/tags 不可用 ({err})")

        return {
            "ok": reachable,
            "api_base": base,
            "configured_model": self.config.model,
            "models": model_ids,
            "details": details,
            "error": "" if reachable else "无法连接当前 API",
        }

    def _messages_have_image(self, messages: Sequence[Dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and str(part.get("type") or "") in (
                        "image_url",
                        "image",
                    ):
                        return True
        return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        timeout_s: Optional[float] = None,
    ) -> str:
        if not self.config.api_key.strip():
            raise RuntimeError(
                f"未配置 API Key，请设置环境变量 {LLM_API_KEY_ENV}，"
                "或使用 --preset / --api-key（本地服务可填 EMPTY）"
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
            "chat_template_kwargs": {
                "enable_thinking": bool(self.config.enable_thinking)
            },
        }
        if timeout_s is None:
            timeout_s = (
                LLM_CHAT_VISION_TIMEOUT_S
                if self._messages_have_image(messages)
                else LLM_CHAT_TIMEOUT_S
            )

        last_err: Optional[BaseException] = None
        attempts = max(1, int(self.config.max_retries) + 1)
        for attempt in range(attempts):
            try:
                body = _http_post_json(
                    url, payload, self._auth_headers(), timeout_s=timeout_s
                )
                return self._extract_reply(body)
            except RuntimeError as exc:
                last_err = exc
                msg = str(exc)
                # 4xx（除 429）一般不重试
                retryable = (
                    "网络错误" in msg
                    or "HTTP 429" in msg
                    or "HTTP 5" in msg
                    or "timed out" in msg.lower()
                )
                if not retryable or attempt + 1 >= attempts:
                    raise
                delay = self.config.retry_backoff_s * (2**attempt)
                print(
                    f"[retry {attempt + 1}/{attempts - 1}] {msg}；{delay:.1f}s 后重试…",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise RuntimeError(str(last_err or "chat 失败"))

    @staticmethod
    def _extract_reply(body: Dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 无有效回复: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        parts: List[str] = []
        if reasoning:
            parts.append(f"[thinking]\n{str(reasoning).strip()}")
        if content:
            text = str(content).strip()
            parts.append(text if not reasoning else f"[answer]\n{text}")
        if not parts:
            raise RuntimeError(f"API 返回空内容: {body}")
        return "\n\n".join(parts).strip()

    def ask(
        self,
        user_text: str,
        *,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """便捷封装：单轮问答，可选附带一张图片。"""
        sys_prompt = (system_prompt or self.config.system_prompt).strip()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
        ]
        if image_path:
            mime, b64 = encode_image_file_b64(image_path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})
        return self.chat(messages)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenAI 兼容远程 API 调用示例（stdlib urllib）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt", nargs="?", default="", help="用户提问文本")
    p.add_argument(
        "--preset",
        choices=sorted(PROVIDER_PRESETS.keys()),
        help="使用内置服务预设",
    )
    p.add_argument("--api-base", default="", help="覆盖 LLM_API_BASE")
    p.add_argument("--model", default="", help="覆盖 LLM_MODEL")
    p.add_argument("--api-key", default="", help=f"覆盖 {LLM_API_KEY_ENV}")
    p.add_argument("--image", default="", help="可选：附带本地图片（多模态）")
    p.add_argument("--system", default="", help="覆盖 system prompt")
    p.add_argument(
        "--thinking", action="store_true", help="开启 thinking（Qwen/Hy-VLM）"
    )
    p.add_argument("--max-tokens", type=int, default=0, help="覆盖 max_tokens")
    p.add_argument("--timeout", type=float, default=0.0, help="覆盖请求超时（秒）")
    p.add_argument("--retries", type=int, default=-1, help="失败重试次数（默认 2）")
    p.add_argument("--probe", action="store_true", help="仅探测服务可达性")
    p.add_argument(
        "--dump-request", action="store_true", help="打印即将发送的消息结构"
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = LlmChatConfig.from_env(preset=args.preset or None)
    if args.api_base:
        cfg.api_base = args.api_base.strip()
    if args.model:
        cfg.model = args.model.strip()
    if args.api_key:
        cfg.api_key = args.api_key.strip()
    if args.system:
        cfg.system_prompt = args.system
    if args.thinking:
        cfg.enable_thinking = True
    if args.max_tokens > 0:
        cfg.max_tokens = args.max_tokens
    if args.retries >= 0:
        cfg.max_retries = args.retries

    client = LlmChatClient(cfg)

    if args.probe:
        info = client.probe()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if info.get("ok") else 2

    if not args.prompt:
        print(
            "请提供 prompt，或使用 --probe。见 --help / README.md",
            file=sys.stderr,
        )
        return 2

    if args.dump_request:
        preview = {
            "api_base": cfg.api_base,
            "model": cfg.model,
            "enable_thinking": cfg.enable_thinking,
            "max_tokens": cfg.max_tokens,
            "has_image": bool(args.image),
            "prompt": args.prompt,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2), file=sys.stderr)

    timeout = args.timeout if args.timeout > 0 else None
    if args.image:
        mime, b64 = encode_image_file_b64(args.image)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": cfg.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ]
        reply = client.chat(messages, timeout_s=timeout)
    else:
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        reply = client.chat(messages, timeout_s=timeout)

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
