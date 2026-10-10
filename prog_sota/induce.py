"""从任务文本（与可选图像路径）归纳手 / 臂动作。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

from prog_sota.catalog import ActionPrim, TaskCard, next_stage, stage_index
from prog_sota.program import summarize_actions


def induce_from_text(
    card: TaskCard,
    text: str = "",
    image_paths: Optional[Sequence[str]] = None,
) -> Tuple[ActionPrim, ...]:
    """规则归纳：结合任务卡阶段与文本关键词，产出当前可用动作序列。"""
    blob = f"{card.text}\n{text or ''}".lower()
    images = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    # 有图像时偏向更完整阶段（视觉证据更充分）
    prefer = len(card.stages) - 1 if images and card.stages else 0
    if any(k in blob for k in ("drawer", "open", "pull", "handle")):
        prefer = max(prefer, min(2, len(card.stages) - 1))
    if any(k in blob for k in ("place", "put", "deposit", "tidy", "lift", "adjust")):
        prefer = max(prefer, min(1, len(card.stages) - 1))
    if "dual" in blob or "aloha" in blob or "two arm" in blob:
        prefer = max(prefer, min(2, len(card.stages) - 1)) if card.stages else 0
    if not card.stages:
        return (_keyword_fallback(blob),)
    prefer = max(0, min(prefer, len(card.stages) - 1))
    return card.stages[prefer]


def _keyword_fallback(blob: str) -> ActionPrim:
    if "drawer" in blob:
        return ActionPrim("pull", "arm", ("drawer",), "fallback")
    if "lift" in blob:
        return ActionPrim("lift", "arm", ("object",), "fallback")
    return ActionPrim("look", "sense", ("scene",), "fallback")


def induce_with_model(
    card: TaskCard,
    *,
    text: str = "",
    image_paths: Optional[Sequence[str]] = None,
    api_base: str = "",
    model: str = "qwen",
    timeout_s: float = 60.0,
) -> Tuple[ActionPrim, ...]:
    """调用 OpenAI 兼容接口，让模型列出手/臂动作 JSON；失败则回退规则。"""
    base = (api_base or "").rstrip("/")
    if not base:
        return induce_from_text(card, text, image_paths)
    images = [p for p in (image_paths or []) if p]
    system = (
        "你是机器人操作规划器。根据仿真任务的文本与图像路径，归纳手臂(arm)与手/夹爪(hand)动作。"
        "只输出 JSON 数组，每项: {\"name\":技能名,\"kind\":\"arm|hand|sense|both\",\"args\":[...],\"note\":\"\"}。"
        "可用 name: look, reach, approach, align, grasp, pinch, release, lift, move_to, "
        "place, transport, orient, pull, push。不要写代码。"
    )
    user = (
        f"backend={card.backend}\ntask={card.key}\ntitle={card.title}\n"
        f"text={card.text}\nextra={text}\nimages={images}\n"
        f"hint_stages=\n{summarize_actions(card.stages[-1] if card.stages else ())}"
    )
    payload = {
        "model": model or "qwen",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = str(body["choices"][0]["message"]["content"])
        parsed = _parse_actions_json(content)
        if parsed:
            return tuple(parsed)
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
        pass
    return induce_from_text(card, text, image_paths)


def _parse_actions_json(text: str) -> List[ActionPrim]:
    raw = (text or "").strip()
    if "```" in raw:
        parts = raw.split("```")
        for chunk in parts[1::2]:
            lines = chunk.splitlines()
            if lines and lines[0].strip().lower() in {"json", "javascript"}:
                lines = lines[1:]
            raw = "\n".join(lines).strip()
            break
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        return []
    data = json.loads(raw[start : end + 1])
    out: List[ActionPrim] = []
    if not isinstance(data, list):
        return []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = str(item.get("kind") or "arm").strip() or "arm"
        args_raw = item.get("args") or []
        args = tuple(str(a) for a in args_raw) if isinstance(args_raw, list) else ()
        note = str(item.get("note") or "")
        out.append(ActionPrim(name=name, kind=kind, args=args, note=note))
    return out


def refine_actions(
    card: TaskCard,
    current: Tuple[ActionPrim, ...],
    *,
    evidence_ok: bool,
) -> List[Tuple[ActionPrim, ...]]:
    """提出下一阶段与 decoy，用于成功率优化。"""
    found: List[Tuple[ActionPrim, ...]] = []
    nxt = next_stage(card, current)
    if nxt is not None and nxt != current:
        found.append(nxt)
    if not evidence_ok and card.stages:
        # 失败时也尝试最终阶段
        final = card.stages[-1]
        if final != current and final not in found:
            found.append(final)
    if card.decoy and card.decoy != current:
        found.append(card.decoy)
    # 若完全对不上阶段，从 H0 升一档
    if stage_index(card, current) is None and card.stages:
        found.insert(0, card.stages[min(1, len(card.stages) - 1)])
    return found
