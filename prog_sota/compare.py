"""与业内评测榜单对照，估计是否逼近 SOTA。"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

EAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EAI_DIR not in sys.path:
    sys.path.insert(0, EAI_DIR)

try:
    from sim_eval_leaderboard import board_for_backend, entries_for_backend
except ImportError:  # pragma: no cover
    def board_for_backend(_backend: str, **_k):  # type: ignore[misc]
        return None

    def entries_for_backend(_backend: str, **_k):  # type: ignore[misc]
        return []


@dataclass
class CompareResult:
    backend: str
    our_sr: float
    board_title: str
    sota_method: str
    sota_score_raw: str
    sota_sr: Optional[float]
    delta_pp: Optional[float]
    rank_estimate: str
    lines: List[str]


_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_sr_percent(score: str) -> Optional[float]:
    """从榜单 score 字段尽量抽出成功率 %。"""
    text = (score or "").strip()
    if not text:
        return None
    # 优先最后一个百分号（常见 Avg 在末尾）
    pcts = _PCT_RE.findall(text)
    if pcts:
        return float(pcts[-1])
    # 「96.85」无百分号但像平均 SR
    if "·" in text:
        tail = text.split("·")[-1].strip()
        m = _NUM_RE.search(tail)
        if m:
            val = float(m.group(1))
            return val if val <= 100 else None
    m = _NUM_RE.search(text)
    if not m:
        return None
    val = float(m.group(1))
    return val if val <= 100 else None


def compare_to_board(backend: str, our_sr: float) -> CompareResult:
    """our_sr: 0~1。"""
    key = (backend or "").strip().lower()
    board = board_for_backend(key) or {}
    entries = entries_for_backend(key)
    title = str(board.get("title") or key)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in entries:
        sr = parse_sr_percent(str(row.get("score") or ""))
        if sr is None:
            continue
        # 人类遥操作参考不参与 SOTA
        method = str(row.get("method") or "")
        if "teleop" in method.lower() or "人类" in method:
            continue
        scored.append((sr, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    our_pct = max(0.0, min(1.0, float(our_sr))) * 100.0
    lines: List[str] = [
        f"后端 {key} · 本地编程策略 SR={our_pct:.1f}%",
        f"榜单: {title}",
    ]
    if board.get("protocol"):
        lines.append(f"协议: {board.get('protocol')}")
    if board.get("source_url"):
        lines.append(f"来源: {board.get('source_url')}")

    if not scored:
        lines.append("榜单无可解析 SR，仅展示原始条目。")
        for row in entries[:8]:
            lines.append(
                f"  #{row.get('rank', '?')} {row.get('method')}  "
                f"{row.get('score')}  ({row.get('org', '')})"
            )
        return CompareResult(
            backend=key,
            our_sr=our_sr,
            board_title=title,
            sota_method="",
            sota_score_raw="",
            sota_sr=None,
            delta_pp=None,
            rank_estimate="n/a",
            lines=lines,
        )

    sota_sr, sota_row = scored[0]
    sota_method = str(sota_row.get("method") or "")
    sota_score_raw = str(sota_row.get("score") or "")
    delta = our_pct - sota_sr
    # 估计排名：比我们高的条目数 + 1
    better = sum(1 for sr, _ in scored if sr > our_pct + 1e-6)
    rank_est = f"~{better + 1} / {len(scored)}（按可解析 SR）"
    lines.append(f"公开 SOTA: {sota_method}  {sota_score_raw}  (≈{sota_sr:.1f}%)")
    if delta >= 0:
        lines.append(f"相对 SOTA: +{delta:.1f} pp  ← 已达到或超过可解析榜首")
    else:
        lines.append(f"相对 SOTA: {delta:.1f} pp  ← 仍差 {-delta:.1f} 个百分点")
    lines.append(f"估计名次: {rank_est}")
    lines.append("—— 榜单前排 ——")
    for sr, row in scored[:8]:
        mark = "◀ ours" if abs(sr - our_pct) < 0.05 else ""
        lines.append(
            f"  #{row.get('rank', '?')} {row.get('method')}  "
            f"{row.get('score')}  ≈{sr:.1f}%  {mark}"
        )
    lines.append(
        "注: 本地 harness 成功率用于快速迭代；冲 SOTA 需在「仿真评测」页跑官方协议。"
    )
    return CompareResult(
        backend=key,
        our_sr=our_sr,
        board_title=title,
        sota_method=sota_method,
        sota_score_raw=sota_score_raw,
        sota_sr=sota_sr,
        delta_pp=delta,
        rank_estimate=rank_est,
        lines=lines,
    )
