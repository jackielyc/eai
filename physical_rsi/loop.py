"""Embodied self-harness：理解 → 执行 → 改写多个候选 → 评测选择 → 继承。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from physical_rsi.tasks import TASKS, TaskSpec, builtin_variants, stage_label
from physical_rsi.world import Rollout, harness_rejected, run_rollout


LogFn = Callable[[str], None]
HarnessFn = Callable[[str], None]


@dataclass
class LoopConfig:
    task: str
    harness: str
    budget: int = 5
    eval_seeds: int = 4
    use_model: bool = False
    api_base: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen"
    n_variants: int = 2
    timeout_s: float = 60.0


@dataclass
class Stats:
    mean_score: float
    sr: float
    n: int
    failures: List[str] = field(default_factory=list)


def evaluate(harness: str, task: str, seeds: Sequence[int]) -> Stats:
    scores: List[float] = []
    wins = 0
    failures: List[str] = []
    for seed in seeds:
        rollout = run_rollout(harness, task, int(seed))
        scores.append(rollout.score)
        wins += int(rollout.success)
        if rollout.failure and rollout.failure not in failures:
            failures.append(rollout.failure)
    n = len(scores) or 1
    return Stats(
        mean_score=sum(scores) / n,
        sr=wins / n,
        n=len(scores),
        failures=failures,
    )


def run_loop(
    cfg: LoopConfig,
    log: Optional[LogFn] = None,
    on_harness: Optional[HarnessFn] = None,
    stop: Optional[threading.Event] = None,
) -> str:
    spec = TASKS[cfg.task]
    emit = log or (lambda _line: None)
    publish = on_harness or (lambda _src: None)
    halted = stop or threading.Event()
    harness = cfg.harness or spec.initial
    emit(f"任务 {spec.title}：{spec.summary}")
    emit("H[k+1] = F(H[k], E+, E-) ；执行 (a, m) = H(o, g, m; S, T)")
    for k in range(max(1, int(cfg.budget))):
        if halted.is_set():
            emit("已停止")
            break
        label = stage_label(spec, harness)
        emit(f"── k={k}  {label}  review → decide → execute → revise")
        emit(f"decide: {spec.plan}")
        evidence_seed = 10_000 + k
        evidence = run_rollout(harness, cfg.task, evidence_seed)
        emit("execute: " + _fmt_rollout(evidence))
        if halted.is_set():
            emit("已停止")
            break
        seeds = [20_000 + k * 100 + i for i in range(max(1, int(cfg.eval_seeds)))]
        candidates: List[tuple] = [("parent", harness)]
        proposed = _propose(cfg, spec, harness, evidence, emit)
        for i, src in enumerate(proposed):
            candidates.append((f"v{i}", src))
        best_name = "parent"
        best_src = harness
        best_stats = evaluate(harness, cfg.task, seeds)
        emit("eval " + _fmt_stats("parent", spec, harness, best_stats))
        for name, src in candidates[1:]:
            if halted.is_set():
                emit("已停止")
                return best_src
            stats = evaluate(src, cfg.task, seeds)
            emit("eval " + _fmt_stats(name, spec, src, stats))
            if stats.mean_score > best_stats.mean_score + 1e-6:
                best_name = name
                best_src = src
                best_stats = stats
        if best_src.strip() == harness.strip():
            emit(
                f"select: 保留 {stage_label(spec, harness)}  "
                f"score {best_stats.mean_score:.1f}  SR {best_stats.sr * 100:.0f}%"
            )
            if best_stats.sr >= 1.0 - 1e-9 or not proposed:
                emit("收敛：没有更优 harness，停止改写")
                break
            emit("本轮对照没有超过当前 harness，进入下一轮")
            continue
        emit(
            f"select: {best_name} {stage_label(spec, best_src)}  "
            f"score {best_stats.mean_score:.1f}  SR {best_stats.sr * 100:.0f}%  → inherit"
        )
        harness = best_src
        publish(harness)
        if best_stats.sr >= 1.0 - 1e-9 and stage_label(spec, harness).startswith("H"):
            if stage_label(spec, harness) == f"H{len(spec.stages) - 1}":
                emit("收敛：评测成功率 100%")
                break
    emit(f"结束于 {stage_label(spec, harness)}")
    return harness


def _propose(
    cfg: LoopConfig,
    spec: TaskSpec,
    harness: str,
    evidence: Rollout,
    emit: LogFn,
) -> List[str]:
    found: List[str] = []
    if cfg.use_model:
        try:
            found.extend(
                _rewrite_with_model(cfg, spec, harness, evidence)
            )
            emit(f"revise: 模型返回 {len(found)} 个候选")
        except Exception as exc:
            emit(f"revise: 模型改写失败（{exc}），改用内置 System 2")
    for src in builtin_variants(spec, harness):
        found.append(src)
    parent = harness.strip()
    unique: List[str] = []
    seen = {parent}
    for src in found:
        reason = harness_rejected(src)
        key = src.strip()
        if not key or key in seen:
            continue
        if reason:
            emit(f"revise: 丢弃候选（{reason}）")
            continue
        seen.add(key)
        unique.append(src)
    if not unique:
        emit("revise: 没有新的候选")
    else:
        names = ", ".join(stage_label(spec, src) for src in unique)
        emit(f"revise: 候选 {names}")
    return unique


def _rewrite_with_model(
    cfg: LoopConfig,
    spec: TaskSpec,
    harness: str,
    evidence: Rollout,
) -> List[str]:
    base = (cfg.api_base or "").rstrip("/")
    if not base:
        raise RuntimeError("API base 为空")
    system = (
        "你是 PhysicalRSI 的 System 2。根据当前 harness 和一次 rollout 的成败，"
        "改写 System 1 控制代码。只输出一个 python 代码块，定义 "
        "def act(o, g, m, skills, tools)。不要 import。用 skills 上已有的方法完成任务。"
        "成功则保持结构，失败则针对 failure 做最小改动。"
    )
    user = (
        f"任务: {spec.title}\n{spec.summary}\n计划: {spec.plan}\n"
        f"可用技能: perceive, match_four, pick_and_place, observe, memorize, cover, "
        f"reveal_named, reveal_guess, carry, nudge_together, swap_direct, touch, "
        f"place_at, approach, align, insert, move_near\n"
        f"rollout success={evidence.success} score={evidence.score} "
        f"failure={evidence.failure or 'ok'} note={evidence.note}\n"
        f"trace={', '.join(evidence.trace)}\n"
        f"当前 harness:\n{harness}"
    )
    out: List[str] = []
    n = max(1, int(cfg.n_variants))
    for i in range(n):
        payload = {
            "model": cfg.model or "qwen",
            "temperature": 0.4 + 0.2 * i,
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
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc)) from exc
        text = str(body["choices"][0]["message"]["content"])
        code = _extract_code(text)
        if code:
            out.append(code if code.endswith("\n") else code + "\n")
    return out


def _extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip() if "def act" in text else ""
    chunks = text.split("```")
    for chunk in chunks[1::2]:
        lines = chunk.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            lines = lines[1:]
        code = "\n".join(lines).strip()
        if "def act" in code:
            return code
    return ""


def _fmt_rollout(rollout: Rollout) -> str:
    flag = "E+" if rollout.success else "E-"
    fail = rollout.failure or "ok"
    trace = ",".join(rollout.trace[:10]) or "-"
    return (
        f"{flag} seed={rollout.seed} score={rollout.score:.0f} "
        f"failure={fail} {rollout.note} trace={trace}"
    )


def _fmt_stats(name: str, spec: TaskSpec, src: str, stats: Stats) -> str:
    fail = ",".join(stats.failures) if stats.failures else "ok"
    return (
        f"{name} {stage_label(spec, src)}  score {stats.mean_score:.1f}  "
        f"SR {stats.sr * 100:.0f}%  n={stats.n}  {fail}"
    )


def self_check() -> None:
    logs: List[str] = []

    def _log(line: str) -> None:
        logs.append(line)
        print(line, flush=True)

    for key, spec in TASKS.items():
        final = run_loop(
            LoopConfig(task=key, harness=spec.initial, budget=6, eval_seeds=3, use_model=False),
            log=_log,
        )
        stats = evaluate(final, key, list(range(50, 55)))
        if final.strip() != spec.stages[-1].strip():
            raise SystemExit(f"{key}: 未继承到最终 harness\n{final}")
        if stats.sr < 1.0 or stats.mean_score < 100 - 1e-6:
            raise SystemExit(f"{key}: 最终评测失败 score={stats.mean_score} sr={stats.sr}")
        print(f"[ok] {spec.title} score={stats.mean_score:.0f} SR={stats.sr:.0%}", flush=True)


if __name__ == "__main__":
    self_check()
