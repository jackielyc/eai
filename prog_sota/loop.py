"""归纳 → 编程 → 评测 → 改写：抬高成功率并对照榜单。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from prog_sota.catalog import ActionPrim, TaskCard, TASKS, stage_label
from prog_sota.compare import compare_to_board
from prog_sota.induce import induce_from_text, induce_with_model, refine_actions
from prog_sota.program import actions_to_source, summarize_actions
from prog_sota.skills import Rollout, run_actions, run_harness


LogFn = Callable[[str], None]
ProgramFn = Callable[[str], None]
ActionsFn = Callable[[Tuple[ActionPrim, ...]], None]


@dataclass
class LoopConfig:
    task: str
    text: str = ""
    image_paths: List[str] = field(default_factory=list)
    actions: Optional[Tuple[ActionPrim, ...]] = None
    program: str = ""
    budget: int = 5
    eval_seeds: int = 4
    use_model: bool = False
    api_base: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen"
    timeout_s: float = 60.0


@dataclass
class Stats:
    mean_score: float
    sr: float
    n: int
    failures: List[str] = field(default_factory=list)


def evaluate_actions(
    card: TaskCard, actions: Sequence[ActionPrim], seeds: Sequence[int]
) -> Stats:
    scores: List[float] = []
    wins = 0
    failures: List[str] = []
    for seed in seeds:
        rollout = run_actions(card, actions, int(seed))
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


def evaluate_program(card: TaskCard, src: str, seeds: Sequence[int]) -> Stats:
    scores: List[float] = []
    wins = 0
    failures: List[str] = []
    for seed in seeds:
        rollout = run_harness(card, src, int(seed))
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
    on_program: Optional[ProgramFn] = None,
    on_actions: Optional[ActionsFn] = None,
    stop: Optional[threading.Event] = None,
) -> Tuple[str, Stats]:
    card = TASKS[cfg.task]
    emit = log or (lambda _line: None)
    publish = on_program or (lambda _src: None)
    publish_actions = on_actions or (lambda _a: None)
    halted = stop or threading.Event()

    emit(f"任务 {card.title}（{card.backend}）")
    emit(f"文本: {(cfg.text or card.text)[:200]}")
    if cfg.image_paths:
        emit(f"图像输入 ×{len(cfg.image_paths)}: " + ", ".join(cfg.image_paths[:3]))

    if cfg.use_model:
        emit("induce: 调用模型归纳手/臂动作…")
        actions = induce_with_model(
            card,
            text=cfg.text,
            image_paths=cfg.image_paths,
            api_base=cfg.api_base,
            model=cfg.model,
            timeout_s=cfg.timeout_s,
        )
    elif cfg.actions:
        actions = cfg.actions
    else:
        emit("induce: 规则归纳手/臂动作…")
        actions = induce_from_text(card, cfg.text, cfg.image_paths)

    publish_actions(actions)
    emit("induce:\n" + summarize_actions(actions))
    program = cfg.program.strip() or actions_to_source(actions, card)
    publish(program)
    emit(f"program: {stage_label(card, actions)}")

    best_actions = actions
    best_program = program
    best_stats = Stats(mean_score=0.0, sr=0.0, n=0)

    for k in range(max(1, int(cfg.budget))):
        if halted.is_set():
            emit("已停止")
            break
        emit(f"── k={k}  {stage_label(card, best_actions)}  induce → program → eval → revise")
        evidence = run_actions(card, best_actions, 10_000 + k)
        emit("execute: " + _fmt_rollout(evidence))
        seeds = [20_000 + k * 100 + i for i in range(max(1, int(cfg.eval_seeds)))]
        parent_stats = evaluate_actions(card, best_actions, seeds)
        emit("eval " + _fmt_stats("parent", card, best_actions, parent_stats))
        best_stats = parent_stats

        candidates = refine_actions(card, best_actions, evidence_ok=evidence.success)
        picked_name = "parent"
        for i, cand in enumerate(candidates):
            if halted.is_set():
                emit("已停止")
                break
            name = f"v{i}"
            stats = evaluate_actions(card, cand, seeds)
            emit("eval " + _fmt_stats(name, card, cand, stats))
            if stats.mean_score > best_stats.mean_score + 1e-6:
                picked_name = name
                best_actions = cand
                best_stats = stats
                best_program = actions_to_source(cand, card)
                publish_actions(best_actions)
                publish(best_program)

        if picked_name == "parent":
            emit(
                f"select: 保留 {stage_label(card, best_actions)}  "
                f"score {best_stats.mean_score:.1f}  SR {best_stats.sr * 100:.0f}%"
            )
            if best_stats.sr >= 1.0 - 1e-9 or not candidates:
                emit("收敛：没有更优动作程序")
                break
            continue
        emit(
            f"select: {picked_name} {stage_label(card, best_actions)}  "
            f"score {best_stats.mean_score:.1f}  SR {best_stats.sr * 100:.0f}%  → inherit"
        )
        if best_stats.sr >= 1.0 - 1e-9 and stage_label(card, best_actions) == f"H{len(card.stages) - 1}":
            emit("收敛：评测成功率 100%")
            break

    # 用编程 harness 再验一次
    prog_stats = evaluate_program(
        card,
        best_program,
        list(range(30_000, 30_000 + max(1, int(cfg.eval_seeds)))),
    )
    emit(
        f"program-check score {prog_stats.mean_score:.1f}  "
        f"SR {prog_stats.sr * 100:.0f}%  n={prog_stats.n}"
    )
    use_stats = prog_stats if prog_stats.n else best_stats

    cmp = compare_to_board(card.backend, use_stats.sr)
    emit("—— 业内榜单对照 ——")
    for line in cmp.lines:
        emit(line)
    emit(f"结束于 {stage_label(card, best_actions)}")
    return best_program, use_stats


def _fmt_rollout(rollout: Rollout) -> str:
    flag = "E+" if rollout.success else "E-"
    fail = rollout.failure or "ok"
    trace = ",".join(rollout.trace[:10]) or "-"
    return (
        f"{flag} seed={rollout.seed} score={rollout.score:.0f} "
        f"failure={fail} {rollout.note} trace={trace}"
    )


def _fmt_stats(name: str, card: TaskCard, actions: Sequence[ActionPrim], stats: Stats) -> str:
    fail = ",".join(stats.failures) if stats.failures else "ok"
    return (
        f"{name} {stage_label(card, tuple(actions))}  score {stats.mean_score:.1f}  "
        f"SR {stats.sr * 100:.0f}%  n={stats.n}  {fail}"
    )


def self_check() -> None:
    for key, card in TASKS.items():
        final, stats = run_loop(
            LoopConfig(task=key, budget=6, eval_seeds=3, use_model=False),
            log=print,
        )
        if stats.sr < 1.0:
            raise SystemExit(f"{key}: SR={stats.sr} program=\n{final}")
        print(f"[ok] {card.title} SR={stats.sr:.0%}", flush=True)


if __name__ == "__main__":
    self_check()
