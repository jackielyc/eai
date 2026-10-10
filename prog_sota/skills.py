"""轻量技能执行器：按动作序列打分，用于本地优化成功率。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prog_sota.catalog import ActionPrim, TaskCard


_SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "enumerate": enumerate,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "sum": sum,
    "any": any,
    "all": all,
    "zip": zip,
    "reversed": reversed,
    "bool": bool,
    "isinstance": isinstance,
    "print": lambda *args, **kwargs: None,
}

_FORBIDDEN = (
    "import ",
    "import\t",
    "__",
    "open(",
    "exec(",
    "eval(",
    "os.",
    "sys.",
    "subprocess",
)


@dataclass
class Rollout:
    task: str
    seed: int
    success: bool
    score: float
    failure: str
    trace: List[str] = field(default_factory=list)
    note: str = ""


class Skills:
    """手 / 臂 / 感知原语；调用会写入 episode 状态。"""

    def __init__(self, ep: "Episode") -> None:
        self.ep = ep

    def look(self, target: str = "scene") -> Dict[str, Any]:
        self.ep.tick("look", target)
        self.ep.seen.add(target)
        return {"seen": sorted(self.ep.seen)}

    def reach(self, target: str) -> bool:
        self.ep.tick("reach", target)
        self.ep.near = target
        return True

    def approach(self, target: str) -> bool:
        self.ep.tick("approach", target)
        self.ep.near = target
        return True

    def align(self, target: str) -> bool:
        self.ep.tick("align", target)
        self.ep.aligned = target
        return True

    def grasp(self, target: str) -> bool:
        self.ep.tick("grasp", target)
        if self.ep.near != target and self.ep.aligned != target:
            self.ep.fail = "grasp_without_reach"
            return False
        self.ep.holding = target
        self.ep.hand_closed = True
        return True

    def pinch(self, target: str) -> bool:
        self.ep.tick("pinch", target)
        if target not in self.ep.seen and self.ep.near != target:
            self.ep.fail = "pinch_unseen"
            return False
        self.ep.holding = target
        self.ep.hand_closed = True
        return True

    def release(self, target: str = "") -> bool:
        self.ep.tick("release", target or self.ep.holding or "")
        held = self.ep.holding
        self.ep.holding = ""
        self.ep.hand_closed = False
        if held and self.ep.goal and self.ep.near == self.ep.goal:
            self.ep.placed.add(held)
        return True

    def lift(self, target: str = "") -> bool:
        t = target or self.ep.holding
        self.ep.tick("lift", t)
        if self.ep.holding != t:
            self.ep.fail = "lift_empty"
            return False
        self.ep.height[t] = self.ep.height.get(t, 0.0) + 1.0
        return True

    def move_to(self, target: str) -> bool:
        self.ep.tick("move_to", target)
        if not self.ep.holding and not self.ep.hand_closed:
            # allow empty move but mark near
            self.ep.near = target
            return True
        self.ep.near = target
        return True

    def place(self, target: str) -> bool:
        self.ep.tick("place", target)
        self.ep.near = target
        if self.ep.holding:
            self.ep.placed.add(self.ep.holding)
        return True

    def transport(self, target: str) -> bool:
        self.ep.tick("transport", target)
        self.ep.near = target
        return True

    def orient(self, pose: str = "upright") -> bool:
        self.ep.tick("orient", pose)
        if not self.ep.holding:
            self.ep.fail = "orient_empty"
            return False
        self.ep.pose[self.ep.holding] = pose
        return True

    def pull(self, target: str) -> bool:
        self.ep.tick("pull", target)
        if self.ep.holding not in {target, "handle"} and self.ep.aligned != "handle":
            if "handle" not in self.ep.trace_names:
                self.ep.fail = "pull_without_grasp"
                return False
        self.ep.open_parts.add(target)
        return True

    def push(self, target: str) -> bool:
        self.ep.tick("push", target)
        self.ep.fail = self.ep.fail or "push_not_goal"
        return False


class Episode:
    def __init__(self, card: TaskCard, seed: int) -> None:
        self.card = card
        self.seed = seed
        self.rng = random.Random(seed)
        self.trace: List[str] = []
        self.trace_names: List[str] = []
        self.seen: set = set()
        self.near = ""
        self.aligned = ""
        self.holding = ""
        self.hand_closed = False
        self.placed: set = set()
        self.open_parts: set = set()
        self.height: Dict[str, float] = {}
        self.pose: Dict[str, str] = {}
        self.fail = ""
        self.goal = _infer_goal(card)
        self.target = _infer_target(card)
        # 轻微噪声：未见布局扰动
        if self.rng.random() < 0.15:
            self.seen.add("distractor")

    def tick(self, name: str, arg: str = "") -> None:
        self.trace.append(f"{name}:{arg}" if arg else name)
        self.trace_names.append(name)

    def observe(self) -> Dict[str, Any]:
        return {
            "backend": self.card.backend,
            "task": self.card.key,
            "text": self.card.text,
            "seen": sorted(self.seen),
            "holding": self.holding,
            "near": self.near,
        }


def _infer_goal(card: TaskCard) -> str:
    gold = card.stages[-1] if card.stages else ()
    for a in reversed(gold):
        if a.name in {"move_to", "place", "transport", "pull"} and a.args:
            return a.args[0]
    text = card.text.lower()
    for token in ("plate", "coaster", "slot", "bin", "goal", "drawer", "table"):
        if token in text or token in card.key:
            return token
    return "goal"


def _infer_target(card: TaskCard) -> str:
    gold = card.stages[-1] if card.stages else ()
    for a in gold:
        if a.name in {"grasp", "pinch"} and a.args:
            return a.args[0]
    for token in ("bowl", "bottle", "cup", "coin", "cube", "object", "clutter", "handle"):
        if token in card.key or token in card.text.lower():
            return token
    return "object"


def harness_rejected(src: str) -> str:
    low = src or ""
    for bad in _FORBIDDEN:
        if bad in low:
            return f"forbidden:{bad.strip()}"
    if "def act" not in low:
        return "missing def act"
    return ""


def run_actions(card: TaskCard, actions: Sequence[ActionPrim], seed: int) -> Rollout:
    ep = Episode(card, seed)
    skills = Skills(ep)
    for act in actions:
        fn = getattr(skills, act.name, None)
        if fn is None:
            ep.fail = f"unknown:{act.name}"
            break
        try:
            if act.args:
                fn(*act.args)
            else:
                fn()
        except TypeError:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                ep.fail = f"call:{exc}"
                break
        if ep.fail:
            break
    return _score(ep)


def run_harness(card: TaskCard, src: str, seed: int) -> Rollout:
    reason = harness_rejected(src)
    if reason:
        return Rollout(
            task=card.key,
            seed=seed,
            success=False,
            score=0.0,
            failure=reason,
            note="rejected",
        )
    ep = Episode(card, seed)
    skills = Skills(ep)
    tools: Dict[str, Any] = {}
    g = {"goal": ep.goal, "target": ep.target, "text": card.text}
    m: Dict[str, Any] = {}
    ns: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(src, ns, ns)  # noqa: S102 — sandboxed builtins
        act = ns.get("act")
        if not callable(act):
            raise RuntimeError("act not callable")
        # 多步直到成功或步数耗尽
        for _ in range(12):
            m = act(ep.observe(), g, m, skills, tools) or m
            if _success(ep):
                break
            if ep.fail:
                break
    except Exception as exc:  # noqa: BLE001
        return Rollout(
            task=card.key,
            seed=seed,
            success=False,
            score=0.0,
            failure=f"exec:{type(exc).__name__}",
            trace=ep.trace,
            note=str(exc)[:120],
        )
    return _score(ep)


def _success(ep: Episode) -> bool:
    card = ep.card
    gold = card.stages[-1] if card.stages else ()
    gold_names = [a.name for a in gold]
    # 覆盖黄金序列关键步骤
    need = set(gold_names)
    have = set(ep.trace_names)
    if not need.issubset(have):
        return False
    if "pull" in need and ep.goal not in ep.open_parts and "drawer" not in ep.open_parts:
        return False
    # place / move_to+release 类任务需要物体落到目标
    place_like = "place" in need or ("move_to" in need and "release" in need)
    if place_like and ep.target and ep.target not in ep.placed:
        return False
    if "lift" in need and ep.height.get(ep.target or ep.holding, 0) < 0.5:
        return False
    if "orient" in need and ep.pose.get(ep.holding or ep.target, "") != "upright":
        # after release, pose may still be recorded
        if "upright" not in ep.pose.values():
            return False
    return not ep.fail


def _score(ep: Episode) -> Rollout:
    gold = ep.card.stages[-1] if ep.card.stages else ()
    gold_names = [a.name for a in gold]
    matched = sum(1 for n in gold_names if n in ep.trace_names)
    total = max(1, len(gold_names))
    partial = 100.0 * matched / total
    ok = _success(ep)
    if ok:
        score = 100.0
        fail = ""
    else:
        score = max(0.0, partial - (20.0 if ep.fail else 0.0))
        fail = ep.fail or "incomplete"
    return Rollout(
        task=ep.card.key,
        seed=ep.seed,
        success=ok,
        score=score,
        failure=fail,
        trace=list(ep.trace),
        note=f"matched={matched}/{total}",
    )
