"""把归纳出的手/臂动作编译为可执行 harness。"""

from __future__ import annotations

from typing import List, Optional, Sequence

from prog_sota.catalog import ActionPrim, TaskCard


def actions_to_source(actions: Sequence[ActionPrim], card: Optional[TaskCard] = None) -> str:
    lines = [
        "def act(o, g, m, skills, tools):",
        "    # programmatic policy from induced arm/hand actions",
    ]
    if card is not None:
        lines.append(f"    # task: {card.backend} / {card.key}")
    if not actions:
        lines.append("    skills.look('scene')")
        lines.append("    return m")
        return "\n".join(lines) + "\n"
    for act in actions:
        lines.append(f"    {act.as_call()}")
    lines.append("    return m")
    return "\n".join(lines) + "\n"


def source_to_action_labels(src: str) -> List[str]:
    out: List[str] = []
    for raw in (src or "").splitlines():
        line = raw.strip()
        if not line.startswith("skills."):
            continue
        out.append(line)
    return out


def summarize_actions(actions: Sequence[ActionPrim]) -> str:
    if not actions:
        return "(empty)"
    return "\n".join(f"{i+1}. {a.label()}" for i, a in enumerate(actions))
