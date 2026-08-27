#!/usr/bin/env python3
"""Convert Steinate/Cortex VN norm_mem JSONL to lake-style Chinese Sys2 SFT JSONL.

Format matches hermas_sys2_*.jsonl:
  - user: 任务 + 上一个子任务，不含全部子任务提示
  - assistant: 所有子任务 + 技能 + 当前子任务 + 上一个子任务 + 下一个子任务（预测目标）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)

DEFAULT_MEMORY = "这是第一个子任务，尚未完成任何子任务。"
LAST_MEMORY = "这是最后一个子任务，没有下一个子任务。"

SKILL_ZH = {
    "Pick": "抓取",
    "Place": "放置",
    "Push": "推动",
    "Move": "推动",
    "Navigate": "推动",
    "Press": "操作",
    "Open": "操作",
    "Close": "操作",
    "Rotate": "操作",
    "Release": "操作",
    "Wipe": "操作",
    "Pour": "操作",
    "Handover": "操作",
    "AdjustPosture": "操作",
    "Manipulate": "操作",
    "Unknown": "操作",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    return str(value).strip()


def _task_instruction(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks") or []
    if tasks:
        text = _as_text(tasks[0])
        if text:
            return text
    detailed = _as_text(episode.get("detailed_task_instruction"))
    if detailed:
        # Prefer first sentence of detailed instruction when task name missing.
        return detailed.split(".")[0].strip() or detailed
    return "执行操作任务。"


def _skill_zh(skill: str) -> str:
    skill = (skill or "").strip()
    return SKILL_ZH.get(skill, SKILL_ZH["Manipulate"])


def _format_all_subtasks(subtasks: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, 1))


def _progress_memory(completed: list[str]) -> str:
    if not completed:
        return DEFAULT_MEMORY
    return completed[-1]


def iter_episode_samples(episode: dict[str, Any]) -> Iterable[dict[str, Any]]:
    actions = episode.get("action_config") or []
    if not actions:
        return

    all_subtasks = [_as_text(a.get("action_text")) for a in actions]
    all_subtasks = [s for s in all_subtasks if s]
    if not all_subtasks:
        return

    task = _task_instruction(episode)
    for idx, action in enumerate(actions):
        subtask = _as_text(action.get("action_text"))
        if not subtask:
            continue
        skill = _skill_zh(_as_text(action.get("skill")))
        user_mem = _progress_memory(all_subtasks[:idx])
        next_mem = all_subtasks[idx + 1] if idx + 1 < len(all_subtasks) else LAST_MEMORY
        user_text = (
            f"任务：{task}\n\n"
            f"上一个子任务：\n{user_mem}\n\n"
            "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
        )
        assistant = (
            f"所有子任务：\n{_format_all_subtasks(all_subtasks)}\n"
            f"技能：{skill}\n"
            f"当前子任务：{subtask}\n"
            f"上一个子任务：{user_mem}\n"
            f"下一个子任务：{next_mem}"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                },
                {"role": "assistant", "content": assistant},
            ]
        }


def convert_file(src: Path, dst: Path, *, max_samples: int | None = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            episode = json.loads(line)
            for sample in iter_episode_samples(episode):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
                if max_samples is not None and n >= max_samples:
                    return n
            if n and n % 200000 == 0:
                print(f"[progress] {src.name}: {n} samples -> {dst}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cortex-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/dataset/Steinate/Cortex"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/data"),
    )
    parser.add_argument("--train-name", default="vn_sys2_train.jsonl")
    parser.add_argument("--val-name", default="vn_sys2_val.jsonl")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    train_src = args.cortex_dir / "vn_norm_mem_train.jsonl"
    val_src = args.cortex_dir / "vn_norm_mem_val.jsonl"
    train_dst = args.output_dir / args.train_name
    val_dst = args.output_dir / args.val_name

    print(f"[info] train {train_src} -> {train_dst}")
    n_train = convert_file(train_src, train_dst, max_samples=args.max_train)
    print(f"[done] train={n_train} -> {train_dst}")

    print(f"[info] val {val_src} -> {val_dst}")
    n_val = convert_file(val_src, val_dst, max_samples=args.max_val)
    print(f"[done] val={n_val} -> {val_dst}")


if __name__ == "__main__":
    main()
