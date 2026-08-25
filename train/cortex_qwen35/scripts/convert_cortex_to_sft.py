#!/usr/bin/env python3
"""Convert Steinate/Cortex JSONL annotations to ShareGPT SFT samples for Cortex System-2."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are Cortex System-2, a robot cognitive orchestrator for long-horizon manipulation. "
    "Given the task instruction and language memory of completed progress, predict the current "
    "executable subtask and the updated language memory after this subtask is planned. "
    "Use only canonical skill primitives and keep the subtask physically executable and unambiguous."
)

DEFAULT_TRAIN_FILES = [
    "agibot26_norm_mem_train.jsonl",
    "agibot_norm_mem_train.jsonl",
    "behavior_norm_mem_train.jsonl",
    "galaxea_norm_mem_train.jsonl",
    "robocerebra_norm_mem.jsonl",
    "robotwin_norm_mem_train.jsonl",
]

DEFAULT_VAL_FILES = [
    "agibot_norm_mem_val.jsonl",
    "behavior_norm_mem_val.jsonl",
    "galaxea_norm_mem_val.jsonl",
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    return str(value).strip()


def episode_instruction(episode: dict[str, Any]) -> str:
    detailed = _as_text(episode.get("detailed_task_instruction"))
    if detailed:
        return detailed
    tasks = episode.get("tasks") or []
    if tasks:
        return _as_text(tasks[0])
    return "Complete the robot manipulation task."


def build_sample(
    instruction: str,
    language_memory: str,
    skill: str,
    subtask: str,
    updated_memory: str,
) -> dict[str, Any]:
    user = (
        f"Task instruction:\n{instruction}\n\n"
        f"Language memory:\n{language_memory}\n\n"
        "Output the current skill, subtask, and updated language memory."
    )
    assistant = f"Skill: {skill}\nSubtask: {subtask}\nMemory: {updated_memory}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def iter_episode_samples(episode: dict[str, Any]) -> Iterable[dict[str, Any]]:
    actions = episode.get("action_config") or []
    if not actions:
        return
    instruction = episode_instruction(episode)
    for idx, action in enumerate(actions):
        skill = _as_text(action.get("skill")) or "Unknown"
        subtask = _as_text(action.get("action_text"))
        memory = _as_text(action.get("language_memory")) or (
            "This is the first subtask, and no subtasks have been completed yet."
        )
        if not subtask:
            continue
        if idx + 1 < len(actions):
            updated_memory = _as_text(actions[idx + 1].get("language_memory"))
        else:
            updated_memory = (
                f"The robot has completed the task. Last subtask: {subtask}."
            )
        if not updated_memory:
            updated_memory = f"The robot has finished: {subtask}."
        yield build_sample(instruction, memory, skill, subtask, updated_memory)


def convert_files(jsonl_files: list[Path], max_samples: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in jsonl_files:
        if not path.exists() or path.stat().st_size == 0:
            print(f"[skip] missing/empty: {path}")
            continue
        n_before = len(samples)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episode = json.loads(line)
                samples.extend(iter_episode_samples(episode))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples[:max_samples]
        print(f"[ok] {path.name}: +{len(samples) - n_before} samples (total={len(samples)})")
    return samples


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[write] {path} ({len(data)} samples)")


def write_dataset_info(path: Path) -> None:
    info = {
        "cortex_sys2_train": {
            "file_name": "cortex_sys2_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "cortex_sys2_val": {
            "file_name": "cortex_sys2_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "cortex_sys2_train_20k": {
            "file_name": "cortex_sys2_train_20k.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "cortex_sys2_val_2k": {
            "file_name": "cortex_sys2_val_2k.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[write] {path}")


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
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/eai/train/cortex_qwen35/data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-train", type=int, default=20000)
    parser.add_argument("--subset-val", type=int, default=2000)
    parser.add_argument("--skip-full", action="store_true", help="Only write subset files")
    args = parser.parse_args()

    random.seed(args.seed)
    train_files = [args.cortex_dir / name for name in DEFAULT_TRAIN_FILES]
    val_files = [args.cortex_dir / name for name in DEFAULT_VAL_FILES]

    print("Converting train split...")
    train_samples = convert_files(train_files)
    print("Converting val split...")
    val_samples = convert_files(val_files)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    write_dataset_info(args.output_dir / "dataset_info.json")
    if not args.skip_full:
        write_json(args.output_dir / "cortex_sys2_train.json", train_samples)
        write_json(args.output_dir / "cortex_sys2_val.json", val_samples)

    write_json(
        args.output_dir / "cortex_sys2_train_20k.json",
        train_samples[: args.subset_train],
    )
    write_json(
        args.output_dir / "cortex_sys2_val_2k.json",
        val_samples[: args.subset_val],
    )


if __name__ == "__main__":
    main()
