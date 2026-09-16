#!/usr/bin/env python3
"""Rebuild VN Sys2 JSONL in English from Cortex EN norm_mem (+ existing image paths).

Chinese vn_sys2_{train,val}.jsonl were produced by translating Cortex action/task EN via
CDB maps. English is recovered from the original Cortex JSONL (same episode/action order
and counts), not by model translation.

Reports missing images / residual Chinese in source EN fields.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_CORTEX = Path(
    "/share_data/projects/mahjong/share/personal/liyichao/dataset/Steinate/Cortex"
)

# Known row counts of Chinese vn_sys2_*.jsonl (verified against Cortex action totals).
EXPECTED_ROWS = {"train": 1739303, "val": 90816}

SYSTEM_PROMPT_EN = (
    "You are a robot manipulation task orchestrator. "
    "Given a scene image, a high-level task name, and the previous subtask, "
    "predict the full list of subtasks under this task, and the currently executable subtask. "
    "First output a numbered list under All subtasks, then answer in four lines labeled "
    "Skill, Current subtask, Previous subtask, and Next subtask."
)
DEFAULT_MEMORY_EN = "This is the first subtask; no subtasks have been completed yet."
LAST_MEMORY_EN = "This is the last subtask; there is no next subtask."


def episode_id(episode: dict[str, Any]) -> str:
    return str(episode.get("source_episode_id") or episode.get("episode_index") or "unk")


def format_all_subtasks(subtasks: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, 1))


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(as_text(v) for v in value if as_text(v)).strip()
    return str(value).strip()


def resolve_task_en(
    episode: dict[str, Any],
    *,
    task_zh2en: dict[str, str],
) -> tuple[str, list[str]]:
    misses: list[str] = []
    tasks = episode.get("tasks") or []
    raw = ((tasks[0] if tasks else "") or "").strip()
    eid = episode_id(episode)
    if raw and not has_chinese(raw):
        return raw, misses

    detailed = as_text(episode.get("detailed_task_instruction"))
    if detailed and not has_chinese(detailed):
        return detailed, misses

    if raw and raw in task_zh2en:
        return task_zh2en[raw], misses

    if raw:
        misses.append(f"task_en_missing eid={eid} zh={raw}")
        return raw, misses
    misses.append(f"task_en_missing eid={eid} zh=")
    return "Perform the manipulation task.", misses


def resolve_action_en(
    text: str,
    *,
    act_zh2en: dict[str, str],
    eid: str,
    idx: int,
) -> tuple[str, list[str]]:
    text = (text or "").strip()
    misses: list[str] = []
    if not text:
        misses.append(f"action_empty eid={eid} idx={idx}")
        return text, misses
    if not has_chinese(text):
        return text, misses
    en = act_zh2en.get(text)
    if en and not has_chinese(en):
        return en, misses
    misses.append(f"action_en_missing eid={eid} idx={idx} zh={text}")
    return text, misses


def index_images(image_dir: Path) -> set[str]:
    print(f"[info] indexing images under {image_dir} ...", flush=True)
    names: set[str] = set()
    with os.scandir(image_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".jpg"):
                names.add(entry.name)
    print(f"[info] image_files={len(names)}", flush=True)
    return names


def iter_en_samples(
    episode: dict[str, Any],
    *,
    image_dir: Path,
    image_names: set[str] | None,
    task_zh2en: dict[str, str],
    act_zh2en: dict[str, str],
) -> list[tuple[str, list[str]]]:
    actions = episode.get("action_config") or []
    eid = episode_id(episode)
    task_en, task_misses = resolve_task_en(episode, task_zh2en=task_zh2en)

    kept: list[tuple[str, str, list[str]]] = []
    for idx, a in enumerate(actions):
        text = (a.get("action_text") or "").strip()
        if not text:
            continue
        skill = (a.get("skill") or "Manipulate").strip() or "Manipulate"
        en, misses = resolve_action_en(text, act_zh2en=act_zh2en, eid=eid, idx=idx)
        kept.append((en, skill, misses))
    if not kept:
        return []

    all_subtasks = [t for t, _, _ in kept]
    all_block = "All subtasks:\n" + format_all_subtasks(all_subtasks)
    image_dir_s = str(image_dir)
    out: list[tuple[str, list[str]]] = []
    for idx, (subtask, skill, act_misses) in enumerate(kept):
        misses = list(task_misses) + list(act_misses)
        if has_chinese(task_en):
            misses.append(f"task_still_zh eid={eid} text={task_en[:80]}")
        if has_chinese(subtask):
            misses.append(f"action_still_zh eid={eid} idx={idx} text={subtask}")

        prev = DEFAULT_MEMORY_EN if idx == 0 else all_subtasks[idx - 1]
        nxt = all_subtasks[idx + 1] if idx + 1 < len(all_subtasks) else LAST_MEMORY_EN
        base = f"{eid}_{idx}.jpg"
        image_path = f"{image_dir_s}/{base}"
        if image_names is not None and base not in image_names:
            misses.append(f"image_missing path={image_path}")

        user_text = (
            f"Task: {task_en}\n\n"
            f"Previous subtask:\n{prev}\n\n"
            "Please output all subtasks, and the current skill, current subtask, "
            "previous subtask, and next subtask."
        )
        assistant = (
            f"{all_block}\n"
            f"Skill: {skill}\n"
            f"Current subtask: {subtask}\n"
            f"Previous subtask: {prev}\n"
            f"Next subtask: {nxt}"
        )
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_EN},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": assistant},
            ]
        }
        out.append((json.dumps(sample, ensure_ascii=False) + "\n", misses))
    return out



def convert_split(
    cortex_path: Path,
    image_dir: Path,
    dst: Path,
    *,
    expected_rows: int | None,
    check_images: bool,
    task_zh2en: dict[str, str],
    act_zh2en: dict[str, str],
) -> dict[str, Any]:
    n_ep = n_out = 0
    miss_counter: dict[str, int] = {}
    miss_examples: list[str] = []
    soft_examples: list[str] = []
    unique_miss_tasks: set[str] = set()
    unique_miss_actions: set[str] = set()
    image_names = index_images(image_dir) if check_images and image_dir.is_dir() else None

    dst.parent.mkdir(parents=True, exist_ok=True)
    with cortex_path.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            episode = json.loads(line)
            n_ep += 1
            for sample_line, misses in iter_en_samples(
                episode,
                image_dir=image_dir,
                image_names=image_names,
                task_zh2en=task_zh2en,
                act_zh2en=act_zh2en,
            ):
                for m in misses:
                    kind = m.split(" ", 1)[0]
                    miss_counter[kind] = miss_counter.get(kind, 0) + 1
                    if kind in ("task_en_missing", "action_en_missing", "image_missing"):
                        if len(miss_examples) < 80:
                            miss_examples.append(m)
                        if kind == "task_en_missing" and "zh=" in m:
                            unique_miss_tasks.add(m.split("zh=", 1)[1])
                        if kind == "action_en_missing" and "zh=" in m:
                            unique_miss_actions.add(m.split("zh=", 1)[1])
                    elif len(soft_examples) < 80:
                        soft_examples.append(m)
                fout.write(sample_line)
                n_out += 1
                if n_out % 200000 == 0:
                    print(f"[progress] {dst.name}: episodes={n_ep} rows={n_out}", flush=True)

    return {
        "cortex": str(cortex_path),
        "dst": str(dst),
        "episodes": n_ep,
        "n_out": n_out,
        "expected_rows": expected_rows,
        "row_mismatch": expected_rows is not None and n_out != expected_rows,
        "miss_counter": miss_counter,
        "miss_examples": miss_examples,
        "soft_examples": soft_examples,
        "unique_miss_tasks": sorted(unique_miss_tasks),
        "unique_miss_actions": sorted(unique_miss_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cortex-dir", type=Path, default=DEFAULT_CORTEX)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--train-dst", type=Path, default=DATA / "vn_sys2_train-en.jsonl")
    parser.add_argument("--val-dst", type=Path, default=DATA / "vn_sys2_val-en.jsonl")
    parser.add_argument("--report", type=Path, default=DATA / "vn_sys2_zh2en_miss_report.json")
    parser.add_argument("--task-zh2en", type=Path, default=DATA / "vn_task_zh2en.json")
    parser.add_argument("--act-zh2en", type=Path, default=DATA / "vn_action_zh2en.json")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Do not index/check image existence (faster).",
    )
    args = parser.parse_args()

    task_zh2en = (
        json.loads(args.task_zh2en.read_text(encoding="utf-8"))
        if args.task_zh2en.exists()
        else {}
    )
    act_zh2en = (
        json.loads(args.act_zh2en.read_text(encoding="utf-8"))
        if args.act_zh2en.exists()
        else {}
    )
    print(f"[info] task_zh2en={len(task_zh2en)} act_zh2en={len(act_zh2en)}", flush=True)

    splits = [
        (
            "train",
            args.cortex_dir / "vn_norm_mem_train.jsonl",
            args.data_dir / "images" / "vn" / "train",
            args.train_dst,
        ),
        (
            "val",
            args.cortex_dir / "vn_norm_mem_val.jsonl",
            args.data_dir / "images" / "vn" / "val",
            args.val_dst,
        ),
    ]

    reports = []
    for name, cortex, image_dir, dst in splits:
        if not cortex.exists():
            raise SystemExit(f"missing cortex EN source: {cortex}")
        expected = EXPECTED_ROWS.get(name)
        print(
            f"[info] converting {name}: cortex={cortex.name} images={image_dir} "
            f"expected_rows={expected}",
            flush=True,
        )
        rep = convert_split(
            cortex,
            image_dir,
            dst,
            expected_rows=expected,
            check_images=not args.skip_image_check,
            task_zh2en=task_zh2en,
            act_zh2en=act_zh2en,
        )
        reports.append(rep)
        print(
            f"[ok] {dst.name}: episodes={rep['episodes']} rows={rep['n_out']} "
            f"mismatch={rep['row_mismatch']} misses={rep['miss_counter']} "
            f"unique_miss_tasks={len(rep['unique_miss_tasks'])} "
            f"unique_miss_actions={len(rep['unique_miss_actions'])}",
            flush=True,
        )

    note = (
        "English primarily from Cortex vn_norm_mem_*.jsonl. "
        "When Cortex task/action text still contains Chinese, lookup uses "
        "detailed_task_instruction (EN) and CDB-derived vn_*_zh2en.json maps. "
        "Image paths reuse lake_qwen35/data/images/vn/{train,val}/{episode}_{idx}.jpg."
    )
    args.report.write_text(
        json.dumps({"note": note, "files": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
