#!/usr/bin/env python3
"""Build VN EN->ZH maps from CDB, then rewrite cortex norm_mem into Chinese lake Sys2 JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)
DEFAULT_MEMORY = "这是第一个子任务，尚未完成任何子任务。"
LAST_MEMORY = "这是最后一个子任务，没有下一个子任务。"

SKILL_EN2ZH = {
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

SKILL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("抓取", re.compile(r"(抓取|拿起|拿取|握住|握紧|握持|拾取|夹取|夹持|grasp|grab|pick\s*up|take|hold|lift)", re.I)),
    ("放置", re.compile(r"(放置|放回|放入|归位|安装|插入|放下|place|put|insert|install|set\s+down)", re.I)),
    ("推动", re.compile(r"(推动|推开|推移|移动|接近|移向|move|slide|shift|approach|navigate|reach|walk|go\s+to)", re.I)),
]


def infer_skill(text: str, en_skill: str = "") -> str:
    text = (text or "").strip()
    for skill, pat in SKILL_PATTERNS:
        if pat.search(text):
            return skill
    return SKILL_EN2ZH.get((en_skill or "").strip(), "操作")


def format_all_subtasks(subtasks: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, 1))


def progress_memory(completed: list[str]) -> str:
    if not completed:
        return DEFAULT_MEMORY
    return completed[-1]


def connect(args: argparse.Namespace):
    import pymysql

    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=600,
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_action_en2zh(cur) -> dict[str, str]:
    print("[info] fetching distinct VN action EN->ZH ...", flush=True)
    cur.execute(
        """
        SELECT cd.description_en AS en, cd.description_zh AS zh
        FROM vn_review_result r
        JOIN vn_annotation_result a
          ON a.input_table_name = 'episode'
         AND a.input_id = r.input_id
         AND a.is_valid = 1
         AND a.result_type = 'action_clip'
        JOIN clip_description cd ON cd.clip_id = a.clip_id
        WHERE r.passed = 1 AND r.is_valid = 1 AND r.input_table_name = 'episode'
          AND IFNULL(cd.description_en,'') <> ''
          AND IFNULL(cd.description_zh,'') <> ''
        GROUP BY cd.description_en, cd.description_zh
        """
    )
    mapping: dict[str, str] = {}
    conflicts = 0
    for row in cur.fetchall():
        en = (row["en"] or "").strip()
        zh = (row["zh"] or "").strip()
        if not en or not zh:
            continue
        if en in mapping and mapping[en] != zh:
            conflicts += 1
            continue
        mapping[en] = zh
    print(f"[info] action_en2zh={len(mapping)} conflicts_skipped={conflicts}", flush=True)
    return mapping


def fetch_task_en2zh(cur) -> dict[str, str]:
    print("[info] fetching task EN->ZH ...", flush=True)
    cur.execute(
        """
        SELECT tn.translated_name AS en, t.name AS zh
        FROM task_name_translation tn
        JOIN task t ON t.id = tn.task_id
        WHERE tn.language = 'en'
          AND IFNULL(tn.translated_name,'') <> ''
          AND IFNULL(t.name,'') <> ''
        """
    )
    mapping: dict[str, str] = {}
    for row in cur.fetchall():
        en = (row["en"] or "").strip()
        zh = (row["zh"] or "").strip()
        if en and zh:
            mapping[en] = zh
    print(f"[info] task_en2zh={len(mapping)}", flush=True)
    return mapping


def translate(text: str, mapping: dict[str, str]) -> str:
    text = (text or "").strip()
    if not text:
        return text
    return mapping.get(text, text)


def rewrite_episode(episode: dict[str, Any], act_map: dict[str, str], task_map: dict[str, str]):
    actions = episode.get("action_config") or []
    if not actions:
        return
    tasks = episode.get("tasks") or []
    task_en = (tasks[0] if tasks else "") or ""
    task = translate(task_en.strip(), task_map) or task_en.strip() or "执行操作任务。"

    all_subtasks: list[str] = []
    skills: list[str] = []
    for a in actions:
        en = (a.get("action_text") or "").strip()
        if not en:
            continue
        zh = translate(en, act_map)
        all_subtasks.append(zh)
        skills.append(infer_skill(zh, (a.get("skill") or "").strip()))

    if not all_subtasks:
        return

    for idx, subtask in enumerate(all_subtasks):
        prev = progress_memory(all_subtasks[:idx])
        next_mem = all_subtasks[idx + 1] if idx + 1 < len(all_subtasks) else LAST_MEMORY
        user_text = (
            f"任务：{task}\n\n"
            f"上一个子任务：\n{prev}\n\n"
            "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
        )
        assistant = (
            f"所有子任务：\n{format_all_subtasks(all_subtasks)}\n"
            f"技能：{skills[idx]}\n"
            f"当前子任务：{subtask}\n"
            f"上一个子任务：{prev}\n"
            f"下一个子任务：{next_mem}"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
                {"role": "assistant", "content": assistant},
            ]
        }


def convert_file(src: Path, dst: Path, act_map: dict[str, str], task_map: dict[str, str]) -> tuple[int, int]:
    n = miss_act = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ep = json.loads(line)
            for a in ep.get("action_config") or []:
                en = (a.get("action_text") or "").strip()
                if en and en not in act_map:
                    miss_act += 1
            for sample in rewrite_episode(ep, act_map, task_map):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
            if n and n % 200000 == 0:
                print(f"[progress] {src.name}: {n}", flush=True)
    return n, miss_act


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="sh-cdb-a3xciow0.sql.tencentcdb.com")
    parser.add_argument("--port", type=int, default=22651)
    parser.add_argument("--user", default="psi_datahub")
    parser.add_argument("--password", default="Q7HV5EXV3vZv")
    parser.add_argument("--database", default="data_hub")
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/data"),
    )
    parser.add_argument("--train-name", default="vn_sys2_train.jsonl")
    parser.add_argument("--val-name", default="vn_sys2_val.jsonl")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    act_cache = args.cache_dir / "vn_action_en2zh.json"
    task_cache = args.cache_dir / "vn_task_en2zh.json"

    if act_cache.exists() and task_cache.exists() and not args.refresh_cache:
        print("[info] loading cached maps", flush=True)
        act_map = json.loads(act_cache.read_text(encoding="utf-8"))
        task_map = json.loads(task_cache.read_text(encoding="utf-8"))
        print(f"[info] cached action={len(act_map)} task={len(task_map)}", flush=True)
    else:
        conn = connect(args)
        cur = conn.cursor()
        act_map = fetch_action_en2zh(cur)
        task_map = fetch_task_en2zh(cur)
        conn.close()
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        act_cache.write_text(json.dumps(act_map, ensure_ascii=False), encoding="utf-8")
        task_cache.write_text(json.dumps(task_map, ensure_ascii=False), encoding="utf-8")
        print(f"[write] {act_cache}", flush=True)
        print(f"[write] {task_cache}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_src = args.cortex_dir / "vn_norm_mem_train.jsonl"
    val_src = args.cortex_dir / "vn_norm_mem_val.jsonl"
    train_dst = args.output_dir / args.train_name
    val_dst = args.output_dir / args.val_name

    n_train, miss_tr = convert_file(train_src, train_dst, act_map, task_map)
    print(f"[done] train={n_train} miss_action_en={miss_tr} -> {train_dst}", flush=True)
    n_val, miss_va = convert_file(val_src, val_dst, act_map, task_map)
    print(f"[done] val={n_val} miss_action_en={miss_va} -> {val_dst}", flush=True)

    # quick zh ratio check
    zh_re = re.compile(r"[\u4e00-\u9fff]")
    with train_dst.open(encoding="utf-8") as f:
        line = f.readline()
    sample = json.loads(line)
    asst = sample["messages"][2]["content"]
    user = sample["messages"][1]["content"][0]["text"]
    print("[sample] user_has_zh=", bool(zh_re.search(user)), "asst_has_zh=", bool(zh_re.search(asst)), flush=True)
    print(user[:200], flush=True)
    print(asst[:300], flush=True)


if __name__ == "__main__":
    # Ensure sibling imports are not required.
    sys.exit(main() or 0)
