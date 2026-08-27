#!/usr/bin/env python3
"""Fetch VN approved annotations from CDB (prefer Chinese) and write lake-style Sys2 JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymysql

SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)

DEFAULT_MEMORY = "这是第一个子任务，尚未完成任何子任务。"
LAST_MEMORY = "这是最后一个子任务，没有下一个子任务。"

SKILL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("抓取", re.compile(r"(抓取|拿起|拿取|握住|握紧|握持|拾取|夹取|夹持|grasp|grab|pick\s*up|take|hold|lift)", re.I)),
    ("放置", re.compile(r"(放置|放回|放入|归位|安装|插入|放下|place|put|insert|install|set\s+down)", re.I)),
    ("推动", re.compile(r"(推动|推开|推移|移动|接近|移向|move|slide|shift|approach|navigate|reach|walk|go\s+to)", re.I)),
]


def infer_skill(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "操作"
    for skill, pat in SKILL_PATTERNS:
        if pat.search(text):
            return skill
    return "操作"


def format_all_subtasks(subtasks: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, 1))


def progress_memory(completed: list[str]) -> str:
    if not completed:
        return DEFAULT_MEMORY
    return completed[-1]


def connect(args: argparse.Namespace):
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


def load_episode_ids_from_cortex(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ep = json.loads(line)
            eid = ep.get("source_episode_id") or ep.get("episode_index")
            if eid is not None:
                ids.add(int(eid))
    return ids


def fetch_approved_episode_ids(cur, limit: int | None) -> list[int]:
    sql = """
        SELECT input_id AS episode_id
        FROM vn_review_result
        WHERE passed = 1 AND is_valid = 1 AND input_table_name = 'episode'
        ORDER BY input_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    return [int(r["episode_id"]) for r in cur.fetchall()]


def fetch_episode_batch(cur, episode_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not episode_ids:
        return {}
    ph = ",".join(["%s"] * len(episode_ids))

    cur.execute(
        f"""
        SELECT e.id AS episode_id, e.task_id,
               t.name AS task_name_zh,
               tn.translated_name AS task_name_en
        FROM episode e
        LEFT JOIN task t ON t.id = e.task_id
        LEFT JOIN task_name_translation tn
          ON tn.task_id = e.task_id AND tn.language = 'en'
        WHERE e.id IN ({ph})
        """,
        episode_ids,
    )
    episodes: dict[int, dict[str, Any]] = {}
    for r in cur.fetchall():
        episodes[int(r["episode_id"])] = {
            "episode_id": int(r["episode_id"]),
            "task_id": r["task_id"],
            "task_name_zh": (r["task_name_zh"] or "").strip(),
            "task_name_en": (r["task_name_en"] or "").strip(),
            "task_desc_zh": "",
            "task_desc_en": "",
            "actions": [],
        }

    cur.execute(
        f"""
        SELECT a.input_id AS episode_id, a.result_type, a.clip_id,
               c.start_ms, c.end_ms,
               IFNULL(cd.description_en,'') AS description_en,
               IFNULL(cd.description_zh,'') AS description_zh
        FROM vn_annotation_result a
        JOIN clip c ON c.id = a.clip_id
        LEFT JOIN clip_description cd ON cd.clip_id = a.clip_id
        WHERE a.is_valid = 1
          AND a.input_table_name = 'episode'
          AND a.result_type IN ('action_clip', 'task_clip')
          AND a.input_id IN ({ph})
        ORDER BY a.input_id, c.start_ms, a.clip_id
        """,
        episode_ids,
    )
    for r in cur.fetchall():
        eid = int(r["episode_id"])
        if eid not in episodes:
            continue
        ep = episodes[eid]
        zh = (r["description_zh"] or "").strip()
        en = (r["description_en"] or "").strip()
        if r["result_type"] == "task_clip":
            if zh:
                ep["task_desc_zh"] = zh
            if en:
                ep["task_desc_en"] = en
            continue
        action = zh or en
        if not action:
            continue
        ep["actions"].append(
            {
                "clip_id": int(r["clip_id"]),
                "start_ms": int(r["start_ms"]),
                "end_ms": int(r["end_ms"]),
                "action_text": action,
            }
        )
    return episodes


def episode_task_name(ep: dict[str, Any]) -> str:
    return (
        ep.get("task_name_zh")
        or ep.get("task_desc_zh")
        or ep.get("task_name_en")
        or ep.get("task_desc_en")
        or "执行操作任务。"
    )


def iter_samples(ep: dict[str, Any]):
    actions = sorted(ep["actions"], key=lambda x: (x["start_ms"], x["end_ms"], x["clip_id"]))
    if not actions:
        return
    all_subtasks = [a["action_text"] for a in actions]
    task = episode_task_name(ep)
    for idx, action in enumerate(actions):
        subtask = action["action_text"]
        skill = infer_skill(subtask)
        user_mem = progress_memory(all_subtasks[:idx])
        next_mem = all_subtasks[idx + 1] if idx + 1 < len(all_subtasks) else LAST_MEMORY
        user_text = (
            f"任务：{task}\n\n"
            f"上一个子任务：\n{user_mem}\n\n"
            "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
        )
        assistant = (
            f"所有子任务：\n{format_all_subtasks(all_subtasks)}\n"
            f"技能：{skill}\n"
            f"当前子任务：{subtask}\n"
            f"上一个子任务：{user_mem}\n"
            f"下一个子任务：{next_mem}"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
                {"role": "assistant", "content": assistant},
            ]
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="sh-cdb-a3xciow0.sql.tencentcdb.com")
    parser.add_argument("--port", type=int, default=22651)
    parser.add_argument("--user", default="psi_datahub")
    parser.add_argument("--password", default="Q7HV5EXV3vZv")
    parser.add_argument("--database", default="data_hub")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cortex-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/dataset/Steinate/Cortex"),
        help="Reuse train/val episode splits from existing vn_norm_mem_*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/data"),
    )
    parser.add_argument("--train-name", default="vn_sys2_train.jsonl")
    parser.add_argument("--val-name", default="vn_sys2_val.jsonl")
    args = parser.parse_args()

    train_ids = load_episode_ids_from_cortex(args.cortex_dir / "vn_norm_mem_train.jsonl")
    val_ids = load_episode_ids_from_cortex(args.cortex_dir / "vn_norm_mem_val.jsonl")
    print(f"[info] cortex split train_eps={len(train_ids)} val_eps={len(val_ids)}")

    conn = connect(args)
    cur = conn.cursor()
    print("[info] fetching approved episode ids...")
    episode_ids = fetch_approved_episode_ids(cur, args.limit)
    print(f"[info] approved episodes: {len(episode_ids)}")

    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_train = n_val = n_skip = n_unassigned = 0
    with train_path.open("w", encoding="utf-8") as ft, val_path.open("w", encoding="utf-8") as fv:
        for i in range(0, len(episode_ids), args.batch_size):
            batch = episode_ids[i : i + args.batch_size]
            eps = fetch_episode_batch(cur, batch)
            for eid in batch:
                ep = eps.get(eid)
                if not ep or not ep["actions"]:
                    n_skip += 1
                    continue
                if eid in val_ids:
                    out = fv
                elif eid in train_ids or not train_ids:
                    out = ft
                else:
                    out = ft
                    n_unassigned += 1
                for sample in iter_samples(ep):
                    out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    if out is fv:
                        n_val += 1
                    else:
                        n_train += 1
            if (i // args.batch_size) % 20 == 0:
                print(
                    f"[progress] {min(i + args.batch_size, len(episode_ids))}/{len(episode_ids)} "
                    f"train={n_train} val={n_val} skip={n_skip} unassigned_eps~={n_unassigned}"
                )

    print(f"[done] train={n_train} -> {train_path}")
    print(f"[done] val={n_val} -> {val_path}")
    print(f"[done] skip={n_skip} unassigned_eps={n_unassigned}")
    conn.close()


if __name__ == "__main__":
    main()
