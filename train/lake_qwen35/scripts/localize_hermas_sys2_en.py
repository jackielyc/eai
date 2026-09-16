#!/usr/bin/env python3
"""Convert Hermes Sys2 Chinese JSONL (20260825-style) to English via lake/CDB lookups.

English sources (no model translation):
  - layer-2 subtask: clip_index.text_embedding_content, else data_hub.clip_description.description_en
  - layer-1 task:    clip_index.upper_text_embedding_content, else upper_clip description_en
  - skills / boilerplate: fixed bilingual templates matching the ZH Sys2 format

Writes new files; does not overwrite the Chinese sources. Emits a miss report.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIP_RE = re.compile(r"/(\d+)\.jpg")

DEFAULT_VIEW = Path(
    "/share_data_lake/hermes-human-ego-10029/views/10029-hermes-data-3_VA48DX"
)
DEFAULT_MYSQL = "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/bin/mysql"
DEFAULT_HOST = "sh-cdb-a3xciow0.sql.tencentcdb.com"
DEFAULT_PORT = "22651"
DEFAULT_USER = "psi_datahub"
DEFAULT_PASSWORD = "Q7HV5EXV3vZv"

SYSTEM_PROMPT_EN = (
    "You are a cognitive orchestrator for robot manipulation tasks. "
    "Given a scene image, a high-level task name, and progress memory, "
    "predict the currently executable layer-2 subtask and update the language memory. "
    "Answer in three lines labeled Skill, Subtask, and Memory."
)
DEFAULT_MEMORY_EN = "This is the first subtask; no subtasks have been completed yet."
DEFAULT_MEMORY_ZH = "这是第一个子任务，尚未完成任何子任务。"

SKILL_ZH2EN = {
    "抓取": "Grasp",
    "放置": "Place",
    "推动": "Push",
    "操作": "Manipulate",
}


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def clip_id_from_sample(sample: dict) -> str | None:
    image = next(
        p["image"] for p in sample["messages"][1]["content"] if p.get("type") == "image"
    )
    match = CLIP_RE.search(image)
    return match.group(1) if match else None


def parse_zh_sample(sample: dict) -> dict[str, str]:
    user_text = next(
        p["text"] for p in sample["messages"][1]["content"] if p.get("type") == "text"
    )
    task = ""
    memory = ""
    lines = user_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("任务："):
            task = line[len("任务：") :].strip()
        if line.startswith("语言记忆："):
            mem_lines: list[str] = []
            for j in range(i + 1, len(lines)):
                if not lines[j].strip():
                    break
                if lines[j].startswith("请输出"):
                    break
                mem_lines.append(lines[j])
            memory = "\n".join(mem_lines).strip()

    asst = sample["messages"][2]["content"]
    skill = subtask = asst_memory = ""
    for line in asst.splitlines():
        if line.startswith("技能："):
            skill = line[len("技能：") :].strip()
        elif line.startswith("子任务："):
            subtask = line[len("子任务：") :].strip()
        elif line.startswith("记忆："):
            asst_memory = line[len("记忆：") :].strip()
    return {
        "task": task,
        "memory": memory,
        "skill": skill,
        "subtask": subtask,
        "asst_memory": asst_memory,
    }


def load_parquet_maps(view_root: Path) -> dict[str, dict[str, str]]:
    """clip_id -> {subtask_en, task_en, task_zh, upper_clip_id}."""
    mapping: dict[str, dict[str, str]] = {}
    cols = [
        "clip_id",
        "task",
        "upper_clip_id",
        "text_embedding_content",
        "upper_text_embedding_content",
    ]
    for split in ("train", "val", "test"):
        path = view_root / f"clip_index_{split}.parquet"
        if not path.exists():
            continue
        print(f"[info] loading {path}", flush=True)
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        for row in df.itertuples(index=False):
            cid = str(row.clip_id)
            mapping[cid] = {
                "subtask_en": (row.text_embedding_content or "").strip(),
                "task_en": (row.upper_text_embedding_content or "").strip(),
                "task_zh": (row.task or "").strip(),
                "upper_clip_id": str(row.upper_clip_id) if row.upper_clip_id is not None else "",
            }
        print(f"[info] parquet map size={len(mapping)} after {split}", flush=True)
    return mapping


def fetch_cdb_descriptions(
    ids: list[str],
    *,
    mysql: str,
    host: str,
    port: str,
    user: str,
    password: str,
    batch_size: int = 2000,
) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        id_list = ",".join(chunk)
        sql = (
            "SELECT clip_id, IFNULL(description_en,''), IFNULL(description_zh,'') "
            f"FROM data_hub.clip_description WHERE clip_id IN ({id_list});"
        )
        proc = subprocess.run(
            [mysql, "-h", host, "-P", port, "-u", user, f"-p{password}", "-N", "-e", sql],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "mysql failed")
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            while len(parts) < 3:
                parts.append("")
            cid, en, zh = parts
            mapping[cid] = {"en": en.strip(), "zh": zh.strip()}
        print(f"[progress] cdb {min(i + batch_size, len(ids))}/{len(ids)}", flush=True)
    return mapping


def build_memory_en(asst_memory_zh: str, subtask_en: str, user_memory_zh: str) -> tuple[str, str]:
    """Return (user_memory_en, assistant_memory_en)."""
    if user_memory_zh.strip() in ("", DEFAULT_MEMORY_ZH):
        user_mem_en = DEFAULT_MEMORY_EN
    elif user_memory_zh.strip() == asst_memory_zh.strip() or not has_chinese(user_memory_zh):
        user_mem_en = user_memory_zh
    else:
        # Progress memory that is itself a prior subtask phrase is uncommon in 20260825
        # (all user mem are DEFAULT); leave marked for report if still Chinese.
        user_mem_en = user_memory_zh

    prefix = "机器人正在执行："
    if asst_memory_zh.startswith(prefix):
        asst_mem_en = f"The robot is currently performing: {subtask_en}"
    elif not has_chinese(asst_memory_zh):
        asst_mem_en = asst_memory_zh
    else:
        asst_mem_en = f"The robot is currently performing: {subtask_en}"
    return user_mem_en, asst_mem_en


def convert_sample(
    sample: dict,
    *,
    clip_map: dict[str, dict[str, str]],
    cdb_map: dict[str, dict[str, str]],
) -> tuple[dict | None, list[str]]:
    misses: list[str] = []
    cid = clip_id_from_sample(sample)
    if not cid:
        return None, ["missing_clip_id"]
    parsed = parse_zh_sample(sample)
    row = clip_map.get(cid, {})
    cdb = cdb_map.get(cid, {})

    subtask_en = (row.get("subtask_en") or cdb.get("en") or "").strip()
    if not subtask_en:
        misses.append(f"subtask_en_missing clip_id={cid} zh={parsed['subtask']}")
    elif has_chinese(subtask_en):
        misses.append(f"subtask_en_still_zh clip_id={cid} en={subtask_en}")

    # Layer-1 short Chinese task names have no short EN column; use upper-layer EN description.
    task_en = (row.get("task_en") or "").strip()
    upper_id = row.get("upper_clip_id") or ""
    if not task_en and upper_id and upper_id in cdb_map:
        task_en = cdb_map[upper_id].get("en", "").strip()
    if not task_en:
        misses.append(f"task_en_missing clip_id={cid} zh_task={parsed['task']}")
    elif has_chinese(task_en):
        misses.append(f"task_en_still_zh clip_id={cid} en={task_en[:80]}")

    skill_en = SKILL_ZH2EN.get(parsed["skill"])
    if not skill_en:
        misses.append(f"skill_unmapped clip_id={cid} zh={parsed['skill']}")
        skill_en = "Manipulate"

    if misses and (not subtask_en or not task_en):
        return None, misses

    user_mem_en, asst_mem_en = build_memory_en(
        parsed["asst_memory"], subtask_en, parsed["memory"]
    )
    if has_chinese(user_mem_en):
        misses.append(f"user_memory_still_zh clip_id={cid}")

    user_text = (
        f"Task: {task_en}\n\n"
        f"Language memory:\n{user_mem_en}\n\n"
        "Please output the current skill, subtask, and updated language memory."
    )
    assistant = f"Skill: {skill_en}\nSubtask: {subtask_en}\nMemory: {asst_mem_en}"

    out = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": [
                    next(p for p in sample["messages"][1]["content"] if p.get("type") == "image"),
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": assistant},
        ]
    }
    return out, misses


def convert_jsonl(
    src: Path,
    dst: Path,
    *,
    clip_map: dict[str, dict[str, str]],
    cdb_map: dict[str, dict[str, str]],
) -> dict:
    n_in = n_out = 0
    miss_rows = 0
    miss_examples: list[str] = []
    miss_counter: Counter[str] = Counter()
    soft_misses: list[str] = []

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            sample = json.loads(line)
            converted, misses = convert_sample(sample, clip_map=clip_map, cdb_map=cdb_map)
            hard = [m for m in misses if m.startswith(("subtask_en_missing", "task_en_missing", "missing_clip_id"))]
            for m in misses:
                kind = m.split(" ", 1)[0]
                miss_counter[kind] += 1
            if hard:
                miss_rows += 1
                if len(miss_examples) < 50:
                    miss_examples.extend(hard[:2])
                continue
            assert converted is not None
            for m in misses:
                if len(soft_misses) < 50:
                    soft_misses.append(m)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 5000 == 0:
                print(f"[progress] {src.name}: {n_in} scanned, {n_out} written", flush=True)

    return {
        "src": str(src),
        "dst": str(dst),
        "n_in": n_in,
        "n_out": n_out,
        "miss_rows": miss_rows,
        "miss_counter": dict(miss_counter),
        "miss_examples": miss_examples,
        "soft_misses": soft_misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--mysql", default=DEFAULT_MYSQL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--skip-cdb", action="store_true")
    parser.add_argument(
        "--train-src",
        type=Path,
        default=DATA / "hermas_sys2_train-20260825.jsonl",
    )
    parser.add_argument(
        "--val-src",
        type=Path,
        default=DATA / "hermas_sys2_val-20260825.jsonl",
    )
    parser.add_argument(
        "--train-dst",
        type=Path,
        default=DATA / "hermas_sys2_train-20260825-en.jsonl",
    )
    parser.add_argument(
        "--val-dst",
        type=Path,
        default=DATA / "hermas_sys2_val-20260825-en.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DATA / "hermas_sys2_20260825_zh2en_miss_report.json",
    )
    args = parser.parse_args()

    clip_map = load_parquet_maps(args.view_root)

    # Collect clip ids (+ upper ids) referenced by sources.
    needed: set[str] = set()
    for src in (args.train_src, args.val_src):
        with src.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                cid = clip_id_from_sample(sample)
                if cid:
                    needed.add(cid)
                    upper = clip_map.get(cid, {}).get("upper_clip_id")
                    if upper:
                        needed.add(upper)

    missing_subtask = [
        cid
        for cid in needed
        if cid in clip_map and not clip_map[cid].get("subtask_en")
    ]
    missing_in_parquet = [cid for cid in needed if cid not in clip_map]
    print(
        f"[info] needed_ids={len(needed)} parquet_hits={len(needed) - len(missing_in_parquet)} "
        f"empty_subtask={len(missing_subtask)} not_in_parquet={len(missing_in_parquet)}",
        flush=True,
    )

    cdb_map: dict[str, dict[str, str]] = {}
    if not args.skip_cdb:
        # Prefer fetching for clips missing EN, plus a sanity sample of all needed layer-2 ids.
        fetch_ids = sorted(set(missing_subtask) | set(missing_in_parquet) | set(needed))
        # Cap: fetching all ~30k is fine
        print(f"[info] fetching CDB descriptions for {len(fetch_ids)} ids", flush=True)
        cdb_map = fetch_cdb_descriptions(
            fetch_ids,
            mysql=args.mysql,
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        # Fill parquet gaps from CDB
        filled = 0
        for cid, info in cdb_map.items():
            en = info.get("en", "")
            if not en:
                continue
            if cid not in clip_map:
                clip_map[cid] = {
                    "subtask_en": en,
                    "task_en": "",
                    "task_zh": info.get("zh", ""),
                    "upper_clip_id": "",
                }
                filled += 1
            elif not clip_map[cid].get("subtask_en"):
                clip_map[cid]["subtask_en"] = en
                filled += 1
            # If this id is an upper clip, propagate as task_en for children later via upper_id lookup
        print(f"[info] cdb filled/updated subtask_en for {filled} clips", flush=True)

    reports = []
    for src, dst in ((args.train_src, args.train_dst), (args.val_src, args.val_dst)):
        print(f"[info] converting {src.name} -> {dst.name}", flush=True)
        reports.append(convert_jsonl(src, dst, clip_map=clip_map, cdb_map=cdb_map))
        print(
            f"[ok] {dst.name}: in={reports[-1]['n_in']} out={reports[-1]['n_out']} "
            f"miss_rows={reports[-1]['miss_rows']}",
            flush=True,
        )

    # Note on task short names
    note = (
        "Layer-1 short Chinese task names (clip_index.task / 任务：) have no short English "
        "column in CDB or parquet. Task English uses upper_text_embedding_content "
        "(or upper_clip description_en), which is the official long English task description."
    )
    payload = {"note": note, "files": reports}
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] report={args.report}", flush=True)
    print(f"[note] {note}", flush=True)


if __name__ == "__main__":
    main()
