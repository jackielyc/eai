#!/usr/bin/env python3
"""Convert hermas_sys2_{train,val}.jsonl (full all-subtasks format) ZH → EN via lake/CDB.

English sources (no model translation):
  - each layer-2 phrase: sibling clip_id → text_embedding_content / clip_description.description_en
  - layer-1 task: upper_text_embedding_content / upper_clip description_en
  - skills + boilerplate templates: fixed map

Uses data/layer2_all_subtasks_by_clip.json for sibling clip order (matches JSONL lists).
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
NUM_RE = re.compile(r"^\d+\.\s*(.*)$")

DEFAULT_VIEW = Path(
    "/share_data_lake/hermes-human-ego-10029/views/10029-hermes-data-3_VA48DX"
)
DEFAULT_MYSQL = "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/bin/mysql"
DEFAULT_HOST = "sh-cdb-a3xciow0.sql.tencentcdb.com"
DEFAULT_PORT = "22651"
DEFAULT_USER = "psi_datahub"
DEFAULT_PASSWORD = "Q7HV5EXV3vZv"

SYSTEM_PROMPT_EN = (
    "You are a robot manipulation task orchestrator. "
    "Given a scene image, a high-level task name, and the previous subtask, "
    "predict the full list of subtasks under this task, and the currently executable subtask. "
    "First output a numbered list under All subtasks, then answer in four lines labeled "
    "Skill, Current subtask, Previous subtask, and Next subtask."
)
DEFAULT_MEMORY_ZH = "这是第一个子任务，尚未完成任何子任务。"
DEFAULT_MEMORY_EN = "This is the first subtask; no subtasks have been completed yet."
LAST_MEMORY_ZH = "这是最后一个子任务，没有下一个子任务。"
LAST_MEMORY_EN = "This is the last subtask; there is no next subtask."

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


def load_parquet_clip_en(view_root: Path) -> dict[str, dict[str, str]]:
    """clip_id -> {subtask_en, task_en, upper_clip_id}."""
    mapping: dict[str, dict[str, str]] = {}
    cols = [
        "clip_id",
        "upper_clip_id",
        "text_embedding_content",
        "upper_text_embedding_content",
    ]
    for split in ("train", "val", "test"):
        path = view_root / f"clip_index_{split}.parquet"
        if not path.exists():
            continue
        print(f"[info] loading {path}", flush=True)
        df = pq.read_table(path, columns=cols).to_pandas()
        for row in df.itertuples(index=False):
            cid = str(row.clip_id)
            mapping[cid] = {
                "subtask_en": (row.text_embedding_content or "").strip(),
                "task_en": (row.upper_text_embedding_content or "").strip(),
                "upper_clip_id": str(row.upper_clip_id)
                if row.upper_clip_id is not None
                else "",
            }
        print(f"[info] parquet clip_en size={len(mapping)} after {split}", flush=True)
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
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        id_list = ",".join(chunk)
        sql = (
            "SELECT clip_id, IFNULL(description_en,'') "
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
            parts = line.split("\t", 1)
            while len(parts) < 2:
                parts.append("")
            cid, en = parts
            if en.strip():
                mapping[cid] = en.strip()
        print(f"[progress] cdb {min(i + batch_size, len(ids))}/{len(ids)}", flush=True)
    return mapping


def parse_zh_sample(sample: dict) -> dict:
    user_text = next(
        p["text"] for p in sample["messages"][1]["content"] if p.get("type") == "text"
    )
    task = ""
    user_prev = ""
    lines = user_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("任务："):
            task = line[len("任务：") :].strip()
        if line.startswith("上一个子任务："):
            mem_lines: list[str] = []
            for j in range(i + 1, len(lines)):
                if not lines[j].strip():
                    break
                if lines[j].startswith("请输出"):
                    break
                mem_lines.append(lines[j])
            user_prev = "\n".join(mem_lines).strip()

    asst = sample["messages"][2]["content"]
    all_subtasks: list[str] = []
    skill = current = prev = nxt = ""
    for line in asst.splitlines():
        m = NUM_RE.match(line)
        if m:
            all_subtasks.append(m.group(1).strip())
            continue
        if line.startswith("技能："):
            skill = line[len("技能：") :].strip()
        elif line.startswith("当前子任务："):
            current = line[len("当前子任务：") :].strip()
        elif line.startswith("上一个子任务："):
            prev = line[len("上一个子任务：") :].strip()
        elif line.startswith("下一个子任务："):
            nxt = line[len("下一个子任务：") :].strip()
    return {
        "task": task,
        "user_prev": user_prev,
        "all_subtasks": all_subtasks,
        "skill": skill,
        "current": current,
        "prev": prev,
        "next": nxt,
    }


def resolve_phrase_en(
    zh: str,
    *,
    zh_list: list[str],
    en_list: list[str],
) -> str | None:
    if zh == DEFAULT_MEMORY_ZH:
        return DEFAULT_MEMORY_EN
    if zh == LAST_MEMORY_ZH:
        return LAST_MEMORY_EN
    if not zh:
        return None
    if not has_chinese(zh):
        return zh
    # Prefer positional match within the episode list (handles duplicate ZH phrases).
    try:
        idx = zh_list.index(zh)
        return en_list[idx]
    except ValueError:
        pass
    # Fallback: any equal EN already resolved for same string in list
    for z, e in zip(zh_list, en_list):
        if z == zh and e:
            return e
    return None


def convert_sample(
    sample: dict,
    *,
    siblings_cache: dict,
    clip_en: dict[str, dict[str, str]],
    cdb_en: dict[str, str],
) -> tuple[dict | None, list[str]]:
    misses: list[str] = []
    cid = clip_id_from_sample(sample)
    if not cid:
        return None, ["missing_clip_id"]
    if cid not in siblings_cache:
        return None, [f"siblings_cache_missing clip_id={cid}"]

    parsed = parse_zh_sample(sample)
    sib = siblings_cache[cid]
    sibling_ids = [str(x) for x in sib["sibling_clip_ids"]]
    zh_list = list(sib["all_subtasks_zh"])
    if parsed["all_subtasks"] != zh_list:
        misses.append(f"all_subtasks_mismatch clip_id={cid}")

    en_list: list[str] = []
    for sid, zh in zip(sibling_ids, zh_list):
        en = (clip_en.get(sid, {}).get("subtask_en") or cdb_en.get(sid) or "").strip()
        if not en:
            misses.append(f"subtask_en_missing sibling={sid} zh={zh}")
            en_list.append("")
        else:
            en_list.append(en)

    if any(not e for e in en_list):
        return None, misses

    row = clip_en.get(cid, {})
    task_en = (row.get("task_en") or "").strip()
    upper_id = row.get("upper_clip_id") or ""
    if not task_en and upper_id:
        task_en = cdb_en.get(upper_id, "").strip()
    if not task_en:
        misses.append(f"task_en_missing clip_id={cid} zh_task={parsed['task']}")
        return None, misses

    skill_en = SKILL_ZH2EN.get(parsed["skill"], "Manipulate")
    if parsed["skill"] not in SKILL_ZH2EN:
        misses.append(f"skill_unmapped clip_id={cid} zh={parsed['skill']}")

    try:
        idx = sibling_ids.index(cid)
    except ValueError:
        return None, misses + [f"clip_not_in_siblings clip_id={cid}"]

    # Prefer sibling index so duplicate ZH phrases stay aligned.
    current_en = en_list[idx]
    if parsed["current"] and parsed["current"] != zh_list[idx]:
        misses.append(
            f"current_zh_index_mismatch clip_id={cid} "
            f"jsonl={parsed['current']} cache={zh_list[idx]}"
        )

    if parsed["prev"] == DEFAULT_MEMORY_ZH:
        prev_en = DEFAULT_MEMORY_EN
    elif idx > 0 and parsed["prev"] == zh_list[idx - 1]:
        prev_en = en_list[idx - 1]
    else:
        prev_en = resolve_phrase_en(parsed["prev"], zh_list=zh_list, en_list=en_list)

    if parsed["next"] == LAST_MEMORY_ZH:
        next_en = LAST_MEMORY_EN
    elif idx + 1 < len(zh_list) and parsed["next"] == zh_list[idx + 1]:
        next_en = en_list[idx + 1]
    else:
        next_en = resolve_phrase_en(parsed["next"], zh_list=zh_list, en_list=en_list)

    if parsed["user_prev"] == DEFAULT_MEMORY_ZH:
        user_prev_en = DEFAULT_MEMORY_EN
    elif parsed["user_prev"] == parsed["prev"]:
        user_prev_en = prev_en
    else:
        user_prev_en = resolve_phrase_en(
            parsed["user_prev"], zh_list=zh_list, en_list=en_list
        )

    for label, val, zh in (
        ("current", current_en, parsed["current"]),
        ("prev", prev_en, parsed["prev"]),
        ("next", next_en, parsed["next"]),
        ("user_prev", user_prev_en, parsed["user_prev"]),
    ):
        if not val:
            misses.append(f"{label}_en_missing clip_id={cid} zh={zh}")
    if any(x is None or x == "" for x in (current_en, prev_en, next_en, user_prev_en)):
        return None, misses

    all_block = "All subtasks:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(en_list, 1))
    user_text = (
        f"Task: {task_en}\n\n"
        f"Previous subtask:\n{user_prev_en}\n\n"
        "Please output all subtasks, and the current skill, current subtask, "
        "previous subtask, and next subtask."
    )
    assistant = (
        f"{all_block}\n"
        f"Skill: {skill_en}\n"
        f"Current subtask: {current_en}\n"
        f"Previous subtask: {prev_en}\n"
        f"Next subtask: {next_en}"
    )
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
    # Soft flags: official EN still contains Chinese chars
    for label, text in (("task_en", task_en), ("subtasks", " | ".join(en_list))):
        if has_chinese(text):
            misses.append(f"{label}_still_zh clip_id={cid}")
    return out, misses


def convert_jsonl(
    src: Path,
    dst: Path,
    *,
    siblings_cache: dict,
    clip_en: dict[str, dict[str, str]],
    cdb_en: dict[str, str],
) -> dict:
    n_in = n_out = miss_rows = 0
    miss_counter: Counter[str] = Counter()
    miss_examples: list[str] = []
    soft_examples: list[str] = []

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            sample = json.loads(line)
            converted, misses = convert_sample(
                sample,
                siblings_cache=siblings_cache,
                clip_en=clip_en,
                cdb_en=cdb_en,
            )
            for m in misses:
                miss_counter[m.split(" ", 1)[0]] += 1
            hard = [
                m
                for m in misses
                if m.startswith(
                    (
                        "missing_clip_id",
                        "siblings_cache_missing",
                        "subtask_en_missing",
                        "task_en_missing",
                        "current_en_missing",
                        "prev_en_missing",
                        "next_en_missing",
                        "user_prev_en_missing",
                        "all_subtasks_mismatch",
                    )
                )
            ]
            if hard or converted is None:
                miss_rows += 1
                if len(miss_examples) < 80:
                    miss_examples.extend(hard[:3])
                continue
            for m in misses:
                if len(soft_examples) < 80 and "still_zh" in m:
                    soft_examples.append(m)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 3000 == 0:
                print(f"[progress] {src.name}: {n_in} scanned, {n_out} written", flush=True)

    return {
        "src": str(src),
        "dst": str(dst),
        "n_in": n_in,
        "n_out": n_out,
        "miss_rows": miss_rows,
        "miss_counter": dict(miss_counter),
        "miss_examples": miss_examples,
        "soft_examples": soft_examples,
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
        "--siblings-cache",
        type=Path,
        default=DATA / "layer2_all_subtasks_by_clip.json",
    )
    parser.add_argument("--train-src", type=Path, default=DATA / "hermas_sys2_train.jsonl")
    parser.add_argument("--val-src", type=Path, default=DATA / "hermas_sys2_val.jsonl")
    parser.add_argument("--train-dst", type=Path, default=DATA / "hermas_sys2_train-en.jsonl")
    parser.add_argument("--val-dst", type=Path, default=DATA / "hermas_sys2_val-en.jsonl")
    parser.add_argument(
        "--report",
        type=Path,
        default=DATA / "hermas_sys2_zh2en_miss_report.json",
    )
    args = parser.parse_args()

    siblings_cache = json.loads(args.siblings_cache.read_text(encoding="utf-8"))
    clip_en = load_parquet_clip_en(args.view_root)

    needed: set[str] = set()
    for src in (args.train_src, args.val_src):
        with src.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                cid = clip_id_from_sample(sample)
                if not cid:
                    continue
                needed.add(cid)
                info = siblings_cache.get(cid)
                if info:
                    needed.update(str(x) for x in info["sibling_clip_ids"])
                upper = clip_en.get(cid, {}).get("upper_clip_id")
                if upper:
                    needed.add(upper)

    missing_en = [
        cid
        for cid in needed
        if not (clip_en.get(cid, {}).get("subtask_en") or "").strip()
    ]
    print(
        f"[info] needed_ids={len(needed)} missing_subtask_en_in_parquet={len(missing_en)}",
        flush=True,
    )

    cdb_en: dict[str, str] = {}
    if not args.skip_cdb:
        print(f"[info] fetching CDB for {len(needed)} ids", flush=True)
        cdb_en = fetch_cdb_descriptions(
            sorted(needed),
            mysql=args.mysql,
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        filled = 0
        for cid, en in cdb_en.items():
            if cid not in clip_en:
                clip_en[cid] = {"subtask_en": en, "task_en": "", "upper_clip_id": ""}
                filled += 1
            elif not clip_en[cid].get("subtask_en"):
                clip_en[cid]["subtask_en"] = en
                filled += 1
        print(f"[info] cdb filled/updated {filled} clip EN entries", flush=True)

    reports = []
    for src, dst in ((args.train_src, args.train_dst), (args.val_src, args.val_dst)):
        print(f"[info] converting {src.name} -> {dst.name}", flush=True)
        reports.append(
            convert_jsonl(
                src,
                dst,
                siblings_cache=siblings_cache,
                clip_en=clip_en,
                cdb_en=cdb_en,
            )
        )
        r = reports[-1]
        print(
            f"[ok] {dst.name}: in={r['n_in']} out={r['n_out']} miss_rows={r['miss_rows']}",
            flush=True,
        )

    note = (
        "Layer-1 short Chinese task names have no short English column; "
        "Task uses upper_text_embedding_content / upper_clip description_en. "
        "All-subtasks English comes from sibling clip_ids via parquet/CDB."
    )
    args.report.write_text(
        json.dumps({"note": note, "files": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] report={args.report}", flush=True)
    print(f"[note] {note}", flush=True)


if __name__ == "__main__":
    main()
