#!/usr/bin/env python3
"""Convert hermas_sys2_*_approved.jsonl ZH → EN via lake parquet (COS mount).

English sources:
  - layer-2: clip_index.text_embedding_content (sibling list under upper_clip_id)
  - layer-1 task: upper_text_embedding_content
  - skills / DEFAULT/LAST memory: fixed templates

Does not use the small layer2_all_subtasks_by_clip.json cache (only ~28k clips).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIP_RE = re.compile(r"/(\d+)\.jpg")
NUM_RE = re.compile(r"^\d+\.\s*(.*)$")

DEFAULT_VIEW = Path(
    "/share_data_lake/hermes-human-ego-10029/views/10029-hermes-data-3_VA48DX"
)

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


def load_parquet(view_root: Path):
    """Return clip_map and upper_siblings.

    clip_map: cid -> {subtask_en, task_en, upper_clip_id}
    upper_siblings: upper_id -> list[cid] ordered by start_idx, clip_id
    """
    clip_map: dict[str, dict[str, str]] = {}
    upper_raw: dict[str, list[tuple[int, str]]] = defaultdict(list)
    cols = [
        "clip_id",
        "upper_clip_id",
        "start_idx",
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
            upper = str(row.upper_clip_id) if row.upper_clip_id is not None else ""
            en = (row.text_embedding_content or "").strip()
            task_en = (row.upper_text_embedding_content or "").strip()
            clip_map[cid] = {
                "subtask_en": en,
                "task_en": task_en,
                "upper_clip_id": upper,
            }
            if upper:
                start = int(row.start_idx) if row.start_idx is not None else 0
                upper_raw[upper].append((start, cid))
        print(
            f"[info] clips={len(clip_map)} uppers={len(upper_raw)} after {split}",
            flush=True,
        )

    upper_siblings: dict[str, list[str]] = {}
    for upper, items in upper_raw.items():
        items.sort(key=lambda x: (x[0], x[1]))
        # de-dup clip ids keeping first occurrence
        seen: set[str] = set()
        ordered: list[str] = []
        for _start, cid in items:
            if cid in seen:
                continue
            seen.add(cid)
            ordered.append(cid)
        upper_siblings[upper] = ordered
    print(f"[info] upper groups={len(upper_siblings)}", flush=True)
    return clip_map, upper_siblings


def parse_zh(sample: dict) -> dict:
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
            mem: list[str] = []
            for j in range(i + 1, len(lines)):
                if not lines[j].strip() or lines[j].startswith("请输出"):
                    break
                mem.append(lines[j])
            user_prev = "\n".join(mem).strip()

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
    image = next(
        p["image"] for p in sample["messages"][1]["content"] if p.get("type") == "image"
    )
    return {
        "image": image,
        "task": task,
        "user_prev": user_prev,
        "all_subtasks": all_subtasks,
        "skill": skill,
        "current": current,
        "prev": prev,
        "next": nxt,
    }


def convert_sample(
    sample: dict,
    *,
    clip_map: dict[str, dict[str, str]],
    upper_siblings: dict[str, list[str]],
    zh2en: dict[str, str],
) -> tuple[str | None, list[str]]:
    """Preserve Chinese all-subtasks list length; map each phrase to EN via clip/CDB cache."""
    misses: list[str] = []
    parsed = parse_zh(sample)
    m = CLIP_RE.search(parsed["image"])
    if not m:
        return None, ["missing_clip_id"]
    cid = m.group(1)
    row = clip_map.get(cid)
    if not row:
        return None, [f"clip_not_in_parquet clip_id={cid}"]

    current_en = (row.get("subtask_en") or "").strip()
    if not current_en:
        return None, [f"subtask_en_missing clip_id={cid} zh={parsed['current']}"]

    task_en = (row.get("task_en") or "").strip()
    if not task_en:
        return None, [f"task_en_missing clip_id={cid} zh_task={parsed['task']}"]

    def phrase_to_en(zh: str) -> str | None:
        if not zh:
            return None
        if zh == DEFAULT_MEMORY_ZH:
            return DEFAULT_MEMORY_EN
        if zh == LAST_MEMORY_ZH:
            return LAST_MEMORY_EN
        if not has_chinese(zh):
            return zh
        if zh == parsed["current"]:
            return current_en
        en = zh2en.get(zh)
        if en:
            return en
        return None

    zh_list = parsed["all_subtasks"] or [parsed["current"]]
    en_list: list[str] = []

    # Optional sibling EN list (same upper) for phrase alignment when lengths match.
    upper = row.get("upper_clip_id") or ""
    siblings = upper_siblings.get(upper) or []
    sibling_pairs = [
        (sid, (clip_map.get(sid, {}).get("subtask_en") or "").strip()) for sid in siblings
    ]
    use_pos = (
        bool(sibling_pairs)
        and all(en for _sid, en in sibling_pairs)
        and len(sibling_pairs) == len(zh_list)
        and cid in siblings
    )
    if use_pos:
        idx = siblings.index(cid)
        en_list = [en for _sid, en in sibling_pairs]
        en_list[idx] = current_en
    else:
        for zh in zh_list:
            en = phrase_to_en(zh)
            if not en:
                misses.append(f"subtask_phrase_en_missing clip_id={cid} zh={zh}")
                en_list.append("")
            else:
                en_list.append(en)
        if any(not e for e in en_list):
            return None, misses

    skill_en = SKILL_ZH2EN.get(parsed["skill"], "Manipulate")
    if parsed["skill"] not in SKILL_ZH2EN:
        misses.append(f"skill_unmapped clip_id={cid} zh={parsed['skill']}")

    # Index of current in ZH list
    try:
        cur_idx = zh_list.index(parsed["current"])
    except ValueError:
        cur_idx = 0
    current_en_out = en_list[cur_idx] if en_list else current_en

    def map_mem(zh: str, fallback_idx: int | None) -> str | None:
        if zh == DEFAULT_MEMORY_ZH:
            return DEFAULT_MEMORY_EN
        if zh == LAST_MEMORY_ZH:
            return LAST_MEMORY_EN
        en = phrase_to_en(zh)
        if en:
            return en
        if fallback_idx is not None and 0 <= fallback_idx < len(en_list):
            return en_list[fallback_idx]
        return None

    prev_en = map_mem(
        parsed["prev"],
        (cur_idx - 1) if cur_idx > 0 and parsed["prev"] != DEFAULT_MEMORY_ZH else None,
    )
    next_en = map_mem(
        parsed["next"],
        (cur_idx + 1)
        if cur_idx + 1 < len(en_list) and parsed["next"] != LAST_MEMORY_ZH
        else None,
    )
    if parsed["next"] == LAST_MEMORY_ZH or (
        next_en is None and cur_idx + 1 >= len(en_list)
    ):
        next_en = LAST_MEMORY_EN

    if parsed["user_prev"] == DEFAULT_MEMORY_ZH:
        user_prev_en = DEFAULT_MEMORY_EN
    elif parsed["user_prev"] == parsed["prev"]:
        user_prev_en = prev_en
    else:
        user_prev_en = map_mem(
            parsed["user_prev"], (cur_idx - 1) if cur_idx > 0 else None
        )
        if user_prev_en is None:
            user_prev_en = DEFAULT_MEMORY_EN

    for label, val, zh in (
        ("current", current_en_out, parsed["current"]),
        ("prev", prev_en, parsed["prev"]),
        ("next", next_en, parsed["next"]),
        ("user_prev", user_prev_en, parsed["user_prev"]),
    ):
        if not val:
            misses.append(f"{label}_en_missing clip_id={cid} zh={zh}")
    if not all([current_en_out, prev_en, next_en, user_prev_en]):
        return None, misses

    all_block = "All subtasks:\n" + "\n".join(
        f"{i}. {s}" for i, s in enumerate(en_list, 1)
    )
    user_text = (
        f"Task: {task_en}\n\n"
        f"Previous subtask:\n{user_prev_en}\n\n"
        "Please output all subtasks, and the current skill, current subtask, "
        "previous subtask, and next subtask."
    )
    assistant = (
        f"{all_block}\n"
        f"Skill: {skill_en}\n"
        f"Current subtask: {current_en_out}\n"
        f"Previous subtask: {prev_en}\n"
        f"Next subtask: {next_en}"
    )
    out = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": parsed["image"]},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": assistant},
        ]
    }
    if has_chinese(task_en):
        misses.append(f"task_en_still_zh clip_id={cid}")
    if any(has_chinese(x) for x in en_list):
        misses.append(f"subtasks_still_zh clip_id={cid}")
    return json.dumps(out, ensure_ascii=False) + "\n", misses


def invert_en2zh(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    en2zh = json.loads(path.read_text(encoding="utf-8"))
    zh2en: dict[str, str] = {}
    for en, zh in en2zh.items():
        en = (en or "").strip()
        zh = (zh or "").strip()
        if en and zh and zh not in zh2en:
            zh2en[zh] = en
    return zh2en


def enrich_zh2en_from_clip_zh(
    zh2en: dict[str, str],
    clip_id2zh_path: Path,
    clip_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Add zh->en pairs by joining layer2_clip_id2zh with parquet EN."""
    if not clip_id2zh_path.exists():
        return zh2en
    clip_zh = json.loads(clip_id2zh_path.read_text(encoding="utf-8"))
    added = 0
    for cid, zh in clip_zh.items():
        zh = (zh or "").strip()
        en = (clip_map.get(str(cid), {}).get("subtask_en") or "").strip()
        if not zh or not en:
            continue
        if zh not in zh2en:
            zh2en[zh] = en
            added += 1
    print(f"[info] enriched zh2en +{added} from clip_id2zh join", flush=True)
    return zh2en


def convert_file(
    src: Path,
    dst: Path,
    *,
    clip_map: dict,
    upper_siblings: dict,
    zh2en: dict[str, str],
) -> dict:
    n_in = n_out = miss_rows = 0
    miss_counter: Counter[str] = Counter()
    miss_examples: list[str] = []
    soft_examples: list[str] = []
    hard_prefixes = (
        "missing_clip_id",
        "clip_not_in_parquet",
        "subtask_en_missing",
        "subtask_phrase_en_missing",
        "task_en_missing",
        "current_en_missing",
        "prev_en_missing",
        "next_en_missing",
        "user_prev_en_missing",
    )

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            sample = json.loads(line)
            out_line, misses = convert_sample(
                sample,
                clip_map=clip_map,
                upper_siblings=upper_siblings,
                zh2en=zh2en,
            )
            for m in misses:
                miss_counter[m.split(" ", 1)[0]] += 1
            hard = [m for m in misses if m.startswith(hard_prefixes)]
            if hard or out_line is None:
                miss_rows += 1
                if len(miss_examples) < 100:
                    miss_examples.extend(hard[:2] or misses[:2])
                continue
            for m in misses:
                if len(soft_examples) < 100:
                    soft_examples.append(m)
            fout.write(out_line)
            n_out += 1
            if n_in % 200000 == 0:
                print(
                    f"[progress] {src.name}: in={n_in} out={n_out} miss={miss_rows}",
                    flush=True,
                )
    tmp.replace(dst)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--train-src",
        type=Path,
        default=DATA / "hermas_sys2_train_approved.jsonl",
    )
    parser.add_argument(
        "--val-src",
        type=Path,
        default=DATA / "hermas_sys2_val_approved.jsonl",
    )
    parser.add_argument(
        "--train-dst",
        type=Path,
        default=DATA / "hermas_sys2_train_approved-en.jsonl",
    )
    parser.add_argument(
        "--val-dst",
        type=Path,
        default=DATA / "hermas_sys2_val_approved-en.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DATA / "hermas_sys2_approved_zh2en_miss_report.json",
    )
    parser.add_argument(
        "--zh2en-cache",
        type=Path,
        default=DATA / "layer2_en2zh.json",
        help="EN->ZH cache to invert for phrase fallback",
    )
    args = parser.parse_args()

    clip_map, upper_siblings = load_parquet(args.view_root)
    zh2en = invert_en2zh(args.zh2en_cache)
    zh2en = enrich_zh2en_from_clip_zh(
        zh2en, DATA / "layer2_clip_id2zh.json", clip_map
    )
    print(f"[info] zh2en phrases={len(zh2en)}", flush=True)

    reports = []
    for src, dst in ((args.val_src, args.val_dst), (args.train_src, args.train_dst)):
        # val first (smaller) then train
        print(f"[info] converting {src.name} -> {dst.name}", flush=True)
        reports.append(
            convert_file(
                src,
                dst,
                clip_map=clip_map,
                upper_siblings=upper_siblings,
                zh2en=zh2en,
            )
        )
        r = reports[-1]
        print(
            f"[ok] {dst.name}: in={r['n_in']} out={r['n_out']} miss_rows={r['miss_rows']} "
            f"counter={r['miss_counter']}",
            flush=True,
        )

    note = (
        "English from lake clip_index parquet (text_embedding_content / "
        "upper_text_embedding_content). All-subtasks = siblings under upper_clip_id. "
        "Short Chinese task names have no short EN column."
    )
    args.report.write_text(
        json.dumps({"note": note, "files": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
