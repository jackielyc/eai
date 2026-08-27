#!/usr/bin/env python3
"""Translate layer-2 English phrases to Chinese and rewrite lake SFT jsonl fully in Chinese."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_CACHE = DATA / "layer2_en2zh.json"
DEFAULT_MODEL = "/share_data/projects/mahjong/share/personal/liyichao/models/Qwen/Qwen3.5-4B"

SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定场景图像、高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)
DEFAULT_MEMORY = "这是第一个子任务，尚未完成任何子任务。"
LAST_MEMORY = "这是最后一个子任务，没有下一个子任务。"

SKILL_RULES = (
    (("pick", "grasp", "grab", "抓取", "拿起", "拿取", "握住", "握紧", "握持", "拾取", "夹取", "夹持"), "抓取"),
    (("place", "put", "insert", "install", "放置", "放回", "放入", "归位", "安装", "插入", "放下"), "放置"),
    (("push", "slide", "推动", "推开", "推移", "移动"), "推动"),
)


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def infer_skill(text: str) -> str:
    lower = text.lower()
    for tokens, skill in SKILL_RULES:
        if any(tok in text or tok in lower for tok in tokens):
            return skill
    if text.startswith("握") or text.startswith("抓"):
        return "抓取"
    if text.startswith("移") or "move" in lower:
        return "推动"
    return "操作"


def collect_en_phrases(paths: list[Path]) -> list[str]:
    phrases: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                asst = sample["messages"][2]["content"]
                for part in asst.split("\n"):
                    if part.startswith("Subtask:"):
                        text = part[len("Subtask:") :].strip()
                        if text and not has_chinese(text):
                            phrases.add(text)
                    elif part.startswith("当前子任务：") or part.startswith("子任务："):
                        prefix = "当前子任务：" if part.startswith("当前子任务：") else "子任务："
                        text = part[len(prefix) :].strip()
                        if text and not has_chinese(text):
                            phrases.add(text)
    return sorted(phrases)


def parse_translations(raw: str, expected: int) -> list[str] | None:
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", line)
        line = line.strip("`\"'“”")
        if line:
            lines.append(line)
    if len(lines) == expected:
        return lines
    # Sometimes model joins with Chinese semicolon / comma
    if len(lines) == 1 and expected > 1:
        parts = re.split(r"[；;]\s*", lines[0])
        if len(parts) == expected:
            return [p.strip() for p in parts]
    return None


def translate_batch(
    model,
    tokenizer,
    phrases: list[str],
    *,
    max_retries: int = 3,
) -> list[str]:
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(phrases))
    prompt = (
        "将下列英文机器人操作短句逐条译成简洁中文动词短语。"
        "要求：保持一一对应；每行一条译文；不要编号、解释或英文原文。\n"
        f"{numbered}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    for attempt in range(max_retries):
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max(64, 24 * len(phrases)),
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = parse_translations(decoded, len(phrases))
        if parsed is not None:
            return parsed
        # fallback: one-by-one for this batch
        if attempt == max_retries - 1:
            results: list[str] = []
            for phrase in phrases:
                one = translate_batch(model, tokenizer, [phrase], max_retries=2)
                results.append(one[0])
            return results
    raise RuntimeError(f"failed to translate batch: {phrases[:3]}...")


def build_cache(
    phrases: list[str],
    cache_path: Path,
    model_path: str,
    batch_size: int,
    device: str,
) -> dict[str, str]:
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache.update(json.loads(cache_path.read_text(encoding="utf-8")))
    todo = [p for p in phrases if p not in cache or not str(cache[p]).strip()]
    print(f"[info] unique={len(phrases)} cached={len(phrases) - len(todo)} todo={len(todo)}")
    if not todo:
        return cache

    print(f"[info] loading model {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    started = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        zh_list = translate_batch(model, tokenizer, batch)
        for en, zh in zip(batch, zh_list):
            cache[en] = zh
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(todo):
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            done = min(i + batch_size, len(todo))
            rate = done / max(time.time() - started, 1e-6)
            print(f"[progress] {done}/{len(todo)} ({rate:.1f} phrases/s)")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def rewrite_sample(sample: dict, cache: dict[str, str]) -> dict:
    user_parts = sample["messages"][1]["content"]
    text_part = next(p for p in user_parts if p.get("type") == "text")
    old_text = text_part["text"]

    task = None
    for line in old_text.splitlines():
        if line.startswith("Task:"):
            task = line[len("Task:") :].strip()
            break
        if line.startswith("任务："):
            task = line[len("任务：") :].strip()
            break
    if not task:
        task = "执行操作任务。"

    asst = sample["messages"][2]["content"]
    subtask = None
    for line in asst.splitlines():
        if line.startswith("Subtask:"):
            subtask = line[len("Subtask:") :].strip()
        elif line.startswith("当前子任务：") or line.startswith("子任务："):
            prefix = "当前子任务：" if line.startswith("当前子任务：") else "子任务："
            subtask = line[len(prefix) :].strip()

    if not subtask:
        subtask = task
    if not has_chinese(subtask):
        subtask = cache.get(subtask, subtask)
    skill = infer_skill(subtask)

    # Preserve existing 所有子任务 block if already in assistant; else single-item list.
    all_block = None
    if "所有子任务：" in asst:
        start = asst.index("所有子任务：")
        end = asst.find("\n技能：", start)
        if end < 0:
            end = asst.find("\nSkill:", start)
        all_block = asst[start:end].rstrip() if end >= 0 else f"所有子任务：\n1. {subtask}"
    if not all_block:
        all_block = f"所有子任务：\n1. {subtask}"

    next_mem = LAST_MEMORY
    numbered = []
    for line in all_block.splitlines():
        m = re.match(r"^\d+\.\s*(.*)$", line)
        if m:
            numbered.append(m.group(1).strip())
    if numbered:
        try:
            idx = numbered.index(subtask)
        except ValueError:
            idx = 0
        if idx + 1 < len(numbered):
            next_mem = numbered[idx + 1]

    text_part["text"] = (
        f"任务：{task}\n\n"
        f"上一个子任务：\n{DEFAULT_MEMORY}\n\n"
        "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
    )
    sample["messages"][0]["content"] = SYSTEM_PROMPT
    sample["messages"][2]["content"] = (
        f"{all_block}\n"
        f"技能：{skill}\n"
        f"当前子任务：{subtask}\n"
        f"上一个子任务：{DEFAULT_MEMORY}\n"
        f"下一个子任务：{next_mem}"
    )
    return sample


def rewrite_jsonl(path: Path, cache: dict[str, str]) -> tuple[int, int]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    still_en = 0
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = rewrite_sample(json.loads(line), cache)
            asst = sample["messages"][2]["content"]
            sub = next(
                p.split("：", 1)[1]
                for p in asst.split("\n")
                if p.startswith("当前子任务：") or (p.startswith("子任务：") and not p.startswith("所有子任务："))
            )
            if not has_chinese(sub):
                still_en += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n, still_en


def write_subset(jsonl: Path, out: Path, limit: int) -> int:
    rows = []
    with jsonl.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-translate", action="store_true")
    args = parser.parse_args()

    train_path = DATA / "hermas_sys2_train.jsonl"
    val_path = DATA / "hermas_sys2_val.jsonl"
    phrases = collect_en_phrases([train_path, val_path])
    print(f"[info] collected {len(phrases)} English layer-2 phrases")

    if args.skip_translate:
        cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    else:
        cache = build_cache(phrases, args.cache, args.model, args.batch_size, args.device)

    missing = [p for p in phrases if p not in cache or not has_chinese(str(cache[p]))]
    if missing:
        print(f"[warn] {len(missing)} phrases still lack Chinese; examples: {missing[:5]}")

    for path in (train_path, val_path):
        n, still_en = rewrite_jsonl(path, cache)
        print(f"[ok] {path.name}: rows={n} still_english_subtask={still_en}")

    n20 = write_subset(train_path, DATA / "hermas_sys2_train_20k.json", 20000)
    n2 = write_subset(val_path, DATA / "hermas_sys2_val_2k.json", 2000)
    print(f"[ok] subsets train_20k={n20} val_2k={n2}")

    with train_path.open("r", encoding="utf-8") as handle:
        sample = json.loads(handle.readline())
    print("--- sample ---")
    print(sample["messages"][0]["content"])
    print(sample["messages"][1]["content"][-1]["text"])
    print(sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
