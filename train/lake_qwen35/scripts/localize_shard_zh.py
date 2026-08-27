#!/usr/bin/env python3
"""Shard helper: translate a slice of EN phrases into a shard cache JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse helpers from localize_dataset_zh
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localize_dataset_zh import (  # noqa: E402
    DEFAULT_MODEL,
    build_cache,
    collect_en_phrases,
    has_chinese,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--merged-cache",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "layer2_en2zh.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    phrases = collect_en_phrases([data / "hermas_sys2_train.jsonl", data / "hermas_sys2_val.jsonl"])
    merged = {}
    if args.merged_cache.exists():
        merged = json.loads(args.merged_cache.read_text(encoding="utf-8"))
    pending = [p for p in phrases if p not in merged or not has_chinese(str(merged[p]))]
    pending.sort()
    mine = pending[args.shard_id :: args.num_shards]
    shard_path = data / f"layer2_en2zh.shard{args.shard_id}.json"
    # Seed shard with empty / existing shard progress
    if not shard_path.exists():
        shard_path.write_text("{}", encoding="utf-8")
    print(
        f"[shard {args.shard_id}/{args.num_shards}] pending_global={len(pending)} "
        f"mine={len(mine)} device={args.device}",
        flush=True,
    )
    build_cache(mine, shard_path, args.model, args.batch_size, args.device)
    print(f"[shard {args.shard_id}] done -> {shard_path}", flush=True)


if __name__ == "__main__":
    main()
