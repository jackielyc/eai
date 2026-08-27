#!/usr/bin/env python3
"""Refresh official layer-2 ZH caches from data_hub.clip_description for clips in lake jsonl."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIP_RE = re.compile(r"/(\d+)\.jpg")
DEFAULT_HOST = "sh-cdb-a3xciow0.sql.tencentcdb.com"
DEFAULT_PORT = "22651"
DEFAULT_USER = "psi_datahub"


def collect_clip_ids(paths: list[Path]) -> list[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                image = next(
                    p["image"] for p in sample["messages"][1]["content"] if p.get("type") == "image"
                )
                match = CLIP_RE.search(image)
                if match:
                    ids.add(match.group(1))
    return sorted(ids)


def fetch_descriptions(
    ids: list[str],
    *,
    mysql: str,
    host: str,
    port: str,
    user: str,
    password: str,
    batch_size: int,
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
            mapping[cid] = {"en": en, "zh": zh}
        print(f"[progress] {min(i + batch_size, len(ids))}/{len(ids)}", flush=True)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mysql",
        default="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/bin/mysql",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default="")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    if not args.password:
        raise SystemExit("Pass --password (CDB password)")

    paths = [
        DATA / "hermas_sys2_train.jsonl",
        DATA / "hermas_sys2_val.jsonl",
        DATA / "hermas_sys2_train_task_100.jsonl",
        DATA / "hermas_sys2_val_task_100.jsonl",
    ]
    ids = collect_clip_ids(paths)
    print(f"[info] clip_ids={len(ids)}")
    mapping = fetch_descriptions(
        ids,
        mysql=args.mysql,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        batch_size=args.batch_size,
    )
    zh_map = {k: v["zh"] for k, v in mapping.items() if v.get("zh")}
    en2zh = {
        v["en"].strip(): v["zh"].strip()
        for v in mapping.values()
        if v.get("en", "").strip() and v.get("zh", "").strip()
    }
    (DATA / "layer2_clip_id2zh.json").write_text(
        json.dumps(zh_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA / "layer2_en2zh_official.json").write_text(
        json.dumps(en2zh, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA / "layer2_en2zh.json").write_text(
        json.dumps(en2zh, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] clips_with_zh={len(zh_map)} en2zh={len(en2zh)}")


if __name__ == "__main__":
    main()
