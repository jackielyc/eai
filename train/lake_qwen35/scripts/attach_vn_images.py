#!/usr/bin/env python3
"""Extract VN ego frames from COS MCAPs and rewrite vn_sys2_{train,val}.jsonl with images."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from qcloud_cos import CosConfig, CosS3Client
from mcap.reader import make_reader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localize_vn_sys2_zh import (
    LAST_MEMORY,
    format_all_subtasks,
    infer_skill,
    progress_memory,
    translate,
)

IMAGE_SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定场景图像、高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)

COS_PREFIX = "/mnt/cos/psi-dc-prod-data/"
BUCKET = "psi-dc-prod-data-1351596430"
REGION = "ap-shanghai"
BLOCK = 8 * 1024 * 1024
BLOCK_CACHE_MAX = 64
FRAME_CACHE_MAX = 16384
TOPIC_PREF = (
    "/hal/camera/head/rgb/color/rect/image/compressed",
    "/hal/camera/left/rgb/color/rect/image/compressed",
    "/hal/camera/front/rgb/color/rect/image/compressed",
    "/hal/camera/left_front/rgb/color/rect/image/compressed",
    "/hal/camera/right/rgb/color/rect/image/compressed",
)

_ACT_MAP: dict[str, str] = {}
_TASK_MAP: dict[str, str] = {}
_PREFER_LOCAL = True
_COS_MOUNT = Path(COS_PREFIX)
_TLS = threading.local()
_FRAME_CACHE: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_FRAME_CACHE_LOCK = threading.Lock()
_FRAME_CACHE_LOCAL = 0
_FRAME_CACHE_COS = 0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEXT_SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。"
    "给定高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)


class EpisodeCheckpoint:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[str] = []
        self._ids: set[str] = set()
        if resume and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    eid = line.strip()
                    if eid:
                        self._ids.add(eid)
        elif not resume and self.path.exists():
            self.path.unlink()

    def __contains__(self, eid: str) -> bool:
        return eid in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, eid: str, *, flush_every: int = 256) -> None:
        if eid in self._ids:
            return
        self._ids.add(eid)
        self._pending.append(eid)
        if len(self._pending) >= flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for eid in self._pending:
                handle.write(f"{eid}\n")
        self._pending.clear()

    def clear(self) -> None:
        self._ids.clear()
        self._pending.clear()
        if self.path.exists():
            self.path.unlink()


def episode_id(episode: dict[str, Any]) -> str:
    return str(episode.get("source_episode_id") or episode.get("episode_index") or "unk")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _cred_expire(cred: dict[str, Any]) -> float:
    raw = cred.get("ExpiredTime") or cred.get("Expiration") or cred.get("expiredTime")
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1e12:
            value /= 1000.0
        return value
    if isinstance(raw, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time() + 3500.0


def _cos_client() -> CosS3Client:
    now = time.time()
    client = getattr(_TLS, "client", None)
    expire = float(getattr(_TLS, "expire", 0.0) or 0.0)
    if client is not None and now < expire - 120:
        return client
    url = os.environ.get("QCLOUD_CONTAINER_INSTANCE_CREDENTIALS_URL")
    if not url:
        raise RuntimeError("QCLOUD_CONTAINER_INSTANCE_CREDENTIALS_URL is not set")
    with urllib.request.urlopen(url, timeout=15) as resp:
        cred = json.loads(resp.read().decode())
    cfg = CosConfig(
        Region=REGION,
        SecretId=cred["TmpSecretId"],
        SecretKey=cred["TmpSecretKey"],
        Token=cred["Token"],
        Scheme="https",
    )
    client = CosS3Client(cfg)
    _TLS.client = client
    _TLS.expire = _cred_expire(cred)
    return client


class CosFile:
    def __init__(self, client: CosS3Client, key: str, size: int):
        self.client = client
        self.key = key
        self.size = size
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.nreq = 0
        self.nbytes = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        self.pos = max(0, min(int(self.pos), self.size))
        return self.pos

    def _blk(self, idx: int) -> bytes:
        if idx not in self.cache:
            start = idx * BLOCK
            end = min(start + BLOCK, self.size) - 1
            self.nreq += 1
            resp = self.client.get_object(Bucket=BUCKET, Key=self.key, Range=f"bytes={start}-{end}")
            body = resp["Body"]
            data = body.get_raw_stream().read() if hasattr(body, "get_raw_stream") else body.read()
            if len(self.cache) >= BLOCK_CACHE_MAX:
                self.cache.clear()
            self.cache[idx] = data
            self.nbytes += len(data)
        return self.cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = int(n)
        if n <= 0 or self.pos >= self.size:
            return b""
        out = bytearray()
        left = n
        while left > 0 and self.pos < self.size:
            idx = self.pos // BLOCK
            off = self.pos % BLOCK
            chunk = self._blk(idx)[off : off + left]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            left -= len(chunk)
        return bytes(out)

    def read1(self, n: int = -1) -> bytes:
        return self.read(n if n and n > 0 else BLOCK)


def jpeg_from_cdr(data: bytes) -> bytes:
    soi = data.find(b"\xff\xd8")
    eoi = data.rfind(b"\xff\xd9")
    if soi < 0 or eoi <= soi:
        raise ValueError("no jpeg in mcap message")
    return data[soi : eoi + 2]


def pick_topic(summary) -> str | None:
    topics = {ch.topic for ch in summary.channels.values()}
    for topic in TOPIC_PREF:
        if topic in topics:
            return topic
    for topic in sorted(topics):
        if topic.endswith("/rgb/color/rect/image/compressed"):
            return topic
    return None


def _extract_frames_from_reader(reader, requests: list[tuple[int, float]]) -> dict[int, bytes]:
    summary = reader.get_summary()
    if summary is None or summary.statistics is None:
        raise RuntimeError("no mcap summary")
    topic = pick_topic(summary)
    if not topic:
        raise RuntimeError("no rgb compressed topic")
    t0 = summary.statistics.message_start_time
    out: dict[int, bytes] = {}
    for frame_idx, fps in requests:
        if frame_idx in out:
            continue
        center = int(t0 + max(frame_idx, 0) * 1e9 / max(fps, 1e-6))
        jpeg = None
        for window in (8e7, 2.5e8, 8e8, 2e9):
            w = int(window)
            for _schema, _channel, message in reader.iter_messages(
                topics=[topic], start_time=center - w, end_time=center + w
            ):
                jpeg = jpeg_from_cdr(message.data)
                break
            if jpeg:
                break
        if jpeg:
            out[frame_idx] = jpeg
    return out


def mcap_local_path(key: str, cos_mount: Path) -> Path | None:
    local = cos_mount / key
    if local.is_file():
        return local
    return None


def _fetch_frames_from_mcap(key: str, requests: list[tuple[int, float]]) -> dict[int, bytes]:
    global _FRAME_CACHE_LOCAL, _FRAME_CACHE_COS
    local = mcap_local_path(key, _COS_MOUNT) if _PREFER_LOCAL else None
    if local is not None:
        _FRAME_CACHE_LOCAL += 1
        with local.open("rb") as fh:
            return _extract_frames_from_reader(make_reader(fh), requests)
    _FRAME_CACHE_COS += 1
    client = _cos_client()
    head = client.head_object(Bucket=BUCKET, Key=key)
    size = int(head["Content-Length"])
    fh = CosFile(client, key, size)
    return _extract_frames_from_reader(make_reader(fh), requests)


def extract_frames(key: str, requests: list[tuple[int, float]]) -> dict[int, bytes]:
    dedup: list[tuple[int, float]] = []
    seen: set[int] = set()
    for frame_idx, fps in requests:
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        dedup.append((frame_idx, fps))

    out: dict[int, bytes] = {}
    missing: list[tuple[int, float]] = []
    with _FRAME_CACHE_LOCK:
        for frame_idx, fps in dedup:
            cached = _FRAME_CACHE.get((key, frame_idx))
            if cached is not None:
                _FRAME_CACHE.move_to_end((key, frame_idx))
                out[frame_idx] = cached
            else:
                missing.append((frame_idx, fps))

    if missing:
        fetched = _fetch_frames_from_mcap(key, missing)
        with _FRAME_CACHE_LOCK:
            for frame_idx, jpeg in fetched.items():
                out[frame_idx] = jpeg
                cache_key = (key, frame_idx)
                _FRAME_CACHE[cache_key] = jpeg
                _FRAME_CACHE.move_to_end(cache_key)
                while len(_FRAME_CACHE) > FRAME_CACHE_MAX:
                    _FRAME_CACHE.popitem(last=False)
    return out


def action_jobs(episode: dict[str, Any], act_map: dict[str, str], task_map: dict[str, str]):
    actions = episode.get("action_config") or []
    if not actions:
        return
    tasks = episode.get("tasks") or []
    task_en = (tasks[0] if tasks else "") or ""
    task = translate(task_en.strip(), task_map) or task_en.strip() or "执行操作任务。"
    kept: list[tuple[int, dict[str, Any], str, str]] = []
    all_subtasks: list[str] = []
    skills: list[str] = []
    for a in actions:
        en = (a.get("action_text") or "").strip()
        if not en:
            continue
        zh = translate(en, act_map)
        all_subtasks.append(zh)
        skills.append(infer_skill(zh, (a.get("skill") or "").strip()))
        kept.append((len(all_subtasks) - 1, a, zh, skills[-1]))
    if not all_subtasks:
        return
    fps = float(episode.get("fps") or 30.0)
    for idx, action, subtask, skill in kept:
        start = int(action.get("start_frame") or 0)
        end = int(action.get("end_frame") or start)
        frame_idx = start + max(end - start, 0) // 2
        prev = progress_memory(all_subtasks[:idx])
        next_mem = all_subtasks[idx + 1] if idx + 1 < len(all_subtasks) else LAST_MEMORY
        user_text = (
            f"任务：{task}\n\n"
            f"上一个子任务：\n{prev}\n\n"
            "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
        )
        assistant = (
            f"所有子任务：\n{format_all_subtasks(all_subtasks)}\n"
            f"技能：{skill}\n"
            f"当前子任务：{subtask}\n"
            f"上一个子任务：{prev}\n"
            f"下一个子任务：{next_mem}"
        )
        yield frame_idx, fps, idx, user_text, assistant


def mcap_key(episode: dict[str, Any]) -> str | None:
    path = (episode.get("source_path") or "").strip()
    if not path:
        return None
    if path.startswith(COS_PREFIX):
        path = path[len(COS_PREFIX) :]
    if path.startswith("/"):
        path = path.lstrip("/")
    if path.endswith(".mp4"):
        path = path[:-4] + ".mcap"
    return path


def _write_jpeg(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def process_episode(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _process_episode(payload)
    except Exception as exc:
        episode = payload.get("episode") or {}
        eid = episode_id(episode)
        return {"samples": [], "n_img": 0, "n_miss": 0, "eid": eid, "error": f"{type(exc).__name__}: {exc}"}


def _process_episode(payload: dict[str, Any]) -> dict[str, Any]:
    episode = payload["episode"]
    image_dir = Path(payload["image_dir"])
    skip_existing = payload.get("skip_existing", True)
    act_map = _ACT_MAP
    task_map = _TASK_MAP
    eid = episode_id(episode)
    key = mcap_key(episode)
    jobs = list(action_jobs(episode, act_map, task_map))
    samples: list[dict[str, Any]] = []
    n_img = n_miss = 0
    error = ""
    missing: list[tuple[int, float, Path]] = []
    paths: dict[int, Path] = {}
    for frame_idx, fps, idx, _user_text, _assistant in jobs:
        jpg = image_dir / f"{eid}_{idx}.jpg"
        paths[idx] = jpg
        if skip_existing and jpg.exists() and jpg.stat().st_size > 0:
            continue
        missing.append((frame_idx, fps, jpg))
    if missing and key:
        try:
            frames = extract_frames(key, [(frame_idx, fps) for frame_idx, fps, _jpg in missing])
            for frame_idx, _fps, jpg in missing:
                jpeg = frames.get(frame_idx)
                if jpeg:
                    _write_jpeg(jpg, jpeg)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    elif missing and not key:
        error = "no mcap key"
    for _frame_idx, _fps, idx, user_text, assistant in jobs:
        jpg = paths[idx]
        ok = jpg.exists() and jpg.stat().st_size > 0
        if ok:
            n_img += 1
            user_content: list[dict[str, Any]] = [
                {"type": "image", "image": str(jpg.resolve())},
                {"type": "text", "text": user_text},
            ]
            sys_prompt = IMAGE_SYSTEM_PROMPT
        else:
            n_miss += 1
            user_content = [{"type": "text", "text": user_text}]
            sys_prompt = TEXT_SYSTEM_PROMPT
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
    return {"samples": samples, "n_img": n_img, "n_miss": n_miss, "eid": eid, "error": error}


def iter_episodes(path: Path, limit: int | None):
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def convert_split(
    *,
    src: Path,
    dst: Path,
    image_dir: Path,
    act_map: dict[str, str],
    task_map: dict[str, str],
    workers: int,
    skip_existing: bool,
    limit: int | None,
    resume: bool,
    split: str,
    output_dir: Path,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = EpisodeCheckpoint(
        output_dir / ".convert_checkpoint" / f"vn_attach_{split}.episode_ids",
        resume=resume,
    )
    if limit is not None:
        dst = dst.with_name(f"{dst.stem}.limit{limit}.jsonl")
        print(f"[info] limit={limit}, writing {dst} (original jsonl untouched)", flush=True)
    elif not resume:
        bak = dst.with_name(f"{dst.stem}.noimage.jsonl")
        if dst.exists() and not bak.exists():
            dst.rename(bak)
            print(f"[info] renamed {dst.name} -> {bak.name}", flush=True)

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if not resume and tmp.exists():
        tmp.unlink()

    n_ep = len(checkpoint)
    n_samp = count_jsonl_lines(tmp) if resume and tmp.exists() else 0
    n_img = n_miss = n_err = 0
    t0 = time.time()
    global _ACT_MAP, _TASK_MAP, _FRAME_CACHE_LOCAL, _FRAME_CACHE_COS
    _ACT_MAP = act_map
    _TASK_MAP = task_map
    _FRAME_CACHE_LOCAL = 0
    _FRAME_CACHE_COS = 0
    last_print = 0.0
    skipped = 0

    def consume(rec: dict[str, Any]) -> None:
        nonlocal n_ep, n_samp, n_img, n_miss, n_err, last_print
        n_ep += 1
        n_img += rec["n_img"]
        n_miss += rec["n_miss"]
        if rec.get("error"):
            n_err += 1
            if n_err <= 20:
                print(f"[warn] eid={rec.get('eid')} {rec['error']}", flush=True)
        for sample in rec["samples"]:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_samp += 1
        checkpoint.add(rec["eid"])
        dt = time.time() - t0
        if n_ep % 50 == 0 or dt - last_print >= 20:
            last_print = dt
            print(
                f"[progress] {src.name} eps={n_ep} samples={n_samp} "
                f"img={n_img} miss={n_miss} err={n_err} "
                f"{dt:.0f}s {n_ep / max(dt, 1e-6):.1f}ep/s "
                f"mcap_local={_FRAME_CACHE_LOCAL} mcap_cos={_FRAME_CACHE_COS}",
                flush=True,
            )

    out_mode = "a" if resume and tmp.exists() else "w"
    if resume:
        print(
            f"[info] resume split={split} checkpoint={len(checkpoint)} "
            f"jsonl_lines={n_samp} mode={out_mode}",
            flush=True,
        )

    max_inflight = max(workers * 2, workers + 8)
    with tmp.open(out_mode, encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=workers) as pool:
        inflight: set = set()
        for ep in iter_episodes(src, limit):
            eid = episode_id(ep)
            if eid in checkpoint:
                skipped += 1
                continue
            inflight.add(
                pool.submit(
                    process_episode,
                    {
                        "episode": ep,
                        "image_dir": str(image_dir),
                        "skip_existing": skip_existing,
                    },
                )
            )
            while len(inflight) >= max_inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    consume(fut.result())
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                consume(fut.result())

    checkpoint.flush()
    tmp.replace(dst)
    checkpoint.clear()
    print(
        f"[done] {dst} eps={n_ep} samples={n_samp} img={n_img} miss={n_miss} "
        f"err={n_err} skipped={skipped} mcap_local={_FRAME_CACHE_LOCAL} "
        f"mcap_cos={_FRAME_CACHE_COS} {time.time() - t0:.0f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cortex-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/dataset/Steinate/Cortex"),
    )
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Max episodes per split (debug)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--cos-mount",
        type=Path,
        default=Path(COS_PREFIX),
        help="Local COS fuse mount; when present, read MCAPs directly instead of HTTP Range.",
    )
    parser.add_argument("--prefer-local", action="store_true", default=True)
    parser.add_argument("--no-prefer-local", action="store_true")
    parser.add_argument("--splits", default="val,train")
    args = parser.parse_args()
    skip_existing = not args.no_skip_existing
    resume = not args.no_resume
    prefer_local = not args.no_prefer_local

    global _PREFER_LOCAL, _COS_MOUNT
    _PREFER_LOCAL = prefer_local and args.cos_mount.is_dir()
    _COS_MOUNT = args.cos_mount

    act_map = json.loads((DATA / "vn_action_en2zh.json").read_text(encoding="utf-8"))
    task_map = json.loads((DATA / "vn_task_en2zh.json").read_text(encoding="utf-8"))
    print(
        f"[info] action_map={len(act_map)} task_map={len(task_map)} workers={args.workers} "
        f"resume={resume} prefer_local={_PREFER_LOCAL} cos_mount={args.cos_mount}",
        flush=True,
    )

    mapping = {
        "train": (args.cortex_dir / "vn_norm_mem_train.jsonl", args.output_dir / "vn_sys2_train.jsonl"),
        "val": (args.cortex_dir / "vn_norm_mem_val.jsonl", args.output_dir / "vn_sys2_val.jsonl"),
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src, dst = mapping[split]
        convert_split(
            src=src,
            dst=dst,
            image_dir=args.output_dir / "images" / "vn" / split,
            act_map=act_map,
            task_map=task_map,
            workers=args.workers,
            skip_existing=skip_existing,
            limit=args.limit,
            resume=resume,
            split=split,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
