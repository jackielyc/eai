#!/usr/bin/env python3
"""Convert /share_data_lake clip index + zarr RGB frames to multimodal ShareGPT JSON."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


SYSTEM_PROMPT = (
    "You are a robot cognitive orchestrator for manipulation tasks. "
    "Given the scene image, task name, and progress memory, predict the current "
    "executable subtask and updated language memory. "
    "Respond with Skill, Subtask, and Memory on separate lines."
)

INDEX_COLUMNS = ("task", "volume_id", "start_idx", "end_idx", "clip_id", "zarr_path")
_ZARR_CACHE: dict[str, tuple[Any, str]] = {}


@dataclass(frozen=True)
class ClipJob:
    split: str
    task: str
    start_idx: int
    end_idx: int
    clip_id: str
    zarr_path: str
    image_path: str
    data_root: str
    datahouse_id: str
    camera: str
    skip_existing: bool


class JsonlWriter:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.count += 1
            self._handle = path.open("a", encoding="utf-8")
        else:
            self._handle = path.open("w", encoding="utf-8")

    def write(self, sample: dict[str, Any]) -> None:
        self._handle.write(json.dumps(sample, ensure_ascii=False))
        self._handle.write("\n")
        self.count += 1

    def close(self) -> None:
        self._handle.close()


class ClipCheckpoint:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[str] = []
        self._ids: set[str] = set()
        if resume and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    clip_id = line.strip()
                    if clip_id:
                        self._ids.add(clip_id)
        elif not resume and self.path.exists():
            self.path.unlink()

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self._ids

    def add(self, clip_id: str, *, flush_every: int = 256) -> None:
        if clip_id in self._ids:
            return
        self._ids.add(clip_id)
        self._pending.append(clip_id)
        if len(self._pending) >= flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for clip_id in self._pending:
                handle.write(f"{clip_id}\n")
        self._pending.clear()

    @property
    def count(self) -> int:
        return len(self._ids)


def _open_zarr_data_group(zarr_path: Path):
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    if "data" in root:
        return root["data"]
    return root


def _read_jpeg_rectified_frame(group, frame_idx: int) -> np.ndarray:
    arr = group["image_rectified"]
    lens = group["image_rectified_len"]
    idx = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
    length = int(lens[idx])
    buf = np.asarray(arr[idx])[:length]
    return np.asarray(Image.open(BytesIO(buf.tobytes())).convert("RGB"))


def _write_jpeg_rectified_frame(group, frame_idx: int, out_path: Path) -> None:
    arr = group["image_rectified"]
    lens = group["image_rectified_len"]
    idx = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
    length = int(lens[idx])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(np.asarray(arr[idx])[:length].tobytes())


def _read_rgb_array_frame(group, camera: str, frame_idx: int) -> np.ndarray:
    if camera not in group:
        available = [k for k in group.keys() if k.startswith("rgb_")]
        if not available:
            raise KeyError(f"No RGB camera in group, keys={list(group.keys())[:20]}")
        camera = available[0]
    arr = group[camera]
    idx = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
    frame = np.asarray(arr[idx])
    if frame.ndim == 1:
        raise ValueError(f"Unexpected frame shape {frame.shape} for {camera}[{idx}]")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resolve_camera(group, camera: str) -> str:
    if camera != "auto":
        return camera
    if "rgb_head" in group:
        return "rgb_head"
    if "image_rectified" in group:
        return "image_rectified"
    rgb_keys = [k for k in group.keys() if k.startswith("rgb_")]
    if rgb_keys:
        return rgb_keys[0]
    raise KeyError(f"No supported image camera in zarr, keys={list(group.keys())[:20]}")


def _get_zarr_group_camera(zarr_path: Path, camera: str) -> tuple[Any, str]:
    key = str(zarr_path)
    cached = _ZARR_CACHE.get(key)
    if cached is not None:
        return cached
    group = _open_zarr_data_group(zarr_path)
    resolved = _resolve_camera(group, camera)
    _ZARR_CACHE[key] = (group, resolved)
    return group, resolved


def _export_frame(group, camera: str, frame_idx: int, out_path: Path) -> None:
    if camera == "image_rectified":
        _write_jpeg_rectified_frame(group, frame_idx, out_path)
        return
    frame = _read_rgb_array_frame(group, camera, frame_idx)
    _save_rgb_jpg(frame, out_path)


def _read_rgb_frame(zarr_path: Path, frame_idx: int, camera: str = "auto") -> np.ndarray:
    group, resolved = _get_zarr_group_camera(zarr_path, camera)
    if resolved == "image_rectified":
        return _read_jpeg_rectified_frame(group, frame_idx)
    return _read_rgb_array_frame(group, resolved, frame_idx)


def _save_rgb_jpg(frame: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path, quality=92)


def _task_to_instruction(task: str) -> str:
    text = task.replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Execute the manipulation task."
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    return text[0].upper() + text[1:]


def _infer_skill(task: str) -> str:
    lower = task.lower()
    if any(token in lower for token in ("pick", "grasp", "抓取", "拿", "取")):
        return "Pick"
    if any(token in task for token in ("place", "放", "归位", "放回", "安装", "插入")):
        return "Place"
    if any(token in task for token in ("push", "推", "移")):
        return "Push"
    return "Manipulate"


def _build_assistant(task: str, clip_length: int) -> str:
    instruction = _task_to_instruction(task)
    skill = _infer_skill(task)
    subtask = instruction
    memory = (
        f"The robot is executing '{instruction}'. "
        f"Current clip covers {clip_length} control steps at the sampled frame."
    )
    return f"Skill: {skill}\nSubtask: {subtask}\nMemory: {memory}"


def _build_sample(
    image_path: Path,
    task: str,
    clip_length: int,
    *,
    memory: str = "This is the first subtask, and no subtasks have been completed yet.",
) -> dict[str, Any]:
    instruction = _task_to_instruction(task)
    user_text = (
        f"Task: {instruction}\n\n"
        f"Language memory:\n{memory}\n\n"
        "Output the current skill, subtask, and updated language memory."
    )
    assistant = _build_assistant(task, clip_length)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": assistant},
        ]
    }


def _resolve_zarr_path(data_root: Path, datahouse_id: str, row: pd.Series) -> Path:
    if "zarr_path" in row and pd.notna(row["zarr_path"]) and str(row["zarr_path"]).strip():
        rel = str(row["zarr_path"]).strip()
        return data_root / datahouse_id / rel
    task = str(row["task"])
    volume_id = str(row["volume_id"])
    return data_root / datahouse_id / "tasks" / task / f"{volume_id}.zarr"


def _parse_task_list(raw: str | None, task_list_file: Path | None) -> list[str] | None:
    tasks: list[str] = []
    if raw:
        tasks.extend(t.strip() for t in raw.split(",") if t.strip())
    if task_list_file is not None:
        text = task_list_file.read_text(encoding="utf-8")
        tasks.extend(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
    if not tasks:
        return None
    # Preserve order while deduplicating.
    return list(dict.fromkeys(tasks))


def _prepare_clip_index(
    df: pd.DataFrame,
    *,
    split: str,
    tasks: list[str] | None = None,
    max_per_task: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    oversample: float = 3.0,
) -> pd.DataFrame:
    total_rows = len(df)
    if tasks:
        task_set = set(tasks)
        df = df[df["task"].isin(task_set)].reset_index(drop=True)
        missing = task_set - set(df["task"].unique())
        print(
            f"[info] {split}: task filter {len(task_set)} tasks -> {len(df)}/{total_rows} rows"
            + (f" ({len(missing)} tasks not in index)" if missing else "")
        )
        if df.empty:
            return df

    if max_per_task is not None:
        parts: list[pd.DataFrame] = []
        for task_name, group in df.groupby("task", sort=False):
            n = min(len(group), max_per_task)
            parts.append(group.sample(n=n, random_state=seed))
        df = pd.concat(parts, ignore_index=True)
        print(
            f"[info] {split}: max_per_task={max_per_task} -> "
            f"{len(df)} rows across {df['task'].nunique()} tasks"
        )

    if max_samples is not None and len(df) > max_samples:
        before = len(df)
        sample_size = min(before, max(int(max_samples * oversample), max_samples))
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"[info] {split}: global cap sampled {sample_size}/{before} rows before decode")

    return df


def _resolve_zarr_path_str(data_root: Path, datahouse_id: str, row_dict: dict[str, Any]) -> Path:
    return _resolve_zarr_path(data_root, datahouse_id, pd.Series(row_dict))


def _row_to_job(
    row_dict: dict[str, Any],
    *,
    split: str,
    data_root: Path,
    datahouse_id: str,
    view_id: str,
    image_dir: Path,
    camera: str,
    skip_existing: bool,
) -> ClipJob | None:
    zarr_path = _resolve_zarr_path_str(data_root, datahouse_id, row_dict)
    if not zarr_path.exists():
        return None
    start_idx = int(row_dict["start_idx"])
    end_idx = int(row_dict["end_idx"])
    clip_length = max(1, end_idx - start_idx + 1)
    clip_id = str(row_dict.get("clip_id", f"{row_dict['volume_id']}_{start_idx}"))
    image_path = image_dir / datahouse_id / view_id / split / f"{clip_id}.jpg"
    return ClipJob(
        split=split,
        task=str(row_dict["task"]),
        start_idx=start_idx,
        end_idx=end_idx,
        clip_id=clip_id,
        zarr_path=str(zarr_path),
        image_path=str(image_path.resolve()),
        data_root=str(data_root),
        datahouse_id=datahouse_id,
        camera=camera,
        skip_existing=skip_existing,
    )


def _process_clip_job(job: ClipJob) -> dict[str, Any] | None:
    image_path = Path(job.image_path)
    clip_length = max(1, job.end_idx - job.start_idx + 1)
    if job.skip_existing and image_path.exists() and image_path.stat().st_size > 0:
        return _build_sample(image_path, job.task, clip_length)

    zarr_path = Path(job.zarr_path)
    frame_idx = job.start_idx + clip_length // 2
    try:
        group, camera = _get_zarr_group_camera(zarr_path, job.camera)
        _export_frame(group, camera, frame_idx, image_path)
    except Exception:
        return None
    return _build_sample(image_path, job.task, clip_length)


def _init_worker() -> None:
    _ZARR_CACHE.clear()


def _load_filtered_index(
    index_path: Path,
    *,
    split: str,
    tasks: list[str] | None,
    max_per_task: int | None,
    max_samples: int | None,
    seed: int,
    oversample: float,
) -> pd.DataFrame | None:
    available_cols = pq.read_schema(index_path).names
    read_cols = [c for c in INDEX_COLUMNS if c in available_cols]
    df = pd.read_parquet(index_path, columns=read_cols)
    required = {"task", "volume_id", "start_idx", "end_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{index_path} missing columns: {sorted(missing)}")
    df = _prepare_clip_index(
        df,
        split=split,
        tasks=tasks,
        max_per_task=max_per_task,
        max_samples=max_samples,
        seed=seed,
        oversample=oversample,
    )
    if df.empty:
        return None
    sort_cols = [c for c in ("zarr_path", "volume_id", "start_idx") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    return df


def _iter_index_rows(
    index_path: Path,
    *,
    split: str,
    tasks: list[str] | None,
    max_per_task: int | None,
    max_samples: int | None,
    seed: int,
    oversample: float,
    stream_batches: bool,
) -> Iterator[dict[str, Any]]:
    if not stream_batches:
        df = _load_filtered_index(
            index_path,
            split=split,
            tasks=tasks,
            max_per_task=max_per_task,
            max_samples=max_samples,
            seed=seed,
            oversample=oversample,
        )
        if df is None:
            return
        yield from df.to_dict(orient="records")
        return

    available_cols = pq.read_schema(index_path).names
    read_cols = [c for c in INDEX_COLUMNS if c in available_cols]
    task_set = set(tasks) if tasks else None
    task_counts: dict[str, int] = defaultdict(int)
    emitted = 0
    pf = pq.ParquetFile(index_path)
    print(f"[info] {split}: streaming parquet batches from {index_path.name}")

    for batch in pf.iter_batches(batch_size=8192, columns=read_cols):
        df = batch.to_pandas()
        if task_set is not None:
            df = df[df["task"].isin(task_set)]
        sort_cols = [c for c in ("zarr_path", "volume_id", "start_idx") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")
        for row_dict in df.to_dict(orient="records"):
            task_name = str(row_dict["task"])
            if max_per_task is not None:
                if task_counts[task_name] >= max_per_task:
                    continue
                task_counts[task_name] += 1
            yield row_dict
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return


def export_view_split(
    data_root: Path,
    datahouse_id: str,
    view_id: str,
    split: str,
    image_dir: Path,
    output_dir: Path,
    *,
    camera: str,
    max_samples: int | None = None,
    tasks: list[str] | None = None,
    max_per_task: int | None = None,
    seed: int = 42,
    oversample: float = 3.0,
    num_workers: int = 1,
    skip_existing: bool = True,
    resume: bool = True,
    stream_batches: bool = False,
) -> int:
    if max_samples is not None and max_samples <= 0:
        print(f"[skip] {split}: max_samples={max_samples}")
        return 0

    view_root = data_root / datahouse_id / "views" / view_id
    index_path = view_root / f"clip_index_{split}.parquet"
    if not index_path.exists() or index_path.stat().st_size == 0:
        print(f"[skip] missing/empty index: {index_path}")
        return 0

    jsonl_path = output_dir / f"lake_sys2_{split}.jsonl"
    checkpoint = ClipCheckpoint(output_dir / ".convert_checkpoint" / f"{split}.clip_ids", resume=resume)
    writer = JsonlWriter(jsonl_path, resume=resume)
    if resume and max_samples is not None and writer.count >= max_samples:
        print(f"[info] {split}: already have {writer.count} samples, target={max_samples}")
        writer.close()
        return min(writer.count, max_samples)

    target = max_samples
    if max_per_task is not None and max_samples is None:
        target = None

    workers = max(1, num_workers)
    started = time.time()
    pending_jobs: list[ClipJob] = []
    batch_size = max(workers * 8, workers)
    processed_batches = 0
    print(
        f"[info] {split}: stream -> {jsonl_path.name} "
        f"(resume={resume}, checkpoint={checkpoint.count}, workers={workers})"
    )

    def _flush_jobs() -> bool:
        nonlocal pending_jobs, processed_batches
        if not pending_jobs:
            return False
        if workers == 1:
            for job in pending_jobs:
                if target is not None and writer.count >= target:
                    return True
                if job.clip_id in checkpoint:
                    continue
                sample = _process_clip_job(job)
                if sample is None:
                    continue
                writer.write(sample)
                checkpoint.add(job.clip_id)
                if writer.count % 1000 == 0:
                    checkpoint.flush()
                    elapsed = max(time.time() - started, 1e-6)
                    print(f"[progress] {split}: {writer.count} samples ({writer.count / elapsed:.1f} clips/s)")
            pending_jobs = []
            processed_batches += 1
            return target is not None and writer.count >= target

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {pool.submit(_process_clip_job, job): job for job in pending_jobs}
            for future in as_completed(futures):
                if target is not None and writer.count >= target:
                    break
                job = futures[future]
                if job.clip_id in checkpoint:
                    continue
                sample = future.result()
                if sample is None:
                    continue
                writer.write(sample)
                checkpoint.add(job.clip_id)
                if writer.count % 1000 == 0:
                    checkpoint.flush()
                    elapsed = max(time.time() - started, 1e-6)
                    print(f"[progress] {split}: {writer.count} samples ({writer.count / elapsed:.1f} clips/s)")
        pending_jobs = []
        processed_batches += 1
        return target is not None and writer.count >= target

    for row_dict in _iter_index_rows(
        index_path,
        split=split,
        tasks=tasks,
        max_per_task=max_per_task,
        max_samples=max_samples if stream_batches else None,
        seed=seed,
        oversample=oversample,
        stream_batches=stream_batches,
    ):
        if target is not None and writer.count >= target:
            break
        job = _row_to_job(
            row_dict,
            split=split,
            data_root=data_root,
            datahouse_id=datahouse_id,
            view_id=view_id,
            image_dir=image_dir,
            camera=camera,
            skip_existing=skip_existing,
        )
        if job is None or job.clip_id in checkpoint:
            continue
        pending_jobs.append(job)
        if len(pending_jobs) >= batch_size:
            if _flush_jobs():
                break

    if pending_jobs and (target is None or writer.count < target):
        _flush_jobs()

    checkpoint.flush()
    writer.close()
    elapsed = max(time.time() - started, 1e-6)
    print(f"[ok] {split}: {writer.count} samples in {jsonl_path.name} ({writer.count / elapsed:.2f} clips/s)")
    return writer.count


def convert_source(
    data_root: Path,
    datahouse_id: str,
    view_id: str,
    output_dir: Path,
    *,
    camera: str = "auto",
    subset_train: int = 20000,
    subset_val: int = 2000,
    skip_full: bool = False,
    tasks: list[str] | None = None,
    max_per_task: int | None = None,
    num_workers: int = 1,
    skip_existing: bool = True,
    resume: bool = True,
    seed: int = 42,
) -> tuple[int, int]:
    image_dir = output_dir / "images"
    train_limit = subset_train if skip_full and max_per_task is None else None
    val_limit = subset_val if skip_full and max_per_task is None else None
    stream_batches = not skip_full and max_per_task is None and not tasks
    if skip_full and max_per_task is not None:
        print("[info] max_per_task set: ignoring SUBSET_TRAIN/SUBSET_VAL global cap during decode")

    common = dict(
        data_root=data_root,
        datahouse_id=datahouse_id,
        view_id=view_id,
        image_dir=image_dir,
        output_dir=output_dir,
        camera=camera,
        tasks=tasks,
        max_per_task=max_per_task,
        num_workers=num_workers,
        skip_existing=skip_existing,
        resume=resume,
        stream_batches=stream_batches,
    )
    train_count = export_view_split(
        split="train",
        max_samples=train_limit,
        seed=seed,
        **common,
    )
    val_count = export_view_split(
        split="val",
        max_samples=val_limit,
        seed=seed + 1,
        **common,
    )
    print(f"[ok] {datahouse_id}/{view_id}: train={train_count}, val={val_count}")
    return train_count, val_count


def jsonl_to_json_array(jsonl_path: Path, json_path: Path, max_samples: int | None = None) -> int:
    if not jsonl_path.exists():
        write_json(json_path, [])
        return 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as src, json_path.open("w", encoding="utf-8") as dst:
        dst.write("[")
        first = True
        for line in src:
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and count >= max_samples:
                break
            if not first:
                dst.write(",")
            dst.write(line)
            first = False
            count += 1
        dst.write("]")
    print(f"[write] {json_path} ({count} samples from {jsonl_path.name})")
    return count


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[write] {path} ({len(data)} samples)")


def write_dataset_info(path: Path) -> None:
    tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    }
    info = {
        name: {
            "file_name": fname,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": tags,
        }
        for name, fname in [
            ("lake_sys2_train", "lake_sys2_train.jsonl"),
            ("lake_sys2_val", "lake_sys2_val.jsonl"),
            ("lake_sys2_train_20k", "lake_sys2_train_20k.json"),
            ("lake_sys2_val_2k", "lake_sys2_val_2k.json"),
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/share_data_lake"),
    )
    parser.add_argument(
        "--datahouse-id",
        type=str,
        default="hermes-human-ego-10029",
    )
    parser.add_argument(
        "--view-id",
        type=str,
        default="10029-hermes-data-3_VA48DX",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/data"),
    )
    parser.add_argument("--camera", type=str, default="auto")
    parser.add_argument("--subset-train", type=int, default=20000)
    parser.add_argument("--subset-val", type=int, default=2000)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to include (exact match on clip_index.task).",
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        default=None,
        help="Text file with one task name per line.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Randomly sample at most N clips per task before decoding images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel workers for image decode/export (default: all CPUs).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse already exported jpg files without re-reading zarr.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from .convert_checkpoint/* and append to existing .jsonl.",
    )
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_filter = _parse_task_list(args.tasks, args.task_list_file)

    train_count, val_count = convert_source(
        args.data_root,
        args.datahouse_id,
        args.view_id,
        args.output_dir,
        camera=args.camera,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        skip_full=args.skip_full,
        tasks=task_filter,
        max_per_task=args.max_per_task,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
        resume=args.resume,
        seed=args.seed,
    )

    write_dataset_info(args.output_dir / "dataset_info.json")
    subset_train = None if args.max_per_task else args.subset_train
    subset_val = None if args.max_per_task else args.subset_val
    jsonl_to_json_array(
        args.output_dir / "lake_sys2_train.jsonl",
        args.output_dir / "lake_sys2_train_20k.json",
        subset_train,
    )
    jsonl_to_json_array(
        args.output_dir / "lake_sys2_val.jsonl",
        args.output_dir / "lake_sys2_val_2k.json",
        subset_val,
    )
    print(
        f"[done] train={train_count} val={val_count} "
        f"(jsonl + subset json; full export trains on *.jsonl directly)"
    )


if __name__ == "__main__":
    main()
