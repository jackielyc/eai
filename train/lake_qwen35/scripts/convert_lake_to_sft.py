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
    "你是机器人操作任务编排器。"
    "给定场景图像、高层任务名称和上一个子任务，预测该任务下的全部子任务列表，"
    "以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「上一个子任务」「下一个子任务」四行作答。"
)

DEFAULT_MEMORY = "这是第一个子任务，尚未完成任何子任务。"
LAST_MEMORY = "这是最后一个子任务，没有下一个子任务。"

SKILL_ZH = {
    "Pick": "抓取",
    "Place": "放置",
    "Push": "推动",
    "Manipulate": "操作",
}

# clip_index rows are layer-2 clips; text_embedding_content is the fine-grained (EN) label.
# Chinese layer-2 is produced via translation cache / online map; task is layer-1 Chinese name.
INDEX_COLUMNS = (
    "task",
    "volume_id",
    "start_idx",
    "end_idx",
    "clip_id",
    "zarr_path",
    "text_embedding_content",
    "upper_text_embedding_content",
)
_ZARR_CACHE: dict[str, tuple[Any, str]] = {}
_LAYER2_ZH_CACHE: dict[str, str] | None = None
_LAYER2_ZH_CACHE_PATH: Path | None = None
_LAYER2_CLIP_ZH_CACHE: dict[str, str] | None = None
_LAYER2_ALL_SUBTASKS_CACHE: dict[str, list[str]] | None = None


@dataclass(frozen=True)
class ClipJob:
    split: str
    task: str
    subtask: str
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
        return "执行操作任务。"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    return text[0].upper() + text[1:]


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _load_layer2_zh_cache(path: Path | None = None) -> dict[str, str]:
    """EN phrase -> ZH (from data_hub.clip_description via layer2_en2zh.json)."""
    global _LAYER2_ZH_CACHE, _LAYER2_ZH_CACHE_PATH
    cache_path = path or _LAYER2_ZH_CACHE_PATH
    if cache_path is None:
        cache_path = Path(__file__).resolve().parents[1] / "data" / "layer2_en2zh.json"
    if _LAYER2_ZH_CACHE is not None and _LAYER2_ZH_CACHE_PATH == cache_path:
        return _LAYER2_ZH_CACHE
    _LAYER2_ZH_CACHE_PATH = cache_path
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        _LAYER2_ZH_CACHE = {str(k): str(v) for k, v in raw.items() if str(v).strip()}
    else:
        _LAYER2_ZH_CACHE = {}
    return _LAYER2_ZH_CACHE


def _load_layer2_clip_zh_cache() -> dict[str, str]:
    """clip_id -> official description_zh from data_hub.clip_description."""
    global _LAYER2_CLIP_ZH_CACHE
    if _LAYER2_CLIP_ZH_CACHE is not None:
        return _LAYER2_CLIP_ZH_CACHE
    cache_path = Path(__file__).resolve().parents[1] / "data" / "layer2_clip_id2zh.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        _LAYER2_CLIP_ZH_CACHE = {str(k): str(v) for k, v in raw.items() if str(v).strip()}
    else:
        _LAYER2_CLIP_ZH_CACHE = {}
    return _LAYER2_CLIP_ZH_CACHE


def _load_all_subtasks_cache() -> dict[str, list[str]]:
    """clip_id -> ordered list of all layer-2 ZH subtasks under the same upper clip."""
    global _LAYER2_ALL_SUBTASKS_CACHE
    if _LAYER2_ALL_SUBTASKS_CACHE is not None:
        return _LAYER2_ALL_SUBTASKS_CACHE
    cache_path = Path(__file__).resolve().parents[1] / "data" / "layer2_all_subtasks_by_clip.json"
    out: dict[str, list[str]] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        for cid, payload in raw.items():
            if isinstance(payload, dict):
                subs = payload.get("all_subtasks_zh") or []
            elif isinstance(payload, list):
                subs = payload
            else:
                subs = []
            cleaned = [str(s).strip() for s in subs if str(s).strip()]
            if cleaned:
                out[str(cid)] = cleaned
    _LAYER2_ALL_SUBTASKS_CACHE = out
    return _LAYER2_ALL_SUBTASKS_CACHE


def _format_all_subtasks(subtasks: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, 1))


def _build_progress_memory(all_subtasks: list[str], current_idx: int) -> str:
    """Language memory is the last completed layer-2 subtask."""
    if current_idx <= 0:
        return DEFAULT_MEMORY
    return all_subtasks[current_idx - 1]


def _next_subtask(all_subtasks: list[str], current_idx: int) -> str:
    if 0 <= current_idx + 1 < len(all_subtasks):
        return all_subtasks[current_idx + 1]
    return LAST_MEMORY


def _to_chinese_layer2(text: str, *, clip_id: str | None = None) -> str:
    text = text.strip()
    if not text:
        return "执行操作任务。"
    if clip_id:
        zh = _load_layer2_clip_zh_cache().get(str(clip_id))
        if zh:
            return zh
    if _has_chinese(text):
        return text
    cache = _load_layer2_zh_cache()
    zh = cache.get(text)
    if zh:
        return zh
    return text


def _infer_skill(task: str) -> str:
    lower = task.lower()
    # Prefer Chinese lexical cues first (avoid bare「移」which appears in many place/move phrases).
    if (
        any(tok in task for tok in ("抓取", "拿起", "拿取", "握住", "握紧", "握持", "拾取", "夹取", "夹持"))
        or task.startswith("握")
        or any(tok in lower for tok in ("pick", "grasp", "grab"))
    ):
        return SKILL_ZH["Pick"]
    if any(tok in task for tok in ("放置", "放回", "放入", "归位", "安装", "插入", "放下")) or any(
        tok in lower for tok in ("place", "put", "insert", "install")
    ):
        return SKILL_ZH["Place"]
    if any(tok in task for tok in ("推动", "推开", "推移")) or any(
        tok in lower for tok in ("push", "slide")
    ):
        return SKILL_ZH["Push"]
    if task.startswith("移") or "移动" in task or "move" in lower:
        return SKILL_ZH["Push"]
    return SKILL_ZH["Manipulate"]


def _layer2_subtask(row_dict: dict[str, Any]) -> str:
    """Official layer-2 Chinese from clip_description; fall back to EN->ZH cache / task."""
    clip_id = row_dict.get("clip_id")
    clip_id_str = str(clip_id).strip() if clip_id is not None and not (isinstance(clip_id, float) and pd.isna(clip_id)) else None
    if clip_id_str:
        zh = _load_layer2_clip_zh_cache().get(clip_id_str)
        if zh:
            return zh
    for key in ("text_embedding_content", "subtask", "task"):
        value = row_dict.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            return _to_chinese_layer2(text, clip_id=clip_id_str)
    return "执行操作任务。"


def _build_assistant(
    subtask: str, all_subtasks: list[str], clip_length: int, memory: str, next_memory: str
) -> str:
    instruction = _task_to_instruction(subtask)
    skill = _infer_skill(subtask)
    return (
        f"所有子任务：\n{_format_all_subtasks(all_subtasks)}\n"
        f"技能：{skill}\n"
        f"当前子任务：{instruction}\n"
        f"上一个子任务：{memory}\n"
        f"下一个子任务：{next_memory}"
    )


def _build_sample(
    image_path: Path,
    task: str,
    subtask: str,
    clip_length: int,
    *,
    memory: str | None = None,
    all_subtasks: list[str] | None = None,
    clip_id: str | None = None,
    current_idx: int | None = None,
) -> dict[str, Any]:
    task_instruction = _task_to_instruction(task)
    subtask_zh = _task_to_instruction(subtask)
    if all_subtasks is None and clip_id is not None:
        cache_path = Path(__file__).resolve().parents[1] / "data" / "layer2_all_subtasks_by_clip.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            payload = raw.get(str(clip_id))
            if isinstance(payload, dict):
                siblings = [str(x) for x in (payload.get("sibling_clip_ids") or [])]
                zh_list = [str(x) for x in (payload.get("all_subtasks_zh") or []) if str(x).strip()]
                if zh_list:
                    all_subtasks = zh_list
                if current_idx is None and siblings and str(clip_id) in siblings:
                    current_idx = siblings.index(str(clip_id))
        if all_subtasks is None:
            all_subtasks = _load_all_subtasks_cache().get(str(clip_id))
    if not all_subtasks:
        all_subtasks = [subtask_zh]
    if memory is None:
        if current_idx is None:
            try:
                current_idx = all_subtasks.index(subtask_zh)
            except ValueError:
                current_idx = 0
        memory = _build_progress_memory(all_subtasks, current_idx)
    if current_idx is None:
        try:
            current_idx = all_subtasks.index(subtask_zh)
        except ValueError:
            current_idx = 0
    next_memory = _next_subtask(all_subtasks, current_idx)
    user_text = (
        f"任务：{task_instruction}\n\n"
        f"上一个子任务：\n{memory}\n\n"
        "请输出全部子任务，以及当前技能、当前子任务、上一个子任务与下一个子任务。"
    )
    assistant = _build_assistant(subtask_zh, all_subtasks, clip_length, memory, next_memory)
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
        subtask=_layer2_subtask(row_dict),
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
        return _build_sample(
            image_path, job.task, job.subtask, clip_length, clip_id=job.clip_id
        )

    zarr_path = Path(job.zarr_path)
    frame_idx = job.start_idx + clip_length // 2
    try:
        group, camera = _get_zarr_group_camera(zarr_path, job.camera)
        _export_frame(group, camera, frame_idx, image_path)
    except Exception:
        return None
    return _build_sample(
        image_path, job.task, job.subtask, clip_length, clip_id=job.clip_id
    )


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

    jsonl_path = output_dir / f"hermas_sys2_{split}.jsonl"
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
            ("hermas_sys2_train", "hermas_sys2_train.jsonl"),
            ("hermas_sys2_val", "hermas_sys2_val.jsonl"),
            ("hermas_sys2_train_20k", "hermas_sys2_train_20k.json"),
            ("hermas_sys2_val_2k", "hermas_sys2_val_2k.json"),
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
        args.output_dir / "hermas_sys2_train.jsonl",
        args.output_dir / "hermas_sys2_train_20k.json",
        subset_train,
    )
    jsonl_to_json_array(
        args.output_dir / "hermas_sys2_val.jsonl",
        args.output_dir / "hermas_sys2_val_2k.json",
        subset_val,
    )
    print(
        f"[done] train={train_count} val={val_count} "
        f"(jsonl + subset json; full export trains on *.jsonl directly)"
    )


if __name__ == "__main__":
    main()
