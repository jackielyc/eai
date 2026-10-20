"""解析 ASPIRE / cap-harness run 产物，供图形化刷新。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".webm", ".mkv"}


@dataclass
class RunSnapshot:
    run_dir: Path
    exists: bool
    status: str = ""
    benchmark: str = ""
    suite: str = ""
    task: str = ""
    seed: Optional[int] = None
    success: Optional[bool] = None
    protocol_success: Optional[bool] = None
    reward: Optional[float] = None
    steps: Optional[int] = None
    termination: str = ""
    summary_lines: List[str] = field(default_factory=list)
    images: List[Path] = field(default_factory=list)
    videos: List[Path] = field(default_factory=list)
    run_json: Dict[str, object] = field(default_factory=dict)
    outcome_json: Dict[str, object] = field(default_factory=dict)
    program_text: str = ""
    program_path: Optional[Path] = None
    error: str = ""


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _collect_media(run: Path, limit: int = 48) -> tuple[List[Path], List[Path]]:
    images: List[Path] = []
    videos: List[Path] = []
    roots = [
        run / "media" / "keyframes",
        run / "media" / "overlays",
        run / "media" / "videos",
        run / "media",
    ]
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if "partial" in path.name:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            ext = path.suffix.lower()
            if ext in _IMAGE_EXTS:
                images.append(path)
            elif ext in _VIDEO_EXTS:
                videos.append(path)
            if len(images) + len(videos) >= limit:
                return images, videos
    return images, videos


def _find_program(run: Path) -> Optional[Path]:
    for path in (
        run / "source" / "program.py",
        run / "source" / "policy.py",
        run / "source" / "generated.py",
    ):
        if path.is_file():
            return path
    src = run / "source"
    if src.is_dir():
        for path in sorted(src.glob("*.py")):
            return path
    return None


def snapshot_run(run_dir: Path | str) -> RunSnapshot:
    run = Path(run_dir).expanduser()
    snap = RunSnapshot(run_dir=run, exists=run.is_dir())
    if not snap.exists:
        snap.error = "run 目录不存在"
        return snap

    run_json = _read_json(run / "run.json")
    outcome = _read_json(run / "outcome.json")
    snap.run_json = run_json
    snap.outcome_json = outcome

    snap.status = str(run_json.get("status") or outcome.get("status") or "")
    task = run_json.get("task") if isinstance(run_json.get("task"), dict) else {}
    snap.benchmark = str(
        run_json.get("benchmark") or task.get("benchmark") or ""
    )
    snap.suite = str(run_json.get("suite") or task.get("suite") or "")
    snap.task = str(
        task.get("name")
        or task.get("slug")
        or run_json.get("task_id")
        or ""
    )
    seed = run_json.get("seed", task.get("seed"))
    try:
        snap.seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        snap.seed = None

    if "task_success" in outcome:
        snap.success = bool(outcome.get("task_success"))
    elif "success" in outcome:
        snap.success = bool(outcome.get("success"))
    if "protocol_success" in outcome:
        snap.protocol_success = bool(outcome.get("protocol_success"))
    reward = outcome.get("reward", outcome.get("aggregate_reward"))
    try:
        snap.reward = float(reward) if reward is not None else None
    except (TypeError, ValueError):
        snap.reward = None
    steps = outcome.get("steps", outcome.get("step_count"))
    try:
        snap.steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        snap.steps = None
    snap.termination = str(outcome.get("termination") or outcome.get("reason") or "")

    prog = _find_program(run)
    snap.program_path = prog
    if prog is not None:
        try:
            snap.program_text = prog.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            snap.error = f"读 program 失败: {exc}"

    snap.images, snap.videos = _collect_media(run)

    lines = [
        f"status: {snap.status or '-'}",
        f"benchmark: {snap.benchmark or '-'}  suite: {snap.suite or '-'}",
        f"task: {snap.task or '-'}  seed: {snap.seed if snap.seed is not None else '-'}",
    ]
    if snap.success is not None:
        lines.append(f"task_success: {snap.success}")
    if snap.protocol_success is not None:
        lines.append(f"protocol_success: {snap.protocol_success}")
    if snap.reward is not None:
        lines.append(f"reward: {snap.reward:.4f}")
    if snap.steps is not None:
        lines.append(f"steps: {snap.steps}")
    if snap.termination:
        lines.append(f"termination: {snap.termination}")
    lines.append(f"images: {len(snap.images)}  videos: {len(snap.videos)}")
    snap.summary_lines = lines
    return snap


def list_recent_runs(output_root: Path | str, limit: int = 40) -> List[Path]:
    root = Path(output_root).expanduser()
    if not root.is_dir():
        return []
    runs: List[Path] = []
    for path in root.rglob("run.json"):
        runs.append(path.parent)
    runs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return runs[:limit]
