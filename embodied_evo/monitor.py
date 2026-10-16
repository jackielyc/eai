"""解析 EmbodiedEvo run 目录产物，供图形化刷新。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ArtifactFile:
    relative: str
    path: Path
    kind: str  # skill | lesson | code | json | other


@dataclass
class SceneScore:
    index: int
    score: Optional[float]
    success: bool
    valid: bool
    label: str


@dataclass
class RunSnapshot:
    run_dir: Path
    exists: bool
    phase: str = ""
    env_task: str = ""
    playground_valid: Optional[bool] = None
    success_rate: Optional[float] = None
    avg_score: Optional[float] = None
    successes: int = 0
    scenes: int = 0
    scored_scenes: int = 0
    scene_scores: List[SceneScore] = field(default_factory=list)
    skills: List[ArtifactFile] = field(default_factory=list)
    lessons: List[ArtifactFile] = field(default_factory=list)
    solution_path: Optional[Path] = None
    solution_text: str = ""
    images: List[Path] = field(default_factory=list)
    protocol: Dict[str, object] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)
    summary_lines: List[str] = field(default_factory=list)
    error: str = ""


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _list_artifacts(root: Path, kind: str) -> List[ArtifactFile]:
    if not root.is_dir():
        return []
    out: List[ArtifactFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        try:
            rel = str(path.relative_to(root.parent))
        except ValueError:
            rel = path.name
        out.append(ArtifactFile(relative=rel, path=path, kind=kind))
    return out


def _find_harness_dirs(run: Path) -> Tuple[Optional[Path], Optional[Path]]:
    candidates = [
        run / "frozen_harness",
        run / "playground" / "workspace",
        run / "workspace",
    ]
    for base in candidates:
        skills = base / "skills"
        lessons = base / "lessons"
        if skills.is_dir() or lessons.is_dir():
            return (
                skills if skills.is_dir() else None,
                lessons if lessons.is_dir() else None,
            )
    return None, None


def _find_solution(run: Path) -> Optional[Path]:
    for path in (
        run / "playground" / "workspace" / "submission" / "solution.py",
        run / "workspace" / "submission" / "solution.py",
        run / "inputs" / "submission" / "solution.py",
    ):
        if path.is_file():
            return path
    # latest test scene
    test = run / "test"
    if test.is_dir():
        scenes = sorted(
            [p for p in test.iterdir() if p.is_dir() and p.name.startswith("scene")],
            reverse=True,
        )
        for scene in scenes:
            cand = scene / "workspace" / "submission" / "solution.py"
            if cand.is_file():
                return cand
    return None


def _collect_images(run: Path, limit: int = 48) -> List[Path]:
    roots = [
        run / "smoke" / "workspace" / "observations",
        run / "playground" / "workspace" / "observations",
        run / "workspace" / "observations",
        run / "smoke" / "raw_observations",
        run / "playground",
        run / "test",
        run / "smoke",
    ]
    found: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
                continue
            # Prefer agent-facing current_cam_*; skip raw duplicates when possible
            name = path.name.lower()
            if name.startswith("raw_cam_"):
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    # Newest first so the live preview follows the latest smoke step
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:limit]


def _scene_scores_from_report(report: Dict[str, object]) -> List[SceneScore]:
    results = report.get("results")
    if not isinstance(results, list):
        return []
    out: List[SceneScore] = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        score_f: Optional[float]
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score_f = float(score)
        else:
            score_f = None
        success = bool(item.get("task_completed"))
        valid = bool(item.get("valid"))
        label = f"scene{i:02d}"
        if score_f is not None:
            label = f"{label}  {score_f:.1f}"
        if success:
            label += " ✓"
        elif not valid:
            label += " ✗"
        out.append(
            SceneScore(
                index=i,
                score=score_f,
                success=success,
                valid=valid,
                label=label,
            )
        )
    return out


def snapshot_run(run_dir: str | Path) -> RunSnapshot:
    run = Path(run_dir).expanduser().resolve()
    snap = RunSnapshot(run_dir=run, exists=run.is_dir())
    if not snap.exists:
        snap.error = "run 目录不存在"
        snap.summary_lines = [snap.error]
        return snap

    protocol = _read_json(run / "protocol.json")
    resolved = _read_json(run / "resolved.json")
    report = _read_json(run / "test.json")
    smoke = _read_json(run / "smoke.json")
    snap.protocol = protocol
    snap.report = report or smoke
    snap.phase = str(protocol.get("phase") or smoke.get("phase") or "")
    if isinstance(resolved.get("simulator"), dict):
        snap.env_task = str(resolved["simulator"].get("env_task") or "")
    if not snap.env_task and isinstance(protocol.get("definition"), dict):
        cfg = protocol["definition"].get("config")
        if isinstance(cfg, dict) and isinstance(cfg.get("simulator"), dict):
            snap.env_task = str(cfg["simulator"].get("env_task") or "")
    if not snap.env_task and smoke.get("env_task"):
        snap.env_task = str(smoke.get("env_task") or "")

    if "playground_valid" in protocol:
        snap.playground_valid = bool(protocol.get("playground_valid"))
    elif smoke:
        snap.playground_valid = bool(smoke.get("passed"))

    if report:
        sr = report.get("success_rate")
        if isinstance(sr, (int, float)) and not isinstance(sr, bool):
            snap.success_rate = float(sr)
        score = report.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            snap.avg_score = float(score)
        snap.successes = int(report.get("successes") or 0)
        snap.scenes = int(report.get("scenes") or 0)
        snap.scored_scenes = int(report.get("scored_scenes") or 0)
        snap.scene_scores = _scene_scores_from_report(report)

    skills_dir, lessons_dir = _find_harness_dirs(run)
    if skills_dir is not None:
        snap.skills = _list_artifacts(skills_dir, "skill")
    if lessons_dir is not None:
        snap.lessons = _list_artifacts(lessons_dir, "lesson")

    sol = _find_solution(run)
    snap.solution_path = sol
    if sol is not None:
        try:
            snap.solution_text = sol.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            snap.solution_text = f"# read error: {exc}"

    snap.images = _collect_images(run)

    lines = [
        f"run: {run}",
        f"phase: {snap.phase or '-'}",
        f"env_task: {snap.env_task or '-'}",
    ]
    if snap.playground_valid is not None:
        lines.append(f"playground_valid: {snap.playground_valid}")
    if smoke:
        lines.append(
            f"smoke: passed={smoke.get('passed')} kind={smoke.get('kind')} "
            f"scored={smoke.get('scored')}"
        )
    if snap.scenes:
        sr_txt = (
            f"{snap.success_rate * 100:.1f}%"
            if snap.success_rate is not None
            else "-"
        )
        score_txt = f"{snap.avg_score:.2f}" if snap.avg_score is not None else "-"
        lines.append(
            f"test: scenes={snap.scenes} successes={snap.successes} "
            f"SR={sr_txt} score={score_txt}"
        )
    lines.append(f"skills={len(snap.skills)} lessons={len(snap.lessons)} images={len(snap.images)}")
    if sol is not None:
        lines.append(f"solution: {sol}")
    snap.summary_lines = lines
    return snap


def list_runs(repo: str | Path, limit: int = 40) -> List[Path]:
    root = Path(repo).expanduser().resolve() / "runs"
    if not root.is_dir():
        return []
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_ui_task") and not p.name.startswith(".")
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]
