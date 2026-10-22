"""后台启动 FLORA 推理子进程（annotate / outcome / process / rubric）。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from flora.catalog import (
    DEFAULT_CACHE,
    DEFAULT_ROOT,
    child_env,
    has_rewarder,
    has_tempseg,
    resolve_python,
    rewarder_dir,
    tempseg_dir,
)

LogFn = Callable[[str], None]


@dataclass
class RunRequest:
    mode: str  # annotate | validate | outcome | process | rubric
    root: Path = field(default_factory=lambda: DEFAULT_ROOT)
    python: str = ""
    gpu: str = "0"
    video: str = ""
    left_wrist: str = ""
    right_wrist: str = ""
    profile: str = "human"
    yam32: bool = False
    instruction: str = ""
    task: str = ""
    robot: str = "bimanual robot"
    cameras: str = "a top (scene) camera, a left-wrist camera, a right-wrist camera"
    rubric_json: str = ""
    output: str = ""
    overwrite: bool = True
    include_audit: bool = False
    with_rationale: bool = False
    dry_run: bool = False
    num_samples: int = 1
    tensor_parallel_size: int = 1


class FloraRunner:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, req: RunRequest, log: LogFn, on_done: Callable[[int], None]) -> None:
        if self.running:
            raise RuntimeError("已有任务在运行")
        self._stop.clear()
        py = resolve_python(req.python)
        req.python = py
        argv, cwd, env = self._build(req, log)

        # Belt-and-suspenders: even if env scrub misses a key, strip VirtualGL
        # injection that causes ``Cannot load symbol cudnnGetVersion``.
        argv = [
            "/usr/bin/env",
            "-u",
            "LD_PRELOAD",
            "-u",
            "MUJOCO_USE_VGL",
            "-u",
            "MUJOCO_VGL_DEVICE",
            *argv,
        ]

        log(f"$ cd {cwd}")
        log("$ " + " ".join(argv))
        log(f"PYTHON={py}")
        log(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"FLORA_MODEL_CACHE={env.get('FLORA_MODEL_CACHE', '')}")
        if env.get("LD_LIBRARY_PATH"):
            n_libs = len([p for p in env["LD_LIBRARY_PATH"].split(os.pathsep) if p])
            log(f"nvidia_lib_dirs={n_libs} (LD_PRELOAD cleared for child)")
        if env.get("VLLM_USE_FLASHINFER_SAMPLER") == "0":
            log("VLLM_USE_FLASHINFER_SAMPLER=0 (Blackwell/cu128 FlashInfer JIT workaround)")

        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._proc = proc

        def _pump() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    if self._stop.is_set():
                        break
                    log(line.rstrip("\n"))
            except Exception as exc:  # noqa: BLE001
                log(f"[reader] {type(exc).__name__}: {exc}")
            code = proc.wait()
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            on_done(code)

        self._thread = threading.Thread(target=_pump, name="flora-runner", daemon=True)
        self._thread.start()

    def stop(self, log: Optional[LogFn] = None) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if log:
            log("正在停止 FLORA 进程组…")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 15
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _build(
        self, req: RunRequest, log: LogFn
    ) -> tuple[List[str], str, dict]:
        root = Path(req.root or DEFAULT_ROOT)
        env = child_env(python=req.python, gpu=req.gpu, root=root)
        Path(DEFAULT_CACHE).mkdir(parents=True, exist_ok=True)

        if req.mode in ("annotate", "validate"):
            return self._annotate_argv(req, root, env, log)
        if req.mode in ("outcome", "process", "rubric"):
            return self._reward_argv(req, root, env, log)
        raise RuntimeError(f"未知模式: {req.mode}")

    def _annotate_argv(
        self, req: RunRequest, root: Path, env: dict, log: LogFn
    ) -> tuple[List[str], str, dict]:
        if not has_tempseg(root):
            raise RuntimeError(f"未找到 TempSeg 检出，请先下载模型: {tempseg_dir(root)}")
        video = Path(req.video).expanduser()
        if not video.is_file():
            raise RuntimeError(f"视频不存在: {video}")
        out = Path(req.output).expanduser() if req.output else Path("annotations.json")
        if not out.is_absolute():
            out = (root / "outputs" / out.name).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cwd = str(tempseg_dir(root))
        argv = [
            req.python,
            "infer.py",
            "--video",
            str(video.resolve()),
            "--profile",
            req.profile or "human",
            "--output",
            str(out),
        ]
        if req.left_wrist.strip():
            argv.extend(["--left-wrist", str(Path(req.left_wrist).expanduser().resolve())])
        if req.right_wrist.strip():
            argv.extend(["--right-wrist", str(Path(req.right_wrist).expanduser().resolve())])
        if req.yam32:
            argv.append("--yam32")
        if req.instruction.strip():
            argv.extend(["--instruction", req.instruction.strip()])
        if req.mode == "validate":
            argv.append("--validate-only")
        if req.include_audit and req.mode == "annotate":
            argv.append("--include-audit")
        if req.overwrite:
            argv.append("--overwrite")
        log(f"annotate output → {out}")
        return argv, cwd, env

    def _reward_argv(
        self, req: RunRequest, root: Path, env: dict, log: LogFn
    ) -> tuple[List[str], str, dict]:
        if not has_rewarder(root):
            raise RuntimeError(f"未找到 Rewarder 检出，请先下载模型: {rewarder_dir(root)}")
        videos = [Path(req.video).expanduser()]
        if req.left_wrist.strip():
            videos.append(Path(req.left_wrist).expanduser())
        if req.right_wrist.strip():
            videos.append(Path(req.right_wrist).expanduser())
        for v in videos:
            if not v.is_file():
                raise RuntimeError(f"视频不存在: {v}")
        task = (req.task or req.instruction or "").strip()
        if not task and not req.dry_run:
            raise RuntimeError("结果/过程/评分规则奖励需要填写任务指令（task）")

        out_name = {
            "outcome": "terminal.jsonl",
            "process": "dense.jsonl",
            "rubric": "rubric.jsonl",
        }[req.mode]
        out = Path(req.output).expanduser() if req.output else Path(out_name)
        if not out.is_absolute():
            out = (root / "outputs" / out.name).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        module = {
            "outcome": "batch_utils.terminal_reward",
            "process": "batch_utils.dense_reward",
            "rubric": "batch_utils.rubric_reward",
        }[req.mode]
        cwd = str(rewarder_dir(root))
        argv = [
            req.python,
            "-m",
            module,
            "--video",
            *[str(v.resolve()) for v in videos],
            "--task",
            task or "(dry-run placeholder)",
            "--output",
            str(out),
            "--tensor-parallel-size",
            str(max(1, int(req.tensor_parallel_size))),
        ]
        if req.overwrite:
            argv.append("--overwrite")
        if req.dry_run:
            argv.append("--dry-run")
        if req.mode == "outcome" and req.with_rationale:
            argv.append("--with-rationale")
        if req.mode == "rubric":
            # rubric 需要 JSONL（含 robot/cameras/rubric）；写临时 jobs
            jobs = out.parent / f"_rubric_job_{int(time.time())}.jsonl"
            cams = [c.strip() for c in (req.cameras or "").split(",") if c.strip()]
            if not cams:
                cams = [f"camera_{i}" for i in range(len(videos))]
            while len(cams) < len(videos):
                cams.append(f"camera_{len(cams)}")
            try:
                rubric = json.loads(req.rubric_json or "[]")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"评分规则 JSON 无效: {exc}") from exc
            if not isinstance(rubric, list) or not rubric:
                raise RuntimeError("评分规则不能为空（需要 steps 列表）")
            record = {
                "id": "episode",
                "task": task or "(dry-run placeholder)",
                "robot": (req.robot or "bimanual robot").strip(),
                "cameras": cams[: len(videos)],
                "videos": [str(v.resolve()) for v in videos],
                "rubric": rubric,
            }
            jobs.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            argv = [
                req.python,
                "-m",
                module,
                "--jobs",
                str(jobs),
                "--output",
                str(out),
                "--tensor-parallel-size",
                str(max(1, int(req.tensor_parallel_size))),
                "--num-samples",
                str(max(1, int(req.num_samples))),
            ]
            if req.overwrite:
                argv.append("--overwrite")
            if req.dry_run:
                argv.append("--dry-run")
            log(f"rubric jobs → {jobs}")
        log(f"reward output → {out}")
        return argv, cwd, env
