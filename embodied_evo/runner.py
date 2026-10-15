"""后台启动子进程执行 EmbodiedEvo explore / test / run。"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from embodied_evo.catalog import host_deps_ok, resolve_cli, resolve_python, write_ui_task_override


LogFn = Callable[[str], None]


@dataclass
class RunRequest:
    repo: Path
    command: str  # explore | test | run | smoke
    env_task: str
    run_dir: Path
    gpu: str = "0"
    test_scenes: int = 5
    test_gpus: str = ""
    test_gpu_slots: int = 1
    timeout_sec: Optional[float] = None
    baseline: bool = False
    use_recipe: bool = True
    model: str = "gpt-6-astra"
    reasoning: str = "xhigh"
    playground_budget: int = 100
    test_budget: int = 100
    python: str = ""
    host_client: bool = True  # smoke 默认无模型 host client


class EmbodiedEvoRunner:
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
        # 强制优先 miniconda3/envs/embodiedevo，避免 GUI 的 ros-humble 缺 gymnasium
        py = resolve_python(req.repo)
        ok, msg = host_deps_ok(py, req.repo)
        if not ok:
            raise RuntimeError(msg + "；请先点「安装 host 依赖」")
        req.python = py
        cwd = str(req.repo.resolve())
        env = _clean_child_env(req.repo, python=py)

        if req.command == "smoke":
            if req.run_dir.exists() and any(req.run_dir.iterdir()):
                raise RuntimeError(f"smoke 需要空的 run-dir: {req.run_dir}")
            env["EMBODIEDEVO_USE_RECIPE"] = "0"
            argv = [
                py,
                "-m",
                "scripts.smoke_session",
                "--task",
                "RoboDojo",
                "--env-task",
                req.env_task,
                "--run-dir",
                str(req.run_dir),
                "--gpu",
                str(req.gpu),
            ]
            if req.host_client:
                argv.append("--host-client")
            log(f"$ cd {cwd}")
            log("$ " + " ".join(argv))
            log("smoke: 无 API / 固定脚本客户端（--host-client）")
            log(f"PYTHONPATH={env.get('PYTHONPATH', '')}")
        else:
            _, prefix = resolve_cli(req.repo, py)
            task_path = write_ui_task_override(
                req.repo,
                baseline=req.baseline,
                env_task=req.env_task,
                model=req.model,
                reasoning=req.reasoning,
                playground_budget=req.playground_budget,
                test_scenes=req.test_scenes,
                test_budget=req.test_budget,
                isolation="host",
            )
            req.run_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                *prefix,
                req.command,
                "--task",
                str(task_path),
                "--env-task",
                req.env_task,
                "--run-dir",
                str(req.run_dir),
                "--gpu",
                str(req.gpu),
                "--test-scenes",
                str(int(req.test_scenes)),
                "--test-gpu-slots",
                str(int(req.test_gpu_slots)),
            ]
            if req.test_gpus.strip():
                argv.extend(["--test-gpus", req.test_gpus.strip()])
            if req.timeout_sec is not None and req.command in {"explore", "run"}:
                argv.extend(["--timeout-sec", str(float(req.timeout_sec))])

            env["EMBODIEDEVO_USE_RECIPE"] = "1" if req.use_recipe else "0"
            log(f"$ cd {cwd}")
            log("$ " + " ".join(argv))
            log(f"临时 task: {task_path}")
            log("agent isolation=host (本机 Codex，不用 Docker 镜像)")
            log(f"USE_RECIPE={env['EMBODIEDEVO_USE_RECIPE']}")
            log(f"PYTHONPATH={env.get('PYTHONPATH', '')}")

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

        self._thread = threading.Thread(target=_pump, name="embodiedevo-runner", daemon=True)
        self._thread.start()

    def stop(self, log: Optional[LogFn] = None) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if log:
            log("正在停止 EmbodiedEvo 进程组…")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 8
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _clean_child_env(repo: Path, *, python: str) -> dict:
    """子进程环境：丢掉 GUI/ros-humble 的 PYTHONPATH，避免串用错误 numpy。"""
    env = os.environ.copy()
    root = str(Path(repo).resolve())
    py_home = str(Path(python).resolve().parent.parent)

    # 这些路径会把 ros-humble 的 site-packages 强塞给 3.12，导致 numpy ABI 崩掉
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONNOUSERSITE",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_PYTHON_EXE",
        "CONDA_SHLVL",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    ):
        env.pop(key, None)

    # 仅保留仓库根，便于 -m orchestrator；绝不继承父进程 site-packages
    env["PYTHONPATH"] = root
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["EMBODIEDEVO_REPO"] = root
    # 显式指向 conda env，避免继承 ros-humble 的 CONDA_PREFIX
    if (Path(py_home) / "bin" / "python").exists() or (
        Path(python).parent.parent / "lib"
    ).is_dir():
        env["VIRTUAL_ENV"] = str(Path(python).parent.parent)
        # PATH 把目标 env 的 bin 放到最前
        bin_dir = str(Path(python).parent)
        path = env.get("PATH", "")
        parts = [p for p in path.split(os.pathsep) if p and p != bin_dir]
        env["PATH"] = os.pathsep.join([bin_dir, *parts])

    return env


def suggest_run_dir(
    repo: Path, env_task: str, *, baseline: bool = False, smoke: bool = False
) -> Path:
    stamp = time.strftime("%Y%m%d%H%M")
    suffix = f"{stamp}_{env_task}"
    if baseline:
        suffix += "_baseline"
    if smoke:
        suffix += "_host_smoke"
    return (repo / "runs" / suffix).resolve()


def readiness_command(repo: Path, python: str = "") -> List[str]:
    cwd, prefix = resolve_cli(repo, python or None)
    return ["bash", "-lc", f"cd {cwd!s} && {' '.join(prefix)} --help"]
