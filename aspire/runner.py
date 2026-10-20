"""后台启动子进程执行 ASPIRE doctor / validate / run / bootstrap。"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from aspire.catalog import (
    child_env,
    find_uv,
    host_deps_ok,
    resolve_cap_harness,
    resolve_python,
    resolve_runtime_python,
)

LogFn = Callable[[str], None]


@dataclass
class RunRequest:
    repo: Path
    command: str  # doctor | validate | run | bootstrap_libero | bootstrap_robosuite
    python: str = ""
    gpu: str = "0"
    # doctor
    doctor_providers: str = ""
    doctor_reset: bool = False
    doctor_output: str = "validation-artifacts/ui/doctor.json"
    # validate
    plan: str = ""
    validate_output_dir: str = "validation-artifacts/ui"
    # run
    benchmark: str = "libero-pro"
    suite: str = "libero_object_swap"
    task_id: int = 0
    seed: int = 1
    program: str = "examples/libero/object_swap_seed1.py"
    output_root: str = "outputs/ui"
    max_steps: int = 2000
    camera_width: int = 800
    camera_height: int = 512
    init_mode: str = "saved"
    no_videos: bool = False


class AspireRunner:
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
        repo = req.repo.resolve()
        if req.command.startswith("bootstrap_"):
            py = resolve_python(repo)
            req.python = py
            argv, env = self._bootstrap_argv(req, repo, log)
        elif req.command == "run":
            py = resolve_runtime_python(repo, req.benchmark)
            req.python = py
            ok, msg = host_deps_ok(py, repo)
            if not ok and not (repo / ".venv-libero" / "bin" / "python").is_file():
                # runtime venv 可能有完整依赖但 host_deps 用不同检查；仍允许 .venv-* 直接跑
                if Path(py).parent.parent.name.startswith(".venv"):
                    log(f"提示: host 检查未通过（{msg}），仍使用 runtime venv: {py}")
                else:
                    raise RuntimeError(msg + "；请先点「安装 host 依赖」或 bootstrap 对应 backend")
            argv, env = self._run_argv(req, repo, py, log)
        else:
            py = resolve_python(repo)
            req.python = py
            ok, msg = host_deps_ok(py, repo)
            if not ok:
                raise RuntimeError(msg + "；请先点「安装 host 依赖」")
            if req.command == "doctor":
                argv, env = self._doctor_argv(req, repo, py, log)
            elif req.command == "validate":
                argv, env = self._validate_argv(req, repo, py, log)
            else:
                raise RuntimeError(f"未知命令: {req.command}")

        cwd = str(repo)
        log(f"$ cd {cwd}")
        log("$ " + " ".join(argv))
        log(f"PYTHON={py}")
        log(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
        log(f"CAP_HARNESS_VENV_ROOT={env.get('CAP_HARNESS_VENV_ROOT', '')}")

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

        self._thread = threading.Thread(target=_pump, name="aspire-runner", daemon=True)
        self._thread.start()

    def stop(self, log: Optional[LogFn] = None) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if log:
            log("正在停止 ASPIRE 进程组…")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 10
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _doctor_argv(
        self, req: RunRequest, repo: Path, py: str, log: LogFn
    ) -> tuple[List[str], dict]:
        env = child_env(repo, python=py, gpu=req.gpu)
        prefix = resolve_cap_harness(py, repo)
        out = Path(req.doctor_output)
        if not out.is_absolute():
            out = repo / out
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [*prefix, "doctor", "--output", str(out)]
        if req.doctor_reset:
            argv.append("--reset")
        providers = req.doctor_providers.strip()
        if providers:
            argv.extend(["--providers", providers])
        log(f"doctor output → {out}")
        return argv, env

    def _validate_argv(
        self, req: RunRequest, repo: Path, py: str, log: LogFn
    ) -> tuple[List[str], dict]:
        env = child_env(repo, python=py, gpu=req.gpu)
        prefix = resolve_cap_harness(py, repo)
        plan = Path(req.plan)
        if not plan.is_absolute():
            plan = repo / plan
        if not plan.is_file():
            raise RuntimeError(f"validation plan 不存在: {plan}")
        out_dir = Path(req.validate_output_dir)
        if not out_dir.is_absolute():
            out_dir = repo / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            *prefix,
            "validate",
            "--plan",
            str(plan),
            "--output-dir",
            str(out_dir),
        ]
        log(f"plan → {plan}")
        log(f"output-dir → {out_dir}")
        return argv, env

    def _run_argv(
        self, req: RunRequest, repo: Path, py: str, log: LogFn
    ) -> tuple[List[str], dict]:
        env = child_env(repo, python=py, gpu=req.gpu)
        prefix = resolve_cap_harness(py, repo)
        program = Path(req.program)
        if not program.is_absolute():
            program = repo / program
        if not program.is_file():
            raise RuntimeError(f"program 不存在: {program}")
        out_root = Path(req.output_root)
        if not out_root.is_absolute():
            out_root = repo / out_root
        out_root.mkdir(parents=True, exist_ok=True)
        argv = [
            *prefix,
            "run",
            "--benchmark",
            req.benchmark,
            "--suite",
            req.suite,
            "--task-id",
            str(int(req.task_id)),
            "--seed",
            str(int(req.seed)),
            "--program",
            str(program),
            "--output-root",
            str(out_root),
            "--max-steps",
            str(int(req.max_steps)),
            "--camera-width",
            str(int(req.camera_width)),
            "--camera-height",
            str(int(req.camera_height)),
            "--init-mode",
            req.init_mode,
        ]
        if req.no_videos:
            argv.append("--no-videos")
        log(f"program → {program}")
        log(f"output-root → {out_root}")
        return argv, env

    def _bootstrap_argv(
        self, req: RunRequest, repo: Path, log: LogFn
    ) -> tuple[List[str], dict]:
        script_name = {
            "bootstrap_libero": "bootstrap_libero.sh",
            "bootstrap_robosuite": "bootstrap_robosuite.sh",
        }.get(req.command)
        if not script_name:
            raise RuntimeError(f"未知 bootstrap: {req.command}")
        script = repo / "scripts" / script_name
        if not script.is_file():
            raise RuntimeError(f"缺少脚本: {script}")
        uv = find_uv()
        if not uv:
            raise RuntimeError(
                "bootstrap 需要 uv。请先: curl -LsSf https://astral.sh/uv/install.sh | sh"
            )
        py = req.python or resolve_python(repo)
        env = child_env(repo, python=py, gpu=req.gpu)
        # 确保 uv 在 PATH
        uv_dir = str(Path(uv).parent)
        env["PATH"] = uv_dir + os.pathsep + env.get("PATH", "")
        env["CAP_HARNESS_PYTHON"] = py
        argv = ["bash", str(script), "--python", py]
        log(f"bootstrap 将把 .venv-* 字节放到 CAP_HARNESS_VENV_ROOT={env['CAP_HARNESS_VENV_ROOT']}")
        log(f"uv={uv}")
        return argv, env
