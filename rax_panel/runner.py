"""后台启动 RAX Operator Dashboard，并解析带 token 的本机 URL。"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from rax_panel.catalog import link_runtime_env

LogFn = Callable[[str], None]
_URL_RE = re.compile(r"https?://127\.0\.0\.1:\d+/#token=\S+")


def env_prefix(python: Path) -> Path:
    """venv/conda 前缀。不要 resolve python，避免落到 bootstrap 解释器。"""
    return Path(python).parent.parent


def child_env(project: Path, python: Path) -> dict:
    """丢掉 GUI 进程的 PYTHONPATH，并让 RAX 脚本使用 miniconda3 环境。"""
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_PYTHON_EXE",
        "CONDA_SHLVL",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    ):
        env.pop(key, None)
    root = str(Path(project))
    prefix = env_prefix(python)
    env["PYTHONPATH"] = root
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["VIRTUAL_ENV"] = str(prefix)
    env["ROBODOJO_ENV_ROOT"] = str(prefix)
    env["ROBODOJO_PYTHON"] = str(prefix / "bin" / "python")
    env["ROBODOJO_RUNTIME_ROOT"] = str(Path(project) / ".local-runtime")
    env["OMNI_KIT_ACCEPT_EULA"] = env.get("OMNI_KIT_ACCEPT_EULA", "Y")
    bin_dir = str(prefix / "bin")
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and p != bin_dir]
    env["PATH"] = os.pathsep.join([bin_dir, *parts])
    return env


class DashboardRunner:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.url = ""

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        project: Path,
        python: Path,
        port: int,
        log: LogFn,
        on_url: Callable[[str], None],
        on_done: Callable[[int], None],
    ) -> None:
        if self.running:
            raise RuntimeError("操作台已在运行")
        ok, msg = link_runtime_env(project, env_prefix(python))
        log(msg)
        if not ok:
            raise RuntimeError(msg)
        linked = project / "runtime" / "envs" / "robodojo" / "bin" / "python"
        if not linked.is_file():
            raise RuntimeError(f"链接后仍找不到 {linked}")
        argv = [
            str(python),
            "-m",
            "services.dashboard.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port)),
            "--project-root",
            str(project),
        ]
        log("$ cd " + str(project))
        log("$ " + " ".join(argv))
        proc = subprocess.Popen(
            argv,
            cwd=str(project),
            env=child_env(project, python),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._proc = proc
            self.url = ""

        def _read() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    text = line.rstrip("\n")
                    log(text)
                    found = _URL_RE.search(text)
                    if found and not self.url:
                        self.url = found.group(0)
                        on_url(self.url)
            finally:
                code = proc.wait()
                on_done(code)

        self._thread = threading.Thread(target=_read, name="rax-dashboard", daemon=True)
        self._thread.start()

    def stop(self, log: Optional[LogFn] = None) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if log is not None:
            log("正在停止 RAX 操作台…")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
