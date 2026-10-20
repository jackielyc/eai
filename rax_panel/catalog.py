"""RAX 运行目录、miniconda3 环境，以及 runtime/envs/robodojo 链接。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_PERSONAL = Path("/share_data/projects/mahjong/share/personal/liyichao")
MINICONDA3 = Path(
    os.environ.get("EAI_MINICONDA3", str(_PERSONAL / "miniconda3"))
)
ENV_NAME = os.environ.get("RAX_CONDA_ENV", "rax")
DEFAULT_INDEX_REPO = Path(os.environ.get("RAX_REPO", str(_PERSONAL / "rax")))
DEFAULT_RUN_ROOT = Path(
    os.environ.get("RAX_RUN_ROOT", str(_PERSONAL / "rax-robodojo-native"))
)
NATIVE_REF = "origin/robodojo-behavior/native"
LOCAL_BRANCH = "rax-robodojo-native"
DASHBOARD_REL = Path("scripts") / "run_operator_dashboard.sh"
PANEL_PACKAGES = (
    "numpy==1.26.0",
    "scipy==1.15.3",
    "pillow",
    "pyyaml",
    "jinja2",
    "h5py",
    "imageio",
    "imageio-ffmpeg",
    "msgpack",
    "omegaconf",
)

LogFn = Callable[[str], None]


def conda_env_prefix() -> Path:
    return MINICONDA3 / "envs" / ENV_NAME


def conda_python() -> Path:
    override = os.environ.get("RAX_PYTHON", "").strip()
    if override:
        return Path(override)
    return conda_env_prefix() / "bin" / "python"


def _emit(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)


def has_dashboard(root: Path) -> bool:
    return (root / DASHBOARD_REL).is_file()


def _git(repo: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ensure_native_checkout(
    index_repo: Path,
    dest: Path,
    *,
    log: Optional[LogFn] = None,
) -> Tuple[bool, str]:
    """准备可运行的 robodojo-behavior/native 检出。

    main 只是索引。已有 dashboard 脚本时直接用；否则从索引仓库加 worktree。
    """
    index_repo = Path(index_repo)
    dest = Path(dest)
    if has_dashboard(dest):
        return True, str(dest)
    if has_dashboard(index_repo):
        return True, str(index_repo.resolve())
    if not (index_repo / ".git").exists():
        return False, f"索引仓库不是 git 检出: {index_repo}"
    if dest.exists() and any(dest.iterdir()):
        return False, f"目标目录已存在且不是 RAX native 检出: {dest}"

    listed = _git(index_repo, "branch", "--list", LOCAL_BRANCH)
    if listed.returncode != 0:
        detail = (listed.stderr or listed.stdout or "").strip()
        return False, detail or "git branch --list 失败"
    has_local = bool(listed.stdout.strip())
    if has_local:
        cmd = ["worktree", "add", str(dest), LOCAL_BRANCH]
    else:
        cmd = ["worktree", "add", "-b", LOCAL_BRANCH, str(dest), NATIVE_REF]
    _emit(log, "git -C " + str(index_repo) + " " + " ".join(cmd))
    proc = _git(index_repo, *cmd, timeout=1800)
    if proc.returncode != 0 or not has_dashboard(dest):
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(detail[-30:] or [f"git exit={proc.returncode}"])
        return False, tail
    return True, str(dest)


def _bootstrap_python() -> Optional[Path]:
    candidates = [
        MINICONDA3 / "envs" / "RoboDojo" / "bin" / "python",
        MINICONDA3 / "envs" / "eai" / "bin" / "python",
        MINICONDA3 / "bin" / "python3",
        MINICONDA3 / "bin" / "python",
        Path("/usr/bin/python3.11"),
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def ensure_conda_env(*, log: Optional[LogFn] = None) -> Tuple[bool, str]:
    """在 miniconda3/envs/rax 安装操作台依赖（Python 3.11）。"""
    env_dir = conda_env_prefix()
    py = env_dir / "bin" / "python"
    if not py.is_file():
        base = _bootstrap_python()
        if base is None:
            return False, f"找不到可用于创建 {env_dir} 的 Python"
        _emit(log, f"用 {base} 创建 {env_dir}")
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [str(base), "-m", "venv", str(env_dir)],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            return False, f"创建 {env_dir} 超时。共享盘较慢，可再点一次「安装环境」。"
        if proc.returncode != 0 or not py.is_file():
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, "\n".join(detail[-20:] or [f"venv exit={proc.returncode}"])
    _emit(log, f"安装依赖到 {py}")
    pip_env = os.environ.copy()
    pip_env["PYTHONNOUSERSITE"] = "1"
    pip_env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                "-i",
                "https://mirrors.aliyun.com/pypi/simple",
                "--trusted-host",
                "mirrors.aliyun.com",
                *PANEL_PACKAGES,
            ],
            capture_output=True,
            text=True,
            timeout=3600,
            env=pip_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"安装依赖超时: {py}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, "\n".join(detail[-30:] or [f"pip exit={proc.returncode}"])
    ok, msg = panel_deps_ok(py)
    if not ok:
        return False, msg
    return True, f"环境就绪: {env_dir}"


def panel_deps_ok(python: Path) -> Tuple[bool, str]:
    if not python.is_file():
        return False, f"Python 不存在: {python}"
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
        }
    }
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run(
            [
                str(python),
                "-c",
                "import numpy, PIL, scipy; "
                "print(f'numpy={numpy.__version__} pillow={PIL.__version__} scipy={scipy.__version__}')",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"依赖检查失败: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, err[-1] if err else f"exit={proc.returncode}"
    return True, (proc.stdout or "").strip()


def link_runtime_env(project: Path, env_prefix: Path) -> Tuple[bool, str]:
    """让 project/runtime/envs/robodojo 指向 miniconda3 里的环境。

    bootstrap 要求 runtime 是指向 .local-runtime 的符号链接，因此不把 runtime
    做成普通目录。
    """
    project = Path(project)
    env_prefix = Path(env_prefix)
    if not (env_prefix / "bin" / "python").is_file():
        return False, f"环境里没有 python: {env_prefix}"
    local = project / ".local-runtime"
    local.mkdir(parents=True, exist_ok=True)
    runtime = project / "runtime"
    if runtime.is_symlink():
        current = Path(os.path.realpath(runtime))
        if current != local.resolve():
            return False, f"runtime 已指向别处: {runtime} -> {os.readlink(runtime)}"
    elif runtime.exists():
        return False, f"runtime 已是普通目录，拒绝覆盖: {runtime}"
    else:
        runtime.symlink_to(".local-runtime", target_is_directory=True)

    envs = local / "envs"
    envs.mkdir(parents=True, exist_ok=True)
    link = envs / "robodojo"
    target = env_prefix.resolve()
    if link.is_symlink():
        if Path(os.path.realpath(link)) == target:
            return True, f"{project / 'runtime' / 'envs' / 'robodojo'} -> {target}"
        link.unlink()
    elif link.exists():
        return False, f"已有非链接环境目录: {link}"
    link.symlink_to(target, target_is_directory=True)
    return True, f"{project / 'runtime' / 'envs' / 'robodojo'} -> {target}"


def probe(index_repo: Path, run_root: Path, python: Path) -> List[str]:
    lines: List[str] = []
    runnable = run_root if has_dashboard(run_root) else (
        index_repo if has_dashboard(index_repo) else None
    )
    if runnable is None:
        lines.append("运行代码: 未检出（点「准备代码」拉取 robodojo-behavior/native）")
    else:
        lines.append(f"运行代码: {runnable}")
    prefix = conda_env_prefix()
    lines.append(f"conda 环境: {prefix}")
    if python.is_file():
        ok, msg = panel_deps_ok(python)
        lines.append(("面板依赖: " if ok else "面板依赖缺失: ") + msg)
    else:
        lines.append(f"面板 Python 不存在: {python}（点「安装环境」）")
    if runnable is not None:
        linked = runnable / "runtime" / "envs" / "robodojo" / "bin" / "python"
        if linked.is_file():
            lines.append(f"runtime/envs/robodojo -> {os.path.realpath(linked)}")
        else:
            lines.append("runtime/envs/robodojo 尚未链到 miniconda3（启动时会自动链接）")
    conda = MINICONDA3 / "bin" / "conda"
    lines.append(f"miniconda3: {MINICONDA3}" + ("" if conda.is_file() else "（未找到 conda）"))
    if shutil.which("bwrap") is None:
        lines.append("未找到 bubblewrap（正式提交程序需要 bwrap；操作台本身可以先打开）")
    return lines
