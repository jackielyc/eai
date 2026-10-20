"""发现 ASPIRE 示例 / 校验计划，并管理 miniconda3/envs/aspire。"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

DEFAULT_REPO = os.environ.get(
    "ASPIRE_REPO",
    "/share_data/projects/mahjong/share/personal/liyichao/ASPIRE",
)
_MINICONDA3 = os.environ.get(
    "EAI_MINICONDA3",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3",
)
DEFAULT_CONDA_ENV = os.environ.get("ASPIRE_CONDA_ENV", "aspire")
DEFAULT_CONDA_PYTHON = Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
# ASPIRE bootstrap 的 .venv-* 字节落到 miniconda3/envs 下（经 CAP_HARNESS_VENV_ROOT）
VENV_BACKING_ROOT = Path(
    os.environ.get("ASPIRE_VENV_BACKING", str(Path(_MINICONDA3) / "envs"))
)

PAGE_URL = "https://research.nvidia.com/labs/gear/aspire/"
ARXIV_URL = "https://arxiv.org/abs/2607.00272"

# 预设示例：(显示名, benchmark, suite, task_id, program_rel, max_steps, camera_w, camera_h)
EXAMPLE_PRESETS: Tuple[Tuple[str, str, str, int, str, int, int, int], ...] = (
    (
        "LIBERO · object_swap seed1",
        "libero-pro",
        "libero_object_swap",
        0,
        "examples/libero/object_swap_seed1.py",
        2000,
        800,
        512,
    ),
    (
        "LIBERO · open_middle_drawer",
        "libero-pro",
        "libero_goal_swap",
        0,
        "examples/libero/open_middle_drawer.py",
        2000,
        800,
        512,
    ),
    (
        "LIBERO · put_bowl_in_bottom_drawer",
        "libero-pro",
        "libero_10_swap",
        3,
        "examples/libero/put_bowl_in_bottom_drawer.py",
        3000,
        800,
        512,
    ),
    (
        "LIBERO · put_wine_bottle_on_drawer",
        "libero-pro",
        "libero_goal_swap",
        2,
        "examples/libero/put_wine_bottle_on_drawer.py",
        3000,
        800,
        512,
    ),
    (
        "LIBERO · put_wine_bottle_on_rack",
        "libero-pro",
        "libero_goal_swap",
        9,
        "examples/libero/put_wine_bottle_on_rack.py",
        3000,
        800,
        512,
    ),
    (
        "LIBERO · close_microwave",
        "libero-pro",
        "libero_90",
        33,
        "examples/libero/close_microwave.py",
        3000,
        800,
        512,
    ),
    (
        "Robosuite · cube_lifting",
        "robosuite",
        "cube_lifting",
        0,
        "examples/robosuite/cube_lifting_seed1.py",
        3000,
        512,
        512,
    ),
    (
        "Robosuite · two_arm_lift",
        "robosuite",
        "two_arm_lift",
        0,
        "examples/robosuite/two_arm_lift_seed1.py",
        3000,
        512,
        512,
    ),
    (
        "Robosuite · two_arm_handover",
        "robosuite",
        "two_arm_handover",
        0,
        "examples/robosuite/two_arm_handover_seed1.py",
        7000,
        512,
        512,
    ),
)

VALIDATION_PLANS: Tuple[Tuple[str, str], ...] = (
    ("LIBERO smoke", "configs/validation/libero-smoke.yaml"),
    ("LIBERO nightly", "configs/validation/libero-nightly.yaml"),
    ("Robosuite structural", "configs/validation/robosuite-structural.yaml"),
    ("Robosuite control", "configs/validation/robosuite-control.yaml"),
    ("Robosuite bimanual", "configs/validation/robosuite-bimanual.yaml"),
    ("BEHAVIOR structural", "configs/validation/behavior-structural.yaml"),
    ("BEHAVIOR pickup", "configs/validation/behavior-pickup.yaml"),
)


@dataclass
class ProbeResult:
    repo: Path
    ok: bool
    lines: List[str] = field(default_factory=list)
    python: str = ""
    has_cap_harness: bool = False
    has_libero_venv: bool = False
    has_robosuite_venv: bool = False
    has_uv: bool = False
    example_count: int = 0


def repo_path(path: Optional[str] = None) -> Path:
    return Path(path or DEFAULT_REPO).expanduser().resolve()


def conda_python() -> Path:
    override = os.environ.get("ASPIRE_PYTHON", "").strip()
    if override:
        return Path(override)
    return Path(DEFAULT_CONDA_PYTHON)


def resolve_python(repo: Optional[Path] = None) -> str:
    """优先 miniconda3/envs/aspire；其次仓库 .venv-libero / .venv。

    不 Path.resolve() python，避免软链落到错误 base 环境。
    """
    root = repo_path(str(repo) if repo else None)
    for candidate in (
        Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python",
        Path("/home/psibot/miniconda3/envs") / DEFAULT_CONDA_ENV / "bin" / "python",
        conda_python(),
        root / ".venv-libero" / "bin" / "python",
        root / ".venv-robosuite" / "bin" / "python",
        root / ".venv" / "bin" / "python",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return str(Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python")


def resolve_runtime_python(repo: Optional[Path], benchmark: str) -> str:
    """跑 episode 时优先对应 backend venv，再回退 aspire conda。"""
    root = repo_path(str(repo) if repo else None)
    bench = (benchmark or "").strip().lower()
    preferred: List[Path] = []
    if bench in {"libero-pro", "libero"}:
        preferred.append(root / ".venv-libero" / "bin" / "python")
    elif bench == "robosuite":
        preferred.append(root / ".venv-robosuite" / "bin" / "python")
    elif bench == "behavior":
        preferred.append(root / ".venv-behavior" / "bin" / "python")
    preferred.append(Path(resolve_python(root)))
    for candidate in preferred:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return resolve_python(root)


def host_deps_ok(python: str, repo: Optional[Path] = None) -> Tuple[bool, str]:
    root = repo_path(str(repo) if repo else None)
    py = Path(python)
    if not py.is_file():
        return (
            False,
            f"Python 不存在: {python}（请先点「安装 host 依赖」创建 "
            f"miniconda3/envs/{DEFAULT_CONDA_ENV}）",
        )
    try:
        clean = {
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
        clean["PYTHONPATH"] = str(root / "src")
        clean["PYTHONNOUSERSITE"] = "1"
        proc = subprocess.run(
            [
                str(py),
                "-c",
                "import numpy, yaml, cap_harness; "
                "from cap_harness import cli; "
                "print(f'cap_harness ok numpy={numpy.__version__}')",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=25,
            env=clean,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"依赖检查失败: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit={proc.returncode}"
        return False, f"编排环境缺依赖: {tail}"
    return True, (proc.stdout or "").strip() or "cap_harness OK"


def _conda_bin() -> Optional[Path]:
    for cand in (
        Path(_MINICONDA3) / "bin" / "conda",
        Path("/home/psibot/miniconda3/bin/conda"),
        Path(shutil.which("conda") or ""),
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def ensure_host_env(
    repo: Optional[Path] = None,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """在 miniconda3/envs/aspire 中安装 ASPIRE host（editable cap-harness）。"""
    root = repo_path(str(repo) if repo else None)
    emit = log or (lambda _m: None)
    if not (root / "pyproject.toml").is_file():
        return False, f"不是 ASPIRE 仓库: {root}"

    py_out = Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
    env_dir = py_out.parent.parent
    if not py_out.is_file():
        # 共享盘上 conda create --copy 极慢；优先用现有 py3.11/3.12 env 做 venv bootstrap
        # （与 EmbodiedEvo 相同策略；目录仍落在 miniconda3/envs/aspire）
        if env_dir.is_dir() and not py_out.is_file():
            emit(f"清理不完整环境: {env_dir}")
            shutil.rmtree(env_dir, ignore_errors=True)
        base_candidates = [
            Path(_MINICONDA3) / "envs" / "eai" / "bin" / "python",
            Path(_MINICONDA3) / "envs" / "libero" / "bin" / "python",
            Path(_MINICONDA3) / "envs" / "embodiedevo" / "bin" / "python",
            Path("/usr/bin/python3.11"),
            Path("/usr/bin/python3.12"),
        ]
        base = next((p for p in base_candidates if p.is_file()), None)
        if base is None:
            conda = _conda_bin()
            if conda is None:
                return False, "找不到 Python 3.11+ / conda 用于创建 miniconda3/envs/aspire"
            emit(f"$ conda create -y -p {env_dir} python=3.11（无本地 base，较慢）")
            proc = subprocess.run(
                [str(conda), "create", "-y", "-p", str(env_dir), "python=3.11"],
                capture_output=True,
                text=True,
                timeout=900,
            )
            if proc.returncode != 0 or not py_out.is_file():
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                return False, "\n".join(detail[-30:] or [f"conda exit={proc.returncode}"])
        else:
            emit(f"用 {base} 创建 {env_dir}")
            proc = subprocess.run(
                [str(base), "-m", "venv", str(env_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0 or not py_out.is_file():
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                return False, "\n".join(detail[-20:] or [f"venv exit={proc.returncode}"])

    emit(f"使用 Python: {py_out}")
    # --no-compile：共享盘上 compileall 极慢
    pip_cmds = (
        [
            str(py_out),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-compile",
            "-i",
            "https://mirrors.aliyun.com/pypi/simple",
            "--trusted-host",
            "mirrors.aliyun.com",
            "numpy>=1.26,<3",
            "requests>=2.32",
            "scipy>=1.12",
            "PyYAML>=6",
            "pillow>=10",
            "fastapi>=0.115",
            "uvicorn>=0.34",
        ],
        [
            str(py_out),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-compile",
            "-i",
            "https://mirrors.aliyun.com/pypi/simple",
            "--trusted-host",
            "mirrors.aliyun.com",
            "-e",
            str(root),
        ],
    )
    for args in pip_cmds:
        emit("$ " + " ".join(args))
        proc = subprocess.run(
            args, cwd=str(root), capture_output=True, text=True, timeout=1200
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, "\n".join(detail[-30:] or [f"exit={proc.returncode}"])
        last = (proc.stdout or "").strip().splitlines()
        if last:
            emit(last[-1])

    ok, msg = host_deps_ok(str(py_out), root)
    return ok, msg if ok else f"安装完成但仍无法 import: {msg}"


def resolve_cap_harness(python: str, repo: Optional[Path] = None) -> List[str]:
    """返回调用 cap-harness 的 argv 前缀。"""
    root = repo_path(str(repo) if repo else None)
    py = Path(python)
    bin_cli = py.parent / "cap-harness"
    if bin_cli.is_file() and os.access(bin_cli, os.X_OK):
        return [str(bin_cli)]
    return [str(py), "-m", "cap_harness.cli"]


def list_example_programs(repo: Optional[Path] = None) -> List[Path]:
    root = repo_path(str(repo) if repo else None)
    out: List[Path] = []
    for sub in ("libero", "robosuite", "behavior"):
        d = root / "examples" / sub
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            out.append(path)
    return out


def list_validation_plans(repo: Optional[Path] = None) -> List[Tuple[str, Path]]:
    root = repo_path(str(repo) if repo else None)
    out: List[Tuple[str, Path]] = []
    for title, rel in VALIDATION_PLANS:
        path = root / rel
        if path.is_file():
            out.append((title, path))
    # 额外扫描
    plan_dir = root / "configs" / "validation"
    known = {p for _, p in out}
    if plan_dir.is_dir():
        for path in sorted(plan_dir.glob("*.yaml")):
            if path in known:
                continue
            out.append((path.stem, path))
    return out


def find_uv() -> Optional[str]:
    for cand in (
        shutil.which("uv"),
        str(Path.home() / ".local" / "bin" / "uv"),
        str(Path(_MINICONDA3) / "bin" / "uv"),
        str(Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "uv"),
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def probe_repo(repo: Optional[Path] = None) -> ProbeResult:
    root = repo_path(str(repo) if repo else None)
    lines: List[str] = []
    py = resolve_python(root)
    result = ProbeResult(repo=root, ok=False, python=py)

    if not root.is_dir():
        lines.append(f"仓库不存在: {root}")
        result.lines = lines
        return result
    if not (root / "pyproject.toml").is_file():
        lines.append(f"缺少 pyproject.toml: {root}")
        result.lines = lines
        return result

    lines.append(f"仓库: {root}")
    lines.append(f"Python: {py}")
    lines.append(f"conda env: {_MINICONDA3}/envs/{DEFAULT_CONDA_ENV}")

    ok, msg = host_deps_ok(py, root)
    result.has_cap_harness = ok
    lines.append(("✓ " if ok else "✗ ") + msg)

    result.has_libero_venv = (root / ".venv-libero" / "bin" / "python").is_file()
    result.has_robosuite_venv = (root / ".venv-robosuite" / "bin" / "python").is_file()
    lines.append(
        ("✓ " if result.has_libero_venv else "○ ")
        + f".venv-libero ({'就绪' if result.has_libero_venv else '未 bootstrap'})"
    )
    lines.append(
        ("✓ " if result.has_robosuite_venv else "○ ")
        + f".venv-robosuite ({'就绪' if result.has_robosuite_venv else '未 bootstrap'})"
    )

    uv = find_uv()
    result.has_uv = bool(uv)
    lines.append(("✓ " if uv else "○ ") + (f"uv: {uv}" if uv else "uv 未安装（bootstrap 需要）"))

    examples = list_example_programs(root)
    result.example_count = len(examples)
    lines.append(f"示例程序: {result.example_count}")

    result.ok = ok
    result.lines = lines
    return result


def child_env(repo: Path, *, python: str, gpu: str = "0") -> dict:
    """为 cap-harness 子进程准备干净环境。"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
        }
    }
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHONNOUSERSITE"] = "1"
    env["CAP_HARNESS_VENV_ROOT"] = str(VENV_BACKING_ROOT)
    env["LIBERO_CONFIG_PATH"] = str(repo / ".libero")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("EGL_PLATFORM", "device")
    gpu_s = str(gpu).strip() or "0"
    env["CUDA_VISIBLE_DEVICES"] = gpu_s
    env["MUJOCO_EGL_DEVICE_ID"] = gpu_s
    # 保证 aspire conda 的 bin（含 cap-harness / uv）优先
    bin_dir = str(Path(python).parent)
    path = env.get("PATH", "")
    if bin_dir and not path.startswith(bin_dir + os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + path
    return env
