"""发现 EmbodiedEvo 任务、读取配置、探测运行前置条件。"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_REPO = os.environ.get(
    "EMBODIEDEVO_REPO",
    "/share_data/projects/mahjong/share/personal/liyichao/EmbodiedEvo",
)
_MINICONDA3 = os.environ.get(
    "EAI_MINICONDA3",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3",
)
DEFAULT_CONDA_ENV = os.environ.get("EMBODIEDEVO_CONDA_ENV", "embodiedevo")
DEFAULT_CONDA_PYTHON = Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
DEFAULT_CODEX_HOME = Path(
    os.environ.get(
        "EMBODIEDEVO_CODEX_HOME",
        str(Path.home() / ".config" / "embodiedevo" / "codex"),
    )
).expanduser()
_PAGE = "https://lab.einsia.ai/embodiedevo/"
_ARXIV = "https://arxiv.org/abs/2610.18867"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    recipe_path: Path
    recipe_text: str = ""


@dataclass
class ProbeResult:
    repo: Path
    ok: bool
    lines: List[str] = field(default_factory=list)
    python: str = ""
    has_venv: bool = False
    has_docker: bool = False
    has_agent_image: bool = False
    has_host_codex: bool = False
    has_codex_home: bool = False
    has_codex_auth: bool = False
    codex_home: str = ""
    isolation: str = "host"
    recipe_count: int = 0


def repo_path(path: Optional[str] = None) -> Path:
    return Path(path or DEFAULT_REPO).expanduser().resolve()


def list_env_tasks(repo: Optional[Path] = None) -> List[TaskSpec]:
    root = repo_path(str(repo) if repo else None)
    recipe_dir = root / "tasks" / "RoboDojo" / "recipe"
    out: List[TaskSpec] = []
    if not recipe_dir.is_dir():
        return out
    for path in sorted(recipe_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        out.append(
            TaskSpec(
                key=path.stem,
                title=path.stem.replace("_", " "),
                recipe_path=path,
                recipe_text=text,
            )
        )
    return out


def read_task_toml(repo: Optional[Path] = None, *, baseline: bool = False) -> Dict[str, object]:
    root = repo_path(str(repo) if repo else None)
    rel = Path("tasks_baseline" if baseline else "tasks") / "RoboDojo" / "task.toml"
    path = root / rel
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11 host
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return {"_raw": path.read_text(encoding="utf-8", errors="replace")}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_raw": path.read_text(encoding="utf-8", errors="replace")}


def conda_python() -> Path:
    override = os.environ.get("EMBODIEDEVO_PYTHON", "").strip()
    if override:
        return Path(override)
    return Path(DEFAULT_CONDA_PYTHON)


def venv_python(repo: Optional[Path] = None) -> Path:
    """兼容旧路径；正式运行优先 conda_python()。"""
    root = repo_path(str(repo) if repo else None)
    return root / ".venv" / "bin" / "python"


def resolve_python(repo: Optional[Path] = None) -> str:
    """优先 miniconda3/envs/embodiedevo，其次仓库 .venv；不回退 ros-humble。

    注意：venv 的 python 常软链到 bootstrap 解释器；不要 Path.resolve()，
    否则会落到 eai 等环境并丢 site-packages。
    """
    root = repo_path(str(repo) if repo else None)
    for candidate in (
        Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python",
        Path("/home/psibot/miniconda3/envs") / DEFAULT_CONDA_ENV / "bin" / "python",
        conda_python(),
        venv_python(root),
        root / ".venv" / "bin" / "python3",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return str(Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python")


def host_deps_ok(python: str, repo: Optional[Path] = None) -> Tuple[bool, str]:
    """检查编排环境能否 import gymnasium / orchestrator。"""
    root = repo_path(str(repo) if repo else None)
    py = Path(python)
    if not py.is_file():
        return (
            False,
            f"Python 不存在: {python}（请先点「安装 host 依赖」创建 "
            f"miniconda3/envs/{DEFAULT_CONDA_ENV}）",
        )
    try:
        # 不要继承 GUI/ros-humble 的 PYTHONPATH，否则会串到错误 numpy
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
        clean["PYTHONPATH"] = str(root)
        clean["PYTHONNOUSERSITE"] = "1"
        proc = subprocess.run(
            [
                str(py),
                "-c",
                "import numpy, gymnasium, orchestrator; "
                "print(f'{gymnasium.__version__} numpy={numpy.__version__}')",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20,
            env=clean,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"依赖检查失败: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit={proc.returncode}"
        return False, f"编排环境缺依赖: {tail}"
    ver = (proc.stdout or "").strip()
    return True, f"gymnasium {ver} + orchestrator OK"


def _conda_bin() -> Optional[Path]:
    for cand in (
        Path(_MINICONDA3) / "bin" / "conda",
        Path("/home/psibot/miniconda3/bin/conda"),
        Path(shutil.which("conda") or ""),
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def ensure_host_venv(
    repo: Optional[Path] = None,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """在 miniconda3/envs/embodiedevo 中安装 EmbodiedEvo host 依赖。"""
    root = repo_path(str(repo) if repo else None)
    emit = log or (lambda _m: None)
    req = root / "requirements-host.txt"
    if not req.is_file():
        return False, f"缺少 {req}"

    py_out = Path(_MINICONDA3) / "envs" / DEFAULT_CONDA_ENV / "bin" / "python"
    env_dir = py_out.parent.parent
    if not py_out.is_file():
        # 共享盘上 conda create 极慢；用现有 py3.12 conda env 做 bootstrap（同 bagel）
        base_candidates = [
            Path(_MINICONDA3) / "envs" / "eai" / "bin" / "python",
            Path(_MINICONDA3) / "envs" / "libero" / "bin" / "python",
            Path("/usr/bin/python3.12"),
        ]
        base = next((p for p in base_candidates if p.is_file()), None)
        if base is None:
            return False, "找不到 Python 3.12 用于创建 miniconda3/envs/embodiedevo"
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
    # 不做 pip self-upgrade（共享盘上易卡死）
    for args in (
        [
            str(py_out),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-i",
            "https://mirrors.aliyun.com/pypi/simple",
            "--trusted-host",
            "mirrors.aliyun.com",
            "-r",
            str(req),
        ],
        [
            str(py_out),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(root),
        ],
    ):
        emit("$ " + " ".join(args))
        proc = subprocess.run(args, cwd=str(root), capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, "\n".join(detail[-20:] or [f"exit={proc.returncode}"])
        last = (proc.stdout or "").strip().splitlines()
        if last:
            emit(last[-1])

    ok, msg = host_deps_ok(str(py_out), root)
    return ok, msg if ok else f"安装完成但仍无法 import: {msg}"


def resolve_cli(repo: Optional[Path] = None, python: Optional[str] = None) -> Tuple[str, List[str]]:
    """返回 (cwd, argv_prefix)。优先 embodiedevo 入口，否则 -m orchestrator。"""
    root = repo_path(str(repo) if repo else None)
    py = python or resolve_python(root)
    bin_cli = Path(py).resolve().parent / "embodiedevo"
    if bin_cli.is_file() and os.access(bin_cli, os.X_OK):
        return str(root), [str(bin_cli)]
    return str(root), [py, "-m", "orchestrator"]


def _docker_has_image(name: str) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        proc = subprocess.run(
            [docker, "image", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_codex_home() -> Path:
    """优先 EMBODIEDEVO_CODEX_HOME，其次默认 ~/.config/embodiedevo/codex。"""
    override = os.environ.get("EMBODIEDEVO_CODEX_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    if DEFAULT_CODEX_HOME.is_dir() or (DEFAULT_CODEX_HOME / "config.toml").is_file():
        return DEFAULT_CODEX_HOME
    # 兼容已有交互式 Codex home（仅当其中已有 config.toml）
    legacy = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    if (legacy / "config.toml").is_file():
        return legacy
    return DEFAULT_CODEX_HOME


def ensure_codex_home(
    repo: Optional[Path] = None,
    *,
    api_key: str = "",
    provider: str = "openai",
    base_url: str = "",
) -> Tuple[bool, str, Path]:
    """创建 EMBODIEDEVO_CODEX_HOME，写入 config.toml；若给了 api_key 则写 auth.json。"""
    import json

    root = repo_path(str(repo) if repo else None)
    home = resolve_codex_home()
    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    config = home / "config.toml"
    if not config.is_file():
        if provider == "compatible" or base_url.strip():
            url = (base_url.strip() or "https://api.openai.com/v1").rstrip("/")
            text = (
                'model = "gpt-6-astra"\n'
                'model_reasoning_effort = "xhigh"\n'
                'model_provider = "compatible"\n'
                'cli_auth_credentials_store = "file"\n'
                "\n"
                "[model_providers.compatible]\n"
                'name = "Responses API gateway"\n'
                f'base_url = "{url}"\n'
                'wire_api = "responses"\n'
                "requires_openai_auth = true\n"
                "supports_websockets = false\n"
            )
            src_note = "compatible template"
        else:
            example = root / "configs" / "codex.openai.example.toml"
            if example.is_file():
                text = example.read_text(encoding="utf-8")
                src_note = str(example)
            else:
                text = (
                    'model = "gpt-6-astra"\n'
                    'model_reasoning_effort = "xhigh"\n'
                    'model_provider = "openai"\n'
                    'cli_auth_credentials_store = "file"\n'
                )
                src_note = "builtin openai defaults"
        config.write_text(text, encoding="utf-8")
        config.chmod(0o600)
    else:
        src_note = "existing config.toml"

    key = (api_key or "").strip()
    auth = home / "auth.json"
    if key:
        auth.write_text(json.dumps({"OPENAI_API_KEY": key}), encoding="utf-8")
        auth.chmod(0o600)
        return True, f"已写入 {home} ({src_note} + auth.json)", home

    has_auth = False
    if auth.is_file():
        try:
            data = json.loads(auth.read_text(encoding="utf-8"))
            has_auth = bool(str(data.get("OPENAI_API_KEY", "")).strip())
        except Exception:
            has_auth = False
    if has_auth:
        return True, f"Codex HOME 就绪: {home}", home
    return (
        False,
        f"已创建 config.toml → {config}；仍缺 API Key（填写后点「保存 Codex 凭证」）",
        home,
    )


def _host_codex_bin(repo: Path) -> Optional[Path]:
    override = os.environ.get("EMBODIEDEVO_CODEX_BIN", "").strip()
    if override and Path(override).is_file():
        return Path(override)
    candidates = [
        repo / ".cache" / "bin" / "codex",
        Path(shutil.which("codex") or ""),
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def probe_repo(repo: Optional[Path] = None) -> ProbeResult:
    root = repo_path(str(repo) if repo else None)
    lines: List[str] = []
    ok = True
    if not root.is_dir():
        return ProbeResult(repo=root, ok=False, lines=[f"仓库不存在: {root}"])
    lines.append(f"仓库: {root}")
    py = resolve_python(root)
    has_conda = Path(py).is_file() and "miniconda3/envs" in str(Path(py).resolve())
    has_venv = Path(py).is_file()
    tag = " (conda)" if has_conda else (" (ok)" if has_venv else f" (期望 {DEFAULT_CONDA_PYTHON})")
    lines.append(f"Python: {py}{tag}")
    deps_ok, deps_msg = host_deps_ok(py, root) if Path(py).is_file() else (False, "无解释器")
    lines.append("Host 依赖: " + deps_msg)
    if not Path(py).is_file() or not deps_ok:
        lines.append(
            f"提示: 点击「安装 host 依赖」创建 miniconda3/envs/{DEFAULT_CONDA_ENV} 并安装 gymnasium 等"
        )
        ok = False
    if not (root / "orchestrator" / "__main__.py").is_file():
        lines.append("缺少 orchestrator/__main__.py")
        ok = False
    if not (root / "tasks" / "RoboDojo" / "task.toml").is_file():
        lines.append("缺少 tasks/RoboDojo/task.toml")
        ok = False
    recipes = list_env_tasks(root)
    lines.append(f"RoboDojo recipes: {len(recipes)}")

    data = read_task_toml(root)
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    isolation = "host"
    if isinstance(agent, dict) and agent.get("isolation"):
        isolation = str(agent["isolation"])
    lines.append(f"Agent isolation: {isolation}")

    # 探测前尽量落盘默认 config.toml，避免仅因目录未创建而中止
    ensure_codex_home(root)
    codex_bin = _host_codex_bin(root)
    has_host_codex = codex_bin is not None
    lines.append(
        "Host Codex CLI: " + (str(codex_bin) if has_host_codex else "未找到 (.cache/bin/codex)")
    )
    codex_home = resolve_codex_home()
    has_codex_config = (codex_home / "config.toml").is_file()
    has_codex_auth = False
    auth_path = codex_home / "auth.json"
    if auth_path.is_file():
        try:
            import json

            data = json.loads(auth_path.read_text(encoding="utf-8"))
            has_codex_auth = bool(str(data.get("OPENAI_API_KEY", "")).strip())
        except Exception:
            has_codex_auth = False
    has_codex_home = has_codex_config and has_codex_auth
    lines.append(
        f"Codex HOME: {codex_home} ("
        + ("config.toml OK" if has_codex_config else "缺少 config.toml")
        + (", auth.json OK" if has_codex_auth else ", 缺少 auth.json/API Key")
        + ")"
    )
    if not has_codex_auth:
        lines.append("提示: 在下方填写 OpenAI API Key，点「保存 Codex 凭证」")

    has_docker = shutil.which("docker") is not None
    image = "embodiedevo-agent:latest"
    if isinstance(agent, dict) and agent.get("image"):
        image = str(agent["image"])
    has_image = _docker_has_image(image) if has_docker else False
    if isolation == "docker":
        lines.append("Docker: " + ("可用" if has_docker else "未找到"))
        lines.append(f"Agent 镜像 {image}: " + ("已构建" if has_image else "未找到"))
        if not has_docker or not has_image:
            ok = False
    else:
        lines.append("Docker agent: 跳过 (isolation=host)")
        if not has_host_codex or not has_codex_home:
            ok = False

    sim_py = root / ".venv-robodojo" / "bin" / "python"
    if sim_py.is_file():
        lines.append(f"RoboDojo 解释器: {sim_py}")
    else:
        robodojo = Path(_MINICONDA3) / "envs" / "RoboDojo" / "bin" / "python"
        if robodojo.is_file():
            lines.append(
                f"RoboDojo 解释器: 未链接 (.venv-robodojo)；可用 conda RoboDojo: {robodojo}"
            )
        else:
            lines.append("RoboDojo 解释器: 未安装 (.venv-robodojo / conda RoboDojo)")
    return ProbeResult(
        repo=root,
        ok=ok and Path(py).is_file() and deps_ok,
        lines=lines,
        python=py,
        has_venv=Path(py).is_file(),
        has_docker=has_docker,
        has_agent_image=has_image,
        has_host_codex=has_host_codex,
        has_codex_home=has_codex_home,
        has_codex_auth=has_codex_auth,
        codex_home=str(codex_home),
        isolation=isolation,
        recipe_count=len(recipes),
    )


def write_ui_task_override(
    repo: Path,
    *,
    baseline: bool,
    env_task: str,
    model: str,
    reasoning: str,
    playground_budget: int,
    test_scenes: int,
    test_budget: int,
    isolation: str = "host",
) -> Path:
    """在 runs/_ui_task_* 下生成带覆盖参数的临时 task 目录，不改动上游 task.toml。"""
    root = repo_path(str(repo))
    src = root / ("tasks_baseline" if baseline else "tasks") / "RoboDojo"
    if not (src / "task.toml").is_file():
        raise FileNotFoundError(f"缺少 task 定义: {src / 'task.toml'}")
    stamp = env_task or "task"
    dest = root / "runs" / f"_ui_task_{'baseline_' if baseline else ''}{stamp}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    toml_path = dest / "task.toml"
    text = toml_path.read_text(encoding="utf-8")
    text = _replace_toml_string(text, "env_task", env_task)
    text = _replace_toml_string(text, "model", model)
    text = _replace_toml_string(text, "reasoning_effort", reasoning)
    text = _replace_toml_string(text, "isolation", isolation)
    if isolation == "host":
        text = _replace_toml_string(text, "network", "none")
    if not baseline:
        text = _replace_toml_int(text, "budget", playground_budget, section="playground")
    text = _replace_toml_int(text, "scenes", test_scenes, section="test")
    text = _replace_toml_int(text, "budget", test_budget, section="test")
    toml_path.write_text(text, encoding="utf-8")
    return dest


def _replace_toml_string(text: str, key: str, value: str) -> str:
    import re

    pattern = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)"[^"]*"', re.M)
    repl = rf'\1"{value}"'
    if pattern.search(text):
        return pattern.sub(repl, text, count=1)
    return text + f'\n{key} = "{value}"\n'


def _replace_toml_int(text: str, key: str, value: int, *, section: str) -> str:
    import re

    # 仅替换指定 [section] 内第一次出现的 key=
    parts = re.split(r"(?=^\[)", text, flags=re.M)
    out: List[str] = []
    for part in parts:
        if part.startswith(f"[{section}]") or part.startswith(f"[{section}."):
            pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)\d+", re.M)
            if pattern.search(part):
                part = pattern.sub(rf"\g<1>{int(value)}", part, count=1)
        out.append(part)
    return "".join(out)


PAGE_URL = _PAGE
ARXIV_URL = _ARXIV
