"""FLORA 本地目录、miniconda3/envs/flora，以及 HF 权重检出。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

_PERSONAL = Path("/share_data/projects/mahjong/share/personal/liyichao")
MINICONDA3 = Path(os.environ.get("EAI_MINICONDA3", str(_PERSONAL / "miniconda3")))
ENV_NAME = os.environ.get("FLORA_CONDA_ENV", "flora")

# 工作目录：三个 HF 仓检出放在 models/Flora-{TempSeg,Captioner,Rewarder}
DEFAULT_ROOT = Path(
    os.environ.get("FLORA_ROOT", str(_PERSONAL / "models"))
)
DEFAULT_CACHE = Path(
    os.environ.get("FLORA_MODEL_CACHE", str(_PERSONAL / "models" / "Flora"))
)

TEMPSEG_REPO_ID = os.environ.get("FLORA_TEMPSEG_REPO", "nvidia/Flora-TempSeg")
CAPTIONER_REPO_ID = os.environ.get("FLORA_CAPTIONER_REPO", "nvidia/Flora-Captioner")
REWARDER_REPO_ID = os.environ.get("FLORA_REWARDER_REPO", "nvidia/Flora-Rewarder")

PAGE_URL = "https://research.nvidia.com/labs/gear/flora/"
HF_COLLECTION = "https://huggingface.co/collections/nvidia/flora"
WECHAT_URL = "https://mp.weixin.qq.com/s/u2hr7Qwx5CMG02DU7yggpQ"

# 文章五项能力 → UI 模式
CAPABILITIES: Tuple[Tuple[str, str, str], ...] = (
    (
        "annotate",
        "①② 时间分段 + 上下文动作描述",
        "TempSeg→Captioner：重叠窗口分段、拼接，再为每个区间生成带上下文的动作指令",
    ),
    (
        "validate",
        "仅校验视频（CPU）",
        "检查媒体/时间戳，不下载权重、不推理",
    ),
    (
        "outcome",
        "③ 结果奖励 Outcome",
        "整段 episode 成功度 1–5（Flora-Rewarder terminal）",
    ),
    (
        "process",
        "④ 过程奖励 Process",
        "每 0.5s 动作质量 1/2/3（Flora-Rewarder dense）",
    ),
    (
        "rubric",
        "⑤ 评分规则奖励 Rubric",
        "按用户自定义步骤与分值打分（Flora-Rewarder rubric）",
    ),
)

PROFILES: Tuple[str, ...] = ("human", "yam", "dex", "tianji")

DEFAULT_RUBRIC_JSON = """[
  {"action": "Grasp the cup", "grading": [
    {"label": "Not attempted", "points": 0.0},
    {"label": "Grasped after re-tries", "points": 0.1},
    {"label": "Clean grasp", "points": 0.2}
  ]},
  {"action": "Set the cup on the tray", "grading": [
    {"label": "Not attempted", "points": 0.0},
    {"label": "Placed but tipped over", "points": 0.4},
    {"label": "Placed upright", "points": 0.8}
  ]}
]"""

# TempSeg / Captioner stack（官方 batch_utils/requirements.txt）
ANNOTATE_PIP = (
    "huggingface_hub==1.33.0",
    "transformers==5.8.1",
    "peft==0.18.1",
    "accelerate==1.12.0",
    "safetensors==0.7.0",
    "av==16.0.1",
    "numpy<3",
    "Pillow>=11,<13",
)
# Rewarder stack
REWARDER_PIP = (
    "vllm==0.24.0",
    "transformers==5.8.1",
    "qwen-vl-utils==0.0.14",
    "decord==0.6.0",
    "av>=15",
    "numpy<3",
    "Pillow>=11,<13",
    "huggingface_hub>=1.0",
)

LogFn = Callable[[str], None]


def conda_env_prefix() -> Path:
    return MINICONDA3 / "envs" / ENV_NAME


def conda_python() -> Path:
    override = os.environ.get("FLORA_PYTHON", "").strip()
    if override:
        return Path(override)
    return conda_env_prefix() / "bin" / "python"


def resolve_python(prefer: str = "") -> str:
    for cand in ((prefer or "").strip(), str(conda_python())):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return str(conda_python())


def tempseg_dir(root: Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / "Flora-TempSeg"


def rewarder_dir(root: Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / "Flora-Rewarder"


def captioner_dir(root: Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT) / "Flora-Captioner"


def _emit(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)


def _bootstrap_python() -> Optional[Path]:
    candidates = [
        MINICONDA3 / "envs" / "aspire" / "bin" / "python",
        MINICONDA3 / "envs" / "eai" / "bin" / "python",
        MINICONDA3 / "bin" / "python3",
        MINICONDA3 / "bin" / "python",
        Path("/usr/bin/python3.12"),
        Path("/usr/bin/python3.11"),
        Path("/usr/bin/python3"),
    ]
    for path in candidates:
        if not (path.is_file() and os.access(path, os.X_OK)):
            continue
        # 跳过半成品 conda env（expat 损坏等）
        try:
            proc = subprocess.run(
                [str(path), "-c", "import xml.parsers.expat"],
                capture_output=True,
                timeout=30,
            )
        except Exception:  # noqa: BLE001
            continue
        if proc.returncode == 0:
            return path
    return None


def _run(
    argv: Sequence[str],
    *,
    log: Optional[LogFn] = None,
    env: Optional[dict] = None,
    timeout: int = 7200,
    cwd: Optional[Path] = None,
) -> Tuple[int, str]:
    _emit(log, "$ " + " ".join(str(x) for x in argv))
    try:
        proc = subprocess.run(
            [str(x) for x in argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired:
        return 1, f"timeout after {timeout}s: {' '.join(str(x) for x in argv)}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if out:
        for line in out.splitlines()[-40:]:
            _emit(log, line)
    return proc.returncode, out


def ensure_conda_env(*, log: Optional[LogFn] = None, stack: str = "both") -> Tuple[bool, str]:
    """在 miniconda3/envs/flora 安装 Python 3.12 + 推理依赖。

    stack: annotate | rewarder | both
    """
    env_dir = conda_env_prefix()
    py = env_dir / "bin" / "python"
    conda = MINICONDA3 / "bin" / "conda"

    if not py.is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        created = False
        last_err = ""
        if conda.is_file():
            # 共享盘上 --copy 易在 link 阶段失败，这里不用 --copy
            _emit(log, f"conda create → {env_dir} (python=3.12)")
            code, out = _run(
                [
                    str(conda),
                    "create",
                    "-y",
                    "-p",
                    str(env_dir),
                    "python=3.12",
                ],
                log=log,
                timeout=3600,
            )
            last_err = out
            created = code == 0 and py.is_file()
            if not created and env_dir.exists() and not py.is_file():
                shutil.rmtree(env_dir, ignore_errors=True)
        if not created:
            base = _bootstrap_python()
            if base is None:
                return False, (
                    f"找不到可用于创建 {env_dir} 的 Python；"
                    f"conda 错误:\n{last_err[-1500:]}"
                )
            _emit(log, f"venv 回退: {base} → {env_dir}")
            if env_dir.exists() and not py.is_file():
                shutil.rmtree(env_dir, ignore_errors=True)
            code, out = _run([str(base), "-m", "venv", str(env_dir)], log=log, timeout=600)
            if code != 0 or not py.is_file():
                return False, out[-2000:] or last_err[-1500:] or f"venv failed exit={code}"

    # 半成品 conda env 有时缺 pip
    code, _ = _run([str(py), "-m", "pip", "--version"], log=log, timeout=60)
    if code != 0:
        _emit(log, "修复 pip (ensurepip)…")
        _run([str(py), "-m", "ensurepip", "--upgrade"], log=log, timeout=300)
        _run(
            [str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
            log=log,
            timeout=600,
        )

    pip_env = os.environ.copy()
    pip_env["PYTHONNOUSERSITE"] = "1"
    pip_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 共享盘 pip 缓存易慢/占满；与其它面板一致走阿里云
    index = [
        "-i",
        "https://mirrors.aliyun.com/pypi/simple",
        "--trusted-host",
        "mirrors.aliyun.com",
    ]

    if stack in ("annotate", "both"):
        _emit(log, "安装 PyTorch（cu128 / torch 2.11，官方验证配置）…")
        code, out = _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                "torch==2.11.0",
                "torchvision==0.26.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            log=log,
            env=pip_env,
            timeout=7200,
        )
        if code != 0:
            _emit(log, "cu128 安装失败，尝试默认 PyPI torch…")
            code, out = _run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--no-compile",
                    *index,
                    "torch",
                    "torchvision",
                ],
                log=log,
                env=pip_env,
                timeout=7200,
            )
            if code != 0:
                return False, out[-3000:] or f"torch install failed exit={code}"

        _emit(log, "安装 TempSeg/Captioner 依赖…")
        code, out = _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                *index,
                *ANNOTATE_PIP,
            ],
            log=log,
            env=pip_env,
            timeout=3600,
        )
        if code != 0:
            return False, out[-3000:] or f"annotate pip failed exit={code}"

    if stack in ("rewarder", "both"):
        _emit(log, "安装 Rewarder 依赖（含 vllm）…")
        code, out = _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                *index,
                *REWARDER_PIP,
            ],
            log=log,
            env=pip_env,
            timeout=7200,
        )
        if code != 0:
            return False, out[-3000:] or f"rewarder pip failed exit={code}"

    ok, msg = deps_ok(py, stack=stack)
    if not ok:
        return False, msg
    return True, f"环境就绪: {env_dir}\n{msg}"


def deps_ok(python: Path, *, stack: str = "both") -> Tuple[bool, str]:
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
    checks: List[str] = []
    if stack in ("annotate", "both"):
        checks.append(
            "import torch, transformers, peft, av; "
            "print(f'torch={torch.__version__} tf={transformers.__version__} peft={peft.__version__}')"
        )
    if stack in ("rewarder", "both"):
        checks.append(
            "import vllm, transformers; "
            "print(f'vllm={vllm.__version__} tf={transformers.__version__}')"
        )
    parts: List[str] = []
    for code in checks:
        try:
            proc = subprocess.run(
                [str(python), "-c", code],
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
        parts.append((proc.stdout or "").strip())
    return True, " | ".join(parts)


def ensure_repos(*, root: Path | None = None, log: Optional[LogFn] = None) -> Tuple[bool, str]:
    """用 huggingface_hub 拉取 TempSeg / Captioner / Rewarder 到 FLORA_ROOT。"""
    root = Path(root or DEFAULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)

    py = Path(resolve_python())
    if not py.is_file():
        return False, f"请先安装环境（需要 python）: {py}"

    # 确保 huggingface_hub 可用
    code, _ = _run(
        [str(py), "-c", "import huggingface_hub"],
        log=log,
        timeout=60,
    )
    if code != 0:
        code, out = _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-i",
                "https://mirrors.aliyun.com/pypi/simple",
                "--trusted-host",
                "mirrors.aliyun.com",
                "huggingface_hub",
            ],
            log=log,
            timeout=600,
        )
        if code != 0:
            return False, out[-1500:] or "pip install huggingface_hub failed"

    targets = (
        (TEMPSEG_REPO_ID, tempseg_dir(root)),
        (CAPTIONER_REPO_ID, captioner_dir(root)),
        (REWARDER_REPO_ID, rewarder_dir(root)),
    )
    script = r"""
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
path = snapshot_download(
    repo_id=repo,
    local_dir=str(dest),
    token=token,
    resume_download=True,
)
print(path)
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # 本机实测官方更快；慢/不通时再 export HF_ENDPOINT=https://hf-mirror.com
    env.setdefault("HF_ENDPOINT", "https://huggingface.co")
    lines: List[str] = []
    for repo_id, dest in targets:
        _emit(log, f"下载 {repo_id} → {dest} (HF_ENDPOINT={env.get('HF_ENDPOINT')})")
        code, out = _run(
            [str(py), "-c", script, repo_id, str(dest)],
            log=log,
            env=env,
            timeout=86400,
        )
        if code != 0:
            return False, f"{repo_id} 下载失败:\n{out[-2000:]}"
        lines.append(f"OK {repo_id} → {dest}")
    return True, "\n".join(lines)


def has_tempseg(root: Path | None = None) -> bool:
    d = tempseg_dir(root)
    return (d / "infer.py").is_file() and (d / "batch_utils" / "__main__.py").is_file()


def has_rewarder(root: Path | None = None) -> bool:
    d = rewarder_dir(root)
    return (d / "batch_utils" / "terminal_reward.py").is_file()


def probe(root: Path | None = None, python: Path | None = None) -> List[str]:
    root = Path(root or DEFAULT_ROOT)
    python = Path(python or resolve_python())
    lines: List[str] = [
        f"工作目录: {root}",
        f"模型缓存 FLORA_MODEL_CACHE: {DEFAULT_CACHE}",
        f"conda 环境: {conda_env_prefix()}",
        f"Python: {python}" + ("" if python.is_file() else "（不存在）"),
    ]
    if python.is_file():
        ok_a, msg_a = deps_ok(python, stack="annotate")
        lines.append(("分段/描述依赖: " if ok_a else "分段/描述依赖缺失: ") + msg_a)
        ok_r, msg_r = deps_ok(python, stack="rewarder")
        lines.append(("奖励依赖: " if ok_r else "奖励依赖缺失: ") + msg_r)
    lines.append(
        "TempSeg 检出: "
        + (str(tempseg_dir(root)) if has_tempseg(root) else "未下载（点「下载模型」）")
    )
    lines.append(
        "Rewarder 检出: "
        + (str(rewarder_dir(root)) if has_rewarder(root) else "未下载（点「下载模型」）")
    )
    cap = captioner_dir(root)
    if (cap / "config.json").is_file():
        lines.append(f"Captioner 检出: {cap}")
    else:
        lines.append("Captioner: 运行 annotate 时会按 models.json 自动拉取到缓存")
    hf = bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        or (Path.home() / ".cache" / "huggingface" / "token").is_file()
    )
    lines.append("HuggingFace 登录: " + ("已配置" if hf else "未检测到 token（公开仓一般可直下）"))
    lines.append(f"项目页: {PAGE_URL}")
    return lines


def _nvidia_lib_dirs(python: str) -> List[str]:
    """site-packages torch + nvidia/*/lib for pip CUDA wheels (cudnn/cublas/…).

    Keep both ``cuda_runtime`` (libcudart.so.12 for torch+cu128) and ``cu13``
    (libcudart.so.13 for vLLM 0.24 ``_C_stable_libtorch``); sonames differ so
    they do not shadow each other. Prefer cuda_runtime ahead of cu13.
    """
    py_bin = Path(python)
    if not py_bin.is_file():
        return []
    prefix = py_bin.parent.parent
    dirs: List[str] = []
    for child in prefix.glob("lib/python*/site-packages/nvidia/*/lib"):
        if child.is_dir():
            dirs.append(str(child.resolve()))

    def _prio(p: str) -> int:
        low = p.replace("\\", "/").lower()
        for i, key in enumerate(
            (
                "/nvidia/cudnn/",
                "/nvidia/cublas/",
                "/nvidia/cuda_runtime/",
                "/nvidia/cu13/",
                "/nvidia/cufft/",
                "/nvidia/cusolver/",
                "/nvidia/cusparse/",
                "/nvidia/nvjitlink/",
                "/nvidia/nccl/",
            )
        ):
            if key in low:
                return i
        return 100

    dirs.sort(key=_prio)
    # Front: conda lib + torch/lib (vLLM stable-libtorch needs libtorch*.so).
    front: List[str] = []
    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        front.append(str(lib_dir))
    for child in prefix.glob("lib/python*/site-packages/torch/lib"):
        if child.is_dir():
            front.append(str(child.resolve()))
    return front + dirs


def child_env(
    *,
    python: str,
    gpu: str = "0",
    root: Path | None = None,
) -> dict:
    """子进程环境：丢掉 GUI/VirtualGL 污染，指向 flora conda 与模型缓存。

    Viewer 在 vglrun 下会带 ``LD_PRELOAD=libdlfaker.so:libvglfaker.so`` 与
    ROS/Qt ``LD_LIBRARY_PATH``；继承后常在 generate 阶段报
    ``Invalid handle. Cannot load symbol cudnnGetVersion``。
    """
    root = Path(root or DEFAULT_ROOT)
    # Drop conda/venv + VirtualGL/Qt/ROS library injection from the GUI parent.
    drop = {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_SHLVL",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "MUJOCO_USE_VGL",
        "MUJOCO_VGL_DEVICE",
        "QT_XCB_GL_INTEGRATION",
        "QT_OPENGL",
        "QT_QUICK_BACKEND",
        "__GL_SYNC_TO_VBLANK",
    }
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = (gpu or "0").strip() or "0"
    env["FLORA_MODEL_CACHE"] = str(DEFAULT_CACHE)
    env["FLORA_TEMPSEG_REPO"] = TEMPSEG_REPO_ID
    env["FLORA_CAPTIONER_REPO"] = CAPTIONER_REPO_ID
    env["FLORA_REWARDER_REPO"] = str(rewarder_dir(root)) if has_rewarder(root) else REWARDER_REPO_ID
    env["HF_HOME"] = env.get("HF_HOME") or str(DEFAULT_CACHE / "hf_home")
    env.setdefault("HF_ENDPOINT", "https://huggingface.co")
    # Blackwell SM 12.x + torch cu128: FlashInfer JIT reports
    # "SM 12.x requires CUDA >= 12.9" then fails check_cuda_arch with a
    # misleading "sm75 or higher". Use vLLM's native top-k/top-p sampler.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # 让 python -m 找到对应 package：cwd 会设为仓库根
    py_bin = Path(python)
    if py_bin.is_file():
        env["PATH"] = str(py_bin.parent) + os.pathsep + env.get("PATH", "")
        prefix = py_bin.parent.parent
        env["VIRTUAL_ENV"] = str(prefix)
    nv_libs = _nvidia_lib_dirs(python)
    if nv_libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(nv_libs)
    return env

