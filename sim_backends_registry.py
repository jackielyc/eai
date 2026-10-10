"""Catalog of EAI 仿真评测 backends: native stack pages + extended benches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SimBackendSpec:
    """One simulation / evaluation backend exposed in the GUI."""

    key: str
    label: str
    kind: str  # native | extended
    stack_page: str  # isaac|mujoco|molmospaces|arena|libero|robotwin|extended
    hint: str
    # EmbodiedBench / official board
    embodied_slug: str = ""
    embodied_metric: str = "Success Rate"
    embodied_setting: str = "Average"
    source_url: str = ""
    source_label: str = ""
    # VLA harness docker tag (allenai/vla-evaluation-harness)
    harness_image: str = ""
    # Default local clone candidates (first existing wins)
    default_roots: Tuple[str, ...] = ()
    # Docs / install
    install_hint: str = ""
    engine: str = ""  # mujoco|sapien|isaac|genesis|coppelia|omni|
    robot_description: str = "a robot manipulator"
    camera_description: str = "the third-person scene camera"
    camera_key: str = ""


_LIY = "/share_data/projects/mahjong/share/personal/liyichao"

# Core backends already wired with dedicated GUI pages.
NATIVE_BACKENDS: Tuple[SimBackendSpec, ...] = (
    SimBackendSpec(
        key="isaac",
        label="Isaac / RoboDojo",
        kind="native",
        stack_page="isaac",
        hint="Isaac / RoboDojo：启动评测后进程常驻，直到「停止评测」。",
        source_url=(
            "https://galaxygeneralrobotics.github.io/astra-policy/"
            "index.html?view=gripper-manipulation"
        ),
        source_label="Astra×RoboDojo",
        harness_image="robodojo",
        engine="isaac",
        robot_description="a dual-arm robot in Isaac Sim / RoboDojo",
        camera_description="the head / third-person camera",
        camera_key="cam_head",
    ),
    SimBackendSpec(
        key="mujoco",
        label="MuJoCo",
        kind="native",
        stack_page="mujoco",
        hint="MuJoCo：本地 MJCF 交互 viewer。榜单默认对照 Meta-World。",
        embodied_slug="meta-world",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/meta-world",
        source_label="EmbodiedBench Meta-World",
        engine="mujoco",
        robot_description="a robot in MuJoCo",
        camera_description="the free / third-person MuJoCo camera",
    ),
    SimBackendSpec(
        key="molmospaces",
        label="MolmoSpaces",
        kind="native",
        stack_page="molmospaces",
        hint="MolmoSpaces：任务下拉框列出可评测 benchmark。",
        source_url="https://molmospaces.allen.ai/leaderboard",
        source_label="molmospaces.allen.ai",
        harness_image="molmospaces",
        engine="mujoco",
        robot_description="a mobile manipulator in MolmoSpaces",
        camera_description="the exo / third-person scene camera",
        camera_key="exo_camera_1",
    ),
    SimBackendSpec(
        key="arena",
        label="IsaacLab-Arena",
        kind="native",
        stack_page="arena",
        hint="IsaacLab-Arena：已注册 environment 列表评测。",
        source_url="https://isaac-sim.github.io/IsaacLab-Arena/main/index.html",
        source_label="Isaac Lab-Arena docs",
        engine="isaac",
        robot_description="a robot in IsaacLab-Arena",
        camera_description="the embodiment-mounted camera",
        camera_key="camera_obs",
    ),
    SimBackendSpec(
        key="libero",
        label="LIBERO",
        kind="native",
        stack_page="libero",
        hint="LIBERO：对齐 RLinf openpi eval。",
        embodied_slug="libero",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/libero",
        source_label="EmbodiedBench LIBERO",
        harness_image="libero",
        engine="mujoco",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    SimBackendSpec(
        key="robotwin",
        label="RoboTwin",
        kind="native",
        stack_page="robotwin",
        hint="RoboTwin：OpenPI + ALOHA / SAPIEN。",
        embodied_slug="robotwin-2",
        source_url="https://robotwin-platform.github.io/leaderboard",
        source_label="RoboTwin / EmbodiedBench",
        harness_image="robotwin",
        engine="sapien",
        robot_description="a dual-arm ALOHA robot workstation",
        camera_description="the third-person head / full scene camera",
        camera_key="full_image",
    ),
)

# Extended benches: share one GUI page + run_external_bench.py.
EXTENDED_BACKENDS: Tuple[SimBackendSpec, ...] = (
    SimBackendSpec(
        key="simpler",
        label="SimplerEnv",
        kind="extended",
        stack_page="extended",
        hint="SimplerEnv：Google Robot / WidowX real-to-sim（SAPIEN）。VLA 论文主对照之一。",
        embodied_slug="simpler-env",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/simpler-env",
        source_label="EmbodiedBench SimplerEnv",
        harness_image="simpler",
        default_roots=(
            f"{_LIY}/SimplerEnv",
            f"{_LIY}/simpler_env",
            f"{_LIY}/ManiSkill/simpler",
        ),
        install_hint=(
            "git clone https://github.com/simpler-env/SimplerEnv\n"
            "或用 AllenAI harness: docker pull ghcr.io/allenai/vla-evaluation-harness/simpler"
        ),
        engine="sapien",
        robot_description="a Google Robot or WidowX arm in SimplerEnv",
        camera_description="the third-person / wrist camera",
    ),
    SimBackendSpec(
        key="calvin",
        label="CALVIN",
        kind="extended",
        stack_page="extended",
        hint="CALVIN：语言条件长程多任务（Avg. Completed Tasks）。",
        embodied_slug="calvin",
        embodied_metric="Avg. Completed Tasks",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/calvin",
        source_label="EmbodiedBench CALVIN",
        harness_image="calvin",
        default_roots=(f"{_LIY}/calvin", f"{_LIY}/CALVIN"),
        install_hint="git clone https://github.com/mees/calvin.git",
        engine="pybullet",
        robot_description="a Franka Panda in CALVIN",
        camera_description="the static / gripper camera",
    ),
    SimBackendSpec(
        key="robocasa",
        label="RoboCasa",
        kind="extended",
        stack_page="extended",
        hint="RoboCasa：厨房家务操作（MuJoCo / robosuite）。",
        embodied_slug="robocasa",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/robocasa",
        source_label="EmbodiedBench RoboCasa",
        harness_image="robocasa",
        default_roots=(f"{_LIY}/robocasa", f"{_LIY}/RoboCasa"),
        install_hint="git clone https://github.com/robocasa/robocasa.git",
        engine="mujoco",
        robot_description="a Franka kitchen robot in RoboCasa",
        camera_description="the third-person kitchen camera",
    ),
    SimBackendSpec(
        key="robocasa365",
        label="RoboCasa365",
        kind="extended",
        stack_page="extended",
        hint="RoboCasa365：更大规模厨房场景套件。",
        embodied_slug="robocasa365",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/robocasa365",
        source_label="EmbodiedBench RoboCasa365",
        harness_image="robocasa365",
        default_roots=(f"{_LIY}/robocasa", f"{_LIY}/RoboCasa"),
        engine="mujoco",
        robot_description="a kitchen robot in RoboCasa365",
        camera_description="the third-person kitchen camera",
    ),
    SimBackendSpec(
        key="maniskill3",
        label="ManiSkill3",
        kind="extended",
        stack_page="extended",
        hint="ManiSkill3：SAPIEN GPU 并行仿真 + 任务套件（含 SimplerEnv GPU port）。",
        embodied_slug="maniskill3",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/maniskill3",
        source_label="EmbodiedBench ManiSkill3",
        harness_image="maniskill2",
        default_roots=(f"{_LIY}/ManiSkill", f"{_LIY}/ManiSkill3"),
        install_hint="pip install mani_skill  # or clone https://github.com/haosulab/ManiSkill",
        engine="sapien",
        robot_description="a robot in ManiSkill3",
        camera_description="the third-person / wrist camera",
    ),
    SimBackendSpec(
        key="maniskill2",
        label="ManiSkill2",
        kind="extended",
        stack_page="extended",
        hint="ManiSkill2：经典 SAPIEN 操作基准（VLA harness 仍广泛使用）。",
        embodied_slug="maniskill2",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/maniskill2",
        source_label="EmbodiedBench ManiSkill2",
        harness_image="maniskill2",
        default_roots=(f"{_LIY}/ManiSkill2", f"{_LIY}/ManiSkill"),
        engine="sapien",
        robot_description="a robot in ManiSkill2",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="metaworld",
        label="Meta-World",
        kind="extended",
        stack_page="extended",
        hint="Meta-World：50 个 MuJoCo 多任务 / meta-RL 操作任务。",
        embodied_slug="meta-world",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/meta-world",
        source_label="EmbodiedBench Meta-World",
        default_roots=(f"{_LIY}/Metaworld", f"{_LIY}/metaworld"),
        install_hint="pip install metaworld",
        engine="mujoco",
        robot_description="a Sawyer arm in Meta-World",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="rlbench",
        label="RLBench",
        kind="extended",
        stack_page="extended",
        hint="RLBench：CoppeliaSim 大规模视觉引导操作。",
        embodied_slug="rlbench",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/rlbench",
        source_label="EmbodiedBench RLBench",
        harness_image="rlbench",
        default_roots=(f"{_LIY}/RLBench", f"{_LIY}/rlbench"),
        install_hint="git clone https://github.com/stepjam/RLBench.git",
        engine="coppelia",
        robot_description="a Franka Panda in RLBench",
        camera_description="the front / wrist camera",
    ),
    SimBackendSpec(
        key="libero_plus",
        label="LIBERO-Plus",
        kind="extended",
        stack_page="extended",
        hint="LIBERO-Plus：视觉 / 物理 / 语言扰动下的鲁棒性扩展。",
        embodied_slug="libero-plus",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/libero-plus",
        source_label="EmbodiedBench LIBERO-Plus",
        harness_image="libero-plus",
        default_roots=(
            f"{_LIY}/LIBERO-plus",
            f"{_LIY}/libero-plus",
            f"{_LIY}/psi-lab/openpi/third_party/libero",
        ),
        engine="mujoco",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    SimBackendSpec(
        key="libero_pro",
        label="LIBERO-Pro",
        kind="extended",
        stack_page="extended",
        hint="LIBERO-Pro：更难的 LIBERO 协议变体。",
        embodied_slug="libero-pro",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/libero-pro",
        source_label="EmbodiedBench LIBERO-Pro",
        harness_image="libero-pro",
        engine="mujoco",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    SimBackendSpec(
        key="libero_mem",
        label="LIBERO-Mem",
        kind="extended",
        stack_page="extended",
        hint="LIBERO-Mem：强调记忆 / 长程依赖的 LIBERO 变体。",
        embodied_slug="libero-mem",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/libero-mem",
        source_label="EmbodiedBench LIBERO-Mem",
        harness_image="libero-mem",
        engine="mujoco",
        robot_description="a Franka single-arm robot",
        camera_description="the third-person agentview camera",
        camera_key="agentview_image",
    ),
    SimBackendSpec(
        key="behavior1k",
        label="BEHAVIOR-1K",
        kind="extended",
        stack_page="extended",
        hint="BEHAVIOR-1K：斯坦福家务长程 Challenge（OmniGibson）。",
        embodied_slug="behavior-1k",
        embodied_metric="Task Success Score",
        source_url="https://behavior.stanford.edu/challenge/index.html",
        source_label="BEHAVIOR Challenge",
        harness_image="behavior1k",
        default_roots=(f"{_LIY}/BEHAVIOR-1K", f"{_LIY}/OmniGibson"),
        install_hint="见 https://behavior.stanford.edu/ — 需 OmniGibson 许可",
        engine="omni",
        robot_description="a mobile manipulator in BEHAVIOR-1K",
        camera_description="the egocentric RGB-D camera",
    ),
    SimBackendSpec(
        key="furniturebench",
        label="FurnitureBench",
        kind="extended",
        stack_page="extended",
        hint="FurnitureBench：真实家具装配仿真。",
        embodied_slug="furniturebench",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/furniturebench",
        source_label="EmbodiedBench FurnitureBench",
        harness_image="furniturebench",
        default_roots=(f"{_LIY}/furniture-bench", f"{_LIY}/FurnitureBench"),
        install_hint="git clone https://github.com/clvrai/furniture-bench.git",
        engine="isaac",
        robot_description="a Franka assembling furniture",
        camera_description="the third-person / wrist camera",
    ),
    SimBackendSpec(
        key="vlabench",
        label="VLABench",
        kind="extended",
        stack_page="extended",
        hint="VLABench：面向 VLA 的综合操作评测套件。",
        embodied_slug="vlabench",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/vlabench",
        source_label="EmbodiedBench VLABench",
        harness_image="vlabench",
        default_roots=(f"{_LIY}/VLABench", f"{_LIY}/vlabench"),
        engine="mujoco",
        robot_description="a robot in VLABench",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="genesis",
        label="Genesis",
        kind="extended",
        stack_page="extended",
        hint="Genesis：多物理求解器仿真后端（软体/流体友好），非单一冻结榜。",
        source_url="https://genesis-embodied-ai.github.io/",
        source_label="Genesis docs",
        default_roots=(f"{_LIY}/Genesis", f"{_LIY}/genesis"),
        install_hint="pip install genesis-world  # https://github.com/Genesis-Embodied-AI/Genesis",
        engine="genesis",
        robot_description="a robot in Genesis",
        camera_description="the free camera",
    ),
    SimBackendSpec(
        key="mikasa",
        label="MIKASA-Robo",
        kind="extended",
        stack_page="extended",
        hint="MIKASA-Robo：较新的机器人操作评测套件。",
        embodied_slug="mikasa-robo",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/mikasa-robo",
        source_label="EmbodiedBench MIKASA-Robo",
        harness_image="mikasa-robo",
        engine="mujoco",
        robot_description="a robot in MIKASA-Robo",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="robocerebra",
        label="RoboCerebra",
        kind="extended",
        stack_page="extended",
        hint="RoboCerebra：认知 / 操作联合评测。",
        embodied_slug="robocerebra",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/robocerebra",
        source_label="EmbodiedBench RoboCerebra",
        harness_image="robocerebra",
        engine="mujoco",
        robot_description="a robot in RoboCerebra",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="robomme",
        label="RoboMME",
        kind="extended",
        stack_page="extended",
        hint="RoboMME：多模态具身评测。",
        embodied_slug="robomme",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/robomme",
        source_label="EmbodiedBench RoboMME",
        harness_image="robomme",
        engine="mujoco",
        robot_description="a robot in RoboMME",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="duobench",
        label="DuoBench",
        kind="extended",
        stack_page="extended",
        hint="DuoBench：双臂 / 双体操作评测。",
        embodied_slug="duobench",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/duobench",
        source_label="EmbodiedBench DuoBench",
        harness_image="duobench",
        engine="sapien",
        robot_description="a dual-arm robot in DuoBench",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="kinetix",
        label="Kinetix",
        kind="extended",
        stack_page="extended",
        hint="Kinetix：动力学 / 控制向仿真评测。",
        embodied_slug="kinetix",
        source_url="https://embodiedbench.vercel.app/en/benchmarks/kinetix",
        source_label="EmbodiedBench Kinetix",
        harness_image="kinetix",
        engine="mujoco",
        robot_description="a robot in Kinetix",
        camera_description="the third-person camera",
    ),
    SimBackendSpec(
        key="vla_harness",
        label="VLA Harness (总榜)",
        kind="extended",
        stack_page="extended",
        hint=(
            "AllenAI VLA Evaluation Harness：统一 Docker 评测 + 跨 bench 总榜。"
            "启动时打印 docker / uv 命令；本页榜单为跨基准聚合入口。"
        ),
        source_url="https://allenai.github.io/vla-evaluation-harness/leaderboard/",
        source_label="AllenAI VLA Leaderboard",
        install_hint=(
            "https://github.com/allenai/vla-evaluation-harness\n"
            "docker pull ghcr.io/allenai/vla-evaluation-harness/base"
        ),
        engine="",
        robot_description="a VLA policy under unified harness evaluation",
        camera_description="benchmark-specific cameras",
    ),
)

ALL_BACKENDS: Tuple[SimBackendSpec, ...] = NATIVE_BACKENDS + EXTENDED_BACKENDS

_BY_KEY: Dict[str, SimBackendSpec] = {b.key: b for b in ALL_BACKENDS}

# Aliases → canonical key
_ALIASES: Dict[str, str] = {
    "robodojo": "isaac",
    "spaces": "molmospaces",
    "simplerenv": "simpler",
    "simpler-env": "simpler",
    "simpler_env": "simpler",
    "meta-world": "metaworld",
    "meta_world": "metaworld",
    "mani_skill3": "maniskill3",
    "mani_skill2": "maniskill2",
    "maniskill": "maniskill3",
    "behavior": "behavior1k",
    "behavior-1k": "behavior1k",
    "behavior_1k": "behavior1k",
    "furniture": "furniturebench",
    "furniture-bench": "furniturebench",
    "libero-plus": "libero_plus",
    "libero-pro": "libero_pro",
    "libero-mem": "libero_mem",
    "mikasa-robo": "mikasa",
    "vla-harness": "vla_harness",
    "vla_eval": "vla_harness",
    "allenai": "vla_harness",
}


def normalize_backend_key(key: str) -> str:
    k = (key or "").strip().lower().replace(" ", "_")
    return _ALIASES.get(k, k)


def get_backend(key: str) -> Optional[SimBackendSpec]:
    return _BY_KEY.get(normalize_backend_key(key))


def list_backend_keys(*, kind: Optional[str] = None) -> List[str]:
    if kind is None:
        return [b.key for b in ALL_BACKENDS]
    return [b.key for b in ALL_BACKENDS if b.kind == kind]


def list_extended_backends() -> List[SimBackendSpec]:
    return list(EXTENDED_BACKENDS)


def embodiedbench_url(slug: str) -> str:
    return f"https://embodiedbench.vercel.app/en/benchmarks/{slug}"


def resolve_default_root(spec: SimBackendSpec) -> str:
    import os

    for cand in spec.default_roots:
        if cand and os.path.isdir(cand):
            return cand
    return spec.default_roots[0] if spec.default_roots else ""
