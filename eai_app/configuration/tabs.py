"""configuration.tabs — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_humanego_root(path: str = "") -> str:
    raw = (path or "").strip() or _eai_configuration_models.HUMANOGO_ROOT_DEFAULT
    return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))


def resolve_humanego_python(repo: str = "") -> str:
    for cand in _eai_configuration_models.HUMANOGO_PYTHON_CANDIDATES:
        if cand and _eai_platform_bootstrap.os.path.isfile(cand):
            return cand
    which = _eai_platform_bootstrap.shutil.which("python3") or _eai_platform_bootstrap.shutil.which("python") or _eai_platform_bootstrap.sys.executable
    return which or _eai_platform_bootstrap.sys.executable


LINGBOT_VISION_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python"
)


LINGBOT_VISION_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_lingbot_vision.sh")


LINGBOT_VISION_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "lingbot_vision")


LINGBOT_VISION_EXAMPLE = _eai_platform_bootstrap.os.path.join(
    _eai_configuration_models.LINGBOT_VISION_ROOT_DEFAULT, "examples", "example.png"
)


LINGBOT_DEPTH_ROOT_DEFAULT = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(
    _eai_platform_bootstrap.os.environ.get("LINGBOT_DEPTH_ROOT") or _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "..", "lingbot-depth")
))


LINGBOT_DEPTH_PYTHON_DEFAULT = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(
    _eai_platform_bootstrap.os.environ.get("LINGBOT_DEPTH_PYTHON") or _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "..", "envs", "lingbot-depth", "bin", "python")
))


LINGBOT_DEPTH_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_lingbot_depth.sh")


LINGBOT_DEPTH_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "lingbot_depth")


LINGBOT_DEPTH_MODEL_DEFAULT = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


LINGBOT_MAP_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/lingbot-map"
)


LINGBOT_MAP_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python"
)


LINGBOT_MAP_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_lingbot_map.sh")


LINGBOT_MAP_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "lingbot_map")


LINGBOT_MAP_MODEL_CACHE = _eai_platform_bootstrap.os.path.join(LINGBOT_MAP_CACHE_DIR, "models")


LINGBOT_MAP_CKPT_DEFAULT = "lingbot-map.pt"


LINGBOT_MAP_PORT_DEFAULT = 8080


LINGBOT_VIDEO_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/lingbot-video"
)


LINGBOT_VIDEO_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python"
)


LINGBOT_VIDEO_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_lingbot_video.sh")


LINGBOT_VIDEO_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "lingbot_video")


LINGBOT_VIDEO_MODEL_CACHE = _eai_platform_bootstrap.os.path.join(LINGBOT_VIDEO_CACHE_DIR, "models")


LINGBOT_VIDEO_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(LINGBOT_VIDEO_CACHE_DIR, "outputs")


LINGBOT_VIDEO_HF_DENSE = "robbyant/lingbot-video-dense-1.3b"


LINGBOT_WORLD_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/lingbot-world-v2"
)


LINGBOT_WORLD_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python"
)


LINGBOT_WORLD_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_lingbot_world.sh")


LINGBOT_WORLD_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "lingbot_world")


LINGBOT_WORLD_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(LINGBOT_WORLD_CACHE_DIR, "outputs")


RLINF_ROOT_DEFAULT = "/share_data/projects/mahjong/share/personal/liyichao/RLinf"


RLINF_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python",
    "/home/psibot/miniconda3/envs/RLinf/bin/python",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python",
    "/home/psibot/miniconda3/envs/eai/bin/python",
)


RLINF_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_reward_model.sh")


RLINF_REWARD_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "reward_model")


RLINF_REWARD_WORKFLOW_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_reward_workflow.sh")


RLINF_REWARD_WORKFLOW_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "reward_workflow")


ROBODOJO_ONLINE_RL_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_robodojo_online_rl.sh")


ROBODOJO_ONLINE_RL_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "robodojo_online_rl")


ROBOMETER_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/robometer-policy-learning"
)


ROBOMETER_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/robometer-policy/bin/python",
)


ROBOMETER_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_robometer_policy.sh")


ROBOMETER_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "robometer_policy")


ROBOMETER_JOBS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str], ...] = (
    ("在线 RL（train.py）", "train", "libero_online_rl"),
    ("DSRL（train_dsrl.py）", "train_dsrl", "dsrl_libero_config"),
    ("评测 Pi0（eval_pi0.py）", "eval_pi0", "eval_pi0"),
    ("评测 DSRL（eval_trained_dsrl.py）", "eval_trained_dsrl", "eval_trained_dsrl"),
    ("异步训练（train_async.py）", "train_async", "config_distributed"),
    ("Reward Relabel 服务", "relabel_server", "reward_relabel_server"),
)


RYNNVALUE_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/RynnValue"
)


RYNNVALUE_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/rynnvalue/bin/python",
)


RYNNVALUE_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_rynnvalue.sh")


RYNNVALUE_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "rynnvalue")


RYNNVALUE_MODEL_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/models/RynnValue-4B"
)


RYNNVALUE_CKPT_DEFAULT = _eai_platform_bootstrap.os.path.join(RYNNVALUE_CACHE_DIR, "checkpoint")


RYNNVALUE_EXAMPLE_VIDEO = _eai_platform_bootstrap.os.path.join(
    RYNNVALUE_ROOT_DEFAULT,
    "example",
    "Put_the_box_in_the_drawer_and_close_it.mp4",
)


RYNNVALUE_DEFAULT_INSTRUCTION = "Put the box in the drawer and close it"


RYNNVALUE_DEFAULT_ROBOT = "an ALOHA dual-arm robot"


RYNNVALUE_DEFAULT_CAMERA = "the top-down camera"


RYNNVALUE_JOBS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("视频推理（rynn_infer）", "infer"),
    ("Reward 服务（start_server）", "reward_server"),
    ("Policy Ranking 评测", "policy_ranking"),
    ("Confusion Matrix 评测", "confusion_matrix"),
)


ROBOICL_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/RoboICL"
)


ROBOICL_SIM_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/env_isaaclab/bin/python",
    "/home/psibot/miniconda3/envs/env_isaaclab/bin/python",
)


ROBOICL_POLICY_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/env_isaaclab/bin/python",
    "/home/psibot/miniconda3/envs/env_isaaclab/bin/python",
)


ROBOICL_RUN_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_roboicl.sh")


ROBOICL_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "roboicl")


ROBOICL_MODES: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("准备 Assets（软链本地）", "prepare_assets"),
    ("预检 dry-run（不启仿真）", "dry_run"),
    ("相机探针 capture-only", "capture_only"),
    ("完整 rollout", "run"),
)


ROBOICL_PROTOCOLS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, int], ...] = (
    ("Zero-shot B25", "configs/protocols/zero_shot_b25.json", 0),
    ("One-shot J12/B12", "configs/protocols/one_shot_j12_b12.json", 1),
)


MUJOCO_VIEWER_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "run_mujoco_viewer.sh")


MUJOCO_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/mujoco"
)


MOLMOSPACES_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/molmospaces"
)


PSI_MOLMOSPACES_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/psi-molmospaces"
)


MOLMOSPACES_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/molmospaces/bin/python"
)


MOLMOSPACES_RUN_RECIPES: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("评测所选任务", "eval_task"),
    ("快速演示 --viewer", "quick_viewer"),
    ("安装/更新依赖", "install_deps"),
    ("打开仓库目录", "open_dir"),
    ("在终端打开", "open_terminal"),
)


MOLMOSPACES_ROBOTS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("Franka Droid", "droid"),
    ("Franka", "franka"),
    ("Floating RUM", "rum"),
    ("RBY1 (导航)", "rby1"),
    ("I2RT YAM", "yam"),
    ("Bimanual YAM", "bimanual_yam"),
    ("Psi R1 (本地 MJCF)", "psi_r1"),
)


MOLMOSPACES_VIEWER_CAMS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("正面第三视角", "front"),
    ("头部第一视角", "head"),
)


MOLMOSPACES_VIEWER_FRONT_DISTANCE = "3.8"


MOLMOSPACES_VIEWER_FRONT_ELEVATION = "-20"


MOLMOSPACES_LOCAL_MJCF_ROBOTS: _eai_platform_bootstrap.Dict[str, str] = {
    "psi_r1": _eai_configuration_models.PSI_R1_MODEL_XML,
}


MOLMOSPACES_QUICK_VIEWER_TASK: _eai_platform_bootstrap.Dict[str, str] = {
    "rby1": "nav_to_obj",  # PickPlanner needs parallel_kinematics (RBY1: N/A)
}


MOLMOSPACES_EVAL_TASKS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str, _eai_platform_bootstrap.Tuple[str, ...]], ...] = (
    (
        "close_v1",
        "Close (Close-v1)",
        "benchmarks/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark",
        (),
    ),
    (
        "open_v1",
        "Open (Open-v1)",
        "benchmarks/molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark",
        (),
    ),
    (
        "pick_v1_1",
        "Pick (Pick-v1.1)",
        "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
        (),
    ),
    (
        "pnp_v1",
        "Pick and Place (PnP-v1)",
        "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark",
        (),
    ),
    (
        "pick_v1_5",
        "Pick-MSProc (Pick-v1.5)",
        "benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
        (),
    ),
    (
        "pick_v2_classic",
        "Pick-Classic (Pick-v2-classic)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        (),
    ),
    (
        "pick_v2_filament",
        "Pick-Filament (Pick-v2-filament)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        ("--use-filament",),
    ),
    (
        "pick_v2_rand_cam",
        "Pick-RandCam (Pick-v2-rand-cam)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        (
            "--use-filament",
            "--camera_names",
            "randomized_zed2_analogue_1",
            "wrist_camera_zed_mini",
        ),
    ),
    (
        "pnp_v2",
        "Pick & Place (PnP-v2)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark",
        ("--use-filament",),
    ),
    (
        "pnp_next_to_v2",
        "Pick & Place-NextTo (PnP-next-to-v2)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark",
        ("--use-filament",),
    ),
    (
        "pnp_color_v2",
        "Pick & Place-Color (PnP-color-v2)",
        "benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark",
        ("--use-filament",),
    ),
)


MOLMOSPACES_EVAL_POLICIES: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    (
        "Pi",
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig",
    ),
    (
        "Psi Policy",
        _eai_adapters_models_discovery.PSI_POLICY_EVAL_ID,
    ),
    (
        "Psi Policy Native6D",
        _eai_adapters_models_discovery.PSI_NATIVE6D_EVAL_ID,
    ),
    (
        "WAM Psi-R1 Remote",
        _eai_adapters_models_discovery.WAM_PSI_R1_EVAL_ID,
    ),
    (
        "Psi R1 Pick Planner",
        "molmo_spaces.evaluation.configs.evaluation_configs:PsiR1PickPlannerEvalConfig",
    ),
    (
        "LingBot-VA (WAM serve)",
        "lingbot_va",
    ),
    (
        "VPP2 (WAM serve)",
        "vpp2",
    ),
    (
        "Dummy（无策略）",
        "molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig",
    ),
    (
        "Teleop",
        "molmo_spaces.evaluation.configs.evaluation_configs:TeleopPolicyEvalConfig",
    ),
)


PSI_REMOTE_CKPT_DEFAULT = _eai_adapters_models_discovery.resolve_default_wam_finetune_checkpoint()


MOLMOSPACES_EVAL_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "molmospaces_eval")


PSI_MOLMOSPACES_EVAL_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "psi_molmospaces_eval")


PSI_MOLMOSPACES_DEFAULT_POLICY = _eai_adapters_models_discovery.WAM_PSI_R1_EVAL_ID


PSI_POLICY_SERVER_PORT_DEFAULT = int(
    _eai_platform_bootstrap.os.environ.get("PSI_POLICY_SERVER_PORT") or _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_PI_PORT") or 8080
)


PSI_POLICY_SERVER_SCRIPT = _eai_platform_bootstrap.os.path.join(
    "psi-policy-server", "scripts", "serve_policy.py"
)


MOLMOSPACES_PI_CKPT_DEFAULT = _eai_configuration_models.resolve_pi_checkpoint(prefer_present_local=True).path


MOLMOSPACES_PI_CONFIG_DEFAULT = _eai_configuration_models.infer_policy_config(
    MOLMOSPACES_PI_CKPT_DEFAULT, _eai_configuration_models.PI_CKPT_CONFIG_DEFAULT
)


MOLMOSPACES_PI_PORT_DEFAULT = int(
    _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_PI_PORT") or _eai_configuration_models.PI_SERVER_PORT_DEFAULT
)


ISAACLAB_ARENA_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/IsaacLab-Arena"
)


ISAACLAB_ARENA_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/IsaacLab-Arena/bin/python",
)


ISAACLAB_ARENA_POLICIES: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("zero_action（零动作）", "zero_action"),
    ("replay", "replay"),
    ("rsl_rl", "rsl_rl"),
    ("LingBot-VA (WAM serve)", "lingbot_va"),
    ("VPP2 (WAM serve)", "vpp2"),
)


LIBERO_CKPT_DEFAULT = _eai_configuration_models.resolve_libero_checkpoint().path


LIBERO_CONFIG_DEFAULT = "pi05_libero"


ROBOTWIN_CKPT_DEFAULT = _eai_configuration_models.resolve_robotwin_checkpoint().path


ROBOTWIN_CONFIG_DEFAULT = "pi05_aloha_robotwin"


ROBOTWIN_TASK_DEFAULT = "adjust_bottle"


MOLMOSPACES_RESOURCES_MIN = "0.0.3a2"


MUJOCO_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/molmospaces/bin/python",
)


MUJOCO_MODEL_PRESETS: _eai_platform_bootstrap.Tuple[str, ...] = (
    _eai_configuration_models.PSI_R1_MODEL_XML,
    "humanoid/humanoid.xml",
    "welcome/welcome.xml",
    "car/car.xml",
    "cube/cube_3x3x3.xml",
    "mug/mug.xml",
    "hammock/hammock.xml",
    "tendon_arm/arm26.xml",
)


RLINF_TELEOP_SCRIPT = _eai_platform_bootstrap.os.path.join(
    RLINF_ROOT_DEFAULT, "examples", "reward", "run_realworld_teleop.sh"
)


LINGBOT_WORLD_CKPT_DEFAULT = _eai_platform_bootstrap.os.path.join(
    LINGBOT_WORLD_ROOT_DEFAULT, "lingbot-world-v2-1.3b-causal-fast"
)


LINGBOT_WORLD_PROMPT_DEFAULT = (
    "A serene lakeside scene with a lone tree standing in calm water, "
    "surrounded by distant snow-capped mountains under a bright blue sky "
    "with drifting white clouds."
)


CONTROL_MISC_TAB_TITLE = "其他"


_EAI_CONTROL_TAB_MIME = "application/x-eai-control-tab"


_EAI_CONTROL_TAB_SRC_MIME = "application/x-eai-control-tab-src"


CONTROL_TAB_TITLES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "大脑",
    "回放",
    "分割",
    "视觉基础模型",
    "空间感知模型",
    "3D重建模型",
    "视频生成模型",
    "世界模型",
    "CAD",
    "训练",
    "手臂/手",
    "手骨架遥控",
    "仿真评测",
    "真机评测",
    "本地摄像头",
    "Reward评测",
    "Reward训练",
    "仿真强化学习训练",
    "RoboMeter",
    "RynnValue",
    "RoboICL",
    "ICL",
    "Astra",
    "PhysicalRSI",
    "EmbodiedEvo",
    "RAX",
    "ASPIRE",
    "FLORA",
    "编程冲SOTA",
    "主机资源",
    "HumanEgo",
    "数据集",
    "sub task",
    "sub image",
    CONTROL_MISC_TAB_TITLE,
)


SIM_BACKEND_SELECTOR_SPECS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str], ...] = (
    ("isaac", "Isaac / RoboDojo", "sim_backend_isaac_radio"),
    ("mujoco", "MuJoCo", "sim_backend_mujoco_radio"),
    ("molmospaces", "MolmoSpaces", "sim_backend_spaces_radio"),
    ("psi_molmospaces", "Psi-MolmoSpaces", "sim_backend_psi_spaces_radio"),
    ("arena", "IsaacLab-Arena", "sim_backend_arena_radio"),
    ("libero", "LIBERO", "sim_backend_libero_radio"),
    ("robotwin", "RoboTwin", "sim_backend_robotwin_radio"),
    ("extended", "扩展评测", "sim_backend_extended_radio"),
)


SIM_BACKEND_ORDER_KEYS: _eai_platform_bootstrap.Tuple[str, ...] = tuple(
    k for k, _label, _attr in SIM_BACKEND_SELECTOR_SPECS
)


ARM_PANEL_SELECTOR_SPECS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str], ...] = (
    ("human_demo", "人类演示模仿", "arm_panel_human_demo_radio"),
    ("mv6d", "多视角6D位姿", "arm_panel_mv6d_radio"),
)


ARM_PANEL_ORDER_KEYS: _eai_platform_bootstrap.Tuple[str, ...] = tuple(
    k for k, _label, _attr in ARM_PANEL_SELECTOR_SPECS
)


CONTROL_TAB_ALIASES: _eai_platform_bootstrap.Dict[str, str] = {
    "brain": "大脑",
    "camera": "大脑",
    "大脑": "大脑",
    "replay": "回放",
    "robot": "回放",
    "回放": "回放",
    "segment": "分割",
    "sam": "分割",
    "分割": "分割",
    "vision": "视觉基础模型",
    "lingbot-vision": "视觉基础模型",
    "视觉基础模型": "视觉基础模型",
    "depth": "空间感知模型",
    "空间感知模型": "空间感知模型",
    "map": "3D重建模型",
    "3d": "3D重建模型",
    "3d重建模型": "3D重建模型",
    "3D重建模型": "3D重建模型",
    "video": "视频生成模型",
    "视频生成模型": "视频生成模型",
    "world": "世界模型",
    "世界模型": "世界模型",
    "bagel": "sub image",
    "Bagel": "sub image",
    "BAGEL": "sub image",
    "sub image": "sub image",
    "subimage": "sub image",
    "sub-image": "sub image",
    "cad": "CAD",
    "CAD": "CAD",
    "train": "训练",
    "训练": "训练",
    "arm": "手臂/手",
    "hand": "手臂/手",
    "手臂/手": "手臂/手",
    "手臂·手": "手臂/手",
    "human_demo": "手臂/手",
    "demo": "手臂/手",
    "人类演示模仿": "手臂/手",
    "mv6d": "手臂/手",
    "6d": "手臂/手",
    "pose6d": "手臂/手",
    "多视角6D位姿": "手臂/手",
    "多视角 6D 位姿": "手臂/手",
    "skeleton": "手骨架遥控",
    "手骨架遥控": "手骨架遥控",
    "test": "sub task",
    "测试": "sub task",
    "sub task": "sub task",
    "subtask": "sub task",
    "sub-task": "sub task",
    "sim": "仿真评测",
    "sim_eval": "仿真评测",
    "仿真评测": "仿真评测",
    "real": "真机评测",
    "real_eval": "真机评测",
    "真机评测": "真机评测",
    "local_cam": "本地摄像头",
    "local_webcam": "本地摄像头",
    "webcam": "本地摄像头",
    "cam_forward": "本地摄像头",
    "本地摄像头": "本地摄像头",
    "reward": "Reward评测",
    "rm": "Reward评测",
    "reward_eval": "Reward评测",
    "reward_model": "Reward评测",
    "Reward评测": "Reward评测",
    "reward_train": "Reward训练",
    "reward_training": "Reward训练",
    "reward_workflow": "Reward训练",
    "rm_train": "Reward训练",
    "Reward训练": "Reward训练",
    "sim_rl_train": "仿真强化学习训练",
    "sim_online_rl_train": "仿真强化学习训练",
    "仿真强化学习训练": "仿真强化学习训练",
    "robodojo_rl": "仿真强化学习训练",
    "robodojo_online": "仿真强化学习训练",
    "robodojo_online_rl": "仿真强化学习训练",
    "rise_online": "仿真强化学习训练",
    "rise_rl": "仿真强化学习训练",
    "dojo_rl": "仿真强化学习训练",
    "RoboDojo在线RL": "仿真强化学习训练",
    "robometer": "RoboMeter",
    "RoboMeter": "RoboMeter",
    "rfm_rl": "RoboMeter",
    "policy_learning": "RoboMeter",
    "robometer-policy": "RoboMeter",
    "rynnvalue": "RynnValue",
    "RynnValue": "RynnValue",
    "rynn_value": "RynnValue",
    "rynn-value": "RynnValue",
    "rynn": "RynnValue",
    "roboicl": "RoboICL",
    "RoboICL": "RoboICL",
    "robo_icl": "RoboICL",
    "robo-icl": "RoboICL",
    "gpt6_astra": "RoboICL",
    "mujoco": "仿真评测",
    "MuJoCo": "仿真评测",
    "mj": "仿真评测",
    "simulate": "仿真评测",
    "molmospaces": "仿真评测",
    "MolmoSpaces": "仿真评测",
    "psi_molmospaces": "仿真评测",
    "Psi-MolmoSpaces": "仿真评测",
    "PsiMolmoSpaces": "仿真评测",
    "spaces": "仿真评测",
    "arena": "仿真评测",
    "IsaacLab-Arena": "仿真评测",
    "isaaclab_arena": "仿真评测",
    "libero": "仿真评测",
    "LIBERO": "仿真评测",
    "robotwin": "仿真评测",
    "RoboTwin": "仿真评测",
    "robo_twin": "仿真评测",
    "rlinf": "仿真评测",
    "simpler": "仿真评测",
    "SimplerEnv": "仿真评测",
    "calvin": "仿真评测",
    "CALVIN": "仿真评测",
    "robocasa": "仿真评测",
    "maniskill3": "仿真评测",
    "maniskill2": "仿真评测",
    "metaworld": "仿真评测",
    "rlbench": "仿真评测",
    "genesis": "仿真评测",
    "behavior1k": "仿真评测",
    "furniturebench": "仿真评测",
    "vlabench": "仿真评测",
    "vla_harness": "仿真评测",
    "扩展评测": "仿真评测",
    "extended": "仿真评测",
    "ctx": "ICL",
    "context": "ICL",
    "icl": "ICL",
    "ICL": "ICL",
    "上下文学习": "ICL",
    "astra": "Astra",
    "Astra": "Astra",
    "physicalrsi": "PhysicalRSI",
    "physical_rsi": "PhysicalRSI",
    "physical-rsi": "PhysicalRSI",
    "rsi": "PhysicalRSI",
    "PhysicalRSI": "PhysicalRSI",
    "embodiedevo": "EmbodiedEvo",
    "embodied_evo": "EmbodiedEvo",
    "embodied-evo": "EmbodiedEvo",
    "eevo": "EmbodiedEvo",
    "EmbodiedEvo": "EmbodiedEvo",
    "rax": "RAX",
    "RAX": "RAX",
    "robot-agent-exploration": "RAX",
    "aspire": "ASPIRE",
    "ASPIRE": "ASPIRE",
    "cap-harness": "ASPIRE",
    "cap_harness": "ASPIRE",
    "flora": "FLORA",
    "FLORA": "FLORA",
    "Flora": "FLORA",
    "flora-tempseg": "FLORA",
    "prog_sota": "编程冲SOTA",
    "progsota": "编程冲SOTA",
    "sota": "编程冲SOTA",
    "code_sota": "编程冲SOTA",
    "编程冲sota": "编程冲SOTA",
    "编程冲SOTA": "编程冲SOTA",
    "system": "主机资源",
    "sysmon": "主机资源",
    "host": "主机资源",
    "gpu": "主机资源",
    "cpu": "主机资源",
    "monitor": "主机资源",
    "system_monitor": "主机资源",
    "主机资源": "主机资源",
    "gpt-6": "Astra",
    "gpt6": "Astra",
    "robodojo": "Astra",
    "llm-as-policies": "Astra",
    "llm_as_policies": "Astra",
    "humanego": "HumanEgo",
    "HumanEgo": "HumanEgo",
    "human-ego": "HumanEgo",
    "human_ego": "HumanEgo",
    "dataset": "数据集",
    "datasets": "数据集",
    "data": "数据集",
    "数据集": "数据集",
    "misc": CONTROL_MISC_TAB_TITLE,
    "other": CONTROL_MISC_TAB_TITLE,
    "others": CONTROL_MISC_TAB_TITLE,
    "其他": CONTROL_MISC_TAB_TITLE,
}


CONTROL_TAB_PREFERRED_ALIAS: _eai_platform_bootstrap.Dict[str, str] = {
    "大脑": "brain",
    "回放": "replay",
    "分割": "segment",
    "视觉基础模型": "vision",
    "空间感知模型": "depth",
    "3D重建模型": "map",
    "视频生成模型": "video",
    "世界模型": "world",
    "CAD": "cad",
    "训练": "train",
    "手臂/手": "arm",
    "手骨架遥控": "skeleton",
    "仿真评测": "sim",
    "真机评测": "real",
    "本地摄像头": "webcam",
    "Reward评测": "reward",
    "Reward训练": "reward_train",
    "仿真强化学习训练": "dojo_rl",
    "RoboMeter": "robometer",
    "RynnValue": "rynn",
    "RoboICL": "roboicl",
    "ICL": "icl",
    "Astra": "astra",
    "PhysicalRSI": "rsi",
    "EmbodiedEvo": "eevo",
    "RAX": "rax",
    "ASPIRE": "aspire",
    "FLORA": "flora",
    "编程冲SOTA": "sota",
    "主机资源": "system",
    "HumanEgo": "humanego",
    "数据集": "dataset",
    "sub task": "test",
    "sub image": "bagel",
    CONTROL_MISC_TAB_TITLE: "misc",
}


def _shell_tab_token(token: str) -> str:
    """CLI 里对 --tab 参数加引号（含空格/中文时）。"""
    t = (token or "").strip()
    if not t:
        return '""'
    if _eai_platform_bootstrap.re.fullmatch(r"[A-Za-z0-9_./+-]+", t):
        return t
    return _eai_platform_bootstrap.shlex.quote(t)


def control_tab_cli_aliases(title: str) -> _eai_platform_bootstrap.List[str]:
    """某 tab 可用于 --tab 的别名列表（首选短英文在前）。"""
    title = (title or "").strip()
    if not title:
        return []
    seen = set()
    ordered: _eai_platform_bootstrap.List[str] = []

    def _push(name: str) -> None:
        n = (name or "").strip()
        if not n or n in seen:
            return
        seen.add(n)
        ordered.append(n)

    pref = CONTROL_TAB_PREFERRED_ALIAS.get(title, "")
    if pref:
        _push(pref)
    ascii_rest: _eai_platform_bootstrap.List[str] = []
    other_rest: _eai_platform_bootstrap.List[str] = []
    for alias, dest in CONTROL_TAB_ALIASES.items():
        if dest != title or alias == title:
            continue
        if alias.isascii() and " " not in alias:
            ascii_rest.append(alias)
        else:
            other_rest.append(alias)
    ascii_rest.sort(key=lambda s: (len(s), s.lower()))
    for alias in ascii_rest:
        _push(alias)
    for alias in other_rest:
        _push(alias)
    _push(title)
    return ordered


def format_control_tab_launch_tooltip(title: str) -> str:
    """鼠标悬停 tab 时展示的启动提示。"""
    title = (title or "").strip()
    if not title or title == "…":
        return "正在构建控制区…"
    if title == CONTROL_MISC_TAB_TITLE:
        return (
            f"「{CONTROL_MISC_TAB_TITLE}」\n"
            "\n"
            "将任意顶栏标签拖到本标签上，即可收入为子标签。\n"
            "子标签可在页内拖动排序，也可拖回顶栏。\n"
            "\n"
            "布局写入 eai/control_tabs_order.json（order + misc_children）"
        )
    aliases = control_tab_cli_aliases(title)
    primary = aliases[0] if aliases else title
    tok = _shell_tab_token(primary)
    alias_line = " / ".join(aliases[:6]) if aliases else title
    return (
        f"「{title}」\n"
        "\n"
        "启动时只显示本页:\n"
        f"  bash run_local.sh --tab {tok}\n"
        f"  bash run_in_docker.sh --tab {tok}\n"
        f"  python3 show_camera_topics.py --tab {tok}\n"
        "\n"
        f"别名: {alias_line}\n"
        f"环境变量: EAI_ONLY_TAB={primary}\n"
        "\n"
        "可拖动标签调整顺序；拖到「其他」可收入为子标签\n"
        "（写入 eai/control_tabs_order.json）"
    )

