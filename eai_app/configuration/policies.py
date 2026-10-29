"""configuration.policies — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

CAD_MESHES_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "meshes")


TEST_IMAGES_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "images")


ISAAC_CAM_BRIDGE_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "scripts/launch/run_isaac_cam_bridge.sh")


ISAAC_CAM_BRIDGE_DIR_DEFAULT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "isaac_cam_bridge")


ISAAC_CAM_BRIDGE_KEYS: _eai_platform_bootstrap.Tuple[str, ...] = (
    "cam_head",
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


try:
    from eai_app.adapters.simulation.camera_sink import (
        clear_gui_robot_cmd,
        clear_sim_control,
        is_sim_paused,
        read_cam_extrinsic,
        read_sim_robot_state,
        sim_state_is_fresh,
        write_gui_robot_cmd,
        write_sim_control,
    )
except ImportError:  # pragma: no cover
    clear_gui_robot_cmd = None  # type: ignore[assignment]
    clear_sim_control = None  # type: ignore[assignment]
    is_sim_paused = None  # type: ignore[assignment]
    read_cam_extrinsic = None  # type: ignore[assignment]
    read_sim_robot_state = None  # type: ignore[assignment]
    sim_state_is_fresh = None  # type: ignore[assignment]
    write_gui_robot_cmd = None  # type: ignore[assignment]
    write_sim_control = None  # type: ignore[assignment]


_SIM_TOPIC_TO_CAM_KEY: _eai_platform_bootstrap.Dict[str, str] = {
    "/camera/head_color": "cam_head",
    "/camera/head_depth": "cam_head",
    "/camera/left_wrist_color": "cam_left_wrist",
    "/camera/left_wrist_depth": "cam_left_wrist",
    "/camera/right_wrist_color": "cam_right_wrist",
    "/camera/right_wrist_depth": "cam_right_wrist",
    # A2D 真机常用 hand_* 命名（与 left_wrist_* 等价）
    "/camera/hand_left_color": "cam_left_wrist",
    "/camera/hand_left_depth": "cam_left_wrist",
    "/camera/hand_right_color": "cam_right_wrist",
    "/camera/hand_right_depth": "cam_right_wrist",
}


ESSENTIAL_A2D_CAMERA_COLOR_TOPICS = frozenset({
    "/camera/head_color",
    "/camera/hand_left_color",
    "/camera/hand_right_color",
    "/camera/left_wrist_color",
    "/camera/right_wrist_color",
})


WRIST_COLOR_TOPIC_ALIASES: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[str, ...]] = {
    "left": ("/camera/left_wrist_color", "/camera/hand_left_color"),
    "right": ("/camera/right_wrist_color", "/camera/hand_right_color"),
}


WRIST_DEPTH_TOPIC_ALIASES: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[str, ...]] = {
    "left": ("/camera/left_wrist_depth", "/camera/hand_left_depth"),
    "right": ("/camera/right_wrist_depth", "/camera/hand_right_depth"),
}


_SIM_FRAME_TO_CAM_KEY: _eai_platform_bootstrap.Dict[str, str] = {
    "camera_frame": "cam_head",
    "isaac_head_camera_frame": "cam_head",
    "head_camera_depth_link": "cam_head",
    "left_wrist_camera_frame": "cam_left_wrist",
    "isaac_left_wrist_camera_frame": "cam_left_wrist",
    "right_wrist_camera_frame": "cam_right_wrist",
    "isaac_right_wrist_camera_frame": "cam_right_wrist",
}


SIM_TELEOP_STATE_MAX_AGE_S = 2.5


SIM_ARM_MOVE_TICK_MS = 50


SIM_PREVIEW_CAM_TOPICS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("cam_left_wrist", "/camera/left_wrist_color"),
    ("cam_head", "/camera/head_color"),
    ("cam_right_wrist", "/camera/right_wrist_color"),
)


SIM_PREVIEW_DEPTH_CAM_TOPICS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("cam_left_wrist", "/camera/left_wrist_depth"),
    ("cam_head", "/camera/head_depth"),
    ("cam_right_wrist", "/camera/right_wrist_depth"),
)


SIM_PREVIEW_TOPICS: _eai_platform_bootstrap.Tuple[str, ...] = tuple(topic for _, topic in SIM_PREVIEW_CAM_TOPICS)


SIM_PREVIEW_DEPTH_TOPICS: _eai_platform_bootstrap.Tuple[str, ...] = tuple(
    topic for _, topic in SIM_PREVIEW_DEPTH_CAM_TOPICS
)


SIM_PREVIEW_ALL_TOPICS: _eai_platform_bootstrap.Tuple[str, ...] = SIM_PREVIEW_TOPICS + SIM_PREVIEW_DEPTH_TOPICS


RYNNVALUE_LIVE_HUD_TOPIC = "/rynnvalue/live_hud"


LIBERO_PREVIEW_TOPIC = "/libero/preview"


RYNNVALUE_LIVE_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "rynnvalue_live")


LIBERO_PREVIEW_JPG = _eai_platform_bootstrap.os.path.join(RYNNVALUE_LIVE_CACHE_DIR, "libero_preview.jpg")


ROBODOJO_EVAL_RESULT_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/eval_result/RoboDojo"
)


ROBODOJO_ROOT_DEFAULT = "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo"


ROBODOJO_POLICY_PORT_DEFAULT = int(
    _eai_platform_bootstrap.os.environ.get("ROBODOJO_POLICY_PORT", "18765") or "18765"
)


ROBODOJO_POLICY_HOST_DEFAULT = (
    _eai_platform_bootstrap.os.environ.get("ROBODOJO_POLICY_HOST", "127.0.0.1").strip() or "127.0.0.1"
)


RISE_ROOT_DEFAULT = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_ROOT_DEFAULT, "XPolicyLab", "policy", "RISE", "RISE"
)


RISE_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/rise/bin/python",
)


RISE_POLICY_CKPT_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_droid_jointpos_polaris",
    "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_droid_jointpos",
    _eai_platform_bootstrap.os.path.join(
        "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo",
        "XPolicyLab",
        "policy",
        "RISE",
        "weights",
        "pi05_base_pytorch",
    ),
)


RISE_REWARD_CKPT_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    # 优先独立 value/reward 训练产物；本机若无则回退到策略 ckpt
    "/share_data/projects/mahjong/share/personal/liyichao/models/value_release",
    "/share_data/projects/mahjong/share/personal/liyichao/models/rise_value_release",
    _eai_platform_bootstrap.os.path.join(
        ROBODOJO_ROOT_DEFAULT,
        "XPolicyLab",
        "policy",
        "RISE",
        "checkpoints",
        "value_release",
    ),
)


ROBODOJO_ENV_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RoboDojo"
)


ROBODOJO_STARVLA_ENV_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/starvla"
)


ROBODOJO_LINGBOT_VA_ENV_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/lingbot_va"
)


ROBODOJO_OPENPI_ENV_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/openpi"
)


ROBODOJO_VPP2_ENV_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/vpp2"
)


def _is_pi05_robodojo_policy(policy_dir: str) -> bool:
    """True for XPolicyLab Pi_05 / Pi_0 / Pi_0_Fast policy dirs."""
    raw = (policy_dir or "").strip().replace("\\", "/")
    if not raw:
        return False
    base = _eai_platform_bootstrap.os.path.basename(raw.rstrip("/")).lower().replace("-", "_")
    return base in ("pi_05", "pi_0", "pi_0_fast", "pi05", "pi0") or "/pi_05" in raw.lower()


def _robodojo_policy_ckpt_dir(policy_rel: str, ckpt_name: str) -> str:
    """Absolute checkpoints/<ckpt_name> under an XPolicyLab policy dir."""
    return _eai_platform_bootstrap.os.path.join(
        ROBODOJO_ROOT_DEFAULT,
        policy_rel.replace("\\", "/").strip("/"),
        "checkpoints",
        (ckpt_name or "").strip(),
    )


PI05_ROBODOJO_CKPT_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "PI05_ROBODOJO_CKPT",
    _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "openpi", "hf", "pi05_droid_jointpos"),
)


def _ckpt_owned_by_lingbot_va_not_pi05(ckpt_name: str) -> bool:
    """True when ckpt exists under LingBot_VA but not under Pi_05 (by name or path)."""
    name = (ckpt_name or "").strip()
    if not name or name == "demo":
        return False
    if _eai_platform_bootstrap.os.path.isabs(name) and _eai_platform_bootstrap.os.path.isdir(name):
        # Absolute OpenPI dirs are Pi_05-compatible, never LingBot leftovers.
        return False
    va = _robodojo_policy_ckpt_dir(_eai_configuration_models.LINGBOT_VA_POLICY_REL, name)
    pi = _robodojo_policy_ckpt_dir("XPolicyLab/policy/Pi_05", name)
    return _eai_platform_bootstrap.os.path.exists(va) and not _eai_platform_bootstrap.os.path.exists(pi)


def _ckpt_dir_abspath(path: str) -> str:
    """Absolute directory path for UI display (resolves symlinks when possible)."""
    raw = _eai_platform_bootstrap.os.path.expanduser((path or "").strip())
    if not raw:
        return ""
    if _eai_platform_bootstrap.os.path.isfile(raw):
        parent = _eai_platform_bootstrap.os.path.dirname(raw)
        if _eai_platform_bootstrap.os.path.basename(parent) == "checkpoints":
            raw = _eai_platform_bootstrap.os.path.dirname(parent)
        else:
            raw = parent
    if _eai_platform_bootstrap.os.path.isdir(raw):
        try:
            return _eai_platform_bootstrap.os.path.realpath(raw)
        except OSError:
            return _eai_platform_bootstrap.os.path.abspath(raw)
    return _eai_platform_bootstrap.os.path.abspath(raw)


def _default_pi05_ckpt_text() -> str:
    """Full OpenPI checkpoint directory for the ckpt text box."""
    prefer = (PI05_ROBODOJO_CKPT_DEFAULT or "").strip()
    if prefer and _eai_platform_bootstrap.os.path.isdir(prefer):
        return _ckpt_dir_abspath(prefer)
    linked = _robodojo_policy_ckpt_dir(
        "XPolicyLab/policy/Pi_05", "pi05_droid_jointpos"
    )
    if _eai_platform_bootstrap.os.path.isdir(linked):
        return _ckpt_dir_abspath(linked)
    return _ckpt_dir_abspath(prefer or linked)


def _default_lingbot_va_ckpt_text() -> str:
    """Full LingBot_VA checkpoint directory for the ckpt text box."""
    name = (
        _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_CKPT", "RoboDojo-cotrain-arx_x5-joint-0").strip()
        or "RoboDojo-cotrain-arx_x5-joint-0"
    )
    if _eai_platform_bootstrap.os.path.isabs(name) and _eai_platform_bootstrap.os.path.isdir(name):
        return _ckpt_dir_abspath(name)
    path = _robodojo_policy_ckpt_dir(_eai_configuration_models.LINGBOT_VA_POLICY_REL, name)
    if _eai_platform_bootstrap.os.path.isdir(path):
        return _ckpt_dir_abspath(path)
    return _ckpt_dir_abspath(path)


ROBODOJO_STARVLA_HF_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo_cache/starvla_hf"
)


ROBODOJO_STARVLA_BASE_VLM_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo_cache/models/"
    "Qwen3-VL-4B-Instruct"
)


ROBODOJO_GUI_EVAL_SCRIPT = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_ROOT_DEFAULT, "scripts", "run_gui_eval.sh"
)


ROBODOJO_GUI_STARVLA_SCRIPT = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_ROOT_DEFAULT, "scripts", "run_gui_starvla_pi_v3.sh"
)


ROBODOJO_STARVLA_HF_EVAL_SCRIPT = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_ROOT_DEFAULT,
    "XPolicyLab",
    "policy",
    "starVLA",
    "scripts",
    "eval_hf_robodojo.sh",
)


ROBODOJO_TASK_CONFIG_DIR = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_ROOT_DEFAULT, "task", "RoboDojo", "config"
)


ROBODOJO_EVAL_NUM_KEEP = "keep"


STARVLA_CKPT_LABEL_DEFAULT = "hf_qwenpi_v3"


STARVLA_PI_V3_RUN_DIR_DEFAULT = _eai_platform_bootstrap.os.path.join(
    ROBODOJO_STARVLA_HF_ROOT_DEFAULT, "pi_v3"
)


STARVLA_PI_V3_CKPT_FILE_DEFAULT = _eai_platform_bootstrap.os.path.join(
    STARVLA_PI_V3_RUN_DIR_DEFAULT, "checkpoints", "steps_100000_pytorch_model.pt"
)


def _default_starvla_ckpt_text() -> str:
    """Full starVLA run directory (contains config.yaml + checkpoints/)."""
    env = _eai_platform_bootstrap.os.environ.get("STARVLA_CKPT", "").strip()
    if env:
        expanded = _ckpt_dir_abspath(env)
        if _eai_platform_bootstrap.os.path.isdir(expanded):
            return expanded
    if _eai_platform_bootstrap.os.path.isdir(STARVLA_PI_V3_RUN_DIR_DEFAULT):
        return _ckpt_dir_abspath(STARVLA_PI_V3_RUN_DIR_DEFAULT)
    linked = _robodojo_policy_ckpt_dir(
        "XPolicyLab/policy/starVLA", STARVLA_CKPT_LABEL_DEFAULT
    )
    if _eai_platform_bootstrap.os.path.isdir(linked):
        return _ckpt_dir_abspath(linked)
    return _ckpt_dir_abspath(STARVLA_PI_V3_RUN_DIR_DEFAULT)


def _looks_like_starvla_ckpt(ckpt: str) -> bool:
    """True when ckpt is a starVLA label or path into starvla_hf / policy checkpoints."""
    raw = (ckpt or "").strip()
    if not raw:
        return False
    low = raw.lower().replace("\\", "/")
    if raw in (STARVLA_CKPT_LABEL_DEFAULT, "pi_v3", "oft", "groot"):
        return True
    if "qwenpi" in low or "starvla_hf" in low or "/starvla/" in low:
        return True
    if _eai_platform_bootstrap.os.path.isfile(raw) and low.endswith((".pt", ".safetensors")):
        parent = _eai_platform_bootstrap.os.path.dirname(raw).replace("\\", "/").lower()
        return "starvla" in parent or parent.endswith("/checkpoints")
    if _eai_platform_bootstrap.os.path.isdir(raw):
        return _eai_platform_bootstrap.os.path.isfile(
            _eai_platform_bootstrap.os.path.join(raw, "checkpoints", "steps_100000_pytorch_model.pt")
        ) or _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(raw, "config.yaml"))
    return False


def resolve_starvla_hf_variant(ckpt: str) -> str:
    """Map UI ckpt label → eval_hf_robodojo variant (oft|groot|pi_v3)."""
    key = (ckpt or "").strip().lower().replace("-", "_")
    if not key or key in ("demo", "default", STARVLA_CKPT_LABEL_DEFAULT.lower()):
        return "pi_v3"
    if "oft" in key and "pi05" not in key:
        return "oft"
    if "groot" in key or "gr00t" in key:
        return "groot"
    # Prefer explicit starVLA names; do not treat OpenPI pi05_* paths as pi_v3.
    if "qwenpi" in key or "pi_v3" in key or key == "pi_v3":
        return "pi_v3"
    if key.startswith("pi_") or "/pi_v3" in key.replace("\\", "/"):
        return "pi_v3"
    return "pi_v3"


def resolve_starvla_ckpt_file(variant: str = "pi_v3", ckpt: str = "") -> str:
    """Locate materialized HF .pt under STARVLA_HF_ROOT/<variant>/checkpoints/.

    Also accepts an absolute ``.pt`` / run-dir path via ``ckpt`` or
    ``STARVLA_CKPT_PATH``.
    """
    env_override = _eai_platform_bootstrap.os.environ.get("STARVLA_CKPT_PATH", "").strip()
    if env_override and _eai_platform_bootstrap.os.path.isfile(env_override):
        return _eai_platform_bootstrap.os.path.abspath(env_override)
    raw = (ckpt or "").strip()
    if raw:
        if _eai_platform_bootstrap.os.path.isfile(raw) and raw.lower().endswith((".pt", ".safetensors")):
            return _eai_platform_bootstrap.os.path.abspath(raw)
        if _eai_platform_bootstrap.os.path.isdir(raw):
            preferred = _eai_platform_bootstrap.os.path.join(raw, "checkpoints", "steps_100000_pytorch_model.pt")
            if _eai_platform_bootstrap.os.path.isfile(preferred):
                return _eai_platform_bootstrap.os.path.abspath(preferred)
            # run dir may already be .../checkpoints
            preferred2 = _eai_platform_bootstrap.os.path.join(raw, "steps_100000_pytorch_model.pt")
            if _eai_platform_bootstrap.os.path.isfile(preferred2):
                return _eai_platform_bootstrap.os.path.abspath(preferred2)
    # Policy-local symlink: checkpoints/hf_qwenpi_v3 → starvla_hf/pi_v3
    linked = _robodojo_policy_ckpt_dir(
        "XPolicyLab/policy/starVLA", STARVLA_CKPT_LABEL_DEFAULT
    )
    linked_pt = _eai_platform_bootstrap.os.path.join(linked, "checkpoints", "steps_100000_pytorch_model.pt")
    if _eai_platform_bootstrap.os.path.isfile(linked_pt):
        return _eai_platform_bootstrap.os.path.abspath(linked_pt)
    if _eai_platform_bootstrap.os.path.isfile(STARVLA_PI_V3_CKPT_FILE_DEFAULT):
        return STARVLA_PI_V3_CKPT_FILE_DEFAULT
    hf_root = _eai_platform_bootstrap.os.environ.get(
        "STARVLA_HF_ROOT", ROBODOJO_STARVLA_HF_ROOT_DEFAULT
    ).strip() or ROBODOJO_STARVLA_HF_ROOT_DEFAULT
    ckpt_dir = _eai_platform_bootstrap.os.path.join(hf_root, variant, "checkpoints")
    if not _eai_platform_bootstrap.os.path.isdir(ckpt_dir):
        return ""
    preferred = _eai_platform_bootstrap.os.path.join(ckpt_dir, "steps_100000_pytorch_model.pt")
    if _eai_platform_bootstrap.os.path.isfile(preferred):
        return preferred
    candidates: _eai_platform_bootstrap.List[str] = []
    try:
        for name in _eai_platform_bootstrap.os.listdir(ckpt_dir):
            lower = name.lower()
            if lower.endswith((".pt", ".safetensors")):
                candidates.append(_eai_platform_bootstrap.os.path.join(ckpt_dir, name))
    except OSError:
        return ""
    if not candidates:
        return ""
    candidates.sort()
    return candidates[-1]


WORKSPACE_DIR = _eai_platform_bootstrap.os.path.dirname(_eai_configuration_models.EAI_DIR)


LOCAL_QWEN_MODELS_ROOT = _eai_platform_bootstrap.os.path.join(WORKSPACE_DIR, "models", "Qwen")


LAKE_QWEN35_OUTPUT_ROOT = (
    "/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/output"
)


REMOTE_LAKE_QWEN_PYTHON = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python"
)


LOCAL_QWEN_DEPLOY_MODELS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str, str], ...] = (
    ("qwen3.5-4b", "Qwen3.5-4B", "qwen3.5-4b", "Qwen3.5-4B"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
    (
        "lake-qwen35-4b-lora",
        "Lake Qwen3.5-4B LoRA",
        "lake-qwen35-4b-lora",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake",
    ),
    (
        "lake-qwen35-4b-lora-smoke",
        "Lake Qwen3.5-4B LoRA (smoke)",
        "lake-qwen35-4b-lora-smoke",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake-smoke",
    ),
)


LOCAL_QWEN_MODEL_DIR_DEFAULT = _eai_platform_bootstrap.os.path.join(
    LOCAL_QWEN_MODELS_ROOT, LOCAL_QWEN_DEPLOY_MODELS[0][3]
)


LOCAL_QWEN_OLLAMA_NAME = _eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_OLLAMA_NAME", "qwen3.5-4b-local")


LOCAL_QWEN_OLLAMA_FALLBACK = _eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_OLLAMA_FALLBACK", "qwen3.5:4b")


LOCAL_QWEN_RUN_SCRIPT_NAME = "scripts/launch/run_local_qwen.sh"


LOCAL_QWEN_PORT_DEFAULT = int(_eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_PORT", "8100"))


LOCAL_QWEN_CTL_PORT_DEFAULT = int(_eai_platform_bootstrap.os.environ.get("LOCAL_QWEN_CTL_PORT", "18101"))


LOCAL_QWEN_CTL_URL_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "LOCAL_QWEN_CTL_URL", f"http://127.0.0.1:{LOCAL_QWEN_CTL_PORT_DEFAULT}"
)


LOCAL_QWEN_MODEL_ID = _eai_platform_bootstrap.os.environ.get(
    "LOCAL_QWEN_MODEL_ID", LOCAL_QWEN_DEPLOY_MODELS[0][2]
)


LOCAL_QWEN_API_BASE_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "LOCAL_QWEN_API_BASE", f"http://127.0.0.1:{LOCAL_QWEN_PORT_DEFAULT}/v1"
)


LOCAL_QWEN_PYTHON_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "LOCAL_QWEN_PYTHON",
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/eai/bin/python",
)


LOCAL_QWEN_CHAT_PRESET_NAME = "本地 Qwen 服务"


REMOTE_QWEN_HOSTS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("psi_motus_2_for_liyichao", "远程 psi_motus（8×A800）"),
    ("tione-develop", "远程 tione-develop（7×A800）"),
    (
        "nb-1668047664206989312-cp0nfah4b37k",
        "远程 nb-cp0nfah4b37k（1×80GB）",
    ),
)


REMOTE_QWEN_HOST_LOCAL_PORTS: _eai_platform_bootstrap.Dict[str, int] = {
    "psi_motus_2_for_liyichao": 18100,
    "tione-develop": 18102,
    "nb-1668047664206989312-cp0nfah4b37k": 18104,
}


REMOTE_QWEN_SSH_HOST = _eai_platform_bootstrap.os.environ.get("REMOTE_QWEN_SSH_HOST", "psi_motus_2_for_liyichao")


REMOTE_QWEN_LOCAL_PORT = int(_eai_platform_bootstrap.os.environ.get("REMOTE_QWEN_LOCAL_PORT", "18100"))


REMOTE_QWEN_API_BASE_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "REMOTE_QWEN_API_BASE", f"http://127.0.0.1:{REMOTE_QWEN_LOCAL_PORT}/v1"
)


REMOTE_QWEN_CHAT_PRESET_NAME = "远程 Qwen 服务"


REMOTE_QWEN_CTL_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "compat/remote_qwen_ctl.py")


REMOTE_QWEN_HOSTCTL_PORT = int(_eai_platform_bootstrap.os.environ.get("REMOTE_QWEN_CTL_PORT", "18103"))


REMOTE_QWEN_HOSTCTL_URL = _eai_platform_bootstrap.os.environ.get(
    "REMOTE_QWEN_CTL_URL", f"http://127.0.0.1:{REMOTE_QWEN_HOSTCTL_PORT}"
).rstrip("/")


CAD_PHOTOGRAMMETRY_PY = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "compat/photogrammetry_reconstruct.py")


CAD_TARGET_EXTENT_DEFAULT_M = 0.15


CAD_POISSON_DEPTH_DEFAULT = 9


CAD_MIN_IMAGES_DEFAULT = 8


CAD_CAPTURE_COUNT_DEFAULT = 12


CAD_CAPTURE_TOPIC_DEFAULT = "/camera/head_color"


CAD_CAPTURE_DEPTH_DEFAULT = "/camera/head_depth"


CAD_CAPTURE_TOPIC_CANDIDATES = (
    "/camera/head_color",
    "/camera/head_rgb",
    "/camera/head/color",
)


CAD_CAPTURE_MIN_DIFF_CORR = 0.97  # 几乎不动才拒绝；头部相机转物通常 corr≈0.87~0.95


CAD_CAPTURE_MIN_DIFF_MAE = 3.0


PSI_POLICY_DIR_ENV = "PSI_POLICY_DIR"


def resolve_cad_mesh_python() -> str:
    env = _eai_platform_bootstrap.os.environ.get("MESH_PYTHON", "").strip()
    if env and _eai_platform_bootstrap.os.path.isfile(env) and _eai_platform_bootstrap.os.access(env, _eai_platform_bootstrap.os.X_OK):
        return env
    candidates = [
        "/opt/miniconda3/envs/foundationpose/bin/python",
        _eai_platform_bootstrap.os.path.expanduser("~/miniconda3/envs/foundationpose/bin/python"),
        _eai_platform_bootstrap.os.path.expanduser("~/anaconda3/envs/foundationpose/bin/python"),
        "/home/psibot/miniconda3/envs/foundationpose/bin/python",
    ]
    for path in candidates:
        if _eai_platform_bootstrap.os.path.isfile(path) and _eai_platform_bootstrap.os.access(path, _eai_platform_bootstrap.os.X_OK):
            return path
    return _eai_platform_bootstrap.shutil.which("python3") or "python3"


def list_reconstructed_meshes(meshes_dir: _eai_platform_bootstrap.Optional[str] = None) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]]:
    """返回 [(显示名, reconstructed.obj 绝对路径), ...]，按修改时间新→旧。"""
    root = meshes_dir or CAD_MESHES_DIR
    if not _eai_platform_bootstrap.os.path.isdir(root):
        return []
    items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[float, str, str]] = []
    for name in _eai_platform_bootstrap.os.listdir(root):
        obj = _eai_platform_bootstrap.os.path.join(root, name, "reconstructed.obj")
        if _eai_platform_bootstrap.os.path.isfile(obj):
            items.append((_eai_platform_bootstrap.os.path.getmtime(obj), name, _eai_platform_bootstrap.os.path.abspath(obj)))
    items.sort(key=lambda x: x[0], reverse=True)
    return [(name, path) for _, name, path in items]


PSI_POLICY_CONFIG_DEFAULT = "example_workspace_imle_rgb"


PSI_POLICY_LOGGING_MODES = ("offline", "online", "disabled")


MAX_GL_SEGMENT_POINTS = 4000


UI_IMAGE_MIN_INTERVAL_S = 1.0 / 5.0


UI_DEPTH_MIN_INTERVAL_S = 1.0 / 2.0


UI_ROBOT_STATE_MIN_INTERVAL_S = 0.5


UI_PREVIEW_MAX_EDGE = 640


UI_DEFAULT_COLOR_TOPIC_SUFFIXES: tuple = ()


UI_TF_LOOKUP_TIMEOUT_S = 0.0


MOVE_TF_LOOKUP_TIMEOUT_S = 1.0


IK_TARGET_FRAME = "base_link"


MINK_FK_FRAME_ALIASES = frozenset({"map", "world"})


SIM_FINGER_MID_OFFSET_IN_EE_M: _eai_platform_bootstrap.Tuple[float, float, float] = (0.08657, 0.0, 0.0)


SIM_ARM_SETTLE_TOL_M = 0.020


SIM_ARM_SETTLE_MAX_S = 8.0


MV6D_MIN_FUSED_POINTS = 80


MV6D_GRASP_Z_LIFT_FRAC = 0.0


MV6D_FILL_GT_MAX_DIST_M = 0.28


MV6D_EST_GT_SNAP_MAX_DIST_M = 0.12


MV6D_WRIST_DEPTH_INVALID_MAX_RATIO = 0.10


MV6D_FUSE_WEIGHT_HEAD = 3.0


MV6D_FUSE_WEIGHT_WRIST = 1.0


MV6D_FUSE_XY_RADIUS_M = 0.11


MV6D_FUSE_TABLE_CLEAR_M = 0.014


SIM_ARM_APPROACH_MIN_DZ_M = 0.03


ROBOT_ARM_TOPIC = "/hal/arm_joint_state"


ROBOT_SERVICE_SCRIPT = "start_ros_service.sh"


STACK_SERVICES_SCRIPT = "nodes/start_arm_hand_services.sh"


STACK_SERVICES_LAUNCH = "arm_hand_services.launch.py"


STACK_SERVICES_LOG = "/var/psi/log/arm_hand_services/latest.log"


LEGACY_BASE_SERVICES_LAUNCH = "base_services.launch.py"


REAL_ROBOT_BACKEND_A2D = "a2d"


REAL_ROBOT_BACKEND_R1 = "r1"


REAL_ROBOT_BACKEND_FRANKA = "franka"


REAL_ROBOT_BACKEND_LABELS = {
    REAL_ROBOT_BACKEND_A2D: "A2D",
    REAL_ROBOT_BACKEND_R1: "R1",
    REAL_ROBOT_BACKEND_FRANKA: "Franka",
}


REAL_ROBOT_EXTERNAL_STACK_BACKENDS = frozenset(
    {REAL_ROBOT_BACKEND_R1, REAL_ROBOT_BACKEND_FRANKA}
)


REAL_ROBOT_SELECTOR_SPECS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str, str], ...] = (
    (REAL_ROBOT_BACKEND_A2D, "A2D", "real_robot_a2d_radio"),
    (REAL_ROBOT_BACKEND_R1, "R1", "real_robot_r1_radio"),
    (REAL_ROBOT_BACKEND_FRANKA, "Franka", "real_robot_franka_radio"),
)


REAL_ROBOT_ORDER_KEYS: _eai_platform_bootstrap.Tuple[str, ...] = tuple(
    k for k, _label, _attr in REAL_ROBOT_SELECTOR_SPECS
)


A2D_DOCKER_CONTAINER_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "A2D_DOCKER_CONTAINER", "a2d-tele-release-2-1-0rc3-latest"
).strip() or "a2d-tele-release-2-1-0rc3-latest"


A2D_PSI_POLICY_ROOT_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "A2D_PSI_POLICY_ROOT",
    "/home/psibot/ceyao_dev/psi-policy-v2",
).strip() or "/home/psibot/ceyao_dev/psi-policy-v2"


A2D_POLICY_CLIENT_DIR_DEFAULT = _eai_platform_bootstrap.os.path.join(
    A2D_PSI_POLICY_ROOT_DEFAULT, "model_inference_client_cena"
)


A2D_CKPT_PATH_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "A2D_CKPT_PATH",
    "/home/psibot/ceyao_dev/psi-policy-v2/outputs/2026-09-17/"
    "18-45-07-pbp-generalization-abs-sh-dc12-hil-teleop-ac48-a2d-act-rgb-3view-proprio/"
    "checkpoints/last/last-step=348000-val_mse=0.008769.ckpt",
).strip()


A2D_RRD_PATH_DEFAULT = _eai_platform_bootstrap.os.environ.get(
    "A2D_RRD_PATH",
    "/root/project/rerun_data/episode_00001.rrd",
).strip() or "/root/project/rerun_data/episode_00001.rrd"


A2D_OPS_DOC_URL = (
    "https://test-d4btraxpj8kh.feishu.cn/wiki/QuFswA02fiHTrfkctIdcmpc7nPh"
)


A2D_HMI_CALIB_TIP = (
    "浏览器打开 HMI → 运维管理 → 本体参数设置 → 头部俯仰角度改为 5° → "
    "点击「真机测试」。标定 Pixel_err 要求 < 3px（文字为绿色）。"
)


ROBOT_FK_JOINTS_TOPIC = "/fk/joint_states"


ROBOT_LEFT_HAND_TOPIC = "/ry_hand/left/joint_states"


ROBOT_RIGHT_HAND_TOPIC = "/ry_hand/right/joint_states"


ROBOT_LEFT_TCP_TOPIC = "/mink_fk/left_tcp_pose"


ROBOT_RIGHT_TCP_TOPIC = "/mink_fk/right_tcp_pose"


ROBOT_TELE_LEFT_TCP_TOPIC = "/tele/fk/left_pose"


ROBOT_TELE_RIGHT_TCP_TOPIC = "/tele/fk/right_pose"


ROBOT_TCP_SOURCE_TOPICS = {
    # mink_fk 优先；tele/fk 常有空端点，放到后面按需订阅
    "left": (ROBOT_LEFT_TCP_TOPIC, ROBOT_TELE_LEFT_TCP_TOPIC),
    "right": (ROBOT_RIGHT_TCP_TOPIC, ROBOT_TELE_RIGHT_TCP_TOPIC),
}


ROBOT_LEFT_HAND_CMD_TOPIC = "/ry_hand/left/set_angles"


ROBOT_RIGHT_HAND_CMD_TOPIC = "/ry_hand/right/set_angles"


WBC_LEFT_EE_CMD_TOPIC = "/wbc/left_ee_command"


WBC_RIGHT_EE_CMD_TOPIC = "/wbc/right_ee_command"


HAND_CMD_RELAY_HOST = _eai_platform_bootstrap.os.environ.get("HAND_CMD_RELAY_HOST", "127.0.0.1")


HAND_CMD_RELAY_PORT = int(_eai_platform_bootstrap.os.environ.get("HAND_CMD_RELAY_PORT", "19283"))


HAND_CMD_RELAY_ENABLED = _eai_platform_bootstrap.os.environ.get("HAND_CMD_RELAY", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)


BASE_LINK_FRAME = "base_link"


LEFT_IK_TARGET_TOPIC = "/ik/left_target"


RIGHT_IK_TARGET_TOPIC = "/ik/right_target"


MODEL_LEFT_TCP_TOPIC = "/model/tcp/left/target_pose"


MODEL_RIGHT_TCP_TOPIC = "/model/tcp/right/target_pose"


WBC_TARGET_JOINTS_TOPIC = "/wbc/target_joints"


CONTROL_MODE_TOPIC = "/control_mode"


ARM_ENABLE_STATE_TOPIC = "/arm/enable_state"


ARM_SET_ENABLE_SERVICE = "/arm/set_enable"


ARM_RESET_TO_INITIAL_SERVICE = "/arm/reset_to_initial"


HAND_ENABLE_STATE_TOPIC = "/hand/enable_state"


HAND_SET_ENABLE_SERVICE = "/hand/set_enable"


REPLAY_START_SERVICE = "/rrd_replay/start_replay"


REPLAY_STOP_SERVICE = "/rrd_replay/stop_replay"


REPLAY_RUNNING_STATE_TOPIC = "/rrd_replay/running_state"


REPLAY_LOOP_COUNT_DEFAULT = 1


REPLAY_LOOP_COUNT_MAX = 99


CTX_DEFAULT_VIDEO_PATH = (
    "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/eval_result/"
    "RoboDojo/pack_objects_into_box/starVLA/arx_x5/"
    "0_ckpt_name=hf_qwenpi_v3,action_type=joint/"
    "gui_starvla_20260907_122225_pack_objects_into_box/"
    "episode_0000000_cam_head_fail.mp4"
)


HUMAN_DEMO_CACHE_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "human_demo")


HUMAN_DEMO_VIDEO_FILTER = (
    "视频 (*.mp4 *.avi *.mkv *.webm *.mov);;所有文件 (*)"
)


REPLAY_STATE_FINISHED = 3


REPLAY_STATE_LABELS = {
    0: "空闲",
    1: "回放中",
    2: "等待条件",
    3: "已完成",
    4: "已停止",
}


UI_TEXT_PRIMARY = "#ececec"


UI_TEXT_SECONDARY = "#c8c8c8"


UI_TEXT_MUTED = "#b0b0b0"


UI_TEXT_PLACEHOLDER = "#9a9a9a"


UI_ACCENT_BLUE = "#7ec8ff"


UI_ACCENT_BLUE_BRIGHT = "#4da3ff"


UI_ACCENT_ORANGE = "#ffb86c"


UI_ACCENT_GREEN = "#50fa7b"


UI_ACCENT_RED = "#ff8888"


UI_BG_WINDOW = "#1e1f22"


UI_BG_PANEL = "#252628"


UI_BG_ELEVATED = "#2c2e32"


UI_BG_INPUT = "#1a1b1e"


UI_BG_HOVER = "#34363c"


UI_BG_LOG = "#1a1b1e"


UI_BORDER = "#3d4048"


UI_BORDER_STRONG = "#52565f"


UI_HIGHLIGHT = "#3d6ea8"


UI_RADIUS = "4px"


UI_MONO_FAMILY = "Monospace"


UI_MONO_SIZE_SMALL = 9


UI_MONO_SIZE_NORMAL = 10


UI_MONO_SIZE_TITLE = 11

