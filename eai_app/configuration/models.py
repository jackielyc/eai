"""configuration.models — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

FP_MESH_DEFAULT = _eai_configuration_perception.resolve_fp_mesh_path()


EAI_DIR = str(PROJECT_ROOT)


_EAI_TOOLS_DIR = _eai_platform_bootstrap.os.path.join(EAI_DIR, "tools")


if _EAI_TOOLS_DIR not in _eai_platform_bootstrap.sys.path:
    _eai_platform_bootstrap.sys.path.insert(0, _EAI_TOOLS_DIR)


if EAI_DIR not in _eai_platform_bootstrap.sys.path:
    _eai_platform_bootstrap.sys.path.insert(0, EAI_DIR)


try:
    from eai_app.ui.simulation.thumbnails import (  # noqa: E402
        bind_eval_task_thumb,
        combo_task_thumb,
        make_eval_task_preview,
        stamp_eval_task_combo,
    )
except ImportError:  # pragma: no cover
    def combo_task_thumb(_combo, _index):  # type: ignore[misc]
        return None

    def stamp_eval_task_combo(_combo) -> None:  # type: ignore[misc]
        return None

    def bind_eval_task_thumb(_combo, _preview) -> None:  # type: ignore[misc]
        return None

    def make_eval_task_preview(_parent=None):  # type: ignore[misc]
        wrap = _eai_platform_bootstrap.QWidget()
        return wrap, _eai_platform_bootstrap.QLabel(), _eai_platform_bootstrap.QLabel()


try:
    from eai_app.ui.simulation.leaderboard import (  # noqa: E402
        board_for_backend as load_sim_eval_board,
        board_for_display as load_sim_eval_board_display,
        clear_leaderboard_cache as clear_sim_eval_leaderboard_cache,
        entry_from_manual as sim_eval_entry_from_manual,
        entry_from_result_json as sim_eval_entry_from_result_json,
        merge_local_entry as merge_sim_eval_local_entry,
        refresh_backend as refresh_sim_eval_backend,
        remove_local_entry as remove_sim_eval_local_entry,
    )
except ImportError:  # pragma: no cover
    def load_sim_eval_board(_backend: str, **_k):  # type: ignore[misc]
        return None

    def load_sim_eval_board_display(_backend: str, **_k):  # type: ignore[misc]
        return None

    def clear_sim_eval_leaderboard_cache() -> None:  # type: ignore[misc]
        return None

    def refresh_sim_eval_backend(_backend: str, **_k):  # type: ignore[misc]
        raise RuntimeError("sim_eval_leaderboard 未找到")

    def sim_eval_entry_from_result_json(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("sim_eval_leaderboard 未找到")

    def sim_eval_entry_from_manual(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("sim_eval_leaderboard 未找到")

    def merge_sim_eval_local_entry(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("sim_eval_leaderboard 未找到")

    def remove_sim_eval_local_entry(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("sim_eval_leaderboard 未找到")


try:
    from eai_app.application.simulation.catalog import (  # noqa: E402
        get_backend as get_sim_backend_spec,
        list_extended_backends,
        normalize_backend_key as normalize_sim_backend_key,
        resolve_default_root as resolve_extended_default_root,
    )
except ImportError:  # pragma: no cover
    def get_sim_backend_spec(_key: str):  # type: ignore[misc]
        return None

    def list_extended_backends():  # type: ignore[misc]
        return []

    def normalize_sim_backend_key(key: str) -> str:  # type: ignore[misc]
        return (key or "").strip().lower()

    def resolve_extended_default_root(_spec):  # type: ignore[misc]
        return ""


try:
    from external_bench_runtime import (  # noqa: E402
        build_eval_argv as build_external_bench_argv,
        install_hint as external_bench_install_hint,
        list_extended_choices,
        probe_deps as probe_external_bench_deps,
        resolve_python as resolve_external_bench_python,
        resolve_root as resolve_external_bench_root,
    )
except ImportError:  # pragma: no cover
    def build_external_bench_argv(**_k):  # type: ignore[misc]
        raise RuntimeError("external_bench_runtime 未找到")

    def external_bench_install_hint(_backend: str, _python_bin: str = "") -> str:  # type: ignore[misc]
        return "external_bench_runtime 未找到"

    def list_extended_choices():  # type: ignore[misc]
        return []

    def probe_external_bench_deps(_backend: str, **_k):  # type: ignore[misc]
        return False, "external_bench_runtime 未找到"

    def resolve_external_bench_python(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or _eai_platform_bootstrap.sys.executable

    def resolve_external_bench_root(_backend: str, prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or ""


try:
    from pi_policy_runtime import (  # noqa: E402
        OPENPI_DATA_HOME_DEFAULT,
        PI_CKPT_CONFIG_DEFAULT,
        PI_CKPT_GS_DEFAULT,
        PI_CKPT_HF_LOCAL_DEFAULT,
        PI_SERVER_HOST_DEFAULT,
        PI_SERVER_PORT_DEFAULT,
        build_serve_policy_argv,
        checkpoint_status,
        ensure_checkpoint_materialized,
        infer_policy_config,
        is_local_checkpoint_usable,
        is_pi_policy_id,
        port_listening,
        probe_pi_server,
        warmup_pi_server_infer,
        pi_server_matches,
        inspect_pi_server_on_port,
        resolve_pi_checkpoint,
        stop_pi_server_on_port,
    )
except ImportError:  # pragma: no cover - keep UI importable if tools missing
    OPENPI_DATA_HOME_DEFAULT = _eai_platform_bootstrap.os.path.join(EAI_DIR, ".cache", "openpi")
    PI_CKPT_CONFIG_DEFAULT = "pi05_droid_jointpos"
    PI_CKPT_GS_DEFAULT = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    PI_CKPT_HF_LOCAL_DEFAULT = _eai_platform_bootstrap.os.path.join(
        EAI_DIR, ".cache", "openpi", "hf", "pi05_droid_jointpos"
    )
    PI_SERVER_HOST_DEFAULT = "localhost"
    PI_SERVER_PORT_DEFAULT = 8080

    def build_serve_policy_argv(**_kwargs):  # type: ignore[misc]
        raise RuntimeError("pi_policy_runtime 未找到")

    def checkpoint_status(path: str) -> str:  # type: ignore[misc]
        return path or ""

    def ensure_checkpoint_materialized(path: str, **_k):  # type: ignore[misc]
        return False, "pi_policy_runtime 未找到"

    def infer_policy_config(path: str, override: str = "") -> str:  # type: ignore[misc]
        return (override or PI_CKPT_CONFIG_DEFAULT).strip() or PI_CKPT_CONFIG_DEFAULT

    def is_local_checkpoint_usable(_path: str) -> bool:  # type: ignore[misc]
        return False

    def is_pi_policy_id(policy_id: str) -> bool:  # type: ignore[misc]
        return "PiPolicyEvalConfig" in (policy_id or "")

    def port_listening(_host: str, _port: int, timeout: float = 0.4) -> bool:  # type: ignore[misc]
        return False

    def probe_pi_server(*_a, **_k):  # type: ignore[misc]
        return False, "pi_policy_runtime 未找到"

    def warmup_pi_server_infer(*_a, **_k):  # type: ignore[misc]
        return False, "pi_policy_runtime 未找到"

    def pi_server_matches(**_k):  # type: ignore[misc]
        return True, "fallback"

    def inspect_pi_server_on_port(port: int = 8080):  # type: ignore[misc]
        return {"pid": "", "config": "", "dir": "", "cmdline": ""}

    def resolve_pi_checkpoint(prefer: str = "", **_k):  # type: ignore[misc]
        class _Ckpt:
            path = prefer or PI_CKPT_GS_DEFAULT
            config = PI_CKPT_CONFIG_DEFAULT
            source = "fallback"

        return _Ckpt()

    def stop_pi_server_on_port(port: int = 8080, **_k):  # type: ignore[misc]
        return f"pi_policy_runtime 未找到 (port={port})"


try:
    from libero_policy_runtime import (  # noqa: E402
        LIBERO_CKPT_GS_DEFAULT,
        LIBERO_EVAL_OUTPUT_DIR,
        LIBERO_HOST_DEFAULT,
        LIBERO_POLICY_CONFIGS,
        LIBERO_PORT_DEFAULT,
        LIBERO_TASK_SUITES,
        build_libero_eval_argv,
        install_hint as libero_install_hint,
        probe_libero_deps,
        resolve_libero_checkpoint,
        resolve_libero_python,
    )
except ImportError:  # pragma: no cover
    LIBERO_CKPT_GS_DEFAULT = "gs://openpi-assets/checkpoints/pi05_libero"
    LIBERO_CKPT_HF_LOCAL_DEFAULT = (
        "/share_data/projects/mahjong/share/personal/liyichao/models/pi05_libero"
    )
    LIBERO_EVAL_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(EAI_DIR, ".cache", "libero_eval")
    LIBERO_HOST_DEFAULT = "localhost"
    LIBERO_PORT_DEFAULT = 8080
    LIBERO_POLICY_CONFIGS = (
        ("π0.5-LIBERO (pi05_libero)", "pi05_libero"),
        ("π0-LIBERO (pi0_libero)", "pi0_libero"),
        ("LingBot-VA (WAM / wan_va)", "lingbot_va"),
        ("VPP2 (WAM)", "vpp2"),
    )
    LIBERO_TASK_SUITES = (
        ("libero_spatial", "libero_spatial"),
        ("libero_object", "libero_object"),
        ("libero_goal", "libero_goal"),
        ("libero_10 (long)", "libero_10"),
        ("libero_90", "libero_90"),
    )

    def build_libero_eval_argv(**_kwargs):  # type: ignore[misc]
        raise RuntimeError("libero_policy_runtime 未找到")

    def libero_install_hint(_python_bin: str = "") -> str:  # type: ignore[misc]
        return "libero_policy_runtime 未找到"

    def probe_libero_deps(_python_bin: str, **_k):  # type: ignore[misc]
        return False, "libero_policy_runtime 未找到"

    def resolve_libero_checkpoint(prefer: str = "", config: str = ""):  # type: ignore[misc]
        class _Ckpt:
            path = prefer or LIBERO_CKPT_HF_LOCAL_DEFAULT
            config = config or "pi05_libero"
            source = "fallback"

        return _Ckpt()

    def resolve_libero_python(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or "python3"


try:
    from robotwin_policy_runtime import (  # noqa: E402
        ROBOTWIN_CKPT_HF_DEFAULT,
        ROBOTWIN_EVAL_OUTPUT_DIR,
        ROBOTWIN_POLICY_CONFIGS,
        ROBOTWIN_ROOT_DEFAULT,
        ROBOTWIN_TASKS,
        build_robotwin_eval_argv,
        build_robotwin_viewer_argv,
        ensure_robotwin_checkpoint,
        install_hint as robotwin_install_hint,
        probe_robotwin_deps,
        resolve_robotwin_assets,
        resolve_robotwin_checkpoint,
        resolve_robotwin_python,
        resolve_robotwin_root,
    )
except ImportError:  # pragma: no cover
    ROBOTWIN_CKPT_HF_DEFAULT = "RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
    ROBOTWIN_EVAL_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(EAI_DIR, ".cache", "robotwin_eval")
    ROBOTWIN_ROOT_DEFAULT = "/share_data/projects/mahjong/share/personal/liyichao/RoboTwin"
    ROBOTWIN_POLICY_CONFIGS = (
        ("π0.5 Aloha (pi05_aloha_robotwin)", "pi05_aloha_robotwin"),
        ("π0 Aloha (pi0_aloha_robotwin)", "pi0_aloha_robotwin"),
        ("LingBot-VA (WAM / wan_va)", "lingbot_va"),
        ("VPP2 (WAM serve)", "vpp2"),
    )
    ROBOTWIN_TASKS = (
        ("adjust_bottle", "adjust_bottle"),
        ("place_empty_cup", "place_empty_cup"),
    )

    def build_robotwin_eval_argv(**_kwargs):  # type: ignore[misc]
        raise RuntimeError("robotwin_policy_runtime 未找到")

    def build_robotwin_viewer_argv(**_kwargs):  # type: ignore[misc]
        raise RuntimeError("robotwin_policy_runtime 未找到")

    def ensure_robotwin_checkpoint(prefer: str = "", config: str = "", **_k):  # type: ignore[misc]
        path = prefer or (
            "/share_data/projects/mahjong/share/personal/liyichao/models/"
            "RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
        )
        if _eai_platform_bootstrap.os.path.isdir(path):
            return True, path
        return False, "robotwin_policy_runtime 未找到"

    def robotwin_install_hint(_python_bin: str = "") -> str:  # type: ignore[misc]
        return "robotwin_policy_runtime 未找到"

    def probe_robotwin_deps(_python_bin: str, **_k):  # type: ignore[misc]
        return False, "robotwin_policy_runtime 未找到"

    def resolve_robotwin_assets(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or ROBOTWIN_ROOT_DEFAULT

    def resolve_robotwin_checkpoint(prefer: str = "", config: str = ""):  # type: ignore[misc]
        class _Ckpt:
            path = prefer or (
                "/share_data/projects/mahjong/share/personal/liyichao/models/"
                "RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
            )
            config = config or "pi05_aloha_robotwin"
            source = "fallback"

        return _Ckpt()

    def resolve_robotwin_python(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or "python3"

    def resolve_robotwin_root(raw: str = ""):  # type: ignore[misc]
        return raw or ROBOTWIN_ROOT_DEFAULT


try:
    from lingbot_va_policy_runtime import (  # noqa: E402
        DEFAULT_VA_PORT as LINGBOT_VA_PORT_DEFAULT,
        LINGBOT_VA_POLICY_ID,
        LINGBOT_VA_POLICY_LABEL,
        LINGBOT_VA_POLICY_REL,
        build_backend_va_argv,
        build_libero_va_eval_argv,
        build_robotwin_va_eval_argv,
        install_hint as lingbot_va_install_hint,
        is_lingbot_va_policy,
        probe_lingbot_va,
        resolve_base_model as resolve_lingbot_va_base_model,
        resolve_checkpoint as resolve_lingbot_va_checkpoint,
        resolve_lingbot_va_conda_env,
        resolve_lingbot_va_python,
        resolve_lingbot_va_root,
        resolve_paths as resolve_lingbot_va_paths,
        robodojo_policy_rel as lingbot_va_robodojo_policy_rel,
        va_config_for_backend,
        va_support_level,
    )
except ImportError:  # pragma: no cover
    LINGBOT_VA_PORT_DEFAULT = 10001
    LINGBOT_VA_POLICY_ID = "lingbot_va"
    LINGBOT_VA_POLICY_LABEL = "LingBot-VA (WAM)"
    LINGBOT_VA_POLICY_REL = "XPolicyLab/policy/LingBot_VA"

    def is_lingbot_va_policy(policy_id_or_dir: str) -> bool:  # type: ignore[misc]
        low = (policy_id_or_dir or "").lower().replace("-", "_")
        return "lingbot_va" in low

    def lingbot_va_install_hint(_python_bin: str = "") -> str:  # type: ignore[misc]
        return "lingbot_va_policy_runtime 未找到"

    def probe_lingbot_va(_paths=None):  # type: ignore[misc]
        return False, "lingbot_va_policy_runtime 未找到"

    def resolve_lingbot_va_base_model(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_BASE_MODEL_PATH", "")

    def resolve_lingbot_va_checkpoint(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_CKPT", "")

    def resolve_lingbot_va_conda_env(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or ""

    def resolve_lingbot_va_python(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or "python3"

    def resolve_lingbot_va_root(raw: str = "", **_k) -> str:  # type: ignore[misc]
        return raw or (
            "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/"
            "XPolicyLab/policy/LingBot_VA"
        )

    def resolve_lingbot_va_paths(**_k):  # type: ignore[misc]
        raise RuntimeError("lingbot_va_policy_runtime 未找到")

    def lingbot_va_robodojo_policy_rel() -> str:  # type: ignore[misc]
        return LINGBOT_VA_POLICY_REL

    def va_config_for_backend(backend: str, override: str = "") -> str:  # type: ignore[misc]
        return (override or "robotwin30_train").strip()

    def va_support_level(_backend: str) -> str:  # type: ignore[misc]
        return "serve_only"

    def build_backend_va_argv(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("lingbot_va_policy_runtime 未找到")

    def build_libero_va_eval_argv(**_k):  # type: ignore[misc]
        raise RuntimeError("lingbot_va_policy_runtime 未找到")

    def build_robotwin_va_eval_argv(**_k):  # type: ignore[misc]
        raise RuntimeError("lingbot_va_policy_runtime 未找到")


try:
    from vpp2_policy_runtime import (  # noqa: E402
        DEFAULT_VPP2_PORT,
        VPP2_POLICY_ID,
        VPP2_POLICY_LABEL,
        VPP2_POLICY_REL,
        build_backend_vpp2_argv,
        ensure_adapter_installed as ensure_vpp2_adapter,
        install_hint as vpp2_install_hint,
        is_vpp2_policy,
        probe_vpp2,
        resolve_paths as resolve_vpp2_paths,
        resolve_vpp2_bundle,
        resolve_vpp2_conda_env,
        resolve_vpp2_python,
        resolve_vpp2_wan_root,
        resolve_vpp2_weights,
        vpp2_support_level,
    )
except ImportError:  # pragma: no cover
    DEFAULT_VPP2_PORT = 19960
    VPP2_POLICY_ID = "vpp2"
    VPP2_POLICY_LABEL = "VPP2 (WAM)"
    VPP2_POLICY_REL = "XPolicyLab/policy/VPP2"

    def is_vpp2_policy(policy_id_or_dir: str) -> bool:  # type: ignore[misc]
        return "vpp2" in (policy_id_or_dir or "").lower()

    def vpp2_install_hint(_python_bin: str = "") -> str:  # type: ignore[misc]
        return "vpp2_policy_runtime 未找到"

    def probe_vpp2(_paths=None):  # type: ignore[misc]
        return False, "vpp2_policy_runtime 未找到"

    def ensure_vpp2_adapter(**_k):  # type: ignore[misc]
        return False, "vpp2_policy_runtime 未找到"

    def resolve_vpp2_paths(**_k):  # type: ignore[misc]
        raise RuntimeError("vpp2_policy_runtime 未找到")

    def resolve_vpp2_bundle(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or ""

    def resolve_vpp2_conda_env(prefer: str = "") -> str:  # type: ignore[misc]
        return prefer or ""

    def resolve_vpp2_python(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or "python3"

    def resolve_vpp2_wan_root(prefer: str = "", **_k) -> str:  # type: ignore[misc]
        return prefer or ""

    def resolve_vpp2_weights(raw: str = "") -> str:  # type: ignore[misc]
        return raw or (
            "/share_data/projects/mahjong/share/personal/liyichao/models/VPP2"
        )

    def vpp2_support_level(_backend: str) -> str:  # type: ignore[misc]
        return "serve_only"

    def build_backend_vpp2_argv(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("vpp2_policy_runtime 未找到")


PSI_R1_MODEL_XML_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.os.path.dirname(EAI_DIR), "psi_r1_ruiyan", "urdf", "model.xml"),
    _eai_platform_bootstrap.os.path.join(EAI_DIR, "assets", "psi_r1_ruiyan", "urdf", "model.xml"),
    _eai_platform_bootstrap.os.path.join(EAI_DIR, "assets", "model.xml"),
)


PSI_R1_MODEL_XML = next(
    (p for p in PSI_R1_MODEL_XML_CANDIDATES if _eai_platform_bootstrap.os.path.isfile(p)),
    PSI_R1_MODEL_XML_CANDIDATES[0],
)


LINGBOT_VISION_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/lingbot-vision"
)


BAGEL_ROOT_DEFAULT = "/share_data/projects/mahjong/share/personal/liyichao/Bagel"


BAGEL_MODEL_DEFAULT = "models/BAGEL-7B-MoT"


BAGEL_SERVER_HOST_DEFAULT = "127.0.0.1"


BAGEL_SERVER_PORT_DEFAULT = 7860


BAGEL_API_PORT_DEFAULT = 7861


BAGEL_NUM_TIMESTEPS_DEFAULT = 25


BAGEL_FAST_NUM_TIMESTEPS = 12


BAGEL_FAST_IMAGE_SIZE = "224"


BAGEL_CFG_INTERVAL_DEFAULT = 0.4


BAGEL_IMAGE_SIZE_DEFAULT = "input"  # 有输入图时跟输入；纯文生图回退 512


BAGEL_IMAGE_SIZE_FALLBACK = "512"


BAGEL_VENV_PYTHON_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/bagel/bin/python"
)


BAGEL_MODEL_CACHE_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/models/BAGEL-7B-MoT"
)


BAGEL_API_OUTPUT_DIR = _eai_platform_bootstrap.os.path.join(EAI_DIR, ".cache", "bagel_tmp")


ASTRA_ROBODOJO_URL = "https://cookiegg.github.io/llm-as-policies/"


ASTRA_ROBODOJO_ZH_URL = "https://cookiegg.github.io/llm-as-policies/zh/"


ASTRA_GALLERY_FOLD_CLOTHES_URL = (
    "https://anonymous-report-421.github.io/public-website/"
    "gallery.html?id=mix__fold_clothes__random__g0__l2"
)


ASTRA_PHYSICAL_RSI_URL = "https://mmlab.hk/research/PhysicalRSI"


ASTRA_PAGE_PRESETS: _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Tuple[str, str], ...] = (
    ("English", ASTRA_ROBODOJO_URL),
    ("中文解读", ASTRA_ROBODOJO_ZH_URL),
    ("Gallery · fold_clothes", ASTRA_GALLERY_FOLD_CLOTHES_URL),
    ("PhysicalRSI", ASTRA_PHYSICAL_RSI_URL),
)


HUMANOGO_ROOT_DEFAULT = (
    "/share_data/projects/mahjong/share/personal/liyichao/HumanEgo"
)


HUMANOGO_WEBSITE_URL = "https://humanego-ai.github.io/"


HUMANOGO_GALLERY_URL = "https://leo-tx-humanego-gallery.static.hf.space/"


HUMANOGO_PYTHON_CANDIDATES: _eai_platform_bootstrap.Tuple[str, ...] = (
    # Exactly one miniconda3 env for HumanEgo (see tools/model_conda_envs.py).
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/humanego/bin/python",
)

