"""configuration.chat — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

MODEL_CONTROL_MODE = 0


TELEOP_CONTROL_MODES = {1, 3}


ARM_MOVE_SPEED_MIN_RAD_S = 0.10


ARM_MOVE_SPEED_MAX_RAD_S = 3.14


ARM_MOVE_SPEED_DEFAULT_RAD_S = 0.45


ARM_MOVE_MIN_DURATION_S = 0.8


ARM_MOVE_MAX_DURATION_S = 10.0


ARM_MOVE_IK_HZ = 20.0


ARM_MOVE_JOINT_HZ = 100.0


ARM_MOVE_WARMUP_STEPS = 4


ARM_MOVE_MODE_BURST = 5


ARM_MOVE_GOAL_SAMPLE_TICKS = 20  # ~1s @ 20Hz，大范围绝对目标需等 IK 收敛


ARM_MOVE_GOAL_MIN_JOINT_DELTA = 1e-3  # 低于此认为 IK 未给出新解


ARM_MOVE_CART_NEAR_M = 0.03  # 笛卡尔已接近则允许关节几乎不变


ARM_MOVE_IK_RELEASE_TICKS = 3


LLM_API_BASE_DEFAULT = _eai_platform_bootstrap.os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")


LLM_MODEL_DEFAULT = _eai_platform_bootstrap.os.environ.get("LLM_MODEL", "gpt-4o-mini")


LLM_API_KEY_ENV = "LLM_API_KEY"


LLM_CHAT_MAX_HISTORY = 24


LLM_CHAT_TIMEOUT_S = 120.0


LLM_CHAT_VISION_TIMEOUT_S = float(_eai_platform_bootstrap.os.environ.get("LLM_CHAT_VISION_TIMEOUT_S", "300"))


LAKE_ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。给定场景图像、高层任务名称和上一个子任务，"
    "预测该任务下的全部子任务列表，以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「"
    "上一个子任务」「下一个子任务」四行作答。"
)


LAKE_ORCHESTRATOR_SYSTEM_PROMPT_EN = (
    "You are a robot manipulation task orchestrator. Given a scene image, "
    "a high-level task name, and the previous subtask, predict the full list "
    "of subtasks under this task, and the currently executable subtask. "
    "First output a numbered list under All subtasks, then answer in four "
    "lines labeled Skill, Current subtask, Previous subtask, and Next subtask."
)


LAKE_ORCHESTRATOR_SYSTEM_PROMPT_EN2 = (
    "Given a scene image and a task name, output two lines: "
    "All subtasks, then Current subtask."
)


LAKE_ORCHESTRATOR_SYSTEM_PROMPT_TRAINING = (
    "你是机器人操作任务的认知编排器。给定场景图像与高层任务名称，"
    "预测该任务下的全部二层（layer-2）子任务列表，以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「子任务」两行作答。"
)


LAKE_DEFAULT_LANGUAGE_MEMORY = "这是第一个子任务，尚未完成任何子任务。"


LAKE_DEFAULT_LANGUAGE_MEMORY_EN = (
    "This is the first subtask; no subtasks have been completed yet."
)


TRADITIONAL_CHAT_INPUT_PLACEHOLDER = "输入问题或指令，Enter 发送（传统对话，无 System/Lake 提示词）"


LAKE_CHAT_INPUT_PLACEHOLDER = "输入高层任务名（如：合上后盖并拧紧）；Enter 发送"


LAKE_CHAT_INPUT_PLACEHOLDER_EN = (
    "Enter a high-level task name (e.g. close the laptop cover); press Enter"
)


LAKE_USER_PROMPT_TEMPLATE = "任务：{task}"


LAKE_USER_PROMPT_TEMPLATE_EN = "Task: {task}"


def lake_orchestrator_prompt_lang(system_prompt: str = "") -> str:
    """根据 system prompt 判断编排语言：'zh' / 'en' / 'en2'。"""
    s = (system_prompt or "").strip()
    if not s:
        return "zh"
    low = s.lower()
    if (
        s == LAKE_ORCHESTRATOR_SYSTEM_PROMPT_EN2
        or "output two lines" in low
        or (
            "given a scene image and a task name" in low
            and "current subtask" in low
            and "previous subtask" not in low
            and "skill" not in low
        )
    ):
        return "en2"
    if (
        s.startswith("You are a robot")
        or "all subtasks" in low
        or "current subtask" in low
        or "previous subtask" in low
    ):
        return "en"
    return "zh"


def lake_orchestrator_system_prompt_for_lang(lang: str) -> str:
    key = (lang or "").strip().lower()
    if key in ("en2", "en_2", "en-2", "current_only", "en_short"):
        return LAKE_ORCHESTRATOR_SYSTEM_PROMPT_EN2
    if key.startswith("en"):
        return LAKE_ORCHESTRATOR_SYSTEM_PROMPT_EN
    return LAKE_ORCHESTRATOR_SYSTEM_PROMPT


def lake_default_previous_subtask(lang: str = "zh") -> str:
    if (lang or "").lower().startswith("en"):
        return LAKE_DEFAULT_LANGUAGE_MEMORY_EN
    return LAKE_DEFAULT_LANGUAGE_MEMORY


def lake_user_output_instruction(system_prompt: str = "") -> str:
    """User 末尾输出要求（已停用，保留函数供旧调用兼容）。"""
    del system_prompt
    return ""


def format_lake_user_prompt(
    task: str,
    memory: str = "",
    *,
    system_prompt: str = "",
) -> str:
    """把用户任务描述包装成 Lake user 文本（仅 Task / 任务行）。"""
    del memory  # 保留参数兼容旧调用，不再写入 prompt
    lang = lake_orchestrator_prompt_lang(system_prompt)
    task_line = _strip_language_memory_from_prompt((task or "").strip())
    task_line = _strip_lake_output_instruction(task_line)
    # 已带「任务：」/ Task: 前缀则剥掉，再按当前语言重包
    if task_line.startswith("任务："):
        task_line = task_line[len("任务：") :].strip()
    elif task_line.startswith("任务:"):
        task_line = task_line[len("任务:") :].strip()
    elif task_line.lower().startswith("task:"):
        task_line = task_line.split(":", 1)[-1].strip()
    # 若用户粘贴了完整多段 prompt，尽量只取任务首段
    for sep in (
        "\n\n上一个子任务",
        "\n\nPrevious subtask",
        "\n\n请输出",
        "\n\nPlease output",
    ):
        if sep in task_line:
            task_line = task_line.split(sep, 1)[0].strip()
            break
    if (lang or "").startswith("en"):
        return LAKE_USER_PROMPT_TEMPLATE_EN.format(task=task_line)
    return LAKE_USER_PROMPT_TEMPLATE.format(task=task_line)


def _strip_lake_output_instruction(text: str) -> str:
    """去掉 user prompt 末尾的「请输出全部子任务…」指令行。"""
    lines = str(text).splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    drop_prefixes = (
        "请输出全部子任务",
        "请输出全部子任务，以及当前技能",
        "please output all subtasks",
    )
    while lines:
        stripped = lines[-1].strip()
        low = stripped.lower()
        if any(stripped.startswith(p) for p in drop_prefixes) or any(
            low.startswith(p) for p in drop_prefixes if p[:1].isascii()
        ):
            lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def _strip_language_memory_from_prompt(text: str) -> str:
    """从手写 prompt 中移除「语言记忆 / Language memory」段落。"""
    lines = str(text).splitlines()
    out: _eai_platform_bootstrap.List[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if (
            stripped.startswith("语言记忆")
            or low.startswith("language memory")
        ):
            skipping = True
            continue
        if skipping:
            # 记忆段直到空行或下一项（输出要求等）结束
            if not stripped:
                skipping = False
                continue
            if (
                stripped.startswith("请输出")
                or stripped.startswith("输出")
                or low.startswith("output")
                or stripped.startswith("任务")
                or low.startswith("task")
            ):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out).strip()


def extract_lake_memory_from_assistant(text: str) -> _eai_platform_bootstrap.Optional[str]:
    """从模型回复中解析「当前子任务」，供下一轮 Previous subtask 使用。"""
    if not text:
        return None
    memory_line: _eai_platform_bootstrap.Optional[str] = None
    current_subtask: _eai_platform_bootstrap.Optional[str] = None
    previous_subtask: _eai_platform_bootstrap.Optional[str] = None
    for raw in str(text).splitlines():
        line = raw.strip()
        low = line.lower()
        if line.startswith("记忆：") or line.startswith("记忆:"):
            mem = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            memory_line = mem.strip() or None
        elif line.startswith("当前子任务：") or line.startswith("当前子任务:"):
            current_subtask = (
                line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            ).strip() or None
        elif line.startswith("上一个子任务：") or line.startswith("上一个子任务:"):
            previous_subtask = (
                line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            ).strip() or None
        elif low.startswith("memory:"):
            memory_line = line.split(":", 1)[-1].strip() or None
        elif low.startswith("current subtask:"):
            current_subtask = line.split(":", 1)[-1].strip() or None
        elif low.startswith("previous subtask:"):
            previous_subtask = line.split(":", 1)[-1].strip() or None
        elif line.startswith("子任务：") or line.startswith("子任务:"):
            # 旧三行版：Subtask / 子任务
            current_subtask = (
                line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            ).strip() or None
        elif low.startswith("subtask:"):
            current_subtask = line.split(":", 1)[-1].strip() or None
    # 下一轮 Previous = 本轮 Current（对齐训练递推）
    if current_subtask:
        return current_subtask
    if memory_line:
        return memory_line
    if previous_subtask:
        return previous_subtask
    return None


def should_use_lake_orchestrator_prompt(api_base: str, model: str) -> bool:
    """本地/远程 Qwen（含 Lake LoRA）走训练对齐提示词。"""
    mid = (model or "").lower()
    base = (api_base or "").lower()
    if "lake" in mid:
        return True
    if any(x in mid for x in ("qwen3.5", "qwen3", "qwen")) and any(
        p in base for p in ("18100", "18102", "18104", "8100", "127.0.0.1")
    ):
        return True
    return False


CHAT_HISTORY_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "chat_history")


BAGEL_HISTORY_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "bagel_history")


EAI_DATASETS_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "datasets")


EAI_IMAGES_DIR = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "images")


CHAT_USER_SETTINGS_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config/chat_user_settings.json")


CONTROL_TABS_ORDER_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config/control_tabs_order.json")


SIM_BACKENDS_ORDER_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config/sim_backends_order.json")


ARM_PANELS_ORDER_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "arm_panels_order.json")


REAL_ROBOT_BACKENDS_ORDER_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "real_robot_backends_order.json")


TEST_QWEN_LAST_CONFIG_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config/test_qwen_last_config.json")


VIEWER_UI_STATE_PATH = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "config/viewer_ui_state.json")


_UI_STATE_SKIP_TEXT_NAME_RE = _eai_platform_bootstrap.re.compile(
    r"(^|_)(log|result|history|status)(_edit|_view|_label)?$",
    _eai_platform_bootstrap.re.IGNORECASE,
)


HY_EMBODIED_VLM_API_BASE = _eai_platform_bootstrap.os.environ.get(
    "HY_EMBODIED_VLM_API_BASE", "http://127.0.0.1:8080/v1"
)


HY_EMBODIED_VLM_MODEL = _eai_platform_bootstrap.os.environ.get("HY_EMBODIED_VLM_MODEL", "hy_a3b")


HY_RXBRAIN_API_BASE = _eai_platform_bootstrap.os.environ.get(
    "HY_RXBRAIN_API_BASE", "http://127.0.0.1:8090/v1"
)


HY_RXBRAIN_MODEL = _eai_platform_bootstrap.os.environ.get("HY_RXBRAIN_MODEL", "hy-rxbrain")


LLM_VISION_PROVIDER_NAMES = {
    _eai_configuration_policies.LOCAL_QWEN_CHAT_PRESET_NAME,
    _eai_configuration_policies.REMOTE_QWEN_CHAT_PRESET_NAME,
    "本地 Ollama · Qwen3-VL-4B",
    "Hy-Embodied-VLM-1.0",
    "Hy-Embodied-RxBrain-1.0",
}


LLM_THINKING_PROVIDER_NAMES = {
    "Hy-Embodied-VLM-1.0",
}


LLM_PROVIDER_PRESETS: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Tuple[str, str, str]] = {
    _eai_configuration_policies.LOCAL_QWEN_CHAT_PRESET_NAME: (
        _eai_configuration_policies.LOCAL_QWEN_API_BASE_DEFAULT,
        _eai_configuration_policies.LOCAL_QWEN_MODEL_ID,
        "EMPTY",
    ),
    _eai_configuration_policies.REMOTE_QWEN_CHAT_PRESET_NAME: (
        _eai_configuration_policies.REMOTE_QWEN_API_BASE_DEFAULT,
        "qwen3.5-35b-a3b",
        "EMPTY",
    ),
    "本地 Ollama · Qwen3.5-4B": (
        "http://127.0.0.1:11434/v1",
        "qwen3.5:4b",
        "ollama",
    ),
    "本地 Ollama · Qwen3-VL-4B": (
        "http://127.0.0.1:11434/v1",
        "qwen3-vl:4b",
        "ollama",
    ),
    "Hy-Embodied-VLM-1.0": (
        HY_EMBODIED_VLM_API_BASE,
        HY_EMBODIED_VLM_MODEL,
        "EMPTY",
    ),
    "Hy-Embodied-RxBrain-1.0": (
        HY_RXBRAIN_API_BASE,
        HY_RXBRAIN_MODEL,
        "EMPTY",
    ),
    "百炼兼容 · Qwen-Plus": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        "",
    ),
    "OpenAI 官方": ("https://api.openai.com/v1", "gpt-4o-mini", ""),
    "自定义": ("", "", ""),
}


IK_SYNC_TIMEOUT_S = 3.0


MANUAL_OFFSET_MAX_M = 0.5


MANUAL_OFFSET_MAX_CM = MANUAL_OFFSET_MAX_M * 100.0  # UI 用 cm


MANUAL_OFFSET_STEP_CM = 1.0


HAND_JOINT_NAMES = [
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
]


LEFT_HAND_JOINT_NAMES = list(HAND_JOINT_NAMES)


RIGHT_HAND_JOINT_NAMES = list(HAND_JOINT_NAMES)


HAND_CMD_VELOCITY = [2000.0] * 6


HAND_CMD_EFFORT = [800.0] * 6


LEFT_HAND_CMD_VELOCITY = HAND_CMD_VELOCITY


LEFT_HAND_CMD_EFFORT = HAND_CMD_EFFORT


HAND_FINGER_GAP_MAX_CM = 12.0


HAND_FINGER_GAP_STEP_CM = 0.1


HAND_FINGER_GAP_SLIDER_MAX = int(round(HAND_FINGER_GAP_MAX_CM / HAND_FINGER_GAP_STEP_CM))


LEFT_HAND_GAP_A_DEFAULT_CM = HAND_FINGER_GAP_MAX_CM  # 张开


LEFT_HAND_GAP_B_DEFAULT_CM = 6.0  # 半开


SKELETON_LOCAL_CAM_PREFIX = "local_webcam:"


SKELETON_NETWORK_CAM_SOURCE = "network_stream"


SKELETON_NETWORK_URL_DEFAULT = "http://127.0.0.1:8090/cam.mjpg"


SKELETON_NETWORK_HELP = (
    "VNC / Remote SSH 不能直接转发摄像头。请把本机画面推成网络流，再在远端使用：\n\n"
    "【本机（有摄像头，macOS / Linux）】\n"
    "1) 启动 MJPEG 服务（在 eai 仓库目录）:\n"
    "   python3 examples/local_webcam_mjpeg_server.py --device 0 --port 8090\n"
    "   macOS：device 一般为 0（内置摄像头）；权限请在「系统设置 → 隐私与安全性 → 摄像头」允许终端。\n"
    "2) SSH 反向隧道（把本机 8090 映射到远端 localhost:8090）:\n"
    "   ssh -N -R 8090:127.0.0.1:8090 -p 10693 root@42.192.34.154\n"
    "   若已用 Cursor/VS Code Remote SSH，另开终端执行上面这条即可（端口冲突时改 8091）。\n\n"
    "【远端 UI】\n"
    "· 手骨架遥控：相机选「网络流」，URL 填 http://127.0.0.1:8090/cam.mjpg → 开始识别\n"
    "· 真机评测：点「本地摄像头预览」（同一 URL）\n\n"
    "也支持 rtsp://...（需 OpenCV/FFmpeg 可读）。"
)


REAL_EVAL_LOCAL_CAM_HELP = SKELETON_NETWORK_HELP


LOCAL_WEBCAM_MJPEG_SCRIPT = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "examples", "local_webcam_mjpeg_server.py")


LOCAL_WEBCAM_FORWARD_UI = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "examples", "local_webcam_forward_ui.py")


LOCAL_WEBCAM_DEFAULT_PORT = 8090


LOCAL_WEBCAM_SSH_TARGET_DEFAULT = "root@42.192.34.154"


LOCAL_WEBCAM_SSH_PORT_DEFAULT = 10693


LOCAL_WEBCAM_SSH_EXTRA_DEFAULT = (
    f"-p {LOCAL_WEBCAM_SSH_PORT_DEFAULT} "
    "-o ServerAliveInterval=30 -o ExitOnForwardFailure=yes"
)


LOCAL_WEBCAM_TAB_HELP = (
    "本页支持两类用法：\n\n"
    "【本机直连监控】摄像头插在当前机器（如 RealSense / USB 摄像头）时：\n"
    "选设备 →「打开本机监控」，直接 V4L2 预览，无需推流/隧道。\n"
    "Linux RealSense 彩色节点多为 /dev/video4（YUYV），深度多为 video0。\n\n"
    "【转发到远端】摄像头在 Mac/笔记本、UI 在 Remote SSH 时：\n"
    "1) 「启动推流」→ 本机 MJPEG：\n"
    "   python3 examples/local_webcam_mjpeg_server.py --device 0 --port 8090\n"
    "2) 「启动隧道」→ SSH 反向端口：\n"
    "   ssh -N -R 8090:127.0.0.1:8090 -p 10693 root@42.192.34.154\n"
    "也可在 Mac 运行: python3 examples/local_webcam_forward_ui.py\n"
    "然后点「打开预览」（URL 默认 http://127.0.0.1:8090/cam.mjpg）。\n\n"
    "macOS 首次推流需在「系统设置 → 隐私与安全性 → 摄像头」允许终端/Python。"
)

