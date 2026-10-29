"""application.cli — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def parse_args() -> _eai_platform_bootstrap.argparse.Namespace:
    parser = _eai_platform_bootstrap.argparse.ArgumentParser(description="PyQt5 图形界面显示 /camera 开头的 ROS2 topic")
    parser.add_argument("--prefix", default="/camera", help="topic 前缀（默认: /camera）")
    parser.add_argument(
        "--llm-api-base",
        default=_eai_platform_bootstrap.os.environ.get("LLM_API_BASE", _eai_configuration_chat.LLM_API_BASE_DEFAULT),
        help="大模型 OpenAI 兼容 API 地址",
    )
    parser.add_argument(
        "--llm-model",
        default=_eai_platform_bootstrap.os.environ.get("LLM_MODEL", _eai_configuration_chat.LLM_MODEL_DEFAULT),
        help="大模型名称",
    )
    parser.add_argument(
        "--tab",
        action="append",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "只展示指定控制区 tab，可同时指定多个。"
            "例: --tab \"sub image\" \"sub task\"  或  --tab bagel,test  或  --tab reward。"
            "可用中文名或英文别名（reward/rm/sim/…）；亦支持环境变量 EAI_ONLY_TAB=a,b。"
        ),
    )
    parser.add_argument(
        "--only-tab",
        dest="tab",
        action="append",
        nargs="+",
        help=_eai_platform_bootstrap.argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_control_tab_title(name: str) -> _eai_platform_bootstrap.Optional[str]:
    raw = (name or "").strip()
    if not raw:
        return None
    # 直接命中正式标题
    if raw in _eai_configuration_tabs.CONTROL_TAB_TITLES:
        return raw
    key = raw.lower().replace(" ", "").replace("_", "-")
    # 别名表（含中文）
    if raw in _eai_configuration_tabs.CONTROL_TAB_ALIASES:
        return _eai_configuration_tabs.CONTROL_TAB_ALIASES[raw]
    if key in _eai_configuration_tabs.CONTROL_TAB_ALIASES:
        return _eai_configuration_tabs.CONTROL_TAB_ALIASES[key]
    # 宽松：去掉间隔符再比
    compact = raw.replace("·", "/").replace("-", "").replace("_", "").replace(" ", "")
    for title in _eai_configuration_tabs.CONTROL_TAB_TITLES:
        if title.replace("/", "").replace("·", "") == compact:
            return title
        if title.lower().replace(" ", "") == key:
            return title
    return None


def _flatten_tab_arg_items(tab_args: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[object]]) -> _eai_platform_bootstrap.List[str]:
    """展开 --tab 的 append+nargs 结果，并拆开逗号/分号。"""
    raw_parts: _eai_platform_bootstrap.List[str] = []

    def _push(text: str) -> None:
        for p in str(text).replace(";", ",").split(","):
            part = p.strip()
            if part:
                raw_parts.append(part)

    if not tab_args:
        return raw_parts
    for item in tab_args:
        if isinstance(item, (list, tuple)):
            for sub in item:
                _push(str(sub))
        else:
            _push(str(item))
    return raw_parts


def parse_only_tabs(tab_args: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[object]]) -> _eai_platform_bootstrap.List[str]:
    raw_parts = _flatten_tab_arg_items(tab_args)
    if not raw_parts:
        env = _eai_platform_bootstrap.os.environ.get("EAI_ONLY_TAB", "").strip()
        if env:
            raw_parts.extend(
                p.strip() for p in env.replace(";", ",").split(",") if p.strip()
            )
    resolved: _eai_platform_bootstrap.List[str] = []
    unknown: _eai_platform_bootstrap.List[str] = []
    for part in raw_parts:
        title = resolve_control_tab_title(part)
        if title is None:
            unknown.append(part)
            continue
        if title not in resolved:
            resolved.append(title)
    if unknown:
        known = " / ".join(_eai_configuration_tabs.CONTROL_TAB_TITLES)
        aliases = (
            "test→sub task, bagel→sub image, reward→Reward评测, sim_rl_train→仿真强化学习训练, mujoco→MuJoCo, sim→仿真评测, world→世界模型, …"
        )
        raise SystemExit(
            f"未知 tab: {', '.join(unknown)}\n"
            f"可用标题: {known}\n"
            f"常用别名: {aliases}"
        )
    return resolved

