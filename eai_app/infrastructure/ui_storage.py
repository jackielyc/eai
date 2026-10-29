"""infrastructure.ui storage — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def ensure_chat_history_dir() -> str:
    _eai_platform_bootstrap.os.makedirs(_eai_configuration_chat.CHAT_HISTORY_DIR, exist_ok=True)
    return _eai_configuration_chat.CHAT_HISTORY_DIR


def load_chat_user_settings() -> _eai_platform_bootstrap.Dict[str, object]:
    """加载对话面板用户设置（system prompt 等）。"""
    try:
        with open(_eai_configuration_chat.CHAT_USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_chat_user_settings(data: _eai_platform_bootstrap.Dict[str, object]) -> str:
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.CHAT_USER_SETTINGS_PATH), exist_ok=True)
    with open(_eai_configuration_chat.CHAT_USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump(data, f, ensure_ascii=False, indent=2)
    return _eai_configuration_chat.CHAT_USER_SETTINGS_PATH


def _sanitize_control_tab_title_list(
    raw: _eai_platform_bootstrap.Sequence[object], *, allow_misc: bool = True
) -> _eai_platform_bootstrap.List[str]:
    out: _eai_platform_bootstrap.List[str] = []
    seen = set()
    known = set(_eai_configuration_tabs.CONTROL_TAB_TITLES)
    for item in raw:
        title = str(item or "").strip()
        if not title or title in seen:
            continue
        if title not in known:
            continue
        if not allow_misc and title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            continue
        seen.add(title)
        out.append(title)
    return out


def load_control_tabs_order() -> _eai_platform_bootstrap.List[str]:
    """加载控制区 tab 自定义顺序；无效则返回空列表。"""
    try:
        with open(_eai_configuration_chat.CONTROL_TABS_ORDER_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("order") or []
        else:
            raw = []
        return _sanitize_control_tab_title_list(raw if isinstance(raw, list) else [])
    except FileNotFoundError:
        return []
    except Exception:
        return []


def load_control_misc_children() -> _eai_platform_bootstrap.List[str]:
    """加载收入「其他」的子标签顺序。"""
    try:
        with open(_eai_configuration_chat.CONTROL_TABS_ORDER_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        if not isinstance(data, dict):
            return []
        raw = data.get("misc_children") or []
        return _sanitize_control_tab_title_list(
            raw if isinstance(raw, list) else [], allow_misc=False
        )
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_control_tabs_order(
    titles: _eai_platform_bootstrap.Sequence[str],
    misc_children: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[str]] = None,
) -> str:
    cleaned = _sanitize_control_tab_title_list(titles)
    if misc_children is None:
        misc_cleaned = load_control_misc_children()
    else:
        misc_cleaned = _sanitize_control_tab_title_list(
            misc_children, allow_misc=False
        )
    # 子标签不应再出现在顶栏顺序里（「其他」本身除外）
    misc_set = set(misc_cleaned)
    cleaned = [
        t for t in cleaned if t not in misc_set or t == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE
    ]
    if _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE not in cleaned and (
        _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE in _eai_configuration_tabs.CONTROL_TAB_TITLES
    ):
        cleaned.append(_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.CONTROL_TABS_ORDER_PATH), exist_ok=True)
    with open(_eai_configuration_chat.CONTROL_TABS_ORDER_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump(
            {"order": cleaned, "misc_children": misc_cleaned},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return _eai_configuration_chat.CONTROL_TABS_ORDER_PATH


def preferred_control_tabs_order(available: _eai_platform_bootstrap.Sequence[str]) -> _eai_platform_bootstrap.List[str]:
    """在 available 中按「已保存顺序 → 默认 CONTROL_TAB_TITLES」排列。"""
    avail = [str(t) for t in available if str(t).strip()]
    avail_set = set(avail)
    saved = load_control_tabs_order()
    ordered: _eai_platform_bootstrap.List[str] = []
    for title in saved:
        if title in avail_set and title not in ordered:
            ordered.append(title)
    for title in _eai_configuration_tabs.CONTROL_TAB_TITLES:
        if title in avail_set and title not in ordered:
            ordered.append(title)
    for title in avail:
        if title not in ordered:
            ordered.append(title)
    return ordered


def load_sim_backends_order() -> _eai_platform_bootstrap.List[str]:
    """加载仿真评测后端选择条自定义顺序；无效则返回空列表。"""
    known = set(_eai_configuration_tabs.SIM_BACKEND_ORDER_KEYS)
    try:
        with open(_eai_configuration_chat.SIM_BACKENDS_ORDER_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("order") or []
        else:
            raw = []
        out: _eai_platform_bootstrap.List[str] = []
        seen = set()
        for item in raw:
            key = str(item or "").strip()
            if not key or key in seen or key not in known:
                continue
            seen.add(key)
            out.append(key)
        return out
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_sim_backends_order(keys: _eai_platform_bootstrap.Sequence[str]) -> str:
    known = set(_eai_configuration_tabs.SIM_BACKEND_ORDER_KEYS)
    cleaned: _eai_platform_bootstrap.List[str] = []
    seen = set()
    for item in keys:
        key = str(item or "").strip()
        if not key or key in seen or key not in known:
            continue
        seen.add(key)
        cleaned.append(key)
    for key in _eai_configuration_tabs.SIM_BACKEND_ORDER_KEYS:
        if key not in seen:
            cleaned.append(key)
            seen.add(key)
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.SIM_BACKENDS_ORDER_PATH), exist_ok=True)
    with open(_eai_configuration_chat.SIM_BACKENDS_ORDER_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump({"order": cleaned}, f, ensure_ascii=False, indent=2)
    return _eai_configuration_chat.SIM_BACKENDS_ORDER_PATH


def preferred_sim_backends_order(available: _eai_platform_bootstrap.Sequence[str] | None = None) -> _eai_platform_bootstrap.List[str]:
    """按「已保存顺序 → 默认 SIM_BACKEND_ORDER_KEYS」排列。"""
    avail = [
        str(k)
        for k in (available if available is not None else _eai_configuration_tabs.SIM_BACKEND_ORDER_KEYS)
        if str(k).strip()
    ]
    avail_set = set(avail)
    saved = load_sim_backends_order()
    ordered: _eai_platform_bootstrap.List[str] = []
    for key in saved:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in _eai_configuration_tabs.SIM_BACKEND_ORDER_KEYS:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in avail:
        if key not in ordered:
            ordered.append(key)
    return ordered


def load_arm_panels_order() -> _eai_platform_bootstrap.List[str]:
    """加载「手臂/手」页功能子面板自定义顺序；无效则返回空列表。"""
    known = set(_eai_configuration_tabs.ARM_PANEL_ORDER_KEYS)
    try:
        with open(_eai_configuration_chat.ARM_PANELS_ORDER_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("order") or []
        else:
            raw = []
        out: _eai_platform_bootstrap.List[str] = []
        seen = set()
        for item in raw:
            key = str(item or "").strip()
            if not key or key in seen or key not in known:
                continue
            seen.add(key)
            out.append(key)
        return out
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_arm_panels_order(keys: _eai_platform_bootstrap.Sequence[str]) -> str:
    known = set(_eai_configuration_tabs.ARM_PANEL_ORDER_KEYS)
    cleaned: _eai_platform_bootstrap.List[str] = []
    seen = set()
    for item in keys:
        key = str(item or "").strip()
        if not key or key in seen or key not in known:
            continue
        seen.add(key)
        cleaned.append(key)
    for key in _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS:
        if key not in seen:
            cleaned.append(key)
            seen.add(key)
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.ARM_PANELS_ORDER_PATH), exist_ok=True)
    with open(_eai_configuration_chat.ARM_PANELS_ORDER_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump({"order": cleaned}, f, ensure_ascii=False, indent=2)
    return _eai_configuration_chat.ARM_PANELS_ORDER_PATH


def preferred_arm_panels_order(available: _eai_platform_bootstrap.Sequence[str] | None = None) -> _eai_platform_bootstrap.List[str]:
    """按「已保存顺序 → 默认 ARM_PANEL_ORDER_KEYS」排列。"""
    avail = [
        str(k)
        for k in (available if available is not None else _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS)
        if str(k).strip()
    ]
    avail_set = set(avail)
    saved = load_arm_panels_order()
    ordered: _eai_platform_bootstrap.List[str] = []
    for key in saved:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in _eai_configuration_tabs.ARM_PANEL_ORDER_KEYS:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in avail:
        if key not in ordered:
            ordered.append(key)
    return ordered


def load_real_robot_backends_order() -> _eai_platform_bootstrap.List[str]:
    """加载真机评测后端选择条自定义顺序；无效则返回空列表。"""
    known = set(_eai_configuration_policies.REAL_ROBOT_ORDER_KEYS)
    try:
        with open(_eai_configuration_chat.REAL_ROBOT_BACKENDS_ORDER_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("order") or []
        else:
            raw = []
        out: _eai_platform_bootstrap.List[str] = []
        seen = set()
        for item in raw:
            key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(item or ""))
            if not key or key in seen or key not in known:
                continue
            seen.add(key)
            out.append(key)
        return out
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_real_robot_backends_order(keys: _eai_platform_bootstrap.Sequence[str]) -> str:
    known = set(_eai_configuration_policies.REAL_ROBOT_ORDER_KEYS)
    cleaned: _eai_platform_bootstrap.List[str] = []
    seen = set()
    for item in keys:
        key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(item or ""))
        if not key or key in seen or key not in known:
            continue
        seen.add(key)
        cleaned.append(key)
    for key in _eai_configuration_policies.REAL_ROBOT_ORDER_KEYS:
        if key not in seen:
            cleaned.append(key)
            seen.add(key)
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.REAL_ROBOT_BACKENDS_ORDER_PATH), exist_ok=True)
    with open(_eai_configuration_chat.REAL_ROBOT_BACKENDS_ORDER_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump({"order": cleaned}, f, ensure_ascii=False, indent=2)
    return _eai_configuration_chat.REAL_ROBOT_BACKENDS_ORDER_PATH


def preferred_real_robot_backends_order(
    available: _eai_platform_bootstrap.Sequence[str] | None = None,
) -> _eai_platform_bootstrap.List[str]:
    """按「已保存顺序 → 默认 REAL_ROBOT_ORDER_KEYS」排列。"""
    avail = [
        _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(k))
        for k in (available if available is not None else _eai_configuration_policies.REAL_ROBOT_ORDER_KEYS)
        if str(k).strip()
    ]
    avail = [k for k in avail if k in _eai_configuration_policies.REAL_ROBOT_ORDER_KEYS]
    avail_set = set(avail)
    saved = load_real_robot_backends_order()
    ordered: _eai_platform_bootstrap.List[str] = []
    for key in saved:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in _eai_configuration_policies.REAL_ROBOT_ORDER_KEYS:
        if key in avail_set and key not in ordered:
            ordered.append(key)
    for key in avail:
        if key not in ordered:
            ordered.append(key)
    return ordered


def load_test_qwen_last_config() -> _eai_platform_bootstrap.Dict[str, object]:
    """加载测试 Tab 上次推理部署配置。"""
    try:
        with open(_eai_configuration_chat.TEST_QWEN_LAST_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_test_qwen_last_config(data: _eai_platform_bootstrap.Dict[str, object]) -> str:
    _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(_eai_configuration_chat.TEST_QWEN_LAST_CONFIG_PATH), exist_ok=True)
    with open(_eai_configuration_chat.TEST_QWEN_LAST_CONFIG_PATH, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump(data, f, ensure_ascii=False, indent=2)
    return _eai_configuration_chat.TEST_QWEN_LAST_CONFIG_PATH


def load_viewer_ui_state() -> _eai_platform_bootstrap.Dict[str, object]:
    """加载主界面控件状态（选项 / 下拉 / 输入等）。"""
    try:
        with open(_eai_configuration_chat.VIEWER_UI_STATE_PATH, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_viewer_ui_state(data: _eai_platform_bootstrap.Dict[str, object]) -> str:
    from eai_app.infrastructure.viewer_preferences import save_main_preferences
    return save_main_preferences(data, _eai_configuration_chat.VIEWER_UI_STATE_PATH)


def _ui_state_skip_text_widget(name: str, widget: _eai_platform_bootstrap.QWidget) -> bool:
    """跳过只读日志/结果类文本框。"""
    if _eai_configuration_chat._UI_STATE_SKIP_TEXT_NAME_RE.search(str(name or "")):
        return True
    try:
        if isinstance(widget, (_eai_platform_bootstrap.QTextEdit, _eai_platform_bootstrap.QPlainTextEdit)) and widget.isReadOnly():
            return True
    except Exception:
        return True
    return False


def _ui_state_read_widget(widget: _eai_platform_bootstrap.QWidget) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]]:
    """读取单个可操作控件的可序列化状态。"""
    try:
        if isinstance(widget, _eai_platform_bootstrap.QComboBox):
            payload: _eai_platform_bootstrap.Dict[str, object] = {
                "kind": "combo",
                "index": int(widget.currentIndex()),
                "text": widget.currentText(),
            }
            data = widget.currentData()
            if data is not None and isinstance(data, (str, int, float, bool)):
                payload["data"] = data
            elif data is not None:
                payload["data"] = str(data)
            if widget.isEditable():
                payload["edit"] = widget.currentText()
            return payload
        if isinstance(widget, _eai_platform_bootstrap.QLineEdit):
            return {"kind": "line", "text": widget.text()}
        if isinstance(widget, _eai_platform_bootstrap.QPlainTextEdit):
            return {"kind": "plain", "text": widget.toPlainText()}
        if isinstance(widget, _eai_platform_bootstrap.QTextEdit):
            return {"kind": "text", "text": widget.toPlainText()}
        if isinstance(widget, _eai_platform_bootstrap.QSpinBox):
            return {"kind": "spin", "value": int(widget.value())}
        if isinstance(widget, _eai_platform_bootstrap.QDoubleSpinBox):
            return {"kind": "dspin", "value": float(widget.value())}
        if isinstance(widget, _eai_platform_bootstrap.QSlider):
            return {"kind": "slider", "value": int(widget.value())}
        if isinstance(widget, _eai_platform_bootstrap.QCheckBox):
            return {"kind": "check", "checked": bool(widget.isChecked())}
        if isinstance(widget, _eai_platform_bootstrap.QRadioButton):
            return {"kind": "radio", "checked": bool(widget.isChecked())}
        if isinstance(widget, (_eai_platform_bootstrap.QToolButton, _eai_platform_bootstrap.QPushButton)) and widget.isCheckable():
            return {"kind": "toggle", "checked": bool(widget.isChecked())}
    except Exception:
        return None
    return None


def _ui_state_write_widget(widget: _eai_platform_bootstrap.QWidget, payload: object) -> bool:
    """把已保存状态写回控件；写期间屏蔽信号，避免误触发网络/任务。"""
    if not isinstance(payload, dict):
        return False
    kind = str(payload.get("kind") or "")
    # radio / 可勾选按钮需要信号以同步互斥组或展开面板
    block = kind not in ("radio", "toggle")
    prev_blocked = widget.blockSignals(True) if block else False
    try:
        if kind == "combo" and isinstance(widget, _eai_platform_bootstrap.QComboBox):
            data = payload.get("data", None)
            text = str(payload.get("text") or "")
            idx = payload.get("index", None)
            matched = -1
            if data is not None:
                for i in range(widget.count()):
                    if widget.itemData(i) == data or str(widget.itemData(i)) == str(data):
                        matched = i
                        break
            if matched < 0 and text:
                matched = widget.findText(text)
            if matched < 0 and isinstance(idx, int) and 0 <= idx < widget.count():
                matched = idx
            if matched >= 0:
                widget.setCurrentIndex(matched)
            if widget.isEditable() and "edit" in payload:
                widget.setEditText(str(payload.get("edit") or ""))
            elif widget.isEditable() and text and matched < 0:
                widget.setEditText(text)
            return matched >= 0 or widget.isEditable()
        if kind == "line" and isinstance(widget, _eai_platform_bootstrap.QLineEdit):
            widget.setText(str(payload.get("text") or ""))
            return True
        if kind == "plain" and isinstance(widget, _eai_platform_bootstrap.QPlainTextEdit):
            widget.setPlainText(str(payload.get("text") or ""))
            return True
        if kind == "text" and isinstance(widget, _eai_platform_bootstrap.QTextEdit):
            widget.setPlainText(str(payload.get("text") or ""))
            return True
        if kind == "spin" and isinstance(widget, _eai_platform_bootstrap.QSpinBox):
            widget.setValue(int(payload.get("value") or 0))
            return True
        if kind == "dspin" and isinstance(widget, _eai_platform_bootstrap.QDoubleSpinBox):
            widget.setValue(float(payload.get("value") or 0.0))
            return True
        if kind == "slider" and isinstance(widget, _eai_platform_bootstrap.QSlider):
            widget.setValue(int(payload.get("value") or 0))
            return True
        if kind == "check" and isinstance(widget, _eai_platform_bootstrap.QCheckBox):
            widget.setChecked(bool(payload.get("checked")))
            return True
        if kind == "radio" and isinstance(widget, _eai_platform_bootstrap.QRadioButton):
            if bool(payload.get("checked")):
                widget.setChecked(True)
            return True
        if kind == "toggle" and isinstance(widget, (_eai_platform_bootstrap.QToolButton, _eai_platform_bootstrap.QPushButton)):
            if widget.isCheckable():
                widget.setChecked(bool(payload.get("checked")))
                return True
    except Exception:
        return False
    finally:
        if block:
            widget.blockSignals(prev_blocked)
    return False


def _ui_state_collect_owner(owner: object, prefix: str = "") -> _eai_platform_bootstrap.Dict[str, object]:
    """从对象属性中采集可操作控件状态。"""
    out: _eai_platform_bootstrap.Dict[str, object] = {}
    try:
        items = list(vars(owner).items())
    except Exception:
        return out
    form_types = (
        _eai_platform_bootstrap.QLineEdit,
        _eai_platform_bootstrap.QTextEdit,
        _eai_platform_bootstrap.QPlainTextEdit,
        _eai_platform_bootstrap.QComboBox,
        _eai_platform_bootstrap.QSpinBox,
        _eai_platform_bootstrap.QDoubleSpinBox,
        _eai_platform_bootstrap.QCheckBox,
        _eai_platform_bootstrap.QRadioButton,
        _eai_platform_bootstrap.QSlider,
        _eai_platform_bootstrap.QToolButton,
        _eai_platform_bootstrap.QPushButton,
    )
    for name, value in items:
        if not isinstance(name, str) or not name or name.startswith("_"):
            continue
        if not isinstance(value, form_types):
            continue
        if isinstance(value, (_eai_platform_bootstrap.QTextEdit, _eai_platform_bootstrap.QPlainTextEdit)) and _ui_state_skip_text_widget(
            name, value
        ):
            continue
        if isinstance(value, _eai_platform_bootstrap.QLineEdit) and _eai_configuration_chat._UI_STATE_SKIP_TEXT_NAME_RE.search(name):
            continue
        if isinstance(value, (_eai_platform_bootstrap.QToolButton, _eai_platform_bootstrap.QPushButton)) and not value.isCheckable():
            continue
        payload = _ui_state_read_widget(value)
        if payload is not None:
            out[f"{prefix}{name}"] = payload
    return out


def _ui_state_apply_owner(
    owner: object, widgets: _eai_platform_bootstrap.Dict[str, object], prefix: str = ""
) -> int:
    """把 widgets 字典中属于该 owner 的项写回控件，返回成功条数。"""
    restored = 0
    try:
        attrs = vars(owner)
    except Exception:
        return 0
    for key, payload in widgets.items():
        if prefix:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
        else:
            if "." in key or key.startswith("tab:"):
                continue
            name = key
        if not name or "." in name:
            continue
        value = attrs.get(name)
        if value is None:
            continue
        if _ui_state_write_widget(value, payload):
            restored += 1
    return restored


def chat_history_path(history_id: str) -> str:
    safe = "".join(ch for ch in history_id if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("invalid history id")
    return _eai_platform_bootstrap.os.path.join(ensure_chat_history_dir(), f"{safe}.json")


def default_chat_history_title(messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            text = " ".join(content.strip().split())
            return text[:40] + ("…" if len(text) > 40 else "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        text = " ".join(text.split())
                        return text[:40] + ("…" if len(text) > 40 else "")
    return f"对话 {_eai_platform_bootstrap.datetime.now().strftime('%m-%d %H:%M')}"


def serialize_chat_messages(messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]:
    """仅持久化可 JSON 化的文本轮次（含时间戳与耗时）。"""
    out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            continue
        item: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, object]] = None
        if isinstance(content, str):
            item = {"role": role, "content": content}
        elif isinstance(content, list):
            texts = [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = "\n".join(t for t in texts if t).strip()
            if text:
                item = {"role": role, "content": text}
        if item is None:
            continue
        ts = msg.get("ts")
        if isinstance(ts, str) and ts.strip():
            item["ts"] = ts.strip()
        latency = msg.get("latency_s")
        if isinstance(latency, (int, float)):
            item["latency_s"] = float(latency)
        out.append(item)
    return out


def format_chat_timestamp(ts: _eai_platform_bootstrap.Optional[str] = None) -> str:
    """本地时间显示；ts 可为 ISO 字符串。"""
    if ts:
        raw = ts.strip()
        try:
            if raw.endswith("Z"):
                dt = _eai_platform_bootstrap.datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
            else:
                dt = _eai_platform_bootstrap.datetime.fromisoformat(raw)
                if dt.tzinfo is not None:
                    dt = dt.astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return raw
    return _eai_platform_bootstrap.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_chat_latency(latency_s: _eai_platform_bootstrap.Optional[float]) -> str:
    if latency_s is None:
        return ""
    try:
        val = float(latency_s)
    except (TypeError, ValueError):
        return ""
    if val < 0:
        return ""
    if val < 1.0:
        return f"{val * 1000.0:.0f} ms"
    return f"{val:.2f} s"


def list_chat_histories() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]:
    root = ensure_chat_history_dir()
    items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
    for name in _eai_platform_bootstrap.os.listdir(root):
        if not name.endswith(".json"):
            continue
        path = _eai_platform_bootstrap.os.path.join(root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            if not isinstance(data, dict):
                continue
            hid = str(data.get("id") or _eai_platform_bootstrap.os.path.splitext(name)[0])
            items.append(
                {
                    "id": hid,
                    "title": str(data.get("title") or hid),
                    "updated_at": str(data.get("updated_at") or ""),
                    "created_at": str(data.get("created_at") or ""),
                    "model": str(data.get("model") or ""),
                    "path": path,
                    "message_count": len(data.get("messages") or []),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return items


def load_chat_history(history_id: str) -> _eai_platform_bootstrap.Dict[str, object]:
    path = chat_history_path(history_id)
    with open(path, "r", encoding="utf-8") as f:
        data = _eai_platform_bootstrap.json.load(f)
    if not isinstance(data, dict):
        raise ValueError("invalid history file")
    data["id"] = str(data.get("id") or history_id)
    return data


def save_chat_history_record(record: _eai_platform_bootstrap.Dict[str, object]) -> str:
    history_id = str(record.get("id") or "").strip() or _eai_platform_bootstrap.uuid.uuid4().hex[:12]
    record["id"] = history_id
    record["updated_at"] = _eai_adapters_models_chat_client._utc_now_iso()
    if not record.get("created_at"):
        record["created_at"] = record["updated_at"]
    path = chat_history_path(history_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump(record, f, ensure_ascii=False, indent=2)
    _eai_platform_bootstrap.os.replace(tmp, path)
    return path


def delete_chat_history(history_id: str) -> None:
    path = chat_history_path(history_id)
    if _eai_platform_bootstrap.os.path.isfile(path):
        _eai_platform_bootstrap.os.remove(path)


def rename_chat_history(history_id: str, title: str) -> None:
    data = load_chat_history(history_id)
    data["title"] = title.strip() or str(data.get("title") or history_id)
    save_chat_history_record(data)


BAGEL_TASK_LABELS = {
    "text2image": "文生图",
    "understand": "图像理解",
    "edit": "图像编辑",
}


def ensure_bagel_history_dir() -> str:
    _eai_platform_bootstrap.os.makedirs(_eai_configuration_chat.BAGEL_HISTORY_DIR, exist_ok=True)
    return _eai_configuration_chat.BAGEL_HISTORY_DIR


def bagel_history_path(history_id: str) -> str:
    safe = "".join(ch for ch in history_id if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("invalid bagel history id")
    return _eai_platform_bootstrap.os.path.join(ensure_bagel_history_dir(), f"{safe}.json")


def bagel_history_asset_dir(history_id: str) -> str:
    safe = "".join(ch for ch in history_id if ch.isalnum() or ch in "-_") or "unknown"
    path = _eai_platform_bootstrap.os.path.join(ensure_bagel_history_dir(), "assets", safe)
    _eai_platform_bootstrap.os.makedirs(path, exist_ok=True)
    return path


def default_bagel_history_title(turns: _eai_platform_bootstrap.Sequence[_eai_platform_bootstrap.Dict[str, object]]) -> str:
    for turn in reversed(list(turns)):
        prompt = str(turn.get("prompt") or "").strip()
        if prompt:
            text = " ".join(prompt.split())
            return text[:40] + ("…" if len(text) > 40 else "")
    return f"Bagel {_eai_platform_bootstrap.datetime.now().strftime('%m-%d %H:%M')}"


def list_bagel_histories() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]:
    root = ensure_bagel_history_dir()
    items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
    for name in _eai_platform_bootstrap.os.listdir(root):
        if not name.endswith(".json"):
            continue
        path = _eai_platform_bootstrap.os.path.join(root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _eai_platform_bootstrap.json.load(f)
            if not isinstance(data, dict):
                continue
            hid = str(data.get("id") or _eai_platform_bootstrap.os.path.splitext(name)[0])
            turns = data.get("turns") or []
            items.append(
                {
                    "id": hid,
                    "title": str(data.get("title") or hid),
                    "updated_at": str(data.get("updated_at") or ""),
                    "created_at": str(data.get("created_at") or ""),
                    "path": path,
                    "turn_count": len(turns) if isinstance(turns, list) else 0,
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return items


def load_bagel_history(history_id: str) -> _eai_platform_bootstrap.Dict[str, object]:
    path = bagel_history_path(history_id)
    with open(path, "r", encoding="utf-8") as f:
        data = _eai_platform_bootstrap.json.load(f)
    if not isinstance(data, dict):
        raise ValueError("invalid bagel history file")
    data["id"] = str(data.get("id") or history_id)
    turns = data.get("turns")
    if not isinstance(turns, list):
        data["turns"] = []
    return data


def save_bagel_history_record(record: _eai_platform_bootstrap.Dict[str, object]) -> str:
    history_id = str(record.get("id") or "").strip() or _eai_platform_bootstrap.uuid.uuid4().hex[:12]
    record["id"] = history_id
    record["updated_at"] = _eai_adapters_models_chat_client._utc_now_iso()
    if not record.get("created_at"):
        record["created_at"] = record["updated_at"]
    if not str(record.get("title") or "").strip():
        turns = record.get("turns") if isinstance(record.get("turns"), list) else []
        record["title"] = default_bagel_history_title(turns)  # type: ignore[arg-type]
    path = bagel_history_path(history_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _eai_platform_bootstrap.json.dump(record, f, ensure_ascii=False, indent=2)
    _eai_platform_bootstrap.os.replace(tmp, path)
    return path


def delete_bagel_history(history_id: str) -> None:
    path = bagel_history_path(history_id)
    if _eai_platform_bootstrap.os.path.isfile(path):
        _eai_platform_bootstrap.os.remove(path)
    asset = _eai_platform_bootstrap.os.path.join(ensure_bagel_history_dir(), "assets", history_id)
    if _eai_platform_bootstrap.os.path.isdir(asset):
        _eai_platform_bootstrap.shutil.rmtree(asset, ignore_errors=True)


def rename_bagel_history(history_id: str, title: str) -> None:
    data = load_bagel_history(history_id)
    data["title"] = title.strip() or str(data.get("title") or history_id)
    save_bagel_history_record(data)


def bagel_persist_media(history_id: str, src: str, prefix: str) -> str:
    """把输入/输出图复制到 bagel_history/assets，返回持久路径。"""
    src = str(src or "").strip()
    if not src or not _eai_platform_bootstrap.os.path.isfile(src):
        return ""
    asset_root = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.join(ensure_bagel_history_dir(), "assets"))
    src_abs = _eai_platform_bootstrap.os.path.abspath(src)
    if src_abs.startswith(asset_root + _eai_platform_bootstrap.os.sep):
        return src_abs
    ext = _eai_platform_bootstrap.os.path.splitext(src)[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        ext = ".png"
    dst = _eai_platform_bootstrap.os.path.join(
        bagel_history_asset_dir(history_id),
        f"{prefix}_{_eai_platform_bootstrap.uuid.uuid4().hex[:10]}{ext}",
    )
    _eai_platform_bootstrap.shutil.copy2(src, dst)
    return dst

