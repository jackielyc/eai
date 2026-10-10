"""把「手臂/手」页的文本指令解析成相对移动和手部开合。

方向相对 base_link：X 前正后负，Y 左正右负，Z 上正下负。
未写距离时手臂默认移动 5cm，单轴最大 0.5m。
手部 0=张开，1=闭合。多条指令用分号、句号或换行分隔，按书写顺序执行。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ARM_TEXT_DEFAULT_STEP_M = 0.05
ARM_TEXT_MAX_OFFSET_M = 0.5

ARM_TEXT_CMD_PLACEHOLDER = "例：左臂向前5cm；右手张开"
ARM_TEXT_CMD_HELP = (
    "输入文字指挥手臂相对移动和手的开合，回车或点「执行」。\n"
    "多条用分号或换行分隔，按书写顺序执行（手臂移动完成后才执行下一条）。\n"
    "\n"
    "手臂（base_link，不写距离默认 5cm，最大 0.5m）：\n"
    "  左臂向前5cm\n"
    "  右臂向上2厘米，向左1cm\n"
    "  双臂抬起\n"
    "\n"
    "手（0=张，1=合）：\n"
    "  左手张开；右手闭合；双手半握\n"
    "  右手合到60%；左手开到0.2\n"
    "  左手食指弯曲；右手拇指张开\n"
    "\n"
    "停止当前移动并清空后续指令：停止\n"
    "\n"
    "未使能时会自动请求启用。\n"
    "仿真评测（任一后端）在「只开界面」或策略=无时，指令写入 gui_robot_cmd.json，"
    "由 Isaac / MuJoCo / LIBERO / RoboTwin 等仿真进程执行。"
)

HAND_JOINT_ORDER = (
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
)

_JOINT_LABELS = {
    "thumb_rotation": "拇指旋转",
    "thumb_bend": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
}

# (短语, 轴, 符号)。长词优先匹配，避免「向前」再吃掉「前」。
_DIR_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("forwards", "x", 1),
    ("forward", "x", 1),
    ("backward", "x", -1),
    ("backwards", "x", -1),
    ("leftward", "y", 1),
    ("rightward", "y", -1),
    ("to the left", "y", 1),
    ("to the right", "y", -1),
    ("向上", "z", 1),
    ("往上", "z", 1),
    ("抬起", "z", 1),
    ("抬高", "z", 1),
    ("升高", "z", 1),
    ("上移", "z", 1),
    ("向下", "z", -1),
    ("往下", "z", -1),
    ("降低", "z", -1),
    ("下降", "z", -1),
    ("下移", "z", -1),
    ("降下", "z", -1),
    ("放下", "z", -1),
    ("向前", "x", 1),
    ("往前", "x", 1),
    ("前进", "x", 1),
    ("前移", "x", 1),
    ("向后", "x", -1),
    ("往后", "x", -1),
    ("后退", "x", -1),
    ("后移", "x", -1),
    ("向左", "y", 1),
    ("往左", "y", 1),
    ("左移", "y", 1),
    ("向右", "y", -1),
    ("往右", "y", -1),
    ("右移", "y", -1),
    ("up", "z", 1),
    ("down", "z", -1),
    ("前", "x", 1),
    ("后", "x", -1),
    ("上", "z", 1),
    ("下", "z", -1),
)

_FINGER_SPECS: Tuple[Tuple[str, str], ...] = (
    ("拇指旋转", "thumb_rotation"),
    ("拇指转动", "thumb_rotation"),
    ("大拇指旋转", "thumb_rotation"),
    ("thumb rotation", "thumb_rotation"),
    ("拇指弯曲", "thumb_bend"),
    ("大拇指", "thumb_bend"),
    ("拇指", "thumb_bend"),
    ("食指", "index"),
    ("中指", "middle"),
    ("无名指", "ring"),
    ("小拇指", "pinky"),
    ("小指", "pinky"),
    ("index", "index"),
    ("middle", "middle"),
    ("ring", "ring"),
    ("pinky", "pinky"),
)

_OPEN_WORDS = (
    "完全张开",
    "张开",
    "打开",
    "松开",
    "放开",
    "摊开",
    "伸开",
    "伸直",
    "open",
    "release",
)
_CLOSE_WORDS = (
    "完全闭合",
    "握拳",
    "握紧",
    "握住",
    "抓住",
    "抓取",
    "闭合",
    "合上",
    "合拢",
    "捏紧",
    "弯曲",
    "握",
    "close",
    "fist",
)
_HALF_WORDS = ("半握", "半开", "半张", "半合")

_SUBJECTS: Tuple[Tuple[str, str, str], ...] = (
    ("左手臂", "left", "arm"),
    ("右手臂", "right", "arm"),
    ("左胳膊", "left", "arm"),
    ("右胳膊", "right", "arm"),
    ("left arm", "left", "arm"),
    ("right arm", "right", "arm"),
    ("both arms", "both", "arm"),
    ("both hands", "both", "hand"),
    ("left hand", "left", "hand"),
    ("right hand", "right", "hand"),
    ("左右臂", "both", "arm"),
    ("双臂", "both", "arm"),
    ("两臂", "both", "arm"),
    ("左右手", "both", "hand"),
    ("两只手", "both", "hand"),
    ("双手", "both", "hand"),
    ("两手", "both", "hand"),
    ("左臂", "left", "arm"),
    ("右臂", "right", "arm"),
    ("左手", "left", "hand"),
    ("右手", "right", "hand"),
)

_STOP_RE = re.compile(
    r"^(?:停止(?:移动|手臂|手)?|停下|取消(?:移动)?|停一下|停|stop|cancel)$",
    re.IGNORECASE,
)
_NUM_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(厘米|公分|cm|毫米|mm|米|m|%)?",
    re.IGNORECASE,
)
_SUBJECT_START_RE = re.compile(
    r"^(?:请|帮我|帮忙|把|将|让)*\s*"
    r"(?:左手臂|右手臂|左胳膊|右胳膊|左臂|右臂|双手|两只手|两手|左右手|"
    r"双臂|两臂|左右臂|左手|右手|left arm|right arm|left hand|right hand|"
    r"both hands|both arms|停止|停下|取消|stop|cancel)",
    re.IGNORECASE,
)


@dataclass
class ArmHandTextAction:
    kind: str  # "arm" | "hand" | "stop"
    side: str  # "left" | "right" | "both"
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    position: Optional[float] = None
    joints: Dict[str, float] = field(default_factory=dict)
    label: str = ""


def canonical_hand_joint(name: str) -> Optional[str]:
    """把关节名归一到 HAND_JOINT_ORDER 中的键。"""
    raw = (name or "").strip().lower().replace("-", "_")
    if not raw:
        return None
    for key in ("thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"):
        if raw == key or raw.endswith("_" + key) or key in raw:
            return key
    return None


def parse_arm_hand_text(text: str) -> Tuple[List[ArmHandTextAction], List[str]]:
    """解析文本。返回 (动作列表, 无法识别的片段)。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u3000", " ").strip()
    if not normalized:
        return [], []
    actions: List[ArmHandTextAction] = []
    errors: List[str] = []
    inherit: Optional[Tuple[str, str]] = None
    for clause in _split_clauses(normalized):
        parsed, err = _parse_clause(clause, inherit)
        if err:
            errors.append(err)
        actions.extend(parsed)
        inherit = _inherit_from(parsed) or inherit
    return actions, errors


def _inherit_from(parsed: List[ArmHandTextAction]) -> Optional[Tuple[str, str]]:
    carried = [item for item in parsed if item.kind in ("arm", "hand")]
    if not carried:
        return None
    if (
        len(carried) >= 2
        and carried[-1].kind == "arm"
        and carried[-2].kind == "arm"
        and {carried[-1].side, carried[-2].side} == {"left", "right"}
        and carried[-1].dx == carried[-2].dx
        and carried[-1].dy == carried[-2].dy
        and carried[-1].dz == carried[-2].dz
    ):
        return ("both", "arm")
    last = carried[-1]
    return (last.side, "arm" if last.kind == "arm" else "hand")


def _split_clauses(text: str) -> List[str]:
    chunks = re.split(r"[\n；;。]+|(?:然后|接着|之后)", text)
    clauses: List[str] = []
    for chunk in chunks:
        parts = re.split(r"[，,、]+", chunk)
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if buf and _SUBJECT_START_RE.match(part):
                clauses.append(buf.strip())
                buf = part
            elif buf:
                buf = f"{buf}，{part}"
            else:
                buf = part
        if buf.strip():
            clauses.append(buf.strip())
    return clauses


def _strip_polite(clause: str) -> str:
    prev = None
    while clause != prev:
        prev = clause
        clause = re.sub(r"^(?:请|帮我|帮忙|把|将|让)\s*", "", clause).strip()
    return clause


def _parse_clause(
    clause: str,
    inherit: Optional[Tuple[str, str]] = None,
) -> Tuple[List[ArmHandTextAction], str]:
    raw = clause.strip()
    clause = _strip_polite(raw)
    if not clause:
        return [], ""
    if _is_stop(clause):
        return [ArmHandTextAction(kind="stop", side="both", label="停止")], ""

    side, prefer = _detect_subject(clause)
    deltas = _extract_deltas(clause)
    fingers = _extract_fingers(clause)
    hand_pos = _parse_hand_position(clause)
    has_arm = deltas is not None
    has_hand = hand_pos is not None or bool(fingers)

    if side is None and not has_arm and not has_hand:
        if re.search(r"手", clause) and "臂" not in clause and not re.search(r"左|右", clause):
            side, prefer = "both", "hand"
            has_hand = hand_pos is not None
        else:
            return [], raw

    if side is None and inherit is not None and (has_arm or has_hand):
        side, prefer = inherit

    if side is None:
        if has_arm and not has_hand:
            return [], f"{raw}（请写明左臂、右臂或双臂）"
        if has_hand and not has_arm:
            return [], f"{raw}（请写明左手、右手或双手）"
        return [], raw

    # 「左手向前」是手臂；「左臂张开」是手。两者都有则按出现顺序执行。
    if prefer == "hand" and has_arm and not has_hand:
        prefer = "arm"
    elif prefer == "arm" and has_hand and not has_arm:
        prefer = "hand"

    arm_action: Optional[ArmHandTextAction] = None
    hand_action: Optional[ArmHandTextAction] = None
    if has_arm and (prefer == "arm" or has_hand):
        dx, dy, dz, clamped = deltas or (0.0, 0.0, 0.0, False)
        if abs(dx) + abs(dy) + abs(dz) < 1e-6:
            return [], f"{raw}（偏移为零）"
        arm_action = _make_arm_action(side, dx, dy, dz, clamped)
    if has_hand and (prefer == "hand" or has_arm):
        if fingers and hand_pos is None:
            return [], f"{raw}（请说明张开、闭合或开合量）"
        pos = 0.0 if hand_pos is None else hand_pos
        joints = {name: pos for name in fingers} if fingers else {}
        hand_side = side
        hand_action = _make_hand_action(hand_side, pos, joints)

    if arm_action is None and hand_action is None:
        return [], raw

    arms = _expand_both_arms(arm_action) if arm_action is not None else []
    hands = [hand_action] if hand_action is not None else []
    if (
        arms
        and hands
        and _hand_word_index(clause) >= 0
        and _hand_word_index(clause) < _dir_word_index(clause)
    ):
        return hands + arms, ""
    return arms + hands, ""


def _is_stop(clause: str) -> bool:
    core = clause
    for phrase, _, _ in _SUBJECTS:
        core = re.sub(re.escape(phrase), " ", core, flags=re.IGNORECASE)
    core = re.sub(r"[\s，,、的了吧啊]+", "", core)
    return bool(_STOP_RE.match(core))


def _detect_subject(clause: str) -> Tuple[Optional[str], str]:
    lower = clause.lower()
    for phrase, side, prefer in _SUBJECTS:
        if phrase.lower() in lower:
            return side, prefer
    if re.search(r"手", clause) and "臂" not in clause and not re.search(r"左|右|left|right", clause, re.I):
        return "both", "hand"
    return None, ""


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _keyword_boundary_ok(text: str, needle: str, start: int, end: int) -> bool:
    """避免「马上」吃掉「上」，或英文单词中间的 up/down。"""
    if needle.isascii():
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if before.isalpha() or after.isalpha():
            return False
        return True
    if len(needle) != 1:
        return True
    if start > 0 and text[start - 1] in "向往":
        return False
    prev = text[start - 1] if start > 0 else ""
    nxt = text[end] if end < len(text) else ""
    prev_ok = start == 0 or not _is_cjk(prev) or prev in "臂手前后上下"
    next_ok = end == len(text) or not _is_cjk(nxt) or nxt in "前后上下"
    return prev_ok and next_ok


def _find_keyword_spans(
    text: str, specs: Tuple[Tuple[str, str, int], ...]
) -> List[Tuple[int, int, str, int]]:
    occupied = [False] * len(text)
    found: List[Tuple[int, int, str, int]] = []
    lower = text.lower()
    for word, axis, sign in sorted(specs, key=lambda item: len(item[0]), reverse=True):
        needle = word.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            start = idx + 1
            if any(occupied[idx:end]):
                continue
            if not _keyword_boundary_ok(text, needle, idx, end):
                continue
            for pos in range(idx, end):
                occupied[pos] = True
            found.append((idx, end, axis, sign))
    found.sort()
    return found


def _extract_deltas(clause: str) -> Optional[Tuple[float, float, float, bool]]:
    spans = _find_keyword_spans(clause, _DIR_SPECS)
    if not spans:
        return None
    dx = dy = dz = 0.0
    clamped = False
    used_num = [False] * len(clause)
    for i, (start, end, axis, sign) in enumerate(spans):
        gap_end = spans[i + 1][0] if i + 1 < len(spans) else min(len(clause), end + 24)
        snippet = clause[end:gap_end]
        meters: Optional[float] = None
        match = _NUM_RE.search(snippet)
        if match:
            abs_start = end + match.start()
            abs_end = end + match.end()
            if not any(used_num[abs_start:abs_end]):
                meters = _length_to_m(float(match.group(1)), match.group(2) or "")
                for pos in range(abs_start, abs_end):
                    used_num[pos] = True
        step = ARM_TEXT_DEFAULT_STEP_M if meters is None else meters
        delta = sign * step
        limited = max(-ARM_TEXT_MAX_OFFSET_M, min(ARM_TEXT_MAX_OFFSET_M, delta))
        if abs(limited - delta) > 1e-9:
            clamped = True
        if axis == "x":
            dx += limited
        elif axis == "y":
            dy += limited
        else:
            dz += limited
    dx = max(-ARM_TEXT_MAX_OFFSET_M, min(ARM_TEXT_MAX_OFFSET_M, dx))
    dy = max(-ARM_TEXT_MAX_OFFSET_M, min(ARM_TEXT_MAX_OFFSET_M, dy))
    dz = max(-ARM_TEXT_MAX_OFFSET_M, min(ARM_TEXT_MAX_OFFSET_M, dz))
    return dx, dy, dz, clamped


def _length_to_m(number: float, unit: str) -> float:
    unit = (unit or "").lower()
    if unit in ("厘米", "公分", "cm"):
        return number / 100.0
    if unit in ("毫米", "mm"):
        return number / 1000.0
    if unit in ("米", "m"):
        return number
    if abs(number) >= 1.0:
        return number / 100.0
    return number


def _extract_fingers(clause: str) -> List[str]:
    lower = clause.lower()
    occupied = [False] * len(clause)
    found: List[str] = []
    for phrase, joint in sorted(_FINGER_SPECS, key=lambda item: len(item[0]), reverse=True):
        needle = phrase.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            start = idx + 1
            if any(occupied[idx:end]):
                continue
            for pos in range(idx, end):
                occupied[pos] = True
            if joint not in found:
                found.append(joint)
    return found


def _parse_hand_position(clause: str) -> Optional[float]:
    match = re.search(
        r"(?:到|至|to)\s*([+-]?\d+(?:\.\d+)?)\s*(%)?",
        clause,
        re.IGNORECASE,
    )
    if match:
        return _clamp01(_hand_number(float(match.group(1)), match.group(2) or ""))
    if any(word in clause for word in _HALF_WORDS):
        return 0.5
    lower = clause.lower()
    if any(word in clause or word in lower for word in _OPEN_WORDS):
        return 0.0
    if any(word in clause or word in lower for word in _CLOSE_WORDS) or clause.rstrip().endswith("合"):
        return 1.0
    return None


def _hand_number(number: float, unit: str) -> float:
    if unit == "%" or abs(number) > 1.0:
        return number / 100.0
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _side_name(side: str, kind: str) -> str:
    if kind == "arm":
        return {"left": "左臂", "right": "右臂", "both": "双臂"}.get(side, side)
    return {"left": "左手", "right": "右手", "both": "双手"}.get(side, side)


def _make_arm_action(
    side: str, dx: float, dy: float, dz: float, clamped: bool
) -> ArmHandTextAction:
    label = (
        f"{_side_name(side, 'arm')} Δ({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m"
    )
    if clamped:
        label += "（已限制到 ±0.5m）"
    return ArmHandTextAction(
        kind="arm", side=side, dx=dx, dy=dy, dz=dz, label=label
    )


def _expand_both_arms(action: ArmHandTextAction) -> List[ArmHandTextAction]:
    if action.side != "both":
        return [action]
    out = []
    for side in ("left", "right"):
        out.append(
            ArmHandTextAction(
                kind="arm",
                side=side,
                dx=action.dx,
                dy=action.dy,
                dz=action.dz,
                label=action.label.replace("双臂", _side_name(side, "arm"), 1),
            )
        )
    return out


def _make_hand_action(
    side: str, position: float, joints: Dict[str, float]
) -> ArmHandTextAction:
    if joints:
        detail = " ".join(
            f"{_JOINT_LABELS.get(name, name)}={val:.2f}" for name, val in joints.items()
        )
        label = f"{_side_name(side, 'hand')} {detail}"
    else:
        label = f"{_side_name(side, 'hand')} 开合={position:.2f}"
    return ArmHandTextAction(
        kind="hand",
        side=side,
        position=position,
        joints=dict(joints),
        label=label,
    )


def _dir_word_index(clause: str) -> int:
    spans = _find_keyword_spans(clause, _DIR_SPECS)
    return spans[0][0] if spans else 10**9


def _hand_word_index(clause: str) -> int:
    indexes = []
    lower = clause.lower()
    for word in list(_FINGER_SPECS) + [(w, "") for w in _OPEN_WORDS + _CLOSE_WORDS + _HALF_WORDS]:
        phrase = word[0]
        idx = lower.find(phrase.lower())
        if idx >= 0:
            indexes.append(idx)
    return min(indexes) if indexes else -1
