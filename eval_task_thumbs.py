"""Illustrated thumbnails for eval-task pickers.

Each task name (plus RoboDojo object categories, when a matching YAML exists)
becomes a small scene: the objects involved and the action, with a short
Chinese caption. The simulation UI shows these in the task dropdown and next
to the current selection.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

THUMB_W = 240
THUMB_H = 136
_OUTLINE = "#243040"
_CFG_DIR = os.environ.get(
    "ROBODOJO_TASK_CONFIG",
    "/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/task/RoboDojo/config",
)

# (needle, kind, priority). Longer needles are matched first.
_OBJECT_RULES: Tuple[Tuple[str, str, int], ...] = (
    ("matryoshka", "doll", 5),
    ("nesting doll", "doll", 5),
    ("xylophone", "xylophone", 5),
    ("tic tac toe", "tictactoe", 5),
    ("checkerboard", "tictactoe", 5),
    ("stacking toy", "ringtoy", 5),
    ("stackingblocks", "ringtoy", 5),
    ("mahjong", "mahjong", 5),
    ("piggy bank", "piggy", 5),
    ("piggy", "piggy", 5),
    ("dustbin", "bin", 4),
    ("toolbox", "toolbox", 4),
    ("toaster", "toaster", 5),
    ("headphone", "headphone", 5),
    ("headset", "headphone", 5),
    ("laptop", "laptop", 5),
    ("conveyor", "conveyor", 4),
    ("wine bottle", "wine", 5),
    ("wuliangye", "bottle", 5),
    ("bottle", "bottle", 5),
    ("goblet", "cup", 4),
    ("ramekin", "ramekin", 4),
    ("mug", "mug", 5),
    ("bowl", "bowl", 5),
    ("plate", "plate", 4),
    ("vase", "vase", 5),
    ("sphere", "ball", 4),
    ("shoe", "shoe", 5),
    ("hammer", "hammer", 5),
    ("wrench", "tool", 4),
    ("plier", "tool", 4),
    ("mallet", "hammer", 4),
    ("broom", "broom", 5),
    ("bell", "bell", 5),
    ("apple", "apple", 5),
    ("tomato", "tomato", 5),
    ("orange juice", "carton", 4),
    ("juice", "carton", 4),
    ("milk", "carton", 4),
    ("butter", "butter", 4),
    ("cheese", "cheese", 4),
    ("pudding", "pudding", 4),
    ("chocolate", "chocolate", 4),
    ("cookie", "cookie", 4),
    ("alphabet soup", "can", 4),
    ("soup", "can", 4),
    ("ketchup", "bottle", 4),
    ("dressing", "bottle", 4),
    ("sauce", "bottle", 4),
    ("moka", "moka", 5),
    ("pot", "pot", 4),
    ("microwave", "microwave", 4),
    ("stove", "stove", 4),
    ("drawer", "drawer", 4),
    ("cabinet", "cabinet", 4),
    ("basket", "basket", 4),
    ("book", "book", 4),
    ("coin", "coin", 5),
    ("key", "key", 5),
    ("test tube", "tube", 5),
    ("charger", "plug", 5),
    ("socket", "socket", 3),
    ("screw", "screw", 5),
    ("bolt", "screw", 5),
    ("factory nut", "nut", 3),
    ("egg", "egg", 5),
    ("oil pen", "pen", 4),
    ("pen", "pen", 4),
    ("bread", "toast", 4),
    ("toast", "toast", 4),
    ("cloth", "cloth", 5),
    ("top long", "cloth", 5),
    ("triangular prism", "prism", 4),
    ("small cube", "block", 3),
    ("block", "block", 3),
    ("cube", "block", 3),
    ("brick", "block", 3),
    ("number card", "digits", 4),
    ("number", "digits", 4),
    ("springbutton", "button", 5),
    ("button", "button", 4),
    ("toy car", "car", 4),
    ("watch", "watch", 4),
    ("phone", "phone", 4),
    ("camera", "camera", 3),
    ("donut", "donut", 4),
    ("ice cream", "icecream", 4),
    ("pepper", "pepper", 3),
    ("toothbrush", "brush", 3),
    ("figurine", "doll", 3),
    ("chessman", "pawn", 4),
    ("can", "can", 4),
    ("cup", "cup", 4),
    ("box", "box", 3),
    ("car", "car", 4),
    ("ball", "ball", 4),
    ("tee", "tee", 5),
)

def _bounded_needle(needle: str) -> str:
    # Optional plural, but not a prefix of a longer word ("pen" must not hit "open").
    return rf"(?<![a-z0-9]){re.escape(needle)}s?(?![a-z0-9])"


_OBJ_RE = re.compile(
    "|".join(
        _bounded_needle(n)
        for n, _k, _p in sorted(_OBJECT_RULES, key=lambda r: -len(r[0]))
    ),
    re.I,
)
_OBJ_MAP = {n: (k, p) for n, k, p in _OBJECT_RULES}

_COLOR_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("black", "黑", "#2c2c2c"),
    ("white", "白", "#f4f4f4"),
    ("yellow", "黄", "#f0c419"),
    ("red", "红", "#d94b3c"),
    ("blue", "蓝", "#3c7ed6"),
    ("green", "绿", "#3e9a4c"),
    ("orange", "橙", "#e57a2a"),
    ("pink", "粉", "#e57aa0"),
    ("brown", "棕", "#8d5a34"),
    ("wooden", "木", "#c49a6c"),
    ("chocolate", "巧克力", "#6b3e26"),
    ("purple", "紫", "#7a5cb8"),
)

_OBJ_ZH = {
    "doll": "套娃",
    "xylophone": "木琴",
    "tictactoe": "井字棋",
    "ringtoy": "套柱玩具",
    "mahjong": "麻将牌",
    "piggy": "存钱罐",
    "bin": "垃圾桶",
    "toolbox": "工具箱",
    "toaster": "烤面包机",
    "headphone": "耳机",
    "laptop": "笔记本",
    "conveyor": "传送带",
    "wine": "酒瓶",
    "bottle": "瓶子",
    "cup": "杯子",
    "ramekin": "小碗",
    "mug": "马克杯",
    "bowl": "碗",
    "plate": "盘子",
    "vase": "花瓶",
    "ball": "球",
    "shoe": "鞋子",
    "hammer": "锤子",
    "tool": "工具",
    "broom": "扫帚",
    "bell": "铃铛",
    "apple": "苹果",
    "tomato": "番茄",
    "carton": "纸盒",
    "butter": "黄油",
    "cheese": "奶酪",
    "pudding": "布丁",
    "chocolate": "巧克力",
    "cookie": "饼干",
    "can": "罐头",
    "moka": "摩卡壶",
    "pot": "锅",
    "microwave": "微波炉",
    "stove": "炉灶",
    "drawer": "抽屉",
    "cabinet": "柜子",
    "basket": "篮子",
    "book": "书",
    "coin": "硬币",
    "key": "钥匙",
    "tube": "试管",
    "plug": "充电器",
    "socket": "插座",
    "screw": "螺丝",
    "nut": "螺母",
    "egg": "鸡蛋",
    "pen": "笔",
    "toast": "面包",
    "cloth": "衣服",
    "prism": "三棱柱",
    "block": "方块",
    "digits": "数字牌",
    "button": "按钮",
    "car": "小车",
    "watch": "手表",
    "phone": "手机",
    "camera": "相机",
    "donut": "甜甜圈",
    "icecream": "冰淇淋",
    "pepper": "胡椒",
    "brush": "牙刷",
    "pawn": "棋子",
    "box": "盒子",
    "tee": "T 形块",
}

_DEFAULT_FILL = {
    "apple": "#e24b3b",
    "tomato": "#d63c30",
    "bottle": "#4aa0e0",
    "wine": "#7a3058",
    "cup": "#f7f7f7",
    "mug": "#f3f1ec",
    "bowl": "#f7f4ef",
    "ramekin": "#f3e6c8",
    "plate": "#fbfbfb",
    "block": "#e07a32",
    "prism": "#5aa0d6",
    "can": "#d24a3a",
    "carton": "#f4f7fb",
    "cheese": "#f2c14e",
    "butter": "#f6de7a",
    "pudding": "#c47a45",
    "chocolate": "#6b3e26",
    "cookie": "#d7a15a",
    "ball": "#3c7ed6",
    "coin": "#e2b84a",
    "egg": "#f3ead4",
    "toast": "#e0b15a",
    "cloth": "#6aa0d4",
    "shoe": "#4a5560",
    "mahjong": "#f7f4ea",
    "doll": "#e25b6a",
    "key": "#e2b84a",
    "pen": "#2f6fad",
    "tube": "#d8f2f6",
    "plug": "#f2f2f2",
    "screw": "#c5ccd4",
    "bell": "#f0c419",
    "hammer": "#8d5a34",
    "tool": "#8aa0b0",
    "broom": "#c49a6c",
    "laptop": "#d5dbe3",
    "headphone": "#3a4250",
    "book": "#3c6eab",
    "bin": "#5f6b76",
    "box": "#c4924a",
    "basket": "#c49a6c",
    "toolbox": "#d6453d",
    "toaster": "#d0d6de",
    "piggy": "#f0a0b4",
    "vase": "#6aa8c8",
    "pot": "#8a9098",
    "moka": "#c0c6ce",
    "cabinet": "#c4a06a",
    "drawer": "#c4a06a",
    "microwave": "#d5dbe3",
    "stove": "#4a515a",
    "car": "#e24b3b",
    "watch": "#2c333c",
    "phone": "#3a4250",
    "camera": "#2c333c",
    "donut": "#e8a05a",
    "icecream": "#f3c1d4",
    "pepper": "#6a9a45",
    "brush": "#4aa0e0",
    "button": "#d6453d",
    "digits": "#f7f4ea",
    "tee": "#3c7ed6",
    "xylophone": "#e07a32",
    "tictactoe": "#f7f4ea",
    "pawn": "#f2f2f2",
    "ringtoy": "#e24b3b",
    "conveyor": "#6a7380",
    "socket": "#f2f2f2",
    "nut": "#c5ccd4",
}

_TAG = {
    "pick": "抓取",
    "place": "放置",
    "pick_place": "抓放",
    "open": "打开",
    "close": "关闭",
    "open_place": "打开放入",
    "place_close": "放入并关",
    "pour": "倾倒",
    "stack": "叠放",
    "push": "推动",
    "swap": "交换",
    "hang": "悬挂",
    "insert": "插入",
    "fold": "折叠",
    "sort": "整理",
    "press": "按压",
    "handover": "交接",
    "cover": "盖住",
    "fill": "装填",
    "adjust": "摆正",
    "align": "对齐",
    "play": "操作",
    "toast": "烤制",
    "sweep": "清扫",
    "lineup": "排齐",
    "equation": "算式",
    "general": "任务",
}

_STRIPE = {
    "pick": "#3a7bd5",
    "place": "#3e9a4c",
    "pick_place": "#1f9d8a",
    "open": "#e09a2b",
    "close": "#c46b2f",
    "open_place": "#d4892a",
    "place_close": "#b86a32",
    "pour": "#3aa0c8",
    "stack": "#7a5ea8",
    "push": "#c45b7a",
    "swap": "#5b8fc4",
    "hang": "#8a6a3b",
    "insert": "#3f8f62",
    "fold": "#b07aaa",
    "sort": "#6a8f3a",
    "press": "#d06060",
    "handover": "#4a6fa5",
    "cover": "#8a7040",
    "fill": "#2f8f8f",
    "adjust": "#c48a3a",
    "align": "#5a7a9a",
    "play": "#d06a9a",
    "toast": "#d08a4a",
    "sweep": "#7a8a4a",
    "lineup": "#6a6a9a",
    "equation": "#3d6ea8",
    "general": "#6a7380",
}

_ACCESSORY_SUFFIXES = (
    "_cushion",
    "_stand",
    "_slot",
    "_shelf",
    "_rack",
    "_holder",
    "_storage",
    "_shovel",
    "_mat",
)

_CAT_CACHE: Dict[str, Tuple[str, ...]] = {}
_PARSE_CACHE: Dict[str, "TaskScene"] = {}
_PIX_CACHE: Dict[str, QPixmap] = {}


@dataclass
class TaskScene:
    action: str = "general"
    objects: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    badge: str = ""
    tag: str = "任务"
    gloss: str = "执行该评测任务"
    source: str = ""


def _normalize(text: str) -> str:
    s = (text or "").lower().replace("_", " ").replace("-", " ")
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    # A few Chinese phrases used in labels.
    repl = (
        ("苹果", " apple "),
        ("番茄", " tomato "),
        ("西红柿", " tomato "),
        ("瓶子", " bottle "),
        ("杯子", " cup "),
        ("碗", " bowl "),
        ("盘子", " plate "),
        ("抽屉", " drawer "),
        ("柜子", " cabinet "),
        ("抓取", " pick "),
        ("拿起", " pick "),
        ("放置", " place "),
        ("打开", " open "),
        ("关闭", " close "),
        ("关上", " close "),
        ("叠", " stack "),
        ("倒", " pour "),
    )
    raw = text or ""
    for zh, en in repl:
        if zh in raw:
            s += en
    return s


def _categories_for(task_name: str) -> Tuple[str, ...]:
    if task_name in _CAT_CACHE:
        return _CAT_CACHE[task_name]
    path = os.path.join(_CFG_DIR, task_name + ".yml")
    found: List[str] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except OSError:
            raw = ""
        for name in re.findall(r'name:\s*"([^"]+)"', raw):
            low = name.lower()
            if any(low.endswith(suf) for suf in _ACCESSORY_SUFFIXES):
                continue
            if name not in found:
                found.append(name)
    got = tuple(found[:8])
    _CAT_CACHE[task_name] = got
    return got


def _enrich(text: str) -> str:
    extra: List[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text or ""):
        for cat in _categories_for(tok):
            if cat not in extra:
                extra.append(cat)
    if not extra:
        return text or ""
    return (text or "") + " " + " ".join(extra)


def _find_objects(norm: str) -> List[str]:
    found: List[Tuple[int, str, int]] = []
    for match in _OBJ_RE.finditer(norm):
        key = match.group(0).lower()
        if key not in _OBJ_MAP and key.endswith("s"):
            key = key[:-1]
        mapped = _OBJ_MAP.get(key)
        if mapped is None:
            continue
        kind, pri = mapped
        found.append((match.start(), kind, pri))
    for match in re.finditer(r"\bt\b", norm):
        found.append((match.start(), "tee", 5))
    found.sort()
    ordered: List[Tuple[str, int]] = []
    seen = set()
    for _pos, kind, pri in found:
        if kind in seen:
            continue
        seen.add(kind)
        ordered.append((kind, pri))
    mains = [k for k, p in ordered if p >= 3]
    if not mains:
        mains = [k for k, _p in ordered]
    # Keep a few extras so a container mentioned later (basket, drawer) is not dropped.
    return mains[:6]


# Color words that belong to an object name, not a tint ("wooden toy", "chocolate bar").
_COLOR_PHRASES = (
    "wooden toy",
    "chocolate bar",
    "chocolate pudding",
    "cream cheese",
)


def _find_colors(norm: str) -> List[str]:
    scan = norm
    for phrase in _COLOR_PHRASES:
        scan = scan.replace(phrase, " ")
    colors: List[str] = []
    for name, _zh, _hex in _COLOR_RULES:
        if re.search(rf"(?<![a-z0-9]){name}(?![a-z0-9])", scan) and name not in colors:
            colors.append(name)
    return colors


# Containers / fixtures. For pick-and-place they are the destination, not the thing held.
_BACKDROP = {
    "conveyor",
    "basket",
    "plate",
    "drawer",
    "cabinet",
    "box",
    "bin",
    "toolbox",
    "stove",
    "microwave",
    "socket",
    "piggy",
    "toaster",
}
_POUR_INTO = {"cup", "mug", "bowl", "vase", "ramekin", "pot"}
_TOOLS = {"broom", "hammer", "tool"}


def _roles(action: str, kinds: Sequence[str]) -> Tuple[List[str], List[str]]:
    if action in ("open", "close"):
        if "drawer" in kinds:
            return ["drawer"], []
        if "cabinet" in kinds:
            return ["cabinet"], []
        return list(kinds[:1]), []
    backdrop = set(_BACKDROP)
    if action == "pour":
        backdrop |= _POUR_INTO
    if action in ("sweep", "press"):
        backdrop |= _TOOLS
    mains = [k for k in kinds if k not in backdrop]
    backs = [k for k in kinds if k in backdrop]
    return mains, backs


def _has(norm: str, *keys: str) -> bool:
    return any(k in norm for k in keys)


def _detect_action(norm: str) -> str:
    has_place = _has(norm, "place", " put", "put ", "pack", "store", "deposit", "move ")
    if norm.startswith("put ") or " put " in f" {norm} ":
        has_place = True
    has_open = bool(re.search(r"\bopen\b", norm))
    has_close = bool(re.search(r"\bclose\b", norm))
    has_pick = _has(norm, "pick", "lift", "pickup", "grasp")
    if _has(norm, "pour"):
        return "pour"
    if _has(norm, "fold"):
        return "fold"
    if _has(norm, "handover"):
        return "handover"
    if _has(norm, "toast", "toaster"):
        return "toast"
    if _has(norm, "equation", "solve"):
        return "equation"
    if _has(norm, "xylophone", "tic tac", "stacking toy", "stackingblocks"):
        return "play"
    if _has(norm, "play"):
        return "play"
    if _has(norm, "kong", "mahjong"):
        return "lineup"
    if _has(norm, "sweep"):
        return "sweep"
    if _has(norm, "fasten", "plug", "insert"):
        return "insert"
    if _has(norm, "hang"):
        return "hang"
    if _has(norm, "swap"):
        return "swap"
    if _has(norm, "push"):
        return "push"
    if _has(norm, "align"):
        return "align"
    if _has(norm, "adjust"):
        return "adjust"
    if _has(norm, "stack", "tower", "build"):
        return "stack"
    if _has(norm, "cover"):
        return "cover"
    if _has(norm, "fill"):
        return "fill"
    if _has(norm, "classify", "arrange", "organize", "imitate", "sorting", "sort"):
        return "sort"
    if has_place and has_close:
        return "place_close"
    if has_open and has_place:
        return "open_place"
    if has_place:
        return "pick_place"
    if has_open:
        return "open"
    if has_close:
        return "close"
    if _has(norm, "press", "click", "beat"):
        return "press"
    if has_pick:
        return "pick"
    return "general"


def _suite_override(norm: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Named benchmark suites that are not a single manipulation phrase."""
    key = norm.strip()
    table = (
        ("simplerenv", "pick", ("can",)),
        ("simpler", "pick", ("can",)),
        ("calvin", "open_place", ("block", "drawer")),
        ("robocasa", "pick_place", ("mug", "cabinet")),
        ("maniskill", "pick", ("block",)),
        ("genesis", "pick", ("block",)),
        ("vla harness", "sort", ("block", "box")),
        ("harness", "sort", ("block", "box")),
    )
    for name, action, objs in table:
        if key == name or key.startswith(name + " "):
            return action, objs
    return None


def _badge(norm: str) -> str:
    if _has(norm, "rand cam", "random camera", "randomized"):
        return "随机视角"
    if _has(norm, "by image", "photo"):
        return "看图"
    if _has(norm, "filament"):
        return "细丝"
    if _has(norm, "language"):
        return "语言"
    if re.search(r"\brandom\b", norm):
        return "随机"
    if _has(norm, "next to", "nextto"):
        return "旁边"
    if re.search(r"\bcolor\b", norm):
        return "颜色"
    if _has(norm, "classic"):
        return "经典"
    return ""


def _obj_phrase(kinds: Sequence[str], colors: Sequence[str]) -> str:
    parts: List[str] = []
    color_zh = {name: zh for name, zh, _hex in _COLOR_RULES}
    for i, kind in enumerate(kinds[:3]):
        name = _OBJ_ZH.get(kind, "物体")
        if i < len(colors):
            name = color_zh.get(colors[i], "") + name
        parts.append(name)
    if not parts:
        return "物体"
    if len(parts) == 1:
        return parts[0]
    return "、".join(parts[:-1]) + "和" + parts[-1]


def _gloss(action: str, obj: str, kinds: Sequence[str], backs: Sequence[str]) -> str:
    if "mahjong" in kinds:
        return "把麻将牌排成一组"
    if action == "equation":
        return "用数字牌和符号拼出算式"
    if "tictactoe" in kinds:
        return "在棋盘上落子"
    if "xylophone" in kinds:
        return "用琴槌敲击木琴"
    if "piggy" in backs or "piggy" in kinds:
        return "把硬币投进存钱罐"
    dest = _OBJ_ZH.get(backs[0], "") if backs else ""
    if action == "pour" and dest:
        return f"把{obj}倒进{dest}"
    if action in ("pick_place", "place_close", "open_place", "place") and dest:
        return f"把{obj}放到{dest}"
    if action == "pick" and "conveyor" in kinds:
        return f"从传送带上拿起{obj}"
    if action == "fill":
        return f"把{obj}装进容器"
    if action == "sort" and dest:
        if obj == dest or not obj:
            return "把散落的物品分类收好"
        return f"把{obj}分类放进{dest}"
    if action == "press" and "hammer" in kinds:
        return f"用锤子敲{obj}"
    if action == "press" and "bell" in kinds:
        return "按响铃铛"
    templates = {
        "pick": "拿起{obj}",
        "place": "把{obj}放到指定位置",
        "pick_place": "拿起{obj}，再放到目标位置",
        "open": "打开{obj}",
        "close": "关上{obj}",
        "open_place": "打开后把{obj}放进去",
        "place_close": "把{obj}放进去并关上",
        "pour": "把{obj}倒进容器",
        "stack": "把{obj}叠起来",
        "push": "把{obj}往前推",
        "swap": "交换{obj}的位置",
        "hang": "把{obj}挂起来",
        "insert": "把{obj}插进去",
        "fold": "折叠{obj}",
        "sort": "把{obj}分类收好",
        "press": "按下{obj}",
        "handover": "把{obj}交到另一只手",
        "cover": "盖住{obj}",
        "fill": "把{obj}装进容器",
        "adjust": "把{obj}摆正",
        "align": "把{obj}排齐",
        "play": "完成{obj}操作",
        "toast": "把面包放进烤面包机",
        "sweep": "把{obj}扫到一起",
        "lineup": "把{obj}排成一排",
        "general": "执行与{obj}相关的操作",
    }
    return templates.get(action, "执行该评测任务").format(obj=obj)


def parse_task_scene(text: str) -> TaskScene:
    key = (text or "").strip()
    cached = _PARSE_CACHE.get(key)
    if cached is not None:
        return cached
    enriched = _enrich(key)
    norm_src = _normalize(key)
    norm = _normalize(enriched)
    action = _detect_action(norm_src)
    objects = _find_objects(norm)
    suite = _suite_override(norm_src)
    if suite is not None:
        action, objects = suite[0], list(suite[1])
    if action == "general":
        # Categories sometimes carry the verb the short label missed.
        action = _detect_action(norm)
    if not objects:
        defaults = {
            "stack": ["block"],
            "pour": ["bottle", "cup"],
            "open": ["cabinet"],
            "close": ["cabinet"],
            "open_place": ["bowl", "drawer"],
            "place_close": ["bowl", "drawer"],
            "fold": ["cloth"],
            "toast": ["toaster", "toast"],
            "sweep": ["broom", "block"],
            "hang": ["mug"],
            "insert": ["key"],
            "press": ["button"],
            "swap": ["block", "block"],
            "push": ["tee"],
            "lineup": ["mahjong"],
            "equation": ["digits"],
            "play": ["block"],
            "cover": ["cup", "block"],
            "fill": ["egg"],
            "handover": ["block"],
            "pick": ["block"],
            "pick_place": ["block", "plate"],
            "sort": ["basket"],
        }
        objects = list(defaults.get(action, ["block"]))
    colors = _find_colors(norm)
    badge = _badge(norm_src) or _badge(norm)
    mains, backs = _roles(action, objects)
    spoken = mains or objects
    # Keep the destination in frame even when several objects were named.
    if backs:
        objects = list(dict.fromkeys([*spoken[:2], backs[0]]))
    else:
        objects = list(spoken[:3])
    obj = _obj_phrase(spoken[:2] or spoken, colors)
    scene = TaskScene(
        action=action,
        objects=objects,
        colors=colors,
        badge=badge,
        tag=_TAG.get(action, "任务"),
        gloss=_gloss(action, obj, objects, backs),
        source=key,
    )
    if len(_PARSE_CACHE) > 800:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[key] = scene
    return scene


def describe_eval_task(text: str) -> str:
    scene = parse_task_scene(text)
    if scene.badge:
        return f"{scene.gloss}（{scene.badge}）"
    return scene.gloss


def combo_item_task_blob(combo, index: int) -> str:
    text = combo.itemText(index)
    data = combo.itemData(index)
    bits = [text]
    if isinstance(data, str) and data and data != text:
        bits.append(data)
    elif isinstance(data, dict):
        for key in ("label", "id"):
            val = str(data.get(key) or "")
            if val and val not in bits:
                bits.append(val)
    return " ".join(bits)


def stamp_eval_task_combo(combo) -> None:
    """Mark a combo for task thumbnails; paint pixmaps lazily on first show."""
    for i in range(combo.count()):
        blob = combo_item_task_blob(combo, i)
        gloss = describe_eval_task(blob)
        combo.setItemData(i, f"{combo.itemText(i)}\n{gloss}", Qt.ToolTipRole)
    # Empty dict = enabled + lazy. Never paint the whole list on backend switch.
    combo._eval_task_thumbs = {}  # type: ignore[attr-defined]


def combo_task_thumb(combo, index: int) -> Optional[QPixmap]:
    thumbs = getattr(combo, "_eval_task_thumbs", None)
    if not isinstance(thumbs, dict):
        return None
    pix = thumbs.get(index)
    if isinstance(pix, QPixmap) and not pix.isNull():
        return pix
    try:
        blob = combo_item_task_blob(combo, index)
    except Exception:
        return None
    pix = eval_task_thumbnail(blob)
    thumbs[index] = pix
    return pix


class EvalTaskThumbLabel(QLabel):
    """Current-task preview. Call ``show_text`` or bind it to a combo."""

    def __init__(self, parent=None, width: int = 168, height: int = 96) -> None:
        super().__init__(parent)
        self._tw = int(width)
        self._th = int(height)
        self._caption: Optional[QLabel] = None
        self.setFixedSize(self._tw, self._th)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background: #1b1d22; border: 1px solid #5c6370; border-radius: 6px; }"
        )

    def set_caption_label(self, label: Optional[QLabel]) -> None:
        self._caption = label

    def show_text(self, text: str) -> None:
        scene_text = (text or "").strip()
        if not scene_text:
            blank = QPixmap(max(1, self._tw - 6), max(1, self._th - 6))
            blank.fill(QColor("#2a2d33"))
            self.setPixmap(blank)
            self.setToolTip("选择一个评测任务")
            if self._caption is not None:
                self._caption.setText("未选择任务\n打开下拉列表可看每个任务的场景缩略图")
            return
        pix = eval_task_thumbnail(scene_text)
        self.setPixmap(
            pix.scaled(
                self._tw - 6,
                self._th - 6,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        gloss = describe_eval_task(scene_text) if scene_text else "选择一个评测任务"
        self.setToolTip(f"{scene_text}\n{gloss}" if scene_text else gloss)
        if self._caption is not None:
            title = scene_text.strip() or "未选择任务"
            if len(title) > 80:
                title = title[:77] + "…"
            self._caption.setText(f"{title}\n{gloss}")

    def show_combo_current(self, combo) -> None:
        idx = combo.currentIndex()
        typed = combo.currentText().strip() if combo.isEditable() else ""
        if idx >= 0 and (not typed or combo.itemText(idx) == typed):
            self.show_text(combo_item_task_blob(combo, idx))
            return
        self.show_text(typed)


def bind_eval_task_thumb(combo, preview: EvalTaskThumbLabel) -> None:
    def _refresh(*_args) -> None:
        preview.show_combo_current(combo)

    combo.currentIndexChanged.connect(_refresh)
    if combo.isEditable():
        combo.currentTextChanged.connect(_refresh)
    _refresh()


def make_eval_task_preview(parent=None) -> Tuple[QWidget, EvalTaskThumbLabel, QLabel]:
    """Horizontal row: thumbnail plus the one-line description."""
    wrap = QWidget(parent)
    layout = QHBoxLayout(wrap)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    thumb = EvalTaskThumbLabel(wrap)
    caption = QLabel("选择一个评测任务", wrap)
    caption.setWordWrap(True)
    caption.setStyleSheet("color: #d0d4dc;")
    thumb.set_caption_label(caption)
    layout.addWidget(thumb, 0, Qt.AlignTop)
    layout.addWidget(caption, 1)
    return wrap, thumb, caption


def _font(px: int, bold: bool = True) -> QFont:
    font = QFont("Droid Sans Fallback")
    font.setPixelSize(px)
    font.setBold(bold)
    return font


def _pen(color: str = _OUTLINE, width: float = 1.6) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    return pen


def _fill(p: QPainter, color: str, outline: str = _OUTLINE, width: float = 1.6) -> None:
    p.setPen(_pen(outline, width))
    p.setBrush(QColor(color))


def _round(p, x, y, w, h, rad, color: str) -> None:
    _fill(p, color)
    p.drawRoundedRect(QRectF(x, y, w, h), rad, rad)


def _oval(p, cx, cy, w, h, color: str) -> None:
    _fill(p, color)
    p.drawEllipse(QRectF(cx - w / 2, cy - h / 2, w, h))


def _poly(p, pts, color: str) -> None:
    _fill(p, color)
    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))


def _lighter(hex_color: str, amount: int = 130) -> str:
    return QColor(hex_color).lighter(amount).name()


def _darker(hex_color: str, amount: int = 140) -> str:
    return QColor(hex_color).darker(amount).name()


def _color_of(scene: TaskScene, index: int, kind: str) -> str:
    if index < len(scene.colors):
        for name, _zh, hex_color in _COLOR_RULES:
            if name == scene.colors[index]:
                return hex_color
    return _DEFAULT_FILL.get(kind, "#d0a060")


def _cube(p, cx, bottom, s, color: str) -> None:
    w, h, d = 30 * s, 24 * s, 12 * s
    x, y = cx - w / 2, bottom - h
    _poly(p, [(x, y + h), (x + w, y + h), (x + w, y), (x, y)], color)
    _poly(
        p,
        [(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)],
        _lighter(color, 125),
    )
    _poly(
        p,
        [(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)],
        _darker(color, 125),
    )


def _bottle(p, cx, bottom, s, color: str, tilt: float = 0) -> None:
    p.save()
    p.translate(cx, bottom)
    p.rotate(tilt)
    _round(p, -9 * s, -34 * s, 18 * s, 34 * s, 6 * s, color)
    _round(p, -5 * s, -48 * s, 10 * s, 16 * s, 3 * s, _lighter(color, 115))
    _round(p, -6 * s, -52 * s, 12 * s, 6 * s, 2 * s, _darker(color, 120))
    p.restore()


def _cup(p, cx, bottom, s, color: str, handle: bool = False) -> None:
    _poly(
        p,
        [
            (cx - 14 * s, bottom - 26 * s),
            (cx + 14 * s, bottom - 26 * s),
            (cx + 11 * s, bottom),
            (cx - 11 * s, bottom),
        ],
        color,
    )
    _oval(p, cx, bottom - 26 * s, 28 * s, 8 * s, _lighter(color, 115))
    if handle:
        p.setBrush(Qt.NoBrush)
        p.setPen(_pen(_darker(color, 140), 2.2))
        p.drawArc(QRectF(cx + 8 * s, bottom - 22 * s, 14 * s, 16 * s), -50 * 16, 200 * 16)


def _bowl(p, cx, bottom, s, color: str) -> None:
    _poly(
        p,
        [
            (cx - 18 * s, bottom - 16 * s),
            (cx + 18 * s, bottom - 16 * s),
            (cx + 11 * s, bottom),
            (cx - 11 * s, bottom),
        ],
        color,
    )
    _oval(p, cx, bottom - 16 * s, 36 * s, 8 * s, _darker(color, 120))


def _plate(p, cx, bottom, s, color: str) -> None:
    _oval(p, cx, bottom - 4 * s, 46 * s, 14 * s, color)
    _oval(p, cx, bottom - 5 * s, 30 * s, 8 * s, _darker(color, 110))


def _apple(p, cx, bottom, s, color: str) -> None:
    _oval(p, cx, bottom - 16 * s, 28 * s, 30 * s, color)
    p.setPen(_pen("#3e7a32", 1.8))
    p.drawLine(QPointF(cx, bottom - 30 * s), QPointF(cx + 2 * s, bottom - 38 * s))
    _oval(p, cx + 8 * s, bottom - 34 * s, 12 * s, 7 * s, "#5eaf45")


def _cabinet(p, cx, bottom, s, color: str, door: float = 0.0) -> None:
    """``door`` 0 = shut, 1 = open to the right."""
    w, h = 54 * s, 48 * s
    x, y = cx - w / 2, bottom - h
    _round(p, x, y, w, h, 3 * s, color)
    _round(p, x + 4 * s, y + 4 * s, w - 8 * s, h * 0.42, 2 * s, _lighter(color, 112))
    _round(p, x + 4 * s, y + h * 0.52, w - 8 * s, h * 0.40, 2 * s, _lighter(color, 112))
    if door > 0.15:
        dw = (w - 10 * s) * (0.35 + 0.45 * door)
        _poly(
            p,
            [
                (x + w - 6 * s, y + 6 * s),
                (x + w - 6 * s + dw, y + 2 * s),
                (x + w - 6 * s + dw, y + h - 4 * s),
                (x + w - 6 * s, y + h - 8 * s),
            ],
            _lighter(color, 140),
        )
    else:
        p.setPen(_pen("#5a4632", 2))
        p.drawLine(QPointF(cx, y + 8 * s), QPointF(cx, bottom - 6 * s))


def _drawer_unit(p, cx, bottom, s, color: str, open_amt: float = 0.0) -> None:
    w, h = 50 * s, 40 * s
    x, y = cx - w / 2, bottom - h
    _round(p, x, y, w, h, 3 * s, color)
    for i in range(3):
        yy = y + 4 * s + i * (h / 3.2)
        _round(p, x + 4 * s, yy, w - 8 * s, h / 4.2, 2 * s, _lighter(color, 115))
    if open_amt > 0.1:
        _round(
            p,
            x + 8 * s,
            y + 6 * s - 10 * s * open_amt,
            w - 16 * s,
            12 * s,
            2 * s,
            "#efe6d4",
        )


def draw_kind(p: QPainter, kind: str, cx: float, bottom: float, s: float, color: str) -> None:
    if kind == "apple":
        _apple(p, cx, bottom, s, color)
    elif kind == "tomato":
        _oval(p, cx, bottom - 14 * s, 28 * s, 26 * s, color)
        _oval(p, cx, bottom - 26 * s, 10 * s, 6 * s, "#3e7a32")
    elif kind in ("bottle", "wine"):
        _bottle(p, cx, bottom, s, color)
    elif kind == "cup":
        _cup(p, cx, bottom, s, color, False)
    elif kind == "mug":
        _cup(p, cx, bottom, s, color, True)
    elif kind in ("bowl", "ramekin"):
        _bowl(p, cx, bottom, s * (0.8 if kind == "ramekin" else 1), color)
    elif kind == "plate":
        _plate(p, cx, bottom, s, color)
    elif kind == "block":
        _cube(p, cx, bottom, s, color)
    elif kind == "prism":
        _poly(
            p,
            [
                (cx - 16 * s, bottom),
                (cx + 16 * s, bottom),
                (cx, bottom - 32 * s),
            ],
            color,
        )
    elif kind in ("can", "pudding"):
        _round(p, cx - 12 * s, bottom - 32 * s, 24 * s, 32 * s, 6 * s, color)
        _oval(p, cx, bottom - 32 * s, 24 * s, 8 * s, _lighter(color, 120))
    elif kind == "carton":
        _round(p, cx - 12 * s, bottom - 36 * s, 24 * s, 36 * s, 2 * s, color)
        _poly(
            p,
            [
                (cx - 12 * s, bottom - 36 * s),
                (cx + 12 * s, bottom - 36 * s),
                (cx + 6 * s, bottom - 46 * s),
                (cx - 6 * s, bottom - 46 * s),
            ],
            _lighter(color, 110),
        )
    elif kind == "cheese":
        _poly(
            p,
            [
                (cx - 16 * s, bottom),
                (cx + 18 * s, bottom),
                (cx + 4 * s, bottom - 28 * s),
            ],
            color,
        )
    elif kind == "butter":
        _round(p, cx - 16 * s, bottom - 14 * s, 32 * s, 14 * s, 2 * s, color)
    elif kind == "chocolate":
        _round(p, cx - 14 * s, bottom - 22 * s, 28 * s, 22 * s, 3 * s, color)
    elif kind == "cookie":
        _oval(p, cx, bottom - 12 * s, 28 * s, 28 * s, color)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6b3e26"))
        for dx, dy in ((-4, -14), (6, -10), (-2, -6)):
            p.drawEllipse(QPointF(cx + dx * s, bottom + dy * s), 2.2 * s, 2.2 * s)
    elif kind == "ball":
        _oval(p, cx, bottom - 14 * s, 28 * s, 28 * s, color)
    elif kind == "coin":
        _oval(p, cx, bottom - 12 * s, 26 * s, 26 * s, color)
        _oval(p, cx, bottom - 12 * s, 14 * s, 14 * s, _lighter(color, 120))
    elif kind == "egg":
        _oval(p, cx, bottom - 16 * s, 20 * s, 28 * s, color)
    elif kind == "toast":
        _round(p, cx - 14 * s, bottom - 26 * s, 28 * s, 26 * s, 8 * s, color)
    elif kind == "cloth":
        _round(p, cx - 20 * s, bottom - 16 * s, 34 * s, 16 * s, 4 * s, color)
        _round(p, cx - 8 * s, bottom - 30 * s, 28 * s, 16 * s, 4 * s, _lighter(color, 115))
    elif kind == "shoe":
        _poly(
            p,
            [
                (cx - 20 * s, bottom),
                (cx + 20 * s, bottom),
                (cx + 16 * s, bottom - 10 * s),
                (cx + 2 * s, bottom - 12 * s),
                (cx - 2 * s, bottom - 24 * s),
                (cx - 16 * s, bottom - 20 * s),
            ],
            color,
        )
    elif kind == "mahjong":
        _round(p, cx - 12 * s, bottom - 32 * s, 24 * s, 32 * s, 3 * s, "#f7f4ea")
        p.setPen(QPen(QColor("#d6453d")))
        p.setFont(_font(int(16 * s)))
        p.drawText(QRectF(cx - 12 * s, bottom - 30 * s, 24 * s, 28 * s), Qt.AlignCenter, "中")
    elif kind == "doll":
        _oval(p, cx, bottom - 10 * s, 30 * s, 22 * s, color)
        _oval(p, cx, bottom - 22 * s, 20 * s, 16 * s, _lighter(color, 120))
        _oval(p, cx, bottom - 32 * s, 12 * s, 10 * s, _lighter(color, 145))
    elif kind == "key":
        _oval(p, cx - 8 * s, bottom - 28 * s, 16 * s, 16 * s, color)
        _round(p, cx - 2 * s, bottom - 22 * s, 6 * s, 22 * s, 2 * s, color)
    elif kind == "pen":
        p.save()
        p.translate(cx, bottom - 16 * s)
        p.rotate(-70)
        _round(p, -4 * s, -18 * s, 8 * s, 36 * s, 3 * s, color)
        p.restore()
    elif kind == "tube":
        _round(p, cx - 6 * s, bottom - 36 * s, 12 * s, 36 * s, 4 * s, color)
        _round(p, cx - 7 * s, bottom - 40 * s, 14 * s, 6 * s, 2 * s, "#e8eef2")
    elif kind == "plug":
        _round(p, cx - 12 * s, bottom - 18 * s, 24 * s, 18 * s, 3 * s, color)
        _round(p, cx - 8 * s, bottom - 32 * s, 5 * s, 16 * s, 1 * s, "#c5ccd4")
        _round(p, cx + 3 * s, bottom - 32 * s, 5 * s, 16 * s, 1 * s, "#c5ccd4")
    elif kind == "socket":
        _round(p, cx - 16 * s, bottom - 28 * s, 32 * s, 28 * s, 3 * s, color)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2c333c"))
        p.drawEllipse(QPointF(cx - 6 * s, bottom - 14 * s), 3 * s, 3 * s)
        p.drawEllipse(QPointF(cx + 6 * s, bottom - 14 * s), 3 * s, 3 * s)
    elif kind in ("screw", "nut"):
        _oval(p, cx, bottom - 8 * s, 22 * s, 22 * s, color)
        _round(p, cx - 3 * s, bottom - 28 * s, 6 * s, 18 * s, 1 * s, _darker(color, 110))
    elif kind == "bell":
        _poly(
            p,
            [
                (cx - 16 * s, bottom - 8 * s),
                (cx + 16 * s, bottom - 8 * s),
                (cx + 8 * s, bottom - 28 * s),
                (cx - 8 * s, bottom - 28 * s),
            ],
            color,
        )
        _oval(p, cx, bottom - 6 * s, 28 * s, 8 * s, _darker(color, 120))
    elif kind == "hammer":
        _round(p, cx - 3 * s, bottom - 34 * s, 6 * s, 34 * s, 2 * s, "#c49a6c")
        _round(p, cx - 16 * s, bottom - 40 * s, 32 * s, 12 * s, 3 * s, "#8aa0b0")
    elif kind == "tool":
        _round(p, cx - 4 * s, bottom - 30 * s, 8 * s, 30 * s, 2 * s, color)
        _oval(p, cx - 8 * s, bottom - 32 * s, 16 * s, 12 * s, color)
    elif kind == "broom":
        _round(p, cx - 3 * s, bottom - 40 * s, 6 * s, 34 * s, 2 * s, "#c49a6c")
        _poly(
            p,
            [
                (cx - 16 * s, bottom),
                (cx + 16 * s, bottom),
                (cx + 6 * s, bottom - 16 * s),
                (cx - 6 * s, bottom - 16 * s),
            ],
            "#c4a06a",
        )
    elif kind == "laptop":
        _round(p, cx - 22 * s, bottom - 8 * s, 44 * s, 8 * s, 2 * s, _darker(color, 120))
        _round(p, cx - 18 * s, bottom - 32 * s, 32 * s, 24 * s, 2 * s, color)
        _round(p, cx - 14 * s, bottom - 28 * s, 24 * s, 16 * s, 1 * s, "#9fd0ea")
    elif kind == "headphone":
        p.setBrush(Qt.NoBrush)
        p.setPen(_pen(color, 4))
        p.drawArc(QRectF(cx - 16 * s, bottom - 36 * s, 32 * s, 28 * s), 20 * 16, 140 * 16)
        _round(p, cx - 20 * s, bottom - 18 * s, 10 * s, 16 * s, 3 * s, color)
        _round(p, cx + 10 * s, bottom - 18 * s, 10 * s, 16 * s, 3 * s, color)
    elif kind == "book":
        _round(p, cx - 16 * s, bottom - 28 * s, 32 * s, 28 * s, 2 * s, color)
        p.setPen(_pen("#f7f4ea", 1.4))
        p.drawLine(QPointF(cx - 12 * s, bottom - 8 * s), QPointF(cx + 10 * s, bottom - 8 * s))
    elif kind == "bin":
        _poly(
            p,
            [
                (cx - 16 * s, bottom - 30 * s),
                (cx + 16 * s, bottom - 30 * s),
                (cx + 12 * s, bottom),
                (cx - 12 * s, bottom),
            ],
            color,
        )
        _round(p, cx - 18 * s, bottom - 36 * s, 36 * s, 8 * s, 2 * s, _darker(color, 120))
    elif kind in ("box", "basket", "toolbox"):
        _round(p, cx - 20 * s, bottom - 24 * s, 40 * s, 24 * s, 3 * s, color)
        if kind == "basket":
            p.setPen(_pen(_darker(color, 130), 1.2))
            p.drawLine(QPointF(cx - 12 * s, bottom - 24 * s), QPointF(cx - 12 * s, bottom))
            p.drawLine(QPointF(cx, bottom - 24 * s), QPointF(cx, bottom))
            p.drawLine(QPointF(cx + 12 * s, bottom - 24 * s), QPointF(cx + 12 * s, bottom))
        if kind == "toolbox":
            p.setBrush(Qt.NoBrush)
            p.setPen(_pen("#2c333c", 2))
            p.drawArc(QRectF(cx - 8 * s, bottom - 34 * s, 16 * s, 14 * s), 0, 180 * 16)
    elif kind == "toaster":
        _round(p, cx - 22 * s, bottom - 26 * s, 44 * s, 26 * s, 4 * s, color)
        _round(p, cx - 14 * s, bottom - 22 * s, 10 * s, 16 * s, 2 * s, "#2c333c")
        _round(p, cx + 4 * s, bottom - 22 * s, 10 * s, 16 * s, 2 * s, "#2c333c")
    elif kind == "piggy":
        _oval(p, cx, bottom - 14 * s, 40 * s, 26 * s, color)
        _oval(p, cx + 16 * s, bottom - 20 * s, 16 * s, 14 * s, color)
        _oval(p, cx + 8 * s, bottom - 28 * s, 8 * s, 8 * s, color)
        _round(p, cx - 8 * s, bottom - 20 * s, 14 * s, 3 * s, 1 * s, "#2c333c")
    elif kind == "vase":
        _poly(
            p,
            [
                (cx - 5 * s, bottom - 40 * s),
                (cx + 5 * s, bottom - 40 * s),
                (cx + 9 * s, bottom - 30 * s),
                (cx + 16 * s, bottom - 16 * s),
                (cx + 8 * s, bottom),
                (cx - 8 * s, bottom),
                (cx - 16 * s, bottom - 16 * s),
                (cx - 9 * s, bottom - 30 * s),
            ],
            color,
        )
        _oval(p, cx, bottom - 40 * s, 14 * s, 6 * s, _darker(color, 115))
    elif kind in ("pot", "moka"):
        _round(p, cx - 16 * s, bottom - 22 * s, 32 * s, 22 * s, 4 * s, color)
        p.setBrush(Qt.NoBrush)
        p.setPen(_pen(_darker(color, 130), 2))
        p.drawArc(QRectF(cx - 28 * s, bottom - 20 * s, 14 * s, 14 * s), -40 * 16, 200 * 16)
        p.drawArc(QRectF(cx + 14 * s, bottom - 20 * s, 14 * s, 14 * s), 20 * 16, 200 * 16)
        if kind == "moka":
            _poly(
                p,
                [
                    (cx - 8 * s, bottom - 22 * s),
                    (cx + 8 * s, bottom - 22 * s),
                    (cx + 4 * s, bottom - 36 * s),
                    (cx - 4 * s, bottom - 36 * s),
                ],
                _lighter(color, 120),
            )
    elif kind in ("cabinet", "microwave"):
        _cabinet(p, cx, bottom, s, color, 0)
    elif kind == "drawer":
        _drawer_unit(p, cx, bottom, s, color, 0)
    elif kind == "stove":
        _round(p, cx - 24 * s, bottom - 28 * s, 48 * s, 28 * s, 3 * s, color)
        _oval(p, cx - 10 * s, bottom - 16 * s, 14 * s, 10 * s, "#2c333c")
        _oval(p, cx + 10 * s, bottom - 16 * s, 14 * s, 10 * s, "#2c333c")
    elif kind == "car":
        _round(p, cx - 20 * s, bottom - 14 * s, 40 * s, 14 * s, 4 * s, color)
        _round(p, cx - 8 * s, bottom - 26 * s, 22 * s, 14 * s, 4 * s, _lighter(color, 130))
        _oval(p, cx - 12 * s, bottom - 4 * s, 10 * s, 10 * s, "#2c333c")
        _oval(p, cx + 12 * s, bottom - 4 * s, 10 * s, 10 * s, "#2c333c")
    elif kind == "watch":
        _round(p, cx - 12 * s, bottom - 28 * s, 24 * s, 28 * s, 6 * s, color)
        _round(p, cx - 8 * s, bottom - 22 * s, 16 * s, 14 * s, 2 * s, "#d8f2f6")
    elif kind == "phone":
        _round(p, cx - 10 * s, bottom - 32 * s, 20 * s, 32 * s, 3 * s, color)
        _round(p, cx - 7 * s, bottom - 28 * s, 14 * s, 22 * s, 1 * s, "#9fd0ea")
    elif kind == "camera":
        _round(p, cx - 18 * s, bottom - 20 * s, 36 * s, 20 * s, 3 * s, color)
        _oval(p, cx + 4 * s, bottom - 12 * s, 14 * s, 14 * s, "#9fd0ea")
    elif kind == "donut":
        _oval(p, cx, bottom - 12 * s, 30 * s, 30 * s, color)
        _oval(p, cx, bottom - 12 * s, 12 * s, 12 * s, "#f4efe6")
    elif kind == "icecream":
        _poly(
            p,
            [(cx - 10 * s, bottom), (cx + 10 * s, bottom), (cx, bottom - 22 * s)],
            "#e7c89a",
        )
        _oval(p, cx, bottom - 26 * s, 22 * s, 18 * s, color)
    elif kind == "pepper":
        _oval(p, cx, bottom - 16 * s, 16 * s, 28 * s, color)
    elif kind == "brush":
        _round(p, cx - 4 * s, bottom - 34 * s, 8 * s, 28 * s, 2 * s, color)
        _round(p, cx - 7 * s, bottom - 14 * s, 14 * s, 14 * s, 3 * s, "#f7f7f7")
    elif kind == "button":
        _round(p, cx - 16 * s, bottom - 16 * s, 32 * s, 16 * s, 8 * s, color)
    elif kind == "digits":
        _round(p, cx - 12 * s, bottom - 32 * s, 24 * s, 32 * s, 3 * s, color)
        p.setPen(QPen(QColor("#243040")))
        p.setFont(_font(int(16 * s)))
        p.drawText(QRectF(cx - 12 * s, bottom - 30 * s, 24 * s, 28 * s), Qt.AlignCenter, "8")
    elif kind == "tee":
        _round(p, cx - 16 * s, bottom - 14 * s, 32 * s, 14 * s, 2 * s, color)
        _round(p, cx - 7 * s, bottom - 36 * s, 14 * s, 24 * s, 2 * s, color)
    elif kind == "xylophone":
        for i, col in enumerate(("#e24b3b", "#e07a32", "#f0c419", "#3e9a4c", "#3c7ed6", "#7a5cb8")):
            _round(p, cx - 28 * s + i * 10 * s, bottom - (16 + i * 3) * s, 8 * s, (16 + i * 3) * s, 2 * s, col)
    elif kind == "tictactoe":
        _round(p, cx - 22 * s, bottom - 40 * s, 44 * s, 40 * s, 3 * s, "#f7f4ea")
        p.setPen(_pen("#243040", 1.6))
        p.drawLine(QPointF(cx - 7 * s, bottom - 36 * s), QPointF(cx - 7 * s, bottom - 4 * s))
        p.drawLine(QPointF(cx + 7 * s, bottom - 36 * s), QPointF(cx + 7 * s, bottom - 4 * s))
        p.drawLine(QPointF(cx - 18 * s, bottom - 26 * s), QPointF(cx + 18 * s, bottom - 26 * s))
        p.drawLine(QPointF(cx - 18 * s, bottom - 14 * s), QPointF(cx + 18 * s, bottom - 14 * s))
        p.setPen(_pen("#d6453d", 2))
        p.drawLine(QPointF(cx - 16 * s, bottom - 34 * s), QPointF(cx - 10 * s, bottom - 28 * s))
        p.drawLine(QPointF(cx - 16 * s, bottom - 28 * s), QPointF(cx - 10 * s, bottom - 34 * s))
    elif kind == "pawn":
        _oval(p, cx, bottom - 8 * s, 18 * s, 10 * s, color)
        _oval(p, cx, bottom - 20 * s, 12 * s, 16 * s, color)
    elif kind == "ringtoy":
        _round(p, cx - 4 * s, bottom - 36 * s, 8 * s, 36 * s, 2 * s, "#c49a6c")
        for i, col in enumerate(("#e24b3b", "#f0c419", "#3c7ed6")):
            _oval(p, cx, bottom - (8 + i * 8) * s, (28 - i * 6) * s, 8 * s, col)
    elif kind == "conveyor":
        _round(p, cx - 36 * s, bottom - 16 * s, 72 * s, 16 * s, 6 * s, color)
        p.setPen(_pen("#d0d6de", 1.4))
        for i in range(4):
            xx = cx - 24 * s + i * 16 * s
            p.drawLine(QPointF(xx, bottom - 14 * s), QPointF(xx, bottom - 2 * s))
    else:
        _round(p, cx - 14 * s, bottom - 28 * s, 28 * s, 28 * s, 6 * s, color)
        p.setPen(QPen(QColor("#243040")))
        p.setFont(_font(int(12 * s)))
        p.drawText(QRectF(cx - 14 * s, bottom - 26 * s, 28 * s, 24 * s), Qt.AlignCenter, kind[:2])


def _gripper(p: QPainter, x: float, y: float, s: float = 1.0, open_amt: float = 1.0) -> None:
    _fill(p, "#5c6b7a")
    p.drawRoundedRect(QRectF(x - 8 * s, y - 6 * s, 16 * s, 10 * s), 2 * s, 2 * s)
    spread = 6 * s * open_amt
    _round(p, x - 10 * s - spread, y + 2 * s, 6 * s, 16 * s, 2 * s, "#5c6b7a")
    _round(p, x + 4 * s + spread, y + 2 * s, 6 * s, 16 * s, 2 * s, "#5c6b7a")


def _arm_to(p: QPainter, x0: float, y0: float, x1: float, y1: float) -> None:
    mid_x = (x0 + x1) / 2
    mid_y = min(y0, y1) - 18
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen("#5c6b7a", 5))
    p.drawLine(QPointF(x0, y0), QPointF(mid_x, mid_y))
    p.drawLine(QPointF(mid_x, mid_y), QPointF(x1, y1))
    _gripper(p, x1, y1)


def _shadow(p: QPainter, cx: float, bottom: float, w: float) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 28))
    p.drawEllipse(QRectF(cx - w / 2, bottom - 4, w, 8))


def _draw_objects_row(
    p: QPainter,
    scene: TaskScene,
    xs: Sequence[float],
    bottom: float,
    scale: float = 1.0,
) -> None:
    for i, (kind, x) in enumerate(zip(scene.objects, xs)):
        _shadow(p, x, bottom, 36 * scale)
        draw_kind(p, kind, x, bottom, scale, _color_of(scene, i, kind))


def _paint_scene(p: QPainter, scene: TaskScene) -> None:
    w, h = THUMB_W, THUMB_H
    table_y = 112
    p.setRenderHint(QPainter.Antialiasing, True)
    grad_top = QColor("#d7e4f2")
    grad_bot = QColor("#f6f1e8")
    p.fillRect(0, 0, w, h, grad_bot)
    p.fillRect(0, 0, w, 78, grad_top)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#c4a574"))
    p.drawRect(0, table_y - 8, w, h - table_y + 8)
    p.setBrush(QColor("#b08960"))
    p.drawRect(0, table_y - 8, w, 6)
    stripe = _STRIPE.get(scene.action, "#6a7380")
    p.fillRect(0, 0, 7, h, QColor(stripe))

    action = scene.action
    kinds = scene.objects
    mains, backs = _roles(action, kinds)
    if action == "stack":
        base = kinds[0] if kinds else "block"
        cols = [_color_of(scene, 0, base), _lighter(_color_of(scene, 0, base), 120), _darker(_color_of(scene, 0, base), 115)]
        step = 18 if base == "bowl" else 22
        for i, col in enumerate(cols):
            draw_kind(p, base if base in ("block", "bowl", "plate", "ringtoy") else "block", 128, table_y - i * step, 0.9, col)
    elif action == "pour":
        src = mains[0] if mains else "bottle"
        dst = backs[0] if backs else "cup"
        draw_kind(p, dst, 156, table_y, 1.0, _color_of(scene, kinds.index(dst) if dst in kinds else 0, dst))
        if src in ("bottle", "wine"):
            _bottle(p, 92, table_y - 6, 1.0, _color_of(scene, 0, src), tilt=55)
        else:
            draw_kind(p, src, 90, table_y - 28, 0.85, _color_of(scene, 0, src))
    elif action in ("open", "close"):
        opened = 0.85 if action == "open" else 0.0
        target = "drawer" if "drawer" in kinds else "cabinet"
        if target == "drawer":
            _drawer_unit(p, 130, table_y, 1.35, _color_of(scene, 0, target), 0.85 if action == "open" else 0.05)
        else:
            _cabinet(p, 130, table_y, 1.45, _color_of(scene, 0, target), opened)
    elif action in ("pick_place", "open_place", "place_close", "place"):
        src = mains[0] if mains else (kinds[0] if kinds else "block")
        dst = backs[0] if backs else (mains[1] if len(mains) > 1 else "plate")
        if dst in ("drawer", "cabinet", "microwave"):
            if dst == "drawer":
                _drawer_unit(p, 168, table_y, 1.15, _color_of(scene, 1, dst), 0.7 if action != "place_close" else 0.25)
            else:
                _cabinet(p, 168, table_y, 1.2, _color_of(scene, 1, dst), 0.55 if action != "place_close" else 0.1)
        else:
            draw_kind(p, dst, 170, table_y, 1.0, _color_of(scene, 1, dst))
        draw_kind(p, src, 78, table_y - 8, 0.95, _color_of(scene, 0, src))
        _arm_to(p, 18, 36, 78, table_y - 46)
        p.setPen(_pen("#1f9d8a", 2.2))
        p.drawLine(QPointF(104, table_y - 30), QPointF(140, table_y - 30))
        p.drawLine(QPointF(132, table_y - 38), QPointF(146, table_y - 30))
        p.drawLine(QPointF(132, table_y - 22), QPointF(146, table_y - 30))
    elif action == "pick":
        kind = kinds[0] if kinds else "block"
        if "conveyor" in kinds:
            draw_kind(p, "conveyor", 130, table_y, 1.0, _DEFAULT_FILL["conveyor"])
        draw_kind(p, kind, 124, table_y - (6 if "conveyor" in kinds else 0), 1.15, _color_of(scene, 0, kind))
        _arm_to(p, 24, 30, 124, table_y - 52)
    elif action == "push":
        kind = "tee" if "tee" in kinds else (kinds[0] if kinds else "block")
        draw_kind(p, kind, 132, table_y, 1.05, _color_of(scene, 0, kind))
        p.setPen(_pen("#c45b7a", 2.4))
        p.drawLine(QPointF(48, table_y - 20), QPointF(96, table_y - 20))
        p.drawLine(QPointF(86, table_y - 28), QPointF(100, table_y - 20))
        p.drawLine(QPointF(86, table_y - 12), QPointF(100, table_y - 20))
    elif action == "swap":
        a = kinds[0] if kinds else "block"
        b = kinds[1] if len(kinds) > 1 else a
        draw_kind(p, a, 78, table_y, 0.9, _color_of(scene, 0, a))
        draw_kind(p, b, 168, table_y, 0.9, _color_of(scene, min(1, len(kinds) - 1), b))
        p.setPen(_pen("#5b8fc4", 2.2))
        p.drawArc(QRectF(86, table_y - 58, 70, 28), 20 * 16, 140 * 16)
        p.drawArc(QRectF(86, table_y - 28, 70, 28), 200 * 16, 140 * 16)
    elif action == "hang":
        p.setPen(_pen("#8a6a3b", 3))
        p.drawLine(QPointF(70, 28), QPointF(190, 28))
        p.drawLine(QPointF(120, 28), QPointF(120, 42))
        kind = "mug" if "mug" in kinds or "cup" in kinds else (kinds[0] if kinds else "mug")
        draw_kind(p, kind, 120, 78, 1.1, _color_of(scene, 0, kind))
    elif action == "insert":
        dst = "socket" if "plug" in kinds else ("nut" if "screw" in kinds else "box")
        src = kinds[0] if kinds else "key"
        _round(p, 150, table_y - 28, 40, 28, 3, "#d9d3c8")
        draw_kind(p, src, 96, table_y, 1.0, _color_of(scene, 0, src))
        p.setPen(_pen("#3f8f62", 2.2))
        p.drawLine(QPointF(120, table_y - 18), QPointF(154, table_y - 18))
    elif action == "fold":
        draw_kind(p, "cloth", 120, table_y, 1.5, _color_of(scene, 0, "cloth"))
    elif action == "sort":
        dest = backs[0] if backs else "basket"
        draw_kind(p, dest, 168, table_y, 1.1, _DEFAULT_FILL.get(dest, "#c49a6c"))
        held = mains or ["cup", "book"]
        draw_kind(p, held[0], 72, table_y, 0.9, _color_of(scene, 0, held[0]))
        if len(held) > 1:
            draw_kind(p, held[1], 114, table_y - 2, 0.75, _color_of(scene, 1, held[1]))
    elif action == "press":
        if "hammer" in kinds:
            _cube(p, 150, table_y, 0.9, _color_of(scene, 0, "block"))
            draw_kind(p, "hammer", 96, table_y - 4, 1.05, _DEFAULT_FILL["hammer"])
        elif "bell" in kinds:
            draw_kind(p, "bell", 130, table_y, 1.35, _color_of(scene, 0, "bell"))
            _gripper(p, 130, table_y - 52, 1.0, 0.3)
        else:
            kind = "button" if "button" in kinds else (kinds[0] if kinds else "button")
            draw_kind(p, kind, 130, table_y, 1.3, _color_of(scene, 0, kind))
            _gripper(p, 130, table_y - 48, 1.1, 0.4)
    elif action == "handover":
        kind = kinds[0] if kinds else "block"
        draw_kind(p, kind, 120, table_y - 6, 0.9, _color_of(scene, 0, kind))
        _gripper(p, 78, table_y - 30, 0.9, 0.8)
        _gripper(p, 162, table_y - 30, 0.9, 0.8)
    elif action == "cover":
        _cube(p, 120, table_y, 0.8, _color_of(scene, 1 if len(kinds) > 1 else 0, "block"))
        _cup(p, 120, table_y - 18, 1.15, _color_of(scene, 0, "cup"), False)
    elif action == "fill":
        container = "basket" if "basket" in kinds else "box"
        draw_kind(p, container, 140, table_y, 1.25, _DEFAULT_FILL.get(container, "#c49a6c"))
        item = next((k for k in kinds if k not in ("basket", "box")), "egg")
        draw_kind(p, item, 128, table_y - 16, 0.55, _color_of(scene, 0, item))
        draw_kind(p, item, 152, table_y - 14, 0.55, _color_of(scene, 0, item))
    elif action == "adjust":
        kind = kinds[0] if kinds else "bottle"
        draw_kind(p, kind, 120, table_y, 1.15, _color_of(scene, 0, kind))
        p.setPen(_pen("#c48a3a", 2.2))
        p.drawArc(QRectF(148, table_y - 48, 36, 36), 40 * 16, 200 * 16)
    elif action == "align":
        _draw_objects_row(p, scene, (70, 120, 170), table_y, 0.85)
    elif action == "lineup":
        for i, x in enumerate((70, 108, 146, 184)):
            draw_kind(p, "mahjong" if "mahjong" in kinds else kinds[0], x, table_y, 0.75, _DEFAULT_FILL.get(kinds[0], "#f7f4ea"))
    elif action == "equation":
        for i, (txt, x) in enumerate((("1", 62), ("+", 102), ("2", 142), ("=", 182))):
            _round(p, x, table_y - 36, 28, 36, 3, "#f7f4ea")
            p.setPen(QPen(QColor("#243040")))
            p.setFont(_font(16))
            p.drawText(QRectF(x, table_y - 34, 28, 32), Qt.AlignCenter, txt)
    elif action == "play":
        kind = kinds[0] if kinds else "block"
        draw_kind(p, kind, 128, table_y, 1.25, _color_of(scene, 0, kind))
        if "xylophone" in kinds:
            draw_kind(p, "hammer", 70, table_y - 6, 0.8, _DEFAULT_FILL["hammer"])
    elif action == "toast":
        draw_kind(p, "toaster", 130, table_y, 1.2, _DEFAULT_FILL["toaster"])
        draw_kind(p, "toast", 130, table_y - 18, 0.7, _DEFAULT_FILL["toast"])
    elif action == "sweep":
        draw_kind(p, "broom", 78, table_y, 1.1, _DEFAULT_FILL["broom"])
        _cube(p, 150, table_y, 0.55, "#e07a32")
        _cube(p, 176, table_y, 0.45, "#3c7ed6")
    else:
        _draw_objects_row(p, scene, (78, 128, 176)[: max(1, len(kinds))], table_y, 0.95)
        _arm_to(p, 20, 34, 78, table_y - 40)

    # Tag pill and optional badge. Drawn last so they stay readable.
    p.setFont(_font(15))
    tag = scene.tag
    tag_w = max(52, p.fontMetrics().boundingRect(tag).width() + 18)
    _fill(p, stripe, stripe, 1)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(stripe))
    p.drawRoundedRect(QRectF(14, 8, tag_w, 24), 8, 8)
    p.setPen(QPen(QColor("#ffffff")))
    p.drawText(QRectF(14, 8, tag_w, 24), Qt.AlignCenter, tag)
    if scene.badge:
        p.setFont(_font(12))
        bw = p.fontMetrics().boundingRect(scene.badge).width() + 14
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 220))
        p.drawRoundedRect(QRectF(w - bw - 8, 8, bw, 22), 7, 7)
        p.setPen(QPen(QColor("#243040")))
        p.drawText(QRectF(w - bw - 8, 8, bw, 22), Qt.AlignCenter, scene.badge)


def eval_task_thumbnail(text: str, width: int = THUMB_W, height: int = THUMB_H) -> QPixmap:
    key = (text or "").strip()
    cached = _PIX_CACHE.get(key)
    if cached is not None and cached.width() == THUMB_W:
        if width == THUMB_W and height == THUMB_H:
            return cached
        return cached.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pix = QPixmap(THUMB_W, THUMB_H)
    pix.fill(QColor("#f6f1e8"))
    painter = QPainter(pix)
    try:
        _paint_scene(painter, parse_task_scene(key))
    finally:
        painter.end()
    if len(_PIX_CACHE) > 500:
        _PIX_CACHE.clear()
    _PIX_CACHE[key] = pix
    if width == THUMB_W and height == THUMB_H:
        return pix
    return pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def thumb_icon(text: str) -> QIcon:
    return QIcon(eval_task_thumbnail(text))
