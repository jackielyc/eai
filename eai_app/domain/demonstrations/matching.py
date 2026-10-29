"""人类演示视频文件名 → LIBERO suite / task_id / language。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# 嵌入一份常用 suite 任务名（与 libero_suite_task_map 对齐），避免强依赖 libero 包。
_LIBERO_TASK_MAP: Dict[str, List[str]] = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}


@dataclass(frozen=True)
class LiberoTaskMatch:
    suite: str
    task_id: int
    language: str
    task_name: str
    score: float
    reason: str

    def summary(self) -> str:
        return f"{self.suite}[{self.task_id}] {self.language}"


def _slug_to_language(name: str) -> str:
    s = name
    # 去掉场景前缀 KITCHEN_SCENE4_
    s = re.sub(r"^[A-Z]+_SCENE\d+_", "", s)
    return s.replace("_", " ").strip()


def list_suite_tasks(suite: str) -> List[Tuple[int, str, str]]:
    """Return [(task_id, task_name, language), ...]."""
    names = _LIBERO_TASK_MAP.get(suite) or []
    out: List[Tuple[int, str, str]] = []
    for i, n in enumerate(names):
        out.append((i, n, _slug_to_language(n)))
    return out


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _score_name(tokens: Sequence[str], task_name: str, language: str) -> float:
    blob = f"{task_name} {language}".lower()
    if not tokens:
        return 0.0
    hit = sum(1 for t in tokens if t in blob)
    bonus = 0.0
    # 强信号加权
    for key, w in (
        ("drawer", 3.0),
        ("bowl", 2.0),
        ("dish", 1.5),
        ("plate", 1.0),
        ("cup", 1.0),
        ("mug", 1.0),
        ("cookie", 1.0),
        ("basket", 0.5),
    ):
        if key in tokens and key in blob:
            bonus += w
    return float(hit) + bonus


def match_video_to_libero_task(video_path: str) -> LiberoTaskMatch:
    """根据文件名关键词匹配最接近的 LIBERO 任务。"""
    base = os.path.splitext(os.path.basename(video_path or ""))[0]
    tokens = _tokenize(base)

    # 规则优先：抽屉 + 碗/盘 → libero_goal put bowl in top drawer
    def _has(*keys: str) -> bool:
        for t in tokens:
            for k in keys:
                if t == k or t.startswith(k):
                    return True
        return False

    if _has("drawer") and _has("bowl", "dish", "plate"):
        suite = "libero_goal"
        task_id = 3
        name = _LIBERO_TASK_MAP[suite][task_id]
        return LiberoTaskMatch(
            suite=suite,
            task_id=task_id,
            language=_slug_to_language(name),
            task_name=name,
            score=100.0,
            reason="keyword:drawer+bowl/dish",
        )
    if _has("drawer"):
        suite = "libero_goal"
        task_id = 3
        name = _LIBERO_TASK_MAP[suite][task_id]
        return LiberoTaskMatch(
            suite=suite,
            task_id=task_id,
            language=_slug_to_language(name),
            task_name=name,
            score=80.0,
            reason="keyword:drawer",
        )
    if _has("cup", "mug", "coffee"):
        # 放杯子：goal 里 put bowl on plate 较接近「放到指定处」
        suite = "libero_goal"
        task_id = 8
        name = _LIBERO_TASK_MAP[suite][task_id]
        return LiberoTaskMatch(
            suite=suite,
            task_id=task_id,
            language=_slug_to_language(name),
            task_name=name,
            score=60.0,
            reason="keyword:cup/mug",
        )
    if _has("cookie"):
        suite = "libero_spatial"
        task_id = 3  # bowl on cookie box → plate（近似）
        name = _LIBERO_TASK_MAP[suite][task_id]
        return LiberoTaskMatch(
            suite=suite,
            task_id=task_id,
            language=_slug_to_language(name),
            task_name=name,
            score=55.0,
            reason="keyword:cookie",
        )

    best: Optional[LiberoTaskMatch] = None
    for suite, names in _LIBERO_TASK_MAP.items():
        if suite == "libero_90":
            continue
        for i, name in enumerate(names):
            lang = _slug_to_language(name)
            sc = _score_name(tokens, name, lang)
            if best is None or sc > best.score:
                best = LiberoTaskMatch(
                    suite=suite,
                    task_id=i,
                    language=lang,
                    task_name=name,
                    score=sc,
                    reason="fuzzy",
                )
    if best is None or best.score <= 0:
        suite = "libero_goal"
        task_id = 3
        name = _LIBERO_TASK_MAP[suite][task_id]
        return LiberoTaskMatch(
            suite=suite,
            task_id=task_id,
            language=_slug_to_language(name),
            task_name=name,
            score=0.0,
            reason="default:drawer_bowl",
        )
    return best


__all__ = [
    "LiberoTaskMatch",
    "list_suite_tasks",
    "match_video_to_libero_task",
]
