"""任务与 harness 阶段。每一阶段都比上一阶段在评测上更优，末段完成任务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _src(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(line[4:] if line.startswith("    ") else line for line in lines).strip() + "\n"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    summary: str
    plan: str
    stages: Tuple[str, ...]
    decoy: str

    @property
    def initial(self) -> str:
        return self.stages[0]


def _norm(src: str) -> str:
    return (src or "").strip()


# 做麻将杠：π0.5 式初稿只抓第一张牌；改写后先配四张再逐张放置。
_KONG_H0 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        if tiles:
            skills.pick_and_place(tiles[0], "row")
        return m
    """
)
_KONG_H1 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        quad = skills.match_four(tiles)
        if quad:
            skills.pick_and_place(quad[0], "row")
        return m
    """
)
_KONG_H2 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        quad = skills.match_four(tiles)
        for tile in quad:
            skills.pick_and_place(tile, "row")
        return m
    """
)
_KONG_DECOY = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        for tile in tiles:
            if tile["value"] == "east":
                skills.pick_and_place(tile, "row")
                break
        return m
    """
)

# 盖住方块：没有记忆时揭错杯子；改写后先记下物体-杯子对应，再按顺序揭开。
_COVER_H0 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.reveal_guess()
        return m
    """
)
_COVER_H1 = _src(
    """
    def act(o, g, m, skills, tools):
        scene = skills.observe()
        skills.memorize(scene, m)
        skills.cover()
        order = g["order"]
        if order:
            skills.reveal_named(m, order[0])
        return m
    """
)
_COVER_H2 = _src(
    """
    def act(o, g, m, skills, tools):
        scene = skills.observe()
        skills.memorize(scene, m)
        skills.cover()
        for name in g["order"]:
            skills.reveal_named(m, name)
        return m
    """
)
_COVER_DECOY = _src(
    """
    def act(o, g, m, skills, tools):
        skills.cover()
        skills.reveal_guess()
        return m
    """
)

# 交换 T：直接对调会丢掉朝向；改写后经空缓冲做三次搬运并保持朝向。
_SWAP_H0 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.nudge_together()
        return m
    """
)
_SWAP_H1 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.carry("A", "buffer")
        skills.carry("B", "home_a")
        return m
    """
)
_SWAP_H2 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.carry("A", "buffer")
        skills.carry("B", "home_a")
        skills.carry("A", "home_b")
        return m
    """
)
_SWAP_DECOY = _src(
    """
    def act(o, g, m, skills, tools):
        skills.swap_direct()
        return m
    """
)

# 排最大数：不排序只碰第一张；改写后按数值降序放进槽位。
_NUM_H0 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        if tiles:
            skills.touch(tiles[0])
        return m
    """
)
_NUM_H1 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        if not tiles:
            return m
        best = max(tiles, key=lambda t: int(t["value"]))
        skills.place_at(best, 0)
        return m
    """
)
_NUM_H2 = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        ordered = sorted(tiles, key=lambda t: int(t["value"]), reverse=True)
        for i, tile in enumerate(ordered):
            skills.place_at(tile, i)
        return m
    """
)
_NUM_DECOY = _src(
    """
    def act(o, g, m, skills, tools):
        tiles = skills.perceive(o)
        ordered = sorted(tiles, key=lambda t: int(t["value"]))
        for i, tile in enumerate(ordered):
            skills.place_at(tile, i)
        return m
    """
)

# 插钥匙：一次挪近对不齐；改写后分成接近、对齐、插入。
_KEY_H0 = _src(
    """
    def act(o, g, m, skills, tools):
        return m
    """
)
_KEY_H1 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.approach("lock")
        return m
    """
)
_KEY_H2 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.approach("lock")
        skills.align("lock")
        return m
    """
)
_KEY_H3 = _src(
    """
    def act(o, g, m, skills, tools):
        skills.approach("lock")
        skills.align("lock")
        skills.insert("lock")
        return m
    """
)
_KEY_DECOY = _src(
    """
    def act(o, g, m, skills, tools):
        skills.insert("lock")
        return m
    """
)


TASKS: Dict[str, TaskSpec] = {
    "make_kong": TaskSpec(
        key="make_kong",
        title="做麻将杠",
        summary="在干扰牌中找出四张相同的牌，并排到目标行。",
        plan="识别牌面，配出四张相同的牌，再逐张抓取放到杠的槽位。",
        stages=(_KONG_H0, _KONG_H1, _KONG_H2),
        decoy=_KONG_DECOY,
    ),
    "cover_blocks": TaskSpec(
        key="cover_blocks",
        title="盖住方块",
        summary="杯子盖住方块后，按指定顺序揭开。对应关系会被遮挡。",
        plan="在遮挡前记下物体和杯子的对应，盖上后再按目标顺序揭开。",
        stages=(_COVER_H0, _COVER_H1, _COVER_H2),
        decoy=_COVER_DECOY,
    ),
    "swap_t": TaskSpec(
        key="swap_t",
        title="交换 T",
        summary="交换两块的位置，并保持各自原来的朝向。中间有一个空缓冲位。",
        plan="经空缓冲做三次搬运：A 到缓冲，B 到 A 的原位，再把 A 放到 B 的原位。",
        stages=(_SWAP_H0, _SWAP_H1, _SWAP_H2),
        decoy=_SWAP_DECOY,
    ),
    "arrange_number": TaskSpec(
        key="arrange_number",
        title="排最大数",
        summary="读出数字，按从大到小放进一排槽位。",
        plan="识别数字，按数值降序排序，再映射到从左到右的槽位。",
        stages=(_NUM_H0, _NUM_H1, _NUM_H2),
        decoy=_NUM_DECOY,
    ),
    "insert_key": TaskSpec(
        key="insert_key",
        title="插钥匙",
        summary="钥匙相对锁有位置误差和朝向误差，需要分阶段插入。",
        plan="先接近，再对齐朝向，最后在容差内插入。",
        stages=(_KEY_H0, _KEY_H1, _KEY_H2, _KEY_H3),
        decoy=_KEY_DECOY,
    ),
}


def stage_index(spec: TaskSpec, src: str) -> Optional[int]:
    cur = _norm(src)
    for i, stage in enumerate(spec.stages):
        if _norm(stage) == cur:
            return i
    return None


def stage_label(spec: TaskSpec, src: str) -> str:
    idx = stage_index(spec, src)
    if idx is not None:
        return f"H{idx}"
    if _norm(src) == _norm(spec.decoy):
        return "decoy"
    return "H?"


def builtin_variants(spec: TaskSpec, src: str) -> List[str]:
    """只提出下一阶段和一条更差的对照，避免一步跳到已知最优。"""
    out: List[str] = []
    idx = stage_index(spec, src)
    if idx is None:
        final = spec.stages[-1]
        if _norm(final) != _norm(src):
            out.append(final)
    elif idx + 1 < len(spec.stages):
        out.append(spec.stages[idx + 1])
    decoy = spec.decoy
    if _norm(decoy) != _norm(src) and all(_norm(decoy) != _norm(v) for v in out):
        out.append(decoy)
    return out
