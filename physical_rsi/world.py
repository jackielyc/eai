"""轻量桌面环境：技能调用改变物体状态，rollout 给出分数和失败标签。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


_SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "enumerate": enumerate,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "sum": sum,
    "any": any,
    "all": all,
    "zip": zip,
    "reversed": reversed,
    "bool": bool,
    "isinstance": isinstance,
    "print": lambda *args, **kwargs: None,
}

_FORBIDDEN = ("import ", "import\t", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess")


@dataclass
class Rollout:
    task: str
    seed: int
    success: bool
    score: float
    failure: str
    trace: List[str] = field(default_factory=list)
    note: str = ""


class Skills:
    def __init__(self, ep: "Episode") -> None:
        self.ep = ep

    def observe(self) -> Dict[str, Any]:
        self.ep.tick("observe")
        return self.ep.observe()

    def perceive(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.ep.tick("perceive")
        return list(obs.get("objects") or [])

    def match_four(self, tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.ep.tick("match_four")
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for tile in tiles:
            groups.setdefault(str(tile.get("value")), []).append(tile)
        for items in groups.values():
            if len(items) >= 4:
                return items[:4]
        return []

    def pick_and_place(self, tile: Any, dest: str = "row") -> bool:
        self.ep.tick("pick_and_place")
        return self.ep.place_tile(tile, dest)

    def memorize(self, obs: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        self.ep.tick("memorize")
        mapping = {}
        for obj in obs.get("objects") or []:
            if obj.get("kind") == "block" and obj.get("cup"):
                mapping[str(obj["id"])] = str(obj["cup"])
        memory["map"] = mapping
        self.ep.remembered = dict(mapping)
        return memory

    def cover(self) -> None:
        self.ep.tick("cover")
        self.ep.phase = "covered"
        for body in self.ep.bodies.values():
            if body["kind"] == "block":
                body["covered"] = True

    def reveal_named(self, memory: Dict[str, Any], block_id: str) -> bool:
        self.ep.tick("reveal_named")
        cup = str((memory.get("map") or {}).get(block_id) or "")
        expected = ""
        body = self.ep.bodies.get(block_id)
        if body is not None:
            expected = str(body.get("cup") or "")
        if not cup or cup != expected:
            self.ep.picked.append("WRONG")
            return False
        self.ep.picked.append(str(block_id))
        if body is not None:
            body["covered"] = False
        return True

    def reveal_guess(self) -> bool:
        self.ep.tick("reveal_guess")
        self.ep.picked.append("WRONG")
        return False

    def carry(self, obj_id: str, place: str) -> bool:
        self.ep.tick(f"carry:{obj_id}->{place}")
        body = self.ep.bodies.get(obj_id)
        spot = self.ep.places.get(place)
        if body is None or spot is None:
            return False
        body["x"], body["y"] = spot
        body["place"] = place
        return True

    def nudge_together(self) -> None:
        self.ep.tick("nudge_together")
        for body in self.ep.bodies.values():
            if body["kind"] != "piece":
                continue
            body["x"] = 0.5
            body["ori"] = float(body["ori"]) + 1.2
            body["place"] = "mid"

    def swap_direct(self) -> None:
        self.ep.tick("swap_direct")
        a = self.ep.bodies.get("A")
        b = self.ep.bodies.get("B")
        if a is None or b is None:
            return
        a["x"], b["x"] = b["x"], a["x"]
        a["place"], b["place"] = b["place"], a["place"]
        a["ori"] = float(a["ori"]) + 1.2
        b["ori"] = float(b["ori"]) + 1.2

    def touch(self, tile: Any) -> None:
        self.ep.tick("touch")
        self.ep.touched = _tile_id(tile)

    def place_at(self, tile: Any, slot_index: int) -> bool:
        self.ep.tick(f"place_at:{slot_index}")
        if slot_index < 0 or slot_index >= len(self.ep.slots):
            return False
        slot = self.ep.slots[slot_index]
        if slot["occ"]:
            return False
        tid = _tile_id(tile)
        body = self.ep.bodies.get(tid)
        if body is None:
            return False
        slot["occ"] = tid
        body["in_slot"] = slot["name"]
        return True

    def approach(self, target_id: str) -> None:
        self.ep.tick("approach")
        self.ep.pos_err = 0.12

    def align(self, target_id: str) -> None:
        self.ep.tick("align")
        self.ep.ori_err = 0.0

    def insert(self, target_id: str) -> bool:
        self.ep.tick("insert")
        if self.ep.pos_err <= 0.2 and self.ep.ori_err <= 0.15:
            self.ep.pos_err = 0.0
            self.ep.inserted = True
            return True
        return False

    def move_near(self, target_id: str) -> None:
        self.ep.tick("move_near")
        self.ep.pos_err = 0.02


class Tools:
    def __init__(self, ep: "Episode") -> None:
        self.ep = ep

    def log(self, text: str) -> None:
        self.ep.tick("tool")
        self.ep.tool_notes.append(str(text))


class Episode:
    MAX_CALLS = 48

    def __init__(self, task: str, seed: int) -> None:
        self.task = task
        self.seed = seed
        self.rng = random.Random(seed)
        self.calls = 0
        self.trace: List[str] = []
        self.bodies: Dict[str, Dict[str, Any]] = {}
        self.slots: List[Dict[str, Any]] = []
        self.places: Dict[str, tuple] = {}
        self.phase = "show"
        self.picked: List[str] = []
        self.remembered: Dict[str, str] = {}
        self.order: List[str] = []
        self.kong_value = ""
        self.touched = ""
        self.pos_err = 0.45
        self.ori_err = 0.9
        self.inserted = False
        self.tool_notes: List[str] = []
        self.error = ""
        self.skills = Skills(self)
        self.tools = Tools(self)
        self._setup()

    def tick(self, name: str) -> None:
        self.calls += 1
        self.trace.append(name)
        if self.calls > self.MAX_CALLS:
            raise RuntimeError("skill call budget exceeded")

    def _setup(self) -> None:
        if self.task == "make_kong":
            self._setup_kong()
        elif self.task == "cover_blocks":
            self._setup_cover()
        elif self.task == "swap_t":
            self._setup_swap()
        elif self.task == "arrange_number":
            self._setup_number()
        elif self.task == "insert_key":
            self._setup_key()
        else:
            raise KeyError(self.task)

    def _setup_kong(self) -> None:
        faces = ["1m", "2m", "3m", "7m", "9m", "east", "red", "green"]
        self.kong_value = self.rng.choice(["5m", "6m", "8m", "white"])
        distractors = self.rng.sample(faces, 6)
        ids = []
        for i, value in enumerate(distractors):
            bid = f"d{i}"
            self.bodies[bid] = {"id": bid, "kind": "tile", "value": value}
            ids.append(bid)
        for i in range(4):
            bid = f"k{i}"
            self.bodies[bid] = {"id": bid, "kind": "tile", "value": self.kong_value}
            ids.append(bid)
        self._perceive_order = ids
        for i in range(4):
            self.slots.append({"name": f"row_{i}", "x": float(i), "y": 0.0, "occ": ""})

    def _setup_cover(self) -> None:
        names = ["red", "green", "blue"]
        self.order = names[:]
        self.rng.shuffle(self.order)
        for i, name in enumerate(names):
            cup = f"cup{i}"
            self.bodies[name] = {
                "id": name,
                "kind": "block",
                "value": name,
                "cup": cup,
                "covered": False,
            }
            self.bodies[cup] = {"id": cup, "kind": "cup", "value": cup, "covers": name}

    def _setup_swap(self) -> None:
        self.places = {"home_a": (0.0, 0.0), "home_b": (1.0, 0.0), "buffer": (0.5, 0.8)}
        ori_a = round(self.rng.uniform(0.2, 1.0), 3)
        ori_b = round(self.rng.uniform(-1.0, -0.2), 3)
        self.bodies["A"] = {
            "id": "A",
            "kind": "piece",
            "x": 0.0,
            "y": 0.0,
            "ori": ori_a,
            "ori0": ori_a,
            "place": "home_a",
            "goal": "home_b",
        }
        self.bodies["B"] = {
            "id": "B",
            "kind": "piece",
            "x": 1.0,
            "y": 0.0,
            "ori": ori_b,
            "ori0": ori_b,
            "place": "home_b",
            "goal": "home_a",
        }

    def _setup_number(self) -> None:
        values = [str(v) for v in self.rng.sample(range(10), 4)]
        self._perceive_order = []
        for i, value in enumerate(values):
            bid = f"n{i}"
            self.bodies[bid] = {"id": bid, "kind": "digit", "value": value}
            self._perceive_order.append(bid)
        for i in range(4):
            self.slots.append({"name": f"slot_{i}", "x": float(i), "y": 0.0, "occ": ""})

    def _setup_key(self) -> None:
        self.pos_err = round(self.rng.uniform(0.35, 0.6), 3)
        self.ori_err = round(self.rng.uniform(0.4, 1.2), 3)
        self.bodies["key"] = {"id": "key", "kind": "key", "value": "key"}
        self.bodies["lock"] = {"id": "lock", "kind": "lock", "value": "lock"}

    def observe(self) -> Dict[str, Any]:
        objects: List[Dict[str, Any]] = []
        if self.task == "make_kong":
            for bid in self._perceive_order:
                body = self.bodies[bid]
                objects.append({"id": body["id"], "kind": "tile", "value": body["value"]})
        elif self.task == "cover_blocks":
            for body in self.bodies.values():
                if body["kind"] == "block" and body.get("covered"):
                    continue
                item = {"id": body["id"], "kind": body["kind"], "value": body["value"]}
                if body["kind"] == "block" and self.phase == "show":
                    item["cup"] = body["cup"]
                objects.append(item)
        elif self.task == "arrange_number":
            for bid in self._perceive_order:
                body = self.bodies[bid]
                objects.append({"id": body["id"], "kind": "digit", "value": body["value"]})
        elif self.task == "swap_t":
            for bid in ("A", "B"):
                body = self.bodies[bid]
                objects.append(
                    {
                        "id": bid,
                        "kind": "piece",
                        "place": body["place"],
                        "ori": body["ori"],
                    }
                )
        elif self.task == "insert_key":
            objects.append({"id": "key", "kind": "key", "pos_err": self.pos_err, "ori_err": self.ori_err})
            objects.append({"id": "lock", "kind": "lock"})
        return {"phase": self.phase, "objects": objects, "task": self.task}

    def goal(self) -> Dict[str, Any]:
        if self.task == "make_kong":
            return {"task": self.task, "row": "row", "need": 4}
        if self.task == "cover_blocks":
            return {"task": self.task, "order": list(self.order)}
        if self.task == "swap_t":
            return {"task": self.task, "buffer": "buffer"}
        if self.task == "arrange_number":
            return {"task": self.task, "order": "desc"}
        return {"task": self.task, "eps_p": 0.05, "eps_r": 0.15}

    def place_tile(self, tile: Any, dest: str) -> bool:
        del dest
        slot = next((s for s in self.slots if not s["occ"]), None)
        if slot is None:
            return False
        tid = _tile_id(tile)
        body = self.bodies.get(tid)
        if body is None:
            return False
        slot["occ"] = tid
        body["in_slot"] = slot["name"]
        return True

    def finish(self) -> Rollout:
        score, success, failure, note = self._score()
        if self.error and not failure:
            failure = "exception"
            note = self.error
        return Rollout(
            task=self.task,
            seed=self.seed,
            success=success,
            score=float(score),
            failure=failure,
            trace=list(self.trace),
            note=note,
        )

    def _score(self):
        if self.task == "make_kong":
            correct = 0
            for slot in self.slots:
                occ = slot["occ"]
                if occ and self.bodies[occ]["value"] == self.kong_value:
                    correct += 1
            score = 25 * correct
            if correct == 0:
                failure = "no_selection"
            elif correct < 4:
                failure = "partial_quad"
            else:
                failure = ""
            return score, correct == 4, failure, f"kong={self.kong_value} placed={correct}/4"
        if self.task == "cover_blocks":
            good = 0
            for got, want in zip(self.picked, self.order):
                if got != want:
                    break
                good += 1
            score = round(100.0 * good / len(self.order))
            if good == 0:
                failure = "memory_miss"
            elif good < len(self.order):
                failure = "partial_sequence"
            else:
                failure = ""
            return score, good == len(self.order), failure, f"order={','.join(self.order)} got={good}"
        if self.task == "swap_t":
            pos_pts = 0
            ori_pts = 0
            for body in (self.bodies["A"], self.bodies["B"]):
                if body["place"] != body["goal"]:
                    continue
                ori_ok = abs(float(body["ori"]) - float(body["ori0"])) <= 0.05
                if ori_ok:
                    pos_pts += 30
                    ori_pts += 20
                else:
                    # 到位但朝向丢了，低于「先把一块完整搬到目标」的部分方案。
                    pos_pts += 15
            score = pos_pts + ori_pts
            if score == 0:
                failure = "no_buffer"
            elif score == 100:
                failure = ""
            elif ori_pts == 0:
                failure = "orientation_miss"
            else:
                failure = "partial_swap"
            return score, score == 100, failure, f"pos={pos_pts} ori={ori_pts}"
        if self.task == "arrange_number":
            target = sorted((int(b["value"]) for b in self.bodies.values()), reverse=True)
            got = []
            for slot in self.slots:
                occ = slot["occ"]
                got.append(int(self.bodies[occ]["value"]) if occ else None)
            correct = sum(1 for a, b in zip(got, target) if a == b)
            score = 25 * correct
            if correct == 0:
                failure = "no_order"
            elif correct < 4:
                failure = "partial_order"
            else:
                failure = ""
            return score, correct == 4, failure, f"target={target} got={got}"
        pos_pts = 40 if self.pos_err <= 0.05 else (20 if self.pos_err <= 0.15 else 0)
        ori_pts = 40 if self.ori_err <= 0.15 else 0
        if self.inserted:
            return 100, True, "", f"pos_err={self.pos_err} ori_err={self.ori_err} inserted"
        failure = "no_stage" if pos_pts == 0 and ori_pts == 0 else (
            "alignment_miss" if ori_pts == 0 else "not_inserted"
        )
        return pos_pts + ori_pts, False, failure, f"pos_err={self.pos_err} ori_err={self.ori_err}"


def _tile_id(tile: Any) -> str:
    if isinstance(tile, dict):
        return str(tile.get("id") or "")
    return str(tile)


def harness_rejected(src: str) -> str:
    text = src or ""
    if "def act" not in text:
        return "缺少 def act"
    for token in _FORBIDDEN:
        if token in text:
            return f"harness 含有不允许的片段: {token.strip()}"
    try:
        compile(text, "<harness>", "exec")
    except SyntaxError as exc:
        return f"语法错误: {exc}"
    return ""


def run_rollout(harness_src: str, task: str, seed: int) -> Rollout:
    ep = Episode(task, seed)
    reason = harness_rejected(harness_src)
    if reason:
        ep.error = reason
        out = ep.finish()
        out.success = False
        out.score = 0.0
        out.failure = "rejected"
        out.note = reason
        return out
    namespace: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(harness_src, namespace, namespace)  # noqa: S102  harness 是被评测的控制代码
        act = namespace.get("act")
        if not callable(act):
            raise TypeError("act 不可调用")
        obs = ep.observe()
        goal = ep.goal()
        memory: Dict[str, Any] = {}
        returned = act(obs, goal, memory, ep.skills, ep.tools)
        if isinstance(returned, dict):
            memory = returned
        del memory
    except Exception as exc:
        ep.error = f"{type(exc).__name__}: {exc}"
    return ep.finish()
