"""把手腕演示轨迹标定到 LIBERO 场景：物体上方 → 抽屉内放置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from human_demo_imitate import DemoTrajectory


@dataclass
class AlignedWaypoint:
    t_sec: float
    ee_pose7_wxyz: Tuple[float, float, float, float, float, float, float]
    gripper01: float  # 0=open 1=closed
    note: str = ""


def _as_xyz(val: Any) -> Optional[np.ndarray]:
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _pick_object_pos(objects: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    if not isinstance(objects, dict) or not objects:
        return None, ""
    prefer = (
        "bowl",
        "akita_black_bowl",
        "plate",
        "cream_cheese",
        "wine_bottle",
        "mug",
    )
    keys = list(objects.keys())
    lower_map = {k.lower(): k for k in keys}
    for p in prefer:
        for lk, orig in lower_map.items():
            if p in lk:
                pos = _as_xyz((objects.get(orig) or {}).get("pos"))
                if pos is not None:
                    return pos, orig
    # 任意有 pos 的物体
    for k, v in objects.items():
        if not isinstance(v, dict):
            continue
        pos = _as_xyz(v.get("pos"))
        if pos is not None and "drawer" not in k.lower() and "cabinet" not in k.lower():
            return pos, k
    return None, ""


def _pick_drawer_place(objects: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    if not isinstance(objects, dict):
        return None, ""
    for k, v in objects.items():
        if not isinstance(v, dict):
            continue
        lk = k.lower()
        if "drawer" not in lk:
            continue
        pos = _as_xyz(v.get("pos"))
        if pos is None:
            continue
        # 抽屉开口上方略内侧：+x（桌面前方）偏置 + 略降
        place = pos.copy()
        place[0] += 0.05
        place[2] += 0.02
        joint = v.get("joint")
        try:
            if joint is not None and float(joint) < 0.05:
                # 未打开时仍给目标，靠偏置靠近抽屉前方
                place[0] += 0.08
        except (TypeError, ValueError):
            pass
        return place, k
    # 无抽屉：用柜体
    for k, v in objects.items():
        if not isinstance(v, dict):
            continue
        if "cabinet" in k.lower():
            pos = _as_xyz(v.get("pos"))
            if pos is not None:
                p = pos.copy()
                p[2] += 0.05
                return p, k
    return None, ""


def _quat_wxyz_from_state(pose7: Sequence[float]) -> Tuple[float, float, float, float]:
    # pose7 = x y z qw qx qy qz
    return (
        float(pose7[3]),
        float(pose7[4]),
        float(pose7[5]),
        float(pose7[6]),
    )


def _collect_wrist_series(
    traj: DemoTrajectory,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times (N,), world_or_img wrists (N,3), grip_close (N,)."""
    ts: List[float] = []
    wrists: List[List[float]] = []
    grips: List[float] = []
    for fr in traj.frames:
        if not fr.hands:
            continue
        # 优先 Right，其次第一只
        hand = None
        for h in fr.hands:
            if h.handedness.lower().startswith("r"):
                hand = h
                break
        if hand is None:
            hand = fr.hands[0]
        if hand.wrist_world is not None and len(hand.wrist_world) >= 3:
            w = [float(x) for x in hand.wrist_world[:3]]
        else:
            w = [float(x) for x in hand.wrist_xyz[:3]]
        g = float(sum(hand.joints) / max(1, len(hand.joints)))
        ts.append(float(fr.t_sec))
        wrists.append(w)
        grips.append(g)
    if not ts:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(ts, dtype=np.float64),
        np.asarray(wrists, dtype=np.float64),
        np.asarray(grips, dtype=np.float64),
    )


def build_scene_aligned_waypoints(
    traj: DemoTrajectory,
    sim_state: Dict[str, Any],
    *,
    approach_height: float = 0.08,
    n_min: int = 24,
) -> List[AlignedWaypoint]:
    """把手腕形状相似变换到「物体上方 → 放置点」。"""
    pose7 = sim_state.get("left_ee_pose") or sim_state.get("right_ee_pose")
    if not isinstance(pose7, (list, tuple)) or len(pose7) < 7:
        raise RuntimeError("sim_robot_state 缺少 EE 位姿")
    objects = sim_state.get("objects") or {}
    if not isinstance(objects, dict) or not objects:
        raise RuntimeError("sim_robot_state 无 objects；请用带任务序号的 LIBERO「只开界面」")

    obj_pos, obj_name = _pick_object_pos(objects)
    place_pos, place_name = _pick_drawer_place(objects)
    if obj_pos is None:
        raise RuntimeError("未找到可放置物体（bowl/plate/…）位姿")
    if place_pos is None:
        # 回退：物体前方
        place_pos = obj_pos.copy()
        place_pos[0] += 0.15
        place_name = "offset_forward"

    start = obj_pos.copy()
    start[2] += float(approach_height)
    end = place_pos.copy()
    end[2] = max(float(end[2]), float(obj_pos[2]) + 0.02)

    times, wrists, grips = _collect_wrist_series(traj)
    if times.size < 2:
        raise RuntimeError("轨迹有效手腕帧不足，请先分析视频")

    # 归一化时间
    t0, t1 = float(times[0]), float(times[-1])
    dur = max(1e-3, t1 - t0)
    alpha = (times - t0) / dur

    # 手腕相对首帧的形状
    w0 = wrists[0]
    shape = wrists - w0
    # 目标主方向
    main = end - start
    main_norm = float(np.linalg.norm(main))
    if main_norm < 1e-4:
        main = np.array([0.12, 0.0, 0.0], dtype=np.float64)
        main_norm = 0.12
    # 取手腕位移最大轴做对齐缩放
    shape_span = float(np.max(np.linalg.norm(shape, axis=1)))
    if shape_span < 1e-6:
        shape_span = 1.0
    scale = main_norm / shape_span
    # 将 shape 投影到主方向 + 保留少量横向
    u = main / main_norm
    # 构造正交基
    tmp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v1 = np.cross(u, tmp)
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(u, v1)

    quat = _quat_wxyz_from_state(pose7)
    waypoints: List[AlignedWaypoint] = []
    for i in range(times.size):
        s = shape[i] * scale
        # 混合：沿主方向按 alpha 插值 + 形状分量
        along = start + u * (main_norm * float(alpha[i]))
        lateral = v1 * float(np.dot(s, v1)) * 0.35 + v2 * float(np.dot(s, v2)) * 0.35
        xyz = along + lateral
        # 夹爪：前 25% 开，之后跟手指闭合度（至少半闭）
        a = float(alpha[i])
        if a < 0.25:
            g = 0.05
        elif a > 0.85:
            g = max(0.55, float(grips[i]))
        else:
            g = float(np.clip(grips[i], 0.2, 1.0))
        pose = (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        )
        waypoints.append(
            AlignedWaypoint(
                t_sec=float(times[i]),
                ee_pose7_wxyz=pose,
                gripper01=g,
                note=f"{obj_name}->{place_name}",
            )
        )

    # 保证最少航点（线性插值补齐）
    if len(waypoints) < n_min:
        denser: List[AlignedWaypoint] = []
        for j in range(n_min):
            a = j / max(1, n_min - 1)
            xyz = start * (1 - a) + end * a
            g = 0.05 if a < 0.25 else (0.7 if a > 0.5 else 0.35)
            denser.append(
                AlignedWaypoint(
                    t_sec=t0 + a * dur,
                    ee_pose7_wxyz=(
                        float(xyz[0]),
                        float(xyz[1]),
                        float(xyz[2]),
                        float(quat[0]),
                        float(quat[1]),
                        float(quat[2]),
                        float(quat[3]),
                    ),
                    gripper01=g,
                    note=f"lerp:{obj_name}->{place_name}",
                )
            )
        waypoints = denser
    return waypoints


__all__ = ["AlignedWaypoint", "build_scene_aligned_waypoints"]
