"""摄像头手骨架检测 → 睿研灵巧手 6 关节开合量。

依赖: mediapipe==0.10.x（solutions.Hands，模型随包自带）。
关节顺序与 show_camera_topics 一致:
  thumb_rotation, thumb_bend, index, middle, ring, pinky
数值约定: 0=张开, 1=闭合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

HAND_JOINT_ORDER = (
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
)

# MediaPipe Hands landmark indices
_WRIST = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_INDEX_MCP, _INDEX_PIP, _INDEX_TIP = 5, 6, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP = 9, 10, 12
_RING_MCP, _RING_PIP, _RING_TIP = 13, 14, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


@dataclass
class DetectedHand:
    """归一化 landmark (x,y,z) 已相对图像；handedness 为人物自身左右。"""

    handedness: str  # "Left" | "Right"
    score: float
    landmarks_norm: np.ndarray  # (21, 3)
    joints: Tuple[float, float, float, float, float, float]  # 6 关节 0~1


def mediapipe_available() -> bool:
    try:
        from mediapipe.python.solutions.hands import Hands  # noqa: F401

        return True
    except Exception:
        return False


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _finger_curl(lm: np.ndarray, mcp: int, tip: int, palm: float) -> float:
    """指尖相对 MCP 收缩程度：张开≈0，握紧≈1。"""
    tip_mcp = _dist(lm[tip], lm[mcp])
    # 张开时 tip-MCP ≈ 0.9~1.1 * palm；握紧时接近 0
    open_ref = max(palm * 0.95, 1e-6)
    return _clip01(1.0 - tip_mcp / open_ref)


def _thumb_bend(lm: np.ndarray, palm: float) -> float:
    tip_mcp = _dist(lm[_THUMB_TIP], lm[_THUMB_MCP])
    open_ref = max(palm * 0.75, 1e-6)
    return _clip01(1.0 - tip_mcp / open_ref)


def _thumb_rotation(lm: np.ndarray, palm: float) -> float:
    """拇指外展：靠近食指≈合(1)，张开≈张(0)。用 tip 到 index MCP 距离。"""
    tip_index = _dist(lm[_THUMB_TIP], lm[_INDEX_MCP])
    open_ref = max(palm * 1.15, 1e-6)
    # 距离大 → 外展 → 开(0)；距离小 → 内收 → 合(1)
    return _clip01(1.0 - tip_index / open_ref)


def landmarks_to_hand_joints(landmarks_norm: np.ndarray) -> Tuple[float, ...]:
    """21 点归一化坐标 → 6 关节开合量。"""
    lm = np.asarray(landmarks_norm, dtype=np.float64)
    if lm.shape != (21, 3) and lm.shape != (21, 2):
        raise ValueError(f"expected landmarks (21,2|3), got {lm.shape}")
    if lm.shape[1] == 2:
        lm = np.concatenate([lm, np.zeros((21, 1), dtype=np.float64)], axis=1)
    palm = _dist(lm[_MIDDLE_MCP], lm[_WRIST])
    if palm < 1e-6:
        palm = 1e-6
    return (
        _thumb_rotation(lm, palm),
        _thumb_bend(lm, palm),
        _finger_curl(lm, _INDEX_MCP, _INDEX_TIP, palm),
        _finger_curl(lm, _MIDDLE_MCP, _MIDDLE_TIP, palm),
        _finger_curl(lm, _RING_MCP, _RING_TIP, palm),
        _finger_curl(lm, _PINKY_MCP, _PINKY_TIP, palm),
    )


def smooth_joints(
    prev: Optional[Sequence[float]],
    cur: Sequence[float],
    alpha: float = 0.35,
) -> Tuple[float, ...]:
    """指数平滑，减轻抖动。alpha 越大越跟手。"""
    if prev is None:
        return tuple(float(v) for v in cur)
    a = float(max(0.0, min(1.0, alpha)))
    return tuple(a * float(c) + (1.0 - a) * float(p) for p, c in zip(prev, cur))


class HandSkeletonDetector:
    """MediaPipe Hands 封装；线程内单实例使用。"""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.python.solutions.hands import Hands

        self._hands = Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._smooth: dict[str, Tuple[float, ...]] = {}

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def detect(
        self,
        bgr: np.ndarray,
        flip_horizontal: bool = False,
        smooth_alpha: float = 0.35,
    ) -> List[DetectedHand]:
        if bgr is None or bgr.size == 0:
            return []
        frame = bgr
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        out: List[DetectedHand] = []
        if not result.multi_hand_landmarks:
            return out
        handedness_list = result.multi_handedness or []
        for i, hand_lms in enumerate(result.multi_hand_landmarks):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float64
            )
            label = "Right"
            score = 0.0
            if i < len(handedness_list):
                cats = handedness_list[i].classification
                if cats:
                    label = str(cats[0].label)
                    score = float(cats[0].score)
            joints = landmarks_to_hand_joints(pts)
            joints = smooth_joints(self._smooth.get(label), joints, alpha=smooth_alpha)
            self._smooth[label] = joints
            out.append(
                DetectedHand(
                    handedness=label,
                    score=score,
                    landmarks_norm=pts,
                    joints=tuple(joints),  # type: ignore[arg-type]
                )
            )
        return out


def draw_hand_skeleton(
    bgr: np.ndarray,
    hands: Sequence[DetectedHand],
    flip_horizontal: bool = False,
) -> np.ndarray:
    """在图像上绘制骨架与关节开合文字。"""
    vis = np.ascontiguousarray(bgr.copy())
    if flip_horizontal:
        vis = cv2.flip(vis, 1)
    h, w = vis.shape[:2]
    colors = {
        "Left": (80, 200, 255),
        "Right": (80, 255, 140),
    }
    for hand in hands:
        color = colors.get(hand.handedness, (255, 255, 100))
        pts = []
        for x, y, _z in hand.landmarks_norm:
            u = int(x * w)
            v = int(y * h)
            pts.append((u, v))
            cv2.circle(vis, (u, v), 3, color, -1, cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(vis, pts[a], pts[b], color, 2, cv2.LINE_AA)
        if pts:
            label = (
                f"{hand.handedness} "
                f"T{hand.joints[0]:.2f}/{hand.joints[1]:.2f} "
                f"I{hand.joints[2]:.2f} M{hand.joints[3]:.2f} "
                f"R{hand.joints[4]:.2f} P{hand.joints[5]:.2f}"
            )
            cv2.putText(
                vis,
                label,
                (pts[0][0] + 8, max(18, pts[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    if flip_horizontal:
        vis = cv2.flip(vis, 1)
    return vis


def map_person_hand_to_robot_side(
    person_handedness: str,
    mirror_mapping: bool,
) -> str:
    """人物 Left/Right → 机器人 left/right。

    mirror_mapping=True 时左右对调（面对机器人时更自然）。
    """
    is_left = person_handedness.lower().startswith("l")
    if mirror_mapping:
        is_left = not is_left
    return "left" if is_left else "right"
