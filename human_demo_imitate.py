"""人类演示视频 → MediaPipe Hands 轨迹 → 机器人手/臂复现。

单目 RGB 无法精确恢复绝对 3D；手腕位移按图像归一化坐标相对首帧做尺度映射，
手指开合复用 hand_skeleton_teleop 的 6 关节映射。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hand_skeleton_teleop import (
    HAND_CONNECTIONS,
    HandSkeletonDetector,
    map_person_hand_to_robot_side,
    mediapipe_available,
)

ProgressCb = Optional[Callable[[float, str], None]]

# 默认：全图宽度约对应 0.30 m 工作空间位移
DEFAULT_ARM_SCALE_M = 0.30
DEFAULT_MAX_DELTA_M = 0.28
DEFAULT_TARGET_FPS = 12.0
# world landmarks 相对位移已是米制，再乘增益对齐工作空间
DEFAULT_WORLD_GAIN = 1.6
TRAJ_CACHE_VERSION = 2


@dataclass
class DemoHandSample:
    handedness: str  # "Left" | "Right"
    score: float
    joints: Tuple[float, float, float, float, float, float]
    wrist_xyz: Tuple[float, float, float]
    landmarks_norm: Optional[List[List[float]]] = None  # 21×3，预览骨架用
    wrist_world: Optional[Tuple[float, float, float]] = None  # MediaPipe 米制


@dataclass
class DemoFrame:
    index: int
    t_sec: float
    hands: List[DemoHandSample] = field(default_factory=list)


@dataclass
class DemoTrajectory:
    video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    sample_stride: int
    frames: List[DemoFrame]
    analyzed_at: float = 0.0

    def summary(self) -> str:
        n = len(self.frames)
        with_hand = sum(1 for f in self.frames if f.hands)
        left_n = sum(1 for f in self.frames for h in f.hands if h.handedness.startswith("L"))
        right_n = sum(1 for f in self.frames for h in f.hands if h.handedness.startswith("R"))
        dur = self.frames[-1].t_sec if self.frames else 0.0
        return (
            f"{n} 采样帧 / {self.frame_count} 总帧 · {dur:.1f}s · "
            f"有手 {with_hand} · L{left_n}/R{right_n} · stride={self.sample_stride}"
        )


def wrist_delta_to_base(
    wrist_now: Sequence[float],
    wrist0: Sequence[float],
    *,
    scale_m: float = DEFAULT_ARM_SCALE_M,
    mirror: bool = True,
    max_delta_m: float = DEFAULT_MAX_DELTA_M,
) -> Tuple[float, float, float]:
    """MediaPipe 手腕相对位移 → base_link (X前, Y左, Z上)。

    假设演示者大致面对相机：画面右→机器人右(−Y，镜像时)、画面上→上(+Z)、
    靠近相机(MediaPipe z 变小)→向前(+X)。
    """
    dx_img = float(wrist_now[0]) - float(wrist0[0])
    dy_img = float(wrist_now[1]) - float(wrist0[1])
    dz_img = float(wrist_now[2]) - float(wrist0[2])
    scale = float(max(1e-4, scale_m))
    # 图像 y 向下；靠近相机时 z 通常更小 → (−dz) 为正向前
    robot_dx = scale * (-dz_img) * 1.2
    robot_dy = scale * ((-dx_img) if mirror else dx_img)
    robot_dz = scale * (-dy_img)
    lim = float(max(0.01, max_delta_m))

    def _clip(v: float) -> float:
        return float(max(-lim, min(lim, v)))

    return (_clip(robot_dx), _clip(robot_dy), _clip(robot_dz))


def wrist_delta_to_base_world(
    wrist_now: Sequence[float],
    wrist0: Sequence[float],
    *,
    wrist_world_now: Optional[Sequence[float]] = None,
    wrist_world0: Optional[Sequence[float]] = None,
    scale_m: float = DEFAULT_ARM_SCALE_M,
    world_gain: float = DEFAULT_WORLD_GAIN,
    mirror: bool = True,
    max_delta_m: float = DEFAULT_MAX_DELTA_M,
) -> Tuple[float, float, float]:
    """优先用 MediaPipe world 手腕位移；缺失时回退图像启发式。

    world 约定（相对首帧）：+x 人物右、+y 上、+z 朝向相机外侧（因模型而异，取相对差）。
    映射到 base：X前 / Y左 / Z上。
    """
    if (
        wrist_world_now is not None
        and wrist_world0 is not None
        and len(wrist_world_now) >= 3
        and len(wrist_world0) >= 3
    ):
        dwx = float(wrist_world_now[0]) - float(wrist_world0[0])
        dwy = float(wrist_world_now[1]) - float(wrist_world0[1])
        dwz = float(wrist_world_now[2]) - float(wrist_world0[2])
        g = float(max(1e-4, world_gain))
        # 靠近相机（world z 变小）→ 向前；画面右 → 机器人右(−Y，镜像)
        robot_dx = g * (-dwz)
        robot_dy = g * ((-dwx) if mirror else dwx)
        robot_dz = g * dwy
        lim = float(max(0.01, max_delta_m))

        def _clip(v: float) -> float:
            return float(max(-lim, min(lim, v)))

        return (_clip(robot_dx), _clip(robot_dy), _clip(robot_dz))
    return wrist_delta_to_base(
        wrist_now,
        wrist0,
        scale_m=scale_m,
        mirror=mirror,
        max_delta_m=max_delta_m,
    )


def first_wrist_origins(traj: DemoTrajectory) -> Dict[str, Tuple[float, float, float]]:
    """每人手 Left/Right 的首个有效手腕（图像归一化，用于相对位移）。"""
    out: Dict[str, Tuple[float, float, float]] = {}
    for fr in traj.frames:
        for hand in fr.hands:
            key = "Left" if hand.handedness.lower().startswith("l") else "Right"
            if key not in out:
                out[key] = tuple(float(v) for v in hand.wrist_xyz)  # type: ignore[assignment]
        if len(out) >= 2:
            break
    return out


def first_wrist_world_origins(
    traj: DemoTrajectory,
) -> Dict[str, Tuple[float, float, float]]:
    """每人手首个有效 world 手腕（米制）。"""
    out: Dict[str, Tuple[float, float, float]] = {}
    for fr in traj.frames:
        for hand in fr.hands:
            key = "Left" if hand.handedness.lower().startswith("l") else "Right"
            if key in out:
                continue
            ww = hand.wrist_world
            if ww is None or len(ww) < 3:
                continue
            out[key] = (float(ww[0]), float(ww[1]), float(ww[2]))
        if len(out) >= 2:
            break
    return out


def analyze_demo_video(
    video_path: str,
    *,
    target_fps: float = DEFAULT_TARGET_FPS,
    max_frames: int = 0,
    flip_horizontal: bool = False,
    smooth_alpha: float = 0.35,
    progress: ProgressCb = None,
) -> DemoTrajectory:
    """离线分析人类演示视频，提取双手关节与手腕轨迹。"""
    if not mediapipe_available():
        raise RuntimeError(
            "未安装 mediapipe。请执行: python -m pip install 'mediapipe==0.10.14'"
        )
    path = os.path.abspath(os.path.expanduser(video_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频不存在: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1e-3:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        stride = max(1, int(round(fps / max(1.0, float(target_fps)))))

        detector = HandSkeletonDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.45,
        )
        frames: List[DemoFrame] = []
        idx = 0
        kept = 0
        t0 = time.time()
        try:
            while True:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                hands_det = detector.detect(
                    bgr, flip_horizontal=flip_horizontal, smooth_alpha=smooth_alpha
                )
                samples: List[DemoHandSample] = []
                for h in hands_det:
                    wrist = (
                        float(h.landmarks_norm[0, 0]),
                        float(h.landmarks_norm[0, 1]),
                        float(h.landmarks_norm[0, 2]),
                    )
                    lm = [
                        [float(x), float(y), float(z)]
                        for x, y, z in np.asarray(h.landmarks_norm, dtype=np.float64)
                    ]
                    wrist_world: Optional[Tuple[float, float, float]] = None
                    lw = getattr(h, "landmarks_world", None)
                    if lw is not None:
                        arr = np.asarray(lw, dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 3:
                            wrist_world = (
                                float(arr[0, 0]),
                                float(arr[0, 1]),
                                float(arr[0, 2]),
                            )
                    samples.append(
                        DemoHandSample(
                            handedness=h.handedness,
                            score=float(h.score),
                            joints=tuple(float(v) for v in h.joints),  # type: ignore[arg-type]
                            wrist_xyz=wrist,
                            landmarks_norm=lm,
                            wrist_world=wrist_world,
                        )
                    )
                t_sec = float(idx) / fps
                frames.append(DemoFrame(index=idx, t_sec=t_sec, hands=samples))
                kept += 1
                idx += 1
                if max_frames > 0 and kept >= max_frames:
                    break
                if progress and kept % 5 == 0:
                    pct = 0.0
                    if total > 0:
                        pct = min(0.99, float(idx) / float(total))
                    progress(pct, f"分析中 {kept} 帧 @ {idx}/{total or '?'}")
        finally:
            detector.close()
    finally:
        cap.release()

    if progress:
        progress(1.0, f"分析完成 {len(frames)} 帧 ({time.time() - t0:.1f}s)")

    return DemoTrajectory(
        video_path=path,
        fps=fps,
        frame_count=total if total > 0 else idx,
        width=width,
        height=height,
        sample_stride=stride,
        frames=frames,
        analyzed_at=time.time(),
    )


def trajectory_to_dict(traj: DemoTrajectory) -> dict:
    return {
        "cache_version": TRAJ_CACHE_VERSION,
        "video_path": traj.video_path,
        "fps": traj.fps,
        "frame_count": traj.frame_count,
        "width": traj.width,
        "height": traj.height,
        "sample_stride": traj.sample_stride,
        "analyzed_at": traj.analyzed_at,
        "frames": [
            {
                "index": f.index,
                "t_sec": f.t_sec,
                "hands": [asdict(h) for h in f.hands],
            }
            for f in traj.frames
        ],
    }


def trajectory_cache_version(data: dict) -> int:
    try:
        return int(data.get("cache_version") or 0)
    except (TypeError, ValueError):
        return 0


def trajectory_from_dict(data: dict) -> DemoTrajectory:
    frames: List[DemoFrame] = []
    for fr in data.get("frames") or []:
        hands = []
        for h in fr.get("hands") or []:
            joints = tuple(float(v) for v in h.get("joints") or (0, 0, 0, 0, 0, 0))
            if len(joints) != 6:
                joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            wrist = tuple(float(v) for v in h.get("wrist_xyz") or (0, 0, 0))
            if len(wrist) != 3:
                wrist = (0.0, 0.0, 0.0)
            lm_raw = h.get("landmarks_norm")
            lm: Optional[List[List[float]]] = None
            if isinstance(lm_raw, list) and lm_raw:
                lm = [[float(a) for a in row[:3]] for row in lm_raw if isinstance(row, (list, tuple))]
            ww_raw = h.get("wrist_world")
            wrist_world: Optional[Tuple[float, float, float]] = None
            if isinstance(ww_raw, (list, tuple)) and len(ww_raw) >= 3:
                wrist_world = (float(ww_raw[0]), float(ww_raw[1]), float(ww_raw[2]))
            hands.append(
                DemoHandSample(
                    handedness=str(h.get("handedness") or "Right"),
                    score=float(h.get("score") or 0.0),
                    joints=joints,  # type: ignore[arg-type]
                    wrist_xyz=wrist,  # type: ignore[arg-type]
                    landmarks_norm=lm,
                    wrist_world=wrist_world,
                )
            )
        frames.append(
            DemoFrame(
                index=int(fr.get("index") or 0),
                t_sec=float(fr.get("t_sec") or 0.0),
                hands=hands,
            )
        )
    return DemoTrajectory(
        video_path=str(data.get("video_path") or ""),
        fps=float(data.get("fps") or 30.0),
        frame_count=int(data.get("frame_count") or 0),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        sample_stride=int(data.get("sample_stride") or 1),
        frames=frames,
        analyzed_at=float(data.get("analyzed_at") or 0.0),
    )


def save_trajectory(traj: DemoTrajectory, path: str) -> str:
    out = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(trajectory_to_dict(traj), f, ensure_ascii=False)
    return out


def load_trajectory(path: str) -> DemoTrajectory:
    with open(os.path.abspath(os.path.expanduser(path)), "r", encoding="utf-8") as f:
        return trajectory_from_dict(json.load(f))


def cache_path_for_video(video_path: str, cache_dir: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)[:80]
    return os.path.join(cache_dir, f"{safe}_hand_traj.json")


def draw_demo_frame(
    bgr: np.ndarray,
    frame: DemoFrame,
    *,
    flip_horizontal: bool = False,
    title: str = "",
) -> np.ndarray:
    """在原帧上画分析得到的骨架（无需再跑 MediaPipe）。"""
    vis = np.ascontiguousarray(bgr.copy())
    if flip_horizontal:
        vis = cv2.flip(vis, 1)
    h, w = vis.shape[:2]
    colors = {"Left": (80, 200, 255), "Right": (80, 255, 140)}
    for hand in frame.hands:
        color = colors.get(hand.handedness, (255, 255, 100))
        pts: List[Tuple[int, int]] = []
        lm = hand.landmarks_norm
        if lm and len(lm) >= 21:
            for x, y, _z in lm[:21]:
                u = int(float(x) * w)
                v = int(float(y) * h)
                pts.append((u, v))
                cv2.circle(vis, (u, v), 3, color, -1, cv2.LINE_AA)
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(vis, pts[a], pts[b], color, 2, cv2.LINE_AA)
        else:
            u = int(hand.wrist_xyz[0] * w)
            v = int(hand.wrist_xyz[1] * h)
            pts.append((u, v))
            cv2.circle(vis, (u, v), 8, color, -1, cv2.LINE_AA)
        if pts:
            j = hand.joints
            label = (
                f"{hand.handedness} "
                f"T{j[0]:.2f}/{j[1]:.2f} "
                f"I{j[2]:.2f} M{j[3]:.2f} "
                f"R{j[4]:.2f} P{j[5]:.2f}"
            )
            cv2.putText(
                vis,
                label,
                (pts[0][0] + 10, max(20, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    if title:
        cv2.putText(
            vis,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
    if flip_horizontal:
        vis = cv2.flip(vis, 1)
    return vis


def read_video_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        return bgr
    finally:
        cap.release()


__all__ = [
    "DEFAULT_ARM_SCALE_M",
    "DEFAULT_MAX_DELTA_M",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_WORLD_GAIN",
    "TRAJ_CACHE_VERSION",
    "DemoFrame",
    "DemoHandSample",
    "DemoTrajectory",
    "analyze_demo_video",
    "cache_path_for_video",
    "draw_demo_frame",
    "first_wrist_origins",
    "first_wrist_world_origins",
    "load_trajectory",
    "map_person_hand_to_robot_side",
    "mediapipe_available",
    "read_video_frame",
    "save_trajectory",
    "trajectory_cache_version",
    "wrist_delta_to_base",
    "wrist_delta_to_base_world",
]
