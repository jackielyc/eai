"""把睿研手骨架从手腕 TCP 投到头/腕相机。

末节和近节共用同一个 0~1 开合量，因为 /ry_hand 只上报 6 个电机。
指尖是末节关节再沿指骨方向延伸的网格长度，不是单独的连杆原点。
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from eai_app.domain.calibration.cameras import CAMERAS, compose_base_camera, load_store

# 电机名 -> (URDF 关节后缀, 上限弧度, 该连杆上的指尖偏移；仅末节有)
# 后缀相对 hand{1|2}_link_
_FINGERS = (
    ("thumb", (255, 140, 0), (
        ("1_1", 2.356, None),
        ("1_2", 0.698, None),
        ("1_3", 1.169, (0.041, -0.02, 0.0)),
    )),
    ("index", (0, 220, 0), (
        ("2_1", 1.518, None),
        ("2_2", 1.623, (0.049, 0.0, 0.0)),
    )),
    ("middle", (220, 180, 0), (
        ("3_1", 1.571, None),
        ("3_2", 1.641, (0.049, 0.0, 0.0)),
    )),
    ("ring", (220, 0, 220), (
        ("4_1", 1.571, None),
        ("4_2", 1.658, (0.049, 0.0, 0.0)),
    )),
    ("pinky", (180, 180, 180), (
        ("5_1", 1.545, None),
        ("5_2", 1.658, (0.049, 0.0, 0.0)),
    )),
)

# 关节原点：parent 是 "palm" 或上一级后缀。右手指骨沿 -X，指尖偏移取反。
_LEFT_JOINTS = {
    "1_1": ("palm", (0.0142087, -0.022, -0.0687731), (0.0, 0.0, 1.5707998)),
    "1_2": ("1_1", (-0.0105, 0.0, 0.0107), (1.5707927, -0.0000001, -3.1415925)),
    "1_3": ("1_2", (0.040457, 0.011223, 0.0), (0.0, 0.0, 0.0)),
    "2_1": ("palm", (-0.0027913, -0.029833, 0.002377), (-0.0000071, -1.4923002, -1.5707928)),
    "2_2": ("2_1", (0.0418999, -0.001, 0.0), (0.0, 0.0000001, 0.0)),
    "3_1": ("palm", (-0.0027913, -0.009945, 0.003427), (-0.000011, -1.5446, -1.5707891)),
    "3_2": ("3_1", (0.0518999, -0.001, 0.0), (0.0, 0.0, 0.0)),
    "4_1": ("palm", (-0.0027913, 0.009945, 0.0024169), (3.141592, -1.5446, 1.5707934)),
    "4_2": ("4_1", (0.0419001, -0.001, 0.0), (0.0, 0.0, 0.0)),
    "5_1": ("palm", (-0.0027913, 0.029624, -0.0006231), (3.141592, -1.4923, 1.5707934)),
    "5_2": ("5_1", (0.0369, -0.001, 0.0), (0.0, 0.0, 0.0)),
}
_RIGHT_JOINTS = {
    "1_1": ("palm", (0.0135, 0.022, 0.0398), (3.1415927, 0.0, -1.5707927)),
    "1_2": ("1_1", (-0.0105, 0.0, -0.0107), (-1.5707927, -0.0000001, -3.1415927)),
    "1_3": ("1_2", (0.040457, 0.011223, 0.0), (0.0, 0.0, 0.0)),
    "2_1": ("palm", (-0.0035, 0.029833, 0.11095), (-3.1415913, 1.4923003, -1.5707915)),
    "2_2": ("2_1", (-0.0419, 0.001, 0.0), (0.0, 0.0000001, 0.0000001)),
    "3_1": ("palm", (-0.0035, 0.009945, 0.112), (3.141592, 1.5446001, -1.5707933)),
    "3_2": ("3_1", (-0.0519, 0.001, 0.0), (0.0, 0.0, 0.0)),
    "4_1": ("palm", (-0.0035, -0.009945, 0.11099), (-0.0000077, 1.5446, 1.5707923)),
    "4_2": ("4_1", (-0.0419, 0.001, 0.0), (0.0, 0.0, 0.0)),
    "5_1": ("palm", (-0.0035, -0.029624, 0.10795), (-0.0000083, 1.4923, 1.5707916)),
    "5_2": ("5_1", (-0.0368999, 0.001, 0.0), (0.0, 0.0000001, 0.0000001)),
}
_PALM = {
    "left": ((0.0002775, 0.0004126, 0.1149153), (0.0000029, 0.0065281, 1.5707946), _LEFT_JOINTS, 1.0),
    "right": ((0.0002782, -0.0004059, 0.0063399), (0.0000035, -0.0016238, -1.5707946), _RIGHT_JOINTS, -1.0),
}
_MOTOR = {
    "1_1": "thumb_rotation",
    "1_2": "thumb_bend",
    "1_3": "thumb_bend",
    "2_1": "index",
    "2_2": "index",
    "3_1": "middle",
    "3_2": "middle",
    "4_1": "ring",
    "4_2": "ring",
    "5_1": "pinky",
    "5_2": "pinky",
}
_NOMINAL_SIZE = {
    "/camera/head_color": (1280, 800),
    "/camera/hand_left_color": (848, 480),
    "/camera/hand_right_color": (848, 480),
}


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pose(xyz, rpy, angle: float = 0.0) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rpy(*rpy) @ _rot_z(angle)
    t[:3, 3] = np.asarray(xyz, dtype=float)
    return t


def _norm(motors: Mapping[str, float], name: str) -> float:
    try:
        value = float(motors.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def skeleton_in_link(side: str, motors: Mapping[str, float]) -> Tuple[Dict[str, np.ndarray], Tuple[Tuple[str, str, Tuple[int, int, int]], ...]]:
    """返回 Link7 坐标系下的关节点，以及要连的边。"""
    palm_xyz, palm_rpy, joints, tip_sign = _PALM[side]
    frames = {"palm": _pose(palm_xyz, palm_rpy)}
    points = {"wrist": np.zeros(3), "palm": frames["palm"][:3, 3].copy()}
    edges = [("wrist", "palm", (255, 255, 255))]
    for finger, color, bones in _FINGERS:
        previous = "palm"
        for suffix, upper, tip in bones:
            parent, xyz, rpy = joints[suffix]
            angle = _norm(motors, _MOTOR[suffix]) * upper
            frames[suffix] = frames[parent] @ _pose(xyz, rpy, angle)
            points[f"{finger}:{suffix}"] = frames[suffix][:3, 3].copy()
            edges.append((previous if previous == "palm" else f"{finger}:{previous}", f"{finger}:{suffix}", color))
            previous = suffix
            if tip is not None:
                local = np.array([tip_sign * tip[0], tip_sign * tip[1], tip[2], 1.0])
                points[f"{finger}:tip"] = (frames[suffix] @ local)[:3]
                edges.append((f"{finger}:{suffix}", f"{finger}:tip", color))
    return points, tuple(edges)


def skeleton_in_base(side: str, tcp_xyz, tcp_xyzw, motors: Mapping[str, float]):
    points, edges = skeleton_in_link(side, motors)
    from eai_app.domain.calibration.cameras import rotation_from_xyzw

    rotation = rotation_from_xyzw(tcp_xyzw)
    origin = np.asarray(tcp_xyz, dtype=float)
    world = {name: origin + rotation @ point for name, point in points.items()}
    return world, edges


def _canonical_topic(topic: str) -> str:
    text = str(topic or "")
    if "hand_left" in text or "left_wrist" in text:
        return "/camera/hand_left_color"
    if "hand_right" in text or "right_wrist" in text:
        return "/camera/hand_right_color"
    if "head" in text:
        return "/camera/head_color"
    return text


def _camera_to_base(topic: str, tcp_for_side) -> Tuple[Optional[np.ndarray], str]:
    topic = _canonical_topic(topic)
    entry = (load_store().get("cameras") or {}).get(topic)
    if not entry:
        return None, "无外参"
    if entry.get("model") == "eye_in_hand":
        tcp = tcp_for_side(entry.get("arm_side"))
        if not tcp or entry.get("T_tcp_camera") is None:
            return None, "无外参"
        return compose_base_camera(tcp["xyz"], tcp["xyzw"], entry["T_tcp_camera"]), "外参"
    if entry.get("T_base_camera") is None:
        return None, "无外参"
    return np.asarray(entry["T_base_camera"], dtype=float), "外参"


def _intrinsics(topic: str, width: int, height: int):
    spec = next((item for item in CAMERAS if item["topic"] == topic), None)
    if spec is None:
        return None
    nominal = _NOMINAL_SIZE.get(topic, (width, height))
    sx = float(width) / float(nominal[0])
    sy = float(height) / float(nominal[1])
    return spec["fx"] * sx, spec["fy"] * sy, spec["cx"] * sx, spec["cy"] * sy


def _project(points: Dict[str, np.ndarray], t_base_camera: np.ndarray, k, width: int, height: int):
    t_camera_base = np.linalg.inv(t_base_camera)
    fx, fy, cx, cy = k
    uv = {}
    for name, point in points.items():
        cam = t_camera_base @ np.array([point[0], point[1], point[2], 1.0])
        if cam[2] <= 0.02:
            continue
        u = fx * cam[0] / cam[2] + cx
        v = fy * cam[1] / cam[2] + cy
        if 0 <= u < width and 0 <= v < height:
            uv[name] = (int(round(u)), int(round(v)))
    return uv


def draw_on_image(image: np.ndarray, topic: str, hands_base, tcp_for_side):
    """hands_base: side -> (points, edges)。返回绘制后的图和这路相机看见的指尖数。"""
    canvas = np.ascontiguousarray(image.copy())
    height, width = canvas.shape[:2]
    t_base, reason = _camera_to_base(topic, tcp_for_side)
    k = _intrinsics(_canonical_topic(topic), width, height)
    if t_base is None or k is None:
        return canvas, {"topic": topic, "ok": False, "reason": reason, "tips": 0, "tips_total": 0}
    import cv2

    tips_total = 0
    tips_seen = 0
    for points, edges in hands_base:
        uv = _project(points, t_base, k, width, height)
        for a, b, color in edges:
            if a in uv and b in uv:
                cv2.line(canvas, uv[a], uv[b], color, 2, cv2.LINE_AA)
        for name, pixel in uv.items():
            if name == "wrist":
                cv2.circle(canvas, pixel, 4, (255, 255, 255), -1, cv2.LINE_AA)
            elif name.endswith(":tip"):
                cv2.circle(canvas, pixel, 5, (0, 255, 255), -1, cv2.LINE_AA)
        for name in points:
            if name.endswith(":tip"):
                tips_total += 1
                if name in uv:
                    tips_seen += 1
    cv2.putText(canvas, "HAND SKELETON", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas, {"topic": topic, "ok": True, "reason": "外参", "tips": tips_seen, "tips_total": tips_total}


def overlay_views(images: Dict[str, np.ndarray], tcp: Mapping[str, dict], motors: Mapping[str, Mapping[str, float]], tcp_for_side):
    hands = []
    missing = []
    for side in ("left", "right"):
        pose = tcp.get(side)
        if not pose:
            missing.append(side)
            continue
        hands.append(skeleton_in_base(side, pose["xyz"], pose["xyzw"], motors.get(side) or {}))
    drawn = {}
    reports = []
    ready = 0
    for topic, image in images.items():
        if image is None:
            continue
        canvas, report = draw_on_image(image, topic, hands, tcp_for_side)
        drawn[topic] = canvas
        reports.append(report)
        if report["ok"]:
            ready += 1
    joint = ready >= 2
    parts = []
    for report in reports:
        short = report["topic"].split("/")[-1]
        if report["ok"]:
            parts.append(f"{short} 指尖 {report['tips']}/{report['tips_total']}")
        else:
            parts.append(f"{short} {report['reason']}")
    if missing:
        parts.append("缺手腕 " + ",".join(missing))
    if joint:
        summary = "联合投影：" + "；".join(parts) + "。末节按开合比例。"
    else:
        summary = "还不能联合：" + "；".join(parts or ["没有画面"]) + "。至少两个相机要有外参。"
    return drawn, summary
