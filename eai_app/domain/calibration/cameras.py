"""手工点对标定：彩色像素 + 深度 + TCP，求解相机外参。

头部是眼在手外，结果为 T_base_camera（只在标定当时的腰/头姿态下有效）。
腕部是眼在手上，结果为 T_tcp_camera；使用时再乘当前 TCP。
"""
from __future__ import annotations

from eai_app.paths import PROJECT_ROOT

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

STORE_PATH = PROJECT_ROOT / "config" / "camera_extrinsics.json"

CAMERAS = (
    {
        "topic": "/camera/head_color",
        "width": 1280, "height": 800,
        "label": "头部",
        "model": "eye_to_hand",
        "arm_side": "",
        # 机器人 /data/parameters/head_intrinsic_params.json，对应 1280×800 彩色。
        "fx": 645.1858520507812,
        "fy": 644.9612426757812,
        "cx": 639.5411987304688,
        "cy": 389.8989562988281,
    },
    {
        "topic": "/camera/hand_left_color",
        "width": 848, "height": 480,
        "label": "左腕",
        "model": "eye_in_hand",
        "arm_side": "left",
        # D405 SN 230322276520，彩色 848×480，从机器人当场枚举。
        "fx": 435.527313232422,
        "fy": 435.177124023438,
        "cx": 419.682006835938,
        "cy": 239.707855224609,
    },
    {
        "topic": "/camera/hand_right_color",
        "width": 848, "height": 480,
        "label": "右腕",
        "model": "eye_in_hand",
        "arm_side": "right",
        # D405 SN 230322274561，彩色 848×480，从机器人当场枚举。
        "fx": 433.735687255859,
        "fy": 433.373382568359,
        "cx": 420.275787353516,
        "cy": 242.550033569336,
    },
)


def rotation_from_xyzw(xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in xyzw]
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-9:
        raise ValueError("TCP 四元数无效")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def point_from_tcp(xyz, xyzw, offset_tcp_m) -> np.ndarray:
    """TCP 原点加上 TCP 坐标系下的点击点偏移。"""
    offset = np.asarray(offset_tcp_m, dtype=float)
    return np.asarray(xyz, dtype=float) + rotation_from_xyzw(xyzw) @ offset


def unproject(u: float, v: float, depth_m: float, k) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in k]
    if min(fx, fy) <= 1.0 or depth_m <= 1e-4:
        raise ValueError("内参或深度无效")
    z = float(depth_m)
    return np.array([(float(u) - cx) * z / fx, (float(v) - cy) * z / fy, z], dtype=float)


def solve_rigid(src: np.ndarray, dst: np.ndarray):
    """src 点经刚体变换后对齐 dst。返回 4×4 与每个点的米制误差。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("至少需要 3 组不共线的三维点")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    if np.linalg.matrix_rank(src - mu_s, tol=1e-3) < 2:
        raise ValueError("点几乎共线，请把位置拉开到不同方向")
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    err = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
    return T, err


def express_in_tcp(point_base, tcp_xyz, tcp_xyzw) -> np.ndarray:
    R = rotation_from_xyzw(tcp_xyzw)
    return R.T @ (np.asarray(point_base, dtype=float) - np.asarray(tcp_xyz, dtype=float))


def compose_base_camera(tcp_xyz, tcp_xyzw, T_tcp_camera) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation_from_xyzw(tcp_xyzw)
    T[:3, 3] = np.asarray(tcp_xyz, dtype=float)
    return T @ np.asarray(T_tcp_camera, dtype=float)


def load_store(path: Optional[os.PathLike] = None) -> dict:
    file = Path(path or STORE_PATH)
    if not file.is_file():
        return {"cameras": {}}
    with file.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        return {"cameras": {}}
    data.setdefault("cameras", {})
    return data


def save_camera(topic: str, entry: dict, path: Optional[os.PathLike] = None) -> str:
    file = Path(path or STORE_PATH)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = load_store(file)
    data["cameras"][topic] = entry
    temporary=file.with_suffix(file.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(file)
    return str(file)


def apply_saved_extrinsic(observation: dict, tcp_for_side) -> bool:
    """TF 缺失时，用已保存的手工标定填写 T_base_camera。"""
    if observation.get("T_base_camera") is not None:
        return True
    cameras = load_store().get("cameras") or {}
    topic = str(observation.get("topic") or "")
    frame = str(observation.get("camera_frame") or observation.get("input_frame") or "").lstrip("/")
    entry = cameras.get(topic)
    if entry is None:
        named = [
            item
            for item in cameras.values()
            if str(item.get("frame_id") or "") == frame and item.get("T_base_camera")
        ]
        if len(named) == 1:
            entry = named[0]
        elif frame == "camera_frame" and cameras.get("/camera/head_color", {}).get("T_base_camera"):
            # 三路图像的 frame_id 都是 camera_frame，抓取缺省用头部结果。
            entry = cameras["/camera/head_color"]
            topic = "/camera/head_color"
    if not entry:
        return False
    model = entry.get("model")
    if model == "eye_in_hand":
        side = entry.get("arm_side")
        tcp = tcp_for_side(side) if side else None
        if not tcp:
            observation["extrinsic_error"] = f"{topic or '腕部相机'} 已有 T_tcp_camera，但当前没有该臂 TCP"
            return False
        T = compose_base_camera(tcp["xyz"], tcp["xyzw"], entry["T_tcp_camera"])
        source = f"manual:{topic or side}:T_tcp_camera"
    else:
        if not entry.get("T_base_camera"):
            return False
        T = np.asarray(entry["T_base_camera"], dtype=float)
        source = f"manual:{topic or entry.get('frame_id')}"
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    observation["T_base_camera"] = T.tolist()
    observation["base_frame"] = "base_link"
    observation["extrinsic_source"] = source
    observation.pop("extrinsic_error", None)
    return True
