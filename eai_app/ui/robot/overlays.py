from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
import ctypes
import html
import json
import math
import os
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from eai_app.perception.geometry import valid_depth_mask, canonical_color_topic
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import (
    QMatrix4x4,
    QSurfaceFormat,
    QVector3D,
)
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QOpenGLWidget, QVBoxLayout, QWidget, QDoubleSpinBox, QGridLayout, QPushButton
from eai_app.infrastructure.module_ref import module_ref

def _qt_matrix(T: np.ndarray) -> QMatrix4x4:
    flat = np.asarray(T, dtype=np.float64).reshape(4, 4).flatten(order="C").tolist()
    return QMatrix4x4(*flat)

_DEPTH_KEYS = (
    ("/camera/head_depth", "head", "头部", "头部深度，用已标定外参画在 base_link 上。"),
    ("/camera/hand_left_depth", "left", "左腕", "左腕深度。标定误差大，点云挂在左手 Link7，可取消勾选。"),
    ("/camera/hand_right_depth", "right", "右腕", "右腕深度。标定误差大，点云挂在右手 Link7，可取消勾选。"),
)

def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读取 trimesh 写出的点云。返回 Nx3 坐标，以及可选的 Nx3 颜色（0–1）。"""
    numpy_of = {
        "char": "<i1", "int8": "<i1",
        "uchar": "<u1", "uint8": "<u1",
        "short": "<i2", "int16": "<i2",
        "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4",
        "uint": "<u4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4",
        "double": "<f8", "float64": "<f8",
    }
    with open(path, "rb") as handle:
        header_lines: List[str] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("ply 头不完整")
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        fmt = ""
        count = 0
        names: List[str] = []
        kinds: List[str] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
                    names = []
                    kinds = []
            elif in_vertex and parts[0] == "property" and len(parts) >= 3:
                kinds.append(parts[1])
                names.append(parts[-1])
        if count <= 0 or not {"x", "y", "z"} <= set(names):
            raise ValueError("ply 没有顶点")
        if fmt == "ascii":
            xyz = np.empty((count, 3), dtype=np.float32)
            has_rgb = {"red", "green", "blue"} <= set(names)
            rgb = np.empty((count, 3), dtype=np.float32) if has_rgb else None
            for i in range(count):
                vals = handle.readline().decode("ascii", errors="replace").split()
                if len(vals) < len(names):
                    raise ValueError("ply 顶点行太短")
                by_name = {name: vals[j] for j, name in enumerate(names)}
                xyz[i] = (float(by_name["x"]), float(by_name["y"]), float(by_name["z"]))
                if rgb is not None:
                    rgb[i] = (
                        float(by_name["red"]) / 255.0,
                        float(by_name["green"]) / 255.0,
                        float(by_name["blue"]) / 255.0,
                    )
            return xyz, rgb
        if fmt != "binary_little_endian":
            raise ValueError(f"不支持的 ply 格式 {fmt or '?'}")
        dtype = np.dtype([(name, numpy_of[kind]) for name, kind in zip(names, kinds)])
        table = np.frombuffer(handle.read(dtype.itemsize * count), dtype=dtype, count=count)
        if table.shape[0] != count:
            raise ValueError("ply 顶点数据不完整")
        xyz = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
        if {"red", "green", "blue"} <= set(names):
            rgb = np.stack(
                [table["red"], table["green"], table["blue"]], axis=1
            ).astype(np.float32) / 255.0
        else:
            rgb = None
        return xyz, rgb

def _placement_for_color(color_topic: str) -> Tuple[str, np.ndarray, str]:
    """与实时深度同一套外参：返回 parent、相机相对 parent 的变换、说明。"""
    from eai_app.adapters.ros.depth_cloud import _CAMERAS, _load_extrinsics, _placement

    color_topic = canonical_color_topic(color_topic)
    spec = None
    for item in _CAMERAS:
        if color_topic in (item.get("color"), item.get("depth")):
            spec = item
            break
    if spec is None:
        return "", np.eye(4, dtype=np.float64), f"未知相机 {color_topic}"
    cam = (_load_extrinsics() or {}).get(spec["color"]) or {}
    parent, xyz, quat, note = _placement(cam, spec)
    return str(parent or ""), _T_from_xyz_quat(xyz, quat), str(note)

def _T_from_xyz_quat(xyz, quat) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = [float(v) for v in xyz]
    return T

def _rgb_at_depth(color, u: np.ndarray, v: np.ndarray, depth_h: int, depth_w: int) -> Optional[np.ndarray]:
    """按深度像素取彩色图。界面里的彩色帧是 BGR。"""
    img = np.asarray(color)
    if img.ndim != 3 or img.shape[2] < 3 or img.shape[0] < 1 or img.shape[1] < 1:
        return None
    ch, cw = int(img.shape[0]), int(img.shape[1])
    if depth_w <= 1 or depth_h <= 1:
        su = u
        sv = v
    else:
        su = u * np.float32((cw - 1) / float(depth_w - 1))
        sv = v * np.float32((ch - 1) / float(depth_h - 1))
    ui = np.clip(np.rint(su).astype(np.int32), 0, cw - 1)
    vi = np.clip(np.rint(sv).astype(np.int32), 0, ch - 1)
    pix = img[vi, ui, :3].astype(np.float32)
    if float(np.nanmax(pix)) > 1.5:
        pix *= np.float32(1.0 / 255.0)
    return pix[:, ::-1]

def _skeleton_lines(side: str, motors: dict) -> np.ndarray:
    from eai_app.domain.robot.hand_geometry import skeleton_in_link

    points, edges = skeleton_in_link(side, motors)
    rows = []
    for start, end, color in edges:
        a = points.get(start)
        b = points.get(end)
        if a is None or b is None:
            continue
        rgb = (float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0)
        rows.append((float(a[0]), float(a[1]), float(a[2]), rgb[0], rgb[1], rgb[2]))
        rows.append((float(b[0]), float(b[1]), float(b[2]), rgb[0], rgb[1], rgb[2]))
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

def format_wrench_html(items: Sequence[dict], *, enabled: bool, now: Optional[float] = None) -> str:
    """左右臂六维力的两行读数。items 含 label/force/torque/stamp/error。"""
    if not enabled:
        return "<span style='color:#9aa0a6;'>六维力已隐藏</span>"
    now = time.time() if now is None else float(now)
    if not items:
        return "<span style='color:#9aa0a6;'>六维力：未订阅</span>"
    lines = []
    colors = {"左": "#ff79c6", "右": "#79d8ff"}
    for item in items:
        label = str(item.get("label") or item.get("side") or "")
        color = colors.get(label, "#e8eaed")
        err = str(item.get("error") or "")
        force = item.get("force")
        torque = item.get("torque")
        if force is None or torque is None:
            text = html.escape(f"{label}臂  {err or '未收到'}")
            lines.append(f"<span style='color:{color};'>{text}</span>")
            continue
        fx, fy, fz = (float(v) for v in force)
        mx, my, mz = (float(v) for v in torque)
        stamp = float(item.get("stamp") or 0.0)
        age = now - stamp if stamp > 0 else 999.0
        stale = f"  停{age:.0f}s" if age > 2.0 else ""
        extra = f"  {err}" if err else ""
        text = (
            f"{label}臂  "
            f"Fx {fx:+7.2f}  Fy {fy:+7.2f}  Fz {fz:+7.2f} N    "
            f"Mx {mx:+6.3f}  My {my:+6.3f}  Mz {mz:+6.3f} N·m"
            f"{stale}{extra}"
        )
        lines.append(f"<span style='color:{color};'>{html.escape(text)}</span>")
    return "<br/>".join(lines)

_WRENCH_FORCE_M_PER_N = 0.02

_WRENCH_TORQUE_M_PER_NM = 0.25

_WRENCH_MAX_ARROW_M = 0.28

_WRENCH_TOPICS = (
    ("left", "/hal/left_arm_data", "Link7_l", "左"),
    ("right", "/hal/right_arm_data", "Link7_r", "右"),
)

def _scaled_arrow(vec: np.ndarray, scale: float) -> np.ndarray:
    end = np.asarray(vec, dtype=np.float64) * float(scale)
    length = float(np.linalg.norm(end))
    if length > _WRENCH_MAX_ARROW_M and length > 1e-9:
        end = end * (_WRENCH_MAX_ARROW_M / length)
    return end

def _append_arrow(rows: list, end: np.ndarray, rgb: Tuple[float, float, float]) -> None:
    if not np.isfinite(end).all():
        return
    if float(np.linalg.norm(end)) < 1e-5:
        return
    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    rows.append((0.0, 0.0, 0.0, *color))
    rows.append((float(end[0]), float(end[1]), float(end[2]), *color))
    # 箭头尖：沿主方向末端画一小段叉。
    direction = end / max(1e-9, float(np.linalg.norm(end)))
    tip = 0.02
    side = np.cross(direction, np.array((0.0, 0.0, 1.0)))
    if float(np.linalg.norm(side)) < 1e-6:
        side = np.cross(direction, np.array((0.0, 1.0, 0.0)))
    side = side / max(1e-9, float(np.linalg.norm(side))) * tip
    up = np.cross(direction, side)
    tip_base = end - direction * tip
    for offset in (side, -side, up, -up):
        rows.append((float(end[0]), float(end[1]), float(end[2]), *color))
        point = tip_base + offset
        rows.append((float(point[0]), float(point[1]), float(point[2]), *color))

def _wrench_lines(force_xyz, torque_xyz) -> np.ndarray:
    """在 Link7 坐标系下画出力与力矩矢量。力用暖色，力矩用冷色。"""
    force = np.asarray(force_xyz, dtype=np.float64).reshape(3)
    torque = np.asarray(torque_xyz, dtype=np.float64).reshape(3)
    rows: list = []
    _append_arrow(rows, _scaled_arrow(force, _WRENCH_FORCE_M_PER_N), (1.0, 0.35, 0.12))
    _append_arrow(rows, _scaled_arrow(torque, _WRENCH_TORQUE_M_PER_NM), (0.15, 0.85, 0.95))
    # 力的三轴分量，方便读六维里的 Fx/Fy/Fz。
    for index, rgb in enumerate(((1.0, 0.2, 0.15), (0.2, 0.85, 0.25), (0.25, 0.45, 1.0))):
        axis = np.zeros(3, dtype=np.float64)
        axis[index] = float(force[index])
        end = _scaled_arrow(axis, _WRENCH_FORCE_M_PER_N)
        if float(np.linalg.norm(end)) < 1e-5:
            continue
        rows.append((0.0, 0.0, 0.0, *rgb))
        rows.append((float(end[0]), float(end[1]), float(end[2]), *rgb))
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

_OBB_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_OBB_SIGNS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)

_POSE_METHODS = (
    ("pca", "PCA", (1.0, 0.86, 0.15)),
    ("foundationpose", "FoundPose", (0.25, 0.85, 1.0)),
    ("any6d", "Any6D", (1.0, 0.5, 0.15)),
    ("grasp_pregrasp", "预抓取（紫）", (0.9, 0.3, 0.9)),
    ("grasp_approach", "接近（绿）", (0.2, 1.0, 0.3)),
    ("grasp_lift", "抬起（黄）", (1.0, 0.9, 0.1)),
)

def _pose6d_lines(xyz, rotation, corners, extents, box_rgb=(1.0, 0.86, 0.15)) -> np.ndarray:
    """base_link 下的包围盒和 RGB 坐标轴。"""
    center = np.asarray(xyz, dtype=np.float64).reshape(3)
    rotation_m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(center).all() or not np.isfinite(rotation_m).all():
        return np.zeros((0, 6), dtype=np.float32)
    box = None
    if corners is not None:
        arr = np.asarray(corners, dtype=np.float64)
        if arr.shape == (8, 3) and np.isfinite(arr).all():
            box = arr
    half = None
    if extents is not None:
        ext = np.asarray(extents, dtype=np.float64).reshape(3)
        if ext.shape == (3,) and np.isfinite(ext).all():
            half = np.abs(ext) * 0.5
    if box is None and half is not None and float(np.max(half)) > 1e-4:
        box = np.stack(
            [center + rotation_m @ (np.asarray(sign) * half) for sign in _OBB_SIGNS],
            axis=0,
        )
    rows = []
    yellow = tuple(float(v) for v in box_rgb)
    if box is not None:
        for a, b in _OBB_EDGES:
            for index in (a, b):
                point = box[index]
                rows.append((float(point[0]), float(point[1]), float(point[2]), *yellow))
    scale = float(np.max(half)) * 1.1 if half is not None else 0.08
    if not np.isfinite(scale) or scale < 0.04:
        scale = 0.08
    for index, rgb in enumerate(((1.0, 0.25, 0.2), (0.2, 0.9, 0.25), (0.25, 0.45, 1.0))):
        end = center + rotation_m[:, index] * scale
        rows.append((float(center[0]), float(center[1]), float(center[2]), *rgb))
        rows.append((float(end[0]), float(end[1]), float(end[2]), *rgb))
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

def _depth_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 4,
    color_bgr: Optional[np.ndarray] = None,
    fallback: Tuple[float, float, float] = (0.7, 0.7, 0.7),
) -> np.ndarray:
    h, w = depth.shape
    sub = depth[::stride, ::stride]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    valid = valid_depth_mask(sub)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)
    z = sub[valid].astype(np.float32)
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    packed = np.zeros((z.shape[0], 6), dtype=np.float32)
    packed[:, 0] = (u - np.float32(cx)) * z / np.float32(fx)
    packed[:, 1] = (v - np.float32(cy)) * z / np.float32(fy)
    packed[:, 2] = z
    rgb = _rgb_at_depth(color_bgr, u, v, h, w) if color_bgr is not None else None
    if rgb is None or rgb.shape[0] != z.shape[0]:
        packed[:, 3] = np.float32(fallback[0])
        packed[:, 4] = np.float32(fallback[1])
        packed[:, 5] = np.float32(fallback[2])
    else:
        packed[:, 3:6] = rgb
    return packed

def _pack_depth_image(
    cam: dict,
    image: np.ndarray,
    encoding: str,
    color_bgr: Optional[np.ndarray] = None,
    fallback: Tuple[float, float, float] = (0.7, 0.7, 0.7),
    intrinsics=None,
) -> Optional[np.ndarray]:
    from eai_app.adapters.ros.depth_cloud import _intrinsics

    img = np.squeeze(np.asarray(image))
    if img.ndim != 2:
        return None
    enc = (encoding or "").upper()
    if enc in ("16UC1", "MONO16") or img.dtype == np.uint16:
        depth = img.astype(np.float32) / 1000.0
    elif enc == "32FC1" or img.dtype == np.float32:
        depth = img.astype(np.float32)
    else:
        return None
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics if intrinsics is not None else _intrinsics(cam, int(width), int(height))
    return _depth_points(
        depth, fx, fy, cx, cy, stride=4, color_bgr=color_bgr, fallback=fallback
    )

