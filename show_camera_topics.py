#!/usr/bin/python3.10
"""
PyQt5 图形界面：显示 ROS2 中以 /camera 开头的 topic 及图像内容。
深度 topic（名称含 depth）以 3D 点云方式显示，支持鼠标旋转/缩放。
点击图像可进行立体分割（3D 点云聚类 refine）并估计 6D 位姿（位置 + RPY + OBB）。
深度 3D 面板叠加显示左右手臂 TCP 与关节状态（/hal/arm_joint_state、/ry_hand/*/joint_states、/mink_fk/*_tcp_pose）。

用法:
  bash run_in_docker.sh          # ROS 在 Docker 内运行时用此方式（推荐）
  bash run.sh                    # ROS 在宿主机直接运行时用此方式
  python3.10 show_camera_topics.py --prefix /camera

顶部控制区按功能分为标签页：相机 / 回放 / 分割 / 手臂·手。
前置条件：robot-service + base_services 已运行，control_mode=0，手臂/手部已使能。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import signal
import sys
import urllib.error
import urllib.request

if sys.version_info[:2] != (3, 10):
    print(
        f"错误: 当前 Python {sys.version_info.major}.{sys.version_info.minor}，"
        "ROS2 Humble 的 rclpy 仅支持 Python 3.10。\n\n"
        "Conda 环境（如 Python 3.12）无法加载 ROS2 C 扩展，请改用系统 Python:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  python3.10 show_camera_topics.py\n\n"
        "或直接使用: ./run.sh",
        file=sys.stderr,
    )
    sys.exit(1)

# opencv-python 可能污染 Qt 插件搜索路径，须在 import PyQt5 之前清除
import os

os.environ.pop("QT_PLUGIN_PATH", None)

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSignal, QObject, QPoint, QEvent
from PyQt5.QtGui import QCloseEvent, QFont, QImage, QMouseEvent, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import UInt32, UInt8
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

ENABLE_STATE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

IMAGE_TYPES = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/Image": Image,
    "sensor_msgs/msg/CompressedImage": CompressedImage,
    "sensor_msgs/CompressedImage": CompressedImage,
}

DEFAULT_MAX_DEPTH_M = 3.0
DEPTH_STRIDE = 4
DEPTH_DISPLAY_STRIDE = 8
DEPTH_PREVIEW_MAX_W = 480
MAX_GL_DEPTH_POINTS = 8000
SEGMENT_DEPTH_TOL_M = 0.025
SEGMENT_COLOR_TOL = 32.0
SEGMENT_MIN_AREA = 40
SEGMENT_3D_EPS_M = 0.018
SEGMENT_3D_MIN_POINTS = 30
SEGMENT_ROI_RADIUS = 120
SEGMENT_MAX_3D_POINTS = 4000
SEGMENT_COMPUTE_MAX_DIM = 640
SEGMENT_OVERLAY_MAX_DIM = 960

SAM3_BACKEND_GEOMETRY = "geometry"
SAM3_BACKEND_POINT = "sam3_point"
SAM3_BACKEND_TEXT = "sam3_text"
SAM3_SEGMENT_BACKENDS = (
    SAM3_BACKEND_GEOMETRY,
    SAM3_BACKEND_POINT,
    SAM3_BACKEND_TEXT,
)
SAM3_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sam3_segment_worker.py"
)
SAM3_PYTHON_DEFAULT = os.environ.get("SAM3_PYTHON", "python3")
SAM3_MODEL_DEFAULT = os.environ.get("SAM3_MODEL", "sam3.pt")
SAM3_SERVER_URL_DEFAULT = os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:8765")
SAM3_USE_HTTP_DEFAULT = os.environ.get("SAM3_USE_HTTP", "0").strip() in ("1", "true", "yes")
SAM3_TIMEOUT_S = float(os.environ.get("SAM3_TIMEOUT_S", "120"))
MAX_GL_SEGMENT_POINTS = 4000
UI_IMAGE_MIN_INTERVAL_S = 1.0 / 15.0
UI_DEPTH_MIN_INTERVAL_S = 1.0 / 8.0
UI_ROBOT_STATE_MIN_INTERVAL_S = 0.25
UI_TF_LOOKUP_TIMEOUT_S = 0.0
MOVE_TF_LOOKUP_TIMEOUT_S = 1.0
IK_TARGET_FRAME = "base_link"
# mink_fk 将 frame_id 标为 map，但与 ik_node /tele/fk 使用同一 fkSolve，数值即 base_link 系
MINK_FK_FRAME_ALIASES = frozenset({"map", "world"})

ROBOT_ARM_TOPIC = "/hal/arm_joint_state"
ROBOT_SERVICE_SCRIPT = "start_ros_service.sh"
BASE_SERVICES_SCRIPT = "nodes/start_base_services.sh"
BASE_SERVICES_LOG = "/var/psi/log/base_services/latest.log"
ROBOT_FK_JOINTS_TOPIC = "/fk/joint_states"
ROBOT_LEFT_HAND_TOPIC = "/ry_hand/left/joint_states"
ROBOT_RIGHT_HAND_TOPIC = "/ry_hand/right/joint_states"
ROBOT_LEFT_TCP_TOPIC = "/mink_fk/left_tcp_pose"
ROBOT_RIGHT_TCP_TOPIC = "/mink_fk/right_tcp_pose"
ROBOT_TELE_LEFT_TCP_TOPIC = "/tele/fk/left_pose"
ROBOT_TELE_RIGHT_TCP_TOPIC = "/tele/fk/right_pose"
ROBOT_TCP_SOURCE_TOPICS = {
    "left": (ROBOT_TELE_LEFT_TCP_TOPIC, ROBOT_LEFT_TCP_TOPIC),
    "right": (ROBOT_TELE_RIGHT_TCP_TOPIC, ROBOT_RIGHT_TCP_TOPIC),
}
ROBOT_LEFT_HAND_CMD_TOPIC = "/ry_hand/left/set_angles"
BASE_LINK_FRAME = "base_link"
LEFT_IK_TARGET_TOPIC = "/ik/left_target"
RIGHT_IK_TARGET_TOPIC = "/ik/right_target"
MODEL_LEFT_TCP_TOPIC = "/model/tcp/left/target_pose"
MODEL_RIGHT_TCP_TOPIC = "/model/tcp/right/target_pose"
WBC_TARGET_JOINTS_TOPIC = "/wbc/target_joints"
CONTROL_MODE_TOPIC = "/control_mode"
ARM_ENABLE_STATE_TOPIC = "/arm/enable_state"
ARM_SET_ENABLE_SERVICE = "/arm/set_enable"
HAND_ENABLE_STATE_TOPIC = "/hand/enable_state"
HAND_SET_ENABLE_SERVICE = "/hand/set_enable"
REPLAY_START_SERVICE = "/rrd_replay/start_replay"
REPLAY_STOP_SERVICE = "/rrd_replay/stop_replay"
REPLAY_RUNNING_STATE_TOPIC = "/rrd_replay/running_state"
REPLAY_LOOP_COUNT_DEFAULT = 1
REPLAY_LOOP_COUNT_MAX = 99
REPLAY_STATE_FINISHED = 3
REPLAY_STATE_LABELS = {
    0: "空闲",
    1: "回放中",
    2: "等待条件",
    3: "已完成",
    4: "已停止",
}
IDLE_CONTROL_MODE = 99
MODEL_CONTROL_MODE = 0
TELEOP_CONTROL_MODES = {1, 3}
ARM_MOVE_SPEED_MIN_RAD_S = 0.10
ARM_MOVE_SPEED_MAX_RAD_S = 1.00
ARM_MOVE_SPEED_DEFAULT_RAD_S = 0.45
ARM_MOVE_MIN_DURATION_S = 0.8
ARM_MOVE_MAX_DURATION_S = 10.0
ARM_MOVE_IK_HZ = 20.0
ARM_MOVE_JOINT_HZ = 100.0
ARM_MOVE_WARMUP_STEPS = 4
ARM_MOVE_MODE_BURST = 5
ARM_MOVE_GOAL_SAMPLE_TICKS = 2
ARM_MOVE_IK_RELEASE_TICKS = 3

LLM_API_BASE_DEFAULT = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_CHAT_MAX_HISTORY = 24
LLM_CHAT_TIMEOUT_S = 120.0
LLM_CHAT_SYSTEM_PROMPT = (
    "你是机器人相机可视化工具中的对话助手，可回答编程、机器人、视觉感知相关问题。"
    "请用用户使用的语言简洁作答。"
)
LLM_PROVIDER_PRESETS: Dict[str, Tuple[str, str, str]] = {
    "本地 Ollama · Qwen3.5-4B": (
        "http://127.0.0.1:11434/v1",
        "qwen3.5:4b",
        "ollama",
    ),
    "本地 Ollama · Qwen3-VL-4B": (
        "http://127.0.0.1:11434/v1",
        "qwen3-vl:4b",
        "ollama",
    ),
    "百炼兼容 · Qwen-Plus": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        "",
    ),
    "OpenAI 官方": ("https://api.openai.com/v1", "gpt-4o-mini", ""),
    "自定义": ("", "", ""),
}
IK_SYNC_TIMEOUT_S = 3.0
MANUAL_OFFSET_MAX_M = 0.5
MANUAL_OFFSET_STEP_M = 0.01

LEFT_HAND_JOINT_NAMES = [
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
]
LEFT_HAND_CMD_VELOCITY = [2000.0] * 6
LEFT_HAND_CMD_EFFORT = [1200.0] * 6
LEFT_HAND_ANGLE_A_DEFAULT = 0
LEFT_HAND_ANGLE_B_DEFAULT = 45

LEFT_ARM_JOINT_NAMES = [f"joint{i}_l" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"joint{i}_r" for i in range(1, 8)]
WAIST_JOINT_NAMES = ["joint_lift_body", "joint_body_pitch"]


def is_depth_topic(topic: str) -> bool:
    name = topic.lower()
    return "depth" in name or name.endswith("_z")


def is_color_image_topic(topic: str, types: List[str]) -> bool:
    """color 图像 topic（名称含 color）。"""
    if not any(t in IMAGE_TYPES for t in types):
        return False
    return "color" in topic.lower()


def default_viewer_geometry() -> Tuple[int, int, int, int]:
    """根据屏幕可用区域返回窗口 (x, y, width, height)。"""
    app = QApplication.instance()
    if app is not None:
        screen = app.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = max(1480, int(avail.width() * 0.96))
            height = max(900, int(avail.height() * 0.92))
            x = avail.x() + max(0, (avail.width() - width) // 2)
            y = avail.y() + max(0, (avail.height() - height) // 2)
            return x, y, width, height
    return 80, 60, 1680, 960


def default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    fx = fy = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def intrinsics_from_camera_info(msg: CameraInfo) -> tuple[float, float, float, float]:
    k = msg.k
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = DEFAULT_MAX_DEPTH_M,
    stride: int = DEPTH_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    depth_f = depth.astype(np.float32)
    if depth.dtype == np.uint16:
        depth_f /= 1000.0

    depth_sub = depth_f[::stride, ::stride]
    h, w = depth_sub.shape
    u_coords, v_coords = np.meshgrid(
        np.arange(0, w, dtype=np.float32) * stride,
        np.arange(0, h, dtype=np.float32) * stride,
    )

    valid = (depth_sub > 0.01) & (depth_sub < max_depth) & np.isfinite(depth_sub)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    z = depth_sub[valid]
    u = u_coords[valid]
    v = v_coords[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    jet = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = jet.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
    alpha = np.ones((len(points), 1), dtype=np.float32)
    colors = np.hstack([rgb, alpha])
    return points, colors


def subsample_points_with_colors(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx], colors[idx]


def build_depth_preview_image(
    depth: np.ndarray,
    max_preview_w: int = DEPTH_PREVIEW_MAX_W,
) -> np.ndarray:
    preview = depth
    if preview.ndim == 2:
        preview_vis = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        preview_vis = cv2.applyColorMap(preview_vis, cv2.COLORMAP_JET)
    else:
        preview_vis = preview.copy()
    ph, pw = preview_vis.shape[:2]
    if pw > max_preview_w:
        scale = max_preview_w / pw
        preview_vis = cv2.resize(
            preview_vis,
            (int(pw * scale), int(ph * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return preview_vis


@dataclass
class DepthVizResult:
    preview_vis: np.ndarray
    preview_shape: Tuple[int, int]
    preview_intrinsics: Tuple[float, float, float, float]
    points: np.ndarray
    colors: np.ndarray
    point_count: int
    full_shape: Tuple[int, int]


def build_depth_viz_data(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> DepthVizResult:
    h, w = depth.shape[:2]
    preview_vis = build_depth_preview_image(depth)
    preview_shape = preview_vis.shape[:2]
    pfx = fx * (preview_shape[1] / w)
    pfy = fy * (preview_shape[0] / h)
    pcx = cx * (preview_shape[1] / w)
    pcy = cy * (preview_shape[0] / h)
    points, colors = depth_to_point_cloud(
        depth, fx, fy, cx, cy, stride=DEPTH_DISPLAY_STRIDE
    )
    points, colors = subsample_points_with_colors(points, colors, MAX_GL_DEPTH_POINTS)
    return DepthVizResult(
        preview_vis=preview_vis,
        preview_shape=preview_shape,
        preview_intrinsics=(pfx, pfy, pcx, pcy),
        points=points,
        colors=colors,
        point_count=len(points),
        full_shape=(h, w),
    )


def camera_info_candidates(depth_topic: str) -> List[str]:
    base = depth_topic.replace("_depth", "")
    return [
        f"{depth_topic}/camera_info",
        f"{base}_color/camera_info",
        f"{base}/camera_info",
    ]


def depth_value_meters(depth: np.ndarray, u: int, v: int) -> float:
    value = float(depth[v, u])
    if depth.dtype == np.uint16:
        return value / 1000.0
    return value


def pixel_to_camera_xyz(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return x, y, depth_m


def format_color_pixel_info(u: int, v: int, image: np.ndarray) -> str:
    h, w = image.shape[:2]
    if u < 0 or v < 0 or u >= w or v >= h:
        return ""
    if image.ndim == 2:
        val = float(image[v, u])
        return f"像素 ({u}, {v})  |  gray={val:.1f}"
    b, g, r = (int(c) for c in image[v, u][:3])
    return f"像素 ({u}, {v})  |  BGR=({b}, {g}, {r})"


def format_depth_pixel_info(
    u: int,
    v: int,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> str:
    h, w = depth.shape[:2]
    if u < 0 or v < 0 or u >= w or v >= h:
        return ""
    depth_m = depth_value_meters(depth, u, v)
    if not np.isfinite(depth_m) or depth_m <= 0.01:
        return f"像素 ({u}, {v})  |  无效深度"
    x, y, z = pixel_to_camera_xyz(u, v, depth_m, fx, fy, cx, cy)
    return (
        f"像素 ({u}, {v})  |  depth={depth_m:.3f} m  |  "
        f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m"
    )


def depth_to_meters_array(depth: np.ndarray) -> np.ndarray:
    depth_m = depth.astype(np.float32)
    if depth.dtype == np.uint16:
        depth_m /= 1000.0
    return depth_m


def scale_uv_to_shape(
    u: int, v: int, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]
) -> Tuple[int, int]:
    sh, sw = src_shape
    dh, dw = dst_shape
    if (sh, sw) == (dh, dw):
        return u, v
    u_d = int(round(u * dw / max(sw, 1)))
    v_d = int(round(v * dh / max(sh, 1)))
    return max(0, min(dw - 1, u_d)), max(0, min(dh - 1, v_d))


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] == (h, w):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return resized > 0


def find_paired_depth_topic(color_topic: str, available: List[str]) -> Optional[str]:
    depth_topics = [t for t in available if is_depth_topic(t)]
    if not depth_topics:
        return None
    candidates = [
        color_topic.replace("_color", "_depth"),
        color_topic.replace("_color", "_depth_z"),
        color_topic.replace("color", "depth"),
    ]
    for c in candidates:
        if c in depth_topics:
            return c
    stem = color_topic.replace("_color", "").replace("color", "").rstrip("_")
    for t in depth_topics:
        if stem and stem in t:
            return t
    return None


def find_paired_color_topic(depth_topic: str, available: List[str]) -> Optional[str]:
    color_topics = [t for t in available if not is_depth_topic(t)]
    if not color_topics:
        return None
    candidates = [
        depth_topic.replace("_depth", "_color"),
        depth_topic.replace("_depth_z", "_color"),
        depth_topic.replace("depth", "color"),
    ]
    for c in candidates:
        if c in color_topics:
            return c
    stem = depth_topic.replace("_depth", "").replace("_depth_z", "").replace("depth", "").rstrip("_")
    for t in color_topics:
        if stem and stem in t and "color" in t.lower():
            return t
    return None


def segment_by_color_floodfill(
    seed_u: int,
    seed_v: int,
    color_bgr: np.ndarray,
    lo_diff: int = 22,
    hi_diff: int = 22,
) -> np.ndarray:
    h, w = color_bgr.shape[:2]
    work = color_bgr.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(
        work,
        flood_mask,
        (seed_u, seed_v),
        (0, 0, 0),
        (lo_diff, lo_diff, lo_diff),
        (hi_diff, hi_diff, hi_diff),
        flags,
    )
    return flood_mask[1:-1, 1:-1] > 0


def _connected_component_at_seed(valid: np.ndarray, seed_u: int, seed_v: int) -> np.ndarray:
    mask_u8 = valid.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8)
    seed_label = labels[seed_v, seed_u]
    if seed_label == 0:
        return np.zeros(valid.shape, dtype=bool)
    return labels == seed_label


def crop_roi(
    u: int, v: int, height: int, width: int, radius: int = SEGMENT_ROI_RADIUS
) -> Tuple[int, int, int, int]:
    u0 = max(0, u - radius)
    u1 = min(width, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(height, v + radius + 1)
    return u0, v0, u1, v1


def subsample_mask(mask: np.ndarray, max_points: int = SEGMENT_MAX_3D_POINTS) -> np.ndarray:
    count = int(mask.sum())
    if count <= max_points:
        return mask
    ys, xs = np.where(mask)
    rng = np.random.default_rng(0)
    pick = rng.choice(count, max_points, replace=False)
    slim = np.zeros_like(mask)
    slim[ys[pick], xs[pick]] = True
    return slim


def subsample_points(points: np.ndarray, max_points: int = MAX_GL_SEGMENT_POINTS) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(1)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def downsample_image_for_compute(
    image: np.ndarray,
    max_dim: int = SEGMENT_COMPUTE_MAX_DIM,
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 0.999:
        return image, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp), scale


def scale_intrinsics_by(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
) -> Tuple[float, float, float, float]:
    if scale >= 0.999:
        return fx, fy, cx, cy
    return fx * scale, fy * scale, cx * scale, cy * scale


def compute_contact_point_3d(
    seed_u: int,
    seed_v: int,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    points_3d: Optional[np.ndarray] = None,
    patch_radius: int = 2,
) -> Optional[Tuple[float, float, float]]:
    """点击像素处的 3D 接触点（相机系），作为左臂 TCP 目标位置。"""
    depth_m = depth_to_meters_array(depth)
    h, w = depth_m.shape
    su = max(0, min(w - 1, int(seed_u)))
    sv = max(0, min(h - 1, int(seed_v)))

    u0 = max(0, su - patch_radius)
    u1 = min(w, su + patch_radius + 1)
    v0 = max(0, sv - patch_radius)
    v1 = min(h, sv + patch_radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[(patch > 0.01) & np.isfinite(patch)]
    if len(valid) >= 1:
        z = float(np.median(valid))
        x = (su - cx) * z / fx
        y = (sv - cy) * z / fy
        return (float(x), float(y), float(z))

    if points_3d is not None and len(points_3d) > 0:
        zs = np.maximum(points_3d[:, 2], 1e-3)
        us = fx * points_3d[:, 0] / zs + cx
        vs = fy * points_3d[:, 1] / zs + cy
        dist = (us - su) ** 2 + (vs - sv) ** 2
        nearest = points_3d[int(np.argmin(dist))]
        return (float(nearest[0]), float(nearest[1]), float(nearest[2]))
    return None


def upscale_segment_result(result: Object6DPoseResult, scale: float) -> Object6DPoseResult:
    if scale >= 0.999:
        return result
    inv = 1.0 / scale
    full_h = max(1, int(round(result.mask.shape[0] * inv)))
    full_w = max(1, int(round(result.mask.shape[1] * inv)))
    result.mask = resize_mask_to_shape(result.mask, (full_h, full_w))
    cu, cv = result.centroid_uv
    result.centroid_uv = (cu * inv, cv * inv)
    cu2, cv2 = result.contact_uv
    result.contact_uv = (cu2 * inv, cv2 * inv)
    return result


def build_pose_overlay_image(
    color_bgr: Optional[np.ndarray],
    result: Object6DPoseResult,
    intrinsics: Tuple[float, float, float, float],
    max_dim: int = SEGMENT_OVERLAY_MAX_DIM,
) -> Optional[np.ndarray]:
    if color_bgr is None:
        return None
    display, scale = downsample_image_for_compute(color_bgr, max_dim)
    fx, fy, cx, cy = scale_intrinsics_by(*intrinsics, scale)
    mask = resize_mask_to_shape(result.mask, display.shape[:2])
    cu = result.centroid_uv[0] * scale
    cv = result.centroid_uv[1] * scale
    contact_u = result.contact_uv[0] * scale
    contact_v = result.contact_uv[1] * scale
    return apply_segment_overlay(
        display,
        mask,
        (cu, cv),
        obb_corners=result.obb_corners,
        intrinsics=(fx, fy, cx, cy),
        contact_uv=(contact_u, contact_v),
    )


def _segment_object_in_crop(
    seed_u: int,
    seed_v: int,
    depth_m: np.ndarray,
    color_bgr: Optional[np.ndarray],
    depth_tol: float,
    color_tol: float,
    min_area: int,
) -> Tuple[np.ndarray, str]:
    h, w = depth_m.shape[:2]
    seed_u = max(0, min(w - 1, seed_u))
    seed_v = max(0, min(h - 1, seed_v))
    seed_d = float(depth_m[seed_v, seed_u])

    if not np.isfinite(seed_d) or seed_d <= 0.01:
        if color_bgr is not None:
            mask = segment_by_color_floodfill(seed_u, seed_v, color_bgr)
            return mask, "color"
        return np.zeros((h, w), dtype=bool), "none"

    depth_diff = np.abs(depth_m - seed_d)
    valid = (depth_m > 0.01) & np.isfinite(depth_m) & (depth_diff <= depth_tol)
    method = "depth"

    if color_bgr is not None:
        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        seed_lab = lab[seed_v, seed_u]
        color_diff = np.linalg.norm(lab - seed_lab, axis=2)
        valid = valid & (color_diff <= color_tol)
        method = "depth+color"

    mask = _connected_component_at_seed(valid, seed_u, seed_v)
    if not mask.any():
        if color_bgr is not None:
            mask = segment_by_color_floodfill(seed_u, seed_v, color_bgr)
            return mask, "color"
        return mask, method

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0

    if int(mask.sum()) < min_area:
        valid_tight = (depth_m > 0.01) & np.isfinite(depth_m) & (depth_diff <= depth_tol * 0.6)
        if color_bgr is not None:
            valid_tight = valid_tight & (color_diff <= color_tol * 0.75)
        mask = _connected_component_at_seed(valid_tight, seed_u, seed_v)

    return mask, method


def segment_object_at_click(
    seed_u: int,
    seed_v: int,
    depth: np.ndarray,
    color_bgr: Optional[np.ndarray] = None,
    depth_tol: float = SEGMENT_DEPTH_TOL_M,
    color_tol: float = SEGMENT_COLOR_TOL,
    min_area: int = SEGMENT_MIN_AREA,
    roi_radius: int = SEGMENT_ROI_RADIUS,
) -> Tuple[np.ndarray, str]:
    """基于深度连续性 + 可选颜色约束分割点击处物体，返回 (mask, method)。"""
    h, w = depth.shape[:2]
    seed_u = max(0, min(w - 1, seed_u))
    seed_v = max(0, min(h - 1, seed_v))
    u0, v0, u1, v1 = crop_roi(seed_u, seed_v, h, w, roi_radius)

    depth_crop = depth[v0:v1, u0:u1]
    depth_m = depth_to_meters_array(depth_crop)
    local_u = seed_u - u0
    local_v = seed_v - v0

    color_crop: Optional[np.ndarray] = None
    if color_bgr is not None:
        ch, cw = color_bgr.shape[:2]
        if (ch, cw) != (h, w):
            cu0, cv0 = scale_uv_to_shape(u0, v0, (h, w), (ch, cw))
            cu1, cv1 = scale_uv_to_shape(u1 - 1, v1 - 1, (h, w), (ch, cw))
            cu1 = min(cw, cu1 + 1)
            cv1 = min(ch, cv1 + 1)
            color_crop = color_bgr[cv0:cv1, cu0:cu1]
            local_u, local_v = scale_uv_to_shape(seed_u, seed_v, (h, w), color_crop.shape[:2])
        else:
            color_crop = color_bgr[v0:v1, u0:u1]

    mask_crop, method = _segment_object_in_crop(
        local_u, local_v, depth_m, color_crop, depth_tol, color_tol, min_area
    )

    mask_full = np.zeros((h, w), dtype=bool)
    mask_full[v0:v1, u0:u1] = mask_crop
    return mask_full, method


@dataclass
class SegmentSettings:
    backend: str = SAM3_BACKEND_GEOMETRY
    sam3_text: str = ""
    sam3_use_http: bool = SAM3_USE_HTTP_DEFAULT
    sam3_server_url: str = SAM3_SERVER_URL_DEFAULT
    sam3_python: str = SAM3_PYTHON_DEFAULT
    sam3_model: str = SAM3_MODEL_DEFAULT


_segment_settings = SegmentSettings()
_segment_settings_lock = threading.Lock()


def get_segment_settings() -> SegmentSettings:
    with _segment_settings_lock:
        return SegmentSettings(
            backend=_segment_settings.backend,
            sam3_text=_segment_settings.sam3_text,
            sam3_use_http=_segment_settings.sam3_use_http,
            sam3_server_url=_segment_settings.sam3_server_url,
            sam3_python=_segment_settings.sam3_python,
            sam3_model=_segment_settings.sam3_model,
        )


def set_segment_settings(**kwargs) -> None:
    with _segment_settings_lock:
        for key, value in kwargs.items():
            if hasattr(_segment_settings, key):
                setattr(_segment_settings, key, value)


def _encode_image_bgr_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("图像 JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_sam3_mask_payload(payload: Dict[str, object]) -> np.ndarray:
    h = int(payload["h"])  # type: ignore[arg-type]
    w = int(payload["w"])  # type: ignore[arg-type]
    packed = np.frombuffer(
        base64.b64decode(str(payload["packed_b64"])), dtype=np.uint8
    )
    flat = np.unpackbits(packed)[: h * w]
    return flat.reshape(h, w).astype(bool)


def _sam3_build_request_payload(
    image_bgr: np.ndarray,
    u: int,
    v: int,
    text: Optional[str],
    model_path: str,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "image_b64": _encode_image_bgr_b64(image_bgr),
        "model": model_path,
    }
    if text and text.strip():
        payload["text"] = text.strip()
    else:
        payload["u"] = int(u)
        payload["v"] = int(v)
    return payload


def _sam3_segment_via_http(
    payload: Dict[str, object],
    server_url: str,
    timeout_s: float,
) -> Tuple[np.ndarray, str]:
    url = server_url.rstrip("/") + "/segment"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SAM3 HTTP {exc.code}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"SAM3 服务不可达 ({server_url})，请先运行: "
            f"python3 sam3_segment_worker.py --serve --model sam3.pt"
        ) from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    mask_payload = body.get("mask")
    if not isinstance(mask_payload, dict):
        raise RuntimeError(f"SAM3 响应缺少 mask: {body}")
    return _decode_sam3_mask_payload(mask_payload), str(body.get("method") or "sam3")


def _sam3_segment_via_subprocess(
    payload: Dict[str, object],
    python_exe: str,
    timeout_s: float,
) -> Tuple[np.ndarray, str]:
    if not os.path.isfile(SAM3_WORKER_SCRIPT):
        raise RuntimeError(f"未找到 worker: {SAM3_WORKER_SCRIPT}")
    proc = subprocess.run(
        [python_exe, SAM3_WORKER_SCRIPT, "--once"],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or stdout or f"SAM3 worker 退出码 {proc.returncode}")
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SAM3 worker 输出非 JSON: {stdout[:200]}") from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    mask_payload = body.get("mask")
    if not isinstance(mask_payload, dict):
        raise RuntimeError(f"SAM3 响应缺少 mask: {body}")
    return _decode_sam3_mask_payload(mask_payload), str(body.get("method") or "sam3")


def run_sam3_segmentation(
    image_bgr: np.ndarray,
    u: int,
    v: int,
    text: Optional[str] = None,
    settings: Optional[SegmentSettings] = None,
) -> Tuple[np.ndarray, str]:
    cfg = settings or get_segment_settings()
    payload = _sam3_build_request_payload(image_bgr, u, v, text, cfg.sam3_model)
    if cfg.sam3_use_http:
        return _sam3_segment_via_http(payload, cfg.sam3_server_url, SAM3_TIMEOUT_S)
    return _sam3_segment_via_subprocess(payload, cfg.sam3_python, SAM3_TIMEOUT_S)


def check_sam3_server_health(server_url: str, timeout_s: float = 2.0) -> bool:
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def segment_object_at_click_with_backend(
    seed_u: int,
    seed_v: int,
    depth: np.ndarray,
    color_bgr: Optional[np.ndarray] = None,
    settings: Optional[SegmentSettings] = None,
    depth_tol: float = SEGMENT_DEPTH_TOL_M,
    color_tol: float = SEGMENT_COLOR_TOL,
    min_area: int = SEGMENT_MIN_AREA,
    roi_radius: int = SEGMENT_ROI_RADIUS,
) -> Tuple[np.ndarray, str]:
    cfg = settings or get_segment_settings()
    if cfg.backend == SAM3_BACKEND_GEOMETRY:
        return segment_object_at_click(
            seed_u,
            seed_v,
            depth,
            color_bgr=color_bgr,
            depth_tol=depth_tol,
            color_tol=color_tol,
            min_area=min_area,
            roi_radius=roi_radius,
        )
    if color_bgr is None:
        raise RuntimeError("SAM3 分割需要彩色图（请同时订阅 color topic）")
    text = cfg.sam3_text.strip() if cfg.backend == SAM3_BACKEND_TEXT else None
    mask, method = run_sam3_segmentation(
        color_bgr, seed_u, seed_v, text=text, settings=cfg
    )
    if mask.shape[:2] != depth.shape[:2]:
        mask = resize_mask_to_shape(mask, depth.shape[:2])
    if int(mask.sum()) < min_area:
        raise RuntimeError(f"SAM3 mask 过小 ({int(mask.sum())} px)")
    return mask, method


def mask_to_points_3d(
    mask: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    zs = depth_m[ys, xs]
    valid = (zs > 0.01) & np.isfinite(zs)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points = np.stack([x, y, zs], axis=-1).astype(np.float32)
    uv = np.stack([xs, ys], axis=-1).astype(np.float32)
    return points, uv


def uv_to_mask(uv: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if len(uv) == 0:
        return mask
    us = np.clip(uv[:, 0].astype(np.int32), 0, w - 1)
    vs = np.clip(uv[:, 1].astype(np.int32), 0, h - 1)
    mask[vs, us] = True
    return mask


def region_grow_3d(
    points: np.ndarray,
    uv: np.ndarray,
    seed_u: float,
    seed_v: float,
    eps: float = SEGMENT_3D_EPS_M,
    max_cluster: int = SEGMENT_MAX_3D_POINTS,
) -> np.ndarray:
    """3D 区域生长，在体素网格上做连通聚类。"""
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n > max_cluster:
        rng = np.random.default_rng(2)
        keep = rng.choice(n, max_cluster, replace=False)
        seed_dist = (uv[:, 0] - seed_u) ** 2 + (uv[:, 1] - seed_v) ** 2
        seed_i_full = int(np.argmin(seed_dist))
        if seed_i_full not in keep:
            keep[0] = seed_i_full
        points = points[keep]
        uv = uv[keep]
        n = len(points)

    seed_i = int(np.argmin((uv[:, 0] - seed_u) ** 2 + (uv[:, 1] - seed_v) ** 2))
    inv_cell = 1.0 / max(eps, 1e-4)
    keys = np.round(points * inv_cell).astype(np.int32)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        grid.setdefault(key, []).append(i)

    visited = np.zeros(n, dtype=bool)
    queue: deque[int] = deque([seed_i])
    visited[seed_i] = True
    cluster: List[int] = []
    eps_sq = eps * eps

    while queue:
        i = queue.popleft()
        cluster.append(i)
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        if visited[j]:
                            continue
                        diff = points[j] - points[i]
                        if float(np.dot(diff, diff)) <= eps_sq:
                            visited[j] = True
                            queue.append(j)
    return np.array(cluster, dtype=np.int64)


def remove_statistical_outliers(points: np.ndarray, k_mad: float = 2.5) -> np.ndarray:
    if len(points) < 8:
        return np.ones(len(points), dtype=bool)
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-6
    return dist <= med + k_mad * mad * 1.4826


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """相机坐标系下 roll(X), pitch(Y), yaw(Z)，单位弧度。"""
    sy = math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy >= 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def quat_xyzw_to_rpy_deg(
    quat_xyzw: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    R = quat_to_rotation_matrix(*quat_xyzw)
    roll, pitch, yaw = rotation_matrix_to_euler_xyz(R)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def format_xyz_rpy_line(
    prefix: str,
    xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
) -> str:
    roll, pitch, yaw = quat_xyzw_to_rpy_deg(quat_xyzw)
    return (
        f"{prefix} XYZ=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m  "
        f"RPY=({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
    )


def format_left_hand_state_line(state: RobotStateSnapshot) -> str:
    if not state.left_hand:
        return "左手: (无数据)"
    vals = [v for _, v in state.left_hand]
    avg = sum(vals) / len(vals)
    return f"左手: avg={avg:.3f} ({len(vals)} 关节)"


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + float(R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + float(R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + float(R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-8
    return qx / norm, qy / norm, qz / norm, qw / norm


def obb_corners(center: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx, sy, sz], dtype=np.float32) * half_extents
                corners.append(center + rotation @ local)
    return np.stack(corners, axis=0).astype(np.float32)


def obb_wireframe_edges(corners: np.ndarray) -> np.ndarray:
    edge_pairs = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = []
    for a, b in edge_pairs:
        lines.append(corners[a])
        lines.append(corners[b])
    return np.stack(lines, axis=0).astype(np.float32)


def pose_axes_lines(center: np.ndarray, rotation: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    origin = center.astype(np.float32)
    lines = []
    colors = []
    axis_colors = [
        (1.0, 0.2, 0.2, 1.0),
        (0.2, 1.0, 0.2, 1.0),
        (0.2, 0.4, 1.0, 1.0),
    ]
    for i in range(3):
        end = origin + rotation[:, i] * scale
        lines.extend([origin, end])
        colors.extend([axis_colors[i], axis_colors[i]])
    return np.stack(lines, axis=0).astype(np.float32), np.stack(colors, axis=0).astype(np.float32)


def estimate_6d_pose_pca(points: np.ndarray) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float], Tuple[float, float, float, float]]
]:
    if len(points) < SEGMENT_3D_MIN_POINTS:
        return None
    centroid = np.median(points, axis=0).astype(np.float32)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    rotation = evecs[:, order].astype(np.float32)
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0

    local = centered @ rotation
    ext_min = np.percentile(local, 5, axis=0)
    ext_max = np.percentile(local, 95, axis=0)
    half_extents = ((ext_max - ext_min) * 0.5).astype(np.float32)
    half_extents = np.maximum(half_extents, 0.005)
    obb_center = centroid + rotation @ ((ext_max + ext_min) * 0.5)

    roll, pitch, yaw = rotation_matrix_to_euler_xyz(rotation)
    quat = rotation_matrix_to_quaternion(rotation)
    return obb_center, rotation, half_extents, (roll, pitch, yaw), quat


def refine_stereo_segment(
    mask: np.ndarray,
    depth: np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D 掩码 → 3D 点云聚类 refine，返回 (refined_mask, points_3d)。"""
    mask = subsample_mask(mask)
    if not mask.any():
        return mask, np.empty((0, 3), dtype=np.float32)

    ys, xs = np.where(mask)
    pad = 10
    h, w = mask.shape
    v0 = max(0, int(ys.min()) - pad)
    v1 = min(h, int(ys.max()) + pad + 1)
    u0 = max(0, int(xs.min()) - pad)
    u1 = min(w, int(xs.max()) + pad + 1)

    mask_crop = mask[v0:v1, u0:u1]
    depth_crop = depth[v0:v1, u0:u1]
    depth_m = depth_to_meters_array(depth_crop)
    local_seed_u = seed_u - u0
    local_seed_v = seed_v - v0
    local_cx = cx - u0
    local_cy = cy - v0

    points, uv = mask_to_points_3d(mask_crop, depth_m, fx, fy, local_cx, local_cy)
    if len(points) == 0:
        return mask, points

    cluster_idx = region_grow_3d(points, uv, float(local_seed_u), float(local_seed_v))
    if len(cluster_idx) < SEGMENT_3D_MIN_POINTS // 2:
        cluster_idx = np.arange(len(points), dtype=np.int64)

    inlier = remove_statistical_outliers(points[cluster_idx])
    cluster_idx = cluster_idx[inlier]
    if len(cluster_idx) < 5:
        return mask, points

    refined_uv = uv[cluster_idx]
    refined_points = points[cluster_idx]
    refined_mask_crop = uv_to_mask(refined_uv, mask_crop.shape)
    refined_mask = np.zeros((h, w), dtype=bool)
    refined_mask[v0:v1, u0:u1] = refined_mask_crop
    return refined_mask, refined_points


def run_stereo_pose_pipeline(
    depth: np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_bgr: Optional[np.ndarray] = None,
) -> Tuple[Optional[Object6DPoseResult], str]:
    try:
        mask, method = segment_object_at_click_with_backend(
            seed_u, seed_v, depth, color_bgr=color_bgr
        )
    except Exception as exc:
        return None, f"点击 ({seed_u}, {seed_v})  |  分割失败: {exc}"
    if not mask.any():
        return None, f"点击 ({seed_u}, {seed_v})  |  立体分割失败（无有效区域）"
    result = compute_stereo_segment_6d_pose(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy, method=method
    )
    if result is None:
        return None, f"点击 ({seed_u}, {seed_v})  |  立体分割失败（深度无效或点云过少）"
    return result, format_pose_6d_info(result, seed_u, seed_v)


def run_stereo_pose_pipeline_resampled(
    depth: np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_bgr: Optional[np.ndarray] = None,
) -> Tuple[Optional[Object6DPoseResult], str]:
    """降采样后在后台线程运行分割，减轻卡顿。"""
    depth_s, scale = downsample_image_for_compute(depth)
    color_s = None
    if color_bgr is not None:
        color_s, _ = downsample_image_for_compute(color_bgr)
    su = max(0, min(depth_s.shape[1] - 1, int(round(seed_u * scale))))
    sv = max(0, min(depth_s.shape[0] - 1, int(round(seed_v * scale))))
    fx_s, fy_s, cx_s, cy_s = scale_intrinsics_by(fx, fy, cx, cy, scale)
    result, info = run_stereo_pose_pipeline(
        depth_s, su, sv, fx_s, fy_s, cx_s, cy_s, color_bgr=color_s
    )
    if result is None:
        return None, info
    if scale < 0.999:
        result = upscale_segment_result(result, scale)
        info = format_pose_6d_info(result, seed_u, seed_v)
    return result, info


@dataclass
class Object6DPoseResult:
    mask: np.ndarray
    points_3d: np.ndarray
    centroid_uv: Tuple[float, float]
    contact_uv: Tuple[float, float]
    contact_xyz: Tuple[float, float, float]
    position_xyz: Tuple[float, float, float]
    euler_rpy_rad: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    rotation_matrix: np.ndarray
    obb_center: Tuple[float, float, float]
    obb_extents: Tuple[float, float, float]
    obb_corners: np.ndarray
    depth_m: float
    pixel_count: int
    point_count: int
    method: str


@dataclass
class SegmentPoseTarget:
    """相机系下分割得到的 6D 位姿目标（接触 TCP → 左臂 IK）。"""

    camera_frame: str
    position_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    source_topic: str
    label: str = ""
    contact_uv: Tuple[float, float] = (0.0, 0.0)
    obb_center_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_pose_result(
        cls,
        result: Object6DPoseResult,
        camera_frame: str,
        source_topic: str,
    ) -> SegmentPoseTarget:
        return cls(
            camera_frame=camera_frame,
            position_xyz=result.contact_xyz,
            quaternion_xyzw=result.quaternion_xyzw,
            source_topic=source_topic,
            contact_uv=result.contact_uv,
            obb_center_xyz=result.position_xyz,
            label=(
                f"{source_topic.split('/')[-1]} 接触TCP "
                f"({result.point_count} pts)"
            ),
        )


@dataclass
class ResolvedArmMoveGoal:
    position_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    label: str


def make_manual_offset_spinbox() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-MANUAL_OFFSET_MAX_M, MANUAL_OFFSET_MAX_M)
    spin.setSingleStep(MANUAL_OFFSET_STEP_M)
    spin.setDecimals(3)
    spin.setSuffix("m")
    spin.setFixedWidth(76)
    return spin


def compute_relative_move_goal(
    current_xyz: Tuple[float, float, float],
    current_quat: Tuple[float, float, float, float],
    dx: float,
    dy: float,
    dz: float,
) -> Optional[ResolvedArmMoveGoal]:
    if abs(dx) + abs(dy) + abs(dz) < 1e-6:
        return None
    goal_xyz = (current_xyz[0] + dx, current_xyz[1] + dy, current_xyz[2] + dz)
    return ResolvedArmMoveGoal(
        position_xyz=goal_xyz,
        quaternion_xyzw=current_quat,
        label=f"相对偏移 Δ({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m",
    )


def compute_stereo_segment_6d_pose(
    mask: np.ndarray,
    depth: np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    method: str = "",
) -> Optional[Object6DPoseResult]:
    refined_mask, points = refine_stereo_segment(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy
    )
    pose = estimate_6d_pose_pca(points)
    if pose is None:
        return None

    obb_center, rotation, half_extents, rpy, quat = pose
    corners = obb_corners(obb_center, rotation, half_extents)
    depth_m = float(obb_center[2])

    us = (points[:, 0] * fx / points[:, 2] + cx) if len(points) else np.array([seed_u])
    vs = (points[:, 1] * fy / points[:, 2] + cy) if len(points) else np.array([seed_v])
    centroid_uv = (float(np.median(us)), float(np.median(vs)))
    contact_xyz = compute_contact_point_3d(
        seed_u, seed_v, depth, fx, fy, cx, cy, points_3d=points
    )
    if contact_xyz is None:
        contact_xyz = (float(obb_center[0]), float(obb_center[1]), float(obb_center[2]))

    return Object6DPoseResult(
        mask=refined_mask,
        points_3d=points,
        centroid_uv=centroid_uv,
        contact_uv=(float(seed_u), float(seed_v)),
        contact_xyz=contact_xyz,
        position_xyz=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=rotation,
        obb_center=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        obb_extents=(float(half_extents[0] * 2), float(half_extents[1] * 2), float(half_extents[2] * 2)),
        obb_corners=corners,
        depth_m=depth_m,
        pixel_count=int(refined_mask.sum()),
        point_count=len(points),
        method=method,
    )


def compute_object_3d_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    seed_u: int = 0,
    seed_v: int = 0,
    method: str = "",
) -> Optional[Object6DPoseResult]:
    return compute_stereo_segment_6d_pose(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy, method=method
    )


def format_pose_6d_info(result: Object6DPoseResult, seed_u: int, seed_v: int) -> str:
    cx, cy, cz = result.contact_xyz
    ox, oy, oz = result.position_xyz
    roll, pitch, yaw = result.euler_rpy_rad
    qx, qy, qz, qw = result.quaternion_xyzw
    dx, dy, dz = result.obb_extents
    r_deg = math.degrees(roll)
    p_deg = math.degrees(pitch)
    y_deg = math.degrees(yaw)
    method = result.method or "stereo3d"
    return (
        f"点击 ({seed_u}, {seed_v})  |  立体分割 {result.point_count} pts / {result.pixel_count} px ({method})\n"
        f"接触 TCP XYZ=({cx:.3f}, {cy:.3f}, {cz:.3f}) m  |  "
        f"OBB中心=({ox:.3f}, {oy:.3f}, {oz:.3f}) m\n"
        f"RPY=({r_deg:.1f}°, {p_deg:.1f}°, {y_deg:.1f}°)  |  "
        f"OBB=({dx:.3f}, {dy:.3f}, {dz:.3f}) m  |  "
        f"四元数 xyzw=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
    )


def format_segment_info(result: Object6DPoseResult, seed_u: int, seed_v: int) -> str:
    return format_pose_6d_info(result, seed_u, seed_v)


def project_points_to_uv(
    points_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    z = np.maximum(points_3d[:, 2], 1e-3)
    u = fx * points_3d[:, 0] / z + cx
    v = fy * points_3d[:, 1] / z + cy
    return np.stack([u, v], axis=-1)


def draw_obb_on_image(
    image: np.ndarray,
    corners_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    if len(corners_3d) == 0:
        return
    uv = project_points_to_uv(corners_3d, fx, fy, cx, cy).astype(np.int32)
    edge_pairs = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    h, w = image.shape[:2]
    for a, b in edge_pairs:
        if corners_3d[a, 2] <= 0.01 or corners_3d[b, 2] <= 0.01:
            continue
        pa = (int(np.clip(uv[a, 0], 0, w - 1)), int(np.clip(uv[a, 1], 0, h - 1)))
        pb = (int(np.clip(uv[b, 0], 0, w - 1)), int(np.clip(uv[b, 1], 0, h - 1)))
        cv2.line(image, pa, pb, (0, 220, 255), 2, cv2.LINE_AA)


def apply_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    centroid_uv: Optional[Tuple[float, float]] = None,
    obb_corners: Optional[np.ndarray] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    contact_uv: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    if mask.shape[:2] != display.shape[:2]:
        mask = resize_mask_to_shape(mask, display.shape[:2])

    if not mask.any():
        return display

    ys, xs = np.where(mask)
    pad = 12
    h, w = display.shape[:2]
    v0 = max(0, int(ys.min()) - pad)
    v1 = min(h, int(ys.max()) + pad + 1)
    u0 = max(0, int(xs.min()) - pad)
    u1 = min(w, int(xs.max()) + pad + 1)

    roi = display[v0:v1, u0:u1].copy()
    roi_mask = mask[v0:v1, u0:u1]
    overlay = roi.copy()
    green = np.array([0, 220, 80], dtype=np.uint8)
    overlay[roi_mask] = (overlay[roi_mask].astype(np.float32) * 0.45 + green * 0.55).astype(np.uint8)
    roi = cv2.addWeighted(roi, 0.35, overlay, 0.65, 0)
    display[v0:v1, u0:u1] = roi

    if obb_corners is not None and intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        draw_obb_on_image(display, obb_corners, fx, fy, cx, cy)

    if centroid_uv is not None:
        cu, cv_pt = int(round(centroid_uv[0])), int(round(centroid_uv[1]))
        cv2.drawMarker(
            display, (cu, cv_pt), (0, 255, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA
        )
        cv2.circle(display, (cu, cv_pt), 6, (0, 255, 255), 1, cv2.LINE_AA)

    if contact_uv is not None:
        tu, tv = int(round(contact_uv[0])), int(round(contact_uv[1]))
        cv2.drawMarker(
            display, (tu, tv), (0, 128, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA
        )
        cv2.circle(display, (tu, tv), 8, (0, 128, 255), 2, cv2.LINE_AA)
    return display


def cv2_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    display = cv_image
    if len(display.shape) == 2:
        display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        display = cv2.applyColorMap(display, cv2.COLORMAP_JET)

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


@dataclass
class TcpPoseSnapshot:
    frame_id: str
    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]
    valid: bool = True


@dataclass
class RobotStateSnapshot:
    left_arm: List[Tuple[str, float]]
    right_arm: List[Tuple[str, float]]
    left_hand: List[Tuple[str, float]]
    right_hand: List[Tuple[str, float]]
    waist: List[Tuple[str, float]]
    left_tcp: Optional[TcpPoseSnapshot]
    right_tcp: Optional[TcpPoseSnapshot]
    updated_at: float = 0.0


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(
    xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    qx, qy, qz, qw = quat_xyzw
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rotation_matrix(qx, qy, qz, qw)
    mat[:3, 3] = xyz
    return mat


def matrix_to_pose(mat: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    xyz = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    R = mat[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-8
    return xyz, (qx / norm, qy / norm, qz / norm, qw / norm)


def transform_matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    return pose_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def transform_tcp_to_camera(
    tcp: Optional[TcpPoseSnapshot],
    camera_frame: str,
    tf_buffer: Optional[Buffer],
    timeout_s: float = UI_TF_LOOKUP_TIMEOUT_S,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if tcp is None or not tcp.valid or not camera_frame or tf_buffer is None:
        return None
    if normalize_frame_id(tcp.frame_id) == normalize_frame_id(camera_frame):
        R = quat_to_rotation_matrix(*tcp.quat_xyzw)
        return np.array(tcp.xyz, dtype=np.float32), R.astype(np.float32)
    try:
        tf_msg = tf_buffer.lookup_transform(
            camera_frame,
            tcp.frame_id,
            Time(),
            timeout=Duration(seconds=timeout_s),
        )
        T_cam_src = transform_matrix_from_tf(tf_msg)
        T_src_tcp = pose_to_matrix(tcp.xyz, tcp.quat_xyzw)
        T_cam_tcp = T_cam_src @ T_src_tcp
        xyz, quat = matrix_to_pose(T_cam_tcp)
        R = quat_to_rotation_matrix(*quat)
        return np.array(xyz, dtype=np.float32), R.astype(np.float32)
    except Exception:
        return None


def quaternion_slerp(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    qa = np.array(q0, dtype=np.float64)
    qb = np.array(q1, dtype=np.float64)
    qa /= np.linalg.norm(qa) + 1e-12
    qb /= np.linalg.norm(qb) + 1e-12
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        out = qa + t * (qb - qa)
        out /= np.linalg.norm(out) + 1e-12
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    out = s0 * qa + s1 * qb
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def normalize_frame_id(frame: str) -> str:
    return (frame or "").strip().strip("/")


def transform_pose_to_base(
    xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    source_frame: str,
    tf_buffer: Optional[Buffer],
    base_frame: str = BASE_LINK_FRAME,
    timeout_s: float = MOVE_TF_LOOKUP_TIMEOUT_S,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    src = normalize_frame_id(source_frame)
    base = normalize_frame_id(base_frame)
    if not src:
        return None
    if src == base:
        return xyz, quat_xyzw
    if tf_buffer is None:
        return None
    try:
        tf_msg = tf_buffer.lookup_transform(
            base_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=timeout_s),
        )
        T_base_src = transform_matrix_from_tf(tf_msg)
        T_src_obj = pose_to_matrix(xyz, quat_xyzw)
        T_base_obj = T_base_src @ T_src_obj
        return matrix_to_pose(T_base_obj)
    except Exception:
        return None


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_interpolated_poses(
    start_xyz: Tuple[float, float, float],
    start_quat: Tuple[float, float, float, float],
    goal_xyz: Tuple[float, float, float],
    goal_quat: Tuple[float, float, float, float],
    steps: int,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    if steps < 1:
        return [(goal_xyz, goal_quat)]
    trajectory: List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = []
    for i in range(1, steps + 1):
        t = _smoothstep01(i / steps)
        xyz = tuple(start_xyz[j] + t * (goal_xyz[j] - start_xyz[j]) for j in range(3))
        quat = quaternion_slerp(start_quat, goal_quat, t)
        trajectory.append((xyz, quat))
    return trajectory


def build_interpolated_joints(
    start_joints: List[float],
    goal_joints: List[float],
    steps: int,
    *,
    linear: bool = True,
) -> List[List[float]]:
    if len(start_joints) < 16 or len(goal_joints) < 16:
        return [list(goal_joints[:16])]
    if steps < 1:
        return [list(goal_joints[:16])]
    trajectory: List[List[float]] = []
    for i in range(1, steps + 1):
        t = (i / steps) if linear else _smoothstep01(i / steps)
        trajectory.append(
            [
                start_joints[j] + t * (goal_joints[j] - start_joints[j])
                for j in range(16)
            ]
        )
    return trajectory


def slider_to_arm_move_speed(slider_val: int) -> float:
    pct = max(10, min(100, int(slider_val))) / 100.0
    return ARM_MOVE_SPEED_MIN_RAD_S + pct * (
        ARM_MOVE_SPEED_MAX_RAD_S - ARM_MOVE_SPEED_MIN_RAD_S
    )


def format_arm_move_speed_label(slider_val: int) -> str:
    speed = slider_to_arm_move_speed(slider_val)
    return f"{speed:.2f} rad/s"


def compute_arm_move_duration_s(
    start_joints: List[float],
    goal_joints: List[float],
    joint_speed_rad_s: float = ARM_MOVE_SPEED_DEFAULT_RAD_S,
) -> float:
    if len(start_joints) < 16 or len(goal_joints) < 16:
        return ARM_MOVE_MIN_DURATION_S
    max_delta = max(abs(goal_joints[i] - start_joints[i]) for i in range(16))
    if max_delta < 1e-4:
        return 0.0
    speed = max(ARM_MOVE_SPEED_MIN_RAD_S, float(joint_speed_rad_s))
    duration = max_delta / speed
    return max(ARM_MOVE_MIN_DURATION_S, min(ARM_MOVE_MAX_DURATION_S, duration))


def _joint_dict_from_msg(msg: JointState) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for i, name in enumerate(msg.name):
        if i < len(msg.position):
            result[name] = float(msg.position[i])
    return result


def _split_arm_joints(joints: Dict[str, float]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    waist: List[Tuple[str, float]] = []
    left: List[Tuple[str, float]] = []
    right: List[Tuple[str, float]] = []
    for name in WAIST_JOINT_NAMES:
        if name in joints:
            waist.append((name, joints[name]))
    for name in LEFT_ARM_JOINT_NAMES:
        if name in joints:
            left.append((name, joints[name]))
    for name in RIGHT_ARM_JOINT_NAMES:
        if name in joints:
            right.append((name, joints[name]))
    return left, right, waist


def _arm_from_raw_positions(names: List[str], positions: List[float]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if names:
        left, right, _ = _split_arm_joints(dict(zip(names, positions)))
        return left, right
    if len(positions) >= 14:
        left = list(zip(LEFT_ARM_JOINT_NAMES, positions[:7]))
        right = list(zip(RIGHT_ARM_JOINT_NAMES, positions[7:14]))
        return left, right
    return [], []


def _pose_from_msg(msg: PoseStamped) -> TcpPoseSnapshot:
    p = msg.pose.position
    q = msg.pose.orientation
    return TcpPoseSnapshot(
        frame_id=msg.header.frame_id or "base_link",
        xyz=(float(p.x), float(p.y), float(p.z)),
        quat_xyzw=(float(q.x), float(q.y), float(q.z), float(q.w)),
        valid=True,
    )


def _tcp_pose_priority(snapshot: Optional[TcpPoseSnapshot]) -> int:
    if snapshot is None or not snapshot.valid:
        return -1
    frame = normalize_frame_id(snapshot.frame_id)
    if frame == normalize_frame_id(IK_TARGET_FRAME):
        return 2
    if frame in MINK_FK_FRAME_ALIASES:
        return 1
    return 0


def _should_accept_tcp_update(
    current: Optional[TcpPoseSnapshot],
    incoming: TcpPoseSnapshot,
) -> bool:
    if current is None or not current.valid:
        return True
    return _tcp_pose_priority(incoming) >= _tcp_pose_priority(current)


def _format_joint_group(title: str, joints: List[Tuple[str, float]], unit: str = "rad") -> str:
    if not joints:
        return f"{title}: (无数据)\n"
    lines = [f"{title}:"]
    for name, val in joints:
        if unit == "rad":
            lines.append(f"  {name}: {val:.3f} rad ({math.degrees(val):.1f}°)")
        elif unit == "mixed":
            if "lift" in name:
                lines.append(f"  {name}: {val:.3f} m")
            else:
                lines.append(f"  {name}: {val:.3f} rad ({math.degrees(val):.1f}°)")
        elif unit == "norm":
            lines.append(f"  {name}: {val:.3f}")
        else:
            lines.append(f"  {name}: {val:.3f} {unit}")
    return "\n".join(lines) + "\n"


def _html_span(text: str, color: str, bold: bool = False) -> str:
    weight = "font-weight:bold;" if bold else ""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return f'<span style="color:{color};{weight}">{safe}</span>'


def _format_joint_group_html(
    title: str,
    joints: List[Tuple[str, float]],
    title_color: str,
    value_color: str,
    unit: str = "rad",
) -> str:
    if not joints:
        return _html_span(f"{title}: (无数据)", "#888888") + "<br/>"
    lines = [_html_span(f"{title}:", title_color, bold=True)]
    for name, val in joints:
        if unit == "rad":
            detail = f"  {name}: {val:.3f} rad ({math.degrees(val):.1f}°)"
        elif unit == "mixed":
            if "lift" in name:
                detail = f"  {name}: {val:.3f} m"
            else:
                detail = f"  {name}: {val:.3f} rad ({math.degrees(val):.1f}°)"
        elif unit == "norm":
            detail = f"  {name}: {val:.3f}"
        else:
            detail = f"  {name}: {val:.3f} {unit}"
        lines.append(_html_span(detail, value_color))
    return "<br/>".join(lines) + "<br/>"


def format_robot_state_html(
    state: RobotStateSnapshot,
    camera_frame: str,
    left_cam: Optional[np.ndarray],
    right_cam: Optional[np.ndarray],
) -> str:
    age = time.time() - state.updated_at if state.updated_at > 0 else -1.0
    header = f"══ 机器人状态 ══  相机系: {camera_frame or '未知'}"
    if age >= 0:
        header += f"  ({age:.1f}s 前)"

    parts = [
        "<div style='font-family: Monospace, Consolas, monospace; font-size: 9pt; "
        "line-height: 1.45;'>",
        _html_span(header, "#f0f0f0", bold=True),
        "<br/><br/>",
    ]

    if left_cam is not None:
        tcp_text = (
            f"左臂 TCP (相机系): XYZ=({left_cam[0]:.3f}, {left_cam[1]:.3f}, {left_cam[2]:.3f}) m"
        )
        parts.append(_html_span(tcp_text, "#ff79c6", bold=True))
    elif state.left_tcp and state.left_tcp.valid:
        x, y, z = state.left_tcp.xyz
        tcp_text = (
            f"左臂 TCP ({state.left_tcp.frame_id}): "
            f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
        parts.append(_html_span(tcp_text, "#ff79c6", bold=True))
    else:
        parts.append(_html_span("左臂 TCP: (无数据)", "#888888"))

    parts.append("<br/>")

    if right_cam is not None:
        tcp_text = (
            f"右臂 TCP (相机系): XYZ=({right_cam[0]:.3f}, {right_cam[1]:.3f}, {right_cam[2]:.3f}) m"
        )
        parts.append(_html_span(tcp_text, "#79d8ff", bold=True))
    elif state.right_tcp and state.right_tcp.valid:
        x, y, z = state.right_tcp.xyz
        tcp_text = (
            f"右臂 TCP ({state.right_tcp.frame_id}): "
            f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
        parts.append(_html_span(tcp_text, "#79d8ff", bold=True))
    else:
        parts.append(_html_span("右臂 TCP: (无数据)", "#888888"))

    parts.append("<br/><br/>")
    parts.append(_format_joint_group_html("左臂关节", state.left_arm, "#ff79c6", "#ffd0ea"))
    parts.append(_format_joint_group_html("右臂关节", state.right_arm, "#79d8ff", "#ccefff"))
    if state.waist:
        parts.append(_format_joint_group_html("腰关节", state.waist, "#f1fa8c", "#fffacd", unit="mixed"))
    parts.append(_format_joint_group_html("左手关节", state.left_hand, "#ffb86c", "#ffe8cc", unit="norm"))
    parts.append(_format_joint_group_html("右手关节", state.right_hand, "#8be9fd", "#d4f7ff", unit="norm"))
    parts.append("</div>")
    return "".join(parts)


def format_robot_state_text(
    state: RobotStateSnapshot,
    camera_frame: str,
    left_cam: Optional[np.ndarray],
    right_cam: Optional[np.ndarray],
) -> str:
    age = time.time() - state.updated_at if state.updated_at > 0 else -1.0
    header = f"══ 机器人状态 ══  相机系: {camera_frame or '未知'}"
    if age >= 0:
        header += f"  ({age:.1f}s 前)"
    parts = [header, ""]

    if left_cam is not None:
        parts.append(
            f"左臂 TCP (相机系): XYZ=({left_cam[0]:.3f}, {left_cam[1]:.3f}, {left_cam[2]:.3f}) m"
        )
    elif state.left_tcp and state.left_tcp.valid:
        x, y, z = state.left_tcp.xyz
        parts.append(
            f"左臂 TCP ({state.left_tcp.frame_id}): XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
    else:
        parts.append("左臂 TCP: (无数据)")

    if right_cam is not None:
        parts.append(
            f"右臂 TCP (相机系): XYZ=({right_cam[0]:.3f}, {right_cam[1]:.3f}, {right_cam[2]:.3f}) m"
        )
    elif state.right_tcp and state.right_tcp.valid:
        x, y, z = state.right_tcp.xyz
        parts.append(
            f"右臂 TCP ({state.right_tcp.frame_id}): XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m  [TF未变换]"
        )
    else:
        parts.append("右臂 TCP: (无数据)")

    parts.append("")
    parts.append(_format_joint_group("左臂关节", state.left_arm))
    parts.append(_format_joint_group("右臂关节", state.right_arm))
    if state.waist:
        parts.append(_format_joint_group("腰关节", state.waist, unit="mixed"))
    parts.append(_format_joint_group("左手关节", state.left_hand, unit="norm"))
    parts.append(_format_joint_group("右手关节", state.right_hand, unit="norm"))
    return "\n".join(parts).strip()


def make_tcp_axis_lines(center: np.ndarray, rotation: np.ndarray, scale: float = 0.08) -> np.ndarray:
    origin = center.astype(np.float32)
    lines = []
    for i in range(3):
        end = origin + rotation[:, i] * scale
        lines.extend([origin, end])
    return np.stack(lines, axis=0).astype(np.float32)


def slider_to_hand_position(slider_pct: int) -> float:
    """游标 0~100 直接映射关节 position：0=完全张开，100=完全闭合。"""
    slider_pct = max(0, min(100, int(slider_pct)))
    return slider_pct / 100.0


def format_hand_angle_label(slider_pct: int) -> str:
    pos = slider_to_hand_position(slider_pct)
    return f"角度 {pos:.2f}"


def make_left_hand_command(position: float) -> JointState:
    """构建睿研左手控制命令，position 由游标直接确定 (0=张, 1=合)。"""
    msg = JointState()
    msg.header.frame_id = "left_hand"
    msg.name = list(LEFT_HAND_JOINT_NAMES)
    pos = max(0.0, min(1.0, float(position)))
    msg.position = [pos] * len(LEFT_HAND_JOINT_NAMES)
    msg.velocity = list(LEFT_HAND_CMD_VELOCITY)
    msg.effort = list(LEFT_HAND_CMD_EFFORT)
    return msg


@dataclass
class LlmChatConfig:
    api_base: str = LLM_API_BASE_DEFAULT
    model: str = LLM_MODEL_DEFAULT
    api_key: str = ""

    @classmethod
    def from_env(cls) -> LlmChatConfig:
        return cls(
            api_base=os.environ.get("LLM_API_BASE", LLM_API_BASE_DEFAULT),
            model=os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
            api_key=os.environ.get(LLM_API_KEY_ENV, "").strip(),
        )


class LlmChatClient:
    """OpenAI 兼容 Chat Completions API 客户端（支持 OpenAI / Ollama / 本地代理等）。"""

    def __init__(self, config: LlmChatConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: List[Dict[str, str]],
        timeout_s: float = LLM_CHAT_TIMEOUT_S,
    ) -> str:
        api_key = self.config.api_key.strip()
        if not api_key:
            raise RuntimeError(
                f"未配置 API Key，请设置环境变量 {LLM_API_KEY_ENV}，"
                "或在对话面板中填写（Ollama 等本地服务可填任意非空字符串）"
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"网络错误: {exc.reason}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 无有效回复: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"API 返回空内容: {body}")
        return str(content).strip()


class LlmChatBridge(QObject):
    finished = pyqtSignal(str, bool)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class ChatPanelWidget(QWidget):
    """文本对话面板：输入问题，后台调用大模型并显示回复。"""

    status_message = pyqtSignal(str)

    def __init__(
        self,
        config: Optional[LlmChatConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or LlmChatConfig.from_env()
        self._client = LlmChatClient(self._config)
        self._messages: List[Dict[str, str]] = [
            {"role": "system", "content": LLM_CHAT_SYSTEM_PROMPT}
        ]
        self._busy = False
        self._bridge = LlmChatBridge()
        self._bridge.finished.connect(self._on_llm_finished)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(QLabel("AI 对话"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(LLM_PROVIDER_PRESETS.keys()))
        self.provider_combo.setToolTip(
            "Qwen-Robot 三模型权重暂未公开；此处可选本地 Qwen 骨干或百炼 API"
        )
        self.provider_combo.currentTextChanged.connect(self._on_provider_preset_changed)
        header.addWidget(self.provider_combo)
        self.model_edit = QLineEdit(self._config.model)
        self.model_edit.setPlaceholderText("模型名")
        self.model_edit.setToolTip("LLM 模型名称，如 gpt-4o-mini 或 qwen2.5")
        self.model_edit.setFixedWidth(140)
        header.addWidget(self.model_edit)
        header.addStretch()
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("API"))
        self.api_base_edit = QLineEdit(self._config.api_base)
        self.api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        self.api_base_edit.setToolTip(
            "OpenAI 兼容 API 地址（Ollama: http://127.0.0.1:11434/v1）"
        )
        settings_row.addWidget(self.api_base_edit, stretch=1)
        layout.addLayout(settings_row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Key"))
        self.api_key_edit = QLineEdit(self._config.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText(f"或设置 ${LLM_API_KEY_ENV}")
        self.api_key_edit.setToolTip(
            f"API Key；本地 Ollama 可填 ollama。也可 export {LLM_API_KEY_ENV}=..."
        )
        key_row.addWidget(self.api_key_edit, stretch=1)
        layout.addLayout(key_row)

        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("对话记录将显示在这里…")
        self.history_view.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #ddd; border: 1px solid #444; }"
        )
        layout.addWidget(self.history_view, stretch=1)

        input_row = QHBoxLayout()
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入问题，Enter 发送，Shift+Enter 换行")
        self.input_edit.setMaximumHeight(80)
        self.input_edit.setStyleSheet(
            "QTextEdit { background-color: #252525; color: #eee; border: 1px solid #555; }"
        )
        self.input_edit.installEventFilter(self)
        input_row.addWidget(self.input_edit, stretch=1)
        send_col = QVBoxLayout()
        self.send_btn = QPushButton("发送")
        self.send_btn.setToolTip("调用大模型获取回复")
        self.send_btn.clicked.connect(self._on_send_clicked)
        send_col.addWidget(self.send_btn)
        send_col.addStretch()
        input_row.addLayout(send_col)
        layout.addLayout(input_row)

        self._append_system_line(
            f"模型: {self._config.model}  |  API: {self._config.api_base}"
        )
        self._append_system_line(
            "Qwen-RobotManip/Nav/World 权重尚未开源；"
            "可先用 Ollama 部署 Qwen3.5 / Qwen3-VL 骨干，或百炼 API"
        )
        if not self._config.api_key:
            self._append_system_line(
                f"提示: 请填写 API Key 或 export {LLM_API_KEY_ENV}=your-key"
            )
        if self._config.api_key and self._config.api_base != LLM_PROVIDER_PRESETS[
            "本地 Ollama · Qwen3.5-4B"
        ][0]:
            idx = self.provider_combo.findText("自定义")
        else:
            idx = self.provider_combo.findText("本地 Ollama · Qwen3.5-4B")
        if idx >= 0:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(idx)
            self.provider_combo.blockSignals(False)
            if idx != self.provider_combo.findText("自定义"):
                self._apply_provider_preset(self.provider_combo.currentText(), silent=True)

    def _apply_provider_preset(self, name: str, silent: bool = False) -> None:
        preset = LLM_PROVIDER_PRESETS.get(name)
        if preset is None or name == "自定义":
            return
        api_base, model, default_key = preset
        if api_base:
            self.api_base_edit.setText(api_base)
        if model:
            self.model_edit.setText(model)
        if default_key and not self.api_key_edit.text().strip():
            self.api_key_edit.setText(default_key)
        if not silent:
            self._append_system_line(f"已切换预设: {name}")

    def _on_provider_preset_changed(self, name: str) -> None:
        self._apply_provider_preset(name)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.input_edit and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
                event.modifiers() & Qt.ShiftModifier
            ):
                self._on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    def _sync_config_from_ui(self) -> None:
        self._config.api_base = self.api_base_edit.text().strip() or LLM_API_BASE_DEFAULT
        self._config.model = self.model_edit.text().strip() or LLM_MODEL_DEFAULT
        key = self.api_key_edit.text().strip()
        if not key:
            key = os.environ.get(LLM_API_KEY_ENV, "").strip()
        self._config.api_key = key
        self._client = LlmChatClient(self._config)

    def _append_system_line(self, text: str) -> None:
        self.history_view.append(f'<span style="color:#888;">[系统] {text}</span>')

    def _append_user_line(self, text: str) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;"><b style="color:#7ec8ff;">你:</b> {safe}</p>'
        )

    def _append_assistant_line(self, text: str) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;"><b style="color:#50fa7b;">AI:</b> {safe}</p>'
        )

    def _append_error_line(self, text: str) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;"><b style="color:#ff5555;">错误:</b> {safe}</p>'
        )

    def _trim_history(self) -> None:
        if len(self._messages) <= 1 + LLM_CHAT_MAX_HISTORY:
            return
        system = self._messages[0]
        self._messages = [system] + self._messages[-(LLM_CHAT_MAX_HISTORY):]

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_btn.setEnabled(not busy)
        self.input_edit.setEnabled(not busy)
        self.send_btn.setText("思考中…" if busy else "发送")

    def _on_send_clicked(self) -> None:
        if self._busy:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self._sync_config_from_ui()
        self.input_edit.clear()
        self._append_user_line(text)
        self._messages.append({"role": "user", "content": text})
        self._trim_history()
        self._set_busy(True)
        self.status_message.emit("正在调用大模型…")
        snapshot = list(self._messages)

        def _work() -> None:
            try:
                reply = self._client.chat(snapshot)
                self._bridge.finished.emit(reply, True)
            except Exception as exc:
                self._bridge.finished.emit(str(exc), False)

        threading.Thread(target=_work, daemon=True).start()

    def _on_llm_finished(self, text: str, ok: bool) -> None:
        self._set_busy(False)
        if ok:
            self._append_assistant_line(text)
            self._messages.append({"role": "assistant", "content": text})
            self._trim_history()
            self.status_message.emit("大模型回复完成")
        else:
            self._append_error_line(text)
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            self.status_message.emit(f"大模型调用失败: {text[:80]}")

    def _clear_chat(self) -> None:
        self._messages = [{"role": "system", "content": LLM_CHAT_SYSTEM_PROMPT}]
        self.history_view.clear()
        self._append_system_line("对话已清空")


class RosBridge(QObject):
    frame_updated = pyqtSignal(str, object)
    topics_updated = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    frame_stats = pyqtSignal(str, int, float)
    robot_state_updated = pyqtSignal()
    arm_enable_changed = pyqtSignal(bool)
    replay_state_changed = pyqtSignal(int)
    left_hand_preset_changed = pyqtSignal(bool)
    slow_motion_progress = pyqtSignal(float, str)
    slow_motion_finished = pyqtSignal(bool, str)


class PoseComputeBridge(QObject):
    """后台位姿计算完成信号（跨线程投递到 UI）。"""

    finished = pyqtSignal(object, str, int, int, object)


class DepthVizBridge(QObject):
    """后台深度预览/点云计算完成信号。"""

    finished = pyqtSignal(object)


class ClickableImageLabel(QLabel):
    """可点击图像，将控件坐标映射回原始像素。"""

    clicked_pixel = pyqtSignal(int, int)

    def __init__(self, placeholder: str = "等待图像...", parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border: 1px solid #444;"
        )
        self.setCursor(Qt.CrossCursor)
        self._source_image: Optional[np.ndarray] = None
        self._latest_pixmap: Optional[QPixmap] = None
        self._segment_mask: Optional[np.ndarray] = None
        self._segment_centroid: Optional[Tuple[float, float]] = None
        self._segment_contact_uv: Optional[Tuple[float, float]] = None
        self._segment_obb: Optional[np.ndarray] = None
        self._segment_intrinsics: Optional[Tuple[float, float, float, float]] = None

    def set_source_image(self, cv_image: np.ndarray, pixmap: Optional[QPixmap] = None) -> None:
        self._source_image = cv_image
        self._refresh_display(pixmap)

    def set_segment_overlay(
        self,
        mask: Optional[np.ndarray],
        centroid_uv: Optional[Tuple[float, float]] = None,
        obb_corners: Optional[np.ndarray] = None,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
        contact_uv: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._segment_mask = mask.copy() if mask is not None else None
        self._segment_centroid = centroid_uv
        self._segment_contact_uv = contact_uv
        self._segment_obb = obb_corners.copy() if obb_corners is not None else None
        self._segment_intrinsics = intrinsics
        self._refresh_display()

    def clear_segment_overlay(self) -> None:
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._refresh_display()

    def set_precomposed_display(self, overlay_bgr: np.ndarray) -> None:
        """直接显示后台线程已合成好的叠加图，避免 UI 线程重复计算。"""
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._latest_pixmap = cv2_to_qpixmap(overlay_bgr)
        self._render_pixmap()

    def _compose_display_image(self) -> np.ndarray:
        if self._source_image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if self._segment_mask is not None and self._segment_mask.any():
            return apply_segment_overlay(
                self._source_image,
                self._segment_mask,
                self._segment_centroid,
                obb_corners=self._segment_obb,
                intrinsics=self._segment_intrinsics,
                contact_uv=self._segment_contact_uv,
            )
        return self._source_image

    def _refresh_display(self, pixmap: Optional[QPixmap] = None) -> None:
        if self._source_image is None:
            return
        if pixmap is not None and self._segment_mask is None:
            self._latest_pixmap = pixmap
        else:
            self._latest_pixmap = cv2_to_qpixmap(self._compose_display_image())
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._latest_pixmap is None:
            return
        target = self.size()
        if target.width() < 16 or target.height() < 16:
            self.setPixmap(self._latest_pixmap)
            return
        scaled = self._latest_pixmap.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def _map_click_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._source_image is None or self._latest_pixmap is None:
            return None

        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self._latest_pixmap.width(), self._latest_pixmap.height()
        scale = min(label_w / pix_w, label_h / pix_h)
        disp_w = int(pix_w * scale)
        disp_h = int(pix_h * scale)
        offset_x = (label_w - disp_w) // 2
        offset_y = (label_h - disp_h) // 2

        lx, ly = pos.x() - offset_x, pos.y() - offset_y
        if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
            return None

        img_h, img_w = self._source_image.shape[:2]
        u = int(lx / scale)
        v = int(ly / scale)
        u = max(0, min(img_w - 1, u))
        v = max(0, min(img_h - 1, v))
        return u, v

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            uv = self._map_click_to_image(event.pos())
            if uv is not None:
                self.clicked_pixel.emit(uv[0], uv[1])
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_pixmap()


class CameraPanel(QWidget):
    def __init__(
        self,
        topic: str,
        get_paired_depth: Optional[Callable[[str], Optional[np.ndarray]]] = None,
        get_intrinsics_for_depth: Optional[
            Callable[[str, int, int], Tuple[float, float, float, float]]
        ] = None,
        get_depth_topic: Optional[Callable[[str], Optional[str]]] = None,
        get_camera_frame_id: Optional[Callable[[str], str]] = None,
        resolve_segment_camera_frame: Optional[
            Callable[[str, Optional[str]], str]
        ] = None,
        is_paired_depth_enabled: Optional[Callable[[str], bool]] = None,
        on_segment_pose: Optional[Callable[[SegmentPoseTarget], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.topic = topic
        self._get_paired_depth = get_paired_depth
        self._get_intrinsics_for_depth = get_intrinsics_for_depth
        self._get_depth_topic = get_depth_topic
        self._get_camera_frame_id = get_camera_frame_id
        self._resolve_segment_camera_frame = resolve_segment_camera_frame
        self._is_paired_depth_enabled = is_paired_depth_enabled
        self._on_segment_pose = on_segment_pose
        self._status_callback = status_callback
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0
        self._latest_image: Optional[np.ndarray] = None

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title = topic.split("/")[-1] or topic
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Monospace", 10, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.image_label = ClickableImageLabel("等待图像...（勾选 depth 后点击物体进行立体分割）")
        self.image_label.clicked_pixel.connect(self._on_pixel_clicked)
        layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel(topic)
        self.info_label.setFont(QFont("Monospace", 8))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #666;")
        self.info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = QLabel("点击图像：需勾选配对 depth topic 后才可进行立体分割")
        self.coord_label.setFont(QFont("Monospace", 8))
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setStyleSheet("color: #2a82da;")
        self.coord_label.setWordWrap(True)
        self.coord_label.setMaximumHeight(40)
        self.coord_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.coord_label)

        self._pose_bridge = PoseComputeBridge()
        self._pose_bridge.finished.connect(self._on_pose_compute_finished)
        self._pose_busy = False

    def _start_pose_compute(
        self,
        depth: np.ndarray,
        color: Optional[np.ndarray],
        u: int,
        v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        display_u: int,
    ) -> None:
        if self._pose_busy:
            return
        self._pose_busy = True
        self.coord_label.setText("立体分割计算中...")

        def _work() -> None:
            overlay = None
            try:
                depth_copy = depth.copy()
                color_copy = color.copy() if color is not None else None
                result, info = run_stereo_pose_pipeline_resampled(
                    depth_copy, u, v, fx, fy, cx, cy, color_bgr=color_copy
                )
                if result is not None and color_copy is not None:
                    overlay = build_pose_overlay_image(
                        color_copy, result, (fx, fy, cx, cy)
                    )
            except Exception as exc:
                result, info, overlay = None, f"点击 ({display_u}, {v})  |  计算错误: {exc}", None
            self._pose_bridge.finished.emit(result, info, display_u, v, overlay)

        threading.Thread(target=_work, daemon=True).start()

    def _on_pose_compute_finished(
        self,
        result: Optional[Object6DPoseResult],
        info: str,
        display_u: int,
        display_v: int,
        overlay: Optional[np.ndarray] = None,
    ) -> None:
        self._pose_busy = False
        if result is None:
            self.image_label.clear_segment_overlay()
        else:
            if overlay is not None:
                self.image_label.set_precomposed_display(overlay)
            else:
                display_mask = resize_mask_to_shape(result.mask, self._latest_image.shape[:2])
                dh, dw = result.mask.shape[:2]
                ch, cw = self._latest_image.shape[:2]
                cu, cv_pt = scale_uv_to_shape(
                    int(round(result.centroid_uv[0])),
                    int(round(result.centroid_uv[1])),
                    (dh, dw),
                    (ch, cw),
                )
                contact_u, contact_v = scale_uv_to_shape(
                    int(round(result.contact_uv[0])),
                    int(round(result.contact_uv[1])),
                    (dh, dw),
                    (ch, cw),
                )
                depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
                if depth_topic and self._get_intrinsics_for_depth:
                    fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
                    self.image_label.set_segment_overlay(
                        display_mask,
                        (float(cu), float(cv_pt)),
                        obb_corners=result.obb_corners,
                        intrinsics=(fx, fy, cx, cy),
                        contact_uv=(float(contact_u), float(contact_v)),
                    )
            self._emit_segment_pose(result)
        self.coord_label.setText(info)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] {info.splitlines()[0]}")

    def _emit_segment_pose(self, result: Object6DPoseResult) -> None:
        if self._on_segment_pose is None:
            return
        depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
        if self._resolve_segment_camera_frame is not None:
            camera_frame = self._resolve_segment_camera_frame(self.topic, depth_topic)
        elif self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(depth_topic or self.topic)
        else:
            return
        if not camera_frame:
            hint = (
                f"[{self.topic}] 分割成功但缺少相机 frame_id，"
                "无法用于左臂移动（请确认 depth/color 已发布 header.frame_id）"
            )
            if self._status_callback:
                self._status_callback(hint)
            return
        target = SegmentPoseTarget.from_pose_result(result, camera_frame, self.topic)
        QTimer.singleShot(0, lambda t=target: self._on_segment_pose(t))

    def _on_pixel_clicked(self, u: int, v: int) -> None:
        if self._latest_image is None or self._pose_busy:
            return

        if self._is_paired_depth_enabled is not None and not self._is_paired_depth_enabled(
            self.topic
        ):
            info = format_color_pixel_info(u, v, self._latest_image)
            self.image_label.clear_segment_overlay()
            self.coord_label.setText(f"{info}  |  未勾选 depth，跳过分割")
            if self._status_callback:
                self._status_callback(f"[{self.topic}] 未勾选 depth，跳过分割")
            return

        depth = self._get_paired_depth(self.topic) if self._get_paired_depth else None
        depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None

        if depth is not None and depth_topic and self._get_intrinsics_for_depth:
            dh, dw = depth.shape[:2]
            fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
            u_d, v_d = scale_uv_to_shape(u, v, self._latest_image.shape[:2], (dh, dw))
            self._start_pose_compute(
                depth, self._latest_image, u_d, v_d, fx, fy, cx, cy, display_u=u
            )
            return

        info = format_color_pixel_info(u, v, self._latest_image)
        self.image_label.clear_segment_overlay()
        self.coord_label.setText(f"{info}  |  无 depth 数据，无法分割")
        if self._status_callback:
            self._status_callback(f"[{self.topic}] {info}")

    def update_frame(self, cv_image: np.ndarray) -> None:
        self._latest_image = cv_image.copy()
        self.image_label.set_source_image(self._latest_image)

        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        h, w = cv_image.shape[:2]
        self.info_label.setText(f"{self.topic}  |  {w}x{h}  |  {self._fps:.1f} Hz")


class DepthPanel3D(QWidget):
    """深度图 3D 点云显示面板。"""

    def __init__(
        self,
        topic: str,
        get_intrinsics: Optional[Callable[[str, int, int], Tuple[float, float, float, float]]] = None,
        get_paired_color: Optional[Callable[[str], Optional[np.ndarray]]] = None,
        get_robot_state: Optional[Callable[[], RobotStateSnapshot]] = None,
        get_camera_frame_id: Optional[Callable[[str], str]] = None,
        resolve_segment_camera_frame: Optional[
            Callable[[str, Optional[str]], str]
        ] = None,
        get_tf_buffer: Optional[Callable[[], Optional[Buffer]]] = None,
        on_segment_pose: Optional[Callable[[SegmentPoseTarget], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.topic = topic
        self._get_intrinsics = get_intrinsics
        self._get_paired_color = get_paired_color
        self._get_robot_state = get_robot_state
        self._get_camera_frame_id = get_camera_frame_id
        self._resolve_segment_camera_frame = resolve_segment_camera_frame
        self._get_tf_buffer = get_tf_buffer
        self._on_segment_pose = on_segment_pose
        self._status_callback = status_callback
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._last_display_time = 0.0
        self._last_robot_overlay_time = 0.0
        self._fps = 0.0
        self._point_count = 0
        self._latest_depth: Optional[np.ndarray] = None
        self._intrinsics: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._depth_preview_raw: Optional[np.ndarray] = None
        self._depth_full_shape: Tuple[int, int] = (0, 0)
        self._preview_shape: Tuple[int, int] = (0, 0)
        self._preview_intrinsics: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._pose_bridge = PoseComputeBridge()
        self._pose_bridge.finished.connect(self._on_pose_compute_finished)
        self._depth_viz_bridge = DepthVizBridge()
        self._depth_viz_bridge.finished.connect(self._on_depth_viz_finished)
        self._pose_busy = False
        self._depth_viz_busy = False
        self._depth_viz_pending = False

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title = topic.split("/")[-1] or topic
        self.title_label = QLabel(f"{title}  [3D]")
        self.title_label.setFont(QFont("Monospace", 10, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.view = gl.GLViewWidget()
        self.view.setMinimumSize(80, 60)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor((30, 30, 30))
        self.view.opts["distance"] = 2.0
        layout.addWidget(self.view, stretch=3)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.2, 0.2)
        self.view.addItem(grid)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32),
            size=3,
            pxMode=True,
        )
        self.view.addItem(self.scatter)

        self.segment_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 0.85, 0.1, 1.0]], dtype=np.float32),
            size=6,
            pxMode=True,
        )
        self.view.addItem(self.segment_scatter)

        self.pose_obb_lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.2, 0.9, 1.0, 1.0),
            width=2,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.pose_obb_lines)

        self.pose_axes_lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            width=3,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.pose_axes_lines)

        self.robot_left_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 0.2, 0.8, 1.0]], dtype=np.float32),
            size=14,
            pxMode=False,
        )
        self.robot_right_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[0.2, 0.85, 1.0, 1.0]], dtype=np.float32),
            size=14,
            pxMode=False,
        )
        self.robot_left_axes = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.3, 0.8, 1.0),
            width=2,
            antialias=True,
            mode="lines",
        )
        self.robot_right_axes = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.3, 0.85, 1.0, 1.0),
            width=2,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.robot_left_scatter)
        self.view.addItem(self.robot_right_scatter)
        self.view.addItem(self.robot_left_axes)
        self.view.addItem(self.robot_right_axes)

        self.depth_preview = ClickableImageLabel("点击深度图：立体分割 + 6D 位姿")
        self.depth_preview.setMinimumHeight(48)
        self.depth_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.depth_preview.clicked_pixel.connect(self._on_depth_pixel_clicked)
        layout.addWidget(self.depth_preview, stretch=1)

        self.info_label = QLabel("等待深度图...")
        self.info_label.setFont(QFont("Monospace", 8))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #666;")
        self.info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = QLabel("点击深度图：立体分割 + 6D 位姿（位置 + RPY + OBB）")
        self.coord_label.setFont(QFont("Monospace", 8))
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setStyleSheet("color: #2a82da;")
        self.coord_label.setWordWrap(True)
        self.coord_label.setMaximumHeight(40)
        self.coord_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.coord_label)

        robot_group = QGroupBox("机器人：左右手 / 手臂 / 关节")
        robot_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        robot_group.setMaximumHeight(148)
        robot_group.setStyleSheet(
            "QGroupBox {"
            "  color: #e8e8e8;"
            "  font-weight: bold;"
            "  border: 1px solid #555;"
            "  border-radius: 4px;"
            "  margin-top: 6px;"
            "  padding-top: 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  padding: 0 4px;"
            "  color: #ffffff;"
            "}"
        )
        robot_layout = QVBoxLayout(robot_group)
        robot_layout.setContentsMargins(6, 4, 6, 4)
        self.robot_info_label = QLabel("等待机器人 joint_states / TCP 数据...")
        self.robot_info_label.setFont(QFont("Monospace", 9))
        self.robot_info_label.setWordWrap(True)
        self.robot_info_label.setTextFormat(Qt.RichText)
        self.robot_info_label.setStyleSheet(
            "color: #f5f5f5;"
            "background-color: #252525;"
            "padding: 4px 6px;"
            "border-radius: 3px;"
        )
        self.robot_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        robot_layout.addWidget(self.robot_info_label)
        layout.addWidget(robot_group)

        self._robot_timer = QTimer(self)
        self._robot_timer.timeout.connect(self._update_robot_overlay)
        self._robot_timer.start(int(UI_ROBOT_STATE_MIN_INTERVAL_S * 1000))

    def stop_robot_timer(self) -> None:
        self._robot_timer.stop()

    def _clear_robot_gl_items(self) -> None:
        hidden = np.zeros((1, 3), dtype=np.float32)
        transparent = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        self.robot_left_scatter.setData(pos=hidden, color=transparent, size=14, pxMode=False)
        self.robot_right_scatter.setData(pos=hidden, color=transparent, size=14, pxMode=False)
        self.robot_left_axes.setData(pos=hidden, color=(1.0, 0.3, 0.8, 0.0), width=2, mode="lines")
        self.robot_right_axes.setData(pos=hidden, color=(0.3, 0.85, 1.0, 0.0), width=2, mode="lines")

    def _set_robot_tcp_gl(
        self,
        scatter_item: gl.GLScatterPlotItem,
        axes_item: gl.GLLinePlotItem,
        cam_pose: Optional[Tuple[np.ndarray, np.ndarray]],
        color_rgba: Tuple[float, float, float, float],
    ) -> None:
        if cam_pose is None:
            hidden = np.zeros((1, 3), dtype=np.float32)
            scatter_item.setData(
                pos=hidden,
                color=np.array([[color_rgba[0], color_rgba[1], color_rgba[2], 0.0]], dtype=np.float32),
                size=14,
                pxMode=False,
            )
            axes_item.setData(pos=hidden, color=(*color_rgba[:3], 0.0), width=2, mode="lines")
            return
        center, rotation = cam_pose
        if center[2] <= 0.02:
            self._set_robot_tcp_gl(scatter_item, axes_item, None, color_rgba)
            return
        scatter_item.setData(
            pos=center.reshape(1, 3),
            color=np.array([color_rgba], dtype=np.float32),
            size=14,
            pxMode=False,
        )
        axis_lines = make_tcp_axis_lines(center, rotation, scale=0.07)
        axes_item.setData(pos=axis_lines, color=color_rgba, width=2, mode="lines")

    def _update_robot_overlay(self) -> None:
        if self._get_robot_state is None or self._pose_busy:
            return
        now = time.time()
        if now - self._last_robot_overlay_time < UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._last_robot_overlay_time = now
        state = self._get_robot_state()
        camera_frame = ""
        if self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(self.topic)
        tf_buffer = self._get_tf_buffer() if self._get_tf_buffer else None

        left_cam = transform_tcp_to_camera(state.left_tcp, camera_frame, tf_buffer)
        right_cam = transform_tcp_to_camera(state.right_tcp, camera_frame, tf_buffer)

        left_xyz = left_cam[0] if left_cam is not None else None
        right_xyz = right_cam[0] if right_cam is not None else None
        self.robot_info_label.setText(
            format_robot_state_html(state, camera_frame, left_xyz, right_xyz)
        )

        self._set_robot_tcp_gl(
            self.robot_left_scatter,
            self.robot_left_axes,
            left_cam,
            (1.0, 0.25, 0.75, 1.0),
        )
        self._set_robot_tcp_gl(
            self.robot_right_scatter,
            self.robot_right_axes,
            right_cam,
            (0.25, 0.85, 1.0, 1.0),
        )

    def _start_pose_compute(self, u_full: int, v_full: int, preview_u: int, preview_v: int) -> None:
        if self._latest_depth is None or self._pose_busy:
            return
        self._pose_busy = True
        self.coord_label.setText("立体分割计算中...")
        fx, fy, cx, cy = self._intrinsics
        preview_intrinsics = self._preview_intrinsics
        get_color = self._get_paired_color
        depth_ref = self._latest_depth

        def _work() -> None:
            overlay = None
            try:
                depth_copy = depth_ref.copy()
                color_copy = None
                if get_color is not None:
                    color = get_color(self.topic)
                    if color is not None:
                        color_copy = color.copy()
                result, info = run_stereo_pose_pipeline_resampled(
                    depth_copy, u_full, v_full, fx, fy, cx, cy, color_bgr=color_copy
                )
                if result is not None:
                    overlay_source = color_copy
                    overlay_intr = preview_intrinsics
                    if overlay_source is None:
                        overlay_source = build_depth_preview_image(depth_copy)
                        overlay_intr = preview_intrinsics
                    overlay = build_pose_overlay_image(
                        overlay_source,
                        result,
                        overlay_intr,
                        max_dim=SEGMENT_OVERLAY_MAX_DIM,
                    )
            except Exception as exc:
                result, info, overlay = None, f"点击 ({preview_u}, {preview_v})  |  计算错误: {exc}", None
            self._pose_bridge.finished.emit(result, info, preview_u, preview_v, overlay)

        threading.Thread(target=_work, daemon=True).start()

    def _on_depth_pixel_clicked(self, u: int, v: int) -> None:
        if self._latest_depth is None or self._pose_busy:
            return
        dh, dw = self._depth_full_shape
        u_full, v_full = scale_uv_to_shape(u, v, self._preview_shape, (dh, dw))
        self._start_pose_compute(u_full, v_full, u, v)

    def _on_pose_compute_finished(
        self,
        result: Optional[Object6DPoseResult],
        info: str,
        preview_u: int,
        preview_v: int,
        overlay: Optional[np.ndarray] = None,
    ) -> None:
        self._pose_busy = False
        if self._depth_viz_pending:
            self._depth_viz_pending = False
            self._request_depth_viz()
        if result is None:
            self.depth_preview.clear_segment_overlay()
            self._clear_pose_visualization()
        else:
            if overlay is not None:
                self.depth_preview.set_precomposed_display(overlay)
            else:
                mask_p = resize_mask_to_shape(result.mask, self._preview_shape)
                cu, cv_pt = scale_uv_to_shape(
                    int(round(result.centroid_uv[0])),
                    int(round(result.centroid_uv[1])),
                    self._depth_full_shape,
                    self._preview_shape,
                )
                contact_u, contact_v = scale_uv_to_shape(
                    int(round(result.contact_uv[0])),
                    int(round(result.contact_uv[1])),
                    self._depth_full_shape,
                    self._preview_shape,
                )
                self.depth_preview.set_segment_overlay(
                    mask_p,
                    (float(cu), float(cv_pt)),
                    obb_corners=result.obb_corners,
                    intrinsics=self._preview_intrinsics,
                    contact_uv=(float(contact_u), float(contact_v)),
                )
            self._show_pose_6d(result)
            self._emit_segment_pose(result)

        self.coord_label.setText(info)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] {info.splitlines()[0]}")

    def _emit_segment_pose(self, result: Object6DPoseResult) -> None:
        if self._on_segment_pose is None:
            return
        if self._resolve_segment_camera_frame is not None:
            camera_frame = self._resolve_segment_camera_frame(self.topic, self.topic)
        elif self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(self.topic)
        else:
            return
        if not camera_frame:
            hint = (
                f"[{self.topic}] 分割成功但缺少相机 frame_id，"
                "无法用于左臂移动（请确认 depth 已发布 header.frame_id）"
            )
            if self._status_callback:
                self._status_callback(hint)
            return
        target = SegmentPoseTarget.from_pose_result(result, camera_frame, self.topic)
        QTimer.singleShot(0, lambda t=target: self._on_segment_pose(t))

    def _clear_pose_visualization(self) -> None:
        self.segment_scatter.setData(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 0.85, 0.1, 0.0]], dtype=np.float32),
            size=6,
            pxMode=True,
        )
        self.pose_obb_lines.setData(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.2, 0.9, 1.0, 0.0),
            width=2,
            mode="lines",
        )
        self.pose_axes_lines.setData(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            width=3,
            mode="lines",
        )

    def _show_pose_6d(self, result: Object6DPoseResult) -> None:
        points = subsample_points(result.points_3d, MAX_GL_SEGMENT_POINTS)
        if len(points) > 0:
            colors = np.tile(
                np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32), (len(points), 1)
            )
            self.segment_scatter.setData(pos=points, color=colors, size=5, pxMode=True)

        obb_lines = obb_wireframe_edges(result.obb_corners)
        self.pose_obb_lines.setData(
            pos=obb_lines,
            color=(0.2, 0.9, 1.0, 1.0),
            width=2,
            mode="lines",
        )

        axis_scale = max(float(max(result.obb_extents)) * 0.6, 0.04)
        center = np.array(result.obb_center, dtype=np.float32)
        axis_lines, _axis_colors = pose_axes_lines(
            center, result.rotation_matrix, axis_scale
        )
        self.pose_axes_lines.setData(
            pos=axis_lines,
            color=(1.0, 0.75, 0.2, 1.0),
            width=3,
            mode="lines",
        )

    def _request_depth_viz(self) -> None:
        if self._latest_depth is None or self._pose_busy:
            return
        now = time.time()
        if now - self._last_display_time < UI_DEPTH_MIN_INTERVAL_S:
            return
        if self._depth_viz_busy:
            self._depth_viz_pending = True
            return
        self._depth_viz_busy = True
        self._last_display_time = now
        snap = self._latest_depth
        fx, fy, cx, cy = self._intrinsics

        def _work() -> None:
            try:
                result = build_depth_viz_data(snap.copy(), fx, fy, cx, cy)
            except Exception:
                result = None
            self._depth_viz_bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def _on_depth_viz_finished(self, result: Optional[DepthVizResult]) -> None:
        self._depth_viz_busy = False
        pending = self._depth_viz_pending
        self._depth_viz_pending = False
        if result is not None and not self._pose_busy:
            self._preview_shape = result.preview_shape
            self._preview_intrinsics = result.preview_intrinsics
            self._depth_preview_raw = result.preview_vis
            self.depth_preview.set_source_image(result.preview_vis)
            self._point_count = result.point_count
            if self._point_count > 0:
                self.scatter.setData(pos=result.points, color=result.colors, size=3, pxMode=True)
                center = result.points.mean(axis=0)
                self.view.opts["center"] = pg.Vector(center[0], center[1], center[2])
            h, w = result.full_shape
            self.info_label.setText(
                f"{self.topic}  |  {w}x{h}  |  {self._point_count} pts  |  {self._fps:.1f} Hz"
            )
        if pending and not self._pose_busy:
            self._request_depth_viz()

    def update_depth(self, depth: np.ndarray) -> None:
        h, w = depth.shape[:2]
        if self._get_intrinsics is not None:
            fx, fy, cx, cy = self._get_intrinsics(self.topic, w, h)
        else:
            fx, fy, cx, cy = default_intrinsics(w, h)
        self._intrinsics = (fx, fy, cx, cy)
        self._depth_full_shape = (h, w)
        self._latest_depth = depth

        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        if not self._pose_busy:
            self._request_depth_viz()
        else:
            self._depth_viz_pending = True


def create_camera_panel(
    topic: str,
    get_intrinsics: Optional[Callable[[str, int, int], Tuple[float, float, float, float]]] = None,
    get_paired_depth: Optional[Callable[[str], Optional[np.ndarray]]] = None,
    get_paired_color: Optional[Callable[[str], Optional[np.ndarray]]] = None,
    get_depth_topic: Optional[Callable[[str], Optional[str]]] = None,
    is_paired_depth_enabled: Optional[Callable[[str], bool]] = None,
    get_robot_state: Optional[Callable[[], RobotStateSnapshot]] = None,
    get_camera_frame_id: Optional[Callable[[str], str]] = None,
    resolve_segment_camera_frame: Optional[
        Callable[[str, Optional[str]], str]
    ] = None,
    get_tf_buffer: Optional[Callable[[], Optional[Buffer]]] = None,
    on_segment_pose: Optional[Callable[[SegmentPoseTarget], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> QWidget:
    if is_depth_topic(topic):
        return DepthPanel3D(
            topic,
            get_intrinsics=get_intrinsics,
            get_paired_color=get_paired_color,
            get_robot_state=get_robot_state,
            get_camera_frame_id=get_camera_frame_id,
            resolve_segment_camera_frame=resolve_segment_camera_frame,
            get_tf_buffer=get_tf_buffer,
            on_segment_pose=on_segment_pose,
            status_callback=status_callback,
        )
    return CameraPanel(
        topic,
        get_paired_depth=get_paired_depth,
        get_intrinsics_for_depth=get_intrinsics,
        get_depth_topic=get_depth_topic,
        get_camera_frame_id=get_camera_frame_id,
        resolve_segment_camera_frame=resolve_segment_camera_frame,
        is_paired_depth_enabled=is_paired_depth_enabled,
        on_segment_pose=on_segment_pose,
        status_callback=status_callback,
    )


class CameraTopicNode(Node):
    def __init__(self, bridge: RosBridge, prefix: str) -> None:
        super().__init__("camera_topic_viewer")
        self.ros_bridge = bridge
        self.prefix = prefix
        self.cv_bridge = CvBridge()
        self.subscriptions_map: Dict[str, object] = {}
        self.camera_info_subs: Dict[str, object] = {}
        self._camera_intrinsics: Dict[str, Tuple[float, float, float, float]] = {}
        self.enabled_topics: set[str] = set()
        self._state_lock = threading.Lock()
        self._pending_prefix: Optional[str] = None
        self._pending_enabled: Optional[set[str]] = None
        self._last_camera_topics: Dict[str, List[str]] = {}
        self._last_status_message = ""
        self._frame_counts: Dict[str, int] = {}
        self._last_frame_time: Dict[str, float] = {}
        self._last_subscribe_time: Dict[str, float] = {}
        self._subscribe_attempts: Dict[str, int] = {}
        self._camera_frame_ids: Dict[str, str] = {}
        self._last_ui_emit_time: Dict[str, float] = {}
        self._robot_subscriptions_map: Dict[str, object] = {}
        self._robot_subscribe_attempts: Dict[str, int] = {}
        self._robot_resubscribe_count: Dict[str, int] = {}
        self._robot_last_subscribe_time: Dict[str, float] = {}
        self._robot_tcp_topic_received: Dict[str, bool] = {}

        self._robot_lock = threading.Lock()
        self._robot_state = RobotStateSnapshot([], [], [], [], [], None, None, 0.0)
        self._last_robot_state_emit = 0.0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._left_hand_cmd_pub = self.create_publisher(
            JointState,
            ROBOT_LEFT_HAND_CMD_TOPIC,
            10,
        )
        self._left_ik_pub = self.create_publisher(
            PoseStamped,
            LEFT_IK_TARGET_TOPIC,
            10,
        )
        self._right_ik_pub = self.create_publisher(
            PoseStamped,
            RIGHT_IK_TARGET_TOPIC,
            10,
        )
        self._model_left_tcp_pub = self.create_publisher(
            PoseStamped,
            MODEL_LEFT_TCP_TOPIC,
            10,
        )
        self._model_right_tcp_pub = self.create_publisher(
            PoseStamped,
            MODEL_RIGHT_TCP_TOPIC,
            10,
        )
        self._control_mode_pub = self.create_publisher(
            UInt32,
            CONTROL_MODE_TOPIC,
            10,
        )
        self._wbc_joint_pub = self.create_publisher(
            JointState,
            WBC_TARGET_JOINTS_TOPIC,
            10,
        )
        self._ik_sync_client = self.create_client(Trigger, "/ik/sync_joint_state")
        self._arm_set_enable_client = self.create_client(SetBool, ARM_SET_ENABLE_SERVICE)
        self._hand_set_enable_client = self.create_client(SetBool, HAND_SET_ENABLE_SERVICE)
        self._replay_start_client = self.create_client(Trigger, REPLAY_START_SERVICE)
        self._replay_stop_client = self.create_client(Trigger, REPLAY_STOP_SERVICE)
        self._wbc_target_joints_sub = self.create_subscription(
            JointState,
            WBC_TARGET_JOINTS_TOPIC,
            self._on_wbc_target_joints,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self._left_hand_at_a = True
        self._slow_motion_active = False
        self._slow_motion_preparing = False
        self._slow_motion_prep: Optional[Dict[str, object]] = None
        self._slow_motion_phase = ""
        self._slow_motion_poses: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
        ] = []
        self._slow_motion_joint_poses: List[List[float]] = []
        self._slow_motion_step = 0
        self._slow_motion_hold_steps = 0
        self._slow_motion_timer = None
        self._slow_motion_prep_timer = None
        self._slow_motion_mode_timer = None
        self._slow_motion_saved_control_mode: Optional[int] = None
        self._slow_motion_right_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
        ] = None
        self._slow_motion_final_left_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
        ] = None
        self._slow_motion_final_joints: Optional[List[float]] = None
        self._slow_motion_wbc_baseline: Optional[List[float]] = None
        self._slow_motion_wbc_changed = False
        self._last_wbc_joints: Optional[List[float]] = None
        self._arm_move_joint_speed_rad_s = ARM_MOVE_SPEED_DEFAULT_RAD_S
        self._hal_arm_received_at = 0.0
        self._control_mode = IDLE_CONTROL_MODE
        self._arm_enabled = False
        self._hand_enabled = False
        self._control_mode_received_at = 0.0
        self._arm_enable_received_at = 0.0
        self._arm_enabled_since = 0.0
        self._replay_arm_settle_sec = 3.0
        self._replay_hand_settle_sec = 3.0
        self._hand_enable_received_at = 0.0
        self._hand_enabled_since = 0.0
        self._replay_running_state = 0
        self._scan_timer = None
        self._setup_robot_subscriptions()

        self._scan_timer = self.create_timer(2.0, self._on_timer)

    def prepare_shutdown(self) -> None:
        """关闭窗口时停止 ROS 定时器与慢速移动，避免退出阻塞。"""
        if self._slow_motion_active or self._slow_motion_preparing:
            self._slow_motion_active = False
            self._slow_motion_preparing = False
            self._slow_motion_prep = None
            self._slow_motion_phase = ""
            self._slow_motion_poses = []
            self._slow_motion_joint_poses = []
            self._slow_motion_step = 0
            self._slow_motion_hold_steps = 0
            self._stop_slow_motion_timer()
            self._stop_slow_motion_prep_timer()
            self._stop_slow_motion_mode_timer()
            self._restore_control_mode_if_needed()
        if self._scan_timer is not None:
            try:
                self.destroy_timer(self._scan_timer)
            except Exception:
                pass
            self._scan_timer = None
        self._tf_listener = None

    def is_left_hand_at_a(self) -> bool:
        return self._left_hand_at_a

    def is_arm_enabled(self) -> bool:
        return self._arm_enabled

    def arm_ready_for_replay(self, settle_sec: Optional[float] = None) -> bool:
        if not self._arm_enabled:
            return False
        settle = self._replay_arm_settle_sec if settle_sec is None else settle_sec
        since = self._arm_enabled_since if self._arm_enabled_since > 0 else self._arm_enable_received_at
        if since <= 0:
            return False
        return (time.time() - since) >= settle

    def is_hal_arm_ready(self) -> bool:
        return self._hal_arm_received_at > 0.0

    def set_arm_move_joint_speed(self, speed_rad_s: float) -> None:
        self._arm_move_joint_speed_rad_s = max(
            ARM_MOVE_SPEED_MIN_RAD_S,
            min(ARM_MOVE_SPEED_MAX_RAD_S, float(speed_rad_s)),
        )

    def get_arm_move_joint_speed(self) -> float:
        return self._arm_move_joint_speed_rad_s

    def get_arm_enable_label(self) -> str:
        if not self._arm_enable_received_at:
            return "手臂: 等待 /arm/enable_state ..."
        return f"手臂: {'已使能' if self._arm_enabled else '未使能 (点「移动」将自动使能)'}"

    def request_arm_enable(self, enable: bool = True) -> str:
        if enable and self._arm_enabled:
            return "手臂已使能"
        if not enable and not self._arm_enabled:
            return "手臂已关闭"
        if not self._arm_set_enable_client.service_is_ready():
            if not self._arm_set_enable_client.wait_for_service(timeout_sec=1.5):
                return f"服务 {ARM_SET_ENABLE_SERVICE} 不可用（请确认 topic_router 已启动）"
        req = SetBool.Request()
        req.data = bool(enable)
        future = self._arm_set_enable_client.call_async(req)

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(
                        f"arm/set_enable({enable}) OK: {result.message}"
                    )
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"arm/set_enable 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"arm/set_enable 异常: {exc}")

        future.add_done_callback(_done)
        if enable:
            return "已请求启用手臂（初始化约需数秒，请等待状态变为「已使能」）"
        return "已请求关闭手臂"

    def is_hand_enabled(self) -> bool:
        return self._hand_enabled

    def hand_ready_for_replay(self, settle_sec: Optional[float] = None) -> bool:
        if not self._hand_enabled:
            return False
        settle = self._replay_hand_settle_sec if settle_sec is None else settle_sec
        since = self._hand_enabled_since if self._hand_enabled_since > 0 else self._hand_enable_received_at
        if since <= 0:
            return False
        return (time.time() - since) >= settle

    def get_hand_enable_label(self) -> str:
        if not self._hand_enable_received_at:
            return "手部: 等待 /hand/enable_state ..."
        return f"手部: {'已使能' if self._hand_enabled else '未使能'}"

    def request_hand_enable(self, enable: bool = True) -> str:
        if enable and self._hand_enabled:
            return "手部已使能"
        if not enable and not self._hand_enabled:
            return "手部已关闭"
        if not self._hand_set_enable_client.service_is_ready():
            if not self._hand_set_enable_client.wait_for_service(timeout_sec=1.5):
                return f"服务 {HAND_SET_ENABLE_SERVICE} 不可用（请确认 topic_router 已启动）"
        req = SetBool.Request()
        req.data = bool(enable)
        future = self._hand_set_enable_client.call_async(req)

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(
                        f"hand/set_enable({enable}) OK: {result.message}"
                    )
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"hand/set_enable 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"hand/set_enable 异常: {exc}")

        future.add_done_callback(_done)
        if enable:
            return "已请求启用手部"
        return "已请求关闭手部"

    def get_replay_state(self) -> int:
        return self._replay_running_state

    def get_replay_state_label(self) -> str:
        return REPLAY_STATE_LABELS.get(
            self._replay_running_state,
            f"未知({self._replay_running_state})",
        )

    def replay_start_service_ready(self) -> bool:
        return self._replay_start_client.service_is_ready()

    def prepare_for_rrd_replay(self) -> List[str]:
        """兼容旧接口：回放前不再自动使能/切模式（与 GUI 行为一致）。"""
        return self.get_replay_prerequisite_warnings()

    def get_replay_prerequisite_warnings(self) -> List[str]:
        warnings: List[str] = []
        if self._control_mode_received_at and self._control_mode != MODEL_CONTROL_MODE:
            warnings.append(f"control_mode={self._control_mode}，需为 {MODEL_CONTROL_MODE}")
        if self._arm_enable_received_at and not self._arm_enabled:
            warnings.append("手臂未使能")
        if self._hand_enable_received_at and not self._hand_enabled:
            warnings.append("手部未使能")
        return warnings

    def request_replay_start(self) -> str:
        if not self._replay_start_client.service_is_ready():
            if not self._replay_start_client.wait_for_service(timeout_sec=0.2):
                return f"服务 {REPLAY_START_SERVICE} 尚未就绪"
        future = self._replay_start_client.call_async(Trigger.Request())

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(f"replay start OK: {result.message}")
                    self.ros_bridge.status_message.emit(
                        result.message or "回放已启动"
                    )
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"replay start 失败: {msg}")
                    self.ros_bridge.status_message.emit(f"启动回放失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"replay start 异常: {exc}")
                self.ros_bridge.status_message.emit(f"启动回放异常: {exc}")

        future.add_done_callback(_done)
        return "已请求开始回放"

    def request_replay_stop(self) -> str:
        if not self._replay_stop_client.service_is_ready():
            if not self._replay_stop_client.wait_for_service(timeout_sec=1.5):
                return f"服务 {REPLAY_STOP_SERVICE} 不可用"
        future = self._replay_stop_client.call_async(Trigger.Request())

        def _done(fut) -> None:
            try:
                result = fut.result()
                if result and result.success:
                    self.get_logger().info(f"replay stop OK: {result.message}")
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"replay stop 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"replay stop 异常: {exc}")

        future.add_done_callback(_done)
        return "已请求停止回放"

    def is_slow_motion_active(self) -> bool:
        return self._slow_motion_active

    def is_slow_motion_preparing(self) -> bool:
        return self._slow_motion_preparing

    def is_slow_motion_busy(self) -> bool:
        return self._slow_motion_active or self._slow_motion_preparing

    def _is_motion_control_locked(self) -> bool:
        return self.is_slow_motion_busy()

    def _tcp_pose_in_ik_frame(
        self,
        side: str,
        timeout_s: float = MOVE_TF_LOOKUP_TIMEOUT_S,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        state = self.get_robot_state()
        tcp = state.left_tcp if side == "left" else state.right_tcp
        if tcp is None or not tcp.valid:
            return None
        transformed = transform_pose_to_base(
            tcp.xyz,
            tcp.quat_xyzw,
            tcp.frame_id,
            self._tf_buffer,
            base_frame=IK_TARGET_FRAME,
            timeout_s=timeout_s,
        )
        if transformed is not None:
            return transformed
        src = normalize_frame_id(tcp.frame_id)
        if src == normalize_frame_id(IK_TARGET_FRAME):
            return tcp.xyz, tcp.quat_xyzw
        if src in MINK_FK_FRAME_ALIASES:
            return tcp.xyz, tcp.quat_xyzw
        return None

    def _get_left_tcp_in_base(
        self,
        timeout_s: float = UI_TF_LOOKUP_TIMEOUT_S,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        return self._tcp_pose_in_ik_frame("left", timeout_s=timeout_s)

    def _get_right_tcp_in_base(
        self,
        timeout_s: float = UI_TF_LOOKUP_TIMEOUT_S,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        return self._tcp_pose_in_ik_frame("right", timeout_s=timeout_s)

    def get_arm_move_blockers(self, tf_timeout_s: float = MOVE_TF_LOOKUP_TIMEOUT_S) -> List[str]:
        blockers: List[str] = []
        if not self._arm_enabled:
            if self._arm_enable_received_at <= 0:
                blockers.append(
                    f"未收到 {ARM_ENABLE_STATE_TOPIC}（点「移动」将尝试自动使能）"
                )
            else:
                blockers.append("手臂未使能（点「移动」将自动使能，或按 F2 /「启用手臂」）")
        state = self.get_robot_state()
        if state.left_tcp is None or not state.left_tcp.valid:
            blockers.append("无左臂 TCP 数据（/tele/fk/left_pose 或 /mink_fk/left_tcp_pose）")
        elif self._tcp_pose_in_ik_frame("left", timeout_s=tf_timeout_s) is None:
            frame = state.left_tcp.frame_id or "未知"
            blockers.append(
                f"左臂 TCP 在 {frame} 系，无法用于 IK（需 base_link 或 /mink_fk）"
            )
        if state.right_tcp is None or not state.right_tcp.valid:
            blockers.append("无右臂 TCP 数据")
        elif self._tcp_pose_in_ik_frame("right", timeout_s=tf_timeout_s) is None:
            frame = state.right_tcp.frame_id or "未知"
            blockers.append(
                f"右臂 TCP 在 {frame} 系，无法用于 IK（需 base_link 或 /mink_fk）"
            )
        return blockers

    def get_arm_move_current_label(self) -> str:
        state = self.get_robot_state()
        hand_line = format_left_hand_state_line(state)
        tcp = self._get_left_tcp_in_base()
        if tcp is not None:
            xyz, quat = tcp
            return f"当前  {hand_line}  |  {format_xyz_rpy_line('左臂', xyz, quat)}"
        if state.left_tcp and state.left_tcp.valid:
            xyz, quat = state.left_tcp.xyz, state.left_tcp.quat_xyzw
            frame = state.left_tcp.frame_id or BASE_LINK_FRAME
            return (
                f"当前  {hand_line}  |  {format_xyz_rpy_line('左臂', xyz, quat)}"
                f"  [{frame}]"
            )
        return f"当前  {hand_line}  |  左臂 TCP: (无数据)"

    def resolve_segment_move_goal(
        self,
        segment: SegmentPoseTarget,
        timeout_s: float = UI_TF_LOOKUP_TIMEOUT_S,
    ) -> Optional[ResolvedArmMoveGoal]:
        goal = transform_pose_to_base(
            segment.position_xyz,
            segment.quaternion_xyzw,
            segment.camera_frame,
            self._tf_buffer,
            base_frame=IK_TARGET_FRAME,
            timeout_s=timeout_s,
        )
        if goal is None:
            return None
        xyz, quat = goal
        return ResolvedArmMoveGoal(
            position_xyz=xyz,
            quaternion_xyzw=quat,
            label=f"接触TCP ({segment.label or segment.source_topic})",
        )

    def get_arm_move_pose_labels(
        self,
        segment: Optional[SegmentPoseTarget],
    ) -> Tuple[str, str]:
        """保留兼容：仅在没有 window 侧解析时使用。"""
        current_line = self.get_arm_move_current_label()
        target_line = "目标  左臂 TCP: (请先设置)"
        if segment is not None:
            resolved = self.resolve_segment_move_goal(segment)
            if resolved is not None:
                target_line = (
                    f"目标  [{resolved.label}]  "
                    f"{format_xyz_rpy_line('左臂', resolved.position_xyz, resolved.quaternion_xyzw)}"
                )
            else:
                target_line = (
                    f"目标  左臂 TCP: TF 不可用 "
                    f"({segment.camera_frame} -> {BASE_LINK_FRAME})"
                )
        return current_line, target_line

    def _make_pose_stamped_msg(
        self,
        xyz: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
    ) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = IK_TARGET_FRAME
        msg.pose.position.x = xyz[0]
        msg.pose.position.y = xyz[1]
        msg.pose.position.z = xyz[2]
        msg.pose.orientation.x = quat_xyzw[0]
        msg.pose.orientation.y = quat_xyzw[1]
        msg.pose.orientation.z = quat_xyzw[2]
        msg.pose.orientation.w = quat_xyzw[3]
        return msg

    def _publish_ik_arm_targets(
        self,
        left_xyz: Tuple[float, float, float],
        left_quat: Tuple[float, float, float, float],
        right_xyz: Tuple[float, float, float],
        right_quat: Tuple[float, float, float, float],
    ) -> None:
        left_msg = self._make_pose_stamped_msg(left_xyz, left_quat)
        right_msg = self._make_pose_stamped_msg(right_xyz, right_quat)
        self._left_ik_pub.publish(left_msg)
        self._right_ik_pub.publish(right_msg)

    def _publish_dual_arm_targets(
        self,
        left_xyz: Tuple[float, float, float],
        left_quat: Tuple[float, float, float, float],
        right_xyz: Tuple[float, float, float],
        right_quat: Tuple[float, float, float, float],
    ) -> None:
        self._publish_ik_arm_targets(left_xyz, left_quat, right_xyz, right_quat)
        left_msg = self._make_pose_stamped_msg(left_xyz, left_quat)
        right_msg = self._make_pose_stamped_msg(right_xyz, right_quat)
        self._model_left_tcp_pub.publish(left_msg)
        self._model_right_tcp_pub.publish(right_msg)

    def _publish_control_mode(self, mode: int) -> None:
        msg = UInt32()
        msg.data = int(mode)
        self._control_mode_pub.publish(msg)
        self._control_mode = int(mode)

    def _restore_control_mode_if_needed(self) -> None:
        if self._slow_motion_saved_control_mode is not None:
            self._publish_control_mode(self._slow_motion_saved_control_mode)
            self._slow_motion_saved_control_mode = None

    def _maintain_model_mode_during_motion(self) -> None:
        if self._control_mode != MODEL_CONTROL_MODE:
            self._publish_control_mode(MODEL_CONTROL_MODE)

    def _start_slow_motion_mode_lock(self) -> None:
        self._stop_slow_motion_mode_timer()
        self._burst_control_mode(MODEL_CONTROL_MODE)
        self._slow_motion_mode_timer = self.create_timer(
            1.0, self._maintain_model_mode_during_motion
        )

    def _stop_slow_motion_mode_timer(self) -> None:
        if self._slow_motion_mode_timer is not None:
            self.destroy_timer(self._slow_motion_mode_timer)
            self._slow_motion_mode_timer = None

    def _publish_slow_motion_targets(
        self,
        left_xyz: Tuple[float, float, float],
        left_quat: Tuple[float, float, float, float],
    ) -> None:
        right_pose = self._slow_motion_right_pose
        if right_pose is None:
            return
        right_xyz, right_quat = right_pose
        self._publish_ik_arm_targets(left_xyz, left_quat, right_xyz, right_quat)

    def _make_wbc_joint_msg(self, positions: List[float]) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = IK_TARGET_FRAME
        msg.position = [float(v) for v in positions[:16]]
        return msg

    def _publish_wbc_joint_target(self, positions: List[float]) -> None:
        self._wbc_joint_pub.publish(self._make_wbc_joint_msg(positions))

    def _stop_slow_motion_prep_timer(self) -> None:
        if self._slow_motion_prep_timer is not None:
            self.destroy_timer(self._slow_motion_prep_timer)
            self._slow_motion_prep_timer = None

    def _release_ik_for_joint_move(self) -> None:
        """切到 IDLE 清除 ik_node 双臂目标，再切回 model 关节模式，避免与关节轨迹抢 /wbc/target_joints。"""
        self._publish_control_mode(IDLE_CONTROL_MODE)
        self._burst_control_mode(MODEL_CONTROL_MODE)

    def _start_joint_move_from_prep(self, prep: Dict[str, object], goal: ResolvedArmMoveGoal) -> None:
        start_joints: List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
        goal_joints: List[float] = prep["goal_joints"]  # type: ignore[assignment]
        duration_s = compute_arm_move_duration_s(
            start_joints, goal_joints, self._arm_move_joint_speed_rad_s
        )
        if duration_s <= 0:
            self._abort_slow_motion_prep("目标关节与当前几乎相同，无需移动")
            return
        max_delta = max(abs(goal_joints[i] - start_joints[i]) for i in range(16))
        steps = max(2, int(duration_s * ARM_MOVE_JOINT_HZ))
        joint_poses = build_interpolated_joints(start_joints, goal_joints, steps, linear=True)
        self.get_logger().info(
            f"关节轨迹: {steps} 步 @ {ARM_MOVE_JOINT_HZ:.0f}Hz / {duration_s:.1f}s, "
            f"最大关节变化 {max_delta:.3f} rad, 发布 {WBC_TARGET_JOINTS_TOPIC}"
        )
        self._slow_motion_joint_poses = joint_poses
        self._slow_motion_final_joints = goal_joints
        self._slow_motion_preparing = False
        self._slow_motion_prep = None
        self._stop_slow_motion_prep_timer()
        self._slow_motion_active = True
        self._slow_motion_phase = "move"
        self._slow_motion_hold_steps = 0
        self._publish_wbc_joint_target(start_joints)
        self._slow_motion_step = 0
        if self._slow_motion_timer is not None:
            self.destroy_timer(self._slow_motion_timer)
        self._slow_motion_timer = self.create_timer(
            1.0 / ARM_MOVE_JOINT_HZ, self._slow_motion_tick
        )
        goal_xyz = prep["goal_xyz"]  # type: ignore[assignment]
        gx, gy, gz = goal_xyz
        saved_mode = int(prep.get("saved_mode", self._control_mode))
        self.get_logger().info(
            f"左臂移动开始 [{goal.label}]: mode {saved_mode}->"
            f"{self._control_mode if self._slow_motion_saved_control_mode is not None else saved_mode}, "
            f"{steps} 关节步 / {duration_s:.1f}s -> {WBC_TARGET_JOINTS_TOPIC}, "
            f"目标 XYZ=({gx:.3f}, {gy:.3f}, {gz:.3f}) [{IK_TARGET_FRAME}]"
        )
        mode_hint = ""
        if self._slow_motion_saved_control_mode is not None:
            mode_hint = "，已临时切到 model 模式(0)"
        message = f"开始移动 [{goal.label}]（约 {duration_s:.1f}s @ {self._arm_move_joint_speed_rad_s:.2f} rad/s）{mode_hint}"
        self.ros_bridge.status_message.emit(message)
        self.ros_bridge.slow_motion_progress.emit(0.0, message)

    def _start_slow_motion_prep_timer(self) -> None:
        self._stop_slow_motion_prep_timer()
        self._slow_motion_prep_timer = self.create_timer(
            1.0 / ARM_MOVE_IK_HZ, self._slow_motion_prep_tick
        )

    def _abort_slow_motion_prep(self, message: str) -> None:
        self._slow_motion_preparing = False
        self._slow_motion_prep = None
        self._slow_motion_right_pose = None
        self._slow_motion_final_left_pose = None
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._stop_slow_motion_prep_timer()
        self._stop_slow_motion_mode_timer()
        self._restore_control_mode_if_needed()
        self.get_logger().warn(message)
        self.ros_bridge.slow_motion_finished.emit(False, message)

    def request_slow_move_to_goal(self, goal: ResolvedArmMoveGoal) -> str:
        if self.is_slow_motion_busy():
            return "已有移动进行中，请等待完成"
        blockers = self.get_arm_move_blockers()
        if blockers:
            msg = "无法移动: " + "；".join(blockers)
            self.get_logger().warn(msg)
            return msg
        self._slow_motion_preparing = True
        self._slow_motion_prep = {
            "goal": goal,
            "phase": "validate",
            "warmup_step": 0,
        }
        self._slow_motion_poses = []
        self._slow_motion_joint_poses = []
        self._slow_motion_step = 0
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._start_slow_motion_prep_timer()
        self.ros_bridge.slow_motion_progress.emit(0.0, "正在准备移动...")
        return "正在准备移动..."

    def _slow_motion_prep_tick(self) -> None:
        prep = self._slow_motion_prep
        if not self._slow_motion_preparing or prep is None:
            self._stop_slow_motion_prep_timer()
            return

        goal: ResolvedArmMoveGoal = prep["goal"]  # type: ignore[assignment]
        phase = str(prep.get("phase", ""))

        if phase == "validate":
            start_pose = self._get_left_tcp_in_base(timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S)
            right_pose = self._get_right_tcp_in_base(timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S)
            if start_pose is None or right_pose is None:
                self._abort_slow_motion_prep("无法获取双臂 TCP（IK 目标坐标系 transform 失败）")
                return
            prep["start_xyz"], prep["start_quat"] = start_pose
            prep["goal_xyz"] = goal.position_xyz
            prep["goal_quat"] = goal.quaternion_xyzw
            prep["right_pose"] = right_pose
            prep["saved_mode"] = self._control_mode
            self._enter_model_mode_for_motion()
            prep["phase"] = "sync_ik"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在同步 IK 关节状态...")
            return

        if phase == "sync_ik":
            sync_future = prep.get("sync_future")
            if sync_future is None:
                if not self._ik_sync_client.wait_for_service(timeout_sec=0.0):
                    self._abort_slow_motion_prep("IK sync 服务不可用，请确认 ik_node 运行")
                    return
                prep["sync_future"] = self._ik_sync_client.call_async(Trigger.Request())
                prep["sync_deadline"] = time.time() + IK_SYNC_TIMEOUT_S
                return
            if not sync_future.done():  # type: ignore[union-attr]
                if time.time() > float(prep.get("sync_deadline", 0.0)):
                    self._abort_slow_motion_prep(
                        f"IK 关节状态同步超时 ({IK_SYNC_TIMEOUT_S:.1f}s)"
                    )
                return
            try:
                result = sync_future.result()  # type: ignore[union-attr]
                if not (result and result.success):
                    msg = result.message if result else "无响应"
                    self._abort_slow_motion_prep(f"IK 关节状态同步失败: {msg}")
                    return
                self.get_logger().info("IK 关节状态已同步 (/ik/sync_joint_state)")
            except Exception as exc:
                self._abort_slow_motion_prep(f"IK sync 异常: {exc}")
                return
            right_xyz, right_quat = prep["right_pose"]  # type: ignore[misc]
            self._slow_motion_right_pose = (right_xyz, right_quat)
            self._slow_motion_final_left_pose = (prep["goal_xyz"], prep["goal_quat"])  # type: ignore[index]
            prep["warmup_step"] = 0
            prep["phase"] = "warmup"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在预热 IK 目标...")
            return

        if phase == "warmup":
            warmup_step = int(prep.get("warmup_step", 0))
            start_xyz, start_quat = prep["start_xyz"], prep["start_quat"]  # type: ignore[misc]
            self._publish_slow_motion_targets(start_xyz, start_quat)
            warmup_step += 1
            prep["warmup_step"] = warmup_step
            if warmup_step >= ARM_MOVE_WARMUP_STEPS:
                prep["phase"] = "check_wbc"
            return

        if phase == "check_wbc":
            if not self._slow_motion_wbc_changed:
                self._abort_slow_motion_prep(
                    f"IK 未驱动机械臂（{WBC_TARGET_JOINTS_TOPIC} 无变化）。"
                    "请确认 F2 已使能、ik_node/arm_control 已运行"
                )
                return
            if self._last_wbc_joints is None or len(self._last_wbc_joints) < 16:
                self._abort_slow_motion_prep("无法读取起始关节状态 (/wbc/target_joints)")
                return
            prep["start_joints"] = list(self._last_wbc_joints[:16])
            prep["goal_published"] = False
            prep["settle_ticks"] = 0
            prep["phase"] = "settle"
            self.ros_bridge.slow_motion_progress.emit(0.0, "正在规划关节轨迹...")
            return

        if phase == "settle":
            if not prep.get("goal_published"):
                prep["motion_start_joints"] = list(self._last_wbc_joints[:16])  # type: ignore[arg-type]
                goal_xyz, goal_quat = prep["goal_xyz"], prep["goal_quat"]  # type: ignore[misc]
                self._publish_slow_motion_targets(goal_xyz, goal_quat)
                prep["goal_published"] = True
                prep["settle_ticks"] = ARM_MOVE_GOAL_SAMPLE_TICKS
                return
            settle_ticks = int(prep.get("settle_ticks", 0)) - 1
            prep["settle_ticks"] = settle_ticks
            if settle_ticks > 0:
                return
            if self._last_wbc_joints is None or len(self._last_wbc_joints) < 16:
                self._abort_slow_motion_prep("IK 未返回目标关节角")
                return
            prep["goal_joints"] = list(self._last_wbc_joints[:16])
            prep["handoff_tick"] = 0
            prep["phase"] = "handoff"
            self.ros_bridge.slow_motion_progress.emit(0.0, "规划完成，释放 IK 控制...")
            return

        if phase == "handoff":
            handoff_tick = int(prep.get("handoff_tick", 0))
            if handoff_tick == 0:
                motion_start: List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
                self._publish_wbc_joint_target(motion_start)
                self._release_ik_for_joint_move()
                prep["handoff_tick"] = 1
                prep["handoff_wait"] = ARM_MOVE_IK_RELEASE_TICKS
                return
            wait = int(prep.get("handoff_wait", 0)) - 1
            prep["handoff_wait"] = wait
            if wait > 0:
                return
            self._start_joint_move_from_prep(prep, goal)
            return

    def start_slow_move_to_goal(self, goal: ResolvedArmMoveGoal) -> str:
        return self.request_slow_move_to_goal(goal)

    def _sync_ik_joint_state_blocking(self, timeout_s: float = IK_SYNC_TIMEOUT_S) -> bool:
        if not self._ik_sync_client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().warn("IK sync 服务不可用，跳过 /ik/sync_joint_state")
            return False
        done = threading.Event()
        ok_box: List[bool] = [False]

        def _done(fut) -> None:
            try:
                result = fut.result()
                ok_box[0] = bool(result and result.success)
                if ok_box[0]:
                    self.get_logger().info("IK 关节状态已同步 (/ik/sync_joint_state)")
                else:
                    msg = result.message if result else "无响应"
                    self.get_logger().warn(f"IK sync 失败: {msg}")
            except Exception as exc:
                self.get_logger().warn(f"IK sync 异常: {exc}")
            finally:
                done.set()

        future = self._ik_sync_client.call_async(Trigger.Request())
        future.add_done_callback(_done)
        if not done.wait(timeout_s):
            self.get_logger().warn(f"IK sync 超时 ({timeout_s:.1f}s)")
            return False
        return ok_box[0]

    def _try_sync_ik_joint_state(self) -> None:
        self._sync_ik_joint_state_blocking()

    def _enter_model_mode_for_motion(self) -> None:
        self._slow_motion_saved_control_mode = self._control_mode
        if self._control_mode != MODEL_CONTROL_MODE:
            self.get_logger().info(
                f"左臂移动: 切换 control_mode {self._slow_motion_saved_control_mode} -> "
                f"{MODEL_CONTROL_MODE}（暂停 tracker→IK 转发，避免覆盖目标）"
            )
        self._start_slow_motion_mode_lock()

    def _burst_control_mode(self, mode: int, count: int = ARM_MOVE_MODE_BURST) -> None:
        for _ in range(count):
            self._publish_control_mode(mode)

    def start_slow_move_to_segment(self, target: SegmentPoseTarget) -> str:
        resolved = self.resolve_segment_move_goal(target, timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S)
        if resolved is None:
            return (
                f"无法将分割位姿从 {target.camera_frame or '未知'} "
                f"变换到 {BASE_LINK_FRAME}（请检查 TF）"
            )
        return self.start_slow_move_to_goal(resolved)

    def _publish_left_ik_target(
        self,
        xyz: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
    ) -> None:
        right_pose = self._get_right_tcp_in_base()
        if right_pose is None:
            return
        self._publish_dual_arm_targets(xyz, quat_xyzw, right_pose[0], right_pose[1])

    def _stop_slow_motion_timer(self) -> None:
        if self._slow_motion_timer is not None:
            self.destroy_timer(self._slow_motion_timer)
            self._slow_motion_timer = None

    def _finish_slow_motion(self, success: bool, message: str) -> None:
        self._slow_motion_active = False
        self._slow_motion_phase = ""
        self._slow_motion_poses = []
        self._slow_motion_joint_poses = []
        self._slow_motion_step = 0
        self._slow_motion_hold_steps = 0
        self._slow_motion_right_pose = None
        self._slow_motion_final_left_pose = None
        self._slow_motion_final_joints = None
        self._slow_motion_wbc_baseline = None
        self._slow_motion_wbc_changed = False
        self._stop_slow_motion_timer()
        self._stop_slow_motion_mode_timer()
        self._restore_control_mode_if_needed()
        self.ros_bridge.slow_motion_finished.emit(success, message)

    def cancel_slow_motion(self) -> str:
        if self._slow_motion_preparing:
            self._abort_slow_motion_prep("移动准备已取消")
            return "移动准备已取消"
        if not self._slow_motion_active:
            return "当前无进行中的移动"
        progress = 0.0
        if self._slow_motion_joint_poses:
            progress = self._slow_motion_step / len(self._slow_motion_joint_poses)
        self.get_logger().info(
            f"移动已取消 ({progress * 100:.0f}%) -> {WBC_TARGET_JOINTS_TOPIC}"
        )
        self._finish_slow_motion(False, f"移动已取消 ({progress * 100:.0f}%)")
        return f"移动已取消 ({progress * 100:.0f}%)"

    def _slow_motion_tick(self) -> None:
        if not self._slow_motion_active:
            return

        if self._slow_motion_step >= len(self._slow_motion_joint_poses):
            final_joints = self._slow_motion_final_joints
            if final_joints is not None:
                self._publish_wbc_joint_target(final_joints)
            self.get_logger().info(f"移动完成 -> {WBC_TARGET_JOINTS_TOPIC}")
            self._finish_slow_motion(True, "移动完成")
            return

        joints = self._slow_motion_joint_poses[self._slow_motion_step]
        self._publish_wbc_joint_target(joints)
        self._slow_motion_step += 1
        progress = self._slow_motion_step / len(self._slow_motion_joint_poses)
        self.ros_bridge.slow_motion_progress.emit(
            progress,
            f"移动中 {progress * 100:.0f}%",
        )

    def apply_left_hand_angle(self, slider_pct: int) -> float:
        """按游标位置发送左手角度命令，返回实际 position。"""
        pos = slider_to_hand_position(slider_pct)
        msg = make_left_hand_command(pos)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._left_hand_cmd_pub.publish(msg)
        self.get_logger().info(
            f"左手命令 -> position={pos:.3f} (游标={slider_pct}): {ROBOT_LEFT_HAND_CMD_TOPIC}"
        )
        return pos

    def toggle_left_hand_between(self, slider_a: int, slider_b: int) -> Tuple[bool, float]:
        """在游标 A/B 两个状态间切换，返回 (当前是否为 A, 发送的 position)。"""
        self._left_hand_at_a = not self._left_hand_at_a
        slider = slider_a if self._left_hand_at_a else slider_b
        preset = "A" if self._left_hand_at_a else "B"
        pos = self.apply_left_hand_angle(slider)
        self.get_logger().info(f"左手切换 -> 状态{preset}")
        self.ros_bridge.left_hand_preset_changed.emit(self._left_hand_at_a)
        return self._left_hand_at_a, pos

    def _setup_robot_subscriptions(self) -> None:
        hand_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        for topic in (ROBOT_ARM_TOPIC, ROBOT_FK_JOINTS_TOPIC):
            self._subscribe_robot_topic(
                JointState,
                topic,
                self._on_arm_joint_state if topic == ROBOT_ARM_TOPIC else self._on_fk_joint_state,
            )
        self.create_subscription(
            JointState,
            ROBOT_LEFT_HAND_TOPIC,
            self._on_left_hand_state,
            hand_qos,
        )
        self.create_subscription(
            JointState,
            ROBOT_RIGHT_HAND_TOPIC,
            self._on_right_hand_state,
            hand_qos,
        )
        for side, topics in ROBOT_TCP_SOURCE_TOPICS.items():
            for topic in topics:
                callback = (
                    (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                    if side == "left"
                    else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
                )
                self._subscribe_robot_topic(PoseStamped, topic, callback)
        for topic, callback in (
            (CONTROL_MODE_TOPIC, self._on_control_mode),
        ):
            self._subscribe_control_topic(UInt32, topic, callback)
        self.create_subscription(
            UInt32,
            ARM_ENABLE_STATE_TOPIC,
            self._on_arm_enable_state,
            ENABLE_STATE_QOS,
        )
        self.create_subscription(
            UInt32,
            HAND_ENABLE_STATE_TOPIC,
            self._on_hand_enable_state,
            ENABLE_STATE_QOS,
        )
        self.create_subscription(
            UInt8,
            REPLAY_RUNNING_STATE_TOPIC,
            self._on_replay_running_state,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.get_logger().info(
            "已订阅机器人状态: arm/hand/tcp/control_mode/replay topics "
            f"(TCP: {ROBOT_LEFT_TCP_TOPIC} / {ROBOT_TELE_LEFT_TCP_TOPIC})"
        )

    def _robot_qos_candidates_for_topic(self, topic: str) -> List[QoSProfile]:
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        derived = self._qos_from_publisher_info(topic)
        if derived is not None:
            return [derived, reliable, best_effort]
        return [reliable, best_effort]

    def _control_qos_candidates_for_topic(self, topic: str) -> List[QoSProfile]:
        candidates: List[QoSProfile] = []
        derived = self._qos_from_publisher_info(topic)
        if derived is not None:
            candidates.append(derived)
        candidates.extend([
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            ),
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ])
        return candidates

    def _robot_qos_for_topic(self, topic: str, attempt: int = 0) -> QoSProfile:
        candidates = self._robot_qos_candidates_for_topic(topic)
        qos = candidates[min(attempt, len(candidates) - 1)]
        self.get_logger().info(
            f"{topic}: robot QoS attempt={attempt} reliability={qos.reliability.name}"
        )
        return qos

    def _subscribe_robot_topic(
        self,
        msg_type: type,
        topic: str,
        callback,
        attempt: int = 0,
    ) -> None:
        if topic in self._robot_subscriptions_map:
            try:
                self.destroy_subscription(self._robot_subscriptions_map.pop(topic))
            except Exception:
                pass
        qos = self._robot_qos_for_topic(topic, attempt)
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self._robot_subscriptions_map[topic] = sub
        self._robot_subscribe_attempts[topic] = attempt
        self._robot_last_subscribe_time[topic] = time.time()

    def _subscribe_control_topic(
        self,
        msg_type: type,
        topic: str,
        callback,
        attempt: int = 0,
    ) -> None:
        if topic in self._robot_subscriptions_map:
            try:
                self.destroy_subscription(self._robot_subscriptions_map.pop(topic))
            except Exception:
                pass
        candidates = self._control_qos_candidates_for_topic(topic)
        qos = candidates[min(attempt, len(candidates) - 1)]
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self._robot_subscriptions_map[topic] = sub
        self._robot_subscribe_attempts[topic] = attempt
        self._robot_last_subscribe_time[topic] = time.time()

    def _maybe_retry_robot_subscriptions(self) -> None:
        now = time.time()
        all_topics = dict(self.get_topic_names_and_types())
        retry_specs = [
            (ROBOT_LEFT_TCP_TOPIC, PoseStamped, "left"),
            (ROBOT_TELE_LEFT_TCP_TOPIC, PoseStamped, "left"),
            (ROBOT_RIGHT_TCP_TOPIC, PoseStamped, "right"),
            (ROBOT_TELE_RIGHT_TCP_TOPIC, PoseStamped, "right"),
        ]
        for topic, msg_type, side in retry_specs:
            if topic not in all_topics:
                continue
            if self._robot_tcp_topic_received.get(topic, False):
                continue
            if topic not in self._robot_subscriptions_map:
                callback = (
                    (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                    if side == "left"
                    else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
                )
                self.get_logger().info(f"{topic}: 检测到发布，补订阅 TCP")
                self._subscribe_robot_topic(msg_type, topic, callback, attempt=0)
                continue
            retries = self._robot_resubscribe_count.get(topic, 0)
            candidates = self._robot_qos_candidates_for_topic(topic)
            if retries >= len(candidates):
                continue
            last_sub = self._robot_last_subscribe_time.get(topic, 0.0)
            if now - last_sub < 5.0:
                continue
            callback = (
                (lambda msg, t=topic: self._on_left_tcp_pose(msg, t))
                if side == "left"
                else (lambda msg, t=topic: self._on_right_tcp_pose(msg, t))
            )
            next_attempt = retries + 1
            self.get_logger().warn(
                f"{topic}: 未收到 TCP 数据，切换 QoS 重订阅 "
                f"({next_attempt}/{len(candidates)})"
            )
            self._robot_resubscribe_count[topic] = next_attempt
            self._subscribe_robot_topic(msg_type, topic, callback, attempt=next_attempt)

    def get_robot_state(self) -> RobotStateSnapshot:
        with self._robot_lock:
            return RobotStateSnapshot(
                left_arm=list(self._robot_state.left_arm),
                right_arm=list(self._robot_state.right_arm),
                left_hand=list(self._robot_state.left_hand),
                right_hand=list(self._robot_state.right_hand),
                waist=list(self._robot_state.waist),
                left_tcp=self._robot_state.left_tcp,
                right_tcp=self._robot_state.right_tcp,
                updated_at=self._robot_state.updated_at,
            )

    def get_camera_frame_id(self, topic: str) -> str:
        return self._camera_frame_ids.get(topic, "")

    def resolve_segment_camera_frame(
        self, source_topic: str, depth_topic: Optional[str] = None
    ) -> str:
        known = list(self._camera_frame_ids.keys())
        candidates: List[str] = []
        if depth_topic:
            candidates.append(depth_topic)
        if source_topic not in candidates:
            candidates.append(source_topic)
        if is_depth_topic(source_topic):
            color = find_paired_color_topic(source_topic, known)
            if color and color not in candidates:
                candidates.append(color)
        else:
            paired_depth = depth_topic or find_paired_depth_topic(source_topic, known)
            if paired_depth and paired_depth not in candidates:
                candidates.insert(0, paired_depth)
        for topic in candidates:
            frame = self._camera_frame_ids.get(topic, "").strip()
            if frame:
                return frame
        return ""

    def get_tf_buffer(self) -> Buffer:
        return self._tf_buffer

    def _touch_robot_state(self) -> None:
        self._robot_state.updated_at = time.time()
        now = time.time()
        if now - self._last_robot_state_emit < UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._last_robot_state_emit = now
        self.ros_bridge.robot_state_updated.emit()

    def _on_arm_joint_state(self, msg: JointState) -> None:
        positions = [float(v) for v in msg.position]
        if self._hal_arm_received_at <= 0:
            self.get_logger().info(f"HAL 关节状态就绪: {ROBOT_ARM_TOPIC}")
        self._hal_arm_received_at = time.time()
        left, right = _arm_from_raw_positions(list(msg.name), positions)
        with self._robot_lock:
            if left:
                self._robot_state.left_arm = left
            if right:
                self._robot_state.right_arm = right
            self._touch_robot_state()

    def _on_fk_joint_state(self, msg: JointState) -> None:
        joints = _joint_dict_from_msg(msg)
        left, right, waist = _split_arm_joints(joints)
        with self._robot_lock:
            if left:
                self._robot_state.left_arm = left
            if right:
                self._robot_state.right_arm = right
            if waist:
                self._robot_state.waist = waist
            self._touch_robot_state()

    def _on_left_hand_state(self, msg: JointState) -> None:
        hand = [(n, float(msg.position[i])) for i, n in enumerate(msg.name) if i < len(msg.position)]
        with self._robot_lock:
            self._robot_state.left_hand = hand
            self._touch_robot_state()

    def _on_right_hand_state(self, msg: JointState) -> None:
        hand = [(n, float(msg.position[i])) for i, n in enumerate(msg.name) if i < len(msg.position)]
        with self._robot_lock:
            self._robot_state.right_hand = hand
            self._touch_robot_state()

    def _on_left_tcp_pose(self, msg: PoseStamped, source_topic: str = ROBOT_LEFT_TCP_TOPIC) -> None:
        incoming = _pose_from_msg(msg)
        with self._robot_lock:
            if not _should_accept_tcp_update(self._robot_state.left_tcp, incoming):
                return
            self._robot_state.left_tcp = incoming
            if not self._robot_tcp_topic_received.get(source_topic, False):
                self._robot_tcp_topic_received[source_topic] = True
                p = msg.pose.position
                frame = msg.header.frame_id or BASE_LINK_FRAME
                self.get_logger().info(
                    f"首帧左臂 TCP [{source_topic}]: "
                    f"frame={frame} XYZ=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
                )
            self._touch_robot_state()

    def _on_right_tcp_pose(self, msg: PoseStamped, source_topic: str = ROBOT_RIGHT_TCP_TOPIC) -> None:
        incoming = _pose_from_msg(msg)
        with self._robot_lock:
            if not _should_accept_tcp_update(self._robot_state.right_tcp, incoming):
                return
            self._robot_state.right_tcp = incoming
            if not self._robot_tcp_topic_received.get(source_topic, False):
                self._robot_tcp_topic_received[source_topic] = True
                p = msg.pose.position
                frame = msg.header.frame_id or BASE_LINK_FRAME
                self.get_logger().info(
                    f"首帧右臂 TCP [{source_topic}]: "
                    f"frame={frame} XYZ=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
                )
            self._touch_robot_state()

    def _on_wbc_target_joints(self, msg: JointState) -> None:
        positions = [float(v) for v in msg.position]
        if len(positions) < 16:
            return
        pos16 = positions[:16]
        self._last_wbc_joints = pos16
        if self._slow_motion_active:
            return
        if self._slow_motion_wbc_baseline is None:
            self._slow_motion_wbc_baseline = pos16
            return
        delta = max(abs(positions[i] - self._slow_motion_wbc_baseline[i]) for i in range(16))
        if delta > 1e-4:
            self._slow_motion_wbc_changed = True

    def _on_control_mode(self, msg: UInt32) -> None:
        if not self._is_motion_control_locked():
            self._control_mode = int(msg.data)
        if self._control_mode_received_at <= 0:
            self._control_mode_received_at = time.time()
            self.get_logger().info(f"control_mode={self._control_mode}")

    def _on_arm_enable_state(self, msg: UInt32) -> None:
        enabled = int(msg.data) != 0
        if self._arm_enable_received_at <= 0 or enabled != self._arm_enabled:
            self.get_logger().info(f"arm/enable_state={'ON' if enabled else 'OFF'}")
        self._arm_enable_received_at = time.time()
        if enabled != self._arm_enabled:
            self._arm_enabled = enabled
            self._arm_enabled_since = time.time() if enabled else 0.0
            self.ros_bridge.arm_enable_changed.emit(enabled)
        else:
            self._arm_enabled = enabled

    def _on_hand_enable_state(self, msg: UInt32) -> None:
        enabled = int(msg.data) != 0
        if self._hand_enable_received_at <= 0 or enabled != self._hand_enabled:
            self.get_logger().info(f"hand/enable_state={'ON' if enabled else 'OFF'}")
        self._hand_enable_received_at = time.time()
        if enabled != self._hand_enabled:
            self._hand_enabled = enabled
            self._hand_enabled_since = time.time() if enabled else 0.0
        else:
            self._hand_enabled = enabled

    def _on_replay_running_state(self, msg: UInt8) -> None:
        state = int(msg.data)
        if state == self._replay_running_state:
            return
        self._replay_running_state = state
        label = REPLAY_STATE_LABELS.get(state, str(state))
        self.get_logger().info(f"replay running_state={state} ({label})")
        self.ros_bridge.replay_state_changed.emit(state)

    def set_prefix(self, prefix: str) -> None:
        with self._state_lock:
            self._pending_prefix = prefix

    def set_enabled_topics(self, topics: set[str]) -> None:
        with self._state_lock:
            self._pending_enabled = set(topics)

    def get_intrinsics(self, topic: str, width: int, height: int) -> Tuple[float, float, float, float]:
        if topic in self._camera_intrinsics:
            return self._camera_intrinsics[topic]
        return default_intrinsics(width, height)

    def _camera_info_callback(self, msg: CameraInfo, depth_topic: str) -> None:
        self._camera_intrinsics[depth_topic] = intrinsics_from_camera_info(msg)
        if msg.header.frame_id:
            self._camera_frame_ids[depth_topic] = msg.header.frame_id

    def _subscribe_camera_info(self, depth_topic: str, qos: QoSProfile) -> None:
        if depth_topic in self.camera_info_subs:
            return
        all_topics = dict(self.get_topic_names_and_types())
        for candidate in camera_info_candidates(depth_topic):
            if candidate not in all_topics:
                continue
            sub = self.create_subscription(
                CameraInfo,
                candidate,
                lambda msg, t=depth_topic: self._camera_info_callback(msg, t),
                qos,
            )
            self.camera_info_subs[depth_topic] = sub
            self.get_logger().info(f"订阅 CameraInfo: {candidate} -> {depth_topic}")
            return

    def _on_timer(self) -> None:
        with self._state_lock:
            if self._pending_prefix is not None:
                self.prefix = self._pending_prefix
                self._pending_prefix = None
            pending_enabled = self._pending_enabled
            self._pending_enabled = None

        if pending_enabled is not None:
            self.enabled_topics = pending_enabled
            self._resubscribe()
            self.get_logger().info(f"subscribed to {len(self.enabled_topics)} image topics")

        self._refresh_topics()
        self._maybe_retry_subscriptions()
        self._maybe_retry_robot_subscriptions()

    def _refresh_topics(self) -> None:
        all_topics = dict(self.get_topic_names_and_types())
        camera_topics = {
            name: types
            for name, types in sorted(all_topics.items())
            if name.startswith(self.prefix)
        }

        if camera_topics != self._last_camera_topics:
            self._last_camera_topics = camera_topics
            self.ros_bridge.topics_updated.emit(camera_topics)

        if not all_topics:
            message = (
                "未发现任何 topic — 请用 bash run.sh 启动（需 ROS_LOCALHOST_ONLY=1 + a2d DDS 配置）"
            )
        elif len(all_topics) <= 3:
            message = (
                f"仅发现 {len(all_topics)} 个 topic（robot 未连通）— 请确认 robot-service 已启动并用 bash run.sh 运行"
            )
        else:
            message = (
                f"共 {len(all_topics)} 个 topic，匹配 {len(camera_topics)} 个（前缀: {self.prefix}）"
            )
        if message != self._last_status_message:
            self._last_status_message = message
            self.ros_bridge.status_message.emit(message)

    def _qos_from_publisher_info(self, topic: str) -> Optional[QoSProfile]:
        """从发布端提取 reliability/durability，构建合法 QoS（不可直接使用 endpoint QoS）。"""
        try:
            publishers = self.get_publishers_info_by_topic(topic)
            if not publishers:
                return None
            pub_qos = publishers[0].qos_profile

            reliability = pub_qos.reliability
            if reliability not in (
                ReliabilityPolicy.RELIABLE,
                ReliabilityPolicy.BEST_EFFORT,
            ):
                reliability = ReliabilityPolicy.RELIABLE

            durability = pub_qos.durability
            if durability not in (
                DurabilityPolicy.VOLATILE,
                DurabilityPolicy.TRANSIENT_LOCAL,
            ):
                durability = DurabilityPolicy.VOLATILE

            return QoSProfile(
                reliability=reliability,
                durability=durability,
                history=HistoryPolicy.KEEP_LAST,
                depth=30,
            )
        except Exception as exc:
            self.get_logger().debug(f"{topic}: 解析发布端 QoS 失败: {exc}")
            return None

    def _qos_candidates_for_topic(self, topic: str) -> List[QoSProfile]:
        candidates: List[QoSProfile] = []

        derived = self._qos_from_publisher_info(topic)
        if derived is not None:
            candidates.append(derived)

        candidates.extend([
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=30,
                durability=DurabilityPolicy.VOLATILE,
            ),
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ])
        return candidates

    def _qos_for_topic(self, topic: str, attempt: int = 0) -> QoSProfile:
        candidates = self._qos_candidates_for_topic(topic)
        idx = min(attempt, len(candidates) - 1)
        qos = candidates[idx]
        self.get_logger().info(
            f"{topic}: QoS attempt={attempt} reliability={qos.reliability.name} depth={qos.depth}"
        )
        return qos

    def _maybe_retry_subscriptions(self) -> None:
        if not self.enabled_topics:
            return
        now = time.time()
        for topic in sorted(self.enabled_topics):
            if self._frame_counts.get(topic, 0) > 0:
                continue
            last_sub = self._last_subscribe_time.get(topic, 0.0)
            if now - last_sub < 5.0:
                continue
            attempt = self._subscribe_attempts.get(topic, 0) + 1
            if attempt > 3:
                continue
            self.get_logger().warn(f"{topic}: 未收到图像，第 {attempt} 次重新订阅")
            self._subscribe_attempts[topic] = attempt
            types = dict(self.get_topic_names_and_types()).get(topic, [])
            if topic in self.subscriptions_map:
                self.destroy_subscription(self.subscriptions_map.pop(topic))
            qos = self._qos_for_topic(topic, attempt - 1)
            if self._subscribe_topic(topic, types, qos):
                self._last_subscribe_time[topic] = now

    def _resubscribe(self) -> None:
        for sub in self.subscriptions_map.values():
            self.destroy_subscription(sub)
        for sub in self.camera_info_subs.values():
            self.destroy_subscription(sub)
        self.subscriptions_map.clear()
        self.camera_info_subs.clear()

        all_topics = dict(self.get_topic_names_and_types())
        for topic in sorted(self.enabled_topics):
            types = all_topics.get(topic, [])
            attempt = self._subscribe_attempts.get(topic, 0)
            qos = self._qos_for_topic(topic, attempt)
            if self._subscribe_topic(topic, types, qos):
                self._last_subscribe_time[topic] = time.time()
                self.get_logger().info(f"订阅成功: {topic}")
                if is_depth_topic(topic):
                    self._subscribe_camera_info(topic, qos)
            else:
                self.get_logger().warn(f"跳过非图像 topic: {topic}")

    def _subscribe_topic(self, topic: str, types: List[str], qos: QoSProfile) -> bool:
        for type_name in types:
            msg_type = IMAGE_TYPES.get(type_name)
            if msg_type is CompressedImage:
                sub = self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, t=topic: self._compressed_callback(msg, t),
                    qos,
                )
                self.subscriptions_map[topic] = sub
                return True
            if msg_type is Image:
                sub = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, t=topic: self._image_callback(msg, t),
                    qos,
                )
                self.subscriptions_map[topic] = sub
                return True
        return False

    def _convert_image(self, msg: Image) -> np.ndarray:
        encoding = (msg.encoding or "bgr8").lower()
        if encoding in ("rgb8", "bgr8"):
            desired = "bgr8" if encoding == "bgr8" else "rgb8"
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=desired)
            if encoding == "rgb8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            return cv_image
        if "16" in encoding or encoding in ("32fc1", "mono16"):
            return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if encoding in ("mono8", "8uc1"):
            gray = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _should_emit_frame_to_ui(self, topic: str) -> bool:
        now = time.time()
        interval = UI_DEPTH_MIN_INTERVAL_S if is_depth_topic(topic) else UI_IMAGE_MIN_INTERVAL_S
        last = self._last_ui_emit_time.get(topic, 0.0)
        if now - last < interval:
            return False
        self._last_ui_emit_time[topic] = now
        return True

    def _publish_frame_to_ui(self, topic: str, cv_image: np.ndarray, now: float) -> None:
        self._frame_counts[topic] = self._frame_counts.get(topic, 0) + 1
        self._last_frame_time[topic] = now
        if not self._should_emit_frame_to_ui(topic):
            return
        self.ros_bridge.frame_updated.emit(topic, cv_image.copy())
        self.ros_bridge.frame_stats.emit(topic, self._frame_counts[topic], now)

    def _image_callback(self, msg: Image, topic: str) -> None:
        try:
            if msg.header.frame_id:
                self._camera_frame_ids[topic] = msg.header.frame_id
            cv_image = self._convert_image(msg)
            now = time.time()
            if self._frame_counts.get(topic, 0) == 0:
                self.get_logger().info(
                    f"首帧 {topic}: {msg.width}x{msg.height} encoding={msg.encoding}"
                )
            self._publish_frame_to_ui(topic, cv_image, now)
        except CvBridgeError as exc:
            self.get_logger().warn(f"{topic}: {exc}")
        except Exception as exc:
            self.get_logger().error(f"{topic}: 处理失败 {exc}")

    def _compressed_callback(self, msg: CompressedImage, topic: str) -> None:
        try:
            if msg.header.frame_id:
                self._camera_frame_ids[topic] = msg.header.frame_id
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            now = time.time()
            if self._frame_counts.get(topic, 0) == 0:
                self.get_logger().info(f"首帧 {topic}: compressed {len(msg.data)} bytes")
            self._publish_frame_to_ui(topic, cv_image, now)
        except CvBridgeError as exc:
            self.get_logger().warn(f"{topic}: {exc}")
        except Exception as exc:
            self.get_logger().error(f"{topic}: 处理失败 {exc}")


def resolve_scripts_pack_dir() -> Optional[str]:
    env_dir = os.environ.get("A2D_SCRIPTS_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "/opt/psi/rt/a2d-tele/install/scripts_pack/share/scripts_pack/scripts",
        os.path.expanduser("~/workspace_liyichao/a2d-tele/install/scripts_pack/share/scripts_pack/scripts"),
        os.path.join(here, "..", "install", "scripts_pack", "share", "scripts_pack", "scripts"),
        os.path.expanduser("~/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"),
    ]
    for path in candidates:
        resolved = os.path.abspath(path)
        if os.path.isdir(resolved):
            return resolved
    return None


def resolve_a2d_workspace_dir() -> str:
    """与 GUI 一致，优先使用 a2d-tele 工作区 install。"""
    for candidate in (
        "/opt/psi/rt/a2d-tele",
        os.path.expanduser("~/workspace_liyichao/a2d-tele"),
    ):
        if os.path.isfile(os.path.join(candidate, "install/setup.bash")):
            return candidate
    return "/opt/psi/rt/a2d-tele"


def default_rrd_dataset_dir() -> str:
    env_dir = os.environ.get("RRD_DATASET_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    for candidate in ("/root/dataset", "/home/psibot/dataset"):
        if os.path.isdir(candidate):
            return candidate
    return "/home/psibot/dataset"


def resolve_rrd_path(path: str) -> str:
    """解析 RRD 路径，兼容容器内 /root/dataset 与 /home/psibot/dataset。"""
    if not path:
        return path
    expanded = os.path.abspath(os.path.expanduser(path.strip()))
    if os.path.isfile(expanded):
        return expanded
    candidates = [expanded]
    if expanded.startswith("/home/psibot/dataset"):
        candidates.append(expanded.replace("/home/psibot/dataset", "/root/dataset", 1))
    elif expanded.startswith("/root/dataset"):
        candidates.append(expanded.replace("/root/dataset", "/home/psibot/dataset", 1))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return expanded


def build_ros_shell_prefix() -> str:
    parts = ["set -e", "source /opt/ros/humble/setup.bash"]
    for setup in (
        "/opt/psi/rt/a2d-tele/install/setup.bash",
        os.path.expanduser("~/workspace_liyichao/a2d-tele/install/setup.bash"),
        os.path.expanduser("~/workspace_liyichao/install/setup.bash"),
    ):
        if os.path.isfile(setup):
            parts.append(f"source {setup}")
    dds = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    if dds and os.path.isfile(dds):
        parts.append(f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds}")
    parts.append("export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}")
    return " && ".join(parts)


def _pgrep_pattern(pattern: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            timeout=0.5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _ros_service_exists(service: str) -> bool:
    workspace = resolve_a2d_workspace_dir()
    dds = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    env_prefix = ""
    if dds and os.path.isfile(dds):
        env_prefix = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && "
    cmd = (
        f"{env_prefix}source /opt/ros/humble/setup.bash && "
        f"cd {workspace} && source install/setup.bash && "
        "ros2 service list"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return service in (result.stdout or "")
    except Exception:
        return False


def _ros2_trigger_service(service: str, timeout_sec: float = 15.0) -> tuple[bool, str]:
    """通过 ros2 CLI 调用 Trigger 服务，避免 Qt 线程内 service client 不可靠。"""
    workspace = resolve_a2d_workspace_dir()
    dds = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    env_prefix = ""
    if dds and os.path.isfile(dds):
        env_prefix = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && "
    cmd = (
        f"{env_prefix}source /opt/ros/humble/setup.bash && "
        f"cd {workspace} && source install/setup.bash && "
        f"ros2 service call {service} std_srvs/srv/Trigger '{{}}'"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return False, f"exit={result.returncode}: {output[:240]}"
        lower = output.lower()
        if "success=false" in lower or "success: false" in lower:
            return False, output[:240]
        if "success=true" in lower or "success: true" in lower:
            return True, output[:240]
        return True, output[:240] or "OK"
    except subprocess.TimeoutExpired:
        return False, f"超时 ({timeout_sec}s)"
    except Exception as exc:
        return False, str(exc)


def _stop_existing_rrd_replay() -> None:
    """停止已有 rrd_replay 进程，避免 GUI/viewer 重复启动导致回放无效。"""
    _ros2_trigger_service(REPLAY_STOP_SERVICE, timeout_sec=5.0)
    for pattern in ("rrd_replay.launch.py", "rrd_replay/rrd_replay_node", "rrd_replay_node"):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=1,
                check=False,
            )
        except Exception:
            pass
    for _ in range(30):
        if not (
            _pgrep_pattern("rrd_replay.launch.py")
            or _pgrep_pattern("rrd_replay_node")
        ):
            break
        time.sleep(0.1)


class RobotStackLauncher(QObject):
    """后台启动 robot-service 与 base_services（不阻塞 UI）。"""

    status_message = pyqtSignal(str)
    stack_status_changed = pyqtSignal(str, str)
    start_robot_on_main = pyqtSignal()
    start_base_on_main = pyqtSignal()

    def __init__(self, node: "CameraTopicNode", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._robot_status = "未运行"
        self._base_status = "未运行"
        self._robot_launch_pending = False
        self._base_launch_pending = False
        self._poll_stop = threading.Event()
        self._status_lock = threading.Lock()
        self.start_robot_on_main.connect(self._launch_robot_service, Qt.QueuedConnection)
        self.start_base_on_main.connect(self._launch_base_services, Qt.QueuedConnection)
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def shutdown(self) -> None:
        self._poll_stop.set()
        self._poll_thread.join(timeout=1.0)

    def get_cached_status(self) -> Tuple[str, str]:
        with self._status_lock:
            return self._robot_status, self._base_status

    def start_stack(self) -> None:
        robot, base = self.get_cached_status()
        if base == "运行中":
            self.status_message.emit("base_services 已在运行")
            return
        if base == "启动中" or self._base_launch_pending:
            self.status_message.emit("base_services 正在启动…")
            return
        if self._node.is_hal_arm_ready() or robot == "就绪":
            self._launch_base_services()
            return
        if robot == "启动中" or self._robot_launch_pending:
            self.status_message.emit("robot-service 启动中，请等待 (~3 分钟)…")
            return
        self._launch_robot_service()

    def _launch_robot_service(self) -> None:
        if self._robot_launch_pending:
            self.status_message.emit("robot-service 已在启动中")
            return
        script = self._resolve_script(ROBOT_SERVICE_SCRIPT)
        if script is None:
            self.status_message.emit("未找到 start_ros_service.sh，请设置 A2D_SCRIPTS_DIR")
            return
        script_dir = os.path.dirname(script)
        cmd = (
            f"{build_ros_shell_prefix()} && cd {script_dir} && "
            f"exec bash {os.path.basename(script)}"
        )
        if not QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self.status_message.emit("robot-service 启动失败（startDetached 返回 false）")
            return
        self._robot_launch_pending = True
        self.status_message.emit("正在启动 robot-service（约 3 分钟）…")

    def _launch_base_services(self) -> None:
        robot, base = self.get_cached_status()
        if base == "运行中":
            self.status_message.emit("base_services 已在运行")
            return
        if self._base_launch_pending or base == "启动中":
            self.status_message.emit("base_services 正在启动…")
            return
        script = self._resolve_script(BASE_SERVICES_SCRIPT)
        if script is None:
            self.status_message.emit("未找到 start_base_services.sh，请设置 A2D_SCRIPTS_DIR")
            return
        script_dir = os.path.dirname(script)
        cmd = (
            f"{build_ros_shell_prefix()} && cd {script_dir} && "
            f"exec bash {os.path.basename(script)}"
        )
        if not QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self.status_message.emit("base_services 启动失败（startDetached 返回 false）")
            return
        self._base_launch_pending = True
        self.status_message.emit("正在启动 base_services…")

    def _resolve_script(self, relative_name: str) -> Optional[str]:
        scripts_dir = resolve_scripts_pack_dir()
        if scripts_dir is None:
            return None
        path = os.path.join(scripts_dir, relative_name)
        return path if os.path.isfile(path) else None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            self._refresh_status_cache()
            robot, base = self.get_cached_status()
            self.stack_status_changed.emit(robot, base)
            interval = 15.0 if base == "运行中" else 5.0
            self._poll_stop.wait(interval)

    def _refresh_status_cache(self) -> None:
        base_running = _pgrep_pattern("base_services.launch.py")
        robot_running = _pgrep_pattern("robot-service")
        hal_ready = self._node.is_hal_arm_ready()

        if base_running:
            self._base_launch_pending = False
            base_status = "运行中"
        elif self._base_launch_pending:
            base_status = "启动中"
        else:
            base_status = "未运行"

        if robot_running:
            robot_status = "启动中"
        elif hal_ready:
            robot_status = "就绪"
        else:
            robot_status = "未运行"

        with self._status_lock:
            self._robot_status = robot_status
            self._base_status = base_status

        self._maybe_continue_stack_startup(base_running, robot_running, hal_ready)

    def _maybe_continue_stack_startup(
        self,
        base_running: bool,
        robot_running: bool,
        hal_ready: bool,
    ) -> None:
        if base_running or self._base_launch_pending:
            return
        if not self._robot_launch_pending:
            return
        if robot_running:
            return
        if not hal_ready:
            return
        self._robot_launch_pending = False
        self.status_message.emit("robot-service 完成，正在启动 base_services…")
        self.start_base_on_main.emit()


class RrdReplayLauncher(QObject):
    """启动/停止 rrd_replay 节点，并在就绪后自动调用 start_replay 服务。"""

    status_message = pyqtSignal(str)
    node_active_changed = pyqtSignal(bool)

    def __init__(self, node: "CameraTopicNode", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._process: Optional[QProcess] = None
        self._rrd_path = ""
        self._target_loops = REPLAY_LOOP_COUNT_DEFAULT
        self._completed_loops = 0
        self._loop_restart_pending = False
        self._finishing_after_complete = False
        self._auto_start_attempts = 0
        self._max_auto_start_attempts = 240
        self._auto_start_min_attempts = 6  # launch 后至少等待 ~3s
        self._auto_start_timer = QTimer()
        self._auto_start_timer.setInterval(500)
        self._auto_start_timer.timeout.connect(self._try_auto_start_replay)
        node.ros_bridge.replay_state_changed.connect(self._on_replay_state_changed)

    def get_loop_progress(self) -> Tuple[int, int]:
        return self._completed_loops, self._target_loops

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == QProcess.Running

    def current_rrd_path(self) -> str:
        return self._rrd_path

    def start(self, rrd_path: str, loop_count: int = REPLAY_LOOP_COUNT_DEFAULT) -> None:
        if self.is_running():
            self.status_message.emit("RRD 回放已在运行")
            return

        self._target_loops = max(1, min(int(loop_count), REPLAY_LOOP_COUNT_MAX))
        self._completed_loops = 0
        self._loop_restart_pending = False

        resolved_path = resolve_rrd_path(rrd_path)
        if not os.path.isfile(resolved_path):
            self.status_message.emit(f"RRD 文件不存在: {resolved_path}")
            return

        if (
            _pgrep_pattern("rrd_replay.launch.py")
            or _pgrep_pattern("rrd_replay_node")
        ):
            self.status_message.emit("正在停止已有 rrd_replay 进程…")
            _stop_existing_rrd_replay()

        self._rrd_path = resolved_path
        quoted = resolved_path.replace("'", "'\\''")
        workspace = resolve_a2d_workspace_dir()
        # 与 a2d-tele GUI 保持一致的 launch 参数
        cmd = (
            f"source /opt/ros/humble/setup.bash && "
            f"cd {workspace} && source install/setup.bash && "
            f"ros2 launch rrd_replay rrd_replay.launch.py "
            f"rrd_path:='{quoted}' force_generate_csv:=false enable_keyboard_stop:=false"
        )
        dds = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
        if dds and os.path.isfile(dds):
            cmd = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && {cmd}"
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(workspace)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.node_active_changed.emit(True)
        self.status_message.emit(
            f"正在启动 RRD 回放: {os.path.basename(resolved_path)} "
            f"（共 {self._target_loops} 次）"
        )
        self._auto_start_attempts = 0
        self._auto_start_timer.start()

    def stop(self, *, auto_complete: bool = False) -> None:
        completed = self._completed_loops
        total = self._target_loops
        self._auto_start_timer.stop()
        self._loop_restart_pending = False
        self._completed_loops = 0
        self._target_loops = REPLAY_LOOP_COUNT_DEFAULT
        if self.is_running() or _ros_service_exists(REPLAY_STOP_SERVICE):
            _ros2_trigger_service(REPLAY_STOP_SERVICE, timeout_sec=5.0)
            self._node.request_replay_stop()
        if self._process is not None:
            if self._process.state() == QProcess.Running:
                self._process.terminate()
                QTimer.singleShot(3000, self._force_kill_process)
            else:
                self._process = None
                self.node_active_changed.emit(False)
        else:
            self.node_active_changed.emit(False)
        if auto_complete:
            self.status_message.emit(f"全部回放完成 ({completed}/{total})")
        else:
            self.status_message.emit("正在停止 RRD 回放…")

    def shutdown(self) -> None:
        self._auto_start_timer.stop()
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None
        self.node_active_changed.emit(False)

    def _try_auto_start_replay(self) -> None:
        self._auto_start_attempts += 1
        if self._auto_start_attempts < self._auto_start_min_attempts:
            return
        if not _ros_service_exists(REPLAY_START_SERVICE):
            if self._auto_start_attempts >= self._max_auto_start_attempts:
                self._auto_start_timer.stop()
                self.status_message.emit("等待 rrd_replay 服务超时（120s）")
            elif self._auto_start_attempts % 4 == 0:
                self.status_message.emit("等待 rrd_replay 服务就绪…")
            return

        ok, detail = _ros2_trigger_service(REPLAY_START_SERVICE, timeout_sec=15.0)
        if ok:
            self._auto_start_timer.stop()
            self._node.get_logger().info(f"auto start_replay OK: {detail}")
            self.status_message.emit(
                f"回放已启动 第 {self._completed_loops + 1}/{self._target_loops} 次"
            )
            return

        self._node.get_logger().warn(f"auto start_replay 失败: {detail}")
        if self._auto_start_attempts >= self._max_auto_start_attempts:
            self._auto_start_timer.stop()
            self.status_message.emit(f"自动启动回放失败: {detail[:120]}")
        elif self._auto_start_attempts % 4 == 0:
            self.status_message.emit(f"重试 start_replay… ({detail[:80]})")

    def _on_replay_state_changed(self, state: int) -> None:
        if state != REPLAY_STATE_FINISHED:
            return
        if self._loop_restart_pending:
            return
        if not self.is_running() and not _ros_service_exists(REPLAY_START_SERVICE):
            return

        self._completed_loops += 1
        self.status_message.emit(
            f"回放完成 {self._completed_loops}/{self._target_loops}"
        )
        self._node.get_logger().info(
            f"rrd replay finished loop {self._completed_loops}/{self._target_loops}"
        )
        if self._completed_loops >= self._target_loops:
            QTimer.singleShot(300, self._finish_all_replay_loops)
            return

        self._loop_restart_pending = True
        QTimer.singleShot(400, self._restart_next_loop)

    def _finish_all_replay_loops(self) -> None:
        if not self.is_running() and not _ros_service_exists(REPLAY_STOP_SERVICE):
            return
        self._finishing_after_complete = True
        self.stop(auto_complete=True)

    def _restart_next_loop(self) -> None:
        if self._completed_loops >= self._target_loops:
            self._loop_restart_pending = False
            return
        ok_stop, stop_detail = _ros2_trigger_service(REPLAY_STOP_SERVICE, timeout_sec=8.0)
        if not ok_stop:
            self._node.get_logger().warn(f"loop stop_replay failed: {stop_detail}")
        QTimer.singleShot(300, self._start_next_loop)

    def _start_next_loop(self) -> None:
        self._loop_restart_pending = False
        if self._completed_loops >= self._target_loops:
            return
        ok, detail = _ros2_trigger_service(REPLAY_START_SERVICE, timeout_sec=15.0)
        if ok:
            self.status_message.emit(
                f"开始第 {self._completed_loops + 1}/{self._target_loops} 次回放"
            )
            self._node.get_logger().info(
                f"rrd replay loop {self._completed_loops + 1}/{self._target_loops} started"
            )
        else:
            self.status_message.emit(f"下一轮回放启动失败: {detail[:120]}")
            self._node.get_logger().warn(f"loop start_replay failed: {detail}")

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            text = line.strip()
            if text:
                self._node.get_logger().info(f"[rrd_replay] {text}")

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._auto_start_timer.stop()
        self._process = None
        self.node_active_changed.emit(False)
        if self._finishing_after_complete:
            self._finishing_after_complete = False
            return
        self.status_message.emit(f"RRD 回放节点已退出 (code={exit_code})")

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.Crashed:
            self.status_message.emit(f"RRD 回放进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.kill()


class CameraTopicWindow(QMainWindow):
    def __init__(
        self,
        node: CameraTopicNode,
        bridge: RosBridge,
        prefix: str,
        llm_config: Optional[LlmChatConfig] = None,
    ) -> None:
        super().__init__()
        self.node = node
        self.bridge = bridge
        self.panels: Dict[str, QWidget] = {}
        self.topic_checks: Dict[str, QCheckBox] = {}
        self._topic_types: Dict[str, List[str]] = {}
        self._frame_cache: Dict[str, np.ndarray] = {}
        self._last_segment_target: Optional[SegmentPoseTarget] = None
        self._robot_ui_last_update = 0.0
        self._pending_arm_move_goal: Optional[ResolvedArmMoveGoal] = None
        self._arm_enable_wait_deadline = 0.0
        self._arm_enable_wait_timer = QTimer(self)
        self._arm_enable_wait_timer.setInterval(200)
        self._arm_enable_wait_timer.timeout.connect(self._on_arm_enable_wait_tick)

        self.setWindowTitle("Camera Topic Viewer")
        win_x, win_y, win_w, win_h = default_viewer_geometry()
        self.setMinimumSize(min(1280, win_w), min(820, win_h))
        self.setGeometry(win_x, win_y, win_w, win_h)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        control_tabs = QTabWidget()
        control_tabs.setDocumentMode(True)
        control_tabs.setTabPosition(QTabWidget.North)

        camera_tab = QWidget()
        camera_layout = QHBoxLayout(camera_tab)
        camera_layout.setContentsMargins(8, 6, 8, 6)
        camera_layout.addWidget(QLabel("Topic 前缀:"))
        self.prefix_edit = QLineEdit(prefix)
        self.prefix_edit.setPlaceholderText("/camera")
        camera_layout.addWidget(self.prefix_edit, 1)
        refresh_btn = QPushButton("刷新 Topic")
        refresh_btn.clicked.connect(self._on_refresh_clicked)
        camera_layout.addWidget(refresh_btn)
        select_all_btn = QPushButton("全选图像")
        select_all_btn.clicked.connect(self._select_all_images)
        camera_layout.addWidget(select_all_btn)
        clear_btn = QPushButton("取消全选")
        clear_btn.clicked.connect(self._clear_selection)
        camera_layout.addWidget(clear_btn)
        camera_layout.addStretch()
        control_tabs.addTab(camera_tab, "相机")

        robot_tab = QWidget()
        robot_layout = QVBoxLayout(robot_tab)
        robot_layout.setContentsMargins(8, 6, 8, 6)
        robot_layout.setSpacing(6)
        stack_row = QHBoxLayout()
        self._stack_launcher = RobotStackLauncher(node, self)
        self.robot_stack_status = QLabel("robot: -- | base: --")
        self.robot_stack_status.setFont(QFont("Monospace", 8))
        self.robot_stack_status.setToolTip(
            f"robot-service 初始化 HAL；base_services 启动 topic_router/ik/fk 等。\n"
            f"日志: tail -f {BASE_SERVICES_LOG}"
        )
        self.robot_stack_btn = QPushButton("启动机器人栈")
        self.robot_stack_btn.setToolTip(
            "依次启动 robot-service（约 3 分钟）与 base_services。\n"
            "若 HAL 已就绪则跳过 robot-service。"
        )
        self.robot_stack_btn.clicked.connect(self._on_robot_stack_clicked)
        stack_row.addWidget(self.robot_stack_status, 1)
        stack_row.addWidget(self.robot_stack_btn)
        robot_layout.addLayout(stack_row)

        replay_row = QHBoxLayout()
        replay_row.setSpacing(6)
        self.replay_status_label = QLabel("回放: --")
        self.replay_status_label.setFont(QFont("Monospace", 8))
        self.replay_status_label.setToolTip(
            "RRD 轨迹回放状态（/rrd_replay/running_state）\n"
            "需 base_services + control_mode=0 + 手/臂使能"
        )
        self.replay_select_btn = QPushButton("选择路径")
        self.replay_select_btn.setToolTip(
            "选择 .rrd 回放文件。\n"
            f"默认数据目录: {default_rrd_dataset_dir()}"
        )
        self.replay_select_btn.clicked.connect(self._on_replay_select_clicked)
        self.replay_start_btn = QPushButton("开始")
        self.replay_start_btn.setToolTip(
            "启动 rrd_replay 节点并开始回放（与 a2d-tele GUI 回放一致）。\n"
            "前置：control_mode=0，手动 F1/F2 使能手/臂。"
        )
        self.replay_start_btn.clicked.connect(self._on_replay_start_clicked)
        self.replay_stop_btn = QPushButton("停止")
        self.replay_stop_btn.setToolTip("停止 rrd_replay 节点与当前回放。")
        self.replay_stop_btn.setStyleSheet("color: #ff8888;")
        self.replay_stop_btn.clicked.connect(self._on_replay_stop_clicked)
        replay_row.addWidget(QLabel("次数"))
        self.replay_count_spin = QSpinBox()
        self.replay_count_spin.setRange(1, REPLAY_LOOP_COUNT_MAX)
        self.replay_count_spin.setValue(REPLAY_LOOP_COUNT_DEFAULT)
        self.replay_count_spin.setFixedWidth(52)
        self.replay_count_spin.setToolTip(
            f"回放次数（1–{REPLAY_LOOP_COUNT_MAX}），每次播完后自动从头再播"
        )
        replay_row.addWidget(self.replay_count_spin)
        replay_row.addWidget(self.replay_status_label, 1)
        replay_row.addWidget(self.replay_select_btn)
        replay_row.addWidget(self.replay_start_btn)
        replay_row.addWidget(self.replay_stop_btn)
        robot_layout.addLayout(replay_row)

        self._selected_rrd_path = ""
        rrd_path_row = QHBoxLayout()
        rrd_path_row.setSpacing(6)
        rrd_path_label = QLabel("回放文件:")
        rrd_path_label.setToolTip("当前已选择、用于回放的 .rrd 文件完整路径")
        rrd_path_row.addWidget(rrd_path_label)
        self.replay_rrd_path_edit = QLineEdit()
        self.replay_rrd_path_edit.setReadOnly(True)
        self.replay_rrd_path_edit.setPlaceholderText("点击「选择路径」选择 .rrd 文件")
        self.replay_rrd_path_edit.setFont(QFont("Monospace", 8))
        rrd_path_row.addWidget(self.replay_rrd_path_edit, 1)
        robot_layout.addLayout(rrd_path_row)
        control_tabs.addTab(robot_tab, "回放")

        self._replay_launcher = RrdReplayLauncher(node, self)

        segment_tab = QWidget()
        segment_layout = QHBoxLayout(segment_tab)
        segment_layout.setContentsMargins(8, 6, 8, 6)
        segment_layout.addWidget(QLabel("分割"))
        self.segment_backend_combo = QComboBox()
        self.segment_backend_combo.addItem("几何(深度+颜色)", SAM3_BACKEND_GEOMETRY)
        self.segment_backend_combo.addItem("SAM3 点提示", SAM3_BACKEND_POINT)
        self.segment_backend_combo.addItem("SAM3 文本", SAM3_BACKEND_TEXT)
        self.segment_backend_combo.setToolTip(
            "SAM3 需单独环境安装 ultralytics 并下载 sam3.pt；"
            "推荐先启动 sam3_segment_worker.py --serve"
        )
        self.segment_backend_combo.currentIndexChanged.connect(
            self._on_segment_settings_changed
        )
        segment_layout.addWidget(self.segment_backend_combo)
        self.sam3_text_edit = QLineEdit()
        self.sam3_text_edit.setPlaceholderText("SAM3 文本概念，如 red cup")
        self.sam3_text_edit.setMinimumWidth(160)
        self.sam3_text_edit.setToolTip("仅在「SAM3 文本」模式下使用")
        self.sam3_text_edit.textChanged.connect(self._on_segment_settings_changed)
        segment_layout.addWidget(self.sam3_text_edit, 1)
        self.sam3_http_check = QCheckBox("HTTP")
        self.sam3_http_check.setChecked(SAM3_USE_HTTP_DEFAULT)
        self.sam3_http_check.setToolTip(
            f"勾选后走 SAM3 常驻服务 ({SAM3_SERVER_URL_DEFAULT})，否则每次 subprocess 调 worker"
        )
        self.sam3_http_check.toggled.connect(self._on_segment_settings_changed)
        segment_layout.addWidget(self.sam3_http_check)
        self.sam3_status_label = QLabel("SAM3: --")
        self.sam3_status_label.setFont(QFont("Monospace", 8))
        self.sam3_status_label.setToolTip("SAM3 HTTP 服务健康检查")
        segment_layout.addWidget(self.sam3_status_label)
        segment_layout.addStretch()
        control_tabs.addTab(segment_tab, "分割")

        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setContentsMargins(8, 6, 8, 6)
        control_layout.setSpacing(6)

        hand_row = QHBoxLayout()
        hand_row.setSpacing(6)
        hand_row.addWidget(QLabel("左手 A:"))
        self.left_hand_slider_a = QSlider(Qt.Horizontal)
        self.left_hand_slider_a.setRange(0, 100)
        self.left_hand_slider_a.setValue(LEFT_HAND_ANGLE_A_DEFAULT)
        self.left_hand_slider_a.setFixedWidth(120)
        self.left_hand_slider_a.setToolTip("左手状态 A 的角度 (0=张, 100=合)")
        self.left_hand_slider_a.valueChanged.connect(self._on_left_hand_sliders_changed)
        hand_row.addWidget(self.left_hand_slider_a)
        self.left_hand_label_a = QLabel(format_hand_angle_label(LEFT_HAND_ANGLE_A_DEFAULT))
        self.left_hand_label_a.setFont(QFont("Monospace", 8))
        self.left_hand_label_a.setMinimumWidth(56)
        hand_row.addWidget(self.left_hand_label_a)

        hand_row.addWidget(QLabel("B:"))
        self.left_hand_slider_b = QSlider(Qt.Horizontal)
        self.left_hand_slider_b.setRange(0, 100)
        self.left_hand_slider_b.setValue(LEFT_HAND_ANGLE_B_DEFAULT)
        self.left_hand_slider_b.setFixedWidth(120)
        self.left_hand_slider_b.setToolTip("左手状态 B 的角度 (0=张, 100=合)")
        self.left_hand_slider_b.valueChanged.connect(self._on_left_hand_sliders_changed)
        hand_row.addWidget(self.left_hand_slider_b)
        self.left_hand_label_b = QLabel(format_hand_angle_label(LEFT_HAND_ANGLE_B_DEFAULT))
        self.left_hand_label_b.setFont(QFont("Monospace", 8))
        self.left_hand_label_b.setMinimumWidth(56)
        hand_row.addWidget(self.left_hand_label_b)

        self.left_hand_toggle_btn = QPushButton("切换")
        self.left_hand_toggle_btn.setToolTip("在游标 A 与 B 设定的两个角度之间切换")
        self.left_hand_toggle_btn.clicked.connect(self._on_left_hand_toggle)
        hand_row.addWidget(self.left_hand_toggle_btn)

        self.left_hand_apply_btn = QPushButton("应用当前")
        self.left_hand_apply_btn.setToolTip("将主游标（当前激活状态 A 或 B）的角度立即发送")
        self.left_hand_apply_btn.clicked.connect(self._on_left_hand_apply_active)
        hand_row.addWidget(self.left_hand_apply_btn)
        hand_row.addStretch()
        control_layout.addLayout(hand_row)

        arm_row1 = QHBoxLayout()
        arm_row1.setSpacing(6)
        self.arm_enable_label = QLabel("手臂: --")
        self.arm_enable_label.setFont(QFont("Monospace", 8))
        self.arm_enable_label.setStyleSheet("color: #ff8888;")
        self.arm_enable_btn = QPushButton("启用手臂")
        self.arm_enable_btn.setToolTip(
            "等同踏板 F2 开启手臂控制（F2 为开关，按一次开、再按一次关）"
        )
        self.arm_enable_btn.clicked.connect(self._on_arm_enable_clicked)
        arm_row1.addWidget(self.arm_enable_label)
        arm_row1.addWidget(self.arm_enable_btn)
        arm_row1.addWidget(QLabel("移动速度"))
        self.arm_move_speed_slider = QSlider(Qt.Horizontal)
        self.arm_move_speed_slider.setRange(10, 100)
        default_speed_slider = int(
            round(
                (ARM_MOVE_SPEED_DEFAULT_RAD_S - ARM_MOVE_SPEED_MIN_RAD_S)
                / (ARM_MOVE_SPEED_MAX_RAD_S - ARM_MOVE_SPEED_MIN_RAD_S)
                * 90
            )
            + 10
        )
        self.arm_move_speed_slider.setValue(default_speed_slider)
        self.arm_move_speed_slider.setFixedWidth(140)
        self.arm_move_speed_slider.setToolTip(
            f"关节空间最大角速度：{ARM_MOVE_SPEED_MIN_RAD_S:.2f}~"
            f"{ARM_MOVE_SPEED_MAX_RAD_S:.2f} rad/s（左慢右快）"
        )
        self.arm_move_speed_slider.valueChanged.connect(self._on_arm_move_speed_changed)
        arm_row1.addWidget(self.arm_move_speed_slider)
        self.arm_move_speed_label = QLabel(format_arm_move_speed_label(default_speed_slider))
        self.arm_move_speed_label.setFont(QFont("Monospace", 8))
        self.arm_move_speed_label.setMinimumWidth(72)
        arm_row1.addWidget(self.arm_move_speed_label)
        arm_row1.addStretch()
        control_layout.addLayout(arm_row1)

        arm_row2 = QHBoxLayout()
        arm_row2.setSpacing(6)
        self.left_arm_move_btn = QPushButton("左臂: 移动")
        self.left_arm_move_btn.setEnabled(False)
        self.left_arm_move_btn.setToolTip(
            "将左臂 TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
            f"目标可为分割位姿，或相对当前位置的手动偏移。\n"
            f"未使能时会自动启用手臂；同时发布左右臂 IK 目标。"
        )
        self.left_arm_move_btn.clicked.connect(self._on_left_arm_move_clicked)
        arm_row2.addWidget(self.left_arm_move_btn)

        pose_info_box = QVBoxLayout()
        pose_info_box.setSpacing(0)
        pose_info_box.setContentsMargins(6, 0, 0, 0)
        self.arm_pose_current_label = QLabel("当前  左臂 TCP: --")
        self.arm_pose_current_label.setFont(QFont("Monospace", 8))
        self.arm_pose_current_label.setStyleSheet("color: #7ec8ff;")
        self.arm_pose_target_label = QLabel("目标  左臂 TCP: --")
        self.arm_pose_target_label.setFont(QFont("Monospace", 8))
        self.arm_pose_target_label.setStyleSheet("color: #ffb86c;")
        pose_info_box.addWidget(self.arm_pose_current_label)
        pose_info_box.addWidget(self.arm_pose_target_label)
        arm_row2.addLayout(pose_info_box, 1)
        control_layout.addLayout(arm_row2)

        target_row = QHBoxLayout()
        target_row.setSpacing(6)
        self.move_target_relative_radio = QRadioButton("相对当前")
        self.move_target_segment_radio = QRadioButton("分割位姿")
        self.move_target_relative_radio.setChecked(True)
        self.move_target_segment_radio.setEnabled(False)
        self._move_target_group = QButtonGroup(self)
        self._move_target_group.addButton(self.move_target_relative_radio)
        self._move_target_group.addButton(self.move_target_segment_radio)
        self.move_target_relative_radio.setToolTip(
            "目标 = 当前 TCP 位置 + 偏移（base_link：X前/后，Y左/右，Z上/下）"
        )
        self.move_target_segment_radio.setToolTip(
            "使用图像点击处的接触 TCP（橙色标记）作为左臂目标，分割完成后自动移动"
        )
        target_row.addWidget(self.move_target_relative_radio)
        target_row.addWidget(self.move_target_segment_radio)
        target_row.addWidget(QLabel("前ΔX"))
        self.offset_x_spin = make_manual_offset_spinbox()
        self.offset_x_spin.setToolTip("沿 base_link X 轴偏移（正=前，负=后）")
        target_row.addWidget(self.offset_x_spin)
        target_row.addWidget(QLabel("左ΔY"))
        self.offset_y_spin = make_manual_offset_spinbox()
        self.offset_y_spin.setToolTip("沿 base_link Y 轴偏移（正=左，负=右）")
        target_row.addWidget(self.offset_y_spin)
        target_row.addWidget(QLabel("上ΔZ"))
        self.offset_z_spin = make_manual_offset_spinbox()
        self.offset_z_spin.setToolTip("沿 base_link Z 轴偏移（正=上，负=下）")
        target_row.addWidget(self.offset_z_spin)
        target_row.addStretch()

        self._move_target_group.buttonToggled.connect(self._on_move_target_params_changed)
        for spin in (self.offset_x_spin, self.offset_y_spin, self.offset_z_spin):
            spin.valueChanged.connect(self._on_move_target_params_changed)
        control_layout.addLayout(target_row)
        control_tabs.addTab(control_tab, "手臂/手")

        self._left_arm_move_btn_idle_style = ""
        self._left_arm_move_btn_cancel_style = "color: #ff8888;"

        root_layout.addWidget(control_tabs)

        bridge.left_hand_preset_changed.connect(self._update_left_hand_toggle_ui)
        bridge.slow_motion_progress.connect(self._on_slow_motion_progress)
        bridge.slow_motion_finished.connect(self._on_slow_motion_finished)
        bridge.robot_state_updated.connect(self._schedule_robot_ui_refresh)
        bridge.arm_enable_changed.connect(self._on_arm_enable_ui_changed)
        self._update_left_hand_toggle_ui(self.node.is_left_hand_at_a())
        self._update_arm_enable_ui()
        self._on_arm_move_speed_changed(self.arm_move_speed_slider.value())
        self._update_left_arm_move_btn_ui(force=True)

        self._main_splitter = QSplitter(Qt.Horizontal)

        left_group = QGroupBox("Topic 列表")
        left_layout = QVBoxLayout(left_group)
        self.topic_list_widget = QWidget()
        self.topic_list_layout = QVBoxLayout(self.topic_list_widget)
        self.topic_list_layout.setAlignment(Qt.AlignTop)

        topic_scroll = QScrollArea()
        topic_scroll.setWidgetResizable(True)
        topic_scroll.setWidget(self.topic_list_widget)
        topic_scroll.setMinimumWidth(220)
        topic_scroll.setMaximumWidth(300)
        left_layout.addWidget(topic_scroll)
        self._main_splitter.addWidget(left_group)

        right_group = QGroupBox("图像预览")
        right_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(4, 8, 4, 4)
        self.grid_widget = QWidget()
        self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(4)
        right_layout.addWidget(self.grid_widget, stretch=1)
        self._main_splitter.addWidget(right_group)

        chat_group = QGroupBox("AI 对话")
        chat_layout = QVBoxLayout(chat_group)
        chat_layout.setContentsMargins(4, 8, 4, 4)
        self.chat_panel = ChatPanelWidget(config=llm_config)
        chat_layout.addWidget(self.chat_panel)
        chat_group.setMinimumWidth(280)
        chat_group.setMaximumWidth(420)
        self._main_splitter.addWidget(chat_group)

        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes([260, max(720, win_w - 560), 320])
        root_layout.addWidget(self._main_splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        self.chat_panel.status_message.connect(self.status_bar.showMessage)
        self._stack_launcher.status_message.connect(self.status_bar.showMessage)
        self._stack_launcher.stack_status_changed.connect(
            self._on_stack_status_changed, Qt.QueuedConnection
        )
        self._replay_launcher.status_message.connect(self.status_bar.showMessage)
        self._replay_launcher.node_active_changed.connect(self._update_replay_ui)
        bridge.replay_state_changed.connect(self._update_replay_ui)

        bridge.topics_updated.connect(self._on_topics_updated)
        bridge.frame_updated.connect(self._on_frame_updated)
        bridge.status_message.connect(self.status_bar.showMessage)
        bridge.frame_stats.connect(self._on_frame_stats)

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_waiting_hint)
        self._ui_timer.start(2000)
        self._sam3_health_timer = QTimer(self)
        self._sam3_health_timer.timeout.connect(self._refresh_sam3_status)
        self._sam3_health_timer.start(5000)
        self._on_segment_settings_changed()
        self._received_topics: Dict[str, int] = {}
        robot, base = self._stack_launcher.get_cached_status()
        self._update_robot_stack_ui(robot, base)
        self._update_replay_ui()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._ui_timer.stop()
        self._sam3_health_timer.stop()
        self._stack_launcher.shutdown()
        self._replay_launcher.shutdown()
        for panel in self.panels.values():
            if isinstance(panel, DepthPanel3D):
                panel.stop_robot_timer()
        self.node.prepare_shutdown()
        super().closeEvent(event)

    def _on_robot_stack_clicked(self) -> None:
        self._stack_launcher.start_stack()

    def _on_replay_select_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return

        initial_dir = default_rrd_dataset_dir()
        if self._selected_rrd_path and os.path.isfile(self._selected_rrd_path):
            initial_dir = os.path.dirname(self._selected_rrd_path)
        elif not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        rrd_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 RRD 回放文件",
            initial_dir,
            "RRD Files (*.rrd);;All Files (*)",
        )
        if not rrd_path:
            return

        resolved_path = resolve_rrd_path(rrd_path)
        self._selected_rrd_path = resolved_path
        self._update_replay_rrd_path_display(resolved_path)
        self._update_replay_ui()

    def _on_replay_start_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return

        path = (self._selected_rrd_path or "").strip()
        if not path:
            QMessageBox.information(
                self,
                "RRD 回放",
                "请先点击「选择路径」选择 .rrd 文件。",
            )
            return
        if not os.path.isfile(path):
            QMessageBox.warning(
                self,
                "RRD 回放",
                f"文件不存在：\n{path}",
            )
            return

        warnings = self.node.get_replay_prerequisite_warnings()
        if warnings:
            reply = QMessageBox.warning(
                self,
                "回放前置条件",
                "当前未满足回放条件：\n\n"
                + "\n".join(f"• {w}" for w in warnings)
                + "\n\n请先手动 F1/F2 使能，或仍要继续启动回放节点？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._replay_launcher.start(
            path, loop_count=self.replay_count_spin.value()
        )

    def _on_replay_stop_clicked(self) -> None:
        if self._replay_launcher.is_running():
            self._replay_launcher.stop()

    def _update_replay_rrd_path_display(self, path: str) -> None:
        text = (path or "").strip()
        if text:
            self.replay_rrd_path_edit.setText(text)
            self.replay_rrd_path_edit.setToolTip(text)
        else:
            self.replay_rrd_path_edit.clear()
            self.replay_rrd_path_edit.setToolTip("当前已选择、用于回放的 .rrd 文件完整路径")

    def _update_replay_ui(self, *_args) -> None:
        state = self.node.get_replay_state()
        label = self.node.get_replay_state_label()
        node_running = self._replay_launcher.is_running()
        completed, total = self._replay_launcher.get_loop_progress()
        loop_hint = ""
        if node_running and total > 1:
            current = min(completed + 1, total) if state in (0, 1, 2) else completed
            loop_hint = f" [{current}/{total}]"
        if node_running:
            self.replay_status_label.setText(f"回放: {label}{loop_hint}")
        else:
            self.replay_status_label.setText(f"回放: {label}")

        path = self._replay_launcher.current_rrd_path() or self._selected_rrd_path
        self._update_replay_rrd_path_display(path)

        has_path = bool((self._selected_rrd_path or "").strip())
        self.replay_count_spin.setEnabled(not node_running)
        self.replay_select_btn.setEnabled(not node_running)
        self.replay_start_btn.setEnabled(not node_running and has_path)
        self.replay_stop_btn.setEnabled(node_running)

        if state == 1:
            color = "#50fa7b"
        elif state in (2, 0) and node_running:
            color = "#ffb86c"
        elif state in (3, 4):
            color = "#888888"
        else:
            color = "#7ec8ff"
        self.replay_status_label.setStyleSheet(f"color: {color};")

    def _on_segment_settings_changed(self, *_args) -> None:
        backend = self.segment_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = SAM3_BACKEND_GEOMETRY
        use_sam3 = backend != SAM3_BACKEND_GEOMETRY
        self.sam3_text_edit.setEnabled(backend == SAM3_BACKEND_TEXT)
        self.sam3_http_check.setEnabled(use_sam3)
        set_segment_settings(
            backend=backend,
            sam3_text=self.sam3_text_edit.text().strip(),
            sam3_use_http=self.sam3_http_check.isChecked() and use_sam3,
            sam3_server_url=SAM3_SERVER_URL_DEFAULT,
        )
        self._refresh_sam3_status()

    def _refresh_sam3_status(self) -> None:
        cfg = get_segment_settings()
        if cfg.backend == SAM3_BACKEND_GEOMETRY:
            self.sam3_status_label.setText("SAM3: 关")
            self.sam3_status_label.setStyleSheet("color: #888;")
            return
        if cfg.sam3_use_http:
            ok = check_sam3_server_health(cfg.sam3_server_url)
            if ok:
                self.sam3_status_label.setText("SAM3: 在线")
                self.sam3_status_label.setStyleSheet("color: #50fa7b;")
            else:
                self.sam3_status_label.setText("SAM3: 离线")
                self.sam3_status_label.setStyleSheet("color: #ff5555;")
        else:
            self.sam3_status_label.setText("SAM3: 子进程")
            self.sam3_status_label.setStyleSheet("color: #ffb86c;")

    def _on_stack_status_changed(self, robot: str, base: str) -> None:
        self._update_robot_stack_ui(robot, base)

    def _update_robot_stack_ui(
        self,
        robot: Optional[str] = None,
        base: Optional[str] = None,
    ) -> None:
        if robot is None or base is None:
            robot, base = self._stack_launcher.get_cached_status()
        self.robot_stack_status.setText(f"robot: {robot} | base: {base}")
        if base == "运行中":
            self.robot_stack_status.setStyleSheet("color: #88ff88;")
        elif base == "启动中" or robot == "启动中":
            self.robot_stack_status.setStyleSheet("color: #ffcc66;")
        else:
            self.robot_stack_status.setStyleSheet("color: #aaaaaa;")
        busy = base in ("运行中", "启动中") or robot == "启动中"
        self.robot_stack_btn.setEnabled(not busy)
        if base == "运行中":
            self.robot_stack_btn.setText("栈已运行")
        elif robot == "启动中":
            self.robot_stack_btn.setText("robot 启动中…")
        elif base == "启动中":
            self.robot_stack_btn.setText("base 启动中…")
        else:
            self.robot_stack_btn.setText("启动机器人栈")

    def _on_refresh_clicked(self) -> None:
        prefix = self.prefix_edit.text().strip() or "/camera"
        if prefix == "/":
            self.status_bar.showMessage("前缀不能为 /，已重置为 /camera")
            prefix = "/camera"
            self.prefix_edit.setText(prefix)
        self.node.set_prefix(prefix)

    def _on_left_hand_sliders_changed(self, _value: int = 0) -> None:
        self.left_hand_label_a.setText(format_hand_angle_label(self.left_hand_slider_a.value()))
        self.left_hand_label_b.setText(format_hand_angle_label(self.left_hand_slider_b.value()))
        self._update_left_hand_toggle_ui(self.node.is_left_hand_at_a())

    def _update_left_hand_toggle_ui(self, at_a: bool) -> None:
        pos_a = slider_to_hand_position(self.left_hand_slider_a.value())
        pos_b = slider_to_hand_position(self.left_hand_slider_b.value())
        next_preset = "B" if at_a else "A"
        next_pos = pos_b if at_a else pos_a
        self.left_hand_toggle_btn.setText(f"左手: 切到{next_preset} ({next_pos:.2f})")
        active_style = "font-weight: bold; color: #7ec8ff;"
        idle_style = "color: #888;"
        self.left_hand_label_a.setStyleSheet(active_style if at_a else idle_style)
        self.left_hand_label_b.setStyleSheet(active_style if not at_a else idle_style)

    def _on_left_hand_toggle(self) -> None:
        at_a, sent_pos = self.node.toggle_left_hand_between(
            self.left_hand_slider_a.value(),
            self.left_hand_slider_b.value(),
        )
        self._update_left_hand_toggle_ui(at_a)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"左手已切到状态{preset} position={sent_pos:.3f} -> {ROBOT_LEFT_HAND_CMD_TOPIC}"
        )

    def _on_move_target_params_changed(self, *_args) -> None:
        if self._last_segment_target is None and self.move_target_segment_radio.isChecked():
            self.move_target_relative_radio.setChecked(True)
        self._update_arm_pose_display()
        self._update_left_arm_move_btn_ui()

    def _resolve_move_goal(self) -> Optional[ResolvedArmMoveGoal]:
        if self.move_target_segment_radio.isChecked() and self._last_segment_target is not None:
            return self.node.resolve_segment_move_goal(self._last_segment_target)
        tcp = self.node._get_left_tcp_in_base(timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
        if tcp is None:
            state = self.node.get_robot_state()
            raw = state.left_tcp
            if raw is not None and raw.valid:
                frame = normalize_frame_id(raw.frame_id)
                if frame == normalize_frame_id(IK_TARGET_FRAME) or frame in MINK_FK_FRAME_ALIASES:
                    tcp = (raw.xyz, raw.quat_xyzw)
        if tcp is None:
            return None
        xyz, quat = tcp
        return compute_relative_move_goal(
            xyz,
            quat,
            self.offset_x_spin.value(),
            self.offset_y_spin.value(),
            self.offset_z_spin.value(),
        )

    def _on_arm_move_speed_changed(self, value: int) -> None:
        speed = slider_to_arm_move_speed(value)
        self.node.set_arm_move_joint_speed(speed)
        self.arm_move_speed_label.setText(format_arm_move_speed_label(value))
        if not self.node.is_slow_motion_busy():
            self._update_left_arm_move_btn_ui()

    def _on_arm_enable_clicked(self) -> None:
        enable = not self.node.is_arm_enabled()
        msg = self.node.request_arm_enable(enable=enable)
        self.status_bar.showMessage(msg)

    def _on_arm_enable_ui_changed(self, enabled: bool) -> None:
        self._update_arm_enable_ui()
        if enabled:
            self._try_finish_pending_arm_move()

    def _try_finish_pending_arm_move(self) -> None:
        if self._pending_arm_move_goal is None or not self.node.is_arm_enabled():
            return
        goal = self._pending_arm_move_goal
        self._pending_arm_move_goal = None
        self._arm_enable_wait_timer.stop()
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_left_arm_move_btn_ui()

    def _on_arm_enable_wait_tick(self) -> None:
        if self._pending_arm_move_goal is None:
            self._arm_enable_wait_timer.stop()
            return
        if self.node.is_arm_enabled():
            self._try_finish_pending_arm_move()
            return
        if time.time() > self._arm_enable_wait_deadline:
            self._pending_arm_move_goal = None
            self._arm_enable_wait_timer.stop()
            self.status_bar.showMessage(
                "自动启用手臂超时，请确认 topic_router 已运行，或手动按 F2 / 点「启用手臂」"
            )
            self._update_left_arm_move_btn_ui()
        self._update_left_arm_move_btn_ui()

    def _update_arm_enable_ui(self) -> None:
        self.arm_enable_label.setText(self.node.get_arm_enable_label())
        enabled = self.node.is_arm_enabled()
        if enabled:
            self.arm_enable_label.setStyleSheet("color: #50fa7b;")
            self.arm_enable_btn.setText("关闭手臂")
        else:
            self.arm_enable_label.setStyleSheet("color: #ff8888;")
            self.arm_enable_btn.setText("启用手臂")

    def _can_start_arm_move(self) -> bool:
        if self.node.is_slow_motion_busy():
            return True
        if self.move_target_segment_radio.isChecked():
            return self._resolve_move_goal() is not None
        state = self.node.get_robot_state()
        left_ok = state.left_tcp is not None and state.left_tcp.valid
        right_ok = state.right_tcp is not None and state.right_tcp.valid
        return left_ok and right_ok

    def _disabled_move_btn_tooltip(self) -> str:
        blockers = self.node.get_arm_move_blockers(tf_timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
        hints: List[str] = []
        if blockers:
            hints.append("点击后可能无法移动:")
            hints.extend(blockers)
        state = self.node.get_robot_state()
        left_ok = state.left_tcp is not None and state.left_tcp.valid
        right_ok = state.right_tcp is not None and state.right_tcp.valid
        if not left_ok or not right_ok:
            hints.append("等待双臂 TCP（/mink_fk/* 或 /tele/fk/*）")
        if self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                hints.append("请先点击图像完成分割")
            elif self._resolve_move_goal() is None:
                frame = self._last_segment_target.camera_frame or "未知"
                hints.append(f"分割 TF 不可用 ({frame} -> {IK_TARGET_FRAME})")
        elif self._resolve_move_goal() is None:
            hints.append("相对当前模式：请设置非零偏移（如 ΔX=0.05 m）")
        if not hints:
            hints.append("未使能时点击「移动」将自动启用手臂")
        return "\n".join(hints)

    def _update_arm_pose_display(self) -> None:
        self.arm_pose_current_label.setText(self.node.get_arm_move_current_label())
        resolved = self._resolve_move_goal()
        if resolved is not None:
            self.arm_pose_target_label.setText(
                f"目标  [{resolved.label}]  "
                f"{format_xyz_rpy_line('左臂', resolved.position_xyz, resolved.quaternion_xyzw)}"
            )
        elif self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                self.arm_pose_target_label.setText("目标  左臂 TCP: (请先点击图像分割)")
            else:
                frame = self._last_segment_target.camera_frame or "未知"
                self.arm_pose_target_label.setText(
                    f"目标  左臂 TCP: TF 不可用 ({frame} -> {IK_TARGET_FRAME})"
                )
        else:
            self.arm_pose_target_label.setText(
                "目标  左臂 TCP: (设置前/后/左/右/上/下偏移，base_link 系)"
            )

    def _schedule_robot_ui_refresh(self, *_args) -> None:
        now = time.time()
        if now - self._robot_ui_last_update < UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._robot_ui_last_update = now
        self._update_arm_pose_display()
        if not self.node.is_slow_motion_busy():
            can_move = self._can_start_arm_move()
            if self.left_arm_move_btn.isEnabled() != can_move:
                self.left_arm_move_btn.setEnabled(can_move)
                if not can_move:
                    self.left_arm_move_btn.setToolTip(self._disabled_move_btn_tooltip())

    def _update_left_arm_move_btn_ui(self, force: bool = False) -> None:
        speed_busy = self.node.is_slow_motion_busy()
        self.arm_move_speed_slider.setEnabled(not speed_busy)
        if self._pending_arm_move_goal is not None and not self.node.is_arm_enabled():
            self.left_arm_move_btn.setText("使能中…")
            self.left_arm_move_btn.setEnabled(True)
            self.left_arm_move_btn.setStyleSheet(self._left_arm_move_btn_idle_style)
            self.left_arm_move_btn.setToolTip("正在自动启用手臂，完成后将开始移动")
            self._update_arm_pose_display()
            return
        if self.node.is_slow_motion_preparing():
            self.left_arm_move_btn.setText("取消准备")
            self.left_arm_move_btn.setEnabled(True)
            self.left_arm_move_btn.setStyleSheet(self._left_arm_move_btn_cancel_style)
            self.left_arm_move_btn.setToolTip("取消正在进行的 IK 同步与轨迹规划")
            self._update_arm_pose_display()
            return
        if self.node.is_slow_motion_active():
            self.left_arm_move_btn.setText("取消移动")
            self.left_arm_move_btn.setEnabled(True)
            self.left_arm_move_btn.setStyleSheet(self._left_arm_move_btn_cancel_style)
            self.left_arm_move_btn.setToolTip("停止当前移动")
            self._update_arm_pose_display()
            return
        self.left_arm_move_btn.setText("左臂: 移动")
        self.left_arm_move_btn.setStyleSheet(self._left_arm_move_btn_idle_style)
        can_move = self._can_start_arm_move()
        self.left_arm_move_btn.setEnabled(can_move)
        if can_move:
            blockers = self.node.get_arm_move_blockers(tf_timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
            extra = ""
            if blockers:
                extra = "\n注意: " + "；".join(blockers)
            elif (
                not self.move_target_segment_radio.isChecked()
                and self._resolve_move_goal() is None
            ):
                extra = "\n请设置非零偏移后再点击"
            self.left_arm_move_btn.setToolTip(
                "将左臂 TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
                f"目标可为分割位姿，或相对当前位置的手动偏移。\n"
                f"未使能时将自动启用手臂；同时发布左右臂 IK 目标。{extra}"
            )
        else:
            self.left_arm_move_btn.setToolTip(self._disabled_move_btn_tooltip())
        self._update_arm_pose_display()

    def _store_segment_target(self, target: SegmentPoseTarget) -> None:
        self._last_segment_target = target
        self.move_target_segment_radio.setEnabled(True)
        self.move_target_segment_radio.setChecked(True)
        self._update_left_arm_move_btn_ui()
        self._update_arm_pose_display()
        self.status_bar.showMessage(
            f"已记录接触 TCP 目标: {target.label or target.source_topic}"
        )
        QTimer.singleShot(0, self._try_auto_move_to_segment_target)

    def _try_auto_move_to_segment_target(self) -> None:
        """分割完成后将接触 TCP 自动传给左臂移动。"""
        if self.node.is_slow_motion_busy():
            return
        if not self.move_target_segment_radio.isChecked():
            return
        goal = self._resolve_move_goal()
        if goal is None:
            blockers = self.node.get_arm_move_blockers(tf_timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
            if blockers:
                self.status_bar.showMessage(
                    "接触 TCP 已记录，暂无法移动: " + "；".join(blockers)
                )
            return
        if not self.node.is_arm_enabled():
            self._pending_arm_move_goal = goal
            msg = self.node.request_arm_enable(True)
            self._arm_enable_wait_deadline = time.time() + 15.0
            self._arm_enable_wait_timer.start()
            self.status_bar.showMessage(f"{msg}，使能后将自动移向接触 TCP…")
            self._update_left_arm_move_btn_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_left_arm_move_btn_ui()

    def _on_left_arm_move_clicked(self) -> None:
        if self.node.is_slow_motion_busy():
            self._pending_arm_move_goal = None
            self._arm_enable_wait_timer.stop()
            msg = self.node.cancel_slow_motion()
            self.status_bar.showMessage(msg)
            self._update_left_arm_move_btn_ui()
            return
        goal = self._resolve_move_goal()
        if goal is None:
            if self.move_target_segment_radio.isChecked():
                if self._last_segment_target is None:
                    self.status_bar.showMessage("请先点击图像完成分割与 6D 位姿估计")
                else:
                    frame = self._last_segment_target.camera_frame or "未知"
                    self.status_bar.showMessage(
                        f"无法将分割位姿变换到 {IK_TARGET_FRAME}，"
                        f"请检查 {frame} 的 TF"
                    )
            else:
                self.status_bar.showMessage("请设置非零偏移量，或等待左臂 TCP 数据")
            return
        if not self.node.is_arm_enabled():
            self._pending_arm_move_goal = goal
            msg = self.node.request_arm_enable(True)
            self._arm_enable_wait_deadline = time.time() + 15.0
            self._arm_enable_wait_timer.start()
            self.status_bar.showMessage(f"{msg}，使能后将自动开始移动…")
            self._update_left_arm_move_btn_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_left_arm_move_btn_ui()

    def _on_slow_motion_progress(self, progress: float, text: str) -> None:
        self.status_bar.showMessage(text)
        self._update_left_arm_move_btn_ui()
        self._update_arm_pose_display()

    def _on_slow_motion_finished(self, ok: bool, text: str) -> None:
        self.status_bar.showMessage(text)
        self._update_left_arm_move_btn_ui()

    def _on_left_hand_apply_active(self) -> None:
        at_a = self.node.is_left_hand_at_a()
        slider = self.left_hand_slider_a.value() if at_a else self.left_hand_slider_b.value()
        sent_pos = self.node.apply_left_hand_angle(slider)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"左手状态{preset} 已应用 position={sent_pos:.3f} -> {ROBOT_LEFT_HAND_CMD_TOPIC}"
        )

    def _normalize_prefix(self, prefix: str) -> str:
        prefix = prefix.strip() or "/camera"
        if prefix == "/":
            return "/camera"
        return prefix

    def _is_image_topic(self, types: List[str]) -> bool:
        return any(t in IMAGE_TYPES for t in types)

    def _on_topics_updated(self, topics: Dict[str, List[str]]) -> None:
        self._topic_types = topics

        while self.topic_list_layout.count():
            item = self.topic_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.topic_checks.clear()

        if not topics:
            empty = QLabel("未发现匹配的 topic")
            empty.setStyleSheet("color: #888; padding: 8px;")
            self.topic_list_layout.addWidget(empty)
            self._rebuild_panels(set())
            return

        default_enabled: set[str] = set()
        for topic, types in topics.items():
            type_str = ", ".join(t.split("/")[-1] for t in types)
            is_default = is_color_image_topic(topic, types)
            label = f"{topic}  [{type_str}]"
            checkbox = QCheckBox(label)
            checkbox.setChecked(is_default)
            checkbox.stateChanged.connect(self._on_selection_changed)
            self.topic_checks[topic] = checkbox
            self.topic_list_layout.addWidget(checkbox)
            if is_default:
                default_enabled.add(topic)

        self.topic_list_layout.addStretch()
        self._apply_selection(default_enabled)

    def _on_selection_changed(self) -> None:
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        self._apply_selection(enabled)

    def _apply_selection(self, enabled: set[str]) -> None:
        self.node.set_enabled_topics(enabled)
        self._rebuild_panels(enabled)

    def _select_all_images(self) -> None:
        for topic, checkbox in self.topic_checks.items():
            types = self._topic_types.get(topic, [])
            checkbox.setChecked(self._is_image_topic(types))

    def _clear_selection(self) -> None:
        for checkbox in self.topic_checks.values():
            checkbox.setChecked(False)

    def _grid_dimensions(self, topic_count: int, has_depth: bool) -> Tuple[int, int]:
        """返回 (rows, cols)，使面板在可视区域内并排显示。"""
        del has_depth  # 深度与彩色面板统一按数量排布
        if topic_count <= 0:
            return 1, 1
        if topic_count == 1:
            return 1, 1
        if topic_count <= 3:
            return 1, topic_count
        if topic_count == 4:
            return 2, 2
        if topic_count <= 6:
            cols = 2
            return (topic_count + cols - 1) // cols, cols
        cols = 3
        return (topic_count + cols - 1) // cols, cols

    def _apply_grid_stretches(self, rows: int, cols: int) -> None:
        for r in range(16):
            self.grid_layout.setRowStretch(r, 1 if r < rows else 0)
        for c in range(16):
            self.grid_layout.setColumnStretch(c, 1 if c < cols else 0)

    def _rebuild_panels(self, enabled: set[str]) -> None:
        for topic in list(self.panels.keys()):
            if topic not in enabled:
                panel = self.panels.pop(topic)
                panel.deleteLater()

        for topic in enabled:
            if topic not in self.panels:
                self.panels[topic] = create_camera_panel(
                    topic,
                    get_intrinsics=self.node.get_intrinsics,
                    get_paired_depth=self._get_paired_depth_frame,
                    get_paired_color=self._get_paired_color_frame,
                    get_depth_topic=self._get_paired_depth_topic_name,
                    is_paired_depth_enabled=self._is_paired_depth_enabled,
                    get_robot_state=self.node.get_robot_state,
                    get_camera_frame_id=self.node.get_camera_frame_id,
                    resolve_segment_camera_frame=self.node.resolve_segment_camera_frame,
                    get_tf_buffer=self.node.get_tf_buffer,
                    on_segment_pose=self._store_segment_target,
                    status_callback=self.status_bar.showMessage,
                )

        while self.grid_layout.count():
            self.grid_layout.takeAt(0)

        topics = sorted(self.panels.keys())
        has_depth = any(is_depth_topic(t) for t in topics)
        rows, cols = self._grid_dimensions(len(topics), has_depth)
        for idx, topic in enumerate(topics):
            row, col = divmod(idx, cols)
            panel = self.panels[topic]
            panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.grid_layout.addWidget(panel, row, col)
        self._apply_grid_stretches(rows, cols)

        self.status_bar.showMessage(f"显示 {len(topics)} 路图像")

    def _cached_topic_names(self) -> List[str]:
        return list(self._frame_cache.keys())

    def _get_paired_depth_topic_name(self, color_topic: str) -> Optional[str]:
        return find_paired_depth_topic(color_topic, self._cached_topic_names())

    def _is_topic_enabled(self, topic: str) -> bool:
        checkbox = self.topic_checks.get(topic)
        return checkbox is not None and checkbox.isChecked()

    def _is_paired_depth_enabled(self, color_topic: str) -> bool:
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if depth_topic is None:
            return False
        return self._is_topic_enabled(depth_topic)

    def _get_paired_depth_frame(self, color_topic: str) -> Optional[np.ndarray]:
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if depth_topic is None:
            return None
        return self._frame_cache.get(depth_topic)

    def _get_paired_color_frame(self, depth_topic: str) -> Optional[np.ndarray]:
        color_topic = find_paired_color_topic(depth_topic, self._cached_topic_names())
        if color_topic is None:
            return None
        frame = self._frame_cache.get(color_topic)
        if frame is None or frame.ndim < 2:
            return None
        return frame

    def _on_frame_updated(self, topic: str, cv_image: object) -> None:
        image = np.asarray(cv_image)
        self._frame_cache[topic] = image
        panel = self.panels.get(topic)
        if panel is None:
            return
        if isinstance(panel, DepthPanel3D):
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            panel.update_depth(image)
        elif isinstance(panel, CameraPanel):
            panel.update_frame(image)

    def _on_frame_stats(self, topic: str, count: int, _timestamp: float) -> None:
        self._received_topics[topic] = count

    def _update_waiting_hint(self) -> None:
        if not self.panels:
            return
        receiving = sum(1 for t in self.panels if self._received_topics.get(t, 0) > 0)
        waiting = len(self.panels) - receiving
        if waiting > 0 and receiving == 0:
            self.status_bar.showMessage(
                f"已订阅 {len(self.panels)} 路，但未收到图像 — 请确认 camera 节点正在发布数据"
            )
        elif waiting > 0:
            self.status_bar.showMessage(
                f"显示 {receiving}/{len(self.panels)} 路图像，{waiting} 路仍等待中"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt5 图形界面显示 /camera 开头的 ROS2 topic")
    parser.add_argument("--prefix", default="/camera", help="topic 前缀（默认: /camera）")
    parser.add_argument(
        "--llm-api-base",
        default=os.environ.get("LLM_API_BASE", LLM_API_BASE_DEFAULT),
        help="大模型 OpenAI 兼容 API 地址",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
        help="大模型名称",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setQuitOnLastWindowClosed(True)

    bridge = RosBridge()
    node = CameraTopicNode(bridge, prefix=args.prefix)
    llm_config = LlmChatConfig(
        api_base=args.llm_api_base,
        model=args.llm_model,
        api_key=os.environ.get(LLM_API_KEY_ENV, "").strip(),
    )

    window = CameraTopicWindow(node, bridge, prefix=args.prefix, llm_config=llm_config)
    window.setAttribute(Qt.WA_QuitOnClose, True)
    window.show()

    shutdown_flag = {"value": False}
    cleaned_up = {"value": False}

    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def cleanup() -> None:
        if cleaned_up["value"]:
            return
        cleaned_up["value"] = True
        shutdown_flag["value"] = True
        sig_timer.stop()
        try:
            node.prepare_shutdown()
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        ros_thread.join(timeout=1.0)
        if rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def ros_spin_wrapper() -> None:
        while rclpy.ok() and not shutdown_flag["value"]:
            try:
                executor.spin_once(timeout_sec=0.05)
            except ExternalShutdownException:
                break
            except Exception:
                if shutdown_flag["value"]:
                    break

    ros_thread = threading.Thread(target=ros_spin_wrapper, daemon=True)
    ros_thread.start()

    def on_sigint(*_args) -> None:
        if shutdown_flag["value"]:
            os._exit(130)
        app.quit()

    signal.signal(signal.SIGINT, on_sigint)

    # Qt 事件循环会阻塞 Python 信号处理，定时器让 Ctrl+C 能及时生效
    sig_timer = QTimer(app)
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(50)

    app.aboutToQuit.connect(cleanup)

    result = 0
    try:
        result = app.exec_()
    except KeyboardInterrupt:
        app.quit()
    finally:
        cleanup()
    return result


if __name__ == "__main__":
    sys.exit(main())
