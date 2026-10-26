"""Calibrated click-to-pose inputs. No robot commands or guessed metric depth.

Calibration describes RECTIFIED color and depth registered to that color camera.
An explicit configuration is required: equal image sizes do not prove alignment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    stamp: float
    received: float
    frame_id: str
    encoding: str
    displayed: object = None


class FrameBuffer:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.frames = {}
        self.lock = threading.Lock()

    def add(self, topic, image, stamp, frame_id, encoding, displayed=None):
        with self.lock:
            self.frames.setdefault(topic, deque(maxlen=self.capacity)).append(
                Frame(image, stamp, time.monotonic(), frame_id, encoding,
                      image if displayed is None else displayed)
            )

    def pair(self, color_topic, depth_topic, displayed, max_age=0.8, max_skew=0.08):
        with self.lock:
            colors = list(self.frames.get(color_topic, ()))
            depths = list(self.frames.get(depth_topic, ()))
        # Use the actual clicked frame, not whichever frame arrived most recently.
        color = next((f for f in reversed(colors) if f.displayed is displayed), None)
        if color is None:
            raise ValueError("点击图像不在实时 ROS 帧缓存中，请等待新帧再点击")
        if not depths:
            raise ValueError("未收到配对深度帧；请检查深度发布、订阅和 QoS")
        if color.stamp <= 0:
            raise ValueError("彩色图缺少有效采集时间戳")
        depths = [f for f in depths if f.stamp > 0]
        if not depths:
            raise ValueError("深度图缺少有效采集时间戳")
        depth = min(depths, key=lambda f: abs(f.stamp - color.stamp))
        now = time.monotonic()
        if max(now - color.received, now - depth.received) > max_age:
            raise ValueError("RGB-D 帧已过期，请等待新帧")
        skew = abs(depth.stamp - color.stamp)
        if skew > max_skew:
            raise ValueError(f"RGB-D 不同步：{skew * 1000:.1f} ms > {max_skew * 1000:.0f} ms")
        return color, depth, skew


def load_calibration(path, topic, shape):
    with open(path, encoding="utf-8") as stream:
        entry = json.load(stream).get("cameras", {}).get(topic)
    if not entry:
        raise ValueError(f"标定文件缺少 {topic}")
    if entry.get("rectified") is not True or entry.get("depth_registered_to_color") is not True:
        raise ValueError("需要已去畸变彩色图及注册到该彩色相机的深度；不能只缩放图像代替对齐")
    h, w = shape[:2]
    if (entry.get("width"), entry.get("height")) != (w, h):
        raise ValueError("图像分辨率与标定不一致；请提供该分辨率对应的内参")
    K = np.asarray(entry["K"], dtype=float).reshape(3, 3)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("相机内参无效")
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8:
        raise ValueError("需要标准零 skew 针孔内参")
    if not entry.get("optical_frame") or not entry.get("depth_topic"):
        raise ValueError("标定缺少 optical_frame 或 depth_topic")
    transform = entry.get("T_base_camera")
    if transform is not None:
        validate_transform(transform)
        if not entry.get("base_frame"):
            raise ValueError("配置外参时必须指定 base_frame")
    return entry, K


def validate_transform(value):
    T = np.asarray(value, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-3)):
        raise ValueError("位姿必须是米制平移和有效右手系刚体旋转")
    return T


def metric_depth(frame, shape):
    if frame.image.shape != tuple(shape[:2]):
        raise ValueError("RGB 与注册深度尺寸不一致；禁止通过 resize 猜测对齐")
    if frame.encoding.upper() == "16UC1" and frame.image.dtype == np.uint16:
        return frame.image.astype(np.float32) / 1000.0
    if frame.encoding.upper() == "32FC1" and frame.image.dtype == np.float32:
        return frame.image.copy()
    raise ValueError(f"不支持的深度编码/类型：{frame.encoding}/{frame.image.dtype}")


def object_points(mask, depth, K, seed, min_points=100, min_fraction=0.6):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape:
        raise ValueError("模型分割尺寸与 RGB-D 不一致")
    u, v = seed
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]) or not mask[v, u]:
        raise ValueError("分割结果未包含点击点，请重新选点")
    count = int(mask.sum())
    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    fraction = float(valid.sum()) / max(count, 1)
    if int(valid.sum()) < min_points or fraction < min_fraction:
        raise ValueError(f"目标深度不足：{int(valid.sum())} 点，有效率 {fraction:.0%}")
    ys, xs = np.where(valid)
    z = depth[ys, xs]
    points = np.column_stack(((xs - K[0, 2]) * z / K[0, 0],
                              (ys - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
    return points, valid, {"valid_depth_fraction": fraction, "point_count": len(points)}


def geometric_preview(points):
    """Visible-surface geometry only; axes have sign/symmetry ambiguity."""
    center = np.median(points, axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
    R = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    local = (points - center) @ R
    lo, hi = np.percentile(local, [5, 95], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center + R @ ((lo + hi) / 2)
    return {"T_camera_object": T.tolist(), "extents_m": (hi - lo).tolist(),
            "method": "visible_surface_pca", "pose_kind": "geometry_preview",
            "orientation_ambiguous": True}


def format_report(result):
    if not result.get("ok"):
        return "点击估计失败：" + str(result.get("error", "未知错误"))
    pose = result["object_pose"]
    xyz = np.asarray(pose["T_camera_object"])[:3, 3]
    lines = [f"物体位姿（{result['camera_frame']}，米）",
             f"XYZ: {xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}",
             f"方法: {pose['method']} / {pose['pose_kind']}",
             f"有效深度: {result['quality']['valid_depth_fraction']:.1%}；"
             f"RGB-D 时间差: {result['skew_ms']:.1f} ms",
             "旋转矩阵:\n" + np.array2string(np.asarray(pose['T_camera_object'])[:3, :3], precision=4),
             "抓取候选: 未生成（需要适配手型的抓取模型）；物体位姿不是 TCP 目标",
             "执行状态: 仅展示，未发送机器人命令"]
    if "T_base_object" in result:
        lines.append(f"机器人坐标 ({result['base_frame']}): " + str(np.asarray(result['T_base_object'])[:3, 3].round(4).tolist()))
    else:
        lines.append("未配置手眼外参，仅输出相机坐标")
    lines.extend(result.get("notes", []))
    grasps = result.get("grasps", {})
    if grasps.get("ok"):
        lines[5:5] = [f"GraspGen 候选: {len(grasps['candidates'])}，手型 {grasps['gripper']}",
                       "模型分数不是成功率；候选尚未完成场景碰撞、IK 与手型实机验证"]
        lines = [line for line in lines if not line.startswith("抓取候选: 未生成")]
        for i, candidate in enumerate(grasps["candidates"][:3]):
            lines.append(f"候选 {i + 1}: score={candidate['model_score']:.3f}, XYZ="
                         + str(np.asarray(candidate['T_camera_grasp'])[:3, 3].round(4).tolist()))
    return "\n".join(lines)


def run_graspgen(points, settings):
    """Execute an explicitly configured, hand-matched model in its own env."""
    required = ("python", "config", "gripper")
    if any(not settings.get(key) for key in required):
        raise ValueError("GraspGen 需要 python、config 和 gripper；不能用通用夹爪替代当前灵巧手")
    worker = str(Path(__file__).with_name("graspgen_worker.py"))
    with tempfile.TemporaryDirectory(prefix="click_grasp_") as directory:
        points_path = os.path.join(directory, "points.npy")
        output = os.path.join(directory, "result.json")
        # Bound IPC size without synthesizing unobserved surfaces.
        indices = np.linspace(0, len(points) - 1, min(len(points), 20000), dtype=int)
        np.save(points_path, points[indices])
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run([settings["python"], worker, "--points", points_path,
                               "--config", settings["config"], "--gripper", settings["gripper"],
                               "--output", output], capture_output=True, text=True, timeout=180, env=env)
        if not os.path.isfile(output):
            raise RuntimeError("GraspGen 未产生结果: " + proc.stderr[-1500:])
        result = json.loads(Path(output).read_text())
        if proc.returncode or not result.get("ok"):
            raise RuntimeError(result.get("error", proc.stderr[-1500:]))
    if result.get("gripper") != settings["gripper"]:
        raise ValueError("抓取模型响应手型不匹配")
    for candidate in result.get("candidates", []):
        validate_transform(candidate["T_camera_grasp"])
        if not np.isfinite(candidate["model_score"]):
            raise ValueError("抓取模型分数无效")
    result["execution_ready"] = False
    return result
