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
_eai_domain_robot_kinematics=module_ref('eai_app.domain.robot.kinematics')
_eai_ui_robot_runtime=module_ref('eai_app.ui.robot.runtime')

try:
    from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer
except Exception:  # pragma: no cover
    QOpenGLShader = None  # type: ignore
    QOpenGLShaderProgram = None  # type: ignore
    QOpenGLBuffer = None  # type: ignore

def _libgl():
    lib = ctypes.CDLL("libGL.so.1")
    lib.glEnable.argtypes = [ctypes.c_uint]
    lib.glViewport.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.glClearColor.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.glClear.argtypes = [ctypes.c_uint]
    lib.glDrawArrays.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int]
    lib.glPointSize.argtypes = [ctypes.c_float]
    lib.glLineWidth.argtypes = [ctypes.c_float]
    return lib

def _mesh_path(filename: str, urdf_path: str) -> str:
    name = (filename or "").strip()
    if name.startswith("file://"):
        return name[len("file://") :]
    root = os.path.dirname(urdf_path)
    marker = "package://wbc/assets/A2D_NoHand/"
    if marker in name:
        return os.path.join(root, name.split(marker, 1)[1])
    if name.startswith("package://"):
        return os.path.join(root, "meshes", os.path.basename(name))
    if os.path.isabs(name):
        return name
    return os.path.join(root, name)

def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx

def _xyz_rpy(text_xyz: str, text_rpy: str) -> np.ndarray:
    xyz = [float(x) for x in (text_xyz or "0 0 0").split()]
    rpy = [float(x) for x in (text_rpy or "0 0 0").split()]
    while len(xyz) < 3:
        xyz.append(0.0)
    while len(rpy) < 3:
        rpy.append(0.0)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz[:3]
    return T

def _parse_scale(text: str) -> np.ndarray:
    vals = [float(x) for x in (text or "1 1 1").split()]
    while len(vals) < 3:
        vals.append(1.0)
    return np.asarray(vals[:3], dtype=np.float64)

def _load_obj(path: str, max_faces: int = _eai_ui_robot_runtime._MAX_FACES) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    i = int(head)
                    if i < 0:
                        i = len(verts) + i
                    else:
                        i -= 1
                    idx.append(i)
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    step = max(1, len(faces) // max_faces)
    if step > 1:
        faces = faces[::step]
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)

def _bake_mesh(verts: np.ndarray, faces: np.ndarray, visual_T: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """返回 Nx6 float32：xyz + 法线，已乘上 visual 原点。"""
    if len(faces) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    tri = verts[faces]  # F,3,3
    tri = tri * scale.astype(np.float32)
    R = visual_T[:3, :3].astype(np.float32)
    t = visual_T[:3, 3].astype(np.float32)
    tri = np.einsum("ij,fkj->fki", R, tri) + t
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-8)
    packed = np.empty((tri.shape[0] * 3, 6), dtype=np.float32)
    packed[:, :3] = tri.reshape(-1, 3)
    packed[:, 3:] = np.repeat(n, 3, axis=0)
    return packed

def _link_color(name: str) -> Tuple[float, float, float]:
    if name.endswith("_l") or name.endswith("_left") or "_l_" in name:
        return (0.35, 0.58, 0.86)
    if name.endswith("_r") or name.endswith("_right") or "_r_" in name:
        return (0.90, 0.52, 0.28)
    return (0.72, 0.74, 0.77)

def load_visuals(urdf_path: Optional[str] = None, hand_only=False) -> List[dict]:
    urdf_path = urdf_path or _eai_domain_robot_kinematics._resolve_urdf()
    root = ET.parse(urdf_path).getroot()
    visuals: List[dict] = []
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        # NoHand's fixed extension frames duplicate the Link7 mesh at an
        # 11 cm offset. Keep the frames in FK, but do not render fake wrists.
        if link_name in ('Link_l_ext', 'Link_r_ext'):
            continue
        if hand_only and not link_name.startswith(('hand1_', 'hand2_', 'wrist_connector_', 'wrist_camera_')):
            continue
        color = _link_color(link_name)
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = _mesh_path(mesh.get("filename") or "", urdf_path)
            if not os.path.isfile(path):
                continue
            origin = visual.find("origin")
            T = _xyz_rpy(
                origin.get("xyz") if origin is not None else "0 0 0",
                origin.get("rpy") if origin is not None else "0 0 0",
            )
            scale = _parse_scale(mesh.get("scale") or "1 1 1")
            verts, faces = _load_obj(path)
            packed = _bake_mesh(verts, faces, T, scale)
            if len(packed) == 0:
                continue
            visuals.append({"link": link_name, "color": color, "packed": packed})
    return visuals

class RuiyanHandModel:
    """Hands, wrist brackets and cameras attached to the existing Link7 frames."""
    def __init__(self):
        self.path = os.path.join(str(PROJECT_ROOT), '..', 'psi-policy', 'assets',
                                 'A2D_RuiYan_D405', 'A2D_RuiYan_D405.urdf')
        self.visuals = load_visuals(self.path, hand_only=True)
        expected = {'wrist_connector_0', 'wrist_connector_1', 'wrist_camera_0', 'wrist_camera_1'}
        if len(self.visuals) < 28 or not expected.issubset({item['link'] for item in self.visuals}):
            raise ValueError('睿研左右手、腕部支架或相机网格不完整')
        self.joints = []
        for joint in ET.parse(self.path).getroot().findall('joint'):
            child = joint.find('child').get('link')
            if not child.startswith(('hand1_', 'hand2_', 'wrist_connector_', 'wrist_camera_')):
                continue
            origin = joint.find('origin')
            limit = joint.find('limit')
            axis = joint.find('axis')
            self.joints.append((joint.find('parent').get('link'), child,
                _xyz_rpy(origin.get('xyz'), origin.get('rpy')),
                None if limit is None else (float(limit.get('lower')), float(limit.get('upper'))),
                np.array([float(x) for x in axis.get('xyz').split()]) if axis is not None else np.array([0., 0., 1.])))

    def poses(self, arm_poses, motors):
        from eai_app.domain.robot.hand_geometry import _MOTOR
        frames = {'arm1_link_7': arm_poses['Link7_l'], 'arm2_link_7': arm_poses['Link7_r']}
        pending = list(self.joints)
        while pending:
            ready = [joint for joint in pending if joint[0] in frames]
            if not ready:
                raise ValueError('睿研手关节树不完整')
            for joint in ready:
                parent, child, origin, limits, axis = joint
                T = origin.copy()
                if limits is not None:
                    side = 'left' if child.startswith('hand1_') else 'right'
                    channel = _MOTOR[child.split('_link_')[1]]
                    value = float(np.clip(motors.get(side, {}).get(channel, 0.), 0., 1.))
                    angle = limits[0] + value * (limits[1] - limits[0])
                    a = axis / np.linalg.norm(axis)
                    cross = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
                    R = np.eye(3) + np.sin(angle)*cross + (1.-np.cos(angle))*(cross @ cross)
                    T[:3, :3] = T[:3, :3] @ R
                frames[child] = frames[parent] @ T
                pending = [entry for entry in pending if entry is not joint]
        return {key: value for key, value in frames.items() if key.startswith(('hand', 'wrist_'))}

def _grid_lines(size: float = 2.0, step: float = 0.25) -> np.ndarray:
    pts = []
    n = int(size / step)
    for i in range(-n, n + 1):
        y = i * step
        pts.extend([(-size, y, 0.0), (size, y, 0.0)])
        x = i * step
        pts.extend([(x, -size, 0.0), (x, size, 0.0)])
    arr = np.asarray(pts, dtype=np.float32)
    packed = np.zeros((len(arr), 6), dtype=np.float32)
    packed[:, :3] = arr
    packed[:, 5] = 1.0
    return packed

