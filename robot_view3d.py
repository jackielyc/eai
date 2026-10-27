"""在界面里画 A2D 模型，关节角来自 host_fk 写的状态文件。

不嵌 RViz：RViz 是另一个进程，塞进 PyQt 后鼠标和 OpenGL 会互相抢。
"""
from __future__ import annotations

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
from perception_geometry import valid_depth_mask, canonical_color_topic
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import (
    QMatrix4x4,
    QSurfaceFormat,
    QVector3D,
)
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QOpenGLWidget, QVBoxLayout, QWidget

try:
    from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer
except Exception:  # pragma: no cover
    QOpenGLShader = None  # type: ignore
    QOpenGLShaderProgram = None  # type: ignore
    QOpenGLBuffer = None  # type: ignore

JOINT_NAMES = [
    "joint_lift_body",
    "joint_body_pitch",
    "joint1_l",
    "joint2_l",
    "joint3_l",
    "joint4_l",
    "joint5_l",
    "joint6_l",
    "joint7_l",
    "joint1_r",
    "joint2_r",
    "joint3_r",
    "joint4_r",
    "joint5_r",
    "joint6_r",
    "joint7_r",
]

_MAX_FACES = 40000

_GL_DEPTH_TEST = 0x0B71
_GL_COLOR_BUFFER_BIT = 0x00004000
_GL_DEPTH_BUFFER_BIT = 0x00000100
_GL_TRIANGLES = 0x0004
_GL_LINES = 0x0001
_GL_POINTS = 0x0000
_GL_FLOAT = 0x1406


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

_VERT = """
attribute vec3 a_pos;
attribute vec3 a_nrm;
uniform mat4 u_mvp;
uniform mat4 u_model;
varying vec3 v_nrm;
varying vec3 v_rgb;
void main() {
    gl_PointSize = 3.0;
    gl_Position = u_mvp * u_model * vec4(a_pos, 1.0);
    v_rgb = a_nrm;
    v_nrm = vec3(
        u_model[0].x * a_nrm.x + u_model[1].x * a_nrm.y + u_model[2].x * a_nrm.z,
        u_model[0].y * a_nrm.x + u_model[1].y * a_nrm.y + u_model[2].y * a_nrm.z,
        u_model[0].z * a_nrm.x + u_model[1].z * a_nrm.y + u_model[2].z * a_nrm.z
    );
}
"""

_FRAG = """
varying vec3 v_nrm;
varying vec3 v_rgb;
uniform vec3 u_color;
uniform float u_lit;
void main() {
    if (u_lit < 0.0) {
        gl_FragColor = vec4(clamp(v_rgb, 0.0, 1.0), 1.0);
    } else {
        vec3 n = normalize(v_nrm);
        float ndl = abs(dot(n, normalize(vec3(0.35, 0.45, 0.82))));
        float shade = mix(1.0, 0.28 + 0.72 * ndl, u_lit);
        gl_FragColor = vec4(u_color * shade, 1.0);
    }
}
"""


def _resolve_urdf() -> str:
    env = os.environ.get("EAI_HOST_FK_URDF", "").strip()
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        env,
        os.path.normpath(
            os.path.join(
                here,
                "..",
                "a2d-tele",
                "modules",
                "control",
                "wbc",
                "assets",
                "A2D_NoHand",
                "A2D_NoHand_Flattened.urdf",
            )
        ),
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("找不到 A2D_NoHand_Flattened.urdf")


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


def _load_obj(path: str, max_faces: int = _MAX_FACES) -> Tuple[np.ndarray, np.ndarray]:
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


def load_visuals(urdf_path: Optional[str] = None) -> List[dict]:
    urdf_path = urdf_path or _resolve_urdf()
    root = ET.parse(urdf_path).getroot()
    visuals: List[dict] = []
    for link in root.findall("link"):
        link_name = link.get("name") or ""
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


class _Kinematics:
    def __init__(self, urdf_path: str) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._q_idx = {}
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            self._q_idx[name] = self.model.joints[jid].idx_q
        self._frame = {}
        for fid in range(self.model.nframes):
            frame = self.model.frames[fid]
            self._frame[frame.name] = fid
        self.q = pin.neutral(self.model)

    def link_poses(self, q16: Sequence[float]) -> Dict[str, np.ndarray]:
        pin = self._pin
        for i, name in enumerate(JOINT_NAMES):
            if i < len(q16):
                self.q[self._q_idx[name]] = float(q16[i])
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        out: Dict[str, np.ndarray] = {}
        for name, fid in self._frame.items():
            out[name] = np.array(self.data.oMf[fid].homogeneous, dtype=np.float64)
        return out


def read_q16(max_age_s: float = 2.0) -> Tuple[Optional[List[float]], float]:
    path = os.environ.get(
        "EAI_HOST_FK_STATE",
        os.path.join(tempfile.gettempdir(), "eai_host_fk_state.json"),
    )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        q = data.get("q16")
        stamp = float(data.get("t") or 0.0)
        age = time.time() - stamp
        if isinstance(q, list) and len(q) >= 16 and age <= max_age_s:
            return [float(x) for x in q[:16]], age
        if isinstance(q, list) and len(q) >= 16:
            return [float(x) for x in q[:16]], age
    except Exception:
        return None, -1.0
    return None, -1.0


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


class _RobotGL(QOpenGLWidget):
    def __init__(self, visuals: List[dict], parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        super().__init__(parent)
        self.setFormat(fmt)
        self._visuals = visuals
        self._poses: Dict[str, np.ndarray] = {}
        self._ready = False
        self._program = None
        self._libgl = None
        self._items: List[dict] = []
        self._grid = None
        self._clouds: List[dict] = []
        self._skeletons: List[dict] = []
        self._wrenches: List[dict] = []
        self._pose_items: List[dict] = []
        self._yaw = 35.0
        self._pitch = 22.0
        self._distance = 3.2
        self._target = QVector3D(0.25, 0.0, 0.85)
        self._last_pos: Optional[QPoint] = None
        self.setMinimumHeight(360)

    def set_poses(self, poses: Dict[str, np.ndarray]) -> None:
        self._poses = poses
        self.update()

    def set_clouds(self, clouds: List[dict]) -> None:
        self._clouds = clouds
        self.update()

    def set_skeletons(self, skeletons: List[dict]) -> None:
        self._skeletons = skeletons
        self.update()

    def set_wrenches(self, wrenches: List[dict]) -> None:
        self._wrenches = wrenches
        self.update()

    def set_pose_items(self, items: List[dict]) -> None:
        self._pose_items = items
        self.update()

    def initializeGL(self) -> None:
        self._libgl = _libgl()
        self._program = QOpenGLShaderProgram(self)
        if not self._program.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERT):
            raise RuntimeError(self._program.log())
        if not self._program.addShaderFromSourceCode(QOpenGLShader.Fragment, _FRAG):
            raise RuntimeError(self._program.log())
        self._program.bindAttributeLocation("a_pos", 0)
        self._program.bindAttributeLocation("a_nrm", 1)
        if not self._program.link():
            raise RuntimeError(self._program.log())
        self._items = [self._upload(item["packed"], item["link"], item["color"]) for item in self._visuals]
        self._grid = self._upload(_grid_lines(), "", (0.28, 0.30, 0.32), lines=True)
        self._ready = True

    def _upload(self, packed: np.ndarray, link: str, color, lines: bool = False) -> dict:
        buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        buf.create()
        buf.bind()
        raw = packed.astype(np.float32, copy=False).tobytes()
        buf.allocate(raw, len(raw))
        buf.release()
        return {
            "buf": buf,
            "count": int(packed.shape[0]),
            "link": link,
            "color": color,
            "lines": lines,
        }

    def resizeGL(self, w: int, h: int) -> None:
        if self._libgl is None:
            return
        self._libgl.glViewport(0, 0, max(1, int(w)), max(1, int(h)))

    def paintGL(self) -> None:
        gl = self._libgl
        if gl is None:
            return
        gl.glViewport(0, 0, int(self.width()), int(self.height()))
        gl.glEnable(_GL_DEPTH_TEST)
        gl.glClearColor(0.12, 0.13, 0.15, 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        if not self._ready or self._program is None:
            return
        mvp = self._mvp()
        prog = self._program
        prog.bind()
        prog.setUniformValue("u_mvp", mvp)
        ident = QMatrix4x4()
        ident.setToIdentity()
        self._draw_item(self._grid, ident, lit=0.0, mode=_GL_LINES)
        for item in self._items:
            pose = self._poses.get(item["link"])
            if pose is None:
                continue
            self._draw_item(item, _qt_matrix(pose), lit=1.0, mode=_GL_TRIANGLES)
        for cloud in self._clouds:
            self._flush_cloud(cloud)
            if not cloud.get("enabled") or cloud.get("count", 0) <= 0:
                continue
            pose = self._poses.get(cloud.get("parent") or "")
            if pose is None:
                continue
            self._draw_item(
                cloud,
                _qt_matrix(pose @ cloud["T"]),
                lit=-1.0,
                mode=_GL_POINTS,
            )
        for skel in self._skeletons:
            self._flush_cloud(skel)
            if not skel.get("enabled") or skel.get("count", 0) <= 0:
                continue
            pose = self._poses.get(skel.get("parent") or "")
            if pose is None:
                continue
            self._draw_item(
                skel,
                _qt_matrix(pose @ skel["T"]),
                lit=-1.0,
                mode=_GL_LINES,
                line_width=2.5,
            )
        for wrench in self._wrenches:
            self._flush_cloud(wrench)
            if not wrench.get("enabled") or wrench.get("count", 0) <= 0:
                continue
            pose = self._poses.get(wrench.get("parent") or "")
            if pose is None:
                continue
            self._draw_item(
                wrench,
                _qt_matrix(pose @ wrench["T"]),
                lit=-1.0,
                mode=_GL_LINES,
                line_width=3.0,
            )
        ident = QMatrix4x4()
        ident.setToIdentity()
        gl.glDisable(_GL_DEPTH_TEST)
        for pose_item in self._pose_items:
            self._flush_cloud(pose_item)
            if not pose_item.get("enabled") or pose_item.get("count", 0) <= 0:
                continue
            parent = pose_item.get("parent") or ""
            if parent:
                pose = self._poses.get(parent)
                if pose is None and parent in ("base_link", "world", "map"):
                    pose = np.eye(4, dtype=np.float64)
                if pose is None:
                    continue
                model = _qt_matrix(pose @ pose_item["T"])
            else:
                model = ident
            draw_mode = int(pose_item.get("draw_mode", _GL_LINES))
            if draw_mode == _GL_POINTS:
                gl.glEnable(_GL_DEPTH_TEST)
            else:
                gl.glDisable(_GL_DEPTH_TEST)
            self._draw_item(
                pose_item,
                model,
                lit=-1.0,
                mode=draw_mode,
                line_width=2.5,
            )
        gl.glEnable(_GL_DEPTH_TEST)
        prog.release()

    def _flush_cloud(self, cloud: dict) -> None:
        lock = cloud.get("lock")
        if lock is None:
            pending = cloud.get("pending")
            cloud["pending"] = None
        else:
            with lock:
                pending = cloud.get("pending")
                cloud["pending"] = None
        if pending is None or pending.shape[0] == 0:
            if pending is not None:
                cloud["count"] = 0
            return
        if cloud.get("buf") is None:
            uploaded = self._upload(pending, "", cloud["color"])
            cloud["buf"] = uploaded["buf"]
            cloud["count"] = uploaded["count"]
            cloud["color"] = uploaded["color"]
            return
        raw = np.ascontiguousarray(pending, dtype=np.float32).tobytes()
        cloud["buf"].bind()
        cloud["buf"].allocate(raw, len(raw))
        cloud["buf"].release()
        cloud["count"] = int(pending.shape[0])

    def _draw_item(
        self,
        item: Optional[dict],
        model: QMatrix4x4,
        lit: float,
        mode: int = _GL_TRIANGLES,
        line_width: float = 1.0,
    ) -> None:
        if item is None or item["count"] <= 0:
            return
        prog = self._program
        prog.setUniformValue("u_model", model)
        prog.setUniformValue("u_color", QVector3D(*item["color"]))
        prog.setUniformValue("u_lit", float(lit))
        item["buf"].bind()
        stride = 6 * 4
        prog.enableAttributeArray(0)
        prog.enableAttributeArray(1)
        prog.setAttributeBuffer(0, _GL_FLOAT, 0, 3, stride)
        prog.setAttributeBuffer(1, _GL_FLOAT, 3 * 4, 3, stride)
        if mode == _GL_POINTS:
            self._libgl.glEnable(0x8642)
            self._libgl.glPointSize(float(item.get("point_size") or 3.0))
        elif mode == _GL_LINES:
            self._libgl.glLineWidth(float(line_width))
        self._libgl.glDrawArrays(mode, 0, item["count"])
        item["buf"].release()

    def _mvp(self) -> QMatrix4x4:
        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)
        cp = math.cos(pitch)
        eye = QVector3D(
            self._target.x() + self._distance * cp * math.cos(yaw),
            self._target.y() + self._distance * cp * math.sin(yaw),
            self._target.z() + self._distance * math.sin(pitch),
        )
        aspect = max(0.2, self.width() / max(1.0, float(self.height())))
        proj = QMatrix4x4()
        proj.perspective(40.0, aspect, 0.05, 40.0)
        view = QMatrix4x4()
        view.lookAt(eye, self._target, QVector3D(0.0, 0.0, 1.0))
        return proj * view

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._distance = float(np.clip(self._distance * factor, 0.4, 12.0))
        self.update()

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            self._last_pos = event.pos()
            return
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        self._last_pos = event.pos()
        if event.buttons() & Qt.RightButton:
            scale = self._distance * 0.002
            yaw = math.radians(self._yaw)
            right = QVector3D(math.sin(yaw), -math.cos(yaw), 0.0)
            up = QVector3D(0.0, 0.0, 1.0)
            self._target = self._target - right * (dx * scale) + up * (dy * scale)
        else:
            self._yaw -= dx * 0.4
            self._pitch = float(np.clip(self._pitch + dy * 0.3, -85.0, 85.0))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_pos = None


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
    from depth_cloud_node import _CAMERAS, _load_extrinsics, _placement

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
    from robot_hand_skeleton import skeleton_in_link

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


# /hal/{left,right}_arm_data 的 force_data = [Fx,Fy,Fz,Mx,My,Mz]，单位 N / N·m。
# force_coordinate=0 时按腕部传感器坐标系画在 Link7 上。
_GENIE_PREFIX_CANDIDATES = (
    "/home/psibot/Projects/rcl/robot-control-library/install/genie_msgs",
    "/home/psibot/a2d_sdk/forwarder/app",
    "/home/psibot/Projects/synrobot/install/genie_msgs",
)


def _genie_prefixes() -> List[str]:
    found: List[str] = []
    env_prefix = os.environ.get("EAI_GENIE_PREFIX", "").strip()
    if env_prefix:
        found.append(env_prefix)
    setup = os.environ.get("EAI_GENIE_SETUP", "").strip()
    if setup:
        parent = os.path.dirname(setup)
        found.append(parent)
        found.append(os.path.join(parent, "genie_msgs"))
    found.extend(_GENIE_PREFIX_CANDIDATES)
    unique: List[str] = []
    for prefix in found:
        if prefix and prefix not in unique and os.path.isdir(prefix):
            unique.append(prefix)
    return unique


def _drop_genie_modules() -> None:
    for name in list(sys.modules):
        if name == "genie_msgs" or name.startswith("genie_msgs."):
            sys.modules.pop(name, None)


def _preload_genie_fastrtps() -> None:
    """按绝对路径载入类型库。进程启动后改 LD_LIBRARY_PATH 不会被 dlopen 看到。"""
    # 先载入 generator，再载入 fastrtps / introspection。RMW 按短文件名 dlopen。
    names = (
        "libgenie_msgs__rosidl_generator_c.so",
        "libgenie_msgs__rosidl_generator_py.so",
        "libgenie_msgs__rosidl_typesupport_c.so",
        "libgenie_msgs__rosidl_typesupport_cpp.so",
        "libgenie_msgs__rosidl_typesupport_fastrtps_c.so",
        "libgenie_msgs__rosidl_typesupport_fastrtps_cpp.so",
        "libgenie_msgs__rosidl_typesupport_introspection_c.so",
        "libgenie_msgs__rosidl_typesupport_introspection_cpp.so",
    )
    mode = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL)
    for prefix in _genie_prefixes():
        libdir = os.path.join(prefix, "lib")
        target = os.path.join(libdir, "libgenie_msgs__rosidl_typesupport_fastrtps_c.so")
        if not os.path.isfile(target):
            continue
        for name in names:
            path = os.path.join(libdir, name)
            if os.path.isfile(path):
                ctypes.CDLL(path, mode=mode)
        return


def import_arm_state():
    """导入 genie_msgs.ArmState。界面进程默认没有这条 PYTHONPATH。"""
    try:
        from genie_msgs.msg import ArmState
        _preload_genie_fastrtps()
        return ArmState
    except Exception as exc:
        last = exc
        _drop_genie_modules()
    for prefix in _genie_prefixes():
        site = os.path.join(prefix, "local", "lib", "python3.10", "dist-packages")
        marker = os.path.join(
            prefix, "share", "ament_index", "resource_index", "packages", "genie_msgs"
        )
        if os.path.isdir(site) and site not in sys.path:
            sys.path.insert(0, site)
        if os.path.isfile(marker):
            current = os.environ.get("AMENT_PREFIX_PATH", "")
            parts = [part for part in current.split(":") if part]
            if prefix not in parts:
                os.environ["AMENT_PREFIX_PATH"] = prefix + ((":" + current) if current else "")
        _drop_genie_modules()
        try:
            from genie_msgs.msg import ArmState
            _preload_genie_fastrtps()
            return ArmState
        except Exception as exc:
            last = exc
            _drop_genie_modules()
    raise ImportError(f"缺少 genie_msgs：{last}")


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
    from depth_cloud_node import _intrinsics

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


class RobotView3D(QWidget):
    """可嵌进手臂页的机器人视图。勾选要显示的深度。set_running(False) 时停止刷新。"""

    def __init__(self, parent=None, ros_node=None) -> None:
        super().__init__(parent)
        self._ros_node = ros_node
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("深度"))
        self._depth_on = {"head": True, "left": True, "right": True}
        self._depth_checks = {}
        for _topic, key, label, tip in _DEPTH_KEYS:
            box = QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(tip)
            box.toggled.connect(lambda checked, name=key: self._set_depth_enabled(name, checked))
            row.addWidget(box)
            self._depth_checks[key] = box
        self._skel_check = QCheckBox("手骨架")
        self._skel_check.setChecked(True)
        self._skel_check.setToolTip("用当前手腕位姿和睿研手的开合量，在左右手上画出估计骨架。")
        self._skel_check.toggled.connect(self._set_skeleton_enabled)
        row.addWidget(self._skel_check)
        self._pose_checks = {}
        for key, label, _rgb in _POSE_METHODS:
            box = QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(f"画出最近一次 {label} 的包围盒和坐标轴。")
            box.toggled.connect(lambda checked, name=key: self._set_pose_enabled(name, checked))
            row.addWidget(box)
            self._pose_checks[key] = box
        self._sam3_check = QCheckBox("SAM3")
        self._sam3_check.setChecked(True)
        self._sam3_check.setToolTip("SAM3 掩码结合深度和相机标定生成的三维点云；二维原始掩码另存于结果目录。")
        self._sam3_check.setToolTip("画出最近一次估计里 SAM3 分出的物体点云。")
        self._sam3_check.toggled.connect(self._set_sam3_enabled)
        row.addWidget(self._sam3_check)
        self._lingbot_check = QCheckBox("LingBot")
        self._lingbot_check.setChecked(True)
        self._lingbot_check.setToolTip(
            "画出最近一次 LingBot-Depth 点云，挂在对应相机上。可取消勾选。"
        )
        self._lingbot_check.toggled.connect(self._set_lingbot_enabled)
        row.addWidget(self._lingbot_check)
        self._wrench_check = QCheckBox("六维力")
        self._wrench_check.setChecked(True)
        self._wrench_check.setToolTip(
            "在左右手腕 Link7 上画出 /hal/*_arm_data 的力（橙）和力矩（青），以及力的 RGB 分量。"
        )
        self._wrench_check.toggled.connect(self._set_wrench_enabled)
        row.addWidget(self._wrench_check)
        row.addStretch(1)
        layout.addLayout(row)
        self._wrench_readout = QLabel("六维力：等待 /hal/left_arm_data、/hal/right_arm_data")
        self._wrench_readout.setTextFormat(Qt.RichText)
        self._wrench_readout.setWordWrap(True)
        self._wrench_readout.setStyleSheet(
            "color: #e8eaed; background: #1e1f22; padding: 4px 8px;"
            "border-radius: 4px; font-family: monospace;"
        )
        layout.addWidget(self._wrench_readout)
        self._status = QLabel("正在加载机器人模型…")
        self._status.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(self._status)
        self._gl: Optional[_RobotGL] = None
        self._kin: Optional[_Kinematics] = None
        self._clouds: List[dict] = []
        self._cloud_by_topic: Dict[str, dict] = {}
        self._skels: List[dict] = []
        self._skel_on = True
        self._wrenches: List[dict] = []
        self._wrench_by_side: Dict[str, dict] = {}
        self._wrench_on = True
        self._pose_items: Dict[str, dict] = {}
        self._pose_on = {key: True for key, _label, _rgb in _POSE_METHODS}
        self._pose_hold: Dict[str, tuple] = {}
        self._pose_note: Dict[str, str] = {}
        self._sam3: Optional[dict] = None
        self._sam3_on = True
        self._sam3_hold = None
        self._sam3_note = ""
        self._lingbot: Optional[dict] = None
        self._lingbot_on = True
        self._lingbot_hold = None
        self._lingbot_colors = None
        self._lingbot_parent = ""
        self._lingbot_T = np.eye(4, dtype=np.float64)
        self._lingbot_note = ""
        self._lock = threading.Lock()
        self._error = ""
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        QTimer.singleShot(0, self._load)

    def _set_depth_enabled(self, key: str, enabled: bool) -> None:
        self._depth_on[key] = bool(enabled)
        for cloud in self._clouds:
            if cloud.get("key") == key:
                cloud["enabled"] = bool(enabled)
        if self._gl is not None:
            self._gl.update()

    def _set_skeleton_enabled(self, enabled: bool) -> None:
        self._skel_on = bool(enabled)
        for skel in self._skels:
            skel["enabled"] = bool(enabled)
        if self._gl is not None:
            self._gl.update()

    def _set_pose_enabled(self, key: str, enabled: bool) -> None:
        self._pose_on[key] = bool(enabled)
        item = self._pose_items.get(key)
        if item is not None:
            item["enabled"] = bool(enabled) and self._pose_hold.get(key) is not None
        if self._gl is not None:
            self._gl.update()

    def _set_sam3_enabled(self, enabled: bool) -> None:
        self._sam3_on = bool(enabled)
        if self._sam3 is not None:
            self._sam3["enabled"] = bool(enabled) and self._sam3_hold is not None
        if self._gl is not None:
            self._gl.update()

    def _set_lingbot_enabled(self, enabled: bool) -> None:
        self._lingbot_on = bool(enabled)
        if self._lingbot is not None:
            self._lingbot["enabled"] = bool(enabled) and self._lingbot_hold is not None
        if self._gl is not None:
            self._gl.update()

    def _set_wrench_enabled(self, enabled: bool) -> None:
        self._wrench_on = bool(enabled)
        for item in self._wrenches:
            item["enabled"] = bool(enabled) and item.get("force") is not None
        self._refresh_wrench_readout()
        if self._gl is not None:
            self._gl.update()

    def _refresh_wrench_readout(self) -> None:
        label = getattr(self, "_wrench_readout", None)
        if label is None:
            return
        label.setText(
            format_wrench_html(self._wrenches, enabled=self._wrench_on)
        )

    def set_sam3_points(self, points) -> None:
        """SAM3 分割点，Nx3，已经在 base_link。空则清掉。"""
        if points is None:
            self._sam3_hold = None
            self._sam3_note = ""
        else:
            arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if arr.shape[0] == 0 or not np.isfinite(arr).all():
                self._sam3_hold = None
                self._sam3_note = ""
            else:
                self._sam3_hold = arr.copy()
                self._sam3_note = f"SAM3 {arr.shape[0]} 点"
        self._sync_sam3()

    def _sync_sam3(self) -> None:
        item = self._sam3
        if item is None:
            return
        item["enabled"] = bool(self._sam3_on) and self._sam3_hold is not None
        if self._sam3_hold is None:
            packed = np.zeros((0, 6), dtype=np.float32)
        else:
            rgb = np.array((0.95, 0.2, 0.85), dtype=np.float32)
            packed = np.zeros((self._sam3_hold.shape[0], 6), dtype=np.float32)
            packed[:, :3] = self._sam3_hold
            packed[:, 3:] = rgb
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def set_lingbot_depth(
        self,
        points=None,
        colors=None,
        *,
        color_topic: str = "",
        note: str = "",
    ) -> None:
        """LingBot-Depth 点云（相机系）。挂在与实时深度同一套外参上。空则清掉。"""
        if points is None:
            self._lingbot_hold = None
            self._lingbot_colors = None
            self._lingbot_parent = ""
            self._lingbot_T = np.eye(4, dtype=np.float64)
            self._lingbot_note = ""
            self._sync_lingbot()
            return
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] == 0 or not np.isfinite(arr).all():
            self._lingbot_hold = None
            self._lingbot_colors = None
            self._lingbot_note = ""
            self._sync_lingbot()
            return
        keep = valid_depth_mask(arr[:, 2])
        if not np.any(keep):
            self.set_lingbot_depth(None)
            return
        if colors is not None:
            color_array = np.asarray(colors).reshape(-1, 3)
            colors = color_array[keep] if len(color_array) == len(arr) else None
        arr = arr[keep]
        if arr.shape[0] > 80000:
            step = int(np.ceil(arr.shape[0] / 80000.0))
            arr = arr[::step]
            if colors is not None:
                colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)[::step]
        rgb = None
        if colors is not None:
            rgb = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            if rgb.shape[0] != arr.shape[0] or not np.isfinite(rgb).all():
                rgb = None
            else:
                if float(np.nanmax(rgb)) > 1.5:
                    rgb = rgb / 255.0
                rgb = np.clip(rgb, 0.0, 1.0)
        topic = (color_topic or "").strip()
        if topic:
            parent, T, place_note = _placement_for_color(topic)
        else:
            parent, T, place_note = "base_link", np.eye(4, dtype=np.float64), "无相机 topic，按 base_link"
        self._lingbot_hold = arr.copy()
        self._lingbot_colors = None if rgb is None else rgb.copy()
        self._lingbot_parent = parent
        self._lingbot_T = np.asarray(T, dtype=np.float64)
        self._lingbot_note = f"{note or f'LingBot {arr.shape[0]} 点'} · {place_note}"
        self._sync_lingbot()

    def set_lingbot_depth_from_ply(self, ply_path: str, *, color_topic: str = "") -> None:
        """从 LingBot-Depth 写出的 point_cloud.ply 加载。"""
        path = os.path.abspath(os.path.expanduser(ply_path or ""))
        if not path or not os.path.isfile(path):
            self.set_lingbot_depth(None)
            return
        xyz, rgb = load_ply_xyz_rgb(path)
        self.set_lingbot_depth(
            xyz,
            rgb,
            color_topic=color_topic,
            note=f"LingBot {xyz.shape[0]} 点",
        )

    def _sync_lingbot(self) -> None:
        item = self._lingbot
        if item is None:
            return
        item["enabled"] = bool(self._lingbot_on) and self._lingbot_hold is not None
        item["parent"] = self._lingbot_parent or ""
        item["T"] = np.asarray(self._lingbot_T, dtype=np.float64)
        if self._lingbot_hold is None:
            packed = np.zeros((0, 6), dtype=np.float32)
        else:
            packed = np.zeros((self._lingbot_hold.shape[0], 6), dtype=np.float32)
            packed[:, :3] = self._lingbot_hold
            if self._lingbot_colors is not None and self._lingbot_colors.shape[0] == packed.shape[0]:
                packed[:, 3:6] = self._lingbot_colors
            else:
                packed[:, 3:6] = (0.35, 0.95, 0.45)
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def set_method_pose(self, key: str, xyz, rotation, corners=None, extents=None) -> None:
        """把某一估计方法的 6D 位姿画在 base_link 上。xyz 为空则清掉该方法。"""
        label = next((text for name, text, _rgb in _POSE_METHODS if name == key), key)
        if xyz is None or rotation is None:
            self._pose_hold.pop(key, None)
            self._pose_note.pop(key, None)
        else:
            point = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
            matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
            box = None if corners is None else np.asarray(corners, dtype=np.float64).copy()
            size = None if extents is None else tuple(float(v) for v in extents)
            self._pose_hold[key] = (point, matrix, box, size)
            self._pose_note[key] = f"{label} ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        self._sync_pose(key)

    def set_estimated_pose(self, xyz, rotation, corners=None, extents=None) -> None:
        self.set_method_pose("pca", xyz, rotation, corners, extents)

    def _sync_pose(self, key: str) -> None:
        item = self._pose_items.get(key)
        if item is None:
            return
        hold = self._pose_hold.get(key)
        item["enabled"] = bool(self._pose_on.get(key, True)) and hold is not None
        if hold is None:
            packed = np.zeros((0, 6), dtype=np.float32)
        else:
            point, matrix, box, size = hold
            packed = _pose6d_lines(point, matrix, box, size, item.get("box_rgb"))
        with item["lock"]:
            item["pending"] = packed
        if self._gl is not None:
            self._gl.update()

    def _sync_pose6d(self) -> None:
        for key, _label, _rgb in _POSE_METHODS:
            self._sync_pose(key)

    def set_running(self, running: bool) -> None:
        if running and self._gl is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._gl is not None:
            self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _load(self) -> None:
        try:
            urdf = _resolve_urdf()
            visuals = load_visuals(urdf)
            if not visuals:
                raise RuntimeError("URDF 里没有可用的视觉网格")
            self._kin = _Kinematics(urdf)
            self._gl = _RobotGL(visuals, self)
            self._clouds = self._make_clouds()
            self._skels = self._make_skeletons()
            self._wrenches = self._make_wrenches()
            self._pose_items = {
                key: self._make_pose6d(rgb) for key, _label, rgb in _POSE_METHODS
            }
            self._sam3 = self._make_sam3()
            self._lingbot = self._make_lingbot()
            self._gl.set_clouds(self._clouds)
            self._gl.set_skeletons(self._skels)
            self._gl.set_wrenches(self._wrenches)
            self._gl.set_pose_items(
                list(self._pose_items.values()) + [self._sam3, self._lingbot]
            )
            self._sync_pose6d()
            self._sync_sam3()
            self._sync_lingbot()
            self.layout().addWidget(self._gl, 1)
            try:
                self._subscribe_depth()
            except Exception as exc:
                self._status.setText(f"深度订阅失败：{exc}")
            try:
                self._subscribe_wrench()
            except Exception as exc:
                self._status.setText(f"六维力订阅失败：{exc}")
            self._status.setText("拖动旋转，右键平移，滚轮缩放。勾选要看的深度。")
            self._tick()
            if self.isVisible():
                self._timer.start()
        except Exception as exc:
            self._error = str(exc)
            self._status.setText(f"机器人视图加载失败：{exc}")

    def _make_clouds(self) -> List[dict]:
        from depth_cloud_node import _CAMERAS, _load_extrinsics, _placement

        cams = _load_extrinsics()
        key_of = {topic: key for topic, key, _label, _tip in _DEPTH_KEYS}
        color_of = {
            "/camera/head_depth": (80 / 255.0, 220 / 255.0, 220 / 255.0),
            "/camera/hand_left_depth": (70 / 255.0, 140 / 255.0, 230 / 255.0),
            "/camera/hand_right_depth": (230 / 255.0, 140 / 255.0, 60 / 255.0),
        }
        clouds: List[dict] = []
        self._cloud_by_topic = {}
        for spec in _CAMERAS:
            cam = dict(cams.get(spec["color"]) or {}, color_topic=spec["color"])
            parent, xyz, quat, note = _placement(cam, spec)
            key = key_of[spec["depth"]]
            label = next(text for topic, name, text, _tip in _DEPTH_KEYS if name == key)
            cloud = {
                "key": key,
                "label": label,
                "topic": spec["depth"],
                "color_topic": spec["color"],
                "parent": parent,
                "T": _T_from_xyz_quat(xyz, quat),
                "color": color_of[spec["depth"]],
                "enabled": bool(self._depth_on.get(key, True)),
                "cam": cam,
                "pending": None,
                "buf": None,
                "count": 0,
                "lock": threading.Lock(),
            }
            clouds.append(cloud)
            self._cloud_by_topic[spec["depth"]] = cloud
            box = self._depth_checks.get(key)
            if box is not None:
                box.setToolTip(note)
        return clouds

    def _make_skeletons(self) -> List[dict]:
        specs = (("left", "Link7_l"), ("right", "Link7_r"))
        skels = []
        for side, parent in specs:
            skels.append(
                {
                    "side": side,
                    "parent": parent,
                    "T": np.eye(4, dtype=np.float64),
                    "color": (1.0, 1.0, 1.0),
                    "enabled": bool(self._skel_on),
                    "pending": None,
                    "buf": None,
                    "count": 0,
                    "lock": threading.Lock(),
                }
            )
        return skels

    def _make_wrenches(self) -> List[dict]:
        items = []
        self._wrench_by_side = {}
        for side, topic, parent, label in _WRENCH_TOPICS:
            item = {
                "side": side,
                "topic": topic,
                "label": label,
                "parent": parent,
                "T": np.eye(4, dtype=np.float64),
                "color": (1.0, 0.35, 0.12),
                "enabled": False,
                "pending": None,
                "buf": None,
                "count": 0,
                "lock": threading.Lock(),
                "force": None,
                "torque": None,
                "force_state": None,
                "stamp": 0.0,
                "error": "",
            }
            items.append(item)
            self._wrench_by_side[side] = item
        return items

    def _make_pose6d(self, box_rgb=(1.0, 0.86, 0.15)) -> dict:
        return {
            "parent": "",
            "T": np.eye(4, dtype=np.float64),
            "color": (1.0, 1.0, 1.0),
            "box_rgb": tuple(box_rgb),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": threading.Lock(),
        }

    def _make_sam3(self) -> dict:
        return {
            "parent": "",
            "T": np.eye(4, dtype=np.float64),
            "color": (0.95, 0.2, 0.85),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": threading.Lock(),
            "draw_mode": _GL_POINTS,
            "point_size": 5.0,
        }

    def _make_lingbot(self) -> dict:
        return {
            "parent": self._lingbot_parent or "",
            "T": np.asarray(self._lingbot_T, dtype=np.float64),
            "color": (0.35, 0.95, 0.45),
            "enabled": False,
            "pending": None,
            "buf": None,
            "count": 0,
            "lock": threading.Lock(),
            "draw_mode": _GL_POINTS,
            "point_size": 3.0,
        }

    def _update_skeleton(self) -> None:
        if not self._skels:
            return
        for skel in self._skels:
            skel["enabled"] = bool(self._skel_on)
        if not self._skel_on:
            return
        motors = {"left": {}, "right": {}}
        node = self._ros_node
        if node is not None and hasattr(node, "get_robot_state"):
            try:
                state = node.get_robot_state()
                motors["left"] = {name: value for name, value in state.left_hand}
                motors["right"] = {name: value for name, value in state.right_hand}
            except Exception:
                pass
        try:
            from robot_hand_skeleton import skeleton_in_link
        except Exception:
            return
        for skel in self._skels:
            try:
                packed = _skeleton_lines(skel["side"], motors.get(skel["side"]) or {})
            except Exception:
                continue
            with skel["lock"]:
                skel["pending"] = packed

    def _subscribe_depth(self) -> None:
        node = self._ros_node
        if node is None or not self._cloud_by_topic:
            return
        enqueue = getattr(node, "enqueue_ros_op", None)
        if callable(enqueue):
            enqueue(self._subscribe_depth_on_ros)
            return
        self._subscribe_depth_on_ros()

    def _subscribe_wrench(self) -> None:
        node = self._ros_node
        if node is None or not self._wrenches:
            return
        enqueue = getattr(node, "enqueue_ros_op", None)
        if callable(enqueue):
            enqueue(self._subscribe_wrench_on_ros)
            return
        self._subscribe_wrench_on_ros()

    def _subscribe_depth_on_ros(self) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = self._ros_node
        self._depth_subs = []
        for topic, cloud in self._cloud_by_topic.items():
            topic_qos = qos
            qos_for = getattr(node, "_qos_for_topic", None)
            if callable(qos_for):
                try:
                    topic_qos = qos_for(topic, 0)
                except Exception:
                    topic_qos = qos
            sub = node.create_subscription(
                Image,
                topic,
                lambda msg, bound=cloud: self._on_depth(bound, msg),
                topic_qos,
            )
            self._depth_subs.append(sub)

    def _subscribe_wrench_on_ros(self) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        try:
            ArmState = import_arm_state()
        except Exception as exc:
            for item in self._wrenches:
                item["error"] = f"缺少 genie_msgs：{exc}"
            return
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = self._ros_node
        self._wrench_subs = []
        for item in self._wrenches:
            topic = str(item.get("topic") or "")
            topic_qos = qos
            qos_for = getattr(node, "_qos_for_topic", None)
            if callable(qos_for):
                try:
                    topic_qos = qos_for(topic, 0)
                except Exception:
                    topic_qos = qos
            # ArmState 发布端是 RELIABLE，再挂一个可靠订阅兜底。
            try:
                from rclpy.qos import ReliabilityPolicy as _Rel

                reliable = QoSProfile(
                    reliability=_Rel.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    durability=DurabilityPolicy.VOLATILE,
                )
            except Exception:
                reliable = topic_qos
            try:
                sub = node.create_subscription(
                    ArmState,
                    topic,
                    lambda msg, bound=item: self._on_wrench(bound, msg),
                    reliable,
                )
            except Exception as exc:
                item["error"] = f"订阅失败：{exc}"
                continue
            self._wrench_subs.append(sub)

    def _on_depth(self, cloud: dict, msg) -> None:
        try:
            from depth_cloud_node import _depth_meters, _intrinsics

            depth = _depth_meters(msg)
            if self._ros_node is not None and hasattr(self._ros_node, "get_intrinsics"):
                fx, fy, cx, cy = self._ros_node.get_intrinsics(cloud["topic"], int(msg.width), int(msg.height))
            else:
                fx, fy, cx, cy = _intrinsics(cloud["cam"], int(msg.width), int(msg.height))
            with cloud["lock"]:
                color_bgr = cloud.get("color_bgr")
                if color_bgr is not None:
                    color_bgr = np.array(color_bgr, copy=True)
            packed = _depth_points(
                depth,
                fx,
                fy,
                cx,
                cy,
                stride=4,
                color_bgr=color_bgr,
                fallback=tuple(cloud.get("color") or (0.7, 0.7, 0.7)),
            )
        except Exception as exc:
            cloud["error"] = str(exc)
            return
        with cloud["lock"]:
            cloud["pending"] = packed
            cloud["error"] = ""

    def _on_wrench(self, item: dict, msg) -> None:
        try:
            data = [float(v) for v in (msg.force_data or [])]
            if len(data) < 6 or not all(np.isfinite(data[:6])):
                raise ValueError(f"force_data 长度 {len(data)}")
            force = (data[0], data[1], data[2])
            torque = (data[3], data[4], data[5])
            packed = _wrench_lines(force, torque)
            state = int(getattr(msg, "force_state", 0) or 0)
        except Exception as exc:
            with item["lock"]:
                item["error"] = str(exc)
            return
        with item["lock"]:
            item["pending"] = packed
            item["force"] = force
            item["torque"] = torque
            item["force_state"] = state
            item["stamp"] = time.time()
            item["error"] = "" if state == 0 else f"force_state={state}"
            item["enabled"] = bool(self._wrench_on)

    def _ingest_cached_depth(self) -> None:
        """用界面已经收到的深度帧。机器人页自己的订阅挂错过节点时，这里仍能画出来。"""
        frames = getattr(self._ros_node, "click_frames", None)
        if frames is None or not self._clouds:
            return
        grabbed = []
        with frames.lock:
            for cloud in self._clouds:
                queue = frames.frames.get(cloud.get("topic"))
                if not queue:
                    continue
                frame = queue[-1]
                color_frame = None
                color_queue = frames.frames.get(cloud.get("color_topic"))
                if color_queue:
                    color_frame = color_queue[-1]
                token = (id(frame), id(color_frame) if color_frame is not None else 0)
                if cloud.get("_token") == token:
                    continue
                color_img = None if color_frame is None else np.array(color_frame.image, copy=True)
                grabbed.append(
                    (cloud, np.array(frame.image, copy=True), str(frame.encoding), color_img, token)
                )
        for cloud, image, encoding, color_img, token in grabbed:
            intr = None
            if self._ros_node is not None and hasattr(self._ros_node, "get_intrinsics"):
                h, w = np.asarray(image).shape[:2]
                intr = self._ros_node.get_intrinsics(cloud["topic"], w, h)
            packed = _pack_depth_image(
                cloud.get("cam") or {},
                image,
                encoding,
                color_bgr=color_img,
                intrinsics=intr,
                fallback=tuple(cloud.get("color") or (0.7, 0.7, 0.7)),
            )
            if packed is None:
                cloud["error"] = f"无法解析 {encoding}"
                continue
            with cloud["lock"]:
                if color_img is not None:
                    cloud["color_bgr"] = color_img
                cloud["pending"] = packed
                cloud["_token"] = token
                cloud["error"] = ""

    def _depth_status(self) -> str:
        parts = []
        poses = self._gl._poses if self._gl is not None else {}
        for cloud in self._clouds:
            if not cloud.get("enabled"):
                continue
            label = cloud.get("label") or cloud.get("key")
            err = cloud.get("error") or ""
            if err:
                parts.append(f"{label} {err}")
                continue
            if cloud.get("parent") not in poses:
                parts.append(f"{label} 缺少 {cloud.get('parent')}")
                continue
            count = int(cloud.get("count") or 0)
            parts.append(f"{label} {count} 点" if count else f"{label} 未收到")
        if not parts:
            return "深度未勾选。"
        return "深度 " + "，".join(parts) + "。"

    def _wrench_status(self) -> str:
        if not self._wrench_on:
            return ""
        parts = []
        now = time.time()
        for item in self._wrenches:
            label = item.get("label") or item.get("side")
            err = item.get("error") or ""
            force = item.get("force")
            torque = item.get("torque")
            stamp = float(item.get("stamp") or 0.0)
            if force is None or torque is None:
                parts.append(f"{label}{' ' + err if err else ' 未收到'}")
                continue
            age = now - stamp if stamp > 0 else 999.0
            f_norm = float(np.linalg.norm(force))
            m_norm = float(np.linalg.norm(torque))
            note = f"{label} F={f_norm:.1f}N M={m_norm:.2f}Nm"
            if age > 2.0:
                note += f" 停{age:.0f}s"
            if err:
                note += f" {err}"
            parts.append(note)
        if not parts:
            return ""
        return "六维力 " + "，".join(parts) + "。"

    def _tick(self) -> None:
        if self._gl is None or self._kin is None:
            return
        q16, age = read_q16()
        self._ingest_cached_depth()
        self._update_skeleton()
        for item in self._wrenches:
            item["enabled"] = bool(self._wrench_on) and item.get("force") is not None
        self._refresh_wrench_readout()
        if q16 is None:
            q16 = [0.0] * 16
            prefix = "没有关节数据。请先启动 host_fk。"
        elif age > 2.0:
            prefix = f"关节数据已停 {age:.0f} 秒。"
        else:
            prefix = ""
        try:
            self._gl.set_poses(self._kin.link_poses(q16))
        except Exception as exc:
            self._status.setText(f"正运动学失败：{exc}")
            return
        pose_note = ""
        shown = [
            self._pose_note[key]
            for key, _label, _rgb in _POSE_METHODS
            if self._pose_on.get(key, True) and self._pose_note.get(key)
        ]
        if shown:
            pose_note = " ".join(shown) + "。"
        sam3_note = ""
        if self._sam3_on and self._sam3_note:
            sam3_note = self._sam3_note + "。"
        lingbot_note = ""
        if self._lingbot_on and self._lingbot_note:
            lingbot_note = self._lingbot_note + "。"
        elif self._lingbot_on and self._lingbot is not None and self._lingbot_hold is None:
            lingbot_note = "LingBot 未收到。"
        self._status.setText(
            prefix
            + self._depth_status()
            + pose_note
            + sam3_note
            + lingbot_note
            + self._wrench_status()
            + " 拖动旋转，滚轮缩放。"
        )
