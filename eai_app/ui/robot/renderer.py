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
_eai_ui_robot_assets=module_ref('eai_app.ui.robot.assets')
_eai_ui_robot_overlays=module_ref('eai_app.ui.robot.overlays')
_eai_ui_robot_runtime=module_ref('eai_app.ui.robot.runtime')

class _RobotGL(QOpenGLWidget):
    def __init__(self, visuals: List[dict], parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        super().__init__(parent)
        self.setFormat(fmt)
        self._visuals = visuals
        self.show_wrist_accessories = True
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
        self._libgl = _eai_ui_robot_assets._libgl()
        self._program = _eai_ui_robot_assets.QOpenGLShaderProgram(self)
        if not self._program.addShaderFromSourceCode(_eai_ui_robot_assets.QOpenGLShader.Vertex, _eai_ui_robot_runtime._VERT):
            raise RuntimeError(self._program.log())
        if not self._program.addShaderFromSourceCode(_eai_ui_robot_assets.QOpenGLShader.Fragment, _eai_ui_robot_runtime._FRAG):
            raise RuntimeError(self._program.log())
        self._program.bindAttributeLocation("a_pos", 0)
        self._program.bindAttributeLocation("a_nrm", 1)
        if not self._program.link():
            raise RuntimeError(self._program.log())
        self._items = [self._upload(item["packed"], item["link"], item["color"]) for item in self._visuals]
        self._grid = self._upload(_eai_ui_robot_assets._grid_lines(), "", (0.28, 0.30, 0.32), lines=True)
        self._ready = True

    def _upload(self, packed: np.ndarray, link: str, color, lines: bool = False) -> dict:
        buf = _eai_ui_robot_assets.QOpenGLBuffer(_eai_ui_robot_assets.QOpenGLBuffer.VertexBuffer)
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
        gl.glEnable(_eai_ui_robot_runtime._GL_DEPTH_TEST)
        gl.glClearColor(0.12, 0.13, 0.15, 1.0)
        gl.glClear(_eai_ui_robot_runtime._GL_COLOR_BUFFER_BIT | _eai_ui_robot_runtime._GL_DEPTH_BUFFER_BIT)
        if not self._ready or self._program is None:
            return
        mvp = self._mvp()
        prog = self._program
        prog.bind()
        prog.setUniformValue("u_mvp", mvp)
        ident = QMatrix4x4()
        ident.setToIdentity()
        self._draw_item(self._grid, ident, lit=0.0, mode=_eai_ui_robot_runtime._GL_LINES)
        for item in self._items:
            if not self.show_wrist_accessories and item['link'].startswith(('wrist_connector_', 'wrist_camera_')):
                continue
            pose = self._poses.get(item["link"])
            if pose is None:
                continue
            self._draw_item(item, _eai_ui_robot_overlays._qt_matrix(pose), lit=1.0, mode=_eai_ui_robot_runtime._GL_TRIANGLES)
        for cloud in self._clouds:
            self._flush_cloud(cloud)
            if not cloud.get("enabled") or cloud.get("count", 0) <= 0:
                continue
            pose = self._poses.get(cloud.get("parent") or "")
            if pose is None:
                continue
            self._draw_item(
                cloud,
                _eai_ui_robot_overlays._qt_matrix(pose @ cloud["T"]),
                lit=-1.0,
                mode=_eai_ui_robot_runtime._GL_POINTS,
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
                _eai_ui_robot_overlays._qt_matrix(pose @ skel["T"]),
                lit=-1.0,
                mode=_eai_ui_robot_runtime._GL_LINES,
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
                _eai_ui_robot_overlays._qt_matrix(pose @ wrench["T"]),
                lit=-1.0,
                mode=_eai_ui_robot_runtime._GL_LINES,
                line_width=3.0,
            )
        ident = QMatrix4x4()
        ident.setToIdentity()
        gl.glDisable(_eai_ui_robot_runtime._GL_DEPTH_TEST)
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
                model = _eai_ui_robot_overlays._qt_matrix(pose @ pose_item["T"])
            else:
                model = ident
            draw_mode = int(pose_item.get("draw_mode", _eai_ui_robot_runtime._GL_LINES))
            if draw_mode == _eai_ui_robot_runtime._GL_POINTS:
                gl.glEnable(_eai_ui_robot_runtime._GL_DEPTH_TEST)
            else:
                gl.glDisable(_eai_ui_robot_runtime._GL_DEPTH_TEST)
            self._draw_item(
                pose_item,
                model,
                lit=-1.0,
                mode=draw_mode,
                line_width=2.5,
            )
        gl.glEnable(_eai_ui_robot_runtime._GL_DEPTH_TEST)
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
        mode: int = _eai_ui_robot_runtime._GL_TRIANGLES,
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
        prog.setAttributeBuffer(0, _eai_ui_robot_runtime._GL_FLOAT, 0, 3, stride)
        prog.setAttributeBuffer(1, _eai_ui_robot_runtime._GL_FLOAT, 3 * 4, 3, stride)
        if mode == _eai_ui_robot_runtime._GL_POINTS:
            self._libgl.glEnable(0x8642)
            self._libgl.glPointSize(float(item.get("point_size") or 3.0))
        elif mode == _eai_ui_robot_runtime._GL_LINES:
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

