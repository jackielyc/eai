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

_MAX_FACES = 40000

_GL_DEPTH_TEST = 0x0B71

_GL_COLOR_BUFFER_BIT = 0x00004000

_GL_DEPTH_BUFFER_BIT = 0x00000100

_GL_TRIANGLES = 0x0004

_GL_LINES = 0x0001

_GL_POINTS = 0x0000

_GL_FLOAT = 0x1406

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

