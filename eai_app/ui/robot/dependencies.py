"""Robot view dependencies, forwarded to their owning implementation modules."""
import importlib
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

_owners={'QOpenGLShader': 'eai_app.ui.robot.assets', 'QOpenGLShaderProgram': 'eai_app.ui.robot.assets', 'QOpenGLBuffer': 'eai_app.ui.robot.assets', 'JOINT_NAMES': 'eai_app.domain.robot.kinematics', '_MAX_FACES': 'eai_app.ui.robot.runtime', '_GL_DEPTH_TEST': 'eai_app.ui.robot.runtime', '_GL_COLOR_BUFFER_BIT': 'eai_app.ui.robot.runtime', '_GL_DEPTH_BUFFER_BIT': 'eai_app.ui.robot.runtime', '_GL_TRIANGLES': 'eai_app.ui.robot.runtime', '_GL_LINES': 'eai_app.ui.robot.runtime', '_GL_POINTS': 'eai_app.ui.robot.runtime', '_GL_FLOAT': 'eai_app.ui.robot.runtime', '_libgl': 'eai_app.ui.robot.assets', '_VERT': 'eai_app.ui.robot.runtime', '_FRAG': 'eai_app.ui.robot.runtime', '_resolve_urdf': 'eai_app.domain.robot.kinematics', '_mesh_path': 'eai_app.ui.robot.assets', '_rpy_matrix': 'eai_app.ui.robot.assets', '_xyz_rpy': 'eai_app.ui.robot.assets', '_parse_scale': 'eai_app.ui.robot.assets', '_load_obj': 'eai_app.ui.robot.assets', '_bake_mesh': 'eai_app.ui.robot.assets', '_link_color': 'eai_app.ui.robot.assets', 'load_visuals': 'eai_app.ui.robot.assets', 'RuiyanHandModel': 'eai_app.ui.robot.assets', '_Kinematics': 'eai_app.domain.robot.kinematics', 'read_q16': 'eai_app.adapters.robot.state_reader', 'sample_reachable_positions': 'eai_app.planning.reachability', '_grid_lines': 'eai_app.ui.robot.assets', '_RobotGL': 'eai_app.ui.robot.renderer', '_qt_matrix': 'eai_app.ui.robot.overlays', '_DEPTH_KEYS': 'eai_app.ui.robot.overlays', 'load_ply_xyz_rgb': 'eai_app.ui.robot.overlays', '_placement_for_color': 'eai_app.ui.robot.overlays', '_T_from_xyz_quat': 'eai_app.ui.robot.overlays', '_rgb_at_depth': 'eai_app.ui.robot.overlays', '_skeleton_lines': 'eai_app.ui.robot.overlays', '_GENIE_PREFIX_CANDIDATES': 'eai_app.adapters.ros.message_types', '_genie_prefixes': 'eai_app.adapters.ros.message_types', '_drop_genie_modules': 'eai_app.adapters.ros.message_types', '_preload_genie_fastrtps': 'eai_app.adapters.ros.message_types', 'import_arm_state': 'eai_app.adapters.ros.message_types', 'format_wrench_html': 'eai_app.ui.robot.overlays', '_WRENCH_FORCE_M_PER_N': 'eai_app.ui.robot.overlays', '_WRENCH_TORQUE_M_PER_NM': 'eai_app.ui.robot.overlays', '_WRENCH_MAX_ARROW_M': 'eai_app.ui.robot.overlays', '_WRENCH_TOPICS': 'eai_app.ui.robot.overlays', '_scaled_arrow': 'eai_app.ui.robot.overlays', '_append_arrow': 'eai_app.ui.robot.overlays', '_wrench_lines': 'eai_app.ui.robot.overlays', '_OBB_EDGES': 'eai_app.ui.robot.overlays', '_OBB_SIGNS': 'eai_app.ui.robot.overlays', '_POSE_METHODS': 'eai_app.ui.robot.overlays', '_pose6d_lines': 'eai_app.ui.robot.overlays', '_depth_points': 'eai_app.ui.robot.overlays', '_pack_depth_image': 'eai_app.ui.robot.overlays'}

def __getattr__(name):
    if name in _owners:return getattr(importlib.import_module(_owners[name]),name)
    raise AttributeError(name)

from eai_app.infrastructure.compat import forward_exports as _forward_exports
_forward_exports(__name__, _owners)
