"""platform.bootstrap — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT

import argparse


import copy


import eai_app.perception.click_pose as click_pose


import base64


import json


import math


import signal


import sys


import urllib.error


import urllib.parse


import urllib.request


from concurrent.futures import ThreadPoolExecutor, as_completed


if sys.version_info[:2] != (3, 10):
    try:
        import rclpy  # noqa: F401
    except Exception as exc:
        print(
            f"错误: 当前 Python {sys.version_info.major}.{sys.version_info.minor}，"
            f"无法加载 rclpy（{exc}）。\n\n"
            "系统 apt Humble 请用 Python 3.10:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  python3.10 compat/show_camera_topics.py\n"
            "  或: ./run.sh\n\n"
            "RoboStack Humble 请用该 conda 环境的 python，例如:\n"
            "  bash scripts/launch/run_local.sh",
            file=sys.stderr,
        )
        sys.exit(1)


import os


import re


os.environ.pop("QT_PLUGIN_PATH", None)


import shlex


import socket


import subprocess


import shutil


import tempfile


import queue


import threading


import time


import uuid


from collections import deque


from dataclasses import dataclass


from datetime import datetime, timezone


from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple


from eai_app.domain.robot.commands import (
    ARM_TEXT_CMD_HELP,
    ARM_TEXT_CMD_PLACEHOLDER,
    HAND_JOINT_ORDER,
    ArmHandTextAction,
    canonical_hand_joint,
    format_arm_abs_pose_text,
    parse_arm_hand_text,
)


cv2: Any = None


np: Any = None


def _clear_opencv_qt_plugin_hijack() -> None:
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        val = os.environ.get(key, "")
        if "/cv2/" in val.replace("\\", "/"):
            os.environ.pop(key, None)


def _ensure_cv2_numpy() -> None:
    """Lazy-import cv2 + numpy (and clear OpenCV Qt plugin hijack)."""
    global cv2, np
    if cv2 is not None and np is not None:
        return
    import numpy as _np
    import cv2 as _cv2

    np = _np
    cv2 = _cv2
    _clear_opencv_qt_plugin_hijack()


pg: Any = None


gl: Any = None


_boot_progress: Optional[Callable[[str], None]] = None


def _boot_tick(msg: str) -> None:
    cb = _boot_progress
    if cb is None:
        return
    try:
        cb(msg)
    except Exception:
        pass


def _ensure_pyqtgraph_opengl() -> Tuple[Any, Any]:
    """Lazy-import pyqtgraph + OpenGL; return (pg, gl)."""
    global pg, gl
    if pg is not None and gl is not None:
        return pg, gl
    import pyqtgraph as _pg
    import pyqtgraph.opengl as _gl

    pg = _pg
    gl = _gl
    return pg, gl


import rclpy


from eai_app.perception.geometry import resolve_intrinsics, sanitize_depth, valid_depth_mask, DEPTH_MIN_M, DEPTH_MAX_M, validated_camera_transform


from PyQt5.QtCore import (
    Qt,
    QProcess,
    QProcessEnvironment,
    QTimer,
    pyqtSignal,
    QObject,
    QPoint,
    QRect,
    QSize,
    QEvent,
    QUrl,
    QByteArray,
    QBuffer,
    QIODevice,
    QMimeData,
)


from PyQt5.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QDrag,
    QFont,
    QIcon,
    QImage,
    QMouseEvent,
    QPixmap,
    QPalette,
    QColor,
)


from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QSizePolicy,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QPlainTextEdit,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)


try:
    # 须在创建 QApplication 前 import，否则部分环境 WebEngine 初始化失败
    from PyQt5.QtWebEngineWidgets import (  # type: ignore
        QWebEngineView,
        QWebEnginePage,
        QWebEngineSettings,
    )

    _HAS_QT_WEBENGINE = True
except Exception:
    QWebEngineView = None  # type: ignore
    QWebEnginePage = None  # type: ignore
    QWebEngineSettings = None  # type: ignore
    _HAS_QT_WEBENGINE = False


from rclpy.node import Node


from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


from rclpy.duration import Duration


from rclpy.time import Time


from std_msgs.msg import UInt32, UInt8


from std_srvs.srv import SetBool, Trigger


from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState


from geometry_msgs.msg import PoseStamped


if TYPE_CHECKING:
    from tf2_ros import Buffer as Buffer
else:
    # 运行时类型占位；真正的 Buffer 在 CameraTopicNode 里 lazy import
    Buffer = Any  # type: ignore[misc, assignment]


_IME_LAST_TEXT_WIDGET = None  # type: ignore


_IME_OPEN_POPUPS = []  # type: ignore

