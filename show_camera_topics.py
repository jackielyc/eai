#!/usr/bin/python3.10
"""
PyQt5 图形界面：显示 ROS2 中以 /camera 开头的 topic 及图像内容。
深度 topic（名称含 depth）以 3D 点云方式显示，支持鼠标旋转/缩放。
点击图像仅选择提示点 (u,v)；需再点「调用分割 / SAM3 / FP」确认执行。
深度 3D 面板叠加显示左右手臂 TCP 与关节状态（/hal/arm_joint_state、/ry_hand/*/joint_states、/mink_fk/*_tcp_pose）。

用法:
  bash run_in_docker.sh          # ROS 在 Docker 内运行时用此方式（推荐）
  bash run.sh                    # ROS 在宿主机直接运行时用此方式
  python3.10 show_camera_topics.py --prefix /camera

顶部控制区按功能分为标签页：大脑 / 回放 / 分割 / CAD / 训练 / 手臂·手 / 手骨架遥控 / 测试。
前置条件：robot-service + 手/臂服务栈已运行，control_mode=0，手臂/手部已使能。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import signal
import sys
import urllib.error
import urllib.parse
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

import shlex
import subprocess
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSignal, QObject, QPoint, QEvent
from PyQt5.QtGui import QCloseEvent, QFont, QImage, QMouseEvent, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
    QToolButton,
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


# 中文输入依赖宿主机 fcitx + run_in_docker.sh 的 dbus abstract 代理。
# 禁止: 全局 im.reset()；获焦时反复开关 WA_InputMethodEnabled。
# 下拉: 用「主窗口内浮层」而非 Qt.Tool 顶层窗（独立 X 窗口会弄死 fcitx IC）。

_IME_LAST_TEXT_WIDGET = None  # type: ignore
_IME_OPEN_POPUPS = []  # type: ignore


def _is_text_ime_widget(widget) -> bool:
    if widget is None:
        return False
    if not isinstance(widget, (QLineEdit, QTextEdit)):
        return False
    try:
        if hasattr(widget, "isReadOnly") and widget.isReadOnly():
            return False
    except Exception:
        pass
    return True


def _remember_text_focus() -> None:
    global _IME_LAST_TEXT_WIDGET
    try:
        fw = QApplication.focusWidget()
        if _is_text_ime_widget(fw):
            _IME_LAST_TEXT_WIDGET = fw
    except Exception:
        pass


def _keep_text_focus() -> None:
    w = _IME_LAST_TEXT_WIDGET
    if not _is_text_ime_widget(w):
        return
    try:
        w.setFocus(Qt.OtherFocusReason)
    except Exception:
        pass


def _revive_ime_after_combo() -> None:
    """下拉关闭后只交还焦点，不重建控件（重建反而常弄死 fcitx）。"""
    _keep_text_focus()


def restore_fcitx_input_method(widget=None) -> None:
    _revive_ime_after_combo()


def _schedule_fcitx_restore(widget=None) -> None:
    try:
        QTimer.singleShot(0, _revive_ime_after_combo)
        QTimer.singleShot(80, _revive_ime_after_combo)
    except Exception:
        pass


class ImeSafeComboBox(QComboBox):
    """鼠标可选下拉：列表做主窗口子控件浮层，不创建新的 X11 窗口。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.setFocusPolicy(Qt.NoFocus)
        self._ime_list: Optional[QListWidget] = None

    def _ensure_list(self) -> QListWidget:
        win = self.window()
        if self._ime_list is not None:
            # 父窗口变了则重建
            if self._ime_list.parent() is win:
                return self._ime_list
            try:
                self._ime_list.hide()
                self._ime_list.setParent(None)
                self._ime_list.deleteLater()
            except Exception:
                pass
            self._ime_list = None
        lw = QListWidget(win)
        # 普通子控件，禁止变顶层 Tool/Popup
        lw.setWindowFlags(Qt.Widget)
        lw.setAttribute(Qt.WA_InputMethodEnabled, False)
        lw.setFocusPolicy(Qt.NoFocus)
        lw.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lw.setStyleSheet(
            "QListWidget { background: #2d2d2d; color: #eee; border: 1px solid #666; }"
            "QListWidget::item:selected { background: #3d6ea8; }"
            "QListWidget::item:hover { background: #454545; }"
        )
        lw.itemClicked.connect(self._on_ime_item_clicked)
        self._ime_list = lw
        return lw

    def showPopup(self) -> None:  # type: ignore[override]
        _remember_text_focus()
        for c in list(_IME_OPEN_POPUPS):
            try:
                if c is not self:
                    c.hidePopup()
            except Exception:
                pass
        lw = self._ensure_list()
        lw.clear()
        cur = self.currentIndex()
        for i in range(self.count()):
            item = QListWidgetItem(self.itemText(i))
            tip = self.itemData(i, Qt.ToolTipRole)
            if tip:
                item.setToolTip(str(tip))
            try:
                enabled = bool(
                    self.model().flags(self.model().index(i, self.modelColumn()))
                    & Qt.ItemIsEnabled
                )
            except Exception:
                enabled = True
            if not enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
            lw.addItem(item)
        if 0 <= cur < lw.count():
            lw.setCurrentRow(cur)
        rows = max(lw.count(), 1)
        row_h = lw.sizeHintForRow(0) if lw.count() else 24
        if row_h <= 0:
            row_h = 24
        visible = min(rows, max(self.maxVisibleItems(), 8))
        lw.setFixedWidth(max(self.width(), 120))
        lw.setFixedHeight(min(280, row_h * visible + 4))
        # 相对主窗口定位（同一 X 窗口内）
        parent = lw.parentWidget()
        gp = self.mapToGlobal(QPoint(0, self.height()))
        if parent is not None:
            lp = parent.mapFromGlobal(gp)
            # 若底部不够，向上展开
            if lp.y() + lw.height() > parent.height():
                gp2 = self.mapToGlobal(QPoint(0, 0))
                lp2 = parent.mapFromGlobal(gp2)
                lp = QPoint(lp2.x(), max(0, lp2.y() - lw.height()))
            lw.move(lp)
        lw.show()
        lw.raise_()
        if self not in _IME_OPEN_POPUPS:
            _IME_OPEN_POPUPS.append(self)
        _keep_text_focus()
        QTimer.singleShot(0, _keep_text_focus)

    def hidePopup(self) -> None:  # type: ignore[override]
        lw = self._ime_list
        if lw is not None:
            lw.hide()
        try:
            _IME_OPEN_POPUPS.remove(self)
        except ValueError:
            pass

    def _on_ime_item_clicked(self, item: QListWidgetItem) -> None:
        if item is None or not (item.flags() & Qt.ItemIsEnabled):
            return
        lw = self._ime_list
        row = lw.row(item) if lw is not None else -1
        if row < 0:
            return
        self.hidePopup()
        if row != self.currentIndex():
            self.setCurrentIndex(row)
        self.activated.emit(row)
        # 选完后再恢复 IC（此时回调已跑完）
        QTimer.singleShot(0, _revive_ime_after_combo)
        QTimer.singleShot(100, _revive_ime_after_combo)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if self._ime_list is not None and self._ime_list.isVisible():
                self.hidePopup()
                QTimer.singleShot(0, _revive_ime_after_combo)
            else:
                self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)


class ImePopupDismissFilter(QObject):
    """点击浮层外关闭；关闭后恢复中文 IC。"""

    def eventFilter(self, obj, event):  # type: ignore[override]
        try:
            et = event.type()
            if et == QEvent.FocusIn and _is_text_ime_widget(obj):
                global _IME_LAST_TEXT_WIDGET
                _IME_LAST_TEXT_WIDGET = obj
            if not _IME_OPEN_POPUPS:
                return False
            if et == QEvent.MouseButtonPress:
                gp = event.globalPos() if hasattr(event, "globalPos") else None
                if gp is None:
                    return False
                closed = False
                for combo in list(_IME_OPEN_POPUPS):
                    lw = getattr(combo, "_ime_list", None)
                    if lw is None or not lw.isVisible():
                        continue
                    if combo.rect().contains(combo.mapFromGlobal(gp)):
                        continue
                    if lw.rect().contains(lw.mapFromGlobal(gp)):
                        continue
                    combo.hidePopup()
                    closed = True
                if closed:
                    QTimer.singleShot(0, _revive_ime_after_combo)
            elif et == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                for combo in list(_IME_OPEN_POPUPS):
                    combo.hidePopup()
                QTimer.singleShot(0, _revive_ime_after_combo)
        except Exception:
            pass
        return False


def install_chinese_ime_guards(app: QApplication) -> None:
    filt = ImePopupDismissFilter(app)
    app.installEventFilter(filt)
    setattr(app, "_ime_popup_dismiss_filter", filt)
    for w in app.allWidgets():
        try:
            if isinstance(w, QComboBox):
                w.setAttribute(Qt.WA_InputMethodEnabled, False)
                w.setFocusPolicy(Qt.NoFocus)
        except Exception:
            pass


try:
    from a2d_head_camera_tf import attach_to_node as attach_head_camera_tf
except Exception:  # pragma: no cover - optional helper
    attach_head_camera_tf = None  # type: ignore

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


def resolve_sam3_model_path() -> str:
    env = os.environ.get("SAM3_MODEL", "").strip()
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "sam3.pt"),
        os.path.expanduser(
            "~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(os.path.expanduser(env or os.path.join(here, "sam3.pt")))


SAM3_MODEL_DEFAULT = resolve_sam3_model_path()
SAM3_SERVER_URL_DEFAULT = os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:8765")
SAM3_USE_HTTP_DEFAULT = os.environ.get("SAM3_USE_HTTP", "0").strip() in ("1", "true", "yes")
SAM3_TIMEOUT_S = float(os.environ.get("SAM3_TIMEOUT_S", "120"))
SAM3_RUN_SCRIPT_NAME = "run_sam3.sh"
SAM3_DEFAULT_PORT = int(os.environ.get("SAM3_PORT", "8765"))
OLLAMA_API_BASE_DEFAULT = "http://127.0.0.1:11434/v1"

POSE_BACKEND_PCA = "pca"
POSE_BACKEND_FOUNDATIONPOSE = "foundationpose"
POSE_BACKENDS = (POSE_BACKEND_PCA, POSE_BACKEND_FOUNDATIONPOSE)
FP_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "foundationpose_worker.py"
)
FP_RUN_SCRIPT_NAME = "run_foundationpose.sh"
FP_PYTHON_DEFAULT = os.environ.get("FP_PYTHON", "python3")
FP_SERVER_URL_DEFAULT = os.environ.get("FP_SERVER_URL", "http://127.0.0.1:8766")
FP_USE_HTTP_DEFAULT = os.environ.get("FP_USE_HTTP", "0").strip() in ("1", "true", "yes")
FP_TIMEOUT_S = float(os.environ.get("FP_TIMEOUT_S", "180"))
FP_DEFAULT_PORT = int(os.environ.get("FP_PORT", "8766"))


def fp_mesh_path_exists(mesh_path: str) -> bool:
    return resolve_fp_mesh_absolute(mesh_path) is not None


def resolve_fp_mesh_absolute(mesh_path: str) -> Optional[str]:
    path = (mesh_path or "").strip()
    if not path:
        path = resolve_fp_mesh_path()
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.expanduser(path)]
    if not os.path.isabs(path):
        candidates.append(os.path.join(here, path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_fp_mesh_for_call(
    mesh_path: str,
    use_http: bool,
    server_url: str,
) -> Tuple[str, str]:
    """解析调用时传给 worker 的 mesh 路径；Docker viewer 可用 worker 侧默认 mesh。"""
    local = resolve_fp_mesh_absolute(mesh_path)
    if local:
        return local, "local"
    if use_http:
        health = fetch_foundationpose_health(server_url, timeout_s=0.8)
        worker_mesh = str(health.get("mesh_resolved") or "").strip()
        if worker_mesh:
            return worker_mesh, "worker"
        worker_default = str(health.get("mesh") or "").strip()
        if worker_default and worker_default != "not_set":
            return worker_default, "worker_default"
    stripped = (mesh_path or "").strip()
    if stripped:
        return stripped, "path"
    return resolve_fp_mesh_path(), "default"


def fp_mesh_available(mesh_path: str, use_http: bool, server_url: str) -> Tuple[bool, bool]:
    """返回 (本地 mesh 可用, 服务/worker mesh 可用)。"""
    local_ok = resolve_fp_mesh_absolute(mesh_path) is not None
    worker_ok = False
    if use_http:
        health = fetch_foundationpose_health(server_url, timeout_s=0.8)
        worker_ok = bool(health.get("mesh_ok")) or bool(
            str(health.get("mesh_resolved") or "").strip()
        )
    return local_ok, worker_ok


def resolve_fp_mesh_path() -> str:
    env = os.environ.get("FP_MESH", "").strip()
    here = os.path.dirname(os.path.abspath(__file__))
    if env:
        for candidate in (
            os.path.expanduser(env),
            os.path.join(here, env),
        ):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    # 方案 A：优先使用 meshes/<name>/reconstructed.obj（按修改时间最新）
    meshes_root = os.path.join(here, "meshes")
    if os.path.isdir(meshes_root):
        recon: list[tuple[float, str]] = []
        for name in os.listdir(meshes_root):
            path = os.path.join(meshes_root, name, "reconstructed.obj")
            if os.path.isfile(path):
                recon.append((os.path.getmtime(path), path))
        if recon:
            recon.sort(key=lambda x: x[0], reverse=True)
            return os.path.abspath(recon[0][1])
    candidates = [
        os.path.join(
            here,
            "FoundationPose",
            "demo_data",
            "mustard0",
            "mesh",
            "textured_simple.obj",
        ),
        os.path.join(here, "demo_data", "mustard0", "mesh", "textured_simple.obj"),
        os.path.expanduser("~/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"),
        os.path.expanduser(
            "~/workspace_liyichao/eai/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj"
        ),
    ]
    fp_root = os.environ.get("FOUNDATIONPOSE_ROOT", "").strip()
    if fp_root:
        candidates.insert(
            0,
            os.path.join(
                fp_root,
                "demo_data",
                "mustard0",
                "mesh",
                "textured_simple.obj",
            ),
        )
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(os.path.expanduser(env or candidates[-1]))


FP_MESH_DEFAULT = resolve_fp_mesh_path()
EAI_DIR = os.path.dirname(os.path.abspath(__file__))
CAD_MESHES_DIR = os.path.join(EAI_DIR, "meshes")
TEST_IMAGES_DIR = os.path.join(EAI_DIR, "images")
WORKSPACE_DIR = os.path.dirname(EAI_DIR)
LOCAL_QWEN_MODELS_ROOT = os.path.join(WORKSPACE_DIR, "models", "Qwen")
LAKE_QWEN35_OUTPUT_ROOT = (
    "/share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35/output"
)
REMOTE_LAKE_QWEN_PYTHON = (
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python"
)
# 可部署的 Qwen 模型：(key, 显示名, API model id, 权重目录名或绝对路径)
LOCAL_QWEN_DEPLOY_MODELS: Tuple[Tuple[str, str, str, str], ...] = (
    ("qwen3.5-4b", "Qwen3.5-4B", "qwen3.5-4b", "Qwen3.5-4B"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"),
    (
        "lake-qwen35-4b-lora",
        "Lake Qwen3.5-4B LoRA",
        "lake-qwen35-4b-lora",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake",
    ),
    (
        "lake-qwen35-4b-lora-smoke",
        "Lake Qwen3.5-4B LoRA (smoke)",
        "lake-qwen35-4b-lora-smoke",
        f"{LAKE_QWEN35_OUTPUT_ROOT}/qwen35-4b-lora-lake-smoke",
    ),
)
LOCAL_QWEN_MODEL_DIR_DEFAULT = os.path.join(
    LOCAL_QWEN_MODELS_ROOT, LOCAL_QWEN_DEPLOY_MODELS[0][3]
)
LOCAL_QWEN_OLLAMA_NAME = os.environ.get("LOCAL_QWEN_OLLAMA_NAME", "qwen3.5-4b-local")
LOCAL_QWEN_OLLAMA_FALLBACK = os.environ.get("LOCAL_QWEN_OLLAMA_FALLBACK", "qwen3.5:4b")
LOCAL_QWEN_RUN_SCRIPT_NAME = "run_local_qwen.sh"
LOCAL_QWEN_PORT_DEFAULT = int(os.environ.get("LOCAL_QWEN_PORT", "8100"))
LOCAL_QWEN_CTL_PORT_DEFAULT = int(os.environ.get("LOCAL_QWEN_CTL_PORT", "18101"))
LOCAL_QWEN_CTL_URL_DEFAULT = os.environ.get(
    "LOCAL_QWEN_CTL_URL", f"http://127.0.0.1:{LOCAL_QWEN_CTL_PORT_DEFAULT}"
)
LOCAL_QWEN_MODEL_ID = os.environ.get(
    "LOCAL_QWEN_MODEL_ID", LOCAL_QWEN_DEPLOY_MODELS[0][2]
)
LOCAL_QWEN_API_BASE_DEFAULT = os.environ.get(
    "LOCAL_QWEN_API_BASE", f"http://127.0.0.1:{LOCAL_QWEN_PORT_DEFAULT}/v1"
)
LOCAL_QWEN_PYTHON_DEFAULT = os.environ.get(
    "LOCAL_QWEN_PYTHON",
    "/home/psibot/miniconda3/envs/psi-policy/bin/python",
)
LOCAL_QWEN_CHAT_PRESET_NAME = "本地 Qwen 服务"
# 远程部署主机列表：(id, 显示名)。路径细节在 remote_qwen_ctl.HOST_PROFILES
REMOTE_QWEN_HOSTS: Tuple[Tuple[str, str], ...] = (
    ("psi_motus_2_for_liyichao", "远程 psi_motus（8×A800）"),
    ("tione-develop", "远程 tione-develop（7×A800）"),
)
REMOTE_QWEN_SSH_HOST = os.environ.get("REMOTE_QWEN_SSH_HOST", "psi_motus_2_for_liyichao")
REMOTE_QWEN_LOCAL_PORT = int(os.environ.get("REMOTE_QWEN_LOCAL_PORT", "18100"))
REMOTE_QWEN_API_BASE_DEFAULT = os.environ.get(
    "REMOTE_QWEN_API_BASE", f"http://127.0.0.1:{REMOTE_QWEN_LOCAL_PORT}/v1"
)
REMOTE_QWEN_CHAT_PRESET_NAME = "远程 Qwen 服务"
REMOTE_QWEN_CTL_SCRIPT = os.path.join(EAI_DIR, "remote_qwen_ctl.py")
REMOTE_QWEN_HOSTCTL_PORT = int(os.environ.get("REMOTE_QWEN_CTL_PORT", "18103"))
REMOTE_QWEN_HOSTCTL_URL = os.environ.get(
    "REMOTE_QWEN_CTL_URL", f"http://127.0.0.1:{REMOTE_QWEN_HOSTCTL_PORT}"
).rstrip("/")

CAD_PHOTOGRAMMETRY_PY = os.path.join(EAI_DIR, "photogrammetry_reconstruct.py")
CAD_TARGET_EXTENT_DEFAULT_M = 0.15
CAD_POISSON_DEPTH_DEFAULT = 9
CAD_MIN_IMAGES_DEFAULT = 8
CAD_CAPTURE_COUNT_DEFAULT = 12
CAD_CAPTURE_TOPIC_DEFAULT = "/camera/head_color"
CAD_CAPTURE_DEPTH_DEFAULT = "/camera/head_depth"
CAD_CAPTURE_TOPIC_CANDIDATES = (
    "/camera/head_color",
    "/camera/head_rgb",
    "/camera/head/color",
)
CAD_CAPTURE_MIN_DIFF_CORR = 0.97  # 几乎不动才拒绝；头部相机转物通常 corr≈0.87~0.95
CAD_CAPTURE_MIN_DIFF_MAE = 3.0
PSI_POLICY_DIR_ENV = "PSI_POLICY_DIR"


def resolve_cad_mesh_python() -> str:
    env = os.environ.get("MESH_PYTHON", "").strip()
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    candidates = [
        os.path.expanduser("~/miniconda3/envs/foundationpose/bin/python"),
        os.path.expanduser("~/anaconda3/envs/foundationpose/bin/python"),
        "/home/psibot/miniconda3/envs/foundationpose/bin/python",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("python3") or "python3"


def list_reconstructed_meshes(meshes_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """返回 [(显示名, reconstructed.obj 绝对路径), ...]，按修改时间新→旧。"""
    root = meshes_dir or CAD_MESHES_DIR
    if not os.path.isdir(root):
        return []
    items: List[Tuple[float, str, str]] = []
    for name in os.listdir(root):
        obj = os.path.join(root, name, "reconstructed.obj")
        if os.path.isfile(obj):
            items.append((os.path.getmtime(obj), name, os.path.abspath(obj)))
    items.sort(key=lambda x: x[0], reverse=True)
    return [(name, path) for _, name, path in items]


PSI_POLICY_CONFIG_DEFAULT = "example_workspace_imle_rgb"
PSI_POLICY_LOGGING_MODES = ("offline", "online", "disabled")
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
STACK_SERVICES_SCRIPT = "nodes/start_arm_hand_services.sh"
STACK_SERVICES_LAUNCH = "arm_hand_services.launch.py"
STACK_SERVICES_LOG = "/var/psi/log/arm_hand_services/latest.log"
# 兼容手动启动完整 base_services 时的状态检测
LEGACY_BASE_SERVICES_LAUNCH = "base_services.launch.py"
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
ROBOT_RIGHT_HAND_CMD_TOPIC = "/ry_hand/right/set_angles"
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

# 深色界面文字对比度（避免 #666/#888 在深色底上难以辨认）
UI_TEXT_PRIMARY = "#ececec"
UI_TEXT_SECONDARY = "#c8c8c8"
UI_TEXT_MUTED = "#b0b0b0"
UI_TEXT_PLACEHOLDER = "#9a9a9a"
UI_ACCENT_BLUE = "#7ec8ff"
UI_ACCENT_BLUE_BRIGHT = "#4da3ff"
UI_ACCENT_ORANGE = "#ffb86c"
UI_ACCENT_GREEN = "#50fa7b"
UI_ACCENT_RED = "#ff8888"
UI_MONO_FAMILY = "Monospace"
UI_MONO_SIZE_SMALL = 9
UI_MONO_SIZE_NORMAL = 10
UI_MONO_SIZE_TITLE = 11

# 测试 Tab：场景示意图像框（2×4）
TEST_SCENARIO_LABELS = (
    "手机装配",
    "商超零售拆盒子",
    "真人数采",
    "THT电容插件装配",
    "叠盒子",
    "麻将机器人",
    "微波炉加热食品",
    "飞机盒装物",
)
# 文件名与场景名不一致时的映射（默认找 images/{场景名}.png）
TEST_SCENARIO_IMAGE_FILES: Dict[str, str] = {}


def qwen_model_path_from_spec(dirname_or_path: str, *, root: str = "") -> str:
    """将部署 spec 解析为绝对路径（相对名则拼到 root / 本地 models）。"""
    if dirname_or_path.startswith("/"):
        return dirname_or_path
    base = root or LOCAL_QWEN_MODELS_ROOT
    return os.path.join(base, dirname_or_path)


def qwen_model_dir_valid(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
        os.path.join(path, "adapter_config.json")
    )


def qwen_model_is_lora(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def qwen_deploy_path_spec_for_key(model_key: str) -> str:
    for key, _label, _mid, spec in LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return spec
    return model_key


def remote_qwen_python_for_key(model_key: str, host_id: str = "") -> Optional[str]:
    if model_key.startswith("lake-"):
        return REMOTE_LAKE_QWEN_PYTHON
    return None


def local_qwen_model_dir_for_key(model_key: str) -> Optional[str]:
    """按部署 key 解析权重目录（全量需 config.json，LoRA 需 adapter_config.json）。"""
    spec = qwen_deploy_path_spec_for_key(model_key)
    if spec.startswith("/"):
        return spec if qwen_model_dir_valid(spec) else None
    for root in (
        LOCAL_QWEN_MODELS_ROOT,
        os.path.join(EAI_DIR, "models", "Qwen"),
    ):
        path = os.path.join(root, spec)
        if qwen_model_dir_valid(path):
            return path
    return None


def local_qwen_model_id_for_key(model_key: str) -> str:
    for key, _label, mid, _dirname in LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return mid
    return LOCAL_QWEN_MODEL_ID


def local_qwen_label_for_key(model_key: str) -> str:
    for key, label, _mid, _dirname in LOCAL_QWEN_DEPLOY_MODELS:
        if key == model_key:
            return label
    return model_key


def local_deploy_scan_roots() -> List[Tuple[str, str]]:
    roots: List[Tuple[str, str]] = [
        (LOCAL_QWEN_MODELS_ROOT, "本地 Qwen 基座"),
    ]
    if os.path.isdir(LAKE_QWEN35_OUTPUT_ROOT):
        roots.append((LAKE_QWEN35_OUTPUT_ROOT, "Lake 训练 output"))
    return roots


def remote_deploy_scan_roots(host_id: str) -> List[Tuple[str, str]]:
    try:
        ctl = load_remote_qwen_ctl()
        return [(r, lbl) for r, lbl in ctl.deploy_scan_roots(host_id)]
    except Exception:
        prof = remote_qwen_profile(host_id)
        return [
            (str(prof.get("model_root") or LOCAL_QWEN_MODELS_ROOT), "Qwen 基座"),
            (LAKE_QWEN35_OUTPUT_ROOT, "Lake 训练 output"),
        ]


def deploy_model_kind_label(kind: str) -> str:
    return "LoRA" if kind == "lora" else "全量"


def fetch_deploy_model_dirs(
    *,
    target: str,
    host_id: str = "",
    root: str = "",
) -> Tuple[bool, List[Dict[str, str]], str]:
    scan_roots = [root.strip()] if root.strip() else None
    try:
        ctl = load_remote_qwen_ctl()
        if target == "local":
            scan = scan_roots or [r for r, _ in local_deploy_scan_roots()]
            models = ctl.list_deploy_model_dirs_local(scan)
            return True, models, ""

        host = (host_id or REMOTE_QWEN_SSH_HOST).strip()

        def _via_ctl() -> Tuple[bool, List[Dict[str, str]], str]:
            if is_running_in_docker():
                return (
                    False,
                    [],
                    "Docker 内列举远程目录需宿主机 remote hostctl (:18103)。"
                    "请重新执行 bash eai/run_in_docker.sh 以重启 hostctl。",
                )
            result = ctl.list_deploy_model_dirs(roots=scan_roots, host_id=host)
            if result.get("ok"):
                raw = result.get("models") or []
                return True, [dict(x) for x in raw if isinstance(x, dict)], ""
            return False, [], str(result.get("message") or "列举失败")

        qs = f"?host={urllib.parse.quote(host)}"
        if root.strip():
            qs += f"&root={urllib.parse.quote(root.strip())}"
        if check_remote_qwen_hostctl_health():
            ok, body = remote_qwen_hostctl_request(
                f"/list_models{qs}",
                method="GET",
                timeout_s=120.0,
            )
            if ok:
                raw = body.get("models") or []
                return True, [dict(x) for x in raw if isinstance(x, dict)], ""
            msg = str(body.get("message") or "list_models 失败")
            if "404" in msg or "unknown /list_models" in msg:
                return _via_ctl()
            return False, [], msg
        return _via_ctl()
    except Exception as exc:
        return False, [], str(exc)


def resolve_deploy_python_for_spec(
    spec: Dict[str, str], *, target: str, host_id: str = ""
) -> str:
    path = str(spec.get("path") or "")
    kind = str(spec.get("kind") or "")
    if target == "local":
        if kind == "lora" or LAKE_QWEN35_OUTPUT_ROOT in path:
            return REMOTE_LAKE_QWEN_PYTHON
        return LOCAL_QWEN_PYTHON_DEFAULT
    if kind == "lora" or LAKE_QWEN35_OUTPUT_ROOT in path or "lake_qwen35" in path:
        return REMOTE_LAKE_QWEN_PYTHON
    prof = remote_qwen_profile(host_id or REMOTE_QWEN_SSH_HOST)
    return str(prof.get("python") or LOCAL_QWEN_PYTHON_DEFAULT)


def resolve_local_qwen_model_dir(model_key: Optional[str] = None) -> Optional[str]:
    """解析本地 Qwen HuggingFace 权重目录。"""
    if model_key:
        found = local_qwen_model_dir_for_key(model_key)
        if found:
            return found
    env = os.environ.get("LOCAL_QWEN_MODEL_DIR", "").strip()
    candidates = []
    if env:
        candidates.append(os.path.abspath(os.path.expanduser(env)))
    candidates.append(LOCAL_QWEN_MODEL_DIR_DEFAULT)
    for _key, _label, _mid, dirname in LOCAL_QWEN_DEPLOY_MODELS:
        candidates.append(os.path.join(LOCAL_QWEN_MODELS_ROOT, dirname))
        candidates.append(os.path.join(EAI_DIR, "models", "Qwen", dirname))
    for path in candidates:
        if qwen_model_dir_valid(path):
            return path
    return None


def resolve_local_qwen_run_script() -> Optional[str]:
    path = os.path.join(EAI_DIR, LOCAL_QWEN_RUN_SCRIPT_NAME)
    return path if os.path.isfile(path) else None


def resolve_local_qwen_bind_host() -> str:
    return "0.0.0.0" if is_running_in_docker() else "127.0.0.1"


def resolve_local_qwen_viewer_api_base() -> str:
    """viewer 访问推理服务的 API。容器为 host 网络时统一用 127.0.0.1。"""
    env = os.environ.get("LOCAL_QWEN_API_BASE", "").strip()
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{LOCAL_QWEN_PORT_DEFAULT}/v1"


def resolve_local_qwen_ctl_url() -> str:
    env = os.environ.get("LOCAL_QWEN_CTL_URL", "").strip()
    if env:
        return env.rstrip("/")
    return LOCAL_QWEN_CTL_URL_DEFAULT.rstrip("/")


def check_local_qwen_server_health(
    api_base: Optional[str] = None, timeout_s: float = 2.0
) -> bool:
    info = fetch_local_qwen_server_info(api_base=api_base, timeout_s=timeout_s)
    return bool(info and info.get("ok"))


def fetch_local_qwen_server_info(
    api_base: Optional[str] = None, timeout_s: float = 2.0
) -> Optional[Dict[str, object]]:
    base = (api_base or resolve_local_qwen_viewer_api_base()).rstrip("/")
    root = base.removesuffix("/v1").rstrip("/")
    url = f"{root}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict):
                return body
    except Exception:
        return None
    return None


def check_local_qwen_hostctl_health(timeout_s: float = 1.0) -> bool:
    url = resolve_local_qwen_ctl_url().rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def local_qwen_hostctl_request(
    path: str,
    method: str = "GET",
    timeout_s: float = 10.0,
    payload: Optional[Dict[str, object]] = None,
) -> Tuple[bool, Dict[str, object]]:
    url = resolve_local_qwen_ctl_url().rstrip("/") + path
    data = None
    if method == "POST":
        data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if method == "POST":
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok", True)), body
    except Exception as exc:
        return False, {"ok": False, "message": str(exc)}


class LocalQwenServiceLauncher(QObject):
    """启动/停止本地 Qwen OpenAI 兼容推理服务。

    Docker 内优先走宿主机 hostctl（:18101），以便使用宿主机 GPU / conda。
    """

    status_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    log_line = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._api_base = resolve_local_qwen_viewer_api_base()
        self._via_hostctl = False
        self._last_model_id = LOCAL_QWEN_MODEL_ID
        self._last_model_label = LOCAL_QWEN_DEPLOY_MODELS[0][1]
        self._starting = False
        self._fail_notified = False

    def api_base(self) -> str:
        return self._api_base

    def last_model_id(self) -> str:
        return self._last_model_id

    def last_model_label(self) -> str:
        return self._last_model_label

    def is_running(self) -> bool:
        if check_local_qwen_server_health(self._api_base):
            return True
        if self._process is not None and self._process.state() == QProcess.Running:
            return True
        return False

    def start(
        self,
        model_key: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
    ) -> None:
        self._api_base = resolve_local_qwen_viewer_api_base()
        key = model_key or LOCAL_QWEN_DEPLOY_MODELS[0][0]
        resolved_dir = model_dir or resolve_local_qwen_model_dir(key)
        resolved_id = model_id or local_qwen_model_id_for_key(key)
        resolved_label = model_label or local_qwen_label_for_key(key)
        self._last_model_id = resolved_id
        self._last_model_label = resolved_label

        if check_local_qwen_server_health(self._api_base):
            info = fetch_local_qwen_server_info(self._api_base) or {}
            running_id = str(info.get("model") or "")
            self._starting = False
            self.running_changed.emit(True)
            if running_id and running_id != resolved_id:
                self.status_message.emit(
                    f"本地 Qwen 已在运行（{running_id}）: {self._api_base}。"
                    f"若要换成 {resolved_label}，请先停止再启动。"
                )
            else:
                self.status_message.emit(
                    f"本地 Qwen 服务已在运行: {self._api_base} ({resolved_label})"
                )
            return

        if resolved_dir is None:
            self._starting = False
            self.running_changed.emit(False)
            self.status_message.emit(
                f"未找到 {resolved_label} 权重（models/Qwen/…）"
            )
            return

        start_payload: Dict[str, object] = {
            "model_dir": resolved_dir,
            "model_id": resolved_id,
            "model_label": resolved_label,
        }
        self._starting = True
        self._fail_notified = False

        if check_local_qwen_hostctl_health():
            self.log_line.emit(
                f"经宿主机 hostctl 启动 {resolved_label}: "
                f"{resolve_local_qwen_ctl_url()}"
            )
            ok, body = local_qwen_hostctl_request(
                "/start",
                method="POST",
                timeout_s=30.0,
                payload=start_payload,
            )
            msg = str(body.get("message") or body)
            self.log_line.emit(msg)
            if ok:
                self._via_hostctl = True
                self.running_changed.emit(True)
                self.status_message.emit(
                    f"已请求宿主机启动 {resolved_label} → {self._api_base}"
                )
            else:
                self._starting = False
                self.running_changed.emit(False)
                self.status_message.emit(f"hostctl 启动失败: {msg}")
            return

        if is_running_in_docker():
            self._starting = False
            self.running_changed.emit(False)
            self.status_message.emit(
                "未检测到宿主机 hostctl。请在宿主机执行: "
                f"python3 {os.path.join(EAI_DIR, 'local_qwen_hostctl.py')} "
                "或重新 bash run_in_docker.sh"
            )
            self.log_line.emit(
                "容器内无 NVIDIA 驱动库，不能直接加载 GPU 模型；"
                "必须由宿主机 hostctl / run_local_qwen.sh 启动。"
            )
            return

        script = resolve_local_qwen_run_script()
        if script is None:
            self.status_message.emit(
                f"未找到 {LOCAL_QWEN_RUN_SCRIPT_NAME}，请确认与 show_camera_topics.py 同目录"
            )
            return

        bind_host = resolve_local_qwen_bind_host()
        script_dir = os.path.dirname(script)
        py = LOCAL_QWEN_PYTHON_DEFAULT
        cmd = (
            f"exec bash {shlex.quote(os.path.basename(script))} "
            f"--host {shlex.quote(bind_host)} "
            f"--port {LOCAL_QWEN_PORT_DEFAULT} "
            f"--model {shlex.quote(resolved_dir)} "
            f"--model-id {shlex.quote(resolved_id)} "
            f"--python {shlex.quote(py)}"
        )
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(script_dir)
        proc.start("bash", ["-lc", cmd])
        self._process = proc
        self._via_hostctl = False
        self.running_changed.emit(True)
        self.status_message.emit(
            f"正在启动 {resolved_label} ({bind_host}:{LOCAL_QWEN_PORT_DEFAULT})，"
            f"viewer API: {self._api_base}"
        )
        self.log_line.emit(f"启动: {cmd}")

    def stop(self) -> None:
        self._starting = False
        if self._via_hostctl or check_local_qwen_hostctl_health():
            ok, body = local_qwen_hostctl_request("/stop", method="POST", timeout_s=15.0)
            msg = str(body.get("message") or body)
            self.log_line.emit(msg)
            self._via_hostctl = False
            self.running_changed.emit(False)
            self.status_message.emit(
                "已停止本地 Qwen 推理服务" if ok else f"停止失败: {msg}"
            )
            return

        if self._process is not None:
            if self._process.state() == QProcess.Running:
                self._process.terminate()
                QTimer.singleShot(3000, self._force_kill_process)
            else:
                self._process = None
        try:
            subprocess.run(
                ["pkill", "-f", "local_qwen_worker.py"],
                capture_output=True,
                timeout=2,
                check=False,
            )
        except Exception:
            pass
        self.running_changed.emit(False)
        self.status_message.emit("已停止本地 Qwen 推理服务")

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            line = line.strip()
            if line:
                self.log_line.emit(line)

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._process = None
        self._starting = False
        self.running_changed.emit(False)
        self.status_message.emit(f"本地 Qwen 服务已退出 (code={exit_code})")

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.Crashed:
            self.status_message.emit(f"本地 Qwen 服务进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.kill()


def load_remote_qwen_ctl():
    """动态加载 remote_qwen_ctl.py（避免 ROS/Qt 环境下的包导入问题）。"""
    import importlib.util

    path = REMOTE_QWEN_CTL_SCRIPT
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {path}")
    spec = importlib.util.spec_from_file_location("remote_qwen_ctl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_remote_qwen_hostctl_url() -> str:
    return REMOTE_QWEN_HOSTCTL_URL.rstrip("/")


def check_remote_qwen_hostctl_health(timeout_s: float = 1.0) -> bool:
    url = resolve_remote_qwen_hostctl_url() + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def remote_qwen_hostctl_request(
    path: str,
    method: str = "GET",
    timeout_s: float = 180.0,
    payload: Optional[Dict[str, object]] = None,
) -> Tuple[bool, Dict[str, object]]:
    url = resolve_remote_qwen_hostctl_url().rstrip("/") + path
    data = None
    if method == "POST":
        data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if method == "POST":
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok", True)), body if isinstance(body, dict) else {}
    except Exception as exc:
        return False, {"ok": False, "message": str(exc)}


def remote_qwen_host_label(host_id: str) -> str:
    for hid, label in REMOTE_QWEN_HOSTS:
        if hid == host_id:
            return label
    return host_id


def remote_qwen_api_base_for_host(host_id: str) -> str:
    try:
        ctl = load_remote_qwen_ctl()
        prof = ctl.apply_host(host_id)
        return str(prof.get("api_base") or REMOTE_QWEN_API_BASE_DEFAULT)
    except Exception:
        port = 18102 if host_id == "tione-develop" else REMOTE_QWEN_LOCAL_PORT
        return f"http://127.0.0.1:{port}/v1"


def remote_qwen_profile(host_id: str) -> Dict[str, object]:
    try:
        ctl = load_remote_qwen_ctl()
        return dict(ctl.apply_host(host_id))
    except Exception:
        return {
            "id": host_id,
            "ssh_host": host_id,
            "api_base": remote_qwen_api_base_for_host(host_id),
            "local_port": 18102 if host_id == "tione-develop" else REMOTE_QWEN_LOCAL_PORT,
            "model_root": "",
            "python": "",
            "remote_work": "",
        }


class RemoteQwenDeployBridge(QObject):
    finished = pyqtSignal(object)


class DeployModelListBridge(QObject):
    finished = pyqtSignal(object)


class RemoteQwenServiceLauncher(QObject):
    """经 SSH 在远程机部署 Qwen，并通过本地端口转发连接。"""

    status_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    log_line = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._starting = False
        self._fail_notified = False
        self._host_id = REMOTE_QWEN_SSH_HOST
        self._last_model_id = "qwen3.5-35b-a3b"
        self._last_model_label = "Qwen3.5-35B-A3B"
        self._last_api_base = remote_qwen_api_base_for_host(self._host_id)
        self._bridge = RemoteQwenDeployBridge()
        self._bridge.finished.connect(self._on_action_finished)
        self._pending_action = ""

    def set_host(self, host_id: str) -> None:
        hid = (host_id or REMOTE_QWEN_SSH_HOST).strip()
        self._host_id = hid
        self._last_api_base = remote_qwen_api_base_for_host(hid)

    def host_id(self) -> str:
        return self._host_id

    def api_base(self) -> str:
        return self._last_api_base or remote_qwen_api_base_for_host(self._host_id)

    def last_model_id(self) -> str:
        return self._last_model_id

    def last_model_label(self) -> str:
        return self._last_model_label

    def is_healthy(self) -> bool:
        return bool(check_local_qwen_server_health(self.api_base()))

    def status(self) -> Dict[str, object]:
        try:
            ctl = load_remote_qwen_ctl()
            ctl.apply_host(self._host_id)
            return dict(ctl.status_payload())
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def start(
        self,
        model_key: Optional[str] = None,
        host_id: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        model_label: Optional[str] = None,
    ) -> None:
        if host_id:
            self.set_host(host_id)
        deploy_key = model_key or "qwen3.5-35b-a3b"
        if model_dir:
            self._last_model_id = str(model_id or os.path.basename(model_dir.rstrip("/")))
            self._last_model_label = str(model_label or self._last_model_id)
        # 先探测隧道：避免上次「等待中」把 _starting 卡死，挡住已就绪的服务
        if self.is_healthy():
            info = fetch_local_qwen_server_info(self.api_base()) or {}
            mid = str(info.get("model") or self._last_model_id)
            self._last_model_id = mid
            self._starting = False
            self._fail_notified = False
            self.running_changed.emit(True)
            self.status_message.emit(
                f"远程 Qwen 已可通过隧道访问: {self.api_base()} ({mid})"
            )
            self.log_line.emit(
                f"远程已就绪: host={self._host_id} model={mid} api={self.api_base()}"
            )
            return
        if self._starting:
            self.status_message.emit("远程部署进行中…")
            return

        self._starting = True
        self._fail_notified = False
        self._pending_action = "deploy"
        self.running_changed.emit(True)
        label = remote_qwen_host_label(self._host_id)
        self.status_message.emit(f"正在远程部署（{label}）…")
        self.log_line.emit(
            f"远程部署开始: host={self._host_id} model_key={deploy_key} "
            f"dir={model_dir or '-'} api={self.api_base()}"
        )
        host = self._host_id
        deploy_payload: Dict[str, object] = {"host_id": host, "model_key": deploy_key}
        if model_dir:
            deploy_payload["model_dir"] = model_dir
        if model_id:
            deploy_payload["model_id"] = model_id
        if model_label:
            deploy_payload["model_label"] = model_label

        def _work() -> None:
            try:
                if check_remote_qwen_hostctl_health():
                    self.log_line.emit(
                        f"经宿主机 remote hostctl 部署: {resolve_remote_qwen_hostctl_url()}"
                    )
                    _ok, result = remote_qwen_hostctl_request(
                        "/deploy",
                        method="POST",
                        timeout_s=300.0,
                        payload=deploy_payload,
                    )
                    if not isinstance(result, dict):
                        result = {"ok": False, "message": str(result)}
                    if not _ok and "ok" not in result:
                        result = {"ok": False, "message": str(result.get("message") or result)}
                else:
                    if is_running_in_docker():
                        result = {
                            "ok": False,
                            "message": (
                                "未检测到宿主机 remote hostctl (:18103)。"
                                "请重新 bash run_in_docker.sh，或在宿主机执行: "
                                f"python3 {os.path.join(EAI_DIR, 'remote_qwen_hostctl.py')}"
                            ),
                        }
                    else:
                        ctl = load_remote_qwen_ctl()
                        ctl.apply_host(host)
                        result = ctl.deploy(
                            model_key=deploy_key,
                            model_dir=model_dir,
                            model_id=model_id,
                            model_label=model_label,
                        )
            except Exception as exc:
                result = {"ok": False, "message": str(exc)}
            self._bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def stop(self, host_id: Optional[str] = None) -> None:
        if host_id:
            self.set_host(host_id)
        self._starting = True
        self._pending_action = "stop"
        self.status_message.emit(f"正在停止远程服务与隧道（{self._host_id}）…")
        host = self._host_id

        def _work() -> None:
            try:
                if check_remote_qwen_hostctl_health():
                    self.log_line.emit(
                        f"经宿主机 remote hostctl 停止: {resolve_remote_qwen_hostctl_url()}"
                    )
                    _ok, result = remote_qwen_hostctl_request(
                        "/stop",
                        method="POST",
                        timeout_s=120.0,
                        payload={"host_id": host},
                    )
                    if not isinstance(result, dict):
                        result = {"ok": False, "message": str(result)}
                else:
                    ctl = load_remote_qwen_ctl()
                    ctl.apply_host(host)
                    result = ctl.stop_all()
            except Exception as exc:
                result = {"ok": False, "message": str(exc)}
            self._bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def _on_action_finished(self, result: object) -> None:
        data = result if isinstance(result, dict) else {"ok": False, "message": str(result)}
        ok = bool(data.get("ok"))
        msg = str(data.get("message") or "")
        action = self._pending_action
        self._pending_action = ""
        self.log_line.emit(msg)

        if data.get("api_base"):
            self._last_api_base = str(data.get("api_base"))
        if data.get("host_id"):
            self._host_id = str(data.get("host_id"))

        if action == "deploy":
            if data.get("model_id"):
                self._last_model_id = str(data.get("model_id"))
            if data.get("model_label"):
                self._last_model_label = str(data.get("model_label"))
            if ok:
                # 模型加载中：隧道可能已通但 /health 尚未 ok
                self._starting = True
                self.status_message.emit(
                    f"远程已启动，等待模型就绪 → {self.api_base()}"
                )
                self.running_changed.emit(True)
            else:
                self._starting = False
                self._fail_notified = True
                self.running_changed.emit(False)
                self.status_message.emit(f"远程部署失败: {msg}")
            return

        # stop
        self._starting = False
        self.running_changed.emit(False)
        self.status_message.emit(msg if ok else f"远程停止异常: {msg}")

    def shutdown(self) -> None:
        # 不在 viewer 退出时强杀远程模型，仅断开当前主机隧道
        try:
            if check_remote_qwen_hostctl_health():
                remote_qwen_hostctl_request(
                    "/stop_tunnel",
                    method="POST",
                    timeout_s=30.0,
                    payload={"host_id": self._host_id},
                )
                return
            ctl = load_remote_qwen_ctl()
            ctl.apply_host(self._host_id)
            ctl.stop_tunnel()
        except Exception:
            pass


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
ARM_MOVE_GOAL_SAMPLE_TICKS = 20  # ~1s @ 20Hz，大范围绝对目标需等 IK 收敛
ARM_MOVE_GOAL_MIN_JOINT_DELTA = 1e-3  # 低于此认为 IK 未给出新解
ARM_MOVE_CART_NEAR_M = 0.03  # 笛卡尔已接近则允许关节几乎不变
ARM_MOVE_IK_RELEASE_TICKS = 3

LLM_API_BASE_DEFAULT = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_CHAT_MAX_HISTORY = 24
LLM_CHAT_TIMEOUT_S = 120.0
LLM_CHAT_VISION_TIMEOUT_S = float(os.environ.get("LLM_CHAT_VISION_TIMEOUT_S", "300"))

# Lake Sys2 提示词（中文；user 输出要求随 system 自动对齐）
LAKE_ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是机器人操作任务编排器。给定场景图像、高层任务名称和上一个子任务，"
    "预测该任务下的全部子任务列表，以及当前可执行的子任务。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「当前子任务」「"
    "上一个子任务」「下一个子任务」四行作答。"
)
LAKE_ORCHESTRATOR_SYSTEM_PROMPT_TRAINING = (
    "你是机器人操作任务的认知编排器。给定场景图像、高层任务名称和进度记忆，"
    "预测该任务下的全部二层（layer-2）子任务列表，以及当前可执行的子任务，并更新语言记忆。"
    "请先输出「所有子任务」编号列表，再分别用「技能」「子任务」「记忆」三行作答。"
)
LAKE_DEFAULT_LANGUAGE_MEMORY = "尚无已完成子任务。"
LAKE_USER_PROMPT_TEMPLATE = (
    "任务：{task}\n"
    "\n"
    "语言记忆：\n"
    "{memory}\n"
    "\n"
    "{output_instruction}"
)


def lake_user_output_instruction(system_prompt: str = "") -> str:
    """根据 system prompt 生成与之匹配的 user 末尾输出要求。"""
    sp = (system_prompt or LAKE_ORCHESTRATOR_SYSTEM_PROMPT).strip()
    if "上一个子任务" in sp and "下一个子任务" in sp:
        return (
            "请输出全部子任务，以及当前技能、当前子任务、"
            "上一个子任务与下一个子任务。"
        )
    if "记忆" in sp and "子任务" in sp:
        return "请输出全部子任务，以及当前技能、子任务与更新后的语言记忆。"
    return (
        "请输出全部子任务，以及当前技能、当前子任务、"
        "上一个子任务与下一个子任务。"
    )


def format_lake_user_prompt(
    task: str,
    memory: str = "",
    *,
    system_prompt: str = "",
) -> str:
    """把用户任务描述包装成与 system 对齐的 user 文本。"""
    task_line = (task or "").strip()
    # 若用户已手写完整训练模板（中/英），直接透传
    if (
        (task_line.startswith("任务：") or task_line.startswith("Task:"))
        and ("语言记忆" in task_line or "Language memory" in task_line)
    ):
        return task_line
    mem = (memory or "").strip() or LAKE_DEFAULT_LANGUAGE_MEMORY
    output_instruction = lake_user_output_instruction(system_prompt)
    return LAKE_USER_PROMPT_TEMPLATE.format(
        task=task_line,
        memory=mem,
        output_instruction=output_instruction,
    )


def extract_lake_memory_from_assistant(text: str) -> Optional[str]:
    """从模型回复中解析记忆字段，供下一轮「语言记忆」使用。"""
    if not text:
        return None
    memory_line: Optional[str] = None
    current_subtask: Optional[str] = None
    previous_subtask: Optional[str] = None
    for raw in str(text).splitlines():
        line = raw.strip()
        low = line.lower()
        if line.startswith("记忆：") or line.startswith("记忆:"):
            mem = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            memory_line = mem.strip() or None
        elif line.startswith("当前子任务：") or line.startswith("当前子任务:"):
            current_subtask = (
                line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            ).strip() or None
        elif line.startswith("上一个子任务：") or line.startswith("上一个子任务:"):
            previous_subtask = (
                line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            ).strip() or None
        elif low.startswith("memory:"):
            memory_line = line.split(":", 1)[-1].strip() or None
    if memory_line:
        return memory_line
    if current_subtask:
        parts: List[str] = []
        empty_prev = {"", "无", "暂无", "—", "-", "none", "null", "n/a"}
        if previous_subtask and previous_subtask.lower() not in empty_prev:
            parts.append(f"机器人已完成：{previous_subtask}。")
        parts.append(f"机器人正在执行：{current_subtask}")
        return "".join(parts)
    return None


def should_use_lake_orchestrator_prompt(api_base: str, model: str) -> bool:
    """本地/远程 Qwen（含 Lake LoRA）走训练对齐提示词。"""
    mid = (model or "").lower()
    base = (api_base or "").lower()
    if "lake" in mid:
        return True
    if any(x in mid for x in ("qwen3.5", "qwen3", "qwen")) and any(
        p in base for p in ("18100", "18102", "8100", "127.0.0.1")
    ):
        return True
    return False


CHAT_HISTORY_DIR = os.path.join(EAI_DIR, "chat_history")
CHAT_USER_SETTINGS_PATH = os.path.join(EAI_DIR, "chat_user_settings.json")
HY_EMBODIED_VLM_API_BASE = os.environ.get(
    "HY_EMBODIED_VLM_API_BASE", "http://127.0.0.1:8080/v1"
)
HY_EMBODIED_VLM_MODEL = os.environ.get("HY_EMBODIED_VLM_MODEL", "hy_a3b")
HY_RXBRAIN_API_BASE = os.environ.get(
    "HY_RXBRAIN_API_BASE", "http://127.0.0.1:8090/v1"
)
HY_RXBRAIN_MODEL = os.environ.get("HY_RXBRAIN_MODEL", "hy-rxbrain")
# 支持附带相机图的预设（OpenAI 兼容 vision / chat completions）
LLM_VISION_PROVIDER_NAMES = {
    LOCAL_QWEN_CHAT_PRESET_NAME,
    REMOTE_QWEN_CHAT_PRESET_NAME,
    "本地 Ollama · Qwen3-VL-4B",
    "Hy-Embodied-VLM-1.0",
    "Hy-Embodied-RxBrain-1.0",
}
# 支持 chat_template_kwargs.enable_thinking 的预设
LLM_THINKING_PROVIDER_NAMES = {
    "Hy-Embodied-VLM-1.0",
}
LLM_PROVIDER_PRESETS: Dict[str, Tuple[str, str, str]] = {
    LOCAL_QWEN_CHAT_PRESET_NAME: (
        LOCAL_QWEN_API_BASE_DEFAULT,
        LOCAL_QWEN_MODEL_ID,
        "EMPTY",
    ),
    REMOTE_QWEN_CHAT_PRESET_NAME: (
        REMOTE_QWEN_API_BASE_DEFAULT,
        "qwen3.5-35b-a3b",
        "EMPTY",
    ),
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
    "Hy-Embodied-VLM-1.0": (
        HY_EMBODIED_VLM_API_BASE,
        HY_EMBODIED_VLM_MODEL,
        "EMPTY",
    ),
    "Hy-Embodied-RxBrain-1.0": (
        HY_RXBRAIN_API_BASE,
        HY_RXBRAIN_MODEL,
        "EMPTY",
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

HAND_JOINT_NAMES = [
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
]
LEFT_HAND_JOINT_NAMES = list(HAND_JOINT_NAMES)
RIGHT_HAND_JOINT_NAMES = list(HAND_JOINT_NAMES)
HAND_CMD_VELOCITY = [2000.0] * 6
HAND_CMD_EFFORT = [1200.0] * 6
LEFT_HAND_CMD_VELOCITY = HAND_CMD_VELOCITY
LEFT_HAND_CMD_EFFORT = HAND_CMD_EFFORT
LEFT_HAND_ANGLE_A_DEFAULT = 0
LEFT_HAND_ANGLE_B_DEFAULT = 45
RIGHT_HAND_ANGLE_A_DEFAULT = 0
RIGHT_HAND_ANGLE_B_DEFAULT = 45

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


def is_head_color_topic(topic: str) -> bool:
    """头部彩色相机（排除 hand_* / wrist 等）。"""
    name = topic.lower().rstrip("/")
    if is_depth_topic(name):
        return False
    if name in {t.lower() for t in CAD_CAPTURE_TOPIC_CANDIDATES}:
        return True
    if "head" not in name:
        return False
    if "hand" in name or "wrist" in name or "fisheye" in name:
        return False
    return "color" in name or name.endswith("/rgb") or name.endswith("_rgb")


def resolve_head_color_topic(available: List[str]) -> Optional[str]:
    """在可用 topic 中优先解析头部彩色相机。"""
    lowered = {t.lower(): t for t in available}
    for pref in CAD_CAPTURE_TOPIC_CANDIDATES:
        hit = lowered.get(pref.lower())
        if hit is not None:
            return hit
    heads = [t for t in available if is_head_color_topic(t)]
    if not heads:
        return None
    heads.sort(key=lambda t: (0 if "head_color" in t.lower() else 1, t))
    return heads[0]


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
        pose_position=np.asarray(result.position_xyz, dtype=np.float32),
        pose_rotation=np.asarray(result.rotation_matrix, dtype=np.float32),
        pose_axis_len=float(max(result.obb_extents) * 0.55),
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


@dataclass
class PoseSettings:
    backend: str = POSE_BACKEND_PCA
    fp_use_http: bool = FP_USE_HTTP_DEFAULT
    fp_server_url: str = FP_SERVER_URL_DEFAULT
    fp_python: str = FP_PYTHON_DEFAULT
    fp_mesh: str = FP_MESH_DEFAULT
    fp_mode: str = "register"


_segment_settings = SegmentSettings()
_segment_settings_lock = threading.Lock()
_pose_settings = PoseSettings()
_pose_settings_lock = threading.Lock()


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


def get_pose_settings() -> PoseSettings:
    with _pose_settings_lock:
        return PoseSettings(
            backend=_pose_settings.backend,
            fp_use_http=_pose_settings.fp_use_http,
            fp_server_url=_pose_settings.fp_server_url,
            fp_python=_pose_settings.fp_python,
            fp_mesh=_pose_settings.fp_mesh,
            fp_mode=_pose_settings.fp_mode,
        )


def set_segment_settings(**kwargs) -> None:
    with _segment_settings_lock:
        for key, value in kwargs.items():
            if hasattr(_segment_settings, key):
                setattr(_segment_settings, key, value)


def set_pose_settings(**kwargs) -> None:
    with _pose_settings_lock:
        for key, value in kwargs.items():
            if hasattr(_pose_settings, key):
                setattr(_pose_settings, key, value)


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


def _summarize_sam3_payload(payload: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key, value in payload.items():
        if key == "image_b64":
            summary[key] = f"<base64 {len(str(value))} chars>"
        else:
            summary[key] = value
    return summary


def _summarize_sam3_mask_result(mask: np.ndarray, method: str) -> Dict[str, object]:
    return {
        "ok": True,
        "method": method,
        "mask": {
            "h": int(mask.shape[0]),
            "w": int(mask.shape[1]),
            "pixels": int(mask.sum()),
        },
    }


def _log_sam3_request_result(
    tag: str,
    request_summary: Dict[str, object],
    result_summary: Dict[str, object],
    elapsed_s: float,
) -> None:
    print(
        f"[show_camera SAM3 {tag}] request: "
        f"{json.dumps(request_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[show_camera SAM3 {tag}] result ({elapsed_s:.3f}s): "
        f"{json.dumps(result_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )


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
    *,
    tag: str = "viewer",
) -> Tuple[np.ndarray, str]:
    cfg = settings or get_segment_settings()
    payload = _sam3_build_request_payload(image_bgr, u, v, text, cfg.sam3_model)
    request_summary = _summarize_sam3_payload(payload)
    request_summary["image_shape"] = [
        int(image_bgr.shape[0]),
        int(image_bgr.shape[1]),
        int(image_bgr.shape[2]),
    ]
    request_summary["transport"] = "HTTP" if cfg.sam3_use_http else "subprocess"
    if cfg.sam3_use_http:
        request_summary["server_url"] = cfg.sam3_server_url
    else:
        request_summary["python"] = cfg.sam3_python
    t0 = time.time()
    try:
        if cfg.sam3_use_http:
            mask, method = _sam3_segment_via_http(
                payload, cfg.sam3_server_url, SAM3_TIMEOUT_S
            )
        else:
            mask, method = _sam3_segment_via_subprocess(
                payload, cfg.sam3_python, SAM3_TIMEOUT_S
            )
        result_summary = _summarize_sam3_mask_result(mask, method)
        _log_sam3_request_result(tag, request_summary, result_summary, time.time() - t0)
        return mask, method
    except Exception as exc:
        _log_sam3_request_result(
            tag,
            request_summary,
            {"ok": False, "error": str(exc)},
            time.time() - t0,
        )
        raise


@dataclass
class Sam3CallResult:
    ok: bool
    topic: str = ""
    u: int = 0
    v: int = 0
    text: str = ""
    method: str = ""
    pixel_count: int = 0
    mask_shape: Tuple[int, int] = (0, 0)
    centroid_uv: Tuple[float, float] = (0.0, 0.0)
    transport: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    mask: Optional[np.ndarray] = None


def format_sam3_call_result_text(result: Sam3CallResult) -> str:
    lines = [
        f"时间: {time.strftime('%H:%M:%S')}",
        f"图像: {result.topic or '--'}",
    ]
    if result.text:
        lines.append(f"文本提示: {result.text}")
    else:
        lines.append(f"点提示: ({result.u}, {result.v})")
    lines.append(f"传输: {result.transport or '--'}")
    if result.ok:
        lines.extend(
            [
                "状态: 成功",
                f"方法: {result.method}",
                f"mask: {result.mask_shape[1]}x{result.mask_shape[0]}",
                f"像素数: {result.pixel_count}",
                f"质心: ({result.centroid_uv[0]:.1f}, {result.centroid_uv[1]:.1f})",
                f"耗时: {result.elapsed_s:.2f}s",
            ]
        )
    else:
        lines.extend(["状态: 失败", f"错误: {result.error}"])
    return "\n".join(lines)


class Sam3CallBridge(QObject):
    finished = pyqtSignal(object)


@dataclass
class FpCallResult:
    ok: bool
    color_topic: str = ""
    depth_topic: str = ""
    u: int = 0
    v: int = 0
    mesh: str = ""
    segment_method: str = ""
    pose_method: str = ""
    transport: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    mask: Optional[np.ndarray] = None
    pose_result: Optional[Object6DPoseResult] = None


def format_fp_call_result_text(result: FpCallResult) -> str:
    lines = [
        f"时间: {time.strftime('%H:%M:%S')}",
        f"彩色: {result.color_topic or '--'}",
        f"深度: {result.depth_topic or '--'}",
        f"提示点: ({result.u}, {result.v})",
        f"mesh: {result.mesh or '--'}",
        f"传输: {result.transport or '--'}",
    ]
    if result.ok and result.pose_result is not None:
        pose = result.pose_result
        lines.extend(
            [
                "状态: 成功",
                f"分割: {result.segment_method}",
                f"位姿: {result.pose_method}",
                f"mask 像素: {pose.pixel_count}",
                f"点云: {pose.point_count} pts",
                f"耗时: {result.elapsed_s:.2f}s",
                format_pose_6d_info(pose, result.u, result.v),
            ]
        )
    elif result.ok:
        lines.extend(["状态: 成功", f"耗时: {result.elapsed_s:.2f}s"])
    else:
        err = format_foundationpose_error(result.error)
        lines.extend(["状态: 失败", f"错误: {err}"])
    return "\n".join(lines)


class FpCallBridge(QObject):
    finished = pyqtSignal(object)


def check_sam3_server_health(server_url: str, timeout_s: float = 2.0) -> bool:
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok"))
    except Exception:
        return False


def check_ollama_server_health(
    api_base: str = OLLAMA_API_BASE_DEFAULT, timeout_s: float = 2.0
) -> bool:
    root = api_base.rstrip("/").removesuffix("/v1").rstrip("/")
    url = f"{root}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_running_in_docker() -> bool:
    return os.path.isfile("/.dockerenv")


def resolve_sam3_run_script() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, SAM3_RUN_SCRIPT_NAME)
    return path if os.path.isfile(path) else None


def resolve_psi_policy_dir(user_dir: Optional[str] = None) -> Optional[str]:
    if user_dir:
        path = os.path.abspath(os.path.expanduser(user_dir.strip()))
        train_py = os.path.join(path, "psi_policy", "train.py")
        if os.path.isfile(train_py):
            return path
    env = os.environ.get(PSI_POLICY_DIR_ENV, "").strip()
    if env and os.path.isdir(env):
        train_py = os.path.join(env, "psi_policy", "train.py")
        if os.path.isfile(train_py):
            return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(here), "psi-policy"),
        os.path.expanduser("~/workspace_liyichao/psi-policy"),
    ]
    for path in candidates:
        train_py = os.path.join(path, "psi_policy", "train.py")
        if os.path.isfile(train_py):
            return os.path.abspath(path)
    return None


def is_valid_psi_policy_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "psi_policy", "train.py"))


def resolve_psi_policy_python(repo_dir: str) -> str:
    env = os.environ.get("PSI_POLICY_PYTHON", "").strip()
    if env:
        return env
    venv_py = os.path.join(repo_dir, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        return venv_py
    return "python3"


def list_psi_policy_workspace_configs(repo_dir: str) -> List[str]:
    config_dir = os.path.join(repo_dir, "psi_policy", "config")
    if not os.path.isdir(config_dir):
        return []
    names: List[str] = []
    for fname in sorted(os.listdir(config_dir)):
        if fname.startswith("example_workspace_") and fname.endswith(".yaml"):
            names.append(fname[:-5])
    return names


def format_hydra_override(key: str, value: str) -> str:
    val = value.strip()
    if not val:
        return ""
    if any(ch in val for ch in ' \t"\'=,'):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key}="{escaped}"'
    return f"{key}={val}"


def resolve_sam3_bind_host() -> str:
    env = os.environ.get("SAM3_HOST", "").strip()
    if env:
        return env
    return "0.0.0.0" if is_running_in_docker() else "127.0.0.1"


def resolve_sam3_viewer_server_url(port: int = SAM3_DEFAULT_PORT) -> str:
    env = os.environ.get("SAM3_SERVER_URL", "").strip()
    if env:
        return env
    if is_running_in_docker():
        candidates = (
            "host.docker.internal",
            "172.17.0.1",
            "127.0.0.1",
        )
        for host in candidates:
            url = f"http://{host}:{port}"
            if check_sam3_server_health(url, timeout_s=0.6):
                return url
        return f"http://172.17.0.1:{port}"
    return f"http://127.0.0.1:{port}"


def _stop_sam3_server_processes() -> None:
    for pattern in ("sam3_segment_worker.py --serve", "sam3_segment_worker --serve"):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=1.0,
                check=False,
            )
        except Exception:
            pass


def _sam3_server_process_running() -> bool:
    return _pgrep_pattern("sam3_segment_worker.py --serve") or _pgrep_pattern(
        "sam3_segment_worker --serve"
    )


def resolve_fp_run_script() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, FP_RUN_SCRIPT_NAME)
    return path if os.path.isfile(path) else None


def resolve_fp_viewer_server_url(port: int = FP_DEFAULT_PORT) -> str:
    env = os.environ.get("FP_SERVER_URL", "").strip()
    if env:
        return env
    if is_running_in_docker():
        candidates = ("host.docker.internal", "172.17.0.1", "127.0.0.1")
        for host in candidates:
            url = f"http://{host}:{port}"
            if check_foundationpose_server_health(url, timeout_s=0.6):
                return url
        return f"http://172.17.0.1:{port}"
    return f"http://127.0.0.1:{port}"


def check_foundationpose_server_health(url: str, timeout_s: float = 1.5) -> bool:
    return fetch_foundationpose_health(url, timeout_s).get("online", False)


def format_foundationpose_error(message: str) -> str:
    msg = (message or "").strip()
    if "nvdiffrast" in msg:
        return (
            f"{msg}\n\n"
            "缺少 nvdiffrast（需在宿主机 foundationpose conda 环境安装）。\n"
            "请在本机终端执行：\n"
            "  cd ~/workspace_liyichao/eai\n"
            "  bash run_foundationpose.sh --install-gpu\n"
            "  bash run_foundationpose.sh --mesh "
            "FoundationPose/demo_data/mustard0/mesh/textured_simple.obj\n"
            "然后 viewer 中重试「调用 FP」。"
        )
    if "pytorch3d" in msg:
        return (
            f"{msg}\n\n"
            "请运行: bash run_foundationpose.sh --install-gpu"
        )
    return msg


def fetch_foundationpose_health(url: str, timeout_s: float = 1.5) -> Dict[str, object]:
    health_url = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        online = resp.status == 200 and bool(body.get("ok", True))
        fp_ready = bool(body.get("fp_ready", body.get("ok", False)))
        missing_deps = body.get("missing_deps") or []
        install_hints = body.get("install_hints") or []
        return {
            "online": online,
            "fp_ready": fp_ready,
            "foundationpose_root": str(body.get("foundationpose_root") or ""),
            "mesh": str(body.get("mesh") or ""),
            "mesh_resolved": str(body.get("mesh_resolved") or ""),
            "missing_deps": list(missing_deps),
            "install_hints": list(install_hints),
        }
    except Exception as exc:
        return {"online": False, "fp_ready": False, "error": str(exc)}


@dataclass
class FpAvailabilityStatus:
    label: str
    detail: str
    tooltip: str
    color: str
    can_invoke: bool
    http_online: bool = False
    fp_ready: bool = False
    mesh_ok: bool = False
    worker_ok: bool = False


def evaluate_fp_availability(
    mesh_path: str,
    use_http: bool,
    server_url: str,
    timeout_s: float = 0.8,
) -> FpAvailabilityStatus:
    worker_ok = os.path.isfile(FP_WORKER_SCRIPT)
    local_mesh_ok, server_mesh_ok = fp_mesh_available(mesh_path, use_http, server_url)
    mesh_ok = local_mesh_ok or server_mesh_ok
    if use_http:
        health = fetch_foundationpose_health(server_url, timeout_s)
        http_online = bool(health.get("online"))
        fp_ready = bool(health.get("fp_ready"))
        fp_root = str(health.get("foundationpose_root") or "")
        mesh_local_tag = "✓" if local_mesh_ok else "✗"
        mesh_srv_tag = "✓" if server_mesh_ok else "✗"
        checks = [
            f"HTTP: {'✓' if http_online else '✗'}",
            f"FP源码: {'✓' if fp_ready else '✗'}",
            f"mesh本地:{mesh_local_tag}",
            f"mesh服务:{mesh_srv_tag}",
        ]
        detail = "  ".join(checks)
        service_ready = http_online and fp_ready
        can_invoke = service_ready
        if service_ready and mesh_ok:
            return FpAvailabilityStatus(
                label="可用",
                detail=detail,
                tooltip=f"FoundationPose 可调用\n{detail}",
                color=UI_ACCENT_GREEN,
                can_invoke=True,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        if service_ready:
            worker_mesh = str(health.get("mesh_resolved") or health.get("mesh") or "")
            return FpAvailabilityStatus(
                label="可用",
                detail=detail,
                tooltip=(
                    "服务已就绪，可点击「调用 FP」尝试。\n"
                    "viewer 本地未找到 mesh（Docker 内常见）；"
                    "将使用 worker 启动时指定的 mesh。\n"
                    + (f"worker mesh: {worker_mesh}\n" if worker_mesh else "")
                    + "若未配置 worker mesh，调用时会报错。"
                ),
                color=UI_ACCENT_GREEN,
                can_invoke=True,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        if http_online:
            missing = health.get("missing_deps") or []
            hints = health.get("install_hints") or []
            hint_text = "\n".join(hints) if hints else (
                "bash run_foundationpose.sh --install-gpu"
            )
            missing_text = ", ".join(missing) if missing else "未知"
            return FpAvailabilityStatus(
                label="未就绪",
                detail=detail,
                tooltip=(
                    "HTTP 服务已启动，但 FoundationPose 依赖未就绪。\n"
                    f"缺少: {missing_text}\n"
                    f"检测: {fp_root or '未找到 eai/FoundationPose'}\n\n"
                    f"请执行:\n{hint_text}"
                ),
                color=UI_ACCENT_ORANGE,
                can_invoke=False,
                http_online=http_online,
                fp_ready=fp_ready,
                mesh_ok=mesh_ok,
                worker_ok=worker_ok,
            )
        err = str(health.get("error") or "")
        return FpAvailabilityStatus(
            label="离线",
            detail=detail,
            tooltip=(
                f"无法连接 {server_url}\n"
                "宿主机请运行: bash run_foundationpose.sh --host 0.0.0.0\n"
                + (f"错误: {err}" if err else "")
            ),
            color=UI_ACCENT_RED,
            can_invoke=False,
            http_online=http_online,
            fp_ready=fp_ready,
            mesh_ok=mesh_ok,
            worker_ok=worker_ok,
        )

    checks = [
        f"worker: {'✓' if worker_ok else '✗'}",
        f"mesh本地:{'✓' if local_mesh_ok else '✗'}",
    ]
    detail = "  ".join(checks)
    can_invoke = worker_ok
    if can_invoke and local_mesh_ok:
        return FpAvailabilityStatus(
            label="可用(本地)",
            detail=detail,
            tooltip=f"子进程模式可调用\n{detail}",
            color=UI_TEXT_SECONDARY,
            can_invoke=True,
            mesh_ok=True,
            worker_ok=worker_ok,
        )
    if can_invoke:
        return FpAvailabilityStatus(
            label="可用(本地)",
            detail=detail,
            tooltip=(
                "子进程 worker 可用，可点击尝试。\n"
                "本地未找到 mesh 时将把路径传给 worker 解析。"
            ),
            color=UI_TEXT_SECONDARY,
            can_invoke=True,
            mesh_ok=local_mesh_ok,
            worker_ok=worker_ok,
        )
    if not worker_ok:
        return FpAvailabilityStatus(
            label="无 worker",
            detail=detail,
            tooltip=f"未找到 {FP_WORKER_SCRIPT}",
            color=UI_ACCENT_ORANGE,
            can_invoke=False,
            mesh_ok=local_mesh_ok,
            worker_ok=worker_ok,
        )
    return FpAvailabilityStatus(
        label="离线",
        detail=detail,
        tooltip="子进程模式不可用",
        color=UI_ACCENT_RED,
        can_invoke=False,
        mesh_ok=local_mesh_ok,
        worker_ok=worker_ok,
    )


def _fp_server_process_running() -> bool:
    return _pgrep_pattern("foundationpose_worker.py --serve") or _pgrep_pattern(
        "foundationpose_worker --serve"
    )


def _encode_depth_b64(depth: np.ndarray) -> Tuple[str, int, int, str]:
    depth_m = depth_to_meters_array(depth)
    h, w = depth_m.shape[:2]
    raw = depth_m.astype(np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii"), h, w, "float32"


def _encode_mask_payload(mask: np.ndarray) -> Dict[str, object]:
    flat = mask.reshape(-1).astype(bool)
    packed = np.packbits(flat)
    return {
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "packed_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _summarize_fp_payload(payload: Dict[str, object]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key, value in payload.items():
        if key in ("rgb_b64", "depth_b64"):
            summary[key] = f"<base64 {len(str(value))} chars>"
        elif key == "mask" and isinstance(value, dict):
            summary[key] = {
                "h": value.get("h"),
                "w": value.get("w"),
                "packed_b64": f"<base64 {len(str(value.get('packed_b64', '')))} chars>",
            }
        else:
            summary[key] = value
    return summary


def _log_fp_request_result(
    tag: str,
    request_summary: Dict[str, object],
    result_summary: Dict[str, object],
    elapsed_s: float,
) -> None:
    print(
        f"[show_camera FP {tag}] request: "
        f"{json.dumps(request_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[show_camera FP {tag}] result ({elapsed_s:.3f}s): "
        f"{json.dumps(result_summary, ensure_ascii=False)}",
        file=sys.stderr,
        flush=True,
    )


def _fp_build_request_payload(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mesh_path: str,
    mode: str = "register",
) -> Dict[str, object]:
    depth_b64, dh, dw, depth_dtype = _encode_depth_b64(depth)
    return {
        "rgb_b64": _encode_image_bgr_b64(color_bgr),
        "depth_b64": depth_b64,
        "depth_h": dh,
        "depth_w": dw,
        "depth_dtype": depth_dtype,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "mask": _encode_mask_payload(mask),
        "mesh": mesh_path,
        "mode": mode,
    }


def _fp_pose_via_http(
    payload: Dict[str, object],
    server_url: str,
    timeout_s: float,
) -> Dict[str, object]:
    url = server_url.rstrip("/") + "/pose"
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
        try:
            err_body = json.loads(detail)
            err_msg = str(err_body.get("error") or detail)
        except json.JSONDecodeError:
            err_msg = detail
        raise RuntimeError(
            format_foundationpose_error(f"FoundationPose HTTP {exc.code}: {err_msg[:400]}")
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"FoundationPose 服务不可达 ({server_url})，请先运行: "
            f"bash run_foundationpose.sh --host 0.0.0.0 --mesh <object.obj>"
        ) from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def _fp_pose_via_subprocess(
    payload: Dict[str, object],
    python_exe: str,
    timeout_s: float,
) -> Dict[str, object]:
    if not os.path.isfile(FP_WORKER_SCRIPT):
        raise RuntimeError(f"未找到 worker: {FP_WORKER_SCRIPT}")
    proc = subprocess.run(
        [python_exe, FP_WORKER_SCRIPT, "--once"],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or stdout or f"FoundationPose worker 退出码 {proc.returncode}")
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FoundationPose worker 输出非 JSON: {stdout[:200]}") from exc
    if not body.get("ok"):
        raise RuntimeError(str(body.get("error") or body))
    return body


def run_foundationpose_estimation(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    settings: Optional[PoseSettings] = None,
    *,
    tag: str = "viewer",
) -> Dict[str, object]:
    cfg = settings or get_pose_settings()
    mesh_path, mesh_source = resolve_fp_mesh_for_call(
        cfg.fp_mesh.strip() or FP_MESH_DEFAULT,
        cfg.fp_use_http,
        cfg.fp_server_url,
    )
    payload = _fp_build_request_payload(
        color_bgr, depth, mask, fx, fy, cx, cy, mesh_path, mode=cfg.fp_mode
    )
    request_summary = _summarize_fp_payload(payload)
    request_summary["mesh_source"] = mesh_source
    request_summary["transport"] = "HTTP" if cfg.fp_use_http else "subprocess"
    if cfg.fp_use_http:
        request_summary["server_url"] = cfg.fp_server_url
    else:
        request_summary["python"] = cfg.fp_python
    t0 = time.time()
    try:
        if cfg.fp_use_http:
            body = _fp_pose_via_http(payload, cfg.fp_server_url, FP_TIMEOUT_S)
        else:
            body = _fp_pose_via_subprocess(payload, cfg.fp_python, FP_TIMEOUT_S)
        result_summary = {
            "ok": True,
            "method": body.get("method"),
            "position_xyz": body.get("position_xyz"),
        }
        _log_fp_request_result(tag, request_summary, result_summary, time.time() - t0)
        return body
    except Exception as exc:
        _log_fp_request_result(
            tag,
            request_summary,
            {"ok": False, "error": str(exc)},
            time.time() - t0,
        )
        raise


def object6d_from_foundationpose(
    mask: np.ndarray,
    depth: np.ndarray,
    seed_u: int,
    seed_v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fp_body: Dict[str, object],
    segment_method: str = "",
) -> Optional[Object6DPoseResult]:
    pose_flat = fp_body.get("pose_matrix")
    if not isinstance(pose_flat, list) or len(pose_flat) != 16:
        return None
    pose = np.asarray(pose_flat, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3].astype(np.float32)
    mesh_origin = pose[:3, 3]
    rpy = rotation_matrix_to_euler_xyz(rotation)
    quat = rotation_matrix_to_quaternion(rotation)

    corners_flat = fp_body.get("obb_corners")
    if isinstance(corners_flat, list) and len(corners_flat) == 24:
        corners = np.asarray(corners_flat, dtype=np.float32).reshape(8, 3)
        # OBB 几何中心（mesh 原点经 to_origin 后可能偏离包围盒中心）
        position = corners.mean(axis=0).astype(np.float64)
    else:
        position = mesh_origin
        half = np.array(fp_body.get("obb_extents") or [0.05, 0.05, 0.05], dtype=np.float32) * 0.5
        corners = obb_corners(position, rotation, half)

    depth_m = depth_to_meters_array(depth)
    refined_mask, points = refine_stereo_segment(
        mask, depth, seed_u, seed_v, fx, fy, cx, cy
    )
    us = (points[:, 0] * fx / points[:, 2] + cx) if len(points) else np.array([seed_u])
    vs = (points[:, 1] * fy / points[:, 2] + cy) if len(points) else np.array([seed_v])
    centroid_uv = (float(np.median(us)), float(np.median(vs)))
    contact_xyz = compute_contact_point_3d(
        seed_u, seed_v, depth, fx, fy, cx, cy, points_3d=points
    )
    if contact_xyz is None:
        contact_xyz = (float(position[0]), float(position[1]), float(position[2]))

    obb_extents = fp_body.get("obb_extents")
    if isinstance(obb_extents, list) and len(obb_extents) == 3:
        extents = tuple(float(x) for x in obb_extents)
    else:
        extents = (
            float(np.linalg.norm(corners[0] - corners[6])),
            float(np.linalg.norm(corners[1] - corners[3])),
            float(np.linalg.norm(corners[0] - corners[4])),
        )

    method = str(fp_body.get("method") or "foundationpose")
    if segment_method:
        method = f"{method}+{segment_method}"

    return Object6DPoseResult(
        mask=refined_mask,
        points_3d=points,
        centroid_uv=centroid_uv,
        contact_uv=(float(seed_u), float(seed_v)),
        contact_xyz=contact_xyz,
        position_xyz=(float(position[0]), float(position[1]), float(position[2])),
        euler_rpy_rad=rpy,
        quaternion_xyzw=quat,
        rotation_matrix=rotation,
        obb_center=(float(position[0]), float(position[1]), float(position[2])),
        obb_extents=extents,
        obb_corners=corners,
        depth_m=float(mesh_origin[2]),
        pixel_count=int(refined_mask.sum()),
        point_count=len(points),
        method=method,
    )


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
        color_bgr, seed_u, seed_v, text=text, settings=cfg, tag="stereo_click"
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
    """8 corners in a fixed order matching draw_obb_on_image / FoundationPose worker."""
    signs = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    corners = []
    for sx, sy, sz in signs:
        local = np.array([sx, sy, sz], dtype=np.float32) * half_extents
        corners.append(center + rotation @ local)
    return np.stack(corners, axis=0).astype(np.float32)


# Cube edge pairs for corner order in obb_corners / FoundationPose worker.
OBB_EDGE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # z-min face
    (4, 5), (5, 6), (6, 7), (7, 4),  # z-max face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def obb_wireframe_edges(corners: np.ndarray) -> np.ndarray:
    lines = []
    for a, b in OBB_EDGE_PAIRS:
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
    pose_settings: Optional[PoseSettings] = None,
) -> Tuple[Optional[Object6DPoseResult], str]:
    pcfg = pose_settings or get_pose_settings()
    try:
        mask, method = segment_object_at_click_with_backend(
            seed_u, seed_v, depth, color_bgr=color_bgr
        )
    except Exception as exc:
        return None, f"点击 ({seed_u}, {seed_v})  |  分割失败: {exc}"
    if not mask.any():
        return None, f"点击 ({seed_u}, {seed_v})  |  立体分割失败（无有效区域）"

    if pcfg.backend == POSE_BACKEND_FOUNDATIONPOSE:
        if color_bgr is None:
            return None, (
                f"点击 ({seed_u}, {seed_v})  |  FoundationPose 需要彩色图"
                "（请同时订阅 color topic）"
            )
        try:
            fp_body = run_foundationpose_estimation(
                color_bgr,
                depth,
                mask,
                fx,
                fy,
                cx,
                cy,
                settings=pcfg,
                tag="stereo_click",
            )
            result = object6d_from_foundationpose(
                mask,
                depth,
                seed_u,
                seed_v,
                fx,
                fy,
                cx,
                cy,
                fp_body,
                segment_method=method,
            )
        except Exception as exc:
            return None, f"点击 ({seed_u}, {seed_v})  |  FoundationPose 失败: {exc}"
        if result is None:
            return None, f"点击 ({seed_u}, {seed_v})  |  FoundationPose 响应无效"
        return result, format_pose_6d_info(result, seed_u, seed_v)

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
    # FoundationPose 物体姿态通常不可直接作为 TCP 姿态；位置绝对、姿态保持当前 TCP
    keep_tcp_orientation: bool = False

    @classmethod
    def from_pose_result(
        cls,
        result: Object6DPoseResult,
        camera_frame: str,
        source_topic: str,
    ) -> SegmentPoseTarget:
        method = (result.method or "").lower()
        # FoundationPose：物体中心绝对位置；几何分割：点击处接触点
        use_object_pose = "foundationpose" in method
        if use_object_pose:
            position = result.obb_center
            label = (
                f"{source_topic.split('/')[-1]} FoundationPose "
                f"({result.point_count} pts)"
            )
        else:
            position = result.contact_xyz
            label = (
                f"{source_topic.split('/')[-1]} 接触TCP "
                f"({result.point_count} pts)"
            )
        return cls(
            camera_frame=camera_frame,
            position_xyz=position,
            quaternion_xyzw=result.quaternion_xyzw,
            source_topic=source_topic,
            contact_uv=result.contact_uv,
            obb_center_xyz=result.obb_center,
            label=label,
            keep_tcp_orientation=use_object_pose,
        )


@dataclass
class ResolvedArmMoveGoal:
    position_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    label: str
    arm_side: str = "left"  # "left" | "right"


def arm_side_label(side: str) -> str:
    return "左臂" if side == "left" else "右臂"


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
    arm_side: str = "left",
) -> Optional[ResolvedArmMoveGoal]:
    if abs(dx) + abs(dy) + abs(dz) < 1e-6:
        return None
    goal_xyz = (current_xyz[0] + dx, current_xyz[1] + dy, current_xyz[2] + dz)
    side_name = arm_side_label(arm_side)
    return ResolvedArmMoveGoal(
        position_xyz=goal_xyz,
        quaternion_xyzw=current_quat,
        label=f"{side_name}相对偏移 Δ({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m",
        arm_side=arm_side,
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
    color: Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 2,
) -> None:
    if len(corners_3d) == 0:
        return
    uv = project_points_to_uv(corners_3d, fx, fy, cx, cy).astype(np.int32)
    h, w = image.shape[:2]
    for a, b in OBB_EDGE_PAIRS:
        if corners_3d[a, 2] <= 0.01 or corners_3d[b, 2] <= 0.01:
            continue
        pa = (int(np.clip(uv[a, 0], 0, w - 1)), int(np.clip(uv[a, 1], 0, h - 1)))
        pb = (int(np.clip(uv[b, 0], 0, w - 1)), int(np.clip(uv[b, 1], 0, h - 1)))
        cv2.line(image, pa, pb, color, thickness, cv2.LINE_AA)


def draw_pose_axes_on_image(
    image: np.ndarray,
    position_xyz: np.ndarray,
    rotation: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    axis_len: float,
) -> None:
    """在图像上画 RGB = XYZ 坐标轴。"""
    if float(position_xyz[2]) <= 0.01:
        return
    origin = np.asarray(position_xyz, dtype=np.float32).reshape(3)
    R = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    # BGR: X红 Y绿 Z蓝
    axis_bgr = ((0, 0, 255), (0, 220, 0), (255, 120, 0))
    labels = ("X", "Y", "Z")
    h, w = image.shape[:2]
    pts = [origin] + [origin + R[:, i] * axis_len for i in range(3)]
    uv = project_points_to_uv(np.stack(pts, axis=0), fx, fy, cx, cy).astype(np.int32)
    o = (int(np.clip(uv[0, 0], 0, w - 1)), int(np.clip(uv[0, 1], 0, h - 1)))
    cv2.circle(image, o, 5, (255, 255, 255), -1, cv2.LINE_AA)
    for i in range(3):
        if pts[i + 1][2] <= 0.01:
            continue
        p = (int(np.clip(uv[i + 1, 0], 0, w - 1)), int(np.clip(uv[i + 1, 1], 0, h - 1)))
        cv2.arrowedLine(image, o, p, axis_bgr[i], 3, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(
            image,
            labels[i],
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            axis_bgr[i],
            2,
            cv2.LINE_AA,
        )


def apply_segment_overlay(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    centroid_uv: Optional[Tuple[float, float]] = None,
    obb_corners: Optional[np.ndarray] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    contact_uv: Optional[Tuple[float, float]] = None,
    seed_uv: Optional[Tuple[float, float]] = None,
    pose_position: Optional[np.ndarray] = None,
    pose_rotation: Optional[np.ndarray] = None,
    pose_axis_len: Optional[float] = None,
) -> np.ndarray:
    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    mask_bool: Optional[np.ndarray] = None
    if mask is not None:
        if mask.shape[:2] != display.shape[:2]:
            mask = resize_mask_to_shape(mask, display.shape[:2])
        mask_bool = np.asarray(mask).astype(bool)

    if mask_bool is not None and mask_bool.any():
        ys, xs = np.where(mask_bool)
        pad = 12
        h, w = display.shape[:2]
        v0 = max(0, int(ys.min()) - pad)
        v1 = min(h, int(ys.max()) + pad + 1)
        u0 = max(0, int(xs.min()) - pad)
        u1 = min(w, int(xs.max()) + pad + 1)

        roi = display[v0:v1, u0:u1]
        roi_mask = mask_bool[v0:v1, u0:u1]
        tint = np.array([40, 220, 80], dtype=np.float32)
        blended = roi.astype(np.float32)
        blended[roi_mask] = blended[roi_mask] * 0.42 + tint * 0.58
        display[v0:v1, u0:u1] = np.clip(blended, 0, 255).astype(np.uint8)

        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(display, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    if obb_corners is not None and intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        draw_obb_on_image(display, obb_corners, fx, fy, cx, cy, color=(0, 220, 255), thickness=2)

    if (
        pose_position is not None
        and pose_rotation is not None
        and intrinsics is not None
    ):
        fx, fy, cx, cy = intrinsics
        axis_len = float(pose_axis_len) if pose_axis_len is not None else 0.05
        if axis_len <= 1e-4:
            axis_len = 0.05
        draw_pose_axes_on_image(
            display,
            np.asarray(pose_position, dtype=np.float32),
            np.asarray(pose_rotation, dtype=np.float32),
            fx,
            fy,
            cx,
            cy,
            axis_len,
        )

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

    if seed_uv is not None:
        su, sv = int(round(seed_uv[0])), int(round(seed_uv[1]))
        cv2.drawMarker(
            display, (su, sv), (0, 80, 255), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA
        )
        cv2.circle(display, (su, sv), 10, (0, 80, 255), 2, cv2.LINE_AA)
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
        return _html_span(f"{title}: (无数据)", UI_TEXT_MUTED) + "<br/>"
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
        "<div style='font-family: Monospace, Consolas, monospace; font-size: 10pt; "
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
        parts.append(_html_span("左臂 TCP: (无数据)", UI_TEXT_MUTED))

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
        parts.append(_html_span("右臂 TCP: (无数据)", UI_TEXT_MUTED))

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


def make_hand_command(
    side: str,
    position: float | Sequence[float],
) -> JointState:
    """构建睿研灵巧手控制命令。

    position: 标量则六指同开合；序列则按 HAND_JOINT_NAMES 顺序逐关节 (0=张, 1=合)。
    """
    side = "right" if side == "right" else "left"
    msg = JointState()
    msg.header.frame_id = f"{side}_hand"
    names = RIGHT_HAND_JOINT_NAMES if side == "right" else LEFT_HAND_JOINT_NAMES
    msg.name = list(names)
    if isinstance(position, (int, float)):
        pos = max(0.0, min(1.0, float(position)))
        msg.position = [pos] * len(names)
    else:
        vals = [max(0.0, min(1.0, float(v))) for v in position]
        if len(vals) < len(names):
            vals = vals + [vals[-1] if vals else 0.0] * (len(names) - len(vals))
        msg.position = vals[: len(names)]
    msg.velocity = list(HAND_CMD_VELOCITY)
    msg.effort = list(HAND_CMD_EFFORT)
    return msg


def make_left_hand_command(position: float | Sequence[float]) -> JointState:
    return make_hand_command("left", position)


def make_right_hand_command(position: float | Sequence[float]) -> JointState:
    return make_hand_command("right", position)


@dataclass
class LlmChatConfig:
    api_base: str = LLM_API_BASE_DEFAULT
    model: str = LLM_MODEL_DEFAULT
    api_key: str = ""
    enable_thinking: bool = False
    max_tokens: int = 1536
    system_prompt: str = LAKE_ORCHESTRATOR_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> LlmChatConfig:
        settings = load_chat_user_settings()
        system_prompt = str(
            settings.get("system_prompt") or LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        ).strip() or LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        return cls(
            api_base=os.environ.get("LLM_API_BASE", LLM_API_BASE_DEFAULT),
            model=os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
            api_key=os.environ.get(LLM_API_KEY_ENV, "").strip(),
            system_prompt=system_prompt,
        )


def encode_bgr_image_jpeg_b64(image_bgr: np.ndarray, max_side: int = 1280) -> str:
    """BGR 图像 → JPEG base64（供 OpenAI vision image_url）。"""
    img = np.asarray(image_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w, 1)))
    if scale < 0.999:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 5.0,
) -> Tuple[bool, object, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(raw), ""
            except Exception:
                return True, raw, ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, None, f"HTTP {exc.code}: {detail[:200]}"
    except Exception as exc:
        return False, None, str(exc)


def classify_llm_service_kind(
    api_base: str,
    model_ids: Sequence[str],
    *,
    ollama_ok: bool = False,
    health: Optional[Dict[str, object]] = None,
    configured_model: str = "",
) -> str:
    base = (api_base or "").lower()
    ids = [str(x).strip() for x in model_ids if str(x).strip()]
    ids_l = " ".join(ids).lower()
    cfg = (configured_model or "").lower()
    health = health or {}

    if "dashscope.aliyuncs.com" in base or "compatible-mode" in base:
        return "阿里云百炼 / DashScope"
    if "api.openai.com" in base:
        return "OpenAI 官方"
    if "api.anthropic.com" in base:
        return "Anthropic"
    if ":8090" in base or "hy-rxbrain" in ids_l or "hy-rxbrain" in cfg:
        return "Hy-Embodied-RxBrain"
    if ":8080" in base or "hy_a3b" in ids_l or "hy_a3b" in cfg or "hy-embodied" in ids_l:
        return "Hy-Embodied-VLM"
    if ollama_ok or ":11434" in base:
        if "qwen3-vl" in ids_l or "qwen3-vl" in cfg:
            return "Ollama · Qwen3-VL"
        if "qwen3.5" in ids_l or "qwen3.5" in cfg:
            return "Ollama · Qwen3.5"
        return "Ollama"
    model_dir = str(health.get("model_dir") or "").lower()
    live = str(health.get("model") or "").lower()
    blob = f"{ids_l} {cfg} {model_dir} {live}"
    if (
        live
        or model_dir
        or "qwen3.5-4b" in blob
        or "qwen3.5-35b" in blob
        or "/models/qwen/" in model_dir
    ):
        if "35b" in blob or "a3b" in blob:
            return "本地 Qwen3.5-35B-A3B 服务"
        if "4b" in blob:
            return "本地 Qwen3.5-4B 服务"
        return "本地 Qwen 推理服务"
    if ids:
        return "OpenAI 兼容服务"
    return "未知服务"


def probe_llm_endpoint(
    api_base: str,
    api_key: str = "",
    configured_model: str = "",
    timeout_s: float = 6.0,
) -> Dict[str, object]:
    """探测 API 地址上的服务类型与可用模型列表。"""
    base = (api_base or "").strip().rstrip("/") or LLM_API_BASE_DEFAULT
    root = base.removesuffix("/v1").rstrip("/")
    key = (api_key or "").strip() or "EMPTY"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    details: List[str] = []
    model_ids: List[str] = []
    health: Dict[str, object] = {}
    ollama_ok = False
    reachable = False

    ok, body, err = _http_get_json(f"{root}/health", headers=headers, timeout_s=timeout_s)
    if ok:
        reachable = True
        details.append("健康检查: /health 可达")
        if isinstance(body, dict):
            health = body
            if body.get("model"):
                mid = str(body.get("model"))
                if mid not in model_ids:
                    model_ids.append(mid)
            if body.get("model_dir"):
                details.append(f"model_dir={body.get('model_dir')}")
            if body.get("device"):
                details.append(f"device={body.get('device')}")
            if body.get("ok") is False and body.get("error"):
                details.append(f"服务未就绪: {body.get('error')}")
    else:
        details.append(f"健康检查: /health 不可用 ({err})")

    ok, body, err = _http_get_json(
        f"{base}/models", headers=headers, timeout_s=timeout_s
    )
    if ok:
        reachable = True
        details.append("模型列表: /models 可达")
        if isinstance(body, dict):
            for item in body.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"])
                    if mid not in model_ids:
                        model_ids.append(mid)
        elif isinstance(body, list):
            for item in body:
                if isinstance(item, dict) and item.get("id"):
                    mid = str(item["id"])
                    if mid not in model_ids:
                        model_ids.append(mid)
    else:
        details.append(f"模型列表: /models 不可用 ({err})")

    ok, body, err = _http_get_json(f"{root}/api/tags", timeout_s=timeout_s)
    if ok:
        reachable = True
        ollama_ok = True
        details.append("Ollama: /api/tags 可达")
        if isinstance(body, dict):
            for item in body.get("models") or []:
                if isinstance(item, dict):
                    mid = str(item.get("name") or item.get("model") or "").strip()
                    if mid and mid not in model_ids:
                        model_ids.append(mid)
    else:
        details.append(f"Ollama: /api/tags 不可用 ({err})")

    kind = classify_llm_service_kind(
        base,
        model_ids,
        ollama_ok=ollama_ok,
        health=health,
        configured_model=configured_model,
    )
    matched = False
    if configured_model:
        cfg_l = configured_model.lower()
        matched = any(cfg_l == m.lower() or cfg_l in m.lower() or m.lower() in cfg_l for m in model_ids)

    return {
        "ok": reachable,
        "kind": kind,
        "api_base": base,
        "root": root,
        "configured_model": configured_model,
        "configured_model_matched": matched,
        "models": model_ids,
        "health": health,
        "details": details,
        "error": "" if reachable else "无法连接当前 API",
    }


def format_chat_prompt_dump(
    messages: List[Dict[str, object]],
    *,
    api_base: str = "",
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 0,
    enable_thinking: bool = False,
    timeout_s: float = 0.0,
) -> str:
    """把即将发给 API 的提示词格式化为可读文本（图片 base64 截断）。"""
    lines: List[str] = [
        "======== 完整请求提示词 ========",
        f"api_base: {api_base}",
        f"model: {model}",
        f"temperature: {temperature}",
        f"max_tokens: {max_tokens}",
        f"chat_template_kwargs.enable_thinking: {bool(enable_thinking)}",
    ]
    if timeout_s > 0:
        lines.append(f"timeout_s: {timeout_s}")
    lines.append(f"messages({len(messages)}):")
    for i, msg in enumerate(messages):
        role = str(msg.get("role") or "")
        content = msg.get("content")
        lines.append(f"----- [{i}] role={role} -----")
        if isinstance(content, str):
            lines.append(content if content.strip() else "(空文本)")
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    lines.append(f"  part[{j}]: {part!r}")
                    continue
                ptype = str(part.get("type") or "")
                if ptype == "text":
                    lines.append(f"  part[{j}] text: {part.get('text') or ''}")
                elif ptype in ("image_url", "image"):
                    url = ""
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url") or "")
                    elif isinstance(image_url, str):
                        url = image_url
                    elif part.get("url"):
                        url = str(part.get("url") or "")
                    if url.startswith("data:"):
                        header, _, rest = url.partition(",")
                        lines.append(
                            f"  part[{j}] image: {header},<base64 len={len(rest)}>"
                        )
                    elif url:
                        lines.append(f"  part[{j}] image_url: {url[:200]}")
                    else:
                        lines.append(f"  part[{j}] image: (无 url)")
                else:
                    lines.append(f"  part[{j}] type={ptype}: {part!r}"[:500])
        elif content is None:
            lines.append("(无 content)")
        else:
            lines.append(repr(content)[:1000])
    lines.append("======== 提示词结束 ========")
    return "\n".join(lines)


class LlmChatClient:
    """OpenAI 兼容 Chat Completions API 客户端（支持 OpenAI / Ollama / Hy-Embodied 等）。"""

    def __init__(self, config: LlmChatConfig) -> None:
        self.config = config

    def _auth_headers(self) -> Dict[str, str]:
        api_key = self.config.api_key.strip() or "EMPTY"
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def probe(self, timeout_s: float = 6.0) -> Dict[str, object]:
        """探测当前 API 可达性与模型种类。"""
        return probe_llm_endpoint(
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            configured_model=self.config.model,
            timeout_s=timeout_s,
        )

    def chat(
        self,
        messages: List[Dict[str, object]],
        timeout_s: Optional[float] = None,
    ) -> str:
        api_key = self.config.api_key.strip()
        if not api_key:
            raise RuntimeError(
                f"未配置 API Key，请设置环境变量 {LLM_API_KEY_ENV}，"
                "或在对话面板中填写（Ollama / Hy-Embodied 本地服务可填 EMPTY）"
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload: Dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": int(self.config.max_tokens),
            # Qwen3.5 默认会进 <think>；未勾选 thinking 时显式关闭，避免长思考挤掉答案
            "chat_template_kwargs": {
                "enable_thinking": bool(self.config.enable_thinking)
            },
        }
        # 带图推理更慢（视觉编码 + 长输出），单独放宽超时
        if timeout_s is None:
            has_image = False
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and str(part.get("type") or "") in (
                            "image_url",
                            "image",
                        ):
                            has_image = True
                            break
                if has_image:
                    break
            timeout_s = (
                LLM_CHAT_VISION_TIMEOUT_S if has_image else LLM_CHAT_TIMEOUT_S
            )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._auth_headers(),
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
        reasoning = message.get("reasoning_content")
        parts: List[str] = []
        if reasoning:
            parts.append(f"[thinking]\n{str(reasoning).strip()}")
        if content:
            parts.append(str(content).strip() if not reasoning else f"[answer]\n{str(content).strip()}")
        if not parts:
            raise RuntimeError(f"API 返回空内容: {body}")
        return "\n\n".join(parts).strip()


class LlmChatBridge(QObject):
    finished = pyqtSignal(str, bool)


class LlmProbeBridge(QObject):
    finished = pyqtSignal(object)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_chat_history_dir() -> str:
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return CHAT_HISTORY_DIR


def load_chat_user_settings() -> Dict[str, object]:
    """加载对话面板用户设置（system prompt 等）。"""
    try:
        with open(CHAT_USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_chat_user_settings(data: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(CHAT_USER_SETTINGS_PATH), exist_ok=True)
    with open(CHAT_USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return CHAT_USER_SETTINGS_PATH


def chat_history_path(history_id: str) -> str:
    safe = "".join(ch for ch in history_id if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("invalid history id")
    return os.path.join(ensure_chat_history_dir(), f"{safe}.json")


def default_chat_history_title(messages: List[Dict[str, object]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            text = " ".join(content.strip().split())
            return text[:40] + ("…" if len(text) > 40 else "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        text = " ".join(text.split())
                        return text[:40] + ("…" if len(text) > 40 else "")
    return f"对话 {datetime.now().strftime('%m-%d %H:%M')}"


def serialize_chat_messages(messages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """仅持久化可 JSON 化的文本轮次（含时间戳与耗时）。"""
    out: List[Dict[str, object]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            continue
        item: Optional[Dict[str, object]] = None
        if isinstance(content, str):
            item = {"role": role, "content": content}
        elif isinstance(content, list):
            texts = [
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = "\n".join(t for t in texts if t).strip()
            if text:
                item = {"role": role, "content": text}
        if item is None:
            continue
        ts = msg.get("ts")
        if isinstance(ts, str) and ts.strip():
            item["ts"] = ts.strip()
        latency = msg.get("latency_s")
        if isinstance(latency, (int, float)):
            item["latency_s"] = float(latency)
        out.append(item)
    return out


def format_chat_timestamp(ts: Optional[str] = None) -> str:
    """本地时间显示；ts 可为 ISO 字符串。"""
    if ts:
        raw = ts.strip()
        try:
            if raw.endswith("Z"):
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
            else:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is not None:
                    dt = dt.astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return raw
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_chat_latency(latency_s: Optional[float]) -> str:
    if latency_s is None:
        return ""
    try:
        val = float(latency_s)
    except (TypeError, ValueError):
        return ""
    if val < 0:
        return ""
    if val < 1.0:
        return f"{val * 1000.0:.0f} ms"
    return f"{val:.2f} s"


def list_chat_histories() -> List[Dict[str, object]]:
    root = ensure_chat_history_dir()
    items: List[Dict[str, object]] = []
    for name in os.listdir(root):
        if not name.endswith(".json"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            hid = str(data.get("id") or os.path.splitext(name)[0])
            items.append(
                {
                    "id": hid,
                    "title": str(data.get("title") or hid),
                    "updated_at": str(data.get("updated_at") or ""),
                    "created_at": str(data.get("created_at") or ""),
                    "model": str(data.get("model") or ""),
                    "path": path,
                    "message_count": len(data.get("messages") or []),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return items


def load_chat_history(history_id: str) -> Dict[str, object]:
    path = chat_history_path(history_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("invalid history file")
    data["id"] = str(data.get("id") or history_id)
    return data


def save_chat_history_record(record: Dict[str, object]) -> str:
    history_id = str(record.get("id") or "").strip() or uuid.uuid4().hex[:12]
    record["id"] = history_id
    record["updated_at"] = _utc_now_iso()
    if not record.get("created_at"):
        record["created_at"] = record["updated_at"]
    path = chat_history_path(history_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def delete_chat_history(history_id: str) -> None:
    path = chat_history_path(history_id)
    if os.path.isfile(path):
        os.remove(path)


def rename_chat_history(history_id: str, title: str) -> None:
    data = load_chat_history(history_id)
    data["title"] = title.strip() or str(data.get("title") or history_id)
    save_chat_history_record(data)


class ChatHistoryDialog(QWidget):
    """主窗口内居中浮层（看起来像弹窗，但不新建 X11 窗口，避免弄坏 fcitx）。"""

    finished = pyqtSignal(int)  # QDialog.Accepted / Rejected

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # parent 必须是顶层窗口，浮层盖在其上
        win = parent.window() if parent is not None else parent
        super().__init__(win)
        self.selected_id: str = ""
        self.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setObjectName("historyMask")
        self.setStyleSheet("#historyMask { background-color: rgba(0,0,0,160); }")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        panel = QFrame()
        panel.setObjectName("historyPanel")
        panel.setStyleSheet(
            "#historyPanel {"
            "  background-color: #2b2b2b;"
            "  border: 1px solid #666;"
            "  border-radius: 6px;"
            "}"
        )
        panel.setMinimumSize(480, 420)
        panel.setMaximumSize(720, 560)
        root.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(panel)
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("历史对话")
        title.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)
        tip = QLabel(
            "选择一条历史对话：双击或点「加载」将把最后一轮问题放入输入框"
            "（不自动请求）；可修改标题或删除。点空白处或「关闭」退出。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        layout.addWidget(tip)
        self.list_widget = QListWidget()
        self.list_widget.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.list_widget.setFocusPolicy(Qt.ClickFocus)
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #1a1a1a; color: #ddd; border: 1px solid #555; }"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_load_clicked)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget, 1)
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("标题"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("选中条目后可改标题（可输入中文）")
        self.title_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.title_edit.setFocusPolicy(Qt.StrongFocus)
        title_row.addWidget(self.title_edit, stretch=1)
        self.save_title_btn = QPushButton("保存标题")
        self.save_title_btn.setFocusPolicy(Qt.NoFocus)
        self.save_title_btn.clicked.connect(self._on_save_title_clicked)
        title_row.addWidget(self.save_title_btn)
        layout.addLayout(title_row)
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("加载")
        self.load_btn.setFocusPolicy(Qt.NoFocus)
        self.load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.load_btn)
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setFocusPolicy(Qt.NoFocus)
        self.delete_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        btn_row.addWidget(self.delete_btn)
        self._delete_armed = False
        btn_row.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.setFocusPolicy(Qt.NoFocus)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {UI_TEXT_MUTED};")
        layout.addWidget(self._status)
        self._panel = panel
        self._reload()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.raise_()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        # 点遮罩空白关闭
        if not self._panel.geometry().contains(event.pos()):
            self.reject()
            event.accept()
            return
        super().mousePressEvent(event)

    def accept(self) -> None:
        self.hide()
        self.finished.emit(int(QDialog.Accepted))

    def reject(self) -> None:
        self.hide()
        self.finished.emit(int(QDialog.Rejected))

    def _reload(self) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        self.list_widget.clear()
        self.title_edit.clear()
        for item in list_chat_histories():
            title = str(item.get("title") or "")
            updated = str(item.get("updated_at") or "")
            model = str(item.get("model") or "")
            count = int(item.get("message_count") or 0)
            label = f"{title}"
            if updated:
                label += f"  ·  {updated}"
            if model:
                label += f"  ·  {model}"
            label += f"  ·  {count} 条"
            row = QListWidgetItem(label)
            row.setData(Qt.UserRole, str(item.get("id") or ""))
            row.setData(Qt.UserRole + 1, title)
            row.setToolTip(str(item.get("path") or ""))
            self.list_widget.addItem(row)
        if self.list_widget.count() == 0:
            empty = QListWidgetItem("（暂无保存的对话）")
            empty.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(empty)
        self._status.setText("")

    def _current_id(self) -> str:
        item = self.list_widget.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _on_current_changed(self, current, _previous) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        if current is None or not (current.flags() & Qt.ItemIsEnabled):
            self.title_edit.clear()
            return
        self.title_edit.setText(str(current.data(Qt.UserRole + 1) or ""))
        self.title_edit.setFocus(Qt.OtherFocusReason)

    def _on_load_clicked(self, *_args) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        self.selected_id = hid
        self.accept()

    def _on_save_title_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        new_title = self.title_edit.text().strip()
        if not new_title:
            self._status.setText("标题不能为空")
            return
        try:
            rename_chat_history(hid, new_title)
        except Exception as exc:
            self._status.setText(f"保存标题失败: {exc}")
            return
        self._reload()
        self._status.setText("标题已保存")

    def _on_delete_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        if not self._delete_armed:
            self._delete_armed = True
            self.delete_btn.setText("再点确认删除")
            self._status.setText("再点一次「再点确认删除」才会删除")
            return
        try:
            delete_chat_history(hid)
        except Exception as exc:
            self._status.setText(f"删除失败: {exc}")
            self._delete_armed = False
            self.delete_btn.setText("删除")
            return
        self._reload()
        self._status.setText("已删除")



class ChatInputEdit(QTextEdit):
    """对话输入框：获得焦点时主动唤醒 fcitx；Enter 发送但不打断中文组字。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._chat_panel: Optional["ChatPanelWidget"] = None
        self._ime_composing = False
        self._ime_guard_until = 0.0

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        global _IME_LAST_TEXT_WIDGET
        _IME_LAST_TEXT_WIDGET = self

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        # 必须在此跟踪组字；用 eventFilter 拦截 Enter 容易抢掉 fcitx 上屏键
        try:
            preedit = bool(event.preeditString())
            commit = bool(event.commitString())
            self._ime_composing = preedit
            if preedit:
                self._ime_guard_until = time.monotonic() + 0.6
            elif commit:
                # 提交后短保护，避免紧随其后的 Enter 被当成「发送」
                self._ime_guard_until = time.monotonic() + 0.2
        except Exception:
            pass
        super().inputMethodEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            if self._ime_composing or time.monotonic() < self._ime_guard_until:
                super().keyPressEvent(event)
                return
            try:
                app = QApplication.instance()
                im = app.inputMethod() if app is not None else None
                if im is not None and im.isVisible():
                    super().keyPressEvent(event)
                    return
            except Exception:
                pass
            panel = self._chat_panel
            if panel is not None:
                if time.monotonic() < getattr(panel, "_suppress_send_until", 0.0):
                    # 历史加载后短暂不发送，交给文本框（换行/IME）
                    super().keyPressEvent(event)
                    return
                panel._on_send_clicked()
                event.accept()
                return
        super().keyPressEvent(event)

    def _notify_input_method(self, *, full_query: bool = False) -> None:
        try:
            app = QApplication.instance()
            if app is None:
                return
            im = app.inputMethod()
            if im is None:
                return
            if full_query:
                im.update(Qt.ImQueryAll)
            else:
                im.update(
                    Qt.ImEnabled
                    | Qt.ImCursorRectangle
                    | Qt.ImHints
                    | Qt.ImInputItemClipRectangle
                )
        except Exception:
            pass


class ChatPanelWidget(QWidget):
    """文本/视觉对话面板：OpenAI 兼容 API（含 Hy-Embodied-VLM / RxBrain）。"""

    status_message = pyqtSignal(str)
    # 附带图片变更：空字符串表示已清除
    attached_image_changed = pyqtSignal(str)

    def __init__(
        self,
        config: Optional[LlmChatConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or LlmChatConfig.from_env()
        saved_settings = load_chat_user_settings()
        saved_system = str(saved_settings.get("system_prompt") or "").strip()
        if saved_system:
            self._config.system_prompt = saved_system
        self._client = LlmChatClient(self._config)
        self._messages: List[Dict[str, object]] = []
        self._lake_language_memory: str = LAKE_DEFAULT_LANGUAGE_MEMORY
        self._busy = False
        self._history_id: str = ""
        self._history_title: str = ""
        self._request_t0: Optional[float] = None
        self._bridge = LlmChatBridge()
        self._bridge.finished.connect(self._on_llm_finished)
        self._probe_bridge = LlmProbeBridge()
        self._probe_bridge.finished.connect(self._on_probe_finished)
        self._camera_frame_provider: Optional[
            Callable[[], Optional[Tuple[str, np.ndarray]]]
        ] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # 紧凑顶栏：预设 / 模型 / 设置 / 清空 —— 把纵向空间留给对话区
        header = QHBoxLayout()
        header.setSpacing(4)
        self.provider_combo = ImeSafeComboBox()
        self.provider_combo.addItems(list(LLM_PROVIDER_PRESETS.keys()))
        self.provider_combo.setToolTip(
            "含腾讯混元 Hy-Embodied-VLM / RxBrain（需先 bash run_hy_embodied_vlm.sh "
            "或 run_hy_rxbrain.sh 启动服务）、本地 Ollama、百炼 API"
        )
        self.provider_combo.currentTextChanged.connect(self._on_provider_preset_changed)
        header.addWidget(self.provider_combo, 1)
        self.model_edit = QLineEdit(self._config.model)
        self.model_edit.setPlaceholderText("模型名")
        self.model_edit.setToolTip(
            "模型名：hy_a3b (VLM) / hy-rxbrain / qwen3-vl:4b / gpt-4o-mini"
        )
        self.model_edit.setMinimumWidth(90)
        self.model_edit.setMaximumWidth(160)
        header.addWidget(self.model_edit)
        self.settings_toggle_btn = QToolButton()
        self.settings_toggle_btn.setText("设置")
        self.settings_toggle_btn.setCheckable(True)
        self.settings_toggle_btn.setChecked(False)
        self.settings_toggle_btn.setToolTip("展开/收起 API、Key 与 System prompt 设置")
        self.settings_toggle_btn.toggled.connect(self._on_settings_toggled)
        header.addWidget(self.settings_toggle_btn)
        self.probe_btn = QPushButton("探测")
        self.probe_btn.setFixedWidth(44)
        self.probe_btn.setToolTip(
            "探测当前 API 地址上的服务类型与可用模型\n"
            "（/health、/models、Ollama /api/tags）"
        )
        self.probe_btn.clicked.connect(self._on_probe_clicked)
        header.addWidget(self.probe_btn)
        self.save_btn = QPushButton("保存")
        self.save_btn.setFixedWidth(44)
        self.save_btn.setToolTip("保存当前对话到本地历史（eai/chat_history）")
        self.save_btn.clicked.connect(self._on_save_chat_clicked)
        header.addWidget(self.save_btn)
        self.history_btn = QPushButton("历史")
        self.history_btn.setFixedWidth(44)
        self.history_btn.setToolTip(
            "加载 / 修改标题 / 删除已保存的对话（加载后问题进输入框，点发送再请求）"
        )
        self.history_btn.clicked.connect(self._on_history_clicked)
        header.addWidget(self.history_btn)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setFixedWidth(44)
        self.clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        self.history_title_label = QLabel("当前: 新对话（未保存）")
        self.history_title_label.setStyleSheet(f"color: {UI_TEXT_MUTED};")
        self.history_title_label.setWordWrap(True)
        layout.addWidget(self.history_title_label)

        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(3)
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("API"))
        self.api_base_edit = QLineEdit(self._config.api_base)
        self.api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        self.api_base_edit.setToolTip(
            "OpenAI 兼容 API：Ollama :11434/v1；Hy-VLM :8080/v1；RxBrain :8090/v1"
        )
        settings_row.addWidget(self.api_base_edit, stretch=1)
        settings_layout.addLayout(settings_row)
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Key"))
        self.api_key_edit = QLineEdit(self._config.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText(f"或设置 ${LLM_API_KEY_ENV}")
        self.api_key_edit.setToolTip(
            f"API Key；Ollama/Hy-Embodied 本地可填 ollama 或 EMPTY。"
            f"也可 export {LLM_API_KEY_ENV}=..."
        )
        key_row.addWidget(self.api_key_edit, stretch=1)
        settings_layout.addLayout(key_row)
        system_label = QLabel("System")
        system_label.setToolTip("Lake / Qwen 推理时作为 system 角色发送；可编辑并保存到本地")
        settings_layout.addWidget(system_label)
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlainText(self._config.system_prompt)
        self.system_prompt_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.system_prompt_edit.setPlaceholderText("System prompt…")
        self.system_prompt_edit.setMinimumHeight(72)
        self.system_prompt_edit.setMaximumHeight(140)
        self.system_prompt_edit.setToolTip(
            "认知编排器 system 提示词；Lake 模式下随请求发送。"
            "点「保存 System」写入 eai/chat_user_settings.json"
        )
        self.system_prompt_edit.setStyleSheet(
            "QTextEdit { background-color: #252525; color: #eee; border: 1px solid #555; }"
        )
        settings_layout.addWidget(self.system_prompt_edit)
        system_btn_row = QHBoxLayout()
        self.save_system_btn = QPushButton("保存 System")
        self.save_system_btn.setToolTip(
            f"保存 system prompt 到 {CHAT_USER_SETTINGS_PATH}"
        )
        self.save_system_btn.clicked.connect(self._on_save_system_prompt_clicked)
        system_btn_row.addWidget(self.save_system_btn)
        self.reset_system_btn = QPushButton("恢复默认")
        self.reset_system_btn.setToolTip(
            "恢复为 Lake 训练默认 system（技能/子任务/记忆 三行版）"
        )
        self.reset_system_btn.clicked.connect(self._on_reset_system_prompt_clicked)
        system_btn_row.addWidget(self.reset_system_btn)
        system_btn_row.addStretch(1)
        settings_layout.addLayout(system_btn_row)
        self.settings_panel.setVisible(False)
        layout.addWidget(self.settings_panel)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(4)
        self.attach_camera_check = QCheckBox("附带相机图")
        self.attach_camera_check.setChecked(True)
        self.attach_camera_check.setToolTip(
            "发送时附带当前彩色相机帧（未在测试页点选场景图时生效；"
            "Hy-Embodied / Qwen 等多模态模型需要）"
        )
        opt_row.addWidget(self.attach_camera_check)
        self.clear_image_btn = QPushButton("清除场景图")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.setToolTip("清除已在测试页点选的场景图")
        self.clear_image_btn.clicked.connect(
            lambda _checked=False: self._on_clear_chat_image_clicked()
        )
        opt_row.addWidget(self.clear_image_btn)
        self.thinking_check = QCheckBox("thinking")
        self.thinking_check.setToolTip(
            "Hy-Embodied-VLM：chat_template_kwargs.enable_thinking=true（慢但推理更强）"
        )
        opt_row.addWidget(self.thinking_check)
        opt_row.addStretch(1)
        self.chat_image_label = QLabel("未选场景图")
        self.chat_image_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.chat_image_label.setWordWrap(False)
        self.chat_image_label.setMinimumWidth(0)
        self.chat_image_label.setToolTip(
            "在「测试」页点击一张场景图，或勾选附带相机图"
        )
        opt_row.addWidget(self.chat_image_label, 1)
        self.chat_image_preview = QLabel()
        self.chat_image_preview.setFixedSize(56, 42)
        self.chat_image_preview.setAlignment(Qt.AlignCenter)
        self.chat_image_preview.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 1px solid #555; color: #888; }"
        )
        self.chat_image_preview.setText("预览")
        opt_row.addWidget(self.chat_image_preview)
        layout.addLayout(opt_row)

        self._chat_attach_image_bgr: Optional[np.ndarray] = None
        self._chat_attach_image_path: str = ""

        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("对话记录将显示在这里…")
        self.history_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.history_view.setMinimumHeight(180)
        self.history_view.setFocusPolicy(Qt.NoFocus)
        self.history_view.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #ddd; border: 1px solid #444; }"
        )
        layout.addWidget(self.history_view, stretch=1)

        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self._input_row = input_row
        self.input_edit = ChatInputEdit()
        self.input_edit._chat_panel = self
        self.input_edit.setPlaceholderText(
            "输入高层任务名（如：合上后盖并拧紧）；Enter 发送"
        )
        self.input_edit.setFixedHeight(96)
        self.input_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.input_edit.setStyleSheet(
            "QTextEdit { background-color: #252525; color: #eee; border: 1px solid #555; }"
        )
        self._suppress_send_until = 0.0
        self._ime_dirty_from_history = False
        self._ime_rebuilding = False
        self._history_dialog = None
        input_row.addWidget(self.input_edit, stretch=1)
        send_col = QVBoxLayout()
        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setToolTip("调用大模型获取回复")
        self.send_btn.setFocusPolicy(Qt.NoFocus)
        self.send_btn.clicked.connect(self._on_send_clicked)
        send_col.addWidget(self.send_btn)
        send_col.addStretch()
        input_row.addLayout(send_col)
        layout.addLayout(input_row)

        self._append_system_line(
            f"模型: {self._config.model}  |  API: {self._config.api_base}"
        )
        self._append_system_line(
            "具身模型: Hy-Embodied-VLM-1.0 / RxBrain-1.0 — "
            "见 HY_EMBODIED.md；先启动对应服务再选预设"
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

    def set_camera_frame_provider(
        self,
        provider: Optional[Callable[[], Optional[Tuple[str, np.ndarray]]]],
    ) -> None:
        self._camera_frame_provider = provider

    def _on_settings_toggled(self, checked: bool) -> None:
        self.settings_panel.setVisible(bool(checked))
        self.settings_toggle_btn.setText("收起" if checked else "设置")

    def _on_probe_clicked(self) -> None:
        if self._busy:
            return
        self._sync_config_from_ui()
        self._set_busy(True)
        self.probe_btn.setText("…")
        self._append_system_line(
            f"正在探测模型服务: {self._config.api_base}  model={self._config.model}"
        )
        self.status_message.emit("正在探测连接的模型…")

        def _work() -> None:
            try:
                result = self._client.probe(timeout_s=6.0)
            except Exception as exc:
                result = {
                    "ok": False,
                    "kind": "探测失败",
                    "api_base": self._config.api_base,
                    "configured_model": self._config.model,
                    "models": [],
                    "details": [],
                    "error": str(exc),
                }
            self._probe_bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def _on_probe_finished(self, result: object) -> None:
        self._set_busy(False)
        self.probe_btn.setText("探测")
        data = result if isinstance(result, dict) else {}
        ok = bool(data.get("ok"))
        kind = str(data.get("kind") or "未知")
        models = [str(m) for m in (data.get("models") or []) if str(m).strip()]
        configured = str(data.get("configured_model") or "")
        matched = bool(data.get("configured_model_matched"))
        details = [str(x) for x in (data.get("details") or [])]
        err = str(data.get("error") or "")

        lines = [
            f"服务类型: {kind}",
            f"API: {data.get('api_base') or self._config.api_base}",
        ]
        if configured:
            mark = "✓ 匹配" if matched else ("未在列表中确认" if models else "待确认")
            lines.append(f"当前配置模型: {configured}（{mark}）")
        if models:
            show = ", ".join(models[:8])
            if len(models) > 8:
                show += f" …共{len(models)}个"
            lines.append(f"可用模型: {show}")
        else:
            lines.append("可用模型: （未列出）")
        for d in details[:6]:
            lines.append(f"- {d}")
        if err:
            lines.append(f"错误: {err}")

        summary = "\n".join(lines)
        if ok:
            self._append_system_line("模型探测完成:\n" + summary)
            self.status_message.emit(f"探测完成: {kind}")
            # 若只发现一个模型且与配置不同，提示可切换
            if len(models) == 1 and configured and models[0] != configured:
                reply = QMessageBox.question(
                    self,
                    "探测模型",
                    f"检测到服务类型: {kind}\n"
                    f"可用模型: {models[0]}\n"
                    f"当前配置: {configured}\n\n"
                    "是否把模型名切换为检测到的模型？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self.model_edit.setText(models[0])
                    self._append_system_line(f"已切换模型名为: {models[0]}")
            else:
                QMessageBox.information(self, "探测模型", summary)
        else:
            self._append_error_line("模型探测失败:\n" + summary)
            self.status_message.emit(f"探测失败: {err or kind}")
            QMessageBox.warning(self, "探测模型", summary)

    def apply_local_ollama_preset(self, silent: bool = False) -> None:
        """切换到本地 Ollama 预设（供「本地部署 AI」按钮调用）。"""
        name = "本地 Ollama · Qwen3.5-4B"
        idx = self.provider_combo.findText(name)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self._apply_provider_preset(name, silent=silent)
        if not silent:
            self._append_system_line(f"已切换对话 API 为 {OLLAMA_API_BASE_DEFAULT}")

    def apply_local_qwen_service_preset(
        self,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        silent: bool = False,
    ) -> None:
        """切换到本地 Qwen 推理服务预设（model id 随已部署模型变化）。"""
        name = LOCAL_QWEN_CHAT_PRESET_NAME
        self.provider_combo.blockSignals(True)
        try:
            idx = self.provider_combo.findText(name)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        finally:
            self.provider_combo.blockSignals(False)
        base = (api_base or resolve_local_qwen_viewer_api_base()).rstrip("/")
        self.api_base_edit.setText(base)
        mid = (model_id or "").strip()
        if not mid:
            info = fetch_local_qwen_server_info(base) or {}
            mid = str(info.get("model") or "").strip()
        if not mid:
            mid = LOCAL_QWEN_MODEL_ID
        self.model_edit.setText(mid)
        if not self.api_key_edit.text().strip():
            self.api_key_edit.setText("EMPTY")
        self.attach_camera_check.setChecked(True)
        self.thinking_check.setChecked(False)
        self.thinking_check.setEnabled(False)
        if not silent:
            self._append_system_line(
                f"已切换对话 API 为本地 Qwen 服务 {base}  model={mid}"
            )

    def apply_remote_qwen_service_preset(
        self,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        silent: bool = False,
    ) -> None:
        """切换到远程 Qwen（SSH 隧道）推理服务预设。"""
        name = REMOTE_QWEN_CHAT_PRESET_NAME
        self.provider_combo.blockSignals(True)
        try:
            idx = self.provider_combo.findText(name)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        finally:
            self.provider_combo.blockSignals(False)
        base = (api_base or REMOTE_QWEN_API_BASE_DEFAULT).rstrip("/")
        self.api_base_edit.setText(base)
        mid = (model_id or "").strip()
        if not mid:
            info = fetch_local_qwen_server_info(base) or {}
            mid = str(info.get("model") or "").strip()
        if not mid:
            mid = "qwen3.5-35b-a3b"
        self.model_edit.setText(mid)
        if not self.api_key_edit.text().strip():
            self.api_key_edit.setText("EMPTY")
        self.attach_camera_check.setChecked(True)
        self.thinking_check.setChecked(False)
        self.thinking_check.setEnabled(False)
        if not silent:
            self._append_system_line(
                f"已切换对话 API 为远程 Qwen 服务 {base}  model={mid}"
            )

    def _apply_provider_preset(self, name: str, silent: bool = False) -> None:
        preset = LLM_PROVIDER_PRESETS.get(name)
        if preset is None or name == "自定义":
            return
        api_base, model, default_key = preset
        if api_base:
            self.api_base_edit.setText(api_base)
        if name == LOCAL_QWEN_CHAT_PRESET_NAME:
            info = fetch_local_qwen_server_info(api_base) or {}
            live_model = str(info.get("model") or "").strip()
            self.model_edit.setText(live_model or model)
        elif name == REMOTE_QWEN_CHAT_PRESET_NAME:
            info = fetch_local_qwen_server_info(api_base) or {}
            live_model = str(info.get("model") or "").strip()
            self.model_edit.setText(live_model or model)
        elif model:
            self.model_edit.setText(model)
        if default_key and not self.api_key_edit.text().strip():
            self.api_key_edit.setText(default_key)
        vision = name in LLM_VISION_PROVIDER_NAMES
        thinking = name in LLM_THINKING_PROVIDER_NAMES
        self.attach_camera_check.setEnabled(True)
        if vision:
            self.attach_camera_check.setChecked(True)
        self.thinking_check.setEnabled(thinking)
        if not thinking:
            self.thinking_check.setChecked(False)
        if not silent:
            hint = ""
            if name == "Hy-Embodied-VLM-1.0":
                hint = "（需: bash run_hy_embodied_vlm.sh，约 4×80GB GPU）"
            elif name == "Hy-Embodied-RxBrain-1.0":
                hint = "（需: bash run_hy_rxbrain.sh，约 1×GPU + 权重）"
            elif name == LOCAL_QWEN_CHAT_PRESET_NAME:
                hint = "（需: 测试 Tab 选择模型并「启动推理服务」）"
            elif name == REMOTE_QWEN_CHAT_PRESET_NAME:
                hosts = " / ".join(h for h, _ in REMOTE_QWEN_HOSTS)
                hint = f"（需: 测试 Tab 部署位置选远程，如 {hosts}）"
            self._append_system_line(f"已切换预设: {name}{hint}")

    def _on_provider_preset_changed(self, name: str) -> None:
        self._apply_provider_preset(name)

    def set_attached_image_from_path(
        self, path: str, display_name: str = ""
    ) -> bool:
        """用测试页场景图路径设置为对话附图；成功返回 True。"""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None or np.asarray(image).size == 0:
            self._append_error_line(f"无法读取场景图: {path}")
            return False
        self._chat_attach_image_bgr = image
        self._chat_attach_image_path = path
        self.clear_image_btn.setEnabled(not self._busy)
        name = display_name or os.path.basename(path)
        h, w = image.shape[:2]
        self.chat_image_label.setText(f"已选场景: {name}  ({w}×{h})")
        self.chat_image_label.setToolTip(path)
        pix = cv2_to_qpixmap(image)
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                self.chat_image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.chat_image_preview.setPixmap(scaled)
            self.chat_image_preview.setText("")
        self._append_system_line(f"已点选场景图: {name}")
        self.attached_image_changed.emit(path)
        return True

    def _on_clear_chat_image_clicked(self, *, notify: bool = True) -> None:
        had_image = self._chat_attach_image_bgr is not None
        self._chat_attach_image_bgr = None
        self._chat_attach_image_path = ""
        self.clear_image_btn.setEnabled(False)
        self.chat_image_label.setText(
            "未选场景图（测试页点选，或勾选附带相机图）"
        )
        self.chat_image_label.setToolTip("")
        self.chat_image_preview.clear()
        self.chat_image_preview.setText("预览")
        if had_image:
            self.attached_image_changed.emit("")
            if notify:
                self._append_system_line("已清除所选场景图")

    def _rebuild_chat_input(self, text: Optional[str] = None) -> "ChatInputEdit":
        """替换输入框，强制 fcitx 重新挂 IC。"""
        global _IME_LAST_TEXT_WIDGET
        if getattr(self, "_ime_rebuilding", False):
            return self.input_edit
        self._ime_rebuilding = True
        try:
            old = self.input_edit
            if text is None:
                text = old.toPlainText()
            placeholder = old.placeholderText()
            style = old.styleSheet()
            height = old.height()
            new = ChatInputEdit(self)
            new._chat_panel = self
            new.setPlaceholderText(placeholder)
            new.setFixedHeight(height if height > 0 else 96)
            new.setAttribute(Qt.WA_InputMethodEnabled, True)
            new.setStyleSheet(style)
            new.setPlainText(text)
            cursor = new.textCursor()
            cursor.movePosition(cursor.End)
            new.setTextCursor(cursor)
            row = getattr(self, "_input_row", None)
            if row is not None:
                row.replaceWidget(old, new)
            old.deleteLater()
            self.input_edit = new
            self._ime_dirty_from_history = False
            _IME_LAST_TEXT_WIDGET = new
            new.setFocus(Qt.OtherFocusReason)
            return new
        finally:
            self._ime_rebuilding = False

    def _focus_chat_input(self) -> None:
        self.input_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.input_edit.setFocus(Qt.OtherFocusReason)

    def _sync_config_from_ui(self) -> None:
        self._config.api_base = self.api_base_edit.text().strip() or LLM_API_BASE_DEFAULT
        self._config.model = self.model_edit.text().strip() or LLM_MODEL_DEFAULT
        key = self.api_key_edit.text().strip()
        if not key:
            key = os.environ.get(LLM_API_KEY_ENV, "").strip()
        self._config.api_key = key
        self._config.enable_thinking = bool(self.thinking_check.isChecked())
        system = self.system_prompt_edit.toPlainText().strip()
        self._config.system_prompt = system or LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        self._client = LlmChatClient(self._config)

    def _on_save_system_prompt_clicked(self) -> None:
        self._sync_config_from_ui()
        try:
            path = save_chat_user_settings(
                {"system_prompt": self._config.system_prompt}
            )
        except Exception as exc:
            QMessageBox.warning(self, "保存 System", f"保存失败: {exc}")
            return
        preview = self._config.system_prompt.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "…"
        aligned = lake_user_output_instruction(self._config.system_prompt)
        self._append_system_line(f"已保存 System prompt: {preview}")
        self._append_system_line(f"User 输出要求已对齐: {aligned}")
        self.status_message.emit(f"System prompt 已保存: {path}")

    def _on_reset_system_prompt_clicked(self) -> None:
        self.system_prompt_edit.setPlainText(LAKE_ORCHESTRATOR_SYSTEM_PROMPT_TRAINING)
        self._sync_config_from_ui()
        self._append_system_line(
            "已恢复训练默认 System（三行版）；user 将自动对齐为「技能/子任务/记忆」"
        )

    def _append_system_line(self, text: str) -> None:
        self.history_view.append(f'<span style="color:{UI_TEXT_MUTED};">[系统] {text}</span>')

    def _append_prompt_dump(self, messages: List[Dict[str, object]]) -> None:
        """在对话框中打印即将发送的全部提示词。"""
        has_image = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and str(part.get("type") or "") in (
                        "image_url",
                        "image",
                    ):
                        has_image = True
                        break
            if has_image:
                break
        timeout_s = (
            LLM_CHAT_VISION_TIMEOUT_S if has_image else LLM_CHAT_TIMEOUT_S
        )
        dump = format_chat_prompt_dump(
            messages,
            api_base=self._config.api_base,
            model=self._config.model,
            temperature=0.2,
            max_tokens=int(self._config.max_tokens),
            enable_thinking=bool(self._config.enable_thinking),
            timeout_s=timeout_s,
        )
        safe = _html_escape(dump).replace("\n", "<br>")
        self.history_view.append(
            f'<pre style="margin:8px 0; padding:8px; white-space:pre-wrap; '
            f'word-wrap:break-word; color:{UI_TEXT_MUTED}; '
            f'background:rgba(127,127,127,0.12); border-radius:6px; '
            f'font-size:11px; font-family:monospace;">{safe}</pre>'
        )

    def _meta_span(self, ts: Optional[str] = None, latency_s: Optional[float] = None) -> str:
        parts = [format_chat_timestamp(ts)]
        lat = format_chat_latency(latency_s)
        if lat:
            parts.append(f"耗时 {lat}")
        meta = " · ".join(parts)
        return (
            f'<span style="color:{UI_TEXT_MUTED}; font-size:11px; margin-left:8px;">'
            f"{_html_escape(meta)}</span>"
        )

    def _append_user_line(self, text: str, ts: Optional[str] = None) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;">'
            f'<b style="color:#7ec8ff;">你:</b>{self._meta_span(ts)}'
            f"<br>{safe}</p>"
        )

    def _append_assistant_line(
        self,
        text: str,
        ts: Optional[str] = None,
        latency_s: Optional[float] = None,
    ) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;">'
            f'<b style="color:#50fa7b;">AI:</b>{self._meta_span(ts, latency_s)}'
            f"<br>{safe}</p>"
        )

    def _append_error_line(
        self,
        text: str,
        ts: Optional[str] = None,
        latency_s: Optional[float] = None,
    ) -> None:
        safe = _html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0;">'
            f'<b style="color:#ff5555;">错误:</b>{self._meta_span(ts, latency_s)}'
            f"<br>{safe}</p>"
        )

    def _trim_history(self) -> None:
        # 只保留最近的 user/assistant 轮次
        keep = [
            m
            for m in self._messages
            if str(m.get("role") or "") in ("user", "assistant")
        ]
        if len(keep) > LLM_CHAT_MAX_HISTORY:
            keep = keep[-LLM_CHAT_MAX_HISTORY:]
        self._messages = keep

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_btn.setEnabled(not busy)
        self.input_edit.setReadOnly(busy)
        self.save_btn.setEnabled(not busy)
        self.history_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.probe_btn.setEnabled(not busy)
        self.clear_image_btn.setEnabled(
            (not busy) and self._chat_attach_image_bgr is not None
        )
        self.send_btn.setText("思考中…" if busy else "发送")
        if not busy:
            QTimer.singleShot(0, self._focus_chat_input)

    def _refresh_history_title_label(self) -> None:
        if self._history_id and self._history_title:
            self.history_title_label.setText(
                f"当前: {self._history_title}  (已保存)"
            )
        elif self._history_title:
            self.history_title_label.setText(f"当前: {self._history_title}")
        else:
            self.history_title_label.setText("当前: 新对话（未保存）")

    def _conversation_message_count(self) -> int:
        return sum(
            1
            for m in self._messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and str(m.get("content") or "").strip()
        )

    def _render_messages_to_view(self) -> None:
        self.history_view.clear()
        for msg in self._messages:
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "system":
                continue
            ts = str(msg.get("ts") or "") or None
            latency_raw = msg.get("latency_s")
            latency_s: Optional[float] = None
            if isinstance(latency_raw, (int, float)):
                latency_s = float(latency_raw)
            if role == "user":
                self._append_user_line(content, ts=ts)
            elif role == "assistant":
                self._append_assistant_line(content, ts=ts, latency_s=latency_s)

    def _on_save_chat_clicked(self) -> None:
        if self._conversation_message_count() == 0:
            QMessageBox.information(self, "保存对话", "当前没有可保存的对话内容")
            return
        self._sync_config_from_ui()
        title = (self._history_title or default_chat_history_title(self._messages)).strip()
        if not title:
            title = time.strftime("对话 %Y-%m-%d %H:%M")
        record: Dict[str, object] = {
            "id": self._history_id or uuid.uuid4().hex[:12],
            "title": title,
            "created_at": "",
            "model": self._config.model,
            "api_base": self._config.api_base,
            "system_prompt": self._config.system_prompt,
            "messages": serialize_chat_messages(self._messages),
        }
        if self._history_id:
            try:
                old = load_chat_history(self._history_id)
                record["created_at"] = str(old.get("created_at") or "")
            except Exception:
                pass
        try:
            path = save_chat_history_record(record)
        except Exception as exc:
            QMessageBox.warning(self, "保存对话", f"保存失败: {exc}")
            return
        self._history_id = str(record["id"])
        self._history_title = title
        self._refresh_history_title_label()
        self._append_system_line(f"已保存对话: {title}")
        self.status_message.emit(f"对话已保存: {path}")

    def _on_history_clicked(self) -> None:
        existing = getattr(self, "_history_dialog", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.title_edit.setFocus(Qt.OtherFocusReason)
                    return
                existing.deleteLater()
            except Exception:
                pass
        dlg = ChatHistoryDialog(self)
        self._history_dialog = dlg

        def _on_finished(result: int) -> None:
            self._history_dialog = None
            accepted = result == int(QDialog.Accepted)
            hid = dlg.selected_id if accepted else ""
            dlg.deleteLater()
            if accepted and hid:
                self._load_history_by_id(hid)
                return
            if not accepted and self._history_id:
                try:
                    load_chat_history(self._history_id)
                except Exception:
                    self._history_id = ""
                    self._history_title = ""
                    self._refresh_history_title_label()
            self._focus_chat_input()

        dlg.finished.connect(_on_finished)
        dlg.show()
        dlg.raise_()
        QTimer.singleShot(0, lambda: dlg.title_edit.setFocus(Qt.OtherFocusReason))

    def _load_history_by_id(self, hid: str) -> None:
        hid = str(hid or "").strip()
        if not hid:
            return
        try:
            data = load_chat_history(hid)
        except Exception as exc:
            QMessageBox.warning(self, "历史对话", f"加载失败: {exc}")
            self._focus_chat_input()
            return
        raw_msgs = data.get("messages") or []
        messages: List[Dict[str, object]] = []
        if isinstance(raw_msgs, list):
            for msg in raw_msgs:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "")
                content = msg.get("content")
                if role not in ("user", "assistant") or not isinstance(content, str):
                    continue
                item: Dict[str, object] = {"role": role, "content": content}
                ts = msg.get("ts")
                if isinstance(ts, str) and ts.strip():
                    item["ts"] = ts.strip()
                latency = msg.get("latency_s")
                if isinstance(latency, (int, float)):
                    item["latency_s"] = float(latency)
                messages.append(item)
        draft = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            content = messages[i].get("content")
            if isinstance(content, str) and content.strip():
                draft = content
                messages = messages[:i]
            break
        self._messages = messages
        self._trim_history()
        self._lake_language_memory = LAKE_DEFAULT_LANGUAGE_MEMORY
        for m in reversed(messages):
            if m.get("role") != "assistant":
                continue
            mem = extract_lake_memory_from_assistant(str(m.get("content") or ""))
            if mem:
                self._lake_language_memory = mem
                break
        self._history_id = str(data.get("id") or hid)
        self._history_title = str(data.get("title") or self._history_id)
        saved_system = str(data.get("system_prompt") or "").strip()
        if saved_system:
            self.system_prompt_edit.setPlainText(saved_system)
            self._config.system_prompt = saved_system
        self._on_clear_chat_image_clicked(notify=False)
        self._render_messages_to_view()
        self._refresh_history_title_label()
        self._suppress_send_until = time.monotonic() + 0.4
        self.input_edit.setPlainText(draft)
        cursor = self.input_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.input_edit.setTextCursor(cursor)
        self._focus_chat_input()
        tip = "已放入输入框，点「发送」后再请求" if draft else "已加载（无用户消息可编辑）"
        self._append_system_line(f"已加载历史对话: {self._history_title} — {tip}")
        self.status_message.emit(f"已加载历史对话: {self._history_title}（待发送）")

    def _clear_chat(self) -> None:
        self._messages = []
        self._lake_language_memory = LAKE_DEFAULT_LANGUAGE_MEMORY
        self.history_view.clear()
        self._history_id = ""
        self._history_title = ""
        self._refresh_history_title_label()
        self._on_clear_chat_image_clicked(notify=False)
        self._append_system_line("对话已清空")

    def _build_api_messages(self, user_text: str) -> List[Dict[str, object]]:
        """历史保持纯文本；当前轮可附带本地选图或相机 JPEG（vision）。

        Lake / 本地远程 Qwen：对齐训练格式
          system + user(任务 / 语言记忆 / 输出要求) + image
        """
        use_lake = should_use_lake_orchestrator_prompt(
            self._config.api_base, self._config.model
        )
        prompt_text = (
            format_lake_user_prompt(
                user_text,
                self._lake_language_memory,
                system_prompt=self._config.system_prompt,
            )
            if use_lake
            else user_text
        )
        if use_lake:
            self._append_system_line(
                "已按 Lake 格式包装提示词（system 与 user 输出要求已对齐）"
            )

        api_messages: List[Dict[str, object]] = []
        system_text = self._config.system_prompt.strip()
        if use_lake and system_text:
            api_messages.append({"role": "system", "content": system_text})
        elif not use_lake:
            api_messages = [
                m
                for m in self._messages[:-1]
                if str(m.get("role") or "") in ("user", "assistant")
            ]

        content: object = prompt_text
        image_bgr: Optional[np.ndarray] = None
        image_tag = ""

        if self._chat_attach_image_bgr is not None:
            image_bgr = self._chat_attach_image_bgr
            image_tag = self._chat_attach_image_path or "本地图片"
        elif (
            self.attach_camera_check.isChecked()
            and self._camera_frame_provider is not None
        ):
            frame = self._camera_frame_provider()
            if frame is not None:
                topic, image_bgr = frame
                image_tag = topic
            else:
                self._append_system_line("未获取到相机帧，仅发送文本")

        if image_bgr is not None:
            try:
                b64 = encode_bgr_image_jpeg_b64(image_bgr)
                # 训练数据为 type=image；OpenAI 兼容接口用 image_url
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ]
                self._append_system_line(f"已附带图片: {image_tag}")
            except Exception as exc:
                self._append_system_line(f"附带图片失败，仅发送文本: {exc}")

        api_messages.append({"role": "user", "content": content})
        return api_messages

    def _on_send_clicked(self) -> None:
        if self._busy:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self._sync_config_from_ui()
        self.input_edit.clear()
        send_ts = _utc_now_iso()
        self._request_t0 = time.perf_counter()
        self._append_user_line(text, ts=send_ts)
        self._messages.append({"role": "user", "content": text, "ts": send_ts})
        self._trim_history()
        api_messages = self._build_api_messages(text)
        self._append_prompt_dump(api_messages)
        self._set_busy(True)
        self.status_message.emit("正在调用大模型…")

        def _work() -> None:
            try:
                reply = self._client.chat(api_messages)
                self._bridge.finished.emit(reply, True)
            except Exception as exc:
                self._bridge.finished.emit(str(exc), False)

        threading.Thread(target=_work, daemon=True).start()

    def _on_llm_finished(self, text: str, ok: bool) -> None:
        self._set_busy(False)
        recv_ts = _utc_now_iso()
        latency_s: Optional[float] = None
        if self._request_t0 is not None:
            latency_s = max(0.0, time.perf_counter() - self._request_t0)
        self._request_t0 = None
        if ok:
            self._append_assistant_line(text, ts=recv_ts, latency_s=latency_s)
            mem = extract_lake_memory_from_assistant(text)
            if mem:
                self._lake_language_memory = mem
            msg: Dict[str, object] = {
                "role": "assistant",
                "content": text,
                "ts": recv_ts,
            }
            if latency_s is not None:
                msg["latency_s"] = float(latency_s)
            self._messages.append(msg)
            self._trim_history()
            lat = format_chat_latency(latency_s)
            self.status_message.emit(
                f"大模型回复完成" + (f"（耗时 {lat}）" if lat else "")
            )
        else:
            self._append_error_line(text, ts=recv_ts, latency_s=latency_s)
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            lat = format_chat_latency(latency_s)
            self.status_message.emit(
                f"大模型调用失败" + (f"（耗时 {lat}）: " if lat else ": ") + text[:80]
            )


class RosBridge(QObject):
    frame_updated = pyqtSignal(str, object)
    topics_updated = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    frame_stats = pyqtSignal(str, int, float)
    robot_state_updated = pyqtSignal()
    arm_enable_changed = pyqtSignal(bool)
    hand_enable_changed = pyqtSignal(bool)
    control_mode_changed = pyqtSignal(int)
    replay_state_changed = pyqtSignal(int)
    left_hand_preset_changed = pyqtSignal(bool)
    right_hand_preset_changed = pyqtSignal(bool)
    slow_motion_progress = pyqtSignal(float, str)
    slow_motion_finished = pyqtSignal(bool, str)


class PoseComputeBridge(QObject):
    """后台位姿计算完成信号（跨线程投递到 UI）。"""

    finished = pyqtSignal(object, str, int, int, object)


class DepthVizBridge(QObject):
    """后台深度预览/点云计算完成信号。"""

    finished = pyqtSignal(object)


def resolve_test_scenario_image_path(title: str) -> Optional[str]:
    """按场景名解析 images/ 下预览图路径；不存在则返回 None。"""
    candidates: List[str] = []
    override = TEST_SCENARIO_IMAGE_FILES.get(title)
    if override:
        candidates.append(override)
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidates.append(f"{title}{ext}")
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(TEST_IMAGES_DIR, name)
        if os.path.isfile(path):
            return path
    return None


class ScaledPixmapLabel(QLabel):
    """保持宽高比缩放显示静态 QPixmap；可选点击选中。"""

    clicked = pyqtSignal()

    def __init__(self, placeholder: str = "暂无图像", parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(140, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source_pixmap: Optional[QPixmap] = None
        self._selected = False
        self._clickable = False
        self._apply_frame_style()

    def set_clickable(self, enabled: bool) -> None:
        self._clickable = enabled
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_frame_style()

    def _apply_frame_style(self) -> None:
        border = "2px solid #4CAF50" if self._selected else "1px solid #555"
        self.setStyleSheet(
            f"QLabel {{ background-color: #1a1a1a; border: {border}; "
            f"color: {UI_TEXT_MUTED}; }}"
        )

    def set_source_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._source_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.clear()
            self.setText("暂无图像")
        else:
            self.setText("")
            self._refresh_scaled()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._clickable
            and event.button() == Qt.LeftButton
            and self._source_pixmap is not None
            and not self._source_pixmap.isNull()
        ):
            self.clicked.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        super().setPixmap(scaled)


class ClickableImageLabel(QLabel):
    """可点击图像，将控件坐标映射回原始像素。"""

    clicked_pixel = pyqtSignal(int, int)

    def __init__(self, placeholder: str = "等待图像...", parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(64, 48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background-color: #1e1e1e; color: {UI_TEXT_MUTED}; border: 1px solid #555;"
        )
        self.setCursor(Qt.CrossCursor)
        self._source_image: Optional[np.ndarray] = None
        self._latest_pixmap: Optional[QPixmap] = None
        self._segment_mask: Optional[np.ndarray] = None
        self._segment_centroid: Optional[Tuple[float, float]] = None
        self._segment_contact_uv: Optional[Tuple[float, float]] = None
        self._segment_seed_uv: Optional[Tuple[float, float]] = None
        self._segment_obb: Optional[np.ndarray] = None
        self._segment_intrinsics: Optional[Tuple[float, float, float, float]] = None
        self._pose_position: Optional[np.ndarray] = None
        self._pose_rotation: Optional[np.ndarray] = None
        self._pose_axis_len: Optional[float] = None

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
        seed_uv: Optional[Tuple[float, float]] = None,
        pose_position: Optional[np.ndarray] = None,
        pose_rotation: Optional[np.ndarray] = None,
        pose_axis_len: Optional[float] = None,
    ) -> None:
        self._segment_mask = mask.copy() if mask is not None else None
        self._segment_centroid = centroid_uv
        self._segment_contact_uv = contact_uv
        self._segment_seed_uv = seed_uv
        self._segment_obb = obb_corners.copy() if obb_corners is not None else None
        self._segment_intrinsics = intrinsics
        self._pose_position = (
            np.asarray(pose_position, dtype=np.float32).reshape(3).copy()
            if pose_position is not None
            else None
        )
        self._pose_rotation = (
            np.asarray(pose_rotation, dtype=np.float32).reshape(3, 3).copy()
            if pose_rotation is not None
            else None
        )
        self._pose_axis_len = pose_axis_len
        self._refresh_display()

    def clear_segment_overlay(self) -> None:
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_seed_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._pose_position = None
        self._pose_rotation = None
        self._pose_axis_len = None
        self._refresh_display()

    def set_precomposed_display(self, overlay_bgr: np.ndarray) -> None:
        """直接显示后台线程已合成好的叠加图，避免 UI 线程重复计算。"""
        # 保留为源图，这样后续帧更新前至少还能看到一次；
        # 真正持久化请用 set_segment_overlay（随视频流重绘）。
        self._source_image = overlay_bgr.copy()
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_seed_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._pose_position = None
        self._pose_rotation = None
        self._pose_axis_len = None
        self._latest_pixmap = cv2_to_qpixmap(overlay_bgr)
        self._render_pixmap()

    def _compose_display_image(self) -> np.ndarray:
        if self._source_image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        has_mask = self._segment_mask is not None and np.asarray(self._segment_mask).any()
        has_pose = self._segment_obb is not None or self._pose_position is not None
        has_seed = self._segment_seed_uv is not None
        if has_mask or has_pose or has_seed:
            return apply_segment_overlay(
                self._source_image,
                self._segment_mask,
                self._segment_centroid,
                obb_corners=self._segment_obb,
                intrinsics=self._segment_intrinsics,
                contact_uv=self._segment_contact_uv,
                seed_uv=self._segment_seed_uv,
                pose_position=self._pose_position,
                pose_rotation=self._pose_rotation,
                pose_axis_len=self._pose_axis_len,
            )
        return self._source_image

    def _refresh_display(self, pixmap: Optional[QPixmap] = None) -> None:
        if self._source_image is None:
            return
        overlay_active = (
            self._segment_mask is not None
            or self._segment_seed_uv is not None
            or self._segment_obb is not None
            or self._pose_position is not None
        )
        if pixmap is not None and not overlay_active:
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
        on_click_uv: Optional[Callable[[str, int, int], None]] = None,
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
        self._on_click_uv = on_click_uv
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
        self.title_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_TITLE, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.image_label = ClickableImageLabel("等待图像...（点击选点，再按「调用分割」确认）")
        self.image_label.clicked_pixel.connect(self._on_pixel_clicked)
        layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel(topic)
        self.info_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = QLabel("点击图像选择提示点；再点「调用分割 / SAM3 / FP」执行")
        self.coord_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setStyleSheet(f"color: {UI_ACCENT_BLUE_BRIGHT};")
        self.coord_label.setWordWrap(True)
        self.coord_label.setMaximumHeight(40)
        self.coord_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.coord_label)

        self._pose_bridge = PoseComputeBridge()
        self._pose_bridge.finished.connect(self._on_pose_compute_finished)
        self._pose_busy = False
        self._pending_seed_uv: Optional[Tuple[int, int]] = None

    def _show_seed_marker(self, u: int, v: int, hint: str) -> None:
        self._pending_seed_uv = (u, v)
        self.image_label.set_segment_overlay(None, seed_uv=(float(u), float(v)))
        self.coord_label.setText(hint)

    def invoke_segment_at_seed(
        self,
        u: Optional[int] = None,
        v: Optional[int] = None,
    ) -> str:
        """按提示点启动立体分割+位姿（由按钮触发，不自动在点击时执行）。"""
        if self._pose_busy:
            return f"[{self.topic}] 分割进行中，请稍候"
        if self._latest_image is None:
            return f"[{self.topic}] 尚无彩色图像"
        if u is None or v is None:
            if self._pending_seed_uv is None:
                return f"[{self.topic}] 请先点击图像选择提示点"
            u, v = self._pending_seed_uv
        h, w = self._latest_image.shape[:2]
        u = max(0, min(w - 1, int(u)))
        v = max(0, min(h - 1, int(v)))
        self._pending_seed_uv = (u, v)
        self.image_label.set_segment_overlay(None, seed_uv=(float(u), float(v)))

        if self._is_paired_depth_enabled is not None and not self._is_paired_depth_enabled(
            self.topic
        ):
            return f"[{self.topic}] 未勾选配对 depth，无法分割"
        depth = self._get_paired_depth(self.topic) if self._get_paired_depth else None
        depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
        if depth is None or not depth_topic or self._get_intrinsics_for_depth is None:
            return f"[{self.topic}] 无 depth 数据，无法分割"
        dh, dw = depth.shape[:2]
        fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
        u_d, v_d = scale_uv_to_shape(u, v, self._latest_image.shape[:2], (dh, dw))
        self._start_pose_compute(
            depth, self._latest_image, u_d, v_d, fx, fy, cx, cy, display_u=u
        )
        return f"[{self.topic}] 已开始分割 @ ({u}, {v})"

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
            # 用 mask 叠加（随视频流持续重绘），避免 set_precomposed 下一帧被冲掉
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
            intr = None
            if depth_topic and self._get_intrinsics_for_depth:
                fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
                # 内参按 color 尺寸缩放，便于画 OBB
                sx = float(cw) / float(max(1, dw))
                sy = float(ch) / float(max(1, dh))
                intr = (fx * sx, fy * sy, cx * sx, cy * sy)
            self.image_label.set_segment_overlay(
                display_mask,
                (float(cu), float(cv_pt)),
                obb_corners=result.obb_corners,
                intrinsics=intr,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(display_u), float(display_v)),
                pose_position=np.asarray(result.position_xyz, dtype=np.float32),
                pose_rotation=np.asarray(result.rotation_matrix, dtype=np.float32),
                pose_axis_len=float(max(result.obb_extents) * 0.55),
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

    def show_sam3_mask(self, mask: np.ndarray, seed_u: int, seed_v: int) -> None:
        if self._latest_image is None or not np.asarray(mask).any():
            self.image_label.clear_segment_overlay()
            return
        display_mask = resize_mask_to_shape(mask, self._latest_image.shape[:2])
        ys, xs = np.where(display_mask)
        if len(xs) > 0:
            cu, cv = float(np.mean(xs)), float(np.mean(ys))
        else:
            cu, cv = float(seed_u), float(seed_v)
        # 随视频流持续重绘（绿填充 + 黄轮廓 + 红星提示点）
        self.image_label.set_segment_overlay(
            display_mask,
            (cu, cv),
            seed_uv=(float(seed_u), float(seed_v)),
        )
        self.coord_label.setText(
            f"SAM3 {int(display_mask.sum())} px @ ({seed_u}, {seed_v})  "
            f"质心 ({cu:.0f}, {cv:.0f})"
        )

    def _on_pixel_clicked(self, u: int, v: int) -> None:
        if self._on_click_uv is not None:
            self._on_click_uv(self.topic, u, v)
        if self._latest_image is None:
            return
        info = format_color_pixel_info(u, v, self._latest_image)
        hint = f"{info}  |  已选提示点，请点「调用分割 / SAM3 / FP」确认"
        self._show_seed_marker(u, v, hint)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] 已选提示点 ({u}, {v})")

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
        on_click_uv: Optional[Callable[[str, int, int], None]] = None,
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
        self._on_click_uv = on_click_uv
        self._status_callback = status_callback
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._last_display_time = 0.0
        self._last_robot_overlay_time = 0.0
        self._fps = 0.0
        self._pending_seed_preview_uv: Optional[Tuple[int, int]] = None
        self._pending_seed_full_uv: Optional[Tuple[int, int]] = None
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
        self.title_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_TITLE, QFont.Bold))
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

        self.depth_preview = ClickableImageLabel("点击深度图选择提示点；再按「调用分割」确认")
        self.depth_preview.setMinimumHeight(48)
        self.depth_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.depth_preview.clicked_pixel.connect(self._on_depth_pixel_clicked)
        layout.addWidget(self.depth_preview, stretch=1)

        self.info_label = QLabel("等待深度图...")
        self.info_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = QLabel("点击深度图选择提示点；再点「调用分割 / SAM3 / FP」执行")
        self.coord_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.coord_label.setAlignment(Qt.AlignCenter)
        self.coord_label.setStyleSheet(f"color: {UI_ACCENT_BLUE_BRIGHT};")
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
        self.robot_info_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_NORMAL))
        self.robot_info_label.setWordWrap(True)
        self.robot_info_label.setTextFormat(Qt.RichText)
        self.robot_info_label.setStyleSheet(
            f"color: {UI_TEXT_PRIMARY};"
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

    def invoke_segment_at_seed(
        self,
        u_full: Optional[int] = None,
        v_full: Optional[int] = None,
    ) -> str:
        """按提示点启动立体分割+位姿（按钮触发）。"""
        if self._latest_depth is None:
            return f"[{self.topic}] 尚无深度图"
        if self._pose_busy:
            return f"[{self.topic}] 分割进行中，请稍候"
        if u_full is None or v_full is None:
            if self._pending_seed_full_uv is None:
                return f"[{self.topic}] 请先点击深度图选择提示点"
            u_full, v_full = self._pending_seed_full_uv
        dh, dw = self._depth_full_shape
        u_full = max(0, min(dw - 1, int(u_full)))
        v_full = max(0, min(dh - 1, int(v_full)))
        preview_u, preview_v = scale_uv_to_shape(
            u_full, v_full, (dh, dw), self._preview_shape
        )
        self._pending_seed_full_uv = (u_full, v_full)
        self._pending_seed_preview_uv = (preview_u, preview_v)
        self.depth_preview.set_segment_overlay(
            None, seed_uv=(float(preview_u), float(preview_v))
        )
        self._start_pose_compute(u_full, v_full, preview_u, preview_v)
        return f"[{self.topic}] 已开始分割 @ depth({u_full}, {v_full})"

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
        if self._latest_depth is None:
            return
        dh, dw = self._depth_full_shape
        u_full, v_full = scale_uv_to_shape(u, v, self._preview_shape, (dh, dw))
        self._pending_seed_preview_uv = (u, v)
        self._pending_seed_full_uv = (u_full, v_full)
        self.depth_preview.set_segment_overlay(None, seed_uv=(float(u), float(v)))
        self.coord_label.setText(
            f"提示点 preview({u}, {v}) / depth({u_full}, {v_full})  |  "
            "请点「调用分割 / SAM3 / FP」确认"
        )
        if self._on_click_uv is not None:
            # 将深度点击映射到配对彩色图坐标，供 SAM3/FP 提示点使用
            color_u, color_v = u_full, v_full
            color_topic = self.topic
            if self._get_paired_color is not None:
                color = self._get_paired_color(self.topic)
                if color is not None:
                    color_u, color_v = scale_uv_to_shape(
                        u_full, v_full, (dh, dw), color.shape[:2]
                    )
            self._on_click_uv(color_topic, int(color_u), int(color_v))
        if self._status_callback:
            self._status_callback(
                f"[{self.topic}] 已选提示点 depth({u_full}, {v_full})"
            )

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
                seed_uv=(float(preview_u), float(preview_v)),
                pose_position=np.asarray(result.position_xyz, dtype=np.float32),
                pose_rotation=np.asarray(result.rotation_matrix, dtype=np.float32),
                pose_axis_len=float(max(result.obb_extents) * 0.55),
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
    on_click_uv: Optional[Callable[[str, int, int], None]] = None,
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
            on_click_uv=on_click_uv,
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
        on_click_uv=on_click_uv,
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
        # A2D: image frame_id 是 camera_frame，但默认 TF 树没有它；自动补发 base_link->camera_frame
        self._head_camera_tf = None
        if attach_head_camera_tf is not None:
            self._head_camera_tf = attach_head_camera_tf(self)
        self._left_hand_cmd_pub = self.create_publisher(
            JointState,
            ROBOT_LEFT_HAND_CMD_TOPIC,
            10,
        )
        self._right_hand_cmd_pub = self.create_publisher(
            JointState,
            ROBOT_RIGHT_HAND_CMD_TOPIC,
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
        self._right_hand_at_a = True
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
        self._replay_prep_mode_timer = None
        self._slow_motion_saved_control_mode: Optional[int] = None
        self._slow_motion_moving_side = "left"
        # 移动时保持另一侧手臂的位姿（IK 双臂目标需要同时发布）
        self._slow_motion_hold_pose: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
        ] = None
        self._slow_motion_final_moving_pose: Optional[
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
            self._stop_replay_prep_mode_timer()
            self._restore_control_mode_if_needed()
        if self._scan_timer is not None:
            try:
                self.destroy_timer(self._scan_timer)
            except Exception:
                pass
            self._scan_timer = None
        self._stop_replay_prep_mode_timer()
        self._tf_listener = None

    def is_left_hand_at_a(self) -> bool:
        return self._left_hand_at_a

    def is_right_hand_at_a(self) -> bool:
        return self._right_hand_at_a

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
            return "手: 等待 /hand/enable_state ..."
        return f"手: {'已使能' if self._hand_enabled else '未使能'}"

    def get_control_mode_label(self) -> str:
        if not self._control_mode_received_at:
            return "mode: 等待 /control_mode ..."
        return f"mode: {self._control_mode}"

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
        """回放前切 model 模式并请求手/臂使能；需 pedal_controller 已运行才会收到 enable_state。"""
        self._start_replay_prep_mode_lock()
        return self.get_replay_prerequisite_warnings()

    def get_replay_prerequisite_warnings(self) -> List[str]:
        warnings: List[str] = []
        if not _is_stack_services_running():
            warnings.append(
                "手/臂服务栈未运行（需 pedal_controller / topic_router，请先点「启动机器人栈」）"
            )
        if not self._control_mode_received_at:
            warnings.append("未收到 /control_mode（等待 pedal_controller 或 viewer 发布）")
        elif self._control_mode != MODEL_CONTROL_MODE:
            warnings.append(f"control_mode={self._control_mode}，需为 {MODEL_CONTROL_MODE}")
        if not self._arm_enable_received_at:
            warnings.append("未收到 /arm/enable_state（请使能手臂）")
        elif not self._arm_enabled:
            warnings.append("手臂未使能")
        if not self._hand_enable_received_at:
            warnings.append("未收到 /hand/enable_state（请使能手部）")
        elif not self._hand_enabled:
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

    def _format_arm_tcp_current_line(self, side: str) -> str:
        state = self.get_robot_state()
        side_name = arm_side_label(side)
        tcp = self._tcp_pose_in_ik_frame(side)
        if tcp is not None:
            xyz, quat = tcp
            return format_xyz_rpy_line(side_name, xyz, quat)
        raw = state.left_tcp if side == "left" else state.right_tcp
        if raw is not None and raw.valid:
            frame = raw.frame_id or BASE_LINK_FRAME
            return f"{format_xyz_rpy_line(side_name, raw.xyz, raw.quat_xyzw)}  [{frame}]"
        return f"{side_name} TCP: (无数据)"

    def get_arm_move_current_label(self, arm_side: str = "left") -> str:
        state = self.get_robot_state()
        hand_line = format_left_hand_state_line(state)
        return f"当前  {hand_line}  |  {self._format_arm_tcp_current_line(arm_side)}"

    def get_arm_move_current_label_both(self) -> str:
        return (
            f"当前  左: {self._format_arm_tcp_current_line('left')}  |  "
            f"右: {self._format_arm_tcp_current_line('right')}"
        )

    def resolve_segment_move_goal(
        self,
        segment: SegmentPoseTarget,
        timeout_s: float = UI_TF_LOOKUP_TIMEOUT_S,
        arm_side: str = "left",
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
        if segment.keep_tcp_orientation:
            tcp = self._tcp_pose_in_ik_frame(arm_side, timeout_s=timeout_s)
            if tcp is None:
                return None
            _, quat = tcp
        kind = "FoundationPose" if "foundationpose" in (segment.label or "").lower() else "分割位姿"
        orient_note = "位置绝对·姿态保持" if segment.keep_tcp_orientation else ""
        side_name = arm_side_label(arm_side)
        label = f"{side_name}{kind} ({segment.label or segment.source_topic})"
        if orient_note:
            label = f"{label} [{orient_note}]"
        return ResolvedArmMoveGoal(
            position_xyz=xyz,
            quaternion_xyzw=quat,
            label=label,
            arm_side=arm_side,
        )

    def get_arm_move_pose_labels(
        self,
        segment: Optional[SegmentPoseTarget],
        arm_side: str = "left",
    ) -> Tuple[str, str]:
        """保留兼容：仅在没有 window 侧解析时使用。"""
        side_name = arm_side_label(arm_side)
        current_line = self.get_arm_move_current_label(arm_side)
        target_line = f"目标  {side_name} TCP: (请先设置)"
        if segment is not None:
            resolved = self.resolve_segment_move_goal(segment, arm_side=arm_side)
            if resolved is not None:
                target_line = (
                    f"目标  [{resolved.label}]  "
                    f"{format_xyz_rpy_line(side_name, resolved.position_xyz, resolved.quaternion_xyzw)}"
                )
            else:
                target_line = (
                    f"目标  {side_name} TCP: TF 不可用 "
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

    def request_model_control_mode(self) -> str:
        """发布 control_mode=0（模型控制），便于手臂移动/回放。"""
        prev = self._control_mode
        self._burst_control_mode(MODEL_CONTROL_MODE)
        self._control_mode_received_at = time.time()
        self.ros_bridge.control_mode_changed.emit(MODEL_CONTROL_MODE)
        self.get_logger().info(
            f"手动切换 control_mode {prev} -> {MODEL_CONTROL_MODE}"
        )
        return f"已切换 control_mode -> {MODEL_CONTROL_MODE}（模型控制）"

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

    def _maintain_replay_prep(self) -> None:
        self._burst_control_mode(MODEL_CONTROL_MODE)
        if self._arm_set_enable_client.service_is_ready() and not self._arm_enabled:
            self.request_arm_enable(True)
        if self._hand_set_enable_client.service_is_ready() and not self._hand_enabled:
            self.request_hand_enable(True)

    def _start_replay_prep_mode_lock(self) -> None:
        self._stop_replay_prep_mode_timer()
        self._burst_control_mode(MODEL_CONTROL_MODE)
        self.request_arm_enable(True)
        self.request_hand_enable(True)
        self._replay_prep_mode_timer = self.create_timer(1.0, self._maintain_replay_prep)

    def _stop_replay_prep_mode_timer(self) -> None:
        if self._replay_prep_mode_timer is not None:
            self.destroy_timer(self._replay_prep_mode_timer)
            self._replay_prep_mode_timer = None

    def _publish_slow_motion_targets(
        self,
        moving_xyz: Tuple[float, float, float],
        moving_quat: Tuple[float, float, float, float],
    ) -> None:
        hold_pose = self._slow_motion_hold_pose
        if hold_pose is None:
            return
        hold_xyz, hold_quat = hold_pose
        if self._slow_motion_moving_side == "right":
            self._publish_ik_arm_targets(hold_xyz, hold_quat, moving_xyz, moving_quat)
        else:
            self._publish_ik_arm_targets(moving_xyz, moving_quat, hold_xyz, hold_quat)

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
        max_delta = max(abs(goal_joints[i] - start_joints[i]) for i in range(16))
        start_xyz = prep.get("start_xyz")
        goal_xyz = prep.get("goal_xyz")
        cart_dist = 0.0
        if (
            isinstance(start_xyz, tuple)
            and isinstance(goal_xyz, tuple)
            and len(start_xyz) == 3
            and len(goal_xyz) == 3
        ):
            cart_dist = float(
                sum((float(goal_xyz[i]) - float(start_xyz[i])) ** 2 for i in range(3)) ** 0.5
            )
        if duration_s <= 0:
            if cart_dist > ARM_MOVE_CART_NEAR_M:
                self._abort_slow_motion_prep(
                    f"IK 未求解到目标（笛卡尔距离 {cart_dist:.3f} m，关节变化 {max_delta:.5f} rad）。"
                    "目标可能不可达，或 IK 未更新 /wbc/target_joints"
                )
            else:
                self._abort_slow_motion_prep("目标关节与当前几乎相同，无需移动")
            return
        steps = max(2, int(duration_s * ARM_MOVE_JOINT_HZ))
        joint_poses = build_interpolated_joints(start_joints, goal_joints, steps, linear=True)
        self.get_logger().info(
            f"关节轨迹: {steps} 步 @ {ARM_MOVE_JOINT_HZ:.0f}Hz / {duration_s:.1f}s, "
            f"最大关节变化 {max_delta:.3f} rad, 笛卡尔 {cart_dist:.3f} m, "
            f"发布 {WBC_TARGET_JOINTS_TOPIC}"
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
        goal_xyz_t = prep["goal_xyz"]  # type: ignore[assignment]
        gx, gy, gz = goal_xyz_t
        saved_mode = int(prep.get("saved_mode", self._control_mode))
        side_name = arm_side_label(goal.arm_side)
        self.get_logger().info(
            f"{side_name}移动开始 [{goal.label}]: mode {saved_mode}->"
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
        self._slow_motion_hold_pose = None
        self._slow_motion_final_moving_pose = None
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
        moving_side = goal.arm_side if goal.arm_side in ("left", "right") else "left"
        self._slow_motion_moving_side = moving_side
        self._slow_motion_preparing = True
        self._slow_motion_prep = {
            "goal": goal,
            "phase": "validate",
            "warmup_step": 0,
            "moving_side": moving_side,
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
            left_pose = self._get_left_tcp_in_base(timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S)
            right_pose = self._get_right_tcp_in_base(timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S)
            if left_pose is None or right_pose is None:
                self._abort_slow_motion_prep("无法获取双臂 TCP（IK 目标坐标系 transform 失败）")
                return
            moving_side = goal.arm_side if goal.arm_side in ("left", "right") else "left"
            if moving_side == "right":
                start_pose, hold_pose = right_pose, left_pose
            else:
                start_pose, hold_pose = left_pose, right_pose
            prep["moving_side"] = moving_side
            prep["start_xyz"], prep["start_quat"] = start_pose
            prep["goal_xyz"] = goal.position_xyz
            prep["goal_quat"] = goal.quaternion_xyzw
            prep["hold_pose"] = hold_pose
            prep["saved_mode"] = self._control_mode
            self._slow_motion_moving_side = moving_side
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
            hold_xyz, hold_quat = prep["hold_pose"]  # type: ignore[misc]
            self._slow_motion_hold_pose = (hold_xyz, hold_quat)
            self._slow_motion_moving_side = str(prep.get("moving_side", "left"))
            self._slow_motion_final_moving_pose = (prep["goal_xyz"], prep["goal_quat"])  # type: ignore[index]
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
                prep["best_goal_joints"] = list(self._last_wbc_joints[:16])
                prep["best_joint_delta"] = 0.0
                sx, sy, sz = prep["start_xyz"]  # type: ignore[misc]
                gx, gy, gz = goal_xyz
                cart = ((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) ** 0.5
                self.get_logger().info(
                    f"IK 目标已发布: start=({sx:.3f},{sy:.3f},{sz:.3f}) "
                    f"goal=({gx:.3f},{gy:.3f},{gz:.3f}) 距离={cart:.3f}m，"
                    f"等待最多 {ARM_MOVE_GOAL_SAMPLE_TICKS} ticks"
                )
                return
            settle_ticks = int(prep.get("settle_ticks", 0)) - 1
            prep["settle_ticks"] = settle_ticks
            if self._last_wbc_joints is not None and len(self._last_wbc_joints) >= 16:
                motion_start: List[float] = prep["motion_start_joints"]  # type: ignore[assignment]
                cur = list(self._last_wbc_joints[:16])
                delta = max(abs(cur[i] - motion_start[i]) for i in range(16))
                if delta > float(prep.get("best_joint_delta", 0.0)):
                    prep["best_joint_delta"] = delta
                    prep["best_goal_joints"] = cur
                # 关节已明显变化则提前结束等待
                if delta >= ARM_MOVE_GOAL_MIN_JOINT_DELTA and settle_ticks > 2:
                    settle_ticks = 0
                    prep["settle_ticks"] = 0
            if settle_ticks > 0:
                return
            if self._last_wbc_joints is None or len(self._last_wbc_joints) < 16:
                self._abort_slow_motion_prep("IK 未返回目标关节角")
                return
            best = prep.get("best_goal_joints")
            prep["goal_joints"] = (
                list(best) if isinstance(best, list) and len(best) >= 16
                else list(self._last_wbc_joints[:16])
            )
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
        side_name = arm_side_label(self._slow_motion_moving_side)
        if self._control_mode != MODEL_CONTROL_MODE:
            self.get_logger().info(
                f"{side_name}移动: 切换 control_mode {self._slow_motion_saved_control_mode} -> "
                f"{MODEL_CONTROL_MODE}（暂停 tracker→IK 转发，避免覆盖目标）"
            )
        self._start_slow_motion_mode_lock()

    def _burst_control_mode(self, mode: int, count: int = ARM_MOVE_MODE_BURST) -> None:
        for _ in range(count):
            self._publish_control_mode(mode)

    def start_slow_move_to_segment(
        self, target: SegmentPoseTarget, arm_side: str = "left"
    ) -> str:
        resolved = self.resolve_segment_move_goal(
            target, timeout_s=MOVE_TF_LOOKUP_TIMEOUT_S, arm_side=arm_side
        )
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
        self._slow_motion_hold_pose = None
        self._slow_motion_final_moving_pose = None
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

    def apply_right_hand_angle(self, slider_pct: int) -> float:
        """按游标位置发送右手角度命令，返回实际 position。"""
        pos = slider_to_hand_position(slider_pct)
        msg = make_right_hand_command(pos)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._right_hand_cmd_pub.publish(msg)
        self.get_logger().info(
            f"右手命令 -> position={pos:.3f} (游标={slider_pct}): {ROBOT_RIGHT_HAND_CMD_TOPIC}"
        )
        return pos

    def apply_hand_joint_positions(
        self, side: str, positions: Sequence[float]
    ) -> List[float]:
        """按 6 关节开合量发送手部命令，返回裁剪后的 position 列表。"""
        side = "right" if side == "right" else "left"
        msg = make_hand_command(side, positions)
        msg.header.stamp = self.get_clock().now().to_msg()
        if side == "right":
            self._right_hand_cmd_pub.publish(msg)
            topic = ROBOT_RIGHT_HAND_CMD_TOPIC
        else:
            self._left_hand_cmd_pub.publish(msg)
            topic = ROBOT_LEFT_HAND_CMD_TOPIC
        vals = [float(v) for v in msg.position]
        self.get_logger().debug(
            f"{'右' if side == 'right' else '左'}手关节命令 {vals} -> {topic}"
        )
        return vals

    def toggle_left_hand_between(self, slider_a: int, slider_b: int) -> Tuple[bool, float]:
        """在游标 A/B 两个状态间切换，返回 (当前是否为 A, 发送的 position)。"""
        self._left_hand_at_a = not self._left_hand_at_a
        slider = slider_a if self._left_hand_at_a else slider_b
        preset = "A" if self._left_hand_at_a else "B"
        pos = self.apply_left_hand_angle(slider)
        self.get_logger().info(f"左手切换 -> 状态{preset}")
        self.ros_bridge.left_hand_preset_changed.emit(self._left_hand_at_a)
        return self._left_hand_at_a, pos

    def toggle_right_hand_between(self, slider_a: int, slider_b: int) -> Tuple[bool, float]:
        """在游标 A/B 两个状态间切换，返回 (当前是否为 A, 发送的 position)。"""
        self._right_hand_at_a = not self._right_hand_at_a
        slider = slider_a if self._right_hand_at_a else slider_b
        preset = "A" if self._right_hand_at_a else "B"
        pos = self.apply_right_hand_angle(slider)
        self.get_logger().info(f"右手切换 -> 状态{preset}")
        self.ros_bridge.right_hand_preset_changed.emit(self._right_hand_at_a)
        return self._right_hand_at_a, pos

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
        new_mode = int(msg.data)
        first_receive = self._control_mode_received_at <= 0
        changed = first_receive or (
            not self._is_motion_control_locked() and new_mode != self._control_mode
        )
        if not self._is_motion_control_locked():
            self._control_mode = new_mode
        if first_receive:
            self._control_mode_received_at = time.time()
            self.get_logger().info(f"control_mode={self._control_mode}")
        elif changed and not self._is_motion_control_locked():
            self.get_logger().info(f"control_mode={self._control_mode}")
        if changed:
            self.ros_bridge.control_mode_changed.emit(self._control_mode)

    def _on_arm_enable_state(self, msg: UInt32) -> None:
        enabled = int(msg.data) != 0
        first_receive = self._arm_enable_received_at <= 0
        if first_receive or enabled != self._arm_enabled:
            self.get_logger().info(f"arm/enable_state={'ON' if enabled else 'OFF'}")
        self._arm_enable_received_at = time.time()
        if first_receive or enabled != self._arm_enabled:
            self._arm_enabled = enabled
            self._arm_enabled_since = time.time() if enabled else 0.0
            self.ros_bridge.arm_enable_changed.emit(enabled)
        else:
            self._arm_enabled = enabled

    def _on_hand_enable_state(self, msg: UInt32) -> None:
        enabled = int(msg.data) != 0
        first_receive = self._hand_enable_received_at <= 0
        if first_receive or enabled != self._hand_enabled:
            self.get_logger().info(f"hand/enable_state={'ON' if enabled else 'OFF'}")
        self._hand_enable_received_at = time.time()
        if first_receive or enabled != self._hand_enabled:
            self._hand_enabled = enabled
            self._hand_enabled_since = time.time() if enabled else 0.0
            self.ros_bridge.hand_enable_changed.emit(enabled)
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
    psibot_home = os.environ.get("PSIBOT_HOME", "/home/psibot").strip() or "/home/psibot"
    candidates = [
        os.path.join(psibot_home, "workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"),
        os.path.abspath(os.path.join(here, "..", "install", "scripts_pack", "share", "scripts_pack", "scripts")),
        os.path.join(psibot_home, "workspace_liyichao/a2d-tele/install/scripts_pack/share/scripts_pack/scripts"),
        os.path.expanduser("~/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts"),
        os.path.expanduser("~/workspace_liyichao/a2d-tele/install/scripts_pack/share/scripts_pack/scripts"),
        "/opt/psi/rt/a2d-tele/install/scripts_pack/share/scripts_pack/scripts",
    ]
    stack_script = STACK_SERVICES_SCRIPT
    resolved_candidates: List[str] = []
    for path in candidates:
        resolved = os.path.abspath(path)
        if resolved in resolved_candidates:
            continue
        resolved_candidates.append(resolved)
        if os.path.isfile(os.path.join(resolved, stack_script)):
            return resolved
    for resolved in resolved_candidates:
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
    scripts_dir = resolve_scripts_pack_dir()
    if scripts_dir:
        parts.append(f"export A2D_SCRIPTS_DIR={scripts_dir}")
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


def _is_stack_services_running() -> bool:
    return _pgrep_pattern(STACK_SERVICES_LAUNCH) or _pgrep_pattern(
        LEGACY_BASE_SERVICES_LAUNCH
    )


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


class LocalAiLauncher(QObject):
    """启动/停止本地 AI：SAM3 分割 HTTP 服务 + Ollama 对话服务。"""

    status_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None

    def is_sam3_running(self) -> bool:
        if self._process is not None and self._process.state() == QProcess.Running:
            return True
        return _sam3_server_process_running()

    def start_deploy(self) -> None:
        self._ensure_ollama()
        self._start_sam3()

    def stop(self) -> None:
        if self._process is not None:
            if self._process.state() == QProcess.Running:
                self._process.terminate()
                QTimer.singleShot(2000, self._force_kill_process)
            else:
                self._process = None
        _stop_sam3_server_processes()
        self.running_changed.emit(False)
        self.status_message.emit("已停止本地 SAM3 服务")

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(1500)
        self._process = None

    def _ensure_ollama(self) -> None:
        if check_ollama_server_health():
            self.status_message.emit("Ollama 已在运行")
            return
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            self.status_message.emit(
                "未检测到 Ollama，请安装: https://ollama.com 后执行 ollama pull qwen3.5:4b"
            )
            return
        if QProcess.startDetached(ollama_bin, ["serve"]):
            self.status_message.emit("正在启动 Ollama 服务 (127.0.0.1:11434)…")
        else:
            self.status_message.emit("Ollama 启动失败")

    def _start_sam3(self) -> None:
        if self.is_sam3_running():
            self.status_message.emit("SAM3 服务已在运行")
            self.running_changed.emit(True)
            return

        script = resolve_sam3_run_script()
        if script is None:
            hint = (
                "cd eai && bash run_sam3.sh --host 0.0.0.0"
                if is_running_in_docker()
                else f"未找到 {SAM3_RUN_SCRIPT_NAME}，请确认与 show_camera_topics.py 同目录"
            )
            self.status_message.emit(f"无法启动 SAM3: {hint}")
            return

        bind_host = resolve_sam3_bind_host()
        script_dir = os.path.dirname(script)
        cmd = f"exec bash {os.path.basename(script)} --host {bind_host}"
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(script_dir)
        proc.start("bash", ["-lc", cmd])
        self._process = proc
        self.running_changed.emit(True)
        viewer_url = resolve_sam3_viewer_server_url()
        self.status_message.emit(
            f"正在启动 SAM3 ({bind_host}:{SAM3_DEFAULT_PORT})，viewer URL: {viewer_url}"
        )

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            text = line.strip()
            if text:
                self.status_message.emit(f"[SAM3] {text}")

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code != 0:
            self.status_message.emit(f"SAM3 进程退出 (code={exit_code})")

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.Crashed:
            self.status_message.emit(f"SAM3 进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.kill()
            self._process = None


class RobotStackLauncher(QObject):
    """后台启动 robot-service 与手/臂精简服务栈（不阻塞 UI）。"""

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
            self.status_message.emit("手/臂服务栈已在运行")
            return
        if base == "启动中" or self._base_launch_pending:
            self.status_message.emit("手/臂服务栈正在启动…")
            return
        if self._node.is_hal_arm_ready() or robot == "就绪":
            self._launch_stack_services()
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

    def _launch_stack_services(self) -> None:
        robot, base = self.get_cached_status()
        if base == "运行中":
            self.status_message.emit("手/臂服务栈已在运行")
            return
        if self._base_launch_pending or base == "启动中":
            self.status_message.emit("手/臂服务栈正在启动…")
            return
        script = self._resolve_script(STACK_SERVICES_SCRIPT)
        if script is None:
            tried = resolve_scripts_pack_dir() or "(未找到 scripts_pack 目录)"
            self.status_message.emit(
                f"未找到 {STACK_SERVICES_SCRIPT}（已查: {tried}）。"
                "请设置 A2D_SCRIPTS_DIR 或 colcon build scripts_pack"
            )
            return
        script_dir = os.path.dirname(script)
        cmd = (
            f"{build_ros_shell_prefix()} && cd {script_dir} && "
            f"exec bash {os.path.basename(script)}"
        )
        if not QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self.status_message.emit("手/臂服务栈启动失败（startDetached 返回 false）")
            return
        self._base_launch_pending = True
        self.status_message.emit(
            "正在启动手/臂服务（pedal_controller、topic_router、ik/fk、arm_control、hand）…"
        )

    def _launch_base_services(self) -> None:
        """兼容旧信号连接，实际启动精简栈。"""
        self._launch_stack_services()

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
        base_running = _is_stack_services_running()
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
        self.status_message.emit("robot-service 完成，正在启动手/臂服务栈…")
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
        self._node.prepare_for_rrd_replay()
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
        self._node._stop_replay_prep_mode_timer()
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
        self._node._stop_replay_prep_mode_timer()
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


class CadMeshLauncher(QObject):
    """启动照片/mesh → FoundationPose .obj 重建，日志转发到 UI。"""

    log_line = pyqtSignal(str)
    status_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)
    mesh_ready = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._last_mesh = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == QProcess.Running

    def last_mesh(self) -> str:
        return self._last_mesh

    def start_from_images(
        self,
        images_dir: str,
        name: str,
        *,
        target_extent_m: float = CAD_TARGET_EXTENT_DEFAULT_M,
        poisson_depth: int = CAD_POISSON_DEPTH_DEFAULT,
        min_images: int = CAD_MIN_IMAGES_DEFAULT,
        out_dir: Optional[str] = None,
    ) -> None:
        images_dir = os.path.abspath(os.path.expanduser(images_dir.strip()))
        if not os.path.isdir(images_dir):
            self.status_message.emit(f"照片目录不存在: {images_dir}")
            return
        self._start(
            [
                "--images",
                images_dir,
                "--name",
                name,
                "--target-extent-m",
                str(float(target_extent_m)),
                "--poisson-depth",
                str(int(poisson_depth)),
                "--min-images",
                str(int(min_images)),
            ],
            out_dir=out_dir,
            expected_name=name,
        )

    def start_from_mesh(
        self,
        mesh_path: str,
        name: str,
        *,
        target_extent_m: float = CAD_TARGET_EXTENT_DEFAULT_M,
        out_dir: Optional[str] = None,
    ) -> None:
        mesh_path = os.path.abspath(os.path.expanduser(mesh_path.strip()))
        if not os.path.isfile(mesh_path):
            self.status_message.emit(f"mesh 不存在: {mesh_path}")
            return
        self._start(
            [
                "--import-mesh",
                mesh_path,
                "--name",
                name,
                "--target-extent-m",
                str(float(target_extent_m)),
            ],
            out_dir=out_dir,
            expected_name=name,
        )

    def _start(
        self,
        extra_args: List[str],
        *,
        out_dir: Optional[str],
        expected_name: str,
    ) -> None:
        if self.is_running():
            self.status_message.emit("CAD 重建已在运行")
            return

        name = (expected_name or "").strip().replace(" ", "_")
        if not name or "/" in name or "\\" in name:
            self.status_message.emit("请填写有效的物体名称")
            return

        out = os.path.abspath(os.path.expanduser(out_dir or CAD_MESHES_DIR))
        os.makedirs(out, exist_ok=True)
        self._expected_obj = os.path.join(out, name, "reconstructed.obj")
        self._last_mesh = ""

        if not os.path.isfile(CAD_PHOTOGRAMMETRY_PY):
            self.status_message.emit(f"未找到重建脚本: {CAD_PHOTOGRAMMETRY_PY}")
            return

        mesh_python = resolve_cad_mesh_python()
        launch_args = [
            CAD_PHOTOGRAMMETRY_PY,
            "--name",
            name,
            "--out-dir",
            out,
        ] + extra_args
        quoted = " ".join(shlex.quote(a) for a in launch_args)
        cmd = (
            f"export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "
            f"MESH_PYTHON={shlex.quote(mesh_python)} "
            f"QT_QPA_PLATFORM=offscreen LIBGL_ALWAYS_SOFTWARE=1 && "
            f"cd {shlex.quote(EAI_DIR)} && "
            f"exec {shlex.quote(mesh_python)} {quoted}"
        )

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(EAI_DIR)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ {mesh_python} {' '.join(launch_args)}")
        self.status_message.emit(f"正在生成 CAD: {name}…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的 CAD 重建")
            return
        self.status_message.emit("正在停止 CAD 重建…")
        if self._process is not None:
            self._process.terminate()
            QTimer.singleShot(3000, self._force_kill_process)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None
        self.running_changed.emit(False)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            text = line.rstrip()
            if not text:
                continue
            self.log_line.emit(text)
            if text.startswith("{") and '"mesh"' in text:
                try:
                    payload = json.loads(text)
                    mesh = str(payload.get("mesh") or "").strip()
                    if mesh and os.path.isfile(mesh):
                        self._last_mesh = os.path.abspath(mesh)
                except Exception:
                    pass

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        mesh = self._last_mesh
        if not mesh and getattr(self, "_expected_obj", "") and os.path.isfile(self._expected_obj):
            mesh = os.path.abspath(self._expected_obj)
            self._last_mesh = mesh
        if exit_code == 0 and mesh:
            self.log_line.emit(f"--- CAD 完成: {mesh} ---")
            self.status_message.emit(f"CAD 已生成: {os.path.basename(os.path.dirname(mesh))}")
            self.mesh_ready.emit(mesh)
        elif exit_code == 0:
            self.log_line.emit("--- CAD 进程正常退出（未找到输出 mesh）---")
            self.status_message.emit("CAD 完成但未找到 reconstructed.obj")
        else:
            self.log_line.emit(f"--- CAD 进程退出 (code={exit_code}) ---")
            self.status_message.emit(f"CAD 重建失败 (code={exit_code})")

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.Crashed:
            self.status_message.emit(f"CAD 重建进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- CAD 进程已被强制终止 ---")
            self.status_message.emit("CAD 重建已强制停止")


class PsiPolicyTrainLauncher(QObject):
    """启动/停止 psi-policy 训练，并将 stdout/stderr 实时转发到 UI。"""

    log_line = pyqtSignal(str)
    status_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == QProcess.Running

    def start(
        self,
        config_name: str,
        datahouse_id: str,
        view_id: str,
        *,
        repo_dir: Optional[str] = None,
        num_epochs: int = 0,
        batch_size: int = 0,
        logging_mode: str = "offline",
        num_processes: int = 1,
    ) -> None:
        if self.is_running():
            self.status_message.emit("psi-policy 训练已在运行")
            return

        repo = repo_dir or resolve_psi_policy_dir()
        if repo is None:
            self.status_message.emit(
                "未找到 psi-policy 目录，请点击「选择路径」指定仓库"
            )
            return

        config = (config_name or PSI_POLICY_CONFIG_DEFAULT).strip()
        if not config:
            self.status_message.emit("请选择训练配置 (config-name)")
            return

        overrides: List[str] = [f"logging.mode={logging_mode}"]
        dh_override = format_hydra_override(
            "task.dataset.sources.0.datahouse_id", datahouse_id
        )
        if dh_override:
            overrides.append(dh_override)
        view_override = format_hydra_override(
            "task.dataset.sources.0.view_id", view_id
        )
        if view_override:
            overrides.append(view_override)
        if num_epochs > 0:
            overrides.append(f"training.num_epochs={int(num_epochs)}")
        if batch_size > 0:
            overrides.append(f"train_dataloader.batch_size={int(batch_size)}")

        python_bin = resolve_psi_policy_python(repo)
        train_args = ["psi_policy/train.py", "--config-name", config] + overrides
        if num_processes > 1:
            launch_args = [
                "-m",
                "accelerate.commands.launch",
                f"--num_processes={int(num_processes)}",
            ] + train_args
        else:
            launch_args = train_args

        quoted_args = " ".join(shlex.quote(arg) for arg in launch_args)
        venv_activate = os.path.join(repo, ".venv", "bin", "activate")
        prefix = ""
        if os.path.isfile(venv_activate):
            prefix = f"source {shlex.quote(venv_activate)} && "
        cmd = (
            f"{prefix}cd {shlex.quote(repo)} && "
            f"export PYTHONUNBUFFERED=1 && "
            f"exec {shlex.quote(python_bin)} {quoted_args}"
        )

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(repo)
        proc.start("setsid", ["bash", "-lc", cmd])
        self._process = proc
        self.running_changed.emit(True)
        self.log_line.emit(f"$ cd {repo}")
        self.log_line.emit(f"$ {python_bin} {' '.join(launch_args)}")
        self.status_message.emit(f"正在启动 psi-policy 训练 ({config})…")

    def stop(self) -> None:
        if not self.is_running():
            self.status_message.emit("当前没有运行中的训练")
            return
        self.status_message.emit("正在停止训练…")
        if self._process is not None:
            self._process.terminate()
            QTimer.singleShot(3000, self._force_kill_process)

    def shutdown(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._process = None
        self.running_changed.emit(False)

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line:
                self.log_line.emit(line.rstrip())

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._process = None
        self.running_changed.emit(False)
        if exit_code == 0:
            self.log_line.emit("--- 训练进程正常退出 ---")
            self.status_message.emit("psi-policy 训练已完成")
        else:
            self.log_line.emit(f"--- 训练进程退出 (code={exit_code}) ---")
            self.status_message.emit(f"psi-policy 训练异常退出 (code={exit_code})")

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.Crashed:
            self.status_message.emit(f"psi-policy 训练进程错误: {error}")

    def _force_kill_process(self) -> None:
        if self._process is not None and self._process.state() == QProcess.Running:
            self._process.kill()
            self._process = None
            self.running_changed.emit(False)
            self.log_line.emit("--- 训练进程已被强制终止 ---")
            self.status_message.emit("训练已强制停止")


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
        self._last_sam3_topic = ""
        self._sam3_call_busy = False
        self._sam3_call_bridge = Sam3CallBridge()
        self._sam3_call_bridge.finished.connect(self._on_sam3_call_finished)
        self._fp_call_busy = False
        self._fp_call_bridge = FpCallBridge()
        self._fp_call_bridge.finished.connect(self._on_fp_call_finished)
        self._robot_ui_last_update = 0.0
        self._pending_arm_move_goal: Optional[ResolvedArmMoveGoal] = None
        self._arm_enable_wait_deadline = 0.0
        self._arm_enable_wait_timer = QTimer(self)
        self._arm_enable_wait_timer.setInterval(200)
        self._arm_enable_wait_timer.timeout.connect(self._on_arm_enable_wait_tick)
        self._psi_policy_dir_override: Optional[str] = None
        self._cad_capture_dir = ""
        self._cad_capture_saved = 0
        self._cad_last_capture_gray: Optional[np.ndarray] = None

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
        control_tabs.addTab(camera_tab, "大脑")

        robot_tab = QWidget()
        robot_layout = QVBoxLayout(robot_tab)
        robot_layout.setContentsMargins(8, 6, 8, 6)
        robot_layout.setSpacing(6)
        stack_row = QHBoxLayout()
        self._stack_launcher = RobotStackLauncher(node, self)
        self.robot_stack_status = QLabel("robot: -- | stack: --")
        self.robot_stack_status.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.robot_stack_status.setToolTip(
            f"robot-service 初始化 HAL；手/臂栈启动 pedal_controller/topic_router/ik/fk/arm/hand。\n"
            f"不启动 VR、扫码、数据采集、RealSense 等。\n"
            f"日志: tail -f {STACK_SERVICES_LOG}"
        )
        self.robot_stack_btn = QPushButton("启动机器人栈")
        self.robot_stack_btn.setToolTip(
            "依次启动 robot-service（约 3 分钟）与手/臂精简服务栈。\n"
            "若 HAL 已就绪则跳过 robot-service。\n"
            "仅含控制手臂与手所需节点。"
        )
        self.robot_stack_btn.clicked.connect(self._on_robot_stack_clicked)
        stack_row.addWidget(self.robot_stack_status, 1)
        stack_row.addWidget(self.robot_stack_btn)
        robot_layout.addLayout(stack_row)

        enable_row = QHBoxLayout()
        enable_row.setSpacing(10)
        self.robot_arm_enable_label = QLabel("手臂: --")
        self.robot_arm_enable_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.robot_arm_enable_label.setToolTip(f"订阅 {ARM_ENABLE_STATE_TOPIC}，1=已使能")
        self.robot_hand_enable_label = QLabel("手: --")
        self.robot_hand_enable_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.robot_hand_enable_label.setToolTip(f"订阅 {HAND_ENABLE_STATE_TOPIC}，1=已使能")
        self.robot_control_mode_label = QLabel("mode: --")
        self.robot_control_mode_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.robot_control_mode_label.setToolTip(
            f"订阅 {CONTROL_MODE_TOPIC}，回放需 mode={MODEL_CONTROL_MODE}"
        )
        enable_row.addWidget(self.robot_arm_enable_label)
        self.robot_arm_enable_btn = QPushButton("启用手臂")
        self.robot_arm_enable_btn.setToolTip(
            "等同踏板 F2 开启/关闭手臂控制"
        )
        self.robot_arm_enable_btn.clicked.connect(self._on_arm_enable_clicked)
        enable_row.addWidget(self.robot_arm_enable_btn)
        enable_row.addSpacing(8)
        enable_row.addWidget(self.robot_hand_enable_label)
        self.robot_hand_enable_btn = QPushButton("启用手")
        self.robot_hand_enable_btn.setToolTip(
            "等同踏板 F1 开启/关闭手部控制"
        )
        self.robot_hand_enable_btn.clicked.connect(self._on_hand_enable_clicked)
        enable_row.addWidget(self.robot_hand_enable_btn)
        enable_row.addSpacing(8)
        enable_row.addWidget(self.robot_control_mode_label)
        self.robot_mode0_btn = QPushButton("切 mode=0")
        self.robot_mode0_btn.setToolTip(
            f"发布 {CONTROL_MODE_TOPIC}={MODEL_CONTROL_MODE}（模型控制，手臂移动/回放需要）"
        )
        self.robot_mode0_btn.clicked.connect(self._on_model_mode_clicked)
        enable_row.addWidget(self.robot_mode0_btn)
        enable_row.addStretch()
        robot_layout.addLayout(enable_row)

        replay_row = QHBoxLayout()
        replay_row.setSpacing(6)
        self.replay_status_label = QLabel("回放: --")
        self.replay_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.replay_status_label.setToolTip(
            "RRD 轨迹回放状态（/rrd_replay/running_state）\n"
            "需手/臂服务栈 + control_mode=0 + 手/臂使能"
        )
        self.replay_select_btn = QPushButton("选择路径")
        self.replay_select_btn.setToolTip(
            "选择 .rrd 回放文件。\n"
            f"默认数据目录: {default_rrd_dataset_dir()}"
        )
        self.replay_select_btn.clicked.connect(self._on_replay_select_clicked)
        self.replay_start_btn = QPushButton("开始")
        self.replay_start_btn.setToolTip(
            "启动 rrd_replay 并开始回放。\n"
            "会自动启动手/臂服务栈（若未运行）、切 control_mode=0 并使能手/臂。"
        )
        self.replay_start_btn.clicked.connect(self._on_replay_start_clicked)
        self.replay_stop_btn = QPushButton("停止")
        self.replay_stop_btn.setToolTip("停止 rrd_replay 节点与当前回放。")
        self.replay_stop_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
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
        self.replay_rrd_path_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        rrd_path_row.addWidget(self.replay_rrd_path_edit, 1)
        robot_layout.addLayout(rrd_path_row)
        control_tabs.addTab(robot_tab, "回放")

        self._replay_launcher = RrdReplayLauncher(node, self)

        segment_tab = QWidget()
        segment_outer = QVBoxLayout(segment_tab)
        segment_outer.setContentsMargins(8, 6, 8, 6)
        segment_outer.setSpacing(6)
        segment_layout = QHBoxLayout()
        segment_layout.setSpacing(6)
        segment_layout.addWidget(QLabel("分割"))
        self.segment_backend_combo = ImeSafeComboBox()
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
        self.sam3_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.sam3_status_label.setToolTip("SAM3 HTTP 服务健康检查")
        segment_layout.addWidget(self.sam3_status_label)
        self._local_ai_launcher = LocalAiLauncher(self)
        self.local_ai_deploy_btn = QPushButton("本地部署 AI")
        self.local_ai_deploy_btn.setToolTip(
            "启动本地 SAM3 分割服务 (run_sam3.sh) 与 Ollama 对话服务。\n"
            "SAM3: 自动勾选 HTTP，分割模式切到 SAM3 点提示。\n"
            "Ollama: 自动切换 AI 对话为本地 Qwen 预设。\n"
            "Docker 内 viewer 时请在宿主机执行: bash run_sam3.sh --host 0.0.0.0"
        )
        self.local_ai_deploy_btn.clicked.connect(self._on_local_ai_deploy_clicked)
        segment_layout.addWidget(self.local_ai_deploy_btn)
        segment_outer.addLayout(segment_layout)

        sam3_invoke_row = QHBoxLayout()
        sam3_invoke_row.setSpacing(6)
        sam3_invoke_row.addWidget(QLabel("提示点"))
        self.sam3_u_spin = QSpinBox()
        self.sam3_u_spin.setRange(0, 9999)
        self.sam3_u_spin.setToolTip("提示点横坐标 u（点击图像只更新此点，不自动分割）")
        sam3_invoke_row.addWidget(QLabel("u"))
        sam3_invoke_row.addWidget(self.sam3_u_spin)
        self.sam3_v_spin = QSpinBox()
        self.sam3_v_spin.setRange(0, 9999)
        self.sam3_v_spin.setToolTip("提示点纵坐标 v（点击图像只更新此点，不自动分割）")
        sam3_invoke_row.addWidget(QLabel("v"))
        sam3_invoke_row.addWidget(self.sam3_v_spin)
        self.stereo_invoke_btn = QPushButton("调用分割")
        self.stereo_invoke_btn.setToolTip(
            "对当前提示点执行立体分割 + 6D 位姿（跟随上方分割/位姿后端设置）。\n"
            "点击图像仅选点，不会自动分割。"
        )
        self.stereo_invoke_btn.clicked.connect(self._on_stereo_invoke_clicked)
        sam3_invoke_row.addWidget(self.stereo_invoke_btn)
        self.sam3_invoke_btn = QPushButton("调用 SAM3")
        self.sam3_invoke_btn.setToolTip(
            "对当前彩色图像调用 SAM3 分割，并在下方显示结果；"
            "成功时会在图像上叠加 mask"
        )
        self.sam3_invoke_btn.clicked.connect(self._on_sam3_invoke_clicked)
        sam3_invoke_row.addWidget(self.sam3_invoke_btn)
        sam3_invoke_row.addStretch()
        segment_outer.addLayout(sam3_invoke_row)

        self.sam3_result_edit = QTextEdit()
        self.sam3_result_edit.setReadOnly(True)
        self.sam3_result_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.sam3_result_edit.setMaximumHeight(108)
        self.sam3_result_edit.setPlaceholderText("SAM3 调用结果将显示在这里…")
        self.sam3_result_edit.setStyleSheet(
            f"QTextEdit {{ color: {UI_TEXT_PRIMARY}; background-color: #252525; "
            "border: 1px solid #555; }}"
        )
        sam3_result_row = QHBoxLayout()
        sam3_result_row.setSpacing(8)
        sam3_result_row.addWidget(self.sam3_result_edit, 1)
        self.sam3_preview_label = QLabel("分割预览")
        self.sam3_preview_label.setAlignment(Qt.AlignCenter)
        self.sam3_preview_label.setMinimumSize(160, 108)
        self.sam3_preview_label.setMaximumSize(240, 160)
        self.sam3_preview_label.setStyleSheet(
            "QLabel { color: #888; background-color: #1a1a1a; border: 1px solid #555; }"
        )
        self.sam3_preview_label.setToolTip("分割 mask 叠加预览（相机画面上也会持续绘制）")
        sam3_result_row.addWidget(self.sam3_preview_label)
        segment_outer.addLayout(sam3_result_row)

        pose_row = QHBoxLayout()
        pose_row.setSpacing(6)
        pose_row.addWidget(QLabel("位姿"))
        self.pose_backend_combo = ImeSafeComboBox()
        self.pose_backend_combo.addItem("PCA(点云)", POSE_BACKEND_PCA)
        self.pose_backend_combo.addItem("FoundationPose", POSE_BACKEND_FOUNDATIONPOSE)
        self.pose_backend_combo.setToolTip(
            "FoundationPose 需 CAD mesh + 彩色/深度/分割 mask；"
            "推荐宿主机运行 run_foundationpose.sh"
        )
        self.pose_backend_combo.currentIndexChanged.connect(self._on_pose_settings_changed)
        pose_row.addWidget(self.pose_backend_combo)
        self.fp_mesh_edit = QLineEdit()
        self.fp_mesh_edit.setPlaceholderText("物体 mesh (.obj)")
        self.fp_mesh_edit.setMinimumWidth(180)
        self.fp_mesh_edit.setText(FP_MESH_DEFAULT)
        self.fp_mesh_edit.textChanged.connect(self._on_pose_settings_changed)
        pose_row.addWidget(self.fp_mesh_edit, 1)
        self.fp_mesh_browse_btn = QPushButton("…")
        self.fp_mesh_browse_btn.setFixedWidth(28)
        self.fp_mesh_browse_btn.setToolTip("选择 mesh 文件")
        self.fp_mesh_browse_btn.clicked.connect(self._on_fp_mesh_browse_clicked)
        pose_row.addWidget(self.fp_mesh_browse_btn)
        self.fp_http_check = QCheckBox("FP HTTP")
        self.fp_http_check.setChecked(FP_USE_HTTP_DEFAULT)
        self.fp_http_check.setToolTip(
            f"勾选后走 FoundationPose 常驻服务 ({FP_SERVER_URL_DEFAULT})"
        )
        self.fp_http_check.toggled.connect(self._on_pose_settings_changed)
        pose_row.addWidget(self.fp_http_check)
        self.fp_status_label = QLabel("FP: --")
        self.fp_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.fp_status_label.setMinimumWidth(72)
        pose_row.addWidget(self.fp_status_label)
        self.fp_avail_detail_label = QLabel("")
        self.fp_avail_detail_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.fp_avail_detail_label.setStyleSheet(f"color: {UI_TEXT_MUTED};")
        self.fp_avail_detail_label.setToolTip("FoundationPose 可用性明细")
        pose_row.addWidget(self.fp_avail_detail_label, 1)
        self.fp_invoke_btn = QPushButton("调用 FP")
        self.fp_invoke_btn.setToolTip(
            "对当前彩色+深度图像调用 FoundationPose：先分割再估计 6D 位姿。\n"
            "点击图像仅选点，需点本按钮确认。\n"
            "mesh 可在 worker 侧配置（run_foundationpose.sh --mesh）；"
            "Docker viewer 本地无 mesh 时仍可使用 worker 默认 mesh"
        )
        self.fp_invoke_btn.clicked.connect(self._on_fp_invoke_clicked)
        pose_row.addWidget(self.fp_invoke_btn)
        segment_outer.addLayout(pose_row)

        self.fp_result_edit = QTextEdit()
        self.fp_result_edit.setReadOnly(True)
        self.fp_result_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.fp_result_edit.setMaximumHeight(120)
        self.fp_result_edit.setPlaceholderText("FoundationPose 调用结果将显示在这里…")
        self.fp_result_edit.setStyleSheet(
            f"QTextEdit {{ color: {UI_TEXT_PRIMARY}; background-color: #252525; "
            "border: 1px solid #555; }}"
        )
        segment_outer.addWidget(self.fp_result_edit)

        control_tabs.addTab(segment_tab, "分割")

        cad_tab = QWidget()
        cad_outer = QVBoxLayout(cad_tab)
        cad_outer.setContentsMargins(8, 6, 8, 6)
        cad_outer.setSpacing(6)

        cad_mode_row = QHBoxLayout()
        cad_mode_row.setSpacing(6)
        cad_mode_row.addWidget(QLabel("模式"))
        self.cad_mode_combo = ImeSafeComboBox()
        self.cad_mode_combo.addItem("多视角照片", "photos")
        self.cad_mode_combo.addItem("导入 Mesh", "mesh")
        self.cad_mode_combo.setToolTip(
            "照片：COLMAP + Poisson 重建（需本机 colmap）。\n"
            "导入：Meshroom / 扫描 / 已有 .obj/.ply 后处理。"
        )
        self.cad_mode_combo.currentIndexChanged.connect(self._on_cad_mode_changed)
        cad_mode_row.addWidget(self.cad_mode_combo)
        cad_mode_row.addWidget(QLabel("名称"))
        self.cad_name_edit = QLineEdit("my_object")
        self.cad_name_edit.setMinimumWidth(120)
        self.cad_name_edit.setToolTip("输出到 meshes/<名称>/reconstructed.obj")
        self.cad_name_edit.editingFinished.connect(self._sync_cad_capture_from_disk)
        cad_mode_row.addWidget(self.cad_name_edit)
        cad_mode_row.addWidget(QLabel("边长(m)"))
        self.cad_extent_spin = QDoubleSpinBox()
        self.cad_extent_spin.setRange(0.01, 2.0)
        self.cad_extent_spin.setSingleStep(0.01)
        self.cad_extent_spin.setDecimals(3)
        self.cad_extent_spin.setValue(CAD_TARGET_EXTENT_DEFAULT_M)
        self.cad_extent_spin.setFixedWidth(80)
        self.cad_extent_spin.setToolTip("后处理：mesh 最大边长缩放到该值（米）")
        cad_mode_row.addWidget(self.cad_extent_spin)
        cad_mode_row.addWidget(QLabel("Poisson"))
        self.cad_poisson_spin = QSpinBox()
        self.cad_poisson_spin.setRange(6, 11)
        self.cad_poisson_spin.setValue(CAD_POISSON_DEPTH_DEFAULT)
        self.cad_poisson_spin.setFixedWidth(52)
        self.cad_poisson_spin.setToolTip("仅照片模式：越大越细越慢")
        cad_mode_row.addWidget(self.cad_poisson_spin)
        cad_mode_row.addStretch()
        cad_outer.addLayout(cad_mode_row)

        cad_src_row = QHBoxLayout()
        cad_src_row.setSpacing(6)
        self.cad_src_label = QLabel("照片目录:")
        cad_src_row.addWidget(self.cad_src_label)
        self.cad_src_edit = QLineEdit()
        self.cad_src_edit.setPlaceholderText("选择多视角照片目录，或导入 .obj/.ply")
        self.cad_src_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        cad_src_row.addWidget(self.cad_src_edit, 1)
        self.cad_src_browse_btn = QPushButton("选择…")
        self.cad_src_browse_btn.setToolTip("选择照片目录或 mesh 文件")
        self.cad_src_browse_btn.clicked.connect(self._on_cad_src_browse_clicked)
        cad_src_row.addWidget(self.cad_src_browse_btn)
        cad_outer.addLayout(cad_src_row)

        cad_capture_row = QHBoxLayout()
        cad_capture_row.setSpacing(6)
        cad_capture_row.addWidget(QLabel("采集"))
        self.cad_capture_target_spin = QSpinBox()
        self.cad_capture_target_spin.setRange(CAD_MIN_IMAGES_DEFAULT, 48)
        self.cad_capture_target_spin.setValue(CAD_CAPTURE_COUNT_DEFAULT)
        self.cad_capture_target_spin.setFixedWidth(52)
        self.cad_capture_target_spin.setToolTip("目标张数；绕物体换视角后逐张拍摄")
        self.cad_capture_target_spin.valueChanged.connect(self._update_cad_capture_btn)
        cad_capture_row.addWidget(self.cad_capture_target_spin)
        self.cad_capture_btn = QPushButton("拍一张 (0/12)")
        self.cad_capture_btn.setToolTip(
            f"固定使用头部 RGB-D：{CAD_CAPTURE_TOPIC_DEFAULT} + {CAD_CAPTURE_DEPTH_DEFAULT}\n"
            "保存彩色+深度到 meshes/<名称>/photos/，并用 TSDF 生成 CAD。\n"
            "重要：每拍一张请明显转动物体（或移动相机），相邻帧不能几乎一样。"
        )
        self.cad_capture_btn.clicked.connect(self._on_cad_capture_clicked)
        cad_capture_row.addWidget(self.cad_capture_btn)
        self.cad_capture_reset_btn = QPushButton("清空采集")
        self.cad_capture_reset_btn.setToolTip("清空当前物体的 photos 目录与计数")
        self.cad_capture_reset_btn.clicked.connect(self._on_cad_capture_reset_clicked)
        cad_capture_row.addWidget(self.cad_capture_reset_btn)
        self.cad_auto_gen_check = QCheckBox("满额后自动生成")
        self.cad_auto_gen_check.setChecked(True)
        self.cad_auto_gen_check.setToolTip("拍满目标张数后自动启动 RGB-D / COLMAP 重建")
        cad_capture_row.addWidget(self.cad_auto_gen_check)
        self.cad_capture_hint = QLabel(
            "头部 RGB-D：每拍一张请转动物体，视角差太小会被拒绝"
        )
        self.cad_capture_hint.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.cad_capture_hint.setStyleSheet(f"color: {UI_TEXT_MUTED};")
        cad_capture_row.addWidget(self.cad_capture_hint, 1)
        cad_outer.addLayout(cad_capture_row)

        cad_action_row = QHBoxLayout()
        cad_action_row.setSpacing(6)
        self.cad_status_label = QLabel("CAD: 空闲")
        self.cad_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.cad_status_label.setToolTip(
            f"输出目录: {CAD_MESHES_DIR}\n"
            f"Python: {resolve_cad_mesh_python()}\n"
            "照片重建需安装 COLMAP"
        )
        cad_action_row.addWidget(self.cad_status_label, 1)
        self.cad_existing_combo = ImeSafeComboBox()
        self.cad_existing_combo.setMinimumWidth(160)
        self.cad_existing_combo.setToolTip("已有 meshes/<name>/reconstructed.obj")
        cad_action_row.addWidget(self.cad_existing_combo)
        self.cad_refresh_btn = QPushButton("刷新")
        self.cad_refresh_btn.clicked.connect(self._refresh_cad_mesh_combo)
        cad_action_row.addWidget(self.cad_refresh_btn)
        self.cad_apply_fp_btn = QPushButton("应用到 FP")
        self.cad_apply_fp_btn.setToolTip("将所选/刚生成的 mesh 填入「分割」页 FoundationPose mesh 路径")
        self.cad_apply_fp_btn.clicked.connect(self._on_cad_apply_fp_clicked)
        cad_action_row.addWidget(self.cad_apply_fp_btn)
        self.cad_start_btn = QPushButton("生成 CAD")
        self.cad_start_btn.setToolTip("开始照片重建或 mesh 后处理")
        self.cad_start_btn.clicked.connect(self._on_cad_start_clicked)
        cad_action_row.addWidget(self.cad_start_btn)
        self.cad_stop_btn = QPushButton("停止")
        self.cad_stop_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
        self.cad_stop_btn.setEnabled(False)
        self.cad_stop_btn.clicked.connect(self._on_cad_stop_clicked)
        cad_action_row.addWidget(self.cad_stop_btn)
        self.cad_clear_log_btn = QPushButton("清空日志")
        self.cad_clear_log_btn.clicked.connect(self._on_cad_clear_log_clicked)
        cad_action_row.addWidget(self.cad_clear_log_btn)
        cad_outer.addLayout(cad_action_row)

        self.cad_log_edit = QTextEdit()
        self.cad_log_edit.setReadOnly(True)
        self.cad_log_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.cad_log_edit.setMinimumHeight(120)
        self.cad_log_edit.setMaximumHeight(200)
        self.cad_log_edit.setPlaceholderText(
            "CAD 重建日志…\n"
            "建议：≥12 张照片、相邻约 60% 重叠；哑光非透明物体更稳。"
        )
        self.cad_log_edit.setStyleSheet(
            f"QTextEdit {{ color: {UI_TEXT_PRIMARY}; background-color: #252525; "
            "border: 1px solid #555; }}"
        )
        cad_outer.addWidget(self.cad_log_edit)

        self._cad_launcher = CadMeshLauncher(self)
        self._on_cad_mode_changed()
        self._refresh_cad_mesh_combo()
        self._sync_cad_capture_from_disk()
        control_tabs.addTab(cad_tab, "CAD")

        train_tab = QWidget()
        train_outer = QVBoxLayout(train_tab)
        train_outer.setContentsMargins(8, 6, 8, 6)
        train_outer.setSpacing(6)

        train_path_row = QHBoxLayout()
        train_path_row.setSpacing(6)
        train_path_row.addWidget(QLabel("psi-policy"))
        self.train_policy_dir_edit = QLineEdit()
        self.train_policy_dir_edit.setReadOnly(True)
        self.train_policy_dir_edit.setPlaceholderText("点击「选择路径」指定 psi-policy 仓库")
        self.train_policy_dir_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        train_path_row.addWidget(self.train_policy_dir_edit, 1)
        self.train_browse_policy_btn = QPushButton("选择路径")
        self.train_browse_policy_btn.setToolTip(
            "选择 psi-policy 仓库根目录（需包含 psi_policy/train.py）"
        )
        self.train_browse_policy_btn.clicked.connect(self._on_train_browse_policy_clicked)
        train_path_row.addWidget(self.train_browse_policy_btn)
        train_outer.addLayout(train_path_row)

        train_row1 = QHBoxLayout()
        train_row1.setSpacing(6)
        train_row1.addWidget(QLabel("配置"))
        self.train_config_combo = ImeSafeComboBox()
        self.train_config_combo.setMinimumWidth(220)
        self.train_config_combo.setToolTip("psi-policy Hydra workspace 配置 (example_workspace_*)")
        psi_repo = resolve_psi_policy_dir()
        train_configs = list_psi_policy_workspace_configs(psi_repo) if psi_repo else []
        if not train_configs:
            train_configs = [PSI_POLICY_CONFIG_DEFAULT]
        for cfg_name in train_configs:
            self.train_config_combo.addItem(cfg_name, cfg_name)
        default_idx = self.train_config_combo.findData(PSI_POLICY_CONFIG_DEFAULT)
        if default_idx >= 0:
            self.train_config_combo.setCurrentIndex(default_idx)
        train_row1.addWidget(self.train_config_combo)
        train_row1.addWidget(QLabel("datahouse_id"))
        self.train_datahouse_edit = QLineEdit()
        self.train_datahouse_edit.setPlaceholderText("数据仓库 ID")
        self.train_datahouse_edit.setMinimumWidth(120)
        train_row1.addWidget(self.train_datahouse_edit, 1)
        train_row1.addWidget(QLabel("view_id"))
        self.train_view_edit = QLineEdit()
        self.train_view_edit.setPlaceholderText("训练视图 ID")
        self.train_view_edit.setMinimumWidth(120)
        train_row1.addWidget(self.train_view_edit, 1)
        train_outer.addLayout(train_row1)

        train_row2 = QHBoxLayout()
        train_row2.setSpacing(6)
        train_row2.addWidget(QLabel("epochs"))
        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(0, 9999)
        self.train_epochs_spin.setValue(0)
        self.train_epochs_spin.setSpecialValueText("默认")
        self.train_epochs_spin.setToolTip("0 表示使用配置文件默认值")
        self.train_epochs_spin.setFixedWidth(72)
        train_row2.addWidget(self.train_epochs_spin)
        train_row2.addWidget(QLabel("batch"))
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(0, 4096)
        self.train_batch_spin.setValue(0)
        self.train_batch_spin.setSpecialValueText("默认")
        self.train_batch_spin.setToolTip("0 表示使用配置文件默认值")
        self.train_batch_spin.setFixedWidth(72)
        train_row2.addWidget(self.train_batch_spin)
        train_row2.addWidget(QLabel("WandB"))
        self.train_logging_combo = ImeSafeComboBox()
        for mode in PSI_POLICY_LOGGING_MODES:
            self.train_logging_combo.addItem(mode, mode)
        self.train_logging_combo.setCurrentIndex(
            self.train_logging_combo.findData("offline")
        )
        self.train_logging_combo.setToolTip("logging.mode：offline / online / disabled")
        train_row2.addWidget(self.train_logging_combo)
        train_row2.addWidget(QLabel("GPU数"))
        self.train_gpu_spin = QSpinBox()
        self.train_gpu_spin.setRange(1, 16)
        self.train_gpu_spin.setValue(1)
        self.train_gpu_spin.setToolTip(">1 时使用 accelerate launch 多卡训练")
        self.train_gpu_spin.setFixedWidth(52)
        train_row2.addWidget(self.train_gpu_spin)
        self.train_status_label = QLabel("训练: 空闲")
        self.train_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        repo_hint = psi_repo or "(未找到 psi-policy，请设置 PSI_POLICY_DIR)"
        self.train_status_label.setToolTip(
            f"psi-policy 目录: {repo_hint}\n"
            "训练日志实时显示在下方"
        )
        train_row2.addWidget(self.train_status_label, 1)
        self.train_start_btn = QPushButton("开始训练")
        self.train_start_btn.setToolTip("启动 psi_policy/train.py")
        self.train_start_btn.clicked.connect(self._on_train_start_clicked)
        train_row2.addWidget(self.train_start_btn)
        self.train_stop_btn = QPushButton("停止")
        self.train_stop_btn.setToolTip("终止当前训练进程")
        self.train_stop_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.clicked.connect(self._on_train_stop_clicked)
        train_row2.addWidget(self.train_stop_btn)
        self.train_clear_log_btn = QPushButton("清空日志")
        self.train_clear_log_btn.clicked.connect(self._on_train_clear_log_clicked)
        train_row2.addWidget(self.train_clear_log_btn)
        train_outer.addLayout(train_row2)

        self.train_log_edit = QTextEdit()
        self.train_log_edit.setReadOnly(True)
        self.train_log_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.train_log_edit.setMinimumHeight(140)
        self.train_log_edit.setMaximumHeight(220)
        self.train_log_edit.setPlaceholderText("训练日志将在此实时显示…")
        self.train_log_edit.setStyleSheet(
            f"QTextEdit {{ color: {UI_TEXT_PRIMARY}; background-color: #252525; "
            "border: 1px solid #555; }}"
        )
        train_outer.addWidget(self.train_log_edit)

        self._train_launcher = PsiPolicyTrainLauncher(self)
        self._update_train_path_ui()
        if self._get_psi_policy_dir() is None:
            self.train_log_edit.setPlainText(
                "未找到 psi-policy 仓库。\n"
                "请点击「选择路径」指定 psi-policy 目录。"
            )
        control_tabs.addTab(train_tab, "训练")

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
        self.left_hand_label_a.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
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
        self.left_hand_label_b.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
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

        right_hand_row = QHBoxLayout()
        right_hand_row.setSpacing(6)
        right_hand_row.addWidget(QLabel("右手 A:"))
        self.right_hand_slider_a = QSlider(Qt.Horizontal)
        self.right_hand_slider_a.setRange(0, 100)
        self.right_hand_slider_a.setValue(RIGHT_HAND_ANGLE_A_DEFAULT)
        self.right_hand_slider_a.setFixedWidth(120)
        self.right_hand_slider_a.setToolTip("右手状态 A 的角度 (0=张, 100=合)")
        self.right_hand_slider_a.valueChanged.connect(self._on_right_hand_sliders_changed)
        right_hand_row.addWidget(self.right_hand_slider_a)
        self.right_hand_label_a = QLabel(format_hand_angle_label(RIGHT_HAND_ANGLE_A_DEFAULT))
        self.right_hand_label_a.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.right_hand_label_a.setMinimumWidth(56)
        right_hand_row.addWidget(self.right_hand_label_a)

        right_hand_row.addWidget(QLabel("B:"))
        self.right_hand_slider_b = QSlider(Qt.Horizontal)
        self.right_hand_slider_b.setRange(0, 100)
        self.right_hand_slider_b.setValue(RIGHT_HAND_ANGLE_B_DEFAULT)
        self.right_hand_slider_b.setFixedWidth(120)
        self.right_hand_slider_b.setToolTip("右手状态 B 的角度 (0=张, 100=合)")
        self.right_hand_slider_b.valueChanged.connect(self._on_right_hand_sliders_changed)
        right_hand_row.addWidget(self.right_hand_slider_b)
        self.right_hand_label_b = QLabel(format_hand_angle_label(RIGHT_HAND_ANGLE_B_DEFAULT))
        self.right_hand_label_b.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.right_hand_label_b.setMinimumWidth(56)
        right_hand_row.addWidget(self.right_hand_label_b)

        self.right_hand_toggle_btn = QPushButton("切换")
        self.right_hand_toggle_btn.setToolTip("在游标 A 与 B 设定的两个角度之间切换")
        self.right_hand_toggle_btn.clicked.connect(self._on_right_hand_toggle)
        right_hand_row.addWidget(self.right_hand_toggle_btn)

        self.right_hand_apply_btn = QPushButton("应用当前")
        self.right_hand_apply_btn.setToolTip("将主游标（当前激活状态 A 或 B）的角度立即发送")
        self.right_hand_apply_btn.clicked.connect(self._on_right_hand_apply_active)
        right_hand_row.addWidget(self.right_hand_apply_btn)
        right_hand_row.addStretch()
        control_layout.addLayout(right_hand_row)

        enable_row = QHBoxLayout()
        enable_row.setSpacing(6)
        self.hand_enable_label = QLabel("手: --")
        self.hand_enable_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.hand_enable_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        self.hand_enable_btn = QPushButton("启用手")
        self.hand_enable_btn.setToolTip(
            "等同踏板 F1 开启手部控制（F1 为开关，按一次开、再按一次关）"
        )
        self.hand_enable_btn.clicked.connect(self._on_hand_enable_clicked)
        enable_row.addWidget(self.hand_enable_label)
        enable_row.addWidget(self.hand_enable_btn)
        enable_row.addSpacing(12)
        self.arm_enable_label = QLabel("手臂: --")
        self.arm_enable_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.arm_enable_label.setStyleSheet(f"color: {UI_ACCENT_RED};")
        self.arm_enable_btn = QPushButton("启用手臂")
        self.arm_enable_btn.setToolTip(
            "等同踏板 F2 开启手臂控制（F2 为开关，按一次开、再按一次关）"
        )
        self.arm_enable_btn.clicked.connect(self._on_arm_enable_clicked)
        enable_row.addWidget(self.arm_enable_label)
        enable_row.addWidget(self.arm_enable_btn)
        enable_row.addSpacing(12)
        self.control_mode_label = QLabel("mode: --")
        self.control_mode_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.control_mode_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.control_mode_label.setToolTip(
            f"当前 {CONTROL_MODE_TOPIC}，模型控制/回放需为 {MODEL_CONTROL_MODE}"
        )
        enable_row.addWidget(self.control_mode_label)
        self.mode0_btn = QPushButton("切 mode=0")
        self.mode0_btn.setToolTip(
            f"发布 {CONTROL_MODE_TOPIC}={MODEL_CONTROL_MODE}（模型控制，手臂移动/回放需要）"
        )
        self.mode0_btn.clicked.connect(self._on_model_mode_clicked)
        enable_row.addWidget(self.mode0_btn)
        enable_row.addStretch()
        control_layout.addLayout(enable_row)

        arm_row1 = QHBoxLayout()
        arm_row1.setSpacing(6)
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
        self.arm_move_speed_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
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
        self.left_arm_move_btn.clicked.connect(
            lambda: self._on_arm_move_clicked("left")
        )
        arm_row2.addWidget(self.left_arm_move_btn)

        self.right_arm_move_btn = QPushButton("右臂: 移动")
        self.right_arm_move_btn.setEnabled(False)
        self.right_arm_move_btn.setToolTip(
            "将右臂 TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
            f"目标可为分割位姿，或相对当前位置的手动偏移。\n"
            f"未使能时会自动启用手臂；同时发布左右臂 IK 目标。"
        )
        self.right_arm_move_btn.clicked.connect(
            lambda: self._on_arm_move_clicked("right")
        )
        arm_row2.addWidget(self.right_arm_move_btn)

        pose_info_box = QVBoxLayout()
        pose_info_box.setSpacing(0)
        pose_info_box.setContentsMargins(6, 0, 0, 0)
        self.arm_pose_current_label = QLabel("当前  左臂/右臂 TCP: --")
        self.arm_pose_current_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.arm_pose_current_label.setStyleSheet(f"color: {UI_ACCENT_BLUE};")
        self.arm_pose_target_label = QLabel("目标  TCP: --")
        self.arm_pose_target_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.arm_pose_target_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
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
            "使用分割/FoundationPose 得到的绝对位姿作为手臂目标（base_link），"
            "不是相对当前 TCP 的偏移。FoundationPose 成功后会自动选中此项。"
        )
        target_row.addWidget(self.move_target_relative_radio)
        target_row.addWidget(self.move_target_segment_radio)
        self.offset_x_label = QLabel("前ΔX")
        target_row.addWidget(self.offset_x_label)
        self.offset_x_spin = make_manual_offset_spinbox()
        self.offset_x_spin.setToolTip("沿 base_link X 轴偏移（正=前，负=后）")
        target_row.addWidget(self.offset_x_spin)
        self.offset_y_label = QLabel("左ΔY")
        target_row.addWidget(self.offset_y_label)
        self.offset_y_spin = make_manual_offset_spinbox()
        self.offset_y_spin.setToolTip("沿 base_link Y 轴偏移（正=左，负=右）")
        target_row.addWidget(self.offset_y_spin)
        self.offset_z_label = QLabel("上ΔZ")
        target_row.addWidget(self.offset_z_label)
        self.offset_z_spin = make_manual_offset_spinbox()
        self.offset_z_spin.setToolTip("沿 base_link Z 轴偏移（正=上，负=下）")
        target_row.addWidget(self.offset_z_spin)
        target_row.addStretch()

        self._move_target_group.buttonToggled.connect(self._on_move_target_params_changed)
        for spin in (self.offset_x_spin, self.offset_y_spin, self.offset_z_spin):
            spin.valueChanged.connect(self._on_move_target_params_changed)
        self._update_move_offset_ui_visibility()
        control_layout.addLayout(target_row)
        control_tabs.addTab(control_tab, "手臂/手")

        skeleton_tab = QWidget()
        skeleton_layout = QVBoxLayout(skeleton_tab)
        skeleton_layout.setContentsMargins(8, 6, 8, 6)
        skeleton_layout.setSpacing(6)

        sk_row1 = QHBoxLayout()
        sk_row1.setSpacing(6)
        sk_row1.addWidget(QLabel("相机:"))
        self.skeleton_cam_combo = ImeSafeComboBox()
        self.skeleton_cam_combo.setMinimumWidth(220)
        self.skeleton_cam_combo.setToolTip("选择用于手骨架识别的彩色图像 topic（需先在左侧勾选订阅）")
        sk_row1.addWidget(self.skeleton_cam_combo, 1)
        self.skeleton_refresh_cam_btn = QPushButton("刷新列表")
        self.skeleton_refresh_cam_btn.clicked.connect(self._refresh_skeleton_camera_list)
        sk_row1.addWidget(self.skeleton_refresh_cam_btn)
        skeleton_layout.addLayout(sk_row1)

        sk_row2 = QHBoxLayout()
        sk_row2.setSpacing(8)
        self.skeleton_flip_check = QCheckBox("水平翻转预览")
        self.skeleton_flip_check.setChecked(True)
        self.skeleton_flip_check.setToolTip("自拍/面对相机时建议开启，便于对照屏幕")
        sk_row2.addWidget(self.skeleton_flip_check)
        self.skeleton_mirror_map_check = QCheckBox("镜像映射到机器人")
        self.skeleton_mirror_map_check.setChecked(True)
        self.skeleton_mirror_map_check.setToolTip(
            "开启：人物左手→机器人右手（面对机器人更自然）\n关闭：人物左手→机器人左手"
        )
        sk_row2.addWidget(self.skeleton_mirror_map_check)
        self.skeleton_ctrl_left_check = QCheckBox("控左手")
        self.skeleton_ctrl_left_check.setChecked(True)
        sk_row2.addWidget(self.skeleton_ctrl_left_check)
        self.skeleton_ctrl_right_check = QCheckBox("控右手")
        self.skeleton_ctrl_right_check.setChecked(True)
        sk_row2.addWidget(self.skeleton_ctrl_right_check)
        sk_row2.addStretch()
        skeleton_layout.addLayout(sk_row2)

        sk_row3 = QHBoxLayout()
        sk_row3.setSpacing(6)
        self.skeleton_track_btn = QPushButton("开始识别")
        self.skeleton_track_btn.setToolTip("启动 MediaPipe Hands 识别（仅预览骨架，不发送命令）")
        self.skeleton_track_btn.clicked.connect(self._on_skeleton_track_toggled)
        sk_row3.addWidget(self.skeleton_track_btn)
        self.skeleton_teleop_check = QCheckBox("开启遥控")
        self.skeleton_teleop_check.setToolTip(
            "勾选后将骨架开合量实时发布到 /ry_hand/*/set_angles。\n请先「启用手」，并注意周围安全。"
        )
        self.skeleton_teleop_check.toggled.connect(self._on_skeleton_teleop_toggled)
        sk_row3.addWidget(self.skeleton_teleop_check)
        sk_row3.addWidget(QLabel("平滑"))
        self.skeleton_smooth_slider = QSlider(Qt.Horizontal)
        self.skeleton_smooth_slider.setRange(10, 90)
        self.skeleton_smooth_slider.setValue(35)
        self.skeleton_smooth_slider.setFixedWidth(100)
        self.skeleton_smooth_slider.setToolTip("跟踪平滑：左小右大（越大越跟手、越抖）")
        sk_row3.addWidget(self.skeleton_smooth_slider)
        self.skeleton_status_label = QLabel("手骨架: 未启动")
        self.skeleton_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.skeleton_status_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        sk_row3.addWidget(self.skeleton_status_label, 1)
        skeleton_layout.addLayout(sk_row3)

        self.skeleton_preview_label = QLabel("勾选彩色相机 topic → 刷新列表 → 开始识别")
        self.skeleton_preview_label.setAlignment(Qt.AlignCenter)
        self.skeleton_preview_label.setMinimumHeight(220)
        self.skeleton_preview_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 1px solid #555; color: #aaa; }"
        )
        skeleton_layout.addWidget(self.skeleton_preview_label, 1)

        self.skeleton_joints_label = QLabel("关节: --")
        self.skeleton_joints_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.skeleton_joints_label.setWordWrap(True)
        skeleton_layout.addWidget(self.skeleton_joints_label)

        control_tabs.addTab(skeleton_tab, "手骨架遥控")

        test_tab = QWidget()
        test_outer = QVBoxLayout(test_tab)
        test_outer.setContentsMargins(8, 6, 8, 6)
        test_outer.setSpacing(6)

        test_hint = QLabel(
            "测试区：点击下方场景图，即可附带到右侧「AI 对话」发送"
        )
        test_hint.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        test_hint.setWordWrap(True)
        test_hint.setToolTip(f"图片目录: {TEST_IMAGES_DIR}")
        test_outer.addWidget(test_hint)

        test_grid_host = QWidget()
        test_grid = QGridLayout(test_grid_host)
        test_grid.setContentsMargins(0, 0, 0, 0)
        test_grid.setHorizontalSpacing(8)
        test_grid.setVerticalSpacing(8)
        self._test_image_labels: Dict[str, ScaledPixmapLabel] = {}
        self._test_image_paths: Dict[str, str] = {}
        self._test_selected_scenario: str = ""
        cols = 4
        for idx, title in enumerate(TEST_SCENARIO_LABELS):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL, QFont.Bold))
            title_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
            title_label.setWordWrap(True)
            cell_layout.addWidget(title_label)
            image_label = ScaledPixmapLabel("暂无图像")
            image_path = resolve_test_scenario_image_path(title)
            if image_path is not None:
                pix = QPixmap(image_path)
                if not pix.isNull():
                    image_label.set_source_pixmap(pix)
                    image_label.set_clickable(True)
                    image_label.setToolTip(
                        f"点击选中「{title}」用于对话\n{image_path}"
                    )
                    self._test_image_paths[title] = image_path
                    image_label.clicked.connect(
                        lambda t=title: self._on_test_scenario_clicked(t)
                    )
                else:
                    image_label.setText("加载失败")
                    image_label.setToolTip(f"{title}\n无法读取: {image_path}")
            else:
                expected = TEST_SCENARIO_IMAGE_FILES.get(title, f"{title}.png")
                image_label.setText("暂无图像")
                image_label.setToolTip(
                    f"{title}\n未找到: {os.path.join(TEST_IMAGES_DIR, expected)}"
                )
            cell_layout.addWidget(image_label, 1)
            self._test_image_labels[title] = image_label
            test_grid.addWidget(cell, idx // cols, idx % cols)
        for c in range(cols):
            test_grid.setColumnStretch(c, 1)
        for r in range((len(TEST_SCENARIO_LABELS) + cols - 1) // cols):
            test_grid.setRowStretch(r, 1)
        test_outer.addWidget(test_grid_host, 1)

        qwen_dir = resolve_local_qwen_model_dir()
        service_row = QHBoxLayout()
        service_row.setSpacing(6)
        service_row.addWidget(QLabel("部署位置"))
        self.test_qwen_target_combo = ImeSafeComboBox()
        self.test_qwen_target_combo.addItem("本地本机", "local")
        for host_id, host_label in REMOTE_QWEN_HOSTS:
            self.test_qwen_target_combo.addItem(host_label, f"remote:{host_id}")
        tip_lines = ["本地：本机 GPU 部署。"]
        for host_id, host_label in REMOTE_QWEN_HOSTS:
            prof = remote_qwen_profile(host_id)
            tip_lines.append(
                f"{host_label}：SSH {host_id}，隧道 "
                f"{prof.get('api_base') or remote_qwen_api_base_for_host(host_id)}"
            )
        tip_lines.append("远程适合 Qwen3.5-35B-A3B 等大模型。")
        self.test_qwen_target_combo.setToolTip("\n".join(tip_lines))
        self.test_qwen_target_combo.currentIndexChanged.connect(
            self._on_test_qwen_target_changed
        )
        service_row.addWidget(self.test_qwen_target_combo)
        service_row.addWidget(QLabel("权重目录"))
        self.test_qwen_root_combo = ImeSafeComboBox()
        self.test_qwen_root_combo.setMinimumWidth(140)
        self.test_qwen_root_combo.currentIndexChanged.connect(
            self._on_test_qwen_root_changed
        )
        service_row.addWidget(self.test_qwen_root_combo)
        self.test_qwen_refresh_btn = QPushButton("刷新")
        self.test_qwen_refresh_btn.setToolTip(
            "扫描所选根目录下含 config.json / adapter_config.json 的子目录"
        )
        self.test_qwen_refresh_btn.clicked.connect(self._refresh_test_qwen_model_list)
        service_row.addWidget(self.test_qwen_refresh_btn)
        service_row.addWidget(QLabel("模型"))
        self.test_qwen_model_combo = ImeSafeComboBox()
        self.test_qwen_model_combo.setMinimumWidth(200)
        self.test_qwen_model_combo.setToolTip(
            "从上方根目录扫描到的可部署权重；选中后点「启动推理服务」。"
        )
        self.test_qwen_model_combo.currentIndexChanged.connect(
            self._on_test_qwen_model_changed
        )
        service_row.addWidget(self.test_qwen_model_combo)
        self.test_model_path_label = QLabel("")
        self.test_model_path_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.test_model_path_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.test_model_path_label.setWordWrap(True)
        service_row.addWidget(self.test_model_path_label, 1)
        self.test_qwen_status_label = QLabel("服务: --")
        self.test_qwen_status_label.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        service_row.addWidget(self.test_qwen_status_label)
        self.test_qwen_start_btn = QPushButton("启动推理服务")
        self.test_qwen_start_btn.setToolTip(
            "按「部署位置」启动本地或远程 OpenAI 兼容推理服务。\n"
            "远程会自动 SSH 同步脚本、拉起模型，并建立本地隧道。\n"
            "首次加载可能需要数分钟。"
        )
        self.test_qwen_start_btn.clicked.connect(self._on_test_qwen_start_clicked)
        service_row.addWidget(self.test_qwen_start_btn)
        self.test_qwen_stop_btn = QPushButton("停止")
        self.test_qwen_stop_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
        self.test_qwen_stop_btn.setToolTip("停止当前部署位置对应的推理服务（远程含隧道）")
        self.test_qwen_stop_btn.clicked.connect(self._on_test_qwen_stop_clicked)
        service_row.addWidget(self.test_qwen_stop_btn)
        self.test_qwen_start_btn.setFocusPolicy(Qt.NoFocus)
        self.test_qwen_stop_btn.setFocusPolicy(Qt.NoFocus)
        self.test_qwen_refresh_btn.setFocusPolicy(Qt.NoFocus)
        test_outer.addLayout(service_row)
        self._refresh_test_qwen_model_path_label()


        self.test_infer_log_edit = QTextEdit()
        self.test_infer_log_edit.setReadOnly(True)
        self.test_infer_log_edit.setFont(QFont(UI_MONO_FAMILY, UI_MONO_SIZE_SMALL))
        self.test_infer_log_edit.setMinimumHeight(80)
        self.test_infer_log_edit.setMaximumHeight(140)
        self.test_infer_log_edit.setPlaceholderText("推理服务日志…")
        self.test_infer_log_edit.setStyleSheet(
            f"QTextEdit {{ color: {UI_TEXT_PRIMARY}; background-color: #252525; "
            "border: 1px solid #555; }}"
        )
        test_outer.addWidget(self.test_infer_log_edit)

        self._local_qwen_launcher = LocalQwenServiceLauncher(self)
        self._local_qwen_launcher.status_message.connect(self._on_test_qwen_status)
        self._local_qwen_launcher.log_line.connect(self._append_test_infer_log)
        self._local_qwen_launcher.running_changed.connect(self._update_test_qwen_ui)
        self._remote_qwen_launcher = RemoteQwenServiceLauncher(self)
        self._remote_qwen_launcher.status_message.connect(self._on_test_qwen_status)
        self._remote_qwen_launcher.log_line.connect(self._append_test_infer_log)
        self._remote_qwen_launcher.running_changed.connect(self._update_test_qwen_ui)
        self._deploy_model_list_bridge = DeployModelListBridge(self)
        self._deploy_model_list_bridge.finished.connect(
            self._on_test_qwen_model_list_ready
        )
        self._test_qwen_model_list_refreshing = False
        self._refresh_test_qwen_scan_roots()
        self._refresh_test_qwen_model_list()
        self._qwen_health_timer = QTimer(self)
        self._qwen_health_timer.timeout.connect(self._refresh_test_qwen_status)
        self._qwen_health_timer.start(4000)
        self._update_test_qwen_ui()
        control_tabs.addTab(test_tab, "测试")

        self._hand_skeleton_detector = None
        self._skeleton_tracking = False
        self._skeleton_busy = False
        self._skeleton_timer = QTimer(self)
        self._skeleton_timer.setInterval(66)  # ~15 Hz
        self._skeleton_timer.timeout.connect(self._on_skeleton_tick)

        self._left_arm_move_btn_idle_style = ""
        self._left_arm_move_btn_cancel_style = f"color: {UI_ACCENT_RED};"
        self._right_arm_move_btn_idle_style = ""
        self._right_arm_move_btn_cancel_style = f"color: {UI_ACCENT_RED};"

        root_layout.addWidget(control_tabs)

        bridge.left_hand_preset_changed.connect(self._update_left_hand_toggle_ui)
        bridge.right_hand_preset_changed.connect(self._update_right_hand_toggle_ui)
        bridge.slow_motion_progress.connect(self._on_slow_motion_progress)
        bridge.slow_motion_finished.connect(self._on_slow_motion_finished)
        bridge.robot_state_updated.connect(self._schedule_robot_ui_refresh)
        bridge.arm_enable_changed.connect(self._on_arm_enable_ui_changed)
        bridge.hand_enable_changed.connect(self._on_hand_enable_ui_changed)
        bridge.control_mode_changed.connect(self._on_control_mode_ui_changed)
        self._update_left_hand_toggle_ui(self.node.is_left_hand_at_a())
        self._update_right_hand_toggle_ui(self.node.is_right_hand_at_a())
        self._update_enable_status_ui()
        self._on_arm_move_speed_changed(self.arm_move_speed_slider.value())
        self._update_arm_move_btns_ui(force=True)

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
        chat_layout.setContentsMargins(4, 6, 4, 4)
        chat_layout.setSpacing(2)
        self.chat_panel = ChatPanelWidget(config=llm_config)
        self.chat_panel.set_camera_frame_provider(self._pick_chat_camera_frame)
        self.chat_panel.attached_image_changed.connect(
            self._on_chat_attached_image_changed
        )
        self.chat_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chat_layout.addWidget(self.chat_panel)
        # 给对话区更大默认宽度，并允许拖拽继续加宽
        chat_group.setMinimumWidth(380)
        chat_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._chat_group = chat_group
        self._main_splitter.addWidget(chat_group)

        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 2)
        self._main_splitter.setStretchFactor(2, 3)
        topic_w = 240
        chat_w = max(480, int(win_w * 0.36))
        preview_w = max(480, win_w - topic_w - chat_w - 40)
        self._main_splitter.setSizes([topic_w, preview_w, chat_w])
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
        self._local_ai_launcher.status_message.connect(self.status_bar.showMessage)
        self._local_ai_launcher.running_changed.connect(self._update_local_ai_deploy_btn)
        self._train_launcher.log_line.connect(self._append_train_log)
        self._train_launcher.status_message.connect(self.status_bar.showMessage)
        self._train_launcher.running_changed.connect(self._update_train_ui)
        self._cad_launcher.log_line.connect(self._append_cad_log)
        self._cad_launcher.status_message.connect(self.status_bar.showMessage)
        self._cad_launcher.running_changed.connect(self._update_cad_ui)
        self._cad_launcher.mesh_ready.connect(self._on_cad_mesh_ready)
        bridge.replay_state_changed.connect(self._update_replay_ui)

        bridge.topics_updated.connect(self._on_topics_updated)
        bridge.frame_updated.connect(self._on_frame_updated)
        bridge.status_message.connect(self.status_bar.showMessage)
        bridge.frame_stats.connect(self._on_frame_stats)

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_waiting_hint)
        self._ui_timer.timeout.connect(self._update_robot_enable_status_ui)
        self._ui_timer.start(2000)
        self._sam3_health_timer = QTimer(self)
        self._sam3_health_timer.timeout.connect(self._refresh_sam3_status)
        self._sam3_health_timer.start(5000)
        self._fp_health_timer = QTimer(self)
        self._fp_health_timer.timeout.connect(self._refresh_fp_status)
        self._fp_health_timer.start(5000)
        self._on_segment_settings_changed()
        self._on_pose_settings_changed()
        self._received_topics: Dict[str, int] = {}
        robot, base = self._stack_launcher.get_cached_status()
        self._update_robot_stack_ui(robot, base)
        self._update_replay_ui()
        self._update_local_ai_deploy_btn()
        self._update_train_ui()
        self._update_cad_ui()

    def _get_psi_policy_dir(self) -> Optional[str]:
        if self._psi_policy_dir_override:
            return resolve_psi_policy_dir(self._psi_policy_dir_override)
        return resolve_psi_policy_dir()

    def _update_train_path_ui(self) -> None:
        repo = self._get_psi_policy_dir()
        if repo:
            self.train_policy_dir_edit.setText(repo)
            self.train_policy_dir_edit.setToolTip(repo)
        else:
            self.train_policy_dir_edit.clear()
            self.train_policy_dir_edit.setToolTip("未找到 psi-policy 仓库")
        self.train_status_label.setToolTip(
            f"psi-policy 目录: {repo or '(未设置)'}\n训练日志实时显示在下方"
        )
        running = self._train_launcher.is_running()
        self.train_start_btn.setEnabled(not running and repo is not None)
        self.train_browse_policy_btn.setEnabled(not running)

    def _refresh_train_config_combo(self) -> None:
        repo = self._get_psi_policy_dir()
        current = str(self.train_config_combo.currentData() or PSI_POLICY_CONFIG_DEFAULT)
        self.train_config_combo.blockSignals(True)
        self.train_config_combo.clear()
        configs = list_psi_policy_workspace_configs(repo) if repo else []
        if not configs:
            configs = [PSI_POLICY_CONFIG_DEFAULT]
        for cfg_name in configs:
            self.train_config_combo.addItem(cfg_name, cfg_name)
        idx = self.train_config_combo.findData(current)
        if idx < 0:
            idx = self.train_config_combo.findData(PSI_POLICY_CONFIG_DEFAULT)
        if idx >= 0:
            self.train_config_combo.setCurrentIndex(idx)
        self.train_config_combo.blockSignals(False)

    def _on_train_browse_policy_clicked(self) -> None:
        if self._train_launcher.is_running():
            return
        current = self._get_psi_policy_dir()
        initial = current if current and os.path.isdir(current) else os.path.expanduser("~")
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 psi-policy 仓库目录",
            initial,
        )
        if not selected:
            return
        if not is_valid_psi_policy_dir(selected):
            QMessageBox.warning(
                self,
                "无效目录",
                f"所选目录不是有效的 psi-policy 仓库：\n{selected}\n\n"
                "请确认目录下存在 psi_policy/train.py",
            )
            return
        self._psi_policy_dir_override = os.path.abspath(selected)
        self._refresh_train_config_combo()
        self._update_train_path_ui()
        self.status_bar.showMessage(f"已选择 psi-policy: {self._psi_policy_dir_override}")

    def _append_train_log(self, line: str) -> None:
        self.train_log_edit.append(line)
        scrollbar = self.train_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_train_ui(self, *_args) -> None:
        running = self._train_launcher.is_running()
        self.train_stop_btn.setEnabled(running)
        if running:
            self.train_status_label.setText("训练: 运行中")
            self.train_status_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        else:
            self.train_status_label.setText("训练: 空闲")
            self.train_status_label.setStyleSheet("")
        self._update_train_path_ui()

    def _on_train_start_clicked(self) -> None:
        config_name = str(self.train_config_combo.currentData() or PSI_POLICY_CONFIG_DEFAULT)
        self._train_launcher.start(
            config_name,
            self.train_datahouse_edit.text(),
            self.train_view_edit.text(),
            repo_dir=self._get_psi_policy_dir(),
            num_epochs=self.train_epochs_spin.value(),
            batch_size=self.train_batch_spin.value(),
            logging_mode=str(self.train_logging_combo.currentData() or "offline"),
            num_processes=self.train_gpu_spin.value(),
        )
        self._update_train_ui()

    def _on_train_stop_clicked(self) -> None:
        self._train_launcher.stop()

    def _on_train_clear_log_clicked(self) -> None:
        self.train_log_edit.clear()

    def _on_test_scenario_clicked(self, title: str) -> None:
        path = self._test_image_paths.get(title, "")
        if not path:
            self.status_bar.showMessage(f"场景「{title}」无可用图片")
            return
        if self.chat_panel.set_attached_image_from_path(path, display_name=title):
            self._set_test_scenario_selection(title)
            self.status_bar.showMessage(f"已选场景图用于对话: {title}")

    def _set_test_scenario_selection(self, title: str) -> None:
        self._test_selected_scenario = title
        for name, label in self._test_image_labels.items():
            label.set_selected(name == title)

    def _on_chat_attached_image_changed(self, path: str) -> None:
        if not path:
            self._set_test_scenario_selection("")
            return
        for name, p in self._test_image_paths.items():
            if os.path.abspath(p) == os.path.abspath(path):
                self._set_test_scenario_selection(name)
                return

    def _append_test_infer_log(self, line: str) -> None:
        self.test_infer_log_edit.append(line)
        scrollbar = self.test_infer_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _selected_test_qwen_deploy_spec(self) -> Dict[str, str]:
        data = self.test_qwen_model_combo.currentData()
        if isinstance(data, dict):
            path = str(data.get("path") or "").strip()
            if path:
                return {
                    "path": path,
                    "model_id": str(data.get("model_id") or os.path.basename(path.rstrip("/"))),
                    "label": str(data.get("label") or data.get("name") or os.path.basename(path.rstrip("/"))),
                    "kind": str(data.get("kind") or ""),
                    "root": str(data.get("root") or ""),
                }
        return {}

    def _selected_test_qwen_model_key(self) -> str:
        spec = self._selected_test_qwen_deploy_spec()
        if spec.get("path"):
            return str(spec.get("model_id") or spec.get("path"))
        key = self.test_qwen_model_combo.currentData()
        if isinstance(key, str) and key:
            return key
        return LOCAL_QWEN_DEPLOY_MODELS[0][0]

    def _selected_test_qwen_scan_root(self) -> str:
        root = self.test_qwen_root_combo.currentData()
        if isinstance(root, str) and root.strip():
            return root.strip()
        return ""

    def _refresh_test_qwen_scan_roots(self) -> None:
        target = self._selected_test_qwen_target()
        prev = self._selected_test_qwen_scan_root()
        self.test_qwen_root_combo.blockSignals(True)
        self.test_qwen_root_combo.clear()
        if target == "local":
            roots = local_deploy_scan_roots()
        else:
            roots = remote_deploy_scan_roots(self._selected_test_qwen_remote_host())
        for path, label in roots:
            self.test_qwen_root_combo.addItem(label, path)
            idx = self.test_qwen_root_combo.count() - 1
            self.test_qwen_root_combo.setItemData(idx, path, Qt.ToolTipRole)
        if prev:
            idx = self.test_qwen_root_combo.findData(prev)
            if idx >= 0:
                self.test_qwen_root_combo.setCurrentIndex(idx)
        self.test_qwen_root_combo.blockSignals(False)

    def _on_test_qwen_root_changed(self, _index: int = 0) -> None:
        self._refresh_test_qwen_model_list()

    def _refresh_test_qwen_model_list(self) -> None:
        if self._test_qwen_model_list_refreshing:
            return
        root = self._selected_test_qwen_scan_root()
        if not root:
            return
        self._test_qwen_model_list_refreshing = True
        # 禁用前把焦点还给文本框，避免 focused+disabled 弄死 fcitx
        fw = QApplication.focusWidget()
        if fw is self.test_qwen_model_combo or (
            fw is not None and self.test_qwen_model_combo.isAncestorOf(fw)
        ):
            target = _IME_LAST_TEXT_WIDGET
            if target is not None and _is_text_ime_widget(target):
                target.setFocus(Qt.OtherFocusReason)
            else:
                self.setFocus(Qt.OtherFocusReason)
        self.test_qwen_refresh_btn.setEnabled(False)
        self.test_qwen_model_combo.setEnabled(False)
        _schedule_fcitx_restore(_IME_LAST_TEXT_WIDGET)
        target = self._selected_test_qwen_target()
        host_id = self._selected_test_qwen_remote_host()
        self._append_test_infer_log(f"扫描可部署目录: {root} …")

        def _work() -> None:
            ok, models, err = fetch_deploy_model_dirs(
                target=target,
                host_id=host_id,
                root=root,
            )
            self._deploy_model_list_bridge.finished.emit(
                {
                    "ok": ok,
                    "models": models,
                    "message": err,
                    "root": root,
                    "target": target,
                }
            )

        threading.Thread(target=_work, daemon=True).start()

    def _on_test_qwen_model_list_ready(self, payload: object) -> None:
        self._test_qwen_model_list_refreshing = False
        data = payload if isinstance(payload, dict) else {}
        ok = bool(data.get("ok"))
        models = data.get("models") if isinstance(data.get("models"), list) else []
        root = str(data.get("root") or self._selected_test_qwen_scan_root())
        prev_path = str(self._selected_test_qwen_deploy_spec().get("path") or "")

        self.test_qwen_model_combo.blockSignals(True)
        self.test_qwen_model_combo.clear()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            name = str(entry.get("name") or os.path.basename(path.rstrip("/")))
            kind = str(entry.get("kind") or "")
            kind_label = deploy_model_kind_label(kind)
            item_label = f"{name} ({kind_label})"
            spec = {
                "path": path,
                "name": name,
                "model_id": str(entry.get("model_id") or name),
                "label": str(entry.get("label") or name),
                "kind": kind,
                "root": str(entry.get("root") or root),
            }
            self.test_qwen_model_combo.addItem(item_label, spec)
            idx = self.test_qwen_model_combo.count() - 1
            tip = (
                f"{name}\nmodel_id={spec['model_id']}\n"
                f"路径: {path}\n类型: {kind_label}\n根目录: {spec['root']}"
            )
            self.test_qwen_model_combo.setItemData(idx, tip, Qt.ToolTipRole)
        if prev_path:
            for i in range(self.test_qwen_model_combo.count()):
                spec = self.test_qwen_model_combo.itemData(i)
                if isinstance(spec, dict) and spec.get("path") == prev_path:
                    self.test_qwen_model_combo.setCurrentIndex(i)
                    break
        elif self.test_qwen_model_combo.count() > 0:
            prefer = self.test_qwen_model_combo.findText(
                "Qwen3.5-35B-A3B", Qt.MatchContains
            )
            self.test_qwen_model_combo.setCurrentIndex(
                prefer if prefer >= 0 else 0
            )
        self.test_qwen_model_combo.blockSignals(False)

        if ok:
            self._append_test_infer_log(
                f"已加载 {len(models)} 个可部署目录（{root}）"
            )
        else:
            msg = str(data.get("message") or "扫描失败")
            self._append_test_infer_log(f"目录扫描失败: {msg}")
            self.status_bar.showMessage(f"目录扫描失败: {msg}")
        self._refresh_test_qwen_model_path_label()
        self._update_test_qwen_ui()

    def _selected_test_qwen_target(self) -> str:
        target = str(self.test_qwen_target_combo.currentData() or "local")
        if target == "local":
            return "local"
        if target.startswith("remote:") or target == "remote":
            return "remote"
        return "local"

    def _selected_test_qwen_remote_host(self) -> str:
        target = str(self.test_qwen_target_combo.currentData() or "")
        if target.startswith("remote:"):
            return target.split(":", 1)[1].strip() or REMOTE_QWEN_SSH_HOST
        return REMOTE_QWEN_SSH_HOST

    def _on_test_qwen_target_changed(self, _index: int = 0) -> None:
        if self._selected_test_qwen_target() == "remote":
            self._remote_qwen_launcher.set_host(self._selected_test_qwen_remote_host())
        self._refresh_test_qwen_scan_roots()
        self._refresh_test_qwen_model_list()
        self._update_test_qwen_ui()

    def _on_test_qwen_model_changed(self, _index: int = 0) -> None:
        self._refresh_test_qwen_model_path_label()

    def _refresh_test_qwen_model_path_label(self) -> None:
        spec = self._selected_test_qwen_deploy_spec()
        target = self._selected_test_qwen_target()
        if spec.get("path"):
            path = spec["path"]
            label = spec.get("label") or os.path.basename(path.rstrip("/"))
            mid = spec.get("model_id") or label
            kind = deploy_model_kind_label(str(spec.get("kind") or ""))
            if target == "remote":
                host_id = self._selected_test_qwen_remote_host()
                prof = remote_qwen_profile(host_id)
                py = resolve_deploy_python_for_spec(
                    spec, target="remote", host_id=host_id
                )
                api = str(prof.get("api_base") or remote_qwen_api_base_for_host(host_id))
                self.test_model_path_label.setText(
                    f"{host_id}:{path}  ·  id={mid}"
                )
                self.test_model_path_label.setStyleSheet(
                    f"color: {UI_TEXT_SECONDARY};"
                )
                self.test_model_path_label.setToolTip(
                    f"远程部署 {label}（{kind}）\nSSH Host: {host_id}\n"
                    f"model_id={mid}\n路径: {path}\n"
                    f"Python: {py}\n"
                    f"工作目录: {prof.get('remote_work')}\n"
                    f"隧道 API: {api}"
                )
            else:
                self.test_model_path_label.setText(f"{path}  ·  id={mid}")
                self.test_model_path_label.setStyleSheet(
                    f"color: {UI_TEXT_SECONDARY};"
                )
                self.test_model_path_label.setToolTip(
                    f"本地部署 {label}（{kind}）\nmodel_id={mid}\n"
                    f"路径: {path}\n"
                    f"API: {LOCAL_QWEN_API_BASE_DEFAULT}"
                )
            return

        key = self._selected_test_qwen_model_key()
        label = local_qwen_label_for_key(key)
        mid = local_qwen_model_id_for_key(key)
        dirname = qwen_deploy_path_spec_for_key(key)
        if target == "remote":
            host_id = self._selected_test_qwen_remote_host()
            prof = remote_qwen_profile(host_id)
            remote_root = str(prof.get("model_root") or "")
            remote_path = qwen_model_path_from_spec(dirname, root=remote_root)
            py = remote_qwen_python_for_key(key) or str(prof.get("python") or "")
            api = str(prof.get("api_base") or remote_qwen_api_base_for_host(host_id))
            kind = "LoRA" if qwen_model_is_lora(remote_path) else "全量"
            self.test_model_path_label.setText(
                f"{host_id}:{remote_path}  ·  id={mid}"
            )
            self.test_model_path_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
            self.test_model_path_label.setToolTip(
                f"远程部署 {label}（{kind}）\nSSH Host: {host_id}\n"
                f"model_id={mid}\n路径: {remote_path}\n"
                f"Python: {py}\n"
                f"工作目录: {prof.get('remote_work')}\n"
                f"隧道 API: {api}"
            )
            return

        path = local_qwen_model_dir_for_key(key)
        missing = qwen_model_path_from_spec(dirname)
        if path:
            self.test_model_path_label.setText(f"{path}  ·  id={mid}")
            self.test_model_path_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        else:
            self.test_model_path_label.setText(f"未找到权重: {missing}")
            self.test_model_path_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        self.test_model_path_label.setToolTip(
            f"本地部署 {label}\nmodel_id={mid}\n"
            f"路径: {path or missing}\n"
            f"API: {LOCAL_QWEN_API_BASE_DEFAULT}"
        )

    def _on_test_qwen_status(self, msg: str) -> None:
        self.test_qwen_status_label.setText(f"服务: {msg}")
        self._append_test_infer_log(msg)
        self.status_bar.showMessage(msg)

    def _refresh_test_qwen_status(self) -> None:
        self._update_test_qwen_ui()

    def _update_test_qwen_ui(self, *_args) -> None:
        target = self._selected_test_qwen_target()
        if target == "remote":
            self._update_test_qwen_ui_remote()
        else:
            self._update_test_qwen_ui_local()

    def _update_test_qwen_ui_remote(self) -> None:
        host_id = self._selected_test_qwen_remote_host()
        self._remote_qwen_launcher.set_host(host_id)
        api = self._remote_qwen_launcher.api_base()
        info = fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        was_starting = bool(getattr(self._remote_qwen_launcher, "_starting", False))
        starting = was_starting and not healthy
        if healthy:
            self._remote_qwen_launcher._starting = False
            starting = False
            if was_starting:
                mid = str(
                    (info or {}).get("model")
                    or self._remote_qwen_launcher.last_model_id()
                )
                self.chat_panel.apply_remote_qwen_service_preset(
                    api_base=api, model_id=mid, silent=False
                )
                self._append_test_infer_log(f"远程 Qwen 已就绪: {api} ({mid})")
        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            self.test_qwen_status_label.setText(f"服务: 远程在线 {api}{suffix}")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        elif starting:
            label = self._remote_qwen_launcher.last_model_label()
            self.test_qwen_status_label.setText(
                f"服务: 远程启动中（{label} @ {host_id}）…"
            )
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        else:
            self.test_qwen_status_label.setText(f"服务: 远程离线（{host_id}）")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.test_qwen_start_btn.setEnabled(not healthy and not starting)
        self.test_qwen_stop_btn.setEnabled(healthy or starting)
        service_busy = healthy or starting
        scanning = bool(self._test_qwen_model_list_refreshing)
        # 仅扫描目录时保持「部署位置/权重目录」可点
        self.test_qwen_model_combo.setEnabled(not service_busy and not scanning)
        self.test_qwen_target_combo.setEnabled(not service_busy)
        self.test_qwen_root_combo.setEnabled(not service_busy)
        self.test_qwen_refresh_btn.setEnabled(not service_busy and not scanning)

    def _update_test_qwen_ui_local(self) -> None:
        api = self._local_qwen_launcher.api_base()
        info = fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        hostctl_running = False
        hostctl_starting = False
        hostctl_error = ""
        if check_local_qwen_hostctl_health():
            _ok, body = local_qwen_hostctl_request("/status", method="GET", timeout_s=2.0)
            if _ok:
                hostctl_running = bool(body.get("process_running"))
                hostctl_starting = bool(body.get("starting")) or (
                    hostctl_running and not healthy
                )
                hostctl_error = str(body.get("last_error") or "").strip()

        if healthy:
            self._local_qwen_launcher._starting = False
        elif getattr(self._local_qwen_launcher, "_starting", False):
            local_proc_alive = (
                self._local_qwen_launcher._process is not None
                and self._local_qwen_launcher._process.state() == QProcess.Running
            )
            if self._local_qwen_launcher._via_hostctl:
                if not hostctl_running and not healthy:
                    self._local_qwen_launcher._starting = False
                    self._local_qwen_launcher._via_hostctl = False
                    if not self._local_qwen_launcher._fail_notified:
                        self._local_qwen_launcher._fail_notified = True
                        err = hostctl_error or (
                            "推理进程已退出，请查看 log/local_qwen_service.log"
                        )
                        self._append_test_infer_log(f"启动失败: {err}")
                        self.status_bar.showMessage("本地 Qwen 启动失败")
            elif not local_proc_alive:
                self._local_qwen_launcher._starting = False

        starting = bool(getattr(self._local_qwen_launcher, "_starting", False)) and not healthy
        if (
            self._local_qwen_launcher._process is not None
            and self._local_qwen_launcher._process.state() == QProcess.Running
            and not healthy
        ):
            starting = True
        if hostctl_starting:
            starting = True
            self._local_qwen_launcher._starting = True

        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            self.test_qwen_status_label.setText(f"服务: 在线 {api}{suffix}")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        elif starting:
            label = self._local_qwen_launcher.last_model_label()
            self.test_qwen_status_label.setText(f"服务: 启动中（加载 {label}）…")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        elif hostctl_error and not healthy:
            short = hostctl_error if len(hostctl_error) <= 80 else hostctl_error[:77] + "…"
            self.test_qwen_status_label.setText(f"服务: 启动失败 · {short}")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_ACCENT_RED};")
            self.test_qwen_status_label.setToolTip(hostctl_error)
        else:
            self.test_qwen_status_label.setText("服务: 离线")
            self.test_qwen_status_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        self.test_qwen_start_btn.setEnabled(not healthy and not starting)
        self.test_qwen_stop_btn.setEnabled(healthy or starting)
        service_busy = healthy or starting
        scanning = bool(self._test_qwen_model_list_refreshing)
        self.test_qwen_model_combo.setEnabled(not service_busy and not scanning)
        self.test_qwen_target_combo.setEnabled(not service_busy)
        self.test_qwen_root_combo.setEnabled(not service_busy)
        self.test_qwen_refresh_btn.setEnabled(not service_busy and not scanning)

    def _on_test_qwen_start_clicked(self) -> None:
        spec = self._selected_test_qwen_deploy_spec()
        key = self._selected_test_qwen_model_key()
        target = self._selected_test_qwen_target()
        self._qwen_switch_tries = 0
        if not spec.get("path") and self.test_qwen_model_combo.count() == 0:
            self._append_test_infer_log("请先点「刷新」加载可部署目录，并选择一个模型。")
            self.status_bar.showMessage("未选择模型")
            return
        if target == "remote":
            host_id = self._selected_test_qwen_remote_host()
            path = spec.get("path") or ""
            if path and LAKE_QWEN35_OUTPUT_ROOT in path and host_id != "psi_motus_2_for_liyichao":
                self._append_test_infer_log(
                    "Lake 训练 output 目前仅在远程 psi_motus 可访问。"
                )
                self.status_bar.showMessage("请切换部署位置为 psi_motus")
                return
            if path and not path.startswith("/share_data") and host_id == "tione-develop":
                self._append_test_infer_log(
                    "提示: 所选目录若不在 tione-develop 本机/共享盘，部署可能失败。"
                )
            api = remote_qwen_api_base_for_host(host_id)
            label = spec.get("label") or key
            self._append_test_infer_log(
                f"远程部署: {host_id} / {label}，隧道 {api}"
            )
            self._remote_qwen_launcher.start(
                model_key=key,
                host_id=host_id,
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
            self._update_test_qwen_ui()
            QTimer.singleShot(2500, self._try_switch_chat_to_remote_qwen)
            return

        path = spec.get("path") or local_qwen_model_dir_for_key(key)
        if spec.get("path"):
            if key.endswith("-a3b") or "35b" in key.lower():
                self._append_test_infer_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self._local_qwen_launcher.start(
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
        else:
            if key == "qwen3.5-35b-a3b":
                self._append_test_infer_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self._local_qwen_launcher.start(model_key=key)
        self._update_test_qwen_ui()
        QTimer.singleShot(2000, self._try_switch_chat_to_local_qwen)

    def _try_switch_chat_to_local_qwen(self) -> None:
        if self._selected_test_qwen_target() != "local":
            return
        api = self._local_qwen_launcher.api_base()
        info = fetch_local_qwen_server_info(api)
        if info and info.get("ok"):
            mid = str(info.get("model") or self._local_qwen_launcher.last_model_id())
            self.chat_panel.apply_local_qwen_service_preset(
                api_base=api, model_id=mid, silent=False
            )
            self._update_test_qwen_ui()
            return

        if check_local_qwen_hostctl_health():
            _ok, body = local_qwen_hostctl_request("/status", method="GET", timeout_s=2.0)
            if _ok and not body.get("process_running") and not body.get("service_healthy"):
                self._local_qwen_launcher._starting = False
                if not self._local_qwen_launcher._fail_notified:
                    self._local_qwen_launcher._fail_notified = True
                    err = str(body.get("last_error") or "推理进程已退出")
                    self._append_test_infer_log(f"启动失败: {err}")
                self._update_test_qwen_ui()
                return

        self._qwen_switch_tries = getattr(self, "_qwen_switch_tries", 0) + 1
        if self._qwen_switch_tries < 180:
            QTimer.singleShot(2000, self._try_switch_chat_to_local_qwen)
        else:
            self._local_qwen_launcher._starting = False
            self._append_test_infer_log(
                "启动超时：模型仍未就绪，请查看 eai/log/local_qwen_service.log"
            )
        self._update_test_qwen_ui()

    def _try_switch_chat_to_remote_qwen(self) -> None:
        if self._selected_test_qwen_target() != "remote":
            return
        api = self._remote_qwen_launcher.api_base()
        info = fetch_local_qwen_server_info(api)
        if info and info.get("ok"):
            self._remote_qwen_launcher._starting = False
            mid = str(info.get("model") or self._remote_qwen_launcher.last_model_id())
            self.chat_panel.apply_remote_qwen_service_preset(
                api_base=api, model_id=mid, silent=False
            )
            self._update_test_qwen_ui()
            return

        if (
            not getattr(self._remote_qwen_launcher, "_starting", False)
            and getattr(self._remote_qwen_launcher, "_fail_notified", False)
        ):
            self._update_test_qwen_ui()
            return

        self._qwen_switch_tries = getattr(self, "_qwen_switch_tries", 0) + 1
        # 远程 35B 多卡加载可能更久
        if self._qwen_switch_tries < 300:
            QTimer.singleShot(2000, self._try_switch_chat_to_remote_qwen)
        else:
            self._remote_qwen_launcher._starting = False
            host_id = self._selected_test_qwen_remote_host()
            prof = remote_qwen_profile(host_id)
            log_hint = str(prof.get("remote_work") or "") + "/local_qwen_service.log"
            self._append_test_infer_log(
                f"远程启动超时（{host_id}）：请查看远程日志 {log_hint}"
            )
        self._update_test_qwen_ui()

    def _on_test_qwen_stop_clicked(self) -> None:
        if self._selected_test_qwen_target() == "remote":
            self._remote_qwen_launcher.stop(
                host_id=self._selected_test_qwen_remote_host()
            )
        else:
            self._local_qwen_launcher.stop()
        self._update_test_qwen_ui()

    def _on_cad_mode_changed(self, *_args) -> None:
        mode = str(self.cad_mode_combo.currentData() or "photos")
        if mode == "mesh":
            self.cad_src_label.setText("Mesh 文件:")
            self.cad_src_edit.setPlaceholderText("选择 .obj / .ply / .glb / .stl")
            self.cad_poisson_spin.setEnabled(False)
            self.cad_capture_btn.setEnabled(False)
            self.cad_capture_reset_btn.setEnabled(False)
        else:
            self.cad_src_label.setText("照片目录:")
            self.cad_src_edit.setPlaceholderText("选择多视角照片目录，或用下方「拍一张」采集")
            self.cad_poisson_spin.setEnabled(True)
            self._update_cad_ui()

    def _cad_photos_dir_for_name(self, name: Optional[str] = None) -> str:
        obj = (name or self.cad_name_edit.text() or "my_object").strip().replace(" ", "_")
        for ch in '/\\:*?"<>|':
            obj = obj.replace(ch, "_")
        if not obj:
            obj = "my_object"
        return os.path.join(CAD_MESHES_DIR, obj, "photos")

    def _count_photos_in_dir(self, photos_dir: str) -> int:
        if not os.path.isdir(photos_dir):
            return 0
        n = 0
        for name in os.listdir(photos_dir):
            lower = name.lower()
            if "_depth" in lower:
                continue
            if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                n += 1
        return n

    def _sync_cad_capture_from_disk(self) -> None:
        photos_dir = self._cad_photos_dir_for_name()
        self._cad_capture_dir = photos_dir
        self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
        if self._cad_capture_saved > 0 and str(
            self.cad_mode_combo.currentData() or "photos"
        ) == "photos":
            self.cad_src_edit.setText(photos_dir)
        self._update_cad_capture_btn()

    def _update_cad_capture_btn(self, *_args) -> None:
        target = int(self.cad_capture_target_spin.value())
        saved = int(self._cad_capture_saved)
        self.cad_capture_btn.setText(f"拍一张 ({saved}/{target})")
        if saved >= target:
            self.cad_capture_hint.setText(f"已满 {target} 张，可点「生成 CAD」或勾选自动生成")
            self.cad_capture_hint.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        elif saved > 0:
            self.cad_capture_hint.setText(
                f"已拍 {saved} 张 — 请换视角后再拍（相邻约 60% 重叠）"
            )
            self.cad_capture_hint.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        else:
            self.cad_capture_hint.setText(
                "头部 RGB-D：每拍一张请转动物体，视角差太小会被拒绝"
            )
            self.cad_capture_hint.setStyleSheet(f"color: {UI_TEXT_MUTED};")

    def _cad_frame_too_similar(self, bgr: np.ndarray) -> Tuple[bool, float, float]:
        """与上一张比较，返回 (太相似?, corr, mae)。"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        gray = cv2.resize(gray, (320, 200), interpolation=cv2.INTER_AREA)
        if self._cad_last_capture_gray is None:
            return False, 0.0, 999.0
        a = self._cad_last_capture_gray.astype(np.float32)
        b = gray.astype(np.float32)
        mae = float(np.mean(np.abs(a - b)))
        aa = (a - a.mean()).ravel()
        bb = (b - b.mean()).ravel()
        corr = float((aa @ bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-9))
        too_similar = corr >= CAD_CAPTURE_MIN_DIFF_CORR or (
            corr >= 0.96 and mae < CAD_CAPTURE_MIN_DIFF_MAE
        )
        return too_similar, corr, mae

    def _ensure_head_rgbd_subscribed(self) -> Tuple[Optional[str], Optional[str]]:
        """确保头部彩色+深度已勾选，返回 (color_topic, depth_topic)。"""
        known = list(self._topic_types.keys()) or list(self.topic_checks.keys())
        color = resolve_head_color_topic(known) or CAD_CAPTURE_TOPIC_DEFAULT
        depth = find_paired_depth_topic(color, known) or CAD_CAPTURE_DEPTH_DEFAULT

        changed = False
        for topic in (color, depth):
            checkbox = self.topic_checks.get(topic)
            if checkbox is not None and not checkbox.isChecked():
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                changed = True
                self._append_cad_log(f"已自动勾选: {topic}")
        if changed:
            enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
            self._apply_selection(enabled)
        return color, depth

    def _pick_head_color_source(
        self,
    ) -> Optional[Tuple[str, np.ndarray, Optional[CameraPanel]]]:
        """CAD 采集专用：只取头部彩色相机画面。"""
        color_topic, _depth_topic = self._ensure_head_rgbd_subscribed()
        if color_topic is None:
            return None

        panel = self.panels.get(color_topic)
        if isinstance(panel, CameraPanel) and panel._latest_image is not None:
            return color_topic, panel._latest_image.copy(), panel

        image = self._frame_cache.get(color_topic)
        if image is not None and np.asarray(image).ndim >= 2:
            return (
                color_topic,
                np.asarray(image).copy(),
                panel if isinstance(panel, CameraPanel) else None,
            )

        for name, cached in self._frame_cache.items():
            if not is_head_color_topic(name):
                continue
            if cached is None or np.asarray(cached).ndim < 2:
                continue
            p = self.panels.get(name)
            return (
                name,
                np.asarray(cached).copy(),
                p if isinstance(p, CameraPanel) else None,
            )
        return None

    def _pick_head_depth_frame(self, color_topic: str) -> Optional[np.ndarray]:
        known = list(self._frame_cache.keys()) + list(self.topic_checks.keys())
        depth_topic = find_paired_depth_topic(color_topic, known) or CAD_CAPTURE_DEPTH_DEFAULT
        depth = self._frame_cache.get(depth_topic)
        if depth is None:
            panel = self.panels.get(depth_topic)
            if isinstance(panel, DepthPanel3D) and getattr(panel, "_latest_depth", None) is not None:
                depth = panel._latest_depth
        if depth is None:
            return None
        arr = np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    def _save_cad_intrinsics(self, photos_dir: str, color_topic: str, width: int, height: int) -> None:
        depth_topic = find_paired_depth_topic(
            color_topic, list(self._frame_cache.keys()) + list(self.topic_checks.keys())
        ) or CAD_CAPTURE_DEPTH_DEFAULT
        fx, fy, cx, cy = self.node.get_intrinsics(depth_topic, width, height)
        # 若只有 color 的 CameraInfo，再试 color
        if abs(fx - 0.9 * max(width, height)) < 1e-3:
            fx, fy, cx, cy = self.node.get_intrinsics(color_topic, width, height)
        meta = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
            "depth_scale": 1000.0,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
        }
        path = os.path.join(photos_dir, "intrinsics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _ensure_cad_capture_session(self) -> str:
        photos_dir = self._cad_photos_dir_for_name()
        if self._cad_capture_dir != photos_dir:
            self._cad_capture_dir = photos_dir
            self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
            self._cad_last_capture_gray = None
        os.makedirs(photos_dir, exist_ok=True)
        return photos_dir

    def _on_cad_capture_clicked(self) -> None:
        if self._cad_launcher.is_running():
            self.status_bar.showMessage("CAD 重建进行中，请稍候再拍")
            return
        if str(self.cad_mode_combo.currentData() or "photos") != "photos":
            self.cad_mode_combo.setCurrentIndex(self.cad_mode_combo.findData("photos"))

        source = self._pick_head_color_source()
        if source is None:
            QMessageBox.information(
                self,
                "CAD 采集",
                "未收到头部相机画面。\n\n"
                f"请确认已发布 {CAD_CAPTURE_TOPIC_DEFAULT}，\n"
                "并在左侧勾选该 topic 后等待图像到达。",
            )
            return

        topic, image, _panel = source
        bgr = np.asarray(image)
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        too_similar, corr, mae = self._cad_frame_too_similar(bgr)
        if too_similar:
            msg = (
                f"视角变化太小（corr={corr:.3f}, mae={mae:.1f}），未保存。\n"
                "请转动物体（或移动头部）后再拍一张。"
            )
            self._append_cad_log(msg.replace("\n", " "))
            self.status_bar.showMessage(msg.split("\n")[0])
            self.cad_capture_hint.setText(msg.split("\n")[0])
            self.cad_capture_hint.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
            return

        depth = self._pick_head_depth_frame(topic)
        if depth is None:
            QMessageBox.information(
                self,
                "CAD 采集",
                "未收到头部深度图。\n\n"
                f"请确认已发布并勾选 {CAD_CAPTURE_DEPTH_DEFAULT}。\n"
                "RGB-D 重建需要彩色+深度。",
            )
            return

        photos_dir = self._ensure_cad_capture_session()
        idx = self._cad_capture_saved + 1
        color_path = os.path.join(photos_dir, f"img_{idx:04d}.jpg")
        depth_path = os.path.join(photos_dir, f"img_{idx:04d}_depth.png")
        while os.path.isfile(color_path):
            idx += 1
            color_path = os.path.join(photos_dir, f"img_{idx:04d}.jpg")
            depth_path = os.path.join(photos_dir, f"img_{idx:04d}_depth.png")

        # 深度对齐到彩色分辨率
        depth_arr = np.asarray(depth)
        if depth_arr.shape[:2] != bgr.shape[:2]:
            depth_arr = cv2.resize(
                depth_arr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if depth_arr.dtype == np.float32 or depth_arr.dtype == np.float64:
            # 米 → mm
            depth_u16 = np.clip(depth_arr * 1000.0, 0, 65535).astype(np.uint16)
        else:
            depth_u16 = depth_arr.astype(np.uint16)

        ok_c = cv2.imwrite(color_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok_d = cv2.imwrite(depth_path, depth_u16)
        if not ok_c or not ok_d:
            QMessageBox.warning(self, "CAD 采集", f"保存失败: {color_path}")
            return

        self._save_cad_intrinsics(photos_dir, topic, bgr.shape[1], bgr.shape[0])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._cad_last_capture_gray = cv2.resize(gray, (320, 200), interpolation=cv2.INTER_AREA)
        self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
        self.cad_src_edit.setText(photos_dir)
        self._update_cad_capture_btn()
        self._append_cad_log(
            f"头部 RGB-D [{self._cad_capture_saved}/{int(self.cad_capture_target_spin.value())}] "
            f"{os.path.basename(color_path)} + depth  ← {topic}  "
            f"({bgr.shape[1]}x{bgr.shape[0]}, Δcorr={corr:.3f})"
        )
        self.status_bar.showMessage(
            f"头部 RGB-D 已拍 {self._cad_capture_saved}/{int(self.cad_capture_target_spin.value())} 张"
        )

        target = int(self.cad_capture_target_spin.value())
        if self._cad_capture_saved >= target and self.cad_auto_gen_check.isChecked():
            self._append_cad_log(f"已满 {target} 张，开始自动生成 CAD（RGB-D TSDF）…")
            self._on_cad_start_clicked()

    def _on_cad_capture_reset_clicked(self) -> None:
        photos_dir = self._cad_photos_dir_for_name()
        if os.path.isdir(photos_dir):
            removed = 0
            for name in os.listdir(photos_dir):
                lower = name.lower()
                if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json")):
                    try:
                        os.remove(os.path.join(photos_dir, name))
                        removed += 1
                    except OSError:
                        pass
            self._append_cad_log(f"已清空采集目录 ({removed} 文件): {photos_dir}")
        self._cad_capture_dir = photos_dir
        self._cad_capture_saved = 0
        self._cad_last_capture_gray = None
        self._update_cad_capture_btn()
        self.status_bar.showMessage("采集已清空")

    def _on_cad_src_browse_clicked(self) -> None:
        mode = str(self.cad_mode_combo.currentData() or "photos")
        current = self.cad_src_edit.text().strip()
        if mode == "mesh":
            initial = current if os.path.isfile(current) else (
                os.path.dirname(current) if current else CAD_MESHES_DIR
            )
            if not os.path.isdir(initial):
                initial = os.path.expanduser("~")
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择 mesh 文件",
                initial,
                "Mesh (*.obj *.ply *.glb *.stl *.off);;All Files (*)",
            )
            if path:
                self.cad_src_edit.setText(path)
                base = os.path.splitext(os.path.basename(path))[0]
                if not self.cad_name_edit.text().strip() or self.cad_name_edit.text() in (
                    "my_object",
                    "imported_object",
                ):
                    self.cad_name_edit.setText(base or "imported_object")
        else:
            initial = current if os.path.isdir(current) else os.path.expanduser("~")
            path = QFileDialog.getExistingDirectory(self, "选择多视角照片目录", initial)
            if path:
                self.cad_src_edit.setText(path)
                base = os.path.basename(path.rstrip(os.sep))
                if not self.cad_name_edit.text().strip() or self.cad_name_edit.text() == "my_object":
                    self.cad_name_edit.setText(base or "my_object")
                self._cad_capture_dir = path
                self._cad_capture_saved = self._count_photos_in_dir(path)
                self._update_cad_capture_btn()

    def _refresh_cad_mesh_combo(self) -> None:
        current = self.cad_existing_combo.currentData()
        self.cad_existing_combo.blockSignals(True)
        self.cad_existing_combo.clear()
        meshes = list_reconstructed_meshes()
        if not meshes:
            self.cad_existing_combo.addItem("(暂无 reconstructed.obj)", "")
        else:
            for name, path in meshes:
                self.cad_existing_combo.addItem(f"{name}", path)
        if current:
            idx = self.cad_existing_combo.findData(current)
            if idx >= 0:
                self.cad_existing_combo.setCurrentIndex(idx)
        self.cad_existing_combo.blockSignals(False)

    def _append_cad_log(self, line: str) -> None:
        self.cad_log_edit.append(line)
        scrollbar = self.cad_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_cad_ui(self, *_args) -> None:
        running = self._cad_launcher.is_running()
        mode_photos = str(self.cad_mode_combo.currentData() or "photos") == "photos"
        self.cad_stop_btn.setEnabled(running)
        self.cad_start_btn.setEnabled(not running)
        self.cad_capture_btn.setEnabled(not running and mode_photos)
        self.cad_capture_reset_btn.setEnabled(not running and mode_photos)
        if running:
            self.cad_status_label.setText("CAD: 运行中")
            self.cad_status_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        else:
            self.cad_status_label.setText("CAD: 空闲")
            self.cad_status_label.setStyleSheet("")
        self._update_cad_capture_btn()

    def _on_cad_start_clicked(self) -> None:
        name = self.cad_name_edit.text().strip() or "my_object"
        src = self.cad_src_edit.text().strip()
        extent = float(self.cad_extent_spin.value())
        mode = str(self.cad_mode_combo.currentData() or "photos")
        if not src:
            # 若已采集照片，自动用 photos 目录
            photos_dir = self._cad_photos_dir_for_name(name)
            if self._count_photos_in_dir(photos_dir) > 0:
                src = photos_dir
                self.cad_src_edit.setText(src)
        if not src:
            QMessageBox.information(
                self,
                "CAD",
                "请先用「拍一张」采集照片，或选择照片目录 / mesh 文件。",
            )
            return
        if mode == "mesh":
            self._cad_launcher.start_from_mesh(src, name, target_extent_m=extent)
        else:
            n_photos = self._count_photos_in_dir(src) if os.path.isdir(src) else 0
            if 0 < n_photos < CAD_MIN_IMAGES_DEFAULT:
                QMessageBox.information(
                    self,
                    "CAD",
                    f"当前仅 {n_photos} 张照片，至少需要 {CAD_MIN_IMAGES_DEFAULT} 张。\n"
                    f"请继续点「拍一张」换视角采集。",
                )
                return
            self._cad_launcher.start_from_images(
                src,
                name,
                target_extent_m=extent,
                poisson_depth=int(self.cad_poisson_spin.value()),
                min_images=CAD_MIN_IMAGES_DEFAULT,
            )
        self._update_cad_ui()

    def _on_cad_stop_clicked(self) -> None:
        self._cad_launcher.stop()

    def _on_cad_clear_log_clicked(self) -> None:
        self.cad_log_edit.clear()

    def _on_cad_apply_fp_clicked(self) -> None:
        mesh = str(self.cad_existing_combo.currentData() or "").strip()
        if not mesh:
            mesh = self._cad_launcher.last_mesh()
        if not mesh or not os.path.isfile(mesh):
            QMessageBox.information(self, "CAD", "没有可用的 reconstructed.obj。")
            return
        self.fp_mesh_edit.setText(mesh)
        idx = self.pose_backend_combo.findData(POSE_BACKEND_FOUNDATIONPOSE)
        if idx >= 0:
            self.pose_backend_combo.setCurrentIndex(idx)
        self._on_pose_settings_changed()
        self.status_bar.showMessage(f"已应用到 FP mesh: {mesh}")

    def _on_cad_mesh_ready(self, mesh_path: str) -> None:
        self._refresh_cad_mesh_combo()
        idx = self.cad_existing_combo.findData(mesh_path)
        if idx < 0:
            # findData 可能因路径规范化不一致失败
            abs_mesh = os.path.abspath(mesh_path)
            for i in range(self.cad_existing_combo.count()):
                if os.path.abspath(str(self.cad_existing_combo.itemData(i) or "")) == abs_mesh:
                    idx = i
                    break
        if idx >= 0:
            self.cad_existing_combo.setCurrentIndex(idx)
        self.fp_mesh_edit.setText(mesh_path)
        self._on_pose_settings_changed()
        self._append_cad_log(f"已自动填入分割页 FP mesh: {mesh_path}")

    def _update_local_ai_deploy_btn(self, *_args) -> None:
        running = self._local_ai_launcher.is_sam3_running()
        if running:
            self.local_ai_deploy_btn.setText("停止 AI")
            self.local_ai_deploy_btn.setStyleSheet(f"color: {UI_ACCENT_RED};")
        else:
            self.local_ai_deploy_btn.setText("本地部署 AI")
            self.local_ai_deploy_btn.setStyleSheet("")

    def _configure_sam3_for_local_http(self) -> None:
        url = resolve_sam3_viewer_server_url()
        idx = self.segment_backend_combo.findData(SAM3_BACKEND_POINT)
        if idx >= 0:
            self.segment_backend_combo.setCurrentIndex(idx)
        self.sam3_http_check.setChecked(True)
        set_segment_settings(
            backend=SAM3_BACKEND_POINT,
            sam3_text=self.sam3_text_edit.text().strip(),
            sam3_use_http=True,
            sam3_server_url=url,
        )
        self._on_segment_settings_changed()

    def _on_local_ai_deploy_clicked(self) -> None:
        if self._local_ai_launcher.is_sam3_running():
            self._local_ai_launcher.stop()
            self._update_local_ai_deploy_btn()
            self._refresh_sam3_status()
            return

        if resolve_sam3_run_script() is None and is_running_in_docker():
            QMessageBox.information(
                self,
                "本地部署 AI",
                "当前 viewer 在 Docker 内运行，无法直接启动宿主机 SAM3。\n\n"
                "请在宿主机终端执行：\n"
                "  cd ~/workspace_liyichao/eai\n"
                "  bash run_sam3.sh --host 0.0.0.0\n\n"
                "启动后本界面会自动通过 HTTP 连接 SAM3，并尝试启动 Ollama。",
            )

        self._local_ai_launcher.start_deploy()
        self._configure_sam3_for_local_http()
        self.chat_panel.apply_local_ollama_preset(silent=True)
        QTimer.singleShot(2500, self._after_local_ai_deploy)

    def _after_local_ai_deploy(self) -> None:
        self._update_local_ai_deploy_btn()
        self._refresh_sam3_status()
        url = resolve_sam3_viewer_server_url()
        if check_sam3_server_health(url):
            self.status_bar.showMessage(f"本地 SAM3 已在线: {url}")
        elif self._local_ai_launcher.is_sam3_running():
            self.status_bar.showMessage(f"SAM3 启动中，请稍候… ({url})")

    def closeEvent(self, event: QCloseEvent) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._stop_skeleton_tracking()
        self._ui_timer.stop()
        self._sam3_health_timer.stop()
        self._fp_health_timer.stop()
        self._stack_launcher.shutdown()
        self._replay_launcher.shutdown()
        self._local_ai_launcher.shutdown()
        self._train_launcher.shutdown()
        self._cad_launcher.shutdown()
        self._local_qwen_launcher.shutdown()
        self._remote_qwen_launcher.shutdown()
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

        robot, stack = self._stack_launcher.get_cached_status()
        if stack != "运行中":
            if self.node.is_hal_arm_ready() or robot in ("就绪", "启动中"):
                self._stack_launcher.start_stack()
                self.status_bar.showMessage(
                    "手/臂服务栈启动中，回放将自动切 model 模式并使能手/臂…",
                    8000,
                )
            else:
                reply = QMessageBox.warning(
                    self,
                    "回放前置条件",
                    "手/臂服务栈未运行，且 HAL 未就绪。\n\n"
                    "请先点击「启动机器人栈」，或仍要继续仅启动回放节点？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        warnings = self.node.prepare_for_rrd_replay()
        if warnings:
            reply = QMessageBox.warning(
                self,
                "回放前置条件",
                "当前未完全满足回放条件（将自动重试使能/切模式）：\n\n"
                + "\n".join(f"• {w}" for w in warnings)
                + "\n\n仍要继续启动回放节点？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                self.node._stop_replay_prep_mode_timer()
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
            color = UI_TEXT_MUTED
        else:
            color = "#7ec8ff"
        self.replay_status_label.setStyleSheet(f"color: {color};")

    def _set_sam3_result_text(self, text: str) -> None:
        self.sam3_result_edit.setPlainText(text)

    def _set_segment_preview(
        self,
        image_bgr: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        seed_uv: Optional[Tuple[float, float]] = None,
        centroid_uv: Optional[Tuple[float, float]] = None,
        obb_corners: Optional[np.ndarray] = None,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
        pose_position: Optional[np.ndarray] = None,
        pose_rotation: Optional[np.ndarray] = None,
        pose_axis_len: Optional[float] = None,
    ) -> None:
        """在分割 Tab 右侧显示 mask / 位姿叠加缩略图。"""
        if image_bgr is None:
            self.sam3_preview_label.clear()
            self.sam3_preview_label.setText("分割预览")
            return
        has_mask = mask is not None and np.asarray(mask).any()
        has_pose = obb_corners is not None or pose_position is not None
        if not has_mask and not has_pose:
            self.sam3_preview_label.clear()
            self.sam3_preview_label.setText("分割预览")
            return
        overlay = apply_segment_overlay(
            image_bgr,
            mask,
            centroid_uv=centroid_uv,
            seed_uv=seed_uv,
            obb_corners=obb_corners,
            intrinsics=intrinsics,
            pose_position=pose_position,
            pose_rotation=pose_rotation,
            pose_axis_len=pose_axis_len,
        )
        max_w, max_h = 240, 160
        h, w = overlay.shape[:2]
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 0.999:
            overlay = cv2.resize(
                overlay,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self.sam3_preview_label.setPixmap(cv2_to_qpixmap(overlay))
        self.sam3_preview_label.setText("")

    def _on_sam3_click_uv(self, topic: str, u: int, v: int) -> None:
        if is_depth_topic(topic):
            color_topic = find_paired_color_topic(topic, self._cached_topic_names())
            if color_topic:
                topic = color_topic
        self._last_sam3_topic = topic
        self.sam3_u_spin.setValue(max(0, int(u)))
        self.sam3_v_spin.setValue(max(0, int(v)))
        panel = self.panels.get(topic)
        if isinstance(panel, CameraPanel) and panel._latest_image is not None:
            h, w = panel._latest_image.shape[:2]
            self.sam3_u_spin.setMaximum(max(u, w - 1))
            self.sam3_v_spin.setMaximum(max(v, h - 1))

    def _on_stereo_invoke_clicked(self) -> None:
        """按钮确认：按提示点执行立体分割 + 6D 位姿。"""
        u = self.sam3_u_spin.value()
        v = self.sam3_v_spin.value()
        topic = self._last_sam3_topic or ""
        panel = self.panels.get(topic) if topic else None

        if isinstance(panel, CameraPanel):
            msg = panel.invoke_segment_at_seed(u, v)
            self.status_bar.showMessage(msg)
            return

        # 深度面板：把彩色提示点映射到 depth 坐标
        if topic and not is_depth_topic(topic):
            depth_topic = self._get_paired_depth_topic_name(topic)
            if depth_topic:
                depth_panel = self.panels.get(depth_topic)
                if isinstance(depth_panel, DepthPanel3D):
                    color = None
                    cam = self.panels.get(topic)
                    if isinstance(cam, CameraPanel) and cam._latest_image is not None:
                        color = cam._latest_image
                    if color is not None and depth_panel._latest_depth is not None:
                        dh, dw = depth_panel._depth_full_shape
                        u_d, v_d = scale_uv_to_shape(
                            u, v, color.shape[:2], (dh, dw)
                        )
                        msg = depth_panel.invoke_segment_at_seed(u_d, v_d)
                        self.status_bar.showMessage(msg)
                        return

        if isinstance(panel, DepthPanel3D):
            msg = panel.invoke_segment_at_seed()
            self.status_bar.showMessage(msg)
            return

        # 回退：任一面板上有待处理提示点
        for p in self.panels.values():
            if isinstance(p, CameraPanel) and p._pending_seed_uv is not None:
                msg = p.invoke_segment_at_seed()
                self.status_bar.showMessage(msg)
                return
            if isinstance(p, DepthPanel3D) and p._pending_seed_full_uv is not None:
                msg = p.invoke_segment_at_seed()
                self.status_bar.showMessage(msg)
                return

        self.status_bar.showMessage("请先点击彩色/深度图像选择提示点，再点「调用分割」")

    def _pick_chat_camera_frame(self) -> Optional[Tuple[str, np.ndarray]]:
        """供 AI 对话附带视觉输入：优先最近点击的彩色 topic / 头部彩色。"""
        source = self._pick_sam3_color_source()
        if source is None:
            return None
        topic, image, _panel = source
        return topic, image

    def _pick_sam3_color_source(
        self,
    ) -> Optional[Tuple[str, np.ndarray, Optional[CameraPanel]]]:
        if self._last_sam3_topic:
            panel = self.panels.get(self._last_sam3_topic)
            if isinstance(panel, CameraPanel) and panel._latest_image is not None:
                return (
                    self._last_sam3_topic,
                    panel._latest_image.copy(),
                    panel,
                )
        for topic, panel in sorted(self.panels.items()):
            if is_depth_topic(topic) or not isinstance(panel, CameraPanel):
                continue
            if panel._latest_image is not None:
                return topic, panel._latest_image.copy(), panel
        for topic in sorted(self._frame_cache.keys()):
            if is_depth_topic(topic):
                continue
            image = self._frame_cache.get(topic)
            if image is None or np.asarray(image).ndim < 2:
                continue
            panel = self.panels.get(topic)
            return (
                topic,
                np.asarray(image).copy(),
                panel if isinstance(panel, CameraPanel) else None,
            )
        return None

    def _on_sam3_invoke_clicked(self) -> None:
        if self._sam3_call_busy:
            return
        source = self._pick_sam3_color_source()
        if source is None:
            self._set_sam3_result_text(
                "错误: 无可用彩色图像\n请勾选 color topic 并等待图像到达"
            )
            self.status_bar.showMessage("SAM3: 无可用彩色图像")
            return

        topic, image, _panel = source
        h, w = image.shape[:2]
        u = max(0, min(w - 1, self.sam3_u_spin.value()))
        v = max(0, min(h - 1, self.sam3_v_spin.value()))
        self.sam3_u_spin.setMaximum(max(u, w - 1))
        self.sam3_v_spin.setMaximum(max(v, h - 1))
        self.sam3_u_spin.setValue(u)
        self.sam3_v_spin.setValue(v)

        backend = self.segment_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = SAM3_BACKEND_POINT
        if backend == SAM3_BACKEND_GEOMETRY:
            backend = SAM3_BACKEND_POINT

        self._on_segment_settings_changed()
        cfg = get_segment_settings()
        text = self.sam3_text_edit.text().strip() if backend == SAM3_BACKEND_TEXT else ""
        transport = "HTTP" if cfg.sam3_use_http else "子进程"

        self._sam3_call_busy = True
        self.sam3_invoke_btn.setEnabled(False)
        self.sam3_invoke_btn.setText("SAM3 调用中…")
        self._set_sam3_result_text(
            f"正在调用 SAM3…\n图像: {topic}\n"
            f"模型: {cfg.sam3_model}\n"
            + (f"文本: {text}\n" if text else f"点: ({u}, {v})\n")
            + f"传输: {transport}"
        )

        def _work() -> None:
            t0 = time.time()
            try:
                mask, method = run_sam3_segmentation(
                    image, u, v, text=text or None, settings=cfg, tag="invoke_btn"
                )
                if not mask.any():
                    raise RuntimeError("SAM3 返回空 mask")
                if int(mask.sum()) < SEGMENT_MIN_AREA:
                    raise RuntimeError(
                        f"SAM3 mask 过小 ({int(mask.sum())} px < {SEGMENT_MIN_AREA})"
                    )
                ys, xs = np.where(mask)
                centroid = (
                    (float(np.mean(xs)), float(np.mean(ys)))
                    if len(xs) > 0
                    else (float(u), float(v))
                )
                result = Sam3CallResult(
                    ok=True,
                    topic=topic,
                    u=u,
                    v=v,
                    text=text,
                    method=method,
                    pixel_count=int(mask.sum()),
                    mask_shape=(mask.shape[0], mask.shape[1]),
                    centroid_uv=centroid,
                    transport=transport,
                    elapsed_s=time.time() - t0,
                    mask=mask,
                )
            except Exception as exc:
                result = Sam3CallResult(
                    ok=False,
                    topic=topic,
                    u=u,
                    v=v,
                    text=text,
                    transport=transport,
                    elapsed_s=time.time() - t0,
                    error=str(exc),
                )
            self._sam3_call_bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def _on_sam3_call_finished(self, result: object) -> None:
        self._sam3_call_busy = False
        self.sam3_invoke_btn.setEnabled(True)
        self.sam3_invoke_btn.setText("调用 SAM3")
        if not isinstance(result, Sam3CallResult):
            return

        self._set_sam3_result_text(format_sam3_call_result_text(result))
        if result.ok and result.mask is not None:
            panel = self.panels.get(result.topic)
            if isinstance(panel, CameraPanel):
                panel.show_sam3_mask(result.mask, result.u, result.v)
                src = panel._latest_image
            else:
                src = None
            self._set_segment_preview(
                src,
                result.mask,
                seed_uv=(float(result.u), float(result.v)),
                centroid_uv=result.centroid_uv,
            )
            self.status_bar.showMessage(
                f"SAM3 成功: {result.pixel_count} px ({result.method})"
            )
        elif result.ok:
            self.status_bar.showMessage("SAM3 成功")
        else:
            self._set_segment_preview(None, None)
            self.status_bar.showMessage(f"SAM3 失败: {result.error[:120]}")

    def _set_fp_result_text(self, text: str) -> None:
        self.fp_result_edit.setPlainText(text)

    def _pick_fp_stereo_source(
        self,
    ) -> Optional[Tuple[str, np.ndarray, str, np.ndarray, int, int]]:
        color_source = self._pick_sam3_color_source()
        if color_source is None:
            return None
        color_topic, color_bgr, _panel = color_source
        if not self._is_paired_depth_enabled(color_topic):
            return None
        depth_topic = self._get_paired_depth_topic_name(color_topic)
        if not depth_topic:
            return None
        depth = self._get_paired_depth_frame(color_topic)
        if depth is None or np.asarray(depth).ndim < 2:
            return None
        depth = np.asarray(depth)
        ch, cw = color_bgr.shape[:2]
        u = max(0, min(cw - 1, self.sam3_u_spin.value()))
        v = max(0, min(ch - 1, self.sam3_v_spin.value()))
        return color_topic, color_bgr, depth_topic, depth, u, v

    def _on_fp_invoke_clicked(self) -> None:
        if self._fp_call_busy:
            return
        self._on_pose_settings_changed()
        pcfg = get_pose_settings()
        mesh_path, mesh_source = resolve_fp_mesh_for_call(
            pcfg.fp_mesh,
            pcfg.fp_use_http,
            pcfg.fp_server_url,
        )
        mesh_display = mesh_path if mesh_source != "local" else pcfg.fp_mesh

        source = self._pick_fp_stereo_source()
        if source is None:
            self._set_fp_result_text(
                "错误: 需要彩色图 + 已勾选的配对 depth topic\n"
                "请勾选 color 与对应 depth，并在图像上设置提示点 (u,v)"
            )
            self.status_bar.showMessage("FoundationPose: 缺少彩色/深度数据")
            return

        color_topic, color_bgr, depth_topic, depth, u, v = source
        dh, dw = depth.shape[:2]
        u_d, v_d = scale_uv_to_shape(u, v, color_bgr.shape[:2], (dh, dw))
        fx, fy, cx, cy = self.node.get_intrinsics(depth_topic, dw, dh)
        transport = "HTTP" if pcfg.fp_use_http else "子进程"

        self._fp_call_busy = True
        self.fp_invoke_btn.setEnabled(False)
        self.fp_invoke_btn.setText("FP 调用中…")
        self._set_fp_result_text(
            f"正在调用 FoundationPose…\n"
            f"彩色: {color_topic}\n"
            f"深度: {depth_topic}\n"
            f"点: color({u},{v}) depth({u_d},{v_d})\n"
            f"mesh: {mesh_display} ({mesh_source})\n"
            f"传输: {transport}"
        )
        scfg = get_segment_settings()

        def _work() -> None:
            t0 = time.time()
            try:
                mask, seg_method = segment_object_at_click_with_backend(
                    u_d,
                    v_d,
                    depth,
                    color_bgr=color_bgr,
                    settings=scfg,
                )
                if not mask.any():
                    raise RuntimeError("分割失败（无有效 mask）")
                fp_body = run_foundationpose_estimation(
                    color_bgr,
                    depth,
                    mask,
                    fx,
                    fy,
                    cx,
                    cy,
                    settings=pcfg,
                    tag="invoke_btn",
                )
                pose_result = object6d_from_foundationpose(
                    mask,
                    depth,
                    u_d,
                    v_d,
                    fx,
                    fy,
                    cx,
                    cy,
                    fp_body,
                    segment_method=seg_method,
                )
                if pose_result is None:
                    raise RuntimeError("FoundationPose 响应无效")
                result = FpCallResult(
                    ok=True,
                    color_topic=color_topic,
                    depth_topic=depth_topic,
                    u=u,
                    v=v,
                    mesh=mesh_display,
                    segment_method=seg_method,
                    pose_method=str(fp_body.get("method") or "foundationpose"),
                    transport=transport,
                    elapsed_s=time.time() - t0,
                    mask=mask,
                    pose_result=pose_result,
                )
            except Exception as exc:
                result = FpCallResult(
                    ok=False,
                    color_topic=color_topic,
                    depth_topic=depth_topic,
                    u=u,
                    v=v,
                    mesh=mesh_display,
                    transport=transport,
                    elapsed_s=time.time() - t0,
                    error=str(exc),
                )
            self._fp_call_bridge.finished.emit(result)

        threading.Thread(target=_work, daemon=True).start()

    def _on_fp_call_finished(self, result: object) -> None:
        self._fp_call_busy = False
        self.fp_invoke_btn.setText("调用 FP")
        if not isinstance(result, FpCallResult):
            self._refresh_fp_status()
            return

        self._set_fp_result_text(format_fp_call_result_text(result))
        if not result.ok or result.pose_result is None:
            if result.ok:
                self.status_bar.showMessage("FoundationPose 成功")
            else:
                self.status_bar.showMessage(f"FoundationPose 失败: {result.error[:120]}")
            self._refresh_fp_status()
            return

        pose = result.pose_result
        color_panel = self.panels.get(result.color_topic)
        if isinstance(color_panel, CameraPanel) and color_panel._latest_image is not None:
            src = color_panel._latest_image
            display_mask = resize_mask_to_shape(pose.mask, src.shape[:2])
            dh, dw = pose.mask.shape[:2]
            ch, cw = src.shape[:2]
            cu, cv_pt = scale_uv_to_shape(
                int(round(pose.centroid_uv[0])),
                int(round(pose.centroid_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            contact_u, contact_v = scale_uv_to_shape(
                int(round(pose.contact_uv[0])),
                int(round(pose.contact_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            # 位姿在 depth 相机坐标系；用 depth 内参并缩放到彩色图尺寸
            fx, fy, cx, cy = self.node.get_intrinsics(result.depth_topic, dw, dh)
            sx = float(cw) / float(max(1, dw))
            sy = float(ch) / float(max(1, dh))
            intr = (fx * sx, fy * sy, cx * sx, cy * sy)
            axis_len = float(max(pose.obb_extents) * 0.55)
            color_panel.image_label.set_segment_overlay(
                display_mask,
                (float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=intr,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(result.u), float(result.v)),
                pose_position=np.asarray(pose.position_xyz, dtype=np.float32),
                pose_rotation=np.asarray(pose.rotation_matrix, dtype=np.float32),
                pose_axis_len=axis_len,
            )
            self._set_segment_preview(
                src,
                display_mask,
                seed_uv=(float(result.u), float(result.v)),
                centroid_uv=(float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=intr,
                pose_position=np.asarray(pose.position_xyz, dtype=np.float32),
                pose_rotation=np.asarray(pose.rotation_matrix, dtype=np.float32),
                pose_axis_len=axis_len,
            )

        depth_panel = self.panels.get(result.depth_topic)
        if isinstance(depth_panel, DepthPanel3D):
            dh, dw = depth_panel._depth_full_shape
            mask_p = resize_mask_to_shape(pose.mask, depth_panel._preview_shape)
            cu, cv_pt = scale_uv_to_shape(
                int(round(pose.centroid_uv[0])),
                int(round(pose.centroid_uv[1])),
                (dh, dw),
                depth_panel._preview_shape,
            )
            contact_u, contact_v = scale_uv_to_shape(
                int(round(pose.contact_uv[0])),
                int(round(pose.contact_uv[1])),
                (dh, dw),
                depth_panel._preview_shape,
            )
            depth_panel.depth_preview.set_segment_overlay(
                mask_p,
                (float(cu), float(cv_pt)),
                obb_corners=pose.obb_corners,
                intrinsics=depth_panel._preview_intrinsics,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(result.u), float(result.v)),
                pose_position=np.asarray(pose.position_xyz, dtype=np.float32),
                pose_rotation=np.asarray(pose.rotation_matrix, dtype=np.float32),
                pose_axis_len=float(max(pose.obb_extents) * 0.55),
            )
            depth_panel._show_pose_6d(pose)

        camera_frame = self.node.resolve_segment_camera_frame(
            result.color_topic, result.depth_topic
        )
        if camera_frame:
            # 切换到「分割位姿」绝对目标（FP 物体 6D），不要走「相对当前」
            self._store_segment_target(
                SegmentPoseTarget.from_pose_result(
                    pose, camera_frame, result.color_topic
                )
            )
            self.status_bar.showMessage(
                f"FoundationPose 成功 ({result.pose_method})，已设为左臂绝对目标"
            )
        else:
            self.status_bar.showMessage(
                f"FoundationPose 成功 ({result.pose_method})，"
                "但缺少相机 frame_id，无法设为左臂绝对目标"
            )
        self._refresh_fp_status()

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
            sam3_server_url=resolve_sam3_viewer_server_url()
            if (self.sam3_http_check.isChecked() and use_sam3)
            else SAM3_SERVER_URL_DEFAULT,
            sam3_model=resolve_sam3_model_path(),
        )
        self._refresh_sam3_status()

    def _refresh_sam3_status(self) -> None:
        cfg = get_segment_settings()
        if cfg.backend == SAM3_BACKEND_GEOMETRY:
            self.sam3_status_label.setText("SAM3: 关")
            self.sam3_status_label.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            return
        if cfg.sam3_use_http:
            ok = check_sam3_server_health(cfg.sam3_server_url)
            if ok:
                self.sam3_status_label.setText("SAM3: 在线")
                self.sam3_status_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
            else:
                self.sam3_status_label.setText("SAM3: 离线")
                self.sam3_status_label.setStyleSheet(f"color: {UI_ACCENT_RED};")
        else:
            self.sam3_status_label.setText("SAM3: 子进程")
            self.sam3_status_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")

    def _on_pose_settings_changed(self, *_args) -> None:
        backend = self.pose_backend_combo.currentData()
        if not isinstance(backend, str):
            backend = POSE_BACKEND_PCA
        self.fp_mesh_edit.setEnabled(True)
        self.fp_mesh_browse_btn.setEnabled(True)
        self.fp_http_check.setEnabled(True)
        self.fp_invoke_btn.setEnabled(not self._fp_call_busy)
        mesh_path = self.fp_mesh_edit.text().strip() or resolve_fp_mesh_path()
        use_http = self.fp_http_check.isChecked()
        server_url = (
            resolve_fp_viewer_server_url()
            if use_http
            else FP_SERVER_URL_DEFAULT
        )
        set_pose_settings(
            backend=backend,
            fp_use_http=use_http,
            fp_server_url=server_url,
            fp_mesh=mesh_path,
        )
        self._refresh_fp_status()

    def _refresh_fp_status(self) -> None:
        cfg = get_pose_settings()
        use_http = self.fp_http_check.isChecked()
        server_url = (
            resolve_fp_viewer_server_url() if use_http else cfg.fp_server_url
        )
        status = evaluate_fp_availability(
            cfg.fp_mesh,
            use_http=use_http,
            server_url=server_url,
        )
        self.fp_status_label.setText(f"FP: {status.label}")
        self.fp_status_label.setStyleSheet(f"color: {status.color};")
        self.fp_status_label.setToolTip(status.tooltip)
        self.fp_avail_detail_label.setText(status.detail)
        self.fp_avail_detail_label.setToolTip(status.tooltip)
        if not self._fp_call_busy:
            self.fp_invoke_btn.setEnabled(status.can_invoke)
        pose_hint = ""
        if cfg.backend == POSE_BACKEND_FOUNDATIONPOSE:
            pose_hint = "选点后点「调用分割」或「调用 FP」将使用 FoundationPose"
        elif status.can_invoke:
            pose_hint = "选点后点「调用分割」用 PCA；「调用 FP」按钮可用"
        if pose_hint and not self._fp_call_busy:
            self.fp_avail_detail_label.setToolTip(f"{status.tooltip}\n{pose_hint}")

    def _on_fp_mesh_browse_clicked(self) -> None:
        initial = self.fp_mesh_edit.text().strip()
        if not initial or not os.path.isfile(initial):
            initial = os.path.dirname(resolve_fp_mesh_path())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择物体 mesh",
            initial if os.path.isdir(initial) else os.path.expanduser("~"),
            "Mesh (*.obj *.OBJ);;All Files (*)",
        )
        if path:
            self.fp_mesh_edit.setText(path)
            self._on_pose_settings_changed()

    def _on_stack_status_changed(self, robot: str, base: str) -> None:
        self._update_robot_stack_ui(robot, base)
        self._update_robot_enable_status_ui()

    def _update_robot_stack_ui(
        self,
        robot: Optional[str] = None,
        base: Optional[str] = None,
    ) -> None:
        if robot is None or base is None:
            robot, base = self._stack_launcher.get_cached_status()
        self.robot_stack_status.setText(f"robot: {robot} | stack: {base}")
        if base == "运行中":
            self.robot_stack_status.setStyleSheet("color: #88ff88;")
        elif base == "启动中" or robot == "启动中":
            self.robot_stack_status.setStyleSheet("color: #ffcc66;")
        else:
            self.robot_stack_status.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")
        busy = base in ("运行中", "启动中") or robot == "启动中"
        self.robot_stack_btn.setEnabled(not busy)
        if base == "运行中":
            self.robot_stack_btn.setText("栈已运行")
        elif robot == "启动中":
            self.robot_stack_btn.setText("robot 启动中…")
        elif base == "启动中":
            self.robot_stack_btn.setText("服务栈启动中…")
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
        active_style = f"font-weight: bold; color: {UI_ACCENT_BLUE};"
        idle_style = f"color: {UI_TEXT_MUTED};"
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

    def _on_right_hand_sliders_changed(self, _value: int = 0) -> None:
        self.right_hand_label_a.setText(format_hand_angle_label(self.right_hand_slider_a.value()))
        self.right_hand_label_b.setText(format_hand_angle_label(self.right_hand_slider_b.value()))
        self._update_right_hand_toggle_ui(self.node.is_right_hand_at_a())

    def _update_right_hand_toggle_ui(self, at_a: bool) -> None:
        pos_a = slider_to_hand_position(self.right_hand_slider_a.value())
        pos_b = slider_to_hand_position(self.right_hand_slider_b.value())
        next_preset = "B" if at_a else "A"
        next_pos = pos_b if at_a else pos_a
        self.right_hand_toggle_btn.setText(f"右手: 切到{next_preset} ({next_pos:.2f})")
        active_style = f"font-weight: bold; color: {UI_ACCENT_BLUE};"
        idle_style = f"color: {UI_TEXT_MUTED};"
        self.right_hand_label_a.setStyleSheet(active_style if at_a else idle_style)
        self.right_hand_label_b.setStyleSheet(active_style if not at_a else idle_style)

    def _on_right_hand_toggle(self) -> None:
        at_a, sent_pos = self.node.toggle_right_hand_between(
            self.right_hand_slider_a.value(),
            self.right_hand_slider_b.value(),
        )
        self._update_right_hand_toggle_ui(at_a)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"右手已切到状态{preset} position={sent_pos:.3f} -> {ROBOT_RIGHT_HAND_CMD_TOPIC}"
        )

    def _update_move_offset_ui_visibility(self) -> None:
        """「分割位姿」时隐藏相对 ΔXYZ；仅「相对当前」显示偏移控件。"""
        show_offset = self.move_target_relative_radio.isChecked()
        for widget in (
            self.offset_x_label,
            self.offset_x_spin,
            self.offset_y_label,
            self.offset_y_spin,
            self.offset_z_label,
            self.offset_z_spin,
        ):
            widget.setVisible(show_offset)
            widget.setEnabled(show_offset)

    def _on_move_target_params_changed(self, *_args) -> None:
        if self._last_segment_target is None and self.move_target_segment_radio.isChecked():
            self.move_target_relative_radio.setChecked(True)
        self._update_move_offset_ui_visibility()
        self._update_arm_pose_display()
        self._update_arm_move_btns_ui()

    def _tcp_for_relative_goal(
        self, arm_side: str
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        tcp = self.node._tcp_pose_in_ik_frame(arm_side, timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
        if tcp is not None:
            return tcp
        state = self.node.get_robot_state()
        raw = state.left_tcp if arm_side == "left" else state.right_tcp
        if raw is not None and raw.valid:
            frame = normalize_frame_id(raw.frame_id)
            if frame == normalize_frame_id(IK_TARGET_FRAME) or frame in MINK_FK_FRAME_ALIASES:
                return (raw.xyz, raw.quat_xyzw)
        return None

    def _resolve_move_goal(self, arm_side: str = "left") -> Optional[ResolvedArmMoveGoal]:
        if self.move_target_segment_radio.isChecked() and self._last_segment_target is not None:
            return self.node.resolve_segment_move_goal(
                self._last_segment_target, arm_side=arm_side
            )
        tcp = self._tcp_for_relative_goal(arm_side)
        if tcp is None:
            return None
        xyz, quat = tcp
        return compute_relative_move_goal(
            xyz,
            quat,
            self.offset_x_spin.value(),
            self.offset_y_spin.value(),
            self.offset_z_spin.value(),
            arm_side=arm_side,
        )

    def _on_arm_move_speed_changed(self, value: int) -> None:
        speed = slider_to_arm_move_speed(value)
        self.node.set_arm_move_joint_speed(speed)
        self.arm_move_speed_label.setText(format_arm_move_speed_label(value))
        if not self.node.is_slow_motion_busy():
            self._update_arm_move_btns_ui()

    def _on_arm_enable_clicked(self) -> None:
        enable = not self.node.is_arm_enabled()
        msg = self.node.request_arm_enable(enable=enable)
        self.status_bar.showMessage(msg)

    def _on_model_mode_clicked(self) -> None:
        msg = self.node.request_model_control_mode()
        self.status_bar.showMessage(msg)
        self._update_enable_status_ui()

    def _on_hand_enable_clicked(self) -> None:
        enable = not self.node.is_hand_enabled()
        msg = self.node.request_hand_enable(enable=enable)
        self.status_bar.showMessage(msg)

    def _on_arm_enable_ui_changed(self, enabled: bool) -> None:
        self._update_enable_status_ui()
        if enabled:
            self._try_finish_pending_arm_move()

    def _on_hand_enable_ui_changed(self, _enabled: bool) -> None:
        self._update_enable_status_ui()

    def _on_control_mode_ui_changed(self, _mode: int) -> None:
        self._update_enable_status_ui()

    @staticmethod
    def _style_enable_label(label: QLabel, received: bool, enabled: bool) -> None:
        if not received:
            label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        elif enabled:
            label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        else:
            label.setStyleSheet(f"color: {UI_ACCENT_RED};")

    def _update_robot_enable_status_ui(self) -> None:
        arm_received = self.node._arm_enable_received_at > 0
        hand_received = self.node._hand_enable_received_at > 0
        arm_enabled = self.node.is_arm_enabled()
        hand_enabled = self.node.is_hand_enabled()

        if not arm_received:
            arm_text = "手臂: 等待"
        else:
            arm_text = f"手臂: {'已使能' if arm_enabled else '未使能'}"
        self.robot_arm_enable_label.setText(arm_text)
        self._style_enable_label(self.robot_arm_enable_label, arm_received, arm_enabled)

        if not hand_received:
            hand_text = "手: 等待"
        else:
            hand_text = f"手: {'已使能' if hand_enabled else '未使能'}"
        self.robot_hand_enable_label.setText(hand_text)
        self._style_enable_label(self.robot_hand_enable_label, hand_received, hand_enabled)

        mode_label = self.node.get_control_mode_label()
        self.robot_control_mode_label.setText(mode_label)
        if self.node._control_mode_received_at <= 0:
            self.robot_control_mode_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        elif self.node._control_mode == MODEL_CONTROL_MODE:
            self.robot_control_mode_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        else:
            self.robot_control_mode_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")

    def _update_hand_enable_ui(self) -> None:
        self.hand_enable_label.setText(self.node.get_hand_enable_label())
        received = self.node._hand_enable_received_at > 0
        enabled = self.node.is_hand_enabled()
        self._style_enable_label(self.hand_enable_label, received, enabled)
        btn_text = "关闭手" if enabled else "启用手"
        self.hand_enable_btn.setText(btn_text)
        if hasattr(self, "robot_hand_enable_btn"):
            self.robot_hand_enable_btn.setText(btn_text)

    def _update_enable_status_ui(self) -> None:
        self._update_arm_enable_ui()
        self._update_hand_enable_ui()
        self._update_robot_enable_status_ui()
        mode_text = self.node.get_control_mode_label()
        self.control_mode_label.setText(mode_text)
        if self.node._control_mode_received_at <= 0:
            self.control_mode_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        elif self.node._control_mode == MODEL_CONTROL_MODE:
            self.control_mode_label.setStyleSheet(f"color: {UI_ACCENT_GREEN};")
        else:
            self.control_mode_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")

    def _try_finish_pending_arm_move(self) -> None:
        if self._pending_arm_move_goal is None or not self.node.is_arm_enabled():
            return
        goal = self._pending_arm_move_goal
        self._pending_arm_move_goal = None
        self._arm_enable_wait_timer.stop()
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

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
            self._update_arm_move_btns_ui()
        self._update_arm_move_btns_ui()

    def _update_arm_enable_ui(self) -> None:
        self.arm_enable_label.setText(self.node.get_arm_enable_label())
        received = self.node._arm_enable_received_at > 0
        enabled = self.node.is_arm_enabled()
        self._style_enable_label(self.arm_enable_label, received, enabled)
        btn_text = "关闭手臂" if enabled else "启用手臂"
        self.arm_enable_btn.setText(btn_text)
        if hasattr(self, "robot_arm_enable_btn"):
            self.robot_arm_enable_btn.setText(btn_text)

    def _can_start_arm_move(self, arm_side: str = "left") -> bool:
        if self.node.is_slow_motion_busy():
            return True
        if self.move_target_segment_radio.isChecked():
            return self._resolve_move_goal(arm_side) is not None
        state = self.node.get_robot_state()
        left_ok = state.left_tcp is not None and state.left_tcp.valid
        right_ok = state.right_tcp is not None and state.right_tcp.valid
        return left_ok and right_ok

    def _disabled_move_btn_tooltip(self, arm_side: str = "left") -> str:
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
        side_name = arm_side_label(arm_side)
        if self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                hints.append("请先点击图像选点，再点「调用分割」")
            elif self._resolve_move_goal(arm_side) is None:
                frame = self._last_segment_target.camera_frame or "未知"
                hints.append(f"分割 TF 不可用 ({frame} -> {IK_TARGET_FRAME})")
        elif self._resolve_move_goal(arm_side) is None:
            hints.append(f"相对当前模式：请为{side_name}设置非零偏移（如 ΔX=0.05 m）")
        if not hints:
            hints.append("未使能时点击「移动」将自动启用手臂")
        return "\n".join(hints)

    def _update_arm_pose_display(self) -> None:
        self.arm_pose_current_label.setText(self.node.get_arm_move_current_label_both())
        left_goal = self._resolve_move_goal("left")
        right_goal = self._resolve_move_goal("right")
        if left_goal is not None or right_goal is not None:
            parts: List[str] = []
            if left_goal is not None:
                parts.append(
                    format_xyz_rpy_line(
                        "左", left_goal.position_xyz, left_goal.quaternion_xyzw
                    )
                )
            if right_goal is not None:
                parts.append(
                    format_xyz_rpy_line(
                        "右", right_goal.position_xyz, right_goal.quaternion_xyzw
                    )
                )
            label_hint = ""
            if left_goal is not None:
                label_hint = left_goal.label
            elif right_goal is not None:
                label_hint = right_goal.label
            self.arm_pose_target_label.setText(
                f"目标  [{label_hint}]  " + "  |  ".join(parts)
            )
        elif self.move_target_segment_radio.isChecked():
            if self._last_segment_target is None:
                self.arm_pose_target_label.setText("目标  TCP: (请先选点并调用分割/FP)")
            else:
                frame = self._last_segment_target.camera_frame or "未知"
                self.arm_pose_target_label.setText(
                    f"目标  TCP: TF 不可用 ({frame} -> {IK_TARGET_FRAME})"
                )
        else:
            self.arm_pose_target_label.setText(
                "目标  TCP: (设置前/后/左/右/上/下偏移，base_link 系)"
            )

    def _schedule_robot_ui_refresh(self, *_args) -> None:
        now = time.time()
        if now - self._robot_ui_last_update < UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._robot_ui_last_update = now
        self._update_arm_pose_display()
        if not self.node.is_slow_motion_busy():
            for side, btn in (
                ("left", self.left_arm_move_btn),
                ("right", self.right_arm_move_btn),
            ):
                can_move = self._can_start_arm_move(side)
                if btn.isEnabled() != can_move:
                    btn.setEnabled(can_move)
                    if not can_move:
                        btn.setToolTip(self._disabled_move_btn_tooltip(side))

    def _update_one_arm_move_btn_ui(self, arm_side: str, force: bool = False) -> None:
        btn = self.left_arm_move_btn if arm_side == "left" else self.right_arm_move_btn
        idle_style = (
            self._left_arm_move_btn_idle_style
            if arm_side == "left"
            else self._right_arm_move_btn_idle_style
        )
        cancel_style = (
            self._left_arm_move_btn_cancel_style
            if arm_side == "left"
            else self._right_arm_move_btn_cancel_style
        )
        side_name = arm_side_label(arm_side)
        pending = self._pending_arm_move_goal
        if pending is not None and not self.node.is_arm_enabled():
            if pending.arm_side == arm_side:
                btn.setText("使能中…")
                btn.setEnabled(True)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("正在自动启用手臂，完成后将开始移动")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在等待使能")
            return
        if self.node.is_slow_motion_preparing():
            if self.node._slow_motion_moving_side == arm_side:
                btn.setText("取消准备")
                btn.setEnabled(True)
                btn.setStyleSheet(cancel_style)
                btn.setToolTip("取消正在进行的 IK 同步与轨迹规划")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在准备移动")
            return
        if self.node.is_slow_motion_active():
            if self.node._slow_motion_moving_side == arm_side:
                btn.setText("取消移动")
                btn.setEnabled(True)
                btn.setStyleSheet(cancel_style)
                btn.setToolTip("停止当前移动")
            else:
                btn.setText(f"{side_name}: 移动")
                btn.setEnabled(False)
                btn.setStyleSheet(idle_style)
                btn.setToolTip("另一侧手臂正在移动")
            return
        btn.setText(f"{side_name}: 移动")
        btn.setStyleSheet(idle_style)
        can_move = self._can_start_arm_move(arm_side)
        btn.setEnabled(can_move)
        if can_move:
            blockers = self.node.get_arm_move_blockers(tf_timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
            extra = ""
            if blockers:
                extra = "\n注意: " + "；".join(blockers)
            elif (
                not self.move_target_segment_radio.isChecked()
                and self._resolve_move_goal(arm_side) is None
            ):
                extra = "\n请设置非零偏移后再点击"
            btn.setToolTip(
                f"将{side_name} TCP 移动到目标位姿（时长随距离与「移动速度」滑块自适应）。\n"
                f"目标可为分割位姿，或相对当前位置的手动偏移。\n"
                f"未使能时将自动启用手臂；同时发布左右臂 IK 目标。{extra}"
            )
        else:
            btn.setToolTip(self._disabled_move_btn_tooltip(arm_side))

    def _update_arm_move_btns_ui(self, force: bool = False) -> None:
        speed_busy = self.node.is_slow_motion_busy()
        self.arm_move_speed_slider.setEnabled(not speed_busy)
        self._update_one_arm_move_btn_ui("left", force=force)
        self._update_one_arm_move_btn_ui("right", force=force)
        self._update_arm_pose_display()

    def _update_left_arm_move_btn_ui(self, force: bool = False) -> None:
        """兼容旧调用点。"""
        self._update_arm_move_btns_ui(force=force)

    def _store_segment_target(self, target: SegmentPoseTarget) -> None:
        self._last_segment_target = target
        self.move_target_segment_radio.setEnabled(True)
        self.move_target_segment_radio.setChecked(True)
        # 选中绝对位姿目标时，偏移量不参与目标计算；清零并隐藏 ΔXYZ
        for spin in (self.offset_x_spin, self.offset_y_spin, self.offset_z_spin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._update_move_offset_ui_visibility()
        self._update_arm_move_btns_ui()
        self._update_arm_pose_display()
        self.status_bar.showMessage(
            f"已记录绝对位姿目标: {target.label or target.source_topic}"
        )
        QTimer.singleShot(0, self._try_auto_move_to_segment_target)

    def _try_auto_move_to_segment_target(self) -> None:
        """分割/FP 完成后，按绝对位姿目标移动左臂（非相对当前）。"""
        if self.node.is_slow_motion_busy():
            return
        if not self.move_target_segment_radio.isChecked():
            return
        goal = self._resolve_move_goal("left")
        if goal is None:
            blockers = self.node.get_arm_move_blockers(tf_timeout_s=UI_TF_LOOKUP_TIMEOUT_S)
            if blockers:
                self.status_bar.showMessage(
                    "绝对位姿已记录，暂无法移动: " + "；".join(blockers)
                )
            return
        if not self.node.is_arm_enabled():
            self._pending_arm_move_goal = goal
            msg = self.node.request_arm_enable(True)
            self._arm_enable_wait_deadline = time.time() + 15.0
            self._arm_enable_wait_timer.start()
            self.status_bar.showMessage(f"{msg}，使能后将自动移向绝对目标…")
            self._update_arm_move_btns_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

    def _on_arm_move_clicked(self, arm_side: str) -> None:
        side_name = arm_side_label(arm_side)
        if self.node.is_slow_motion_busy():
            if self.node._slow_motion_moving_side != arm_side:
                return
            self._pending_arm_move_goal = None
            self._arm_enable_wait_timer.stop()
            msg = self.node.cancel_slow_motion()
            self.status_bar.showMessage(msg)
            self._update_arm_move_btns_ui()
            return
        goal = self._resolve_move_goal(arm_side)
        if goal is None:
            if self.move_target_segment_radio.isChecked():
                if self._last_segment_target is None:
                    self.status_bar.showMessage("请先选点并点「调用分割 / FP」完成位姿估计")
                else:
                    frame = self._last_segment_target.camera_frame or "未知"
                    self.status_bar.showMessage(
                        f"无法将分割位姿变换到 {IK_TARGET_FRAME}，"
                        f"请检查 {frame} 的 TF"
                    )
            else:
                self.status_bar.showMessage(
                    f"请设置非零偏移量，或等待{side_name} TCP 数据"
                )
            return
        if not self.node.is_arm_enabled():
            self._pending_arm_move_goal = goal
            msg = self.node.request_arm_enable(True)
            self._arm_enable_wait_deadline = time.time() + 15.0
            self._arm_enable_wait_timer.start()
            self.status_bar.showMessage(f"{msg}，使能后将自动开始移动…")
            self._update_arm_move_btns_ui()
            return
        msg = self.node.request_slow_move_to_goal(goal)
        self.status_bar.showMessage(msg)
        self._update_arm_move_btns_ui()

    def _on_left_arm_move_clicked(self) -> None:
        self._on_arm_move_clicked("left")

    def _on_slow_motion_progress(self, progress: float, text: str) -> None:
        self.status_bar.showMessage(text)
        self._update_arm_move_btns_ui()
        self._update_arm_pose_display()

    def _on_slow_motion_finished(self, ok: bool, text: str) -> None:
        self.status_bar.showMessage(text)
        self._update_arm_move_btns_ui()

    def _on_left_hand_apply_active(self) -> None:
        at_a = self.node.is_left_hand_at_a()
        slider = self.left_hand_slider_a.value() if at_a else self.left_hand_slider_b.value()
        sent_pos = self.node.apply_left_hand_angle(slider)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"左手状态{preset} 已应用 position={sent_pos:.3f} -> {ROBOT_LEFT_HAND_CMD_TOPIC}"
        )

    def _on_right_hand_apply_active(self) -> None:
        at_a = self.node.is_right_hand_at_a()
        slider = self.right_hand_slider_a.value() if at_a else self.right_hand_slider_b.value()
        sent_pos = self.node.apply_right_hand_angle(slider)
        preset = "A" if at_a else "B"
        self.status_bar.showMessage(
            f"右手状态{preset} 已应用 position={sent_pos:.3f} -> {ROBOT_RIGHT_HAND_CMD_TOPIC}"
        )

    def _refresh_skeleton_camera_list(self) -> None:
        current = self.skeleton_cam_combo.currentData()
        self.skeleton_cam_combo.blockSignals(True)
        self.skeleton_cam_combo.clear()
        topics = []
        for topic, types in sorted(self._topic_types.items()):
            if not self._is_image_topic(types):
                continue
            if is_depth_topic(topic):
                continue
            topics.append(topic)
        if not topics:
            for topic in sorted(self._frame_cache.keys()):
                if not is_depth_topic(topic):
                    topics.append(topic)
        for topic in topics:
            self.skeleton_cam_combo.addItem(topic, topic)
        if current:
            idx = self.skeleton_cam_combo.findData(current)
            if idx >= 0:
                self.skeleton_cam_combo.setCurrentIndex(idx)
        self.skeleton_cam_combo.blockSignals(False)
        if self.skeleton_cam_combo.count() == 0:
            self.skeleton_status_label.setText("手骨架: 无可用彩色相机（请先勾选并订阅）")

    def _stop_skeleton_tracking(self) -> None:
        self._skeleton_tracking = False
        self._skeleton_timer.stop()
        self.skeleton_teleop_check.setChecked(False)
        if self._hand_skeleton_detector is not None:
            try:
                self._hand_skeleton_detector.close()
            except Exception:
                pass
            self._hand_skeleton_detector = None
        self.skeleton_track_btn.setText("开始识别")
        self.skeleton_status_label.setText("手骨架: 已停止")

    def _on_skeleton_track_toggled(self) -> None:
        if self._skeleton_tracking:
            self._stop_skeleton_tracking()
            return
        try:
            from hand_skeleton_teleop import HandSkeletonDetector, mediapipe_available
        except Exception as exc:
            QMessageBox.warning(
                self,
                "手骨架遥控",
                f"无法加载 hand_skeleton_teleop:\n{exc}",
            )
            return
        if not mediapipe_available():
            QMessageBox.warning(
                self,
                "手骨架遥控",
                "未安装 mediapipe。请执行:\n"
                "  python3.10 -m pip install 'mediapipe==0.10.14'\n"
                "并保持 numpy<2（与 ROS2 cv_bridge 兼容）。",
            )
            return
        self._refresh_skeleton_camera_list()
        if self.skeleton_cam_combo.count() == 0:
            self.status_bar.showMessage("请先在左侧勾选彩色相机 topic")
            return
        try:
            self._hand_skeleton_detector = HandSkeletonDetector(max_num_hands=2)
        except Exception as exc:
            QMessageBox.warning(self, "手骨架遥控", f"初始化 MediaPipe Hands 失败:\n{exc}")
            self._hand_skeleton_detector = None
            return
        self._skeleton_tracking = True
        self.skeleton_track_btn.setText("停止识别")
        self.skeleton_status_label.setText("手骨架: 识别中…")
        self._skeleton_timer.start()

    def _on_skeleton_teleop_toggled(self, checked: bool) -> None:
        if checked and not self._skeleton_tracking:
            self.skeleton_teleop_check.blockSignals(True)
            self.skeleton_teleop_check.setChecked(False)
            self.skeleton_teleop_check.blockSignals(False)
            self.status_bar.showMessage("请先点击「开始识别」")
            return
        if checked and not self.node.is_hand_enabled():
            msg = self.node.request_hand_enable(True)
            self.status_bar.showMessage(f"{msg}（遥控需要手部使能）")
        if checked:
            self.skeleton_status_label.setText("手骨架: 识别中 + 遥控中")
            self.skeleton_status_label.setStyleSheet(f"color: {UI_ACCENT_ORANGE};")
        elif self._skeleton_tracking:
            self.skeleton_status_label.setText("手骨架: 识别中（仅预览）")
            self.skeleton_status_label.setStyleSheet(f"color: {UI_TEXT_SECONDARY};")

    def _on_skeleton_tick(self) -> None:
        if not self._skeleton_tracking or self._skeleton_busy:
            return
        if self._hand_skeleton_detector is None:
            return
        topic = self.skeleton_cam_combo.currentData()
        if not topic:
            return
        frame = self._frame_cache.get(topic)
        if frame is None or getattr(frame, "size", 0) == 0:
            self.skeleton_status_label.setText(f"手骨架: 等待图像 {topic}")
            return
        self._skeleton_busy = True
        try:
            from hand_skeleton_teleop import (
                draw_hand_skeleton,
                map_person_hand_to_robot_side,
            )

            flip = self.skeleton_flip_check.isChecked()
            alpha = self.skeleton_smooth_slider.value() / 100.0
            hands = self._hand_skeleton_detector.detect(
                frame, flip_horizontal=flip, smooth_alpha=alpha
            )
            vis = draw_hand_skeleton(frame, hands, flip_horizontal=flip)
            # 预览缩放到合适宽度
            max_w = max(320, self.skeleton_preview_label.width() - 8)
            h, w = vis.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                vis = cv2.resize(
                    vis,
                    (max_w, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            self.skeleton_preview_label.setPixmap(cv2_to_qpixmap(vis))

            mirror = self.skeleton_mirror_map_check.isChecked()
            lines: List[str] = []
            teleop = self.skeleton_teleop_check.isChecked()
            ctrl_left = self.skeleton_ctrl_left_check.isChecked()
            ctrl_right = self.skeleton_ctrl_right_check.isChecked()
            for hand in hands:
                robot_side = map_person_hand_to_robot_side(hand.handedness, mirror)
                j = hand.joints
                lines.append(
                    f"人{hand.handedness}→机{robot_side}: "
                    f"rot={j[0]:.2f} bend={j[1]:.2f} "
                    f"I={j[2]:.2f} M={j[3]:.2f} R={j[4]:.2f} P={j[5]:.2f}"
                )
                if teleop:
                    if robot_side == "left" and ctrl_left:
                        self.node.apply_hand_joint_positions("left", j)
                    elif robot_side == "right" and ctrl_right:
                        self.node.apply_hand_joint_positions("right", j)
            if lines:
                self.skeleton_joints_label.setText(" | ".join(lines))
                mode = "遥控" if teleop else "预览"
                self.skeleton_status_label.setText(
                    f"手骨架: {mode} 检测到 {len(hands)} 只手 @ {topic}"
                )
            else:
                self.skeleton_joints_label.setText("关节: (未检测到手)")
                self.skeleton_status_label.setText(f"手骨架: 未检测到手 @ {topic}")
        except Exception as exc:
            self.skeleton_status_label.setText(f"手骨架错误: {exc}")
        finally:
            self._skeleton_busy = False

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
            empty.setStyleSheet(f"color: {UI_TEXT_MUTED}; padding: 8px;")
            self.topic_list_layout.addWidget(empty)
            self._rebuild_panels(set())
            self._refresh_skeleton_camera_list()
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
        self._refresh_skeleton_camera_list()

    def _on_selection_changed(self) -> None:
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        self._apply_selection(enabled)
        self._refresh_skeleton_camera_list()

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
                    on_click_uv=self._on_sam3_click_uv,
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


def apply_viewer_theme(app: QApplication) -> None:
    """统一深色主题，提高标签/输入框/标签页文字对比度。"""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#2b2b2b"))
    palette.setColor(QPalette.WindowText, QColor(UI_TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor("#1e1e1e"))
    palette.setColor(QPalette.AlternateBase, QColor("#353535"))
    palette.setColor(QPalette.Text, QColor(UI_TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor("#3a3a3a"))
    palette.setColor(QPalette.ButtonText, QColor(UI_TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight, QColor("#3d6ea8"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.ToolTipText, QColor(UI_TEXT_PRIMARY))
    palette.setColor(QPalette.PlaceholderText, QColor(UI_TEXT_PLACEHOLDER))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QWidget {{
            font-size: {UI_MONO_SIZE_NORMAL}pt;
        }}
        QLabel {{
            color: {UI_TEXT_PRIMARY};
        }}
        QTabWidget::pane {{
            border: 1px solid #555;
            background: #2b2b2b;
        }}
        QTabBar::tab {{
            color: {UI_TEXT_SECONDARY};
            background: #333333;
            padding: 6px 14px;
            margin-right: 2px;
            border: 1px solid #555;
        }}
        QTabBar::tab:selected {{
            color: #ffffff;
            background: #3d3d3d;
            font-weight: bold;
        }}
        QGroupBox {{
            color: {UI_TEXT_PRIMARY};
            border: 1px solid #555;
            margin-top: 8px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #f5f5f5;
        }}
        QCheckBox, QRadioButton {{
            color: {UI_TEXT_PRIMARY};
            spacing: 6px;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            color: {UI_TEXT_PRIMARY};
            background-color: #2d2d2d;
            border: 1px solid #555;
            padding: 2px 4px;
            selection-background-color: #3d6ea8;
        }}
        QComboBox QAbstractItemView {{
            color: {UI_TEXT_PRIMARY};
            background-color: #2d2d2d;
            selection-background-color: #3d6ea8;
        }}
        QPushButton {{
            color: {UI_TEXT_PRIMARY};
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 4px 10px;
        }}
        QPushButton:hover {{
            background-color: #454545;
        }}
        QPushButton:disabled {{
            color: {UI_TEXT_MUTED};
            background-color: #2a2a2a;
        }}
        QStatusBar {{
            color: {UI_TEXT_PRIMARY};
            background: #252525;
        }}
        QScrollArea {{
            border: none;
        }}
        """
    )


def configure_qt_ime_for_chinese() -> None:
    """为 Docker/本机启用 fcitx 中文输入（须在创建 QApplication 之前调用）。"""
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    os.environ["LANG"] = "zh_CN.UTF-8"
    os.environ["LC_ALL"] = "zh_CN.UTF-8"
    os.environ["QT_IM_MODULE"] = "fcitx"
    os.environ["XMODIFIERS"] = "@im=fcitx"
    os.environ["GTK_IM_MODULE"] = "fcitx"

    # PyQt5 自带 Qt 库优先，避免 fcitx 插件与系统 Qt 混库
    try:
        import PyQt5

        qt_lib = os.path.join(
            os.path.dirname(os.path.abspath(PyQt5.__file__)), "Qt5", "lib"
        )
        if os.path.isdir(qt_lib):
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            parts = [p for p in cur.split(":") if p and p != qt_lib]
            os.environ["LD_LIBRARY_PATH"] = ":".join([qt_lib] + parts)
    except Exception:
        pass

    try:
        import PyQt5
        from PyQt5.QtCore import QCoreApplication

        plug = os.path.join(
            os.path.dirname(os.path.abspath(PyQt5.__file__)),
            "Qt5",
            "plugins",
        )
        if os.path.isdir(plug):
            QCoreApplication.addLibraryPath(plug)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    configure_qt_ime_for_chinese()
    rclpy.init()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_viewer_theme(app)
    install_chinese_ime_guards(app)
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
