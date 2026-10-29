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
from eai_app.infrastructure.module_ref import module_ref

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

