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
_eai_domain_robot_kinematics=module_ref('eai_app.domain.robot.kinematics')

def sample_reachable_positions(urdf_path, q16, count=6000):
    """Sample Link7 positions within URDF joint limits at a fixed waist pose.

    Position samples only: no collision, orientation or trajectory guarantee.
    Own model/data so the rendering FK is never modified by the worker.
    """
    kin = _eai_domain_robot_kinematics._Kinematics(urdf_path)
    pin, model = kin._pin, kin.model
    q = kin.q.copy()
    for name, value in zip(_eai_domain_robot_kinematics.JOINT_NAMES, q16):
        q[kin._q_idx[name]] = value
    rng = np.random.default_rng(17)
    result = {}
    for side in ('l', 'r'):
        indices = [kin._q_idx[f'joint{i}_{side}'] for i in range(1, 8)]
        lo, hi = model.lowerPositionLimit[indices], model.upperPositionLimit[indices]
        if not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError('URDF 手臂关节限位无效')
        samples = rng.uniform(lo, hi, (count, 7))
        points = np.empty((count, 3))
        fid = kin._frame[f'Link7_{side}']
        for i, values in enumerate(samples):
            q[indices] = values
            pin.forwardKinematics(model, kin.data, q)
            pin.updateFramePlacement(model, kin.data, fid)
            points[i] = kin.data.oMf[fid].translation
        result[side] = points
    return result

