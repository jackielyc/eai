"""Shared camera calibration and valid-depth rules for perception and display."""
from __future__ import annotations

import numpy as np
from eai_app.domain.calibration.cameras import CAMERAS, load_store

DEPTH_MIN_M = 0.08
DEPTH_MAX_M = 2.5


def canonical_color_topic(topic: str) -> str:
    topic = str(topic or '')
    aliases = {
        '/camera/left_wrist_color': '/camera/hand_left_color',
        '/camera/right_wrist_color': '/camera/hand_right_color',
    }
    if topic.endswith('_depth'):
        topic = topic[:-6] + '_color'
    return aliases.get(topic, topic)


def scaled_intrinsics(k, source_size, target_size):
    sw, sh = map(float, source_size)
    w, h = map(float, target_size)
    fx, fy, cx, cy = map(float, k)
    if min(sw, sh, w, h, fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy, sw, sh]).all():
        raise ValueError('相机内参或标定分辨率无效')
    return fx * w / sw, fy * h / sh, cx * w / sw, cy * h / sh


def calibrated_intrinsics(entry, width, height, topic=''):
    """Legacy A2D records use documented native sizes, never 2*principal-point."""
    entry = entry or {}
    topic = canonical_color_topic(topic or entry.get('color_topic', ''))
    spec = next((s for s in CAMERAS if s['topic'] == topic), None)
    intr = entry.get('intrinsics') or spec or {}
    if not all(k in intr for k in ('fx', 'fy', 'cx', 'cy')):
        return None
    sw = intr.get('width') or entry.get('image_width') or (spec or {}).get('width')
    sh = intr.get('height') or entry.get('image_height') or (spec or {}).get('height')
    if not sw or not sh:
        return None
    return scaled_intrinsics([intr[k] for k in ('fx', 'fy', 'cx', 'cy')], (sw, sh), (width, height))


def resolve_intrinsics(topic, width, height, *, live=None, live_sizes=None, cameras=None):
    color = canonical_color_topic(topic)
    keys = list(dict.fromkeys((topic, color, color.replace('_color', '_depth'))))
    for key in keys:
        k = (live or {}).get(key)
        size = (live_sizes or {}).get(key)
        if k is not None and size:
            return scaled_intrinsics(k, size, (width, height)), 'CameraInfo'
    if cameras is None:
        cameras = load_store().get('cameras') or {}
    entry = cameras.get(color) or {}
    k = calibrated_intrinsics(entry, width, height, color)
    if k is not None:
        return k, '标定内参' if entry.get('intrinsics') else '相机出厂内参'
    return (0.9 * max(width, height), 0.9 * max(width, height), width / 2., height / 2.), '估算内参'


def valid_depth_mask(depth_m, min_m=DEPTH_MIN_M, max_m=DEPTH_MAX_M):
    depth_m = np.asarray(depth_m)
    return np.isfinite(depth_m) & (depth_m > min_m) & (depth_m < max_m)


def sanitize_depth(depth, min_m=DEPTH_MIN_M, max_m=DEPTH_MAX_M):
    arr = np.asarray(depth)
    meters = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        meters /= 1000.
    return np.ascontiguousarray(np.where(valid_depth_mask(meters, min_m, max_m), meters, 0.), dtype=np.float32)


def validated_camera_transform(entry):
    """Use the same acceptance criteria for pose estimation and cloud display."""
    if not entry:
        return None, '缺少外参'
    model = entry.get('model')
    if model not in ('eye_to_hand', 'eye_in_hand'):
        return None, '外参类型未知'
    try:
        rms = float(entry.get('rms_m', float('inf')))
        if not np.isfinite(rms) or rms > .02:
            return None, f'标定 RMS={rms:.3f}m，超过 0.02m'
        key = 'T_tcp_camera' if model == 'eye_in_hand' else 'T_base_camera'
        t = np.asarray(entry.get(key), dtype=np.float64)
        if t.shape != (4,4) or not np.isfinite(t).all():
            return None, '外参矩阵无效'
        r=t[:3,:3]
        if not np.allclose(t[3],[0,0,0,1],atol=1e-6) or not np.allclose(r.T@r,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(r),1.,atol=1e-3):
            return None, '外参不是刚体变换'
        if model=='eye_in_hand' and np.linalg.norm(t[:3,3]) > .35:
            return None, '腕部相机相对 TCP 平移超过 0.35m'
        return t, ''
    except (TypeError,ValueError):
        return None, '外参数据无效'
