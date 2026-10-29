"""Geometry-based hand-shape suggestions; calibrated presets are optional."""
import numpy as np


RUIYAN_FACTORY_HAND_MODEL = 'ruiyan_6dof_v1'
RUIYAN_FACTORY_MAX_APERTURE_M = .12
RUIYAN_PREOPEN_CLEARANCE_M = .01
RUIYAN_FACTORY_OPEN = np.asarray([0., 0., 0., 0., 0., 0.])
RUIYAN_FACTORY_CLOSE = np.asarray([.9, .9, 1., 1., 1., 1.])
_RUIYAN_ACTIVE_CHANNELS = {
    'edge_pinch': (0, 1, 2),
    'precision_pinch': (0, 1, 2),
    'tripod': (0, 1, 2, 3),
    'wrap': (0, 1, 2, 3, 4, 5),
    'power_wrap': (0, 1, 2, 3, 4, 5),
}


def suggest_hand_shape(observation):
    if not observation or not observation.get('ok'):
        raise ValueError('请先完成视觉6D估计')
    sizes = np.asarray(observation.get('object_pose', {}).get('extents_m', []), dtype=float)
    if sizes.shape != (3,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('视觉结果缺少有效物体尺寸，不能仅凭位置推算手型')
    order = np.argsort(sizes)
    thin, width, length = sizes[order]
    if thin < .012 and width > 3*thin:
        key, name, fingers = 'edge_pinch', '薄片边缘捏取', ['拇指', '食指']
        note = '目标较薄，需选择露出的边缘；桌面接触处不能直接插入手指。'
    elif width < .025:
        key, name, fingers = 'precision_pinch', '两指精细捏取', ['拇指', '食指']
        note = '横截面较小，优先小范围对向接触；细长物体应靠近重心抓取。'
    elif length > 2.5*width:
        key, name, fingers = 'wrap', '细长物体包覆抓取', ['拇指', '食指', '中指', '无名指', '小指']
        note = '长宽比大，建议沿横截面包覆；包围盒不能证明物体为圆柱。'
    elif width < .065:
        key, name, fingers = 'tripod', '三指捏取', ['拇指', '食指', '中指']
        note = '中小尺寸目标，优先三点支撑，避免其余手指干涉。'
    else:
        key, name, fingers = 'power_wrap', '掌心包覆抓取', ['拇指', '食指', '中指', '无名指', '小指']
        note = '目标较大，先核对实际最大开口，不能仅靠增大闭合量抓紧。'
    return dict(key=key, name=name, fingers=fingers, extents_m=sizes.tolist(),
                suggested_aperture_m=float(width+RUIYAN_PREOPEN_CLEARANCE_M), closing_axis_index=int(order[1]),
                note=note, method='obb_geometry_heuristic', execution_ready=False,
                limitations='仅根据可见点云包围盒的启发式建议；未识别材质、重量、摩擦、空心结构或接触力。')


def calibrated_preset(profile, suggestion):
    from eai_app.domain.grasp.execution import hand_positions
    preset = profile.get('hand_shape_presets', {}).get(suggestion['key'])
    if preset and preset.get('calibrated') is True:
        lo, hi = map(float, preset['aperture_range_m'])
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            raise ValueError('手型适用开口范围无效')
        if not lo <= suggestion['suggested_aperture_m'] <= hi:
            raise ValueError('物体所需开口超出该手型已标定范围')
        return hand_positions(preset['open_positions']), hand_positions(preset['close_positions'])

    opened = np.asarray(hand_positions(profile.get('open_positions')), dtype=float)
    closed = np.asarray(hand_positions(profile.get('close_positions')), dtype=float)
    is_factory_ruiyan = (
        profile.get('hand_model') == RUIYAN_FACTORY_HAND_MODEL
        and np.allclose(opened, RUIYAN_FACTORY_OPEN)
        and np.allclose(closed, RUIYAN_FACTORY_CLOSE)
    )
    if not is_factory_ruiyan:
        raise ValueError('该手型尚无已标定开合映射，请在抓取标定页加载默认值或配置 hand_shape_presets。')

    aperture = float(suggestion['suggested_aperture_m'])
    if not np.isfinite(aperture) or not 0 < aperture <= RUIYAN_FACTORY_MAX_APERTURE_M:
        raise ValueError('物体所需开口超出睿研默认最大开口 120 mm')
    active = _RUIYAN_ACTIVE_CHANNELS.get(suggestion['key'])
    if active is None:
        raise ValueError('睿研默认配置不支持该手型')

    # The existing Ruiyan UI defines a linear 120 mm full-open to 0 mm
    # full-close scale.  Pre-open with 10 mm clearance, then close only to the
    # estimated object width.  Actual contact still needs operator observation.
    preopen_fraction = 1.0 - aperture / RUIYAN_FACTORY_MAX_APERTURE_M
    object_width = max(0.0, aperture - RUIYAN_PREOPEN_CLEARANCE_M)
    contact_fraction = 1.0 - object_width / RUIYAN_FACTORY_MAX_APERTURE_M
    preopen = opened.copy()
    target = opened.copy()
    for index in active:
        travel = closed[index] - opened[index]
        preopen[index] = opened[index] + preopen_fraction * travel
        target[index] = opened[index] + contact_fraction * travel
    return hand_positions(preopen.tolist()), hand_positions(target.tolist())
