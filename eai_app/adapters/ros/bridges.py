"""adapters.ros.bridges — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class RosBridge(_eai_platform_bootstrap.QObject):
    frame_updated = _eai_platform_bootstrap.pyqtSignal(str, object)
    # 合并投递：避免跨线程 QueuedConnection 堆积大量图像拷贝导致 UI 卡死/内存暴涨
    frame_tick = _eai_platform_bootstrap.pyqtSignal()
    topics_updated = _eai_platform_bootstrap.pyqtSignal(dict)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    frame_stats = _eai_platform_bootstrap.pyqtSignal(str, int, float)
    robot_state_updated = _eai_platform_bootstrap.pyqtSignal()
    arm_enable_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    hand_enable_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    control_mode_changed = _eai_platform_bootstrap.pyqtSignal(int)
    replay_state_changed = _eai_platform_bootstrap.pyqtSignal(int)
    left_hand_preset_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    right_hand_preset_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    slow_motion_progress = _eai_platform_bootstrap.pyqtSignal(float, str)
    slow_motion_finished = _eai_platform_bootstrap.pyqtSignal(bool, str)


class PoseComputeBridge(_eai_platform_bootstrap.QObject):
    """后台位姿计算完成信号（跨线程投递到 UI）。"""

    finished = _eai_platform_bootstrap.pyqtSignal(object, str, int, int, object)


class DepthVizBridge(_eai_platform_bootstrap.QObject):
    """后台深度预览/点云计算完成信号。"""

    finished = _eai_platform_bootstrap.pyqtSignal(object)

