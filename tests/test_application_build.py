"""Construct every page with external I/O blocked; never starts a ROS node."""
import os
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication, QWidget
import eai_app.application.viewer as viewer


class InlineExecutor:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try: future.set_result(fn(*args, **kwargs))
        except Exception as exc: future.set_exception(exc)
        return future


class ApplicationBuildTests(unittest.TestCase):
    def test_unconfirmed_stop_and_grip_block_hot_update(self):
        from types import SimpleNamespace as NS
        from eai_app.development.qt_controller import DevelopmentController
        panel = NS(sequence=NS(state='stop_unconfirmed'), other_busy=lambda:False)
        controller = NS(window=NS(grasp_execution_panel=panel))
        self.assertIn('停止尚未确认', DevelopmentController.busy_reason(controller))
        panel.sequence.state = 'idle'
        panel._unverified_grips = {'right'}
        self.assertTrue(DevelopmentController.busy_reason(controller))
        panel._unverified_grips.clear()
        self.assertEqual(DevelopmentController.busy_reason(controller), '')

    def test_all_pages_and_development_menu_build_without_external_io(self):
        app = QApplication.instance() or QApplication([])
        viewer._ensure_cv2_numpy()
        node = MagicMock()
        node.is_arm_enabled.return_value = node.is_hand_enabled.return_value = False
        node.is_slow_motion_busy.return_value = False
        node._grasp_execution_reserved = False
        node.list_topics.return_value = []
        for name in ('get_arm_enable_label', 'get_hand_enable_label', 'get_control_mode_label',
                     'get_arm_move_current_label_both'):
            getattr(node, name).return_value = ''
        node._tcp_pose_in_ik_frame.return_value = None
        node._arm_enable_received_at = node._hand_enable_received_at = node._control_mode_received_at = 0.
        node.get_robot_state.return_value = viewer.RobotStateSnapshot([], [], [], [], [], None, None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(viewer, 'VIEWER_UI_STATE_PATH', directory+'/state.json'), \
             patch.object(viewer, 'ThreadPoolExecutor', InlineExecutor), \
             patch('threading.Thread.start'), \
             patch('subprocess.Popen', side_effect=RuntimeError('external processes blocked')), \
             patch('subprocess.run', return_value=MagicMock(returncode=1, stdout='', stderr='')), \
             patch('subprocess.check_output', return_value=''), \
             patch('urllib.request.urlopen', side_effect=RuntimeError('network blocked')), \
             patch('socket.socket.connect', side_effect=OSError('network blocked')), \
             patch.object(viewer.QTimer, 'start'), patch.object(viewer.QTimer, 'singleShot'), \
             patch.object(viewer.cv2, 'VideoCapture', side_effect=RuntimeError('camera blocked')), \
             patch.object(viewer, 'QWebEngineView', None):
            window = viewer.CameraTopicWindow(node, viewer.RosBridge(), '/camera')
            try:
                window.build_heavy_ui()
                self.assertEqual(window.control_tabs.count(), 35)
                self.assertGreaterEqual(len(window.findChildren(QWidget)), 3118)
                self.assertTrue(window._development_controller.automatic.isChecked())
                self.assertEqual(window._development_controller.reloader.changed_paths(), [])
            finally:
                from eai_app.development import qt_controller, source_guard
                window._viewer_ui_state_saved = True
                window._development_controller.timer.stop()
                qt_controller._controller = None
                source_guard.install_check(None)
                window.deleteLater()
