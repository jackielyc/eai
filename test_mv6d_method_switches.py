"""Independent method selection: real UI state, mocked inference only."""
import itertools
import os
import unittest
from types import SimpleNamespace as NS
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt5.QtWidgets import QApplication,QCheckBox,QComboBox,QPushButton
import show_camera_topics as m
from test_mv6d_lingbot_pipeline import Harness

ATTRS = {'pca':'mv6d_pca_enabled_check',
         'foundationpose':'mv6d_foundationpose_enabled_check',
         'any6d':'mv6d_any6d_enabled_check'}


class InferenceSwitchTests(unittest.TestCase):
    def configure(self,h,enabled):
        for backend,attr in ATTRS.items():
            setattr(h,attr,NS(isChecked=lambda backend=backend:backend in enabled))

    def test_every_nonempty_subset_calls_only_enabled_methods(self):
        keys=tuple(ATTRS)
        for n in range(1,4):
            for enabled in itertools.combinations(keys,n):
                with self.subTest(enabled=enabled):
                    # PCA is the saved result source even when it is disabled.
                    h=Harness(backend='pca');self.configure(h,enabled)
                    result=h.run()
                    self.assertTrue(result['ok'],result.get('error'))
                    self.assertEqual([name for name,_ in h.model_depths],list(enabled))
                    self.assertEqual(h.snapshot_inputs['pose_backend'],enabled[0])
                    self.assertEqual(set(result['method_poses']),set(enabled))

    def test_all_off_skips_depth_segmentation_and_pose(self):
        h=Harness();self.configure(h,())
        h._start_mv6d_estimate(silent=True)
        self.assertEqual(h.events,[]);self.assertEqual(h.outputs,[])
        self.assertFalse(h._mv6d_busy)


class UiHarness:
    _mv6d_enabled_methods=m.CameraTopicWindow._mv6d_enabled_methods
    _on_mv6d_methods_changed=m.CameraTopicWindow._on_mv6d_methods_changed
    def __init__(self):
        self._mv6d_busy=False
        self.cleared=[]
        self._robot_method_poses={key:('old',) for key in ATTRS}
        self._robot_view=NS(set_method_pose=lambda key,*args:self.cleared.append(key))
        self.mv6d_once_btn=QPushButton()
        self.mv6d_enable_check=QCheckBox();self.mv6d_enable_check.setChecked(True)
        self.mv6d_method_combo=QComboBox()
        for key in ATTRS:self.mv6d_method_combo.addItem(key,key)
        for key,attr in ATTRS.items():
            check=QCheckBox(key);check.setChecked(True);setattr(self,attr,check)
            check.toggled.connect(self._on_mv6d_methods_changed)


class UiSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_disabled_primary_moves_to_enabled_and_clears_its_cloud(self):
        h=UiHarness();h.mv6d_pca_enabled_check.setChecked(False)
        self.assertEqual(h.mv6d_method_combo.currentData(),'foundationpose')
        self.assertFalse(h.mv6d_method_combo.model().item(0).isEnabled())
        self.assertNotIn('pca',h._robot_method_poses)
        self.assertIn('pca',h.cleared)

    def test_all_off_stops_realtime_and_reenable_restores_estimate(self):
        h=UiHarness()
        for attr in ATTRS.values():getattr(h,attr).setChecked(False)
        self.assertFalse(h.mv6d_enable_check.isChecked())
        self.assertFalse(h.mv6d_once_btn.isEnabled())
        self.assertFalse(h.mv6d_method_combo.isEnabled())
        h.mv6d_any6d_enabled_check.setChecked(True)
        self.assertTrue(h.mv6d_once_btn.isEnabled())
        self.assertEqual(h.mv6d_method_combo.currentData(),'any6d')

    def test_switches_are_locked_during_inference(self):
        h=UiHarness();h._mv6d_busy=True;h._on_mv6d_methods_changed()
        for attr in ATTRS.values():self.assertFalse(getattr(h,attr).isEnabled())
        self.assertFalse(h.mv6d_method_combo.isEnabled())
        h._mv6d_busy=False;h._on_mv6d_methods_changed()
        for attr in ATTRS.values():self.assertTrue(getattr(h,attr).isEnabled())

    def test_preferences_round_trip_through_existing_state_storage(self):
        h=UiHarness();h.mv6d_foundationpose_enabled_check.setChecked(False)
        h.mv6d_any6d_enabled_check.setChecked(False)
        saved=m._ui_state_collect_owner(h)
        other=UiHarness();m._ui_state_apply_owner(other,saved);other._on_mv6d_methods_changed()
        self.assertEqual(other._mv6d_enabled_methods(),('pca',))


if __name__=='__main__':unittest.main()
