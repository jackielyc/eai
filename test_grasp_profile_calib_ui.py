import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication

from grasp_profile_calib_ui import GraspProfileCalibrationPanel


class ProfileCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        template = Path(__file__).with_name('config') / 'grasp_execution.example.json'
        self.path = Path(self.tmp.name) / 'grasp.json'
        self.path.write_text(template.read_text())
        self.saved = []
        self.panel = GraspProfileCalibrationPanel(str(self.path), self.saved.append)

    def tearDown(self):
        self.panel.deleteLater()
        self.tmp.cleanup()

    def test_needs_all_explicit_operator_confirmations(self):
        self.assertFalse(self.panel.save_button.isEnabled())
        self.panel.tcp_checked.setChecked(True)
        self.panel.hand_checked.setChecked(True)
        self.assertFalse(self.panel.save_button.isEnabled())
        self.panel.workspace_checked.setChecked(True)
        self.assertTrue(self.panel.save_button.isEnabled())

    def test_save_marks_profile_only_after_confirmation(self):
        for box in (self.panel.tcp_checked, self.panel.hand_checked, self.panel.workspace_checked):
            box.setChecked(True)
        self.panel.hand_model.setText('ruiyan_6dof_v1')
        self.panel.tcp_translation[0].setValue(8.0)
        self.panel.save()
        profile = json.loads(self.path.read_text())
        self.assertTrue(profile['calibrated'])
        self.assertEqual(profile['hand_model'], 'ruiyan_6dof_v1')
        self.assertAlmostEqual(profile['T_grasp_tcp'][0][3], .08)
        self.assertEqual(self.saved, [str(self.path)])


if __name__ == '__main__':
    unittest.main()
