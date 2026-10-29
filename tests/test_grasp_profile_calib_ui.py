from eai_app.paths import PROJECT_ROOT, TEST_DATA_ROOT
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
        template = (PROJECT_ROOT / 'config') / 'grasp_execution.example.json'
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

    def test_ruiyan_factory_defaults_fill_only_hand_fields(self):
        self.panel.tcp_translation[0].setValue(8.0)
        self.panel.workspace[0][0].setValue(0.1)
        for box in (self.panel.tcp_checked, self.panel.hand_checked, self.panel.workspace_checked):
            box.setChecked(True)

        self.panel.load_ruiyan_factory_defaults()

        self.assertEqual(self.panel.hand_model.text(), 'ruiyan_6dof_v1')
        self.assertEqual([box.value() for box in self.panel.open_positions], [0.0] * 6)
        self.assertEqual(
            [box.value() for box in self.panel.close_positions],
            [0.9, 0.9, 1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(self.panel.tcp_translation[0].value(), 8.0)
        self.assertEqual(self.panel.workspace[0][0].value(), 0.1)
        self.assertFalse(self.panel.hand_checked.isChecked())
        self.assertFalse(self.panel.save_button.isEnabled())

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
