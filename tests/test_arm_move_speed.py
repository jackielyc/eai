from eai_app.paths import PROJECT_ROOT, TEST_DATA_ROOT
import unittest
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import show_camera_topics as m


class ArmSpeedTests(unittest.TestCase):
    def test_slider_endpoints_and_default_roundtrip(self):
        self.assertAlmostEqual(m.slider_to_arm_move_speed(10), .1)
        self.assertAlmostEqual(m.slider_to_arm_move_speed(100), 3.14)
        self.assertAlmostEqual(m.slider_to_arm_move_speed(1000), 3.14)
        self.assertLess(abs(m.slider_to_arm_move_speed(m.arm_move_speed_to_slider(.45))-.45), .02)

    def test_maximum_agrees_with_model_and_controller(self):
        root = PROJECT_ROOT.parent / 'a2d-tele/modules/control/wbc'
        urdf = ET.parse(root/'assets/A2D_NoHand/A2D_NoHand_Flattened.urdf')
        names = {f'joint{i}_{s}' for s in ('l','r') for i in range(1,8)}
        limits = [float(j.find('limit').get('velocity')) for j in urdf.findall('joint') if j.get('name') in names]
        self.assertEqual(len(limits),14)
        self.assertLessEqual(m.ARM_MOVE_SPEED_MAX_RAD_S, min(limits))
        import yaml
        config=yaml.safe_load((root/'config/a2d_tele.robot.yaml').read_text())
        values=config['/**']['ros__parameters']['robot']['joint_limits']['velocity'][2:]
        self.assertEqual(len(values),14)
        self.assertLessEqual(m.ARM_MOVE_SPEED_MAX_RAD_S,min(map(float,values)))

    def test_discrete_playback_respects_selected_limit_even_on_long_moves(self):
        start=np.zeros(16)
        goal=start.copy();goal[2]=3.;goal[0]=.5
        for speed in (.1, 1.,3.14):
            duration=m.compute_arm_move_duration_s(start.tolist(),goal.tolist(),speed)
            steps=math.ceil(duration*m.ARM_MOVE_JOINT_HZ)
            rows=np.array([start.tolist()]+m.build_interpolated_joints(start.tolist(),goal.tolist(),steps))
            velocities=np.abs(np.diff(rows,axis=0))*m.ARM_MOVE_JOINT_HZ
            self.assertLessEqual(velocities[:,2:].max(),speed+1e-8)
            self.assertLessEqual(velocities[:,0].max(),.5+1e-8)
        self.assertGreater(m.compute_arm_move_duration_s(start.tolist(),goal.tolist(),.1),10)


if __name__ == '__main__': unittest.main()
