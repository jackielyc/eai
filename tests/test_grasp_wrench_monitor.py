import threading
import unittest
from types import SimpleNamespace

from show_camera_topics import CameraTopicWindow


class WrenchMonitorTests(unittest.TestCase):
    def test_arm_state_is_cached_by_side_with_units_unchanged(self):
        owner = SimpleNamespace(_grasp_wrench_lock=threading.Lock(), _grasp_wrench_cache={})
        msg = SimpleNamespace(force_data=[3., 4., 0., .1, .2, .3], force_state=0)
        CameraTopicWindow._on_grasp_wrench(owner, 'right', msg)
        sample = CameraTopicWindow._grasp_wrench_feedback(owner, 'right')
        self.assertEqual(sample['force'], [3., 4., 0.])
        self.assertEqual(sample['torque'], [.1, .2, .3])
        self.assertEqual(sample['force_state'], 0)
        self.assertGreater(sample['stamp'], 0)
        self.assertIsNone(CameraTopicWindow._grasp_wrench_feedback(owner, 'left'))

    def test_invalid_data_is_fail_safe(self):
        owner = SimpleNamespace(_grasp_wrench_lock=threading.Lock(), _grasp_wrench_cache={})
        CameraTopicWindow._on_grasp_wrench(
            owner, 'left', SimpleNamespace(force_data=[1.], force_state=-1))
        sample = CameraTopicWindow._grasp_wrench_feedback(owner, 'left')
        self.assertIsNone(sample['force'])
        self.assertEqual(sample['force_state'], -1)


if __name__ == '__main__':
    unittest.main()
