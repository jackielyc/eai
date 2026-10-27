"""Local kinematic simulation; no ROS publishers or hardware commands."""
import hashlib
import time
from pathlib import Path
import numpy as np


class GraspSimulation:
    def __init__(self, profile, seed, display):
        from robot_view3d import _Kinematics
        self.kin = _Kinematics(profile['urdf_path'])
        self.model_hash = hashlib.sha256(Path(profile['urdf_path']).read_bytes()).hexdigest()
        self.q = list(seed)
        self.hands = {}
        self.display = display

    def feedback(self):
        poses = self.kin.link_poses(self.q)
        stamp = time.time()
        result = dict(q16=list(self.q), urdf_sha256=self.model_hash,
                      arm_received_at=stamp, waist_received_at=stamp)
        for side, link in (('left', 'Link7_l'), ('right', 'Link7_r')):
            T = poses[link]
            quat = self.kin._pin.Quaternion(T[:3, :3]).coeffs()
            result[side] = dict(xyz=T[:3, 3].tolist(), xyzw=np.asarray(quat).tolist())
        return result

    def publish(self, q):
        self.q = list(q)
        self.display(self.q, self.hands)

    def hand(self, side, values):
        self.hands[side] = list(values)
        self.display(self.q, self.hands)
